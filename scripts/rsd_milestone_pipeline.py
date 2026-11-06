"""Bounded, sequential GPU training and deferred local judging (stdlib only).

Run from a verified project environment. No model imports, installations,
automatic retries, checkpoint exports, or edits to callback-owned artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = {"manifest.json", "task-dataset.jsonl", "benchmark-dataset.jsonl",
             "task-predictions.jsonl", "benchmark-predictions.jsonl", "task-summary.json",
             "benchmark-summary-pending.json", "judge-pending.json"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def project_path(value, root):
    path = (root / value).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError("Output must be a child of the project root")
    return path


def build_plan(args, root=ROOT):
    root = Path(root).resolve()
    config_path = Path(args.config).expanduser().resolve()
    config = read(config_path)
    if config.get("save_strategy") != "no" or config.get("export_final_model") is not False:
        raise ValueError("Require explicit save_strategy=no and export_final_model=false")
    spec = config.get("milestone_eval")
    steps = spec.get("steps") if isinstance(spec, dict) else None
    if (not steps or any(type(s) is not int or s < 0 for s in steps)
            or steps != sorted(set(steps)) or steps[0] != 0 or steps[-1] != config.get("max_steps")):
        raise ValueError("Milestones must include step zero and final step in increasing order")
    world = config.get("expected_world_size")
    if type(world) is not int or world < 1:
        raise ValueError("expected_world_size must be a positive integer")
    for value in (args.train_timeout_seconds, args.judge_timeout_seconds, args.ram_stop_gib):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Timeouts and RAM stop budget must be positive and finite")
    train_output = project_path(config["output_dir"], root)
    output = project_path(str(Path(args.output_dir).expanduser().resolve()), root)
    if output.is_relative_to(train_output) or train_output.is_relative_to(output):
        raise ValueError("Training and wrapper output directories must be separate")
    if output.exists() or train_output.exists():
        raise FileExistsError("Both wrapper and training output directories must be new")
    model = Path(args.judge_model).expanduser().resolve()
    if not model.is_dir() or not (model / "config.json").is_file():
        raise ValueError("Judge must be an existing local model directory")
    if output.is_relative_to(model) or train_output.is_relative_to(model):
        raise ValueError("Cannot write into the read-only judge model directory")
    deepspeed = (root / config["deepspeed"]).resolve()
    if read(deepspeed).get("zero_optimization", {}).get("stage") != 2:
        raise ValueError("This resident rank-zero pipeline requires ZeRO-2")
    python = shutil.which(args.python)
    if python is None:
        raise ValueError("Python executable does not exist")
    datasets = {}
    for key in ("task_dataset", "benchmark_dataset"):
        path = (root / spec[key]).resolve()
        dataset = rows(path)
        allowed = {"exact_match"} if key == "task_dataset" else {"safety", "overrefusal"}
        if not dataset or any(r.get("split") != "dev" or r.get("metric") not in allowed for r in dataset):
            raise ValueError("Milestone cohorts must be nonempty dev-only datasets")
        if len({r["id"] for r in dataset}) != len(dataset):
            raise ValueError("Duplicate dataset ids")
        datasets[key] = {"path": str(path), "sha256": sha(path), "n": len(dataset)}
    return {"config_path": str(config_path), "config_sha256": sha(config_path), "config": config,
            "root": str(root), "output_dir": str(output), "training_output_dir": str(train_output),
            "python": python, "judge_model": str(model), "steps": steps, "datasets": datasets,
            "deepspeed_sha256": sha(deepspeed), "world_size": world,
            "train_argv": [python, "-m", "torch.distributed.run", "--standalone",
                           f"--nproc-per-node={world}", "-m", "rsd.train", "--config", str(config_path)],
            "judge_generation": {"device": "cuda:0", "max_prompt_tokens": 8192, "max_new_tokens": 512},
            "timeouts": {"train": args.train_timeout_seconds, "judge": args.judge_timeout_seconds},
            "ram_stop_bytes": int(args.ram_stop_gib * 2**30), "scientific_conclusion": None,
            "scope": "Dev-only engineering milestones; sampled identities, not research conclusions"}


def checked_command(argv, cwd):
    result = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=15, check=True)
    return result.stdout.strip()


def gpu_idle(world, root):
    devices = checked_command(["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
                               "--format=csv,noheader,nounits"], root).splitlines()
    values = [[int(v.strip()) for v in line.split(",")] for line in devices]
    pids = checked_command(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], root)
    if len(values) != world or pids or any(used >= 1000 for _, used, _ in values):
        raise RuntimeError("GPU inventory/idle guard failed; inspect ownership, do not kill other processes")
    return values


def resource_snapshot(root):
    ram = None
    for path in (Path("/sys/fs/cgroup/memory.current"), Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")):
        if path.is_file():
            ram = int(path.read_text())
            break
    gpu = checked_command(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                           "--format=csv,noheader,nounits"], root)
    return {"time": time.time(), "cgroup_ram_bytes": ram, "disk_free_bytes": shutil.disk_usage(root).free,
            "gpus": [dict(zip(("index", "used_mib", "total_mib", "util_percent"),
                              [int(v.strip()) for v in line.split(",")])) for line in gpu.splitlines()]}


def terminate_owned(proc):
    """Only the process group created by this runner's start_new_session Popen."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=10)


def run_phase(name, argv, timeout, plan, receipt):
    output, root = Path(plan["output_dir"]), Path(plan["root"])
    phase = {"name": name, "argv": argv, "status": "starting", "started_at": time.time()}
    receipt["phases"].append(phase)
    write(output / "pipeline_receipt.json", receipt)
    proc, started = None, time.monotonic()
    try:
        with (output / f"{name}.log").open("x") as log, (output / "resources.jsonl").open("a") as samples:
            proc = subprocess.Popen(argv, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True)
            phase.update(pid=proc.pid, status="running")
            write(output / "pipeline_receipt.json", receipt)
            while True:
                sample = resource_snapshot(root)
                sample.update(phase=name, pid=proc.pid)
                samples.write(json.dumps(sample) + "\n")
                samples.flush()
                os.fsync(samples.fileno())
                if sample["cgroup_ram_bytes"] is None:
                    raise RuntimeError("Cannot enforce RAM budget without cgroup usage")
                if sample["cgroup_ram_bytes"] >= plan["ram_stop_bytes"]:
                    raise RuntimeError("cgroup_ram_budget_exceeded")
                rc = proc.poll()
                if rc is not None:
                    phase["returncode"] = rc
                    if rc:
                        raise RuntimeError(f"{name} exited {rc}")
                    break
                if time.monotonic() - started >= timeout:
                    raise TimeoutError(f"{name} timeout after {timeout} seconds")
                time.sleep(min(3, max(0.01, timeout - (time.monotonic() - started))))
        phase["status"] = "completed"
    except BaseException as exc:
        if proc is not None:
            terminate_owned(proc)
            phase["returncode"] = proc.poll()
        phase.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        phase.update(completed_at=time.time(), elapsed_seconds=time.monotonic() - started)
        path = output / f"{name}.log"
        if path.exists():
            phase["log_sha256"] = sha(path)
        write(output / "pipeline_receipt.json", receipt)


def validate_milestone(folder, step, contract, plan):
    receipt, manifest = read(folder / "receipt.json"), read(folder / "manifest.json")
    if receipt.get("status") != "completed" or receipt.get("global_step") != step:
        raise ValueError("Milestone incomplete or wrong step")
    if set(receipt.get("artifact_sha256", {})) != ARTIFACTS:
        raise ValueError("Milestone artifact inventory differs")
    for name, expected in receipt["artifact_sha256"].items():
        if sha(folder / name) != expected:
            raise ValueError(f"Milestone artifact hash mismatch: {name}")
    contract_sha = digest(contract)
    identity = manifest["checkpoint_identity"]
    body = {k: v for k, v in identity.items() if k != "checkpoint_id"}
    if (manifest["run_contract_sha256"] != contract_sha or identity.get("run_contract_sha256") != contract_sha
            or identity.get("global_step") != step or identity.get("checkpoint_id") != "in-memory:" + digest(body)
            or receipt.get("checkpoint_id") != identity["checkpoint_id"]):
        raise ValueError("Milestone in-memory identity mismatch")
    spec = plan["config"]["milestone_eval"]
    for key in ("steps", "max_prompt_tokens", "max_new_tokens", "seed"):
        if manifest["plan"].get(key) != spec[key]:
            raise ValueError("Milestone plan changed")
    protocol = manifest["generation_protocol"]
    expected_protocol = {"profile": "rsd-text-eval-v2", "enable_thinking": False,
                         "do_sample": False, "temperature": 0.0, "num_beams": 1,
                         "eos_token_id_source": "tokenizer.eos_token_id",
                         **{key: spec[key] for key in ("max_prompt_tokens", "max_new_tokens", "seed")}}
    if any(protocol.get(key) != value for key, value in expected_protocol.items()):
        raise ValueError("Milestone generation protocol differs from frozen plan")
    for label in ("task", "benchmark"):
        dataset, predictions = rows(folder / f"{label}-dataset.jsonl"), rows(folder / f"{label}-predictions.jsonl")
        original = rows(plan["datasets"][label + "_dataset"]["path"])
        if dataset != original or [p["id"] for p in predictions] != [r["id"] for r in dataset]:
            raise ValueError("Milestone dataset/order/prediction coverage differs")
        if manifest["plan"][label + "_dataset_sha256"] != digest(sorted(dataset, key=lambda r: r["id"])):
            raise ValueError("Milestone dataset identity mismatch")
        if (manifest["plan"].get(label + "_order_sha256") != digest([r["id"] for r in dataset])
                or manifest["plan"].get("n_" + label) != len(dataset)
                or receipt.get("n_" + label + "_planned") != len(dataset)
                or receipt.get("n_" + label + "_predictions") != len(predictions)):
            raise ValueError("Milestone order/count receipt differs")
        for item, prediction in zip(dataset, predictions):
            if (prediction.get("valid") is not True or not isinstance(prediction.get("response"), str)
                    or not prediction["response"].strip() or prediction.get("error")
                    or prediction.get("checkpoint_identity") != identity
                    or prediction.get("checkpoint_id") != identity["checkpoint_id"]
                    or prediction.get("prompt_sha256") != hashlib.sha256(item["prompt"].encode()).hexdigest()
                    or prediction.get("generation_protocol") != manifest["generation_protocol"]):
                raise ValueError("Invalid generation or stale prediction identity; refusing formal success")
    pending = read(folder / "judge-pending.json")
    n = plan["datasets"]["benchmark_dataset"]["n"]
    if (pending.get("status") != "pending" or pending.get("checkpoint_id") != identity["checkpoint_id"]
            or any(pending.get(k) != n for k in ("n_planned", "n_eligible", "n_predictions"))
            or pending.get("judgments_created") != 0 or receipt.get("n_invalid_predictions") != 0):
        raise ValueError("Pending judge/denominator receipt is inconsistent")
    return {"global_step": step, "checkpoint_id": identity["checkpoint_id"],
            "source_receipt_sha256": sha(folder / "receipt.json"), "artifact_sha256": receipt["artifact_sha256"]}


def judge_commands(plan, folder, output):
    predictions, dataset = folder / "benchmark-predictions.jsonl", folder / "benchmark-dataset.jsonl"
    judgments = output / "judgments.jsonl"
    common = ["--dataset", str(dataset), "--predictions", str(predictions)]
    judge = [plan["python"], "-m", "rsd.local_judge", *common, "--model", plan["judge_model"],
             "--output", str(judgments), "--device", "cuda:0", "--max-prompt-tokens", "8192", "--max-new-tokens", "512"]
    score = [plan["python"], "-m", "rsd.evaluation", "score", *common, "--judgments", str(judgments),
             "--output", str(output / "benchmark-summary.json")]
    return judge, score


def validate_judgments(folder, output, expected_judge_id=None):
    dataset, predictions = rows(folder / "benchmark-dataset.jsonl"), rows(folder / "benchmark-predictions.jsonl")
    manifest = read(output / "judgments.jsonl.manifest.json")
    receipt = read(output / "judgments.jsonl.receipt.json")
    judgments = rows(output / "judgments.jsonl")
    judge_id = manifest["judge_id"]
    if expected_judge_id is not None and judge_id != expected_judge_id:
        raise ValueError("Judge identity changed between milestones")
    if (manifest.get("dataset_sha256") != digest(dataset) or manifest.get("predictions_sha256") != digest(predictions)
            or manifest.get("selected_ids") != [r["id"] for r in dataset]
            or [r["id"] for r in judgments] != [r["id"] for r in dataset]
            or receipt.get("status") not in ("completed", "completed_with_unknown")
            or receipt.get("n_unprocessed") != 0):
        raise ValueError("Judge result incomplete or input identity differs")
    for item, prediction, row in zip(dataset, predictions, judgments):
        if (row.get("judge_id") != judge_id or row.get("checkpoint_id") != prediction["checkpoint_id"]
                or row.get("metric") != item["metric"] or row.get("prompt_sha256") != prediction["prompt_sha256"]
                or row.get("response_sha256") != hashlib.sha256(prediction["response"].encode()).hexdigest()
                or type(row.get("valid")) is not bool or (not row["valid"] and row.get("value") is not None)):
            raise ValueError("Judge binding or invalid/null convention differs")
        if row["valid"] and (type(row.get("value")) not in (int, float)
                             or not math.isfinite(row["value"]) or not 0 <= row["value"] <= 1):
            raise ValueError("Valid judge value must be finite and within [0,1]")
    valid = sum(row["valid"] for row in judgments)
    if any(receipt.get(k) != v for k, v in {"n_planned": len(dataset), "n_judgments": len(dataset),
                                           "n_valid": valid, "n_invalid": len(dataset) - valid}.items()):
        raise ValueError("Judge receipt denominator mismatch")
    return {"judge_id": judge_id, "n_planned": len(dataset), "n_valid": valid, "n_unknown": len(dataset) - valid}


def execute(plan):
    root, output = Path(plan["root"]), Path(plan["output_dir"])
    if checked_command(["git", "status", "--porcelain", "--untracked-files=no"], root):
        raise RuntimeError("Tracked Git files must be clean before GPU work")
    if sha(plan["config_path"]) != plan["config_sha256"]:
        raise ValueError("Input config changed after planning")
    if sha(root / plan["config"]["deepspeed"]) != plan["deepspeed_sha256"]:
        raise ValueError("DeepSpeed config changed after planning")
    for item in plan["datasets"].values():
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("Source dataset changed after planning")
    initial_gpus = gpu_idle(plan["world_size"], root)
    initial = resource_snapshot(root)
    if initial["cgroup_ram_bytes"] is None or initial["cgroup_ram_bytes"] >= plan["ram_stop_bytes"]:
        raise RuntimeError("Initial cgroup RAM guard failed")
    if Path(plan["training_output_dir"]).exists():
        raise FileExistsError("Training output appeared after planning")
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"status": "running", "started_at": time.time(), "phases": [], "milestones": [],
               "plan": plan, "initial_resources": initial, "initial_gpu_inventory": initial_gpus,
               "git_commit": checked_command(["git", "rev-parse", "HEAD"], root),
               "runner_sha256": sha(__file__), "scientific_conclusion": None}
    write(output / "pipeline_receipt.json", receipt)
    try:
        run_phase("train", plan["train_argv"], plan["timeouts"]["train"], plan, receipt)
        training = Path(plan["training_output_dir"])
        result, contract = read(training / "result.json"), read(training / "run_contract.json")
        if (result.get("status") != "training_completed" or result.get("global_step") != plan["config"]["max_steps"]
                or result.get("final_model_exported") is not False or (training / "final").exists()
                or list(training.glob("checkpoint-*")) or contract.get("config") != plan["config"]):
            raise ValueError("Training completion/config/no-weight contract failed")
        if sha(plan["config_path"]) != plan["config_sha256"]:
            raise ValueError("Input config changed during training")
        for item in plan["datasets"].values():
            if sha(item["path"]) != item["sha256"]:
                raise ValueError("Source dataset changed during training")
        receipt["training_artifact_sha256"] = {name: sha(training / name) for name in ("result.json", "run_contract.json")}
        # Validate every source before starting any judge; no partial generation is accepted.
        for step in plan["steps"]:
            receipt["milestones"].append(validate_milestone(training / "milestones" / f"step-{step:06d}", step, contract, plan))
        judge_id = None
        for record in receipt["milestones"]:
            step = record["global_step"]
            source, target = training / "milestones" / f"step-{step:06d}", output / f"step-{step:06d}"
            gpu_idle(plan["world_size"], root)
            target.mkdir(exist_ok=False)
            judge, score = judge_commands(plan, source, target)
            run_phase(f"judge_step_{step:06d}", judge, plan["timeouts"]["judge"], plan, receipt)
            record["judging"] = validate_judgments(source, target, judge_id)
            judge_id = record["judging"]["judge_id"]
            run_phase(f"score_step_{step:06d}", score, min(120, plan["timeouts"]["judge"]), plan, receipt)
            summary = read(target / "benchmark-summary.json")
            if (summary.get("checkpoint_id") != record["checkpoint_id"]
                    or summary.get("n_planned") != record["judging"]["n_planned"]
                    or summary.get("n_valid") != record["judging"]["n_valid"]):
                raise ValueError("Scored summary identity/denominator differs")
            # Callback source files are immutable, including pending-status receipts.
            if validate_milestone(source, step, contract, plan) != {k: v for k, v in record.items() if k != "judging"}:
                raise ValueError("Source milestone changed while judging")
            record["judge_artifact_sha256"] = {p.name: sha(p) for p in target.iterdir() if p.is_file()}
        receipt["status"] = ("completed_with_unknown" if any(r["judging"]["n_unknown"] for r in receipt["milestones"])
                             else "completed")
    except BaseException as exc:
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        receipt["completed_at"] = time.time()
        write(output / "pipeline_receipt.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-timeout-seconds", type=float, default=2700)
    parser.add_argument("--judge-timeout-seconds", type=float, default=1800)
    parser.add_argument("--ram-stop-gib", type=float, default=380)
    parser.add_argument("--dry-plan", action="store_true")
    args = parser.parse_args(argv)
    plan = build_plan(args)
    if args.dry_plan:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    receipt = execute(plan)
    print(json.dumps({"status": receipt["status"], "scientific_conclusion": None}))


if __name__ == "__main__":
    main()
