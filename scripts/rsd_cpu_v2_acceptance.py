#!/usr/bin/env python3
"""Six-run CPU plumbing acceptance with random tiny GPT2; never a research run.

Uses repository test fixtures so the reference setup stays shared with tests.
Untrained tiny input assets are retained for reconstruction; trained weights are
never exported. Explicit CPU calls and offline Hub flags are mandatory here.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    output = Path(args.output).absolute()
    if output.exists() or Path(args.receipt).exists():
        raise FileExistsError("Acceptance outputs must be new; preserve previous attempts")
    if any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError("Acceptance path cannot traverse a symlink")
    import torch
    from test_rsd_training import tiny_assets
    from test_rsd_confirmation import fixture_plan
    from rsd.confirmation import (release_confirmation, score_released_task,
                                   export_released_judge_inputs, write_new, sha)
    from rsd.task_analysis import analyze_released_tasks
    from rsd.train import run
    torch.set_num_threads(1)
    output.mkdir(parents=True)
    tiny_assets(output / "tiny-model")
    plan_path, configs = fixture_plan(output)
    observed = []
    for config in configs:
        run_id = config["experiment"]["run_id"]
        write_new(output / (run_id + ".config.json"), config)
        trainer = run(config, use_cpu=True)
        directory = Path(config["output_dir"])
        result = json.loads((directory / "result.json").read_text())
        telemetry = [json.loads(line) for line in (directory / "optimizer_rank0.jsonl").read_text().splitlines()]
        assert result["global_step"] == 2 and result["final_model_exported"] is False
        assert result["parameter_probe"]["changed_coordinates"] > 0
        if config["teacher_mode"] == "frozen":
            assert result["frozen_teacher_probe"]["changed_coordinates"] == 0
        assert len(telemetry) == 2 and telemetry[-1]["attempt_global_sample_exposures"] == 8
        assert not list(directory.rglob("*.safetensors")) and not list(directory.rglob("*.pt"))
        observed.append({"run_id": run_id, "teacher_mode": config["teacher_mode"], "seed": config["training_seed"],
                         "optimizer_steps": 2, "effective_batch": 4, "sample_exposures": 8,
                         "valid_training_tokens": telemetry[-1]["attempt_global_valid_tokens"],
                         "student_changed_sampled_coordinates": result["parameter_probe"]["changed_coordinates"],
                         "frozen_teacher_changed_sampled_coordinates": None if config["teacher_mode"] != "frozen" else 0,
                         "result_sha256": sha(directory / "result.json"),
                         "contract_sha256": sha(directory / "run_contract.json")})
        del trainer
        gc.collect()
    release_path = output / "release.json"
    release = release_confirmation(plan_path, output / "runs", release_path, root=ROOT)
    assert release["purpose"] == "cpu_fixture" and release["scientific_ready"] is False
    scores = []
    for config in configs:
        run_id = config["experiment"]["run_id"]
        score = score_released_task(plan_path, output / "runs", release_path,
            output / (run_id + ".released-task-score.json"), root=ROOT, run_id=run_id, step=2)
        assert score["confirmation_release_gate_verified"] and score["n_planned"] == 1
        scores.append({"run_id": run_id, "n_planned": score["n_planned"], "n_unknown": score["n_unknown"]})
    analysis_path = output / "released-task-paired.json"
    analysis = analyze_released_tasks(plan_path, output / "runs", release_path, analysis_path, root=ROOT, step=2)
    assert analysis["confirmation_release_gate_verified"] and analysis["scientific_conclusion"] is None
    export_released_judge_inputs(plan_path, output / "runs", release_path, output / "released-judge-inputs",
        root=ROOT, run_id=configs[0]["experiment"]["run_id"], step=2)
    receipt = {"schema_version": "rsd-cpu-v2-pipeline-acceptance", "status": "PASS_CPU_FIXTURE_ONLY",
               "purpose": "random tiny GPT2 plumbing acceptance; no Qwen/Gemma research evidence",
               "model": "random tiny GPT2, 1 layer, hidden size 16", "device": "CPU", "runs": observed,
               "total_optimizer_steps": 12, "total_sample_exposures": 48, "dev_nodes_per_run": [0, 1, 2],
               "confirmation_nodes_per_run": [0, 2], "confirmation_release_verified": True,
               "confirmation_task_scores": scores, "paired_task_analysis_status": analysis["status"],
               "sealed_judge_inputs_exported_after_release": True, "judge_model_calls": 0,
               "gpu_executions": 0, "pretrained_weights_loaded": 0, "trained_weights_persisted": False,
               "retained_untrained_tiny_input_assets": True,
               "plan_sha256": sha(plan_path), "release_sha256": sha(release_path),
               "paired_analysis_sha256": sha(analysis_path),
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "scientific_ready": False, "scientific_conclusion": None,
               "limits": ["CUDA/DeepSpeed and long outputs not exercised.",
                          "Fixture judge profile is not an actual calibrated judge; no benchmark judge results produced.",
                          "Single-process integration uses effective batch4; actual 2-rank CPU batch8 parity is recorded separately."]}
    write_new(output / "acceptance-receipt.json", receipt)
    write_new(args.receipt, receipt)
    print(json.dumps({"status": receipt["status"], "runs": 6, "optimizer_steps": 12, "scientific_ready": False}))


if __name__ == "__main__":
    main()
