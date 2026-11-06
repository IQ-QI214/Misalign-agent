#!/usr/bin/env python3
"""Collect a local GPU environment receipt without installing or downloading.

Default: system/package inventory, Torch CUDA availability, and a real CPU
Dataset -> NumPy -> Torch -> NumPy interoperability check.
Optional, explicitly requested exercises run in isolated, bounded subprocesses:
    --check-kernels        import the actual FLA and causal-conv1d functions
    --exercise-optimizer   build/load CPUAdam and update one tiny CPU parameter
    --exercise-model       random tiny Qwen3.5 CUDA forward/backward (no weights)

All script-controlled writes, logs, compiler caches and temporary files go below
--output-dir. Offline environment variables are set before optional imports.
This is cache routing, not an OS sandbox for arbitrary third-party code.
Passing this script never establishes full 9B or distributed training readiness.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone


# This must also take effect when this module is imported by a test/launcher.
sys.dont_write_bytecode = True
SCRIPT_PATH = Path(__file__).resolve()
PACKAGES = (
    "torch", "transformers", "trl", "accelerate", "peft", "datasets",
    "deepspeed", "flash-linear-attention", "causal-conv1d", "flash-attn",
    "triton", "ninja", "packaging", "safetensors", "numpy", "pyarrow", "pandas",
)
CACHE_PATHS = {
    "TMPDIR": "tmp", "TMP": "tmp", "TEMP": "tmp",
    "XDG_CACHE_HOME": "cache", "XDG_CONFIG_HOME": "config",
    "XDG_DATA_HOME": "share", "XDG_STATE_HOME": "state",
    "HF_HOME": "cache/huggingface", "HF_HUB_CACHE": "cache/huggingface/hub",
    "HF_DATASETS_CACHE": "cache/huggingface/datasets",
    "HF_MODULES_CACHE": "cache/huggingface/modules",
    "TORCH_HOME": "cache/torch", "TORCH_EXTENSIONS_DIR": "cache/torch_extensions",
    "TORCHINDUCTOR_CACHE_DIR": "cache/torchinductor",
    "TRITON_CACHE_DIR": "cache/triton", "TRITON_DUMP_DIR": "cache/triton_dump",
    "TRITON_OVERRIDE_DIR": "cache/triton_override",
    "CUDA_CACHE_PATH": "cache/cuda", "NUMBA_CACHE_DIR": "cache/numba",
    "MPLCONFIGDIR": "config/matplotlib",
    "WANDB_DIR": "state/wandb", "WANDB_CACHE_DIR": "cache/wandb",
    "WANDB_CONFIG_DIR": "config/wandb",
}
OFFLINE_ENV = {
    "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1",
    "WANDB_MODE": "disabled", "TOKENIZERS_PARALLELISM": "false",
}


def confined_path(root: Path, relative: str) -> Path:
    """Reject existing symlinks that would send our writes outside the root."""
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"Output path escapes --output-dir: {relative}")
    return candidate


def configure_output(output_dir: str | Path) -> tuple[Path, dict[str, str]]:
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Validate all paths first, before creating any cache directory.
    paths = {key: confined_path(root, value) for key, value in CACHE_PATHS.items()}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    configured = {key: str(value) for key, value in paths.items()}
    configured.update(OFFLINE_ENV)
    configured["MAX_JOBS"] = os.environ.get("MAX_JOBS", "2")
    os.environ.update(configured)
    tempfile.tempdir = str(paths["TMPDIR"])
    os.chdir(root)  # Catch relative compiler/third-party output as well.
    return root, configured


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None


def finite_limit(value: str | None) -> int | None:
    if value is None or value == "max":
        return None
    try:
        number = int(value)
    except ValueError:
        return None
    # cgroup v1 uses a huge page-aligned integer to mean unlimited.
    return number if 0 <= number < 2**60 else None


def cgroup_memory(proc_root: Path = Path("/proc"),
                  cgroup_root: Path = Path("/sys/fs/cgroup")) -> dict:
    """Read namespace root and current-group ancestors for v1 and v2 limits."""
    candidates = {cgroup_root, cgroup_root / "memory"}
    membership = read_text(proc_root / "self/cgroup")
    if membership:
        for line in membership.splitlines():
            parts = line.split(":", 2)
            if len(parts) != 3 or (parts[1] and "memory" not in parts[1].split(",")):
                continue
            relative = Path(parts[2].lstrip("/"))
            # Some cgroup namespaces expose host-relative paths with '..'.
            # In that case the namespace mount root is the authoritative read.
            if ".." in relative.parts:
                continue
            base = cgroup_root if not parts[1] else cgroup_root / "memory"
            current = base / relative
            while current.is_relative_to(base):
                candidates.add(current)
                if current == base:
                    break
                current = current.parent
    readings = []
    for directory in sorted(candidates):
        for version, limit_file, current_file in (
            (2, "memory.max", "memory.current"),
            (1, "memory.limit_in_bytes", "memory.usage_in_bytes"),
        ):
            raw_limit = read_text(directory / limit_file)
            if raw_limit is None:
                continue
            limit = finite_limit(raw_limit)
            current = finite_limit(read_text(directory / current_file))
            readings.append({
                "version": version, "path": str(directory),
                "raw_limit": raw_limit, "limit_bytes": limit,
                "usage_bytes": current,
                "remaining_bytes": max(0, limit - current)
                if limit is not None and current is not None else None,
            })
    limits = [item["limit_bytes"] for item in readings if item["limit_bytes"] is not None]
    remaining = [item["remaining_bytes"] for item in readings if item["remaining_bytes"] is not None]
    return {
        "status": "observed" if readings else "unavailable",
        "membership": membership, "readings": readings,
        "effective_limit_bytes": min(limits) if limits else None,
        "effective_remaining_bytes": min(remaining) if remaining else None,
        "note": "Observed namespace/current-group limits; not a RAM reservation.",
    }


def system_inventory(root: Path) -> dict:
    memory = {}
    for line in (read_text(Path("/proc/meminfo")) or "").splitlines():
        parts = line.replace(":", "").split()
        if len(parts) >= 2 and parts[0] in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
            memory[parts[0] + "_bytes"] = int(parts[1]) * 1024
    cgroup = cgroup_memory()
    available = [value for value in (
        memory.get("MemAvailable_bytes"), cgroup["effective_remaining_bytes"]
    ) if value is not None]
    disk = shutil.disk_usage(root)
    memlock = None
    try:
        import resource  # Standard library, unavailable on some platforms.
        soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        memlock = {"soft_bytes": soft if soft != resource.RLIM_INFINITY else "unlimited",
                   "hard_bytes": hard if hard != resource.RLIM_INFINITY else "unlimited"}
    except (ImportError, AttributeError, OSError, ValueError):
        pass
    return {
        "hostname": socket.gethostname(), "platform": platform.platform(),
        "python": sys.version, "executable": sys.executable, "pid": os.getpid(),
        "cpu_count": os.cpu_count(),
        "cpu_affinity_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "host_memory": memory, "cgroup_memory": cgroup,
        "effective_available_ram_bytes": min(available) if available else None,
        "memlock": memlock,
        "disk": {"path": str(root), "total_bytes": disk.total,
                 "used_bytes": disk.used, "free_bytes": disk.free},
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def package_inventory() -> dict:
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = {"status": "installed", "version": importlib.metadata.version(package)}
        except importlib.metadata.PackageNotFoundError:
            versions[package] = {"status": "not_installed", "version": None}
    return versions


def probe_torch_cuda(_args) -> dict:
    import torch

    available = torch.cuda.is_available()
    devices = []
    if available:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            with torch.cuda.device(index):
                free, total = torch.cuda.mem_get_info()
                bf16 = torch.cuda.is_bf16_supported()
            devices.append({
                "index": index, "name": props.name,
                "total_memory_bytes": props.total_memory,
                "capability": [props.major, props.minor],
                "bf16_supported": bf16,
                "free_memory_bytes": free, "runtime_total_memory_bytes": total,
            })
    driver = None
    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True,
                timeout=20, check=False,
            )
            driver = {"returncode": result.returncode, "csv": result.stdout.strip(),
                      "stderr": result.stderr.strip()[:2000]}
        except (OSError, subprocess.TimeoutExpired) as exc:
            driver = {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "status": "passed" if available else "unavailable",
        "scope": "Torch import, CUDA visibility and device inventory only",
        "torch_version": torch.__version__, "torch_path": torch.__file__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": available, "visible_device_count": torch.cuda.device_count(),
        "devices": devices, "nvidia_smi": driver,
    }


def probe_data_stack(_args) -> dict:
    """Exercise the binary interfaces; installed version metadata is insufficient."""
    import numpy as np
    import pyarrow as pa
    from datasets import Dataset
    import torch

    dataset = Dataset.from_list([{"value": 1.5}, {"value": -2.25}, {"value": 0.0}])
    array = np.asarray(dataset.with_format("numpy")["value"], dtype=np.float32)
    tensor = torch.from_numpy(array)
    tensor.add_(0.5)
    roundtrip = tensor.numpy()
    expected = np.array([2.0, -1.75, 0.5], dtype=np.float32)
    values_match = bool(np.array_equal(roundtrip, expected))
    shared_mutation = bool(np.array_equal(array, expected))
    return {
        "status": "passed" if values_match and shared_mutation else "failed",
        "scope": "Real CPU Dataset/Arrow -> NumPy -> torch.from_numpy -> tensor.numpy interoperability",
        "versions": {"numpy": np.__version__, "pyarrow": pa.__version__,
                     "torch": str(torch.__version__)},
        "module_paths": {"numpy": np.__file__, "pyarrow": pa.__file__, "torch": torch.__file__},
        "dataset_rows": len(dataset), "arrow_schema": str(dataset.data.schema),
        "tensor_device": str(tensor.device), "tensor_dtype": str(tensor.dtype),
        "values_match": values_match, "shared_mutation_observed": shared_mutation,
        "roundtrip_values": roundtrip.tolist(),
        "gpu_exercised": False,
    }


def probe_kernels(args) -> dict:
    if not args.check_kernels:
        raise ValueError("Kernel imports require --check-kernels")
    imports = []
    for module_name, names in (
        ("fla.ops.gated_delta_rule", ("chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule")),
        ("causal_conv1d", ("causal_conv1d_fn", "causal_conv1d_update")),
    ):
        try:
            module = importlib.import_module(module_name)
            for name in names:
                if not callable(getattr(module, name)):
                    raise TypeError(f"{module_name}.{name} is not callable")
            imports.append({"module": module_name, "functions": list(names),
                            "status": "passed", "path": getattr(module, "__file__", None)})
        except Exception as exc:
            imports.append({"module": module_name, "functions": list(names), "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}"})
    return {
        "status": "passed" if all(item["status"] == "passed" for item in imports) else "failed",
        "scope": "Actual function imports only; does not prove CUDA kernels execute or are selected",
        "imports": imports,
    }


def probe_cpu_adam(args) -> dict:
    if not args.exercise_optimizer:
        raise ValueError("CPUAdam exercise requires --exercise-optimizer")
    import torch
    from deepspeed.ops.adam import DeepSpeedCPUAdam

    parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0], device="cpu", dtype=torch.float32))
    initial = parameter.detach().clone()
    optimizer = DeepSpeedCPUAdam([parameter], lr=0.01)
    loss = parameter.square().sum()
    loss.backward()
    optimizer.step()
    finite = bool(torch.isfinite(parameter).all())
    changed = not torch.equal(initial, parameter.detach())
    state = optimizer.state[parameter]
    result = {
        "status": "passed" if finite and changed else "failed",
        "scope": "One float32 CPU parameter, one DeepSpeedCPUAdam update; no ZeRO/distributed/offload training",
        "parameter_device": str(parameter.device), "loss": float(loss.detach()),
        "finite_parameters": finite, "parameter_changed": changed,
        "optimizer_state_keys": sorted(state),
        "optimizer_state_devices": {key: str(value.device) for key, value in state.items()
                                    if isinstance(value, torch.Tensor)},
    }
    del optimizer
    return result


def probe_tiny_qwen(args) -> dict:
    if not args.exercise_model:
        raise ValueError("Tiny CUDA model exercise requires --exercise-model")
    import torch
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    if not torch.cuda.is_available():
        return {"status": "unavailable", "reason": "torch.cuda.is_available() is false"}
    if not 0 <= args.device_index < torch.cuda.device_count():
        raise ValueError(f"CUDA device index out of range: {args.device_index}")
    torch.cuda.set_device(args.device_index)
    if not torch.cuda.is_bf16_supported():
        return {"status": "unavailable", "reason": "Selected CUDA device does not support bfloat16"}
    torch.manual_seed(17)
    config = Qwen3_5TextConfig(
        vocab_size=128, hidden_size=128, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=64, linear_conv_kernel_dim=4,
        linear_key_head_dim=64, linear_value_head_dim=64,
        linear_num_key_heads=2, linear_num_value_heads=2,
        layer_types=["linear_attention", "full_attention"],
        max_position_embeddings=128, pad_token_id=0, eos_token_id=2,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 0.5,
                         "mrope_section": [8, 4, 4], "mrope_interleaved": True},
        use_cache=False,
    )
    config._attn_implementation = "sdpa"
    device = torch.device("cuda", args.device_index)
    model = Qwen3_5ForCausalLM(config).to(device=device, dtype=torch.bfloat16)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    ids = torch.arange(3, 35, device=device).unsqueeze(0)
    torch.cuda.reset_peak_memory_stats(device)
    loss = model(input_ids=ids, labels=ids, use_cache=False).loss
    loss.backward()
    torch.cuda.synchronize(device)
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    finite = bool(torch.isfinite(loss)) and bool(gradients) and all(
        bool(torch.isfinite(gradient).all()) for gradient in gradients
    )
    nonzero = any(bool(torch.count_nonzero(gradient)) for gradient in gradients)
    return {
        "status": "passed" if finite and nonzero else "failed",
        "scope": "Random two-layer Qwen3.5 BF16 CUDA forward/backward with SDPA and gradient checkpointing",
        "model_class": type(model).__name__, "model_config": config.to_dict(),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device), "dtype": "bfloat16",
        "loss": float(loss.detach()) if bool(torch.isfinite(loss)) else None,
        "finite_loss_and_gradients": finite, "nonzero_gradients": nonzero,
        "parameters_with_gradients": len(gradients), "batch_size": 1, "sequence_length": ids.shape[1],
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "not_tested": ["published_9B_weights", "tokenizer", "generation_cache", "ZeRO",
                       "multi_GPU", "teacher", "optimizer_step", "checkpoint_resume"],
        "kernel_selection": "not_instrumented; inspect logs, import success does not establish kernel selection",
    }


PROBES = {"torch_cuda": probe_torch_cuda, "data_stack": probe_data_stack, "kernels": probe_kernels,
          "cpu_adam": probe_cpu_adam, "tiny_qwen_cuda": probe_tiny_qwen}


def write_json(path: Path, data: dict) -> None:
    # No external temporary directory or partially written final receipt.
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def stop_process(process) -> None:
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
        process.wait()


def run_probe(name: str, args, root: Path, receipt_dir: Path) -> dict:
    result_path = receipt_dir / (name + ".json")
    log_path = receipt_dir / (name + ".log")
    command = [sys.executable, "-B", str(SCRIPT_PATH), "--output-dir", str(root),
               "--internal-probe", name, "--probe-output", str(result_path),
               "--device-index", str(args.device_index)]
    for flag in ("check_kernels", "exercise_optimizer", "exercise_model"):
        if getattr(args, flag):
            command.append("--" + flag.replace("_", "-"))
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, cwd=root,
                                   start_new_session=os.name == "posix")
        try:
            returncode = process.wait(timeout=args.timeout_seconds)
        except subprocess.TimeoutExpired:
            stop_process(process)
            return {"status": "timed_out", "timeout_seconds": args.timeout_seconds,
                    "log": str(log_path), "elapsed_seconds": time.monotonic() - started}
        except BaseException:
            stop_process(process)
            raise
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        result = {"status": "failed", "error": f"No valid probe receipt: {exc}"}
    if returncode and result.get("status") == "passed":
        result["status"] = "failed"
        result["error"] = "Probe reported success but its process exited unsuccessfully"
    result.update(returncode=returncode, log=str(log_path), elapsed_seconds=time.monotonic() - started)
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--check-kernels", action="store_true")
    result.add_argument("--exercise-optimizer", action="store_true")
    result.add_argument("--exercise-model", action="store_true")
    result.add_argument("--device-index", type=int, default=0, help="Logical CUDA device for tiny model (default 0)")
    result.add_argument("--timeout-seconds", type=float, default=300, help="Per-probe timeout, including compilation")
    result.add_argument("--internal-probe", choices=tuple(PROBES), help=argparse.SUPPRESS)
    result.add_argument("--probe-output", help=argparse.SUPPRESS)
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0 or args.device_index < 0:
        raise ValueError("--timeout-seconds must be positive and --device-index nonnegative")
    root, configured = configure_output(args.output_dir)
    if args.internal_probe:
        if not args.probe_output:
            raise ValueError("Internal probe requires --probe-output")
        target = Path(args.probe_output).resolve()
        if not target.is_relative_to(root):
            raise ValueError("Probe receipt must be below --output-dir")
        try:
            result = PROBES[args.internal_probe](args)
        except Exception as exc:
            traceback.print_exc()
            result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        write_json(target, result)
        return 0 if result["status"] == "passed" else 1

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    receipt_dir = confined_path(root, f"receipts/{timestamp}-{os.getpid()}")
    receipt_dir.mkdir(parents=True, exist_ok=False)
    report_path = receipt_dir / "preflight.json"
    report = {
        "schema_version": 1, "created_at_utc": timestamp,
        "output_dir": str(root), "receipt_path": str(report_path),
        "system": system_inventory(root), "packages": package_inventory(),
        "configured_environment": configured,
        "scope": "Local environment checks, no downloads, no published model weights",
        "full_9b_training_readiness": "not_established",
        "distributed_training_tested": False, "probes": {},
    }
    requested = {"torch_cuda": True, "data_stack": True, "kernels": args.check_kernels,
                 "cpu_adam": args.exercise_optimizer, "tiny_qwen_cuda": args.exercise_model}
    for name, enabled in requested.items():
        if enabled:
            report["probes"][name] = run_probe(name, args, root, receipt_dir)
        else:
            report["probes"][name] = {"status": "not_requested"}
        write_json(report_path, report)
    report["failed_requested_probes"] = [
        name for name, enabled in requested.items()
        if enabled and report["probes"][name]["status"] != "passed"
    ]
    report["requested_checks_passed"] = not report["failed_requested_probes"]
    write_json(report_path, report)
    print(json.dumps({"receipt": str(report_path),
                      "requested_checks_passed": report["requested_checks_passed"],
                      "probes": {name: value["status"] for name, value in report["probes"].items()},
                      "full_9b_training_readiness": "not_established"}, ensure_ascii=False))
    return 0 if report["requested_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
