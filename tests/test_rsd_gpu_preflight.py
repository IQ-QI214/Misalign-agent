"""Preflight checks; optional libraries stay in child processes, no GPU is used."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import pytest

from scripts import rsd_gpu_preflight as preflight


@pytest.fixture
def isolated_environment(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(os, "environ", os.environ.copy())
    monkeypatch.setattr(tempfile, "tempdir", None)


def test_help_is_stdlib_only_and_does_not_create_output(tmp_path):
    output = tmp_path / "unused"
    result = subprocess.run(
        [sys.executable, "-B", "-S", str(preflight.SCRIPT_PATH),
         "--output-dir", str(output), "--help"],
        capture_output=True, text=True, timeout=10, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--exercise-model" in result.stdout
    assert not output.exists()


def test_cache_routing_and_escape_rejection(isolated_environment, tmp_path):
    root, configured = preflight.configure_output(tmp_path / "output")
    assert Path.cwd() == root
    assert Path(tempfile.gettempdir()).is_relative_to(root)
    for name in preflight.CACHE_PATHS:
        assert Path(configured[name]).is_relative_to(root)
        assert Path(configured[name]).is_dir()
    assert configured["HF_HUB_OFFLINE"] == "1"
    assert "HOME" not in configured
    escape_root = tmp_path / "escape"
    escape_root.mkdir()
    (escape_root / "cache").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        preflight.configure_output(escape_root)
    assert not (tmp_path / "torch_extensions").exists()


def test_cgroup_hierarchy_and_unlimited_v1(tmp_path):
    proc = tmp_path / "proc"
    cgroup = tmp_path / "cgroup"
    (proc / "self").mkdir(parents=True)
    (proc / "self/cgroup").write_text("0::/team/job\n3:memory:/team/job\n")
    (cgroup / "team/job").mkdir(parents=True)
    (cgroup / "memory/team/job").mkdir(parents=True)
    (cgroup / "memory/memory.limit_in_bytes").write_text("9223372036854771712")
    (cgroup / "memory/memory.usage_in_bytes").write_text("500")
    for relative, limit, current in (("", "max", "900"), ("team", "1000", "750"),
                                     ("team/job", "800", "300")):
        directory = cgroup / relative
        (directory / "memory.max").write_text(limit)
        (directory / "memory.current").write_text(current)
    info = preflight.cgroup_memory(proc, cgroup)
    assert info["effective_limit_bytes"] == 800
    assert info["effective_remaining_bytes"] == 250  # Parent group is tighter.
    assert len(info["readings"]) == 4
    assert preflight.cgroup_memory(tmp_path / "missing", tmp_path / "absent")["status"] == "unavailable"


def test_exercises_require_explicit_flags_before_imports():
    args = SimpleNamespace(check_kernels=False, exercise_optimizer=False, exercise_model=False)
    for probe, flag in ((preflight.probe_kernels, "--check-kernels"),
                        (preflight.probe_cpu_adam, "--exercise-optimizer"),
                        (preflight.probe_tiny_qwen, "--exercise-model")):
        with pytest.raises(ValueError, match=flag):
            probe(args)


def test_child_failure_has_json_and_log_without_exercising_optimizer(
    isolated_environment, tmp_path,
):
    root, _ = preflight.configure_output(tmp_path / "output")
    receipt_dir = root / "receipts"
    receipt_dir.mkdir()
    args = SimpleNamespace(check_kernels=False, exercise_optimizer=False, exercise_model=False,
                           device_index=0, timeout_seconds=10)
    result = preflight.run_probe("cpu_adam", args, root, receipt_dir)
    assert result["status"] == "failed"
    assert result["returncode"] == 1
    assert "requires --exercise-optimizer" in result["error"]
    assert Path(result["log"]).is_relative_to(root)
    assert "requires --exercise-optimizer" in Path(result["log"]).read_text()


def test_real_data_stack_roundtrip_runs_on_cpu_in_child(isolated_environment, tmp_path):
    root, _ = preflight.configure_output(tmp_path / "output")
    receipt_dir = root / "receipts"
    receipt_dir.mkdir()
    args = SimpleNamespace(check_kernels=False, exercise_optimizer=False, exercise_model=False,
                           device_index=0, timeout_seconds=60)
    result = preflight.run_probe("data_stack", args, root, receipt_dir)
    assert result["status"] == "passed", result
    assert result["returncode"] == 0
    assert result["tensor_device"] == "cpu"
    assert result["values_match"] and result["shared_mutation_observed"]
    assert result["roundtrip_values"] == [2.0, -1.75, 0.5]
    assert result["gpu_exercised"] is False


def test_data_stack_import_failure_isolated_and_preserved(tmp_path):
    fake_modules = tmp_path / "fake_modules"
    fake_modules.mkdir()
    (fake_modules / "numpy.py").write_text('raise RuntimeError("synthetic NumPy ABI failure")\n')
    output = tmp_path / "output"
    receipt = output / "data_stack.json"
    original_numpy = sys.modules.get("numpy")
    result = subprocess.run(
        [sys.executable, "-B", str(preflight.SCRIPT_PATH), "--output-dir", str(output),
         "--internal-probe", "data_stack", "--probe-output", str(receipt)],
        env={**os.environ, "PYTHONPATH": str(fake_modules)},
        capture_output=True, text=True, timeout=10, check=False,
    )
    assert result.returncode == 1
    report = json.loads(receipt.read_text())
    assert report["status"] == "failed"
    assert report["error"] == "RuntimeError: synthetic NumPy ABI failure"
    assert sys.modules.get("numpy") is original_numpy


def test_receipt_keeps_requested_failure_and_unrequested_checks_separate(
    isolated_environment, monkeypatch, tmp_path,
):
    calls = []

    def fake_probe(name, args, root, receipt_dir):
        calls.append(name)
        return {"status": "failed" if name == "kernels" else "passed"}

    monkeypatch.setattr(preflight, "run_probe", fake_probe)
    monkeypatch.setattr(preflight, "package_inventory", lambda: {})
    result = preflight.main(["--output-dir", str(tmp_path / "output"), "--check-kernels"])
    assert result == 1
    assert calls == ["torch_cuda", "data_stack", "kernels"]
    report_path = next((tmp_path / "output/receipts").glob("*/preflight.json"))
    report = json.loads(report_path.read_text())
    assert report["failed_requested_probes"] == ["kernels"]
    assert report["probes"]["cpu_adam"]["status"] == "not_requested"
    assert report["probes"]["tiny_qwen_cuda"]["status"] == "not_requested"
    assert report["full_9b_training_readiness"] == "not_established"
    assert report["distributed_training_tested"] is False
