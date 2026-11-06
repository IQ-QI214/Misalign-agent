"""Import the pinned, unmodified OPSD trainer, keeping its Apache header intact."""

from pathlib import Path
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "references" / "opsd_upstream"
COMMIT = "ae7d2519e94920c4eb6206c0c26de46d9c50abae"


def verify_upstream():
    lock = json.loads((ROOT / "configs/rsd/upstream.lock.json").read_text())
    for filename, digest in lock["sha256"].items():
        path = UPSTREAM / filename
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Pinned OPSD source missing or modified: {path}")
    if (UPSTREAM / ".git").exists():
        head = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
        if head != COMMIT:
            raise RuntimeError(f"Unexpected upstream revision: {head}")
    return lock


def import_trainer():
    verify_upstream()
    sys.path.insert(0, str(UPSTREAM))
    from opsd_trainer import OPSDTrainer
    return OPSDTrainer
