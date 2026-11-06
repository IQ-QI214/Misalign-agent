"""Prepare evaluation-only benchmark prompts from pinned official sources.

No model, judge, training, or credentials are used. SORRY-Bench is access-gated:
import an authorized local question.jsonl with an explicit Hub revision and hash.
The deterministic dev/test partition is ours, not any paper's original split.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import ssl
import unicodedata
from urllib.request import Request, urlopen


VERSION = "rsd-benchmarks-v1"
SPLIT_SALT = "rsd-benchmarks-prompt-split-v1"
SOURCES = {
    "xstest_safe": {
        "repository": "https://github.com/paul-rottger/xstest",
        "revision": "d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d",
        "file": "xstest_prompts.csv",
        "sha256": "11783fb294ed017473ee53c207d71f2161c7672c8d0b037501e78387f801cb5a",
        "raw_rows": 450, "expected_rows": 250,
        "filter": "Keep rows whose type does not start with contrast_; exclude all 200 unsafe contrast rows.",
        "license_note": "Prompts: CC-BY-4.0; see the pinned official readme.",
    },
    "strongreject": {
        "repository": "https://github.com/alexandrasouly/strongreject",
        "revision": "f7cad6c17e624e21d8df2278e918ae1dddb4cb56",
        "file": "strongreject_dataset/strongreject_dataset.csv",
        "sha256": "4dd70357e4ff8b5d0ba5ebafecab5d6dd5633ce8046e3dd1c8bd93e64de44381",
        "raw_rows": 313, "expected_rows": 313, "filter": "All full-dataset rows; no small-subset substitution.",
        "official_loader": "https://github.com/dsbowen/strong_reject/blob/7a551d5b440ec7b75d4f6f5bb7c1719965b76b47/strong_reject/load_datasets.py",
        "license_note": "Original authored data/code MIT; prompts inherited from prior work retain original licenses.",
    },
}
for _source in SOURCES.values():
    _repo = _source["repository"].removeprefix("https://github.com/")
    _source["url"] = f"https://raw.githubusercontent.com/{_repo}/{_source['revision']}/{_source['file']}"


def sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def normalise_prompt(prompt: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", prompt).split()).casefold()


def prompt_source(prompt: str) -> str:
    return "prompt-sha256:" + sha256(normalise_prompt(prompt).encode("utf-8"))


def split_for_prompt(prompt: str) -> str:
    digest = sha256((SPLIT_SALT + "\n" + normalise_prompt(prompt)).encode("utf-8"))
    return "dev" if int(digest[:16], 16) % 5 == 0 else "test"


def _row(benchmark, native_id, category, prompt, metric):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"{benchmark}: empty/non-string prompt")
    if native_id is None or not str(native_id).strip():
        raise ValueError(f"{benchmark}: missing native identity")
    if not isinstance(category, str) or not category.strip():
        raise ValueError(f"{benchmark}: missing category")
    return {"id": f"{benchmark}/{native_id}", "source_id": prompt_source(prompt),
            "family": f"{benchmark}/{category}", "split": split_for_prompt(prompt),
            "prompt": prompt, "metric": metric, "benchmark": benchmark,
            "native_id": str(native_id), "native_category": category,
            "evaluation_only": True}


def parse_xstest(blob, expected_total=450, expected_kept=250):
    rows = list(csv.DictReader(io.StringIO(blob.decode("utf-8-sig"))))
    if len(rows) != expected_total:
        raise ValueError(f"XSTest raw count {len(rows)} != {expected_total}")
    result = []
    for row in rows:
        if not row.get("type"):
            raise ValueError("XSTest missing type")
        if not row["type"].startswith("contrast_"):
            result.append(_row("xstest_safe", row["id"], row["type"], row["prompt"], "overrefusal"))
    if len(result) != expected_kept:
        raise ValueError(f"XSTest safe count {len(result)} != {expected_kept}")
    return result


def parse_strongreject(blob, expected_total=313):
    rows = list(csv.DictReader(io.StringIO(blob.decode("utf-8-sig"))))
    if len(rows) != expected_total:
        raise ValueError(f"StrongREJECT count {len(rows)} != {expected_total}")
    result = []
    for number, row in enumerate(rows, 1):
        item = _row("strongreject", f"row-{number:04d}", row["category"], row["forbidden_prompt"], "safety")
        item["native_dataset_source"] = row["source"]
        result.append(item)
    return result


def parse_sorry(blob, expected_total=440):
    rows = [json.loads(line) for line in blob.decode("utf-8-sig").splitlines() if line.strip()]
    if len(rows) != expected_total:
        raise ValueError(f"SORRY-Bench count {len(rows)} != {expected_total}")
    result = []
    for row in rows:
        turns = row.get("turns")
        if not isinstance(turns, list) or len(turns) != 1:
            raise ValueError("SORRY-Bench base file requires exactly one turn per question")
        result.append(_row("sorry_bench", row["question_id"], str(row["category"]), turns[0], "safety"))
    return result


def validate_rows(rows, training_rows=()):
    """Keep duplicate benchmark prompts together; reject train/eval overlap."""
    ids, ownership = set(), {"prompt": {}, "source_id": {}}
    for row in rows:
        for key in ("id", "source_id", "family", "prompt"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Invalid evaluation field: {key}")
        if row["id"] in ids:
            raise ValueError(f"Duplicate benchmark id: {row['id']}")
        ids.add(row["id"])
        if row.get("split") not in ("dev", "test") or row.get("metric") not in ("safety", "overrefusal"):
            raise ValueError("Benchmarks are evaluation-only dev/test safety or overrefusal data")
        for key, value in (("prompt", normalise_prompt(row["prompt"])), ("source_id", row["source_id"])):
            previous = ownership[key].setdefault(value, row["split"])
            if previous != row["split"]:
                raise ValueError(f"Cross-split {key} overlap")
    for row in training_rows:
        prompt = row.get("prompt")
        if prompt is None and isinstance(row.get("turns"), list) and len(row["turns"]) == 1:
            prompt = row["turns"][0]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Training overlap check requires prompt or a single turns entry")
        if normalise_prompt(prompt) in ownership["prompt"] or row.get("source_id") in ownership["source_id"]:
            raise ValueError("Training/evaluation overlap detected")
    return {"rows": len(rows), "unique_prompts": len(ownership["prompt"]),
            "duplicate_prompt_rows": len(rows) - len(ownership["prompt"]),
            "split_counts": dict(Counter(row["split"] for row in rows))}


def fetch_pinned(source):
    request = Request(source["url"], headers={"User-Agent": "Safe-Agent-RSD-benchmark-preparer/1"})
    context = ssl.create_default_context()
    # Python.org macOS installs may lack a populated system CA file. Add the
    # installed certifi roots when available; certificate checks stay enabled.
    try:
        import certifi
    except ImportError:
        pass
    else:
        context.load_verify_locations(cafile=certifi.where())
    with urlopen(request, timeout=30, context=context) as response:
        blob = response.read(5_000_001)
    if len(blob) > 5_000_000:
        raise ValueError("Benchmark download exceeds the expected small-file limit")
    if sha256(blob) != source["sha256"]:
        raise ValueError(f"Official raw file SHA256 mismatch: {source['url']}")
    return blob


def _jsonl(rows):
    return ("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)).encode("utf-8")


def prepare(output_dir, train_files=(), sorry_file=None, sorry_revision=None, sorry_sha256=None):
    datasets, raw, sources = {}, {}, {}
    for name, parser in (("xstest_safe", parse_xstest), ("strongreject", parse_strongreject)):
        source = SOURCES[name]
        blob = fetch_pinned(source)
        raw[name] = blob
        datasets[name] = parser(blob)
        sources[name] = {**source, "status": "prepared", "download_bytes": len(blob),
                         "counts": validate_rows(datasets[name])}
    if sorry_file is not None:
        if not isinstance(sorry_revision, str) or not re.fullmatch(r"[0-9a-f]{40}", sorry_revision):
            raise ValueError("SORRY local import requires its 40-character official Hub commit")
        if not isinstance(sorry_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", sorry_sha256):
            raise ValueError("SORRY local import requires an explicit expected SHA256")
        blob = Path(sorry_file).read_bytes()
        if sha256(blob) != sorry_sha256:
            raise ValueError("SORRY local file SHA256 mismatch")
        raw["sorry_bench"] = blob
        datasets["sorry_bench"] = parse_sorry(blob)
        sources["sorry_bench"] = {
            "status": "prepared", "repository": "https://huggingface.co/datasets/sorry-bench/sorry-bench-202503",
            "url": f"https://huggingface.co/datasets/sorry-bench/sorry-bench-202503/resolve/{sorry_revision}/question.jsonl",
            "revision": sorry_revision, "sha256": sorry_sha256, "expected_rows": 440,
            "filter": "All 440 base question.jsonl rows; no linguistic mutations.",
            "provenance": "Caller-provided authorized local file and revision; remote contents not independently fetched.",
            "counts": validate_rows(datasets["sorry_bench"]),
        }
    else:
        if sorry_revision is not None or sorry_sha256 is not None:
            raise ValueError("SORRY revision/hash provided without --sorry-file")
        sources["sorry_bench"] = {
            "status": "missing_gated_source", "expected_rows": 440, "prepared_rows": 0,
            "repository": "https://huggingface.co/datasets/sorry-bench/sorry-bench-202503",
            "note": "Official source is gated. Supply an authorized local question.jsonl, revision and expected SHA256. No unofficial mirror is substituted.",
        }
    all_rows = [row for name in sorted(datasets) for row in datasets[name]]
    train_hashes, training_rows = {}, []
    for path in train_files:
        blob = Path(path).read_bytes()
        train_hashes[str(path)] = sha256(blob)
        training_rows.extend(json.loads(line) for line in blob.decode("utf-8").splitlines() if line.strip())
    counts = validate_rows(all_rows, training_rows)
    payloads = {f"{name}.jsonl": _jsonl(rows) for name, rows in datasets.items()}
    for split in ("dev", "test"):
        payloads[f"{split}.jsonl"] = _jsonl([row for row in all_rows if row["split"] == split])
    for name, blob in raw.items():
        suffix = "jsonl" if name == "sorry_bench" else "csv"
        payloads[f"raw/{name}.{suffix}"] = blob
    manifest = {
        "schema_version": VERSION, "status": "prepared" if sorry_file else "partial_sorry_missing",
        "evaluation_only": True, "sources": sources, "counts": counts,
        "split": {"origin": "our_deterministic_split_not_paper_split", "salt": SPLIT_SALT,
                  "rule": "SHA256(salt + newline + NFKC/whitespace/casefold prompt), first 16 hex modulo 5; 0=dev, others=test",
                  "source_grouping": "Identical normalized prompts share source_id and split across all benchmarks; semantic overlap beyond exact normalization is not established."},
        "training_overlap_check": {"status": "checked" if train_files else "not_requested",
                                   "files_sha256": train_hashes, "rows": len(training_rows)},
        "evaluator_note": "These are prompt datasets only. A custom safety judge is not the official SORRY/StrongREJECT score; record the chosen evaluator profile separately.",
        "files": {name: {"sha256": sha256(blob), "bytes": len(blob)} for name, blob in sorted(payloads.items())},
    }
    output = Path(output_dir)
    manifest_blob = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    payloads["manifest.json"] = manifest_blob
    # Do not silently mix a changed source/split with a prior preparation.
    for name, blob in payloads.items():
        target = output / name
        if target.exists() and target.read_bytes() != blob:
            raise ValueError(f"Output differs from existing preparation: {target}; use a new directory")
    output.mkdir(parents=True, exist_ok=True)
    for name, blob in payloads.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    command = subparsers.add_parser("prepare")
    command.add_argument("--output-dir", default="data/rsd_benchmarks")
    command.add_argument("--train-file", action="append", default=[])
    command.add_argument("--sorry-file")
    command.add_argument("--sorry-revision")
    command.add_argument("--sorry-sha256")
    args = parser.parse_args()
    manifest = prepare(args.output_dir, args.train_file, args.sorry_file, args.sorry_revision, args.sorry_sha256)
    print(json.dumps({"status": manifest["status"], "output_dir": args.output_dir,
                      "counts": manifest["counts"],
                      "sources": {k: v["status"] for k, v in manifest["sources"].items()},
                      "training_overlap_check": manifest["training_overlap_check"],
                      "gpu_used": False, "judge_called": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
