"""Assemble reviewed/observed A/B/C materials without generating or labeling them.

The current real panel is A64 only. Pending B attempts and unadmitted C sources
remain visible as coverage gaps. A full panel needs the declared per-stratum
metric/source quotas; merely reaching 200 rows cannot release it.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from . import judge_validation as jv
from .evaluation import text_digest

VERSION = "judge-validation-panel-v1"
TARGETS = {"A": {"overrefusal": 48, "safety": 16, "sources": 32},
           "B": {"overrefusal": 48, "safety": 48, "sources": 48},
           "C": {"overrefusal": 20, "safety": 20, "sources": 40}}


def _plain_path(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Panel evidence must not traverse symlinks")
    return path


def coverage(rows):
    indexed, sources = {}, defaultdict(list)
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get("id"), str)
                or not row["id"] or row["id"] in indexed
                or not isinstance(row.get("source_id"), str) or not row["source_id"]
                or not isinstance(row.get("origin"), str) or not row["origin"]
                or type(row.get("model_output")) is not bool
                or row.get("stratum") not in TARGETS
                or row.get("metric") not in {"overrefusal", "safety"}):
            raise ValueError("Invalid/duplicate panel case identity")
        for key in ("prompt", "response"):
            if (not isinstance(row.get(key), str) or not row[key].strip()
                    or row.get(key + "_sha256") != text_digest(row[key])):
                raise ValueError("Panel text/hash binding differs")
        if row.get("human_gold") is not None or row.get("human_gold_status") != "UNKNOWN":
            raise ValueError("Candidate materials cannot contain human gold")
        indexed[row["id"]] = row
        sources[row["source_id"]].append(row)
    representatives = []
    for group in sources.values():
        first = group[0]
        if any(any(r[k] != first[k] for k in ("stratum", "metric", "prompt_sha256")) for r in group):
            raise ValueError("Source cluster spans different strata/metrics/prompts")
        expected = 1 if first["stratum"] == "C" else 2
        if len(group) > expected or (first["stratum"] == "A" and len(group) != 2):
            raise ValueError("Source has unexpected answer multiplicity")
        if first["stratum"] == "A" and {r.get("variant") for r in group} != {"original", "evaluator_directive"}:
            raise ValueError("A source must retain both original and directive variant")
        for previous in representatives:
            if (jv._normalized(first["prompt"]) == jv._normalized(previous["prompt"])
                    or jv._similar(first["prompt"], previous["prompt"])):
                raise ValueError("Panel has exact/near duplicate questions across distinct sources")
        representatives.append(first)
    result = {}
    for layer, target in TARGETS.items():
        subset = [r for r in rows if r["stratum"] == layer]
        counts = Counter(r["metric"] for r in subset)
        actual = {m: counts[m] for m in ("overrefusal", "safety")}
        actual["sources"] = len({r["source_id"] for r in subset})
        result[layer] = {"target": target, "actual": actual,
                         "missing": {k: max(0, target[k] - actual[k]) for k in target},
                         "excess": {k: max(0, actual[k] - target[k]) for k in target},
                         "exact_coverage": actual == target}
    return result


def _source(path):
    path = _plain_path(path)
    return {"path": str(path), "manifest_sha256": jv._file_sha(path / "manifest.json")}


def _collect(sources):
    for entry in sources.values():
        path = _plain_path(entry["path"])
        if jv._file_sha(path / "manifest.json") != entry["manifest_sha256"]:
            raise ValueError("Source bundle manifest changed")
    raw_a = json.loads((Path(sources["A"]["path"]) / "manifest.json").read_text())
    if raw_a.get("version") != jv.VERSION:
        raise ValueError("A must reference an original material bundle, not another assembled panel")
    am, authored = jv.load_bundle(sources["A"]["path"])
    if am.get("version") != jv.VERSION or len(authored) != 64 or any(r.get("stratum") != "A" for r in authored):
        raise ValueError("A must be the original 64-case material bundle")
    rows = list(authored)
    pending = {"B": {"planned_slots": 96, "prepared_answers": 0, "remaining_slots": 96},
               "C": {"target_answers": 40, "admitted_answers": 0}}
    if "B" in sources:
        from .judge_real_data import load_candidates
        bm, answers = load_candidates(sources["B"]["path"])
        if bm.get("purpose") != "research":
            raise ValueError("CPU fixture generations cannot supply a research panel")
        if bm.get("source_validation_bundle_manifest_sha256") != sources["A"]["manifest_sha256"]:
            raise ValueError("B answers belong to a different original A/B selection")
        slots = json.loads((Path(sources["A"]["path"]) / "real_dev_generation_slots.json").read_text())
        expected = {r["id"]: r for r in slots}
        for r in answers:
            original = expected.get(r["id"])
            if original is None or any(r.get(k) != original[k] for k in ("source_id", "prompt", "prompt_sha256", "metric")):
                raise ValueError("B answer does not match the selected slot")
        rows.extend(answers)
        pending["B"].update(prepared_answers=len(answers), remaining_slots=96-len(answers))
    if "C" in sources:
        from .judge_reference_admission import load_admission
        cm, decisions, answers = load_admission(sources["C"]["path"])
        if any(r.get("accepted_for_validation") is not True for r in answers):
            raise ValueError("C response has not passed prompt eligibility admission")
        rows.extend(answers)
        pending["C"].update(admitted_answers=len(answers), candidate_decisions=len(decisions))
    return rows, pending


def assemble(authored_bundle, output_dir, *, b_bundle=None, c_admission=None):
    sources = {"A": _source(authored_bundle)}
    if b_bundle is not None:
        sources["B"] = _source(b_bundle)
    if c_admission is not None:
        sources["C"] = _source(c_admission)
    rows, pending = _collect(sources)
    counts = coverage(rows)
    complete = all(item["exact_coverage"] for item in counts.values())
    output = _plain_path(output_dir)
    if output.exists():
        raise FileExistsError("Panel output must be a new evidence directory")
    output.mkdir(parents=True, mode=0o700)
    payloads = {"candidates.private.json": rows,
                "reviewer_1.json": jv.blind_sheet(rows, 1), "reviewer_2.json": jv.blind_sheet(rows, 2)}
    for name, value in payloads.items():
        jv._write(output / name, value)
    manifest = {"version": VERSION,
                "status": "MATERIALS_COMPLETE_UNLABELED" if complete else "PARTIAL_MATERIALS_UNLABELED",
                "planned_records": 200, "prepared_answer_records": len(rows),
                "prepared_source_clusters": len({r["source_id"] for r in rows}),
                "prepared_human_gold_records": 0, "coverage": counts,
                "full_panel_materials_complete": complete, "pending": pending,
                "source_bundles": sources,
                "files_sha256": {name: jv._file_sha(output / name) for name in payloads},
                "implementation_sha256": jv._file_sha(__file__),
                "model_calls_executed": 0, "judge_batch_ready": False,
                "scientific_conclusion": None}
    jv._write(output / "manifest.json", manifest)
    return manifest


def load_panel(bundle_dir):
    root = _plain_path(bundle_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("version") != VERSION:
        raise ValueError("Unknown assembled panel version")
    expected_files = {"candidates.private.json", "reviewer_1.json", "reviewer_2.json"}
    if set(manifest.get("files_sha256", {})) != expected_files:
        raise ValueError("Incomplete assembled panel artifacts")
    for name, expected in manifest["files_sha256"].items():
        path = _plain_path(root / name)
        if jv._file_sha(path) != expected:
            raise ValueError("Assembled panel artifact changed")
    if not {"A"} <= set(manifest.get("source_bundles", {})) <= {"A", "B", "C"}:
        raise ValueError("Invalid panel source inventory")
    rows, pending = _collect(manifest["source_bundles"])
    stored = json.loads((root / "candidates.private.json").read_text())
    counts = coverage(rows)
    complete = all(x["exact_coverage"] for x in counts.values())
    for slot in (1, 2):
        if json.loads((root / f"reviewer_{slot}.json").read_text()) != jv.blind_sheet(rows, slot):
            raise ValueError("Frozen blank review projection differs; save reviews as new submissions")
    if (stored != rows or manifest.get("coverage") != counts or manifest.get("pending") != pending
            or manifest.get("planned_records") != 200 or manifest.get("prepared_answer_records") != len(rows)
            or manifest.get("prepared_source_clusters") != len({r["source_id"] for r in rows})
            or manifest.get("prepared_human_gold_records") != 0
            or manifest.get("full_panel_materials_complete") is not complete
            or manifest.get("status") != ("MATERIALS_COMPLETE_UNLABELED" if complete else "PARTIAL_MATERIALS_UNLABELED")
            or manifest.get("judge_batch_ready") is not False
            or type(manifest.get("model_calls_executed")) is not int or manifest["model_calls_executed"] != 0
            or manifest.get("scientific_conclusion") is not None):
        raise ValueError("Panel completeness/source binding mismatch")
    return manifest, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authored-bundle", required=True)
    parser.add_argument("--b-bundle")
    parser.add_argument("--c-admission")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = assemble(args.authored_bundle, args.output_dir, b_bundle=args.b_bundle, c_admission=args.c_admission)
    print(json.dumps({k: result[k] for k in ("status", "prepared_answer_records", "coverage", "pending")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
