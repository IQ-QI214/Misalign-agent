"""Make an explicit, answer-only synthetic reference fixture for the DCE probe.

This is NOT OpenThoughts or a reproduction of DCE's gold reasoning solutions.
Production verified-solution JSONL can use the same schema without this helper.
"""
import argparse
import hashlib
import json
from pathlib import Path


def prepare(source, output):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    rows = [json.loads(x) for x in source.read_text().splitlines() if x.strip()]
    if not rows:
        raise ValueError("Empty reference source")
    for row in rows:
        if row.get("split") != "train" or row.get("metric") != "exact_match" or "expected_answer" not in row:
            raise ValueError("Only labelled synthetic training tasks can become this fixture")
        if not row.get("feedback_evidence"):
            raise ValueError("A verified synthetic rule source is required")
        row["verified_solution"] = f"The independently computed correct final answer is {row['expected_answer']}."
        row["reference_provenance"] = "synthetic_pilot_answer_only_v1; not a full reference rationale"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
    manifest = {"kind": "synthetic_reference_fixture", "n_records": len(rows),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "scientific_scope": "mechanics only; not DCE benchmark reproduction"}
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/rsd_pilot/train.jsonl")
    parser.add_argument("--output", default="data/rsd_verified/train.jsonl")
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), indent=2))
