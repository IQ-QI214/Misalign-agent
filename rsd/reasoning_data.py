"""CPU-only, pinned GSM8K/ARC preparation; model inputs never contain gold.

Raw source caches and derived prompts/gold are private, ignored artifacts. The
public audit contains counts and hashes, not benchmark answers. No model runs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import unicodedata
from urllib.request import Request, urlopen

VERSION = "rsd-reasoning-data-v2"
SALT = "rsd-reasoning-v2-selection-20261106"
TEACHER_CUE = "Please use the information provided in the question and follow the requested answer format."
QUOTAS = {"train": {"gsm8k": 512, "arc_challenge": 512},
          "dev": {"gsm8k": 128, "arc_challenge": 128},
          "test": {"gsm8k": 256, "arc_challenge": 256}}
# Conservative topic exclusion, not a harmfulness classifier or human approval.
TOPIC_PATTERN = r"\b(?:weapons?|guns?|bombs?|explosives?|suicid\w*|poisons?|gambl\w*|cigarettes?|alcohol|drugs?|murders?|kills?|killing|firearms?|pistols?|rifles?|knives|knife)\b"
VISUAL_PATTERN = r"\b(?:diagram|figure|picture|graph|illustration|table (?:below|above)|shown (?:below|above)|(?:in|from|using) the (?:table|chart))\b"
POLICY = {"selection_salt": SALT, "quota_per_split": QUOTAS,
          "topic_screen_regex": TOPIC_PATTERN, "missing_visual_regex": VISUAL_PATTERN,
          "filter_applies_to": ["train", "dev", "test"],
          "deduplication": "Quarantine every row sharing a normalized question stem, across all native files, even if choices differ.",
          "near_duplicate_screen": "Quarantine both ends of question-stem token-set Jaccard >= 0.85, including the same native file; no semantic-independence claim.",
          "selection": "Hash rank per benchmark; GSM dev first then train from native train; ARC native splits preserved; final order hash-ranked.",
          "teacher_cue": TEACHER_CUE,
          "scoring_protocol": "final-answer-v1",
          "human_benign_review": "NOT_PERFORMED", "pretraining_contamination": "UNKNOWN"}


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def norm(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def parse_source(rows, benchmark, native_split):
    parsed = []
    for index, raw in enumerate(rows):
        question = raw.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Missing question")
        item = {"benchmark": benchmark, "native_split": native_split,
                "question": question, "source_id": "question-sha256:" + sha(norm(question).encode()),
                "choice_labels": []}
        if benchmark == "gsm8k":
            answer = raw.get("answer", "")
            if not isinstance(answer, str) or answer.count("####") != 1:
                raise ValueError("GSM reference requires exactly one final-answer delimiter")
            gold = answer.rsplit("####", 1)[1].strip()
            if not re.fullmatch(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", gold):
                raise ValueError("Unexpected GSM reference numeric format")
            Fraction(gold.replace(",", ""))
            native_id = f"{native_split}:{index:05d}"
            item.update(expected_answer=gold, answer_type="numeric", native_id=native_id,
                        visible_question=question)
        elif benchmark == "arc_challenge":
            choices = raw.get("choices", {})
            labels, texts = choices.get("label", []), choices.get("text", [])
            if (not 3 <= len(labels) <= 5 or len(labels) != len(texts)
                    or len(set(labels)) != len(labels)
                    or any(not isinstance(x, str) or not re.fullmatch(r"[A-E1-5]", x) for x in labels)
                    or any(not isinstance(t, str) or not t.strip() for t in texts)
                    or raw.get("answerKey") not in labels
                    or not isinstance(raw.get("id"), str) or not raw["id"]):
                raise ValueError("Invalid ARC identity, options, or answer key")
            item.update(expected_answer=raw["answerKey"], answer_type="choice", native_id=raw["id"],
                        choice_labels=labels,
                        visible_question=question + "\n\n" + "\n".join(f"{label}. {text}" for label, text in zip(labels, texts)))
        else:
            raise ValueError("Unsupported source")
        item["id"] = f"{benchmark}/{native_split}/{item['native_id']}"
        parsed.append(item)
    if len({r["id"] for r in parsed}) != len(parsed):
        raise ValueError("Duplicate native identity within source file")
    return parsed


def near_duplicate_ids(rows, threshold=0.85):
    """Exact token-set similarity screening through an inverted index.

    Include same-native-file pairs: GSM derived train and dev share one native
    pool. Lexical similarity is still not a semantic-equivalence oracle.
    """
    tokens = [set(re.findall(r"\w+", norm(r["question"]))) for r in rows]
    inverted = defaultdict(list)
    excluded, pairs = set(), []
    for i, terms in enumerate(tokens):
        candidates = set()
        for term in terms:
            candidates.update(inverted[term])
        for j in candidates:
            other = tokens[j]
            if not terms or min(len(terms), len(other)) / max(len(terms), len(other)) < threshold:
                continue
            score = len(terms & other) / len(terms | other)
            if score >= threshold:
                excluded.update((rows[i]["id"], rows[j]["id"]))
                pairs.append({"first": rows[j]["id"], "second": rows[i]["id"], "similarity": score})
        for term in terms:
            inverted[term].append(i)
    return excluded, sorted(pairs, key=lambda p: (p["first"], p["second"]))


def filter_candidates(rows):
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate combined row identity")
    stems = Counter(r["source_id"] for r in rows)
    native_ids = Counter((r["benchmark"], r["native_id"]) for r in rows)
    near_ids, pairs = near_duplicate_ids(rows)
    kept, excluded = [], []
    for row in rows:
        reasons = []
        if stems[row["source_id"]] > 1:
            reasons.append("duplicate_question_stem")
        if native_ids[row["benchmark"], row["native_id"]] > 1:
            reasons.append("duplicate_native_id")
        if row["id"] in near_ids:
            reasons.append("lexical_near_duplicate")
        if re.search(TOPIC_PATTERN, row["visible_question"], re.I):
            reasons.append("conservative_topic_screen")
        if re.search(VISUAL_PATTERN, row["visible_question"], re.I):
            reasons.append("possible_missing_visual")
        if reasons:
            excluded.append({"id": row["id"], "reasons": reasons})
        else:
            kept.append(row)
    return kept, excluded, pairs


def rank(row):
    return sha((SALT + "\n" + row["source_id"]).encode())


def select(rows, quotas=None):
    quotas = QUOTAS if quotas is None else quotas
    pools = defaultdict(list)
    for row in rows:
        pools[row["benchmark"], row["native_split"]].append(row)
    for pool in pools.values():
        pool.sort(key=rank)
    selected, used = {}, set()
    for split in ("dev", "test", "train"):
        picked = []
        for benchmark, amount in quotas[split].items():
            native_split = ("train" if split in ("train", "dev") else "test") if benchmark == "gsm8k" else ("validation" if split == "dev" else split)
            candidates = [r for r in pools[benchmark, native_split] if r["source_id"] not in used]
            if len(candidates) < amount:
                raise ValueError(f"Insufficient {benchmark}/{split}: {len(candidates)} < {amount}; no split fallback")
            for row in candidates[:amount]:
                picked.append({**row, "split": split})
                used.add(row["source_id"])
        selected[split] = sorted(picked, key=rank)
    return selected


def render(row):
    instruction = ("You may reason briefly. Put your final numeric answer on the last line in the form FINAL: <number>. Do not include units on that line."
                   if row["answer_type"] == "numeric" else
                   "You may reason briefly. Put the original label of your selected option on the last line in the form FINAL: <label>.")
    prompt = row["visible_question"] + "\n\n" + instruction
    visible = {k: row[k] for k in ("id", "source_id", "benchmark", "native_id", "native_split", "split")}
    visible.update(prompt=prompt, prompt_sha256=sha(prompt.encode()), family=row["benchmark"],
                   task_type="ordinary", answer_type=row["answer_type"], choice_labels=row["choice_labels"])
    if row["split"] == "train":
        visible["teacher_context"] = TEACHER_CUE
    gold = {k: visible[k] for k in ("id", "source_id", "benchmark", "split", "answer_type", "choice_labels", "prompt_sha256")}
    gold["expected_answer"] = row["expected_answer"]
    return visible, gold


def write_private(path, blob):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Preserve a different preparation rather than silently replacing it.
    if path.exists() and path.read_bytes() != blob:
        raise ValueError(f"Refusing to overwrite different data artifact: {path}")
    path.write_bytes(blob)
    path.chmod(0o600)
    return {"sha256": sha(blob), "bytes": len(blob)}


def jsonl(rows):
    return b"".join(canonical(row) + b"\n" for row in rows)


def prepare(lock, cache, output, *, download=False, benchmark_rows=()):
    import pyarrow.parquet as pq
    cache, output = Path(cache), Path(output)
    if lock.get("preparation_policy") != POLICY:
        raise ValueError("Lock policy differs from implemented preparation policy")
    cache.mkdir(parents=True, exist_ok=True)
    all_rows, source_receipts = [], []
    for source in lock["files"]:
        path = cache / source["file"]
        if not path.exists():
            if not download:
                raise ValueError(f"Missing source cache {path}; explicit --download required")
            req = Request(source["url"], headers={"User-Agent": VERSION})
            with urlopen(req, timeout=60) as response:
                blob = response.read()
            if sha(blob) != source["sha256"]:
                raise ValueError("Downloaded source SHA mismatch")
            write_private(path, blob)
        blob = path.read_bytes()
        if sha(blob) != source["sha256"] or len(blob) != source["bytes"]:
            raise ValueError(f"Source bytes differ from lock: {source['file']}")
        raw = pq.read_table(path).to_pylist()
        if len(raw) != source["rows"]:
            raise ValueError("Native row count mismatch")
        all_rows.extend(parse_source(raw, source["benchmark"], source["native_split"]))
        source_receipts.append({k: source[k] for k in ("file", "sha256", "rows", "bytes")})
    kept, excluded, near_pairs = filter_candidates(all_rows)
    selected = select(kept)
    benchmark_stems = {norm(row["prompt"]) for row in benchmark_rows}
    overlaps = [r["id"] for group in selected.values() for r in group
                if norm(r["question"]) in benchmark_stems or norm(r["visible_question"]) in benchmark_stems]
    if overlaps:
        raise ValueError("Selected task stem overlaps frozen safety benchmark")
    output.mkdir(parents=True, exist_ok=True)
    (output / "sealed").mkdir(exist_ok=True)
    (output / "sealed").chmod(0o700)
    files, split_audit = {}, {}
    for split in ("train", "dev", "test"):
        converted = [render(r) for r in selected[split]]
        prefix = "sealed/" if split == "test" else ""
        name = prefix + split + ".prompts.jsonl"
        files[name] = write_private(output / name, jsonl([r[0] for r in converted]))
        # Training gold and GSM solutions are never copied to derived training data.
        if split != "train":
            name = prefix + split + ".gold.jsonl"
            files[name] = write_private(output / name, jsonl([r[1] for r in converted]))
        split_audit[split] = {"rows": len(converted), "unique_source_ids": len({r[0]["source_id"] for r in converted}),
                              "benchmarks": dict(Counter(r[0]["benchmark"] for r in converted)),
                              "native_splits": dict(Counter(r[0]["benchmark"] + "/" + r[0]["native_split"] for r in converted)),
                              "ordered_id_sha256": sha(canonical([r[0]["id"] for r in converted]))}
    lineage = {"exclusions": excluded, "near_duplicate_pairs": near_pairs,
               "selected": {split: [{k: r[k] for k in ("id", "source_id", "native_split", "benchmark")} for r in rows]
                            for split, rows in selected.items()}}
    files["lineage.json"] = write_private(output / "lineage.json", canonical(lineage) + b"\n")
    audit = {"version": VERSION, "state": "DATA_PREPARED_NO_MODEL_EXECUTION",
             "implementation_sha256": sha(Path(__file__).read_bytes()),
             "source_lock_sha256": sha(canonical(lock)), "policy": POLICY, "source_files": source_receipts,
             "raw_rows": len(all_rows), "eligible_rows": len(kept), "excluded_rows": len(excluded),
             "exclusion_reasons_nonexclusive": dict(Counter(reason for r in excluded for reason in r["reasons"])),
             "near_duplicate_pairs": len(near_pairs), "splits": split_audit, "artifacts": files,
             "checks": {"selected_source_ids_unique_across_splits": True,
                        "selected_question_stems_unique_across_splits": True,
                        "training_file_contains_no_gold_or_solution_fields": True,
                        "explicit_neutral_teacher_cue_on_every_train_row": True,
                        "safety_benchmark_rows_checked": len(benchmark_rows),
                        "safety_benchmark_canonical_sha256": sha(canonical(list(benchmark_rows))),
                        "safety_exact_stem_overlap": len(overlaps)},
             "limitations": ["No human harmlessness certification; conservative lexical screen only.",
                             "Exact stem and specified near-duplicate rules do not prove semantic independence.",
                             "Paraphrases below the lexical threshold may remain, including within GSM native train.",
                             "Public benchmark pretraining contamination unknown.",
                             "This filtered balanced subset is not an official benchmark result.",
                             "sealed/ is an operational separation, not cryptographic access control.",
                             "No model performance or output-length distribution measured."]}
    write_private(output / "manifest.json", canonical(audit) + b"\n")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", default="configs/rsd/reasoning_sources_v2.lock.json")
    parser.add_argument("--source-cache", default=".runtime/reasoning_v2_sources")
    parser.add_argument("--output", default="data/rsd_reasoning_v2")
    parser.add_argument("--audit", default="validation/reasoning_data_v2.json")
    parser.add_argument("--benchmark-dataset", action="append", default=[])
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    benchmarks = []
    for name in args.benchmark_dataset:
        benchmarks.extend(json.loads(line) for line in Path(name).read_text().splitlines() if line.strip())
    result = prepare(json.loads(Path(args.lock).read_text()), args.source_cache, args.output,
                     download=args.download, benchmark_rows=benchmarks)
    audit = Path(args.audit)
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"state": result["state"], "splits": result["splits"], "audit": str(audit)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
