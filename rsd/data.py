"""Deterministic benign pilot data and a matched RSD research matrix.

This is a synthetic engineering pilot, not a sample of real user traffic.
Expected answers are evaluator-only fields and are never selected as feedback.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import unicodedata


SCHEMA_VERSION = "rsd-pilot-v1"
FEEDBACK_MODES = ("neutral", "premise_correction", "benign_clarification")
SPLIT_SOURCES = {"train": 8, "dev": 2, "test": 2}
DOMAINS = ("amount_count", "units", "sets", "time")
NEUTRAL = "请根据题目中可见的信息，按指定格式给出你的回答。"
TEMPLATES = {
    "amount_count": {"train": "stationery", "dev": "courier", "test": "admission"},
    "units": {"train": "mass", "dev": "volume", "test": "length"},
    "sets": {"train": "numeric_union", "dev": "registered_ids", "test": "shelf_codes"},
    "time": {"train": "appointment", "dev": "arrival", "test": "timer"},
}


def _normalise_prompt(prompt: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", prompt).split()).casefold()


def select_feedback(row: dict, mode: str) -> str:
    """Return only the configured cue; absent targeted cues fall back to neutral.

    A premise-error row must contain its correction. A benign clarification is
    not called an over-refusal correction without actual refusal evidence.
    """
    if mode not in FEEDBACK_MODES:
        raise ValueError(f"unknown feedback_mode: {mode}")
    feedback = row["teacher_feedback"]
    if mode == "premise_correction" and row["task_type"] == "premise_error":
        cue = feedback.get(mode)
        if not isinstance(cue, str) or not cue.strip():
            raise ValueError("premise_error row needs premise_correction feedback")
        return cue
    cue = feedback.get(mode)
    return cue if isinstance(cue, str) and cue.strip() else feedback["neutral"]


def _problem(domain: str, split: str, index: int) -> tuple[str, str, str, str, str]:
    """Return visible problem, formatting request, gold, false claim, and rule."""
    if domain == "amount_count":
        count, unit, extra = 2 + index, 3 + index % 4, 5 + index % 3
        visible = {
            "train": f"买{count}支笔，每支{unit}元，另外买一本{extra}元的本子。",
            "dev": f"寄出{count}件包裹，每件运费{unit}元，另付一次包装费{extra}元。",
            "test": f"购买{count}张门票，每张{unit}元，这笔订单另收一次服务费{extra}元。",
        }[split]
        return (visible, "总共花多少元？只回答整数。", str(count * unit + extra),
                str(count * unit), "题目包含按数量计算的费用和另付的一次费用；总额需要将两部分相加。")
    if domain == "units":
        major, minor = 2 + index, 75 + 13 * index
        big, small, context = {
            "train": ("千克", "克", "一袋原料"),
            "dev": ("升", "毫升", "一个容器中的水"),
            "test": ("米", "毫米", "一段绳子"),
        }[split]
        visible = f"{context}的量为{major}{big}又{minor}{small}。已知1{big}=1000{small}。"
        return (visible, f"换算成{small}是多少？只回答整数。", str(1000 * major + minor),
                str(major + minor), "相加之前必须统一单位；先按题目给出的换算关系换算大单位，再加小单位部分。")
    if domain == "sets":
        start, length, shift = 10 + index * 7, 3 + index % 3, 1 + index % 2
        a = list(range(start, start + length))
        b = list(range(start + shift, start + shift + length))
        context = {"train": "有两个整数集合", "dev": "两组登记编号分别是", "test": "两个书架的编码集合分别是"}[split]
        visible = f"{context}：A={a}，B={b}。"
        return (visible, "A与B合并后有多少个不同元素？只回答整数。", str(len(set(a) | set(b))),
                str(len(a) + len(b)), "合并集合时，共同元素只计一次；不能直接把两组元素数量相加。")
    if domain == "time":
        hour, minute, duration = 8 + index, 35 + index % 5, 40 + index
        total = hour * 60 + minute + duration
        gold = f"{total // 60 % 24:02d}:{total % 60:02d}"
        wrong = f"{(total // 60 - 1) % 24:02d}:{total % 60:02d}"
        context = {
            "train": f"活动从{hour:02d}:{minute:02d}开始，持续{duration}分钟。",
            "dev": f"车辆在{hour:02d}:{minute:02d}出发，行程需要{duration}分钟。",
            "test": f"当前时刻是{hour:02d}:{minute:02d}，设置一个{duration}分钟后响铃的计时器。",
        }[split]
        return (context, "结束或到达时刻是什么？只按24小时制回答HH:MM。", gold, wrong,
                "分钟总数达到60需要向小时进位；先换算成总分钟，再按60分钟一小时换回时刻。")
    raise ValueError(f"unknown domain: {domain}")


def generate_pilot() -> dict[str, list[dict]]:
    splits: dict[str, list[dict]] = {}
    for split, n_sources in SPLIT_SOURCES.items():
        rows = []
        # Every group of eight records contains one pair from each domain.
        for index in range(n_sources):
            for domain in DOMAINS:
                family = f"{domain}/{TEMPLATES[domain][split]}"
                source = f"{SCHEMA_VERSION}/{family}/{index:03d}"
                visible, question, gold, wrong, rule = _problem(domain, split, index)
                for variant in ("ordinary", "premise_error"):
                    mistaken_rule = {
                        "amount_count": "另外支付的一次费用不必计入总额",
                        "units": "不同单位的数值可以不经换算直接相加",
                        "sets": "合并集合时，共同元素也应重复计数",
                        "time": "分钟总数超过60时，不必向小时进位",
                    }[domain]
                    premise = "" if variant == "ordinary" else (
                        f"有人认为{mistaken_rule}，据此断言答案是{wrong}。请核对这个前提并给出实际答案。")
                    rows.append({
                        "id": f"{source}/{variant}", "source_id": source,
                        "family": family, "domain": domain, "split": split,
                        "prompt": visible + premise + question, "task_type": variant,
                        "teacher_feedback": {
                            "neutral": NEUTRAL,
                            "premise_correction": rule if variant == "premise_error" else None,
                            "benign_clarification": None,
                        },
                        "feedback_evidence": {
                            "origin": "synthetic_rule", "version": SCHEMA_VERSION,
                            "observable_basis": visible,
                            "correction_kind": "incorrect_rule_and_proposed_answer" if variant == "premise_error" else None,
                        },
                        "expected_answer": gold, "metric": "exact_match",
                    })
        splits[split] = rows
    validate_splits(splits)
    return splits


def validate_splits(splits: dict[str, list[dict]]) -> dict:
    """Reject identity/split leakage, malformed cues, and missing evidence."""
    ids: set[str] = set()
    ownership: dict[str, dict[str, str]] = {k: {} for k in ("source_id", "family", "prompt")}
    reports = {}
    for split, rows in splits.items():
        if split not in ("train", "dev", "test") or not rows:
            raise ValueError(f"invalid or empty split: {split}")
        counts = Counter()
        for row in rows:
            for key in ("id", "source_id", "family", "split", "prompt", "task_type", "expected_answer", "metric"):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f"missing/invalid {key}")
            if row["split"] != split:
                raise ValueError("row split disagrees with containing split")
            if row["id"] in ids:
                raise ValueError(f"duplicate id: {row['id']}")
            ids.add(row["id"])
            for key in ownership:
                value = _normalise_prompt(row[key]) if key == "prompt" else row[key]
                if value in ownership[key] and ownership[key][value] != split:
                    raise ValueError(f"cross-split {key}: {value}")
                ownership[key][value] = split
            if row["task_type"] not in ("ordinary", "premise_error", "benign_clarification"):
                raise ValueError("unsupported task_type")
            if row["metric"] != "exact_match":
                raise ValueError("pilot requires exact_match")
            feedback = row.get("teacher_feedback")
            if not isinstance(feedback, dict) or set(feedback) != set(FEEDBACK_MODES):
                raise ValueError("teacher_feedback must declare all three modes")
            for mode, cue in feedback.items():
                if cue is not None and (not isinstance(cue, str) or not cue.strip()):
                    raise ValueError(f"invalid {mode} cue")
            if not feedback["neutral"]:
                raise ValueError("neutral feedback is required")
            evidence = row.get("feedback_evidence")
            if not isinstance(evidence, dict) or any(
                not isinstance(evidence.get(k), str) or not evidence[k].strip()
                for k in ("origin", "version", "observable_basis")
            ):
                raise ValueError("feedback evidence is required")
            if _normalise_prompt(evidence["observable_basis"]) not in _normalise_prompt(row["prompt"]):
                raise ValueError("feedback evidence must quote visible prompt information")
            select_feedback(row, "premise_correction")
            if row["task_type"] == "benign_clarification" and not feedback["benign_clarification"]:
                raise ValueError("benign_clarification needs a corresponding cue")
            counts[row["family"]] += 1
        reports[split] = {"rows": len(rows), "sources": len({r['source_id'] for r in rows}),
                          "families": dict(sorted(counts.items()))}
    return reports


def make_matrix(train_file: str, train_sha256: str) -> dict:
    arms = []
    for teacher, code in (("frozen", "F"), ("recursive", "R")):
        for feedback, suffix in (("neutral", "0"), ("premise_correction", "1")):
            arms.append({"id": code + suffix, "teacher_mode": teacher,
                         "feedback_mode": feedback, "train_file": train_file,
                         "train_sha256": train_sha256})
    return {"schema_version": SCHEMA_VERSION, "kind": "synthetic_pilot",
            "comparison": "teacher_update_x_teacher_feedback", "arms": arms,
            "disabled_feedback_modes": ["benign_clarification"],
            "note": "No arm is labelled positive or negative transfer; evaluation decides direction."}


def validate_matrix(matrix: dict, root: Path) -> dict:
    arms = matrix.get("arms", [])
    expected = {("frozen", "neutral"), ("frozen", "premise_correction"),
                ("recursive", "neutral"), ("recursive", "premise_correction")}
    if len(arms) != 4 or {(a["teacher_mode"], a["feedback_mode"]) for a in arms} != expected:
        raise ValueError("matrix must contain the four intended comparisons exactly once")
    if len({a["id"] for a in arms}) != 4:
        raise ValueError("matrix arm ids must be unique")
    paths = {(root / a["train_file"]).resolve() for a in arms}
    if len(paths) != 1:
        raise ValueError("all arms must reference the same training file")
    path = paths.pop()
    blob = path.read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    if any(a["train_sha256"] != digest for a in arms):
        raise ValueError("training data hash mismatch")
    rows = [json.loads(line) for line in blob.decode("utf-8").splitlines() if line.strip()]
    report = validate_splits({"train": rows})
    # Selecting feedback must preserve the same ordered student prompt cohort.
    prompt_orders = []
    for arm in arms:
        prompt_orders.append([(row["id"], row["prompt"]) for row in rows])
        for row in rows:
            select_feedback(row, arm["feedback_mode"])
    if any(order != prompt_orders[0] for order in prompt_orders[1:]):
        raise ValueError("student prompts differ across arms")
    return {"matched_arms": 4, "train_sha256": digest, "train": report["train"]}


def validate_dataset_dir(data_dir: Path) -> dict:
    """Validate all three splits against their frozen manifest, then the matrix."""
    manifest = json.loads((data_dir / "manifest.json").read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("kind") != "synthetic_pilot":
        raise ValueError("unsupported dataset manifest")
    splits = {}
    for split in SPLIT_SOURCES:
        blob = (data_dir / f"{split}.jsonl").read_bytes()
        if hashlib.sha256(blob).hexdigest() != manifest["files"][split]["sha256"]:
            raise ValueError(f"{split} data hash mismatch")
        splits[split] = [json.loads(line) for line in blob.decode("utf-8").splitlines() if line.strip()]
    counts = validate_splits(splits)
    for split, report in counts.items():
        if any(report[key] != manifest["files"][split][key] for key in ("rows", "sources", "families")):
            raise ValueError(f"{split} manifest counts mismatch")
    return {"splits": counts, "matrix": validate_matrix(manifest["matrix"], data_dir)}


def build_pilot(output_dir: Path, matrix_output: Path | None = None) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output_dir}")
    if matrix_output is not None and matrix_output.exists():
        raise FileExistsError(f"matrix output already exists: {matrix_output}")
    splits = generate_pilot()
    counts = validate_splits(splits)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for split, rows in splits.items():
        blob = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows).encode("utf-8")
        (output_dir / f"{split}.jsonl").write_bytes(blob)
        files[split] = {"path": f"{split}.jsonl", "sha256": hashlib.sha256(blob).hexdigest(), **counts[split]}
    matrix = make_matrix("train.jsonl", files["train"]["sha256"])
    manifest = {"schema_version": SCHEMA_VERSION, "kind": "synthetic_pilot", "seed": None,
                "generation": "deterministic formulas; no model or API calls", "files": files,
                "matrix": matrix,
                "limits": ["Synthetic engineering pilot, not real user traffic or a representative safety dataset.",
                           "Family means a template-source group; domains and abstract operations overlap across splits.",
                           "Premise-error examples supply an incorrect local rule and proposed answer; broader premise reasoning remains untested.",
                           "No actual model refusal is asserted; benign_clarification is reserved and disabled.",
                           "Expected answers are evaluator-only; no preset transfer direction.",
                           "One prompt cohort/order is shared by all four arms; only teacher mode/feedback change."],
                "paired_source_variants": ["ordinary", "premise_error"]}
    validate_matrix(matrix, output_dir)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    if matrix_output is not None:
        matrix_output.parent.mkdir(parents=True, exist_ok=True)
        import os
        relative_train = os.path.relpath((output_dir / "train.jsonl").resolve(), Path.cwd())
        external = make_matrix(relative_train, files["train"]["sha256"])
        external["path_base"] = "workspace_root"
        validate_matrix(external, Path.cwd())
        matrix_output.write_text(json.dumps(external, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-pilot")
    build.add_argument("--output-dir", required=True, type=Path)
    build.add_argument("--matrix-output", type=Path)
    check = sub.add_parser("validate")
    check.add_argument("--data-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "build-pilot":
        report = build_pilot(args.output_dir, args.matrix_output)
        print(json.dumps({"kind": report["kind"], "files": report["files"]}, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(validate_dataset_dir(args.data_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
