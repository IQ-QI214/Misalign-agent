import csv
import io
import json

import pytest

from rsd import benchmarks as b


def csv_bytes(fields, rows):
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().encode()


def test_xstest_filters_contrasts_and_preserves_native_ids():
    raw = csv_bytes(["id", "prompt", "type"], [
        {"id": "1", "prompt": "Sample A", "type": "category"},
        {"id": "2", "prompt": "Sample B", "type": "contrast_category"},
        {"id": "3", "prompt": "Sample C", "type": "another"},
    ])
    rows = b.parse_xstest(raw, expected_total=3, expected_kept=2)
    assert [row["native_id"] for row in rows] == ["1", "3"]
    assert all(row["metric"] == "overrefusal" for row in rows)
    with pytest.raises(ValueError, match="safe count"):
        b.parse_xstest(raw, expected_total=3, expected_kept=3)


def test_strongreject_csv_multiline_is_one_record():
    raw = csv_bytes(["category", "source", "forbidden_prompt"],
                    [{"category": "test", "source": "fixture", "forbidden_prompt": "Line one\nLine two"}])
    rows = b.parse_strongreject(raw, expected_total=1)
    assert len(rows) == 1 and rows[0]["prompt"] == "Line one\nLine two"
    assert rows[0]["metric"] == "safety"


def test_sorry_base_single_turn_only():
    raw = json.dumps({"question_id": 1, "category": "fixture", "turns": ["Example"]}).encode()
    assert b.parse_sorry(raw, expected_total=1)[0]["native_id"] == "1"
    with pytest.raises(ValueError, match="exactly one turn"):
        b.parse_sorry(json.dumps({"question_id": 1, "category": "x", "turns": ["A", "B"]}).encode(), 1)


def test_split_and_sources_follow_normalised_prompt_across_benchmarks():
    first = b._row("a", "1", "x", "  Ａlpha\nBeta ", "safety")
    second = b._row("b", "2", "y", "alpha beta", "safety")
    assert first["source_id"] == second["source_id"]
    assert first["split"] == second["split"]
    assert b.validate_rows([first, second])["duplicate_prompt_rows"] == 1
    second["split"] = "test" if first["split"] == "dev" else "dev"
    with pytest.raises(ValueError, match="Cross-split"):
        b.validate_rows([first, second])


@pytest.mark.parametrize("training", [{"prompt": " SAMPLE "}, {"turns": ["sample"]}])
def test_training_overlap_is_rejected(training):
    row = b._row("a", "1", "x", "sample", "safety")
    with pytest.raises(ValueError, match="Training/evaluation overlap"):
        b.validate_rows([row], [training])


def test_training_source_overlap_and_train_split_rejected():
    row = b._row("a", "1", "x", "sample", "safety")
    with pytest.raises(ValueError, match="overlap"):
        b.validate_rows([row], [{"prompt": "Different", "source_id": row["source_id"]}])
    row["split"] = "train"
    with pytest.raises(ValueError, match="evaluation-only"):
        b.validate_rows([row])


def test_fetch_requires_pinned_hash(monkeypatch):
    class Response(io.BytesIO):
        pass
    monkeypatch.setattr(b, "urlopen", lambda *args, **kwargs: Response(b"content"))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        b.fetch_pinned({"url": "https://example.invalid/data", "sha256": "0" * 64})


def test_prepare_reports_missing_gated_source_and_checks_before_write(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "fetch_pinned", lambda source: b"fixture")
    monkeypatch.setattr(b, "parse_xstest", lambda raw: [b._row("xstest_safe", "1", "x", "safe fixture", "overrefusal")])
    monkeypatch.setattr(b, "parse_strongreject", lambda raw: [b._row("strongreject", "1", "y", "safety fixture", "safety")])
    training = tmp_path / "train.jsonl"
    training.write_text(json.dumps({"prompt": "safe fixture"}) + "\n")
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="overlap"):
        b.prepare(output, [training])
    assert not output.exists()
    manifest = b.prepare(output)
    assert manifest["status"] == "partial_sorry_missing"
    assert manifest["sources"]["sorry_bench"]["status"] == "missing_gated_source"
    assert not (output / "sorry_bench.jsonl").exists()
    assert manifest["split"]["origin"] == "our_deterministic_split_not_paper_split"
    assert (output / "manifest.json").is_file()


def test_preparation_rejects_changed_existing_output(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "fetch_pinned", lambda source: b"fixture")
    monkeypatch.setattr(b, "parse_xstest", lambda raw: [b._row("xstest_safe", "1", "x", "A", "overrefusal")])
    monkeypatch.setattr(b, "parse_strongreject", lambda raw: [b._row("strongreject", "1", "y", "B", "safety")])
    b.prepare(tmp_path)
    (tmp_path / "dev.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="differs"):
        b.prepare(tmp_path)
