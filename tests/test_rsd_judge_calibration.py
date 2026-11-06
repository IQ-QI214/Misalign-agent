"""CPU-only constructed-control contracts; no GPU or private test data needed."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from rsd import judge_calibration as jc
from rsd.evaluation import digest, judge_inputs, read_jsonl, text_digest
from rsd.judge_protocols import PROFILE_NAME, parse_judge_output, load_source_lock


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


class JudgeCalibrationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.spec = jc.load_controls()
        self.source = self.root / "frozen-dev.jsonl"
        source_ids = sorted({row["source_id"] for row in self.spec["controls"]})
        self.source_rows = [{"id": item_id, "source_id": "source-" + item_id,
                             "family": "fixture", "split": "dev", "prompt": "Private fixture " + item_id,
                             "metric": "safety" if item_id.startswith("strongreject") else "overrefusal"}
                            for item_id in source_ids]
        write_jsonl(self.source, self.source_rows)
        self.spec["source_dev_sha256"] = jc._sha(self.source)
        self.controls_path = self.root / "controls.json"
        self.controls_path.write_text(json.dumps(self.spec, indent=2) + "\n")
        self.bundle = self.root / "bundle"
        self.manifest = jc.build(self.source, self.bundle, controls_path=self.controls_path)
        self.dataset = read_jsonl(self.bundle / "dataset.jsonl")
        self.predictions = read_jsonl(self.bundle / "predictions.jsonl")

    def make_round(self, number, *, raw_overrides=None, missing=(), profile_change=None, limit=None):
        requirements = self.manifest["judge_requirements"]
        profile = {"name": requirements["profile_name"], "parser_version": requirements["parser_version"],
                   "source_lock": load_source_lock(), "alternative_judge_model": True,
                   "protocol_implementation_sha256": requirements["protocol_implementation_sha256"],
                   "model_identity": {"kind": "CPU_fixture_never_loaded"}, "generation": {"seed": 0}}
        if profile_change:
            profile.update(profile_change)
        judge_id = PROFILE_NAME + ":" + digest(profile)
        rows = []
        inputs = judge_inputs(self.dataset, self.predictions)
        for control, item in zip(self.spec["controls"], inputs):
            if control["id"] in missing:
                continue
            raw = control["hard"].get("raw_label")
            if raw is None:
                raw = f"1.b {control['hard']['components']['refusal']}\n2.b 4\n3.b 4"
            if raw_overrides and control["id"] in raw_overrides:
                raw = raw_overrides[control["id"]]
            parsed = parse_judge_output(item["metric"], raw)
            row = {key: item[key] for key in
                   ("id", "metric", "checkpoint_id", "prompt_sha256", "response_sha256")}
            row.update(judge_id=judge_id, judge_profile=profile, raw_judge_text=raw,
                       parser_result=parsed, stopped_on_eos=True, reached_token_limit=False, **parsed)
            rows.append(row)
        path = self.root / f"round{number}.jsonl"
        write_jsonl(path, rows)
        sidecar = {"judge_id": judge_id, "judge_profile": profile,
                   "dataset_sha256": self.manifest["dataset_sha256"],
                   "predictions_sha256": self.manifest["predictions_sha256"],
                   "selected_ids": [row["id"] for row in self.dataset], "limit": limit, "allow_test": False}
        path.with_name(path.name + ".manifest.json").write_text(json.dumps(sidecar))
        return path

    def analyze(self, paths):
        return jc.analyze(self.bundle, paths, controls_path=self.controls_path)

    def test_builder_frozen_source_and_existing_judge_schema(self):
        self.assertEqual(len(judge_inputs(self.dataset, self.predictions)), 16)
        self.assertEqual(self.manifest["n_planned_judgments"], 32)
        self.assertFalse(self.manifest["subject_model_performance"])
        for row, pred, control in zip(self.dataset, self.predictions, self.spec["controls"]):
            self.assertEqual(pred["prompt_sha256"], text_digest(row["prompt"]))
            self.assertEqual(pred["response_sha256"], text_digest(control["response"]))
            self.assertTrue(pred["checkpoint_id"].startswith("constructed-controls:sha256:"))
            self.assertEqual(row["source_dev_id"], control["source_id"])
            self.assertNotIn("hard", row)
        self.assertEqual(sum(row["cross_rubric_benign_anchor"] for row in self.dataset), 3)
        self.assertTrue(all("prompt" not in row for row in jc.load_controls()["controls"]))
        with self.assertRaises(FileExistsError):
            jc.build(self.source, self.bundle, controls_path=self.controls_path)

    def test_rejects_changed_source_before_parsing_and_never_accepts_test(self):
        self.source.write_text("not even JSON")
        with self.assertRaisesRegex(ValueError, "frozen dev SHA256"):
            jc.build(self.source, self.root / "other", controls_path=self.controls_path)
        self.source_rows[0]["split"] = "test"
        write_jsonl(self.source, self.source_rows)
        self.spec["source_dev_sha256"] = jc._sha(self.source)
        self.controls_path.write_text(json.dumps(self.spec))
        with self.assertRaisesRegex(ValueError, "dev records only"):
            jc.build(self.source, self.root / "other", controls_path=self.controls_path)
        self.assertFalse((self.root / "other").exists())

    def test_all_hard_expectations_pass_only_finite_controls(self):
        report = self.analyze([self.make_round(1), self.make_round(2)])
        self.assertTrue(report["finite_controls_passed"])
        self.assertEqual(report["summary"]["n_valid"], 32)
        self.assertEqual(report["summary"]["n_hard_passed"], 32)
        self.assertEqual(report["summary"]["n_soft_cases"], 6)
        self.assertEqual(len(report["repeat_stability"]), 16)
        self.assertEqual(len(report["paired_variants"]), 12)
        self.assertFalse(report["formal_judge_reliability_established"])
        self.assertFalse(report["parser_fixtures_in_semantic_denominator"])
        self.assertIsNone(report["real_harmful_compliance_accuracy"])
        json.dumps(report, allow_nan=False)

    def test_partial_and_full_refusal_are_not_hidden_by_same_binary_score(self):
        report = self.analyze([self.make_round(1), self.make_round(2, raw_overrides={"X03": "2_full_refusal"})])
        self.assertFalse(report["finite_controls_passed"])
        self.assertEqual(report["summary"]["n_hard_failed"], 1)
        case = report["rounds"][1]["cases"][2]
        self.assertEqual(case["value"], 1)
        self.assertFalse(case["hard_passed"])

    def test_real_driver_manifest_explicit_full_cohort_limit(self):
        # run_local_judge records the supplied CLI --limit, even when its slice
        # retains every case. Mirror those exact manifest fields and JSON types.
        first, second = self.make_round(1, limit=16), self.make_round(2, limit=16)
        contract = json.loads(second.with_name(second.name + ".manifest.json").read_text())
        self.assertEqual(set(contract), {"judge_id", "judge_profile", "dataset_sha256",
                                        "predictions_sha256", "selected_ids", "limit", "allow_test"})
        self.assertEqual(contract["limit"], len(contract["selected_ids"]))
        report = self.analyze([first, second])
        self.assertTrue(report["finite_controls_passed"])
        self.assertEqual(report["summary"]["n_planned"], 32)
        self.assertEqual(report["summary"]["n_valid"], 32)

    def test_explicit_limit_cannot_hide_truncation_or_missing_judgments(self):
        first = self.make_round(1, limit=16)
        for limit in (15, 17, True, 16.0, "16"):
            with self.subTest(limit=limit):
                second = self.make_round(2, limit=limit)
                with self.assertRaisesRegex(ValueError, "all fixed dev controls"):
                    self.analyze([first, second])
        second = self.make_round(2, limit=16)
        sidecar = second.with_name(second.name + ".manifest.json")
        contract = json.loads(sidecar.read_text())
        contract["selected_ids"] = contract["selected_ids"][:-1]
        sidecar.write_text(json.dumps(contract))
        with self.assertRaisesRegex(ValueError, "all fixed dev controls"):
            self.analyze([first, second])
        second = self.make_round(2, limit=16, missing=("X01",))
        report = self.analyze([first, second])
        self.assertFalse(report["finite_controls_passed"])
        self.assertEqual(report["summary"]["n_missing"], 1)
        self.assertEqual(report["summary"]["n_valid"], 31)

    def test_invalid_and_missing_stay_unknown_not_measured_zero(self):
        report = self.analyze([self.make_round(1), self.make_round(2, missing=("S01",),
                              raw_overrides={"X01": "1_full_compliance\nReasoning: extra"})])
        self.assertFalse(report["finite_controls_passed"])
        self.assertEqual(report["summary"]["n_valid"], 30)
        self.assertEqual(report["summary"]["n_invalid"], 1)
        self.assertEqual(report["summary"]["n_missing"], 1)
        for case in report["rounds"][1]["cases"]:
            if case["control_id"] in {"X01", "S01"}:
                self.assertIsNone(case["value"])
                self.assertIsNone(case["hard_passed"])
        empty = self.make_round(3, missing=tuple(row["id"] for row in self.spec["controls"]))
        empty2 = self.make_round(4, missing=tuple(row["id"] for row in self.spec["controls"]))
        empty_report = self.analyze([empty, empty2])
        self.assertEqual(empty_report["summary"]["n_missing"], 32)
        self.assertIsNone(empty_report["summary"]["hard_pass_fraction_when_valid"])

    def test_soft_ranges_and_component_drift_are_separate_from_hard_gate(self):
        report = self.analyze([self.make_round(1), self.make_round(2, raw_overrides={
            "S01": "1.b 1\n2.b 3\n3.b 4", "S04": "1.b 0\n2.b 1\n3.b 1"})])
        self.assertTrue(report["finite_controls_passed"])
        self.assertEqual(report["summary"]["n_soft_failed"], 1)
        repeat = next(row for row in report["repeat_stability"] if row["control_id"] == "S01")
        self.assertFalse(repeat["components_identical"])
        self.assertTrue(repeat["value_identical"])

    def test_same_file_and_different_judge_identity_cannot_form_repeats(self):
        first = self.make_round(1)
        with self.assertRaisesRegex(ValueError, "two distinct"):
            self.analyze([first, first])
        second = self.make_round(2, profile_change={"generation": {"seed": 1}})
        with self.assertRaisesRegex(ValueError, "exact same judge identity"):
            self.analyze([first, second])

    def test_tampered_scores_bindings_duplicates_and_foreign_ids_fail_closed(self):
        first, second = self.make_round(1), self.make_round(2)
        original = read_jsonl(second)
        for mutation in ("value", "prompt_sha256", "response_sha256", "checkpoint_id", "duplicate", "foreign"):
            with self.subTest(mutation=mutation):
                rows = copy.deepcopy(original)
                if mutation == "duplicate":
                    rows.append(rows[0])
                elif mutation == "foreign":
                    rows[0]["id"] = "foreign"
                else:
                    rows[0][mutation] = 1 if mutation == "value" else "foreign"
                write_jsonl(second, rows)
                with self.assertRaises(ValueError):
                    self.analyze([first, second])

    def test_truncated_judge_output_retains_parser_result_but_invalid_value(self):
        first, second = self.make_round(1), self.make_round(2)
        rows = read_jsonl(second)
        rows[0].update(stopped_on_eos=False, reached_token_limit=True, valid=False,
                       value=None, error="judge_output_token_limit")
        write_jsonl(second, rows)
        report = self.analyze([first, second])
        self.assertFalse(report["finite_controls_passed"])
        self.assertEqual(report["summary"]["n_invalid"], 1)
        self.assertIsNone(report["rounds"][1]["cases"][0]["value"])

    def test_changed_bundle_or_round_selection_rejected(self):
        first, second = self.make_round(1), self.make_round(2)
        sidecar = second.with_name(second.name + ".manifest.json")
        contract = json.loads(sidecar.read_text())
        contract["selected_ids"] = contract["selected_ids"][:-1]
        sidecar.write_text(json.dumps(contract))
        with self.assertRaisesRegex(ValueError, "all fixed dev controls"):
            self.analyze([first, second])
        with (self.bundle / "predictions.jsonl").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
            self.analyze([first, second])

    def test_cli_analyze_and_import_are_stdlib_only(self):
        first, second = self.make_round(1), self.make_round(2)
        output = self.root / "analysis.json"
        subprocess.run([sys.executable, "-B", "-m", "rsd.judge_calibration", "analyze",
                        "--bundle-dir", str(self.bundle), "--controls", str(self.controls_path),
                        "--rounds", str(first), str(second), "--output", str(output)],
                       cwd=jc.ROOT, check=True, capture_output=True, text=True)
        self.assertTrue(json.loads(output.read_text())["finite_controls_passed"])
        script = ("import sys; import rsd.judge_calibration; "
                  "assert not {'torch','transformers','openai','requests'} & set(sys.modules)")
        subprocess.run([sys.executable, "-B", "-c", script], cwd=jc.ROOT, check=True)


if __name__ == "__main__":
    unittest.main()
