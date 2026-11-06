"""Offline checks of cohort identity, information boundaries, and split leakage."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from rsd.data import build_pilot, generate_pilot, select_feedback, validate_dataset_dir, validate_matrix, validate_splits


class RSDDataTests(unittest.TestCase):
    def setUp(self):
        self.splits = generate_pilot()

    def test_pilot_counts_pairs_and_family_exposure(self):
        report = validate_splits(self.splits)
        self.assertEqual({k: v["rows"] for k, v in report.items()}, {"train": 64, "dev": 16, "test": 16})
        for split, rows in self.splits.items():
            expected_per_family = 16 if split == "train" else 4
            self.assertEqual(set(report[split]["families"].values()), {expected_per_family})
            for source in {r["source_id"] for r in rows}:
                pair = [r for r in rows if r["source_id"] == source]
                self.assertEqual({r["task_type"] for r in pair}, {"ordinary", "premise_error"})
                self.assertEqual(len({r["expected_answer"] for r in pair}), 1)
                self.assertEqual(len({r["family"] for r in pair}), 1)

    def test_hand_checked_first_examples(self):
        normal = {r["domain"]: r for r in self.splits["train"][:8] if r["task_type"] == "ordinary"}
        self.assertEqual(normal["amount_count"]["expected_answer"], "11")
        self.assertEqual(normal["units"]["expected_answer"], "2075")
        self.assertEqual(normal["sets"]["expected_answer"], "4")
        self.assertEqual(normal["time"]["expected_answer"], "09:15")

    def test_gold_is_not_a_feedback_input(self):
        for row in self.splits["train"]:
            altered = deepcopy(row)
            altered["expected_answer"] = "DO_NOT_DISCLOSE_EVALUATOR_ONLY_GOLD"
            for mode in ("neutral", "premise_correction", "benign_clarification"):
                self.assertEqual(select_feedback(row, mode), select_feedback(altered, mode))
                self.assertNotIn(altered["expected_answer"], select_feedback(altered, mode))
            if row["task_type"] == "ordinary":
                self.assertEqual(select_feedback(row, "premise_correction"), select_feedback(row, "neutral"))

    def test_duplicate_id_is_rejected(self):
        self.splits["train"][1]["id"] = self.splits["train"][0]["id"]
        with self.assertRaisesRegex(ValueError, "duplicate id"):
            validate_splits(self.splits)

    def test_source_or_family_cannot_cross_splits(self):
        for key in ("source_id", "family"):
            with self.subTest(key=key):
                changed = deepcopy(self.splits)
                changed["dev"][0][key] = changed["train"][0][key]
                with self.assertRaisesRegex(ValueError, f"cross-split {key}"):
                    validate_splits(changed)

    def test_normalised_prompts_cannot_cross_splits(self):
        self.splits["dev"][0]["prompt"] = "  " + self.splits["train"][0]["prompt"] + "\n"
        with self.assertRaisesRegex(ValueError, "cross-split prompt"):
            validate_splits(self.splits)

    def test_correction_needs_visible_evidence_and_cue(self):
        for change in ("missing_evidence", "invisible_evidence", "missing_cue"):
            with self.subTest(change=change):
                altered = deepcopy(self.splits)
                row = altered["train"][1]
                if change == "missing_evidence":
                    row.pop("feedback_evidence")
                elif change == "invisible_evidence":
                    row["feedback_evidence"]["observable_basis"] = "private answer unavailable to the student"
                else:
                    row["teacher_feedback"]["premise_correction"] = None
                with self.assertRaises(ValueError):
                    validate_splits(altered)

    def test_build_is_deterministic_and_matrix_pins_shared_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a", Path(tmp) / "b"
            ma, mb = build_pilot(a), build_pilot(b)
            self.assertEqual(ma, mb)
            for filename in ("train.jsonl", "dev.jsonl", "test.jsonl", "manifest.json"):
                self.assertEqual((a / filename).read_bytes(), (b / filename).read_bytes())
            self.assertEqual(validate_matrix(ma["matrix"], a)["matched_arms"], 4)
            self.assertEqual({arm["train_file"] for arm in ma["matrix"]["arms"]}, {"train.jsonl"})
            with self.assertRaises(FileExistsError):
                build_pilot(a)
            # Content drift invalidates every arm even when the filename is unchanged.
            with (a / "train.jsonl").open("a") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                validate_matrix(ma["matrix"], a)

    def test_matrix_refuses_a_different_prompt_cohort(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            matrix = build_pilot(root)["matrix"]
            changed = deepcopy(matrix)
            changed["arms"][1]["train_file"] = "dev.jsonl"
            with self.assertRaisesRegex(ValueError, "same training file"):
                validate_matrix(changed, root)
            changed = deepcopy(matrix)
            changed["arms"][1]["feedback_mode"] = "benign_clarification"
            with self.assertRaisesRegex(ValueError, "four intended"):
                validate_matrix(changed, root)

    def test_heldout_data_is_also_hash_pinned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_pilot(root)
            self.assertEqual(validate_dataset_dir(root)["splits"]["test"]["rows"], 16)
            rows = [json.loads(line) for line in (root / "test.jsonl").read_text().splitlines()]
            rows[0]["expected_answer"] = "incorrect evaluation gold"
            (root / "test.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "test data hash mismatch"):
                validate_dataset_dir(root)

    def test_unknown_feedback_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            select_feedback(self.splits["train"][0], "positive")


if __name__ == "__main__":
    unittest.main()
