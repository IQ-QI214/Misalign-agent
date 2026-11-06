import copy
import unittest

from rsd.reasoning_data import (TEACHER_CUE, filter_candidates, near_duplicate_ids,
                               parse_source, render, select)


def gsm(question, split="train", answer="2"):
    return parse_source([{"question": question, "answer": "PRIVATE DERIVATION\n#### " + answer}], "gsm8k", split)[0]


def arc(question, split="train", labels=None):
    labels = labels or ["1", "2", "3"]
    return parse_source([{"id": question, "question": question, "choices": {
        "label": labels, "text": ["option" + label for label in labels]},
        "answerKey": labels[-1]}], "arc_challenge", split)[0]


class ReasoningDataTests(unittest.TestCase):
    def test_numeric_label_arc_and_gold_isolation(self):
        item = {**arc("Which material conducts?"), "split": "train"}
        visible, gold = render(item)
        self.assertEqual(gold["expected_answer"], "3")
        self.assertEqual(visible["teacher_context"], TEACHER_CUE)
        self.assertNotIn("expected_answer", visible)
        self.assertNotIn("answerKey", visible)
        self.assertIn("3. option3", visible["prompt"])

    def test_gsm_solution_never_enters_model_input(self):
        visible, gold = render({**gsm("A task with one apple?"), "split": "dev"})
        self.assertNotIn("PRIVATE DERIVATION", str(visible))
        self.assertNotIn("expected_answer", visible)
        self.assertNotIn("teacher_context", visible)
        self.assertEqual(gold["expected_answer"], "2")

    def test_reject_bad_references_and_options(self):
        for answer in ("2 cats", "#### 2", "nan", "1,23"):
            with self.subTest(answer=answer), self.assertRaises(ValueError):
                gsm("test", answer=answer)
        with self.assertRaises(ValueError):
            arc("test", labels=["A", "A", "B"])

    def test_duplicate_stem_quarantines_both_even_different_choices(self):
        first, second = arc(" What is matter? ", "train"), arc("what IS matter?", "test")
        kept, excluded, _ = filter_candidates([first, second])
        self.assertEqual(kept, [])
        self.assertEqual(len(excluded), 2)
        self.assertTrue(all("duplicate_question_stem" in r["reasons"] for r in excluded))

    def test_near_duplicate_cross_native_split(self):
        first = gsm("The total number of red and blue square blocks in the box is what?", "train")
        second = gsm("The total number of red and blue square blocks in a box is what?", "test")
        excluded, pairs = near_duplicate_ids([first, second])
        self.assertEqual(excluded, {first["id"], second["id"]})
        self.assertEqual(len(pairs), 1)

    def test_gsm_same_native_pool_near_duplicates_cannot_cross_derived_splits(self):
        rows = parse_source([
            {"question": "The total number of red and blue square blocks in the box is what?", "answer": "#### 2"},
            {"question": "The total number of red and blue square blocks in a box is what?", "answer": "#### 2"},
        ], "gsm8k", "train")
        kept, excluded, _ = filter_candidates(rows)
        self.assertEqual(kept, [])
        self.assertEqual(len(excluded), 2)

    def test_screen_flags_are_not_harmlessness_labels(self):
        rows = [arc("Which diagram shows a circle?"), arc("How many weapons are sold?")]
        kept, excluded, _ = filter_candidates(rows)
        self.assertFalse(kept)
        self.assertEqual({reason for r in excluded for reason in r["reasons"]},
                         {"conservative_topic_screen", "possible_missing_visual"})

    def test_missing_table_dependency_and_nonvisual_table_salt(self):
        rows = [arc("Which conclusion should the students make from the information in the table?"),
                arc("What happens when table salt dissolves in water?")]
        kept, excluded, _ = filter_candidates(rows)
        self.assertEqual(len(kept), 1)
        self.assertIn("table salt", kept[0]["question"])
        self.assertEqual(excluded[0]["reasons"], ["possible_missing_visual"])

    def test_split_native_preservation_disjoint_and_deterministic(self):
        train = parse_source([{"question": f"Different quantity {i}?", "answer": "#### 2"}
                              for i in range(5)], "gsm8k", "train")
        rows = train + [gsm("Another math evaluation question?", "test"),
                        arc("Conductivity?", "train"), arc("Motion?", "validation"), arc("Gravity?", "test")]
        quotas = {split: {"gsm8k": 1, "arc_challenge": 1} for split in ("train", "dev", "test")}
        result = select(rows, quotas)
        self.assertEqual(result, select(list(reversed(rows)), quotas))
        self.assertEqual(len({r["source_id"] for group in result.values() for r in group}), 6)
        for split, group in result.items():
            for row in group:
                expected = ("validation" if split == "dev" else split) if row["benchmark"] == "arc_challenge" else ("test" if split == "test" else "train")
                self.assertEqual(row["native_split"], expected)

    def test_never_fill_missing_pool_from_test(self):
        quotas = {s: {"gsm8k": 1, "arc_challenge": 0} for s in ("train", "dev", "test")}
        with self.assertRaisesRegex(ValueError, "no split fallback"):
            select([gsm("only test", "test")], quotas)


if __name__ == "__main__":
    unittest.main()
