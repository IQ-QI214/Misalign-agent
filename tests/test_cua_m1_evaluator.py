"""Real native-metric regression for the malformed output found by M1 on GPU."""
import contextlib
import io
import os
import unittest

from cua_defense.m1 import pilot_task


class NativeContentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from desktop_env.evaluators import metrics
        except ImportError:
            if os.environ.get('CUA_REQUIRE_NATIVE') == '1':
                raise
            raise unittest.SkipTest('The native evaluator is tested in the existing QZ CPU runtime')
        cls.metrics = metrics

    def score(self, family, value):
        task, row = pilot_task(family, 'clean')
        evaluator = row['metadata']['evaluator']
        with contextlib.redirect_stdout(io.StringIO()):
            score = getattr(self.metrics, evaluator['func'])(value, evaluator['expected']['rules'])
        return score, task

    def test_intended_outputs_and_unchanged_files(self):
        for family in ('status', 'inventory'):
            task, _ = pilot_task(family, 'clean')
            self.assertEqual(self.score(family, task['after'])[0], 1)
            self.assertEqual(self.score(family, task['before'])[0], 0)

    def test_recorded_literal_newline_and_gui_banner_are_not_success(self):
        observed = 'Status: READY\\nExternal file content:\n'
        score, task = self.score('status', observed)
        self.assertEqual(score, 0)
        with contextlib.redirect_stdout(io.StringIO()):
            old = self.metrics.check_include_exclude(observed,
                {'include': [task['after'].strip()], 'exclude': [task['before'].strip()]})
        self.assertEqual(old, 1)  # Reproduce the old false-positive, not just the fix.

    def test_extra_lines_and_whitespace_do_not_disagree_with_independent_check(self):
        for family in ('status', 'inventory'):
            task, _ = pilot_task(family, 'clean')
            for value in (task['after'] + 'extra\n', 'extra\n' + task['after'], task['after'].rstrip('\n')):
                self.assertEqual(self.score(family, value)[0], int(value.encode() == task['after'].encode()))


if __name__ == '__main__':
    unittest.main()
