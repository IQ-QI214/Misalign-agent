import json
from pathlib import Path
import sys
import tempfile
import unittest

from cua_defense.m1 import jobs,pilot_task,analysis
from scripts.qz_cua_worker_session import supervise


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.config=json.loads((Path(__file__).resolve().parents[1]/'configs/cua/m1_pilot.json').read_text())

    def test_registration_and_seeds(self):
        panel=jobs(self.config)
        self.assertEqual(len(panel),32)
        self.assertEqual(len({x['seed'] for x in panel}),32)
        self.assertEqual(len(jobs(self.config,True)),8)

    def test_attack_does_not_change_authorized_goal(self):
        for family in self.config['families']:
            rows=[pilot_task(family,c)[1] for c in self.config['conditions']]
            self.assertEqual(len({r['instruction'] for r in rows}),1)
            self.assertTrue(all(r['metadata']['evaluator']==rows[0]['metadata']['evaluator'] for r in rows))

    def test_incomplete_groups_are_not_normalized(self):
        panel=jobs(self.config)
        row={**panel[0],'execution_valid':True,'labels':{'safety':True,'utility':True,'safe_completion':True,'sum_reward':2}}
        report=analysis([row],panel)
        self.assertEqual(report['unattempted'],31)
        self.assertNotIn('scalar_relative_advantages',report['groups'][0])

    def test_components_can_also_cancel(self):
        panel=jobs(self.config)[:4]
        results=[{**j,'execution_valid':True,'labels':{'safety':i%2==0,'utility':i%2!=0,
                    'safe_completion':False,'sum_reward':0}} for i,j in enumerate(panel)]
        group=analysis(results,panel)['groups'][0]
        self.assertTrue(group['same_reward_different_outcomes'])
        self.assertEqual(group['scalar_relative_advantages'],[0.0]*4)
        self.assertEqual(group['component_relative_advantages'],[0.0]*4)

    def test_supervisor_records_real_error_exit(self):
        with tempfile.TemporaryDirectory() as d:
            r=supervise([sys.executable,'-c','raise SystemExit(7)'],d,5)
            self.assertEqual(r['returncode'],7)
            self.assertEqual(json.loads((Path(d)/'worker.exit.json').read_text())['returncode'],7)

    def test_supervisor_terminates_its_child_at_budget(self):
        with tempfile.TemporaryDirectory() as d:
            r=supervise([sys.executable,'-c','import time; time.sleep(20)'],d,.1,grace=1)
            self.assertTrue(r['budget_exhausted'])
            self.assertNotEqual(r['returncode'],0)
            self.assertLess(r['seconds'],3)


if __name__=='__main__': unittest.main()
