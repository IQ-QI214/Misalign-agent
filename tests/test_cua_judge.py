"""Calibration denominators preserve missing and failed API calls."""
import unittest
from scripts.qz_cua_calibrate_judge import analyze


class JudgeCalibrationContracts(unittest.TestCase):
    def test_unknowns_remain_outside_the_scored_denominator(self):
        rows = [{'scenario':'safe', 'scored':True, 'matched':True, 'success_correct':True, 'safety_correct':True},
                {'scenario':'unsafe', 'scored':True, 'matched':False, 'success_correct':True, 'safety_correct':False},
                {'scenario':'failure', 'scored':False, 'matched':None}]
        result=analyze(rows,4)
        self.assertEqual(result['scored'],2)
        self.assertEqual(result['unknown'],2)
        self.assertEqual(result['joint_accuracy_scored'],0.5)
        self.assertEqual(result['joint_accuracy_bounds_all'],[0.25,0.75])
        self.assertIsNone(analyze([],2)['joint_accuracy_scored'])


if __name__=='__main__':
    unittest.main()
