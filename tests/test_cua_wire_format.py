"""Exercise the pinned native Qwen3.5 parser on scripted control payloads."""
import shlex
import unittest

from scripts.qz_cua_focus_diagnostic import action_response
from scripts.qz_cua_m0 import control_actions
from scripts.qz_cua_judge_controls import TASKS

try:
    from lite.agents.models.qwen3_5.adapter import _parse_xml_tool_calls
except ModuleNotFoundError as error:
    if error.name != 'lite':
        raise
    _parse_xml_tool_calls = None


@unittest.skipIf(_parse_xml_tool_calls is None, 'Native CUA-Lite parser runs in the existing QZ environment')
class NativeWireTests(unittest.TestCase):
    def test_text_quotes_and_backslashes_are_preserved(self):
        for text in ['"quoted"', "'quoted'", 'a\\nb', 'line1\nline2', '中文与 "quotes"']:
            parsed = _parse_xml_tool_calls(action_response('type', text=text))
            self.assertEqual(parsed[0]['arguments'], {'action': 'type', 'text': text})

    def test_every_m0_command_parses_as_valid_python(self):
        scenarios = ['safe_completion', 'unsafe_completion', 'ordinary_failure',
                     'unsafe_failure', 'violation_then_restore']
        for task in TASKS[:2]:
            for scenario in scenarios:
                parsed = _parse_xml_tool_calls(control_actions(task, scenario)[1])
                text = parsed[0]['arguments']['text']
                tokens = shlex.split(text)
                self.assertEqual(tokens[:2], ['python3', '-c'])
                compile(tokens[2], '<scripted-control>', 'exec')

    def test_key_arrays_remain_typed_arrays(self):
        parsed = _parse_xml_tool_calls(action_response('key', keys=['ctrl', 'c']))
        self.assertEqual(parsed[0]['arguments']['keys'], ['ctrl', 'c'])


if __name__ == '__main__':
    unittest.main()
