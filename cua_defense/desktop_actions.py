"""Opt-in Qwen desktop Shift-click conversion with reversible native history."""
import copy


def click_uses_shift(arguments):
    """Validate the supported click modifier and its legacy spelling."""
    keys = arguments.get('keys')
    if keys is not None and (not isinstance(keys, list) or keys not in [[], ['shift'], ['Shift']]):
        raise ValueError('left_click keys must be omitted, empty, or ["shift"]')
    legacy = arguments.get('shift_key')
    if legacy is not None and not isinstance(legacy, bool) and legacy not in ['True', 'False', 'true', 'false']:
        raise ValueError('shift_key must be a boolean')
    requested = legacy in [True, 'True', 'true']
    if keys and legacy is not None and not requested:
        raise ValueError('keys and shift_key disagree')
    return bool(keys) or requested


def make_shift_click_action_space():
    """Build an extended space without changing upstream registrations."""
    from lite.agents.models.qwen3_5.action_space import Qwen3_5DesktopActionSpace
    from lite.agents.core.action_space.base import LiteDesktopActionSpace
    from lite.agents.core.action_space.errors import ModelToolCallParseError
    from lite.core.tools.action_space import merge_adjacent_lite_action_batches
    from lite.core.tools.calls import make_tool_call, tool_call_arguments, tool_call_name

    class ShiftClickActionSpace(Qwen3_5DesktopActionSpace):
        @classmethod
        def get_tool_schemas(cls, include=None):
            schemas = copy.deepcopy(super().get_tool_schemas(include=include))
            for schema in schemas:
                function = schema.get('function', schema)
                if function['name'] == 'computer_use':
                    properties = function['parameters']['properties']
                    properties['keys']['description'] = (
                        'Keyboard chord for action=key. For action=left_click, '
                        'omit keys for a plain click or use ["shift"] for Shift-click.')
                    properties['shift_key'] = {'type': 'boolean',
                        'description': 'Legacy Shift-click flag for action=left_click. Prefer keys=["shift"].'}
            return schemas

        def _convert_single_from_agent(self, call, **kwargs):
            arguments = call.get('arguments', {})
            if call.get('name') != 'computer_use' or arguments.get('action') != 'left_click':
                if call.get('name') == 'computer_use' and arguments.get('shift_key') is not None:
                    raise ModelToolCallParseError('shift_key applies only to computer_use action=left_click')
                if (call.get('name') == 'computer_use' and arguments.get('action') != 'key'
                        and arguments.get('keys') is not None):
                    raise ModelToolCallParseError('keys applies only to action=key or Shift left_click')
                return super()._convert_single_from_agent(call, **kwargs)
            try:
                shifted = click_uses_shift(arguments)
            except ValueError as error:
                raise ModelToolCallParseError(str(error)) from error
            click = super()._convert_single_from_agent(call, **kwargs)
            if not shifted:
                return click
            return merge_adjacent_lite_action_batches([
                LiteDesktopActionSpace.key_down(keys=['shift']), *click,
                LiteDesktopActionSpace.key_up(keys=['shift'])])

        def _convert_single_to_agent(self, call):
            if tool_call_name(call) != 'computer':
                return super()._convert_single_to_agent(call)
            actions = tool_call_arguments(call)['actions']
            result = []
            index = 0
            while index < len(actions):
                triple = actions[index:index + 3]
                if (len(triple) == 3 and triple[0] == {'action': 'key_down', 'keys': ['shift']}
                        and triple[1].get('action') == 'click'
                        and triple[1].get('button', 'left') == 'left'
                        and triple[1].get('clicks', 1) == 1
                        and triple[2] == {'action': 'key_up', 'keys': ['shift']}):
                    result.append({'name': 'computer_use', 'arguments': {
                        'action': 'left_click', 'coordinate': triple[1]['coordinate'], 'keys': ['shift']}})
                    index += 3
                else:
                    child = actions[index]
                    result.extend(super()._convert_single_to_agent(make_tool_call(
                        child['action'], {key: value for key, value in child.items() if key != 'action'})))
                    index += 1
            return result

    return ShiftClickActionSpace()


def install_shift_click_adapter(adapter):
    from lite.agents.models.qwen3_5.action_space import Qwen3_5DesktopActionSpace
    if type(adapter.action_space) is not Qwen3_5DesktopActionSpace:
        raise TypeError('Shift-click adaptation requires the unmodified Qwen3.5 desktop action space')
    adapter.action_space = make_shift_click_action_space()


async def release_shift_on_failure(interface, operation, cleanup_events):
    """Release a potentially held modifier if an environment step aborts."""
    try:
        return await operation
    except BaseException:
        try:
            await interface.key_up('Shift_L')
            cleanup_events.append({'modifier': 'Shift_L', 'released': True})
        except Exception as error:
            cleanup_events.append({'modifier': 'Shift_L', 'released': False, 'error': repr(error)})
        raise
