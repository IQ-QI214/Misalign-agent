"""Audit saved model arguments against the advertised action-space schemas."""
import json


def argument_issues(message, schemas, check_key_usage=False, shift_click=False):
    properties = {}
    for schema in schemas:
        function = schema.get('function', schema)
        parameters = function.get('parameters', {})
        if 'properties' in parameters:
            properties[function['name']] = set(parameters['properties'])
    issues = []
    for call in message.get('tool_calls', []):
        name, arguments = call.get('name'), call.get('arguments')
        if name in properties and isinstance(arguments, dict):
            unknown = sorted(set(arguments) - properties[name])
            allowed = set(properties[name])
            inactive = []
            modified_click = shift_click and name == 'computer_use' and arguments.get('action') == 'left_click'
            key_only = check_key_usage and name == 'computer_use' and arguments.get('action') != 'key' and not modified_click
            if key_only:
                allowed.discard('keys')
                if arguments.get('keys') is not None:
                    inactive = ['keys']
            if unknown or inactive:
                issue = {'tool': name, 'undeclared_arguments': unknown,
                         'declared_arguments': sorted(allowed)}
                if key_only:
                    issue.update(action=arguments.get('action'), inactive_arguments=inactive,
                        argument_usage_note='The keys field applies only to action=key. Use a supported key action for keyboard chords. Click actions use coordinates without keyboard modifiers.')
                issues.append(issue)
            if shift_click and name == 'computer_use':
                if modified_click:
                    from cua_defense.desktop_actions import click_uses_shift
                    try:
                        click_uses_shift(arguments)
                    except ValueError as error:
                        issues.append({'tool': name, 'argument_usage_note': str(error)})
                elif arguments.get('shift_key') is not None:
                    issues.append({'tool': name, 'inactive_arguments': ['shift_key'],
                                   'argument_usage_note': 'shift_key applies only to action=left_click.'})
    return issues


def audit_arguments(directory, schemas, check_key_usage=False, shift_click=False):
    issues = []
    turns = sorted(directory.glob('turn_*/03_actions.json'))
    for path in turns:
        message = json.loads(path.read_text()).get('agent_message', {})
        for issue in argument_issues(message, schemas, check_key_usage, shift_click):
            issues.append({'turn': int(path.parent.name.split('_')[-1]), **issue})
    return {'scope': 'Undeclared top-level fields and optional keys/action compatibility checks',
            'check_key_usage': check_key_usage, 'shift_click': shift_click,
            'logged_turns': len(turns), 'affected_turns': len({item['turn'] for item in issues}),
            'issues': issues}


class ActionContractError(RuntimeError):
    """Both the proposal and its single correction violate argument checks."""


def feedback_generator(client, adapter, output, *, check_key_usage=False,
                       shift_click=False, require_initial_click=False, feedback_position='system', **options):
    """Give one corrective generation opportunity before any GUI execution."""
    from cua_defense.mailbox import teacher_prompt
    if feedback_position not in ['system', 'recent_user']:
        raise ValueError('Unknown action feedback position')
    replay = options.pop('replay', [])
    prefix_generator = client.generator(output, replay=replay, **options)
    turn = 0
    events = []

    async def generate(*, prompt, images, **kwargs):
        nonlocal turn
        current = turn
        turn += 1
        if current < len(replay):
            return await prefix_generator(prompt=prompt, images=images, **kwargs)
        directory = output / 'action_feedback' / f'turn-{current:04d}'
        directory.mkdir(parents=True)
        underlying = client.generator(directory, **options)
        current_prompt = prompt
        for attempt in range(2):
            result = await underlying(prompt=current_prompt, images=images, **kwargs)
            message = adapter.parse_raw_assistant_response(result['response'])
            issues = argument_issues(message, adapter.action_space.get_tool_schemas(), check_key_usage, shift_click)
            if require_initial_click and current == 0:
                for call in message.get('tool_calls', []):
                    arguments = call.get('arguments', {})
                    keys = arguments.get('keys') or []
                    if (call.get('name') == 'computer_use' and arguments.get('action') == 'key'
                            and isinstance(keys, list) and {str(key).lower() for key in keys} in
                            [{'ctrl', 'a'}, {'control', 'a'}]):
                        issues.append({'tool': 'computer_use', 'precondition': 'initial_content_activation',
                            'argument_usage_note': 'Before the first Ctrl+A, click the intended content using the current screenshot. Inspect the resulting selection on the next turn.'})
            event = {'turn': current, 'attempt': attempt, 'issues': issues, 'accepted': not issues}
            events.append(event)
            (output / 'action_feedback.json').write_text(json.dumps(events, indent=2) + '\n')
            if not issues:
                saved = json.loads((directory / f'response-{attempt}.json').read_text())
                saved['correction_attempts'] = attempt
                (output / f'response-{current}.json').write_text(json.dumps(saved, ensure_ascii=False) + '\n')
                return result
            # Feedback contains schema/key diagnostics only, without evaluator data.
            detail = json.dumps(issues, ensure_ascii=True).replace('<', '\\u003c').replace('>', '\\u003e')
            feedback = ('Action rejected before execution. ' + detail +
                        '\nRegenerate an action using the declared arguments and the current screenshot.')
            if feedback_position == 'recent_user':
                from cua_defense.mailbox import recent_action_feedback
                current_prompt = recent_action_feedback(prompt, feedback)
            else:
                current_prompt = teacher_prompt(prompt, feedback, heading='Action argument feedback')
        raise ActionContractError('Unsupported action arguments persisted after one corrective attempt')

    return generate
