"""Audit saved model arguments against the advertised action-space schemas."""
import json


def undeclared_arguments(message, schemas):
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
            if unknown:
                issues.append({'tool': name, 'undeclared_arguments': unknown,
                               'declared_arguments': sorted(properties[name])})
    return issues


def audit_arguments(directory, schemas):
    issues = []
    turns = sorted(directory.glob('turn_*/03_actions.json'))
    for path in turns:
        message = json.loads(path.read_text()).get('agent_message', {})
        for issue in undeclared_arguments(message, schemas):
            issues.append({'turn': int(path.parent.name.split('_')[-1]), **issue})
    return {'scope': 'Undeclared top-level arguments of recognized model-facing tools',
            'logged_turns': len(turns), 'affected_turns': len({item['turn'] for item in issues}),
            'issues': issues}


class ActionContractError(RuntimeError):
    """Both the initial proposal and its single correction use undeclared keys."""


def feedback_generator(client, adapter, output, **options):
    """Give one corrective generation opportunity before any GUI execution."""
    from cua_defense.mailbox import teacher_prompt
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
            issues = undeclared_arguments(message, adapter.action_space.get_tool_schemas())
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
            current_prompt = teacher_prompt(prompt,
                'Action rejected before execution. ' + detail +
                '\nRegenerate an action using the declared arguments and the current screenshot.',
                heading='Action argument feedback')
        raise ActionContractError('Undeclared arguments persisted after one corrective attempt')

    return generate
