"""Audit saved model arguments against the advertised action-space schemas."""
import json


def audit_arguments(directory, schemas):
    properties = {}
    for schema in schemas:
        function = schema.get('function', schema)
        parameters = function.get('parameters', {})
        if 'properties' in parameters:
            properties[function['name']] = set(parameters['properties'])
    issues = []
    turns = sorted(directory.glob('turn_*/03_actions.json'))
    for path in turns:
        message = json.loads(path.read_text()).get('agent_message', {})
        for call in message.get('tool_calls', []):
            name, arguments = call.get('name'), call.get('arguments')
            if name in properties and isinstance(arguments, dict):
                unknown = sorted(set(arguments) - properties[name])
                if unknown:
                    issues.append({'turn': int(path.parent.name.split('_')[-1]),
                                   'tool': name, 'undeclared_arguments': unknown})
    return {'scope': 'Undeclared top-level arguments of recognized model-facing tools',
            'logged_turns': len(turns), 'affected_turns': len({item['turn'] for item in issues}),
            'issues': issues}
