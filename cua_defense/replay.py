"""Operational replay checks over visible pixels and task-file content."""
import hashlib
import io
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile


def file_fingerprint(data):
    if data is None:
        return None
    if not zipfile.is_zipfile(io.BytesIO(data)):
        return hashlib.sha256(data).hexdigest()
    digest = hashlib.sha256()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for name in sorted(archive.namelist()):
            content = archive.read(name)
            if name == 'docProps/core.xml':
                root = ET.fromstring(content)
                for element in list(root):
                    if element.tag.rsplit('}', 1)[-1] in ['created', 'modified']:
                        root.remove(element)
                content = ET.tostring(root)
            digest.update(name.encode())
            digest.update(content)
    return digest.hexdigest()


def visual_difference(before, after):
    from PIL import ImageChops
    if before.size != after.size:
        return 1.0
    width, height = before.size
    box = (0, int(height * 0.04), width, height)
    diff = ImageChops.difference(before.convert('RGB').crop(box), after.convert('RGB').crop(box))
    histogram = diff.point(lambda value: 255 if value > 32 else 0).convert('L').histogram()
    return sum(histogram[1:]) / (width * (height - box[1]))


async def focus_state(interface):
    try:
        result = await interface.run_command(
            'export DISPLAY=${DISPLAY:-:1}; '
            'xdotool getactivewindow getwindowname; '
            'xdotool getwindowfocus getwindowname; '
            'xprop -id "$(xdotool getwindowfocus)" WM_CLASS', timeout=10)
        return {'stdout': result.stdout, 'stderr': result.stderr, 'returncode': result.returncode}
    except Exception as error:
        return {'error': f'{type(error).__name__}: {error}', 'returncode': None}


async def checkpoint(env, row, image, directory, turn):
    import json
    import shlex
    result = row['metadata'].get('evaluator', {}).get('result', {})
    paths = []
    entries = result if isinstance(result, list) else [result]
    for entry in entries:
        if entry.get('type') == 'vm_file':
            paths.append(entry['path'])
    paths += row['metadata'].get('others', {}).get('research_checkpoint_paths', [])
    hashes = {}
    snapshots = {}
    for path in dict.fromkeys(paths):
        test = await env._computer.interface.run_command('test -f ' + shlex.quote(path))
        content = await env._computer.interface.read_bytes(path) if test.returncode == 0 else None
        hashes[path] = file_fingerprint(content)
        if content is not None:
            filename = f'checkpoint-{turn}-file-{len(snapshots)}' + Path(path).suffix
            (directory / filename).write_bytes(content)
            snapshots[path] = filename
    record = {'files': hashes, 'visual_size': image.size,
              'file_snapshots': snapshots,
              'focus': await focus_state(env._computer.interface),
              'coverage': 'Task output files, excluding ZIP container timestamps and core created/modified fields; visible image excludes the top 4% desktop panel.'}
    image.save(directory / f'checkpoint-{turn}.png')
    (directory / f'checkpoint-{turn}.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


def verify_checkpoint(reference, current, directory, turn):
    import json
    from PIL import Image
    old = json.loads((reference / f'checkpoint-{turn}.json').read_text())
    if old['files'] != current['files']:
        raise ValueError('Replayed task-file state differs at the branch point')
    with Image.open(reference / f'checkpoint-{turn}.png') as before, Image.open(directory / f'checkpoint-{turn}.png') as after:
        difference = visual_difference(before, after)
    if difference > 0.005:
        raise ValueError(f'Replayed visible state differs at branch point: {difference:.4f}')
    return {'file_match': True, 'changed_pixel_fraction': difference, 'threshold': 0.005}
