"""Versioned observation preparation for the synthetic terminal pilot only."""
import copy
import hashlib
import json
import shlex

PROTOCOL = {
    'id': 'synthetic_terminal_v1', 'image_size': [1920, 1088],
    'mask_top_pixels': 32, 'mask_rgb': [0, 0, 0],
    'terminal_geometry_request': [200, 100, 900, 600],
    'scope': 'Synthetic terminal tasks only; top system panel is outside the task and attack surface.',
    'coordinate_transform': 'identity', 'raw_images_retained': True,
}


def pixel_hash(image):
    return hashlib.sha256((image.mode + str(image.size)).encode() + image.tobytes()).hexdigest()


def model_image(image):
    from PIL import ImageDraw
    if list(image.size) != PROTOCOL['image_size']:
        raise ValueError('Observation protocol requires the registered screen dimensions')
    result = image.convert('RGB').copy()
    ImageDraw.Draw(result).rectangle((0, 0, result.width - 1, PROTOCOL['mask_top_pixels'] - 1),
                                    fill=tuple(PROTOCOL['mask_rgb']))
    return result


def save_inputs(images, directory, turn):
    """Persist exactly the raw and model-visible images without coordinate changes."""
    rows, prepared = [], []
    for index, raw in enumerate(images):
        image = model_image(raw)
        raw_name, model_name = f'raw-{turn}-{index}.png', f'input-{turn}-{index}.png'
        raw.save(directory / raw_name)
        image.save(directory / model_name)
        rows.append({'raw': raw_name, 'input': model_name,
                     'raw_pixel_sha256': pixel_hash(raw), 'input_pixel_sha256': pixel_hash(image)})
        prepared.append(image)
    (directory / f'observation-{turn}.json').write_text(json.dumps(
        {'protocol': PROTOCOL, 'images': rows}, indent=2) + '\n')
    return prepared


def stable_task(row):
    result = copy.deepcopy(row)
    code = '''import subprocess,time
def run(*args):
    return subprocess.check_output(args,text=True).strip()
run('gsettings','set','org.gnome.desktop.interface','enable-animations','false')
run('gsettings','set','org.gnome.desktop.interface','cursor-blink','false')
profile=run('gsettings','get','org.gnome.Terminal.ProfilesList','default').strip("'")
schema='org.gnome.Terminal.Legacy.Profile:/org/gnome/terminal/legacy/profiles:/:'+profile+'/'
run('gsettings','set',schema,'cursor-blink-mode','off')
windows=run('xdotool','search','--onlyvisible','--class','gnome-terminal').splitlines()
assert len(windows)==1, 'Exactly one visible terminal is required'
window=windows[0]
run('xdotool','windowsize','--sync',window,'900','600')
run('xdotool','windowmove','--sync',window,'200','100')
run('xdotool','windowactivate','--sync',window)
run('xdotool','mousemove','1800','1000')
time.sleep(1)
'''
    result['metadata']['config'].append({'type': 'execute', 'parameters': {
        'command': 'python3 -c ' + shlex.quote(code), 'shell': True}})
    result['metadata']['others']['research_observation_protocol'] = PROTOCOL['id']
    return result


async def geometry(interface):
    code = '''import subprocess,json
def run(*args):
    return subprocess.check_output(args,text=True).strip()
active=run('xdotool','getactivewindow')
focus=run('xdotool','getwindowfocus')
raw=dict(line.split('=',1) for line in run('xdotool','getwindowgeometry','--shell',active).splitlines())
print(json.dumps({'geometry':{key:int(raw[key]) for key in ('X','Y','WIDTH','HEIGHT','SCREEN')},
 'title':run('xdotool','getwindowname',active), 'class':run('xprop','-id',active,'WM_CLASS'),
 'active_equals_focus':active==focus}))
'''
    result = await interface.run_command('python3 -c ' + shlex.quote(code), timeout=10)
    if result.returncode:
        raise RuntimeError('Cannot capture active window geometry: ' + result.stderr)
    return json.loads(result.stdout)
