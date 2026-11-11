"""Serialized GPU requests shared by concurrently running CPU environments."""
import asyncio
import hashlib
import itertools
import fcntl
import json
from pathlib import Path
import time


def teacher_prompt(prompt, feedback):
    if not feedback:
        return prompt
    if not prompt.startswith('<|im_start|>system\n'):
        raise ValueError('Repair feedback requires the Qwen system-message template')
    index = prompt.index('<|im_end|>')
    return prompt[:index] + '\nOffline repair feedback about prior execution:\n' + feedback + '\n' + prompt[index:]


class MailboxClient:
    def __init__(self, directory, timeout=600):
        self.directory = Path(directory)
        self.timeout = timeout
        self.lock = (self.directory / '.client.lock').open('a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except Exception:
            self.lock.close()
            raise
        numbers = [int(path.stem.split('-')[-1]) for path in self.directory.glob('request-*.json')]
        self.counter = itertools.count(max(numbers, default=-1) + 1)

    def close(self):
        self.lock.close()

    def generator(self, output, seed=0, feedback='', replay=None, sampling=None):
        output = Path(output)
        replay = replay or []
        local_turn = 0

        async def generate(*, prompt, images, **kwargs):
            nonlocal local_turn
            turn = local_turn
            local_turn += 1
            if turn < len(replay):
                result = replay[turn]
                await asyncio.sleep(result.get('replay_wait_seconds', 0))
                (output / f'response-{turn}.json').write_text(json.dumps({**result, 'source': 'prefix_replay'}) + '\n')
                return {'response': result['response'], 'finish_reason': result['finish_reason']}
            number = next(self.counter)
            try:
                paths = []
                for index, image in enumerate(images or []):
                    path = self.directory / f'image-{number}-{index}.png'
                    image.save(path)
                    paths.append(str(path))
                generated_prompt = teacher_prompt(prompt, feedback)
                request = {'prompt': generated_prompt, 'images': paths, 'seed': seed,
                           'sampling_params': sampling or {'temperature': 0.0, 'max_new_tokens': 1024},
                           'student_prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                           'generation_prompt_sha256': hashlib.sha256(generated_prompt.encode()).hexdigest()}
                temporary = self.directory / f'request-{number}.tmp'
                temporary.write_text(json.dumps(request) + '\n')
                temporary.replace(self.directory / f'request-{number}.json')
                deadline = time.monotonic() + self.timeout
                response = self.directory / f'response-{number}.json'
                while not response.exists():
                    error = self.directory / 'error.json'
                    if error.exists():
                        raise RuntimeError(error.read_text())
                    if (self.directory / 'stopped.json').exists():
                        raise RuntimeError('GPU worker stopped before responding')
                    if time.monotonic() > deadline:
                        raise TimeoutError('GPU response timed out')
                    await asyncio.sleep(0.25)
                result = json.loads(response.read_text())
                if 'error' in result:
                    raise RuntimeError(result['error'])
                target = output / f'response-{turn}.json'
                if target.exists():
                    raise ValueError('Episode response already exists; resume via the batch ledger')
                target.write_text(json.dumps({**result, 'request_id': number,
                    'source': 'model', 'student_prompt_sha256': request['student_prompt_sha256'],
                    'generation_prompt_sha256': request['generation_prompt_sha256']}, ensure_ascii=False) + '\n')
                return {'response': result['response'], 'finish_reason': result['finish_reason']}
            except BaseException:
                (self.directory / 'stop').touch()
                raise
        return generate
