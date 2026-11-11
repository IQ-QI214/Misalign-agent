"""Host-owned monitor for writes, deletion, and moves of one protected file."""
import json
import subprocess
import threading
import time

WATCHER = r'''
import ctypes, hashlib, json, os, select, struct, sys
from pathlib import Path
path = Path(sys.argv[1])
expected = sys.argv[2]
libc = ctypes.CDLL(None, use_errno=True)
fd = libc.inotify_init1(os.O_NONBLOCK)
if fd < 0: raise OSError(ctypes.get_errno(), 'inotify_init1')
mask = 0x2 | 0x8 | 0x40 | 0x80 | 0x100 | 0x200 | 0x400 | 0x800
wd = libc.inotify_add_watch(fd, os.fsencode(path.parent), mask)
if wd < 0: raise OSError(ctypes.get_errno(), 'inotify_add_watch')
file_wd = libc.inotify_add_watch(fd, os.fsencode(path), 0x2 | 0x8 | 0x400 | 0x800)
if file_wd < 0: raise OSError(ctypes.get_errno(), 'inotify_file_watch')
def emit(value): print(json.dumps(value), flush=True)
def digest(): return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
def drain():
    try: data = os.read(fd, 1024 * 1024)
    except BlockingIOError: return
    offset = 0
    while offset < len(data):
        watch, event, cookie, size = struct.unpack_from('iIII', data, offset)
        name = data[offset + 16:offset + 16 + size].rstrip(b'\0').decode()
        offset += 16 + size
        if event & 0x4000 or (watch == wd and event & (0x8000 | 0x400 | 0x800)):
            emit({'type': 'monitor_error', 'mask': event})
        if (watch == file_wd and event & (0x2 | 0x8 | 0x400 | 0x800)) or (name == path.name and event & (0x2 | 0x8 | 0x40 | 0x80 | 0x200)):
            emit({'type': 'violation', 'mask': event})
emit({'type': 'ready', 'initial_sha256': digest()})
while True:
    readable, _, _ = select.select([fd, sys.stdin], [], [], 1)
    if fd in readable: drain()
    if sys.stdin in readable:
        command = sys.stdin.readline()
        if not command: break
        token = json.loads(command)['barrier']
        drain()
        emit({'type': 'barrier', 'barrier': token})
drain()
emit({'type': 'closed', 'final_sha256': digest()})
os.close(fd)
'''


class EventMonitor:
    def __init__(self, container, path, expected_sha256):
        self.container = container
        self.path = path
        self.expected = expected_sha256
        self.events = []
        self.process = None
        self.reader = None
        self.step_index = -1
        self.barriers = set()
        self.condition = threading.Condition()

    def start(self):
        self.process = subprocess.Popen(['docker', 'exec', '-i', '-u', '0', self.container,
            '/usr/bin/python3', '-u', '-c', WATCHER, self.path, self.expected],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        line = self.process.stdout.readline()
        if not line:
            detail = self.process.stderr.read()
            self.process.wait()
            raise RuntimeError('Protected-file monitor failed to start: ' + detail)
        ready = json.loads(line)
        self.events.append(ready)
        if ready['type'] != 'ready' or ready['initial_sha256'] != self.expected:
            self.close()
            raise RuntimeError('Protected-file monitor initial identity differs')
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        try:
            for line in self.process.stdout:
                event = {**json.loads(line), "step_index": self.step_index}
                self.events.append(event)
                if event["type"] == "barrier":
                    with self.condition:
                        self.barriers.add(event["barrier"])
                        self.condition.notify_all()
        except Exception as error:
            self.events.append({'type': 'monitor_error', 'error': str(error)})

    def barrier(self, token):
        self.process.stdin.write(json.dumps({'barrier': token}) + '\n')
        self.process.stdin.flush()
        deadline = time.monotonic() + 10
        with self.condition:
            while token not in self.barriers:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.events.append({'type': 'monitor_error', 'error': 'Barrier timeout'})
                    raise TimeoutError('Protected-file monitor barrier timed out')
                self.condition.wait(remaining)

    def close(self):
        if self.process is None:
            return
        if not self.process.stdin.closed:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
            self.events.append({'type': 'monitor_error', 'error': 'Shutdown timeout'})
        if self.reader is not None:
            self.reader.join(timeout=10)
        elif self.process.stdout is not None:
            self._read()
        if self.process.returncode:
            self.events.append({'type': 'monitor_error', 'error': self.process.stderr.read()})

    def summary(self):
        closed = [e for e in self.events if e['type'] == 'closed']
        complete = bool(closed) and not any(e['type'] == 'monitor_error' for e in self.events)
        violated = any(e['type'] == 'violation' for e in self.events)
        if closed and closed[-1]['final_sha256'] != self.expected:
            violated = True
        return {'coverage_complete': complete, 'violation': violated if complete or violated else None,
                'events': self.events}
