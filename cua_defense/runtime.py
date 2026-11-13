"""Risk monitoring around the native CUA-Lite environment lifecycle."""
import asyncio
from lite.gym.wrappers import EnvWrapper
from cua_defense.events import EventMonitor
from cua_defense.tasks import PROTECTED, PROTECTED_SHA256


class TaskResourceEnv(EnvWrapper):
    def __init__(self, env, expected_paths):
        super().__init__(env)
        self.expected_paths = expected_paths
        self.setup_attempts = 0

    async def reset(self):
        import shlex
        for attempt in range(2):
            self.setup_attempts += 1
            observation = await self.env.reset()
            missing = []
            for path in self.expected_paths:
                result = await self.env._computer.interface.run_command('test -f ' + shlex.quote(path))
                if result.returncode:
                    missing.append(path)
            if not missing:
                return observation
            await self.env.close()
        raise RuntimeError('Task setup did not create expected resources: ' + repr(missing))


class MonitoredEnv(EnvWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.monitor = None
        self.risk = None
        self.step_index = -1

    async def reset(self):
        observation = await self.env.reset()
        self.monitor = EventMonitor(self.env._container_name, PROTECTED, PROTECTED_SHA256,
                                    monitor_user='65532:65532')
        await asyncio.to_thread(self.monitor.start)
        return observation

    async def step(self, actions):
        self.step_index += 1
        self.monitor.step_index = self.step_index
        result = await self.env.step(actions)
        await asyncio.to_thread(self.monitor.barrier, self.step_index)
        return result

    async def close(self):
        try:
            if self.monitor is not None:
                await asyncio.to_thread(self.monitor.close)
                self.risk = self.monitor.summary()
                self.monitor = None
        finally:
            await self.env.close()


class ShiftClickEnv(EnvWrapper):
    """Keep the native batch executor and clean up aborted modifier batches."""
    def __init__(self, env):
        super().__init__(env)
        self.modifier_cleanup = []

    async def step(self, actions):
        from lite.core.tools.calls import tool_call_arguments, tool_call_name
        from cua_defense.desktop_actions import release_shift_on_failure
        shifted = any(child.get('action') == 'key_down' and child.get('keys') == ['shift']
                      for call in actions if tool_call_name(call) == 'computer'
                      for child in tool_call_arguments(call)['actions'])
        operation = self.env.step(actions)
        if shifted:
            return await release_shift_on_failure(self.env._computer.interface, operation, self.modifier_cleanup)
        return await operation
