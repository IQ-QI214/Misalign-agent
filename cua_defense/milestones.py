"""Evaluation handoff between a training process and the CPU GUI controller."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from transformers import TrainerCallback
from cua_defense.inference import serve_requests


class MilestoneEvaluation(TrainerCallback):
    def __init__(self, root, processor, milestones, budget_minutes):
        self.root = Path(root)
        self.processor = processor
        self.milestones = set(milestones)
        self.deadline = time.monotonic() + budget_minutes * 60
        self.completed = set()

    def evaluate(self, state, control, model):
        step = state.global_step
        if step in self.completed or step not in self.milestones:
            return control
        directory = self.root / f'step-{step:06d}'
        directory.mkdir(parents=True, exist_ok=False)
        model.eval()
        (directory / 'ready.json').write_text(json.dumps({'model_run': True, 'step': step,
            'time_utc': datetime.now(timezone.utc).isoformat(), 'model_in_training_process': True}) + '\n')
        try:
            serve_requests(model, self.processor, directory,
                           idle_timeout=max(1, min(600, self.deadline - time.monotonic())), deadline=self.deadline)
            if not (directory / 'evaluation_result.json').exists():
                raise RuntimeError('CPU evaluation ended without a result record')
            self.completed.add(step)
        except Exception as error:
            (directory / 'error.json').write_text(json.dumps({'error': str(error)}) + '\n')
            control.should_training_stop = True
        finally:
            (directory / 'stopped.json').write_text(json.dumps({'step': step}) + '\n')
            model.train()
        if time.monotonic() >= self.deadline:
            control.should_training_stop = True
        return control

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        return self.evaluate(state, control, model)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if time.monotonic() >= self.deadline:
            control.should_training_stop = True
            return control
        return self.evaluate(state, control, model)
