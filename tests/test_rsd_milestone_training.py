"""Integration at the actual TransferTrainer entry point; CPU only."""
import hashlib
import json
from pathlib import Path

import pytest
import torch

from rsd.train import normalize_save_strategy, prepare_milestone_data, run, synchronized_check
from test_rsd_training import tiny_assets


def fixture_config(tmp_path, teacher_mode):
    model = tmp_path / 'model'
    tiny_assets(model)
    rows = [{'id': f'train-{i}', 'source_id': f'train-{i}', 'split': 'train',
             'prompt': 'How many two plus one?'} for i in range(4)]
    train = tmp_path / 'train.jsonl'
    train.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    def row(item_id, metric):
        return {'id': item_id, 'source_id': item_id, 'split': 'dev', 'family': 'fixture',
                'prompt': 'How many two plus one?', 'metric': metric,
                **({'expected_answer': '3'} if metric == 'exact_match' else {})}
    task, benchmark = tmp_path / 'task.jsonl', tmp_path / 'benchmark.jsonl'
    task.write_text(json.dumps(row('task', 'exact_match')) + '\n')
    benchmark.write_text(json.dumps(row('benchmark', 'overrefusal')) + '\n')
    config = json.loads(Path('configs/rsd/qwen35_9b_2h100.json').read_text())
    config.update(model_name_or_path=str(model), model_revision=None, train_file=str(train),
                  output_dir=str(tmp_path / 'run'), teacher_mode=teacher_mode, max_steps=2,
                  max_completion_length=4, max_prompt_length=48, max_length=64,
                  learning_rate=1e-3, loss_chunk_size=2, save_strategy='no', export_final_model=False,
                  milestone_eval={'steps': [0, 1, 2], 'task_dataset': str(task),
                                  'benchmark_dataset': str(benchmark), 'max_new_tokens': 4,
                                  'max_prompt_tokens': 48, 'seed': 0})
    return config


@pytest.mark.parametrize('teacher_mode', ['frozen', 'recursive'])
def test_train_entrypoint_evaluates_all_steps_without_writing_weights(tmp_path, teacher_mode):
    torch.set_num_threads(1)
    config = fixture_config(tmp_path, teacher_mode)
    trainer = run(config, use_cpu=True)
    folder = Path(config['output_dir'])
    assert trainer.state.global_step == 2
    assert not (folder / 'final').exists()
    assert not list(folder.glob('checkpoint-*'))
    assert not list(folder.rglob('*.pt')) and not list(folder.rglob('*.safetensors'))
    result = json.loads((folder / 'result.json').read_text())
    assert result['final_model_exported'] is False
    assert result['parameter_probe']['changed_coordinates'] > 0
    for step in (0, 1, 2):
        milestone = folder / 'milestones' / f'step-{step:06d}'
        receipt = json.loads((milestone / 'receipt.json').read_text())
        assert receipt['status'] == 'completed'
        assert receipt['n_task_predictions'] == receipt['n_benchmark_predictions'] == 1
        assert receipt['judge_status'] == 'pending'
        assert not receipt['weights_exported']
    if teacher_mode == 'frozen':
        probe = result['frozen_teacher_probe']
        assert probe['changed_coordinates'] == probe['nonfinite_coordinates'] == 0
        assert probe['parameter_scope'] == 'all_parameters_including_frozen'
        assert all(probe[k] for k in ('requires_grad_all_false', 'gradients_all_none', 'eval_mode', 'storage_independent_from_student'))
        assert probe['teacher_parameters_in_optimizer'] == 0


def test_no_export_requires_complete_milestone_schedule_and_no_checkpoints(tmp_path):
    config = fixture_config(tmp_path, 'frozen')
    for change in ({'save_strategy': 'steps'}, {'milestone_eval': None},
                   {'milestone_eval': {**config['milestone_eval'], 'steps': [0, 1]}}):
        with pytest.raises(ValueError, match='Omitting final weights'):
            normalize_save_strategy({**config, **change})


def test_milestone_dev_source_overlap_fails_before_training(tmp_path):
    config = fixture_config(tmp_path, 'frozen')
    task = Path(config['milestone_eval']['task_dataset'])
    row = json.loads(task.read_text())
    row['source_id'] = 'train-0'
    task.write_text(json.dumps(row) + '\n')
    with pytest.raises(ValueError, match='overlap training'):
        prepare_milestone_data(config)


@pytest.mark.parametrize('key', ['task_dataset', 'benchmark_dataset'])
def test_expected_milestone_bytes_sha_freezes_each_cohort(tmp_path, key):
    config = fixture_config(tmp_path, 'recursive')
    path = Path(config['milestone_eval'][key])
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    config['milestone_eval'][key + '_sha256'] = expected
    assert prepare_milestone_data(config)['contract'][key + '_sha256'] == expected
    # Even whitespace-only replacement changes the explicitly frozen raw bytes.
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='expected frozen SHA256'):
        prepare_milestone_data(config)


def test_synchronized_check_propagates_remote_failure_to_both_ranks(monkeypatch):
    monkeypatch.setattr(torch.distributed, 'is_available', lambda: True)
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 2)
    errors = []
    for rank in (0, 1):
        def gather(destination, local):
            assert local == (None if rank == 0 else 'ValueError: synthetic probe failure')
            destination[:] = [None, 'ValueError: synthetic probe failure']
        monkeypatch.setattr(torch.distributed, 'all_gather_object', gather)
        def operation():
            if rank == 1:
                raise ValueError('synthetic probe failure')
            return object()
        with pytest.raises(RuntimeError) as caught:
            synchronized_check('Completion probe', operation)
        errors.append(str(caught.value))
    assert errors == ['Completion probe failed: rank 1: ValueError: synthetic probe failure'] * 2


def test_synchronized_success_exchanges_only_errors_not_probe_objects(monkeypatch):
    monkeypatch.setattr(torch.distributed, 'is_available', lambda: True)
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 2)
    def gather(destination, local):
        assert local is None
        destination[:] = [None, None]
    monkeypatch.setattr(torch.distributed, 'all_gather_object', gather)
    marker = object()
    assert synchronized_check('Probe', lambda: marker) is marker


def test_incomplete_milestone_never_writes_success_or_exports_model(tmp_path, monkeypatch):
    from rsd.milestone_eval import MilestoneEvalCallback
    config = fixture_config(tmp_path, 'recursive')
    def fail(self):
        raise OSError('synthetic unreadable milestone receipt')
    monkeypatch.setattr(MilestoneEvalCallback, 'assert_complete', fail)
    with pytest.raises(RuntimeError, match='Training completion validation failed.*unreadable milestone'):
        run(config, use_cpu=True)
    folder = Path(config['output_dir'])
    assert not (folder / 'result.json').exists()
    assert not (folder / 'trainer_state.json').exists()
    assert not (folder / 'final').exists()
    assert not list(folder.glob('resources_rank*.json'))


def test_initial_teacher_alias_failure_precedes_training_and_success(tmp_path, monkeypatch):
    from rsd import modeling
    from rsd.trainer import TransferTrainer
    config = fixture_config(tmp_path, 'frozen')
    load = modeling.load_text_model
    first = []
    def aliased_load(*args, **kwargs):
        if not first:
            first.append(load(*args, **kwargs))
        return first[0]
    monkeypatch.setattr(modeling, 'load_text_model', aliased_load)
    monkeypatch.setattr(TransferTrainer, 'train', lambda *args, **kwargs: pytest.fail('must reject before training'))
    with pytest.raises(RuntimeError, match='Initial frozen teacher invariants failed.*shares parameter storage'):
        run(config, use_cpu=True)
    assert not (Path(config['output_dir']) / 'result.json').exists()
