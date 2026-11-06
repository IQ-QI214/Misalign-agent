import json
from pathlib import Path
from unittest.mock import patch

import pytest

from rsd.train import (load_config, make_training_args, run,
                       validate_batch_config, validate_training_geometry)


def config():
    return json.loads(Path("configs/rsd/qwen35_9b_2h100.json").read_text())


def test_accumulation4_config_propagates_into_framework(tmp_path):
    cfg = config()
    cfg.update(gradient_accumulation_steps=4, data_seed=17)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    loaded = load_config(path)
    args = make_training_args(loaded, tmp_path / "run", use_cpu=True)
    assert args.gradient_accumulation_steps == 4
    assert args.data_seed == 17
    rows = [{"id": str(i)} for i in range(1024)]
    cfg.update(max_steps=512, effective_global_batch=8, planned_exposures=4096)
    dose = validate_training_geometry(cfg, rows, 2)
    assert dose["effective_global_batch"] == 8
    assert dose["planned_sample_exposures"] == 4096


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "4", None])
def test_accumulation_requires_positive_integer(value):
    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        validate_batch_config({**config(), "gradient_accumulation_steps": value})


@pytest.mark.parametrize("value", [True, 2, 1.0, None])
def test_microbatch_one_has_explicit_integer_contract(value):
    with pytest.raises(ValueError, match="microbatch"):
        validate_batch_config({**config(), "per_device_train_batch_size": value})


def test_partial_epoch_and_distributed_padding_rejected():
    cfg = {**config(), "gradient_accumulation_steps": 4}
    for size in (0, 1, 7, 9, 1023):
        with pytest.raises(ValueError, match="divisible"):
            validate_training_geometry(cfg, [{}] * size, 2)


def test_declared_dose_cannot_silently_disagree():
    cfg = {**config(), "gradient_accumulation_steps": 4, "effective_global_batch": 2}
    with pytest.raises(ValueError, match="effective_global_batch"):
        validate_training_geometry(cfg, [{}] * 16, 2)
    cfg.update(effective_global_batch=8, planned_exposures=3)
    with pytest.raises(ValueError, match="planned_exposures"):
        validate_training_geometry(cfg, [{}] * 16, 2)


def test_explicit_cue_cannot_fall_back_or_select_another_feedback():
    cfg = {**config(), "teacher_context": "frozen cue"}
    assert validate_training_geometry(cfg, [{"teacher_context": "frozen cue"}] * 2, 2)
    for row in ({}, {"teacher_context": "Excellent"},
                {"teacher_context": "frozen cue", "teacher_feedback": {"neutral": "other"}}):
        with pytest.raises(ValueError, match="Every training row"):
            validate_training_geometry(cfg, [row] * 2, 2)


def test_disabled_preparation_config_refuses_before_model_loading():
    with patch("rsd.modeling.load_text_model", side_effect=AssertionError("must not load")):
        with pytest.raises(ValueError, match="Training is disabled"):
            run({**config(), "training_enabled": False}, use_cpu=True)


def test_nonconstant_schedule_and_thinking_are_not_silently_ignored():
    for change in ({"thinking": True}, {"lr_scheduler_type": "linear"}):
        with pytest.raises(ValueError):
            validate_batch_config({**config(), **change})
