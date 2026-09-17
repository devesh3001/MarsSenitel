"""Tests for backward-compatibility and configuration validation contracts."""
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import pytest

from src.train import normalized_config, validate_config, TrainConfig


# ---------------------------------------------------------------------------
# Test 1: augment backward compat — old checkpoints that pre-date 'augment'
# ---------------------------------------------------------------------------

def test_augment_key_missing_from_old_checkpoint_does_not_raise():
    """Older checkpoints (v1–v6) don't store 'augment'; normalized_config must
    supply a default of False instead of raising KeyError."""
    old_style = {
        'version': 'v3',
        'latent_dim': 256,
        'epochs': 15,
        'batch_size': 16,
        'learning_rate': 0.0003,
        'structural_weight': 0.1,
        # 'gradient_weight' and 'augment' deliberately absent
        'seed': 2026,
        'split_seed': 2026,
        'preprocessing': 'raw',
        'workers': 0,
        'patience': 5,
        'max_batches': 0,
    }
    result = normalized_config(old_style)
    assert result['augment'] is False, "Missing 'augment' should default to False"
    assert result['gradient_weight'] == 0.0, "Missing 'gradient_weight' should default to 0.0"


def test_augment_key_present_is_preserved():
    """If augment IS present in the checkpoint, its value must be kept."""
    config_with_augment = asdict(TrainConfig())
    config_with_augment['augment'] = True
    result = normalized_config(config_with_augment)
    assert result['augment'] is True


def test_validate_config_passes_for_old_checkpoint_on_resume():
    """validate_config should not raise when an old checkpoint is resumed with
    the equivalent modern config — the defaults fill in the missing keys."""
    saved = {
        'version': 'v3',
        'latent_dim': 256,
        'epochs': 15,
        'batch_size': 16,
        'learning_rate': 0.0003,
        'structural_weight': 0.1,
        # gradient_weight and augment absent
        'seed': 2026,
        'split_seed': 2026,
        'preprocessing': 'raw',
        'workers': 0,
        'patience': 5,
        'max_batches': 0,
    }
    requested = TrainConfig(
        version='v3', latent_dim=256, epochs=20,  # extending epochs is allowed on resume
        structural_weight=0.1, gradient_weight=0.0, augment=False,
        seed=2026, split_seed=2026, preprocessing='raw',
    )
    # Should not raise
    validate_config(saved, requested, resume=True)


# ---------------------------------------------------------------------------
# Test 2: config mismatch is detected
# ---------------------------------------------------------------------------

def test_validate_config_raises_on_mismatch():
    """Attempting to resume with a different latent_dim must raise."""
    saved = asdict(TrainConfig(latent_dim=128))
    requested = TrainConfig(latent_dim=256)
    with pytest.raises(ValueError, match='latent_dim'):
        validate_config(saved, requested, resume=False)


# ---------------------------------------------------------------------------
# Test 3: parse_bool CLI helper
# ---------------------------------------------------------------------------

def test_parse_bool_handles_string_false():
    """parse_bool('False') must return False, not True (the old bug)."""
    from src.train import parse_bool
    assert parse_bool('False') is False
    assert parse_bool('false') is False
    assert parse_bool('0') is False
    assert parse_bool('True') is True
    assert parse_bool('true') is True
    assert parse_bool('1') is True
    assert parse_bool(True) is True
    assert parse_bool(False) is False
