import pytest
import torch
from src.train import atomic_checkpoint


def test_partial_write_preserves_previous_checkpoint_and_allows_retry(tmp_path, monkeypatch):
    destination = tmp_path/'best.pt'
    atomic_checkpoint({'epoch': 1}, destination)
    original_save = torch.save

    def fail_after_partial_write(state, path):
        path.write_bytes(b'incomplete checkpoint')
        raise OSError('simulated full disk')

    monkeypatch.setattr(torch, 'save', fail_after_partial_write)
    with pytest.raises(OSError, match='simulated full disk'):
        atomic_checkpoint({'epoch': 2}, destination)
    assert torch.load(destination, weights_only=True)['epoch'] == 1
    monkeypatch.setattr(torch, 'save', original_save)
    atomic_checkpoint({'epoch': 2}, destination)
    assert torch.load(destination, weights_only=True)['epoch'] == 2
    assert not (tmp_path/'best.tmp.pt').exists()
