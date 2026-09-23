'''Fast tensor tests for the simplified DMD math.'''

import importlib.util
from pathlib import Path

import torch

_UTIL_PATH = Path(__file__).resolve().parents[1] / 'basicsr' / 'utils' / 'dmd_util.py'
_SPEC = importlib.util.spec_from_file_location('dmd_util_under_test', _UTIL_PATH)
_DMD_UTIL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_DMD_UTIL)

distributed_softmax_weights = _DMD_UTIL.distributed_softmax_weights
residual_cosine_scores = _DMD_UTIL.residual_cosine_scores
shift_timesteps = _DMD_UTIL.shift_timesteps
standard_dmd_loss = _DMD_UTIL.standard_dmd_loss


def test_active_basicsr_sources_compile():
    root = Path(__file__).resolve().parents[1]
    package = root / 'basicsr'
    source_files = list(package.glob('*.py'))
    for name in ('archs', 'data', 'losses', 'metrics', 'models', 'ops', 'utils'):
        source_files.extend((package / name).rglob('*.py'))

    for source_file in source_files:
        source = source_file.read_text(encoding='utf-8')
        compile(source, str(source_file), 'exec')


def test_shift_timesteps_identity_and_bounds():
    base = torch.arange(0, 200)
    identity = shift_timesteps(base, 0, 200, shift=1.0)
    assert torch.equal(identity, base)
    shifted = shift_timesteps(base, 0, 200, shift=4.0)
    assert shifted.min().item() == 0
    assert shifted.max().item() == 199
    assert torch.all(shifted[1:] >= shifted[:-1])


def test_direction_weights_and_original_cosine_range():
    scores = torch.tensor([-1.0, 0.0, 1.0])
    weights = distributed_softmax_weights(
        scores, temperature=1.0, min_weight=0.0, max_weight=10.0)
    assert torch.all(weights[1:] > weights[:-1])
    assert torch.allclose(weights.mean(), torch.tensor(1.0))

    noise = torch.randn(4, 2, 3, 3)
    fake = torch.randn_like(noise)
    real = torch.randn_like(noise)
    cosine = residual_cosine_scores(fake, real, noise)
    assert cosine.shape == (4,)
    assert torch.all(cosine >= -1.0)
    assert torch.all(cosine <= 1.0)


def test_direction_weight_is_one_for_batch_size_one():
    weight = distributed_softmax_weights(
        torch.tensor([0.37]), temperature=0.2,
        min_weight=0.2, max_weight=3.0)
    assert torch.equal(weight, torch.ones_like(weight))


def test_standard_dmd_loss_backpropagates_only_to_generated_latent():
    generated = torch.randn(2, 4, 4, 4, requires_grad=True)
    noisy = torch.randn_like(generated)
    real_prediction = torch.randn_like(generated, requires_grad=True)
    fake_prediction = torch.randn_like(generated, requires_grad=True)
    alphas = torch.tensor([0.8, 0.4])
    weights = torch.tensor([0.5, 1.5])

    loss, stats = standard_dmd_loss(
        generated, noisy, real_prediction, fake_prediction,
        alphas, sample_weights=weights)
    loss.backward()
    assert torch.isfinite(loss)
    assert generated.grad is not None
    assert torch.isfinite(generated.grad).all()
    assert real_prediction.grad is None
    assert fake_prediction.grad is None
    assert 'dmd_grad_rms_mean' in stats


if __name__ == '__main__':
    test_active_basicsr_sources_compile()
    test_shift_timesteps_identity_and_bounds()
    test_direction_weights_and_original_cosine_range()
    test_direction_weight_is_one_for_batch_size_one()
    test_standard_dmd_loss_backpropagates_only_to_generated_latent()
    print('DMD math tests passed.')
