'''Tensor-only helpers for distribution matching distillation (DMD).'''

from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
from torch.nn import functional as F


def shift_timesteps(
        base_timesteps: torch.Tensor,
        min_timestep: int,
        max_timestep: int,
        shift: float,
        eps: float = 1e-8) -> torch.Tensor:
    '''Apply the original monotonic noise-shift mapping.

    ``max_timestep`` is exclusive, matching ``torch.randint``.
    '''
    if max_timestep <= min_timestep:
        raise ValueError('max_timestep must be greater than min_timestep.')
    shift = max(float(shift), float(eps))
    span = max(float(max_timestep - min_timestep - 1), 1.0)
    normalized = (base_timesteps.float() - float(min_timestep)) / span
    shifted = (shift * normalized) / (1.0 + (shift - 1.0) * normalized + eps)
    shifted = torch.round(shifted * span + float(min_timestep)).long()
    return shifted.clamp_(min_timestep, max_timestep - 1)


def sample_shifted_timesteps(
        batch_size: int,
        min_timestep: int,
        max_timestep: int,
        shift: float,
        device: torch.device,
        eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    '''Sample base timesteps and return ``(shifted, base)``.'''
    if batch_size < 1:
        raise ValueError('batch_size must be positive.')
    base = torch.randint(
        min_timestep, max_timestep, (batch_size,), device=device, dtype=torch.long)
    return shift_timesteps(base, min_timestep, max_timestep, shift, eps), base


def distributed_softmax_weights(
        scores: torch.Tensor,
        temperature: float,
        min_weight: float,
        max_weight: float,
        eps: float = 1e-8) -> torch.Tensor:
    '''Make mean-one softmax weights, then clamp without renormalizing.'''
    scores = scores.reshape(-1)
    if scores.numel() == 0:
        return scores
    if min_weight > max_weight:
        raise ValueError('min_weight must not exceed max_weight.')

    scaled = scores / max(float(temperature), float(eps))
    if dist.is_available() and dist.is_initialized():
        local_size = torch.tensor([scaled.numel()], device=scaled.device, dtype=torch.long)
        sizes = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
        dist.all_gather(sizes, local_size)
        sizes = [int(value.item()) for value in sizes]
        if any(size != sizes[0] for size in sizes):
            raise RuntimeError(
                'Directional weighting requires equal local batch sizes; '
                f'got {sizes}. Enable drop_last for training.')
        gathered = [torch.zeros_like(scaled) for _ in sizes]
        dist.all_gather(gathered, scaled.detach())
        global_scores = torch.cat(gathered, dim=0)
        global_weights = torch.softmax(global_scores, dim=0) * global_scores.numel()
        start = dist.get_rank() * scaled.numel()
        weights = global_weights[start:start + scaled.numel()]
    else:
        weights = torch.softmax(scaled, dim=0) * scaled.numel()
    return weights.clamp(min=float(min_weight), max=float(max_weight))


def residual_cosine_scores(
        fake_prediction: torch.Tensor,
        real_prediction: torch.Tensor,
        noise: torch.Tensor,
        eps: float = 1e-8) -> torch.Tensor:
    '''Cosine similarity of fake-image and real-image denoising residuals.'''
    if fake_prediction.shape != real_prediction.shape or fake_prediction.shape != noise.shape:
        raise ValueError('Predictions and noise must have identical shapes.')
    batch_size = fake_prediction.shape[0]
    fake_residual = (fake_prediction.detach() - noise.detach()).reshape(batch_size, -1).float()
    real_residual = (real_prediction.detach() - noise.detach()).reshape(batch_size, -1).float()
    return F.cosine_similarity(real_residual, fake_residual, dim=1, eps=eps)


def standard_dmd_loss(
        generated_latent: torch.Tensor,
        noisy_latent: torch.Tensor,
        real_noise_prediction: torch.Tensor,
        fake_noise_prediction: torch.Tensor,
        alphas_cumprod: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
        norm_eps: float = 1e-4,
        grad_rms_max: Optional[float] = 0.5,
        loss_clip_max: Optional[float] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    '''Return the standard latent-space DMD surrogate loss and diagnostics.'''
    batch_size = generated_latent.shape[0]
    if alphas_cumprod.numel() != batch_size:
        raise ValueError('alphas_cumprod must contain one value per sample.')

    alpha = alphas_cumprod.to(device=generated_latent.device, dtype=torch.float32)
    sqrt_alpha = alpha.sqrt().view(-1, 1, 1, 1)
    sqrt_one_minus_alpha = (1.0 - alpha).sqrt().view(-1, 1, 1, 1)
    with torch.no_grad():
        noisy = noisy_latent.float()
        pred_real_x0 = (
            noisy - sqrt_one_minus_alpha * real_noise_prediction.float()
        ) / (sqrt_alpha + 1e-8)
        pred_fake_x0 = (
            noisy - sqrt_one_minus_alpha * fake_noise_prediction.float()
        ) / (sqrt_alpha + 1e-8)
        raw_gradient = pred_fake_x0 - pred_real_x0
        norm = (generated_latent.detach().float() - pred_real_x0).abs().mean(
            dim=(1, 2, 3), keepdim=True).clamp_min(float(norm_eps))
        dmd_gradient = raw_gradient / norm
        raw_rms = torch.sqrt(
            dmd_gradient.square().mean(dim=(1, 2, 3), keepdim=True) + 1e-12)
        scale = torch.ones_like(raw_rms)
        if grad_rms_max is not None and float(grad_rms_max) > 0:
            scale = torch.clamp(float(grad_rms_max) / raw_rms, max=1.0)
            dmd_gradient = dmd_gradient * scale
        target = generated_latent.detach().float() - dmd_gradient

    per_sample = 0.5 * F.mse_loss(
        generated_latent.float(), target, reduction='none').mean(dim=(1, 2, 3))
    if sample_weights is None:
        sample_weights = torch.ones_like(per_sample)
    sample_weights = sample_weights.detach().to(device=per_sample.device, dtype=per_sample.dtype)
    if sample_weights.numel() != batch_size:
        raise ValueError('sample_weights must contain one value per sample.')

    unweighted = per_sample.mean()
    raw_loss = (per_sample * sample_weights.reshape(-1)).mean()
    loss = raw_loss
    clip_hit = raw_loss.new_tensor(0.0)
    if loss_clip_max is not None and float(loss_clip_max) > 0:
        clip_hit = (raw_loss > float(loss_clip_max)).float().detach()
        loss = torch.clamp(raw_loss, max=float(loss_clip_max))

    final_rms = torch.sqrt(
        dmd_gradient.square().mean(dim=(1, 2, 3), keepdim=True) + 1e-12)
    stats = {
        'dmd_standard_norm_mean': norm.mean().detach(),
        'dmd_grad_raw_rms_mean': raw_rms.mean().detach(),
        'dmd_grad_rms_mean': final_rms.mean().detach(),
        'dmd_grad_clip_rate': (scale < 1.0).float().mean().detach(),
        'dmd_loss_raw': raw_loss.detach(),
        'dmd_loss_unweighted': unweighted.detach(),
        'dmd_loss_p95': torch.quantile(per_sample.detach(), 0.95),
        'dmd_loss_clip_hit': clip_hit,
    }
    return loss, stats
