'''Minimal REPA loss used by the simplified DMD training model.'''

import importlib
import math
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from basicsr.utils.registry import LOSS_REGISTRY


def _project_path(value):
    if value is None or str(value).lower() in ('', 'none', 'null', '~'):
        return None
    path = Path(value)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    return path.resolve()


def _load_local_dino_backbone(repo_dir, model_name, weights):
    repo_string = str(repo_dir)
    if repo_string not in sys.path:
        sys.path.insert(0, repo_string)

    module = importlib.import_module('dinov3.hub.backbones')
    module_path = Path(module.__file__).resolve()
    expected_package = (repo_dir / 'dinov3').resolve()
    if expected_package not in module_path.parents:
        raise ImportError(
            f'dinov3 resolved outside the configured repository: {module_path}')
    try:
        builder = getattr(module, model_name)
    except AttributeError as error:
        raise ValueError(f'Unknown DINOv3 backbone: {model_name}') from error
    return builder(pretrained=True, weights=str(weights))


@LOSS_REGISTRY.register()
class DMDRepaLoss(nn.Module):
    '''Align one score-UNet feature map with frozen DINOv3 patch tokens.'''

    def __init__(
            self,
            in_dim=320,
            target_dim=768,
            hidden_dim=1024,
            loss_weight=0.1,
            target_range='zero_one',
            loss_mode='neg_cosine',
            dino_repo_dir='dinov3',
            dino_weights=None,
            dino_model_name='dinov3_vitb16'):
        super().__init__()
        self.loss_weight = float(loss_weight)
        self.target_range = target_range
        self.loss_mode = loss_mode

        repo_dir = _project_path(dino_repo_dir)
        weights = _project_path(dino_weights)
        if repo_dir is None or not repo_dir.is_dir():
            raise FileNotFoundError(f'DINOv3 repository not found: {repo_dir}')
        if weights is None or not weights.is_file():
            raise FileNotFoundError(
                'score.repa.dino_weights must point to a local DINOv3 checkpoint; '
                f'got {weights}.')

        self.backbone = _load_local_dino_backbone(
            repo_dir, dino_model_name, weights)
        self.backbone.requires_grad_(False).eval()

        output_dim = getattr(self.backbone, 'embed_dim', None)
        if target_dim in (None, 'auto'):
            target_dim = output_dim
        if target_dim is None:
            raise ValueError('Cannot infer DINOv3 output dimension; set target_dim explicitly.')
        if output_dim is not None and int(target_dim) != int(output_dim):
            raise ValueError(
                f'target_dim={target_dim} does not match DINOv3 embed_dim={output_dim}.')

        patch_size = getattr(self.backbone, 'patch_size', 16)
        if isinstance(patch_size, (tuple, list)):
            if patch_size[0] != patch_size[1]:
                raise ValueError(f'Only square DINO patches are supported: {patch_size}')
            patch_size = patch_size[0]
        self.patch_size = int(patch_size)
        self.proj = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(target_dim)),
        )
        self.register_buffer(
            'image_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False)
        self.register_buffer(
            'image_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def _preprocess_target(self, image):
        if self.target_range == 'minus_one_one':
            image = (image + 1.0) * 0.5
        elif self.target_range != 'zero_one':
            raise ValueError(f'Unsupported target_range: {self.target_range}')
        image = image.clamp(0.0, 1.0)
        image = (image - self.image_mean) / self.image_std

        height, width = image.shape[-2:]
        pad_h = (self.patch_size - height % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - width % self.patch_size) % self.patch_size
        if pad_h or pad_w:
            mode = 'reflect' if height > pad_h and width > pad_w else 'replicate'
            image = F.pad(image, (0, pad_w, 0, pad_h), mode=mode)
        return image

    @staticmethod
    def _tokens_to_grid(tokens):
        batch, token_count, channels = tokens.shape
        side = int(math.sqrt(token_count))
        if side * side != token_count:
            raise ValueError(f'Cannot infer a square grid from {token_count} source tokens.')
        return tokens.transpose(1, 2).reshape(batch, channels, side, side)

    def _source_tokens(self, source, target_tokens):
        if source.dim() == 3:
            source = self._tokens_to_grid(source)
        elif source.dim() != 4:
            raise ValueError(f'Unsupported REPA source shape: {tuple(source.shape)}')

        target_count = target_tokens.shape[1]
        target_side = int(math.sqrt(target_count))
        if target_side * target_side != target_count:
            raise ValueError(f'Cannot infer a square grid from {target_count} DINO tokens.')
        if source.shape[-2:] != (target_side, target_side):
            source = F.adaptive_avg_pool2d(source, (target_side, target_side))
        return source.flatten(2).transpose(1, 2)

    def forward(self, source_feature, target_image):
        with torch.no_grad():
            target = self._preprocess_target(target_image)
            target_tokens = self.backbone.forward_features(target)['x_norm_patchtokens']

        source_tokens = self._source_tokens(source_feature, target_tokens)
        projected = F.normalize(self.proj(source_tokens.float()), dim=-1)
        target_tokens = F.normalize(target_tokens.detach().float(), dim=-1)
        similarity = (projected * target_tokens).sum(dim=-1).mean()

        if self.loss_mode == 'neg_cosine':
            loss = -similarity
        elif self.loss_mode == 'one_minus_cosine':
            loss = 1.0 - similarity
        else:
            raise ValueError(f'Unsupported loss_mode: {self.loss_mode}')
        return loss * self.loss_weight
