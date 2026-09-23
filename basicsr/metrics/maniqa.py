'''Optional PyIQA-backed validation metrics.'''

import numpy as np
import torch

from basicsr.metrics.metric_util import reorder_image
from basicsr.utils.registry import METRIC_REGISTRY


_PYIQA_MODEL_CACHE = {}


def _resolve_device(device):
    if device in (None, 'auto'):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(device)


def _get_pyiqa_model(metric_name, device):
    try:
        import pyiqa
    except ImportError as error:
        raise ImportError(
            'PyIQA metrics require the optional pyiqa package. '
            'Install the dependencies from requirements.txt.') from error

    device = _resolve_device(device)
    cache_key = (metric_name, str(device))
    if cache_key not in _PYIQA_MODEL_CACHE:
        model = pyiqa.create_metric(
            metric_name, device=device, as_loss=False).eval()
        _PYIQA_MODEL_CACHE[cache_key] = model
    return _PYIQA_MODEL_CACHE[cache_key], device


def _prepare_tensor_image(
        image, crop_border, input_order, device, bgr2rgb=True):
    image = reorder_image(image, input_order=input_order)
    if crop_border:
        image = image[
            crop_border:-crop_border, crop_border:-crop_border, ...]
    if image.ndim == 2:
        image = image[..., None]
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    if image.shape[2] != 3:
        raise ValueError(f'Expected one or three channels, got {image.shape}.')
    if bgr2rgb:
        image = image[..., ::-1]

    image = np.ascontiguousarray(image.transpose(2, 0, 1))
    tensor = torch.from_numpy(image).float().unsqueeze(0)
    if tensor.max().item() > 1.0:
        tensor = tensor / 255.0
    return tensor.clamp_(0.0, 1.0).to(device)


def _scalar_score(value):
    return torch.as_tensor(value).detach().float().mean().item()


@METRIC_REGISTRY.register()
def calculate_maniqa(
        img,
        img2=None,
        crop_border=0,
        input_order='HWC',
        device='auto',
        bgr2rgb=True,
        model_name='maniqa-pipal',
        **kwargs):
    '''Calculate the no-reference MANIQA score; higher is better.'''
    model, target_device = _get_pyiqa_model(model_name, device)
    image = _prepare_tensor_image(
        img, crop_border, input_order, target_device, bgr2rgb)
    with torch.no_grad():
        return _scalar_score(model(image))


@METRIC_REGISTRY.register()
def calculate_dists(
        img,
        img2,
        crop_border=0,
        input_order='HWC',
        device='auto',
        bgr2rgb=True,
        **kwargs):
    '''Calculate the full-reference DISTS score; lower is better.'''
    if img2 is None:
        raise ValueError('DISTS requires a reference image.')
    if img.shape != img2.shape:
        raise ValueError(
            f'Image shapes differ: {img.shape} versus {img2.shape}.')
    model, target_device = _get_pyiqa_model('dists', device)
    image = _prepare_tensor_image(
        img, crop_border, input_order, target_device, bgr2rgb)
    reference = _prepare_tensor_image(
        img2, crop_border, input_order, target_device, bgr2rgb)
    with torch.no_grad():
        return _scalar_score(model(image, reference))


@METRIC_REGISTRY.register()
def calculate_musiq(
        img,
        img2=None,
        crop_border=0,
        input_order='HWC',
        device='auto',
        bgr2rgb=True,
        **kwargs):
    '''Calculate the no-reference MUSIQ score; higher is better.'''
    model, target_device = _get_pyiqa_model('musiq', device)
    image = _prepare_tensor_image(
        img, crop_border, input_order, target_device, bgr2rgb)
    with torch.no_grad():
        return _scalar_score(model(image))


@METRIC_REGISTRY.register()
def calculate_niqe_ours(
        img,
        img2=None,
        crop_border=0,
        input_order='HWC',
        device='auto',
        bgr2rgb=True,
        **kwargs):
    '''Calculate the PyIQA NIQE score; lower is better.'''
    model, target_device = _get_pyiqa_model('niqe', device)
    image = _prepare_tensor_image(
        img, crop_border, input_order, target_device, bgr2rgb)
    with torch.no_grad():
        return _scalar_score(model(image))
