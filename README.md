# PhoenixSR

**PhoenixSR: Generative Heterogeneous Distillation Unleashes Efficient Models
for Real-World Super-Resolution**

Anonymous research implementation of distribution matching distillation for
real-world image super-resolution. The training pipeline is built on BasicSR
and combines a SwinIR generator, RealESRGAN-style degradation and adversarial
training, latent-space DMD, directional sample weighting, and optional
DINOv3-based representation alignment (REPA).

[中文说明](README_CN.md)

## Method overview

The compact implementation keeps one training entry point:
`basicsr/models/dmd_sr_model.py`.

- The generator is trained with pixel, perceptual, GAN, and DMD objectives.
- A fully trainable fake-score UNet models generated images.
- A LoRA-adapted real-score UNet models the real-image distribution.
- Standard latent DMD uses the difference between the two predicted clean
  latents as its surrogate gradient.
- Optional REPA aligns a fake-score feature map with frozen DINOv3 patch
  tokens.
- Validation supports PSNR, SSIM, NIQE, and the no-reference MANIQA metric.

For directional weighting, the raw per-sample score is

```text
s_i = cosine(real-image denoising residual, fake-image denoising residual)
```

so the original similarity is in `[-1, 1]`. The implementation converts the
scores into global, mean-one weights with
`N * softmax(s / temperature)`, then clamps them to
`[min_weight, max_weight]`. With a global batch size of one, the softmax
weight is always one.

## Repository layout

```text
basicsr/models/dmd_sr_model.py       compact training model
basicsr/utils/dmd_util.py            timestep, weighting, and DMD math
basicsr/losses/dmd_repa_loss.py      optional DINOv3 REPA objective
basicsr/metrics/maniqa.py            lazy PyIQA metric wrappers
options/train/OURS/dmd_sr_swinir.yml canonical experiment configuration
tests/test_dmd_util.py                unit tests for the DMD utilities
check_anonymity.py                    release privacy audit
```

Reference trees, local model repositories, datasets, checkpoints, and
experiment outputs are intentionally excluded from the anonymous release.

## Installation

Python 3.10 or 3.11 is recommended. Install the PyTorch build appropriate for
the local CUDA runtime first, then run:

```bash
pip install -r requirements.txt
pip install -e .
```

PyIQA loads its metric weights on first use. Disable the MANIQA entry in the
configuration when network access and a cached checkpoint are unavailable.

## External assets

No datasets or pretrained weights are committed. Prepare the following local
layout, or change the corresponding relative paths in
`options/train/OURS/dmd_sr_swinir.yml`:

```text
datasets/
├── train/HR/
└── RealSR/
    ├── LR/
    └── HR/

pretrained_models/
├── stable-diffusion-2-1/
├── dinov3_vitb16_pretrain_lvd1689m.pth
└── swinir_real_sr_x4.pth

dinov3/
└── dinov3/                         compatible DINOv3 source package
```

The Stable Diffusion directory must be loadable by Diffusers. The DINOv3
checkpoint must match `dino_model_name` and `target_dim`. Set
`score.repa.enabled: false` to train without DINOv3/REPA.

## Training

```bash
python basicsr/train.py \
  -opt options/train/OURS/dmd_sr_swinir.yml \
  --launcher none
```

For distributed training, use the normal BasicSR launcher and keep equal local
batch sizes on every rank. Directional weighting performs a global softmax and
therefore requires equal per-rank sample counts.

## Verification

```bash
python tests/test_dmd_util.py
python tests/test_metrics/test_maniqa.py
python -m py_compile \
  basicsr/models/dmd_sr_model.py \
  basicsr/utils/dmd_util.py \
  basicsr/losses/dmd_repa_loss.py \
  basicsr/metrics/maniqa.py
python check_anonymity.py
```

The full training smoke test additionally requires the external assets listed
above and a CUDA environment.

## Anonymous release checklist

1. Put any private names, account handles, hostnames, or organization terms
   into `.private_terms.txt`, one term per line. This local file is
   ignored by Git.
2. Run `python check_anonymity.py`.
3. Initialize a fresh repository instead of reusing personal commit history.
4. Configure a non-identifying Git author name and no-reply address locally.
5. Stage the release, then run
   `python check_anonymity.py --staged` before every push.
6. Inspect the staged file list and remote URL manually. GitHub account names,
   commit signatures, issue links, and release assets can reveal identity even
   when source files are clean.

## License and third-party code

This repository is distributed under the Apache-2.0 license. It builds on
BasicSR and contains code adapted from several open-source projects. Required
third-party notices and licenses are retained under `LICENSE/` and
`LICENSE.txt`. Those attribution records identify upstream projects and must
not be removed as part of author anonymization.
