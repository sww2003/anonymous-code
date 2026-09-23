'''Compact RealESRGAN + distribution matching distillation training model.'''

import copy
import os
import time
from collections import OrderedDict

import torch
from torch.nn import functional as F

from diffusers import DDPMScheduler, StableDiffusionPipeline
from peft import LoraConfig, get_peft_model

from basicsr.losses import build_loss
from basicsr.losses.loss_util import get_refined_artifact_map
from basicsr.models.realesrgan_model import RealESRGANModel as RealESRGANBaseModel
from basicsr.utils import get_root_logger
from basicsr.utils.dist_util import master_only
from basicsr.utils.dmd_util import (
    distributed_softmax_weights,
    residual_cosine_scores,
    sample_shifted_timesteps,
    standard_dmd_loss,
)
from basicsr.utils.registry import MODEL_REGISTRY


VAE_SCALING = 0.18215


def _torch_dtype(name):
    mapping = {
        'float32': torch.float32,
        'fp32': torch.float32,
        'float16': torch.float16,
        'fp16': torch.float16,
        'bfloat16': torch.bfloat16,
        'bf16': torch.bfloat16,
    }
    key = str(name).lower()
    if key not in mapping:
        raise ValueError(f'Unsupported score torch_dtype: {name}')
    return mapping[key]


@MODEL_REGISTRY.register()
class DMDSRModel(RealESRGANBaseModel):
    '''SwinIR/RealESRGAN training with fake-score, real-score, REPA, and DMD.'''

    def __init__(self, opt):
        super().__init__(opt)
        if not self.is_train:
            return
        if not hasattr(self, 'cri_gan'):
            raise ValueError('DMDSRModel requires train.gan_opt.')

        self._read_dmd_options()
        self._setup_score_models()
        self._setup_repa()
        self._log_batch_weighting_warning()

    @staticmethod
    def _optional_float(value):
        return None if value is None else float(value)

    def _read_dmd_options(self):
        dmd = self.opt.get('dmd', {}) or {}
        noise_shift = dmd.get('noise_shift', {}) or {}
        direction = dmd.get('direction', {}) or {}
        standard = dmd.get('standard', {}) or {}
        updates = self.opt.get('updates', {}) or {}
        score = self.opt.get('score', {}) or {}

        self.dmd_loss_weight = float(dmd.get(
            'loss_weight', self.opt.get('dmd_loss_weight', 1.0)))
        self.min_timestep = int(dmd.get(
            'min_timestep', self.opt.get('min_timesteps', 0)))
        self.max_timestep = int(dmd.get(
            'max_timestep', self.opt.get('max_timesteps', 200)))
        self.dmd_norm_eps = float(standard.get(
            'norm_eps', self.opt.get('dmd_standard_norm_eps', 1e-4)))
        self.dmd_grad_rms_max = self._optional_float(standard.get(
            'grad_rms_max', self.opt.get('dmd_standard_grad_rms_max', 0.5)))
        self.dmd_loss_clip_max = self._optional_float(standard.get(
            'loss_clip_max', self.opt.get('dmd_clip_raw_max', 0.5)))

        self.noise_shift_enabled = bool(noise_shift.get(
            'enabled', self.opt.get('noise_shift_enable', True)))
        self.noise_shift_switch_iter = int(noise_shift.get(
            'switch_iter', self.opt.get('noise_shift_switch_iter', 100000)))
        self.noise_shift_stage1 = float(noise_shift.get(
            'stage1', self.opt.get('noise_shift_s_stage1', 4.0)))
        self.noise_shift_stage2 = float(noise_shift.get(
            'stage2', self.opt.get('noise_shift_s_stage2', 0.5)))

        self.direction_enabled = bool(direction.get(
            'enabled', self.opt.get('dmd_directional_weighting', True)))
        self.direction_temperature = float(direction.get(
            'temperature', self.opt.get('dmd_directional_temperature', 0.2)))
        self.direction_min_weight = float(direction.get(
            'min_weight', self.opt.get('dmd_directional_min_weight', 0.2)))
        self.direction_max_weight = float(direction.get(
            'max_weight', self.opt.get('dmd_directional_max_weight', 3.0)))
        self.direction_eps = float(direction.get(
            'eps', self.opt.get('dmd_directional_eps', 1e-8)))

        self.g_interval = max(1, int(updates.get(
            'generator_interval', self.opt.get('net_g_update_interval', self.net_d_iters))))
        after_switch = updates.get(
            'generator_interval_after_switch',
            self.opt.get('net_g_update_interval_after_switch'))
        self.g_interval_after_switch = None if after_switch is None else max(1, int(after_switch))
        self.interval_switch_iter = int(updates.get(
            'switch_iter', self.opt.get('fake_g_ratio_switch_iter', 0)))
        self.score_interval = max(1, int(updates.get(
            'score_interval', self.opt.get('score_update_interval', 1))))
        self.d_interval = max(1, int(updates.get(
            'discriminator_interval', self.opt.get('net_d_update_interval', 1))))
        self.g_init_iters = max(
            int(updates.get('freeze_generator_iters', self.opt.get('freeze_net_g_iters', 0))),
            int(self.net_d_init_iters))
        self.d_init_iters = max(
            int(updates.get('freeze_discriminator_iters', self.opt.get('freeze_net_d_iters', 0))),
            int(self.net_d_init_iters))

        self.score_model_path = score.get('model_path', self.opt.get('sd_model_path'))
        if not self.score_model_path:
            raise ValueError('score.model_path must point to a Stable Diffusion directory.')
        self.score_dtype = _torch_dtype(score.get('torch_dtype', 'float32'))
        self.fixed_prompt = str(score.get(
            'prompt', self.opt.get('fixed_prompts', '')))
        self.fake_score_cfg = score.get('fake', {}) or {}
        self.real_score_cfg = score.get('real', {}) or {}
        self.repa_cfg = score.get('repa', {}) or {}
        self.score_checkpoint_name = str(score.get('checkpoint_name', 'score_latest.pth'))

    def _log_batch_weighting_warning(self):
        train_data = self.opt.get('datasets', {}).get('train', {}) or {}
        local_batch = int(train_data.get('batch_size_per_gpu', 1))
        world_size = int(self.opt.get('world_size', 1))
        if self.direction_enabled and local_batch * world_size == 1:
            get_root_logger().warning(
                'Directional weighting is enabled with global batch size 1; '
                'its softmax weight is always exactly 1.')

    @staticmethod
    def _optimizer_config(config, default_lr):
        result = dict(config or {})
        result.setdefault('type', 'AdamW')
        result.setdefault('lr', float(default_lr))
        result.setdefault('betas', (0.9, 0.999))
        result.setdefault('eps', 1e-8)
        result.setdefault('weight_decay', 0.01)
        return result

    def _build_optimizer(self, config, parameters):
        config = dict(config)
        optimizer_type = config.pop('type')
        return self.get_optimizer(optimizer_type, parameters, **config)

    def _setup_score_models(self):
        logger = get_root_logger()
        logger.info(f'Loading score backbone from {self.score_model_path}')
        pipe = StableDiffusionPipeline.from_pretrained(
            self.score_model_path,
            torch_dtype=self.score_dtype,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
        self.score_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

        self.score_vae = pipe.vae.requires_grad_(False).eval().to(self.device)
        pipe.vae = None

        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder.requires_grad_(False).eval().to(self.device)
        tokens = tokenizer(
            [self.fixed_prompt],
            padding='max_length',
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt',
        )
        with torch.no_grad():
            self.prompt_embedding = text_encoder(
                tokens.input_ids.to(self.device))[0].detach()
        text_encoder.to('cpu')
        pipe.text_encoder = None
        pipe.tokenizer = None
        del text_encoder, tokenizer

        base_unet = pipe.unet
        pipe.unet = None
        del pipe
        base_unet.requires_grad_(False).eval()

        fake_unet = copy.deepcopy(base_unet).requires_grad_(True).train()
        self.fake_score_unet = self.model_to_device(fake_unet)

        lora_targets = self.real_score_cfg.get(
            'target_modules',
            self.opt.get(
                'real_lora_target_modules',
                ['to_q', 'to_k', 'to_v', 'to_out.0', 'proj_in', 'proj_out']))
        lora_config = LoraConfig(
            r=int(self.real_score_cfg.get(
                'rank', self.opt.get('real_lora_rank', 16))),
            lora_alpha=int(self.real_score_cfg.get(
                'alpha', self.opt.get('real_lora_alpha', 32))),
            lora_dropout=float(self.real_score_cfg.get(
                'dropout', self.opt.get('real_lora_dropout', 0.0))),
            target_modules=list(lora_targets),
            bias='none',
        )
        real_unet = get_peft_model(base_unet, lora_config).train()
        self.real_score_unet = self.model_to_device(real_unet)

        fake_lr = float(self.fake_score_cfg.get(
            'lr', self.opt.get('lr_fake_score', 3e-5)))
        real_lr = float(self.real_score_cfg.get(
            'lr', self.opt.get('lr_real_score', fake_lr)))
        fake_optim_cfg = self._optimizer_config(
            self.fake_score_cfg.get('optimizer'), fake_lr)
        real_optim_cfg = self._optimizer_config(
            self.real_score_cfg.get('optimizer'), real_lr)
        self.optimizer_fake_score = self._build_optimizer(
            fake_optim_cfg, self.fake_score_unet.parameters())
        real_parameters = [
            parameter for parameter in self.real_score_unet.parameters()
            if parameter.requires_grad]
        if not real_parameters:
            raise RuntimeError('The real-score LoRA model has no trainable parameters.')
        self.optimizer_real_score = self._build_optimizer(real_optim_cfg, real_parameters)

        self.score_alphas_cumprod = self.score_scheduler.alphas_cumprod.to(
            device=self.device, dtype=torch.float32)
        self.score_dtype = next(
            self.get_bare_model(self.fake_score_unet).parameters()).dtype
        fake_count = sum(
            parameter.numel() for parameter in self.fake_score_unet.parameters()
            if parameter.requires_grad)
        real_count = sum(parameter.numel() for parameter in real_parameters)
        logger.info(
            f'Score models ready: fake full={fake_count:,}, real LoRA={real_count:,}')

    @staticmethod
    def _select_hook_output(value, output_path):
        if output_path is None:
            return value
        keys = output_path if isinstance(output_path, (list, tuple)) else [output_path]
        for key in keys:
            if isinstance(value, (list, tuple)):
                value = value[int(key)]
            elif isinstance(value, dict):
                value = value[key]
            else:
                value = getattr(value, str(key))
        return value

    def _resolve_fake_score_module(self, module_path):
        modules = dict(self.get_bare_model(self.fake_score_unet).named_modules())
        if module_path in modules:
            return modules[module_path]
        matches = [
            (name, module) for name, module in modules.items()
            if name.endswith(f'.{module_path}')]
        if len(matches) == 1:
            return matches[0][1]
        if len(matches) > 1:
            raise KeyError(
                f'Ambiguous REPA module {module_path}: {[name for name, _ in matches]}')
        raise KeyError(
            f'Cannot find REPA module {module_path}. '
            f'Examples: {[name for name in modules if name][:30]}')

    def _setup_repa(self):
        self.cri_score_repa = None
        self._repa_feature = None
        self._capture_repa = False
        self._repa_hook = None
        if not self.repa_cfg or not bool(self.repa_cfg.get('enabled', True)):
            return

        config = dict(self.repa_cfg)
        config.pop('enabled', None)
        module_path = config.pop('module', 'down_blocks.0')
        output_path = config.pop('output_path', 0)
        config.pop('name', None)
        config.setdefault('type', 'DMDRepaLoss')
        self.cri_score_repa = self.model_to_device(build_loss(config))
        trainable = [
            parameter for parameter in self.cri_score_repa.parameters()
            if parameter.requires_grad]
        if not trainable:
            raise RuntimeError('DMDRepaLoss has no trainable projection parameters.')
        self.optimizer_fake_score.add_param_group({'params': trainable})

        module = self._resolve_fake_score_module(module_path)

        def capture_feature(_module, _inputs, output):
            if self._capture_repa:
                self._repa_feature = self._select_hook_output(output, output_path)

        self._repa_hook = module.register_forward_hook(capture_feature)
        get_root_logger().info(f'REPA hook: fake score {module_path} -> {output_path}')

    def _fake_score_training_forward(self, noisy_latent, timesteps, text_embedding):
        self._repa_feature = None
        self._capture_repa = self.cri_score_repa is not None
        try:
            prediction = self.fake_score_unet(
                noisy_latent, timesteps,
                encoder_hidden_states=text_embedding).sample
        finally:
            self._capture_repa = False
        if self.cri_score_repa is not None and self._repa_feature is None:
            raise RuntimeError('REPA is enabled, but its fake-score hook captured no feature.')
        return prediction

    def _compute_repa_loss(self, target_image):
        if self.cri_score_repa is None:
            return None
        return self.cri_score_repa(self._repa_feature, target_image)

    @staticmethod
    def _updates_at(current_iter, interval, init_iters=0):
        return current_iter > init_iters and current_iter % interval == 0

    def _current_g_interval(self, current_iter):
        if (
                self.g_interval_after_switch is not None
                and self.interval_switch_iter > 0
                and current_iter >= self.interval_switch_iter):
            return self.g_interval_after_switch
        return self.g_interval

    def _current_noise_shift(self, current_iter):
        if not self.noise_shift_enabled:
            return 1.0
        if current_iter <= self.noise_shift_switch_iter:
            return self.noise_shift_stage1
        return self.noise_shift_stage2

    def _sample_timesteps(self, batch_size, current_iter):
        total = int(self.score_scheduler.config.num_train_timesteps)
        minimum = max(0, min(self.min_timestep, total - 1))
        maximum = min(max(self.max_timestep, minimum + 1), total)
        shift = self._current_noise_shift(current_iter)
        timesteps, base = sample_shifted_timesteps(
            batch_size, minimum, maximum, shift, self.device)
        stats = {
            'dmd_timestep_mean': timesteps.float().mean().detach(),
            'dmd_timestep_min': timesteps.float().min().detach(),
            'dmd_timestep_max': timesteps.float().max().detach(),
            'dmd_timestep_base_mean': base.float().mean().detach(),
            'dmd_noise_shift': timesteps.new_tensor(shift, dtype=torch.float32),
        }
        return timesteps, stats

    def _batch_prompt_embedding(self, batch_size):
        return self.prompt_embedding.to(dtype=self.score_dtype).expand(
            batch_size, -1, -1)

    def _encode_image(self, image, require_gradient):
        vae_dtype = next(self.score_vae.parameters()).dtype
        image = (image * 2.0 - 1.0).to(dtype=vae_dtype)
        if require_gradient:
            latent = self.score_vae.encode(image).latent_dist.sample()
        else:
            with torch.no_grad():
                latent = self.score_vae.encode(image).latent_dist.sample()
        return latent * VAE_SCALING

    @staticmethod
    def _match_real_image(real_image, generated_image):
        real_image = real_image.detach()
        if real_image.shape[-2:] != generated_image.shape[-2:]:
            real_image = F.interpolate(
                real_image, size=generated_image.shape[-2:],
                mode='bicubic', align_corners=False)
        return real_image.clamp(0.0, 1.0)

    def _bare_score_forward(self, model, latent, timesteps, text_embedding):
        return self.get_bare_model(model)(
            latent, timesteps, encoder_hidden_states=text_embedding).sample

    @staticmethod
    def _prediction_mean(prediction):
        if isinstance(prediction, (list, tuple)):
            values = []
            for item in prediction:
                if isinstance(item, (list, tuple)):
                    item = item[-1]
                values.append(item.detach().mean())
            return sum(values) / len(values)
        return prediction.detach().mean()

    def _compute_score_and_dmd(
            self, generated_image, real_image, current_iter, update_score, update_g):
        generated_image = generated_image.clamp(0.0, 1.0)
        real_image = self._match_real_image(real_image, generated_image)
        batch_size = generated_image.shape[0]
        generated_latent = self._encode_image(
            generated_image, require_gradient=update_g)
        timesteps, stats = self._sample_timesteps(batch_size, current_iter)
        text_embedding = self._batch_prompt_embedding(batch_size)
        noise = torch.randn_like(generated_latent)
        noisy_generated = self.score_scheduler.add_noise(
            generated_latent, noise, timesteps)

        fake_prediction_train = None
        fake_loss = None
        if update_score:
            fake_prediction_train = self._fake_score_training_forward(
                noisy_generated.detach(), timesteps, text_embedding)
            fake_noise_loss = F.mse_loss(
                fake_prediction_train.float(), noise.float())
            repa_loss = self._compute_repa_loss(generated_image.detach())
            self._repa_feature = None
            fake_loss = fake_noise_loss
            if repa_loss is not None:
                fake_loss = fake_loss + repa_loss
                stats['l_score_fake_repa'] = repa_loss.detach()
            stats['l_score_fake_noise'] = fake_noise_loss.detach()

        need_real_latent = update_score or (update_g and self.direction_enabled)
        real_latent = None
        if need_real_latent:
            real_latent = self._encode_image(real_image, require_gradient=False)

        real_loss = None
        if update_score:
            real_noise = torch.randn_like(real_latent)
            noisy_real_train = self.score_scheduler.add_noise(
                real_latent, real_noise, timesteps)
            real_prediction_train = self.real_score_unet(
                noisy_real_train.detach(), timesteps,
                encoder_hidden_states=text_embedding).sample
            real_loss = F.mse_loss(
                real_prediction_train.float(), real_noise.float())
            stats['l_score_real'] = real_loss.detach()

        result = {
            'fake_loss': fake_loss,
            'real_loss': real_loss,
            'dmd_loss': None,
            'stats': stats,
        }
        if not update_g:
            return result

        with torch.no_grad():
            if fake_prediction_train is None:
                fake_prediction_dmd = self._bare_score_forward(
                    self.fake_score_unet, noisy_generated.detach(),
                    timesteps, text_embedding)
            else:
                fake_prediction_dmd = fake_prediction_train.detach()

            real_prediction_dmd = self._bare_score_forward(
                self.real_score_unet, noisy_generated.detach(),
                timesteps, text_embedding)

            if self.direction_enabled:
                noisy_real_direction = self.score_scheduler.add_noise(
                    real_latent, noise.detach(), timesteps)
                fake_prediction_real = self._bare_score_forward(
                    self.fake_score_unet, noisy_real_direction,
                    timesteps, text_embedding)
                cosine = residual_cosine_scores(
                    fake_prediction_dmd, fake_prediction_real, noise,
                    eps=self.direction_eps)
                sample_weights = distributed_softmax_weights(
                    cosine,
                    temperature=self.direction_temperature,
                    min_weight=self.direction_min_weight,
                    max_weight=self.direction_max_weight,
                    eps=self.direction_eps)
                stats.update({
                    'dmd_direction_cosine': cosine.mean().detach(),
                    'dmd_direction_weight': sample_weights.mean().detach(),
                    'dmd_direction_weight_min': sample_weights.min().detach(),
                    'dmd_direction_weight_max': sample_weights.max().detach(),
                })
            else:
                sample_weights = generated_latent.new_ones(batch_size)

            fake_error = F.mse_loss(
                fake_prediction_dmd.float(), noise.float())
            real_error = F.mse_loss(
                real_prediction_dmd.float(), noise.float())
            stats['dmd_fake_to_noise_mse'] = fake_error.detach()
            stats['dmd_real_to_noise_mse'] = real_error.detach()
            stats['dmd_real_fake_score_gap'] = (fake_error - real_error).detach()

        dmd_loss, dmd_stats = standard_dmd_loss(
            generated_latent=generated_latent,
            noisy_latent=noisy_generated,
            real_noise_prediction=real_prediction_dmd,
            fake_noise_prediction=fake_prediction_dmd,
            alphas_cumprod=self.score_alphas_cumprod[timesteps],
            sample_weights=sample_weights,
            norm_eps=self.dmd_norm_eps,
            grad_rms_max=self.dmd_grad_rms_max,
            loss_clip_max=self.dmd_loss_clip_max,
        )
        stats.update(dmd_stats)
        result['dmd_loss'] = dmd_loss
        return result

    def optimize_parameters(self, current_iter):
        l1_gt = self.gt_usm if self.opt.get('l1_gt_usm', True) else self.gt
        percep_gt = self.gt_usm if self.opt.get('percep_gt_usm', True) else self.gt
        gan_gt = self.gt_usm if self.opt.get('gan_gt_usm', False) else self.gt

        g_interval = self._current_g_interval(current_iter)
        update_g = self._updates_at(current_iter, g_interval, self.g_init_iters)
        update_score = self._updates_at(current_iter, self.score_interval)
        update_d = self._updates_at(current_iter, self.d_interval, self.d_init_iters)

        for parameter in self.net_d.parameters():
            parameter.requires_grad_(False)
        self.optimizer_g.zero_grad(set_to_none=True)
        if update_g:
            self.output = self.net_g(self.lq)
        else:
            with torch.no_grad():
                self.output = self.net_g(self.lq)

        loss_dict = OrderedDict()
        loss_dict['update_g'] = self.output.new_tensor(float(update_g))
        loss_dict['update_score'] = self.output.new_tensor(float(update_score))
        loss_dict['update_d'] = self.output.new_tensor(float(update_d))
        loss_dict['g_interval'] = self.output.new_tensor(float(g_interval))

        if update_score:
            self.optimizer_fake_score.zero_grad(set_to_none=True)
            self.optimizer_real_score.zero_grad(set_to_none=True)

        score_result = None
        if update_score or update_g:
            score_result = self._compute_score_and_dmd(
                self.output, self.gt, current_iter, update_score, update_g)
            for name, value in score_result['stats'].items():
                loss_dict[name] = value

        if update_score:
            fake_loss = score_result['fake_loss']
            real_loss = score_result['real_loss']
            fake_loss.backward()
            self.optimizer_fake_score.step()
            real_loss.backward()
            self.optimizer_real_score.step()
            loss_dict['l_score_fake'] = fake_loss.detach()
            loss_dict['l_score_real'] = real_loss.detach()

        if update_g:
            generator_total = self.output.new_zeros(())
            if self.cri_pix:
                pixel_loss = self.cri_pix(self.output, l1_gt)
                generator_total = generator_total + pixel_loss
                loss_dict['l_g_pix'] = pixel_loss

            if self.cri_ldl:
                with torch.no_grad():
                    self.output_ema = self.net_g_ema(self.lq)
                pixel_weight = get_refined_artifact_map(
                    self.gt, self.output, self.output_ema, 7)
                ldl_loss = self.cri_ldl(
                    pixel_weight * self.output, pixel_weight * self.gt)
                generator_total = generator_total + ldl_loss
                loss_dict['l_g_ldl'] = ldl_loss

            if self.cri_perceptual:
                perceptual_loss, style_loss = self.cri_perceptual(
                    self.output, percep_gt)
                if perceptual_loss is not None:
                    generator_total = generator_total + perceptual_loss
                    loss_dict['l_g_percep'] = perceptual_loss
                if style_loss is not None:
                    generator_total = generator_total + style_loss
                    loss_dict['l_g_style'] = style_loss

            fake_prediction = self.net_d(self.output)
            gan_loss = self.cri_gan(fake_prediction, True, is_disc=False)
            generator_total = generator_total + gan_loss
            loss_dict['l_g_gan'] = gan_loss

            weighted_dmd = score_result['dmd_loss'] * self.dmd_loss_weight
            generator_total = generator_total + weighted_dmd
            loss_dict['l_dmd'] = weighted_dmd
            loss_dict['l_g_total'] = generator_total

            generator_total.backward()
            self.optimizer_g.step()

        for parameter in self.net_d.parameters():
            parameter.requires_grad_(True)

        if update_d:
            self.optimizer_d.zero_grad(set_to_none=True)
            real_prediction = self.net_d(gan_gt)
            discriminator_real = self.cri_gan(
                real_prediction, True, is_disc=True)
            discriminator_real.backward()
            loss_dict['l_d_real'] = discriminator_real
            loss_dict['out_d_real'] = self._prediction_mean(real_prediction)

            fake_prediction = self.net_d(self.output.detach().clone())
            discriminator_fake = self.cri_gan(
                fake_prediction, False, is_disc=True)
            discriminator_fake.backward()
            self.optimizer_d.step()
            loss_dict['l_d_fake'] = discriminator_fake
            loss_dict['out_d_fake'] = self._prediction_mean(fake_prediction)

        if update_g and self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
        self.log_dict = self.reduce_loss_dict(loss_dict)

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network(
                [self.net_g, self.net_g_ema], 'net_g', current_iter,
                param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_network(self.net_d, 'net_d', current_iter)
        self._save_score_checkpoint(current_iter)
        self.save_training_state(epoch, current_iter)

    @staticmethod
    def _cpu_state_dict(module):
        state = {}
        for name, value in module.state_dict().items():
            if name.startswith('module.'):
                name = name[7:]
            state[name] = value.detach().cpu()
        return state

    @staticmethod
    def _cpu_trainable_state(module):
        return {
            name: parameter.detach().cpu()
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        }

    @master_only
    def _save_score_checkpoint(self, current_iter):
        if current_iter == -1:
            return
        save_path = os.path.join(
            self.opt['path']['models'], self.score_checkpoint_name)
        temporary_path = save_path + '.tmp'
        checkpoint = {
            'iter': current_iter,
            'student_unet': self._cpu_state_dict(
                self.get_bare_model(self.fake_score_unet)),
            'real_lora': self._cpu_trainable_state(
                self.get_bare_model(self.real_score_unet)),
            'optimizer_student': self.optimizer_fake_score.state_dict(),
            'optimizer_real': self.optimizer_real_score.state_dict(),
        }
        if self.cri_score_repa is not None:
            checkpoint['student_repa_loss_trainable'] = self._cpu_trainable_state(
                self.get_bare_model(self.cri_score_repa))

        last_error = None
        for attempt in range(3):
            try:
                torch.save(checkpoint, temporary_path)
                os.replace(temporary_path, save_path)
                return
            except Exception as error:
                last_error = error
                get_root_logger().warning(
                    f'Cannot save score checkpoint, attempt {attempt + 1}/3: {error}')
                time.sleep(1)
        raise IOError(f'Cannot save score checkpoint to {save_path}') from last_error

    @master_only
    def save_training_state(self, epoch, current_iter):
        if current_iter == -1:
            return
        state = {
            'epoch': epoch,
            'iter': current_iter,
            'optimizers': [optimizer.state_dict() for optimizer in self.optimizers],
            'schedulers': [scheduler.state_dict() for scheduler in self.schedulers],
            'score_checkpoint': self.score_checkpoint_name,
            'score_checkpoint_iter': current_iter,
        }
        save_path = os.path.join(
            self.opt['path']['training_states'], f'{current_iter}.state')
        last_error = None
        for attempt in range(3):
            try:
                torch.save(state, save_path)
                return
            except Exception as error:
                last_error = error
                get_root_logger().warning(
                    f'Cannot save training state, attempt {attempt + 1}/3: {error}')
                time.sleep(1)
        raise IOError(f'Cannot save training state to {save_path}') from last_error

    def resume_training(self, resume_state):
        super().resume_training(resume_state)
        score_name = resume_state.get(
            'score_checkpoint', self.score_checkpoint_name)
        score_path = score_name
        if not os.path.isabs(score_path):
            score_path = os.path.join(self.opt['path']['models'], score_path)
        if not os.path.isfile(score_path):
            raise FileNotFoundError(
                f'Score checkpoint required by the training state was not found: {score_path}')

        checkpoint = torch.load(score_path, map_location='cpu')
        expected_iter = resume_state.get(
            'score_checkpoint_iter', resume_state.get('iter'))
        score_iter = checkpoint.get('iter')
        if expected_iter is not None and score_iter is not None:
            if int(expected_iter) != int(score_iter):
                raise RuntimeError(
                    f'Score checkpoint iteration mismatch: expected {expected_iter}, '
                    f'found {score_iter} in {score_path}.')

        self.get_bare_model(self.fake_score_unet).load_state_dict(
            checkpoint['student_unet'], strict=True)
        self.get_bare_model(self.real_score_unet).load_state_dict(
            checkpoint['real_lora'], strict=False)
        if self.cri_score_repa is not None:
            repa_state = checkpoint.get('student_repa_loss_trainable')
            if repa_state is None:
                get_root_logger().warning(
                    'Score checkpoint has no REPA projection state; using initialization.')
            else:
                self.get_bare_model(self.cri_score_repa).load_state_dict(
                    repa_state, strict=False)

        self.optimizer_fake_score.load_state_dict(
            checkpoint['optimizer_student'])
        self.optimizer_real_score.load_state_dict(
            checkpoint['optimizer_real'])
        get_root_logger().info(f'Loaded score checkpoint from {score_path}')
