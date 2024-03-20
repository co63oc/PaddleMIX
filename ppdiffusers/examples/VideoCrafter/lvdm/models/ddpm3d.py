# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle

"""
wild mixture of
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/CompVis/taming-transformers
-- merci
"""
import logging
from contextlib import contextmanager
from functools import partial

import numpy as np
from einops import rearrange, repeat
from tqdm import tqdm

mainlogger = logging.getLogger("mainlogger")
from lvdm.basics import disabled_train
from lvdm.common import default, exists, extract_into_tensor, noise_like
from lvdm.distributions import DiagonalGaussianDistribution
from lvdm.ema import LitEma
from lvdm.models.utils_diffusion import make_beta_schedule
from lvdm.modules.encoders.ip_resampler import ImageProjModel, Resampler
from utils.utils import instantiate_from_config, randn_tensor_mod

from ppdiffusers.pipelines.lvdm.video_save import make_grid

__conditioning_keys__ = {"concat": "c_concat", "crossattn": "c_crossattn", "adm": "y"}


class DDPM(paddle.nn.Layer):
    def __init__(
        self,
        unet_config,
        timesteps=1000,
        beta_schedule="linear",
        loss_type="l2",
        ckpt_path=None,
        ignore_keys=[],
        load_only_unet=False,
        monitor=None,
        use_ema=True,
        first_stage_key="image",
        image_size=256,
        channels=3,
        log_every_t=100,
        clip_denoised=True,
        linear_start=0.0001,
        linear_end=0.02,
        cosine_s=0.008,
        given_betas=None,
        original_elbo_weight=0.0,
        v_posterior=0.0,
        l_simple_weight=1.0,
        conditioning_key=None,
        parameterization="eps",
        scheduler_config=None,
        use_positional_encodings=False,
        learn_logvar=False,
        logvar_init=0.0,
    ):
        super().__init__()
        assert parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        mainlogger.info(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.channels = channels
        self.temporal_length = unet_config.params.temporal_length
        self.image_size = image_size
        if isinstance(self.image_size, int):
            self.image_size = [self.image_size, self.image_size]
        self.use_positional_encodings = use_positional_encodings
        self.model = DiffusionWrapper(unet_config, conditioning_key)
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            mainlogger.info(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")
        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config
        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)
        self.register_schedule(
            given_betas=given_betas,
            beta_schedule=beta_schedule,
            timesteps=timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )
        self.loss_type = loss_type
        self.learn_logvar = learn_logvar
        self.logvar = paddle.full(fill_value=logvar_init, shape=(self.num_timesteps,))
        if self.learn_logvar:
            out_0 = paddle.create_parameter(
                shape=self.logvar.shape,
                dtype=self.logvar.numpy().dtype,
                default_initializer=paddle.nn.initializer.Assign(self.logvar),
            )
            out_0.stop_gradient = not True
            self.logvar = out_0

    def register_schedule(
        self,
        given_betas=None,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=0.0001,
        linear_end=0.02,
        cosine_s=0.008,
    ):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(
                beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s
            )
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        (timesteps,) = tuple(betas.shape)
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert tuple(alphas_cumprod.shape)[0] == self.num_timesteps, "alphas have to be defined for each timestep"
        to_paddle = partial(paddle.to_tensor, dtype="float32")
        self.register_buffer(name="betas", tensor=to_paddle(betas))
        self.register_buffer(name="alphas_cumprod", tensor=to_paddle(alphas_cumprod))
        self.register_buffer(name="alphas_cumprod_prev", tensor=to_paddle(alphas_cumprod_prev))
        self.register_buffer(name="sqrt_alphas_cumprod", tensor=to_paddle(np.sqrt(alphas_cumprod)))
        self.register_buffer(name="sqrt_one_minus_alphas_cumprod", tensor=to_paddle(np.sqrt(1.0 - alphas_cumprod)))
        self.register_buffer(name="log_one_minus_alphas_cumprod", tensor=to_paddle(np.log(1.0 - alphas_cumprod)))
        self.register_buffer(name="sqrt_recip_alphas_cumprod", tensor=to_paddle(np.sqrt(1.0 / alphas_cumprod)))
        self.register_buffer(name="sqrt_recipm1_alphas_cumprod", tensor=to_paddle(np.sqrt(1.0 / alphas_cumprod - 1)))
        posterior_variance = (1 - self.v_posterior) * betas * (1.0 - alphas_cumprod_prev) / (
            1.0 - alphas_cumprod
        ) + self.v_posterior * betas
        self.register_buffer(name="posterior_variance", tensor=to_paddle(posterior_variance))
        self.register_buffer(
            name="posterior_log_variance_clipped", tensor=to_paddle(np.log(np.maximum(posterior_variance, 1e-20)))
        )
        self.register_buffer(
            name="posterior_mean_coef1",
            tensor=to_paddle(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            name="posterior_mean_coef2",
            tensor=to_paddle((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)),
        )
        if self.parameterization == "eps":
            lvlb_weights = self.betas**2 / (
                2 * self.posterior_variance * to_paddle(alphas) * (1 - self.alphas_cumprod)
            )
        elif self.parameterization == "x0":
            lvlb_weights = (
                0.5
                * np.sqrt(paddle.to_tensor(data=alphas_cumprod))
                / (2.0 * 1 - paddle.to_tensor(data=alphas_cumprod))
            )
        else:
            raise NotImplementedError("mu not supported")
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer(name="lvlb_weights", tensor=lvlb_weights, persistable=False)
        assert not paddle.isnan(x=self.lvlb_weights).astype("bool").all()

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                mainlogger.info(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    mainlogger.info(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = paddle.load(path=path)
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    mainlogger.info("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = (
            self.set_state_dict(state_dict=sd, use_structured_name=False)
            if not only_model
            else self.model.set_state_dict(state_dict=sd, use_structured_name=False)
        )
        mainlogger.info(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            mainlogger.info(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            mainlogger.info(f"Unexpected Keys: {unexpected}")

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = extract_into_tensor(self.sqrt_alphas_cumprod, t, tuple(x_start.shape)) * x_start
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, tuple(x_start.shape))
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, tuple(x_start.shape))
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, tuple(x_t.shape)) * x_t
            - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, tuple(x_t.shape)) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, tuple(x_t.shape)) * x_start
            + extract_into_tensor(self.posterior_mean_coef2, t, tuple(x_t.shape)) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, tuple(x_t.shape))
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, tuple(x_t.shape))
        return (posterior_mean, posterior_variance, posterior_log_variance_clipped)

    def p_mean_variance(self, x, t, clip_denoised: bool):
        model_out = self.model(x, t)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clip_(min=-1.0, max=1.0)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @paddle.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_, device = *tuple(x.shape), x.place
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(tuple(x.shape), device, repeat_noise)
        nonzero_mask = (1 - (t == 0).astype(dtype="float32")).reshape([b, *((1,) * (len(tuple(x.shape)) - 1))])
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @paddle.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):
        # device = self.betas.place
        b = shape[0]
        img = randn_tensor_mod(shape=shape)
        intermediates = [img]
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc="Sampling t", total=self.num_timesteps):
            img = self.p_sample(
                img, paddle.full(shape=(b,), fill_value=i, dtype="int64"), clip_denoised=self.clip_denoised
            )
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    @paddle.no_grad()
    def sample(self, batch_size=16, return_intermediates=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop(
            (batch_size, channels, image_size, image_size), return_intermediates=return_intermediates
        )

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: randn_tensor_mod(shape=x_start.shape, dtype=x_start.dtype))
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, tuple(x_start.shape))
            * x_start
            * extract_into_tensor(self.scale_arr, t, tuple(x_start.shape))
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, tuple(x_start.shape)) * noise
        )

    def get_input(self, batch, k):
        x = batch[k]
        x = x.to().astype(dtype="float32")
        return x

    def _get_rows_from_list(self, samples):
        n_imgs_per_row = len(samples)
        denoise_grid = rearrange(samples, "n b c h w -> b n c h w")
        denoise_grid = rearrange(denoise_grid, "b n c h w -> (b n) c h w")
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    @paddle.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs):
        log = dict()
        x = self.get_input(batch, self.first_stage_key)
        N = min(tuple(x.shape)[0], N)
        n_row = min(tuple(x.shape)[0], n_row)
        x = x.to(self.device)[:N]
        log["inputs"] = x
        diffusion_row = list()
        x_start = x[:n_row]
        for t in range(self.num_timesteps):
            if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                t = repeat(paddle.to_tensor(data=[t]), "1 -> b", b=n_row)
                t = t.to(self.device).astype(dtype="int64")
                noise = randn_tensor_mod(shape=x_start.shape, dtype=x_start.dtype)
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                diffusion_row.append(x_noisy)
        log["diffusion_row"] = self._get_rows_from_list(diffusion_row)
        if sample:
            with self.ema_scope("Plotting"):
                samples, denoise_row = self.sample(batch_size=N, return_intermediates=True)
            log["samples"] = samples
            log["denoise_row"] = self._get_rows_from_list(denoise_row)
        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log


class LatentDiffusion(DDPM):
    """main class"""

    def __init__(
        self,
        first_stage_config,
        cond_stage_config,
        num_timesteps_cond=None,
        cond_stage_key="caption",
        cond_stage_trainable=False,
        cond_stage_forward=None,
        conditioning_key=None,
        uncond_prob=0.2,
        uncond_type="empty_seq",
        scale_factor=1.0,
        scale_by_std=False,
        encoder_type="2d",
        only_model=False,
        use_scale=False,
        scale_a=1,
        scale_b=0.3,
        mid_step=400,
        fix_scale_bug=False,
        *args,
        **kwargs
    ):
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= kwargs["timesteps"]
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        conditioning_key = default(conditioning_key, "crossattn")
        super().__init__(*args, conditioning_key=conditioning_key, **kwargs)
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        self.use_scale = use_scale
        if self.use_scale:
            self.scale_a = scale_a
            self.scale_b = scale_b
            if fix_scale_bug:
                scale_step = self.num_timesteps - mid_step
            else:
                scale_step = self.num_timesteps
            scale_arr1 = np.linspace(scale_a, scale_b, mid_step)
            scale_arr2 = np.full(scale_step, scale_b)
            scale_arr = np.concatenate((scale_arr1, scale_arr2))
            # scale_arr_prev = np.append(scale_a, scale_arr[:-1])
            to_paddle = partial(paddle.to_tensor, dtype="float32")
            self.register_buffer(name="scale_arr", tensor=to_paddle(scale_arr))
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer(name="scale_factor", tensor=paddle.to_tensor(data=scale_factor))
        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        self.first_stage_config = first_stage_config
        self.cond_stage_config = cond_stage_config
        self.clip_denoised = False
        self.cond_stage_forward = cond_stage_forward
        self.encoder_type = encoder_type
        assert encoder_type in ["2d", "3d"]
        self.uncond_prob = uncond_prob
        self.classifier_free_guidance = True if uncond_prob > 0 else False
        assert uncond_type in ["zero_embed", "empty_seq"]
        self.uncond_type = uncond_type
        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys, only_model=only_model)
            self.restarted_from_ckpt = True

    def make_cond_schedule(self):
        self.cond_ids = paddle.full(shape=(self.num_timesteps,), fill_value=self.num_timesteps - 1, dtype="int64")
        ids = paddle.round(paddle.linspace(start=0, stop=self.num_timesteps - 1, num=self.num_timesteps_cond)).astype(
            dtype="int64"
        )
        self.cond_ids[: self.num_timesteps_cond] = ids

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: randn_tensor_mod(shape=x_start.shape, dtype=x_start.dtype))
        if self.use_scale:
            return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, tuple(x_start.shape))
                * x_start
                * extract_into_tensor(self.scale_arr, t, tuple(x_start.shape))
                + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, tuple(x_start.shape)) * noise
            )
        else:
            return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, tuple(x_start.shape)) * x_start
                + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, tuple(x_start.shape)) * noise
            )

    def _freeze_model(self):
        for name, para in self.model.diffusion_model.named_parameters():
            para.stop_gradient = not False

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.stop_gradient = not False

    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            model = instantiate_from_config(config)
            self.cond_stage_model = model.eval()
            self.cond_stage_model.train = disabled_train
            for param in self.cond_stage_model.parameters():
                param.stop_gradient = not False
        else:
            model = instantiate_from_config(config)
            self.cond_stage_model = model

    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, "encode") and callable(self.cond_stage_model.encode):
                c = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    def get_first_stage_encoding(self, encoder_posterior, noise=None):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample(noise=noise)
        elif isinstance(encoder_posterior, paddle.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    @paddle.no_grad()
    def encode_first_stage(self, x):
        if self.encoder_type == "2d" and x.dim() == 5:
            b, _, t, _, _ = tuple(x.shape)
            x = rearrange(x, "b c t h w -> (b t) c h w")
            reshape_back = True
        else:
            reshape_back = False
        encoder_posterior = self.first_stage_model.encode(x)
        results = self.get_first_stage_encoding(encoder_posterior).detach()
        if reshape_back:
            results = rearrange(results, "(b t) c h w -> b c t h w", b=b, t=t)
        return results

    @paddle.no_grad()
    def encode_first_stage_2DAE(self, x):
        b, _, t, _, _ = tuple(x.shape)
        results = paddle.concat(
            x=[
                self.get_first_stage_encoding(self.first_stage_model.encode(x[:, :, i])).detach().unsqueeze(axis=2)
                for i in range(t)
            ],
            axis=2,
        )
        return results

    def decode_core(self, z, **kwargs):
        if self.encoder_type == "2d" and z.dim() == 5:
            b, _, t, _, _ = tuple(z.shape)
            z = rearrange(z, "b c t h w -> (b t) c h w")
            reshape_back = True
        else:
            reshape_back = False
        z = 1.0 / self.scale_factor * z
        results = self.first_stage_model.decode(z, **kwargs)
        if reshape_back:
            results = rearrange(results, "(b t) c h w -> b c t h w", b=b, t=t)
        return results

    @paddle.no_grad()
    def decode_first_stage(self, z, **kwargs):
        return self.decode_core(z, **kwargs)

    def apply_model(self, x_noisy, t, cond, **kwargs):
        if isinstance(cond, dict):
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = "c_concat" if self.model.conditioning_key == "concat" else "c_crossattn"
            cond = {key: cond}
        x_recon = self.model(x_noisy, t, **cond, **kwargs)
        if isinstance(x_recon, tuple):
            return x_recon[0]
        else:
            return x_recon

    def _get_denoise_row_from_list(self, samples, desc=""):
        denoise_row = []
        for zd in tqdm(samples, desc=desc):
            denoise_row.append(self.decode_first_stage(zd.to(self.device)))
        n_log_timesteps = len(denoise_row)
        denoise_row = paddle.stack(x=denoise_row)
        if denoise_row.dim() == 5:
            denoise_grid = rearrange(denoise_row, "n b c h w -> b n c h w")
            denoise_grid = rearrange(denoise_grid, "b n c h w -> (b n) c h w")
            denoise_grid = make_grid(denoise_grid, nrow=n_log_timesteps)
        elif denoise_row.dim() == 6:
            video_length = tuple(denoise_row.shape)[3]
            denoise_grid = rearrange(denoise_row, "n b c t h w -> b n c t h w")
            denoise_grid = rearrange(denoise_grid, "b n c t h w -> (b n) c t h w")
            denoise_grid = rearrange(denoise_grid, "n c t h w -> (n t) c h w")
            denoise_grid = make_grid(denoise_grid, nrow=video_length)
        else:
            raise ValueError
        return denoise_grid

    @paddle.no_grad()
    def decode_first_stage_2DAE(self, z, **kwargs):
        b, _, t, _, _ = tuple(z.shape)
        z = 1.0 / self.scale_factor * z
        results = paddle.concat(
            x=[self.first_stage_model.decode(z[:, :, i], **kwargs).unsqueeze(axis=2) for i in range(t)], axis=2
        )
        return results

    def p_mean_variance(
        self, x, c, t, clip_denoised: bool, return_x0=False, score_corrector=None, corrector_kwargs=None, **kwargs
    ):
        t_in = t
        model_out = self.apply_model(x, t_in, c, **kwargs)
        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(self, model_out, x, t, c, **corrector_kwargs)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()
        if clip_denoised:
            x_recon.clip_(min=-1.0, max=1.0)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        if return_x0:
            return (model_mean, posterior_variance, posterior_log_variance, x_recon)
        else:
            return model_mean, posterior_variance, posterior_log_variance

    @paddle.no_grad()
    def p_sample(
        self,
        x,
        c,
        t,
        clip_denoised=False,
        repeat_noise=False,
        return_x0=False,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        **kwargs
    ):
        b, *_, device = *tuple(x.shape), x.place
        outputs = self.p_mean_variance(
            x=x,
            c=c,
            t=t,
            clip_denoised=clip_denoised,
            return_x0=return_x0,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
            **kwargs,
        )
        if return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs
        noise = noise_like(tuple(x.shape), device, repeat_noise) * temperature
        if noise_dropout > 0.0:
            noise = paddle.nn.functional.dropout(x=noise, p=noise_dropout)
        nonzero_mask = (1 - (t == 0).astype(dtype="float32")).reshape([b, *((1,) * (len(tuple(x.shape)) - 1))])
        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @paddle.no_grad()
    def p_sample_loop(
        self,
        cond,
        shape,
        return_intermediates=False,
        x_T=None,
        verbose=True,
        callback=None,
        timesteps=None,
        mask=None,
        x0=None,
        img_callback=None,
        start_T=None,
        log_every_t=None,
        **kwargs
    ):
        if not log_every_t:
            log_every_t = self.log_every_t
        # device = self.betas.place
        b = shape[0]
        if x_T is None:
            img = randn_tensor_mod(shape=shape)
        else:
            img = x_T
        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps
        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = (
            tqdm(reversed(range(0, timesteps)), desc="Sampling t", total=timesteps)
            if verbose
            else reversed(range(0, timesteps))
        )
        if mask is not None:
            assert x0 is not None
            assert tuple(x0.shape)[2:3] == tuple(mask.shape)[2:3]
        for i in iterator:
            ts = paddle.full(shape=(b,), fill_value=i, dtype="int64")
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != "hybrid"
                tc = self.cond_ids[ts].to(cond.place)
                noise = randn_tensor_mod(shape=cond.shape, dtype=cond.dtype)
                cond = self.q_sample(x_start=cond, t=tc, noise=noise)
            img = self.p_sample(img, cond, ts, clip_denoised=self.clip_denoised, **kwargs)
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1.0 - mask) * img
            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback:
                callback(i)
            if img_callback:
                img_callback(img, i)
        if return_intermediates:
            return img, intermediates
        return img


class LatentVisualDiffusion(LatentDiffusion):
    def __init__(self, cond_img_config, finegrained=False, random_cond=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.random_cond = random_cond
        self.instantiate_img_embedder(cond_img_config, freeze=True)
        num_tokens = 16 if finegrained else 4
        self.image_proj_model = self.init_projector(
            use_finegrained=finegrained, num_tokens=num_tokens, input_dim=1024, cross_attention_dim=1024, dim=1280
        )

    def instantiate_img_embedder(self, config, freeze=True):
        embedder = instantiate_from_config(config)
        if freeze:
            self.embedder = embedder.eval()
            self.embedder.train = disabled_train
            for param in self.embedder.parameters():
                param.stop_gradient = not False

    def init_projector(self, use_finegrained, num_tokens, input_dim, cross_attention_dim, dim):
        if not use_finegrained:
            image_proj_model = ImageProjModel(
                clip_extra_context_tokens=num_tokens,
                cross_attention_dim=cross_attention_dim,
                clip_embeddings_dim=input_dim,
            )
        else:
            image_proj_model = Resampler(
                dim=input_dim,
                depth=4,
                dim_head=64,
                heads=12,
                num_queries=num_tokens,
                embedding_dim=dim,
                output_dim=cross_attention_dim,
                ff_mult=4,
            )
        return image_proj_model

    def get_image_embeds(self, batch_imgs):
        img_token = self.embedder(batch_imgs)
        img_emb = self.image_proj_model(img_token)
        return img_emb


class DiffusionWrapper(paddle.nn.Layer):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key

    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None, c_adm=None, s=None, mask=None, **kwargs):
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t)
        elif self.conditioning_key == "concat":
            xc = paddle.concat(x=[x] + c_concat, axis=1)
            out = self.diffusion_model(xc, t, **kwargs)
        elif self.conditioning_key == "crossattn":
            cc = paddle.concat(x=c_crossattn, axis=1)
            out = self.diffusion_model(x, t, context=cc, **kwargs)
        elif self.conditioning_key == "hybrid":
            xc = paddle.concat(x=[x] + c_concat, axis=1)
            cc = paddle.concat(x=c_crossattn, axis=1)
            out = self.diffusion_model(xc, t, context=cc)
        elif self.conditioning_key == "resblockcond":
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, context=cc)
        elif self.conditioning_key == "adm":
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)
        elif self.conditioning_key == "hybrid-adm":
            assert c_adm is not None
            xc = paddle.concat(x=[x] + c_concat, axis=1)
            cc = paddle.concat(x=c_crossattn, axis=1)
            out = self.diffusion_model(xc, t, context=cc, y=c_adm)
        elif self.conditioning_key == "hybrid-time":
            assert s is not None
            xc = paddle.concat(x=[x] + c_concat, axis=1)
            cc = paddle.concat(x=c_crossattn, axis=1)
            out = self.diffusion_model(xc, t, context=cc, s=s)
        elif self.conditioning_key == "concat-time-mask":
            xc = paddle.concat(x=[x] + c_concat, axis=1)
            out = self.diffusion_model(xc, t, context=None, s=s, mask=mask)
        elif self.conditioning_key == "concat-adm-mask":
            if c_concat is not None:
                xc = paddle.concat(x=[x] + c_concat, axis=1)
            else:
                xc = x
            out = self.diffusion_model(xc, t, context=None, y=s, mask=mask)
        elif self.conditioning_key == "hybrid-adm-mask":
            cc = paddle.concat(x=c_crossattn, axis=1)
            if c_concat is not None:
                xc = paddle.concat(x=[x] + c_concat, axis=1)
            else:
                xc = x
            out = self.diffusion_model(xc, t, context=cc, y=s, mask=mask)
        elif self.conditioning_key == "hybrid-time-adm":
            assert c_adm is not None
            xc = paddle.concat(x=[x] + c_concat, axis=1)
            cc = paddle.concat(x=c_crossattn, axis=1)
            out = self.diffusion_model(xc, t, context=cc, s=s, y=c_adm)
        else:
            raise NotImplementedError()
        return out
