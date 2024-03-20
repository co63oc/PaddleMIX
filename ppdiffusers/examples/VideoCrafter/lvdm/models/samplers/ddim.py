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

import numpy as np
import paddle
from lvdm.common import noise_like
from lvdm.models.utils_diffusion import (
    make_ddim_sampling_parameters,
    make_ddim_timesteps,
)
from tqdm import tqdm
from utils.utils import randn_tensor_mod


def to_paddle(x):
    if type(x) != paddle.Tensor:
        x = paddle.to_tensor(x)
    return x.clone().detach().to("float32")


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.counter = 0

    def register_buffer(self, name, attr):
        if type(attr) == paddle.Tensor:
            if attr.place != str("cuda").replace("cuda", "gpu"):
                attr = attr.to(str("cuda").replace("cuda", "gpu"))
        else:
            attr = paddle.to_tensor(attr)
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0.0, verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method=ddim_discretize,
            num_ddim_timesteps=ddim_num_steps,
            num_ddpm_timesteps=self.ddpm_num_timesteps,
            verbose=verbose,
        )
        alphas_cumprod = self.model.alphas_cumprod
        assert tuple(alphas_cumprod.shape)[0] == self.ddpm_num_timesteps, "alphas have to be defined for each timestep"
        self.register_buffer(name="betas", attr=to_paddle(self.model.betas))
        self.register_buffer(name="alphas_cumprod", attr=to_paddle(alphas_cumprod))
        self.register_buffer(name="alphas_cumprod_prev", attr=to_paddle(self.model.alphas_cumprod_prev))
        self.use_scale = self.model.use_scale
        print("DDIM scale", self.use_scale)
        if self.use_scale:
            self.register_buffer(name="scale_arr", attr=to_paddle(self.model.scale_arr))
            ddim_scale_arr = self.scale_arr.cpu()[self.ddim_timesteps]
            self.register_buffer(name="ddim_scale_arr", attr=ddim_scale_arr)
            ddim_scale_arr = np.asarray(
                [self.scale_arr.cpu()[0]] + self.scale_arr.cpu()[self.ddim_timesteps[:-1]].tolist()
            )
            self.register_buffer(name="ddim_scale_arr_prev", attr=ddim_scale_arr)
        self.register_buffer(name="sqrt_alphas_cumprod", attr=to_paddle(np.sqrt(alphas_cumprod.cpu().numpy())))
        self.register_buffer(
            name="sqrt_one_minus_alphas_cumprod", attr=to_paddle(np.sqrt(1.0 - alphas_cumprod.cpu().numpy()))
        )
        self.register_buffer(
            name="log_one_minus_alphas_cumprod", attr=to_paddle(np.log(1.0 - alphas_cumprod.cpu().numpy()))
        )
        self.register_buffer(
            name="sqrt_recip_alphas_cumprod", attr=to_paddle(np.sqrt(1.0 / alphas_cumprod.cpu().numpy()))
        )
        self.register_buffer(
            name="sqrt_recipm1_alphas_cumprod", attr=to_paddle(np.sqrt(1.0 / alphas_cumprod.cpu().numpy() - 1))
        )
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=alphas_cumprod.cpu(), ddim_timesteps=self.ddim_timesteps, eta=ddim_eta, verbose=verbose
        )
        self.register_buffer(name="ddim_sigmas", attr=ddim_sigmas)
        self.register_buffer(name="ddim_alphas", attr=ddim_alphas)
        self.register_buffer(name="ddim_alphas_prev", attr=ddim_alphas_prev)
        self.register_buffer(name="ddim_sqrt_one_minus_alphas", attr=paddle.sqrt(1.0 - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * paddle.sqrt(
            x=(1 - self.alphas_cumprod_prev)
            / (1 - self.alphas_cumprod)
            * (1 - self.alphas_cumprod / self.alphas_cumprod_prev)
        )
        self.register_buffer(name="ddim_sigmas_for_original_num_steps", attr=sigmas_for_original_sampling_steps)

    @paddle.no_grad()
    def sample(
        self,
        S,
        batch_size,
        shape,
        conditioning=None,
        callback=None,
        normals_sequence=None,
        img_callback=None,
        quantize_x0=False,
        eta=0.0,
        mask=None,
        x0=None,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        verbose=True,
        schedule_verbose=False,
        x_T=None,
        log_every_t=100,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        **kwargs
    ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                try:
                    cbs = tuple(conditioning[list(conditioning.keys())[0]][0].shape)[0]
                except:
                    cbs = tuple(conditioning[list(conditioning.keys())[0]][0]["last_hidden_state"].shape)[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            elif tuple(conditioning.shape)[0] != batch_size:
                print(f"Warning: Got {tuple(conditioning.shape)[0]} conditionings but batch-size is {batch_size}")
        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=schedule_verbose)
        if len(shape) == 3:
            C, H, W = shape
            size = batch_size, C, H, W
        elif len(shape) == 4:
            C, T, H, W = shape
            size = batch_size, C, T, H, W
        samples, intermediates = self.ddim_sampling(
            conditioning,
            size,
            callback=callback,
            img_callback=img_callback,
            quantize_denoised=quantize_x0,
            mask=mask,
            x0=x0,
            ddim_use_original_steps=False,
            noise_dropout=noise_dropout,
            temperature=temperature,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
            x_T=x_T,
            log_every_t=log_every_t,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
            verbose=verbose,
            **kwargs,
        )
        return samples, intermediates

    @paddle.no_grad()
    def ddim_sampling(
        self,
        cond,
        shape,
        x_T=None,
        ddim_use_original_steps=False,
        callback=None,
        timesteps=None,
        quantize_denoised=False,
        mask=None,
        x0=None,
        img_callback=None,
        log_every_t=100,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        verbose=True,
        cond_tau=1.0,
        target_size=None,
        start_timesteps=None,
        **kwargs
    ):
        device = self.model.betas.place
        print("ddim device", device)
        b = shape[0]
        if x_T is None:
            img = randn_tensor_mod(shape=shape)
        else:
            img = x_T
        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = (
                int(min(timesteps / tuple(self.ddim_timesteps.shape)[0], 1) * tuple(self.ddim_timesteps.shape)[0]) - 1
            )
            timesteps = self.ddim_timesteps[:subset_end]
        intermediates = {"x_inter": [img], "pred_x0": [img]}
        time_range = reversed(range(0, timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else tuple(timesteps.shape)[0]
        if verbose:
            iterator = tqdm(time_range, desc="DDIM Sampler", total=total_steps)
        else:
            iterator = time_range
        init_x0 = False
        clean_cond = kwargs.pop("clean_cond", False)
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = paddle.full(shape=(b,), fill_value=step, dtype="int64")
            if start_timesteps is not None:
                assert x0 is not None
                if step > start_timesteps * time_range[0]:
                    continue
                elif not init_x0:
                    img = self.model.q_sample(x0, ts)
                    init_x0 = True
            if mask is not None:
                assert x0 is not None
                if clean_cond:
                    img_orig = x0
                else:
                    img_orig = self.model.q_sample(x0, ts)
                img = img_orig * mask + (1.0 - mask) * img
            index_clip = int((1 - cond_tau) * total_steps)
            if index <= index_clip and target_size is not None:
                target_size_ = [target_size[0], target_size[1] // 8, target_size[2] // 8]
                img = paddle.nn.functional.interpolate(x=img, size=target_size_, mode="nearest")
            outs = self.p_sample_ddim(
                img,
                cond,
                ts,
                index=index,
                use_original_steps=ddim_use_original_steps,
                quantize_denoised=quantize_denoised,
                temperature=temperature,
                noise_dropout=noise_dropout,
                score_corrector=score_corrector,
                corrector_kwargs=corrector_kwargs,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
                x0=x0,
                **kwargs,
            )
            img, pred_x0 = outs
            if callback:
                callback(i)
            if img_callback:
                img_callback(pred_x0, i)
            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates["x_inter"].append(img)
                intermediates["pred_x0"].append(pred_x0)
        return img, intermediates

    @paddle.no_grad()
    def p_sample_ddim(
        self,
        x,
        c,
        t,
        index,
        repeat_noise=False,
        use_original_steps=False,
        quantize_denoised=False,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        uc_type=None,
        conditional_guidance_scale_temporal=None,
        **kwargs
    ):
        b, *_, device = *tuple(x.shape), x.place
        if x.dim() == 5:
            is_video = True
        else:
            is_video = False
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.0:
            e_t = self.model.apply_model(x, t, c, **kwargs)
        else:
            if isinstance(c, paddle.Tensor):
                e_t = self.model.apply_model(x, t, c, **kwargs)
                e_t_uncond = self.model.apply_model(x, t, unconditional_conditioning, **kwargs)
            elif isinstance(c, dict):
                e_t = self.model.apply_model(x, t, c, **kwargs)
                e_t_uncond = self.model.apply_model(x, t, unconditional_conditioning, **kwargs)
            else:
                raise NotImplementedError
            if uc_type is None:
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
            elif uc_type == "cfg_original":
                e_t = e_t + unconditional_guidance_scale * (e_t - e_t_uncond)
            elif uc_type == "cfg_ours":
                e_t = e_t + unconditional_guidance_scale * (e_t_uncond - e_t)
            else:
                raise NotImplementedError
            if conditional_guidance_scale_temporal is not None:
                e_t_temporal = self.model.apply_model(x, t, c, **kwargs)
                e_t_image = self.model.apply_model(x, t, c, no_temporal_attn=True, **kwargs)
                e_t = e_t + conditional_guidance_scale_temporal * (e_t_temporal - e_t_image)
        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)
        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = (
            self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        )
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        if is_video:
            size = b, 1, 1, 1, 1
        else:
            size = b, 1, 1, 1
        a_t = paddle.full(shape=size, fill_value=alphas[index])
        a_prev = paddle.full(shape=size, fill_value=alphas_prev[index])
        sigma_t = paddle.full(shape=size, fill_value=sigmas[index])
        sqrt_one_minus_at = paddle.full(shape=size, fill_value=sqrt_one_minus_alphas[index])
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        dir_xt = (1.0 - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(tuple(x.shape), device, repeat_noise) * temperature
        if noise_dropout > 0.0:
            noise = paddle.nn.functional.dropout(x=noise, p=noise_dropout)
        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        if self.use_scale:
            scale_arr = self.model.scale_arr if use_original_steps else self.ddim_scale_arr
            scale_t = paddle.full(shape=size, fill_value=scale_arr[index])
            scale_arr_prev = self.model.scale_arr_prev if use_original_steps else self.ddim_scale_arr_prev
            scale_t_prev = paddle.full(shape=size, fill_value=scale_arr_prev[index])
            pred_x0 /= scale_t
            x_prev = a_prev.sqrt() * scale_t_prev * pred_x0 + dir_xt + noise
        else:
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0

    @paddle.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = paddle.sqrt(x=self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas
        if noise is None:
            noise = randn_tensor_mod(shape=x0.shape, dtype=x0.dtype)

        def extract_into_tensor(a, t, x_shape):
            b, *_ = tuple(t.shape)
            out = a.take_along_axis(axis=-1, indices=t)
            return out.reshape([b, *((1,) * (len(x_shape) - 1))])

        return (
            extract_into_tensor(sqrt_alphas_cumprod, t, tuple(x0.shape)) * x0
            + extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, tuple(x0.shape)) * noise
        )

    @paddle.no_grad()
    def decode(
        self,
        x_latent,
        cond,
        t_start,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        use_original_steps=False,
    ):
        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]
        time_range = np.flip(timesteps)
        total_steps = tuple(timesteps.shape)[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")
        iterator = tqdm(time_range, desc="Decoding image", total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = paddle.full(shape=(tuple(x_latent.shape)[0],), fill_value=step, dtype="int64")
            x_dec, _ = self.p_sample_ddim(
                x_dec,
                cond,
                ts,
                index=index,
                use_original_steps=use_original_steps,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
            )
        return x_dec
