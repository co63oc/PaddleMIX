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

# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import argparse
import os
import sys

import paddle
from cog import BasePredictor, Input, Path
from omegaconf import OmegaConf

sys.path.insert(0, "scripts/evaluation")
from funcs import batch_ddim_sampling, load_image_batch, load_model_checkpoint
from utils.utils import instantiate_from_config

from ppdiffusers.pipelines.lvdm.video_save import make_grid, write_video


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        ckpt_path_base = "checkpoints/base_1024_v1/model.ckpt"
        config_base = "configs/inference_t2v_1024_v1.0.yaml"
        ckpt_path_i2v = "checkpoints/i2v_512_v1/model.ckpt"
        config_i2v = "configs/inference_i2v_512_v1.0.yaml"
        config_base = OmegaConf.load(config_base)
        model_config_base = config_base.pop("model", OmegaConf.create())
        self.model_base = instantiate_from_config(model_config_base)
        self.model_base = self.model_base.cuda(blocking=True)
        self.model_base = load_model_checkpoint(self.model_base, ckpt_path_base)
        self.model_base.eval()
        config_i2v = OmegaConf.load(config_i2v)
        model_config_i2v = config_i2v.pop("model", OmegaConf.create())
        self.model_i2v = instantiate_from_config(model_config_i2v)
        self.model_i2v = self.model_i2v.cuda(blocking=True)
        self.model_i2v = load_model_checkpoint(self.model_i2v, ckpt_path_i2v)
        self.model_i2v.eval()

    def predict(
        self,
        task: str = Input(description="Choose the task.", choices=["text2video", "image2video"], default="text2video"),
        prompt: str = Input(
            description="Prompt for video generation.",
            default="A tiger walks in the forest, photorealistic, 4k, high definition.",
        ),
        image: Path = Input(description="Input image for image2video task.", default=None),
        ddim_steps: int = Input(description="Number of denoising steps.", default=50),
        unconditional_guidance_scale: float = Input(description="Classifier-free guidance scale.", default=12.0),
        seed: int = Input(description="Random seed. Leave blank to randomize the seed", default=None),
        save_fps: int = Input(description="Frame per second for the generated video.", default=10),
    ) -> Path:
        width = 1024 if task == "text2video" else 512
        height = 576 if task == "text2video" else 320
        model = self.model_base if task == "text2video" else self.model_i2v
        if task == "image2video":
            assert image is not None, "Please provide image for image2video generation."
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        paddle.seed(seed)
        args = argparse.Namespace(
            mode="base" if task == "text2video" else "i2v",
            savefps=save_fps,
            n_samples=1,
            ddim_steps=ddim_steps,
            ddim_eta=1.0,
            bs=1,
            height=height,
            width=width,
            frames=-1,
            fps=28 if task == "text2video" else 8,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_guidance_scale_temporal=None,
        )
        h, w = args.height // 8, args.width // 8
        frames = model.temporal_length if args.frames < 0 else args.frames
        channels = model.channels
        batch_size = 1
        noise_shape = [batch_size, channels, frames, h, w]
        fps = paddle.to_tensor(data=[args.fps] * batch_size).to(model.place).astype(dtype="int64")
        prompts = [prompt]
        text_emb = model.get_learned_conditioning(prompts)
        if args.mode == "base":
            cond = {"c_crossattn": [text_emb], "fps": fps}
        elif args.mode == "i2v":
            cond_images = load_image_batch([str(image)], (args.height, args.width))
            cond_images = cond_images.to(model.place)
            img_emb = model.get_image_embeds(cond_images)
            imtext_cond = paddle.concat(x=[text_emb, img_emb], axis=1)
            cond = {"c_crossattn": [imtext_cond], "fps": fps}
        else:
            raise NotImplementedError
        batch_samples = batch_ddim_sampling(
            model, cond, noise_shape, args.n_samples, args.ddim_steps, args.ddim_eta, args.unconditional_guidance_scale
        )
        out_path = "/tmp/output.mp4"
        vid_tensor = batch_samples[0]
        video = vid_tensor.detach().cpu()
        video = paddle.clip(x=video.astype(dtype="float32"), min=-1.0, max=1.0)
        video = video.transpose(perm=[2, 0, 1, 3, 4])
        frame_grids = [make_grid(framesheet, nrow=int(args.n_samples)) for framesheet in video]
        grid = paddle.stack(x=frame_grids, axis=0)
        grid = (grid + 1.0) / 2.0
        grid = (grid * 255).to("uint8").transpose(perm=[0, 2, 3, 1])
        write_video(out_path, grid, fps=args.savefps, video_codec="h264", options={"crf": "10"})
        return Path(out_path)
