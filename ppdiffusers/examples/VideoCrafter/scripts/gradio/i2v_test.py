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

import os
import time

import paddle
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from scripts.evaluation.funcs import (
    batch_ddim_sampling,
    load_model_checkpoint,
    save_videos,
)
from utils.utils import instantiate_from_config


class Image2Video:
    def __init__(self, result_dir="./tmp/", gpu_num=1) -> None:
        self.download_model()
        self.result_dir = result_dir
        if not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)
        ckpt_path = "checkpoints/i2v_512_v1/model.ckpt"
        config_file = "configs/inference_i2v_512_v1.0.yaml"
        config = OmegaConf.load(config_file)
        model_config = config.pop("model", OmegaConf.create())
        model_config["params"]["unet_config"]["params"]["use_checkpoint"] = False
        model_list = []
        for gpu_id in range(gpu_num):
            model = instantiate_from_config(model_config)
            assert os.path.exists(ckpt_path), "Error: checkpoint Not Found!"
            model = load_model_checkpoint(model, ckpt_path)
            model.eval()
            model_list.append(model)
        self.model_list = model_list
        self.save_fps = 8

    def get_image(self, image, prompt, steps=50, cfg_scale=12.0, eta=1.0, fps=16):
        paddle.device.cuda.empty_cache()
        print("start:", prompt, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
        start = time.time()
        gpu_id = 0
        if steps > 60:
            steps = 60
        model = self.model_list[gpu_id]
        model = model.cuda(blocking=True)
        batch_size = 1
        channels = model.model.diffusion_model.in_channels
        frames = model.temporal_length
        h, w = 320 // 8, 512 // 8
        noise_shape = [batch_size, channels, frames, h, w]
        text_emb = model.get_learned_conditioning([prompt])
        img_tensor = paddle.to_tensor(data=image).transpose(perm=[2, 0, 1]).astype(dtype="float32")
        img_tensor = (img_tensor / 255.0 - 0.5) * 2
        img_tensor = img_tensor.unsqueeze(axis=0)
        cond_images = img_tensor.to(model.place)
        img_emb = model.get_image_embeds(cond_images)
        imtext_cond = paddle.concat(x=[text_emb, img_emb], axis=1)
        cond = {"c_crossattn": [imtext_cond], "fps": fps}
        batch_samples = batch_ddim_sampling(
            model, cond, noise_shape, n_samples=1, ddim_steps=steps, ddim_eta=eta, cfg_scale=cfg_scale
        )
        prompt_str = prompt.replace("/", "_slash_") if "/" in prompt else prompt
        prompt_str = prompt_str.replace(" ", "_") if " " in prompt else prompt_str
        prompt_str = prompt_str[:30]
        save_videos(batch_samples, self.result_dir, filenames=[prompt_str], fps=self.save_fps)
        print(f"Saved in {prompt_str}. Time used: {time.time() - start:.2f} seconds")
        model = model.cpu()
        return os.path.join(self.result_dir, f"{prompt_str}.mp4")

    def download_model(self):
        REPO_ID = "VideoCrafter/Image2Video-512"
        filename_list = ["model.ckpt"]
        if not os.path.exists("./checkpoints/i2v_512_v1/"):
            os.makedirs("./checkpoints/i2v_512_v1/")
        for filename in filename_list:
            local_file = os.path.join("./checkpoints/i2v_512_v1/", filename)
            if not os.path.exists(local_file):
                hf_hub_download(
                    repo_id=REPO_ID,
                    filename=filename,
                    local_dir="./checkpoints/i2v_512_v1/",
                    local_dir_use_symlinks=False,
                )


if __name__ == "__main__":
    i2v = Image2Video()
    video_path = i2v.get_image("prompts/i2v_prompts/horse.png", "horses are walking on the grassland")
    print("done", video_path)
