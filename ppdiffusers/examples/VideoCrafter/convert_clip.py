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
import argparse

import paddle
import torch

paddle.set_device("cpu")

# i2v includes t2v
linear_list = [
    "visual.transformer.resblocks.0.attn.out_proj",
    "visual.transformer.resblocks.0.mlp.c_fc",
    "visual.transformer.resblocks.0.mlp.c_proj",
    "visual.transformer.resblocks.1.attn.out_proj",
    "visual.transformer.resblocks.1.mlp.c_fc",
    "visual.transformer.resblocks.1.mlp.c_proj",
    "visual.transformer.resblocks.2.attn.out_proj",
    "visual.transformer.resblocks.2.mlp.c_fc",
    "visual.transformer.resblocks.2.mlp.c_proj",
    "visual.transformer.resblocks.3.attn.out_proj",
    "visual.transformer.resblocks.3.mlp.c_fc",
    "visual.transformer.resblocks.3.mlp.c_proj",
    "visual.transformer.resblocks.4.attn.out_proj",
    "visual.transformer.resblocks.4.mlp.c_fc",
    "visual.transformer.resblocks.4.mlp.c_proj",
    "visual.transformer.resblocks.5.attn.out_proj",
    "visual.transformer.resblocks.5.mlp.c_fc",
    "visual.transformer.resblocks.5.mlp.c_proj",
    "visual.transformer.resblocks.6.attn.out_proj",
    "visual.transformer.resblocks.6.mlp.c_fc",
    "visual.transformer.resblocks.6.mlp.c_proj",
    "visual.transformer.resblocks.7.attn.out_proj",
    "visual.transformer.resblocks.7.mlp.c_fc",
    "visual.transformer.resblocks.7.mlp.c_proj",
    "visual.transformer.resblocks.8.attn.out_proj",
    "visual.transformer.resblocks.8.mlp.c_fc",
    "visual.transformer.resblocks.8.mlp.c_proj",
    "visual.transformer.resblocks.9.attn.out_proj",
    "visual.transformer.resblocks.9.mlp.c_fc",
    "visual.transformer.resblocks.9.mlp.c_proj",
    "visual.transformer.resblocks.10.attn.out_proj",
    "visual.transformer.resblocks.10.mlp.c_fc",
    "visual.transformer.resblocks.10.mlp.c_proj",
    "visual.transformer.resblocks.11.attn.out_proj",
    "visual.transformer.resblocks.11.mlp.c_fc",
    "visual.transformer.resblocks.11.mlp.c_proj",
    "visual.transformer.resblocks.12.attn.out_proj",
    "visual.transformer.resblocks.12.mlp.c_fc",
    "visual.transformer.resblocks.12.mlp.c_proj",
    "visual.transformer.resblocks.13.attn.out_proj",
    "visual.transformer.resblocks.13.mlp.c_fc",
    "visual.transformer.resblocks.13.mlp.c_proj",
    "visual.transformer.resblocks.14.attn.out_proj",
    "visual.transformer.resblocks.14.mlp.c_fc",
    "visual.transformer.resblocks.14.mlp.c_proj",
    "visual.transformer.resblocks.15.attn.out_proj",
    "visual.transformer.resblocks.15.mlp.c_fc",
    "visual.transformer.resblocks.15.mlp.c_proj",
    "visual.transformer.resblocks.16.attn.out_proj",
    "visual.transformer.resblocks.16.mlp.c_fc",
    "visual.transformer.resblocks.16.mlp.c_proj",
    "visual.transformer.resblocks.17.attn.out_proj",
    "visual.transformer.resblocks.17.mlp.c_fc",
    "visual.transformer.resblocks.17.mlp.c_proj",
    "visual.transformer.resblocks.18.attn.out_proj",
    "visual.transformer.resblocks.18.mlp.c_fc",
    "visual.transformer.resblocks.18.mlp.c_proj",
    "visual.transformer.resblocks.19.attn.out_proj",
    "visual.transformer.resblocks.19.mlp.c_fc",
    "visual.transformer.resblocks.19.mlp.c_proj",
    "visual.transformer.resblocks.20.attn.out_proj",
    "visual.transformer.resblocks.20.mlp.c_fc",
    "visual.transformer.resblocks.20.mlp.c_proj",
    "visual.transformer.resblocks.21.attn.out_proj",
    "visual.transformer.resblocks.21.mlp.c_fc",
    "visual.transformer.resblocks.21.mlp.c_proj",
    "visual.transformer.resblocks.22.attn.out_proj",
    "visual.transformer.resblocks.22.mlp.c_fc",
    "visual.transformer.resblocks.22.mlp.c_proj",
    "visual.transformer.resblocks.23.attn.out_proj",
    "visual.transformer.resblocks.23.mlp.c_fc",
    "visual.transformer.resblocks.23.mlp.c_proj",
    "visual.transformer.resblocks.24.attn.out_proj",
    "visual.transformer.resblocks.24.mlp.c_fc",
    "visual.transformer.resblocks.24.mlp.c_proj",
    "visual.transformer.resblocks.25.attn.out_proj",
    "visual.transformer.resblocks.25.mlp.c_fc",
    "visual.transformer.resblocks.25.mlp.c_proj",
    "visual.transformer.resblocks.26.attn.out_proj",
    "visual.transformer.resblocks.26.mlp.c_fc",
    "visual.transformer.resblocks.26.mlp.c_proj",
    "visual.transformer.resblocks.27.attn.out_proj",
    "visual.transformer.resblocks.27.mlp.c_fc",
    "visual.transformer.resblocks.27.mlp.c_proj",
    "visual.transformer.resblocks.28.attn.out_proj",
    "visual.transformer.resblocks.28.mlp.c_fc",
    "visual.transformer.resblocks.28.mlp.c_proj",
    "visual.transformer.resblocks.29.attn.out_proj",
    "visual.transformer.resblocks.29.mlp.c_fc",
    "visual.transformer.resblocks.29.mlp.c_proj",
    "visual.transformer.resblocks.30.attn.out_proj",
    "visual.transformer.resblocks.30.mlp.c_fc",
    "visual.transformer.resblocks.30.mlp.c_proj",
    "visual.transformer.resblocks.31.attn.out_proj",
    "visual.transformer.resblocks.31.mlp.c_fc",
    "visual.transformer.resblocks.31.mlp.c_proj",
    "transformer.resblocks.0.attn.out_proj",
    "transformer.resblocks.0.mlp.c_fc",
    "transformer.resblocks.0.mlp.c_proj",
    "transformer.resblocks.1.attn.out_proj",
    "transformer.resblocks.1.mlp.c_fc",
    "transformer.resblocks.1.mlp.c_proj",
    "transformer.resblocks.2.attn.out_proj",
    "transformer.resblocks.2.mlp.c_fc",
    "transformer.resblocks.2.mlp.c_proj",
    "transformer.resblocks.3.attn.out_proj",
    "transformer.resblocks.3.mlp.c_fc",
    "transformer.resblocks.3.mlp.c_proj",
    "transformer.resblocks.4.attn.out_proj",
    "transformer.resblocks.4.mlp.c_fc",
    "transformer.resblocks.4.mlp.c_proj",
    "transformer.resblocks.5.attn.out_proj",
    "transformer.resblocks.5.mlp.c_fc",
    "transformer.resblocks.5.mlp.c_proj",
    "transformer.resblocks.6.attn.out_proj",
    "transformer.resblocks.6.mlp.c_fc",
    "transformer.resblocks.6.mlp.c_proj",
    "transformer.resblocks.7.attn.out_proj",
    "transformer.resblocks.7.mlp.c_fc",
    "transformer.resblocks.7.mlp.c_proj",
    "transformer.resblocks.8.attn.out_proj",
    "transformer.resblocks.8.mlp.c_fc",
    "transformer.resblocks.8.mlp.c_proj",
    "transformer.resblocks.9.attn.out_proj",
    "transformer.resblocks.9.mlp.c_fc",
    "transformer.resblocks.9.mlp.c_proj",
    "transformer.resblocks.10.attn.out_proj",
    "transformer.resblocks.10.mlp.c_fc",
    "transformer.resblocks.10.mlp.c_proj",
    "transformer.resblocks.11.attn.out_proj",
    "transformer.resblocks.11.mlp.c_fc",
    "transformer.resblocks.11.mlp.c_proj",
    "transformer.resblocks.12.attn.out_proj",
    "transformer.resblocks.12.mlp.c_fc",
    "transformer.resblocks.12.mlp.c_proj",
    "transformer.resblocks.13.attn.out_proj",
    "transformer.resblocks.13.mlp.c_fc",
    "transformer.resblocks.13.mlp.c_proj",
    "transformer.resblocks.14.attn.out_proj",
    "transformer.resblocks.14.mlp.c_fc",
    "transformer.resblocks.14.mlp.c_proj",
    "transformer.resblocks.15.attn.out_proj",
    "transformer.resblocks.15.mlp.c_fc",
    "transformer.resblocks.15.mlp.c_proj",
    "transformer.resblocks.16.attn.out_proj",
    "transformer.resblocks.16.mlp.c_fc",
    "transformer.resblocks.16.mlp.c_proj",
    "transformer.resblocks.17.attn.out_proj",
    "transformer.resblocks.17.mlp.c_fc",
    "transformer.resblocks.17.mlp.c_proj",
    "transformer.resblocks.18.attn.out_proj",
    "transformer.resblocks.18.mlp.c_fc",
    "transformer.resblocks.18.mlp.c_proj",
    "transformer.resblocks.19.attn.out_proj",
    "transformer.resblocks.19.mlp.c_fc",
    "transformer.resblocks.19.mlp.c_proj",
    "transformer.resblocks.20.attn.out_proj",
    "transformer.resblocks.20.mlp.c_fc",
    "transformer.resblocks.20.mlp.c_proj",
    "transformer.resblocks.21.attn.out_proj",
    "transformer.resblocks.21.mlp.c_fc",
    "transformer.resblocks.21.mlp.c_proj",
    "transformer.resblocks.22.attn.out_proj",
    "transformer.resblocks.22.mlp.c_fc",
    "transformer.resblocks.22.mlp.c_proj",
    "transformer.resblocks.23.attn.out_proj",
    "transformer.resblocks.23.mlp.c_fc",
    "transformer.resblocks.23.mlp.c_proj",
]


def convert_to_ppdiffusers(vae_or_unet, dtype=None):
    need_transpose = []
    with_name = False
    if "state_dict" in vae_or_unet:
        with_name = True
        vae_or_unet = vae_or_unet["state_dict"]
    for k, v in vae_or_unet.items():
        if k.replace(".weight", "") in linear_list:
            need_transpose.append(k)
        if "in_proj_weight" in k:
            need_transpose.append(k)

    new_vae_or_unet = {}
    for k, v in vae_or_unet.items():
        if not isinstance(v, torch.Tensor):
            print(v)
            continue
        if k not in need_transpose:
            v2 = v
        else:
            v2 = v.t()
        new_vae_or_unet[k] = paddle.to_tensor(v2.cpu().numpy())
        if dtype is not None:
            new_vae_or_unet[k] = new_vae_or_unet[k].astype(dtype)
    if not with_name:
        return new_vae_or_unet
    return {"state_dict": new_vae_or_unet}


def convert_diffusers_stable_diffusion_to_ppdiffusers(pretrained_model_name_or_path, output_path=None):
    model = torch.load(pretrained_model_name_or_path)
    state_dict = convert_to_ppdiffusers(model)
    paddle.save(state_dict, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pytorch model weights to Paddle model weights.")
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default="/data/VideoCrafter/checkpoints/i2v_512_v1/model.ckpt",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="checkpoints/i2v_512_v1/model.ckpt",
        help="The model output path.",
    )
    # args = parser.parse_args()
    # convert_diffusers_stable_diffusion_to_ppdiffusers(args.pretrained_path, args.output_path)

    convert_diffusers_stable_diffusion_to_ppdiffusers(
        "/root/.cache/huggingface/hub/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K/snapshots/de081ac0a0ca8dc9d1533eed1ae884bb8ae1404b/open_clip_pytorch_model.bin",
        "checkpoints/open_clip_paddle_model.bin",
    )
    print("1")

    # convert_diffusers_stable_diffusion_to_ppdiffusers(
    #     "/nfs/model/ImageVideo/model.ckpt",
    #     "checkpoints/i2v_512_v1/model.ckpt",
    # )
    # print("2")
