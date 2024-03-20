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

import glob
import os
import sys
from collections import OrderedDict

import cv2
import numpy as np
import paddle
from decord import VideoReader, cpu

sys.path.insert(1, os.path.join(sys.path[0], "..", ".."))
from lvdm.models.samplers.ddim import DDIMSampler

from ppdiffusers.pipelines.lvdm.video_save import make_grid, write_video


def batch_ddim_sampling(
    model,
    cond,
    noise_shape,
    n_samples=1,
    ddim_steps=50,
    ddim_eta=1.0,
    cfg_scale=1.0,
    temporal_cfg_scale=None,
    **kwargs
):
    ddim_sampler = DDIMSampler(model)
    uncond_type = model.uncond_type
    batch_size = noise_shape[0]
    if cfg_scale != 1.0:
        if uncond_type == "empty_seq":
            prompts = batch_size * [""]
            uc_emb = model.get_learned_conditioning(prompts)
        elif uncond_type == "zero_embed":
            c_emb = cond["c_crossattn"][0] if isinstance(cond, dict) else cond
            uc_emb = paddle.zeros_like(x=c_emb)
        if hasattr(model, "embedder"):
            uc_img = paddle.zeros(shape=[noise_shape[0], 3, 224, 224])  # .to(model.place)
            uc_img = model.get_image_embeds(uc_img)
            uc_emb = paddle.concat(x=[uc_emb, uc_img], axis=1)
        if isinstance(cond, dict):
            uc = {key: cond[key] for key in cond.keys()}
            uc.update({"c_crossattn": [uc_emb]})
        else:
            uc = uc_emb
    else:
        uc = None
    x_T = None
    batch_variants = []
    for _ in range(n_samples):
        if ddim_sampler is not None:
            kwargs.update({"clean_cond": True})
            samples, _ = ddim_sampler.sample(
                S=ddim_steps,
                conditioning=cond,
                batch_size=noise_shape[0],
                shape=noise_shape[1:],
                verbose=False,
                unconditional_guidance_scale=cfg_scale,
                unconditional_conditioning=uc,
                eta=ddim_eta,
                temporal_length=noise_shape[2],
                conditional_guidance_scale_temporal=temporal_cfg_scale,
                x_T=x_T,
                **kwargs,
            )
            samples = samples.astype(model._dtype)
        batch_images = model.decode_first_stage_2DAE(samples)
        batch_variants.append(batch_images)
    batch_variants = paddle.stack(x=batch_variants, axis=1)
    return batch_variants


def get_filelist(data_dir, ext="*"):
    file_list = glob.glob(os.path.join(data_dir, "*.%s" % ext))
    file_list.sort()
    return file_list


def get_dirlist(path):
    list = []
    if os.path.exists(path):
        files = os.listdir(path)
        for file in files:
            m = os.path.join(path, file)
            if os.path.isdir(m):
                list.append(m)
    list.sort()
    return list


def load_model_checkpoint(model, ckpt):
    def load_checkpoint(model, ckpt, full_strict):
        state_dict = paddle.load(path=ckpt)
        try:
            new_pl_sd = OrderedDict()
            for key in state_dict["module"].keys():
                new_pl_sd[key[16:]] = state_dict["module"][key]
            model.set_state_dict(state_dict=new_pl_sd, use_structured_name=full_strict)
        except:
            if "state_dict" in list(state_dict.keys()):
                state_dict = state_dict["state_dict"]
            dtype = None
            for i in state_dict:
                if isinstance(state_dict[i], paddle.Tensor):
                    dtype = state_dict[i].dtype
                break
            if dtype:
                model.to(dtype=dtype)
                print("model dtype: " + str(dtype))
            model.set_state_dict(state_dict=state_dict, use_structured_name=full_strict)
        return model

    load_checkpoint(model, ckpt, full_strict=True)
    print(">>> model checkpoint loaded.")
    return model


def load_prompts(prompt_file):
    f = open(prompt_file, "r")
    prompt_list = []
    for idx, line in enumerate(f.readlines()):
        l = line.strip()
        if len(l) != 0:
            prompt_list.append(l)
        f.close()
    return prompt_list


def load_video_batch(filepath_list, frame_stride, video_size=(256, 256), video_frames=16):
    """
    Notice about some special cases:
    1. video_frames=-1 means to take all the frames (with fs=1)
    2. when the total video frames is less than required, padding strategy will be used (repreated last frame)
    """
    fps_list = []
    batch_tensor = []
    assert frame_stride > 0, "valid frame stride should be a positive interge!"
    for filepath in filepath_list:
        padding_num = 0
        vidreader = VideoReader(filepath, ctx=cpu(0), width=video_size[1], height=video_size[0])
        fps = vidreader.get_avg_fps()
        total_frames = len(vidreader)
        max_valid_frames = (total_frames - 1) // frame_stride + 1
        if video_frames < 0:
            required_frames = total_frames
            frame_stride = 1
        else:
            required_frames = video_frames
        query_frames = min(required_frames, max_valid_frames)
        frame_indices = [(frame_stride * i) for i in range(query_frames)]
        frames = vidreader.get_batch(frame_indices)
        frame_tensor = paddle.to_tensor(data=frames.asnumpy()).transpose(perm=[3, 0, 1, 2]).astype(dtype="float32")
        frame_tensor = (frame_tensor / 255.0 - 0.5) * 2
        if max_valid_frames < required_frames:
            padding_num = required_frames - max_valid_frames
            frame_tensor = paddle.concat(x=[frame_tensor, *([frame_tensor[:, -1:, :, :]] * padding_num)], axis=1)
            print(f"{os.path.split(filepath)[1]} is not long enough: {padding_num} frames padded.")
        batch_tensor.append(frame_tensor)
        sample_fps = int(fps / frame_stride)
        fps_list.append(sample_fps)
    return paddle.stack(x=batch_tensor, axis=0)


from PIL import Image


def load_image_batch(filepath_list, image_size=(256, 256)):
    batch_tensor = []
    for filepath in filepath_list:
        _, filename = os.path.split(filepath)
        _, ext = os.path.splitext(filename)
        if ext == ".mp4":
            vidreader = VideoReader(filepath, ctx=cpu(0), width=image_size[1], height=image_size[0])
            frame = vidreader.get_batch([0])
            img_tensor = (
                paddle.to_tensor(data=frame.asnumpy())
                .squeeze(axis=0)
                .transpose(perm=[2, 0, 1])
                .astype(dtype="float32")
            )
        elif ext == ".png" or ext == ".jpg":
            img = Image.open(filepath).convert("RGB")
            rgb_img = np.array(img, np.float32)
            rgb_img = cv2.resize(rgb_img, (image_size[1], image_size[0]), interpolation=cv2.INTER_LINEAR)
            img_tensor = paddle.to_tensor(data=rgb_img).transpose(perm=[2, 0, 1]).astype(dtype="float32")
        else:
            print(f"ERROR: <{ext}> image loading only support format: [mp4], [png], [jpg]")
            raise NotImplementedError
        img_tensor = (img_tensor / 255.0 - 0.5) * 2
        batch_tensor.append(img_tensor)
    return paddle.stack(x=batch_tensor, axis=0)


def save_videos(batch_tensors, savedir, filenames, fps=10):
    n_samples = tuple(batch_tensors.shape)[1]
    for idx, vid_tensor in enumerate(batch_tensors):
        video = vid_tensor.detach().cpu()
        video = paddle.clip(x=video.astype(dtype="float32"), min=-1.0, max=1.0)
        video = video.transpose(perm=[2, 0, 1, 3, 4])
        frame_grids = [make_grid(framesheet, nrow=int(n_samples)) for framesheet in video]
        grid = paddle.stack(x=frame_grids, axis=0)
        grid = (grid + 1.0) / 2.0
        grid = (grid * 255).to("uint8").transpose(perm=[0, 2, 3, 1])
        savepath = os.path.join(savedir, f"{filenames[idx]}.mp4")
        write_video(savepath, grid, fps=fps, video_codec="h264", options={"crf": "10"})
