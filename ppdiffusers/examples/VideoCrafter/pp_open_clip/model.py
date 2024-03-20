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

from .multihead import MultiHeadAttention

""" CLIP Model

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
import copy
import logging
import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

# from .hf_model import HFTextEncoder
# from .modified_resnet import ModifiedResNet
# from .timm_model import TimmModel
from .transformer import (
    Attention,
    LayerNorm,
    LayerNormFp32,
    QuickGELU,
    TextTransformer,
    VisionTransformer,
    text_global_pool,
)
from .utils import to_2tuple


@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224
    ls_init_value: Optional[float] = None
    patch_dropout: float = 0.0
    attentional_pool: bool = False
    attn_pooler_queries: int = 256
    attn_pooler_heads: int = 8
    no_ln_pre: bool = False
    pos_embed_type: str = "learnable"
    final_ln_after_pool: bool = False
    pool_type: str = "tok"
    output_tokens: bool = False
    act_kwargs: Optional[dict] = None
    norm_kwargs: Optional[dict] = None
    timm_model_name: Optional[str] = None
    timm_model_pretrained: bool = False
    timm_pool: str = "avg"
    timm_proj: str = "linear"
    timm_proj_bias: bool = False
    timm_drop: float = 0.0
    timm_drop_path: Optional[float] = None


@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    hf_tokenizer_name: Optional[str] = None
    tokenizer_kwargs: Optional[dict] = None
    width: int = 512
    heads: int = 8
    layers: int = 12
    mlp_ratio: float = 4.0
    ls_init_value: Optional[float] = None
    embed_cls: bool = False
    pad_id: int = 0
    no_causal_mask: bool = False
    final_ln_after_pool: bool = False
    pool_type: str = "argmax"
    proj_bias: bool = False
    output_tokens: bool = False
    act_kwargs: dict = None
    norm_kwargs: dict = None
    hf_model_name: Optional[str] = None
    hf_model_pretrained: bool = True
    hf_proj_type: str = "mlp"
    hf_pooler_type: str = "mean_pooler"


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16":
        cast_dtype = "bfloat16"
    elif precision == "fp16":
        cast_dtype = "float16"
    return cast_dtype


def get_input_dtype(precision: str):
    input_dtype = None
    if precision in ("bf16", "pure_bf16"):
        input_dtype = "bfloat16"
    elif precision in ("fp16", "pure_fp16"):
        input_dtype = "float16"
    return input_dtype


def _build_vision_tower(
    embed_dim: int, vision_cfg: CLIPVisionCfg, quick_gelu: bool = False, cast_dtype: Optional[paddle.dtype] = None
):
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)
    act_layer = QuickGELU if quick_gelu else paddle.nn.GELU

    vision_heads = vision_cfg.width // vision_cfg.head_width
    norm_layer = LayerNormFp32 if cast_dtype in ("float16", "bfloat16") else LayerNorm
    if vision_cfg.norm_kwargs:
        norm_layer = partial(norm_layer, **vision_cfg.norm_kwargs)
    if vision_cfg.act_kwargs is not None:
        act_layer = partial(act_layer, **vision_cfg.act_kwargs)
    visual = VisionTransformer(
        image_size=vision_cfg.image_size,
        patch_size=vision_cfg.patch_size,
        width=vision_cfg.width,
        layers=vision_cfg.layers,
        heads=vision_heads,
        mlp_ratio=vision_cfg.mlp_ratio,
        ls_init_value=vision_cfg.ls_init_value,
        patch_dropout=vision_cfg.patch_dropout,
        attentional_pool=vision_cfg.attentional_pool,
        attn_pooler_queries=vision_cfg.attn_pooler_queries,
        attn_pooler_heads=vision_cfg.attn_pooler_heads,
        pos_embed_type=vision_cfg.pos_embed_type,
        no_ln_pre=vision_cfg.no_ln_pre,
        final_ln_after_pool=vision_cfg.final_ln_after_pool,
        pool_type=vision_cfg.pool_type,
        output_tokens=vision_cfg.output_tokens,
        output_dim=embed_dim,
        act_layer=act_layer,
        norm_layer=norm_layer,
    )
    return visual


def _build_text_tower(
    embed_dim: int, text_cfg: CLIPTextCfg, quick_gelu: bool = False, cast_dtype: Optional[paddle.dtype] = None
):
    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)

    act_layer = QuickGELU if quick_gelu else paddle.nn.GELU
    norm_layer = LayerNormFp32 if cast_dtype in ("float16", "bfloat16") else LayerNorm
    if text_cfg.norm_kwargs:
        norm_layer = partial(norm_layer, **text_cfg.norm_kwargs)
    if text_cfg.act_kwargs is not None:
        act_layer = partial(act_layer, **text_cfg.act_kwargs)
    text = TextTransformer(
        context_length=text_cfg.context_length,
        vocab_size=text_cfg.vocab_size,
        width=text_cfg.width,
        heads=text_cfg.heads,
        layers=text_cfg.layers,
        mlp_ratio=text_cfg.mlp_ratio,
        ls_init_value=text_cfg.ls_init_value,
        output_dim=embed_dim,
        embed_cls=text_cfg.embed_cls,
        no_causal_mask=text_cfg.no_causal_mask,
        pad_id=text_cfg.pad_id,
        pool_type=text_cfg.pool_type,
        proj_bias=text_cfg.proj_bias,
        output_tokens=text_cfg.output_tokens,
        act_layer=act_layer,
        norm_layer=norm_layer,
    )
    return text


class CLIP(paddle.nn.Layer):
    output_dict: []

    def __init__(
        self,
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
        init_logit_scale: float = np.log(1 / 0.07),
        init_logit_bias: Optional[float] = None,
        cast_dtype: Optional[paddle.dtype] = None,
        output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.transformer = text.transformer
        self.context_length = text.context_length
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.text_pool_type = text.pool_type
        self.register_buffer(name="attn_mask", tensor=text.attn_mask, persistable=False)
        out_17 = paddle.create_parameter(
            shape=(paddle.ones(shape=[]) * init_logit_scale).shape,
            dtype=(paddle.ones(shape=[]) * init_logit_scale).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(paddle.ones(shape=[]) * init_logit_scale),
        )
        out_17.stop_gradient = not True
        self.logit_scale = out_17
        if init_logit_bias is not None:
            out_18 = paddle.create_parameter(
                shape=(paddle.ones(shape=[]) * init_logit_bias).shape,
                dtype=(paddle.ones(shape=[]) * init_logit_bias).numpy().dtype,
                default_initializer=paddle.nn.initializer.Assign(paddle.ones(shape=[]) * init_logit_bias),
            )
            out_18.stop_gradient = not True
            self.logit_bias = out_18
        else:
            self.logit_bias = None

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.transformer.grad_checkpointing = enable

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return paddle.nn.functional.normalize(x=features, axis=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()
        x = self.token_embedding(text).to(cast_dtype)
        x = x + self.positional_embedding.to(cast_dtype)
        x = x.transpose(perm=[1, 0, 2])
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.transpose(perm=[1, 0, 2])
        x = self.ln_final(x)
        x, _ = text_global_pool(x, text, self.text_pool_type)
        if self.text_projection is not None:
            if isinstance(self.text_projection, paddle.nn.Linear):
                x = self.text_projection(x)
            else:
                x = x @ self.text_projection
        return paddle.nn.functional.normalize(x=x, axis=-1) if normalize else x

    def get_logits(self, image, text):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        image_logits = self.logit_scale.exp() * image_features @ text_features.T
        if self.logit_bias is not None:
            image_logits += self.logit_bias
        text_logits = image_logits.T
        return image_logits, text_logits

    def forward(self, image: Optional[paddle.Tensor] = None, text: Optional[paddle.Tensor] = None):
        image_features = self.encode_image(image, normalize=True) if image is not None else None
        text_features = self.encode_text(text, normalize=True) if text is not None else None
        if self.output_dict:
            out_dict = {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp(),
            }
            if self.logit_bias is not None:
                out_dict["logit_bias"] = self.logit_bias
            return out_dict
        if self.logit_bias is not None:
            return image_features, text_features, self.logit_scale.exp(), self.logit_bias
        return image_features, text_features, self.logit_scale.exp()


class CustomTextCLIP(paddle.nn.Layer):
    output_dict: []

    def __init__(
        self,
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
        init_logit_scale: float = np.log(1 / 0.07),
        init_logit_bias: Optional[float] = None,
        cast_dtype: Optional[paddle.dtype] = None,
        output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        self.text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.context_length = self.text.context_length
        self.vocab_size = self.text.vocab_size
        out_19 = paddle.create_parameter(
            shape=(paddle.ones(shape=[]) * init_logit_scale).shape,
            dtype=(paddle.ones(shape=[]) * init_logit_scale).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(paddle.ones(shape=[]) * init_logit_scale),
        )
        out_19.stop_gradient = not True
        self.logit_scale = out_19
        if init_logit_bias is not None:
            out_20 = paddle.create_parameter(
                shape=(paddle.ones(shape=[]) * init_logit_bias).shape,
                dtype=(paddle.ones(shape=[]) * init_logit_bias).numpy().dtype,
                default_initializer=paddle.nn.initializer.Assign(paddle.ones(shape=[]) * init_logit_bias),
            )
            out_20.stop_gradient = not True
            self.logit_bias = out_20
        else:
            self.logit_bias = None

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    def lock_text_tower(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        self.text.lock(unlocked_layers, freeze_layer_norm)

    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return paddle.nn.functional.normalize(x=features, axis=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        features = self.text(text)
        return paddle.nn.functional.normalize(x=features, axis=-1) if normalize else features

    def get_logits(self, image, text):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        image_logits = self.logit_scale.exp() * image_features @ text_features.T
        if self.logit_bias is not None:
            image_logits += self.logit_bias
        text_logits = image_logits.T
        return image_logits, text_logits

    def forward(self, image: Optional[paddle.Tensor] = None, text: Optional[paddle.Tensor] = None):
        image_features = self.encode_image(image, normalize=True) if image is not None else None
        text_features = self.encode_text(text, normalize=True) if text is not None else None
        if self.output_dict:
            out_dict = {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp(),
            }
            if self.logit_bias is not None:
                out_dict["logit_bias"] = self.logit_bias
            return out_dict
        if self.logit_bias is not None:
            return image_features, text_features, self.logit_scale.exp(), self.logit_bias
        return image_features, text_features, self.logit_scale.exp()


def convert_weights_to_lp(model: paddle.nn.Layer, dtype="float16"):
    """Convert applicable model parameters to low-precision (bf16 or fp16)"""

    def _convert_weights(l):
        if isinstance(l, (paddle.nn.Conv1D, paddle.nn.Conv2D, paddle.nn.Linear)):
            l.weight.data = l.weight.data.to(dtype)
            if l.bias is not None:
                l.bias.data = l.bias.data.to(dtype)
        if isinstance(l, (MultiHeadAttention, Attention)):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.to(dtype)
        if isinstance(l, (CLIP, TextTransformer)):
            attr = getattr(l, "text_projection", None)
            if attr is not None:
                attr.data = attr.data.to(dtype)
        if isinstance(l, VisionTransformer):
            attr = getattr(l, "proj", None)
            if attr is not None:
                attr.data = attr.data.to(dtype)

    model.apply(_convert_weights)


convert_weights_to_fp16 = convert_weights_to_lp


def convert_to_custom_text_state_dict(state_dict: dict):
    if "text_projection" in state_dict:
        new_state_dict = {}
        for k, v in state_dict.items():
            if any(
                k.startswith(p)
                for p in ("text_projection", "positional_embedding", "token_embedding", "transformer", "ln_final")
            ):
                k = "text." + k
            new_state_dict[k] = v
        return new_state_dict
    return state_dict


def resize_pos_embed(state_dict, model, interpolation: str = "bicubic", antialias: bool = True):
    old_pos_embed = state_dict.get("visual.positional_embedding", None)
    if old_pos_embed is None or not hasattr(model.visual, "grid_size"):
        return
    grid_size = to_2tuple(model.visual.grid_size)
    extra_tokens = 1
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
    if new_seq_len == tuple(old_pos_embed.shape)[0]:
        return
    if extra_tokens:
        pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[extra_tokens:]
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed
    old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))
    logging.info("Resizing position embedding grid-size from %s to %s", old_grid_size, grid_size)
    pos_emb_img = pos_emb_img.reshape([1, old_grid_size[0], old_grid_size[1], -1]).transpose(perm=[0, 3, 1, 2])
    pos_emb_img = paddle.nn.functional.interpolate(
        pos_emb_img, size=grid_size, mode=interpolation, antialias=antialias, align_corners=False
    )
    pos_emb_img = pos_emb_img.transpose(perm=[0, 2, 3, 1]).reshape([1, grid_size[0] * grid_size[1], -1])[0]
    if pos_emb_tok is not None:
        new_pos_embed = paddle.concat(x=[pos_emb_tok, pos_emb_img], axis=0)
    else:
        new_pos_embed = pos_emb_img
    state_dict["visual.positional_embedding"] = new_pos_embed


def resize_text_pos_embed(state_dict, model, interpolation: str = "linear", antialias: bool = False):
    old_pos_embed = state_dict.get("positional_embedding", None)
    if old_pos_embed is None:
        return
    model_pos_embed = getattr(model, "positional_embedding", None)
    if model_pos_embed is None:
        model_pos_embed = getattr(model.text, "positional_embedding", None)
    old_num_pos = tuple(old_pos_embed.shape)[0]
    old_width = tuple(old_pos_embed.shape)[1]
    num_pos = tuple(model_pos_embed.shape)[0]
    width = tuple(model_pos_embed.shape)[1]
    assert old_width == width, "text pos_embed width changed!"
    if old_num_pos == num_pos:
        return
    logging.info("Resizing text position embedding num_pos from %s to %s", old_num_pos, num_pos)
    old_pos_embed = old_pos_embed.reshape([1, old_num_pos, old_width]).transpose(perm=[0, 2, 1])
    old_pos_embed = paddle.nn.functional.interpolate(
        old_pos_embed, size=num_pos, mode=interpolation, antialias=antialias, align_corners=False
    )
    old_pos_embed = old_pos_embed.transpose(perm=[0, 2, 1])[0]
    new_pos_embed = old_pos_embed
    state_dict["positional_embedding"] = new_pos_embed


def get_model_preprocess_cfg(model):
    module = getattr(model, "visual", model)
    preprocess_cfg = getattr(module, "preprocess_cfg", {})
    if not preprocess_cfg:
        size = getattr(module, "image_size")
        if size is not None:
            preprocess_cfg["size"] = size
        mean = getattr(module, "image_mean", None)
        if mean is not None:
            preprocess_cfg["mean"] = mean
        std = getattr(module, "image_std", None)
        if std is not None:
            preprocess_cfg["std"] = std
    return preprocess_cfg


def set_model_preprocess_cfg(model, preprocess_cfg: Dict[str, Any]):
    module = getattr(model, "visual", model)
    module.image_mean = preprocess_cfg["mean"]
    module.image_std = preprocess_cfg["std"]
    module.preprocess_cfg = copy.deepcopy(preprocess_cfg)


def get_model_tokenize_cfg(model):
    module = getattr(model, "text", model)
    cfg = {}
    context_length = getattr(module, "context_length", None)
    if context_length is not None:
        cfg["context_length"] = context_length
    vocab_size = getattr(module, "vocab_size", None)
    if vocab_size is not None:
        cfg["vocab_size"] = vocab_size
    return cfg
