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

import math
from typing import Callable, Optional, Sequence, Tuple

import paddle
from utils.utils import randn_tensor_mod

from .multihead import MultiHeadAttention
from .pos_embed import get_2d_sincos_pos_embed
from .utils import to_2tuple


class LayerNormFp32(paddle.nn.LayerNorm):
    """Subclass paddle's LayerNorm to handle fp16 (by casting to float32 and back)."""

    def forward(self, x: paddle.Tensor):
        orig_type = x.dtype
        x = paddle.nn.functional.layer_norm(
            x=x.to("float32"),
            normalized_shape=self.normalized_shape,
            weight=self.weight,
            bias=self.bias,
            epsilon=self.eps,
        )
        return x.to(orig_type)


class LayerNorm(paddle.nn.LayerNorm):
    """Subclass paddle's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: paddle.Tensor):
        orig_type = x.dtype
        x = paddle.nn.functional.layer_norm(
            x=x, normalized_shape=self._normalized_shape, weight=self.weight, bias=self.bias, epsilon=self._epsilon
        )
        return x.to(orig_type)


class QuickGELU(paddle.nn.Layer):
    def forward(self, x: paddle.Tensor):
        return x * paddle.nn.functional.sigmoid(x=1.702 * x)


class LayerScale(paddle.nn.Layer):
    def __init__(self, dim, init_values=1e-05, inplace=False):
        super().__init__()
        self.inplace = inplace
        out_3 = paddle.create_parameter(
            shape=(init_values * paddle.ones(shape=dim)).shape,
            dtype=(init_values * paddle.ones(shape=dim)).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(init_values * paddle.ones(shape=dim)),
        )
        out_3.stop_gradient = not True
        self.gamma = out_3

    def forward(self, x):
        return x.multiply_(y=paddle.to_tensor(self.gamma)) if self.inplace else x * self.gamma


class PatchDropout(paddle.nn.Layer):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.0
        self.prob = prob
        self.exclude_first_token = exclude_first_token

    def forward(self, x):
        if not self.training or self.prob == 0.0:
            return x
        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = paddle.jit.annotate(paddle.Tensor, x[:, :1])
        batch = tuple(x.shape)[0]
        num_tokens = tuple(x.shape)[1]
        batch_indices = paddle.arange(end=batch)
        batch_indices = batch_indices[..., None]
        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))
        rand = randn_tensor_mod(shape=[batch, num_tokens])
        patch_indices_keep = rand.topk(k=num_patches_keep, axis=-1).indices
        x = x[batch_indices, patch_indices_keep]
        if self.exclude_first_token:
            x = paddle.concat(x=(cls_tokens, x), axis=1)
        return x


class Attention(paddle.nn.Layer):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        scaled_cosine=False,
        scale_heads=False,
        logit_scale_max=math.log(1.0 / 0.01),
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.scaled_cosine = scaled_cosine
        self.scale_heads = scale_heads
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.logit_scale_max = logit_scale_max
        out_4 = paddle.create_parameter(
            shape=(paddle.randn(shape=(dim * 3, dim)) * self.scale).shape,
            dtype=(paddle.randn(shape=(dim * 3, dim)) * self.scale).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(paddle.randn(shape=(dim * 3, dim)) * self.scale),
        )
        out_4.stop_gradient = not True
        self.in_proj_weight = out_4
        if qkv_bias:
            out_5 = paddle.create_parameter(
                shape=paddle.zeros(shape=dim * 3).shape,
                dtype=paddle.zeros(shape=dim * 3).numpy().dtype,
                default_initializer=paddle.nn.initializer.Assign(paddle.zeros(shape=dim * 3)),
            )
            out_5.stop_gradient = not True
            self.in_proj_bias = out_5
        else:
            self.in_proj_bias = None
        if self.scaled_cosine:
            out_6 = paddle.create_parameter(
                shape=paddle.log(x=10 * paddle.ones(shape=(num_heads, 1, 1))).shape,
                dtype=paddle.log(x=10 * paddle.ones(shape=(num_heads, 1, 1))).numpy().dtype,
                default_initializer=paddle.nn.initializer.Assign(
                    paddle.log(x=10 * paddle.ones(shape=(num_heads, 1, 1)))
                ),
            )
            out_6.stop_gradient = not True
            self.logit_scale = out_6
        else:
            self.logit_scale = None
        self.attn_drop = paddle.nn.Dropout(p=attn_drop)
        if self.scale_heads:
            out_7 = paddle.create_parameter(
                shape=paddle.ones(shape=(num_heads, 1, 1)).shape,
                dtype=paddle.ones(shape=(num_heads, 1, 1)).numpy().dtype,
                default_initializer=paddle.nn.initializer.Assign(paddle.ones(shape=(num_heads, 1, 1))),
            )
            out_7.stop_gradient = not True
            self.head_scale = out_7
        else:
            self.head_scale = None
        self.out_proj = paddle.nn.Linear(in_features=dim, out_features=dim)
        self.out_drop = paddle.nn.Dropout(p=proj_drop)

    def forward(self, x, attn_mask: Optional[paddle.Tensor] = None):
        L, N, C = tuple(x.shape)
        q, k, v = paddle.nn.functional.linear(weight=self.in_proj_weight.T, bias=self.in_proj_bias, x=x).chunk(
            chunks=3, axis=-1
        )
        x = q.reshape([L, N * self.num_heads, -1])
        perm_0 = list(range(x.ndim))
        perm_0[0] = 1
        perm_0[1] = 0
        q = x.transpose(perm=perm_0)
        x = k.reshape([L, N * self.num_heads, -1])
        perm_1 = list(range(x.ndim))
        perm_1[0] = 1
        perm_1[1] = 0
        k = x.transpose(perm=perm_1)
        x = v.reshape([L, N * self.num_heads, -1])
        perm_2 = list(range(x.ndim))
        perm_2[0] = 1
        perm_2[1] = 0
        v = x.transpose(perm=perm_2)
        if self.logit_scale is not None:
            x = paddle.nn.functional.normalize(x=k, axis=-1)
            perm_3 = list(range(x.ndim))
            perm_3[-1] = -2
            perm_3[-2] = -1
            attn = paddle.bmm(x=paddle.nn.functional.normalize(x=q, axis=-1), y=x.transpose(perm=perm_3))
            logit_scale = paddle.clip(x=self.logit_scale, max=self.logit_scale_max).exp()
            attn = attn.reshape([N, self.num_heads, L, L]) * logit_scale
            attn = attn.reshape([-1, L, L])
        else:
            q = q * self.scale
            x = k
            perm_4 = list(range(x.ndim))
            perm_4[-1] = -2
            perm_4[-2] = -1
            attn = paddle.bmm(x=q, y=x.transpose(perm=perm_4))
        if attn_mask is not None:
            if attn_mask.dtype == "bool":
                new_attn_mask = paddle.zeros_like(x=attn_mask, dtype=q.dtype)
                new_attn_mask.masked_fill_(mask=attn_mask, value=float("-inf"))
                attn_mask = new_attn_mask
            attn += attn_mask
        attn = paddle.nn.functional.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        x = paddle.bmm(x=attn, y=v)
        if self.head_scale is not None:
            x = x.reshape([N, self.num_heads, L, C]) * self.head_scale
            x = x.reshape([-1, L, C])
        x = x
        perm_5 = list(range(x.ndim))
        perm_5[0] = 1
        perm_5[1] = 0
        x = x.transpose(perm=perm_5).reshape([L, N, C])
        x = self.out_proj(x)
        x = self.out_drop(x)
        return x


class AttentionalPooler(paddle.nn.Layer):
    def __init__(
        self, d_model: int, context_dim: int, n_head: int = 8, n_queries: int = 256, norm_layer: Callable = LayerNorm
    ):
        super().__init__()
        out_8 = paddle.create_parameter(
            shape=paddle.randn(shape=[n_queries, d_model]).shape,
            dtype=paddle.randn(shape=[n_queries, d_model]).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(paddle.randn(shape=[n_queries, d_model])),
        )
        out_8.stop_gradient = not True
        self.query = out_8
        self.attn = MultiHeadAttention(d_model, n_head, kdim=context_dim, vdim=context_dim)
        self.ln_q = norm_layer(d_model)
        self.ln_k = norm_layer(context_dim)

    def forward(self, x: paddle.Tensor):
        x = self.ln_k(x).transpose(perm=[1, 0, 2])
        N = tuple(x.shape)[1]
        q = self.ln_q(self.query)
        out = self.attn(q.unsqueeze(axis=1).expand(shape=[-1, N, -1]), x, x, need_weights=False)[0]
        return out.transpose(perm=[1, 0, 2])


class ResidualAttentionBlock(paddle.nn.Layer):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer: Callable = paddle.nn.GELU,
        norm_layer: Callable = LayerNorm,
        is_cross_attention: bool = False,
    ):
        super().__init__()
        self.ln_1 = norm_layer(d_model)
        self.attn = MultiHeadAttention(d_model, n_head)
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else paddle.nn.Identity()
        if is_cross_attention:
            self.ln_1_kv = norm_layer(d_model)
        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = paddle.nn.Sequential(
            *[
                ("c_fc", paddle.nn.Linear(in_features=d_model, out_features=mlp_width)),
                ("gelu", act_layer()),
                ("c_proj", paddle.nn.Linear(in_features=mlp_width, out_features=d_model)),
            ]
        )
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else paddle.nn.Identity()

    def attention(
        self,
        q_x: paddle.Tensor,
        k_x: Optional[paddle.Tensor] = None,
        v_x: Optional[paddle.Tensor] = None,
        attn_mask: Optional[paddle.Tensor] = None,
    ):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x
        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        return self.attn(q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(
        self,
        q_x: paddle.Tensor,
        k_x: Optional[paddle.Tensor] = None,
        v_x: Optional[paddle.Tensor] = None,
        attn_mask: Optional[paddle.Tensor] = None,
    ):
        k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None
        x = q_x + self.ls_1(self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


class CustomResidualAttentionBlock(paddle.nn.Layer):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer: Callable = paddle.nn.GELU,
        norm_layer: Callable = LayerNorm,
        scale_cosine_attn: bool = False,
        scale_heads: bool = False,
        scale_attn: bool = False,
        scale_fc: bool = False,
    ):
        super().__init__()
        self.ln_1 = norm_layer(d_model)
        self.attn = Attention(d_model, n_head, scaled_cosine=scale_cosine_attn, scale_heads=scale_heads)
        self.ln_attn = norm_layer(d_model) if scale_attn else paddle.nn.Identity()
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else paddle.nn.Identity()
        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = paddle.nn.Sequential(
            *[
                ("c_fc", paddle.nn.Linear(in_features=d_model, out_features=mlp_width)),
                ("gelu", act_layer()),
                ("ln", norm_layer(mlp_width) if scale_fc else paddle.nn.Identity()),
                ("c_proj", paddle.nn.Linear(in_features=mlp_width, out_features=d_model)),
            ]
        )
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else paddle.nn.Identity()

    def forward(self, x: paddle.Tensor, attn_mask: Optional[paddle.Tensor] = None):
        x = x + self.ls_1(self.ln_attn(self.attn(self.ln_1(x), attn_mask=attn_mask)))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


def _expand_token(token, batch_size: int):
    return token.reshape([1, 1, -1]).expand(shape=[batch_size, -1, -1])


class Transformer(paddle.nn.Layer):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer: Callable = paddle.nn.GELU,
        norm_layer: Callable = LayerNorm,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False
        self.resblocks = paddle.nn.LayerList(
            sublayers=[
                ResidualAttentionBlock(
                    width, heads, mlp_ratio, ls_init_value=ls_init_value, act_layer=act_layer, norm_layer=norm_layer
                )
                for _ in range(layers)
            ]
        )

    def get_cast_dtype(self) -> paddle.dtype:
        if hasattr(self.resblocks[0].mlp.c_fc, "int8_original_dtype"):
            return self.resblocks[0].mlp.c_fc.int8_original_dtype
        return self.resblocks[0].mlp.c_fc.weight.dtype

    def forward(self, x: paddle.Tensor, attn_mask: Optional[paddle.Tensor] = None):
        for r in self.resblocks:
            # if self.grad_checkpointing and not paddle.jit.is_scripting():
            #     x = paddle.utils.checkpoint.checkpoint(r, x, None, None,
            #         attn_mask)
            # else:
            #     x = r(x, attn_mask=attn_mask)
            x = r(x, attn_mask=attn_mask)
        return x


class VisionTransformer(paddle.nn.Layer):
    output_tokens: []

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float,
        ls_init_value: float = None,
        attentional_pool: bool = False,
        attn_pooler_queries: int = 256,
        attn_pooler_heads: int = 8,
        output_dim: int = 512,
        patch_dropout: float = 0.0,
        no_ln_pre: bool = False,
        pos_embed_type: str = "learnable",
        pool_type: str = "tok",
        final_ln_after_pool: bool = False,
        act_layer: Callable = paddle.nn.GELU,
        norm_layer: Callable = LayerNorm,
        output_tokens: bool = False,
    ):
        super().__init__()
        assert pool_type in ("tok", "avg", "none")
        self.output_tokens = output_tokens
        image_height, image_width = self.image_size = to_2tuple(image_size)
        patch_height, patch_width = self.patch_size = to_2tuple(patch_size)
        self.grid_size = (image_height // patch_height, image_width // patch_width)
        self.final_ln_after_pool = final_ln_after_pool
        self.output_dim = output_dim
        self.conv1 = paddle.nn.Conv2D(
            in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias_attr=False
        )
        scale = width**-0.5
        param1 = scale * paddle.randn(shape=[width])
        out_9 = paddle.create_parameter(
            shape=param1.shape, dtype=param1.dtype, default_initializer=paddle.nn.initializer.Assign(param1)
        )
        out_9.stop_gradient = not True
        self.class_embedding = out_9
        if pos_embed_type == "learnable":
            param2 = scale * paddle.randn(shape=[self.grid_size[0] * self.grid_size[1] + 1, width])
            out_10 = paddle.create_parameter(
                shape=param2.shape, dtype=param2.dtype, default_initializer=paddle.nn.initializer.Assign(param2)
            )
            out_10.stop_gradient = not True
            self.positional_embedding = out_10
        elif pos_embed_type == "sin_cos_2d":
            assert (
                self.grid_size[0] == self.grid_size[1]
            ), "currently sin cos 2d pos embedding only supports square input"
            param1 = paddle.zeros(shape=[self.grid_size[0] * self.grid_size[1] + 1, width])
            out_11 = paddle.create_parameter(
                shape=param1.shape, dtype=param1.dtype, default_initializer=paddle.nn.initializer.Assign(param1)
            )
            out_11.stop_gradient = not False
            self.positional_embedding = out_11
            pos_embed_type = get_2d_sincos_pos_embed(width, self.grid_size[0], cls_token=True)
            paddle.assign(
                paddle.to_tensor(data=pos_embed_type).astype(dtype="float32"), output=self.positional_embedding.data
            )
        else:
            raise ValueError
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0.0 else paddle.nn.Identity()
        self.ln_pre = paddle.nn.Identity() if no_ln_pre else norm_layer(width)
        self.transformer = Transformer(
            width, layers, heads, mlp_ratio, ls_init_value=ls_init_value, act_layer=act_layer, norm_layer=norm_layer
        )
        if attentional_pool:
            if isinstance(attentional_pool, str):
                self.attn_pool_type = attentional_pool
                self.pool_type = "none"
                if attentional_pool in ("parallel", "cascade"):
                    self.attn_pool = AttentionalPooler(
                        output_dim, width, n_head=attn_pooler_heads, n_queries=attn_pooler_queries
                    )
                    self.attn_pool_contrastive = AttentionalPooler(
                        output_dim, width, n_head=attn_pooler_heads, n_queries=1
                    )
                else:
                    assert False
            else:
                self.attn_pool_type = ""
                self.pool_type = pool_type
                self.attn_pool = AttentionalPooler(
                    output_dim, width, n_head=attn_pooler_heads, n_queries=attn_pooler_queries
                )
                self.attn_pool_contrastive = None
            pool_dim = output_dim
        else:
            self.attn_pool = None
            pool_dim = width
            self.pool_type = pool_type
        self.ln_post = norm_layer(pool_dim)
        param1 = scale * paddle.randn(shape=[pool_dim, output_dim])
        out_12 = paddle.create_parameter(
            shape=param1.shape, dtype=param1.dtype, default_initializer=paddle.nn.initializer.Assign(param1)
        )
        out_12.stop_gradient = not True
        self.proj = out_12
        self.init_parameters()

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        for param in self.parameters():
            param.stop_gradient = not False
        if unlocked_groups != 0:
            groups = [
                [self.conv1, self.class_embedding, self.positional_embedding, self.ln_pre],
                *self.transformer.resblocks[:-1],
                [self.transformer.resblocks[-1], self.ln_post],
                self.proj,
            ]

            def _unlock(x):
                if isinstance(x, Sequence):
                    for g in x:
                        _unlock(g)
                elif isinstance(x, paddle.nn.Parameter):
                    x.stop_gradient = not True
                else:
                    for p in x.parameters():
                        p.stop_gradient = not True

            _unlock(groups[-unlocked_groups:])

    def init_parameters(self):
        pass

    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def _global_pool(self, x: paddle.Tensor) -> Tuple[paddle.Tensor, paddle.Tensor]:
        if self.pool_type == "avg":
            pooled, tokens = x[:, 1:].mean(axis=1), x[:, 1:]
        elif self.pool_type == "tok":
            pooled, tokens = x[:, 0], x[:, 1:]
        else:
            pooled = tokens = x
        return pooled, tokens

    def forward(self, x: paddle.Tensor):
        x = self.conv1(x)
        x = x.reshape([tuple(x.shape)[0], tuple(x.shape)[1], -1])
        x = x.transpose(perm=[0, 2, 1])
        x = paddle.concat(x=[_expand_token(self.class_embedding, tuple(x.shape)[0]).to(dtype=x.dtype), x], axis=1)
        x = x + self.positional_embedding.to(dtype=x.dtype)
        x = self.patch_dropout(x)
        x = self.ln_pre(x)
        x = x.transpose(perm=[1, 0, 2])
        x = self.transformer(x)
        x = x.transpose(perm=[1, 0, 2])
        if self.attn_pool is not None:
            if self.attn_pool_contrastive is not None:
                x = self.ln_post(x)
                tokens = self.attn_pool(x)
                if self.attn_pool_type == "parallel":
                    pooled = self.attn_pool_contrastive(x)
                else:
                    assert self.attn_pool_type == "cascade"
                    pooled = self.attn_pool_contrastive(tokens)
            else:
                x = self.attn_pool(x)
                x = self.ln_post(x)
                pooled, tokens = self._global_pool(x)
        elif self.final_ln_after_pool:
            pooled, tokens = self._global_pool(x)
            pooled = self.ln_post(pooled)
        else:
            x = self.ln_post(x)
            pooled, tokens = self._global_pool(x)
        if self.proj is not None:
            pooled = pooled @ self.proj
        if self.output_tokens:
            return pooled, tokens
        return pooled


def text_global_pool(x, text: Optional[paddle.Tensor] = None, pool_type: str = "argmax"):
    if pool_type == "first":
        pooled, tokens = x[:, 0], x[:, 1:]
    elif pool_type == "last":
        pooled, tokens = x[:, -1], x[:, :-1]
    elif pool_type == "argmax":
        assert text is not None
        pooled, tokens = x[paddle.arange(end=tuple(x.shape)[0]), text.argmax(axis=-1)], x
    else:
        pooled = tokens = x
    return pooled, tokens


class TextTransformer(paddle.nn.Layer):
    output_tokens: []

    def __init__(
        self,
        context_length: int = 77,
        vocab_size: int = 49408,
        width: int = 512,
        heads: int = 8,
        layers: int = 12,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        output_dim: int = 512,
        embed_cls: bool = False,
        no_causal_mask: bool = False,
        pad_id: int = 0,
        pool_type: str = "argmax",
        proj_bias: bool = False,
        act_layer: Callable = paddle.nn.GELU,
        norm_layer: Callable = LayerNorm,
        output_tokens: bool = False,
    ):
        super().__init__()
        assert pool_type in ("first", "last", "argmax", "none")
        self.output_tokens = output_tokens
        self.num_pos = self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.output_dim = output_dim
        self.heads = heads
        self.pad_id = pad_id
        self.pool_type = pool_type
        self.token_embedding = paddle.nn.Embedding(num_embeddings=vocab_size, embedding_dim=width)
        if embed_cls:
            out_13 = paddle.create_parameter(
                shape=paddle.empty(shape=width).shape,
                dtype=paddle.empty(shape=width).numpy().dtype,
                default_initializer=paddle.nn.initializer.Assign(paddle.empty(shape=width)),
            )
            out_13.stop_gradient = not True
            self.cls_emb = out_13
            self.num_pos += 1
        else:
            self.cls_emb = None
        out_14 = paddle.create_parameter(
            shape=paddle.empty(shape=[self.num_pos, width]).shape,
            dtype=paddle.empty(shape=[self.num_pos, width]).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(paddle.empty(shape=[self.num_pos, width])),
        )
        out_14.stop_gradient = not True
        self.positional_embedding = out_14
        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        self.ln_final = norm_layer(width)
        if no_causal_mask:
            self.attn_mask = None
        else:
            self.register_buffer(name="attn_mask", tensor=self.build_causal_mask(), persistable=False)
        if proj_bias:
            self.text_projection = paddle.nn.Linear(in_features=width, out_features=output_dim)
        else:
            out_15 = paddle.create_parameter(
                shape=paddle.empty(shape=[width, output_dim]).shape,
                dtype=paddle.empty(shape=[width, output_dim]).numpy().dtype,
                default_initializer=paddle.nn.initializer.Assign(paddle.empty(shape=[width, output_dim])),
            )
            out_15.stop_gradient = not True
            self.text_projection = out_15
        self.init_parameters()

    def init_parameters(self):
        init_Normal = paddle.nn.initializer.Normal(std=0.02)
        init_Normal(self.token_embedding.weight)
        init_Normal = paddle.nn.initializer.Normal(std=0.01)
        init_Normal(self.positional_embedding)
        if self.cls_emb is not None:
            init_Normal = paddle.nn.initializer.Normal(std=0.01)
            init_Normal(self.cls_emb)
        proj_std = self.transformer.width**-0.5 * (2 * self.transformer.layers) ** -0.5
        attn_std = self.transformer.width**-0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            init_Normal = paddle.nn.initializer.Normal(std=attn_std)
            # init_Normal(block.attn.in_proj_weight)
            init_Normal = paddle.nn.initializer.Normal(std=proj_std)
            # init_Normal(block.attn.out_proj.weight)
            init_Normal = paddle.nn.initializer.Normal(std=fc_std)
            init_Normal(block.mlp.c_fc.weight)
            init_Normal = paddle.nn.initializer.Normal(std=proj_std)
            init_Normal(block.mlp.c_proj.weight)
        if self.text_projection is not None:
            if isinstance(self.text_projection, paddle.nn.Linear):
                init_Normal = paddle.nn.initializer.Normal(std=self.transformer.width**-0.5)
                init_Normal(self.text_projection.weight)
                if self.text_projection.bias is not None:
                    init_Constant = paddle.nn.initializer.Constant(value=0.0)
                    init_Constant(self.text_projection.bias)
            else:
                init_Normal = paddle.nn.initializer.Normal(std=self.transformer.width**-0.5)
                init_Normal(self.text_projection)

    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def build_causal_mask(self):
        mask = paddle.empty(shape=[self.num_pos, self.num_pos])
        mask.fill_(value=float("-inf"))
        mask.triu_(diagonal=1)
        return mask

    def build_cls_mask(self, text, cast_dtype: paddle.dtype):
        cls_mask = (text != self.pad_id).unsqueeze(axis=1)
        cls_mask = paddle.nn.functional.pad(pad=(1, 0, tuple(cls_mask.shape)[2], 0), value=True, x=cls_mask)
        additive_mask = paddle.empty(shape=tuple(cls_mask.shape), dtype=cast_dtype)
        additive_mask.fill_(value=0)
        additive_mask.masked_fill_(mask=~cls_mask, value=float("-inf"))
        additive_mask = paddle.repeat_interleave(x=additive_mask, repeats=self.heads, axis=0)
        return additive_mask

    def forward(self, text):
        cast_dtype = self.transformer.get_cast_dtype()
        seq_len = tuple(text.shape)[1]
        x = self.token_embedding(text).to(cast_dtype)
        attn_mask = self.attn_mask
        if self.cls_emb is not None:
            seq_len += 1
            x = paddle.concat(x=[x, _expand_token(self.cls_emb, tuple(x.shape)[0])], axis=1)
            cls_mask = self.build_cls_mask(text, cast_dtype)
            if attn_mask is not None:
                attn_mask = attn_mask[None, :seq_len, :seq_len] + cls_mask[:, :seq_len, :seq_len]
        x = x + self.positional_embedding[:seq_len].to(cast_dtype)
        x = x.transpose(perm=[1, 0, 2])
        x = self.transformer(x, attn_mask=attn_mask)
        x = x.transpose(perm=[1, 0, 2])
        if self.cls_emb is not None:
            pooled, tokens = text_global_pool(x, pool_type="last")
            pooled = self.ln_final(pooled)
        else:
            x = self.ln_final(x)
            pooled, tokens = text_global_pool(x, text, pool_type=self.pool_type)
        if self.text_projection is not None:
            if isinstance(self.text_projection, paddle.nn.Linear):
                pooled = self.text_projection(pooled)
            else:
                pooled = pooled @ self.text_projection
        if self.output_tokens:
            return pooled, tokens
        return pooled


class MultimodalTransformer(Transformer):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        context_length: int = 77,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer: Callable = paddle.nn.GELU,
        norm_layer: Callable = LayerNorm,
        output_dim: int = 512,
    ):
        super().__init__(
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        self.context_length = context_length
        self.cross_attn = paddle.nn.LayerList(
            sublayers=[
                ResidualAttentionBlock(
                    width,
                    heads,
                    mlp_ratio,
                    ls_init_value=ls_init_value,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    is_cross_attention=True,
                )
                for _ in range(layers)
            ]
        )
        self.register_buffer(name="attn_mask", tensor=self.build_attention_mask(), persistable=False)
        self.ln_final = norm_layer(width)
        out_16 = paddle.create_parameter(
            shape=paddle.empty(shape=[width, output_dim]).shape,
            dtype=paddle.empty(shape=[width, output_dim]).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(paddle.empty(shape=[width, output_dim])),
        )
        out_16.stop_gradient = not True
        self.text_projection = out_16

    def init_parameters(self):
        proj_std = self.transformer.width**-0.5 * (2 * self.transformer.layers) ** -0.5
        attn_std = self.transformer.width**-0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            init_Normal = paddle.nn.initializer.Normal(std=attn_std)
            # init_Normal(block.attn.in_proj_weight)
            init_Normal = paddle.nn.initializer.Normal(std=proj_std)
            # init_Normal(block.attn.out_proj.weight)
            init_Normal = paddle.nn.initializer.Normal(std=fc_std)
            init_Normal(block.mlp.c_fc.weight)
            init_Normal = paddle.nn.initializer.Normal(std=proj_std)
            init_Normal(block.mlp.c_proj.weight)
        for block in self.transformer.cross_attn:
            init_Normal = paddle.nn.initializer.Normal(std=attn_std)
            # init_Normal(block.attn.in_proj_weight)
            init_Normal = paddle.nn.initializer.Normal(std=proj_std)
            # init_Normal(block.attn.out_proj.weight)
            init_Normal = paddle.nn.initializer.Normal(std=fc_std)
            init_Normal(block.mlp.c_fc.weight)
            init_Normal = paddle.nn.initializer.Normal(std=proj_std)
            init_Normal(block.mlp.c_proj.weight)
        if self.text_projection is not None:
            init_Normal = paddle.nn.initializer.Normal(std=self.transformer.width**-0.5)
            init_Normal(self.text_projection)

    def build_attention_mask(self):
        mask = paddle.empty(shape=[self.context_length, self.context_length])
        mask.fill_(value=float("-inf"))
        mask.triu_(diagonal=1)
        return mask

    def forward(self, image_embs, text_embs):
        text_embs = text_embs.transpose(perm=[1, 0, 2])
        image_embs = image_embs.transpose(perm=[1, 0, 2])
        seq_len = tuple(text_embs.shape)[0]
        for resblock, cross_attn in zip(self.resblocks, self.cross_attn):
            if self.grad_checkpointing and not paddle.jit.is_scripting():
                text_embs = paddle.utils.checkpoint.checkpoint(
                    resblock, text_embs, None, None, self.attn_mask[:seq_len, :seq_len]
                )
                text_embs = paddle.utils.checkpoint.checkpoint(cross_attn, text_embs, image_embs, image_embs, None)
            else:
                text_embs = resblock(text_embs, attn_mask=self.attn_mask[:seq_len, :seq_len])
                text_embs = cross_attn(text_embs, k_x=image_embs, v_x=image_embs)
        x = text_embs.transpose(perm=[1, 0, 2])
        x = self.ln_final(x)
        if self.text_projection is not None:
            x = x @ self.text_projection
        return x

    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable
