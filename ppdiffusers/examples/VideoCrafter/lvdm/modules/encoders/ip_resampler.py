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

# modified from https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/helpers.py

import math

import paddle
import utils.paddle_aux  # noqa:*


class ImageProjModel(paddle.nn.Layer):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = paddle.nn.Linear(
            in_features=clip_embeddings_dim, out_features=self.clip_extra_context_tokens * cross_attention_dim
        )
        self.norm = paddle.nn.LayerNorm(normalized_shape=cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds.astype(list(self.proj.parameters())[0].dtype)
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return paddle.nn.Sequential(
        paddle.nn.LayerNorm(normalized_shape=dim),
        paddle.nn.Linear(in_features=dim, out_features=inner_dim, bias_attr=False),
        paddle.nn.GELU(),
        paddle.nn.Linear(in_features=inner_dim, out_features=dim, bias_attr=False),
    )


def reshape_tensor(x, heads):
    bs, length, width = tuple(x.shape)
    x = x.view(bs, length, heads, -1)
    x = x
    perm_0 = list(range(x.ndim))
    perm_0[1] = 2
    perm_0[2] = 1
    x = x.transpose(perm=perm_0)
    x = x.reshape(bs, heads, length, -1)
    return x


class PerceiverAttention(paddle.nn.Layer):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm1 = paddle.nn.LayerNorm(normalized_shape=dim)
        self.norm2 = paddle.nn.LayerNorm(normalized_shape=dim)
        self.to_q = paddle.nn.Linear(in_features=dim, out_features=inner_dim, bias_attr=False)
        self.to_kv = paddle.nn.Linear(in_features=dim, out_features=inner_dim * 2, bias_attr=False)
        self.to_out = paddle.nn.Linear(in_features=inner_dim, out_features=dim, bias_attr=False)

    def forward(self, x, latents):
        """
        Args:
            x (paddle.Tensor): image features
                shape (b, n1, D)
            latent (paddle.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)
        b, l, _ = tuple(latents.shape)
        q = self.to_q(latents)
        kv_input = paddle.concat(x=(x, latents), axis=-2)
        k, v = self.to_kv(kv_input).chunk(chunks=2, axis=-1)
        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        x = k * scale
        perm_1 = list(range(x.ndim))
        perm_1[-2] = -1
        perm_1[-1] = -2
        weight = q * scale @ x.transpose(perm=perm_1)
        weight = paddle.nn.functional.softmax(x=weight.astype(dtype="float32"), axis=-1).astype(weight.dtype)
        out = weight @ v
        out = out.transpose(perm=[0, 2, 1, 3]).reshape(b, l, -1)
        return self.to_out(out)


class Resampler(paddle.nn.Layer):
    def __init__(
        self, dim=1024, depth=8, dim_head=64, heads=16, num_queries=8, embedding_dim=768, output_dim=1024, ff_mult=4
    ):
        super().__init__()
        out_1 = paddle.create_parameter(
            shape=(paddle.randn(shape=[1, num_queries, dim]) / dim**0.5).shape,
            dtype=(paddle.randn(shape=[1, num_queries, dim]) / dim**0.5).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(paddle.randn(shape=[1, num_queries, dim]) / dim**0.5),
        )
        out_1.stop_gradient = not True
        self.latents = out_1
        self.proj_in = paddle.nn.Linear(in_features=embedding_dim, out_features=dim)
        self.proj_out = paddle.nn.Linear(in_features=dim, out_features=output_dim)
        self.norm_out = paddle.nn.LayerNorm(normalized_shape=output_dim)
        self.layers = paddle.nn.LayerList(sublayers=[])
        for _ in range(depth):
            self.layers.append(
                paddle.nn.LayerList(
                    sublayers=[
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x):
        latents = self.latents.repeat(x.shape[0], 1, 1)
        x = self.proj_in(x)
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        latents = self.proj_out(latents)
        return self.norm_out(latents)
