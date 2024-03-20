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

from functools import partial

import paddle
import utils.paddle_aux  # noqa
from einops import rearrange, repeat

try:
    import xformers
    import xformers.ops
except:
    pass

XFORMERS_IS_AVAILBLE = False
from lvdm.basics import zero_module
from lvdm.common import checkpoint, default, exists


class RelativePosition(paddle.nn.Layer):
    """https://github.com/evelinehong/Transformer_Relative_Position_PyTorch/blob/master/relative_position.py"""

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        out_10 = paddle.create_parameter(
            shape=paddle.empty(shape=[max_relative_position * 2 + 1, num_units]).shape,
            dtype=paddle.empty(shape=[max_relative_position * 2 + 1, num_units]).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(
                paddle.empty(shape=[max_relative_position * 2 + 1, num_units])
            ),
        )
        out_10.stop_gradient = not True
        self.embeddings_table = out_10
        init_XavierUniform = paddle.nn.initializer.XavierUniform()
        init_XavierUniform(self.embeddings_table)

    def forward(self, length_q, length_k):
        # device = self.embeddings_table.place
        range_vec_q = paddle.arange(end=length_q)
        range_vec_k = paddle.arange(end=length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = paddle.clip(
            x=distance_mat, min=-self.max_relative_position, max=self.max_relative_position
        )
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = final_mat.astype(dtype="int64")
        embeddings = self.embeddings_table[final_mat]
        return embeddings


class CrossAttention(paddle.nn.Layer):
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        relative_position=False,
        temporal_length=None,
        img_cross_attention=False,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head**-0.5
        self.heads = heads
        self.dim_head = dim_head
        self.to_q = paddle.nn.Linear(in_features=query_dim, out_features=inner_dim, bias_attr=False)
        self.to_k = paddle.nn.Linear(in_features=context_dim, out_features=inner_dim, bias_attr=False)
        self.to_v = paddle.nn.Linear(in_features=context_dim, out_features=inner_dim, bias_attr=False)
        self.to_out = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=inner_dim, out_features=query_dim), paddle.nn.Dropout(p=dropout)
        )
        self.image_cross_attention_scale = 1.0
        self.text_context_len = 77
        self.img_cross_attention = img_cross_attention
        if self.img_cross_attention:
            self.to_k_ip = paddle.nn.Linear(in_features=context_dim, out_features=inner_dim, bias_attr=False)
            self.to_v_ip = paddle.nn.Linear(in_features=context_dim, out_features=inner_dim, bias_attr=False)
        self.relative_position = relative_position
        if self.relative_position:
            assert temporal_length is not None
            self.relative_position_k = RelativePosition(num_units=dim_head, max_relative_position=temporal_length)
            self.relative_position_v = RelativePosition(num_units=dim_head, max_relative_position=temporal_length)
        elif XFORMERS_IS_AVAILBLE and temporal_length is None:
            self.forward = self.efficient_forward

    def forward(self, x, context=None, mask=None):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        if context is not None and self.img_cross_attention:
            context, context_img = context[:, : self.text_context_len, :], context[:, self.text_context_len :, :]
            k = self.to_k(context)
            v = self.to_v(context)
            k_ip = self.to_k_ip(context_img)
            v_ip = self.to_v_ip(context_img)
        else:
            k = self.to_k(context)
            v = self.to_v(context)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))
        sim = paddle.einsum("b i d, b j d -> b i j", q, k) * self.scale
        if self.relative_position:
            len_q, len_k, len_v = tuple(q.shape)[1], tuple(k.shape)[1], tuple(v.shape)[1]
            k2 = self.relative_position_k(len_q, len_k)
            sim2 = paddle.einsum("b t d, t s d -> b t s", q, k2) * self.scale
            sim += sim2
        del k
        if exists(mask):
            max_neg_value = -paddle.finfo(paddle_aux._STR_2_PADDLE_DTYPE(sim.dtype)).max  # noqa
            mask = repeat(mask, "b i j -> (b h) i j", h=h)
            sim.masked_fill_(mask=~(mask > 0.5), value=max_neg_value)
        sim = paddle.nn.functional.softmax(sim, axis=-1)
        out = paddle.einsum("b i j, b j d -> b i d", sim, v)
        if self.relative_position:
            v2 = self.relative_position_v(len_q, len_v)
            out2 = paddle.einsum("b t s, t s d -> b t d", sim, v2)
            out += out2
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        if context is not None and self.img_cross_attention:
            k_ip, v_ip = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (k_ip, v_ip))
            sim_ip = paddle.einsum("b i d, b j d -> b i j", q, k_ip) * self.scale
            del k_ip
            sim_ip = paddle.nn.functional.softmax(sim_ip, axis=-1)
            out_ip = paddle.einsum("b i j, b j d -> b i d", sim_ip, v_ip)
            out_ip = rearrange(out_ip, "(b h) n d -> b n (h d)", h=h)
            out = out + self.image_cross_attention_scale * out_ip
        del q
        return self.to_out(out)

    def efficient_forward(self, x, context=None, mask=None):
        q = self.to_q(x)
        context = default(context, x)
        if context is not None and self.img_cross_attention:
            context, context_img = context[:, : self.text_context_len, :], context[:, self.text_context_len :, :]
            k = self.to_k(context)
            v = self.to_v(context)
            k_ip = self.to_k_ip(context_img)
            v_ip = self.to_v_ip(context_img)
        else:
            k = self.to_k(context)
            v = self.to_v(context)
        b, _, _ = tuple(q.shape)
        q, k, v = map(
            lambda t: t.unsqueeze(axis=3)
            .reshape(b, tuple(t.shape)[1], self.heads, self.dim_head)
            .transpose(perm=[0, 2, 1, 3])
            .reshape(b * self.heads, tuple(t.shape)[1], self.dim_head),
            (q, k, v),
        )
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=None)
        if context is not None and self.img_cross_attention:
            k_ip, v_ip = map(
                lambda t: t.unsqueeze(axis=3)
                .reshape(b, tuple(t.shape)[1], self.heads, self.dim_head)
                .transpose(perm=[0, 2, 1, 3])
                .reshape(b * self.heads, tuple(t.shape)[1], self.dim_head),
                (k_ip, v_ip),
            )
            out_ip = xformers.ops.memory_efficient_attention(q, k_ip, v_ip, attn_bias=None, op=None)
            out_ip = (
                out_ip.unsqueeze(axis=0)
                .reshape(b, self.heads, tuple(out.shape)[1], self.dim_head)
                .transpose(perm=[0, 2, 1, 3])
                .reshape(b, tuple(out.shape)[1], self.heads * self.dim_head)
            )
        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(axis=0)
            .reshape(b, self.heads, tuple(out.shape)[1], self.dim_head)
            .transpose(perm=[0, 2, 1, 3])
            .reshape(b, tuple(out.shape)[1], self.heads * self.dim_head)
        )
        if context is not None and self.img_cross_attention:
            out = out + self.image_cross_attention_scale * out_ip
        return self.to_out(out)


class BasicTransformerBlock(paddle.nn.Layer):
    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        disable_self_attn=False,
        attention_cls=None,
        img_cross_attention=False,
    ):
        super().__init__()
        attn_cls = CrossAttention if attention_cls is None else attention_cls
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else None,
        )
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            img_cross_attention=img_cross_attention,
        )
        self.norm1 = paddle.nn.LayerNorm(normalized_shape=dim)
        self.norm2 = paddle.nn.LayerNorm(normalized_shape=dim)
        self.norm3 = paddle.nn.LayerNorm(normalized_shape=dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None, mask=None):
        input_tuple = (x,)
        if context is not None:
            input_tuple = x, context
        if mask is not None:
            forward_mask = partial(self._forward, mask=mask)
            return checkpoint(forward_mask, (x,), self.parameters(), self.checkpoint)
        if context is not None and mask is not None:
            input_tuple = x, context, mask
        return checkpoint(self._forward, input_tuple, self.parameters(), self.checkpoint)

    def _forward(self, x, context=None, mask=None):
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None, mask=mask) + x
        x = self.attn2(self.norm2(x), context=context, mask=mask) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(paddle.nn.Layer):
    """
    Transformer block for image-like data in spatial axis.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        use_checkpoint=True,
        disable_self_attn=False,
        use_linear=False,
        img_cross_attention=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = paddle.nn.GroupNorm(
            num_groups=32, num_channels=in_channels, epsilon=1e-06, weight_attr=True, bias_attr=True
        )
        if not use_linear:
            self.proj_in = paddle.nn.Conv2D(
                in_channels=in_channels, out_channels=inner_dim, kernel_size=1, stride=1, padding=0
            )
        else:
            self.proj_in = paddle.nn.Linear(in_features=in_channels, out_features=inner_dim)
        self.transformer_blocks = paddle.nn.LayerList(
            sublayers=[
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim,
                    img_cross_attention=img_cross_attention,
                    disable_self_attn=disable_self_attn,
                    checkpoint=use_checkpoint,
                )
                for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = zero_module(
                paddle.nn.Conv2D(in_channels=inner_dim, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.proj_out = zero_module(paddle.nn.Linear(in_features=inner_dim, out_features=in_channels))
        self.use_linear = use_linear

    def forward(self, x, context=None):
        b, c, h, w = tuple(x.shape)
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context)
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in


class TemporalTransformer(paddle.nn.Layer):
    """
    Transformer block for image-like data in temporal axis.
    First, reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        use_checkpoint=True,
        use_linear=False,
        only_self_att=True,
        causal_attention=False,
        relative_position=False,
        temporal_length=None,
    ):
        super().__init__()
        self.only_self_att = only_self_att
        self.relative_position = relative_position
        self.causal_attention = causal_attention
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = paddle.nn.GroupNorm(
            num_groups=32, num_channels=in_channels, epsilon=1e-06, weight_attr=True, bias_attr=True
        )
        self.proj_in = paddle.nn.Conv1D(
            in_channels=in_channels, out_channels=inner_dim, kernel_size=1, stride=1, padding=0
        )
        if not use_linear:
            self.proj_in = paddle.nn.Conv1D(
                in_channels=in_channels, out_channels=inner_dim, kernel_size=1, stride=1, padding=0
            )
        else:
            self.proj_in = paddle.nn.Linear(in_features=in_channels, out_features=inner_dim)
        if relative_position:
            assert temporal_length is not None
            attention_cls = partial(CrossAttention, relative_position=True, temporal_length=temporal_length)
        else:
            attention_cls = None
        if self.causal_attention:
            assert temporal_length is not None
            self.mask = paddle.tril(x=paddle.ones(shape=[1, temporal_length, temporal_length]))
        if self.only_self_att:
            context_dim = None
        self.transformer_blocks = paddle.nn.LayerList(
            sublayers=[
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim,
                    attention_cls=attention_cls,
                    checkpoint=use_checkpoint,
                )
                for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = zero_module(
                paddle.nn.Conv1D(in_channels=inner_dim, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.proj_out = zero_module(paddle.nn.Linear(in_features=inner_dim, out_features=in_channels))
        self.use_linear = use_linear

    def forward(self, x, context=None):
        b, c, t, h, w = tuple(x.shape)
        x_in = x
        x = self.norm(x)
        x = rearrange(x, "b c t h w -> (b h w) c t").contiguous()
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, "bhw c t -> bhw t c").contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        if self.causal_attention:
            mask = self.mask.to(x.place)
            mask = repeat(mask, "l i j -> (l bhw) i j", bhw=b * h * w)
        else:
            mask = None
        if self.only_self_att:
            for i, block in enumerate(self.transformer_blocks):
                x = block(x, mask=mask)
            x = rearrange(x, "(b hw) t c -> b hw t c", b=b).contiguous()
        else:
            x = rearrange(x, "(b hw) t c -> b hw t c", b=b).contiguous()
            context = rearrange(context, "(b t) l con -> b t l con", t=t).contiguous()
            for i, block in enumerate(self.transformer_blocks):
                for j in range(b):
                    context_j = repeat(context[j], "t l con -> (t r) l con", r=h * w // t, t=t).contiguous()
                    x[j] = block(x[j], context=context_j)
        if self.use_linear:
            x = self.proj_out(x)
            x = rearrange(x, "b (h w) t c -> b c t h w", h=h, w=w).contiguous()
        if not self.use_linear:
            x = rearrange(x, "b hw t c -> (b hw) c t").contiguous()
            x = self.proj_out(x)
            x = rearrange(x, "(b h w) c t -> b c t h w", b=b, h=h, w=w).contiguous()
        return x + x_in


class GEGLU(paddle.nn.Layer):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = paddle.nn.Linear(in_features=dim_in, out_features=dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(chunks=2, axis=-1)
        return x * paddle.nn.functional.gelu(x=gate)


class FeedForward(paddle.nn.Layer):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = (
            paddle.nn.Sequential(paddle.nn.Linear(in_features=dim, out_features=inner_dim), paddle.nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )
        self.net = paddle.nn.Sequential(
            project_in, paddle.nn.Dropout(p=dropout), paddle.nn.Linear(in_features=inner_dim, out_features=dim_out)
        )

    def forward(self, x):
        return self.net(x)


class LinearAttention(paddle.nn.Layer):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = paddle.nn.Conv2D(in_channels=dim, out_channels=hidden_dim * 3, kernel_size=1, bias_attr=False)
        self.to_out = paddle.nn.Conv2D(in_channels=hidden_dim, out_channels=dim, kernel_size=1)

    def forward(self, x):
        b, c, h, w = tuple(x.shape)
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, "b (qkv heads c) h w -> qkv b heads c (h w)", heads=self.heads, qkv=3)
        k = paddle.nn.functional.softmax(k, axis=-1)
        context = paddle.einsum("bhdn,bhen->bhde", k, v)
        out = paddle.einsum("bhde,bhdn->bhen", context, q)
        out = rearrange(out, "b heads c (h w) -> b (heads c) h w", heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(paddle.nn.Layer):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = paddle.nn.GroupNorm(
            num_groups=32, num_channels=in_channels, epsilon=1e-06, weight_attr=True, bias_attr=True
        )
        self.q = paddle.nn.Conv2D(
            in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = paddle.nn.Conv2D(
            in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = paddle.nn.Conv2D(
            in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = paddle.nn.Conv2D(
            in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, c, h, w = tuple(q.shape)
        q = rearrange(q, "b c h w -> b (h w) c")
        k = rearrange(k, "b c h w -> b c (h w)")
        w_ = paddle.einsum("bij,bjk->bik", q, k)
        w_ = w_ * int(c) ** -0.5
        w_ = paddle.nn.functional.softmax(x=w_, axis=2)
        v = rearrange(v, "b c h w -> b c (h w)")
        w_ = rearrange(w_, "b i j -> b j i")
        h_ = paddle.einsum("bij,bjk->bik", v, w_)
        h_ = rearrange(h_, "b c (h w) -> b c h w", h=h)
        h_ = self.proj_out(h_)
        return x + h_
