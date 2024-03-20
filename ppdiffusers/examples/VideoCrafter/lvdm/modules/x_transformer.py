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

"""shout-out to https://github.com/lucidrains/x-transformers/tree/main/x_transformers"""


from collections import namedtuple
from functools import partial
from inspect import isfunction

import paddle
from einops import rearrange, repeat
from utils import paddle_aux  # noqa

DEFAULT_DIM_HEAD = 64
Intermediates = namedtuple("Intermediates", ["pre_softmax_attn", "post_softmax_attn"])
LayerIntermediates = namedtuple("Intermediates", ["hiddens", "attn_intermediates"])


class AbsolutePositionalEmbedding(paddle.nn.Layer):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.emb = paddle.nn.Embedding(num_embeddings=max_seq_len, embedding_dim=dim)
        self.init_()

    def init_(self):
        init_Normal = paddle.nn.initializer.Normal(std=0.02)
        init_Normal(self.emb.weight)

    def forward(self, x):
        n = paddle.arange(end=tuple(x.shape)[1])
        return self.emb(n)[None, :, :]


class FixedPositionalEmbedding(paddle.nn.Layer):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / 10000 ** (paddle.arange(start=0, end=dim, step=2).astype(dtype="float32") / dim)
        self.register_buffer(name="inv_freq", tensor=inv_freq)

    def forward(self, x, seq_dim=1, offset=0):
        t = paddle.arange(end=tuple(x.shape)[seq_dim]).astype(dtype=self.inv_freq.dtype) + offset
        sinusoid_inp = paddle.einsum("i , j -> i j", t, self.inv_freq)
        emb = paddle.concat(x=(sinusoid_inp.sin(), sinusoid_inp.cos()), axis=-1)
        return emb[None, :, :]


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def always(val):
    def inner(*args, **kwargs):
        return val

    return inner


def not_equals(val):
    def inner(x):
        return x != val

    return inner


def equals(val):
    def inner(x):
        return x == val

    return inner


def max_neg_value(tensor):
    return -paddle.finfo(paddle_aux._STR_2_PADDLE_DTYPE(tensor.dtype)).max


def pick_and_pop(keys, d):
    values = list(map(lambda key: d.pop(key), keys))
    return dict(zip(keys, values))


def group_dict_by_key(cond, d):
    return_val = [dict(), dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)


def string_begins_with(prefix, str):
    return str.startswith(prefix)


def group_by_key_prefix(prefix, d):
    return group_dict_by_key(partial(string_begins_with, prefix), d)


def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix) :], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs


class Scale(paddle.nn.Layer):
    def __init__(self, value, fn):
        super().__init__()
        self.value = value
        self.fn = fn

    def forward(self, x, **kwargs):
        x, *rest = self.fn(x, **kwargs)
        return x * self.value, *rest


class Rezero(paddle.nn.Layer):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        out_2 = paddle.create_parameter(
            shape=paddle.zeros(shape=[1]).shape,
            dtype=paddle.zeros(shape=[1]).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(paddle.zeros(shape=[1])),
        )
        out_2.stop_gradient = not True
        self.g = out_2

    def forward(self, x, **kwargs):
        x, *rest = self.fn(x, **kwargs)
        return x * self.g, *rest


class ScaleNorm(paddle.nn.Layer):
    def __init__(self, dim, eps=1e-05):
        super().__init__()
        self.scale = dim**-0.5
        self.eps = eps
        out_3 = paddle.create_parameter(
            shape=paddle.ones(shape=[1]).shape,
            dtype=paddle.ones(shape=[1]).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(paddle.ones(shape=[1])),
        )
        out_3.stop_gradient = not True
        self.g = out_3

    def forward(self, x):
        norm = paddle.linalg.norm(x=x, axis=-1, keepdim=True) * self.scale
        return x / norm.clip(min=self.eps) * self.g


class RMSNorm(paddle.nn.Layer):
    def __init__(self, dim, eps=1e-08):
        super().__init__()
        self.scale = dim**-0.5
        self.eps = eps
        out_4 = paddle.create_parameter(
            shape=paddle.ones(shape=dim).shape,
            dtype=paddle.ones(shape=dim).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(paddle.ones(shape=dim)),
        )
        out_4.stop_gradient = not True
        self.g = out_4

    def forward(self, x):
        norm = paddle.linalg.norm(x=x, axis=-1, keepdim=True) * self.scale
        return x / norm.clip(min=self.eps) * self.g


class Residual(paddle.nn.Layer):
    def forward(self, x, residual):
        return x + residual


class GRUGating(paddle.nn.Layer):
    def __init__(self, dim):
        super().__init__()
        self.gru = paddle.nn.GRUCell(input_size=dim, hidden_size=dim)

    def forward(self, x, residual):
        gated_output = self.gru(rearrange(x, "b n d -> (b n) d"), rearrange(residual, "b n d -> (b n) d"))
        return gated_output.reshape(x.shape)


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


class Attention(paddle.nn.Layer):
    def __init__(
        self,
        dim,
        dim_head=DEFAULT_DIM_HEAD,
        heads=8,
        causal=False,
        mask=None,
        talking_heads=False,
        sparse_topk=None,
        use_entmax15=False,
        num_mem_kv=0,
        dropout=0.0,
        on_attn=False,
    ):
        super().__init__()
        if use_entmax15:
            raise NotImplementedError("Check out entmax activation instead of softmax activation!")
        self.scale = dim_head**-0.5
        self.heads = heads
        self.causal = causal
        self.mask = mask
        inner_dim = dim_head * heads
        self.to_q = paddle.nn.Linear(in_features=dim, out_features=inner_dim, bias_attr=False)
        self.to_k = paddle.nn.Linear(in_features=dim, out_features=inner_dim, bias_attr=False)
        self.to_v = paddle.nn.Linear(in_features=dim, out_features=inner_dim, bias_attr=False)
        self.dropout = paddle.nn.Dropout(p=dropout)
        self.talking_heads = talking_heads
        if talking_heads:
            out_5 = paddle.create_parameter(
                shape=paddle.randn(shape=[heads, heads]).shape,
                dtype=paddle.randn(shape=[heads, heads]).numpy().dtype,
                default_initializer=paddle.nn.initializer.Assign(paddle.randn(shape=[heads, heads])),
            )
            out_5.stop_gradient = not True
            self.pre_softmax_proj = out_5
            out_6 = paddle.create_parameter(
                shape=paddle.randn(shape=[heads, heads]).shape,
                dtype=paddle.randn(shape=[heads, heads]).numpy().dtype,
                default_initializer=paddle.nn.initializer.Assign(paddle.randn(shape=[heads, heads])),
            )
            out_6.stop_gradient = not True
            self.post_softmax_proj = out_6
        self.sparse_topk = sparse_topk
        self.attn_fn = paddle.nn.functional.softmax
        self.num_mem_kv = num_mem_kv
        if num_mem_kv > 0:
            out_7 = paddle.create_parameter(
                shape=paddle.randn(shape=[heads, num_mem_kv, dim_head]).shape,
                dtype=paddle.randn(shape=[heads, num_mem_kv, dim_head]).numpy().dtype,
                default_initializer=paddle.nn.initializer.Assign(paddle.randn(shape=[heads, num_mem_kv, dim_head])),
            )
            out_7.stop_gradient = not True
            self.mem_k = out_7
            out_8 = paddle.create_parameter(
                shape=paddle.randn(shape=[heads, num_mem_kv, dim_head]).shape,
                dtype=paddle.randn(shape=[heads, num_mem_kv, dim_head]).numpy().dtype,
                default_initializer=paddle.nn.initializer.Assign(paddle.randn(shape=[heads, num_mem_kv, dim_head])),
            )
            out_8.stop_gradient = not True
            self.mem_v = out_8
        self.attn_on_attn = on_attn
        self.to_out = (
            paddle.nn.Sequential(paddle.nn.Linear(in_features=inner_dim, out_features=dim * 2), paddle.nn.GLU())
            if on_attn
            else paddle.nn.Linear(in_features=inner_dim, out_features=dim)
        )

    def forward(
        self,
        x,
        context=None,
        mask=None,
        context_mask=None,
        rel_pos=None,
        sinusoidal_emb=None,
        prev_attn=None,
        mem=None,
    ):
        b, n, _, h, talking_heads, _ = *tuple(x.shape), self.heads, self.talking_heads, x.place
        kv_input = default(context, x)
        q_input = x
        k_input = kv_input
        v_input = kv_input
        if exists(mem):
            k_input = paddle.concat(x=(mem, k_input), axis=-2)
            v_input = paddle.concat(x=(mem, v_input), axis=-2)
        if exists(sinusoidal_emb):
            offset = tuple(k_input.shape)[-2] - tuple(q_input.shape)[-2]
            q_input = q_input + sinusoidal_emb(q_input, offset=offset)
            k_input = k_input + sinusoidal_emb(k_input)
        q = self.to_q(q_input)
        k = self.to_k(k_input)
        v = self.to_v(v_input)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
        input_mask = None
        if any(map(exists, (mask, context_mask))):
            q_mask = default(mask, lambda: paddle.ones(shape=(b, n)).astype(dtype="bool"))
            k_mask = q_mask if not exists(context) else context_mask
            k_mask = default(k_mask, lambda: paddle.ones(shape=(b, tuple(k.shape)[-2])).astype(dtype="bool"))
            q_mask = rearrange(q_mask, "b i -> b () i ()")
            k_mask = rearrange(k_mask, "b j -> b () () j")
            input_mask = q_mask * k_mask
        if self.num_mem_kv > 0:
            mem_k, mem_v = map(lambda t: repeat(t, "h n d -> b h n d", b=b), (self.mem_k, self.mem_v))
            k = paddle.concat(x=(mem_k, k), axis=-2)
            v = paddle.concat(x=(mem_v, v), axis=-2)
            if exists(input_mask):
                input_mask = paddle_aux._FUNCTIONAL_PAD(pad=(self.num_mem_kv, 0), value=True, x=input_mask)
        dots = paddle.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        mask_value = max_neg_value(dots)
        if exists(prev_attn):
            dots = dots + prev_attn
        pre_softmax_attn = dots
        if talking_heads:
            dots = paddle.einsum("b h i j, h k -> b k i j", dots, self.pre_softmax_proj)
        if exists(rel_pos):
            dots = rel_pos(dots)
        if exists(input_mask):
            dots.masked_fill_(mask=~input_mask, value=mask_value)
            del input_mask
        if self.causal:
            i, j = tuple(dots.shape)[-2:]
            r = paddle.arange(end=i)
            mask = rearrange(r, "i -> () () i ()") < rearrange(r, "j -> () () () j")
            mask = paddle_aux._FUNCTIONAL_PAD(pad=(j - i, 0), value=False, x=mask)
            dots.masked_fill_(mask=mask, value=mask_value)
            del mask
        if exists(self.sparse_topk) and self.sparse_topk < tuple(dots.shape)[-1]:
            top, _ = dots.topk(k=self.sparse_topk, axis=-1)
            vk = top[..., -1].unsqueeze(axis=-1).expand_as(y=dots)
            mask = dots < vk
            dots.masked_fill_(mask=mask, value=mask_value)
            del mask
        attn = self.attn_fn(dots, dim=-1)
        post_softmax_attn = attn
        attn = self.dropout(attn)
        if talking_heads:
            attn = paddle.einsum("b h i j, h k -> b k i j", attn, self.post_softmax_proj)
        out = paddle.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        intermediates = Intermediates(pre_softmax_attn=pre_softmax_attn, post_softmax_attn=post_softmax_attn)
        return self.to_out(out), intermediates


class AttentionLayers(paddle.nn.Layer):
    def __init__(
        self,
        dim,
        depth,
        heads=8,
        causal=False,
        cross_attend=False,
        only_cross=False,
        use_scalenorm=False,
        use_rmsnorm=False,
        use_rezero=False,
        rel_pos_num_buckets=32,
        rel_pos_max_distance=128,
        position_infused_attn=False,
        custom_layers=None,
        sandwich_coef=None,
        par_ratio=None,
        residual_attn=False,
        cross_residual_attn=False,
        macaron=False,
        pre_norm=True,
        gate_residual=False,
        **kwargs
    ):
        super().__init__()
        ff_kwargs, kwargs = groupby_prefix_and_trim("ff_", kwargs)
        attn_kwargs, _ = groupby_prefix_and_trim("attn_", kwargs)
        # dim_head = attn_kwargs.get("dim_head", DEFAULT_DIM_HEAD)
        self.dim = dim
        self.depth = depth
        self.layers = paddle.nn.LayerList(sublayers=[])
        self.has_pos_emb = position_infused_attn
        self.pia_pos_emb = FixedPositionalEmbedding(dim) if position_infused_attn else None
        self.rotary_pos_emb = always(None)
        assert (
            rel_pos_num_buckets <= rel_pos_max_distance
        ), "number of relative position buckets must be less than the relative position max distance"
        self.rel_pos = None
        self.pre_norm = pre_norm
        self.residual_attn = residual_attn
        self.cross_residual_attn = cross_residual_attn
        norm_class = ScaleNorm if use_scalenorm else paddle.nn.LayerNorm
        norm_class = RMSNorm if use_rmsnorm else norm_class
        norm_fn = partial(norm_class, dim)
        norm_fn = paddle.nn.Identity if use_rezero else norm_fn
        branch_fn = Rezero if use_rezero else None
        if cross_attend and not only_cross:
            default_block = "a", "c", "f"
        elif cross_attend and only_cross:
            default_block = "c", "f"
        else:
            default_block = "a", "f"
        if macaron:
            default_block = ("f",) + default_block
        if exists(custom_layers):
            layer_types = custom_layers
        elif exists(par_ratio):
            par_depth = depth * len(default_block)
            assert 1 < par_ratio <= par_depth, "par ratio out of range"
            default_block = tuple(filter(not_equals("f"), default_block))
            par_attn = par_depth // par_ratio
            depth_cut = par_depth * 2 // 3
            par_width = (depth_cut + depth_cut // par_attn) // par_attn
            assert len(default_block) <= par_width, "default block is too large for par_ratio"
            par_block = default_block + ("f",) * (par_width - len(default_block))
            par_head = par_block * par_attn
            layer_types = par_head + ("f",) * (par_depth - len(par_head))
        elif exists(sandwich_coef):
            assert sandwich_coef > 0 and sandwich_coef <= depth, "sandwich coefficient should be less than the depth"
            layer_types = ("a",) * sandwich_coef + default_block * (depth - sandwich_coef) + ("f",) * sandwich_coef
        else:
            layer_types = default_block * depth
        self.layer_types = layer_types
        self.num_attn_layers = len(list(filter(equals("a"), layer_types)))
        for layer_type in self.layer_types:
            if layer_type == "a":
                layer = Attention(dim, heads=heads, causal=causal, **attn_kwargs)
            elif layer_type == "c":
                layer = Attention(dim, heads=heads, **attn_kwargs)
            elif layer_type == "f":
                layer = FeedForward(dim, **ff_kwargs)
                layer = layer if not macaron else Scale(0.5, layer)
            else:
                raise Exception(f"invalid layer type {layer_type}")
            if isinstance(layer, Attention) and exists(branch_fn):
                layer = branch_fn(layer)
            if gate_residual:
                residual_fn = GRUGating(dim)
            else:
                residual_fn = Residual()
            self.layers.append(paddle.nn.LayerList(sublayers=[norm_fn(), layer, residual_fn]))

    def forward(self, x, context=None, mask=None, context_mask=None, mems=None, return_hiddens=False):
        hiddens = []
        intermediates = []
        prev_attn = None
        prev_cross_attn = None
        mems = mems.copy() if exists(mems) else [None] * self.num_attn_layers
        for ind, (layer_type, (norm, block, residual_fn)) in enumerate(zip(self.layer_types, self.layers)):
            is_last = ind == len(self.layers) - 1
            if layer_type == "a":
                hiddens.append(x)
                layer_mem = mems.pop(0)
            residual = x
            if self.pre_norm:
                x = norm(x)
            if layer_type == "a":
                out, inter = block(
                    x,
                    mask=mask,
                    sinusoidal_emb=self.pia_pos_emb,
                    rel_pos=self.rel_pos,
                    prev_attn=prev_attn,
                    mem=layer_mem,
                )
            elif layer_type == "c":
                out, inter = block(x, context=context, mask=mask, context_mask=context_mask, prev_attn=prev_cross_attn)
            elif layer_type == "f":
                out = block(x)
            x = residual_fn(out, residual)
            if layer_type in ("a", "c"):
                intermediates.append(inter)
            if layer_type == "a" and self.residual_attn:
                prev_attn = inter.pre_softmax_attn
            elif layer_type == "c" and self.cross_residual_attn:
                prev_cross_attn = inter.pre_softmax_attn
            if not self.pre_norm and not is_last:
                x = norm(x)
        if return_hiddens:
            intermediates = LayerIntermediates(hiddens=hiddens, attn_intermediates=intermediates)
            return x, intermediates
        return x


class Encoder(AttentionLayers):
    def __init__(self, **kwargs):
        assert "causal" not in kwargs, "cannot set causality on encoder"
        super().__init__(causal=False, **kwargs)


class TransformerWrapper(paddle.nn.Layer):
    def __init__(
        self,
        *,
        num_tokens,
        max_seq_len,
        attn_layers,
        emb_dim=None,
        max_mem_len=0.0,
        emb_dropout=0.0,
        num_memory_tokens=None,
        tie_embedding=False,
        use_pos_emb=True
    ):
        super().__init__()
        assert isinstance(attn_layers, AttentionLayers), "attention layers must be one of Encoder or Decoder"
        dim = attn_layers.dim
        emb_dim = default(emb_dim, dim)
        self.max_seq_len = max_seq_len
        self.max_mem_len = max_mem_len
        self.num_tokens = num_tokens
        self.token_emb = paddle.nn.Embedding(num_embeddings=num_tokens, embedding_dim=emb_dim)
        self.pos_emb = (
            AbsolutePositionalEmbedding(emb_dim, max_seq_len)
            if use_pos_emb and not attn_layers.has_pos_emb
            else always(0)
        )
        self.emb_dropout = paddle.nn.Dropout(p=emb_dropout)
        self.project_emb = (
            paddle.nn.Linear(in_features=emb_dim, out_features=dim) if emb_dim != dim else paddle.nn.Identity()
        )
        self.attn_layers = attn_layers
        self.norm = paddle.nn.LayerNorm(normalized_shape=dim)
        self.init_()
        self.to_logits = (
            paddle.nn.Linear(in_features=dim, out_features=num_tokens)
            if not tie_embedding
            else lambda t: t @ self.token_emb.weight.t()
        )
        num_memory_tokens = default(num_memory_tokens, 0)
        self.num_memory_tokens = num_memory_tokens
        if num_memory_tokens > 0:
            out_9 = paddle.create_parameter(
                shape=paddle.randn(shape=[num_memory_tokens, dim]).shape,
                dtype=paddle.randn(shape=[num_memory_tokens, dim]).numpy().dtype,
                default_initializer=paddle.nn.initializer.Assign(paddle.randn(shape=[num_memory_tokens, dim])),
            )
            out_9.stop_gradient = not True
            self.memory_tokens = out_9
            if hasattr(attn_layers, "num_memory_tokens"):
                attn_layers.num_memory_tokens = num_memory_tokens

    def init_(self):
        init_Normal = paddle.nn.initializer.Normal(std=0.02)
        init_Normal(self.token_emb.weight)

    def forward(
        self, x, return_embeddings=False, mask=None, return_mems=False, return_attn=False, mems=None, **kwargs
    ):
        b, _, _, num_mem = *tuple(x.shape), x.place, self.num_memory_tokens
        x = self.token_emb(x)
        x += self.pos_emb(x)
        x = self.emb_dropout(x)
        x = self.project_emb(x)
        if num_mem > 0:
            mem = repeat(self.memory_tokens, "n d -> b n d", b=b)
            x = paddle.concat(x=(mem, x), axis=1)
            if exists(mask):
                mask = paddle_aux._FUNCTIONAL_PAD(pad=(num_mem, 0), value=True, x=mask)
        x, intermediates = self.attn_layers(x, mask=mask, mems=mems, return_hiddens=True, **kwargs)
        x = self.norm(x)
        mem, x = x[:, :num_mem], x[:, num_mem:]
        out = self.to_logits(x) if not return_embeddings else x
        if return_mems:
            hiddens = intermediates.hiddens
            new_mems = (
                list(map(lambda pair: paddle.concat(x=pair, axis=-2), zip(mems, hiddens))) if exists(mems) else hiddens
            )
            new_mems = list(map(lambda t: t[..., -self.max_mem_len :, :].detach(), new_mems))
            return out, new_mems
        if return_attn:
            attn_maps = list(map(lambda t: t.post_softmax_attn, intermediates.attn_intermediates))
            return out, attn_maps
        return out
