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

import numpy as np
import paddle
import utils.paddle_aux  # noqa
from lvdm.modules.attention import LinearAttention

# diffusion + derived encoder decoder


# from utils.utils import instantiate_from_config


def nonlinearity(x):
    return x * paddle.nn.functional.sigmoid(x=x)


def Normalize(in_channels, num_groups=32):
    return paddle.nn.GroupNorm(
        num_groups=num_groups, num_channels=in_channels, epsilon=1e-06, weight_attr=True, bias_attr=True
    )


class LinAttnBlock(LinearAttention):
    """to match AttnBlock usage"""

    def __init__(self, in_channels):
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels)


class AttnBlock(paddle.nn.Layer):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)
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
        q = q.reshape(b, c, h * w)
        q = q.transpose(perm=[0, 2, 1])
        k = k.reshape(b, c, h * w)
        w_ = paddle.bmm(x=q, y=k)
        w_ = w_ * int(c) ** -0.5
        w_ = paddle.nn.functional.softmax(x=w_, axis=2)
        v = v.reshape(b, c, h * w)
        w_ = w_.transpose(perm=[0, 2, 1])
        h_ = paddle.bmm(x=v, y=w_)
        h_ = h_.reshape(b, c, h, w)
        h_ = self.proj_out(h_)
        return x + h_


def make_attn(in_channels, attn_type="vanilla"):
    assert attn_type in ["vanilla", "linear", "none"], f"attn_type {attn_type} unknown"
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    elif attn_type == "none":
        return paddle.nn.Identity(in_channels)
    else:
        return LinAttnBlock(in_channels)


class Downsample(paddle.nn.Layer):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        self.in_channels = in_channels
        if self.with_conv:
            self.conv = paddle.nn.Conv2D(
                in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=2, padding=0
            )

    def forward(self, x):
        if self.with_conv:
            pad = 0, 1, 0, 1
            x = paddle_aux._FUNCTIONAL_PAD(pad=pad, mode="constant", value=0, x=x)  # noqa
            x = self.conv(x)
        else:
            x = paddle.nn.functional.avg_pool2d(kernel_size=2, stride=2, x=x, exclusive=False)
        return x


class Upsample(paddle.nn.Layer):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        self.in_channels = in_channels
        if self.with_conv:
            self.conv = paddle.nn.Conv2D(
                in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x):
        x = paddle.nn.functional.interpolate(x=x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(tuple(timesteps.shape)) == 1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = paddle.exp(x=paddle.arange(dtype="float32", end=half_dim) * -emb)
    emb = emb.to(device=timesteps.place)
    emb = timesteps.astype(dtype="float32")[:, None] * emb[None, :]
    emb = paddle.concat(x=[paddle.sin(x=emb), paddle.cos(x=emb)], axis=1)
    if embedding_dim % 2 == 1:
        emb = paddle_aux._FUNCTIONAL_PAD(pad=(0, 1, 0, 0), x=emb)  # noqa
    return emb


class ResnetBlock(paddle.nn.Layer):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.norm1 = Normalize(in_channels)
        self.conv1 = paddle.nn.Conv2D(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1
        )
        if temb_channels > 0:
            self.temb_proj = paddle.nn.Linear(in_features=temb_channels, out_features=out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = paddle.nn.Dropout(p=dropout)
        self.conv2 = paddle.nn.Conv2D(
            in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = paddle.nn.Conv2D(
                    in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = paddle.nn.Conv2D(
                    in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h


class Model(paddle.nn.Layer):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        use_timestep=True,
        use_linear_attn=False,
        attn_type="vanilla"
    ):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = self.ch * 4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.use_timestep = use_timestep
        if self.use_timestep:
            self.temb = paddle.nn.Layer()
            self.temb.dense = paddle.nn.LayerList(
                sublayers=[
                    paddle.nn.Linear(in_features=self.ch, out_features=self.temb_ch),
                    paddle.nn.Linear(in_features=self.temb_ch, out_features=self.temb_ch),
                ]
            )
        self.conv_in = paddle.nn.Conv2D(
            in_channels=in_channels, out_channels=self.ch, kernel_size=3, stride=1, padding=1
        )
        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = paddle.nn.LayerList()
        for i_level in range(self.num_resolutions):
            block = paddle.nn.LayerList()
            attn = paddle.nn.LayerList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = paddle.nn.Layer()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)
        self.mid = paddle.nn.Layer()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
        )
        self.up = paddle.nn.LayerList()
        for i_level in reversed(range(self.num_resolutions)):
            block = paddle.nn.LayerList()
            attn = paddle.nn.LayerList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                block.append(
                    ResnetBlock(
                        in_channels=block_in + skip_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = paddle.nn.Layer()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)
        self.norm_out = Normalize(block_in)
        self.conv_out = paddle.nn.Conv2D(in_channels=block_in, out_channels=out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t=None, context=None):
        if context is not None:
            x = paddle.concat(x=(x, context), axis=1)
        if self.use_timestep:
            assert t is not None
            temb = get_timestep_embedding(t, self.ch)
            temb = self.temb.dense[0](temb)
            temb = nonlinearity(temb)
            temb = self.temb.dense[1](temb)
        else:
            temb = None
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](paddle.concat(x=[h, hs.pop()], axis=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

    def get_last_layer(self):
        return self.conv_out.weight


class Encoder(paddle.nn.Layer):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        double_z=True,
        use_linear_attn=False,
        attn_type="vanilla",
        **ignore_kwargs
    ):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.conv_in = paddle.nn.Conv2D(
            in_channels=in_channels, out_channels=self.ch, kernel_size=3, stride=1, padding=1
        )
        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = paddle.nn.LayerList()
        for i_level in range(self.num_resolutions):
            block = paddle.nn.LayerList()
            attn = paddle.nn.LayerList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = paddle.nn.Layer()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)
        self.mid = paddle.nn.Layer()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
        )
        self.norm_out = Normalize(block_in)
        self.conv_out = paddle.nn.Conv2D(
            in_channels=block_in,
            out_channels=2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        temb = None
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(paddle.nn.Layer):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        give_pre_end=False,
        tanh_out=False,
        use_linear_attn=False,
        attn_type="vanilla",
        **ignorekwargs
    ):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        # in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = 1, z_channels, curr_res, curr_res
        print("AE working on z of shape {} = {} dimensions.".format(self.z_shape, np.prod(self.z_shape)))
        self.conv_in = paddle.nn.Conv2D(
            in_channels=z_channels, out_channels=block_in, kernel_size=3, stride=1, padding=1
        )
        self.mid = paddle.nn.Layer()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
        )
        self.up = paddle.nn.LayerList()
        for i_level in reversed(range(self.num_resolutions)):
            block = paddle.nn.LayerList()
            attn = paddle.nn.LayerList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = paddle.nn.Layer()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            if len(self.up) > 0:
                self.up.insert(0, up)
            else:
                self.up.append(up)
        self.norm_out = Normalize(block_in)
        self.conv_out = paddle.nn.Conv2D(in_channels=block_in, out_channels=out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        self.last_z_shape = tuple(z.shape)
        temb = None
        h = self.conv_in(z)
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        if self.give_pre_end:
            return h
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = paddle.nn.functional.tanh(x=h)
        return h


class SimpleDecoder(paddle.nn.Layer):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__()
        self.model = paddle.nn.LayerList(
            sublayers=[
                paddle.nn.Conv2D(in_channels=in_channels, out_channels=in_channels, kernel_size=1),
                ResnetBlock(in_channels=in_channels, out_channels=2 * in_channels, temb_channels=0, dropout=0.0),
                ResnetBlock(in_channels=2 * in_channels, out_channels=4 * in_channels, temb_channels=0, dropout=0.0),
                ResnetBlock(in_channels=4 * in_channels, out_channels=2 * in_channels, temb_channels=0, dropout=0.0),
                paddle.nn.Conv2D(in_channels=2 * in_channels, out_channels=in_channels, kernel_size=1),
                Upsample(in_channels, with_conv=True),
            ]
        )
        self.norm_out = Normalize(in_channels)
        self.conv_out = paddle.nn.Conv2D(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        for i, layer in enumerate(self.model):
            if i in [1, 2, 3]:
                x = layer(x, None)
            else:
                x = layer(x)
        h = self.norm_out(x)
        h = nonlinearity(h)
        x = self.conv_out(h)
        return x


class UpsampleDecoder(paddle.nn.Layer):
    def __init__(self, in_channels, out_channels, ch, num_res_blocks, resolution, ch_mult=(2, 2), dropout=0.0):
        super().__init__()
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        block_in = in_channels
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.res_blocks = paddle.nn.LayerList()
        self.upsample_blocks = paddle.nn.LayerList()
        for i_level in range(self.num_resolutions):
            res_block = []
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                res_block.append(
                    ResnetBlock(
                        in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout
                    )
                )
                block_in = block_out
            self.res_blocks.append(paddle.nn.LayerList(sublayers=res_block))
            if i_level != self.num_resolutions - 1:
                self.upsample_blocks.append(Upsample(block_in, True))
                curr_res = curr_res * 2
        self.norm_out = Normalize(block_in)
        self.conv_out = paddle.nn.Conv2D(
            in_channels=block_in, out_channels=out_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        h = x
        for k, i_level in enumerate(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.res_blocks[i_level][i_block](h, None)
            if i_level != self.num_resolutions - 1:
                h = self.upsample_blocks[k](h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class LatentRescaler(paddle.nn.Layer):
    def __init__(self, factor, in_channels, mid_channels, out_channels, depth=2):
        super().__init__()
        self.factor = factor
        self.conv_in = paddle.nn.Conv2D(
            in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1
        )
        self.res_block1 = paddle.nn.LayerList(
            sublayers=[
                ResnetBlock(in_channels=mid_channels, out_channels=mid_channels, temb_channels=0, dropout=0.0)
                for _ in range(depth)
            ]
        )
        self.attn = AttnBlock(mid_channels)
        self.res_block2 = paddle.nn.LayerList(
            sublayers=[
                ResnetBlock(in_channels=mid_channels, out_channels=mid_channels, temb_channels=0, dropout=0.0)
                for _ in range(depth)
            ]
        )
        self.conv_out = paddle.nn.Conv2D(in_channels=mid_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv_in(x)
        for block in self.res_block1:
            x = block(x, None)
        x = paddle.nn.functional.interpolate(
            x=x, size=(int(round(tuple(x.shape)[2] * self.factor)), int(round(tuple(x.shape)[3] * self.factor)))
        )
        x = self.attn(x)
        for block in self.res_block2:
            x = block(x, None)
        x = self.conv_out(x)
        return x


class MergedRescaleEncoder(paddle.nn.Layer):
    def __init__(
        self,
        in_channels,
        ch,
        resolution,
        out_ch,
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        ch_mult=(1, 2, 4, 8),
        rescale_factor=1.0,
        rescale_module_depth=1,
    ):
        super().__init__()
        intermediate_chn = ch * ch_mult[-1]
        self.encoder = Encoder(
            in_channels=in_channels,
            num_res_blocks=num_res_blocks,
            ch=ch,
            ch_mult=ch_mult,
            z_channels=intermediate_chn,
            double_z=False,
            resolution=resolution,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            resamp_with_conv=resamp_with_conv,
            out_ch=None,
        )
        self.rescaler = LatentRescaler(
            factor=rescale_factor,
            in_channels=intermediate_chn,
            mid_channels=intermediate_chn,
            out_channels=out_ch,
            depth=rescale_module_depth,
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.rescaler(x)
        return x


class MergedRescaleDecoder(paddle.nn.Layer):
    def __init__(
        self,
        z_channels,
        out_ch,
        resolution,
        num_res_blocks,
        attn_resolutions,
        ch,
        ch_mult=(1, 2, 4, 8),
        dropout=0.0,
        resamp_with_conv=True,
        rescale_factor=1.0,
        rescale_module_depth=1,
    ):
        super().__init__()
        tmp_chn = z_channels * ch_mult[-1]
        self.decoder = Decoder(
            out_ch=out_ch,
            z_channels=tmp_chn,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            resamp_with_conv=resamp_with_conv,
            in_channels=None,
            num_res_blocks=num_res_blocks,
            ch_mult=ch_mult,
            resolution=resolution,
            ch=ch,
        )
        self.rescaler = LatentRescaler(
            factor=rescale_factor,
            in_channels=z_channels,
            mid_channels=tmp_chn,
            out_channels=tmp_chn,
            depth=rescale_module_depth,
        )

    def forward(self, x):
        x = self.rescaler(x)
        x = self.decoder(x)
        return x


class Upsampler(paddle.nn.Layer):
    def __init__(self, in_size, out_size, in_channels, out_channels, ch_mult=2):
        super().__init__()
        assert out_size >= in_size
        num_blocks = int(np.log2(out_size // in_size)) + 1
        factor_up = 1.0 + out_size % in_size
        print(
            f"Building {self.__class__.__name__} with in_size: {in_size} --> out_size {out_size} and factor {factor_up}"
        )
        self.rescaler = LatentRescaler(
            factor=factor_up, in_channels=in_channels, mid_channels=2 * in_channels, out_channels=in_channels
        )
        self.decoder = Decoder(
            out_ch=out_channels,
            resolution=out_size,
            z_channels=in_channels,
            num_res_blocks=2,
            attn_resolutions=[],
            in_channels=None,
            ch=in_channels,
            ch_mult=[ch_mult for _ in range(num_blocks)],
        )

    def forward(self, x):
        x = self.rescaler(x)
        x = self.decoder(x)
        return x


class Resize(paddle.nn.Layer):
    def __init__(self, in_channels=None, learned=False, mode="bilinear"):
        super().__init__()
        self.with_conv = learned
        self.mode = mode
        if self.with_conv:
            print(f"Note: {self.__class__.__name} uses learned downsampling and will ignore the fixed {mode} mode")
            raise NotImplementedError()
            assert in_channels is not None
            self.conv = paddle.nn.Conv2D(
                in_channels=in_channels, out_channels=in_channels, kernel_size=4, stride=2, padding=1
            )

    def forward(self, x, scale_factor=1.0):
        if scale_factor == 1.0:
            return x
        else:
            x = paddle.nn.functional.interpolate(x=x, mode=self.mode, align_corners=False, scale_factor=scale_factor)
        return x


# class FirstStagePostProcessor(paddle.nn.Layer):
#     def __init__(
#         self,
#         ch_mult: list,
#         in_channels,
#         pretrained_model: paddle.nn.Layer = None,
#         reshape=False,
#         n_channels=None,
#         dropout=0.0,
#         pretrained_config=None,
#     ):
#         super().__init__()
#         if pretrained_config is None:
#             assert pretrained_model is not None, 'Either "pretrained_model" or "pretrained_config" must not be None'
#             self.pretrained_model = pretrained_model
#         else:
#             assert pretrained_config is not None, 'Either "pretrained_model" or "pretrained_config" must not be None'
#             self.instantiate_pretrained(pretrained_config)
#         self.do_reshape = reshape
#         if n_channels is None:
#             n_channels = self.pretrained_model.encoder.ch
#         self.proj_norm = Normalize(in_channels, num_groups=in_channels // 2)
#         self.proj = paddle.nn.Conv2D(
#             in_channels=in_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=1
#         )
#         blocks = []
#         downs = []
#         ch_in = n_channels
#         for m in ch_mult:
#             blocks.append(ResnetBlock(in_channels=ch_in, out_channels=m * n_channels, dropout=dropout))
#             ch_in = m * n_channels
#             downs.append(Downsample(ch_in, with_conv=False))
#         self.model = paddle.nn.LayerList(sublayers=blocks)
#         self.downsampler = paddle.nn.LayerList(sublayers=downs)

#     def instantiate_pretrained(self, config):
#         model = instantiate_from_config(config)
#         self.pretrained_model = model.eval()
#         for param in self.pretrained_model.parameters():
#             param.stop_gradient = not False

#     @paddle.no_grad()
#     def encode_with_pretrained(self, x):
#         c = self.pretrained_model.encode(x)
#         if isinstance(c, DiagonalGaussianDistribution):
#             c = c.mode()
#         return c

#     def forward(self, x):
#         z_fs = self.encode_with_pretrained(x)
#         z = self.proj_norm(z_fs)
#         z = self.proj(z)
#         z = nonlinearity(z)
#         for submodel, downmodel in zip(self.model, self.downsampler):
#             z = submodel(z, temb=None)
#             z = downmodel(z)
#         if self.do_reshape:
#             z = rearrange(z, "b c h w -> b (h w) c")
#         return z
