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
from typing import Optional, Tuple, Type

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class MLPBlock(paddle.nn.Layer):
    def __init__(self, embedding_dim: int, mlp_dim: int, act: Type[paddle.nn.Layer] = paddle.nn.GELU) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class LayerNorm2d(paddle.nn.Layer):
    def __init__(self, num_channels: int, epsilon: float = 1e-06) -> None:
        super().__init__()
        self.weight = paddle.base.framework.EagerParamBase.from_tensor(tensor=paddle.ones(shape=num_channels))
        self.bias = paddle.base.framework.EagerParamBase.from_tensor(tensor=paddle.zeros(shape=num_channels))
        self.epsilon = epsilon

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        u = x.mean(axis=1, keepdim=True)
        s = (x - u).pow(y=2).mean(axis=1, keepdim=True)
        x = (x - u) / paddle.sqrt(x=s + self.epsilon)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ImageEncoderViT(paddle.nn.Layer):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Layer] = nn.LayerNorm,
        act_layer: Type[nn.Layer] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Layer): Normalization layer.
            act_layer (nn.Layer): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = img_size

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[paddle.base.framework.EagerParamBase.from_tensor] = None
        if use_abs_pos:
            self.pos_embed = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=paddle.zeros(shape=[1, img_size // patch_size, img_size // patch_size, embed_dim])
            )

        self.blocks = paddle.nn.LayerList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2D(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias_attr=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2D(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias_attr=False,
            ),
            LayerNorm2d(out_chans),
        )

        self.net_2 = nn.Conv2D(256, 512, kernel_size=3, stride=2, padding=1, bias_attr=False)
        self.net_3 = nn.Conv2D(512, 1024, kernel_size=3, stride=2, padding=1, bias_attr=False)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.neck(x.transpose([0, 3, 1, 2]))
        x = self.net_2(x)
        x = self.net_3(x)
        return x


class Block(paddle.nn.Layer):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Layer] = nn.LayerNorm,
        act_layer: Type[nn.Layer] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Layer): Normalization layer.
            act_layer (nn.Layer): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class Attention(paddle.nn.Layer):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None, "Input size must be provided if using relative positional encoding."
            self.rel_pos_h = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=paddle.zeros(shape=[2 * input_size[0] - 1, head_dim])
            )
            self.rel_pos_w = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=paddle.zeros(shape=[2 * input_size[1] - 1, head_dim])
            )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        B, H, W, _ = tuple(x.shape)
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape([B, H * W, 3, self.num_heads, -1]).transpose([2, 0, 3, 1, 4])
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape([3, B * self.num_heads, H * W, -1]).unbind(axis=0)

        attn = (q * self.scale) @ k.transpose([0, 2, 1])  # [-2, -1]

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = F.softmax(attn, axis=-1)
        x = (attn @ v).reshape([B, self.num_heads, H, W, -1]).transpose([0, 2, 3, 1, 4]).reshape([B, H, W, -1])
        x = self.proj(x)

        return x


def window_partition(x: paddle.Tensor, window_size: int) -> Tuple[paddle.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = tuple(x.shape)

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    # x.shape [1, 64, 64, 768]
    if pad_h > 0 or pad_w > 0:
        # x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h)) # torch
        x = F.pad(x, pad=(0, pad_w, 0, pad_h), data_format="NHWC")  # default NCHW
    Hp, Wp = H + pad_h, W + pad_w

    x = x.reshape([B, Hp // window_size, window_size, Wp // window_size, window_size, C])
    windows = x.transpose([0, 1, 3, 2, 4, 5]).reshape([-1, window_size, window_size, C])
    return windows, (Hp, Wp)


def window_unpartition(
    windows: paddle.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> paddle.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = tuple(windows.shape)[0] // (Hp * Wp // window_size // window_size)
    x = windows.reshape([B, Hp // window_size, Wp // window_size, window_size, window_size, -1])
    x = x.transpose([0, 1, 3, 2, 4, 5]).reshape([B, Hp, Wp, -1])
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :]
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: paddle.Tensor) -> paddle.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if tuple(rel_pos.shape)[0] != max_rel_dist:
        rel_pos_resized = paddle.nn.functional.interpolate(
            rel_pos.reshape([1, tuple(rel_pos.shape)[0], -1]).transpose([0, 2, 1]),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape([-1, max_rel_dist]).transpose([1, 0])
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = paddle.arange(end=q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = paddle.arange(end=k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = q_coords - k_coords + (k_size - 1) * max(q_size / k_size, 1.0)
    return rel_pos_resized[relative_coords.astype(dtype="int64")]


def add_decomposed_rel_pos(
    attn: paddle.Tensor,
    q: paddle.Tensor,
    rel_pos_h: paddle.Tensor,
    rel_pos_w: paddle.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> paddle.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = tuple(q.shape)
    r_q = q.reshape([B, q_h, q_w, dim])
    rel_h = paddle.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = paddle.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (attn.reshape([B, q_h, q_w, k_h, k_w]) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]).reshape(
        [B, q_h * q_w, k_h * k_w]
    )

    return attn


class PatchEmbed(paddle.nn.Layer):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()
        self.proj = nn.Conv2D(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.transpose([0, 2, 3, 1])
        return x


def build_GOT_vit_b():
    return _build_GOT_vision(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
    )


def _build_GOT_vision(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_encoder = ImageEncoderViT(
        depth=encoder_depth,
        embed_dim=encoder_embed_dim,
        img_size=image_size,
        mlp_ratio=4,
        norm_layer=partial(paddle.nn.LayerNorm, epsilon=1e-6),
        num_heads=encoder_num_heads,
        patch_size=vit_patch_size,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=encoder_global_attn_indexes,
        window_size=14,
        out_chans=prompt_embed_dim,
    )

    return image_encoder
