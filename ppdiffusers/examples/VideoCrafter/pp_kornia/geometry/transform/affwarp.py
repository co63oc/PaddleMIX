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

from typing import Optional, Tuple, Union

import paddle

from ... import filters, utils

__all__ = ["resize"]


def _side_to_image_size(side_size: int, aspect_ratio: float, side: str = "short") -> Tuple[int, int]:
    if side not in ("short", "long", "vert", "horz"):
        raise ValueError(f"side can be one of 'short', 'long', 'vert', and 'horz'. Got '{side}'")
    if side == "vert":
        return side_size, int(side_size * aspect_ratio)
    if side == "horz":
        return int(side_size / aspect_ratio), side_size
    if (side == "short") ^ (aspect_ratio < 1.0):
        return side_size, int(side_size * aspect_ratio)
    return int(side_size / aspect_ratio), side_size


@utils.image.perform_keep_shape_image
def resize(
    input: paddle.Tensor,
    size: Union[int, Tuple[int, int]],
    interpolation: str = "bilinear",
    align_corners: Optional[bool] = None,
    side: str = "short",
    antialias: bool = False,
) -> paddle.Tensor:
    """Resize the input torch.Tensor to the given size.

    .. image:: _static/img/resize.png

    Args:
        tensor: The image tensor to be skewed with shape of :math:`(..., H, W)`.
            `...` means there can be any number of dimensions.
        size: Desired output size. If size is a sequence like (h, w),
            output size will be matched to this. If size is an int, smaller edge of the image will
            be matched to this number. i.e, if height > width, then image will be rescaled
            to (size * height / width, size)
        interpolation:  algorithm used for upsampling: ``'nearest'`` | ``'linear'`` | ``'bilinear'`` |
            'bicubic' | 'trilinear' | 'area'.
        align_corners: interpolation flag.
        side: Corresponding side if ``size`` is an integer. Can be one of ``'short'``, ``'long'``, ``'vert'``,
            or ``'horz'``.
        antialias: if True, then image will be filtered with Gaussian before downscaling.
            No effect for upscaling.

    Returns:
        The resized tensor with the shape as the specified size.

    Example:
        >>> img = torch.rand(1, 3, 4, 4)
        >>> out = resize(img, (6, 8))
        >>> print(out.shape)
        torch.Size([1, 3, 6, 8])
    """
    if not isinstance(input, paddle.Tensor):
        raise TypeError(f"Input tensor type is not a torch.Tensor. Got {type(input)}")
    if len(tuple(input.shape)) < 2:
        raise ValueError(f"Input tensor must have at least two dimensions. Got {len(tuple(input.shape))}")
    input_size = h, w = tuple(input.shape)[-2:]
    if isinstance(size, int):
        aspect_ratio = w / h
        size = _side_to_image_size(size, aspect_ratio, side)
    if size == input_size:
        return input
    factors = h / size[0], w / size[1]
    antialias = antialias and max(factors) > 1
    if antialias:
        sigmas = max((factors[0] - 1.0) / 2.0, 0.001), max((factors[1] - 1.0) / 2.0, 0.001)
        ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))
        if ks[0] % 2 == 0:
            ks = ks[0] + 1, ks[1]
        if ks[1] % 2 == 0:
            ks = ks[0], ks[1] + 1
        input = filters.gaussian_blur2d(input, ks, sigmas)
    output = paddle.nn.functional.interpolate(x=input, size=size, mode=interpolation, align_corners=align_corners)
    return output
