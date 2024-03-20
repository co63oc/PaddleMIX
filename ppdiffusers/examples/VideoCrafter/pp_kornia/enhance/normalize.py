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
import utils.paddle_aux  # noqa

"""Module containing functionals for intensity normalisation."""
from typing import List, Tuple, Union

__all__ = ["normalize", "normalize_min_max", "denormalize", "Normalize", "Denormalize"]


class Normalize(paddle.nn.Layer):
    """Normalize a tensor image with mean and standard deviation.

    .. math::
        \\text{input[channel] = (input[channel] - mean[channel]) / std[channel]}

    Where `mean` is :math:`(M_1, ..., M_n)` and `std` :math:`(S_1, ..., S_n)` for `n` channels,

    Args:
        mean: Mean for each channel.
        std: Standard deviations for each channel.

    Shape:
        - Input: Image tensor of size :math:`(*, C, ...)`.
        - Output: Normalised tensor with same size as input :math:`(*, C, ...)`.

    Examples:
        >>> x = torch.rand(1, 4, 3, 3)
        >>> out = Normalize(0.0, 255.)(x)
        >>> out.shape
        torch.Size([1, 4, 3, 3])

        >>> x = torch.rand(1, 4, 3, 3)
        >>> mean = torch.zeros(4)
        >>> std = 255. * torch.ones(4)
        >>> out = Normalize(mean, std)(x)
        >>> out.shape
        torch.Size([1, 4, 3, 3])
    """

    def __init__(
        self,
        mean: Union[paddle.Tensor, Tuple[float], List[float], float],
        std: Union[paddle.Tensor, Tuple[float], List[float], float],
    ) -> None:
        super().__init__()
        if isinstance(mean, (int, float)):
            mean = paddle.to_tensor(data=[mean])
        if isinstance(std, (int, float)):
            std = paddle.to_tensor(data=[std])
        if isinstance(mean, (tuple, list)):
            mean = paddle.to_tensor(data=mean)
        if isinstance(std, (tuple, list)):
            std = paddle.to_tensor(data=std)
        self.mean = mean
        self.std = std

    def forward(self, input: paddle.Tensor) -> paddle.Tensor:
        return normalize(input, self.mean, self.std)

    def __repr__(self) -> str:
        repr = f"(mean={self.mean}, std={self.std})"
        return self.__class__.__name__ + repr


def normalize(data: paddle.Tensor, mean: paddle.Tensor, std: paddle.Tensor) -> paddle.Tensor:
    """Normalize an image/video tensor with mean and standard deviation.

    .. math::
        \\text{input[channel] = (input[channel] - mean[channel]) / std[channel]}

    Where `mean` is :math:`(M_1, ..., M_n)` and `std` :math:`(S_1, ..., S_n)` for `n` channels,

    Args:
        data: Image tensor of size :math:`(B, C, *)`.
        mean: Mean for each channel.
        std: Standard deviations for each channel.

    Return:
        Normalised tensor with same size as input :math:`(B, C, *)`.

    Examples:
        >>> x = torch.rand(1, 4, 3, 3)
        >>> out = normalize(x, torch.tensor([0.0]), torch.tensor([255.]))
        >>> out.shape
        torch.Size([1, 4, 3, 3])

        >>> x = torch.rand(1, 4, 3, 3)
        >>> mean = torch.zeros(4)
        >>> std = 255. * torch.ones(4)
        >>> out = normalize(x, mean, std)
        >>> out.shape
        torch.Size([1, 4, 3, 3])
    """
    shape = tuple(data.shape)
    if len(tuple(mean.shape)) == 0 or tuple(mean.shape)[0] == 1:
        mean = mean.expand(shape=shape[1])
    if len(tuple(std.shape)) == 0 or tuple(std.shape)[0] == 1:
        std = std.expand(shape=shape[1])
    if tuple(mean.shape) and tuple(mean.shape)[0] != 1:
        if tuple(mean.shape)[0] != tuple(data.shape)[1] and tuple(mean.shape)[:2] != tuple(data.shape)[:2]:
            raise ValueError(
                f"mean length and number of channels do not match. Got {tuple(mean.shape)} and {tuple(data.shape)}."
            )
    if tuple(std.shape) and tuple(std.shape)[0] != 1:
        if tuple(std.shape)[0] != tuple(data.shape)[1] and tuple(std.shape)[:2] != tuple(data.shape)[:2]:
            raise ValueError(
                f"std length and number of channels do not match. Got {tuple(std.shape)} and {tuple(data.shape)}."
            )
    mean = paddle.to_tensor(data=mean, place=data.place, dtype=data.dtype)
    std = paddle.to_tensor(data=std, place=data.place, dtype=data.dtype)
    if tuple(mean.shape):
        mean = mean[..., :, None]
    if tuple(std.shape):
        std = std[..., :, None]
    out: paddle.Tensor = (data.view(shape[0], shape[1], -1) - mean) / std
    return out.view(shape)


class Denormalize(paddle.nn.Layer):
    """Denormalize a tensor image with mean and standard deviation.

    .. math::
        \\text{input[channel] = (input[channel] * std[channel]) + mean[channel]}

    Where `mean` is :math:`(M_1, ..., M_n)` and `std` :math:`(S_1, ..., S_n)` for `n` channels,

    Args:
        mean: Mean for each channel.
        std: Standard deviations for each channel.

    Shape:
        - Input: Image tensor of size :math:`(*, C, ...)`.
        - Output: Denormalised tensor with same size as input :math:`(*, C, ...)`.

    Examples:
        >>> x = torch.rand(1, 4, 3, 3)
        >>> out = Denormalize(0.0, 255.)(x)
        >>> out.shape
        torch.Size([1, 4, 3, 3])

        >>> x = torch.rand(1, 4, 3, 3, 3)
        >>> mean = torch.zeros(1, 4)
        >>> std = 255. * torch.ones(1, 4)
        >>> out = Denormalize(mean, std)(x)
        >>> out.shape
        torch.Size([1, 4, 3, 3, 3])
    """

    def __init__(self, mean: Union[paddle.Tensor, float], std: Union[paddle.Tensor, float]) -> None:
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, input: paddle.Tensor) -> paddle.Tensor:
        return denormalize(input, self.mean, self.std)

    def __repr__(self) -> str:
        repr = f"(mean={self.mean}, std={self.std})"
        return self.__class__.__name__ + repr


def denormalize(
    data: paddle.Tensor, mean: Union[paddle.Tensor, float], std: Union[paddle.Tensor, float]
) -> paddle.Tensor:
    """Denormalize an image/video tensor with mean and standard deviation.

    .. math::
        \\text{input[channel] = (input[channel] * std[channel]) + mean[channel]}

    Where `mean` is :math:`(M_1, ..., M_n)` and `std` :math:`(S_1, ..., S_n)` for `n` channels,

    Args:
        input: Image tensor of size :math:`(B, C, *)`.
        mean: Mean for each channel.
        std: Standard deviations for each channel.

    Return:
        Denormalised tensor with same size as input :math:`(B, C, *)`.

    Examples:
        >>> x = torch.rand(1, 4, 3, 3)
        >>> out = denormalize(x, 0.0, 255.)
        >>> out.shape
        torch.Size([1, 4, 3, 3])

        >>> x = torch.rand(1, 4, 3, 3, 3)
        >>> mean = torch.zeros(1, 4)
        >>> std = 255. * torch.ones(1, 4)
        >>> out = denormalize(x, mean, std)
        >>> out.shape
        torch.Size([1, 4, 3, 3, 3])
    """
    shape = tuple(data.shape)
    if isinstance(mean, float):
        mean = paddle.to_tensor(data=[mean] * shape[1], dtype=data.dtype, place=data.place)
    if isinstance(std, float):
        std = paddle.to_tensor(data=[std] * shape[1], dtype=data.dtype, place=data.place)
    if not isinstance(data, paddle.Tensor):
        raise TypeError(f"data should be a tensor. Got {type(data)}")
    if not isinstance(mean, paddle.Tensor):
        raise TypeError(f"mean should be a tensor or a float. Got {type(mean)}")
    if not isinstance(std, paddle.Tensor):
        raise TypeError(f"std should be a tensor or float. Got {type(std)}")
    if tuple(mean.shape) and tuple(mean.shape)[0] != 1:
        if tuple(mean.shape)[0] != tuple(data.shape)[-3] and tuple(mean.shape)[:2] != tuple(data.shape)[:2]:
            raise ValueError(
                f"mean length and number of channels do not match. Got {tuple(mean.shape)} and {tuple(data.shape)}."
            )
    if tuple(std.shape) and tuple(std.shape)[0] != 1:
        if tuple(std.shape)[0] != tuple(data.shape)[-3] and tuple(std.shape)[:2] != tuple(data.shape)[:2]:
            raise ValueError(
                f"std length and number of channels do not match. Got {tuple(std.shape)} and {tuple(data.shape)}."
            )
    mean = paddle.to_tensor(data=mean, place=data.place, dtype=data.dtype)
    std = paddle.to_tensor(data=std, place=data.place, dtype=data.dtype)
    if tuple(mean.shape):
        mean = mean[..., :, None]
    if tuple(std.shape):
        std = std[..., :, None]
    out: paddle.Tensor = data.view(shape[0], shape[1], -1) * std + mean
    return out.view(shape)


def normalize_min_max(
    x: paddle.Tensor, min_val: float = 0.0, max_val: float = 1.0, eps: float = 1e-06
) -> paddle.Tensor:
    """Normalise an image/video tensor by MinMax and re-scales the value between a range.

    The data is normalised using the following formulation:

    .. math::
        y_i = (b - a) * \\frac{x_i - \\text{min}(x)}{\\text{max}(x) - \\text{min}(x)} + a

    where :math:`a` is :math:`\\text{min_val}` and :math:`b` is :math:`\\text{max_val}`.

    Args:
        x: The image tensor to be normalised with shape :math:`(B, C, *)`.
        min_val: The minimum value for the new range.
        max_val: The maximum value for the new range.
        eps: Float number to avoid zero division.

    Returns:
        The normalised image tensor with same shape as input :math:`(B, C, *)`.

    Example:
        >>> x = torch.rand(1, 5, 3, 3)
        >>> x_norm = normalize_min_max(x, min_val=-1., max_val=1.)
        >>> x_norm.min()
        tensor(-1.)
        >>> x_norm.max()
        tensor(1.0000)
    """
    if not isinstance(x, paddle.Tensor):
        raise TypeError(f"data should be a tensor. Got: {type(x)}.")
    if not isinstance(min_val, float):
        raise TypeError(f"'min_val' should be a float. Got: {type(min_val)}.")
    if not isinstance(max_val, float):
        raise TypeError(f"'b' should be a float. Got: {type(max_val)}.")
    if len(tuple(x.shape)) < 3:
        raise ValueError(f"Input shape must be at least a 3d tensor. Got: {tuple(x.shape)}.")
    shape = tuple(x.shape)
    B, C = shape[0], shape[1]
    x_min: paddle.Tensor = x.view(B, C, -1).min(-1)[0].view(B, C, 1)
    x_max: paddle.Tensor = x.view(B, C, -1).max(-1)[0].view(B, C, 1)
    x_out: paddle.Tensor = (max_val - min_val) * (x.view(B, C, -1) - x_min) / (x_max - x_min + eps) + min_val
    return x_out.view(shape)
