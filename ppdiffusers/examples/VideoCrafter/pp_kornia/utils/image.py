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

from functools import wraps
from typing import Any, Callable

from .. import core


def _to_bchw(tensor: core.Tensor) -> core.Tensor:
    """Convert a PyTorch tensor image to BCHW format.

    Args:
        tensor (torch.Tensor): image of the form :math:`(*, H, W)`.

    Returns:
        input tensor of the form :math:`(B, C, H, W)`.
    """
    if not isinstance(tensor, core.Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(tensor)}")
    if len(tuple(tensor.shape)) < 2:
        raise ValueError(f"Input size must be a two, three or four dimensional tensor. Got {tuple(tensor.shape)}")
    if len(tuple(tensor.shape)) == 2:
        tensor = tensor.unsqueeze(axis=0)
    if len(tuple(tensor.shape)) == 3:
        tensor = tensor.unsqueeze(axis=0)
    if len(tuple(tensor.shape)) > 4:
        tensor = tensor.reshape([-1, tuple(tensor.shape)[-3], tuple(tensor.shape)[-2], tuple(tensor.shape)[-1]])
    return tensor


def _to_bcdhw(tensor: core.Tensor) -> core.Tensor:
    """Convert a PyTorch tensor image to BCDHW format.

    Args:
        tensor (torch.Tensor): image of the form :math:`(*, D, H, W)`.

    Returns:
        input tensor of the form :math:`(B, C, D, H, W)`.
    """
    if not isinstance(tensor, core.Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(tensor)}")
    if len(tuple(tensor.shape)) < 3:
        raise ValueError(f"Input size must be a three, four or five dimensional tensor. Got {tuple(tensor.shape)}")
    if len(tuple(tensor.shape)) == 3:
        tensor = tensor.unsqueeze(axis=0)
    if len(tuple(tensor.shape)) == 4:
        tensor = tensor.unsqueeze(axis=0)
    if len(tuple(tensor.shape)) > 5:
        tensor = tensor.reshape(
            [-1, tuple(tensor.shape)[-4], tuple(tensor.shape)[-3], tuple(tensor.shape)[-2], tuple(tensor.shape)[-1]]
        )
    return tensor


def perform_keep_shape_image(f: Callable[..., core.Tensor]) -> Callable[..., core.Tensor]:
    """A decorator that enable `f` to be applied to an image of arbitrary leading dimensions `(*, C, H, W)`.

    It works by first viewing the image as `(B, C, H, W)`, applying the function and re-viewing the image as original
    shape.
    """

    @wraps(f)
    def _wrapper(input: core.Tensor, *args: Any, **kwargs: Any) -> core.Tensor:
        if not isinstance(input, core.Tensor):
            raise TypeError(f"Input input type is not a Tensor. Got {type(input)}")
        if input.size == 0:
            raise ValueError("Invalid input tensor, it is empty.")
        input_shape = tuple(input.shape)
        input = _to_bchw(input)
        output = f(input, *args, **kwargs)
        if len(input_shape) == 3:
            output = output[0]
        if len(input_shape) == 2:
            output = output[0, 0]
        if len(input_shape) > 4:
            output = output.reshape(input_shape[:-3] + tuple(output.shape)[-3:])
        return output

    return _wrapper
