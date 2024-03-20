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
import pp_kornia as kornia
import utils.paddle_aux  # noqa

from .kernels import normalize_kernel2d

_VALID_BORDERS = {"constant", "reflect", "replicate", "circular"}
_VALID_PADDING = {"valid", "same"}
_VALID_BEHAVIOUR = {"conv", "corr"}


def _compute_padding(kernel_size: list[int]) -> list[int]:
    """Compute padding tuple."""
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [(k - 1) for k in kernel_size]
    out_padding = 2 * len(kernel_size) * [0]
    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]
        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front
        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear
    return out_padding


def filter2d(
    input: kornia.core.Tensor,
    kernel: kornia.core.Tensor,
    border_type: str = "reflect",
    normalized: bool = False,
    padding: str = "same",
    behaviour: str = "corr",
) -> kornia.core.Tensor:
    """Convolve a tensor with a 2d kernel.

    The function applies a given kernel to a tensor. The kernel is applied
    independently at each depth channel of the tensor. Before applying the
    kernel, the function applies padding according to the specified mode so
    that the output remains in the same shape.

    Args:
        input: the input tensor with shape of
          :math:`(B, C, H, W)`.
        kernel: the kernel to be convolved with the input
          tensor. The kernel shape must be :math:`(1, kH, kW)` or :math:`(B, kH, kW)`.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``.
        normalized: If True, kernel will be L1 normalized.
        padding: This defines the type of padding.
          2 modes available ``'same'`` or ``'valid'``.
        behaviour: defines the convolution mode -- correlation (default), using pytorch conv2d,
        or true convolution (kernel is flipped). 2 modes available ``'corr'`` or ``'conv'``.


    Return:
        Tensor: the convolved tensor of same size and numbers of channels
        as the input with shape :math:`(B, C, H, W)`.

    Example:
        >>> input = torch.tensor([[[
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 5., 0., 0.],
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 0., 0., 0.],]]])
        >>> kernel = torch.ones(1, 3, 3)
        >>> filter2d(input, kernel, padding='same')
        tensor([[[[0., 0., 0., 0., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 0., 0., 0., 0.]]]])
    """
    kornia.core.check.KORNIA_CHECK_IS_TENSOR(input)
    kornia.core.check.KORNIA_CHECK_SHAPE(input, ["B", "C", "H", "W"])
    kornia.core.check.KORNIA_CHECK_IS_TENSOR(kernel)
    kornia.core.check.KORNIA_CHECK_SHAPE(kernel, ["B", "H", "W"])
    kornia.core.check.KORNIA_CHECK(
        str(border_type).lower() in _VALID_BORDERS,
        f"Invalid border, gotcha {border_type}. Expected one of {_VALID_BORDERS}",
    )
    kornia.core.check.KORNIA_CHECK(
        str(padding).lower() in _VALID_PADDING,
        f"Invalid padding mode, gotcha {padding}. Expected one of {_VALID_PADDING}",
    )
    kornia.core.check.KORNIA_CHECK(
        str(behaviour).lower() in _VALID_BEHAVIOUR,
        f"Invalid padding mode, gotcha {behaviour}. Expected one of {_VALID_BEHAVIOUR}",
    )
    b, c, h, w = tuple(input.shape)
    if str(behaviour).lower() == "conv":
        tmp_kernel = kernel.flip(axis=(-2, -1))[:, None, ...].to(device=input.place, dtype=input.dtype)
    else:
        tmp_kernel = kernel[:, None, ...].to(device=input.place, dtype=input.dtype)
    if normalized:
        tmp_kernel = normalize_kernel2d(tmp_kernel)
    tmp_kernel = tmp_kernel.expand(shape=[-1, c, -1, -1])
    height, width = tuple(tmp_kernel.shape)[-2:]
    if padding == "same":
        padding_shape: list[int] = _compute_padding([height, width])
        input = kornia.core.pad(input, padding_shape, mode=border_type)
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.shape[0], input.shape[-2], input.shape[-1])
    output = paddle.nn.functional.conv2d(x=input, weight=tmp_kernel, groups=tmp_kernel.shape[0], padding=0, stride=1)
    if padding == "same":
        out = output.view(b, c, h, w)
    else:
        out = output.view(b, c, h - height + 1, w - width + 1)
    return out


def filter2d_separable(
    input: kornia.core.Tensor,
    kernel_x: kornia.core.Tensor,
    kernel_y: kornia.core.Tensor,
    border_type: str = "reflect",
    normalized: bool = False,
    padding: str = "same",
) -> kornia.core.Tensor:
    """Convolve a tensor with two 1d kernels, in x and y directions.

    The function applies a given kernel to a tensor. The kernel is applied
    independently at each depth channel of the tensor. Before applying the
    kernel, the function applies padding according to the specified mode so
    that the output remains in the same shape.

    Args:
        input: the input tensor with shape of
          :math:`(B, C, H, W)`.
        kernel_x: the kernel to be convolved with the input
          tensor. The kernel shape must be :math:`(1, kW)` or :math:`(B, kW)`.
        kernel_y: the kernel to be convolved with the input
          tensor. The kernel shape must be :math:`(1, kH)` or :math:`(B, kH)`.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``.
        normalized: If True, kernel will be L1 normalized.
        padding: This defines the type of padding.
          2 modes available ``'same'`` or ``'valid'``.

    Return:
        Tensor: the convolved tensor of same size and numbers of channels
        as the input with shape :math:`(B, C, H, W)`.

    Example:
        >>> input = torch.tensor([[[
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 5., 0., 0.],
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 0., 0., 0.],]]])
        >>> kernel = torch.ones(1, 3)

        >>> filter2d_separable(input, kernel, kernel, padding='same')
        tensor([[[[0., 0., 0., 0., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 0., 0., 0., 0.]]]])
    """
    out_x = filter2d(input, kernel_x[..., None, :], border_type, normalized, padding)
    out = filter2d(out_x, kernel_y[..., None], border_type, normalized, padding)
    return out


def filter3d(
    input: kornia.core.Tensor, kernel: kornia.core.Tensor, border_type: str = "replicate", normalized: bool = False
) -> kornia.core.Tensor:
    """Convolve a tensor with a 3d kernel.

    The function applies a given kernel to a tensor. The kernel is applied
    independently at each depth channel of the tensor. Before applying the
    kernel, the function applies padding according to the specified mode so
    that the output remains in the same shape.

    Args:
        input: the input tensor with shape of
          :math:`(B, C, D, H, W)`.
        kernel: the kernel to be convolved with the input
          tensor. The kernel shape must be :math:`(1, kD, kH, kW)`  or :math:`(B, kD, kH, kW)`.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``,
          ``'replicate'`` or ``'circular'``.
        normalized: If True, kernel will be L1 normalized.

    Return:
        the convolved tensor of same size and numbers of channels
        as the input with shape :math:`(B, C, D, H, W)`.

    Example:
        >>> input = torch.tensor([[[
        ...    [[0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.]],
        ...    [[0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 5., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.]],
        ...    [[0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.]]
        ... ]]])
        >>> kernel = torch.ones(1, 3, 3, 3)
        >>> filter3d(input, kernel)
        tensor([[[[[0., 0., 0., 0., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 0., 0., 0., 0.]],
        <BLANKLINE>
                  [[0., 0., 0., 0., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 0., 0., 0., 0.]],
        <BLANKLINE>
                  [[0., 0., 0., 0., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 0., 0., 0., 0.]]]]])
    """
    kornia.core.check.KORNIA_CHECK_IS_TENSOR(input)
    kornia.core.check.KORNIA_CHECK_SHAPE(input, ["B", "C", "D", "H", "W"])
    kornia.core.check.KORNIA_CHECK_IS_TENSOR(kernel)
    kornia.core.check.KORNIA_CHECK_SHAPE(kernel, ["B", "D", "H", "W"])
    kornia.core.check.KORNIA_CHECK(
        str(border_type).lower() in _VALID_BORDERS,
        f"Invalid border, gotcha {border_type}. Expected one of {_VALID_BORDERS}",
    )
    b, c, d, h, w = tuple(input.shape)
    tmp_kernel = kernel[:, None, ...].to(device=input.place, dtype=input.dtype)
    if normalized:
        bk, dk, hk, wk = tuple(kernel.shape)
        tmp_kernel = normalize_kernel2d(tmp_kernel.view(bk, dk, hk * wk)).view_as(tmp_kernel)
    tmp_kernel = tmp_kernel.expand(shape=[-1, c, -1, -1, -1])
    depth, height, width = tuple(tmp_kernel.shape)[-3:]
    padding_shape: list[int] = _compute_padding([depth, height, width])
    input_pad = kornia.core.pad(input, padding_shape, mode=border_type)
    tmp_kernel = tmp_kernel.reshape(-1, 1, depth, height, width)
    input_pad = input_pad.view(-1, tmp_kernel.shape[0], input_pad.shape[-3], input_pad.shape[-2], input_pad.shape[-1])
    output = paddle.nn.functional.conv3d(
        x=input_pad, weight=tmp_kernel, groups=tmp_kernel.shape[0], padding=0, stride=1
    )
    return output.view(b, c, d, h, w)
