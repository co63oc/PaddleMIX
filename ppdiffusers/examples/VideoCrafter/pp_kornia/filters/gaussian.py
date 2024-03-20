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

from typing import Any

import pp_kornia as kornia
import pp_kornia.core.check as check
from paddle import Tensor

from .filter import filter2d, filter2d_separable
from .kernels import _unpack_2d_ks, get_gaussian_kernel1d, get_gaussian_kernel2d


def gaussian_blur2d(
    input: kornia.core.Tensor,
    kernel_size: (tuple[int, int] | int),
    sigma: (tuple[float, float] | Tensor),
    border_type: str = "reflect",
    separable: bool = True,
) -> kornia.core.Tensor:
    """Create an operator that blurs a tensor using a Gaussian filter.

    .. image:: _static/img/gaussian_blur2d.png

    The operator smooths the given tensor with a gaussian kernel by convolving
    it to each channel. It supports batched operation.

    Arguments:
        input: the input tensor with shape :math:`(B,C,H,W)`.
        kernel_size: the size of the kernel.
        sigma: the standard deviation of the kernel.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        separable: run as composition of two 1d-convolutions.

    Returns:
        the blurred tensor with shape :math:`(B, C, H, W)`.

    .. note::
       See a working example `here <https://kornia.github.io/tutorials/nbs/gaussian_blur.html>`__.

    Examples:
        >>> input = torch.rand(2, 4, 5, 5)
        >>> output = gaussian_blur2d(input, (3, 3), (1.5, 1.5))
        >>> output.shape
        torch.Size([2, 4, 5, 5])

        >>> output = gaussian_blur2d(input, (3, 3), torch.tensor([[1.5, 1.5]]))
        >>> output.shape
        torch.Size([2, 4, 5, 5])
    """
    check.KORNIA_CHECK_IS_TENSOR(input)
    if isinstance(sigma, tuple):
        sigma = kornia.core.tensor([sigma], place=input.place, dtype=input.dtype)
    else:
        check.KORNIA_CHECK_IS_TENSOR(sigma)
        sigma = sigma.to(device=input.place, dtype=input.dtype)
    if separable:
        ky, kx = _unpack_2d_ks(kernel_size)
        bs = tuple(sigma.shape)[0]
        kernel_x = get_gaussian_kernel1d(kx, sigma[:, 1].view(bs, 1))
        kernel_y = get_gaussian_kernel1d(ky, sigma[:, 0].view(bs, 1))
        out = filter2d_separable(input, kernel_x, kernel_y, border_type)
    else:
        kernel = get_gaussian_kernel2d(kernel_size, sigma)
        out = filter2d(input, kernel, border_type)
    return out


class GaussianBlur2d(kornia.core.Module):
    """Create an operator that blurs a tensor using a Gaussian filter.

    The operator smooths the given tensor with a gaussian kernel by convolving
    it to each channel. It supports batched operation.

    Arguments:
        kernel_size: the size of the kernel.
        sigma: the standard deviation of the kernel.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        separable: run as composition of two 1d-convolutions.

    Returns:
        the blurred tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples::

        >>> input = torch.rand(2, 4, 5, 5)
        >>> gauss = GaussianBlur2d((3, 3), (1.5, 1.5))
        >>> output = gauss(input)  # 2x4x5x5
        >>> output.shape
        torch.Size([2, 4, 5, 5])
    """

    def __init__(
        self,
        kernel_size: (tuple[int, int] | int),
        sigma: (tuple[float, float] | Tensor),
        border_type: str = "reflect",
        separable: bool = True,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.border_type = border_type
        self.separable = separable

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(kernel_size={self.kernel_size}, sigma={self.sigma}, border_type={self.border_type}, separable={self.separable})"

    def forward(self, input: kornia.core.Tensor) -> kornia.core.Tensor:
        return gaussian_blur2d(input, self.kernel_size, self.sigma, self.border_type, self.separable)


# @kornia.utils.deprecated(replace_with='gaussian_blur2d', version='6.9.10')
def gaussian_blur2d_t(*args: Any, **kwargs: Any) -> kornia.core.Tensor:
    return gaussian_blur2d(*args, **kwargs)
