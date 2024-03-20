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

from math import sqrt
from typing import Optional

import paddle
import pp_kornia as kornia
from pp_kornia.core import Tensor


def _check_kernel_size(kernel_size: (tuple[int, ...] | int), min_value: int = 0, allow_even: bool = False) -> None:
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,)
    fmt = "even or odd" if allow_even else "odd"
    for size in kernel_size:
        kornia.core.check.KORNIA_CHECK(
            isinstance(size, int) and ((size % 2 == 1 or allow_even) and size > min_value),
            f"Kernel size must be an {fmt} integer bigger than {min_value}. Gotcha {size} on {kernel_size}",
        )


def _unpack_2d_ks(kernel_size: (tuple[int, int] | int)) -> tuple[int, int]:
    if isinstance(kernel_size, int):
        ky = kx = kernel_size
    else:
        kornia.core.check.KORNIA_CHECK(len(kernel_size) == 2, "2D Kernel size should have a length of 2.")
        ky, kx = kernel_size
    ky = int(ky)
    kx = int(kx)
    return ky, kx


def _unpack_3d_ks(kernel_size: (tuple[int, int, int] | int)) -> tuple[int, int, int]:
    if isinstance(kernel_size, int):
        kz = ky = kx = kernel_size
    else:
        kornia.core.check.KORNIA_CHECK(len(kernel_size) == 3, "3D Kernel size should have a length of 3.")
        kz, ky, kx = kernel_size
    kz = int(kz)
    ky = int(ky)
    kx = int(kx)
    return kz, ky, kx


def normalize_kernel2d(input: kornia.core.Tensor) -> kornia.core.Tensor:
    """Normalize both derivative and smoothing kernel."""
    kornia.core.check.KORNIA_CHECK_SHAPE(input, ["*", "H", "W"])
    norm = input.abs().sum(axis=-1).sum(axis=-1)
    return input / norm[..., None, None]


def gaussian(
    window_size: int,
    sigma: (Tensor | float),
    *,
    device: Optional[kornia.core.Device] = None,
    dtype: Optional[kornia.core.Dtype] = None
) -> kornia.core.Tensor:
    """Compute the gaussian values based on the window and sigma values.

    Args:
        window_size: the size which drives the filter amount.
        sigma: gaussian standard deviation. If a tensor, should be in a shape :math:`(B, 1)`
        device: This value will be used if sigma is a float. Device desired to compute.
        dtype: This value will be used if sigma is a float. Dtype desired for compute.
    Returns:
        A tensor withshape :math:`(B, 	ext{kernel_size})`, with Gaussian values.
    """
    if isinstance(sigma, float):
        sigma = kornia.core.tensor([[sigma]], device=device, dtype=dtype)
    kornia.core.check.KORNIA_CHECK_IS_TENSOR(sigma)
    kornia.core.check.KORNIA_CHECK_SHAPE(sigma, ["B", "1"])
    batch_size = tuple(sigma.shape)[0]
    x = (paddle.arange(dtype=sigma.dtype, end=window_size) - window_size // 2).expand(shape=[batch_size, -1])
    if window_size % 2 == 0:
        x = x + 0.5
    gauss = paddle.exp(x=-x.pow(y=2.0) / (2 * sigma.pow(y=2.0)))
    return gauss / gauss.sum(axis=-1, keepdim=True)


def gaussian_discrete_erf(
    window_size: int,
    sigma: (Tensor | float),
    *,
    device: Optional[kornia.core.Device] = None,
    dtype: Optional[kornia.core.Dtype] = None
) -> kornia.core.Tensor:
    """Discrete Gaussian by interpolating the error function.

    Adapted from: https://github.com/Project-MONAI/MONAI/blob/master/monai/networks/layers/convutils.py
    Args:
        window_size: the size which drives the filter amount.
        sigma: gaussian standard deviation. If a tensor, should be in a shape :math:`(B, 1)`
        device: This value will be used if sigma is a float. Device desired to compute.
        dtype: This value will be used if sigma is a float. Dtype desired for compute.
    Returns:
        A tensor withshape :math:`(B, \\text{kernel_size})`, with discrete Gaussian values computed by approximation of
        the error function.
    """
    if isinstance(sigma, float):
        sigma = kornia.core.tensor([[sigma]], device=device, dtype=dtype)
    kornia.core.check.KORNIA_CHECK_SHAPE(sigma, ["B", "1"])
    batch_size = tuple(sigma.shape)[0]
    x = (paddle.arange(dtype=sigma.dtype, end=window_size) - window_size // 2).expand(shape=[batch_size, -1])
    t = 0.70710678 / sigma.abs()
    gauss = 0.5 * ((t * (x + 0.5)).erf() - (t * (x - 0.5)).erf())
    gauss = gauss.clip(min=0)
    return gauss / gauss.sum(axis=-1, keepdim=True)


def _modified_bessel_0(x: kornia.core.Tensor) -> kornia.core.Tensor:
    """Adapted from:

    https://github.com/Project-MONAI/MONAI/blob/master/monai/networks/layers/convutils.py
    """
    ax = paddle.abs(x=x)
    out = kornia.core.zeros_like(x)
    idx_a = ax < 3.75
    if idx_a.astype("bool").any():
        y = x[idx_a] / 3.75 * (x[idx_a] / 3.75)
        out[idx_a] = 1.0 + y * (
            3.5156229 + y * (3.0899424 + y * (1.2067492 + y * (0.2659732 + y * (0.0360768 + y * 0.0045813))))
        )
    idx_b = ~idx_a
    if idx_b.astype("bool").any():
        y = 3.75 / ax[idx_b]
        ans = 0.00916281 + y * (-0.02057706 + y * (0.02635537 + y * (-0.01647633 + y * 0.00392377)))
        coef = 0.39894228 + y * (0.01328592 + y * (0.00225319 + y * (-0.00157565 + y * ans)))
        out[idx_b] = ax[idx_b].exp() / ax[idx_b].sqrt() * coef
    return out


def _modified_bessel_1(x: kornia.core.Tensor) -> kornia.core.Tensor:
    """Adapted from:

    https://github.com/Project-MONAI/MONAI/blob/master/monai/networks/layers/convutils.py
    """
    ax = paddle.abs(x=x)
    out = kornia.core.zeros_like(x)
    idx_a = ax < 3.75
    if idx_a.astype("bool").any():
        y = x[idx_a] / 3.75 * (x[idx_a] / 3.75)
        ans = 0.51498869 + y * (0.15084934 + y * (0.02658733 + y * (0.00301532 + y * 0.00032411)))
        out[idx_a] = ax[idx_a] * (0.5 + y * (0.87890594 + y * ans))
    idx_b = ~idx_a
    if idx_b.astype("bool").any():
        y = 3.75 / ax[idx_b]
        ans = 0.02282967 + y * (-0.02895312 + y * (0.01787654 - y * 0.00420059))
        ans = 0.39894228 + y * (-0.03988024 + y * (-0.00362018 + y * (0.00163801 + y * (-0.01031555 + y * ans))))
        ans = ans * ax[idx_b].exp() / ax[idx_b].sqrt()
        out[idx_b] = kornia.core.where(x[idx_b] < 0, -ans, ans)
    return out


def _modified_bessel_i(n: int, x: kornia.core.Tensor) -> kornia.core.Tensor:
    """Adapted from:

    https://github.com/Project-MONAI/MONAI/blob/master/monai/networks/layers/convutils.py
    """
    kornia.core.check.KORNIA_CHECK(n >= 2, "n must be greater than 1.99")
    if (x == 0.0).astype("bool").all():
        return x
    batch_size = tuple(x.shape)[0]
    tox = 2.0 / x.abs()
    ans = kornia.core.zeros(batch_size, 1, device=x.place, dtype=x.dtype)
    bip = kornia.core.zeros(batch_size, 1, device=x.place, dtype=x.dtype)
    bi = paddle.ones(shape=[batch_size, 1], dtype=x.dtype)
    m = int(2 * (n + int(sqrt(40.0 * n))))
    for j in range(m, 0, -1):
        bim = bip + float(j) * tox * bi
        bip = bi
        bi = bim
        idx = bi.abs() > 10000000000.0
        if idx.astype("bool").any():
            ans[idx] = ans[idx] * 1e-10
            bi[idx] = bi[idx] * 1e-10
            bip[idx] = bip[idx] * 1e-10
        if j == n:
            ans = bip
    out = ans * _modified_bessel_0(x) / bi
    if n % 2 == 1:
        out = kornia.core.where(x < 0.0, -out, out)
    out = kornia.core.where(x == 0.0, x, out)
    return out


def gaussian_discrete(
    window_size: int,
    sigma: (Tensor | float),
    *,
    device: Optional[kornia.core.Device] = None,
    dtype: Optional[kornia.core.Dtype] = None
) -> kornia.core.Tensor:
    """Discrete Gaussian kernel based on the modified Bessel functions.

    Adapted from: https://github.com/Project-MONAI/MONAI/blob/master/monai/networks/layers/convutils.py
    Args:
        window_size: the size which drives the filter amount.
        sigma: gaussian standard deviation. If a tensor, should be in a shape :math:`(B, 1)`
        device: This value will be used if sigma is a float. Device desired to compute.
        dtype: This value will be used if sigma is a float. Dtype desired for compute.
    Returns:
        A tensor withshape :math:`(B, \\text{kernel_size})`, with discrete Gaussian values computed by modified Bessel
        function.
    """
    if isinstance(sigma, float):
        sigma = kornia.core.tensor([[sigma]], device=device, dtype=dtype)
    kornia.core.check.KORNIA_CHECK_SHAPE(sigma, ["B", "1"])
    sigma2 = sigma * sigma
    tail = int(window_size // 2) + 1
    bessels = [
        _modified_bessel_0(sigma2),
        _modified_bessel_1(sigma2),
        *(_modified_bessel_i(k, sigma2) for k in range(2, tail)),
    ]
    out = kornia.core.concatenate(bessels[:0:-1] + bessels, -1) * sigma2.exp()
    return out / out.sum(axis=-1, keepdim=True)


def laplacian_1d(
    window_size: int, *, device: Optional[kornia.core.Device] = None, dtype: kornia.core.Dtype = "float32"
) -> kornia.core.Tensor:
    """One could also use the Laplacian of Gaussian formula to design the filter."""
    filter_1d = paddle.ones(shape=window_size, dtype=dtype)
    middle = window_size // 2
    filter_1d[middle] = 1 - window_size
    return filter_1d


def get_box_kernel1d(
    kernel_size: int, *, device: Optional[kornia.core.Device] = None, dtype: Optional[kornia.core.Dtype] = None
) -> kornia.core.Tensor:
    """Utility function that returns a 1-D box filter.

    Args:
        kernel_size: the size of the kernel.
        device: the desired device of returned tensor.
        dtype: the desired data type of returned tensor.
    Returns:
        A tensor with shape :math:`(1, \\text{kernel\\_size})`, filled with the value
        :math:`\\frac{1}{\\text{kernel\\_size}}`.
    """
    scale = kornia.core.tensor(1.0 / kernel_size, device=device, dtype=dtype)
    return scale.expand(shape=[1, kernel_size])


def get_box_kernel2d(
    kernel_size: (tuple[int, int] | int),
    *,
    device: Optional[kornia.core.Device] = None,
    dtype: Optional[kornia.core.Dtype] = None
) -> kornia.core.Tensor:
    """Utility function that returns a 2-D box filter.

    Args:
        kernel_size: the size of the kernel.
        device: the desired device of returned tensor.
        dtype: the desired data type of returned tensor.
    Returns:
        A tensor with shape :math:`(1, \\text{kernel\\_size}[0], \\text{kernel\\_size}[1])`,
        filled with the value :math:`\\frac{1}{\\text{kernel\\_size}[0] \\times \\text{kernel\\_size}[1]}`.
    """
    ky, kx = _unpack_2d_ks(kernel_size)
    scale = kornia.core.tensor(1.0 / (kx * ky), device=device, dtype=dtype)
    return scale.expand(shape=[1, ky, kx])


def get_binary_kernel2d(
    window_size: (tuple[int, int] | int),
    *,
    device: Optional[kornia.core.Device] = None,
    dtype: kornia.core.Dtype = "float32"
) -> kornia.core.Tensor:
    """Create a binary kernel to extract the patches.

    If the window size is HxW will create a (H*W)x1xHxW kernel.
    """
    ky, kx = _unpack_2d_ks(window_size)
    window_range = kx * ky
    kernel = kornia.core.zeros((window_range, window_range), device=device, dtype=dtype)
    idx = paddle.arange(end=window_range)
    kernel[idx, idx] += 1.0
    return kernel.view(window_range, 1, ky, kx)


def get_sobel_kernel_3x3(
    *, device: Optional[kornia.core.Device] = None, dtype: Optional[kornia.core.Dtype] = None
) -> kornia.core.Tensor:
    """Utility function that returns a sobel kernel of 3x3."""
    return kornia.core.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=device, dtype=dtype)


def get_sobel_kernel_5x5_2nd_order(
    *, device: Optional[kornia.core.Device] = None, dtype: Optional[kornia.core.Dtype] = None
) -> kornia.core.Tensor:
    """Utility function that returns a 2nd order sobel kernel of 5x5."""
    return kornia.core.tensor(
        [
            [-1.0, 0.0, 2.0, 0.0, -1.0],
            [-4.0, 0.0, 8.0, 0.0, -4.0],
            [-6.0, 0.0, 12.0, 0.0, -6.0],
            [-4.0, 0.0, 8.0, 0.0, -4.0],
            [-1.0, 0.0, 2.0, 0.0, -1.0],
        ],
        device=device,
        dtype=dtype,
    )


def _get_sobel_kernel_5x5_2nd_order_xy(
    *, device: Optional[kornia.core.Device] = None, dtype: Optional[kornia.core.Dtype] = None
) -> kornia.core.Tensor:
    """Utility function that returns a 2nd order sobel kernel of 5x5."""
    return kornia.core.tensor(
        [
            [-1.0, -2.0, 0.0, 2.0, 1.0],
            [-2.0, -4.0, 0.0, 4.0, 2.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 4.0, 0.0, -4.0, -2.0],
            [1.0, 2.0, 0.0, -2.0, -1.0],
        ],
        device=device,
        dtype=dtype,
    )


def get_diff_kernel_3x3(
    *, device: Optional[kornia.core.Device] = None, dtype: Optional[kornia.core.Dtype] = None
) -> kornia.core.Tensor:
    """Utility function that returns a first order derivative kernel of 3x3."""
    return kornia.core.tensor([[-0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [-0.0, 0.0, 0.0]], device=device, dtype=dtype)


def get_diff_kernel3d(
    device: Optional[kornia.core.Device] = None, dtype: Optional[kornia.core.Dtype] = None
) -> kornia.core.Tensor:
    """Utility function that returns a first order derivative kernel of 3x3x3."""
    kernel = kornia.core.tensor(
        [
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [-0.5, 0.0, 0.5], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, -0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.5, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [0.0, -0.5, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.0]],
            ],
        ],
        device=device,
        dtype=dtype,
    )
    return kernel[:, None, ...]


def get_diff_kernel3d_2nd_order(
    device: Optional[kornia.core.Device] = None, dtype: Optional[kornia.core.Dtype] = None
) -> kornia.core.Tensor:
    """Utility function that returns a first order derivative kernel of 3x3x3."""
    kernel = kornia.core.tensor(
        [
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [0.0, -2.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, -2.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[1.0, 0.0, -1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 1.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, -1.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
            ],
        ],
        device=device,
        dtype=dtype,
    )
    return kernel[:, None, ...]


def get_sobel_kernel2d(
    *, device: Optional[kornia.core.Device] = None, dtype: Optional[kornia.core.Dtype] = None
) -> kornia.core.Tensor:
    kernel_x = get_sobel_kernel_3x3(device=device, dtype=dtype)
    x = kernel_x
    perm_73 = list(range(x.ndim))
    perm_73[0] = 1
    perm_73[1] = 0
    kernel_y = x.transpose(perm=perm_73)
    return kornia.core.stack([kernel_x, kernel_y])


def get_diff_kernel2d(
    *, device: Optional[kornia.core.Device] = None, dtype: Optional[kornia.core.Dtype] = None
) -> kornia.core.Tensor:
    kernel_x = get_diff_kernel_3x3(device=device, dtype=dtype)
    x = kernel_x
    perm_74 = list(range(x.ndim))
    perm_74[0] = 1
    perm_74[1] = 0
    kernel_y = x.transpose(perm=perm_74)
    return kornia.core.stack([kernel_x, kernel_y])


def get_sobel_kernel2d_2nd_order(
    *, device: Optional[kornia.core.Device] = None, dtype: Optional[kornia.core.Dtype] = None
) -> kornia.core.Tensor:
    gxx = get_sobel_kernel_5x5_2nd_order(device=device, dtype=dtype)
    x = gxx
    perm_75 = list(range(x.ndim))
    perm_75[0] = 1
    perm_75[1] = 0
    gyy = x.transpose(perm=perm_75)
    gxy = _get_sobel_kernel_5x5_2nd_order_xy(device=device, dtype=dtype)
    return kornia.core.stack([gxx, gxy, gyy])


def get_diff_kernel2d_2nd_order(
    *, device: Optional[kornia.core.Device] = None, dtype: Optional[kornia.core.Dtype] = None
) -> kornia.core.Tensor:
    gxx = kornia.core.tensor([[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]], device=device, dtype=dtype)
    x = gxx
    perm_76 = list(range(x.ndim))
    perm_76[0] = 1
    perm_76[1] = 0
    gyy = x.transpose(perm=perm_76)
    gxy = kornia.core.tensor([[-1.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, -1.0]], device=device, dtype=dtype)
    return kornia.core.stack([gxx, gxy, gyy])


def get_spatial_gradient_kernel2d(
    mode: str, order: int, *, device: Optional[kornia.core.Device] = None, dtype: Optional[kornia.core.Dtype] = None
) -> kornia.core.Tensor:
    """Function that returns kernel for 1st or 2nd order image gradients, using one of the following operators:

    sobel, diff.
    """
    kornia.core.check.KORNIA_CHECK(mode.lower() in {"sobel", "diff"}, f"Mode should be `sobel` or `diff`. Got {mode}")
    kornia.core.check.KORNIA_CHECK(order in {1, 2}, f"Order should be 1 or 2. Got {order}")
    if mode == "sobel" and order == 1:
        kernel: Tensor = get_sobel_kernel2d(device=device, dtype=dtype)
    elif mode == "sobel" and order == 2:
        kernel = get_sobel_kernel2d_2nd_order(device=device, dtype=dtype)
    elif mode == "diff" and order == 1:
        kernel = get_diff_kernel2d(device=device, dtype=dtype)
    elif mode == "diff" and order == 2:
        kernel = get_diff_kernel2d_2nd_order(device=device, dtype=dtype)
    else:
        raise NotImplementedError(f"Not implemented for order {order} on mode {mode}")
    return kernel


def get_spatial_gradient_kernel3d(
    mode: str, order: int, device: Optional[kornia.core.Device] = None, dtype: Optional[kornia.core.Dtype] = None
) -> kornia.core.Tensor:
    """Function that returns kernel for 1st or 2nd order scale pyramid gradients, using one of the following
    operators: sobel, diff."""
    kornia.core.check.KORNIA_CHECK(mode.lower() in {"sobel", "diff"}, f"Mode should be `sobel` or `diff`. Got {mode}")
    kornia.core.check.KORNIA_CHECK(order in {1, 2}, f"Order should be 1 or 2. Got {order}")
    if mode == "diff" and order == 1:
        kernel = get_diff_kernel3d(device=device, dtype=dtype)
    elif mode == "diff" and order == 2:
        kernel = get_diff_kernel3d_2nd_order(device=device, dtype=dtype)
    else:
        raise NotImplementedError(f"Not implemented 3d gradient kernel for order {order} on mode {mode}")
    return kernel


def get_gaussian_kernel1d(
    kernel_size: int,
    sigma: (float | Tensor),
    force_even: bool = False,
    *,
    device: Optional[kornia.core.Device] = None,
    dtype: Optional[kornia.core.Dtype] = None
) -> kornia.core.Tensor:
    """Function that returns Gaussian filter coefficients.

    Args:
        kernel_size: filter size. It should be odd and positive.
        sigma: gaussian standard deviation.
        force_even: overrides requirement for odd kernel size.
        device: This value will be used if sigma is a float. Device desired to compute.
        dtype: This value will be used if sigma is a float. Dtype desired for compute.

    Returns:
        gaussian filter coefficients with shape :math:`(B, \\text{kernel_size})`.

    Examples:
        >>> get_gaussian_kernel1d(3, 2.5)
        tensor([[0.3243, 0.3513, 0.3243]])
        >>> get_gaussian_kernel1d(5, 1.5)
        tensor([[0.1201, 0.2339, 0.2921, 0.2339, 0.1201]])
        >>> get_gaussian_kernel1d(5, torch.tensor([[1.5], [0.7]]))
        tensor([[0.1201, 0.2339, 0.2921, 0.2339, 0.1201],
                [0.0096, 0.2054, 0.5699, 0.2054, 0.0096]])
    """
    _check_kernel_size(kernel_size, allow_even=force_even)
    return gaussian(kernel_size, sigma, device=device, dtype=dtype)


def get_gaussian_discrete_kernel1d(
    kernel_size: int,
    sigma: (float | Tensor),
    force_even: bool = False,
    *,
    device: Optional[kornia.core.Device] = None,
    dtype: Optional[kornia.core.Dtype] = None
) -> kornia.core.Tensor:
    """Function that returns Gaussian filter coefficients based on the modified Bessel functions.

    Adapted from: https://github.com/Project-MONAI/MONAI/blob/master/monai/networks/layers/convutils.py.

    Args:
        kernel_size: filter size. It should be odd and positive.
        sigma: gaussian standard deviation. If a tensor, should be in a shape :math:`(B, 1)`
        force_even: overrides requirement for odd kernel size.
        device: This value will be used if sigma is a float. Device desired to compute.
        dtype: This value will be used if sigma is a float. Dtype desired for compute.

    Returns:
        1D tensor with gaussian filter coefficients. With shape :math:`(B, 	ext{kernel_size})`

    Examples:
        >>> get_gaussian_discrete_kernel1d(3, 2.5)
        tensor([[0.3235, 0.3531, 0.3235]])
        >>> get_gaussian_discrete_kernel1d(5, 1.5)
        tensor([[0.1096, 0.2323, 0.3161, 0.2323, 0.1096]])
        >>> get_gaussian_discrete_kernel1d(5, torch.tensor([[1.5],[2.4]]))
        tensor([[0.1096, 0.2323, 0.3161, 0.2323, 0.1096],
                [0.1635, 0.2170, 0.2389, 0.2170, 0.1635]])
    """
    _check_kernel_size(kernel_size, allow_even=force_even)
    return gaussian_discrete(kernel_size, sigma, device=device, dtype=dtype)


def get_gaussian_erf_kernel1d(
    kernel_size: int,
    sigma: (float | Tensor),
    force_even: bool = False,
    *,
    device: Optional[kornia.core.Device] = None,
    dtype: Optional[kornia.core.Dtype] = None
) -> kornia.core.Tensor:
    """Function that returns Gaussian filter coefficients by interpolating the error function.

    Adapted from: https://github.com/Project-MONAI/MONAI/blob/master/monai/networks/layers/convutils.py.

    Args:
        kernel_size: filter size. It should be odd and positive.
        sigma: gaussian standard deviation. If a tensor, should be in a shape :math:`(B, 1)`
        force_even: overrides requirement for odd kernel size.
        device: This value will be used if sigma is a float. Device desired to compute.
        dtype: This value will be used if sigma is a float. Dtype desired for compute.

    Returns:
        1D tensor with gaussian filter coefficients. Shape :math:`(B, 	ext{kernel_size})`

    Examples:
        >>> get_gaussian_erf_kernel1d(3, 2.5)
        tensor([[0.3245, 0.3511, 0.3245]])
        >>> get_gaussian_erf_kernel1d(5, 1.5)
        tensor([[0.1226, 0.2331, 0.2887, 0.2331, 0.1226]])
        >>> get_gaussian_erf_kernel1d(5, torch.tensor([[1.5], [2.1]]))
        tensor([[0.1226, 0.2331, 0.2887, 0.2331, 0.1226],
                [0.1574, 0.2198, 0.2456, 0.2198, 0.1574]])
    """
    _check_kernel_size(kernel_size, allow_even=force_even)
    return gaussian_discrete_erf(kernel_size, sigma, device=device, dtype=dtype)


def get_gaussian_kernel2d(
    kernel_size: (tuple[int, int] | int),
    sigma: (tuple[float, float] | Tensor),
    force_even: bool = False,
    *,
    device: Optional[kornia.core.Device] = None,
    dtype: Optional[kornia.core.Dtype] = None
) -> kornia.core.Tensor:
    """Function that returns Gaussian filter matrix coefficients.

    Args:
        kernel_size: filter sizes in the y and x direction. Sizes should be odd and positive.
        sigma: gaussian standard deviation in the y and x.
        force_even: overrides requirement for odd kernel size.
        device: This value will be used if sigma is a float. Device desired to compute.
        dtype: This value will be used if sigma is a float. Dtype desired for compute.

    Returns:
        2D tensor with gaussian filter matrix coefficients.

    Shape:
        - Output: :math:`(B, \\text{kernel_size}_x, \\text{kernel_size}_y)`

    Examples:
        >>> get_gaussian_kernel2d((5, 5), (1.5, 1.5))
        tensor([[[0.0144, 0.0281, 0.0351, 0.0281, 0.0144],
                 [0.0281, 0.0547, 0.0683, 0.0547, 0.0281],
                 [0.0351, 0.0683, 0.0853, 0.0683, 0.0351],
                 [0.0281, 0.0547, 0.0683, 0.0547, 0.0281],
                 [0.0144, 0.0281, 0.0351, 0.0281, 0.0144]]])
        >>> get_gaussian_kernel2d((3, 5), (1.5, 1.5))
        tensor([[[0.0370, 0.0720, 0.0899, 0.0720, 0.0370],
                 [0.0462, 0.0899, 0.1123, 0.0899, 0.0462],
                 [0.0370, 0.0720, 0.0899, 0.0720, 0.0370]]])
        >>> get_gaussian_kernel2d((5, 5), torch.tensor([[1.5, 1.5]]))
        tensor([[[0.0144, 0.0281, 0.0351, 0.0281, 0.0144],
                 [0.0281, 0.0547, 0.0683, 0.0547, 0.0281],
                 [0.0351, 0.0683, 0.0853, 0.0683, 0.0351],
                 [0.0281, 0.0547, 0.0683, 0.0547, 0.0281],
                 [0.0144, 0.0281, 0.0351, 0.0281, 0.0144]]])
    """
    if isinstance(sigma, tuple):
        sigma = kornia.core.tensor([sigma], device=device, dtype=dtype)
    kornia.core.check.KORNIA_CHECK_IS_TENSOR(sigma)
    kornia.core.check.KORNIA_CHECK_SHAPE(sigma, ["B", "2"])
    ksize_y, ksize_x = _unpack_2d_ks(kernel_size)
    sigma_y, sigma_x = sigma[:, 0, None], sigma[:, 1, None]
    kernel_y = get_gaussian_kernel1d(ksize_y, sigma_y, force_even, device=device, dtype=dtype)[..., None]
    kernel_x = get_gaussian_kernel1d(ksize_x, sigma_x, force_even, device=device, dtype=dtype)[..., None]
    return kernel_y * kernel_x.view(-1, 1, ksize_x)


def get_gaussian_kernel3d(
    kernel_size: (tuple[int, int, int] | int),
    sigma: (tuple[float, float, float] | Tensor),
    force_even: bool = False,
    *,
    device: Optional[kornia.core.Device] = None,
    dtype: Optional[kornia.core.Dtype] = None
) -> kornia.core.Tensor:
    """Function that returns Gaussian filter matrix coefficients.

    Args:
        kernel_size: filter sizes in the z, y and x direction. Sizes should be odd and positive.
        sigma: gaussian standard deviation in the z, y and x direction.
        force_even: overrides requirement for odd kernel size.
        device: This value will be used if sigma is a float. Device desired to compute.
        dtype: This value will be used if sigma is a float. Dtype desired for compute.

    Returns:
        3D tensor with gaussian filter matrix coefficients.

    Shape:
        - Output: :math:`(B, \\text{kernel_size}_x, \\text{kernel_size}_y,  \\text{kernel_size}_z)`

    Examples:
        >>> get_gaussian_kernel3d((3, 3, 3), (1.5, 1.5, 1.5))
        tensor([[[[0.0292, 0.0364, 0.0292],
                  [0.0364, 0.0455, 0.0364],
                  [0.0292, 0.0364, 0.0292]],
        <BLANKLINE>
                 [[0.0364, 0.0455, 0.0364],
                  [0.0455, 0.0568, 0.0455],
                  [0.0364, 0.0455, 0.0364]],
        <BLANKLINE>
                 [[0.0292, 0.0364, 0.0292],
                  [0.0364, 0.0455, 0.0364],
                  [0.0292, 0.0364, 0.0292]]]])
        >>> get_gaussian_kernel3d((3, 3, 3), (1.5, 1.5, 1.5)).sum()
        tensor(1.)
        >>> get_gaussian_kernel3d((3, 3, 3), (1.5, 1.5, 1.5)).shape
        torch.Size([1, 3, 3, 3])
        >>> get_gaussian_kernel3d((3, 7, 5), torch.tensor([[1.5, 1.5, 1.5]])).shape
        torch.Size([1, 3, 7, 5])
    """
    if isinstance(sigma, tuple):
        sigma = kornia.core.tensor([sigma], device=device, dtype=dtype)
    kornia.core.check.KORNIA_CHECK_IS_TENSOR(sigma)
    kornia.core.check.KORNIA_CHECK_SHAPE(sigma, ["B", "3"])
    ksize_z, ksize_y, ksize_x = _unpack_3d_ks(kernel_size)
    sigma_z, sigma_y, sigma_x = sigma[:, 0, None], sigma[:, 1, None], sigma[:, 2, None]
    kernel_z = get_gaussian_kernel1d(ksize_z, sigma_z, force_even, device=device, dtype=dtype)
    kernel_y = get_gaussian_kernel1d(ksize_y, sigma_y, force_even, device=device, dtype=dtype)
    kernel_x = get_gaussian_kernel1d(ksize_x, sigma_x, force_even, device=device, dtype=dtype)
    return kernel_z.view(-1, ksize_z, 1, 1) * kernel_y.view(-1, 1, ksize_y, 1) * kernel_x.view(-1, 1, 1, ksize_x)
