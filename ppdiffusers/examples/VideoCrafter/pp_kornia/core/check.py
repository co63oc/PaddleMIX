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

import pp_kornia as kornia
import utils.paddle_aux  # noqa

"""The testing package contains testing-specific utilities."""
from typing import Any, Optional, TypeVar, cast

from typing_extensions import TypeGuard

__all__ = [
    "KORNIA_CHECK_SHAPE",
    "KORNIA_CHECK",
    "KORNIA_UNWRAP",
    "KORNIA_CHECK_TYPE",
    "KORNIA_CHECK_IS_TENSOR",
]


def KORNIA_CHECK_SHAPE(x: kornia.core.Tensor, shape: list[str], raises: bool = True) -> bool:
    """Check whether a tensor has a specified shape.

    The shape can be specified with a implicit or explicit list of strings.
    The guard also check whether the variable is a type `Tensor`.

    Args:
        x: the tensor to evaluate.
        shape: a list with strings with the expected shape.
        raises: bool indicating whether an exception should be raised upon failure.

    Raises:
        Exception: if the input tensor is has not the expected shape and raises is True.

    Example:
        >>> x = torch.rand(2, 3, 4, 4)
        >>> KORNIA_CHECK_SHAPE(x, ["B", "C", "H", "W"])  # implicit
        True

        >>> x = torch.rand(2, 3, 4, 4)
        >>> KORNIA_CHECK_SHAPE(x, ["2", "3", "H", "W"])  # explicit
        True
    """
    if "*" == shape[0]:
        shape_to_check = shape[1:]
        x_shape_to_check = tuple(x.shape)[-len(shape) + 1 :]
    elif "*" == shape[-1]:
        shape_to_check = shape[:-1]
        x_shape_to_check = tuple(x.shape)[: len(shape) - 1]
    else:
        shape_to_check = shape
        x_shape_to_check = tuple(x.shape)
    if len(x_shape_to_check) != len(shape_to_check):
        if raises:
            raise TypeError(f"{x} shape must be [{shape}]. Got {tuple(x.shape)}")
        else:
            return False
    for i in range(len(x_shape_to_check)):
        dim_: str = shape_to_check[i]
        if not dim_.isnumeric():
            continue
        dim = int(dim_)
        if x_shape_to_check[i] != dim:
            if raises:
                raise TypeError(f"{x} shape must be [{shape}]. Got {tuple(x.shape)}")
            else:
                return False
    return True


def KORNIA_CHECK(condition: bool, msg: Optional[str] = None, raises: bool = True) -> bool:
    """Check any arbitrary boolean condition.

    Args:
        condition: the condition to evaluate.
        msg: message to show in the exception.
        raises: bool indicating whether an exception should be raised upon failure.

    Raises:
        Exception: if the condition is met and raises is True.

    Example:
        >>> x = torch.rand(2, 3, 3)
        >>> KORNIA_CHECK(x.shape[-2:] == (3, 3), "Invalid homography")
        True
    """
    if not condition:
        if raises:
            raise Exception(f"{condition} not true.\n{msg}")
        return False
    return True


def KORNIA_UNWRAP(maybe_obj: object, typ: Any) -> Any:
    """Unwraps an optional contained value that may or not be present.

    Args:
        maybe_obj: the object to unwrap.
        typ: expected type after unwrap.
    """
    return cast(typ, maybe_obj)


T = TypeVar("T", bound=type)


def KORNIA_CHECK_TYPE(
    x: object, typ: (T | tuple[T, ...]), msg: Optional[str] = None, raises: bool = True
) -> TypeGuard[T]:
    """Check the type of an aribratry variable.

    Args:
        x: any input variable.
        typ: the expected type of the variable.
        msg: message to show in the exception.
        raises: bool indicating whether an exception should be raised upon failure.

    Raises:
        TypeException: if the input variable does not match with the expected and raises is True.

    Example:
        >>> KORNIA_CHECK_TYPE("foo", str, "Invalid string")
        True
    """
    if not isinstance(x, typ):
        if raises:
            raise TypeError(f"Invalid type: {type(x)}.\n{msg}")
        return False
    return True


def KORNIA_CHECK_IS_TENSOR(x: object, msg: Optional[str] = None, raises: bool = True) -> TypeGuard[kornia.core.Tensor]:
    """Check the input variable is a Tensor.

    Args:
        x: any input variable.
        msg: message to show in the exception.
        raises: bool indicating whether an exception should be raised upon failure.

    Raises:
        TypeException: if the input variable does not match with the expected and raises is True.

    Example:
        >>> x = torch.rand(2, 3, 3)
        >>> KORNIA_CHECK_IS_TENSOR(x, "Invalid tensor")
        True
    """
    if not isinstance(x, kornia.core.Tensor):
        if raises:
            raise TypeError(f"Not a Tensor type. Got: {type(x)}.\n{msg}")
        return False
    return True
