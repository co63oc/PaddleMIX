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

from ._backend import (
    Device,
    Dtype,
    Module,
    ModuleList,
    Tensor,
    arange,
    as_tensor,
    complex,
    concatenate,
    cos,
    diag,
    einsum,
    eye,
    linspace,
    map_coordinates,
    normalize,
    ones,
    ones_like,
    pad,
    rand,
    sin,
    softmax,
    stack,
    tan,
    tensor,
    where,
    zeros,
    zeros_like,
)
from .tensor_wrapper import TensorWrapper

__all__ = [
    "arange",
    "concatenate",
    "Device",
    "Dtype",
    "Module",
    "ModuleList",
    "Tensor",
    "tensor",
    "normalize",
    "pad",
    "stack",
    "softmax",
    "as_tensor",
    "rand",
    "cos",
    "sin",
    "tan",
    "where",
    "eye",
    "ones",
    "ones_like",
    "einsum",
    "zeros",
    "complex",
    "zeros_like",
    "linspace",
    "diag",
    "TensorWrapper",
    "map_coordinates",
]
