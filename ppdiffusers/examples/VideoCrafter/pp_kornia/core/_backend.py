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

from typing import Union

import paddle

Tensor = paddle.Tensor
tensor = paddle.to_tensor
Module = paddle.nn.Layer
ModuleList = paddle.nn.LayerList
# Parameter = paddle.nn.Parameter
arange = paddle.arange
concatenate = paddle.concat
stack = paddle.stack
linspace = paddle.linspace
normalize = paddle.nn.functional.normalize
pad = paddle.nn.functional.pad
eye = paddle.eye
einsum = paddle.einsum
zeros = paddle.zeros
zeros_like = paddle.zeros_like
ones = paddle.ones
ones_like = paddle.ones_like
where = paddle.where
complex = paddle.complex
diag = paddle.diag
softmax = paddle.nn.functional.softmax
map_coordinates = paddle.nn.functional.grid_sample
cos = paddle.cos
sin = paddle.sin
tan = paddle.tan
as_tensor = paddle.to_tensor
rand = paddle.rand
Device = Union[str, str, None]
Dtype = Union[paddle.dtype, None]
