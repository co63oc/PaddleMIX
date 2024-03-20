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

import math
from inspect import isfunction

import paddle
from utils.utils import randn_tensor_mod


def gather_data(data, return_np=True):
    """gather data from multiple processes to one list"""
    data_list = [paddle.zeros_like(x=data) for _ in range(paddle.distributed.get_world_size())]
    paddle.distributed.all_gather(tensor_list=data_list, tensor=data)
    if return_np:
        data_list = [data.cpu().numpy() for data in data_list]
    return data_list


def autocast(f):
    def do_autocast(*args, **kwargs):
        # with paddle.amp.auto_cast(enable=True, dtype=paddle.
        #     get_autocast_gpu_dtype()):
        #     return f(*args, **kwargs)
        with paddle.amp.auto_cast(enable=True):
            return f(*args, **kwargs)

    return do_autocast


def extract_into_tensor(a, t, x_shape):
    b, *_ = tuple(t.shape)
    out = a.take_along_axis(axis=-1, indices=t)
    return out.reshape([b, *((1,) * (len(x_shape) - 1))])


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: randn_tensor_mod(shape=(1, *shape[1:])).tile([shape[0], *((1,) * (len(shape) - 1))])
    noise = lambda: randn_tensor_mod(shape=shape)
    return repeat_noise() if repeat else noise()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def exists(val):
    return val is not None


def identity(*args, **kwargs):
    return paddle.nn.Identity()


def uniq(arr):
    return {el: (True) for el in arr}.keys()


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(axis=list(range(1, len(tuple(tensor.shape)))))


def ismap(x):
    if not isinstance(x, paddle.Tensor):
        return False
    return len(tuple(x.shape)) == 4 and tuple(x.shape)[1] > 3


def isimage(x):
    if not isinstance(x, paddle.Tensor):
        return False
    return len(tuple(x.shape)) == 4 and (tuple(x.shape)[1] == 3 or tuple(x.shape)[1] == 1)


def max_neg_value(t):
    return -paddle.finfo(paddle_aux._STR_2_PADDLE_DTYPE(t.dtype)).max  # noqa


def shape_to_str(x):
    shape_str = "x".join([str(x) for x in tuple(x.shape)])
    return shape_str


def init_(tensor):
    dim = tuple(tensor.shape)[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(min=-std, max=std)
    return tensor


# ckpt = torch.utils.checkpoint.checkpoint


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    # if flag:
    #     return ckpt(func, *inputs)
    # else:
    #     return func(*inputs)
    return func(*inputs)
