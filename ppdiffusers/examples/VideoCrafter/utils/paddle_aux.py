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


def reshape(self, *args, **kwargs):
    if args:
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            return paddle.reshape(self, args[0])
        else:
            return paddle.reshape(self, list(args))
    elif kwargs:
        assert "shape" in kwargs
        return paddle.reshape(self, shape=kwargs["shape"])


setattr(paddle.Tensor, "reshape", reshape)


def min_class_func(self, *args, **kwargs):
    if "other" in kwargs:
        kwargs["y"] = kwargs.pop("other")
        ret = paddle.minimum(self, *args, **kwargs)
    elif len(args) == 1 and isinstance(args[0], paddle.Tensor):
        ret = paddle.minimum(self, *args, **kwargs)
    else:
        if "dim" in kwargs:
            kwargs["axis"] = kwargs.pop("dim")

        if "axis" in kwargs or len(args) >= 1:
            ret = paddle.min(self, *args, **kwargs), paddle.argmin(self, *args, **kwargs)
        else:
            ret = paddle.min(self, *args, **kwargs)

    return ret


def max_class_func(self, *args, **kwargs):
    if "other" in kwargs:
        kwargs["y"] = kwargs.pop("other")
        ret = paddle.maximum(self, *args, **kwargs)
    elif len(args) == 1 and isinstance(args[0], paddle.Tensor):
        ret = paddle.maximum(self, *args, **kwargs)
    else:
        if "dim" in kwargs:
            kwargs["axis"] = kwargs.pop("dim")

        if "axis" in kwargs or len(args) >= 1:
            ret = paddle.max(self, *args, **kwargs), paddle.argmax(self, *args, **kwargs)
        else:
            ret = paddle.max(self, *args, **kwargs)

    return ret


setattr(paddle.Tensor, "min", min_class_func)
setattr(paddle.Tensor, "max", max_class_func)


def repeat(self, *args, **kwargs):
    if args:
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            return paddle.tile(self, args[0])
        else:
            return paddle.tile(self, list(args))
    elif kwargs:
        assert "repeats" in kwargs
        return paddle.tile(self, repeat_times=kwargs["repeats"])


setattr(paddle.Tensor, "repeat", repeat)


def _STR_2_PADDLE_DTYPE(type):
    type_map = {
        "uint8": paddle.uint8,
        "int8": paddle.int8,
        "int16": paddle.int16,
        "int32": paddle.int32,
        "int64": paddle.int64,
        "float16": paddle.float16,
        "float32": paddle.float32,
        "float64": paddle.float64,
        "bfloat16": paddle.bfloat16,
    }
    return type_map.get(type)


def view(self, *args, **kwargs):
    if args:
        if len(args) == 1:
            if isinstance(args[0], (tuple, list)):
                return paddle.reshape(self, args[0])  # To change reshape => view
            elif isinstance(args[0], str):
                return paddle.view(self, args[0])
            else:
                return paddle.reshape(self, list(args))  # To change reshape => view
        else:
            return paddle.reshape(self, list(args))  # To change reshape => view
    elif kwargs:
        key = [k for k in kwargs.keys()]
        if "dtype" in kwargs:
            return paddle.view(self, shape_or_dtype=kwargs[key[0]])
        else:
            return paddle.reshape(self, shape=kwargs[key[0]])  # To change reshape => view


setattr(paddle.Tensor, "view", view)


def _FUNCTIONAL_PAD(x, pad, mode="constant", value=0.0, data_format="NCHW"):
    if len(x.shape) * 2 == len(pad) and mode == "constant":
        pad = paddle.to_tensor(pad, dtype="int32").reshape((-1, 2)).flip([0]).flatten().tolist()
    return paddle.nn.functional.pad(x, pad, mode, value, data_format)
