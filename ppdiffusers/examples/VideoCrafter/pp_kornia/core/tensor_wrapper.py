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

import collections

import paddle
import utils.paddle_aux  # noqa


def wrap(v, cls):
    if type(v) in {tuple, list}:
        return type(v)(wrap(vi, cls) for vi in v)
    return cls(v) if isinstance(v, paddle.Tensor) else v


def unwrap(v):
    if type(v) in {tuple, list}:
        return type(v)(unwrap(vi) for vi in v)
    return v._data if isinstance(v, TensorWrapper) else v


class TensorWrapper:
    def __init__(self, data: paddle.Tensor) -> None:
        self.__dict__["_data"] = data
        self.__dict__["used_attrs"] = set()
        self.__dict__["used_calls"] = set()

    def unwrap(self):
        return unwrap(self)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __repr__(self) -> str:
        return f"{self._data}"

    def __getattr__(self, name):
        if name == "data":
            return self._data
        elif name in self.__dict__:
            return self.__dict__[name]
        self.used_attrs.add(name)
        val = getattr(self._data, name)
        return wrap(val, type(self))

    def __setattr__(self, name, value) -> None:
        if name in self.__dict__:
            self.__dict__[name] = value
        self.used_attrs.add(name)
        setattr(self._data, name, value)

    def __setitem__(self, key, value) -> None:
        self._data[key] = value

    def __getitem__(self, key):
        return wrap(self._data[key], type(self))

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        args_of_this_cls = []
        for a in args:
            if isinstance(a, cls):
                args_of_this_cls.append(a)
            elif isinstance(a, collections.abc.Sequence):
                args_of_this_cls.extend(el for el in a if isinstance(el, cls))
        for a in args_of_this_cls:
            a.used_calls.add(func)
        args = unwrap(tuple(args))
        kwargs = {k: unwrap(v) for k, v in kwargs.items()}
        return wrap(func(*args, **kwargs), cls)

    def __add__(self, other):
        return self.__unary_op__(paddle.add, other)

    def __radd__(self, other):
        return self.__unary_op__(paddle.add, other)

    def __mul__(self, other):
        return self.__unary_op__(paddle.multiply, other)

    def __rmul__(self, other):
        return self.__unary_op__(paddle.multiply, other)

    def __sub__(self, other):
        return self.__unary_op__(paddle.subtract, other)

    def __rsub__(self, other):
        return self.__unary_op__(paddle.subtract, other)

    def __truediv__(self, other):
        return self.__unary_op__(paddle.divide, other)

    def __floordiv__(self, other):
        return self.__unary_op__(paddle.floor_divide, other)

    def __ge__(self, other):
        return self.__unary_op__(paddle.greater_equal, other)

    def __gt__(self, other):
        return self.__unary_op__(paddle.greater_than, other)

    def __lt__(self, other):
        return self.__unary_op__(paddle.less_than, other)

    def __le__(self, other):
        return self.__unary_op__(paddle.less_equal, other)

    def __eq__(self, other):
        return self.__unary_op__(paddle.equal, other)

    def __ne__(self, other):
        return self.__unary_op__(paddle.not_equal, other)

    def __bool__(self) -> bool:
        return self.__unary_op__(paddle.Tensor.__bool__)

    def __int__(self) -> int:
        return self.__unary_op__(paddle.Tensor.__int__)

    def __neg__(self):
        return self.__unary_op__(paddle.Tensor.neg)

    def __unary_op__(self, func, other=None):
        args = (self, other) if other is not None else (self,)
        return self.__torch_function__(func, (type(self),), args)

    def __len__(self) -> int:
        return len(self._data)
