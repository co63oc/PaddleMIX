# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import inspect
import io
import json
import os
from collections import defaultdict
from collections import OrderedDict
from paddlenlp.transformers.auto.configuration import AutoConfig as PPNLPAutoConfig
from paddlenlp.utils.import_utils import import_module
import importlib

from ..model_utils import PretrainedConfig, PretrainedModel
CONFIG_MAPPING_NAMES = OrderedDict(
    [
                ("deit", "DeiTConfig"),
                        ("trocr", "TrOCRConfig"),
                                ("vision-encoder-decoder", "VisionEncoderDecoderConfig"),
    ]
)
MODEL_NAMES_MAPPING = OrderedDict(
    [
                ("deit", "DeiT"),
                        ("trocr", "TrOCR"),
                                ("vision-encoder-decoder", "Vision Encoder decoder"),
    ]
)
DEPRECATED_MODELS = [
]
SPECIAL_MODEL_TYPE_TO_MODULE_NAME = OrderedDict([])

def model_type_to_module_name(key):
    """Converts a config key to the corresponding module."""
    if key in SPECIAL_MODEL_TYPE_TO_MODULE_NAME:
        key = SPECIAL_MODEL_TYPE_TO_MODULE_NAME[key]
        if key in DEPRECATED_MODELS:
            key = f"deprecated.{key}"
        return key
    key = key.replace("-", "_")
    if key in DEPRECATED_MODELS:
        key = f"deprecated.{key}"
    return key


def config_class_to_model_type(config):
    """Converts a config class name to the corresponding model type"""
    for key, cls in CONFIG_MAPPING_NAMES.items():
        if cls == config:
            return key
    for key, cls in CONFIG_MAPPING._extra_content.items():
        if cls.__name__ == config:
            return key
    return None

class _LazyConfigMapping(OrderedDict):
    """
    A dictionary that lazily load its values when they are requested.
    """

    def __init__(self, mapping):
        self._mapping = mapping
        self._extra_content = {}
        self._modules = {}

    def __getitem__(self, key):
        if key in self._extra_content:
            return self._extra_content[key]
        if key not in self._mapping:
            raise KeyError(key)
        value = self._mapping[key]
        module_name = model_type_to_module_name(key)
        if module_name not in self._modules:
            self._modules[module_name] = importlib.import_module(
                f".{module_name}", "ppdiffusers.transformers"
            )
        if hasattr(self._modules[module_name], value):
            return getattr(self._modules[module_name], value)
        transformers_module = importlib.import_module("ppdiffusers.transformers")
        return getattr(transformers_module, value)

    def keys(self):
        return list(self._mapping.keys()) + list(self._extra_content.keys())

    def values(self):
        return [self[k] for k in self._mapping.keys()] + list(
            self._extra_content.values()
        )

    def items(self):
        return [(k, self[k]) for k in self._mapping.keys()] + list(
            self._extra_content.items()
        )

    def __iter__(self):
        return iter(list(self._mapping.keys()) + list(self._extra_content.keys()))

    def __contains__(self, item):
        return item in self._mapping or item in self._extra_content

    def register(self, key, value, exist_ok=False):
        """
        Register a new configuration in this mapping.
        """
        if key in self._mapping.keys() and not exist_ok:
            raise ValueError(
                f"'{key}' is already used by a Transformers config, pick another name."
            )
        self._extra_content[key] = value


CONFIG_MAPPING = _LazyConfigMapping(CONFIG_MAPPING_NAMES)

def get_configurations():
    """load the configurations of PretrainedConfig mapping: {<model-name>: [<class-name>, <class-name>, ...], }

    Returns:
        dict[str, str]: the mapping of model-name to model-classes
    """
    # 1. search the subdir<model-name> to find model-names
    ppdiffusers_transformers_dir = os.path.dirname(os.path.dirname(__file__))
    exclude_models = ["auto"]

    mappings = defaultdict(list)
    for model_name in os.listdir(ppdiffusers_transformers_dir):
        if model_name in exclude_models:
            continue

        model_dir = os.path.join(ppdiffusers_transformers_dir, model_name)
        if not os.path.isdir(model_dir):
            continue

        # 2. find the `configuration.py` file as the identifier of PretrainedConfig class
        configuration_path = os.path.join(model_dir, "configuration.py")
        if not os.path.exists(configuration_path):
            continue

        for package in ["paddlenlp", "ppdiffusers"]:
            configuration_module = import_module(f"{package}.transformers.{model_name}.configuration")
            for key in dir(configuration_module):
                value = getattr(configuration_module, key)
                if inspect.isclass(value) and issubclass(value, PretrainedConfig):
                    mappings[model_name].append(value)
    return mappings


class AutoConfig(PPNLPAutoConfig):
    MAPPING_NAMES = get_configurations()

    @classmethod
    def for_model(cls, model_type: str, *args, **kwargs):
        if model_type in CONFIG_MAPPING:
            config_class = CONFIG_MAPPING[model_type]
            return config_class(*args, **kwargs)
        raise ValueError(
            f"Unrecognized model identifier: {model_type}. Should contain one of {', '.join(CONFIG_MAPPING.keys())}"
        )
        
    @classmethod
    def _get_config_class_from_config(cls, pretrained_model_name_or_path: str, config_file_path: str):
        with io.open(config_file_path, encoding="utf-8") as f:
            config = json.load(f)

        # add support for legacy config
        if "init_class" in config:
            architectures = [config.pop("init_class")]
        else:
            architectures = config.pop("architectures", None)
            if architectures is None:
                return cls

        model_name = architectures[0]
        for package in ["ppdiffusers", "paddlenlp"]:
            model_class = import_module(f"{package}.transformers.{model_name}")
            if model_class is not None:
                break
        if model_class is None:
            raise ImportError(f"Cannot find the {model_class} from paddlenlp or ppdiffusers.")
        assert inspect.isclass(model_class) and issubclass(
            model_class, PretrainedModel
        ), f"<{model_class}> should be a PretarinedModel class, but <{type(model_class)}>"

        return cls if model_class.config_class is None else model_class.config_class
