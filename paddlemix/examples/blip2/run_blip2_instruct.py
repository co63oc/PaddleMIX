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

# import paddle  # noqa: F401
from PIL import Image

from paddlemix.models.blip2.blip2_vicuna_instruct import Blip2VicunaInstruct

raw_image = Image.open("Confusing-Pictures.jpg").convert("RGB")
model = Blip2VicunaInstruct(vit_precision="fp32")
print("start paddle-----")
for name, value in model.named_parameters():
    print(name, value.shape)
print("start lavis-----")

from lavis.models import load_model_and_preprocess

# loads InstructBLIP model
# model, vis_processors, _ = load_model_and_preprocess(name="blip2_opt", model_type="pretrain_opt2.7b", is_eval=True, device=device)
model, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True)
# prepare the image
image = vis_processors["eval"](raw_image).unsqueeze(0)
for name, value in model.named_parameters():
    print(name, value.shape)

# import numpy as np
# t1 = np.load("image.np.npy")
# raw_image = paddle.to_tensor(t1)

ret = model.generate({"image": image, "prompt": "What is unusual about this image?"})
print(ret)
