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
import pp_kornia
import pp_open_clip
import utils.paddle_aux  # noqa:*
from lvdm.common import autocast
from utils.utils import count_params

from ppdiffusers import transformers


class AbstractEncoder(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class IdentityEncoder(AbstractEncoder):
    def encode(self, x):
        return x


class ClassEmbedder(paddle.nn.Layer):
    def __init__(self, embed_dim, n_classes=1000, key="class", ucg_rate=0.1):
        super().__init__()
        self.key = key
        self.embedding = paddle.nn.Embedding(num_embeddings=n_classes, embedding_dim=embed_dim)
        self.n_classes = n_classes
        self.ucg_rate = ucg_rate

    def forward(self, batch, key=None, disable_dropout=False):
        if key is None:
            key = self.key
        c = batch[key][:, None]
        if self.ucg_rate > 0.0 and not disable_dropout:
            mask = 1.0 - paddle.bernoulli(x=paddle.ones_like(x=c) * self.ucg_rate)
            c = mask * c + (1 - mask) * paddle.ones_like(x=c) * (self.n_classes - 1)
            c = c.astype(dtype="int64")
        c = self.embedding(c)
        return c

    def get_unconditional_conditioning(self, bs, device="gpu"):
        uc_class = self.n_classes - 1
        uc = paddle.ones(shape=(bs,)) * uc_class
        uc = {self.key: uc}
        return uc


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class FrozenT5Embedder(AbstractEncoder):
    """Uses the T5 transformer encoder for text"""

    def __init__(self, version="google/t5-v1_1-large", device="gpu", max_length=77, freeze=True):
        super().__init__()
        self.tokenizer = transformers.T5Tokenizer.from_pretrained(version)
        self.transformer = transformers.T5EncoderModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.stop_gradient = not False

    def forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)
        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)


class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from huggingface)"""

    LAYERS = ["last", "pooled", "hidden"]

    def __init__(
        self,
        version="openai/clip-vit-large-patch14",
        device="gpu",
        max_length=77,
        freeze=True,
        layer="last",
        layer_idx=None,
    ):
        super().__init__()
        assert layer in self.LAYERS
        self.tokenizer = transformers.CLIPTokenizer.from_pretrained(version)
        self.transformer = transformers.CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = layer_idx
        if layer == "hidden":
            assert layer_idx is not None
            assert 0 <= abs(layer_idx) <= 12

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.stop_gradient = not False

    def forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens, output_hidden_states=self.layer == "hidden")
        if self.layer == "last":
            z = outputs.last_hidden_state
        elif self.layer == "pooled":
            z = outputs.pooler_output[:, None, :]
        else:
            z = outputs.hidden_states[self.layer_idx]
        return z

    def encode(self, text):
        return self(text)


class ClipImageEmbedder(paddle.nn.Layer):
    def __init__(
        self,
        model,
        jit=False,
        device="gpu" if paddle.device.cuda.device_count() >= 1 else "cpu",
        antialias=True,
        ucg_rate=0.0,
    ):
        super().__init__()
        from clip import load as load_clip

        self.model, _ = load_clip(name=model, device=device, jit=jit)
        self.antialias = antialias
        self.register_buffer(
            name="mean",
            tensor=paddle.to_tensor(data=[0.48145466, 0.4578275, 0.40821073], dtype="float32"),
            persistable=False,
        )
        self.register_buffer(
            name="std",
            tensor=paddle.to_tensor(data=[0.26862954, 0.26130258, 0.27577711], dtype="float32"),
            persistable=False,
        )
        self.ucg_rate = ucg_rate

    def preprocess(self, x):
        x = pp_kornia.geometry.resize(
            x, (224, 224), interpolation="bicubic", align_corners=True, antialias=self.antialias
        )
        x = (x + 1.0) / 2.0
        x = pp_kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def forward(self, x, no_dropout=False):
        out = self.model.encode_image(self.preprocess(x))
        out = out.to(dtype=x.dtype)
        if self.ucg_rate > 0.0 and not no_dropout:
            out = paddle.bernoulli(x=(1.0 - self.ucg_rate) * paddle.ones(shape=tuple(out.shape)[0]))[:, None] * out
        return out


class FrozenOpenCLIPEmbedder(AbstractEncoder):
    """
    Uses the OpenCLIP transformer encoder for text
    """

    LAYERS = ["last", "penultimate"]

    def __init__(
        self, arch="ViT-H-14", version="laion2b_s32b_b79k", device="gpu", max_length=77, freeze=True, layer="last"
    ):
        super().__init__()
        assert layer in self.LAYERS
        model, _, _ = pp_open_clip.create_model_and_transforms(arch, device=str("cpu").replace("cuda", "gpu"))

        del model.visual
        self.model = model
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.stop_gradient = not False

    def forward(self, text):
        # self.device = self.model.positional_embedding.place
        tokens = pp_open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens)
        return z

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)
        x = x + self.model.positional_embedding
        x = x.transpose(perm=[1, 0, 2])
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.transpose(perm=[1, 0, 2])
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: paddle.Tensor, attn_mask=None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            # if (self.model.transformer.grad_checkpointing and not torch.jit
            #     .is_scripting()):
            #     x = torch.utils.checkpoint.checkpoint(r, x, attn_mask)
            # else:
            #     x = r(x, attn_mask=attn_mask)
            x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)


class FrozenOpenCLIPImageEmbedder(AbstractEncoder):
    """
    Uses the OpenCLIP vision transformer encoder for images
    """

    def __init__(
        self,
        arch="ViT-H-14",
        version="laion2b_s32b_b79k",
        device="gpu",
        max_length=77,
        freeze=True,
        layer="pooled",
        antialias=True,
        ucg_rate=0.0,
    ):
        super().__init__()
        model, _, _ = pp_open_clip.create_model_and_transforms(
            arch, device=str("cpu").replace("cuda", "gpu"), pretrained=version
        )
        del model.transformer
        self.model = model
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "penultimate":
            raise NotImplementedError()
            self.layer_idx = 1
        self.antialias = antialias
        self.register_buffer(
            name="mean",
            tensor=paddle.to_tensor(data=[0.48145466, 0.4578275, 0.40821073], dtype="float32"),
            persistable=False,
        )
        self.register_buffer(
            name="std",
            tensor=paddle.to_tensor(data=[0.26862954, 0.26130258, 0.27577711], dtype="float32"),
            persistable=False,
        )
        self.ucg_rate = ucg_rate

    def preprocess(self, x):
        x = pp_kornia.geometry.resize(
            x, (224, 224), interpolation="bicubic", align_corners=True, antialias=self.antialias
        )
        x = (x + 1.0) / 2.0
        x = pp_kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.stop_gradient = not False

    @autocast
    def forward(self, image, no_dropout=False):
        z = self.encode_with_vision_transformer(image)
        if self.ucg_rate > 0.0 and not no_dropout:
            z = paddle.bernoulli(x=(1.0 - self.ucg_rate) * paddle.ones(shape=tuple(z.shape)[0]))[:, None] * z
        return z

    def encode_with_vision_transformer(self, img):
        img = self.preprocess(img)
        x = self.model.visual(img)
        return x

    def encode(self, text):
        return self(text)


class FrozenOpenCLIPImageEmbedderV2(AbstractEncoder):
    """
    Uses the OpenCLIP vision transformer encoder for images
    """

    def __init__(
        self, arch="ViT-H-14", version="laion2b_s32b_b79k", device="gpu", freeze=True, layer="pooled", antialias=True
    ):
        super().__init__()
        # has converted to paddle model
        # version = "checkpoints/open_clip_paddle_model.bin"
        model, _, _ = pp_open_clip.create_model_and_transforms(
            arch, device=str("cpu").replace("cuda", "gpu"), pretrained=version
        )
        del model.transformer
        self.model = model
        self.device = device
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "penultimate":
            raise NotImplementedError()
            self.layer_idx = 1
        self.antialias = antialias
        self.register_buffer(
            name="mean",
            tensor=paddle.to_tensor(data=[0.48145466, 0.4578275, 0.40821073], dtype="float32"),
            persistable=False,
        )
        self.register_buffer(
            name="std",
            tensor=paddle.to_tensor(data=[0.26862954, 0.26130258, 0.27577711], dtype="float32"),
            persistable=False,
        )

    def preprocess(self, x):
        x = pp_kornia.geometry.resize(
            x, (224, 224), interpolation="bicubic", align_corners=True, antialias=self.antialias
        )
        x = (x + 1.0) / 2.0
        x = pp_kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def freeze(self):
        self.model = self.model.eval()
        for param in self.model.parameters():
            param.stop_gradient = not False

    def forward(self, image, no_dropout=False):
        z = self.encode_with_vision_transformer(image)
        return z

    def encode_with_vision_transformer(self, x):
        if x.dtype != self.model.visual._dtype:
            x = x.astype(self.model.visual._dtype)
        x = self.preprocess(x)
        if hasattr(self.model.visual, "input_patchnorm") and self.model.visual.input_patchnorm:
            x = x.reshape(
                tuple(x.shape)[0],
                tuple(x.shape)[1],
                self.model.visual.grid_size[0],
                self.model.visual.patch_size[0],
                self.model.visual.grid_size[1],
                self.model.visual.patch_size[1],
            )
            x = x.transpose(perm=[0, 2, 4, 1, 3, 5])
            x = x.reshape(tuple(x.shape)[0], self.model.visual.grid_size[0] * self.model.visual.grid_size[1], -1)
            x = self.model.visual.patchnorm_pre_ln(x)
            x = self.model.visual.conv1(x)
        else:
            x = self.model.visual.conv1(x)
            x = x.reshape(tuple(x.shape)[0], tuple(x.shape)[1], -1)
            x = x.transpose(perm=[0, 2, 1])
        x = paddle.concat(
            x=[
                self.model.visual.class_embedding.to(dtype=x.dtype)
                + paddle.zeros(shape=[tuple(x.shape)[0], 1, tuple(x.shape)[-1]], dtype=x.dtype),
                x,
            ],
            axis=1,
        )
        x = x + self.model.visual.positional_embedding.to(dtype=x.dtype)
        x = self.model.visual.patch_dropout(x)
        x = self.model.visual.ln_pre(x)
        x = x.transpose(perm=[1, 0, 2])
        x = self.model.visual.transformer(x)
        x = x.transpose(perm=[1, 0, 2])
        return x


class FrozenCLIPT5Encoder(AbstractEncoder):
    def __init__(
        self,
        clip_version="openai/clip-vit-large-patch14",
        t5_version="google/t5-v1_1-xl",
        device="gpu",
        clip_max_length=77,
        t5_max_length=77,
    ):
        super().__init__()
        self.clip_encoder = FrozenCLIPEmbedder(clip_version, device, max_length=clip_max_length)
        self.t5_encoder = FrozenT5Embedder(t5_version, device, max_length=t5_max_length)
        print(
            f"{self.clip_encoder.__class__.__name__} has {count_params(self.clip_encoder) * 1e-06:.2f} M parameters, {self.t5_encoder.__class__.__name__} comes with {count_params(self.t5_encoder) * 1e-06:.2f} M params."
        )

    def encode(self, text):
        return self(text)

    def forward(self, text):
        clip_z = self.clip_encoder.encode(text)
        t5_z = self.t5_encoder.encode(text)
        return [clip_z, t5_z]
