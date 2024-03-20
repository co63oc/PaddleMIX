<https://github.com/AILab-CVC/VideoCrafter>

## 🔆 Introduction

🤗🤗🤗 VideoCrafter is an open-source video generation and editing toolbox for crafting video content.
It currently includes the Text2Video and Image2Video models:

## 💫 Inference

### 1. Text-to-Video

1) Download pretrained T2V models via [Hugging Face](https://huggingface.co/co63oc/VideoCrafter2/checkpoints/base_512_v2/model.ckpt), and put the `model.ckpt` in `checkpoints/base_512_v2/model.ckpt`.
2) Input the following commands in terminal.

```bash
  sh scripts/run_text2video.sh
```

### 2. Image-to-Video

1) Download pretrained I2V models via [Hugging Face](https://huggingface.co/co63oc/VideoCrafter2/checkpoints/i2v_512_v1/model.ckpt), and put the `model.ckpt` in `checkpoints/i2v_512_v1/model.ckpt`.
2) Input the following commands in terminal.

```bash
  sh scripts/run_image2video.sh
```

### 3. Local Gradio demo

1. Download the pretrained T2V and I2V models and put them in the corresponding directory according to the previous guidelines.
2. Input the following commands in terminal.

```bash
  python gradio_app.py
```
