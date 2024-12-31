# Anime Flow

A simple implementation of conditional flow matching for generating anime faces. The model architecture closely follows the Diffusion Transformer model (DiT) found at https://github.com/facebookresearch/DiT/blob/main/models.py.

## Train model

```bash
pip install uv
uv run train.py --config ./config.yaml
```

## Generate images

```bash
uv run sample.py --ckpt ./state_1000000.ckpt --config ./config.yaml --seed 0
```

![image](https://github.com/user-attachments/assets/d7372aaa-4409-47d8-bb92-a212d8d17717)
