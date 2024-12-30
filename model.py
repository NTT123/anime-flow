import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


@dataclass
class DiTConfig:
    input_dim: int
    hidden_dim: int
    num_blocks: int
    num_heads: int
    patch_size: int
    patch_stride: int
    time_freq_dim: int
    time_max_period: int
    mlp_ratio: int
    use_bias: bool
    padding: str
    pos_embed_cls_token: bool
    pos_embed_extra_tokens: int


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = jnp.arange(grid_size, dtype=jnp.float32)
    grid_w = jnp.arange(grid_size, dtype=jnp.float32)
    grid = jnp.meshgrid(grid_w, grid_h)  # here w goes first
    grid = jnp.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = jnp.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = jnp.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 16**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = jnp.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = jnp.sin(out)  # (M, D/2)
    emb_cos = jnp.cos(out)  # (M, D/2)

    emb = jnp.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class PatchEmbedding(nnx.Module):
    """Patch embedding module."""

    def __init__(self, config: DiTConfig, *, rngs: nnx.Rngs):
        super().__init__()
        self.cnn = nnx.Conv(
            config.input_dim,
            config.hidden_dim,
            kernel_size=(config.patch_size, config.patch_size),
            strides=(config.patch_stride, config.patch_stride),
            padding=config.padding,
            use_bias=config.use_bias,
            rngs=rngs,
        )

    def __call__(self, x):
        return self.cnn(x)


class TimeEmbedding(nnx.Module):
    """Time embedding module."""

    def __init__(self, config: DiTConfig, *, rngs: nnx.Rngs):
        super().__init__()
        self.freq_dim = config.time_freq_dim
        self.max_period = config.time_max_period
        self.fc1 = nnx.Linear(
            self.freq_dim, config.hidden_dim, use_bias=config.use_bias, rngs=rngs
        )
        self.fc2 = nnx.Linear(
            config.hidden_dim, config.hidden_dim, use_bias=config.use_bias, rngs=rngs
        )

    @staticmethod
    def cosine_embedding(t, dim, max_period):
        assert dim % 2 == 0
        half = dim // 2
        freqs = jnp.exp(
            -math.log(max_period)
            * jnp.arange(start=0, stop=half, dtype=jnp.float32)
            / half
        )
        args = t[:, None] * freqs[None, :] * 1024
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        return embedding

    def __call__(self, t):
        t_freq = self.cosine_embedding(t, self.freq_dim, self.max_period)
        t_embed = self.fc1(t_freq)
        t_embed = nnx.silu(t_embed)
        t_embed = self.fc2(t_embed)
        return t_embed


class MLP(nnx.Module):
    """MLP module."""

    def __init__(self, config: DiTConfig, *, rngs: nnx.Rngs):
        super().__init__()
        self.fc1 = nnx.Linear(
            config.hidden_dim,
            config.hidden_dim * config.mlp_ratio,
            use_bias=config.use_bias,
            rngs=rngs,
        )
        self.fc2 = nnx.Linear(
            config.hidden_dim * config.mlp_ratio,
            config.hidden_dim,
            use_bias=config.use_bias,
            rngs=rngs,
        )

    def __call__(self, x):
        x = self.fc1(x)
        x = nnx.silu(x)
        x = self.fc2(x)
        return x


class SelfAttention(nnx.Module):
    """Self attention module."""

    def __init__(self, config: DiTConfig, *, rngs: nnx.Rngs):
        super().__init__()
        self.fc = nnx.Linear(
            config.hidden_dim,
            3 * config.hidden_dim,
            use_bias=config.use_bias,
            rngs=rngs,
        )
        self.heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        assert config.hidden_dim % config.num_heads == 0
        self.q_norm = nnx.RMSNorm(num_features=self.head_dim, use_scale=True, rngs=rngs)
        self.k_norm = nnx.RMSNorm(num_features=self.head_dim, use_scale=True, rngs=rngs)

    def __call__(self, x):
        q, k, v = jnp.split(self.fc(x), 3, axis=-1)
        # reshape q, k v, to N, T, H, D
        q = q.reshape(q.shape[0], q.shape[1], self.heads, self.head_dim)
        k = k.reshape(k.shape[0], k.shape[1], self.heads, self.head_dim)
        v = v.reshape(v.shape[0], v.shape[1], self.heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        o = jax.nn.dot_product_attention(q, k, v, is_causal=False)
        o = o.reshape(o.shape[0], o.shape[1], self.heads * self.head_dim)
        return o


def modulate(x, shift, scale):
    return x * (1 + scale[:, None, :]) + shift[:, None, :]


class TransformerBlock(nnx.Module):
    """Transformer block."""

    def __init__(self, config: DiTConfig, *, rngs: nnx.Rngs):
        super().__init__()
        self.norm1 = nnx.RMSNorm(
            num_features=config.hidden_dim, use_scale=False, rngs=rngs
        )
        self.attn = SelfAttention(config, rngs=rngs)
        self.norm2 = nnx.RMSNorm(
            num_features=config.hidden_dim, use_scale=False, rngs=rngs
        )
        self.mlp = MLP(config, rngs=rngs)
        self.adalm_modulation = nnx.Sequential(
            nnx.silu,
            nnx.Linear(
                config.hidden_dim,
                6 * config.hidden_dim,
                use_bias=config.use_bias,
                rngs=rngs,
            ),
        )

    def __call__(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(
            self.adalm_modulation(c), 6, axis=-1
        )
        attn_x = self.norm1(x)
        attn_x = modulate(attn_x, shift_msa, scale_msa)
        x = x + gate_msa[:, None, :] * self.attn(attn_x)
        mlp_x = self.norm2(x)
        mlp_x = modulate(mlp_x, shift_mlp, scale_mlp)
        x = x + gate_mlp[:, None, :] * self.mlp(mlp_x)
        return x


class FinalLayer(nnx.Module):
    """Final layer."""

    def __init__(self, config: DiTConfig, *, rngs: nnx.Rngs):
        super().__init__()
        self.norm = nnx.RMSNorm(
            num_features=config.hidden_dim, use_scale=False, rngs=rngs
        )
        self.conv = nnx.ConvTranspose(
            config.hidden_dim,
            config.input_dim,
            kernel_size=(config.patch_size, config.patch_size),
            strides=(config.patch_stride, config.patch_stride),
            padding=config.padding,
            use_bias=config.use_bias,
            rngs=rngs,
        )
        self.adalm_modulation = nnx.Sequential(
            nnx.silu,
            nnx.Linear(
                config.hidden_dim,
                2 * config.hidden_dim,
                use_bias=config.use_bias,
                rngs=rngs,
            ),
        )

    def __call__(self, x, c):
        shift, scale = jnp.split(self.adalm_modulation(c), 2, axis=-1)
        x = self.norm(x)
        x = modulate(x, shift, scale)
        # reshape to N, H, W, C
        H = W = int(x.shape[1] ** 0.5)
        x = x.reshape(x.shape[0], H, W, x.shape[-1])
        x = self.conv(x)
        return x


class DiT(nnx.Module):
    """Diffusion Transformer"""

    def __init__(self, config: DiTConfig, *, rngs: nnx.Rngs):
        super().__init__()
        self.config = config
        self.time_embedding = TimeEmbedding(config, rngs=rngs)
        self.patch_embedding = PatchEmbedding(config, rngs=rngs)
        self.blocks = [
            TransformerBlock(config, rngs=rngs) for _ in range(config.num_blocks)
        ]
        self.final_layer = FinalLayer(config, rngs=rngs)

    def __call__(self, xt, t):
        t = self.time_embedding(t)
        x = self.patch_embedding(xt)
        N, H, W, D = x.shape
        x = x.reshape(N, H * W, D)
        x = x + get_2d_sincos_pos_embed(
            D,
            H,
            cls_token=self.config.pos_embed_cls_token,
            extra_tokens=self.config.pos_embed_extra_tokens,
        ).reshape(1, H * W, D)
        c = t
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        return x
