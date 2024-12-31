"""
Generate images from trained model
"""

import argparse
import pickle

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import yaml
from flax import nnx
from jax.experimental import ode

from model import DiT, DiTConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--ckpt", type=str, default=None, help="Path to checkpoint file"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


@jax.jit
def sample_images(graphdef, state, rng):
    flow = nnx.merge(graphdef, state)

    def flow_fn(y, t):
        o = flow(y, t[None])
        return o

    x = jax.random.normal(rng, shape=(16, 64, 64, 3), dtype=jnp.float32)
    o = ode.odeint(flow_fn, x, jnp.linspace(0, 1, 2))
    o = jnp.clip(o[-1], 0, 1)
    return o


def plot_new_images(graphdef, state, seed):
    images = sample_images(graphdef, state, nnx.Rngs(seed)())

    plt.figure(figsize=(2, 2))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(images[i])
        plt.axis("off")
    plt.subplots_adjust(left=0, bottom=0, top=1, right=1, wspace=0, hspace=0)
    plt.savefig(f"samples.png")
    plt.close()


def main():
    args = parse_args()
    config = load_config(args.config)

    dit_config = DiTConfig(
        input_dim=config["model"]["input_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_blocks=config["model"]["num_blocks"],
        num_heads=config["model"]["num_heads"],
        patch_size=config["model"]["patch_size"],
        patch_stride=config["model"]["patch_stride"],
        time_freq_dim=config["model"]["time_freq_dim"],
        time_max_period=config["model"]["time_max_period"],
        mlp_ratio=config["model"]["mlp_ratio"],
        use_bias=config["model"]["use_bias"],
        padding=config["model"]["padding"],
        pos_embed_cls_token=config["model"]["pos_embed_cls_token"],
        pos_embed_extra_tokens=config["model"]["pos_embed_extra_tokens"],
    )

    abstract_flow = nnx.eval_shape(lambda: DiT(dit_config, rngs=nnx.Rngs(0)))
    graphdef, _ = nnx.split(abstract_flow)
    with open(args.ckpt, "rb") as f:
        state = pickle.load(f, fix_imports=True)
        if "time_embedding" not in state:
            state = state[0]
    plot_new_images(graphdef, state, args.seed)


if __name__ == "__main__":
    main()
