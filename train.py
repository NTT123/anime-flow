"""
A simple implementation of conditional flow matching for generating anime faces.
"""

import argparse
import pickle
import random
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import optax
import ot
import yaml
from flax import nnx
from jax.experimental import ode
from PIL import Image
from tqdm.cli import tqdm

from model import DiT, DiTConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    return parser.parse_args()


def load_config(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def gen_data_batches(data, batch_size):
    N = data.shape[0]
    while True:
        random_indices = np.random.choice(N, size=batch_size, replace=False)
        batch = data[random_indices]
        batch = batch.astype(np.float32) / 256
        yield batch


def loss_fn(flow, batch):
    xt, t, vt = batch
    velocity = flow(xt, t)
    loss = jnp.mean(jnp.square(velocity - vt))
    return loss


def train_step(flow, optimizer, rngs, batch):
    x0, x1 = batch
    noise = jax.random.uniform(rngs(), shape=x1.shape, minval=0, maxval=1 / 256)
    x1 = x1 + noise
    # randomize t
    t = jax.random.uniform(rngs(), (x1.shape[0],), minval=0, maxval=1)
    # randomize x0
    xt = x0 + (x1 - x0) * t[:, None, None, None]
    vt = x1 - x0
    batch = (xt, t, vt)
    loss, grads = nnx.value_and_grad(loss_fn)(flow, batch)
    optimizer.update(grads)
    return loss


@jax.jit
def train_step_raw(graphdef, state, batch):
    flow, optimizer, rngs = nnx.merge(graphdef, state)
    loss = train_step(flow, optimizer, rngs, batch)
    _, state = nnx.split((flow, optimizer, rngs))
    return state, loss


@jax.jit
def sample_images(graphdef, state):
    flow, _, _ = nnx.merge(graphdef, state)

    def flow_fn(y, t):
        o = flow(y, t[None])
        return o

    x = jax.random.normal(nnx.Rngs(0)(), shape=(16, 64, 64, 3), dtype=jnp.float32)
    o = ode.odeint(flow_fn, x, jnp.linspace(0, 1, 1000))
    o = jnp.clip(o[-1], 0, 1)
    return o


def generate_ot_pairs(x1):
    n = x1.shape[0]
    x0 = np.random.randn(*x1.shape)
    d1 = x1.reshape(n, -1)
    d0 = x0.reshape(n, -1)
    # loss matrix
    M = ot.dist(d0, d1)
    a, b = np.ones((n,)), np.ones((n,))
    G0 = ot.emd(a, b, M)
    d1 = np.matmul(G0, d1)
    x1 = d1.reshape(*x1.shape)
    return x0, x1


def plot_new_images(step: int, graphdef, state):
    images = sample_images(graphdef, state)

    plt.figure(figsize=(2, 2))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(images[i])
        plt.axis("off")
    plt.subplots_adjust(left=0, bottom=0, top=1, right=1, wspace=0, hspace=0)
    plt.savefig(f"images_{step:06d}.png")
    plt.close()


args = parse_args()
config = load_config(args.config)

# Download latest version
path = kagglehub.dataset_download("thimac/anime-face-64")
data_path = Path(path) / "64x64"
print("Path to dataset files:", data_path)

data_dir = data_path
image_files = sorted(data_dir.glob("*.jpg"))
random.Random(config["data"]["random_seed"]).shuffle(image_files)
N = len(image_files)
dataset = np.empty((N, 64, 64, 3), dtype=np.uint8)
for i, file_path in enumerate(tqdm(image_files)):
    dataset[i] = Image.open(file_path)

L = int(N * config["data"]["train_split"])
train_data = dataset[:L]
test_data = dataset[L:]

plt.figure(figsize=(2, 2))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(train_data[i])
    plt.axis("off")
plt.subplots_adjust(left=0, bottom=0, top=1, right=1, wspace=0, hspace=0)
plt.savefig("train_data_samples.png")
plt.close()

scheduler = optax.cosine_onecycle_schedule(
    transition_steps=config["training"]["num_steps"],
    peak_value=config["training"]["learning_rate"],
    pct_start=config["training"]["warmup_pct"],
)

gradient_transform = optax.chain(
    optax.clip_by_global_norm(config["training"]["grad_clip_norm"]),
    optax.scale_by_adam(),
    optax.scale_by_schedule(scheduler),
    optax.add_decayed_weights(config["training"]["weight_decay"]),
    optax.scale(-1.0),
)

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

flow = DiT(dit_config, rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(flow, gradient_transform)

rngs = nnx.Rngs(0)
graphdef, state = nnx.split((flow, optimizer, rngs))
train_data_iter = gen_data_batches(train_data, config["training"]["batch_size"])

start = time.perf_counter()
losses = []
ckpt_path = config["checkpointing"].get("resume_from_checkpoint")
if ckpt_path:
    del state
    with open(ckpt_path, "rb") as f:
        state = pickle.load(f)
    print(f"Resuming from checkpoint {ckpt_path}")
    step_str = Path(ckpt_path).stem.split("_")[-1]
    start_step = int(step_str) + 1
else:
    start_step = 1

for step, batch in enumerate(train_data_iter, start=start_step):
    x0, x1 = generate_ot_pairs(batch)
    state, loss = train_step_raw(graphdef, state, (x0, x1))

    if step % 100 == 0:
        losses.append(loss.item())

    if step % config["checkpointing"]["log_every"] == 0:
        end = time.perf_counter()
        duration = end - start
        loss = sum(losses) / len(losses)
        start = time.perf_counter()
        losses = []
        print(f"step {step:06d}  loss {loss:.3f}  duration {duration:.3f}s", flush=True)

    if step % config["checkpointing"]["plot_every"] == 0:
        plot_new_images(step, graphdef, state)

    if step % config["checkpointing"]["save_every"] == 0:
        # save checkpoint
        with open(f"state_{step:06d}.ckpt", "wb") as f:
            pickle.dump(state, f)
