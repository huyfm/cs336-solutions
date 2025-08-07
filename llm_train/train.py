import argparse
import time
from pydantic import BaseModel
from cs336_basics.modules import Transformer, cross_entropy, AdamW, cosine_lr
from cs336_basics.utils import get_batch
import torch
import numpy as np


# parser = argparse.ArgumentParser(description="CS336 Transformer training CLI")

# parser.add_argument("--train_path", type=str, required=True, help="filepath to training dataset")
# parser.add_argument("--valid_path", type=str, required=True, help="filepath to validation dataset")
# parser.add_argument("--batch_size", type=int, required=True, help="batch size")
# parser.add_argument("--lr_min", type=float, required=True, help="minimum learning rate")
# parser.add_argument("--lr_max", type=float, required=True, help="maximum learning rate")
# parser.add_argument("--num_iters", type=int, required=True, help="number of iterations")
# parser.add_argument("--num_warmup_iters", type=int, default=10, help="number of warming up iterations") 
# parser.add_argument("--weight_decay", type=float, default=1e-3, help="weight decay regularization factor")

# args = parser.parse_args()

# Hyperparameters
# lr_min = 
# lr_max =
# num_warmup =

B = 32
T = 32
num_iters = 100
weight_decay = 100


if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(device)


class Config(BaseModel):
    vocab_size: int = 50257
    ctx_len: int = 1024
    num_layers: int
    num_heads: int
    d_model: int
    d_ff: int
    rope_theta: float = 1e5


config = Config(num_layers=12, d_model=768, num_heads=12, d_ff=3072)
model = Transformer(**config.model_dump(), dtype=torch.float32, device=torch.device("cuda:0"))

Xtrain = np.memmap("data/TinyStoriesV2-GPT4-valid.dat", dtype=np.uint16)[:1000]
# Xval = np.memmap(valid_path, dtype=np.uint16)

optimizer = AdamW(model.parameters(), lr=3e-4, betas=(0.95, 0.99), eps=1e-5, weight_decay=weight_decay)

for step in range(1, num_iters):
    t0 = time.perf_counter()
    x, y = get_batch(Xtrain, B, T, device)
    logits = model(x)
    loss = cross_entropy(logits, y)
    optimizer.zero_grad()
    loss.backward()

    # Compute cosine annealing learning rate and update parameters.
    # lr = cosine_lr(step, lr_min, lr_max, warmup_iters, num_iters)
    # for group in optimizer.param_groups:
    #     group["lr"] = lr
    optimizer.step()

    dt = time.perf_counter() - t0
    tok_per_sec = x.numel() / dt
    print(f"step: {step}, loss: {loss.item():.4f}, dt: {dt * 1000 :.2f} ms, tok/sec: {tok_per_sec}")
