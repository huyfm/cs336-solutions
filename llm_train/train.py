import os
import time

import torch

from cs336_basics.modules import Transformer, cosine_lr, cross_entropy
from cs336_basics.utils import DataLoader

LOG_DIR = "llm_train/logs"

device = torch.device("cuda:1")
print("device:", device)

trainpath = "data/TinyStoriesV2-GPT4-train.dat"
validpath = "data/TinyStoriesV2-GPT4-valid.dat"

config = {
    "vocab_size": 50257,
    "ctx_len": 256,
    "num_layers": 4,
    "num_heads": 16,
    "d_model": 512,
    "d_ff": 1344,
    "rope_theta": 1e5,
}
model = Transformer(**config, device=device)
nparams = 0
for m in model.parameters():
    nparams += m.numel()
print(f"loaded model: {nparams // 10**6}M params")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)

model.compile()
torch.backends.cuda.enable_flash_sdp(True)
torch.set_float32_matmul_precision("high")

B = 64
T = 256
train_loader = DataLoader(trainpath, B, T)
val_loader = DataLoader(validpath, B, T)

lr_max = 6e-4
lr_min = 0.1 * lr_max
max_iters = len(train_loader) // (B * T)
val_iters = len(val_loader) // (B * T)
warmup_iters = 100
print(f"1 batch = {B * T} tokens")
print(f"train: {max_iters} iters, warmup: {warmup_iters} iters, val: {val_iters} iters")

ckpt_dir = LOG_DIR

for step in range(1, max_iters + 1):
    # Checkpoint model once in a while
    if step % 500 == 0 or step == max_iters:
        val_loss_accum = 0.0
        val_loader.reset()
        model.eval()
        for _ in range(val_iters):
            x, y = val_loader.next_batch()
            with torch.no_grad():
                logits = model(x)
                loss = cross_entropy(logits, y)
            val_loss_accum += loss.item()
        val_loss = val_loss_accum / val_iters
        checkpoint = {
            "model": model.state_dict(),
            "step": step,
            "val_loss": val_loss,
        }
        ckpt_name = f"model-step={step}-val_loss={val_loss:.4f}.pt"
        torch.save(checkpoint, os.path.join(ckpt_dir, ckpt_name))

    # Do one optimization step
    model.train()
    t0 = time.perf_counter()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(x)
        loss = cross_entropy(logits, y)
    optimizer.zero_grad()
    loss.backward()

    lr = cosine_lr(step, lr_min, lr_max, warmup_iters, max_iters)
    for group in optimizer.param_groups:
        group["lr"] = lr

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    tok_per_sec = x.numel() / dt
    print(
        f"step: {step} | loss: {loss.item():6.4f} | lr: {lr:.2e} | norm: {norm:.4f} | "
        f"dt: {dt * 1000:.2f} ms | tok/sec: {tok_per_sec:.0f}"
    )


# @torch.no_grad()
# def infer():


