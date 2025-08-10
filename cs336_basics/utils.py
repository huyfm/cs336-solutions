import os
from typing import IO, BinaryIO

import numpy as np
import tiktoken
import torch

from cs336_basics.modules import softmax


class DataLoader:
    def __init__(self, fpath: str, B: int, T: int):
        self.data = np.memmap(fpath, dtype=np.uint16)
        self.B = B
        self.T = T
        self.cur_idx = 0

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = self.B, self.T
        # input batch
        x_np = self.data[self.cur_idx : self.cur_idx + B * T]
        x = torch.from_numpy(x_np).to(dtype=torch.int64).view(B, T)
        # target batch: input batch shift by one.
        y_np = self.data[self.cur_idx + 1 : self.cur_idx + B * T + 1]
        y = torch.from_numpy(y_np).to(dtype=torch.int64).view(B, T)

        # advance to next batch
        self.cur_idx += B * T
        if self.cur_idx >= len(self.data):
            self.cur_idx = 0

        return x, y

    def __len__(self):
        return len(self.data)

    def reset(self):
        self.cur_idx = 0


def get_batch(data: np.ndarray, B: int, T: int, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
    ids = np.random.randint(len(data) - T, size=B)
    x_np = np.stack([data[i : i + T] for i in ids], axis=0)
    x = torch.from_numpy(x_np).to(dtype=torch.int64, device=device)  # (B, T)
    y_np = np.stack([data[i + 1 : i + T + 1] for i in ids], axis=0)
    y = torch.from_numpy(y_np).to(dtype=torch.int64, device=device)  # (B, T)
    return x, y


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    state_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(state_dict, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str | torch.device,
):
    state_dict = torch.load(src, map_location=device)

    model.load_state_dict(state_dict["model"])
    optimizer.load_state_dict(state_dict["optimizer"])

    return state_dict["iteration"]


@torch.no_grad()
def complete(model: torch.nn.Module, prefix: str, num_choices: int, max_length: int, device: str) -> list[str]:
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(prefix)
    x = torch.tensor(tokens, dtype=torch.int64).repeat(num_choices, 1)
    xgen = x.to(device)
    model.eval()

    while xgen.size(1) < max_length:
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model(xgen)  # (B, T, vocab_size)
        # take logits of the last step.
        logits = logits[:, -1, :]  # (B, vocab_size)
        # compute predicted distribution over the next token.
        probs = softmax(logits, dim=1)
        # do the topk sampling.
        topk_probs, topk_ids = torch.topk(probs, k=50, dim=1)  # each tensor: (B, k)
        # generate indices from topk probabilities.
        idx = torch.multinomial(topk_probs, 1)  # (B, 1)
        # gather corresponding tokens from indices.
        xnext = torch.gather(topk_ids, -1, idx)  # (B, 1)
        # concat new token to current token sequence.
        xgen = torch.cat([xgen, xnext], dim=1)  # (B, T+1)

    # print generated text
    endoftext_tok = 50256
    out: list[str] = []
    for i in range(num_choices):
        tokens = xgen[i].tolist()
        # cap the text to <|endoftext|> token.
        try:
            endoftext_idx = tokens.index(endoftext_tok)
            tokens = tokens[:endoftext_idx]
        except ValueError:
            pass
        decoded = enc.decode(tokens)
        out.append(decoded)
    return out
