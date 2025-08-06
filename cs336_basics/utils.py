import os
from typing import IO, BinaryIO

import numpy as np
import torch
from torch import Tensor


def get_batch(data: np.ndarray, batch_size: int, ctx_len: int, device: str) -> tuple[Tensor, Tensor]:
    ids = np.random.randint(len(data) - ctx_len, size=batch_size)

    xs = [torch.from_numpy(data[i : i + ctx_len]) for i in ids]
    inputs = torch.stack(xs, dim=0).to(device)

    # targets are inputs shifted by 1.
    ys = [torch.from_numpy(data[i + 1 : i + 1 + ctx_len]) for i in ids]
    targets = torch.stack(ys, dim=0).to(device)

    return inputs, targets


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
