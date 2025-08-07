import os
from typing import IO, BinaryIO

import numpy as np
import torch


def get_batch(data: np.ndarray, B: int, T: int, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
    ids = np.random.randint(len(data) - T, size=B)
    x_np = np.stack([data[i : i + T] for i in ids], axis=0)
    x = torch.from_numpy(x_np).to(device)  # (B, T)
    y_np = np.stack([data[i + 1 : i + T + 1] for i in ids], axis=0)
    y = torch.from_numpy(y_np).to(device)  # (B, T)
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
