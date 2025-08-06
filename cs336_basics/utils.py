import numpy as np
import torch
from torch import Tensor


def get_batch(data: np.ndarray, batch_size: int, ctx_len: int, device: str) -> tuple[Tensor, Tensor]:
    start_ids = np.random.randint(len(data) - ctx_len, size=batch_size)

    xs = [torch.from_numpy(data[i : i + ctx_len]) for i in start_ids]
    inputs = torch.stack(xs, dim=0).to(device)

    ys = [torch.from_numpy(data[i + 1 : i + 1 + ctx_len]) for i in start_ids]
    targets = torch.stack(ys, dim=0).to(device)

    return inputs, targets
