import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(
    state: dict,
    checkpoint_dir: str | Path,
    filename: str,
    is_best: bool = False,
) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / filename
    torch.save(state, path)
    if is_best:
        best_path = checkpoint_dir / f"best_{filename}"
        shutil.copyfile(path, best_path)


def copy_checkpoint_to_drive(
    src_path: str | Path,
    drive_dir: str | Path,
) -> None:
    """Copy a checkpoint to Google Drive (Colab only)."""
    drive_dir = Path(drive_dir)
    drive_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, drive_dir / Path(src_path).name)


def save_metrics(metrics: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
