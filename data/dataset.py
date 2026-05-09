import os
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

# One-hot columns in the partition xlsx files, in class-index order
LABEL_COLUMNS = ["NC", "G3", "G4", "G5"]
CLASS_NAMES = LABEL_COLUMNS


class SICAPv2Dataset(Dataset):
    """
    Reads patch paths and labels from a SICAPv2 partition xlsx file.

    The xlsx has columns: image_name, NC, G3, G4, G5, G4C
    Labels are one-hot over NC/G3/G4/G5; we take argmax → class index 0-3.
    """

    def __init__(
        self,
        xlsx_path: str | Path,
        images_dir: str | Path,
        transform=None,
    ):
        self.images_dir = Path(images_dir)
        self.transform = transform

        df = pd.read_excel(xlsx_path)
        self.image_names = df["image_name"].tolist()
        self.labels = df[LABEL_COLUMNS].values.argmax(axis=1).tolist()

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int):
        img_path = self.images_dir / self.image_names[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights for weighted CrossEntropyLoss."""
        counts = torch.zeros(len(LABEL_COLUMNS))
        for lbl in self.labels:
            counts[lbl] += 1
        weights = 1.0 / counts
        return weights / weights.sum() * len(LABEL_COLUMNS)
