"""
Evaluate trained models on the held-out test set.

Usage:
    python evaluation/metrics.py --config training/config.yaml --model resnet50
    python evaluation/metrics.py --config training/config.yaml --model all
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import CLASS_NAMES, SICAPv2Dataset
from data.transforms import get_val_transforms
from models.load_models import create_model


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def compute_accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc = correct / total
    per_class = _per_class_accuracy(all_labels, all_preds)
    return acc, per_class


def _per_class_accuracy(labels, preds):
    counts = {c: [0, 0] for c in range(len(CLASS_NAMES))}
    for lbl, pred in zip(labels, preds):
        counts[lbl][1] += 1
        if lbl == pred:
            counts[lbl][0] += 1
    return {
        CLASS_NAMES[c]: round(counts[c][0] / counts[c][1], 4) if counts[c][1] > 0 else 0.0
        for c in counts
    }


def evaluate_model(model_key: str, cfg: dict, project_root: Path) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timm_name = cfg["models"][model_key]["timm_name"]
    model = create_model(timm_name, num_classes=cfg["num_classes"], pretrained=False).to(device)

    ckpt_dir = project_root / cfg["checkpoint_dir"] / model_key
    best_ckpts = sorted(ckpt_dir.glob("best_*.pt"))
    if not best_ckpts:
        raise FileNotFoundError(f"No best checkpoint found in {ckpt_dir}")
    ckpt = torch.load(best_ckpts[-1], map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    print(f"Loaded {model_key} from epoch {ckpt['epoch']} (val_acc={ckpt['val_acc']:.4f})")

    test_ds = SICAPv2Dataset(
        xlsx_path=project_root / cfg["test_xlsx"],
        images_dir=project_root / cfg["images_dir"],
        transform=get_val_transforms(cfg["input_size"]),
    )
    loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=4)

    acc, per_class = compute_accuracy(model, loader, device)
    result = {
        "test_accuracy": round(acc, 4),
        "per_class_accuracy": per_class,
        "num_params": sum(p.numel() for p in model.parameters()),
    }
    print(f"  {model_key}: test_acc={acc:.4f}  per_class={per_class}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/config.yaml")
    parser.add_argument("--model", default="all", choices=["resnet50", "tiny_vit_11m", "fastvit_t8", "all"])
    parser.add_argument("--project_root", default=None)
    args = parser.parse_args()

    project_root = Path(args.project_root) if args.project_root else Path(__file__).resolve().parent.parent
    cfg = load_config(project_root / args.config if not Path(args.config).is_absolute() else args.config)

    model_keys = list(cfg["models"].keys()) if args.model == "all" else [args.model]

    metrics_path = project_root / cfg["metrics_file"]
    existing = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            existing = json.load(f)

    for key in model_keys:
        result = evaluate_model(key, cfg, project_root)
        existing.setdefault(key, {}).update(result)

    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
