"""
Main training loop for all three models.

Usage:
    python training/train.py --config training/config.yaml --model resnet50
    python training/train.py --config training/config.yaml --model tiny_vit_11m
    python training/train.py --config training/config.yaml --model fastvit_t8
    python training/train.py --config training/config.yaml --model all
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import SICAPv2Dataset
from data.transforms import get_train_transforms, get_val_transforms
from models.classifier_heads import freeze_backbone, unfreeze_all
from models.load_models import create_model
from training.utils import AverageMeter, copy_checkpoint_to_drive, save_checkpoint, save_metrics, set_seed


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_dataloaders(cfg: dict, project_root: Path):
    train_ds = SICAPv2Dataset(
        xlsx_path=project_root / cfg["train_xlsx"],
        images_dir=project_root / cfg["images_dir"],
        transform=get_train_transforms(cfg["input_size"]),
    )
    val_ds = SICAPv2Dataset(
        xlsx_path=project_root / cfg["val_xlsx"],
        images_dir=project_root / cfg["images_dir"],
        transform=get_val_transforms(cfg["input_size"]),
    )

    class_weights = train_ds.get_class_weights()
    sample_weights = [class_weights[lbl].item() for lbl in train_ds.labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], sampler=sampler,
        num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=4, pin_memory=True,
    )
    return train_loader, val_loader, class_weights


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    loss_meter = AverageMeter()
    correct = total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with autocast(enabled=scaler is not None):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        loss_meter.update(loss.item(), labels.size(0))

    return loss_meter.avg, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    correct = total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        loss_meter.update(loss.item(), labels.size(0))

    return loss_meter.avg, correct / total


def train_model(model_key: str, cfg: dict, project_root: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg["seed"])

    timm_name = cfg["models"][model_key]["timm_name"]
    model = create_model(timm_name, num_classes=cfg["num_classes"]).to(device)

    train_loader, val_loader, class_weights = build_dataloaders(cfg, project_root)
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    scaler = GradScaler() if cfg["mixed_precision"] and device.type == "cuda" else None

    freeze_backbone(model)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    checkpoint_dir = project_root / cfg["checkpoint_dir"] / model_key
    best_val_acc = 0.0
    history = []

    print(f"\n{'='*60}")
    print(f"Training: {model_key}  ({timm_name})")
    print(f"Device: {device}")
    print(f"{'='*60}")

    for epoch in range(1, cfg["epochs"] + 1):
        if epoch == cfg["unfreeze_epoch"]:
            print(f"\nEpoch {epoch}: Unfreezing full model.")
            unfreeze_all(model)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=cfg["learning_rate"],
                weight_decay=cfg["weight_decay"],
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg["epochs"] - epoch + 1
            )

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc

        state = {
            "epoch": epoch,
            "model_key": model_key,
            "state_dict": model.state_dict(),
            "val_acc": val_acc,
            "optimizer": optimizer.state_dict(),
        }
        ckpt_name = f"epoch_{epoch:02d}.pt"
        save_checkpoint(state, checkpoint_dir, ckpt_name, is_best=is_best)

        if cfg.get("save_to_drive"):
            copy_checkpoint_to_drive(
                checkpoint_dir / ckpt_name,
                Path(cfg["drive_checkpoint_dir"]) / model_key,
            )

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{cfg['epochs']}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
            + ("  *" if is_best else "")
        )

    return {"model": model_key, "best_val_acc": round(best_val_acc, 4), "history": history}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/config.yaml")
    parser.add_argument(
        "--model",
        default="all",
        choices=["resnet50", "tiny_vit_11m", "fastvit_t8", "all"],
    )
    parser.add_argument("--project_root", default=None)
    args = parser.parse_args()

    project_root = Path(args.project_root) if args.project_root else Path(__file__).resolve().parent.parent
    cfg = load_config(project_root / args.config if not Path(args.config).is_absolute() else args.config)

    model_keys = list(cfg["models"].keys()) if args.model == "all" else [args.model]

    all_results = {}
    for key in model_keys:
        result = train_model(key, cfg, project_root)
        all_results[key] = result

    metrics_path = project_root / cfg["metrics_file"]
    existing = {}
    if metrics_path.exists():
        import json
        with open(metrics_path) as f:
            existing = json.load(f)
    existing.update(all_results)
    save_metrics(existing, metrics_path)
    print(f"\nMetrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
