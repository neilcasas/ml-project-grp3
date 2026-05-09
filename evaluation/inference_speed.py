"""
Benchmark inference time (ms/image) for all three models.

Usage:
    python evaluation/inference_speed.py --config training/config.yaml
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.load_models import create_model

WARMUP_RUNS = 20
TIMED_RUNS = 100


def benchmark(model, input_size: int, device: torch.device) -> float:
    model.eval()
    dummy = torch.randn(1, 3, input_size, input_size, device=device)

    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(TIMED_RUNS):
            model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return (elapsed / TIMED_RUNS) * 1000  # ms per image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/config.yaml")
    parser.add_argument("--project_root", default=None)
    args = parser.parse_args()

    project_root = Path(args.project_root) if args.project_root else Path(__file__).resolve().parent.parent
    config_path = project_root / args.config if not Path(args.config).is_absolute() else Path(args.config)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking on: {device}\n")

    speed_results = {}
    for key, model_cfg in cfg["models"].items():
        model = create_model(model_cfg["timm_name"], num_classes=cfg["num_classes"], pretrained=False).to(device)
        ms = benchmark(model, cfg["input_size"], device)
        num_params = sum(p.numel() for p in model.parameters())
        speed_results[key] = {
            "inference_ms_per_image": round(ms, 3),
            "num_params": num_params,
        }
        print(f"  {key}: {ms:.2f} ms/image  |  {num_params/1e6:.1f}M params")

    metrics_path = project_root / cfg["metrics_file"]
    existing = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            existing = json.load(f)

    for key, vals in speed_results.items():
        existing.setdefault(key, {}).update(vals)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nSpeed results saved to {metrics_path}")


if __name__ == "__main__":
    main()
