"""
Grad-CAM visualization for ResNet-50, TinyViT-11M, and FastViT-T8.

Hook targets per architecture:
  resnet50:       model.layer4          (last conv block)
  tiny_vit_11m:   model.stages[-1]      (last attention stage)
  fastvit_t8:     model.stages[-1]      (last reparameterized stage)

Usage:
    python evaluation/gradcam.py --config training/config.yaml --model all
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import CLASS_NAMES, SICAPv2Dataset
from data.transforms import get_val_transforms, IMAGENET_MEAN, IMAGENET_STD
from models.load_models import create_model

SAMPLES_PER_CLASS = 2  # number of representative patches per class


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        # output may be a tensor or a tuple (some ViT stages)
        if isinstance(output, tuple):
            output = output[0]
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        grad = grad_output[0]
        if grad is not None:
            self.gradients = grad.detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[0, class_idx]
        score.backward()

        acts = self.activations  # (1, C, H, W) or (1, N, C) for attention
        grads = self.gradients

        if acts is None or grads is None:
            raise RuntimeError("Activations or gradients not captured. Check target layer.")

        # Handle (B, N, C) sequence output from ViT stages
        if acts.dim() == 3:
            weights = grads.mean(dim=1, keepdim=True)  # (1, 1, C)
            cam = (weights * acts).sum(dim=-1)          # (1, N)
            n = cam.shape[1]
            s = int(n ** 0.5)
            cam = cam.view(1, 1, s, s)
        else:
            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam = (weights * acts).sum(dim=1, keepdim=True)

        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def remove(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def _get_target_layer(model, model_key: str):
    if model_key == "resnet50":
        return model.layer4
    # Both TinyViT and FastViT expose stages; hook last stage
    return model.stages[-1]


def _denormalize(tensor: torch.Tensor) -> np.ndarray:
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * std + mean
    return np.clip(img, 0, 1)


def _select_representative_patches(dataset: SICAPv2Dataset, samples_per_class: int) -> list[int]:
    selected = []
    seen = {c: 0 for c in range(len(CLASS_NAMES))}
    for idx, lbl in enumerate(dataset.labels):
        if seen[lbl] < samples_per_class:
            selected.append(idx)
            seen[lbl] += 1
        if all(v >= samples_per_class for v in seen.values()):
            break
    return selected


def generate_gradcam_grid(
    model,
    model_key: str,
    dataset: SICAPv2Dataset,
    device: torch.device,
    figures_dir: Path,
    sample_indices: list[int],
):
    target_layer = _get_target_layer(model, model_key)
    gradcam = GradCAM(model, target_layer)
    model.eval()

    n = len(sample_indices)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = [axes]

    for row, idx in enumerate(sample_indices):
        image_tensor, label = dataset[idx]
        input_tensor = image_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(input_tensor)
        pred_class = logits.argmax(dim=1).item()

        cam = gradcam.generate(input_tensor, pred_class)
        img_np = _denormalize(image_tensor)

        ax_img, ax_cam, ax_overlay = axes[row]

        ax_img.imshow(img_np)
        ax_img.set_title(f"True: {CLASS_NAMES[label]}")
        ax_img.axis("off")

        ax_cam.imshow(cam, cmap="jet")
        ax_cam.set_title(f"Grad-CAM (pred: {CLASS_NAMES[pred_class]})")
        ax_cam.axis("off")

        overlay = img_np.copy()
        heatmap = plt.cm.jet(cam)[..., :3]
        overlay = 0.6 * overlay + 0.4 * heatmap
        ax_overlay.imshow(overlay)
        ax_overlay.set_title("Overlay")
        ax_overlay.axis("off")

    plt.suptitle(f"Grad-CAM — {model_key}", fontsize=14, y=1.01)
    plt.tight_layout()

    out_path = figures_dir / f"gradcam_{model_key}.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")
    gradcam.remove()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/config.yaml")
    parser.add_argument("--model", default="all", choices=["resnet50", "tiny_vit_11m", "fastvit_t8", "all"])
    parser.add_argument("--project_root", default=None)
    args = parser.parse_args()

    project_root = Path(args.project_root) if args.project_root else Path(__file__).resolve().parent.parent
    config_path = project_root / args.config if not Path(args.config).is_absolute() else Path(args.config)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    figures_dir = project_root / cfg["figures_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)

    test_ds = SICAPv2Dataset(
        xlsx_path=project_root / cfg["test_xlsx"],
        images_dir=project_root / cfg["images_dir"],
        transform=get_val_transforms(cfg["input_size"]),
    )
    sample_indices = _select_representative_patches(test_ds, SAMPLES_PER_CLASS)

    model_keys = list(cfg["models"].keys()) if args.model == "all" else [args.model]

    for key in model_keys:
        model_cfg = cfg["models"][key]
        model = create_model(model_cfg["timm_name"], num_classes=cfg["num_classes"], pretrained=False).to(device)

        ckpt_dir = project_root / cfg["checkpoint_dir"] / key
        best_ckpts = sorted(ckpt_dir.glob("best_*.pt"))
        if not best_ckpts:
            print(f"  WARNING: no checkpoint for {key}, using random weights.")
        else:
            ckpt = torch.load(best_ckpts[-1], map_location=device)
            model.load_state_dict(ckpt["state_dict"])

        print(f"Generating Grad-CAM for {key}...")
        generate_gradcam_grid(model, key, test_ds, device, figures_dir, sample_indices)


if __name__ == "__main__":
    main()
