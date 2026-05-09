"""
Utilities for freezing/unfreezing backbone parameters during two-phase training.

Phase 1 (epochs 1-freeze_epochs):  freeze backbone, train head only.
Phase 2 (epoch unfreeze_epoch+):   unfreeze full model, fine-tune end-to-end.
"""

import torch.nn as nn


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all parameters except the classifier head."""
    for name, param in model.named_parameters():
        if not name.startswith("head"):
            param.requires_grad = False


def unfreeze_all(model: nn.Module) -> None:
    """Unfreeze all model parameters."""
    for param in model.parameters():
        param.requires_grad = True


def get_head_params(model: nn.Module) -> list:
    return [p for n, p in model.named_parameters() if n.startswith("head")]


def get_backbone_params(model: nn.Module) -> list:
    return [p for n, p in model.named_parameters() if not n.startswith("head")]
