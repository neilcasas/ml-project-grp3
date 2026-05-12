"""
Utilities for freezing/unfreezing backbone parameters during two-phase training.

Phase 1 (epochs 1-freeze_epochs):  freeze backbone, train classifier head only.
Phase 2 (epoch unfreeze_epoch+):   unfreeze full model, fine-tune end-to-end.

Uses timm's `get_classifier()` to identify head parameters generically — works
for ResNet (.fc), TinyViT (.head), FastViT (.head), etc.
"""

import torch.nn as nn


def _classifier_param_ids(model: nn.Module) -> set:
    return {id(p) for p in model.get_classifier().parameters()}


def freeze_backbone(model: nn.Module) -> None:
    head_ids = _classifier_param_ids(model)
    for p in model.parameters():
        p.requires_grad = id(p) in head_ids


def unfreeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True


def get_head_params(model: nn.Module) -> list:
    return list(model.get_classifier().parameters())


def get_backbone_params(model: nn.Module) -> list:
    head_ids = _classifier_param_ids(model)
    return [p for p in model.parameters() if id(p) not in head_ids]
