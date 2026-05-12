import timm
import torch.nn as nn


def create_model(name: str, num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    """
    Load a timm model with a classifier head sized for `num_classes`.

    timm handles head replacement per-architecture internally:
      - ResNet:    model.fc
      - TinyViT:   model.head
      - FastViT:   model.head
    """
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)


def load_all_models(num_classes: int = 4, pretrained: bool = True) -> dict[str, nn.Module]:
    timm_names = {
        "resnet50":     "resnet50",
        "tiny_vit_11m": "tiny_vit_11m_224",
        "fastvit_t8":   "fastvit_t8",
    }
    return {
        key: create_model(timm_name, num_classes=num_classes, pretrained=pretrained)
        for key, timm_name in timm_names.items()
    }
