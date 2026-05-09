import timm
import torch.nn as nn


def create_model(name: str, num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    """
    Load a timm model and replace its classifier head for num_classes output.

    Supported names: 'resnet50', 'tiny_vit_11m_224', 'fastvit_t8'
    """
    model = timm.create_model(name, pretrained=pretrained, num_classes=0)

    in_features = model.num_features

    model.head = nn.Linear(in_features, num_classes)

    return model


def load_all_models(num_classes: int = 4, pretrained: bool = True) -> dict[str, nn.Module]:
    timm_names = {
        "resnet50":       "resnet50",
        "tiny_vit_11m":   "tiny_vit_11m_224",
        "fastvit_t8":     "fastvit_t8",
    }
    return {
        key: create_model(timm_name, num_classes=num_classes, pretrained=pretrained)
        for key, timm_name in timm_names.items()
    }
