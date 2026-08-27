import torch.nn as nn
from torchvision.models import resnet18


def get_model(architecture: str = "resnet18", num_classes: int = 10) -> nn.Module:
    """
    Create a CIFAR-10 image classification model.

    Args:
        architecture: Model architecture name. Currently supports resnet18.
        num_classes: Number of output classes.

    Returns:
        A PyTorch classification model.
    """
    if architecture != "resnet18":
        raise ValueError(
            f"Unsupported architecture: {architecture}. "
            "Supported architectures: resnet18."
        )

    model = resnet18(weights=None)

    # Adapt ResNet-18 for CIFAR-10's 32x32 images.
    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )

    model.maxpool = nn.Identity()

    # Replace ImageNet's 1000-class classifier.
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
