import torch
import pytest

from src.model import get_model


def test_resnet18_can_be_created():
    model = get_model("resnet18", 10)

    assert model is not None
    assert isinstance(model, torch.nn.Module)


def test_resnet18_cifar10_output_shape():
    model = get_model("resnet18", 10)
    model.eval()

    inputs = torch.randn(4, 3, 32, 32)

    with torch.no_grad():
        outputs = model(inputs)

    assert outputs.shape == (4, 10)


def test_model_runs_on_cpu():
    model = get_model("resnet18", 10)
    model.eval()

    inputs = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        outputs = model(inputs)

    assert outputs.device.type == "cpu"
    assert outputs.shape == (2, 10)


def test_invalid_architecture_is_rejected():
    with pytest.raises(ValueError):
        get_model("invalid_model", 10)


def test_model_has_trainable_parameters():
    model = get_model("resnet18", 10)

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    assert len(trainable_parameters) > 0


def test_cifar10_transforms_produce_expected_shape():
    from src.dataset import get_transforms

    transform = get_transforms(train=False)

    # CIFAR-10 sample represented as a PIL image.
    from PIL import Image
    import numpy as np

    image = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    tensor = transform(image)

    assert tensor.shape == (3, 32, 32)
