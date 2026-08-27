import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml

from dataset import get_dataloaders
from model import get_model
import os
import json
import random


def load_config(config_path: str) -> dict[str, Any]:
    """Load YAML training configuration."""
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Training configuration must be a YAML mapping.")

    return config


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible training."""
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training epoch."""
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)

        predictions = outputs.argmax(dim=1)
        total += targets.size(0)
        correct += predictions.eq(targets).sum().item()

    if total == 0:
        raise RuntimeError("Training loader returned zero samples.")

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate the model."""
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)

        predictions = outputs.argmax(dim=1)
        total += targets.size(0)
        correct += predictions.eq(targets).sum().item()

    if total == 0:
        raise RuntimeError("Validation loader returned zero samples.")

    return total_loss / total, correct / total


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    val_accuracy: float,
    save_path: Path,
) -> None:
    """Save the model and optimizer state."""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        },
        save_path,
    )


def main() -> None:
    # Support Kubernetes/Docker mounted configuration and local development.
    config_path = Path("/app/configs/training_config.yaml")

    if not config_path.exists():
        config_path = Path("configs/training_config.yaml")

    config = load_config(str(config_path))
    config["data"]["data_dir"] = os.getenv(
    "MLOPS_DATA_DIR",
    config["data"]["data_dir"],
)

    config["output"]["checkpoint_dir"] = os.getenv(
    "MLOPS_CHECKPOINT_DIR",
    config["output"]["checkpoint_dir"],
)

    seed = int(config["training"].get("seed", 42))
    set_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = get_model(
        architecture=config["model"]["architecture"],
        num_classes=int(config["model"]["num_classes"]),
    ).to(device)

    train_loader, val_loader = get_dataloaders(
        data_dir=config["data"]["data_dir"],
        batch_size=int(config["training"]["batch_size"]),
        num_workers=int(config["training"].get("num_workers", 2)),
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
    )

    criterion = nn.CrossEntropyLoss()
    checkpoint_dir = Path(config["output"]["checkpoint_dir"])
    checkpoint_path = checkpoint_dir / config["output"]["model_name"]

    

    checkpoint_path = checkpoint_dir / config["output"]["model_name"]

    epochs = int(config["training"]["epochs"])
    patience = int(config["training"]["early_stopping_patience"])

    best_val_loss = float("inf")
    patience_counter = 0

    print(
        json.dumps(
            {
                "event": "training_started",
                "device": str(device),
                "epochs": epochs,
                "batch_size": int(config["training"]["batch_size"]),
            }
        ),
        flush=True,
    )

    for epoch in range(epochs):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        log_entry = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_accuracy, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_accuracy, 4),
        }

        print(json.dumps(log_entry), flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
                save_path=checkpoint_path,
            )

            print(
                json.dumps(
                    {
                        "event": "checkpoint_saved",
                        "path": str(checkpoint_path),
                    }
                ),
                flush=True,
            )

        else:
            patience_counter += 1

            if patience_counter >= patience:
                print(
                    json.dumps(
                        {
                            "event": "early_stopping",
                            "epoch": epoch + 1,
                        }
                    ),
                    flush=True,
                )
                break

    print(
        json.dumps(
            {
                "event": "training_complete",
                "best_val_loss": round(best_val_loss, 4),
                "checkpoint": str(checkpoint_path),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
