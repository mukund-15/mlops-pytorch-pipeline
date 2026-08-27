import io
import os
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from .model import get_model
from .dataset import get_transforms
from contextlib import asynccontextmanager


app = FastAPI(title="MLOps PyTorch Classifier")


CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


CHECKPOINT_PATH = Path(
    os.getenv(
        "MLOPS_CHECKPOINT_PATH",
        "checkpoints/classifier_v1.pt",
    )
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None


def load_model() -> torch.nn.Module:
    """Load the trained model checkpoint."""
    global model

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    loaded_model = get_model(
        architecture="resnet18",
        num_classes=10,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )

    loaded_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    loaded_model.to(DEVICE)
    loaded_model.eval()

    model = loaded_model

    return loaded_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_model()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load model from {CHECKPOINT_PATH}: {exc}"
        ) from exc

    yield


app = FastAPI(
    title="MLOps PyTorch Classifier",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    """Health endpoint."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    return {
        "status": "ok",
        "model_loaded": True,
    }


@app.post("/predict")
async def predict(image: UploadFile = File(...)) -> dict:
    """Run image classification."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    if image.content_type is None or not image.content_type.startswith(
        "image/"
    ):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image",
        )

    try:
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        transform = get_transforms(train=False)
        tensor = transform(pil_image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]

        predicted_index = int(torch.argmax(probabilities).item())

        class_probabilities = {
            CLASS_NAMES[index]: round(
                float(probabilities[index].item()),
                6,
            )
            for index in range(len(CLASS_NAMES))
        }

        return {
            "predicted_class": CLASS_NAMES[predicted_index],
            "confidence": round(
                float(probabilities[predicted_index].item()),
                6,
            ),
            "probabilities": class_probabilities,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to process image: {exc}",
        ) from exc
