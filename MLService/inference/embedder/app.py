import os
import base64
import time
from contextlib import asynccontextmanager
from io import BytesIO

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
from transformers import AutoProcessor, AutoModel

MODEL_NAME = os.environ.get("SIGLIP_MODEL", "google/siglip2-base-patch16-224")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

processor = None
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global processor, model
    print(f"Loading {MODEL_NAME} on {DEVICE}...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    print("Model ready.")
    yield


app = FastAPI(title="SigLIP 2 Embedder", lifespan=lifespan)


class TextRequest(BaseModel):
    text: str


class ImageRequest(BaseModel):
    image: str  # base64-encoded JPEG/PNG


class BatchImageRequest(BaseModel):
    images: list[str]  # list of base64-encoded images


def decode_image(b64: str) -> Image.Image:
    try:
        data = base64.b64decode(b64)
        return Image.open(BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "device": DEVICE}


@app.post("/embed/text")
def embed_text(req: TextRequest):
    inputs = processor(text=[req.text], return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        features = model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
    return {"embedding": features[0].cpu().tolist(), "model": MODEL_NAME}


@app.post("/embed/image")
def embed_image(req: ImageRequest):
    image = decode_image(req.image)
    inputs = processor(images=[image], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        features = model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
    return {"embedding": features[0].cpu().tolist(), "model": MODEL_NAME}


@app.post("/embed/images")
def embed_images(req: BatchImageRequest):
    if not req.images:
        raise HTTPException(status_code=400, detail="images list is empty")
    images = [decode_image(b64) for b64 in req.images]
    inputs = processor(images=images, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        features = model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
    return {"embeddings": features.cpu().tolist(), "model": MODEL_NAME}
