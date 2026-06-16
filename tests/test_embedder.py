"""
Integration tests for the embedder service (http://localhost:8002).
Run: pip install pytest httpx Pillow && pytest tests/test_embedder.py -v
"""
import base64
import math
from io import BytesIO

import httpx
import pytest
from PIL import Image

BASE = "http://localhost:8002"


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def dummy_b64_image(color=(255, 0, 0), size=(64, 64)):
    img = Image.new("RGB", size, color)
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture(scope="session")
def client():
    c = httpx.Client(base_url=BASE, timeout=30)
    resp = c.get("/health")
    assert resp.status_code == 200, f"Embedder not ready: {resp.text}"
    return c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "model" in body


def test_embed_text_returns_vector(client):
    r = client.post("/embed/text", json={"text": "a dog running in a park"})
    assert r.status_code == 200
    emb = r.json()["embedding"]
    assert isinstance(emb, list)
    assert len(emb) > 0


def test_embed_text_is_normalized(client):
    r = client.post("/embed/text", json={"text": "hello world"})
    emb = r.json()["embedding"]
    norm = math.sqrt(sum(x * x for x in emb))
    assert abs(norm - 1.0) < 1e-4, f"Expected unit vector, norm={norm}"


def test_embed_text_similarity_ordering(client):
    """Semantically close texts should score higher than unrelated ones."""
    r_dog = client.post("/embed/text", json={"text": "a dog"})
    r_puppy = client.post("/embed/text", json={"text": "a puppy"})
    r_car = client.post("/embed/text", json={"text": "a racing car"})

    emb_dog = r_dog.json()["embedding"]
    emb_puppy = r_puppy.json()["embedding"]
    emb_car = r_car.json()["embedding"]

    sim_dog_puppy = cosine(emb_dog, emb_puppy)
    sim_dog_car = cosine(emb_dog, emb_car)
    assert sim_dog_puppy > sim_dog_car, (
        f"Expected dog↔puppy ({sim_dog_puppy:.3f}) > dog↔car ({sim_dog_car:.3f})"
    )


def test_embed_image_returns_vector(client):
    r = client.post("/embed/image", json={"image": dummy_b64_image()})
    assert r.status_code == 200
    emb = r.json()["embedding"]
    assert isinstance(emb, list)
    assert len(emb) > 0


def test_embed_image_is_normalized(client):
    r = client.post("/embed/image", json={"image": dummy_b64_image()})
    emb = r.json()["embedding"]
    norm = math.sqrt(sum(x * x for x in emb))
    assert abs(norm - 1.0) < 1e-4


def test_embed_images_batch(client):
    images = [dummy_b64_image((255, 0, 0)), dummy_b64_image((0, 255, 0))]
    r = client.post("/embed/images", json={"images": images})
    assert r.status_code == 200
    embeddings = r.json()["embeddings"]
    assert len(embeddings) == 2
    for emb in embeddings:
        norm = math.sqrt(sum(x * x for x in emb))
        assert abs(norm - 1.0) < 1e-4


def test_embed_images_empty_list(client):
    r = client.post("/embed/images", json={"images": []})
    assert r.status_code == 400


def test_embed_text_consistent(client):
    """Same text twice should yield identical vectors."""
    text = "consistency check"
    e1 = client.post("/embed/text", json={"text": text}).json()["embedding"]
    e2 = client.post("/embed/text", json={"text": text}).json()["embedding"]
    assert e1 == e2


def test_embed_image_bad_b64(client):
    r = client.post("/embed/image", json={"image": "not-valid-base64!!!"})
    assert r.status_code == 400
