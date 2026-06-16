"""
Pipeline integration test: uploads a tiny real video to MinIO, fires
video.uploaded into Redis, then polls until video.indexed appears.

Requirements:
  pip install pytest redis boto3 httpx qdrant-client

Run:
  pytest tests/test_pipeline.py -v -s

The test uses the same env vars as the compose stack.
"""
import io
import os
import struct
import time
import uuid

import boto3
import httpx
import pytest
import redis
from qdrant_client import QdrantClient

# ── Config ──────────────────────────────────────────────────────────────────

REDIS_URI   = os.getenv("REDIS_URI",        "redis://localhost:6379")
S3_ENDPOINT = os.getenv("AWS_S3_ENDPOINT",  "http://localhost:9000")
S3_BUCKET   = os.getenv("MINIO_BUCKET",     "video-vault")
AWS_KEY     = os.getenv("AWS_ACCESS_KEY_ID",     "minioadmin")
AWS_SECRET  = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
QDRANT_URL  = os.getenv("QDRANT_URL",       "http://localhost:6333")
EMBEDDER    = os.getenv("EMBEDDER_URL",     "http://localhost:8002")

STREAM      = "pipeline.events"
TIMEOUT_S   = 120   # max seconds to wait for indexing


def minimal_mp4() -> bytes:
    """Return the smallest valid MP4 that ffmpeg can process (~1 second, 1x1 px)."""
    # ftyp box
    ftyp = b'\x00\x00\x00\x1cftypisom\x00\x00\x02\x00isomiso2avc1mp41'
    # mdat box (empty)
    mdat = b'\x00\x00\x00\x08mdat'
    # moov box (minimal, single black frame)
    moov = (
        b'\x00\x00\x00\x6cmoov'
        b'\x00\x00\x00\x6cmvhd\x00\x00\x00\x00'
        b'\x00\x00\x00\x00\x00\x00\x00\x00'
        b'\x00\x00\x03\xe8'         # timescale 1000
        b'\x00\x00\x03\xe8'         # duration  1000 ms
        + b'\x00' * 76
    )
    return ftyp + mdat


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def r():
    client = redis.from_url(REDIS_URI, decode_responses=True)
    client.ping()
    return client


@pytest.fixture(scope="session")
def s3():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=AWS_KEY,
        aws_secret_access_key=AWS_SECRET,
        region_name="us-east-1",
    )


@pytest.fixture(scope="session")
def qdrant():
    host, port = QDRANT_URL.replace("http://", "").split(":")
    return QdrantClient(host=host, port=int(port))


@pytest.fixture(scope="session")
def embedder():
    c = httpx.Client(base_url=EMBEDDER, timeout=30)
    resp = c.get("/health")
    assert resp.status_code == 200, "Embedder not ready"
    return c


# ── Tests ────────────────────────────────────────────────────────────────────

def test_redis_reachable(r):
    assert r.ping()


def test_minio_reachable(s3):
    buckets = [b["Name"] for b in s3.list_buckets().get("Buckets", [])]
    assert S3_BUCKET in buckets, f"Bucket '{S3_BUCKET}' not found; got {buckets}"


def test_qdrant_collections(qdrant):
    names = [c.name for c in qdrant.get_collections().collections]
    assert "chunks" in names and "frames" in names, (
        f"Expected 'chunks' and 'frames' collections, got {names}"
    )


def test_embedder_health(embedder):
    r = embedder.get("/health")
    assert r.status_code == 200


def test_full_pipeline(r, s3, qdrant, embedder):
    """
    End-to-end: upload video → fire event → wait for indexed → search.
    """
    video_id = str(uuid.uuid4())
    user_id  = "test-user-integration"
    s3_key   = f"raw/default/{user_id}/{video_id}/original.mp4"

    # 1. Upload a tiny MP4 to MinIO
    s3.upload_fileobj(io.BytesIO(minimal_mp4()), S3_BUCKET, s3_key,
                      ExtraArgs={"ContentType": "video/mp4"})
    print(f"\n[pipeline] uploaded {s3_key}")

    # 2. Publish video.uploaded into the pipeline stream
    r.xadd(STREAM, {
        "type":       "video.uploaded",
        "video_id":   video_id,
        "user_id":    user_id,
        "tenant_id":  "default",
        "profile":    "general",
        "s3_raw_path": s3_key,
    })
    print(f"[pipeline] published video.uploaded for {video_id}")

    # 3. Poll ui.status.<videoId> until 'indexed' or timeout
    status_stream = f"ui.status.{video_id}"
    deadline = time.time() + TIMEOUT_S
    last_id  = "0"
    stages_seen = []

    while time.time() < deadline:
        msgs = r.xread({status_stream: last_id}, count=10, block=2000)
        if msgs:
            for _, entries in msgs:
                for msg_id, data in entries:
                    last_id = msg_id
                    stage = data.get("stage", "")
                    stages_seen.append(stage)
                    print(f"[pipeline] stage={stage} data={data}")
                    if stage == "indexed":
                        break
            if "indexed" in stages_seen:
                break

    assert "indexed" in stages_seen, (
        f"Pipeline did not reach 'indexed' within {TIMEOUT_S}s. Stages seen: {stages_seen}"
    )

    # 4. Verify frames are in Qdrant scoped to this user
    resp = embedder.post("/embed/text", json={"text": "test query"})
    assert resp.status_code == 200
    query_vector = resp.json()["embedding"]

    results = qdrant.search(
        collection_name="frames",
        query_vector=query_vector,
        query_filter={
            "must": [
                {"key": "user_id",  "match": {"value": user_id}},
                {"key": "video_id", "match": {"value": video_id}},
            ]
        },
        limit=5,
    )
    assert len(results) > 0, "No frames indexed in Qdrant for this video"
    print(f"[pipeline] {len(results)} frames found in Qdrant")
