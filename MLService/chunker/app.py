import json
import os
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import boto3
import redis
from botocore.config import Config
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

# boto3 default has *no* read timeout — a hung TCP connection blocks forever.
# Set hard timeouts and retries so a stalled download self-heals.
S3_CLIENT_CONFIG = Config(
    connect_timeout=10,
    read_timeout=60,
    retries={"max_attempts": 5, "mode": "standard"},
    max_pool_connections=10,
)

REDIS_URI = os.environ.get("REDIS_URI", "redis://redis:6379")
S3_ENDPOINT = os.environ.get("AWS_S3_ENDPOINT")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
S3_BUCKET = os.environ.get("S3_BUCKET", os.environ.get("MINIO_BUCKET", "video-vault"))
STREAM = "pipeline.events"
GROUP = "cg-chunker"
CONSUMER = f"chunker-{uuid.uuid4().hex[:8]}"
CHUNK_MAX_SEC = 30
CCTV_CHUNK_SEC = 15
MIN_CHUNK_SEC = 1.0
PENDING_IDLE_MS = 60_000
FAST_FIXED_WINDOW_PIXELS = int(os.environ.get("FAST_FIXED_WINDOW_PIXELS", str(1920 * 1080)))
FAST_FIXED_WINDOW_DURATION_MS = int(os.environ.get("FAST_FIXED_WINDOW_DURATION_MS", "180000"))


def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        region_name=AWS_REGION,
        config=S3_CLIENT_CONFIG,
    )


def redis_client():
    return redis.from_url(REDIS_URI, decode_responses=True)


def download_from_s3(s3, bucket, key, dest):
    s3.download_file(bucket, key, dest)


def upload_to_s3(s3, local_path, bucket, key):
    s3.upload_file(local_path, bucket, key)


def emit_status(r, video_id, stage, **fields):
    payload = {"stage": stage}
    payload.update({k: str(v) for k, v in fields.items() if v is not None})
    r.xadd(f"ui.status.{video_id}", payload)


def ffprobe_duration(video_path):
    result_proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True, check=True,
    )
    return float(result_proc.stdout.strip() or "0")


def parse_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def ffprobe_metadata(video_path):
    try:
        proc = subprocess.run([
            "ffprobe", "-v", "error",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path,
        ], capture_output=True, text=True, check=True, timeout=30)
        raw = json.loads(proc.stdout or "{}")
        fmt = raw.get("format", {}) or {}
        video_stream = next((s for s in raw.get("streams", []) if s.get("codec_type") == "video"), {})
        rate = video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate") or ""
        fps = 0.0
        if "/" in rate:
            numerator, denominator = rate.split("/", 1)
            fps = parse_float(numerator) / (parse_float(denominator, 1.0) or 1.0)
        else:
            fps = parse_float(rate)
        duration = parse_float(video_stream.get("duration")) or parse_float(fmt.get("duration"))
        return {
            "duration_ms": int(duration * 1000),
            "width": int(parse_float(video_stream.get("width"))),
            "height": int(parse_float(video_stream.get("height"))),
            "fps": round(fps, 3),
            "codec": str(video_stream.get("codec_name") or ""),
            "format": str(fmt.get("format_name") or ""),
        }
    except Exception as e:
        print(f"[chunker] ffprobe metadata failed: {e}", flush=True)
        return {"duration_ms": 0, "width": 0, "height": 0, "fps": 0, "codec": "", "format": ""}


def normalize_windows(windows, duration):
    cleaned = []
    for start, end in windows:
        start_s = max(float(start or 0), 0.0)
        end_s = duration if end is None else min(float(end), duration)
        if end_s <= start_s:
            continue
        if cleaned and end_s - start_s < MIN_CHUNK_SEC:
            prev_start, _ = cleaned[-1]
            cleaned[-1] = (prev_start, end_s)
        else:
            cleaned.append((start_s, end_s))

    if len(cleaned) > 1 and cleaned[0][1] - cleaned[0][0] < MIN_CHUNK_SEC:
        _, first_end = cleaned.pop(0)
        next_start, next_end = cleaned[0]
        cleaned[0] = (min(next_start, first_end), next_end)

    capped = []
    for start_s, end_s in cleaned or [(0.0, duration)]:
        while end_s - start_s > CHUNK_MAX_SEC:
            capped.append((start_s, start_s + CHUNK_MAX_SEC))
            start_s += CHUNK_MAX_SEC
        capped.append((start_s, end_s))
    return capped


def detect_scenes(video_path):
    """Return list of (start_sec, end_sec) capped at CHUNK_MAX_SEC."""
    duration = ffprobe_duration(video_path)
    video = open_video(video_path)
    manager = SceneManager()
    manager.add_detector(ContentDetector(threshold=27.0))
    manager.detect_scenes(video)
    scenes = manager.get_scene_list()
    windows = [(start.get_seconds(), end.get_seconds()) for start, end in scenes]
    return normalize_windows(windows or [(0, duration)], duration)


def fixed_windows(video_path, window_sec):
    """Return list of (start_sec, end_sec) for fixed-size windows."""
    duration = ffprobe_duration(video_path)
    windows = []
    t = 0.0
    while t < duration:
        windows.append((t, min(t + window_sec, duration)))
        t += window_sec
    return normalize_windows(windows or [(0, duration)], duration)


def split_chunk(video_path, start_sec, end_sec, out_path):
    duration = max(end_sec - start_sec, MIN_CHUNK_SEC)
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_sec:.3f}",
        "-i", video_path,
        "-t", f"{duration:.3f}",
        "-map", "0:v:0",
        # Emit the canonical normalized shape so the normalizer can fast-copy.
        "-vf", "setpts=PTS-STARTPTS,scale=640:360:force_original_aspect_ratio=decrease,pad=640:360:(ow-iw)/2:(oh-ih)/2",
        "-r", "25",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "26",
        "-pix_fmt", "yuv420p",
        "-threads", "2",
        "-an",
        "-movflags", "+faststart",
        out_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=180)


def process(msg_id, data, s3, r):
    if data.get("type") != "video.uploaded":
        r.xack(STREAM, GROUP, msg_id)
        return

    video_id = data["video_id"]
    user_id = data["user_id"]
    tenant_id = data.get("tenant_id", "default")
    profile = data.get("profile", "general")
    s3_raw_path = data["s3_raw_path"]  # e.g. raw/default/{user_id}/{video_id}/original.mp4
    source_file = data.get("source_file", "video.mp4")

    with tempfile.TemporaryDirectory() as tmpdir:
        src = str(Path(tmpdir) / "original.mp4")
        # s3_raw_path is a bucket-relative key
        bucket_key = s3_raw_path.lstrip("/")
        download_from_s3(s3, S3_BUCKET, bucket_key, src)
        video_metadata = ffprobe_metadata(src)
        video_metadata["source_file"] = source_file
        video_metadata["profile"] = profile
        video_metadata_json = json.dumps(video_metadata)

        pixels = int(video_metadata.get("width", 0)) * int(video_metadata.get("height", 0))
        duration_ms = int(video_metadata.get("duration_ms", 0))
        use_fast_fixed_windows = (
            profile != "cctv"
            and (pixels >= FAST_FIXED_WINDOW_PIXELS or duration_ms >= FAST_FIXED_WINDOW_DURATION_MS)
        )

        if profile == "cctv":
            windows = fixed_windows(src, CCTV_CHUNK_SEC)
        elif use_fast_fixed_windows:
            print(
                f"[chunker] {video_id}: using fixed windows for large source "
                f"({video_metadata.get('width')}x{video_metadata.get('height')}, {duration_ms} ms)",
                flush=True,
            )
            windows = fixed_windows(src, CHUNK_MAX_SEC)
        else:
            windows = detect_scenes(src)

        chunks = []
        emit_status(r, video_id, "chunked", chunk_count=len(windows), indexed_count=0, failed_count=0)
        r.xadd(STREAM, {
            "type": "video.chunked",
            "video_id": video_id,
            "user_id": user_id,
            "tenant_id": tenant_id,
            "profile": profile,
            "chunk_count": str(len(windows)),
            "source_file": source_file,
            "video_metadata_json": video_metadata_json,
        })

        for index, (start_sec, end_sec) in enumerate(windows):
            chunk_id = str(uuid.uuid4())
            chunk_file = str(Path(tmpdir) / f"{chunk_id}.mp4")
            print(f"[chunker] {video_id}/{chunk_id}: splitting {start_sec:.3f}-{end_sec:.3f}", flush=True)
            split_chunk(src, start_sec, end_sec, chunk_file)
            s3_key = f"chunks/{tenant_id}/{user_id}/{video_id}/{chunk_id}.mp4"
            print(f"[chunker] {video_id}/{chunk_id}: uploading {s3_key}", flush=True)
            upload_to_s3(s3, chunk_file, S3_BUCKET, s3_key)
            chunks.append({
                "chunk_id": chunk_id,
                "t_start_ms": int(start_sec * 1000),
                "t_end_ms": int((end_sec or 0) * 1000),
                "s3_path": s3_key,
            })

            r.xadd(STREAM, {
                "type": "chunk.created",
                "video_id": video_id,
                "user_id": user_id,
                "tenant_id": tenant_id,
                "profile": profile,
                "source_file": source_file,
                "video_metadata_json": video_metadata_json,
                "chunk_id": chunk_id,
                "chunk_index": str(index),
                "chunk_count": str(len(windows)),
                "t_start_ms": str(int(start_sec * 1000)),
                "t_end_ms": str(int(end_sec * 1000)),
                "s3_chunk_path": s3_key,
                "attempt": "0",
            })
            emit_status(
                r, video_id, "chunk.created",
                chunk_id=chunk_id, chunk_index=index, chunk_count=len(windows),
                t_start_ms=int(start_sec * 1000), t_end_ms=int(end_sec * 1000),
            )

        r.xadd(STREAM, {
            "type": "video.chunks_ready",
            "video_id": video_id,
            "user_id": user_id,
            "tenant_id": tenant_id,
            "profile": profile,
            "chunk_count": str(len(chunks)),
            "chunks_json": json.dumps(chunks),
            "source_file": source_file,
            "video_metadata_json": video_metadata_json,
        })

    r.xack(STREAM, GROUP, msg_id)
    print(f"[chunker] {video_id}: {len(chunks)} chunks", flush=True)


def claim_stale(r):
    try:
        claimed = r.xautoclaim(STREAM, GROUP, CONSUMER, min_idle_time=PENDING_IDLE_MS, start_id="0-0", count=5)
        return claimed[1] if isinstance(claimed, tuple) and len(claimed) > 1 else []
    except Exception:
        return []


def main():
    r = redis_client()
    s3 = s3_client()
    # Ensure consumer group exists
    try:
        r.xgroup_create(STREAM, GROUP, id="0", mkstream=True)
    except redis.exceptions.ResponseError:
        pass  # already exists

    print(f"[chunker] listening as {CONSUMER}")
    while True:
        for msg_id, data in claim_stale(r):
            try:
                process(msg_id, data, s3, r)
            except Exception as e:
                print(f"[chunker] stale error on {msg_id}: {e}", flush=True)
                r.xadd(STREAM, {"type": "video.failed", "video_id": data.get("video_id", ""), "stage": "chunker", "error": str(e)})
                emit_status(r, data.get("video_id", ""), "failed", stage_detail="chunker", error=str(e))
                r.xack(STREAM, GROUP, msg_id)

        messages = r.xreadgroup(GROUP, CONSUMER, {STREAM: ">"}, count=1, block=5000)
        if not messages:
            continue
        for stream_name, entries in messages:
            for msg_id, data in entries:
                try:
                    process(msg_id, data, s3, r)
                except Exception as e:
                    print(f"[chunker] error on {msg_id}: {e}", flush=True)
                    r.xadd(STREAM, {"type": "video.failed", "video_id": data.get("video_id", ""), "stage": "chunker", "error": str(e)})
                    emit_status(r, data.get("video_id", ""), "failed", stage_detail="chunker", error=str(e))
                    r.xack(STREAM, GROUP, msg_id)


if __name__ == "__main__":
    time.sleep(5)  # let Redis settle
    main()
