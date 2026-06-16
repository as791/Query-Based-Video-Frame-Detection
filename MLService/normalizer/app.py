import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import boto3
import redis
from botocore.config import Config

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
GROUP = "cg-normalizer"
CONSUMER = f"normalizer-{uuid.uuid4().hex[:8]}"
PENDING_IDLE_MS = 60_000


class FfmpegError(RuntimeError):
    def __init__(self, message, stderr=""):
        super().__init__(message)
        self.stderr = stderr


def s3_client():
    return boto3.client("s3", endpoint_url=S3_ENDPOINT, region_name=AWS_REGION,
                        config=S3_CLIENT_CONFIG)


def redis_client():
    return redis.from_url(REDIS_URI, decode_responses=True)


def emit_status(r, video_id, stage, **fields):
    payload = {"stage": stage}
    payload.update({k: str(v) for k, v in fields.items() if v is not None})
    r.xadd(f"ui.status.{video_id}", payload)


def tail(text, limit=1800):
    text = text or ""
    return text[-limit:]


def run_ffmpeg(cmd):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise FfmpegError(f"ffmpeg exited {proc.returncode}", tail(proc.stderr))


def parse_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def ffprobe_video(src):
    try:
        proc = subprocess.run([
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,pix_fmt,width,height,avg_frame_rate,r_frame_rate",
            "-of", "json",
            src,
        ], capture_output=True, text=True, check=True, timeout=20)
        raw = json.loads(proc.stdout or "{}")
        stream = (raw.get("streams") or [{}])[0]
        rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or ""
        fps = 0.0
        if "/" in rate:
            numerator, denominator = rate.split("/", 1)
            fps = parse_float(numerator) / (parse_float(denominator, 1.0) or 1.0)
        else:
            fps = parse_float(rate)
        return {
            "codec": str(stream.get("codec_name") or ""),
            "pix_fmt": str(stream.get("pix_fmt") or ""),
            "width": int(parse_float(stream.get("width"))),
            "height": int(parse_float(stream.get("height"))),
            "fps": fps,
        }
    except Exception as e:
        print(f"[normalizer] ffprobe fast-copy check failed: {e}", flush=True)
        return {}


def already_normalized(src):
    meta = ffprobe_video(src)
    return (
        meta.get("codec") == "h264"
        and meta.get("pix_fmt") == "yuv420p"
        and meta.get("width") == 640
        and meta.get("height") == 360
        and abs(float(meta.get("fps") or 0) - 25.0) <= 0.75
    )


def normalize_chunk(src, dst):
    if already_normalized(src):
        shutil.copyfile(src, dst)
        return

    primary = [
        "ffmpeg", "-y", "-i", src,
        "-vf", "scale=640:360:force_original_aspect_ratio=decrease,pad=640:360:(ow-iw)/2:(oh-ih)/2",
        "-r", "25",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-an",
        dst,
    ]
    fallback = [
        "ffmpeg", "-y", "-fflags", "+genpts+discardcorrupt", "-err_detect", "ignore_err", "-i", src,
        "-vf", "setpts=PTS-STARTPTS,scale=640:360:force_original_aspect_ratio=decrease,pad=640:360:(ow-iw)/2:(oh-ih)/2",
        "-r", "25",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "25",
        "-pix_fmt", "yuv420p",
        "-an",
        "-movflags", "+faststart",
        dst,
    ]
    try:
        run_ffmpeg(primary)
    except FfmpegError as first:
        try:
            run_ffmpeg(fallback)
        except FfmpegError as second:
            raise FfmpegError("normalization failed after retry", second.stderr or first.stderr) from second


def extract_thumbnail(src, dst):
    run_ffmpeg([
        "ffmpeg", "-y", "-i", src,
        "-vframes", "1", "-q:v", "2",
        dst,
    ])


def process(msg_id, data, s3, r):
    event_type = data.get("type")
    if event_type == "video.chunked":
        r.xack(STREAM, GROUP, msg_id)
        return
    if event_type == "video.chunks_ready":
        r.xack(STREAM, GROUP, msg_id)
        return
    if event_type != "chunk.created":
        r.xack(STREAM, GROUP, msg_id)
        return

    video_id = data["video_id"]
    user_id = data["user_id"]
    tenant_id = data.get("tenant_id", "default")
    profile = data.get("profile", "general")
    chunk_id = data["chunk_id"]
    chunk_index = data.get("chunk_index", "")
    chunk_count = data.get("chunk_count", "")
    t_start_ms = data["t_start_ms"]
    t_end_ms = data["t_end_ms"]
    s3_chunk_key = data.get("s3_chunk_path") or data.get("s3_path")
    source_file = data.get("source_file", "video.mp4")
    video_metadata_json = data.get("video_metadata_json", "{}")

    emit_status(
        r, video_id, "normalizing",
        chunk_id=chunk_id, chunk_index=chunk_index, chunk_count=chunk_count,
        t_start_ms=t_start_ms, t_end_ms=t_end_ms,
    )

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            src = str(Path(tmpdir) / "chunk.mp4")
            dst = str(Path(tmpdir) / "processed.mp4")
            thumb = str(Path(tmpdir) / "thumb.jpg")

            s3.download_file(S3_BUCKET, s3_chunk_key, src)
            normalize_chunk(src, dst)
            extract_thumbnail(dst, thumb)

            s3_processed_key = f"processed/{tenant_id}/{user_id}/{video_id}/{chunk_id}.mp4"
            s3_thumb_key = f"thumbs/{tenant_id}/{user_id}/{video_id}/{chunk_id}.jpg"
            s3.upload_file(dst, S3_BUCKET, s3_processed_key)
            s3.upload_file(thumb, S3_BUCKET, s3_thumb_key)

        r.xadd(STREAM, {
            "type": "chunk.normalized",
            "video_id": video_id,
            "user_id": user_id,
            "tenant_id": tenant_id,
            "profile": profile,
            "source_file": source_file,
            "video_metadata_json": video_metadata_json,
            "chunk_id": chunk_id,
            "chunk_index": str(chunk_index),
            "chunk_count": str(chunk_count),
            "t_start_ms": str(t_start_ms),
            "t_end_ms": str(t_end_ms),
            "s3_processed_path": s3_processed_key,
            "s3_thumb_path": s3_thumb_key,
        })
        emit_status(
            r, video_id, "normalized",
            chunk_id=chunk_id, chunk_index=chunk_index, chunk_count=chunk_count,
            t_start_ms=t_start_ms, t_end_ms=t_end_ms,
        )
        print(f"[normalizer] {video_id}/{chunk_id}: normalized")
    except Exception as e:
        error = getattr(e, "stderr", "") or str(e)
        r.xadd(STREAM, {
            "type": "chunk.failed",
            "video_id": video_id,
            "user_id": user_id,
            "tenant_id": tenant_id,
            "profile": profile,
            "chunk_id": chunk_id,
            "chunk_index": str(chunk_index),
            "chunk_count": str(chunk_count),
            "stage": "normalizer",
            "error": tail(error),
        })
        emit_status(
            r, video_id, "failed",
            stage_detail="normalizer", chunk_id=chunk_id, chunk_index=chunk_index,
            chunk_count=chunk_count, error=tail(error),
        )
        print(f"[normalizer] {video_id}/{chunk_id}: failed {tail(error, 300)}")

    r.xack(STREAM, GROUP, msg_id)


def claim_stale(r):
    try:
        claimed = r.xautoclaim(STREAM, GROUP, CONSUMER, min_idle_time=PENDING_IDLE_MS, start_id="0-0", count=5)
        return claimed[1] if isinstance(claimed, tuple) and len(claimed) > 1 else []
    except Exception:
        return []


def main():
    r = redis_client()
    s3 = s3_client()
    try:
        r.xgroup_create(STREAM, GROUP, id="0", mkstream=True)
    except redis.exceptions.ResponseError:
        pass

    print(f"[normalizer] listening as {CONSUMER}")
    while True:
        for msg_id, data in claim_stale(r):
            try:
                process(msg_id, data, s3, r)
            except Exception as e:
                print(f"[normalizer] stale error on {msg_id}: {e}")
                r.xadd(STREAM, {"type": "chunk.failed", "video_id": data.get("video_id", ""), "chunk_id": data.get("chunk_id", ""), "stage": "normalizer", "error": str(e)})
                emit_status(r, data.get("video_id", ""), "failed", stage_detail="normalizer", chunk_id=data.get("chunk_id", ""), error=str(e))
                r.xack(STREAM, GROUP, msg_id)

        messages = r.xreadgroup(GROUP, CONSUMER, {STREAM: ">"}, count=1, block=5000)
        if not messages:
            continue
        for _, entries in messages:
            for msg_id, data in entries:
                try:
                    process(msg_id, data, s3, r)
                except Exception as e:
                    print(f"[normalizer] error on {msg_id}: {e}")
                    r.xadd(STREAM, {"type": "chunk.failed", "video_id": data.get("video_id", ""), "chunk_id": data.get("chunk_id", ""), "stage": "normalizer", "error": str(e)})
                    emit_status(r, data.get("video_id", ""), "failed", stage_detail="normalizer", chunk_id=data.get("chunk_id", ""), error=str(e))
                    r.xack(STREAM, GROUP, msg_id)


if __name__ == "__main__":
    time.sleep(5)
    main()
