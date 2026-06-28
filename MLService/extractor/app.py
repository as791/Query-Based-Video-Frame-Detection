import base64
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import time
import uuid
from io import BytesIO
from pathlib import Path

import boto3
import redis
import requests
from botocore.config import Config
from PIL import Image, ImageDraw
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct

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
EMBEDDER_URL = os.environ.get("EMBEDDER_URL", "http://embedder:8002")
VLM_URL = os.environ.get("VLM_URL", "http://host.docker.internal:11434")
VLM_MODEL = os.environ.get("VLM_MODEL", "qwen2.5vl:7b")
VLM_ANALYSIS_ENABLED = os.environ.get("VLM_ANALYSIS_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
ACTION_LABELS_KEY = os.environ.get("ACTION_LABELS_KEY", "taxonomy:action-labels")
ACTION_LABELS_REFRESH_SEC = float(os.environ.get("ACTION_LABELS_REFRESH_SEC", "30"))
ACTION_SCORING_ENABLED = os.environ.get("ACTION_SCORING_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
ACTION_SCORING_TOP_K = int(os.environ.get("ACTION_SCORING_TOP_K", "3"))
ACTION_SCORING_FRAME_TOP_K = int(os.environ.get("ACTION_SCORING_FRAME_TOP_K", "3"))
ACTION_SCORING_MIN_RAW_SCORE = float(os.environ.get("ACTION_SCORING_MIN_RAW_SCORE", "0.08"))
EXTRACTOR_BENCHMARK_RUN_FILTER = {
    item.strip()
    for item in os.environ.get("EXTRACTOR_BENCHMARK_RUN_FILTER", "").split(",")
    if item.strip()
}
ACTION_EMBEDDINGS_PROMPT_VERSION = "v2-ensemble"
FEW_SHOT_SCORING_ENABLED = os.environ.get("FEW_SHOT_SCORING_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
FEW_SHOT_MIN_SCORE = float(os.environ.get("FEW_SHOT_MIN_SCORE", "0.78"))
FEW_SHOT_MAX_EXAMPLE_FRAMES = int(os.environ.get("FEW_SHOT_MAX_EXAMPLE_FRAMES", "240"))
VLM_FAILURE_THRESHOLD = int(os.environ.get("VLM_FAILURE_THRESHOLD", "3"))
VLM_FAILURE_WINDOW_SEC = int(os.environ.get("VLM_FAILURE_WINDOW_SEC", "120"))
VLM_FAILURE_BACKOFF_SEC = int(os.environ.get("VLM_FAILURE_BACKOFF_SEC", "0"))
VLM_NUM_PREDICT = int(os.environ.get("VLM_NUM_PREDICT", "320"))
VLM_CONTACT_SHEET_FRAMES = int(os.environ.get("VLM_CONTACT_SHEET_FRAMES", "6"))
QDRANT_HOST = os.environ.get("QDRANT_HOST", "qdrant")
STREAM = "pipeline.events"
GROUP = "cg-extractor"
CONSUMER = f"extractor-{uuid.uuid4().hex[:8]}"
FPS_GENERAL = 0.5   # 1 frame per 2s
FPS_CCTV = 1.0      # 1 frame per 1s
VLM_TIMEOUT_SEC = float(os.environ.get("VLM_TIMEOUT_SEC", "8"))
PENDING_IDLE_MS = 60_000
DEFAULT_ACTION_TAXONOMY = [
    {"label": "abuse", "description": "a person being abused or attacked"},
    {"label": "arrest", "description": "a person being arrested"},
    {"label": "arson", "description": "a person starting a fire"},
    {"label": "assault", "description": "a person assaulting another person"},
    {"label": "burglary", "description": "a person breaking into a building"},
    {"label": "explosion", "description": "an explosion or blast"},
    {"label": "fighting", "description": "people fighting"},
    {"label": "normal", "description": "normal CCTV footage with no suspicious action"},
    {"label": "roadaccidents", "description": "a road accident involving vehicles or pedestrians"},
    {"label": "robbery", "description": "a robbery or theft in progress"},
    {"label": "shooting", "description": "a person shooting a weapon"},
    {"label": "shoplifting", "description": "a person shoplifting in a store"},
    {"label": "stealing", "description": "a person stealing something"},
    {"label": "vandalism", "description": "a person vandalizing property"},
]
_ACTION_TAXONOMY = DEFAULT_ACTION_TAXONOMY
_ACTION_TAXONOMY_LOADED_AT = 0.0
_ACTION_EMBEDDINGS_SIGNATURE = ""
_ACTION_EMBEDDINGS = []
ACTION_EMBEDDINGS_CACHE_PREFIX = "extractor:action_embeddings:"
VLM_FAILURES_KEY = "extractor:vlm_failures"
VLM_BACKOFF_KEY = "extractor:vlm_backoff_until"
STOP_WORDS = {
    "the", "and", "for", "with", "from", "that", "this", "there", "their", "into",
    "near", "over", "under", "while", "video", "frame", "scene", "shows", "showing",
    "chunk", "segment", "person", "people", "image", "camera", "view",
}


def s3_client():
    return boto3.client("s3", endpoint_url=S3_ENDPOINT, region_name=AWS_REGION,
                        config=S3_CLIENT_CONFIG)


def redis_client():
    return redis.from_url(REDIS_URI, decode_responses=True)


def qdrant_client():
    return QdrantClient(host=QDRANT_HOST, port=6333)


def vlm_backoff_active(r):
    if VLM_FAILURE_BACKOFF_SEC <= 0:
        return False
    try:
        backoff_until = float(r.get(VLM_BACKOFF_KEY) or "0")
        return backoff_until > time.time()
    except Exception:
        return False


def note_vlm_failure(r, reason):
    if VLM_FAILURE_BACKOFF_SEC <= 0:
        return
    try:
        failures = r.incr(VLM_FAILURES_KEY)
        if failures == 1:
            r.expire(VLM_FAILURES_KEY, VLM_FAILURE_WINDOW_SEC)
        if failures >= VLM_FAILURE_THRESHOLD:
            backoff_until = time.time() + VLM_FAILURE_BACKOFF_SEC
            r.setex(VLM_BACKOFF_KEY, VLM_FAILURE_BACKOFF_SEC, str(backoff_until))
            print(f"[extractor] VLM cooldown {VLM_FAILURE_BACKOFF_SEC}s after {failures} failures: {reason}")
    except Exception as e:
        print(f"[extractor] could not record VLM failure: {e}")


def note_vlm_success(r):
    if VLM_FAILURE_BACKOFF_SEC <= 0:
        return
    try:
        r.delete(VLM_FAILURES_KEY)
        r.delete(VLM_BACKOFF_KEY)
    except Exception:
        pass


def emit_status(r, video_id, stage, **fields):
    payload = {"stage": stage}
    payload.update({k: str(v) for k, v in fields.items() if v is not None})
    r.xadd(f"ui.status.{video_id}", payload)


def tail(text, limit=1800):
    text = text or ""
    return text[-limit:]


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


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
        fps = 0.0
        rate = video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate") or ""
        if "/" in rate:
            numerator, denominator = rate.split("/", 1)
            denom = parse_float(denominator, 1.0) or 1.0
            fps = parse_float(numerator) / denom
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
        print(f"[extractor] ffprobe metadata failed: {e}")
        return {
            "duration_ms": 0,
            "width": 0,
            "height": 0,
            "fps": 0,
            "codec": "",
            "format": "",
        }


def keyword_tags(text, limit=10):
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", (text or "").lower())
    tags = []
    for word in words:
        if word in STOP_WORDS or word in tags:
            continue
        tags.append(word)
        if len(tags) >= limit:
            break
    return tags


def estimate_motion(frames):
    if len(frames) < 2:
        return 0.0, "unknown motion"
    diffs = []
    previous = None
    for _, _, _, img, _ in frames[:12]:
        gray = img.resize((64, 36)).convert("L")
        pixels = list(gray.getdata())
        if previous is not None:
            diff = sum(abs(a - b) for a, b in zip(previous, pixels)) / (255 * len(pixels))
            diffs.append(diff)
        previous = pixels
    score = clamp(sum(diffs) / len(diffs) if diffs else 0.0)
    if score >= 0.22:
        label = "high motion"
    elif score >= 0.1:
        label = "moderate motion"
    elif score >= 0.035:
        label = "low motion"
    else:
        label = "mostly static"
    return round(score, 4), label


def extract_json_object(text):
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]
    return json.loads(text)


def clean_list(value, limit=12):
    if isinstance(value, str):
        value = re.split(r"[,;\n]", value)
    if not isinstance(value, list):
        return []
    cleaned = []
    for item in value:
        item = str(item).strip().lower()
        if item and item not in cleaned:
            cleaned.append(item)
        if len(cleaned) >= limit:
            break
    return cleaned


def normalize_label(value):
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def normalize_taxonomy(value):
    if not isinstance(value, list):
        return DEFAULT_ACTION_TAXONOMY
    out = []
    seen = set()
    for item in value:
        if isinstance(item, dict):
            label = normalize_label(item.get("label"))
            description = str(item.get("description") or "").strip()
        else:
            label = normalize_label(item)
            description = ""
        if label and label not in seen:
            out.append({"label": label, "description": description})
            seen.add(label)
    return out or DEFAULT_ACTION_TAXONOMY


def load_action_taxonomy(r):
    global _ACTION_TAXONOMY, _ACTION_TAXONOMY_LOADED_AT
    now = time.time()
    if now - _ACTION_TAXONOMY_LOADED_AT < ACTION_LABELS_REFRESH_SEC:
        return _ACTION_TAXONOMY
    try:
        raw = r.get(ACTION_LABELS_KEY)
        _ACTION_TAXONOMY = normalize_taxonomy(json.loads(raw)) if raw else DEFAULT_ACTION_TAXONOMY
    except Exception as e:
        print(f"[extractor] action taxonomy fallback: {e}")
        _ACTION_TAXONOMY = DEFAULT_ACTION_TAXONOMY
    _ACTION_TAXONOMY_LOADED_AT = now
    return _ACTION_TAXONOMY


def action_label_names():
    return {item["label"] for item in _ACTION_TAXONOMY if item.get("label")}


def add_runtime_action_label(label, description=""):
    global _ACTION_TAXONOMY
    normalized = normalize_label(label)
    if not normalized or normalized in action_label_names():
        return
    _ACTION_TAXONOMY = list(_ACTION_TAXONOMY) + [{"label": normalized, "description": description}]


def clean_action_label(value):
    label = normalize_label(value)
    aliases = {
        "roadaccident": "roadaccidents",
        "roadaccidents": "roadaccidents",
        "accident": "roadaccidents",
        "vehicleaccident": "roadaccidents",
        "fight": "fighting",
        "theft": "stealing",
        "steal": "stealing",
        "shot": "shooting",
        "gunfire": "shooting",
    }
    label = aliases.get(label, label)
    return label if label in action_label_names() else ""


def clean_action_labels(value, limit=3):
    labels = []
    for item in clean_list(value, limit=limit * 2):
        label = clean_action_label(item)
        if label and label not in labels:
            labels.append(label)
        if len(labels) >= limit:
            break
    return labels


def normalize_action_scores(value):
    if not isinstance(value, dict):
        return {}
    scores = {}
    for raw_label, raw_score in value.items():
        label = clean_action_label(raw_label)
        if not label:
            continue
        scores[label] = round(clamp(parse_float(raw_score)), 3)
    return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))


def word_count(value):
    return len(re.findall(r"[a-zA-Z0-9]+", str(value or "")))


def descriptive_fallback_caption(fallback):
    scene = str(fallback.get("scene") or "video").strip()
    motion = str(fallback.get("motion") or "unknown motion").strip()
    return (
        f"The clip shows a {scene} scene with {motion} across the sampled frames. "
        "The specific people, objects, environment, and activity are not confidently identified by the visual model."
    )


def normalize_caption_text(caption, fallback):
    caption = re.sub(r"\s+", " ", str(caption or "").strip())
    too_generic = normalize_label(caption) in {"normal", "activity", "motion", "scene", "video", "unknown"}
    if word_count(caption) < 8 or too_generic:
        return descriptive_fallback_caption(fallback)
    return caption[:700]


def normalize_phrase(value, fallback_value, min_words=2):
    text = re.sub(r"\s+", " ", str(value or "").strip())
    if word_count(text) < min_words:
        text = re.sub(r"\s+", " ", str(fallback_value or "").strip())
    return text[:160]


def normalize_analysis(value, fallback):
    if not isinstance(value, dict):
        value = {}
    caption = normalize_caption_text(value.get("caption") or fallback.get("caption"), fallback)
    main_activity = normalize_phrase(value.get("main_activity"), fallback.get("main_activity"), min_words=2)
    scene = normalize_phrase(value.get("scene"), fallback.get("scene"), min_words=2)
    motion = normalize_phrase(value.get("motion"), fallback.get("motion"), min_words=2)
    tags = clean_list(value.get("tags")) or fallback.get("tags", [])
    objects = clean_list(value.get("objects")) or fallback.get("objects", [])
    confidence = clamp(parse_float(value.get("confidence"), fallback.get("confidence", 0.35)))
    action_scores = normalize_action_scores(value.get("action_scores")) or fallback.get("action_scores", {})
    action_label_limit = max(1, ACTION_SCORING_TOP_K)
    requested_labels = clean_action_labels(value.get("action_labels"), limit=action_label_limit) or fallback.get("action_labels", [])
    action_labels = []
    for label, score in action_scores.items():
        if score >= 0.45 and label not in action_labels:
            action_labels.append(label)
        if len(action_labels) >= action_label_limit:
            break
    for label in clean_action_labels(requested_labels, limit=action_label_limit):
        if label not in action_labels:
            action_labels.append(label)
        if len(action_labels) >= action_label_limit:
            break
    action_confidence = clamp(parse_float(
        value.get("action_confidence"),
        fallback.get("action_confidence", max(action_scores.values(), default=0.0)),
    ))
    action_top = next(iter(action_scores.keys()), "") if action_scores else (action_labels[0] if action_labels else "")
    return {
        "caption": caption,
        "main_activity": main_activity,
        "tags": clean_list(tags),
        "objects": clean_list(objects),
        "scene": scene,
        "motion": motion,
        "confidence": round(confidence, 3),
        "action_labels": clean_action_labels(action_labels, limit=action_label_limit),
        "action_top": action_top,
        "action_scores": action_scores,
        "action_confidence": round(action_confidence, 3),
    }


def parse_metadata_json(value):
    try:
        parsed = json.loads(value or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def extract_frames(video_path, fps, start_ms):
    """Return list of (frame_id, absolute_t_ms, chunk_t_ms, PIL.Image)."""
    out_dir = Path(video_path).parent / "frames"
    out_dir.mkdir(exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"fps={fps}",
        "-q:v", "2",
        str(out_dir / "%05d.jpg"),
    ], check=True, capture_output=True)
    frames = []
    for i, f in enumerate(sorted(out_dir.glob("*.jpg"))):
        chunk_t_ms = int(i * (1000 / fps))
        t_ms = start_ms + chunk_t_ms
        img = Image.open(f).convert("RGB")
        frames.append((str(uuid.uuid4()), t_ms, chunk_t_ms, img, str(f)))
    return frames


def image_to_b64(img):
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def jpeg_b64_from_bytes(raw, max_size=384):
    img = Image.open(BytesIO(raw)).convert("RGB")
    img.thumbnail((max_size, max_size))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=82, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def contact_sheet_b64(frames, max_frames=6, cell_size=(256, 144)):
    """Build a small ordered frame strip so the VLM can infer action over time."""
    if not frames:
        return ""
    max_frames = max(1, min(max_frames, len(frames)))
    if len(frames) <= max_frames:
        selected = frames
    else:
        selected = []
        for i in range(max_frames):
            idx = round(i * (len(frames) - 1) / (max_frames - 1))
            selected.append(frames[idx])

    cell_w, cell_h = cell_size
    label_h = 18
    cols = min(3, len(selected))
    rows = math.ceil(len(selected) / cols)
    sheet = Image.new("RGB", (cols * cell_w, rows * (cell_h + label_h)), (8, 8, 10))
    draw = ImageDraw.Draw(sheet)
    for index, (_, t_ms, chunk_t_ms, img, _) in enumerate(selected):
        thumb = img.copy()
        thumb.thumbnail((cell_w, cell_h))
        x = (index % cols) * cell_w
        y = (index // cols) * (cell_h + label_h)
        paste_x = x + (cell_w - thumb.width) // 2
        sheet.paste(thumb, (paste_x, y))
        label = f"frame {index + 1}  t={chunk_t_ms / 1000:.1f}s"
        draw.rectangle((x, y + cell_h, x + cell_w, y + cell_h + label_h), fill=(18, 18, 22))
        draw.text((x + 6, y + cell_h + 3), label, fill=(235, 235, 235))

    buf = BytesIO()
    sheet.save(buf, format="JPEG", quality=84, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def embed_images(images):
    resp = requests.post(
        f"{EMBEDDER_URL}/embed/images",
        json={"images": [image_to_b64(img) for img in images]},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"]


def embed_text(text):
    resp = requests.post(
        f"{EMBEDDER_URL}/embed/text",
        json={"text": text},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def dot(a, b):
    return sum(float(x) * float(y) for x, y in zip(a, b))


def action_prompt(item):
    label = item.get("label", "")
    description = (item.get("description") or label).strip()
    return (
        "A CCTV security video frame where the main visible action is "
        f"{description}. Action label: {label}."
    )


def action_prompts(item):
    label = item.get("label", "")
    description = (item.get("description") or label).strip()
    return [
        action_prompt(item),
        f"A CCTV frame of {label}.",
        f"A person {label}.",
        f"The visible action is {label}.",
        f"A video frame showing {description}.",
        f"Security camera footage of {description}.",
    ]


def average_embedding(vectors):
    vectors = [v for v in vectors if v]
    if not vectors:
        return []
    length = len(vectors[0])
    averaged = [sum(vec[i] for vec in vectors) / len(vectors) for i in range(length)]
    norm = math.sqrt(sum(value * value for value in averaged)) or 1.0
    return [value / norm for value in averaged]


def action_taxonomy_signature(action_taxonomy):
    parts = [
        f"{item.get('label', '')}:{item.get('description', '')}"
        for item in action_taxonomy
        if item.get("label")
    ]
    return ACTION_EMBEDDINGS_PROMPT_VERSION + "|" + "|".join(parts)


def cached_action_text_embeddings(signature, r):
    if r is None or not signature:
        return []
    key = ACTION_EMBEDDINGS_CACHE_PREFIX + hashlib.sha1(signature.encode("utf-8")).hexdigest()
    try:
        raw = r.get(key)
        if raw:
            cached = json.loads(raw)
            if isinstance(cached, list):
                return cached
    except Exception as e:
        print(f"[extractor] action embedding cache read failed: {e}")
    return []


def store_action_text_embeddings(signature, r, embeddings):
    if r is None or not signature or not embeddings:
        return
    key = ACTION_EMBEDDINGS_CACHE_PREFIX + hashlib.sha1(signature.encode("utf-8")).hexdigest()
    try:
        r.setex(key, 3600, json.dumps(embeddings))
    except Exception as e:
        print(f"[extractor] action embedding cache write failed: {e}")


def action_text_embeddings(action_taxonomy, r=None):
    global _ACTION_EMBEDDINGS_SIGNATURE, _ACTION_EMBEDDINGS
    signature = action_taxonomy_signature(action_taxonomy)
    if signature and signature == _ACTION_EMBEDDINGS_SIGNATURE and _ACTION_EMBEDDINGS:
        return _ACTION_EMBEDDINGS

    cached = cached_action_text_embeddings(signature, r)
    if cached:
        _ACTION_EMBEDDINGS_SIGNATURE = signature
        _ACTION_EMBEDDINGS = cached
        return cached

    lock_key = ACTION_EMBEDDINGS_CACHE_PREFIX + hashlib.sha1(signature.encode("utf-8")).hexdigest() + ":lock" if signature else ""
    have_lock = False
    if r is not None and lock_key:
        try:
            have_lock = bool(r.set(lock_key, CONSUMER, nx=True, ex=120))
        except Exception:
            have_lock = False
        if not have_lock:
            deadline = time.time() + 45
            while time.time() < deadline:
                time.sleep(0.5)
                cached = cached_action_text_embeddings(signature, r)
                if cached:
                    _ACTION_EMBEDDINGS_SIGNATURE = signature
                    _ACTION_EMBEDDINGS = cached
                    return cached

    embeddings = []
    try:
        for item in action_taxonomy:
            label = item.get("label", "")
            if not label:
                continue
            try:
                vectors = [embed_text(prompt) for prompt in action_prompts(item)]
                embeddings.append({
                    "label": label,
                    "description": item.get("description", ""),
                    "embedding": average_embedding(vectors),
                })
            except Exception as e:
                print(f"[extractor] action text embedding failed for {label}: {e}")
        _ACTION_EMBEDDINGS_SIGNATURE = signature
        _ACTION_EMBEDDINGS = embeddings
        store_action_text_embeddings(signature, r, embeddings)
        return embeddings
    finally:
        if have_lock and r is not None and lock_key:
            try:
                r.delete(lock_key)
            except Exception:
                pass


def score_actions_from_embeddings(frame_embeddings, action_taxonomy, r=None):
    if not ACTION_SCORING_ENABLED or not frame_embeddings or not action_taxonomy:
        return {"action_labels": [], "action_top": "", "action_scores": {}, "action_confidence": 0.0}

    text_embeddings = action_text_embeddings(action_taxonomy, r)
    if not text_embeddings:
        return {"action_labels": [], "action_top": "", "action_scores": {}, "action_confidence": 0.0}

    frame_top_k = max(1, min(ACTION_SCORING_FRAME_TOP_K, len(frame_embeddings)))
    raw_scores = []
    for item in text_embeddings:
        similarities = sorted((dot(frame_emb, item["embedding"]) for frame_emb in frame_embeddings), reverse=True)
        score = sum(similarities[:frame_top_k]) / frame_top_k
        raw_scores.append((item["label"], score))

    raw_scores.sort(key=lambda item: item[1], reverse=True)
    if not raw_scores:
        return {"action_labels": [], "action_top": "", "action_scores": {}, "action_confidence": 0.0}

    top_k = max(1, min(ACTION_SCORING_TOP_K, len(raw_scores)))
    selected = raw_scores[:top_k]
    best = selected[0][1]
    if best < ACTION_SCORING_MIN_RAW_SCORE:
        return {"action_labels": [], "action_top": "", "action_scores": {}, "action_confidence": 0.0}
    worst = raw_scores[-1][1]
    spread = max(1e-6, best - worst)
    action_scores = {
        label: round(clamp((score - worst) / spread), 3)
        for label, score in selected
    }
    action_labels = [label for label, _ in selected]
    second = raw_scores[1][1] if len(raw_scores) > 1 else worst
    confidence = clamp(0.55 + ((best - second) * 12.0) + (max(0.0, best) * 0.35), 0.05, 0.95)
    return {
        "action_labels": clean_action_labels(action_labels, limit=top_k),
        "action_top": clean_action_label(action_labels[0]),
        "action_scores": normalize_action_scores(action_scores),
        "action_confidence": round(confidence, 3),
    }


def few_shot_action_analysis(qdrant, user_id, domain_id, current_video_id, frame_embeddings):
    if not FEW_SHOT_SCORING_ENABLED or not user_id or not frame_embeddings:
        return {"action_labels": [], "action_top": "", "action_scores": {}, "action_confidence": 0.0}
    try:
        points, _ = qdrant.scroll(
            collection_name="frames",
            scroll_filter=Filter(must=[
                FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                FieldCondition(key="domain_id", match=MatchValue(value=domain_id or "general")),
                FieldCondition(key="few_shot_example", match=MatchValue(value=True)),
            ]),
            limit=FEW_SHOT_MAX_EXAMPLE_FRAMES,
            with_vectors=True,
            with_payload=True,
        )
    except Exception as e:
        print(f"[extractor] few-shot prototype lookup failed: {e}")
        return {"action_labels": [], "action_top": "", "action_scores": {}, "action_confidence": 0.0}

    best_by_label = {}
    for point in points:
        payload = point.payload or {}
        if payload.get("video_id") == current_video_id:
            continue
        label = normalize_label(payload.get("few_shot_label"))
        if not label or not point.vector:
            continue
        add_runtime_action_label(label, f"user provided examples for {label}")
        best = max(dot(frame_emb, point.vector) for frame_emb in frame_embeddings)
        if best > best_by_label.get(label, -1.0):
            best_by_label[label] = best

    if not best_by_label:
        return {"action_labels": [], "action_top": "", "action_scores": {}, "action_confidence": 0.0}

    ranked = sorted(best_by_label.items(), key=lambda item: item[1], reverse=True)
    if ranked[0][1] < FEW_SHOT_MIN_SCORE:
        return {"action_labels": [], "action_top": "", "action_scores": {}, "action_confidence": 0.0}

    selected = ranked[:max(1, ACTION_SCORING_TOP_K)]
    scores = {
        label: round(clamp((score - FEW_SHOT_MIN_SCORE) / max(1e-6, 1.0 - FEW_SHOT_MIN_SCORE)), 3)
        for label, score in selected
    }
    labels = [label for label, _ in selected]
    return {
        "action_labels": labels,
        "action_top": labels[0],
        "action_scores": normalize_action_scores(scores),
        "action_confidence": round(clamp(ranked[0][1]), 3),
    }


def merge_action_evidence(vlm_analysis, embedding_analysis):
    vlm_scores = normalize_action_scores(vlm_analysis.get("action_scores", {}))
    embedding_scores = normalize_action_scores(embedding_analysis.get("action_scores", {}))
    merged_scores = dict(embedding_scores)
    for label, score in vlm_scores.items():
        # Require weak VLM-only labels to compete with embedding evidence instead of
        # automatically taking top position from prompt ordering.
        merged_scores[label] = round(max(score * 0.95, merged_scores.get(label, 0.0)), 3)
    merged_scores = dict(sorted(merged_scores.items(), key=lambda item: item[1], reverse=True))

    labels = []
    for source in (
        list(merged_scores.keys()),
        [embedding_analysis.get("action_top")],
        embedding_analysis.get("action_labels") or [],
        [vlm_analysis.get("action_top")],
        vlm_analysis.get("action_labels") or [],
    ):
        for item in source:
            label = clean_action_label(item)
            if label and label not in labels:
                labels.append(label)
            if len(labels) >= max(1, ACTION_SCORING_TOP_K):
                break
        if len(labels) >= max(1, ACTION_SCORING_TOP_K):
            break

    action_top = next(iter(merged_scores.keys()), "") if merged_scores else (labels[0] if labels else "")
    best = next(iter(merged_scores.values()), 0.0) if merged_scores else 0.0
    second = list(merged_scores.values())[1] if len(merged_scores) > 1 else 0.0
    margin_confidence = clamp(0.45 + (best * 0.35) + max(0.0, best - second) * 0.35)
    confidence = max(
        parse_float(vlm_analysis.get("action_confidence")),
        parse_float(embedding_analysis.get("action_confidence")),
        margin_confidence,
    )
    return {
        "action_labels": clean_action_labels(labels, limit=max(1, ACTION_SCORING_TOP_K)),
        "action_top": action_top,
        "action_scores": normalize_action_scores(merged_scores),
        "action_confidence": round(clamp(confidence), 3),
    }


def caption_chunk(s3, s3_thumb_path):
    """Download the thumbnail from S3, send it to the VLM as image content,
    return a caption. Returns empty string on any failure (caption is optional)."""
    try:
        # Fetch thumbnail bytes from S3
        buf = BytesIO()
        s3.download_fileobj(S3_BUCKET, s3_thumb_path, buf)
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        data_url = f"data:image/jpeg;base64,{img_b64}"

        resp = requests.post(
            f"{VLM_URL}/v1/chat/completions",
            json={
                "model": VLM_MODEL,
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Describe this video scene for retrieval in one specific sentence. "
                                "Include visible people/entities, action, movement, environment, and important objects. "
                                "Avoid one-word or generic captions."
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }],
                "max_tokens": 180,
                "stream": False,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[extractor] caption_chunk failed: {e}")
        return ""


def taxonomy_prompt(action_taxonomy):
    return "; ".join(
        f"{item['label']}: {item.get('description') or item['label']}"
        for item in action_taxonomy
        if item.get("label")
    )


def analyze_chunk(r, s3, s3_thumb_path, profile, video_metadata, source_file, motion_label, motion_score, action_taxonomy, frames=None):
    fallback_scene = f"{profile} camera view".strip()
    fallback_caption = (
        f"The clip shows a {fallback_scene} with {motion_label} across the sampled frames. "
        "The exact people, objects, environment, and activity require visual model confirmation."
    )
    fallback_tags = clean_list([profile, motion_label.replace(" ", "-"), "video"])
    fallback = {
        "caption": fallback_caption,
        "main_activity": f"{motion_label} activity",
        "tags": fallback_tags,
        "objects": [],
        "scene": fallback_scene,
        "motion": motion_label,
        "confidence": 0.35 + min(0.2, motion_score),
        "action_labels": [],
        "action_top": "",
        "action_scores": {},
        "action_confidence": 0.0,
    }
    if not VLM_ANALYSIS_ENABLED:
        return normalize_analysis(fallback, fallback)
    if vlm_backoff_active(r):
        return normalize_analysis(fallback, fallback)
    try:
        img_b64 = contact_sheet_b64(frames or [], max_frames=VLM_CONTACT_SHEET_FRAMES)
        if not img_b64:
            buf = BytesIO()
            s3.download_fileobj(S3_BUCKET, s3_thumb_path, buf)
            img_b64 = jpeg_b64_from_bytes(buf.getvalue(), max_size=768)
        data_url = f"data:image/jpeg;base64,{img_b64}"
        prompt = (
            "Analyze the provided CCTV/video frame sequence for retrieval. "
            "The image may be a contact sheet ordered left-to-right, top-to-bottom with timestamps; use it to infer action and movement over time. "
            "Return strict JSON only with keys: caption, main_activity, tags, objects, scene, motion, confidence, "
            "action_labels, action_scores, action_confidence. "
            "caption must be one or two specific sentences, 25 to 60 words, describing what is actually happening: "
            "people/entities, their actions and movement, interactions, visible objects, environment, camera viewpoint, and any temporal change. "
            "Do not return one-word captions or generic phrases such as normal, person, activity, video, or scene. "
            "main_activity must be a concise 4 to 12 word activity phrase. "
            "scene must describe the environment in 4 to 12 words. "
            "motion must describe visible movement or posture changes in 4 to 12 words. "
            "objects must be concrete visible entities/objects as short lowercase strings, including people, vehicles, weapons, bags, doors, counters, roads, furniture, or other relevant items. "
            "tags must be 6 to 12 lowercase retrieval terms covering actions, entities, objects, and environment. "
            "confidence must be 0 to 1. "
            "action_labels must contain up to three labels from this configured taxonomy only: "
            f"{taxonomy_prompt(action_taxonomy)}. "
            "action_scores must be an object whose keys are labels from that closed set and values are 0 to 1. "
            "Do not infer labels from filenames or paths; use only visual evidence in the image. "
            f"Known metadata: profile={profile}, "
            f"duration_ms={video_metadata.get('duration_ms', 0)}, "
            f"resolution={video_metadata.get('width', 0)}x{video_metadata.get('height', 0)}, "
            f"estimated_motion={motion_label}."
        )
        try:
            resp = requests.post(
                f"{VLM_URL}/api/chat",
                json={
                    "model": VLM_MODEL,
                    "messages": [{
                        "role": "user",
                        "content": prompt,
                        "images": [img_b64],
                    }],
                    "options": {"temperature": 0, "num_predict": VLM_NUM_PREDICT},
                    "stream": False,
                },
                timeout=VLM_TIMEOUT_SEC,
            )
            resp.raise_for_status()
            content = resp.json().get("message", {}).get("content", "")
        except Exception as native_error:
            if isinstance(native_error, requests.Timeout):
                raise native_error
            resp = requests.post(
                f"{VLM_URL}/v1/chat/completions",
                json={
                    "model": VLM_MODEL,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }],
                    "max_tokens": VLM_NUM_PREDICT,
                    "temperature": 0,
                    "stream": False,
                },
                timeout=VLM_TIMEOUT_SEC,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            print(f"[extractor] native VLM fallback used: {native_error}")
        analysis = normalize_analysis(extract_json_object(content), fallback)
        note_vlm_success(r)
        if not analysis["tags"]:
            analysis["tags"] = keyword_tags(" ".join([
                analysis["caption"], analysis["main_activity"], analysis["scene"], analysis["motion"],
            ]))
        return analysis
    except Exception as e:
        note_vlm_failure(r, str(e))
        print(f"[extractor] analyze_chunk fallback: {e}")
        return normalize_analysis(fallback, fallback)


def mean_vector(vectors):
    n = len(vectors)
    dim = len(vectors[0])
    mean = [sum(v[i] for v in vectors) / n for i in range(dim)]
    norm = sum(x * x for x in mean) ** 0.5 or 1.0
    return [x / norm for x in mean]


def process(msg_id, data, s3, r, qdrant):
    event_type = data.get("type")
    if event_type == "chunk.failed":
        r.xack(STREAM, GROUP, msg_id)
        return
    if event_type not in ("chunk.normalized", "video.normalized"):
        r.xack(STREAM, GROUP, msg_id)
        return

    video_id = data["video_id"]
    user_id = data["user_id"]
    tenant_id = data.get("tenant_id", "default")
    domain_id = data.get("domain_id", "general")
    profile = data.get("profile", "general")
    chunk_id = data["chunk_id"]
    benchmark_run_id = data.get("benchmark_run_id", "")
    few_shot_example = str(data.get("few_shot_example", "")).lower() in {"1", "true", "yes", "on"}
    few_shot_label = normalize_label(data.get("few_shot_label", ""))
    few_shot_model_id = data.get("few_shot_model_id", "")
    if EXTRACTOR_BENCHMARK_RUN_FILTER and benchmark_run_id not in EXTRACTOR_BENCHMARK_RUN_FILTER:
        r.xack(STREAM, GROUP, msg_id)
        return
    t_start_ms = int(data["t_start_ms"])
    t_end_ms = int(data["t_end_ms"])
    s3_processed_path = data["s3_processed_path"]
    s3_thumb_path = data["s3_thumb_path"]
    source_file = data.get("source_file") or data.get("file_name") or "original.mp4"
    original_video_metadata = parse_metadata_json(data.get("video_metadata_json", "{}"))

    fps = FPS_CCTV if profile == "cctv" else FPS_GENERAL
    emit_status(
        r, video_id, "indexing",
        chunk_id=chunk_id, chunk_index=data.get("chunk_index", ""),
        chunk_count=data.get("chunk_count", ""), t_start_ms=t_start_ms, t_end_ms=t_end_ms,
    )

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            chunk_file = str(Path(tmpdir) / "chunk.mp4")
            s3.download_file(S3_BUCKET, s3_processed_path, chunk_file)

            frames = extract_frames(chunk_file, fps, t_start_ms)
            if not frames:
                raise RuntimeError("no frames extracted")
            chunk_metadata = ffprobe_metadata(chunk_file)
            video_metadata = dict(original_video_metadata)
            video_metadata["source_file"] = source_file
            video_metadata["profile"] = profile
            video_metadata["chunk_duration_ms"] = chunk_metadata.get("duration_ms", 0)
            video_metadata["processed_width"] = chunk_metadata.get("width", 0)
            video_metadata["processed_height"] = chunk_metadata.get("height", 0)
            video_metadata["processed_fps"] = chunk_metadata.get("fps", 0)
            motion_score, motion_label = estimate_motion(frames)

            frame_ids, frame_t_ms, frame_chunk_t_ms, frame_images, frame_paths = zip(*frames)

            # Embed all frames in one batch
            embeddings = embed_images(list(frame_images))

            # Upload frames to S3
            frame_s3_keys = []
            for fid, fpath in zip(frame_ids, frame_paths):
                key = f"frames/{tenant_id}/{user_id}/{video_id}/{chunk_id}/{fid}.jpg"
                s3.upload_file(fpath, S3_BUCKET, key)
                frame_s3_keys.append(key)

        action_taxonomy = load_action_taxonomy(r)
        if few_shot_label:
            add_runtime_action_label(few_shot_label, f"user provided examples for {few_shot_label}")
            action_taxonomy = _ACTION_TAXONOMY
        embedding_action_analysis = score_actions_from_embeddings(embeddings, action_taxonomy, r)
        prototype_action_analysis = (
            {
                "action_labels": [few_shot_label],
                "action_top": few_shot_label,
                "action_scores": {few_shot_label: 1.0},
                "action_confidence": 1.0,
            }
            if few_shot_label and few_shot_example
            else few_shot_action_analysis(qdrant, user_id, domain_id, video_id, embeddings)
        )
        embedding_action_analysis = merge_action_evidence(prototype_action_analysis, embedding_action_analysis)
        analysis = analyze_chunk(
            r, s3, s3_thumb_path, profile, video_metadata, source_file,
            motion_label, motion_score, action_taxonomy, frames=frames,
        )
        analysis.update(merge_action_evidence(analysis, embedding_action_analysis))
        caption = analysis["caption"]
        tags = clean_list(analysis.get("tags", []) + keyword_tags(f"{caption} {analysis.get('main_activity', '')}"))
        objects = clean_list(analysis.get("objects", []))
        action_labels = clean_action_labels(analysis.get("action_labels", []), limit=max(1, ACTION_SCORING_TOP_K))
        action_top = clean_action_label(analysis.get("action_top")) or (action_labels[0] if action_labels else "")
        action_scores = normalize_action_scores(analysis.get("action_scores", {}))
        action_confidence = clamp(parse_float(analysis.get("action_confidence"), max(action_scores.values(), default=0.0)))

        # Chunk embedding = mean of frame embeddings
        chunk_vector = mean_vector(embeddings)

        # Write chunk to Qdrant
        qdrant.upsert(
            collection_name="chunks",
            points=[PointStruct(
                id=str(uuid.uuid4()),
                vector=chunk_vector,
                payload={
                    "user_id": user_id,
                    "tenant_id": tenant_id,
                    "domain_id": domain_id,
                    "video_id": video_id,
                    "chunk_id": chunk_id,
                    "t_start_ms": t_start_ms,
                    "t_end_ms": t_end_ms,
                    "s3_chunk_path": s3_processed_path,
                    "s3_thumb_path": s3_thumb_path,
                    "caption": caption,
                    "main_activity": analysis["main_activity"],
                    "tags": tags,
                    "objects": objects,
                    "scene": analysis["scene"],
                    "motion": analysis["motion"],
                    "analysis_confidence": analysis["confidence"],
                    "action_labels": action_labels,
                    "action_top": action_top,
                    "action_scores": action_scores,
                    "action_confidence": round(action_confidence, 3),
                    "motion_score": motion_score,
                    "source_file": source_file,
                    "benchmark_run_id": benchmark_run_id,
                    "few_shot_example": few_shot_example,
                    "few_shot_label": few_shot_label,
                    "few_shot_model_id": few_shot_model_id,
                    "video_metadata": video_metadata,
                    "chunk_metadata": chunk_metadata,
                    "duration_ms": video_metadata.get("duration_ms", 0),
                    "width": video_metadata.get("width", 0),
                    "height": video_metadata.get("height", 0),
                    "fps": video_metadata.get("fps", 0),
                    "profile": profile,
                },
            )],
        )

        # Write frames to Qdrant
        frame_points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=emb,
                payload={
                    "user_id": user_id,
                    "tenant_id": tenant_id,
                    "domain_id": domain_id,
                    "video_id": video_id,
                    "chunk_id": chunk_id,
                    "frame_id": fid,
                    "t_ms": t_ms,
                    "t_chunk_ms": chunk_t_ms,
                    "s3_frame_path": s3_key,
                    "caption": caption,
                    "main_activity": analysis["main_activity"],
                    "tags": tags,
                    "objects": objects,
                    "scene": analysis["scene"],
                    "motion": analysis["motion"],
                    "analysis_confidence": analysis["confidence"],
                    "action_labels": action_labels,
                    "action_top": action_top,
                    "action_scores": action_scores,
                    "action_confidence": round(action_confidence, 3),
                    "motion_score": motion_score,
                    "source_file": source_file,
                    "benchmark_run_id": benchmark_run_id,
                    "few_shot_example": few_shot_example,
                    "few_shot_label": few_shot_label,
                    "few_shot_model_id": few_shot_model_id,
                    "video_metadata": video_metadata,
                    "chunk_metadata": chunk_metadata,
                    "duration_ms": video_metadata.get("duration_ms", 0),
                    "width": video_metadata.get("width", 0),
                    "height": video_metadata.get("height", 0),
                    "fps": video_metadata.get("fps", 0),
                    "profile": profile,
                },
            )
            for fid, t_ms, chunk_t_ms, emb, s3_key in zip(frame_ids, frame_t_ms, frame_chunk_t_ms, embeddings, frame_s3_keys)
        ]
        qdrant.upsert(collection_name="frames", points=frame_points)

        r.xadd(STREAM, {
            "type": "chunk.indexed",
            "video_id": video_id,
            "user_id": user_id,
            "tenant_id": tenant_id,
            "chunk_id": chunk_id,
            "chunk_index": str(data.get("chunk_index", "")),
            "chunk_count": str(data.get("chunk_count", "")),
        })
        emit_status(
            r, video_id, "indexed",
            chunk_id=chunk_id, chunk_index=data.get("chunk_index", ""),
            chunk_count=data.get("chunk_count", ""), frame_count=len(frames),
        )
        print(f"[extractor] {video_id}/{chunk_id}: {len(frames)} frames indexed")
    except Exception as e:
        error = tail(str(e))
        r.xadd(STREAM, {
            "type": "chunk.failed",
            "video_id": video_id,
            "user_id": user_id,
            "tenant_id": tenant_id,
            "profile": profile,
            "chunk_id": chunk_id,
            "chunk_index": str(data.get("chunk_index", "")),
            "chunk_count": str(data.get("chunk_count", "")),
            "stage": "extractor",
            "error": error,
        })
        emit_status(
            r, video_id, "failed",
            stage_detail="extractor", chunk_id=chunk_id,
            chunk_index=data.get("chunk_index", ""), chunk_count=data.get("chunk_count", ""),
            error=error,
        )
        print(f"[extractor] {video_id}/{chunk_id}: failed {tail(error, 300)}")

    r.xack(STREAM, GROUP, msg_id)


def claim_stale(r):
    try:
        claimed = r.xautoclaim(STREAM, GROUP, CONSUMER, min_idle_time=PENDING_IDLE_MS, start_id="0-0", count=5)
        if isinstance(claimed, (list, tuple)) and len(claimed) > 1:
            return claimed[1] or []
        return []
    except Exception:
        return []


def main():
    r = redis_client()
    s3 = s3_client()
    qdrant = qdrant_client()
    try:
        r.xgroup_create(STREAM, GROUP, id="0", mkstream=True)
    except redis.exceptions.ResponseError:
        pass

    print(f"[extractor] listening as {CONSUMER}")
    while True:
        for msg_id, data in claim_stale(r):
            try:
                process(msg_id, data, s3, r, qdrant)
            except Exception as e:
                print(f"[extractor] stale error on {msg_id}: {e}")
                r.xadd(STREAM, {"type": "chunk.failed", "video_id": data.get("video_id", ""), "chunk_id": data.get("chunk_id", ""), "stage": "extractor", "error": str(e)})
                emit_status(r, data.get("video_id", ""), "failed", stage_detail="extractor", chunk_id=data.get("chunk_id", ""), error=str(e))
                r.xack(STREAM, GROUP, msg_id)

        messages = r.xreadgroup(GROUP, CONSUMER, {STREAM: ">"}, count=1, block=5000)
        if not messages:
            continue
        for _, entries in messages:
            for msg_id, data in entries:
                try:
                    process(msg_id, data, s3, r, qdrant)
                except Exception as e:
                    print(f"[extractor] error on {msg_id}: {e}")
                    r.xadd(STREAM, {"type": "chunk.failed", "video_id": data.get("video_id", ""), "chunk_id": data.get("chunk_id", ""), "stage": "extractor", "error": str(e)})
                    emit_status(r, data.get("video_id", ""), "failed", stage_detail="extractor", chunk_id=data.get("chunk_id", ""), error=str(e))
                    r.xack(STREAM, GROUP, msg_id)


if __name__ == "__main__":
    time.sleep(10)  # wait for embedder + qdrant
    main()
