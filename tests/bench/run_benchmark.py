#!/usr/bin/env python3
"""Run the documented CCTV benchmark against a live VideoVault API."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import ssl
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
DATASET_ID = "jonathannield/cctv-action-recognition-dataset"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://localhost:8080", help="VideoVault API base URL")
    parser.add_argument("--data-dir", default="tests/bench/data", help="Dataset directory")
    parser.add_argument("--runs-dir", default="tests/bench/runs", help="Benchmark output directory")
    parser.add_argument("--cookie", default=os.environ.get("VIDEOVAULT_COOKIE", ""), help="Auth cookie, e.g. SESSION=...")
    parser.add_argument("--benchmark-sub", default=os.environ.get("VIDEOVAULT_BENCHMARK_SUB", ""), help="Google subject for local benchmark auth")
    parser.add_argument("--download", action="store_true", help="Download/unzip the Kaggle dataset first")
    parser.add_argument("--limit", type=int, default=0, help="Limit clips for a smoke benchmark")
    parser.add_argument("--wait-timeout-sec", type=int, default=7200)
    parser.add_argument("--search-limit", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=0, help="Upload and wait in batches; 0 uploads everything before waiting")
    parser.add_argument("--upload-workers", type=int, default=8, help="Parallel upload workers inside each batch")
    parser.add_argument("--skip-upload", action="store_true", help="Use an existing labels.json in the run dir")
    parser.add_argument("--existing-run-dir", default="", help="Existing run directory for eval-only mode")
    parser.add_argument("--labels-from-run-dir", default="", help="Read labels.json from this run dir while writing outputs to a new/current run dir")
    parser.add_argument("--eval-limit", type=int, default=0, help="Evaluate only the first N dataset-order videos from labels.json")
    parser.add_argument("--benchmark-run-id", default="", help="Stable run id used to isolate indexed benchmark payloads")
    parser.add_argument("--skip-action-label-sync", action="store_true", help="Do not sync benchmark labels into the configured action taxonomy")
    parser.add_argument("--few-shot-per-label", type=int, default=0, help="Upload this many held-out labeled examples per label before the eval clips")
    parser.add_argument("--few-shot-run-id", default="", help="Run id used for held-out few-shot example uploads")
    parser.add_argument("--seed-few-shot-only", action="store_true", help="Upload/wait for held-out few-shot examples and exit before eval upload/search")
    parser.add_argument("--domain-id", default="", help="Optional domain profile id for upload/search isolation")
    args = parser.parse_args()

    root = Path.cwd()
    data_dir = (root / args.data_dir).resolve()
    runs_dir = (root / args.runs_dir).resolve()
    run_dir = Path(args.existing_run_dir).resolve() if args.existing_run_dir else runs_dir / datetime.now().strftime("%Y-%m-%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    benchmark_run_id = args.benchmark_run_id or run_dir.name

    if args.download:
        download_dataset(data_dir)

    all_clips = list(iter_dataset(data_dir))
    few_shot_clips: list[tuple[Path, str]] = []
    if args.few_shot_per_label > 0 and not args.skip_upload:
        few_shot_clips, all_clips = split_few_shot_examples(all_clips, args.few_shot_per_label)
    clips = all_clips
    if args.limit:
        clips = clips[: args.limit]
    if not clips and not args.skip_upload:
        raise SystemExit(f"No video clips found under {data_dir}")
    benchmark_env = load_env_file(root / "compose/.benchmark.env")

    write_json(run_dir / "run.json", {
        "dataset": f"kaggle:{DATASET_ID}",
        "data_dir": str(data_dir),
        "clip_count": len(clips),
        "api": args.api,
        "profile": "cctv",
        "search_limit": args.search_limit,
        "batch_size": args.batch_size,
        "upload_workers": args.upload_workers,
        "benchmark_run_id": benchmark_run_id,
        "action_label_sync": not args.skip_action_label_sync,
            "few_shot": {
            "enabled": args.few_shot_per_label > 0 and not args.skip_upload,
            "examples_per_label": args.few_shot_per_label,
            "example_count": len(few_shot_clips),
            "run_id": args.few_shot_run_id or f"{benchmark_run_id}-fewshot",
            },
            "domain_id": args.domain_id,
        "config": {
            "vlm_analysis_enabled": env_value("VLM_ANALYSIS_ENABLED", benchmark_env),
            "vlm_timeout_sec": env_value("VLM_TIMEOUT_SEC", benchmark_env),
            "vlm_failure_threshold": env_value("VLM_FAILURE_THRESHOLD", benchmark_env),
            "vlm_failure_window_sec": env_value("VLM_FAILURE_WINDOW_SEC", benchmark_env),
            "vlm_failure_backoff_sec": env_value("VLM_FAILURE_BACKOFF_SEC", benchmark_env),
            "action_taxonomy_enabled": True,
            "action_scoring_enabled": env_value("ACTION_SCORING_ENABLED", benchmark_env),
            "action_scoring_top_k": env_value("ACTION_SCORING_TOP_K", benchmark_env),
            "action_scoring_frame_top_k": env_value("ACTION_SCORING_FRAME_TOP_K", benchmark_env),
            "action_scoring_min_raw_score": env_value("ACTION_SCORING_MIN_RAW_SCORE", benchmark_env),
            "search_llm_rerank_enabled": env_value("SEARCH_LLM_RERANK_ENABLED", benchmark_env),
            "search_chunk_candidate_limit": env_value("SEARCH_CANDIDATES_CHUNK_LIMIT", benchmark_env),
            "search_frame_candidate_limit": env_value("SEARCH_CANDIDATES_FRAME_LIMIT", benchmark_env),
            "search_action_candidate_limit": env_value("SEARCH_CANDIDATES_ACTION_LIMIT", benchmark_env),
            "autoscaler_poll_interval_sec": env_value("AUTOSCALER_POLL_INTERVAL_SEC", benchmark_env),
            "chunker_min_replicas": env_value("CHUNKER_MIN_REPLICAS", benchmark_env),
            "chunker_max_replicas": env_value("CHUNKER_MAX_REPLICAS", benchmark_env),
            "normalizer_min_replicas": env_value("NORMALIZER_MIN_REPLICAS", benchmark_env),
            "normalizer_max_replicas": env_value("NORMALIZER_MAX_REPLICAS", benchmark_env),
            "extractor_min_replicas": env_value("EXTRACTOR_MIN_REPLICAS", benchmark_env),
            "extractor_max_replicas": env_value("EXTRACTOR_MAX_REPLICAS", benchmark_env),
        },
    })

    if args.skip_upload:
        labels_dir = Path(args.labels_from_run_dir).resolve() if args.labels_from_run_dir else run_dir
        labels = read_json(labels_dir / "labels.json")
        if args.eval_limit:
            labels = limit_labels_by_dataset_order(labels, clips, args.eval_limit)
            write_json(run_dir / f"labels.first-{args.eval_limit}.json", labels)
        else:
            write_json(run_dir / "labels.json", labels)
    else:
        require_auth(args.cookie, args.benchmark_sub)
        auth = Auth(args.cookie, args.benchmark_sub)
        if not args.skip_action_label_sync:
            pre_queries = load_queries(root / "tests/bench/queries.yaml", labels_for_clips(clips))
            sync_action_labels(args.api, auth, pre_queries)
        if few_shot_clips:
            few_shot_run_id = args.few_shot_run_id or f"{benchmark_run_id}-fewshot"
            print(f"uploading {len(few_shot_clips)} held-out few-shot examples under run id {few_shot_run_id}", flush=True)
            few_shot_labels = upload_clips(
                args.api,
                auth,
                few_shot_clips,
                run_dir,
                few_shot_run_id,
                offset=0,
                total=len(few_shot_clips),
                workers=args.upload_workers,
                use_few_shot_labels=True,
                domain_id=args.domain_id,
            )
            write_json(run_dir / "few_shot_labels.json", few_shot_labels)
            wait_ready(args.api, auth, few_shot_labels, run_dir, args.wait_timeout_sec, label="few-shot examples")
            if args.seed_few_shot_only:
                print(f"Few-shot examples written to {run_dir}")
                return 0
        if args.batch_size > 0:
            labels = upload_batches(args.api, auth, clips, run_dir, benchmark_run_id, args.batch_size, args.upload_workers, args.wait_timeout_sec, args.domain_id)
        else:
            labels = upload_clips(args.api, auth, clips, run_dir, benchmark_run_id, offset=0, total=len(clips), workers=args.upload_workers, domain_id=args.domain_id)
            wait_ready(args.api, auth, labels, run_dir, args.wait_timeout_sec)
        write_json(run_dir / "labels.json", labels)

    queries = load_queries(root / "tests/bench/queries.yaml", labels)
    auth = Auth(args.cookie, args.benchmark_sub)
    search_raw = replay_search(args.api, auth, labels, queries, run_dir, args.search_limit, benchmark_run_id, args.domain_id)
    metrics, per_class = compute_search_metrics(labels, search_raw)
    write_json(run_dir / "per_class.json", per_class)
    write_json(run_dir / "metrics.json", metrics)
    print(f"Benchmark run written to {run_dir}")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


def download_dataset(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    if not shutil_which("kaggle"):
        raise SystemExit("Kaggle CLI is not installed. Install it with: python3 -m pip install kaggle")
    token = Path.home() / ".kaggle/kaggle.json"
    if not token.exists():
        raise SystemExit("Missing Kaggle token at ~/.kaggle/kaggle.json")
    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", DATASET_ID,
        "-p", str(data_dir),
        "--unzip",
    ], check=True)


def shutil_which(name: str) -> str | None:
    for part in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(part) / name
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


def env_value(name: str, env_file: dict[str, str]) -> str:
    return os.environ.get(name, env_file.get(name, ""))


def iter_dataset(data_dir: Path) -> list[tuple[Path, str]]:
    clips: list[tuple[Path, str]] = []
    for path in sorted(data_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
            clips.append((path, infer_label(data_dir, path)))
    return clips


def infer_label(data_dir: Path, clip: Path) -> str:
    rel = clip.relative_to(data_dir)
    if len(rel.parts) > 2 and rel.parts[0].lower() == "videos" and rel.parts[1].lower() == "videos":
        return normalize_label(rel.parts[2])
    if len(rel.parts) > 1:
        return normalize_label(rel.parts[0])
    return normalize_label(clip.stem.split("_")[0])


def normalize_label(value: str) -> str:
    return "".join(ch for ch in value.lower().strip() if ch.isalnum())


def limit_labels_by_dataset_order(labels: dict[str, dict[str, str]], clips: list[tuple[Path, str]], limit: int) -> dict[str, dict[str, str]]:
    by_path = {record["path"]: (video_id, record) for video_id, record in labels.items()}
    limited: dict[str, dict[str, str]] = {}
    for clip, _ in clips:
        found = by_path.get(str(clip))
        if not found:
            continue
        video_id, record = found
        limited[video_id] = record
        if len(limited) >= limit:
            break
    if len(limited) < limit:
        raise SystemExit(f"Only found {len(limited)} labeled videos for eval limit {limit}")
    return limited


def labels_for_clips(clips: list[tuple[Path, str]]) -> dict[str, dict[str, str]]:
    return {
        f"clip-{idx}": {"label": label, "path": str(path), "source_file": path.name}
        for idx, (path, label) in enumerate(clips)
    }


def split_few_shot_examples(
        clips: list[tuple[Path, str]],
        examples_per_label: int) -> tuple[list[tuple[Path, str]], list[tuple[Path, str]]]:
    selected: list[tuple[Path, str]] = []
    counts: dict[str, int] = {}
    remaining: list[tuple[Path, str]] = []
    for clip, label in clips:
        if counts.get(label, 0) < examples_per_label:
            selected.append((clip, label))
            counts[label] = counts.get(label, 0) + 1
        else:
            remaining.append((clip, label))
    missing = sorted({label for _, label in clips if counts.get(label, 0) < examples_per_label})
    if missing:
        raise SystemExit(f"Not enough few-shot examples for labels: {', '.join(missing)}")
    return selected, remaining


class Auth:
    def __init__(self, cookie: str = "", benchmark_sub: str = "") -> None:
        self.cookie = cookie
        self.benchmark_sub = benchmark_sub


def require_auth(cookie: str, benchmark_sub: str) -> None:
    if not cookie and not benchmark_sub:
        raise SystemExit("Missing auth. Pass --cookie 'SESSION=...' or --benchmark-sub with BENCHMARK_AUTH_ENABLED=true.")


def sync_action_labels(api: str, auth: Auth, queries: dict[str, str]) -> None:
    existing = request_json("GET", f"{api}/v1/action-labels", auth=auth)
    existing_labels = {
        str(item.get("label", ""))
        for item in existing.get("labels", [])
        if isinstance(item, dict)
    } if isinstance(existing, dict) else set()
    for label, description in sorted(queries.items()):
        if label in existing_labels:
            continue
        request_json("POST", f"{api}/v1/action-labels", auth=auth, body={
            "label": label,
            "description": description,
        })
        existing_labels.add(label)


def upload_batches(
        api: str,
        auth: Auth,
        clips: list[tuple[Path, str]],
        run_dir: Path,
        benchmark_run_id: str,
        batch_size: int,
        workers: int,
        wait_timeout_sec: int,
        domain_id: str = "") -> dict[str, dict[str, str]]:
    labels: dict[str, dict[str, str]] = {}
    for start in range(0, len(clips), batch_size):
        batch_no = (start // batch_size) + 1
        batch = clips[start:start + batch_size]
        end = start + len(batch)
        print(f"batch {batch_no}: uploading clips {start + 1}-{end} of {len(clips)}", flush=True)
        batch_labels = upload_clips(api, auth, batch, run_dir, benchmark_run_id, offset=start, total=len(clips), workers=workers, domain_id=domain_id)
        labels.update(batch_labels)
        write_json(run_dir / "labels.json", labels)
        wait_ready(args_api=api, auth=auth, labels=batch_labels, run_dir=run_dir, timeout_sec=wait_timeout_sec, label=f"batch {batch_no}")
    return labels


def upload_clips(
        api: str,
        auth: Auth,
        clips: list[tuple[Path, str]],
        run_dir: Path,
        benchmark_run_id: str,
        offset: int,
        total: int,
        workers: int,
        use_few_shot_labels: bool = False,
        domain_id: str = "") -> dict[str, dict[str, str]]:
    labels: dict[str, dict[str, str]] = {}
    workers = max(1, min(workers, len(clips) or 1))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(upload_one_clip, api, auth, clip, label, benchmark_run_id, offset + idx, total, label if use_few_shot_labels else "", domain_id): (clip, label)
            for idx, (clip, label) in enumerate(clips, start=1)
        }
        for future in as_completed(futures):
            video_id, record = future.result()
            labels[video_id] = record
            write_json(run_dir / "labels.json", labels)
    return labels


def upload_one_clip(api: str, auth: Auth, clip: Path, label: str, benchmark_run_id: str, idx: int, total: int, few_shot_label: str = "", domain_id: str = "") -> tuple[str, dict[str, str]]:
    print(f"[{idx}/{total}] uploading {clip.name} ({label})", flush=True)
    init = request_json(
        "POST",
        f"{api}/v1/video/multipart/initiate?{urllib.parse.urlencode({'profile': 'cctv', 'fileName': clip.name, 'contentType': content_type(clip)})}",
        auth=auth,
    )
    parts = upload_parts(api, auth, clip, init)
    body = {
        "s3Key": init["s3Key"],
        "uploadId": init["uploadId"],
        "profile": "cctv",
        "sourceFile": clip.name,
        "benchmarkRunId": benchmark_run_id,
        "parts": parts,
    }
    if few_shot_label:
        body["fewShotLabel"] = few_shot_label
    if domain_id:
        body["domainId"] = domain_id
    request_json("POST", f"{api}/v1/video/multipart/{init['videoId']}/complete", auth=auth, body=body)
    record = {"label": label, "path": str(clip), "source_file": clip.name, "benchmark_run_id": benchmark_run_id}
    if few_shot_label:
        record["few_shot_label"] = few_shot_label
    if domain_id:
        record["domain_id"] = domain_id
    return init["videoId"], record


def upload_parts(api: str, auth: Auth, clip: Path, init: dict[str, str]) -> list[dict[str, object]]:
    part_size = int(init.get("partSize", 8 * 1024 * 1024))
    parts: list[dict[str, object]] = []
    with clip.open("rb") as handle:
        part_number = 1
        while True:
            chunk = handle.read(part_size)
            if not chunk:
                break
            qs = urllib.parse.urlencode({
                "s3Key": init["s3Key"],
                "uploadId": init["uploadId"],
                "partNumber": part_number,
            })
            part = request_json("GET", f"{api}/v1/video/multipart/{init['videoId']}/part?{qs}", auth=auth)
            req = urllib.request.Request(part["url"], data=chunk, method="PUT")
            etag = upload_part_with_retry(req)
            parts.append({"partNumber": part_number, "eTag": etag})
            part_number += 1
    return parts


def upload_part_with_retry(req: urllib.request.Request, attempts: int = 4) -> str:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=300, context=ssl_context()) as resp:
                return resp.headers.get("ETag", "").strip('"')
        except (urllib.error.URLError, ConnectionResetError, TimeoutError) as exc:
            last_error = exc
            if attempt == attempts:
                break
            time.sleep(min(20, 2 ** attempt))
    raise last_error or RuntimeError("part upload failed")


def wait_ready(
        args_api: str,
        auth: Auth,
        labels: dict[str, dict[str, str]],
        run_dir: Path,
        timeout_sec: int,
        label: str = "run") -> None:
    deadline = time.time() + timeout_sec
    pending = set(labels)
    with (run_dir / "status_snapshots.jsonl").open("a", encoding="utf-8") as out:
        while pending and time.time() < deadline:
            for video_id in list(pending):
                snap = request_json("GET", f"{args_api}/v1/video/{video_id}/status/snapshot", auth=auth)
                snap["video_id"] = video_id
                snap["observed_at"] = datetime.now().isoformat(timespec="seconds")
                out.write(json.dumps(snap, sort_keys=True) + "\n")
                out.flush()
                if snap.get("stage") in {"ready", "partial", "failed"}:
                    pending.remove(video_id)
            if pending:
                print(f"{label}: waiting for {len(pending)} videos to finish indexing", flush=True)
                time.sleep(10)
    if pending:
        raise SystemExit(f"Timed out waiting for videos: {sorted(pending)}")


def replay_search(api: str, auth: Auth, labels: dict[str, dict[str, str]], queries: dict[str, str], run_dir: Path, limit: int, benchmark_run_id: str, domain_id: str = "") -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with (run_dir / "search_raw.jsonl").open("w", encoding="utf-8") as out:
        for label in sorted({item["label"] for item in labels.values()}):
            query = queries.get(label, f"a person {label}")
            started = time.perf_counter()
            body = {
                "query": query,
                "limit": limit,
                "minConfidence": 0,
                "benchmarkRunId": benchmark_run_id,
            }
            if domain_id:
                body["domainId"] = domain_id
            raw_hits = request_json("POST", f"{api}/v1/search", auth=auth, body=body)
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            allowed_video_ids = set(labels)
            hits = [hit for hit in raw_hits if isinstance(hit, dict) and str(hit.get("video_id", "")) in allowed_video_ids]
            row = {"label": label, "query": query, "latency_ms": elapsed_ms, "hits": hits}
            rows.append(row)
            out.write(json.dumps(row, sort_keys=True) + "\n")
    return rows


def compute_search_metrics(labels: dict[str, dict[str, str]], search_raw: list[dict[str, object]]) -> tuple[dict[str, object], dict[str, object]]:
    label_by_video = {video_id: item["label"] for video_id, item in labels.items()}
    per_class: dict[str, object] = {}
    ap_values: list[float] = []
    rr_values: list[float] = []
    recall_at = {1: [], 5: [], 20: []}
    precision_at = {5: []}
    latencies = [float(row["latency_ms"]) for row in search_raw]

    for row in search_raw:
        label = str(row["label"])
        hits = row.get("hits") if isinstance(row.get("hits"), list) else []
        relevant_total = sum(1 for item in labels.values() if item["label"] == label)
        ranked = [str(hit.get("video_id", "")) for hit in hits if isinstance(hit, dict)]
        relevant_flags = [label_by_video.get(video_id) == label for video_id in ranked]
        first_rank = next((idx + 1 for idx, ok in enumerate(relevant_flags) if ok), 0)
        precisions = []
        found = 0
        seen_relevant_videos: set[str] = set()
        for idx, ok in enumerate(relevant_flags, start=1):
            video_id = ranked[idx - 1]
            if ok and video_id not in seen_relevant_videos:
                seen_relevant_videos.add(video_id)
                found += 1
                precisions.append(found / idx)
        average_precision = sum(precisions) / max(1, relevant_total)
        reciprocal_rank = 0.0 if first_rank == 0 else 1.0 / first_rank
        ap_values.append(average_precision)
        rr_values.append(reciprocal_rank)
        per_class[label] = {
            "query": row["query"],
            "relevant_total": relevant_total,
            "first_rank": first_rank,
            "average_precision": round(average_precision, 4),
            "mrr": round(reciprocal_rank, 4),
            "recall@1": recall(ranked, label_by_video, label, relevant_total, 1),
            "recall@5": recall(ranked, label_by_video, label, relevant_total, 5),
            "recall@20": recall(ranked, label_by_video, label, relevant_total, 20),
            "precision@5": precision(relevant_flags, 5),
        }
        for k in recall_at:
            recall_at[k].append(per_class[label][f"recall@{k}"])
        precision_at[5].append(per_class[label]["precision@5"])

    metrics = {
        "dataset": f"kaggle:{DATASET_ID}",
        "profile": "cctv",
        "ingest": {"videos_total": len(labels)},
        "search": {
            "recall@1": mean(recall_at[1]),
            "recall@5": mean(recall_at[5]),
            "recall@20": mean(recall_at[20]),
            "precision@5": mean(precision_at[5]),
            "map": mean(ap_values),
            "mrr": mean(rr_values),
            "latency_ms": {"p50": percentile(latencies, 50), "p95": percentile(latencies, 95)},
        },
    }
    return metrics, per_class


def recall(ranked: list[str], label_by_video: dict[str, str], label: str, total: int, k: int) -> float:
    found = {video_id for video_id in ranked[:k] if label_by_video.get(video_id) == label}
    return round(min(1.0, len(found) / max(1, total)), 4)


def precision(flags: list[bool], k: int) -> float:
    return round(sum(flags[:k]) / max(1, min(k, len(flags))), 4)


def mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def percentile(values: list[float], pct: int) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return round(values[0], 2)
    return round(statistics.quantiles(values, n=100, method="inclusive")[pct - 1], 2)


def load_queries(path: Path, labels: dict[str, dict[str, str]]) -> dict[str, str]:
    queries: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            queries[normalize_label(key)] = value.strip().strip("'\"")
    for label in {item["label"] for item in labels.values()}:
        queries.setdefault(label, f"a person {label}")
    return queries


def request_json(method: str, url: str, auth: Auth | None = None, body: object | None = None) -> object:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Accept", "application/json")
    if body is not None:
        req.add_header("Content-Type", "application/json")
    if auth and auth.cookie:
        req.add_header("Cookie", auth.cookie)
    if auth and auth.benchmark_sub:
        req.add_header("X-Benchmark-Google-Sub", auth.benchmark_sub)
    last_error: Exception | None = None
    for attempt in range(1, 5):
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise SystemExit(f"{method} {url} failed with HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, ConnectionResetError, TimeoutError) as exc:
            last_error = exc
            if attempt == 4:
                break
            time.sleep(min(20, 2 ** attempt))
    raise last_error or RuntimeError(f"{method} {url} failed")


def content_type(path: Path) -> str:
    return mimetypes.guess_type(path.name)[0] or "video/mp4"


def ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
