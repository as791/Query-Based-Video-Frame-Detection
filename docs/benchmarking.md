# VideoVault — End-to-End Benchmarking

This doc defines how we benchmark the full VideoVault pipeline
(ingest → embed → index → search → chat) using a real CCTV
workload.

**Dataset:** [CCTV Action Recognition Dataset](https://www.kaggle.com/datasets/jonathannield/cctv-action-recognition-dataset)
by Jonathan Nield (Kaggle).

The dataset is a fit because:

- It is **CCTV footage** with **action labels per clip**, which exercises
  the `cctv` profile end-to-end: 15s fixed-window chunking, 1 fps frame
  sampling, optional face/plate blur in the normalizer.
- Labels give us a ground-truth retrieval signal: for each action class
  we can issue a text query and measure whether the top-K frames returned
  by `/v1/search` fall inside a clip with that label.
- Clip lengths are short enough to load a meaningful library on a single
  GPU dev box.

This doc has three parts:

1. [Setup](#1-setup) — get the dataset onto disk and uploaded.
2. [Metrics](#2-metrics) — what we measure and where the numbers come from.
3. [Reporting](#3-reporting) — the result template every benchmark run fills in.

---

## 1. Setup

### 1.1 Acquire the dataset

```bash
# Requires a Kaggle API token at ~/.kaggle/kaggle.json
pip install kaggle
mkdir -p tests/bench/data
kaggle datasets download -d jonathannield/cctv-action-recognition-dataset \
  -p tests/bench/data --unzip
```

Inspect the layout once unzipped — the directory structure on Kaggle
typically organizes clips by action class. Capture the exact layout in
the run report (see §3) so future runs can be diffed.

```bash
find tests/bench/data -maxdepth 3 -type d | head
find tests/bench/data -name '*.mp4' | wc -l   # clip count for the report
```

### 1.2 Bulk-ingest into VideoVault

Use the same multipart path the UI uses
(`POST /v1/video/multipart/initiate` → presigned PUTs → `complete` →
`finalize`); never bypass it, or the benchmark stops representing the
real flow.

`tests/bench/upload.py` (to be added) drives this loop:

```python
# Pseudocode — full script lives in tests/bench/upload.py
for clip_path, action_label in iter_dataset("tests/bench/data"):
    init = api.post("/v1/video/multipart/initiate",
                    params={"profile": "cctv",
                            "fileName": clip_path.name})
    upload_parts(clip_path, init)
    api.post(f"/v1/video/multipart/{init['videoId']}/complete", json=parts)
    api.post(f"/v1/video/{init['videoId']}/finalize")
    record(video_id=init["videoId"], label=action_label, path=clip_path)
```

The mapping `{video_id → ground-truth action label}` is written to
`tests/bench/runs/<run-id>/labels.json` and is the ground truth all
retrieval metrics are computed against.

### 1.3 Wait for indexing

Tail the SSE status stream until every uploaded video reaches `Ready`:

```bash
curl -N -H "Cookie: SESSION=…" \
     http://localhost:8080/v1/video/$VIDEO_ID/status
```

A worker script in `tests/bench/wait_ready.py` polls
`/v1/video/{id}/status/snapshot` for every video and exits when all are
terminal (`Ready` or `Failed`). Failed videos are recorded but excluded
from quality metrics.

---

## 2. Metrics

Every metric below is recorded per run into
`tests/bench/runs/<run-id>/metrics.json`. Stage-level numbers are
scraped from the `ui.status.{video_id}` Redis stream — every worker
already emits `chunking → chunked → normalizing → normalized → indexing
→ indexed` events with timestamps, so we get them for free.

### 2.1 Ingest throughput

| Metric | Definition |
|---|---|
| `videos_total` | Number of clips uploaded |
| `videos_failed` | Clips whose terminal status is `Failed` |
| `videos_per_min` | `videos_total / wall_minutes_to_all_ready` |
| `chunks_total` | Sum of `chunks_emitted` over all videos |
| `chunks_per_sec` | `chunks_total / wall_seconds` (extractor throughput proxy) |

### 2.2 Per-stage latency (per chunk)

Computed from `ui.status.{video_id}` event timestamps.

| Stage | Start event | End event |
|---|---|---|
| Chunking | `video.uploaded` | `video.chunked` |
| Normalize | `video.chunked` | `video.normalized` (per chunk) |
| Extract | `video.normalized` | `video.indexed` (per chunk) |
| **End-to-end** | `video.uploaded` | last `video.indexed` |

Report P50 / P95 / P99 for each stage. Extract is usually the slowest
stage and dominates total latency on single-GPU dev hardware.

### 2.3 Component-level numbers

- **Embedder QPS** — hit `embedder:8002/embed` with a fixed batch
  (`tests/bench/embedder_load.py`) and report sustained QPS + P95
  latency for `image` and `text` separately. SigLIP 2 image embeds
  dominate extractor cost.
- **VLM caption latency** — captured per-frame in the extractor logs
  (`vlm_ms`). Report P50 / P95 + timeout rate (fraction of frames where
  `VLM_TIMEOUT_SEC` triggered the soft-fallback).
- **GPU utilization** — `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 1`
  sampled into `tests/bench/runs/<run-id>/gpu.csv`.

### 2.4 Search quality (the load-bearing number)

For each action class `c` in the dataset:

1. Build a text query from the label (e.g. `walking → "a person walking"`,
   `loitering → "a person loitering near an entrance"`). The exact
   prompt template lives in `tests/bench/queries.yaml` so quality
   numbers are reproducible across runs.
2. Issue `POST /v1/search {q, k=20}` filtered to the run's `user_id`.
3. A hit is **relevant** iff its `video_id` belongs to a clip labeled `c`.

Metrics:

| Metric | What it tells us |
|---|---|
| **Recall@K** (K ∈ {1, 5, 20}) | Did *any* clip from class `c` show up? |
| **Precision@K** | Of the top-K, what fraction were correct class? |
| **mAP** | Class-balanced average — single headline number per run |
| **MRR** | Rank of the first relevant hit; sensitive to the re-ranker |

Run `tests/bench/eval_search.py` to compute these from
`labels.json` + the raw search responses; aggregated numbers go into
`metrics.json` and per-class numbers into `per_class.json`.

### 2.5 Search latency

`POST /v1/search` is timed end-to-end client-side and broken down
server-side via Spring `@Timed`:

| Hop | Metric |
|---|---|
| Embed (text → 768-d) | `search.embed_ms` |
| Qdrant chunks query | `search.chunks_ms` |
| Qdrant frames query | `search.frames_ms` |
| Presign frame URLs | `search.presign_ms` |
| **Total wall** | `search.total_ms` |

Report P50 / P95 over a fixed query mix (queries.yaml replayed twice,
warm-cache pass and cold-cache pass).

### 2.6 Chat (RAG) quality + latency

For a curated set of CCTV-flavored prompts
(`tests/bench/chat_prompts.yaml`, e.g. *"summarize any loitering near
the entrance"*, *"how many people walked past between 10:00 and 10:30?"*):

| Metric | Definition |
|---|---|
| **TTFT** | Time-to-first-token from `POST /v1/chat` or `/ws/chat` |
| **Total stream time** | First token → `[DONE]` |
| **Citation precision** | Fraction of `citations` in the final message whose underlying frames are labeled with an action that matches the prompt class (manually labeled once per prompt) |
| **VLM fallback rate** | Fraction of prompts where the handler degraded to retrieval-only |

---

## 3. Reporting

Each benchmark run produces a directory:

```
tests/bench/runs/<YYYY-MM-DD-HHmm-shortsha>/
├── run.yaml          # config: profile, K, query set, model versions
├── labels.json       # {video_id: action_class}
├── status_stream.jsonl   # raw ui.status.* events for every video
├── search_raw.jsonl  # raw /v1/search responses
├── chat_raw.jsonl    # raw /ws/chat transcripts
├── gpu.csv
├── per_class.json    # recall@K / precision@K / mAP per class
└── metrics.json      # headline numbers (the table below)
```

Headline `metrics.json` schema:

```json
{
  "dataset": "kaggle:jonathannield/cctv-action-recognition-dataset",
  "profile": "cctv",
  "models": {
    "embedder": "google/siglip2-base-patch16-224",
    "vlm": "qwen2.5vl:7b@ollama"
  },
  "ingest": {
    "videos_total": 0,
    "videos_failed": 0,
    "videos_per_min": 0.0,
    "chunks_per_sec": 0.0,
    "stage_latency_ms": {
      "chunking":   {"p50": 0, "p95": 0, "p99": 0},
      "normalize":  {"p50": 0, "p95": 0, "p99": 0},
      "extract":    {"p50": 0, "p95": 0, "p99": 0},
      "end_to_end": {"p50": 0, "p95": 0, "p99": 0}
    },
    "vlm_timeout_rate": 0.0
  },
  "search": {
    "recall@1": 0.0, "recall@5": 0.0, "recall@20": 0.0,
    "precision@5": 0.0, "map": 0.0, "mrr": 0.0,
    "latency_ms": {"p50": 0, "p95": 0}
  },
  "chat": {
    "ttft_ms":  {"p50": 0, "p95": 0},
    "total_ms": {"p50": 0, "p95": 0},
    "citation_precision": 0.0,
    "vlm_fallback_rate": 0.0
  },
  "gpu": {"avg_util_pct": 0, "peak_mem_mb": 0}
}
```

A run is considered a regression if any of the following degrade vs. the
previous run on the same hardware:

- `search.map` drops > 2 percentage points
- `ingest.stage_latency_ms.extract.p95` grows > 20%
- `chat.ttft_ms.p95` grows > 20%
- `ingest.videos_failed` is non-zero

---

## 4. Known caveats

- **Quality vs. profile mismatch.** Numbers from this dataset only
  validate the `cctv` profile. The `general` profile (shot detection,
  0.5 fps, no blur) should be benchmarked on a separate consumer-video
  set; do not mix the two in headline numbers.
- **Class → query prompt** is an authored mapping. Two reasonable
  prompts for the same class can produce noticeably different recall;
  freeze `queries.yaml` per run and version it in git.
- **Single-GPU dev hardware.** Extractor + VLM share the GPU locally;
  in prod they are separate. Compare like-with-like — do not regress
  prod against local numbers.
- **VLM availability.** When Ollama is down, the extractor falls back
  to no caption (embedding-only). Track `vlm_timeout_rate`; high
  values invalidate the chat quality numbers for that run.
