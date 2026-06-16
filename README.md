# VideoVault — Query-Based Video Frame Detection

VideoVault turns a library of uploaded videos into a searchable, askable index.
Users sign in with Google, upload a video, and within seconds they can:

- **Search** the library by free-text query and jump to matching frames.
- **Ask** questions about indexed videos and stream a grounded answer back
  (RAG over per-chunk + per-frame embeddings).

This README documents the system end-to-end: high-level architecture, the
low-level path a single request takes through each service, and a worked
example tied to the recording in
[`e2e-recordings/2026-06-16-sso-e2e/`](e2e-recordings/2026-06-16-sso-e2e/).

The design rationale and locked decisions live in
[`docs/architecture.md`](docs/architecture.md). Read that first if you want
the *why*; this README is the *what* and *how*.

---

## 1. High-Level Design (HLD)

### 1.1 Component map

```
                       ┌─────────────────────────────┐
                       │           Browser           │
                       │  Next.js 15  (web, :3000)   │
                       └────────────┬────────────────┘
                                    │  HTTPS / SSE / WebSocket
                                    ▼
                       ┌─────────────────────────────┐
                       │   Spring Boot API  (:8080)  │
                       │  auth · presign · status    │
                       │  search · chat (RAG)        │
                       └─┬─────────┬─────────┬───────┘
                         │         │         │
                         │ presign │ enqueue │ query
                         ▼         ▼         ▼
                   ┌─────────┐ ┌───────┐ ┌────────┐
                   │   S3 /  │ │ Redis │ │ Qdrant │
                   │  MinIO  │ │Streams│ │ vectors│
                   └────┬────┘ └───┬───┘ └────┬───┘
                        │          │          ▲
                        │          ▼          │
                        │   ┌────────────────────────────┐
                        │   │     ML workers (Python)    │
                        │   │  chunker → normalizer →    │
                        │   │  extractor (SigLIP 2 +     │
                        │   │  Qwen2.5-VL via Ollama)    │
                        │   └────────────┬───────────────┘
                        │                │ embed
                        ▼                ▼
                   ┌─────────┐     ┌──────────┐
                   │ MongoDB │     │ Embedder │
                   │metadata │     │ FastAPI  │
                   └─────────┘     │  :8002   │
                                   └──────────┘
```

All services run from `compose/docker-compose.yml` on a private `internal`
network; only `web` (3000) and `api` (8080) bind to host ports.

### 1.2 Service responsibilities

| Service | Tech | Job |
|---|---|---|
| `web` | Next.js 15 + Tailwind + shadcn/ui | Glass-UI library, search grid, chat (`/library`, `/ask`) |
| `api` | Spring Boot, Spring Security OAuth2 | Auth, presigned URLs, status SSE, search proxy, chat WebSocket |
| `mongo` | MongoDB 7 | Video metadata (owner, profile, status, S3 keys) |
| `postgres` | Postgres 16 | User accounts, sessions, chat history |
| `redis` | Redis 7 (Streams) | `pipeline.events` + per-video `ui.status.{video_id}` |
| `qdrant` | Qdrant | `chunks` + `frames` collections, 768-d SigLIP 2 vectors |
| S3 / `minio` | S3 API | `raw/ chunks/ processed/ thumbs/ frames/` per user |
| `chunker` | Python + FFmpeg + PySceneDetect | Split raw video into chunks (15s CCTV / shot-cap 30s general) |
| `normalizer` | Python + FFmpeg | Resize / fps / codec, write `processed/` chunks, emit thumbs |
| `extractor` | Python + Qdrant + Ollama | Sample frames, embed via SigLIP 2, caption via Qwen2.5-VL, upsert to Qdrant |
| `embedder` | FastAPI + SigLIP 2 | `/embed` endpoint, image + text in the same 768-d space |
| VLM | Ollama on host (`qwen2.5vl:7b`) | Captions + chat responses; vLLM in prod (see compose comment) |

### 1.3 Trust and isolation

- Google OAuth2 (authorization code) terminates at Spring Security. The
  session cookie pins a server-side `User` row keyed by `googleSub`.
- Every controller derives `user_id` from the `@AuthenticationPrincipal`;
  request bodies/paths never supply it. Cross-user reads are rejected at
  the filter layer.
- **Per-user KMS CMK** (`alias/vvault/user/<user_id>`) is provisioned on
  first login. Every `PutObject` (presigned or backend-streamed) pins
  `SSEAlgorithm.KMS` to that user's CMK. Locally, MinIO falls back to
  SSE-S3.
- Qdrant searches always carry a `must` filter on `user_id`. No shared ANN
  graph; see [`docs/architecture.md` §6](docs/architecture.md).

### 1.4 Event bus

Single durable Redis Stream `pipeline.events`, consumed by three groups:

```
video.uploaded   ─►  cg-chunker      (chunker)
video.chunked    ─►  cg-normalizer   (normalizer)
video.normalized ─►  cg-extractor    (extractor)  ─►  video.indexed | video.failed
```

UI status is a separate per-video stream `ui.status.{video_id}` that the
backend tails and forwards to the browser over SSE.

---

## 2. Low-Level Design (LLD)

### 2.1 Public API surface (Spring Boot, `:8080`)

| Verb | Path | Purpose |
|---|---|---|
| GET | `/v1/auth/me` | Current user from session (`AuthController`) |
| GET | `/login/oauth2/code/google` | OAuth callback; provisions `User` + KMS CMK on first login (`CustomOAuth2UserService`) |
| POST | `/v1/video/presignUpload` | Presigned PUT URL (KMS-pinned, 10 min TTL) |
| POST | `/v1/video/upload` | Multipart upload fallback for small files |
| POST | `/v1/video/multipart/initiate` | Start S3 multipart upload (8 MB parts) |
| GET | `/v1/video/multipart/{id}/part` | Presigned PUT for a single part |
| POST | `/v1/video/multipart/{id}/complete` | Finish multipart upload |
| POST | `/v1/video/{id}/finalize` | Publish `video.uploaded` to `pipeline.events` |
| GET | `/v1/video/{id}/status` (SSE) | Live pipeline status |
| GET | `/v1/video/{id}/status/snapshot` | Latest status (one-shot) |
| POST | `/v1/search` | Two-stage Qdrant retrieval, returns ranked frames |
| GET | `/v1/search/frames/context` | Neighboring frames around a hit (used by frame modal) |
| POST | `/v1/chat` (SSE) | RAG chat (token stream) |
| WS | `/ws/chat` | Bidirectional chat session (used by `/ask` UI) |
| GET | `/health` | Liveness for compose healthcheck |

### 2.2 Upload path (frontend → S3)

1. `web` calls `POST /v1/video/multipart/initiate` with `{profile, fileName, contentType}`.
2. Spring mints `videoId = UUID`, resolves the user's CMK, and asks S3 for
   an `uploadId`. Returns `{videoId, s3Key, uploadId, partSize=8MiB}`.
3. Browser chunks the file and, for each part, calls
   `GET /v1/video/multipart/{id}/part?partNumber=N` to get a presigned PUT
   URL. Bytes go straight to S3; the API never proxies the payload.
4. On completion the browser posts the part ETags to
   `POST /v1/video/multipart/{id}/complete`. Spring calls
   `CompleteMultipartUpload`.
5. Browser calls `POST /v1/video/{id}/finalize` → Spring writes
   `{event: "video.uploaded", video_id, user_id, profile, s3_raw_path}`
   to `pipeline.events`.

For files under the multipart threshold, `presignUpload` returns a single
presigned PUT and the rest of the flow is identical from `finalize` onward.

### 2.3 ML pipeline (Redis Streams)

Each worker uses a Redis consumer group with `XREADGROUP` + `XACK`, with
`PENDING_IDLE_MS = 60s` for crash recovery.

**Chunker** (`MLService/chunker/app.py`):
- Downloads `raw/.../original.mp4` from S3.
- `cctv` profile: fixed 15s windows. `general`: PySceneDetect content
  detection capped at 30s (with a fast-path fixed window for very large or
  very long videos).
- Uploads each chunk to `chunks/{tenant}/{user_id}/{video_id}/{chunk_id}.mp4`.
- Emits `video.chunked` with the chunk list + timestamps; emits
  `ui.status.{video_id}` events as `chunking → chunked`.

**Normalizer** (`MLService/normalizer/app.py`):
- Re-encodes each chunk to a deterministic profile (resize, fps cap,
  H.264). Writes to `processed/...`.
- Optional face/plate blur for `cctv` profile.
- Emits one `video.normalized` per chunk.

**Extractor** (`MLService/extractor/app.py`):
- Samples frames at `FPS_GENERAL = 0.5` or `FPS_CCTV = 1.0`.
- For each frame: SigLIP 2 image embed (via `embedder` `/embed`) →
  Qwen2.5-VL caption (via Ollama, with a soft timeout) → text embed.
- Upserts to Qdrant `frames` collection with payload
  `{user_id, video_id, chunk_id, frame_id, t_ms, s3_frame_path, parent_chunk_id, profile, tenant_id}`.
- Aggregates per-chunk embedding + caption into Qdrant `chunks` collection.
- Emits `video.indexed` per chunk; `video.failed` on terminal error.

### 2.4 Search path

`POST /v1/search` (handled by `VectorSearchController` →
`SearchRerankService`) is two-stage:

1. Text → SigLIP 2 embedding via `embedder`.
2. Qdrant `chunks` query, top-50, filtered by `user_id`, optional
   `video_id ∈ {…}`, optional `t_start_ms` range, `profile`.
3. Qdrant `frames` query restricted to those `parent_chunk_id`s, top-K.
4. Per-frame presigned GET URL (10-minute TTL, key-scoped to the caller's
   prefix). Hits ship with caption, timestamp, parent chunk, and a
   re-rank score.

The frame modal calls `GET /v1/search/frames/context?frame_id=…` to fetch
N neighbours on either side for prev/next navigation.

### 2.5 Chat path (`/ask`)

`/ask` opens a WebSocket to `/ws/chat` (`ChatWebSocketHandler`):

1. Browser sends `{type: "prompt", text, video_ids?}`.
2. Handler authenticates the session, creates / loads a `ChatSession` in
   Postgres, appends the user `ChatMessage`.
3. Embeds the prompt → two-stage Qdrant retrieval scoped to `user_id` (and
   `video_ids` if provided) → assembles a prompt with frame captions +
   timestamps as citations.
4. Streams Qwen2.5-VL output from Ollama, forwarding each token frame to
   the client as `{type: "token", delta}`.
5. On `[DONE]`, persists the assistant message and emits
   `{type: "done", message_id, citations}`.

If the VLM is unreachable the handler falls back to a retrieval-only
answer (top captions + timestamps), surfaced in the UI as a soft-degrade.

### 2.6 Storage layout

```
s3://video-vault/
  raw/{tenant}/{user_id}/{video_id}/original.mp4
  chunks/{tenant}/{user_id}/{video_id}/{chunk_id}.mp4
  processed/{tenant}/{user_id}/{video_id}/{chunk_id}.mp4
  thumbs/{tenant}/{user_id}/{video_id}/{chunk_id}.jpg
  frames/{tenant}/{user_id}/{video_id}/{chunk_id}/{frame_id}.jpg
```

Qdrant payload schema lives in
[`docs/architecture.md` §6](docs/architecture.md#6-qdrant-schema).

---

## 3. Running locally

```bash
cp compose/.env.example compose/.env   # fill GOOGLE_CLIENT_ID/SECRET, AWS keys or MinIO creds
cd compose
docker compose up --build
# web      → http://localhost:3000
# api      → http://localhost:8080
# qdrant   → internal only
```

GPU box: also include the override.

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

The local VLM uses **Ollama on the host** (`qwen2.5vl:7b`) by default
(`VLM_URL=http://host.docker.internal:11434`). Start it with
`ollama pull qwen2.5vl:7b && ollama serve`. Production swaps it for the
commented-out `vlm` service (vLLM + Qwen2.5-VL-7B-AWQ).

---

## 4. End-to-end example — the SSO recording

The recording at
[`e2e-recordings/2026-06-16-sso-e2e/videovault-sso-e2e-recording.mp4`](e2e-recordings/2026-06-16-sso-e2e/videovault-sso-e2e-recording.mp4)
walks through a single user session end to end. Each numbered frame in
the recording corresponds to a concrete code path:

| # | Frame | What the user sees | What the system does |
|---|---|---|---|
| 1 | `001-signed-out-landing` | Landing page at `/` | Next.js renders the unauthenticated marketing surface |
| 2 | `002-google-account-chooser` | Google account picker | Redirect to `/oauth2/authorization/google`; Spring Security drives the authorization-code flow |
| 3 | `003-authenticated-library` | `/library` after redirect | First-login `CustomOAuth2UserService` upserts the `User` row, `KmsService` provisions `alias/vvault/user/<id>`; cookie session begins |
| 4 | `004-upload-started` | "Upload started" toast | Browser drags `5977704-hd_1366_586_30fps.mp4` → `POST /v1/video/multipart/initiate` → parallel presigned PUTs → `complete` → `finalize` (publishes `video.uploaded`) |
| 5 | `005-processing-1` | `normalized · 0/3 indexed` | `chunker` produced 3 chunks (general profile, shot-detected, ≤30s); `normalizer` finished all 3; extractor SSE shows `0/3` indexed |
| 6 | `005-processing-2` | `indexing · 1/3 indexed` | `extractor` sampled frames for chunk 1, embedded via SigLIP 2, captioned via Ollama, upserted to Qdrant `frames` + `chunks`; emitted `video.indexed` |
| 7 | `005-processing-3` | `Ready (3/3 chunks indexed)` | All chunks indexed; library card flips to **Ready**; `VideoStatus` snapshot resolves to terminal state |
| 8 | `006-search-query-entered` | "low motion" in search bar | Browser holds the query; nothing on the wire yet |
| 9 | `007-search-results` | Frame grid of matches | `POST /v1/search {q: "low motion"}` → embed → Qdrant chunks top-50 → frames top-K (filtered by `user_id`) → presigned GETs returned |
|10 | `008-frame-modal-open` | Frame modal opens | `GET /v1/search/frames/context?frame_id=…` for the clicked hit; modal shows caption + timestamp + neighbours |
|11 | `009-frame-modal-previous` | Modal walked back one frame | Same endpoint, `cursor=prev`; modal swaps to the neighbouring frame in the same chunk |
|12 | `010-ask-websocket-connected` | `/ask` page, "connected" indicator | WebSocket handshake to `/ws/chat`; `ChatWebSocketHandler` authenticates the session, creates a `ChatSession` row |
|13 | `011-chat-prompt-entered` | "summarize low motion" | Client sends `{type: "prompt", text: "summarize low motion"}` |
|14 | `012-chat-streaming` | Tokens streaming | Server: embed prompt → Qdrant two-stage retrieval scoped to user → build context with captions/timestamps → stream Qwen2.5-VL tokens from Ollama → forward `{type: "token", delta}` frames |
|15 | `013-chat-completed` | Final answer + input reset | `[DONE]` from VLM → `ChatMessage` persisted → `{type: "done", message_id, citations}` |

The full machine-readable trace lives in
[`e2e-recordings/2026-06-16-sso-e2e/manifest.json`](e2e-recordings/2026-06-16-sso-e2e/manifest.json),
including timestamps for every frame, which is useful when diffing
pipeline latency across releases.

---

## 5. Repo layout

```
Query-Based-Video-Frame-Detection/
├── BackendService/               Spring Boot API (auth, presign, status, search, chat)
├── MLService/
│   ├── chunker/                  FFmpeg + PySceneDetect worker
│   ├── normalizer/               FFmpeg re-encode + optional blur
│   ├── extractor/                SigLIP 2 + Qwen captioning + Qdrant writes
│   ├── inference/embedder/       SigLIP 2 FastAPI server (/embed)
│   └── src/                      legacy monolith (kept for reference)
├── FrontEndService/              Streamlit (deprecated; deleted once /library + /ask reach parity)
├── web/                          Next.js 15 app
├── compose/
│   ├── docker-compose.yml
│   ├── docker-compose.gpu.yml
│   ├── .env.example
│   └── init/                     qdrant-init.py, redis-init.sh, minio-init.sh
├── docs/architecture.md          Design doc — locked decisions, trust model, schemas
├── e2e-recordings/               Captured end-to-end runs (video + manifest)
└── tests/                        Smoke + pipeline integration tests
```

---

## 6. Benchmarking

End-to-end benchmarks (ingest throughput, per-stage latency, search
recall/mAP, chat TTFT) run against the
[Kaggle CCTV Action Recognition dataset](https://www.kaggle.com/datasets/jonathannield/cctv-action-recognition-dataset).
See [`docs/benchmarking.md`](docs/benchmarking.md) for the metric
definitions, run layout, and regression rules.

## 7. Pointers

- Architecture, rationale, open questions → [`docs/architecture.md`](docs/architecture.md)
- Benchmarking methodology → [`docs/benchmarking.md`](docs/benchmarking.md)
- Latest E2E capture → [`e2e-recordings/2026-06-16-sso-e2e/`](e2e-recordings/2026-06-16-sso-e2e/)
- Smoke/integration tests → [`tests/smoke_test.sh`](tests/smoke_test.sh), [`tests/test_pipeline.py`](tests/test_pipeline.py)
- Compose environment template → [`compose/.env.example`](compose/.env.example)
