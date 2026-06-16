# Architecture Plan — Query-Based Video Frame Detection (v2)

Status: planning doc, drafted 2026-06-14. Implementation has not started.
Next step on resume: **Step 1 — Security baseline**.

---

## 1. Goal

Evolve the project from a synchronous "upload → caption every frame → return one frame"
pipeline into a real-time, async, multi-tenant capable platform with two user flows:

1. **Library flow** — upload many videos, search by text query within a time window,
   browse matching frames.
2. **Ask flow** — chat against one or more indexed videos
   (RAG over video chunks), with timestamp citations that jump the player.

Primary use cases: CCTV/surveillance **and** general consumer video. Profile is
chosen per upload.

---

## 2. Locked decisions

| Area | Decision |
|---|---|
| Embedding model | **SigLIP 2** (replaces BLIP-2 + LoRA-BERT for retrieval) |
| VLM for chat/captioning | **Qwen2.5-VL-7B-AWQ** via vLLM, self-hosted |
| API gateway | **Keep Spring Boot**, extend with new endpoints |
| Workers | New Python services: `chunker`, `normalizer`, `extractor` |
| Orchestration | **Redis Streams** (durable, consumer groups), not pub/sub |
| Vector DB | **Qdrant** (time-range payload filtering, easy self-host) |
| Object storage | **MinIO** in local dev (S3-compatible), real S3 in prod |
| Metadata DB | MongoDB (kept) |
| Frontend | **Next.js 15 + Tailwind + shadcn/ui**, glass UI for chrome |
| Live updates | **SSE** (server → client only; sufficient for status + chat streaming) |
| Chunking | **Hybrid: shot detection (PySceneDetect) with 30s max cap** for `general`; **fixed 15s** for `cctv` |
| Embedding granularity | **Per-chunk + per-frame hybrid** (two Qdrant collections) |
| Use-case profile | Configurable per upload: `cctv` or `general` |
| Deployment target | Local `docker-compose` for now |
| Auth | **Google SSO (OAuth2)** → session JWT carrying `user_id` |
| Tenant model | Single-tenant, `tenant_id="default"`; revisit when a second tenant is real |
| Per-user isolation | **Per-user KMS CMK** (not per-tenant) + `user_id` scoping on every read/query |
| Trust model | Strong (a): users cannot see each other's data; operator can decrypt (audited). Not zero-knowledge. |

---

## 3. Service topology (docker-compose)

```
web                Next.js 15 (port 3000)
api                Spring Boot (port 8080) — extended
minio              S3-compatible local storage
mongo              video metadata
qdrant             vector store
redis              Streams + cache
chunker            Python worker — FFmpeg + PySceneDetect
normalizer         Python worker — resize/fps/codec + optional blur
extractor          Python worker (GPU) — SigLIP 2 embed + Qwen captioning + Qdrant writes
embedder           SigLIP 2 inference server (FastAPI or TorchServe)
vlm                vLLM serving Qwen2.5-VL-7B-AWQ on :8001 (lazy load on single-GPU dev)
```

GPU is shared between `extractor` and `vlm` in single-GPU dev. Use a
`docker-compose.gpu.yml` override.

---

## 4. Event flow (Redis Streams)

```
stream: pipeline.events
  ├─ video.uploaded     {video_id, s3_raw_path, profile, tenant_id}
  ├─ video.chunked      {video_id, chunks: [{chunk_id, t_start_ms, t_end_ms, s3_path}]}
  ├─ video.normalized   {video_id, chunk_id, s3_processed_path}
  ├─ video.indexed      {video_id, chunk_id}
  └─ video.failed       {video_id, stage, error}

stream: ui.status.{video_id}   # per-video stream tailed by SSE for live UI
```

Consumer groups: `cg-chunker`, `cg-normalizer`, `cg-extractor`. Failures go to a
DLQ stream and surface in the UI.

---

## 5. Storage layout

```
s3://video-vault/
  raw/{tenant}/{user_id}/{video_id}/original.mp4
  chunks/{tenant}/{user_id}/{video_id}/{chunk_id}.mp4
  processed/{tenant}/{user_id}/{video_id}/{chunk_id}.mp4
  thumbs/{tenant}/{user_id}/{video_id}/{chunk_id}.jpg
  frames/{tenant}/{user_id}/{video_id}/{chunk_id}/{frame_id}.jpg
```

Encryption:
- Local: MinIO with SSE-S3 (KMS not available locally; segregation enforced by path + authz).
- Prod: SSE-KMS with **one CMK per user**, alias `alias/vvault/user/<user_id>`. Every
  `PutObject` pins `x-amz-server-side-encryption-aws-kms-key-id` to that user's CMK ARN.
  Revoking a user's CMK cryptographically kills their data, including in backups.

---

## 6. Qdrant schema

**Collection `chunks`** (768-d SigLIP 2)
- payload: `user_id`, `video_id`, `chunk_id`, `t_start_ms`, `t_end_ms`, `s3_chunk_path`,
  `s3_thumb_path`, `caption`, `profile`, `tenant_id`

**Collection `frames`** (768-d SigLIP 2)
- payload: `user_id`, `video_id`, `chunk_id`, `frame_id`, `t_ms`, `s3_frame_path`,
  `parent_chunk_id`, `profile`, `tenant_id`

**Two-stage retrieval** (every stage filters by `user_id` from JWT — never trust path/body):
1. Embed query → top-50 chunks (filtered by `tenant_id`, **`user_id`**, `video_id` set,
   `t_start_ms` range, `profile`).
2. Re-rank by per-frame search restricted to those chunk IDs (filter still includes
   `user_id`) → top-K frames.
3. Return frames with timestamps + parent chunk references.

No shared ANN graph across users — `user_id` is a mandatory payload filter on
every search. Index partitioning per user can be revisited if leakage via
neighbor-graph statistics becomes a concern.

---

## 7. Per-profile defaults

| Field | `cctv` | `general` |
|---|---|---|
| Chunking | Fixed 15s windows | Shot detection, 30s cap |
| Frame sampling | 1 fps | 1 frame / 2s + shot keyframes |
| Face/plate blur | On | Off |
| Chat system prompt | "surveillance analyst" | "video assistant" |
| Default retention | 90d | user-controlled |

Selected in the upload modal; persisted on the video metadata document.

---

## 8. Spring Boot API additions

Keep `VideoController`, add:

- `POST /v1/auth/google` → OAuth2 authorization-code exchange. On first login:
  create Mongo `users` doc, provision KMS CMK (`alias/vvault/user/<user_id>`),
  attach key policy granting decrypt to the worker role. Returns session JWT
  carrying `user_id` (+ short TTL, refresh-token rotation).
- `GET /v1/auth/me` → current user from JWT.
- `POST /v1/video/presignUpload` → `{video_id, presignedPutUrl, expiresAt}`.
  Presigned PUT is pinned to the caller's CMK
  (`x-amz-server-side-encryption=aws:kms`,
  `x-amz-server-side-encryption-aws-kms-key-id=<user CMK ARN>`) and to the
  `raw/default/<user_id>/<video_id>/original.mp4` key. Client uploads directly
  to S3; bytes never tunnel through Spring.
- `POST /v1/video/{id}/finalize` → backend writes `video.uploaded` to Redis
  Streams (event payload includes `user_id`).
- `GET /v1/video/{id}/status` (SSE) → proxies `ui.status.{video_id}` after
  verifying the video belongs to the JWT user.
- `POST /v1/chat` (SSE) → RAG: embed query → Qdrant two-stage search (filtered
  by `user_id`) → vLLM stream → forward tokens to client.

**Authz filter** (Spring Security): every endpoint derives `user_id` from JWT;
any request whose path/body `user_id` ≠ JWT `user_id` is rejected at the filter
before controller code runs. `tenant_id="default"` is server-injected, never
client-supplied.

---

## 9. Glass UI design

Pragmatic glassmorphism:
- **Chrome glass**: sidebar, top bar, modals, chat panel
  → `bg-white/8 backdrop-blur-2xl border border-white/15`.
- **Content surfaces**: solid for readability (search grid, video player).
- **Background**: animated gradient mesh; replaced by blurred current video
  when a player is open.
- Routes:
  - `/library` — upload, list, search, grid of result frames.
  - `/ask/[videoId?]` — chat with optional scoped video set.
- **Live pipeline timeline** component per video, updated via SSE.

---

## 10. Security baseline (Step 1, do first)

Issues in current code to fix before any new features:

1. Hardcoded AWS keys in
   `MLService/src/NearestFrameToContext.py` (`self.aws_access_key_id = 'key'`)
   and `BackendService/src/main/resources/application.yml`
   (`accessKey: <accessKey>`).
   → Move to env vars; in prod use IAM roles / Secrets Manager.
2. **Google SSO** (OAuth2 authorization-code + PKCE on web). Spring Security
   resource server validates session JWT on every request. JWT carries
   `user_id` (internal UUID, not Google `sub`) + short TTL + refresh rotation.
3. **Per-user KMS CMK provisioning** on first login. Alias scheme
   `alias/vvault/user/<user_id>`. Key policy grants `kms:Encrypt`/`kms:Decrypt`
   to the API role (for presign + status) and the worker role (for processing).
   Worker IAM policy condition restricts `kms:Decrypt` to CMKs tagged
   `app=vvault`; explicit deny on `kms:Decrypt` against `Resource: "*"`.
4. **Authz middleware**: every endpoint rejects requests where path/body
   `user_id` ≠ JWT `user_id`. `tenant_id` is server-injected.
5. **Cross-user authz tests** (must exist before launch): user A cannot
   (a) GET user B's video by ID, (b) presign against user B's path,
   (c) get hits from user B's embeddings via `/v1/chat` or search.
6. Presigned URL TTL is 1 hour for results in `VideoService.searchOnVideo`.
   → Shorten to 10 minutes, GET-only, key-scoped to the user's prefix.
7. Redis / MongoDB / MinIO must bind to docker network only — no host ports
   in prod-like compose.
8. Add SSE-S3 on MinIO bucket by default; production uses SSE-KMS with the
   caller's per-user CMK (see §5).
9. Optional: face/license-plate blurring stage in `normalizer` for the `cctv`
   profile (e.g., `deface` or a YOLO-based detector).

---

## 11. Repo layout target

```
Query-Based-Video-Frame-Detection/
├── BackendService/                 # Spring Boot, extended
├── MLService/
│   ├── chunker/                    # FFmpeg + PySceneDetect
│   ├── normalizer/                 # resize/fps/codec + optional blur
│   ├── extractor/                  # SigLIP 2 + Qwen captioning + Qdrant writes
│   ├── inference/
│   │   ├── embedder/               # SigLIP 2 server
│   │   └── vlm/                    # vLLM Qwen2.5-VL
│   └── shared/                     # schemas, redis client, s3 client
├── FrontEndService/                # Streamlit — delete once Next.js reaches parity
├── web/                            # Next.js 15 app (new)
├── compose/
│   ├── docker-compose.yml
│   ├── docker-compose.gpu.yml
│   └── .env.example
└── docs/
    └── architecture.md             # this file
```

---

## 12. Execution order

1. **Security baseline** (Step 1, next session start).
2. Add MinIO + Qdrant + Redis Streams to compose; define schemas.
3. Stand up SigLIP 2 `embedder` service with `/embed` endpoint.
4. Refactor ML monolith into `chunker` → `normalizer` → `extractor` workers
   wired via Redis Streams. Implement per-profile branches. Write per-chunk
   and per-frame embeddings to Qdrant.
5. Extend Spring Boot: presign, finalize, SSE status, vector-search proxy.
6. Add vLLM container + `POST /v1/chat` RAG endpoint with SSE streaming.
7. Build Next.js glass UI: `/library` and `/ask` routes, SSE wiring, video
   player with timestamp jumps.
8. Delete Streamlit frontend.
9. (Later) LoRA fine-tune SigLIP 2 on collected query logs for domain
   specificity.

---

## 13. Open questions deferred to later

- Exact vLLM version pinning for Qwen2.5-VL — verify at Step 6.
- Audit log destination — CloudTrail (all `kms:Decrypt` calls, per-user via
  CMK ARN) + a Mongo collection vs. a dedicated append-only store. Decide
  before going to production. Per-user decrypt audit is now load-bearing
  given the trust model.
- Whether to keep MongoDB long-term or fold metadata into Qdrant payload.
- CMK lifecycle: deletion window (default 30d), rotation cadence, and how
  "delete my account" maps to `ScheduleKeyDeletion` + S3 object cleanup.
