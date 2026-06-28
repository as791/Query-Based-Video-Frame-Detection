// Server-side (SSR/RSC inside Docker): use internal network URL.
// Client-side (browser): use the public-facing URL (or proxy via next rewrites).
const API =
  typeof window === 'undefined'
    ? (process.env.API_URL || 'http://localhost:8080')
    : (process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080')

const WS_API = API.replace(/^http/, 'ws')

/**
 * Get current authenticated user. Works in both server and client contexts.
 * On the server, forwards the incoming request's cookies to the backend.
 */
export async function getMe() {
  const headers: Record<string, string> = {}

  // In SSR, `credentials: 'include'` does nothing — we must manually forward cookies.
  // Dynamic import avoids bundling `next/headers` into client-side code.
  if (typeof window === 'undefined') {
    try {
      const { cookies } = await import('next/headers')
      const cookieStore = await cookies()
      const cookieHeader = cookieStore.getAll()
        .map(c => `${c.name}=${c.value}`)
        .join('; ')
      if (cookieHeader) {
        headers['cookie'] = cookieHeader
      }
    } catch {
      // cookies() throws outside of request context (e.g. during build)
    }
  }

  const res = await fetch(`${API}/v1/auth/me`, {
    credentials: 'include',
    headers,
  })
  if (res.status === 401 || res.status === 403) return null
  if (!res.ok) return null
  return res.json()
}

export async function presignUpload(profile = 'general', fileName = 'video.mp4', contentType = 'video/mp4') {
  const params = new URLSearchParams({ profile, fileName, contentType })
  const res = await fetch(`${API}/v1/video/presignUpload?${params}`, {
    method: 'POST',
    credentials: 'include',
  })
  if (!res.ok) {
    const detail = await res.text().catch(() => '')
    throw new Error(`Failed to get presigned URL (${res.status})${detail ? `: ${detail}` : ''}`)
  }
  return res.json() as Promise<{
    videoId: string
    presignedPutUrl: string
    s3Key: string
    [key: `uploadHeader:${string}`]: string
  }>
}

export async function finalizeUpload(videoId: string, s3Key: string, profile = 'general', sourceFile = 'video.mp4', fewShotLabel = '', domainId = 'general') {
  const params = new URLSearchParams({ s3Key, profile, sourceFile })
  if (fewShotLabel) params.set('fewShotLabel', fewShotLabel)
  if (domainId) params.set('domainId', domainId)
  const res = await fetch(
    `${API}/v1/video/${videoId}/finalize?${params}`,
    { method: 'POST', credentials: 'include' }
  )
  if (!res.ok) {
    const detail = await res.text().catch(() => '')
    throw new Error(`Failed to finalize upload (${res.status})${detail ? `: ${detail}` : ''}`)
  }
  return res.json()
}

export async function uploadViaBackend(file: File, profile = 'general', fewShotLabel = '', domainId = 'general') {
  const form = new FormData()
  form.append('file', file)
  const params = new URLSearchParams({ profile })
  if (fewShotLabel) params.set('fewShotLabel', fewShotLabel)
  if (domainId) params.set('domainId', domainId)
  const res = await fetch(`${API}/v1/video/upload?${params}`, {
    method: 'POST',
    credentials: 'include',
    body: form,
  })
  if (!res.ok) {
    const detail = await res.text().catch(() => '')
    throw new Error(`Backend upload failed (${res.status})${detail ? `: ${detail}` : ''}`)
  }
  return res.json() as Promise<{ videoId: string; s3Key: string; status: string }>
}

export async function initiateMultipartUpload(file: File, profile = 'general') {
  const params = new URLSearchParams({
    profile,
    fileName: file.name,
    contentType: file.type || 'video/mp4',
  })
  const res = await fetch(`${API}/v1/video/multipart/initiate?${params}`, {
    method: 'POST',
    credentials: 'include',
  })
  if (!res.ok) {
    const detail = await res.text().catch(() => '')
    throw new Error(`Failed to initiate multipart upload (${res.status})${detail ? `: ${detail}` : ''}`)
  }
  return res.json() as Promise<{ videoId: string; s3Key: string; uploadId: string; partSize: string }>
}

export async function presignMultipartPart(videoId: string, s3Key: string, uploadId: string, partNumber: number) {
  const params = new URLSearchParams({ s3Key, uploadId, partNumber: String(partNumber) })
  const res = await fetch(`${API}/v1/video/multipart/${videoId}/part?${params}`, {
    credentials: 'include',
  })
  if (!res.ok) {
    const detail = await res.text().catch(() => '')
    throw new Error(`Failed to sign upload part (${res.status})${detail ? `: ${detail}` : ''}`)
  }
  return res.json() as Promise<{ url: string; partNumber: string }>
}

export async function completeMultipartUpload(
  videoId: string,
  s3Key: string,
  uploadId: string,
  parts: Array<{ partNumber: number; eTag: string }>,
  profile = 'general',
  sourceFile = 'video.mp4',
  fewShotLabel = '',
  domainId = 'general'
) {
  const res = await fetch(`${API}/v1/video/multipart/${videoId}/complete`, {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ s3Key, uploadId, parts, profile, sourceFile, fewShotLabel, domainId }),
  })
  if (!res.ok) {
    const detail = await res.text().catch(() => '')
    throw new Error(`Failed to complete multipart upload (${res.status})${detail ? `: ${detail}` : ''}`)
  }
  return res.json() as Promise<{ videoId: string; s3Key: string; status: string }>
}

export async function abortMultipartUpload(videoId: string, s3Key: string, uploadId: string) {
  const params = new URLSearchParams({ s3Key, uploadId })
  await fetch(`${API}/v1/video/multipart/${videoId}?${params}`, {
    method: 'DELETE',
    credentials: 'include',
  }).catch(() => {})
}

export type FrameResult = {
  s3_frame_path: string
  url: string
  clip_url?: string
  t_ms: number
  t_chunk_ms?: number
  video_id: string
  chunk_id: string
  frame_id?: string
  selected?: boolean
  caption?: string
  chunk_caption?: string
  main_activity?: string
  scene?: string
  motion?: string
  source_file?: string
  tags?: string[]
  objects?: string[]
  confidence?: number
  relevance_reason?: string
  rerank_source?: string
  initial_score?: number
  vector_score?: number
  tag_score?: number
  action_score?: number
  action_top?: string
  action_labels?: string[]
  metadata_score?: number
  domain_id?: string
  domain_score?: number
  feedback_score?: number
  model_version?: string
}

export type ActionLabel = {
  label: string
  description: string
}

export type FewShotLabel = {
  label: string
  description: string
  exampleCount: number
  status: string
}

export type FewShotExample = {
  videoId: string
  label: string
  sourceFile: string
  s3Key: string
  status: string
  createdAt: string
}

export type DomainSummary = {
  domainId: string
  domainName: string
  description: string
  labelCount: number
  exampleCount: number
  modelVersion: string
  updatedAt: string
}

export type DomainLabel = {
  label: string
  description: string
  exampleCount: number
  status: string
}

export type DomainModel = {
  schemaVersion: number
  tenantId: string
  userId: string
  domainId: string
  domainName: string
  modelVersion: string
  labels: DomainLabel[]
  examples: FewShotExample[]
  feedbackStats: Record<string, number>
  config: Record<string, unknown>
  updatedAt: string
}

export async function listDomains() {
  const res = await fetch(`${API}/v1/domains`, {
    credentials: 'include',
  })
  if (!res.ok) throw new Error('Failed to load domains')
  return res.json() as Promise<{ domains: DomainSummary[] }>
}

export async function createDomain(name: string, description = '') {
  const res = await fetch(`${API}/v1/domains`, {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, description }),
  })
  if (!res.ok) {
    const detail = await res.text().catch(() => '')
    throw new Error(`Failed to create domain (${res.status})${detail ? `: ${detail}` : ''}`)
  }
  return res.json() as Promise<DomainSummary>
}

export async function getDomainModel(domainId: string) {
  const res = await fetch(`${API}/v1/domains/${encodeURIComponent(domainId)}/model`, {
    credentials: 'include',
  })
  if (!res.ok) throw new Error('Failed to load domain model')
  return res.json() as Promise<DomainModel>
}

export async function addDomainLabel(domainId: string, label: string, description = '') {
  const res = await fetch(`${API}/v1/domains/${encodeURIComponent(domainId)}/labels`, {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ label, description }),
  })
  if (!res.ok) {
    const detail = await res.text().catch(() => '')
    throw new Error(`Failed to save domain label (${res.status})${detail ? `: ${detail}` : ''}`)
  }
  return res.json() as Promise<DomainLabel>
}

export async function listActionLabels() {
  const res = await fetch(`${API}/v1/action-labels`, {
    credentials: 'include',
  })
  if (!res.ok) throw new Error('Failed to load action labels')
  return res.json() as Promise<{ labels: ActionLabel[] }>
}

export async function addActionLabel(label: string, description = '') {
  const res = await fetch(`${API}/v1/action-labels`, {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ label, description }),
  })
  if (!res.ok) {
    const detail = await res.text().catch(() => '')
    throw new Error(`Failed to save action label (${res.status})${detail ? `: ${detail}` : ''}`)
  }
  return res.json() as Promise<ActionLabel>
}

export async function listFewShotModel(domainId = 'general') {
  const params = new URLSearchParams()
  if (domainId) params.set('domainId', domainId)
  const res = await fetch(`${API}/v1/few-shot?${params}`, {
    credentials: 'include',
  })
  if (!res.ok) throw new Error('Failed to load few-shot model')
  return res.json() as Promise<{ labels: FewShotLabel[]; examples: FewShotExample[] }>
}

export async function addFewShotLabel(label: string, description = '', domainId = 'general') {
  const params = new URLSearchParams()
  if (domainId) params.set('domainId', domainId)
  const res = await fetch(`${API}/v1/few-shot/labels?${params}`, {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ label, description }),
  })
  if (!res.ok) {
    const detail = await res.text().catch(() => '')
    throw new Error(`Failed to save few-shot label (${res.status})${detail ? `: ${detail}` : ''}`)
  }
  return res.json() as Promise<FewShotLabel>
}

export async function search(query: string, videoId?: string, minConfidence = 0.8, limit = 12, domainId = '') {
  const body: Record<string, unknown> = { query, videoId, minConfidence, limit }
  if (domainId) body.domainId = domainId
  const res = await fetch(`${API}/v1/search`, {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error('Search failed')
  return res.json() as Promise<FrameResult[]>
}

export async function submitSearchFeedback(input: {
  domainId: string
  query: string
  videoId: string
  frameId?: string
  chunkId?: string
  relevant: boolean
}) {
  const res = await fetch(`${API}/v1/search/feedback`, {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input),
  })
  if (!res.ok) {
    const detail = await res.text().catch(() => '')
    throw new Error(`Failed to save feedback (${res.status})${detail ? `: ${detail}` : ''}`)
  }
  return res.json() as Promise<DomainModel>
}

export async function undoSearchFeedback(input: {
  domainId: string
  query: string
  videoId: string
  frameId?: string
  chunkId?: string
  relevant: boolean
}) {
  const res = await fetch(`${API}/v1/search/feedback/undo`, {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input),
  })
  if (!res.ok) {
    const detail = await res.text().catch(() => '')
    throw new Error(`Failed to undo feedback (${res.status})${detail ? `: ${detail}` : ''}`)
  }
  return res.json() as Promise<DomainModel>
}

export async function frameContext(videoId: string, chunkId: string, tMs: number, window = 12) {
  const params = new URLSearchParams({ videoId, chunkId, tMs: String(tMs), window: String(window) })
  const res = await fetch(`${API}/v1/search/frames/context?${params}`, {
    credentials: 'include',
  })
  if (!res.ok) throw new Error('Failed to load frame context')
  return res.json() as Promise<FrameResult[]>
}

export function statusStream(videoId: string) {
  return new EventSource(`${API}/v1/video/${videoId}/status`, { withCredentials: true })
}

export async function videoStatusSnapshot(videoId: string) {
  const res = await fetch(`${API}/v1/video/${videoId}/status/snapshot`, {
    credentials: 'include',
  })
  if (!res.ok) throw new Error('Failed to load video status')
  return res.json() as Promise<Record<string, string>>
}

export function chatStream(query: string, videoId?: string) {
  // We need POST+SSE — use fetch with ReadableStream
  return fetch(`${API}/v1/chat`, {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, videoId }),
  })
}

export function openChatSocket() {
  return new WebSocket(`${WS_API}/ws/chat`)
}
