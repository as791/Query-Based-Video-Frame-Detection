'use client'
import { useEffect, useRef, useState } from 'react'
import {
  abortMultipartUpload,
  completeMultipartUpload,
  finalizeUpload,
  frameContext,
  FrameResult,
  initiateMultipartUpload,
  presignMultipartPart,
  presignUpload,
  search,
  statusStream,
  uploadViaBackend,
  videoStatusSnapshot,
} from '@/lib/api'

type UploadState = 'queued' | 'uploading' | 'processing' | 'ready' | 'partial' | 'failed' | 'cancelled'
type UploadItem = {
  id: string
  file: File
  state: UploadState
  progress: number
  status: string
  videoId?: string
  error?: string
}

const MULTIPART_THRESHOLD = 64 * 1024 * 1024
const MULTIPART_CONCURRENCY = 4

export default function LibraryPage() {
  const [query, setQuery] = useState('')
  const [frames, setFrames] = useState<FrameResult[]>([])
  const [uploads, setUploads] = useState<UploadItem[]>([])
  const [dragging, setDragging] = useState(false)
  const [searching, setSearching] = useState(false)
  const [searchError, setSearchError] = useState('')
  const [hasSearched, setHasSearched] = useState(false)
  const [minConfidence, setMinConfidence] = useState(0.8)
  const [selectedFrame, setSelectedFrame] = useState<FrameResult | null>(null)
  const [timeline, setTimeline] = useState<FrameResult[]>([])
  const fileRef = useRef<HTMLInputElement>(null)
  const frameContextCache = useRef<Record<string, FrameResult[]>>({})
  const frameContextRequest = useRef(0)
  const aborts = useRef<Record<string, AbortController>>({})
  const multipart = useRef<Record<string, { videoId: string; s3Key: string; uploadId: string }>>({})

  useEffect(() => {
    if (!selectedFrame) return
    if (timeline.some(frame => frameKey(frame) === frameKey(selectedFrame)) && timeline.length > 1) return

    const cacheKey = `${selectedFrame.video_id}:${Math.round(selectedFrame.t_ms / 1000)}`
    const cached = frameContextCache.current[cacheKey]
    if (cached?.length) {
      setTimeline(cached)
      return
    }

    const requestId = ++frameContextRequest.current
    frameContext(selectedFrame.video_id, selectedFrame.chunk_id, selectedFrame.t_ms, 8)
      .then(items => {
        if (requestId !== frameContextRequest.current) return
        const nextTimeline = items.length ? items : [selectedFrame]
        frameContextCache.current[cacheKey] = nextTimeline
        setTimeline(nextTimeline)
      })
      .catch(() => {
        if (requestId === frameContextRequest.current) setTimeline([selectedFrame])
      })
  }, [selectedFrame])

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (!selectedFrame || timeline.length === 0) return
      const index = timeline.findIndex(f => frameKey(f) === frameKey(selectedFrame))
      if (e.key === 'Escape') setSelectedFrame(null)
      if (e.key === 'ArrowLeft') setSelectedFrame(timeline[Math.max(0, index - 1)])
      if (e.key === 'ArrowRight') setSelectedFrame(timeline[Math.min(timeline.length - 1, index + 1)])
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [selectedFrame, timeline])

  function updateUpload(id: string, patch: Partial<UploadItem>) {
    setUploads(prev => prev.map(item => item.id === id ? { ...item, ...patch } : item))
  }

  function enqueue(files: FileList | File[]) {
    const next = Array.from(files)
      .filter(file => file.type.startsWith('video/') || /\.(mp4|mov|m4v|webm)$/i.test(file.name))
      .map(file => ({
        id: crypto.randomUUID(),
        file,
        state: 'queued' as UploadState,
        progress: 0,
        status: 'Queued',
      }))
    setUploads(prev => [...next, ...prev])
    next.forEach(item => void uploadFile(item.id, item.file))
  }

  async function uploadFile(id: string, file: File) {
    updateUpload(id, { state: 'uploading', progress: 2, status: 'Preparing upload', error: undefined })
    const controller = new AbortController()
    aborts.current[id] = controller
    try {
      let videoId: string
      if (file.size >= MULTIPART_THRESHOLD) {
        videoId = await uploadMultipart(id, file, controller)
      } else {
        videoId = await uploadSingle(id, file, controller)
      }
      updateUpload(id, { videoId, state: 'processing', progress: 82, status: 'Processing video chunks' })
      watchProcessing(id, videoId)
    } catch (err) {
      if (controller.signal.aborted) {
        updateUpload(id, { state: 'cancelled', status: 'Cancelled', progress: 0 })
      } else {
        updateUpload(id, {
          state: 'failed',
          status: 'Upload failed',
          error: err instanceof Error ? err.message : 'Upload failed',
        })
      }
    } finally {
      delete aborts.current[id]
    }
  }

  async function uploadSingle(id: string, file: File, controller: AbortController) {
    try {
      const uploadRequest = await presignUpload('general', file.name, file.type || 'video/mp4')
      const uploadHeaders = Object.fromEntries(
        Object.entries(uploadRequest)
          .filter(([key]) => key.startsWith('uploadHeader:'))
          .map(([key, value]) => [key.slice('uploadHeader:'.length), value])
      )
      updateUpload(id, { progress: 20, status: 'Uploading to S3' })
      const uploadRes = await fetch(uploadRequest.presignedPutUrl, {
        method: 'PUT',
        body: file,
        headers: uploadHeaders,
        signal: controller.signal,
      })
      if (!uploadRes.ok) throw new Error(`S3 upload failed (${uploadRes.status})`)
      updateUpload(id, { progress: 72, status: 'Finalizing upload' })
      await finalizeUpload(uploadRequest.videoId, uploadRequest.s3Key, 'general', file.name)
      return uploadRequest.videoId
    } catch (err) {
      if (controller.signal.aborted) throw err
      updateUpload(id, { progress: 20, status: 'Direct upload failed, using backend fallback' })
      const fallback = await uploadViaBackend(file, 'general')
      return fallback.videoId
    }
  }

  async function uploadMultipart(id: string, file: File, controller: AbortController) {
    const init = await initiateMultipartUpload(file, 'general')
    multipart.current[id] = init
    const requestAbort = new AbortController()
    const partSize = Number(init.partSize)
    const parts: Array<{ partNumber: number; eTag: string }> = []
    const totalParts = Math.ceil(file.size / partSize)
    const partProgress = new Array(totalParts).fill(0)
    let nextPartIndex = 0
    let completedParts = 0
    let failed = false

    function updateMultipartProgress(activePart?: number) {
      const uploadedBytes = partProgress.reduce((sum, value) => sum + value, 0)
      const uploadPct = file.size ? uploadedBytes / file.size : 0
      const progress = Math.max(10, Math.min(72, Math.round(10 + uploadPct * 62)))
      updateUpload(id, {
        progress,
        status: activePart
          ? `Uploading ${completedParts}/${totalParts} parts · active ${activePart}`
          : `Uploading ${completedParts}/${totalParts} parts`,
      })
    }

    async function worker() {
      while (!failed && nextPartIndex < totalParts) {
        const partIndex = nextPartIndex
        nextPartIndex += 1
        const partNumber = partIndex + 1
        const start = partIndex * partSize
        const end = Math.min(file.size, (partIndex + 1) * partSize)
        const chunk = file.slice(start, end)
        updateMultipartProgress(partNumber)

        const signed = await presignMultipartPart(init.videoId, init.s3Key, init.uploadId, partNumber)
        const eTag = await uploadPartWithProgress(
          signed.url,
          chunk,
          [controller.signal, requestAbort.signal],
          loaded => {
            partProgress[partIndex] = Math.min(chunk.size, loaded)
            updateMultipartProgress(partNumber)
          }
        )
        partProgress[partIndex] = chunk.size
        parts.push({ partNumber, eTag })
        completedParts += 1
        updateMultipartProgress()
      }
    }

    try {
      const workers = Array.from({ length: Math.min(MULTIPART_CONCURRENCY, totalParts) }, () => worker())
      await Promise.all(workers)
      updateUpload(id, { progress: 74, status: 'Completing multipart upload' })
      await completeMultipartUpload(init.videoId, init.s3Key, init.uploadId, parts, 'general', file.name)
      delete multipart.current[id]
      return init.videoId
    } catch (err) {
      failed = true
      requestAbort.abort()
      await abortMultipartUpload(init.videoId, init.s3Key, init.uploadId)
      delete multipart.current[id]
      throw err
    }
  }

  function watchProcessing(id: string, videoId: string) {
    const es = statusStream(videoId)
    let done = false
    let pollTimer: ReturnType<typeof setInterval> | undefined

    function applyStatus(data: Record<string, string>) {
      const chunkCount = Number(data.chunk_count || 0)
      const indexed = Number(data.indexed_count || 0)
      const failed = Number(data.failed_count || 0)
      const terminal = Number(data.terminal_count || 0)
      const pct = chunkCount ? Math.min(99, 82 + Math.round((terminal / chunkCount) * 17)) : 88
      if (data.stage === 'ready' || data.stage === 'partial') {
        done = true
        es.close()
        if (pollTimer) clearInterval(pollTimer)
        updateUpload(id, {
          state: data.stage,
          progress: 100,
          status: data.stage === 'ready'
            ? `Ready (${indexed}/${chunkCount} chunks indexed)`
            : `Partial (${indexed} indexed, ${failed} failed)`,
        })
      } else {
        updateUpload(id, {
          state: 'processing',
          progress: pct,
          status: chunkCount ? `${data.stage} · ${indexed}/${chunkCount} indexed · ${failed} failed` : data.stage,
        })
      }
    }

    async function pollStatus() {
      if (done) return
      try {
        applyStatus(await videoStatusSnapshot(videoId))
      } catch {
        if (!done) updateUpload(id, { status: 'Processing status unavailable' })
      }
    }

    es.onmessage = (ev) => {
      try {
        applyStatus(JSON.parse(ev.data))
      } catch {
        void pollStatus()
      }
    }
    es.onerror = () => {
      es.close()
      if (!done) updateUpload(id, { status: 'Realtime status lost, polling', state: 'processing' })
    }
    pollTimer = setInterval(() => void pollStatus(), 3000)
    void pollStatus()
  }

  async function cancelUpload(item: UploadItem) {
    aborts.current[item.id]?.abort()
    const activeMultipart = multipart.current[item.id]
    if (activeMultipart) {
      await abortMultipartUpload(activeMultipart.videoId, activeMultipart.s3Key, activeMultipart.uploadId)
      delete multipart.current[item.id]
    }
    updateUpload(item.id, { state: 'cancelled', status: 'Cancelled', progress: 0 })
  }

  async function retryUpload(item: UploadItem) {
    updateUpload(item.id, { state: 'queued', status: 'Queued', progress: 0, error: undefined })
    await uploadFile(item.id, item.file)
  }

  function openFrame(frame: FrameResult) {
    setTimeline(current => current.some(item => frameKey(item) === frameKey(frame)) ? current : [frame])
    setSelectedFrame(frame)
  }

  async function handleSearch(e: React.FormEvent) {
    e.preventDefault()
    if (!query.trim()) return
    setSearching(true)
    setSearchError('')
    setHasSearched(true)
    try {
      const results = await search(query, undefined, minConfidence, 12)
      setFrames(results)
    } catch (err) {
      setSearchError(err instanceof Error ? err.message : 'Search failed')
      setFrames([])
    } finally {
      setSearching(false)
    }
  }

  const activeIndex = selectedFrame ? timeline.findIndex(f => frameKey(f) === frameKey(selectedFrame)) : -1

  return (
    <div className="space-y-5">
      <section
        onDragEnter={(e) => { e.preventDefault(); setDragging(true) }}
        onDragOver={(e) => e.preventDefault()}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => { e.preventDefault(); setDragging(false); enqueue(e.dataTransfer.files) }}
        className={`glass p-4 ${dragging ? 'border-cyan-300/45 bg-cyan-300/10' : ''}`}
      >
        <div className="flex flex-col gap-3 xl:flex-row xl:items-center">
          <form onSubmit={handleSearch} className="flex min-w-0 flex-1 gap-2">
            <input
              value={query}
              onChange={e => setQuery(e.target.value)}
              placeholder="Search across indexed frames"
              className="min-w-0 flex-1 rounded-lg border border-white/10 bg-black/30 px-3 py-2 text-sm text-white placeholder:text-white/35 focus:border-white/35 focus:outline-none"
            />
            <button type="submit" disabled={searching} className="icon-button w-24">
              {searching ? '...' : 'Search'}
            </button>
          </form>
          <label className="glass-sm flex items-center gap-3 px-3 py-2 text-xs text-white/60">
            <span className="whitespace-nowrap">Min confidence</span>
            <input
              type="range"
              min="0.6"
              max="0.95"
              step="0.05"
              value={minConfidence}
              onChange={e => setMinConfidence(Number(e.target.value))}
              className="confidence-slider w-28"
            />
            <span className="w-9 text-right text-cyan-200">{Math.round(minConfidence * 100)}%</span>
          </label>
          <button onClick={() => fileRef.current?.click()} className="icon-button">
            Upload
          </button>
          <input ref={fileRef} type="file" accept="video/*" multiple className="hidden" onChange={e => e.target.files && enqueue(e.target.files)} />
        </div>
      </section>

      {uploads.length > 0 && (
        <section className="space-y-2">
          {uploads.map(item => (
            <div key={item.id} className="glass-sm p-3">
              <div className="flex items-center gap-3">
                <div className="min-w-0 flex-1">
                  <div className="truncate text-sm text-white/90">{item.file.name}</div>
                  <div className="mt-1 text-xs text-white/45">{item.status}</div>
                  {item.error && <div className="mt-1 text-xs text-red-300">{item.error}</div>}
                </div>
                <div className="w-28 text-right text-xs text-white/45">{formatSize(item.file.size)}</div>
                {(item.state === 'uploading' || item.state === 'queued') && (
                  <button className="icon-button h-8 px-3 text-xs" onClick={() => cancelUpload(item)}>Cancel</button>
                )}
                {(item.state === 'failed' || item.state === 'cancelled') && (
                  <button className="icon-button h-8 px-3 text-xs" onClick={() => retryUpload(item)}>Retry</button>
                )}
              </div>
              <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-white/8">
                <div className={`h-full rounded-full ${item.state === 'failed' ? 'bg-red-400' : 'progress-fill'}`} style={{ width: `${item.progress}%` }} />
              </div>
            </div>
          ))}
        </section>
      )}

      {searchError && <div className="glass-sm p-3 text-sm text-red-300">{searchError}</div>}

      {frames.length > 0 ? (
        <section className="space-y-3">
          <div className="flex items-center justify-between gap-3">
            <div className="text-xs uppercase tracking-[0.18em] text-white/35">High confidence frames</div>
            <div className="text-xs text-white/35">{frames.length} above {Math.round(minConfidence * 100)}%</div>
          </div>
          <div className="flex gap-3 overflow-x-auto pb-3">
            {frames.map((frame) => (
              <button key={frameKey(frame)} onClick={() => openFrame(frame)} className="frame-card group w-64 shrink-0 text-left">
                <div className="relative">
                  <img src={frame.url} alt="" className="aspect-video w-full object-cover" />
                  <span className="confidence-badge absolute right-2 top-2">{formatConfidence(frame.confidence)}</span>
                </div>
                <div className="space-y-2 px-2 py-2">
                  <div className="flex items-center justify-between text-xs text-white/55">
                    <span>{formatTime(frame.t_ms)}</span>
                    <span>{frame.video_id.slice(0, 8)}</span>
                  </div>
                  {(frame.main_activity || frame.caption) && (
                    <div className="line-clamp-2 text-xs text-white/72">{frame.main_activity || frame.caption}</div>
                  )}
                  <TagList tags={frame.tags} />
                </div>
              </button>
            ))}
          </div>
        </section>
      ) : (
        !searchError && (
          <div className="glass flex h-64 items-center justify-center text-sm text-white/28">
            {hasSearched ? `No frames passed the ${Math.round(minConfidence * 100)}% confidence threshold` : 'Drop videos here, then search indexed frames'}
          </div>
        )
      )}

      {selectedFrame && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4 backdrop-blur-md" onClick={() => setSelectedFrame(null)}>
          <div className="glass max-h-[92vh] w-full max-w-5xl overflow-hidden" onClick={e => e.stopPropagation()}>
            <div className="flex items-center justify-between border-b border-white/10 px-4 py-3">
              <div className="min-w-0">
                <div className="text-sm text-white/85">{formatTime(selectedFrame.t_ms)} · {selectedFrame.video_id.slice(0, 8)}</div>
                <div className="mt-1 text-xs text-white/38">{selectedFrame.source_file || selectedFrame.scene || 'Indexed frame'}</div>
              </div>
              <div className="flex items-center gap-2">
                <span className="confidence-badge">{formatConfidence(selectedFrame.confidence)}</span>
                <button className="icon-button h-8 px-3 text-xs" onClick={() => setSelectedFrame(null)}>Close</button>
              </div>
            </div>
            <div className="relative bg-black">
              <img src={selectedFrame.url} alt="" className="mx-auto max-h-[62vh] w-full object-contain" />
              <button className="modal-nav left-3" disabled={activeIndex <= 0} onClick={() => setSelectedFrame(timeline[Math.max(0, activeIndex - 1)])}>‹</button>
              <button className="modal-nav right-3" disabled={activeIndex < 0 || activeIndex >= timeline.length - 1} onClick={() => setSelectedFrame(timeline[Math.min(timeline.length - 1, activeIndex + 1)])}>›</button>
            </div>
            <div className="grid gap-3 border-t border-white/10 p-4 text-sm md:grid-cols-[1.2fr_0.8fr]">
              <div>
                <div className="text-white/82">{selectedFrame.caption || selectedFrame.main_activity || 'No caption stored for this frame yet.'}</div>
                {selectedFrame.relevance_reason && (
                  <div className="mt-2 text-xs text-cyan-100/75">{selectedFrame.relevance_reason}</div>
                )}
              </div>
              <div className="space-y-2 text-xs text-white/48">
                <div>{selectedFrame.scene || 'Unknown scene'} · {selectedFrame.motion || 'unknown motion'}</div>
                <TagList tags={selectedFrame.tags} />
              </div>
            </div>
            <div className="flex gap-2 overflow-x-auto border-t border-white/10 p-3">
              {timeline.map(frame => (
                <button
                  key={frameKey(frame)}
                  onClick={() => setSelectedFrame(frame)}
                  className={`h-20 w-36 shrink-0 overflow-hidden rounded-lg border ${frameKey(frame) === frameKey(selectedFrame) ? 'border-white/70' : 'border-white/10'}`}
                >
                  <img src={frame.url} alt="" className="h-full w-full object-cover" />
                </button>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function uploadPartWithProgress(
  url: string,
  chunk: Blob,
  signals: AbortSignal[],
  onProgress: (loaded: number) => void
) {
  return new Promise<string>((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    let settled = false
    const abort = () => {
      if (settled) return
      settled = true
      xhr.abort()
      reject(new Error('Upload cancelled'))
    }

    for (const signal of signals) {
      if (signal.aborted) {
        abort()
        return
      }
      signal.addEventListener('abort', abort, { once: true })
    }

    xhr.upload.onprogress = event => {
      if (event.lengthComputable) onProgress(event.loaded)
    }
    xhr.onload = () => {
      settled = true
      for (const signal of signals) signal.removeEventListener('abort', abort)
      if (xhr.status >= 200 && xhr.status < 300) {
        const eTag = xhr.getResponseHeader('ETag')
        if (!eTag) {
          reject(new Error('Uploaded part missing ETag'))
          return
        }
        resolve(eTag)
      } else {
        reject(new Error(`Part upload failed (${xhr.status})`))
      }
    }
    xhr.onerror = () => {
      settled = true
      for (const signal of signals) signal.removeEventListener('abort', abort)
      reject(new Error('Part upload failed'))
    }
    xhr.onabort = () => {
      if (settled) return
      settled = true
      for (const signal of signals) signal.removeEventListener('abort', abort)
      reject(new Error('Upload cancelled'))
    }
    xhr.open('PUT', url)
    xhr.send(chunk)
  })
}

function frameKey(frame: FrameResult) {
  return `${frame.video_id}:${frame.chunk_id}:${frame.frame_id || frame.t_ms}`
}

function formatTime(ms: number) {
  const total = Math.max(0, Math.round(ms / 1000))
  const minutes = Math.floor(total / 60)
  const seconds = total % 60
  return `${minutes}:${seconds.toString().padStart(2, '0')}`
}

function formatConfidence(value?: number) {
  if (typeof value !== 'number' || Number.isNaN(value)) return 'score --'
  return `${Math.round(value * 100)}%`
}

function formatSize(bytes: number) {
  if (bytes > 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024 / 1024).toFixed(1)} GB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

function TagList({ tags }: { tags?: string[] }) {
  const visible = Array.isArray(tags) ? tags.filter(Boolean).slice(0, 4) : []
  if (visible.length === 0) return null
  return (
    <div className="flex flex-wrap gap-1">
      {visible.map(tag => (
        <span key={tag} className="tag-chip">{tag}</span>
      ))}
    </div>
  )
}
