'use client'
import { useEffect, useMemo, useRef, useState } from 'react'
import {
  abortMultipartUpload,
  completeMultipartUpload,
  DomainSummary,
  finalizeUpload,
  frameContext,
  FrameResult,
  initiateMultipartUpload,
  listDomains,
  presignMultipartPart,
  presignUpload,
  search,
  statusStream,
  submitSearchFeedback,
  undoSearchFeedback,
  uploadViaBackend,
  videoStatusSnapshot,
} from '@/lib/api'

type UploadState = 'queued' | 'uploading' | 'processing' | 'ready' | 'partial' | 'failed' | 'cancelled'
type ResultLayout = 'grid' | 'list'
type FeedbackFilter = 'all' | 'unrated' | 'liked' | 'disliked'
type UploadItem = {
  id: string
  file: File
  domainId: string
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
  const [domains, setDomains] = useState<DomainSummary[]>([])
  const [selectedDomainId, setSelectedDomainId] = useState('general')
  const [uploads, setUploads] = useState<UploadItem[]>([])
  const [dragging, setDragging] = useState(false)
  const [searching, setSearching] = useState(false)
  const [searchError, setSearchError] = useState('')
  const [hasSearched, setHasSearched] = useState(false)
  const [minConfidence, setMinConfidence] = useState(0.8)
  const [selectedFrame, setSelectedFrame] = useState<FrameResult | null>(null)
  const [timeline, setTimeline] = useState<FrameResult[]>([])
  const [feedbackState, setFeedbackState] = useState<Record<string, string>>({})
  const [feedbackNotice, setFeedbackNotice] = useState<Record<string, string>>({})
  const [resultLayout, setResultLayout] = useState<ResultLayout>('grid')
  const [feedbackFilter, setFeedbackFilter] = useState<FeedbackFilter>('all')
  const [actionFilter, setActionFilter] = useState('all')
  const [videoFilter, setVideoFilter] = useState('all')
  const [activeFrameKey, setActiveFrameKey] = useState('')
  const fileRef = useRef<HTMLInputElement>(null)
  const frameContextCache = useRef<Record<string, FrameResult[]>>({})
  const frameContextRequest = useRef(0)
  const aborts = useRef<Record<string, AbortController>>({})
  const multipart = useRef<Record<string, { videoId: string; s3Key: string; uploadId: string }>>({})
  const feedbackNoticeTimers = useRef<Record<string, ReturnType<typeof setTimeout>>>({})
  const selectedDomain = domains.find(domain => domain.domainId === selectedDomainId)
  const actionOptions = useMemo(() => {
    const values = new Set<string>()
    frames.forEach(frame => (frame.action_labels || []).forEach(label => label && values.add(label)))
    return Array.from(values).sort()
  }, [frames])
  const videoOptions = useMemo(() => Array.from(new Set(frames.map(frame => frame.video_id).filter(Boolean))).sort(), [frames])
  const visibleFrames = useMemo(() => {
    return frames.filter(frame => {
      const key = frameKey(frame)
      const liked = feedbackState[`${key}:yes`] === 'saved'
      const disliked = feedbackState[`${key}:no`] === 'saved'
      if (feedbackFilter === 'liked' && !liked) return false
      if (feedbackFilter === 'disliked' && !disliked) return false
      if (feedbackFilter === 'unrated' && (liked || disliked)) return false
      if (actionFilter !== 'all' && !(frame.action_labels || []).includes(actionFilter)) return false
      if (videoFilter !== 'all' && frame.video_id !== videoFilter) return false
      return true
    })
  }, [actionFilter, feedbackFilter, feedbackState, frames, videoFilter])
  const activeVisibleIndex = visibleFrames.findIndex(frame => frameKey(frame) === activeFrameKey)

  useEffect(() => {
    void loadDomains()
    const cachedLayout = localStorage.getItem('videovault:resultLayout')
    if (cachedLayout === 'grid' || cachedLayout === 'list') setResultLayout(cachedLayout)
  }, [])

  useEffect(() => {
    localStorage.setItem('videovault:resultLayout', resultLayout)
  }, [resultLayout])

  useEffect(() => {
    return () => {
      Object.values(feedbackNoticeTimers.current).forEach(clearTimeout)
    }
  }, [])

  useEffect(() => {
    if (!selectedDomainId) return
    localStorage.setItem('videovault:selectedDomainId', selectedDomainId)
  }, [selectedDomainId])

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
        domainId: selectedDomainId,
        state: 'queued' as UploadState,
        progress: 0,
        status: 'Queued',
      }))
    setUploads(prev => [...next, ...prev])
    next.forEach(item => void uploadFile(item.id, item.file, item.domainId))
  }

  async function loadDomains() {
    try {
      const data = await listDomains()
      const nextDomains = data.domains || []
      setDomains(nextDomains)
      localStorage.setItem('videovault:domainSummaries', JSON.stringify(nextDomains))
      const cached = localStorage.getItem('videovault:selectedDomainId') || ''
      const preferred = nextDomains.find(item => item.domainId === cached)?.domainId || nextDomains[0]?.domainId || 'general'
      setSelectedDomainId(preferred)
    } catch {
      const cachedSummary = localStorage.getItem('videovault:domainSummaries')
      if (cachedSummary) {
        try {
          const parsed = JSON.parse(cachedSummary) as DomainSummary[]
          setDomains(parsed)
          setSelectedDomainId(parsed[0]?.domainId || 'general')
        } catch {
          setSelectedDomainId('general')
        }
      }
    }
  }

  async function uploadFile(id: string, file: File, domainId: string) {
    updateUpload(id, { state: 'uploading', progress: 2, status: 'Preparing upload', error: undefined })
    const controller = new AbortController()
    aborts.current[id] = controller
    try {
      let videoId: string
      if (file.size >= MULTIPART_THRESHOLD) {
        videoId = await uploadMultipart(id, file, controller, domainId)
      } else {
        videoId = await uploadSingle(id, file, controller, domainId)
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

  async function uploadSingle(id: string, file: File, controller: AbortController, domainId: string) {
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
      await finalizeUpload(uploadRequest.videoId, uploadRequest.s3Key, 'general', file.name, '', domainId)
      return uploadRequest.videoId
    } catch (err) {
      if (controller.signal.aborted) throw err
      updateUpload(id, { progress: 20, status: 'Direct upload failed, using backend fallback' })
      const fallback = await uploadViaBackend(file, 'general', '', domainId)
      return fallback.videoId
    }
  }

  async function uploadMultipart(id: string, file: File, controller: AbortController, domainId: string) {
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
      await completeMultipartUpload(init.videoId, init.s3Key, init.uploadId, parts, 'general', file.name, '', domainId)
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
    await uploadFile(item.id, item.file, item.domainId)
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
      const scopedVideoId = videoFilter !== 'all' ? videoFilter : undefined
      const results = await search(query, scopedVideoId, minConfidence, 24, selectedDomainId)
      setFrames(results)
      setActiveFrameKey(results[0] ? frameKey(results[0]) : '')
    } catch (err) {
      setSearchError(err instanceof Error ? err.message : 'Search failed')
      setFrames([])
      setActiveFrameKey('')
    } finally {
      setSearching(false)
    }
  }

  async function handleFeedback(frame: FrameResult, relevant: boolean) {
    const key = `${frameKey(frame)}:${relevant ? 'yes' : 'no'}`
    const oppositeKey = `${frameKey(frame)}:${relevant ? 'no' : 'yes'}`
    const noticeKey = frameKey(frame)
    setFeedbackState(prev => ({ ...prev, [key]: 'saving' }))
    setFeedbackNotice(prev => {
      const next = { ...prev }
      delete next[noticeKey]
      return next
    })
    try {
      await submitSearchFeedback({
        domainId: selectedDomainId,
        query,
        videoId: frame.video_id,
        frameId: frame.frame_id,
        chunkId: frame.chunk_id,
        relevant,
      })
      setFeedbackState(prev => {
        const next = { ...prev, [key]: 'saved' }
        delete next[oppositeKey]
        return next
      })
      setFeedbackNotice(prev => ({ ...prev, [noticeKey]: relevant ? 'Saved. Future results adapt to this domain.' : 'Saved. Similar results will rank lower.' }))
      if (feedbackNoticeTimers.current[noticeKey]) clearTimeout(feedbackNoticeTimers.current[noticeKey])
      feedbackNoticeTimers.current[noticeKey] = setTimeout(() => {
        setFeedbackNotice(prev => {
          const next = { ...prev }
          delete next[noticeKey]
          return next
        })
        delete feedbackNoticeTimers.current[noticeKey]
      }, 1800)
    } catch (err) {
      setFeedbackState(prev => ({ ...prev, [key]: err instanceof Error ? err.message : 'failed' }))
      setFeedbackNotice(prev => ({ ...prev, [noticeKey]: 'Feedback failed' }))
    }
  }

  async function clearFeedback(frame: FrameResult) {
    const key = frameKey(frame)
    const relevant = feedbackState[`${key}:yes`] === 'saved'
    const hasVote = relevant || feedbackState[`${key}:no`] === 'saved'
    if (!hasVote) return
    setFeedbackNotice(prev => ({ ...prev, [key]: 'Undoing feedback...' }))
    try {
      await undoSearchFeedback({
        domainId: selectedDomainId,
        query,
        videoId: frame.video_id,
        frameId: frame.frame_id,
        chunkId: frame.chunk_id,
        relevant,
      })
    } catch (err) {
      setFeedbackNotice(prev => ({ ...prev, [key]: err instanceof Error ? err.message : 'Undo failed' }))
      return
    }
    setFeedbackState(prev => {
      const next = { ...prev }
      delete next[`${key}:yes`]
      delete next[`${key}:no`]
      return next
    })
    setFeedbackNotice(prev => ({ ...prev, [key]: 'Feedback undone' }))
    if (feedbackNoticeTimers.current[key]) clearTimeout(feedbackNoticeTimers.current[key])
    feedbackNoticeTimers.current[key] = setTimeout(() => {
      setFeedbackNotice(prev => {
        const next = { ...prev }
        delete next[key]
        return next
      })
      delete feedbackNoticeTimers.current[key]
    }, 1400)
  }

  function feedbackButtonClass(frame: FrameResult, relevant: boolean) {
    const state = feedbackState[`${frameKey(frame)}:${relevant ? 'yes' : 'no'}`]
    const selected = state === 'saved'
    const error = state && state !== 'saving' && state !== 'saved'
    return [
      'icon-button h-8 min-h-8 flex-1 px-2 text-base leading-none',
      selected ? 'border-cyan-300/55 bg-cyan-300/16 text-cyan-100 shadow-[0_0_18px_rgba(103,232,249,0.13)]' : '',
      error ? 'border-red-300/45 bg-red-400/12 text-red-200' : '',
    ].filter(Boolean).join(' ')
  }

  useEffect(() => {
    if (visibleFrames.length === 0) return
    if (!activeFrameKey || !visibleFrames.some(frame => frameKey(frame) === activeFrameKey)) {
      setActiveFrameKey(frameKey(visibleFrames[0]))
    }
  }, [activeFrameKey, visibleFrames])

  useEffect(() => {
    function onReviewKey(e: KeyboardEvent) {
      const target = e.target as HTMLElement | null
      if (target && ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName)) return
      if (selectedFrame) return
      if (visibleFrames.length === 0) return
      const currentIndex = activeVisibleIndex >= 0 ? activeVisibleIndex : 0
      if (e.key === 'j' || e.key === 'ArrowDown') {
        e.preventDefault()
        setActiveFrameKey(frameKey(visibleFrames[Math.min(visibleFrames.length - 1, currentIndex + 1)]))
      } else if (e.key === 'k' || e.key === 'ArrowUp') {
        e.preventDefault()
        setActiveFrameKey(frameKey(visibleFrames[Math.max(0, currentIndex - 1)]))
      } else if (e.key === 'u') {
        e.preventDefault()
        void handleFeedback(visibleFrames[currentIndex], true)
      } else if (e.key === 'd') {
        e.preventDefault()
        void handleFeedback(visibleFrames[currentIndex], false)
      } else if (e.key === 'Enter') {
        e.preventDefault()
        openFrame(visibleFrames[currentIndex])
      }
    }
    window.addEventListener('keydown', onReviewKey)
    return () => window.removeEventListener('keydown', onReviewKey)
  }, [activeVisibleIndex, selectedFrame, visibleFrames])

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
          <label className="glass-sm min-w-[12rem] px-3 py-2 text-xs text-white/60">
            <span className="mb-1 block whitespace-nowrap">Domain</span>
            <select
              value={selectedDomainId}
              onChange={e => setSelectedDomainId(e.target.value)}
              className="w-full rounded-md border border-white/10 bg-black/35 px-2 py-1 text-xs text-white focus:border-white/35 focus:outline-none"
            >
              {domains.map(domain => (
                <option key={domain.domainId} value={domain.domainId}>{domain.domainName}</option>
              ))}
              {domains.length === 0 && <option value="general">General</option>}
            </select>
          </label>
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

      <section className="grid gap-3 md:grid-cols-[minmax(0,1fr)_auto]">
        <div className="glass-sm p-3">
          <div className="flex flex-wrap items-center gap-2 text-xs text-white/45">
            <span className="text-white/75">{selectedDomain?.domainName || 'General'}</span>
            <span>{selectedDomain?.labelCount || 0} labels</span>
            <span>{selectedDomain?.exampleCount || 0} examples</span>
            {selectedDomain?.modelVersion && <span>model {selectedDomain.modelVersion.slice(0, 8)}</span>}
            {selectedDomain?.updatedAt && <span>updated {shortDateTime(selectedDomain.updatedAt)}</span>}
          </div>
          <div className="mt-2 text-xs text-cyan-100/70">
            Feedback and examples in this domain influence future ranking.
          </div>
        </div>
        <a href="/features" className="icon-button h-full px-4 text-xs">
          Manage learning
        </a>
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
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <div className="text-xs uppercase tracking-[0.18em] text-white/35">High confidence frames</div>
              <div className="mt-1 text-xs text-white/35">{visibleFrames.length}/{frames.length} shown above {Math.round(minConfidence * 100)}%</div>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <label className="glass-sm px-2 py-1 text-xs text-white/45">
                <span className="sr-only">Feedback filter</span>
                <select value={feedbackFilter} onChange={e => setFeedbackFilter(e.target.value as FeedbackFilter)} className="bg-transparent text-xs text-white focus:outline-none">
                  <option value="all">All feedback</option>
                  <option value="unrated">Unrated</option>
                  <option value="liked">Thumbs up</option>
                  <option value="disliked">Thumbs down</option>
                </select>
              </label>
              <label className="glass-sm px-2 py-1 text-xs text-white/45">
                <span className="sr-only">Action filter</span>
                <select value={actionFilter} onChange={e => setActionFilter(e.target.value)} className="max-w-[9rem] bg-transparent text-xs text-white focus:outline-none">
                  <option value="all">All actions</option>
                  {actionOptions.map(action => <option key={action} value={action}>{action}</option>)}
                </select>
              </label>
              <label className="glass-sm px-2 py-1 text-xs text-white/45">
                <span className="sr-only">Video filter</span>
                <select value={videoFilter} onChange={e => setVideoFilter(e.target.value)} className="max-w-[9rem] bg-transparent text-xs text-white focus:outline-none">
                  <option value="all">All videos</option>
                  {videoOptions.map(video => <option key={video} value={video}>{video.slice(0, 8)}</option>)}
                </select>
              </label>
              <div className="glass-sm inline-flex w-fit gap-1 p-1" aria-label="Result layout">
                <button
                  type="button"
                  onClick={() => setResultLayout('grid')}
                  className={`flex h-8 w-9 items-center justify-center rounded-md text-sm transition-colors ${resultLayout === 'grid' ? 'bg-cyan-300/18 text-cyan-100' : 'text-white/45 hover:bg-white/8 hover:text-white/80'}`}
                  aria-label="Grid view"
                  title="Grid view"
                >
                  ▦
                </button>
                <button
                  type="button"
                  onClick={() => setResultLayout('list')}
                  className={`flex h-8 w-9 items-center justify-center rounded-md text-sm transition-colors ${resultLayout === 'list' ? 'bg-cyan-300/18 text-cyan-100' : 'text-white/45 hover:bg-white/8 hover:text-white/80'}`}
                  aria-label="List view"
                  title="List view"
                >
                  ☰
                </button>
              </div>
            </div>
          </div>
          <div className="glass-sm flex flex-wrap items-center gap-2 px-3 py-2 text-xs text-white/45">
            <span>Review:</span>
            <span>j/k move</span>
            <span>u thumbs up</span>
            <span>d thumbs down</span>
            <span>enter open</span>
          </div>
          <div className={resultLayout === 'grid' ? 'grid gap-3 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4' : 'space-y-2'}>
            {visibleFrames.map((frame) => (
              <div
                key={frameKey(frame)}
                className={`frame-card group relative text-left ${frameKey(frame) === activeFrameKey ? 'frame-card-active' : ''} ${resultLayout === 'list' ? 'grid sm:grid-cols-[minmax(0,1fr)_6.5rem]' : ''}`}
                onMouseEnter={() => setActiveFrameKey(frameKey(frame))}
              >
                <button type="button" onClick={() => { setActiveFrameKey(frameKey(frame)); openFrame(frame) }} className={`w-full text-left ${resultLayout === 'list' ? 'grid grid-cols-[9rem_minmax(0,1fr)] sm:grid-cols-[12rem_minmax(0,1fr)]' : 'block'}`}>
                  <div className="relative">
                    <img src={frame.url} alt="" className={`${resultLayout === 'list' ? 'h-full min-h-28' : 'aspect-video'} w-full object-cover`} />
                    <span className="confidence-badge absolute right-2 top-2">{formatConfidence(frame.confidence)}</span>
                  </div>
                  <div className="min-w-0 space-y-2 px-2 py-2">
                    <div className="flex items-center justify-between text-xs text-white/55">
                      <span>{formatTime(frame.t_ms)}</span>
                      <span>{frame.video_id.slice(0, 8)}</span>
                    </div>
                    {(frame.main_activity || frame.caption) && (
                      <div className={`${resultLayout === 'list' ? 'line-clamp-1' : 'line-clamp-2'} text-xs text-white/72`}>{frame.main_activity || frame.caption}</div>
                    )}
                    <MatchExplanation frame={frame} query={query} compact={resultLayout === 'list'} />
                    <TagList tags={frame.action_labels} tone="cyan" />
                    {resultLayout === 'grid' && <TagList tags={frame.tags} />}
                  </div>
                </button>
                <div className={`${resultLayout === 'list' ? 'justify-end px-2 pb-2 sm:flex-col sm:justify-center sm:px-2 sm:py-2' : 'px-2 pb-2'} flex gap-2`}>
                  <button
                    type="button"
                    onClick={() => void handleFeedback(frame, true)}
                    disabled={feedbackState[`${frameKey(frame)}:yes`] === 'saving'}
                    className={feedbackButtonClass(frame, true)}
                    aria-label="Thumbs up"
                    title="Thumbs up"
                  >
                    <span aria-hidden="true">{feedbackState[`${frameKey(frame)}:yes`] === 'saving' ? '...' : '👍'}</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => void handleFeedback(frame, false)}
                    disabled={feedbackState[`${frameKey(frame)}:no`] === 'saving'}
                    className={feedbackButtonClass(frame, false)}
                    aria-label="Thumbs down"
                    title="Thumbs down"
                  >
                    <span aria-hidden="true">{feedbackState[`${frameKey(frame)}:no`] === 'saving' ? '...' : '👎'}</span>
                  </button>
                  {(feedbackState[`${frameKey(frame)}:yes`] === 'saved' || feedbackState[`${frameKey(frame)}:no`] === 'saved') && (
                    <button
                      type="button"
                      onClick={() => void clearFeedback(frame)}
                      className="icon-button h-8 min-h-8 px-2 text-xs"
                      title="Undo feedback"
                    >
                      Undo
                    </button>
                  )}
                </div>
                {feedbackNotice[frameKey(frame)] && (
                  <div className="absolute bottom-2 right-2 rounded-full border border-cyan-300/30 bg-black/72 px-2 py-1 text-[0.68rem] font-medium text-cyan-100 shadow-lg">
                    {feedbackNotice[frameKey(frame)]}
                  </div>
                )}
              </div>
            ))}
          </div>
          {visibleFrames.length === 0 && (
            <div className="glass flex h-44 flex-col items-center justify-center gap-3 text-sm text-white/35">
              <span>No results match the active filters.</span>
              <button type="button" onClick={() => { setFeedbackFilter('all'); setActionFilter('all'); setVideoFilter('all') }} className="icon-button h-8 min-h-8 px-3 text-xs">
                Clear filters
              </button>
            </div>
          )}
        </section>
      ) : (
        !searchError && (
          <div className="glass flex min-h-64 flex-col items-center justify-center gap-3 p-4 text-center text-sm text-white/35">
            <div>{hasSearched ? `No frames passed the ${Math.round(minConfidence * 100)}% confidence threshold` : 'Drop videos here, then search indexed frames'}</div>
            {hasSearched ? (
              <div className="flex flex-wrap justify-center gap-2">
                <button type="button" onClick={() => setMinConfidence(0.6)} className="icon-button h-8 min-h-8 px-3 text-xs">Lower confidence</button>
                <button type="button" onClick={() => setSelectedDomainId('general')} className="icon-button h-8 min-h-8 px-3 text-xs">Search general</button>
                <a href="/features" className="icon-button h-8 min-h-8 px-3 text-xs">Add examples</a>
              </div>
            ) : (
              <a href="/features" className="icon-button h-8 min-h-8 px-3 text-xs">Set up a domain</a>
            )}
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
                <button
                  className="icon-button h-8 px-3 text-xs"
                  onClick={() => {
                    setVideoFilter(selectedFrame.video_id)
                    setSelectedFrame(null)
                  }}
                >
                  This video
                </button>
                <button className="icon-button h-8 px-3 text-xs" onClick={() => setSelectedFrame(null)}>Close</button>
              </div>
            </div>
            <div className="relative bg-black">
              {selectedFrame.clip_url ? (
                <video
                  key={selectedFrame.clip_url}
                  src={selectedFrame.clip_url}
                  controls
                  autoPlay
                  muted
                  loop
                  playsInline
                  poster={selectedFrame.url}
                  className="mx-auto max-h-[62vh] w-full bg-black object-contain"
                />
              ) : (
                <img src={selectedFrame.url} alt="" className="mx-auto max-h-[62vh] w-full object-contain" />
              )}
              <button className="modal-nav left-3" disabled={activeIndex <= 0} onClick={() => setSelectedFrame(timeline[Math.max(0, activeIndex - 1)])}>‹</button>
              <button className="modal-nav right-3" disabled={activeIndex < 0 || activeIndex >= timeline.length - 1} onClick={() => setSelectedFrame(timeline[Math.min(timeline.length - 1, activeIndex + 1)])}>›</button>
            </div>
            <div className="grid gap-3 border-t border-white/10 p-4 text-sm md:grid-cols-[1.2fr_0.8fr]">
              <div>
                <div className="text-white/82">{selectedFrame.caption || selectedFrame.main_activity || 'No caption stored for this frame yet.'}</div>
                <div className="mt-3">
                  <MatchExplanation frame={selectedFrame} query={query} compact />
                </div>
                {selectedFrame.relevance_reason && (
                  <div className="mt-2 text-xs text-cyan-100/75">{selectedFrame.relevance_reason}</div>
                )}
              </div>
              <div className="space-y-2 text-xs text-white/48">
                <div>{selectedFrame.scene || 'Unknown scene'} · {selectedFrame.motion || 'unknown motion'}</div>
                <div className="grid grid-cols-2 gap-2">
                  <MetricMini label="Domain" value={formatConfidence(selectedFrame.domain_score)} />
                  <MetricMini label="Feedback" value={formatConfidence(selectedFrame.feedback_score)} />
                  <MetricMini label="Visual" value={formatConfidence(selectedFrame.vector_score || selectedFrame.initial_score)} />
                  <MetricMini label="Action" value={formatConfidence(selectedFrame.action_score)} />
                </div>
                <TagList tags={selectedFrame.action_labels} tone="cyan" />
                <TagList tags={selectedFrame.tags} />
                <TagList tags={selectedFrame.objects} />
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

function scorePercent(value?: number) {
  if (typeof value !== 'number' || Number.isNaN(value) || value <= 0) return ''
  return `${Math.round(value * 100)}%`
}

function formatSize(bytes: number) {
  if (bytes > 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024 / 1024).toFixed(1)} GB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

function shortDateTime(value: string) {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return ''
  return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}

function MetricMini({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-white/10 bg-white/[0.04] px-2 py-1">
      <div className="text-[0.65rem] uppercase tracking-[0.12em] text-white/30">{label}</div>
      <div className="mt-0.5 text-xs text-white/76">{value}</div>
    </div>
  )
}

function matchExplanations(frame: FrameResult, query: string) {
  const items: string[] = []
  const action = frame.action_top || frame.action_labels?.[0]
  if (action) items.push(`Action ${action}`)

  const domainScore = scorePercent(frame.domain_score)
  if (domainScore) items.push(`Domain ${domainScore}`)

  const feedbackScore = scorePercent(frame.feedback_score)
  if (feedbackScore) items.push(`Feedback ${feedbackScore}`)

  const visualScore = scorePercent(frame.vector_score || frame.initial_score)
  if (visualScore && items.length < 3) items.push(`Visual ${visualScore}`)

  const searchableText = `${frame.caption || ''} ${frame.main_activity || ''} ${frame.tags?.join(' ') || ''}`.toLowerCase()
  const queryTokens = query.toLowerCase().split(/\s+/).filter(token => token.length > 2)
  if (queryTokens.some(token => searchableText.includes(token))) items.push('Caption match')

  if (frame.relevance_reason && items.length < 4) items.push(frame.relevance_reason)
  return items.slice(0, 4)
}

function MatchExplanation({ frame, query, compact = false }: { frame: FrameResult; query: string; compact?: boolean }) {
  const items = matchExplanations(frame, query)
  if (items.length === 0) return null
  return (
    <div className={compact ? 'flex flex-wrap gap-1' : 'space-y-1'}>
      {items.map(item => (
        <span key={item} className="match-chip">{item}</span>
      ))}
    </div>
  )
}

function TagList({ tags, tone = 'default' }: { tags?: string[]; tone?: 'default' | 'cyan' }) {
  const visible = Array.isArray(tags) ? tags.filter(Boolean).slice(0, 4) : []
  if (visible.length === 0) return null
  return (
    <div className="flex flex-wrap gap-1">
      {visible.map(tag => (
        <span key={tag} className={`tag-chip ${tone === 'cyan' ? 'border-cyan-300/25 bg-cyan-300/10 text-cyan-100/80' : ''}`}>{tag}</span>
      ))}
    </div>
  )
}
