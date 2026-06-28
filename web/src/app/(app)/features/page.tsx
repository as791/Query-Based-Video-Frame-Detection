'use client'

import { FormEvent, useEffect, useRef, useState } from 'react'
import {
  addDomainLabel,
  createDomain,
  DomainModel,
  DomainSummary,
  FewShotExample,
  FewShotLabel,
  finalizeUpload,
  getDomainModel,
  listFewShotModel,
  listDomains,
  presignUpload,
  statusStream,
  uploadViaBackend,
  videoStatusSnapshot,
} from '@/lib/api'

type ExampleUpload = {
  id: string
  file: File
  label: string
  domainId: string
  progress: number
  status: string
  state: 'uploading' | 'processing' | 'ready' | 'failed'
  videoId?: string
  error?: string
}

export default function FeaturesPage() {
  const [domains, setDomains] = useState<DomainSummary[]>([])
  const [selectedDomainId, setSelectedDomainId] = useState('general')
  const [domainName, setDomainName] = useState('')
  const [domainDescription, setDomainDescription] = useState('')
  const [labels, setLabels] = useState<FewShotLabel[]>([])
  const [examples, setExamples] = useState<FewShotExample[]>([])
  const [domainModel, setDomainModel] = useState<DomainModel | null>(null)
  const [label, setLabel] = useState('')
  const [description, setDescription] = useState('')
  const [selectedLabel, setSelectedLabel] = useState('')
  const [uploads, setUploads] = useState<ExampleUpload[]>([])
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const fileRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    void loadDomains()
  }, [])

  useEffect(() => {
    if (!selectedDomainId) return
    localStorage.setItem('videovault:selectedDomainId', selectedDomainId)
    void refresh()
  }, [selectedDomainId])

  async function loadDomains() {
    setError('')
    try {
      const data = await listDomains()
      const nextDomains = data.domains || []
      setDomains(nextDomains)
      localStorage.setItem('videovault:domainSummaries', JSON.stringify(nextDomains))
      const cached = localStorage.getItem('videovault:selectedDomainId') || ''
      const preferred = nextDomains.find(item => item.domainId === cached)?.domainId || nextDomains[0]?.domainId || 'general'
      setSelectedDomainId(preferred)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load domains')
      setSelectedDomainId(current => current || 'general')
    }
  }

  async function refresh() {
    setLoading(true)
    setError('')
    try {
      const data = await listFewShotModel(selectedDomainId)
      const model = await getDomainModel(selectedDomainId).catch(() => null)
      const nextLabels = data.labels || []
      const nextExamples = await reconcileExamples(data.examples || [])
      setLabels(nextLabels)
      setExamples(nextExamples)
      setDomainModel(model)
      settleUploadsFromExamples(nextExamples)
      if (!nextLabels.some(item => item.label === selectedLabel)) {
        setSelectedLabel(nextLabels[0]?.label || '')
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load few-shot model')
    } finally {
      setLoading(false)
    }
  }

  async function saveDomain(e: FormEvent) {
    e.preventDefault()
    if (!domainName.trim() || saving) return
    setSaving(true)
    setError('')
    try {
      const created = await createDomain(domainName, domainDescription)
      setDomainName('')
      setDomainDescription('')
      await loadDomains()
      setSelectedDomainId(created.domainId)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create domain')
    } finally {
      setSaving(false)
    }
  }

  async function saveLabel(e: FormEvent) {
    e.preventDefault()
    const normalized = normalizeLabel(label)
    if (!normalized || saving) return
    setSaving(true)
    setError('')
    try {
      await addDomainLabel(selectedDomainId, label, description)
      setLabel('')
      setDescription('')
      setSelectedLabel(normalized)
      await refresh()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save label')
    } finally {
      setSaving(false)
    }
  }

  async function uploadExamples(files: FileList | File[]) {
    const activeLabel = selectedLabel || labels[0]?.label || ''
    if (!activeLabel) {
      setError('Add a label before uploading examples')
      return
    }
    const next = Array.from(files).filter(file => file.type.startsWith('video/') || /\.(mp4|mov|avi|mkv|webm)$/i.test(file.name))
    for (const file of next) {
      const id = crypto.randomUUID()
      const item: ExampleUpload = {
        id,
        file,
        label: activeLabel,
        domainId: selectedDomainId,
        progress: 0,
        status: 'Queued',
        state: 'uploading',
      }
      setUploads(prev => [item, ...prev])
      void uploadOne(item)
    }
  }

  async function uploadOne(item: ExampleUpload) {
    updateUpload(item.id, { progress: 4, status: 'Preparing upload' })
    try {
      const signed = await presignUpload('cctv', item.file.name, item.file.type || 'video/mp4')
      updateUpload(item.id, { progress: 18, status: 'Uploading example' })
      const headers = uploadHeaders(signed)
      const put = await fetch(signed.presignedPutUrl, {
        method: 'PUT',
        headers,
        body: item.file,
      })
      if (!put.ok) throw new Error(`S3 upload failed (${put.status})`)
      updateUpload(item.id, { progress: 78, status: 'Indexing example', state: 'processing' })
      await finalizeUpload(signed.videoId, signed.s3Key, 'cctv', item.file.name, item.label, item.domainId)
      updateUpload(item.id, { videoId: signed.videoId })
      watchStatus(item.id, signed.videoId)
      await refresh()
    } catch (err) {
      try {
        updateUpload(item.id, { progress: 18, status: 'Using backend upload' })
        const uploaded = await uploadViaBackend(item.file, 'cctv', item.label, item.domainId)
        updateUpload(item.id, { progress: 78, status: 'Indexing example', state: 'processing', videoId: uploaded.videoId })
        watchStatus(item.id, uploaded.videoId)
        await refresh()
      } catch (fallbackErr) {
        updateUpload(item.id, {
          state: 'failed',
          status: 'Upload failed',
          error: fallbackErr instanceof Error ? fallbackErr.message : err instanceof Error ? err.message : 'Upload failed',
        })
      }
    }
  }

  function watchStatus(id: string, videoId: string) {
    const events = statusStream(videoId)
    let done = false
    let pollTimer: ReturnType<typeof setInterval> | undefined

    function applyStatus(data: Record<string, string>) {
      const chunkCount = Number(data.chunk_count || 0)
      const indexed = Number(data.indexed_count || 0)
      const failed = Number(data.failed_count || 0)
      const terminal = Number(data.terminal_count || 0)
      const pct = chunkCount ? Math.min(99, 78 + Math.round((terminal / chunkCount) * 21)) : 86
      if (data.stage === 'ready' || data.stage === 'partial') {
        done = true
        events.close()
        if (pollTimer) clearInterval(pollTimer)
        if (indexed > 0) {
          updateUpload(id, { progress: 100, state: 'ready', status: data.stage === 'partial' ? `Ready with ${failed} failed chunks` : 'Ready' })
          setTimeout(() => setUploads(prev => prev.filter(item => item.id !== id)), 900)
          void refresh()
        } else {
          updateUpload(id, { state: 'failed', progress: 100, status: 'Indexing failed', error: 'No chunks were indexed successfully' })
        }
      } else if (data.stage === 'failed') {
        done = true
        events.close()
        if (pollTimer) clearInterval(pollTimer)
        updateUpload(id, { state: 'failed', status: 'Indexing failed', error: data.error || 'Processing failed' })
      } else {
        updateUpload(id, { progress: pct, status: data.stage || 'Processing', state: 'processing' })
      }
    }

    async function pollStatus() {
      if (done) return
      try {
        applyStatus(await videoStatusSnapshot(videoId))
      } catch {
        if (!done) updateUpload(id, { status: 'Indexing example' })
      }
    }

    events.onmessage = event => {
      try {
        applyStatus(JSON.parse(event.data) as Record<string, string>)
      } catch {
        void pollStatus()
      }
    }
    events.onerror = () => {
      events.close()
      if (!done) {
        updateUpload(id, { status: 'Indexing example' })
        void pollStatus()
      }
    }
    pollTimer = setInterval(() => void pollStatus(), 3000)
    void pollStatus()
  }

  function updateUpload(id: string, patch: Partial<ExampleUpload>) {
    setUploads(prev => prev.map(item => item.id === id ? { ...item, ...patch } : item))
  }

  async function reconcileExamples(items: FewShotExample[]) {
    const reconciled = await Promise.all(items.map(async item => {
      if (item.status !== 'processing') return item
      try {
        const status = await videoStatusSnapshot(item.videoId)
        const indexed = Number(status.indexed_count || 0)
        const failed = Number(status.failed_count || 0)
        if (status.stage === 'ready') return { ...item, status: 'ready' }
        if (status.stage === 'partial') return { ...item, status: indexed > 0 ? 'partial' : 'failed' }
        if (status.stage === 'failed') return { ...item, status: 'failed' }
        if (failed > 0 && indexed === 0 && Number(status.terminal_count || 0) > 0) return { ...item, status: 'failed' }
        return item
      } catch {
        return item
      }
    }))
    return reconciled
  }

  function settleUploadsFromExamples(items: FewShotExample[]) {
    const byVideo = new Map(items.map(item => [item.videoId, item]))
    setUploads(prev => prev.flatMap(item => {
      if (!item.videoId) return [item]
      const example = byVideo.get(item.videoId)
      if (!example) return [item]
      if (example.status === 'ready' || example.status === 'partial') return []
      if (example.status === 'failed') {
        return [{ ...item, state: 'failed' as const, progress: 100, status: 'Indexing failed', error: 'Example failed during processing' }]
      }
      return [item]
    }))
  }

  const selectedDomain = domains.find(domain => domain.domainId === selectedDomainId)
  const positiveFeedback = domainModel?.feedbackStats?.positive || 0
  const negativeFeedback = domainModel?.feedbackStats?.negative || 0
  const readyLabels = labels.filter(item => item.status === 'ready').length
  const labelsNeedingExamples = labels.filter(item => item.exampleCount < 5).slice(0, 4)
  const topLabels = [...labels].sort((a, b) => b.exampleCount - a.exampleCount).slice(0, 5)
  const onboarding = [
    { label: 'Create a domain', done: domains.length > 0 },
    { label: 'Add search labels', done: labels.length > 0 },
    { label: 'Upload 3-5 examples per label', done: labels.length > 0 && labels.every(item => item.exampleCount >= 3) },
    { label: 'Run searches and rate results', done: positiveFeedback + negativeFeedback > 0 },
  ]

  return (
    <div className="space-y-5">
      <section className="grid gap-4 xl:grid-cols-[minmax(360px,0.9fr)_minmax(460px,1.1fr)]">
        <form onSubmit={saveDomain} className="glass p-4">
          <div className="mb-4 flex items-center justify-between gap-3">
            <div>
              <div className="text-sm font-medium text-white/90">Domains</div>
              <div className="mt-1 text-xs text-white/40">{domains.length} profiles</div>
            </div>
            <button type="button" onClick={loadDomains} disabled={loading || saving} className="icon-button h-9 px-3 text-xs">
              Refresh
            </button>
          </div>
          <div className="grid gap-3">
            <label>
              <span className="mb-1 block text-xs uppercase tracking-[0.16em] text-white/35">Active domain</span>
              <select
                value={selectedDomainId}
                onChange={e => setSelectedDomainId(e.target.value)}
                className="w-full rounded-lg border border-white/10 bg-black/30 px-3 py-2 text-sm text-white focus:border-white/35 focus:outline-none"
              >
                {domains.map(domain => (
                  <option key={domain.domainId} value={domain.domainId}>{domain.domainName}</option>
                ))}
                {domains.length === 0 && <option value="general">General</option>}
              </select>
            </label>
            <label>
              <span className="mb-1 block text-xs uppercase tracking-[0.16em] text-white/35">New domain</span>
              <input
                value={domainName}
                onChange={e => setDomainName(e.target.value)}
                placeholder="Retail theft"
                className="w-full rounded-lg border border-white/10 bg-black/30 px-3 py-2 text-sm text-white placeholder:text-white/35 focus:border-white/35 focus:outline-none"
              />
            </label>
            <label>
              <span className="mb-1 block text-xs uppercase tracking-[0.16em] text-white/35">Context</span>
              <input
                value={domainDescription}
                onChange={e => setDomainDescription(e.target.value)}
                placeholder="shoplifting, aisle behavior, checkout exceptions"
                className="w-full rounded-lg border border-white/10 bg-black/30 px-3 py-2 text-sm text-white placeholder:text-white/35 focus:border-white/35 focus:outline-none"
              />
            </label>
            <button type="submit" disabled={!domainName.trim() || saving} className="icon-button justify-self-start px-5">
              {saving ? '...' : 'Create domain'}
            </button>
          </div>
        </form>

        <div className="glass p-4">
          <div className="flex items-start justify-between gap-3">
            <div>
              <div className="text-sm font-medium text-white/90">Selected model</div>
              <div className="mt-1 text-xs text-white/40">{selectedDomain?.domainName || domainModel?.domainName || selectedDomainId}</div>
            </div>
            {domainModel?.modelVersion && <span className="confidence-badge">{domainModel.modelVersion.slice(0, 8)}</span>}
          </div>
          <div className="mt-3 grid gap-2 md:grid-cols-4">
            <Metric label="Labels" value={labels.length} />
            <Metric label="Examples" value={examples.length} />
            <Metric label="Ready labels" value={readyLabels} />
            <Metric label="Feedback" value={positiveFeedback + negativeFeedback} />
          </div>
          <div className="mt-4 grid gap-3 md:grid-cols-2">
            <div className="glass-sm p-3">
              <div className="text-xs uppercase tracking-[0.16em] text-white/35">Feedback signal</div>
              <div className="mt-2 flex gap-2 text-xs">
                <span className="rounded-full border border-cyan-300/25 bg-cyan-300/10 px-2 py-1 text-cyan-100/80">up {positiveFeedback}</span>
                <span className="rounded-full border border-pink-300/25 bg-pink-300/10 px-2 py-1 text-pink-100/80">down {negativeFeedback}</span>
              </div>
            </div>
            <div className="glass-sm p-3">
              <div className="text-xs uppercase tracking-[0.16em] text-white/35">Learning status</div>
              <div className="mt-2 text-xs leading-5 text-white/55">
                {labelsNeedingExamples.length > 0
                  ? `${labelsNeedingExamples.map(item => item.label).join(', ')} need more examples`
                  : 'Label coverage is ready for review feedback'}
              </div>
            </div>
          </div>
          <div className="mt-4 text-xs leading-5 text-white/45">
            {selectedDomain?.description || 'Domain state is stored server-side and cached locally only for selector speed.'}
          </div>
        </div>
      </section>

      <section className="grid gap-4 xl:grid-cols-[minmax(360px,0.9fr)_minmax(460px,1.1fr)]">
        <div className="glass p-4">
          <div className="text-sm font-medium text-white/90">Domain onboarding</div>
          <div className="mt-3 grid gap-2">
            {onboarding.map(item => (
              <div key={item.label} className="flex items-center gap-3 text-sm">
                <span className={`flex h-6 w-6 items-center justify-center rounded-full border text-xs ${item.done ? 'border-cyan-300/45 bg-cyan-300/12 text-cyan-100' : 'border-white/12 bg-white/5 text-white/35'}`}>
                  {item.done ? '✓' : '•'}
                </span>
                <span className={item.done ? 'text-white/75' : 'text-white/45'}>{item.label}</span>
              </div>
            ))}
          </div>
        </div>
        <div className="glass p-4">
          <div className="text-sm font-medium text-white/90">Learned label coverage</div>
          <div className="mt-3 flex flex-wrap gap-2">
            {topLabels.length > 0 ? topLabels.map(item => (
              <span key={item.label} className="tag-chip">
                {item.label} · {item.exampleCount}
              </span>
            )) : (
              <span className="text-sm text-white/32">Add labels and examples to start adapting this domain.</span>
            )}
          </div>
          <div className="mt-4 text-xs leading-5 text-white/45">
            Search feedback updates positive and negative centroids for this domain; examples provide the few-shot prototype signal.
          </div>
        </div>
      </section>

      <section className="grid gap-4 xl:grid-cols-[minmax(360px,0.9fr)_minmax(460px,1.1fr)]">
        <form onSubmit={saveLabel} className="glass p-4">
          <div className="mb-4 flex items-center justify-between gap-3">
            <div>
              <div className="text-sm font-medium text-white/90">Few-shot labels</div>
              <div className="mt-1 text-xs text-white/40">{labels.length} labels</div>
            </div>
            <button type="button" onClick={refresh} disabled={loading || saving} className="icon-button h-9 px-3 text-xs">
              Refresh
            </button>
          </div>
          <div className="grid gap-3">
            <label>
              <span className="mb-1 block text-xs uppercase tracking-[0.16em] text-white/35">Label</span>
              <input
                value={label}
                onChange={e => setLabel(e.target.value)}
                placeholder="fall"
                className="w-full rounded-lg border border-white/10 bg-black/30 px-3 py-2 text-sm text-white placeholder:text-white/35 focus:border-white/35 focus:outline-none"
              />
            </label>
            <label>
              <span className="mb-1 block text-xs uppercase tracking-[0.16em] text-white/35">Model context</span>
              <input
                value={description}
                onChange={e => setDescription(e.target.value)}
                placeholder="a person falling down or losing balance"
                className="w-full rounded-lg border border-white/10 bg-black/30 px-3 py-2 text-sm text-white placeholder:text-white/35 focus:border-white/35 focus:outline-none"
              />
            </label>
            <button type="submit" disabled={!normalizeLabel(label) || saving} className="icon-button justify-self-start px-5">
              {saving ? '...' : 'Add label'}
            </button>
          </div>
        </form>

        <section
          onDragOver={e => e.preventDefault()}
          onDrop={e => { e.preventDefault(); void uploadExamples(e.dataTransfer.files) }}
          className="glass p-4"
        >
          <div className="flex flex-col gap-3 md:flex-row md:items-end">
            <label className="min-w-0 flex-1">
              <span className="mb-1 block text-xs uppercase tracking-[0.16em] text-white/35">Training label</span>
              <select
                value={selectedLabel}
                onChange={e => setSelectedLabel(e.target.value)}
                className="w-full rounded-lg border border-white/10 bg-black/30 px-3 py-2 text-sm text-white focus:border-white/35 focus:outline-none"
              >
                {labels.map(item => (
                  <option key={item.label} value={item.label}>{item.label}</option>
                ))}
              </select>
            </label>
            <button type="button" onClick={() => fileRef.current?.click()} disabled={!selectedLabel} className="icon-button px-5">
              Upload examples
            </button>
            <input
              ref={fileRef}
              type="file"
              accept="video/*"
              multiple
              className="hidden"
              onChange={e => e.target.files && void uploadExamples(e.target.files)}
            />
          </div>
          <div className="mt-4 grid gap-2 md:grid-cols-3">
            <Metric label="Labels" value={labels.length} />
            <Metric label="Examples" value={examples.length} />
            <Metric label="Ready labels" value={labels.filter(item => item.status === 'ready').length} />
          </div>
        </section>
      </section>

      {error && <div className="glass-sm p-3 text-sm text-red-300">{error}</div>}

      <section className="glass p-4">
        <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <div>
            <div className="text-sm font-medium text-white/90">Search quality signals</div>
            <div className="mt-1 text-xs text-white/40">Use this to decide what examples or feedback to add next.</div>
          </div>
          <a href="/library" className="icon-button h-9 px-3 text-xs">Review results</a>
        </div>
        <div className="mt-4 grid gap-3 md:grid-cols-3">
          <QualityCard
            title="Low coverage labels"
            value={labelsNeedingExamples.length}
            detail={labelsNeedingExamples.length ? labelsNeedingExamples.map(item => item.label).join(', ') : 'All labels have initial examples'}
          />
          <QualityCard
            title="Feedback balance"
            value={`${positiveFeedback}/${negativeFeedback}`}
            detail="thumbs up / thumbs down"
          />
          <QualityCard
            title="Next action"
            value={labels.length === 0 ? 'labels' : examples.length < labels.length * 3 ? 'examples' : 'review'}
            detail={labels.length === 0 ? 'Add labels for this domain' : examples.length < labels.length * 3 ? 'Upload more labeled clips' : 'Rate search results in Library'}
          />
        </div>
      </section>

      <section className="grid gap-4 xl:grid-cols-[minmax(360px,0.9fr)_minmax(460px,1.1fr)]">
        <div className="space-y-3">
          <div className="text-xs uppercase tracking-[0.18em] text-white/35">Models</div>
          {labels.length > 0 ? (
            <div className="grid gap-3">
              {labels.map(item => (
                <button
                  key={item.label}
                  onClick={() => setSelectedLabel(item.label)}
                  className={`glass-sm p-3 text-left transition-colors ${selectedLabel === item.label ? 'border-cyan-300/45 bg-cyan-300/10' : 'hover:border-white/22'}`}
                >
                  <div className="flex items-center justify-between gap-3">
                    <div className="truncate text-sm font-medium text-white/90">{item.label}</div>
                    <span className="confidence-badge">{item.exampleCount}</span>
                  </div>
                  <div className="mt-2 line-clamp-2 min-h-[2.5rem] text-xs leading-5 text-white/50">{item.description || 'No context'}</div>
                  <div className="mt-2 text-xs text-cyan-100/70">{formatStatus(item.status)}</div>
                </button>
              ))}
            </div>
          ) : (
            <div className="glass flex h-44 items-center justify-center text-sm text-white/28">No few-shot labels</div>
          )}
        </div>

        <div className="space-y-3">
          <div className="text-xs uppercase tracking-[0.18em] text-white/35">Examples</div>
          {uploads.map(item => (
            <div key={item.id} className="glass-sm p-3">
              <div className="flex items-center gap-3">
                <div className="min-w-0 flex-1">
                  <div className="truncate text-sm text-white/90">{item.file.name}</div>
                  <div className="mt-1 text-xs text-white/45">{item.label} · {item.status}</div>
                  {item.error && <div className="mt-1 text-xs text-red-300">{item.error}</div>}
                </div>
                <span className="text-xs text-white/40">{item.progress}%</span>
              </div>
              <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-white/8">
                <div className={`h-full rounded-full ${item.state === 'failed' ? 'bg-red-400' : 'progress-fill'}`} style={{ width: `${item.progress}%` }} />
              </div>
            </div>
          ))}
          {examples.map(example => (
            <div key={example.videoId} className="glass-sm flex items-center justify-between gap-3 p-3">
              <div className="min-w-0">
                <div className="truncate text-sm text-white/85">{example.sourceFile || example.videoId}</div>
                <div className="mt-1 text-xs text-white/45">{example.label} · {example.status}</div>
              </div>
              <span className="text-xs text-white/35">{shortDate(example.createdAt)}</span>
            </div>
          ))}
          {uploads.length === 0 && examples.length === 0 && (
            <div className="glass flex h-44 items-center justify-center text-sm text-white/28">No examples uploaded</div>
          )}
        </div>
      </section>
    </div>
  )
}

function Metric({ label, value }: { label: string; value: number }) {
  return (
    <div className="glass-sm px-3 py-2">
      <div className="text-lg font-semibold text-white/90">{value}</div>
      <div className="text-xs text-white/40">{label}</div>
    </div>
  )
}

function QualityCard({ title, value, detail }: { title: string; value: string | number; detail: string }) {
  return (
    <div className="glass-sm p-3">
      <div className="text-xs uppercase tracking-[0.16em] text-white/35">{title}</div>
      <div className="mt-2 text-xl font-semibold text-white/90">{value}</div>
      <div className="mt-1 line-clamp-2 text-xs leading-5 text-white/45">{detail}</div>
    </div>
  )
}

function uploadHeaders(response: Record<string, string>) {
  const headers: Record<string, string> = {}
  for (const [key, value] of Object.entries(response)) {
    if (key.startsWith('uploadHeader:')) headers[key.slice('uploadHeader:'.length)] = value
  }
  return headers
}

function normalizeLabel(value: string) {
  return value.toLowerCase().replace(/[^a-z0-9]/g, '')
}

function formatStatus(value: string) {
  if (value === 'ready') return 'ready'
  if (value === 'partial') return 'partial'
  if (value === 'failed') return 'failed'
  if (value === 'processing') return 'processing'
  if (value === 'needs_more_examples') return 'needs more examples'
  return 'empty'
}

function shortDate(value: string) {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return ''
  return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}
