'use client'

import { FormEvent, useEffect, useMemo, useState } from 'react'
import { ActionLabel, addActionLabel, listActionLabels } from '@/lib/api'

export default function LabelsPage() {
  const [labels, setLabels] = useState<ActionLabel[]>([])
  const [label, setLabel] = useState('')
  const [description, setDescription] = useState('')
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')

  const normalized = useMemo(() => normalizeLabel(label), [label])

  useEffect(() => {
    refresh()
  }, [])

  async function refresh() {
    setLoading(true)
    setError('')
    try {
      const data = await listActionLabels()
      setLabels(data.labels || [])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load labels')
    } finally {
      setLoading(false)
    }
  }

  async function save(e: FormEvent) {
    e.preventDefault()
    if (!normalized || saving) return
    setSaving(true)
    setError('')
    try {
      await addActionLabel(label, description)
      setLabel('')
      setDescription('')
      await refresh()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save label')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="space-y-5">
      <section className="glass p-4">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-end">
          <form onSubmit={save} className="grid min-w-0 flex-1 gap-3 md:grid-cols-[minmax(180px,260px)_minmax(260px,1fr)_auto]">
            <label className="min-w-0">
              <span className="mb-1 block text-xs uppercase tracking-[0.16em] text-white/35">Label</span>
              <input
                value={label}
                onChange={e => setLabel(e.target.value)}
                placeholder="loitering"
                className="w-full rounded-lg border border-white/10 bg-black/30 px-3 py-2 text-sm text-white placeholder:text-white/35 focus:border-white/35 focus:outline-none"
              />
            </label>
            <label className="min-w-0">
              <span className="mb-1 block text-xs uppercase tracking-[0.16em] text-white/35">Description</span>
              <input
                value={description}
                onChange={e => setDescription(e.target.value)}
                placeholder="a person staying in one area for an unusual amount of time"
                className="w-full rounded-lg border border-white/10 bg-black/30 px-3 py-2 text-sm text-white placeholder:text-white/35 focus:border-white/35 focus:outline-none"
              />
            </label>
            <button type="submit" disabled={!normalized || saving} className="icon-button self-end px-5">
              {saving ? '...' : 'Add'}
            </button>
          </form>
          <button onClick={refresh} disabled={loading || saving} className="icon-button self-start lg:self-end">
            Refresh
          </button>
        </div>
        {label && (
          <div className="mt-3 text-xs text-white/45">
            Saved label id: <span className="text-cyan-200">{normalized || 'invalid'}</span>
          </div>
        )}
      </section>

      {error && <div className="glass-sm p-3 text-sm text-red-300">{error}</div>}

      <section className="space-y-3">
        <div className="flex items-center justify-between gap-3">
          <div className="text-xs uppercase tracking-[0.18em] text-white/35">Action taxonomy</div>
          <div className="text-xs text-white/35">{labels.length} labels</div>
        </div>
        {loading ? (
          <div className="glass flex h-48 items-center justify-center text-sm text-white/28">Loading labels</div>
        ) : labels.length > 0 ? (
          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
            {labels.map(item => (
              <div key={item.label} className="glass-sm p-3">
                <div className="flex items-center justify-between gap-3">
                  <div className="truncate text-sm font-medium text-white/90">{item.label}</div>
                  <span className="confidence-badge shrink-0">active</span>
                </div>
                <div className="mt-2 line-clamp-3 min-h-[2.75rem] text-xs leading-5 text-white/50">
                  {item.description || 'No description'}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="glass flex h-48 items-center justify-center text-sm text-white/28">No labels configured</div>
        )}
      </section>
    </div>
  )
}

function normalizeLabel(value: string) {
  return value.toLowerCase().replace(/[^a-z0-9]/g, '')
}
