'use client'
import { FormEvent, useEffect, useMemo, useRef, useState } from 'react'
import { openChatSocket } from '@/lib/api'

type ChatSession = { id: string; title: string; videoId?: string | null; updatedAt: string }
type Message = {
  id: string
  role: 'user' | 'assistant'
  content: string
  status?: string
  error?: boolean
  sourcesJson?: string
}

const ACTIVE_CHAT_KEY = 'videovault:activeChatSessionId'

export default function AskPage() {
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [connected, setConnected] = useState(false)
  const [error, setError] = useState('')
  const socketRef = useRef<WebSocket | null>(null)
  const activeSessionIdRef = useRef<string | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)

  const streaming = useMemo(() => messages.some(m => m.role === 'assistant' && m.status === 'streaming'), [messages])

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])

  useEffect(() => {
    let disposed = false
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null

    function connect() {
      if (disposed) return
      const socket = openChatSocket()
      socketRef.current = socket
      socket.onopen = () => {
        setConnected(true)
        setError('')
        socket.send(JSON.stringify({ type: 'session.list' }))
      }
      socket.onclose = () => {
        if (socketRef.current === socket) socketRef.current = null
        setConnected(false)
        if (!disposed) {
          setError('Chat connection closed. Reconnecting...')
          reconnectTimer = setTimeout(connect, 1200)
        }
      }
      socket.onerror = () => {
        setConnected(false)
        setError('Chat connection failed')
      }
      socket.onmessage = (event) => handleSocketMessage(JSON.parse(event.data))
    }

    connect()
    return () => {
      disposed = true
      if (reconnectTimer) clearTimeout(reconnectTimer)
      socketRef.current?.close()
    }
  }, [])

  function sendSocket(payload: Record<string, unknown>) {
    const socket = socketRef.current
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      setError('Chat is not connected')
      return
    }
    socket.send(JSON.stringify(payload))
  }

  function handleSocketMessage(data: any) {
    if (data.type === 'session.listed') {
      const nextSessions = data.sessions || []
      setSessions(nextSessions)
      const target = chooseSessionId(nextSessions, data.activeSessionId)
      if (target) {
        openSession(target, { updateServer: false })
      }
    }
    if (data.type === 'session.created') {
      setSessions(prev => upsertSession(prev, data.session))
      setActiveSession(data.session.id)
      setMessages([])
      sendSocket({ type: 'session.subscribe', sessionId: data.session.id })
    }
    if (data.type === 'session.activated') {
      setSessions(prev => upsertSession(prev, data.session))
    }
    if (data.type === 'messages.listed') {
      setActiveSession(data.sessionId)
      setMessages((data.messages || []).map(toMessage))
      sendSocket({ type: 'session.subscribe', sessionId: data.sessionId })
    }
    if (data.type === 'message.started') {
      if (data.session) setSessions(prev => upsertSession(prev, data.session))
      if (data.session?.id && (!activeSessionIdRef.current || data.session.id === activeSessionIdRef.current)) {
        setActiveSession(data.session.id)
      }
      if (!isActiveEvent(data)) return
      setMessages(prev => mergeMessages(prev, [toMessage(data.userMessage), toMessage(data.assistantMessage)]))
    }
    if (data.type === 'assistant.sources') {
      if (!isActiveEvent(data)) return
      setMessages(prev => prev.map(m => m.id === data.messageId ? { ...m, sourcesJson: JSON.stringify(data.sources || []) } : m))
    }
    if (data.type === 'assistant.delta') {
      if (!isActiveEvent(data)) return
      setMessages(prev => prev.map(m => m.id === data.messageId ? { ...m, content: m.content + data.delta, status: 'streaming' } : m))
    }
    if (data.type === 'assistant.completed') {
      if (!isActiveEvent(data)) return
      setMessages(prev => prev.map(m => m.id === data.messageId ? { ...m, status: 'complete' } : m))
    }
    if (data.type === 'assistant.error') {
      if (data.sessionId && !isActiveEvent(data)) return
      setError(data.message || 'Chat failed')
      setMessages(prev => {
        if (!prev.length || prev[prev.length - 1].role !== 'assistant') return prev
        const next = [...prev]
        next[next.length - 1] = { ...next[next.length - 1], content: next[next.length - 1].content || data.message, error: true, status: 'complete' }
        return next
      })
    }
  }

  function chooseSessionId(nextSessions: ChatSession[], backendActiveSessionId?: string) {
    const validIds = new Set(nextSessions.map(session => session.id))
    const urlSessionId = new URLSearchParams(window.location.search).get('sessionId') || ''
    const cachedSessionId = localStorage.getItem(ACTIVE_CHAT_KEY) || ''
    if (urlSessionId && validIds.has(urlSessionId)) return urlSessionId
    if (backendActiveSessionId && validIds.has(backendActiveSessionId)) return backendActiveSessionId
    if (cachedSessionId && validIds.has(cachedSessionId)) return cachedSessionId
    return nextSessions[0]?.id || ''
  }

  function isActiveEvent(data: any) {
    return !data.sessionId || data.sessionId === activeSessionIdRef.current
  }

  function setActiveSession(id: string) {
    activeSessionIdRef.current = id
    setActiveSessionId(id)
    localStorage.setItem(ACTIVE_CHAT_KEY, id)
    const url = new URL(window.location.href)
    url.pathname = '/ask'
    url.searchParams.set('sessionId', id)
    window.history.replaceState(null, '', `${url.pathname}?${url.searchParams.toString()}`)
  }

  function openSession(id: string, options: { updateServer: boolean }) {
    if (!id) return
    setActiveSession(id)
    if (options.updateServer) sendSocket({ type: 'session.activate', sessionId: id })
    sendSocket({ type: 'messages.list', sessionId: id })
    sendSocket({ type: 'session.subscribe', sessionId: id })
  }

  function createSession() {
    sendSocket({ type: 'session.create' })
  }

  function loadSession(id: string) {
    openSession(id, { updateServer: true })
  }

  function send(e: FormEvent) {
    e.preventDefault()
    const content = input.trim()
    if (!content || streaming) return
    setInput('')
    setError('')
    sendSocket({
      type: 'message.send',
      sessionId: activeSessionIdRef.current,
      content,
      clientMessageId: crypto.randomUUID(),
    })
  }

  return (
    <div className="grid h-full grid-cols-1 gap-4 lg:grid-cols-[260px_minmax(0,1fr)]">
      <aside className="glass flex min-h-0 flex-col p-3">
        <button onClick={createSession} className="icon-button mb-3 w-full">
          New chat
        </button>
        <div className="min-h-0 flex-1 space-y-1 overflow-y-auto">
          {sessions.map(session => (
            <button
              key={session.id}
              onClick={() => loadSession(session.id)}
              className={`w-full truncate rounded-lg px-3 py-2 text-left text-sm transition-colors ${activeSessionId === session.id ? 'bg-white/14 text-white' : 'text-white/55 hover:bg-white/8 hover:text-white'}`}
            >
              {session.title || 'New chat'}
            </button>
          ))}
        </div>
        <div className="mt-3 rounded-lg border border-white/10 bg-black/25 px-3 py-2 text-xs text-white/45">
          {connected ? 'WebSocket connected' : 'Disconnected'}
        </div>
      </aside>

      <main className="flex min-h-0 flex-col gap-3">
        <div className="glass min-h-0 flex-1 overflow-y-auto p-5">
          {messages.length === 0 && (
            <div className="flex h-full items-center justify-center text-sm text-white/28">
              Ask about uploaded and indexed videos
            </div>
          )}
          <div className="space-y-4">
            {messages.map(message => (
              <div key={message.id} className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`max-w-[78%] rounded-lg px-4 py-3 text-sm leading-6 whitespace-pre-wrap ${message.role === 'user' ? 'bg-white/14 text-white' : 'glass-sm text-white/82'} ${message.error ? 'border-red-400/40 text-red-300' : ''}`}>
                  {message.content || <span className="opacity-35">▋</span>}
                </div>
              </div>
            ))}
          </div>
          <div ref={bottomRef} />
        </div>

        {error && <div className="glass-sm px-3 py-2 text-sm text-red-300">{error}</div>}

        <form onSubmit={send} className="glass flex gap-2 p-3">
          <input
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder={connected ? 'Ask about your videos' : 'Connecting...'}
            className="min-w-0 flex-1 bg-transparent text-sm text-white placeholder:text-white/35 focus:outline-none"
            disabled={!connected}
          />
          <button type="submit" disabled={!connected || streaming || !input.trim()} className="icon-button w-24">
            {streaming ? '...' : 'Send'}
          </button>
        </form>
      </main>
    </div>
  )
}

function toMessage(raw: any): Message {
  return {
    id: raw.id,
    role: raw.role,
    content: raw.content || '',
    status: raw.status,
    sourcesJson: raw.sourcesJson,
  }
}

function upsertSession(sessions: ChatSession[], session: ChatSession) {
  return [session, ...sessions.filter(item => item.id !== session.id)]
}

function mergeMessages(existing: Message[], incoming: Message[]) {
  const byId = new Map(existing.map(message => [message.id, message]))
  for (const message of incoming) {
    if (!byId.has(message.id)) byId.set(message.id, message)
  }
  return Array.from(byId.values())
}
