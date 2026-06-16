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
}

export default function AskPage() {
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [connected, setConnected] = useState(false)
  const [error, setError] = useState('')
  const socketRef = useRef<WebSocket | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)

  const streaming = useMemo(() => messages.some(m => m.role === 'assistant' && m.status === 'streaming'), [messages])

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])

  useEffect(() => {
    const socket = openChatSocket()
    socketRef.current = socket
    socket.onopen = () => {
      setConnected(true)
      setError('')
      socket.send(JSON.stringify({ type: 'session.list' }))
    }
    socket.onclose = () => {
      setConnected(false)
      setError('Chat connection closed')
    }
    socket.onerror = () => {
      setConnected(false)
      setError('Chat connection failed')
    }
    socket.onmessage = (event) => {
      const data = JSON.parse(event.data)
      if (data.type === 'session.listed') {
        setSessions(data.sessions || [])
      }
      if (data.type === 'session.created') {
        setSessions(prev => upsertSession(prev, data.session))
        setActiveSessionId(data.session.id)
        setMessages([])
      }
      if (data.type === 'messages.listed') {
        setActiveSessionId(data.sessionId)
        setMessages((data.messages || []).map(toMessage))
      }
      if (data.type === 'message.started') {
        setSessions(prev => upsertSession(prev, data.session))
        setActiveSessionId(data.session.id)
        setMessages(prev => {
          const next = [...prev]
          if (!next.some(m => m.id === data.userMessage.id)) next.push(toMessage(data.userMessage))
          if (!next.some(m => m.id === data.assistantMessage.id)) next.push(toMessage(data.assistantMessage))
          return next
        })
      }
      if (data.type === 'assistant.delta') {
        setMessages(prev => prev.map(m => m.id === data.messageId ? { ...m, content: m.content + data.delta, status: 'streaming' } : m))
      }
      if (data.type === 'assistant.completed') {
        setMessages(prev => prev.map(m => m.id === data.messageId ? { ...m, status: 'complete' } : m))
      }
      if (data.type === 'assistant.error') {
        setError(data.message || 'Chat failed')
        setMessages(prev => {
          if (!prev.length || prev[prev.length - 1].role !== 'assistant') return prev
          const next = [...prev]
          next[next.length - 1] = { ...next[next.length - 1], content: next[next.length - 1].content || data.message, error: true, status: 'complete' }
          return next
        })
      }
    }
    return () => socket.close()
  }, [])

  function sendSocket(payload: Record<string, unknown>) {
    const socket = socketRef.current
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      setError('Chat is not connected')
      return
    }
    socket.send(JSON.stringify(payload))
  }

  function createSession() {
    sendSocket({ type: 'session.create' })
  }

  function loadSession(id: string) {
    sendSocket({ type: 'messages.list', sessionId: id })
  }

  function send(e: FormEvent) {
    e.preventDefault()
    const content = input.trim()
    if (!content || streaming) return
    setInput('')
    setError('')
    sendSocket({
      type: 'message.send',
      sessionId: activeSessionId,
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
  }
}

function upsertSession(sessions: ChatSession[], session: ChatSession) {
  return [session, ...sessions.filter(item => item.id !== session.id)]
}
