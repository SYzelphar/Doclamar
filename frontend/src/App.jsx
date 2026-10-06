import { useCallback, useEffect, useRef, useState } from 'react'
import './App.css'
import { api, isDesktop, storage } from './api'
import Sidebar from './components/Sidebar'
import ChatWindow from './components/ChatWindow'
import QueryInput from './components/QueryInput'
import Login from './components/Login'
import Home from './components/Home'
import SettingsModal from './components/SettingsModal'

const USER_KEY = 'doclamar_username'
const VIEW_KEY = 'doclamar_view'
const FOLDER_KEY = 'doclamar_folder'

let nextId = 0
const newId = () => `${Date.now()}-${nextId++}`

function messagesFromHistory(rows) {
  return rows.map((m) => ({
    id: newId(),
    role: m.role,
    content: m.content,
    citations: m.citations || [],
    timestamp: m.created_at ? `${m.created_at.replace(' ', 'T')}Z` : null,
  }))
}

function App() {
  // Read the saved name synchronously: the old app started with '' and lost
  // the user's chat history on every restart.
  const [username, setUsername] = useState(() => storage.get(USER_KEY) || '')
  const [view, setView] = useState(() => {
    if (!storage.get(USER_KEY)) return 'login'
    return storage.get(VIEW_KEY) === 'app' ? 'app' : 'home'
  })
  const [health, setHealth] = useState(null)
  const [showSettings, setShowSettings] = useState(false)

  const [folder, setFolder] = useState(() => storage.get(FOLDER_KEY) || '')
  const [indexStatus, setIndexStatus] = useState(null)
  const [indexError, setIndexError] = useState(null)
  const [session, setSession] = useState(null)
  const [messages, setMessages] = useState([])
  const [sessions, setSessions] = useState([])
  const [busy, setBusy] = useState(false)
  const [notice, setNotice] = useState(null)

  const ready = health?.status === 'ready'
  const pollTimer = useRef(null)

  useEffect(() => storage.set(USER_KEY, username), [username])
  useEffect(() => storage.set(VIEW_KEY, view), [view])
  useEffect(() => storage.set(FOLDER_KEY, folder), [folder])

  // ------------------------------------------------------------ engine health
  const refreshHealth = useCallback(async () => {
    try {
      setHealth(await api('/health'))
    } catch {
      setHealth(null)
    }
  }, [])

  useEffect(() => {
    let cancelled = false
    let timer
    const tick = async () => {
      await refreshHealth()
      if (!cancelled) timer = setTimeout(tick, ready ? 15000 : 1500)
    }
    tick()
    return () => {
      cancelled = true
      clearTimeout(timer)
    }
  }, [ready, refreshHealth])

  // ----------------------------------------------------------------- history
  const refreshSessions = useCallback(async () => {
    if (!username) return
    try {
      const data = await api('/sessions', { params: { username } })
      setSessions(data.sessions)
    } catch (e) {
      setNotice(e.message)
    }
  }, [username])

  useEffect(() => {
    if (ready && username) refreshSessions()
  }, [ready, username, refreshSessions])

  // ---------------------------------------------------------------- indexing
  const pollIndex = useCallback(async (path) => {
    clearTimeout(pollTimer.current)
    try {
      const status = await api('/index/status', { params: { folder: path } })
      setIndexStatus(status)
      setIndexError(null)
      const state = status.job?.state
      if (state === 'queued' || state === 'scanning' || state === 'indexing') {
        pollTimer.current = setTimeout(() => pollIndex(path), 800)
      }
    } catch (e) {
      setIndexError(e.message)
    }
  }, [])

  const syncFolder = useCallback(async (path) => {
    if (!path) return
    setIndexError(null)
    try {
      setIndexStatus(await api('/index', { method: 'POST', body: { folder: path } }))
      pollIndex(path)
    } catch (e) {
      setIndexStatus(null)
      setIndexError(e.message)
    }
  }, [pollIndex])

  // Re-sync whenever the folder changes (and once the engine is up). Syncs are
  // incremental, so this only re-reads files that changed since last time.
  useEffect(() => {
    if (ready && folder) syncFolder(folder)
    return () => clearTimeout(pollTimer.current)
  }, [ready, folder, syncFolder])

  // ----------------------------------------------------------------- actions
  const newChat = () => {
    setSession(null)
    setMessages([])
  }

  const selectFolder = (path) => {
    if (!path) return
    newChat()
    setIndexStatus(null)
    setFolder(path)
  }

  const send = async (question) => {
    if (busy) return
    const inFileMode = session?.mode === 'file'
    if (!inFileMode && !folder && !session?.folder) {
      setNotice('Choose a folder of documents first.')
      return
    }
    const pendingId = newId()
    setMessages((m) => [
      ...m,
      { id: newId(), role: 'user', content: question, timestamp: new Date().toISOString() },
      { id: pendingId, role: 'assistant', pending: true },
    ])
    setBusy(true)
    try {
      const data = await api('/chat', {
        method: 'POST',
        body: {
          question,
          username,
          session_id: session?.id ?? null,
          folder: session?.folder || folder || null,
        },
      })
      setMessages((m) => m.map((msg) => (msg.id === pendingId
        ? { id: pendingId, role: 'assistant', content: data.answer, citations: data.citations,
            status: data.status, timestamp: new Date().toISOString() }
        : msg)))
      if (!session) setSession({ id: data.session_id, mode: 'folder', folder, title: question })
      if (data.status === 'indexing') pollIndex(folder)
      refreshSessions()
    } catch (e) {
      setMessages((m) => m.map((msg) => (msg.id === pendingId
        ? { id: pendingId, role: 'assistant', error: e.message, retryQuestion: question }
        : msg)))
    } finally {
      setBusy(false)
    }
  }

  const retry = (question) => {
    setMessages((m) => {
      const i = m.findIndex((msg) => msg.retryQuestion === question)
      return i > 0 ? m.slice(0, i - 1).concat(m.slice(i + 1)) : m
    })
    send(question)
  }

  const openSession = async (id) => {
    if (busy) return
    try {
      const data = await api(`/sessions/${id}`, { params: { username } })
      setSession(data.session)
      setMessages(messagesFromHistory(data.messages))
      if (data.session.mode === 'folder' && data.session.folder && data.session.folder !== folder) {
        setIndexStatus(null)
        setFolder(data.session.folder)
      }
    } catch (e) {
      setNotice(e.message)
    }
  }

  const deleteSession = async (id) => {
    try {
      await api(`/sessions/${id}`, { method: 'DELETE', params: { username } })
      if (session?.id === id) newChat()
      refreshSessions()
    } catch (e) {
      setNotice(e.message)
    }
  }

  const chatWithFile = async (filePath) => {
    if (!filePath || busy) return
    setBusy(true)
    setMessages([{ id: newId(), role: 'system',
      content: 'Reading the file… scanned documents and images are run through OCR, which can take a moment.' }])
    try {
      const data = await api('/sessions/file', { method: 'POST', body: { file_path: filePath, username } })
      setSession(data.session)
      const pages = data.file.pages ? ` (${data.file.pages} pages)` : ''
      setMessages([{ id: newId(), role: 'system',
        content: `Answering only from **${data.file.name}**${pages}. Ask anything about it.` }])
      refreshSessions()
    } catch (e) {
      setMessages([])
      setNotice(`Couldn't open that file: ${e.message}`)
    } finally {
      setBusy(false)
    }
  }

  const logout = () => {
    newChat()
    setSessions([])
    setUsername('')
    setView('login')
  }

  // ------------------------------------------------------------------ render
  const settingsModal = showSettings && (
    <SettingsModal
      onClose={() => setShowSettings(false)}
      onSaved={() => {
        setShowSettings(false)
        refreshHealth()
      }}
    />
  )

  if (view === 'login') {
    return <Login onLogin={(name) => { setUsername(name); setView('home') }} />
  }

  if (view === 'home') {
    return (
      <>
        <Home
          username={username}
          health={health}
          onEnterApp={() => setView('app')}
          onOpenSettings={() => setShowSettings(true)}
          onLogout={logout}
        />
        {settingsModal}
      </>
    )
  }

  const fileMode = session?.mode === 'file'
  const canAsk = ready && !busy && (fileMode || Boolean(folder || session?.folder))

  return (
    <div className="app-container">
      <Sidebar
        username={username}
        folder={folder}
        onSelectFolder={selectFolder}
        indexStatus={indexStatus}
        indexError={indexError}
        onRescan={() => syncFolder(folder)}
        session={session}
        sessions={sessions}
        onOpenSession={openSession}
        onDeleteSession={deleteSession}
        onNewChat={newChat}
        onOpenFile={async () => chatWithFile(isDesktop ? await window.doclamar.openFile() : null)}
        onOpenSettings={() => setShowSettings(true)}
        onLogout={logout}
        llmConfigured={Boolean(health?.llm?.configured)}
        busy={busy}
      />
      <main className="main-content">
        {notice && (
          <div className="notice" role="alert">
            <span>{notice}</span>
            <button type="button" onClick={() => setNotice(null)} aria-label="Dismiss">×</button>
          </div>
        )}
        {!ready && <div className="notice subtle">Connecting to the DocLamar engine…</div>}
        <ChatWindow
          messages={messages}
          folder={folder}
          session={session}
          onChatWithFile={chatWithFile}
          onRetry={retry}
          onPickFolder={async () => isDesktop && selectFolder(await window.doclamar.openDirectory())}
          onOpenSettings={() => setShowSettings(true)}
        />
        <QueryInput
          onSubmit={send}
          disabled={!canAsk}
          placeholder={
            !ready ? 'Waiting for the engine…'
              : busy ? 'Working on it…'
                : canAsk ? (fileMode ? `Ask about ${session.title}…` : 'Ask a question about your documents…')
                  : 'Choose a folder to start'
          }
        />
      </main>
      {settingsModal}
    </div>
  )
}

export default App
