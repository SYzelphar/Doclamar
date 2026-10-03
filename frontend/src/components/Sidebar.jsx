import { useEffect, useState } from 'react'
import { isDesktop } from '../api'

const ACTIVE = ['queued', 'scanning', 'indexing']

const baseName = (p) => (p ? p.split(/[\\/]/).filter(Boolean).pop() : '')

function formatDate(value) {
  if (!value) return ''
  const date = new Date(`${value.replace(' ', 'T')}${value.endsWith('Z') || value.includes('+') ? '' : 'Z'}`)
  return Number.isNaN(date.getTime()) ? '' : date.toLocaleDateString()
}

function IndexStatus({ status, error, onRescan }) {
  const [showIssues, setShowIssues] = useState(false)
  if (error) return <div className="index-status error">{error}</div>
  if (!status) return null

  const job = status.job
  if (job && ACTIVE.includes(job.state)) {
    const total = job.to_index || 0
    const pct = total ? Math.round((job.processed / total) * 100) : 0
    return (
      <div className="index-status">
        <div className="index-line">
          {job.state === 'indexing'
            ? `Indexing ${job.processed} / ${total} files…`
            : 'Scanning folder…'}
        </div>
        <div className="progress"><div style={{ width: `${pct}%` }} /></div>
        {job.current_file && <div className="index-file" title={job.current_file}>{baseName(job.current_file)}</div>}
      </div>
    )
  }

  const issues = status.issues || []
  return (
    <div className="index-status">
      {job?.state === 'error' && <div className="index-line error">Indexing failed: {job.error}</div>}
      <div className="index-line">
        <span className="ok-dot" /> {status.files_indexed} file{status.files_indexed === 1 ? '' : 's'} indexed
        · {status.chunks.toLocaleString()} passages
        <button type="button" className="link-btn" onClick={onRescan} title="Look for new or changed files">Rescan</button>
      </div>
      {issues.length > 0 && (
        <>
          <button type="button" className="link-btn" onClick={() => setShowIssues((v) => !v)}>
            {issues.length} file{issues.length === 1 ? '' : 's'} skipped {showIssues ? '▴' : '▾'}
          </button>
          {showIssues && (
            <ul className="issues">
              {issues.map((i) => (
                <li key={i.path} title={i.path}><b>{baseName(i.path)}</b>: {i.error || i.status}</li>
              ))}
            </ul>
          )}
        </>
      )}
      {status.files_total === 0 && <div className="index-line muted">No PDF, DOCX, TXT or MD files found here.</div>}
    </div>
  )
}

const Sidebar = ({
  username, folder, onSelectFolder, indexStatus, indexError, onRescan,
  session, sessions, onOpenSession, onDeleteSession, onNewChat, onOpenFile,
  onOpenSettings, onLogout, llmConfigured, busy,
}) => {
  const [typedPath, setTypedPath] = useState(folder)
  useEffect(() => setTypedPath(folder), [folder])

  const browse = async () => {
    const picked = await window.doclamar.openDirectory()
    if (picked) onSelectFolder(picked)
  }

  const submitPath = (e) => {
    e.preventDefault()
    const path = typedPath.trim().replace(/^"(.*)"$/, '$1')
    if (path && path !== folder) onSelectFolder(path)
  }

  const fileMode = session?.mode === 'file'

  return (
    <aside className="sidebar">
      <section className="sidebar-section">
        <h3>Folder</h3>
        <form className="directory-form" onSubmit={submitPath}>
          <input
            className="directory-input"
            value={typedPath}
            onChange={(e) => setTypedPath(e.target.value)}
            placeholder="Paste a folder path and press Enter"
            spellCheck={false}
          />
          {isDesktop && (
            <button type="button" className="icon-btn" onClick={browse} title="Browse for a folder">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" /></svg>
            </button>
          )}
        </form>
        {folder && <IndexStatus status={indexStatus} error={indexError} onRescan={onRescan} />}
      </section>

      {fileMode && (
        <section className="file-mode">
          <div>Chatting with one file</div>
          <b title={session.file_path}>{session.title}</b>
          <button type="button" className="link-btn" onClick={onNewChat}>← Back to folder search</button>
        </section>
      )}

      <section className="sidebar-actions">
        <button type="button" className="primary-btn" onClick={onNewChat} disabled={busy}>+ New chat</button>
        {isDesktop && (
          <button type="button" className="secondary-btn" onClick={onOpenFile} disabled={busy}
            title="Ask questions about a single document">
            Chat with a file…
          </button>
        )}
      </section>

      <section className="chat-history">
        <h3>History</h3>
        {sessions.length === 0 ? (
          <p className="empty-message">No chats yet.</p>
        ) : (
          sessions.map((s) => (
            <div key={s.id} className={`chat-item ${session?.id === s.id ? 'active' : ''}`}>
              <button type="button" className="chat-item-content" onClick={() => onOpenSession(s.id)} disabled={busy}>
                <span className="chat-item-icon">{s.mode === 'file' ? '📄' : '📁'}</span>
                <span className="chat-item-text">
                  <span className="chat-item-title">{s.title}</span>
                  <span className="chat-item-meta">
                    {baseName(s.mode === 'file' ? s.file_path : s.folder) || 'folder'} · {formatDate(s.updated_at)}
                  </span>
                </span>
              </button>
              <button
                type="button"
                className="delete-chat-btn"
                title="Delete chat"
                onClick={() => { if (window.confirm('Delete this chat?')) onDeleteSession(s.id) }}
              >
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="3 6 5 6 21 6" /><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" /></svg>
              </button>
            </div>
          ))
        )}
      </section>

      <footer className="sidebar-footer">
        <span className="user-name" title="Profile">{username}</span>
        <button type="button" className="link-btn" onClick={onOpenSettings}>
          {llmConfigured ? 'Settings' : <span className="warn">Add API key</span>}
        </button>
        <button type="button" className="link-btn" onClick={onLogout}>Switch user</button>
      </footer>
    </aside>
  )
}

export default Sidebar
