const Home = ({ username, health, onEnterApp, onOpenSettings, onLogout }) => {
  const ready = health?.status === 'ready'
  const llm = health?.llm

  return (
    <div className="fullscreen-view home">
      <header className="home-header">
        <h2>Hello, {username}</h2>
        <div className="engine-status">
          <span className={`status-dot ${ready ? 'on' : 'waiting'}`} />
          Engine: {ready ? `online (v${health.version})` : 'starting…'}
        </div>
      </header>

      <div className="glass-panel home-card">
        <h3>Ready to work?</h3>
        <p className="muted">Point DocLamar at a folder of PDFs, Word, text or Markdown files and ask questions.
          Answers cite the exact file and page.</p>

        {ready && !llm?.configured && (
          <div className="callout">
            No LLM API key yet — search will work, but answers need a key.
            <button type="button" className="link-btn" onClick={onOpenSettings}>Add one in Settings</button>
          </div>
        )}
        {ready && llm?.configured && (
          <p className="muted small">Answers by <b>{llm.provider}</b> · {llm.model}{' '}
            <button type="button" className="link-btn" onClick={onOpenSettings}>Change</button>
          </p>
        )}

        <button type="button" className="primary-btn large" onClick={onEnterApp} disabled={!ready}>
          {ready ? 'Open workspace' : 'Starting engine…'}
        </button>
        <button type="button" className="link-btn" onClick={onLogout}>Not {username}? Switch user</button>
      </div>
    </div>
  )
}

export default Home
