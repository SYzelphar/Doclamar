import { useState } from 'react'

// A local profile, not an account: it only keeps each person's chat history
// separate on a shared computer. Nothing is sent anywhere.
const Login = ({ onLogin }) => {
  const [name, setName] = useState('')

  const handleSubmit = (e) => {
    e.preventDefault()
    if (name.trim()) onLogin(name.trim())
  }

  return (
    <div className="fullscreen-view centered">
      <form onSubmit={handleSubmit} className="glass-panel login-card">
        <h1>DocLamar</h1>
        <p className="muted">Who's using DocLamar?</p>
        <input
          type="text"
          placeholder="Your name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          className="text-input"
          autoFocus
        />
        <button type="submit" className="primary-btn large" disabled={!name.trim()}>Continue</button>
        <p className="muted small">Profiles only separate chat history on this computer.</p>
      </form>
    </div>
  )
}

export default Login
