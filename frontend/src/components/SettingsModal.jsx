import { useEffect, useState } from 'react'
import { api, ipcErrorMessage, isDesktop } from '../api'

const PROVIDERS = {
  groq: { label: 'Groq', defaultModel: 'openai/gpt-oss-120b', keyUrl: 'https://console.groq.com/keys' },
  gemini: { label: 'Google Gemini', defaultModel: 'gemini-2.5-flash', keyUrl: 'https://aistudio.google.com/apikey' },
}

const SettingsModal = ({ onClose, onSaved }) => {
  const [provider, setProvider] = useState('groq')
  const [model, setModel] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [hasKey, setHasKey] = useState(false)
  const [savedProvider, setSavedProvider] = useState(null)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!isDesktop) return
    window.doclamar.getSettings().then((s) => {
      setProvider(s.provider)
      setSavedProvider(s.provider)
      setModel(s.model)
      setHasKey(s.hasKey)
    })
  }, [])

  useEffect(() => {
    const onKey = (e) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const keepsSavedKey = hasKey && provider === savedProvider

  const save = async (e) => {
    e.preventDefault()
    setSaving(true)
    setError(null)
    try {
      if (isDesktop) {
        await window.doclamar.saveSettings({ provider, apiKey: apiKey.trim(), model: model.trim() })
      } else {
        await api('/config/llm', { method: 'POST', body: { provider, api_key: apiKey.trim(), model: model.trim() || null } })
      }
      onSaved()
    } catch (err) {
      setError(ipcErrorMessage(err))
    } finally {
      setSaving(false)
    }
  }

  const info = PROVIDERS[provider]

  return (
    <div className="modal-backdrop" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <form className="modal" onSubmit={save}>
        <h2>Language model</h2>
        <p className="muted small">
          DocLamar searches your files locally. Only the question and the few passages needed to answer it are
          sent to the provider you choose.
        </p>

        <label>
          Provider
          <select value={provider} onChange={(e) => { setProvider(e.target.value); setModel('') }} className="text-input">
            {Object.entries(PROVIDERS).map(([id, p]) => <option key={id} value={id}>{p.label}</option>)}
          </select>
        </label>

        <label>
          API key
          <input
            type="password"
            className="text-input"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder={keepsSavedKey ? '•••••••• (saved — leave blank to keep)' : 'Paste your API key'}
            autoComplete="off"
          />
          <a className="small" href={info.keyUrl} target="_blank" rel="noreferrer">Get a {info.label} key</a>
        </label>

        <label>
          <span>Model <span className="muted">(optional)</span></span>
          <input
            className="text-input"
            value={model}
            onChange={(e) => setModel(e.target.value)}
            placeholder={info.defaultModel}
            spellCheck={false}
          />
        </label>

        {isDesktop
          ? <p className="muted small">The key is checked with {info.label}, then stored encrypted on this computer.</p>
          : <p className="muted small">Browser mode: the key is kept by the running engine until it restarts.</p>}
        {error && <div className="form-error">{error}</div>}

        <div className="modal-actions">
          <button type="button" className="secondary-btn" onClick={onClose}>Cancel</button>
          <button type="submit" className="primary-btn" disabled={saving || (!apiKey.trim() && !keepsSavedKey)}>
            {saving ? 'Checking key…' : 'Save'}
          </button>
        </div>
      </form>
    </div>
  )
}

export default SettingsModal
