import { useEffect, useRef, useState } from 'react'

const QueryInput = ({ onSubmit, disabled, placeholder }) => {
  const [query, setQuery] = useState('')
  const ref = useRef(null)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`
  }, [query])

  useEffect(() => {
    if (!disabled) ref.current?.focus()
  }, [disabled])

  const submit = (e) => {
    e?.preventDefault()
    const text = query.trim()
    if (!text || disabled) return
    onSubmit(text)
    setQuery('')
  }

  return (
    <div className="query-input-container">
      <form onSubmit={submit} className="input-wrapper">
        <textarea
          ref={ref}
          rows={1}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) submit(e)
          }}
          placeholder={placeholder}
          className="query-input"
          disabled={disabled}
        />
        <button type="submit" className="send-button" disabled={disabled || !query.trim()} title="Send (Enter) · New line (Shift+Enter)">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="22" y1="2" x2="11" y2="13" />
            <polygon points="22 2 15 22 11 13 2 9 22 2" />
          </svg>
        </button>
      </form>
    </div>
  )
}

export default QueryInput
