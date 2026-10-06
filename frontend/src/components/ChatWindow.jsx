import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import 'katex/dist/katex.min.css'
import { isDesktop } from '../api'

// Code and math are left untouched when linking citations (x[1] in an equation is not a citation).
const PROTECTED = /(```[\s\S]*?```|`[^`\n]*`|\$\$[\s\S]*?\$\$)/g

// Models write LaTeX as \( \) and \[ \]; remark-math wants $$. Single $ stays
// literal so prices like "$31.00" are not turned into math.
function prepareMarkdown(text) {
  const withMath = text
    .replace(/\\\[([\s\S]+?)\\\]/g, (_, m) => `\n$$\n${m.trim()}\n$$\n`)
    .replace(/\\\(([\s\S]+?)\\\)/g, (_, m) => `$$${m.trim()}$$`)
  return withMath
    .split(PROTECTED)
    .map((part, i) => (i % 2 ? part : part.replace(/(?<![\\\]])\[(\d{1,2})\](?![(:])/g, '[\\[$1\\]](#cite-$1)')))
    .join('')
}

const remarkPlugins = [remarkGfm, [remarkMath, { singleDollarTextMath: false }]]
const rehypePlugins = [rehypeKatex]

function Markdown({ text, onCite }) {
  return (
    <div className="markdown-body">
      <ReactMarkdown
        remarkPlugins={remarkPlugins}
        rehypePlugins={rehypePlugins}
        components={{
          a: ({ href, children }) => (href?.startsWith('#cite-')
            ? <button type="button" className="cite-ref" onClick={() => onCite?.(Number(href.slice(6)))}>{children}</button>
            : <a href={href} target="_blank" rel="noreferrer">{children}</a>),
        }}
      >
        {prepareMarkdown(text || '')}
      </ReactMarkdown>
    </div>
  )
}

function Sources({ citations, highlight, onChatWithFile, fileMode }) {
  const [showAll, setShowAll] = useState(false)
  const [open, setOpen] = useState(() => new Set())
  const refs = useRef({})
  const highlighted = highlight?.ref ?? null

  // highlight = { ref, at }: "at" changes on every click so re-clicking [n] scrolls again.
  useEffect(() => {
    if (!highlight) return
    if (!citations?.some((c) => c.cited && c.ref === highlight.ref)) setShowAll(true)
    setOpen((prev) => new Set(prev).add(highlight.ref))
    requestAnimationFrame(() => refs.current[highlight.ref]?.scrollIntoView({ behavior: 'smooth', block: 'nearest' }))
  }, [highlight, citations])

  if (!citations?.length) return null
  const cited = citations.filter((c) => c.cited)
  const shown = showAll || cited.length === 0 ? citations : cited

  const toggle = (ref) => setOpen((prev) => {
    const next = new Set(prev)
    if (next.has(ref)) next.delete(ref)
    else next.add(ref)
    return next
  })

  return (
    <div className="citations-container">
      <h4>Sources</h4>
      <div className="citations-list">
        {shown.map((c) => (
          <div key={c.ref} ref={(el) => { refs.current[c.ref] = el }}
            className={`citation-item ${highlighted === c.ref ? 'highlight' : ''}`}>
            <button type="button" className="citation-head" onClick={() => toggle(c.ref)}>
              <span className="cite-num">{c.ref}</span>
              <span className="citation-file">{c.file}</span>
              {c.ocr && <span className="ocr-badge" title="Read from a scan or image with OCR; may contain recognition errors">OCR</span>}
              <span className="citation-where">{[c.pages, c.section].filter(Boolean).join(' · ')}</span>
              <span className="chevron">{open.has(c.ref) ? '▴' : '▾'}</span>
            </button>
            {open.has(c.ref) && (
              <>
                <div className="citation-snippet">{c.snippet}{c.snippet?.length >= 400 ? '…' : ''}</div>
                <div className="citation-actions">
                  {isDesktop && (
                    <button type="button" className="link-btn" onClick={() => window.doclamar.openPath(c.path)}>Open file</button>
                  )}
                  {isDesktop && (
                    <button type="button" className="link-btn" onClick={() => window.doclamar.showInFolder(c.path)}>Show in folder</button>
                  )}
                  {!fileMode && (
                    <button type="button" className="link-btn" onClick={() => onChatWithFile(c.path)}>Chat with this file</button>
                  )}
                </div>
              </>
            )}
          </div>
        ))}
      </div>
      {cited.length > 0 && cited.length < citations.length && (
        <button type="button" className="link-btn" onClick={() => setShowAll((v) => !v)}>
          {showAll ? 'Show cited sources only' : `Show all ${citations.length} retrieved passages`}
        </button>
      )}
    </div>
  )
}

const STATUS_NOTE = {
  not_found: 'Nothing relevant found',
  indexing: 'Still indexing',
  no_llm: 'No API key — showing passages only',
  llm_error: 'The language model request failed',
}

function AssistantMessage({ msg, onRetry, onChatWithFile, fileMode }) {
  const [highlight, setHighlight] = useState(null)

  if (msg.pending) {
    return (
      <div className="message assistant-message">
        <div className="message-text pending"><span className="dots"><i /><i /><i /></span> Searching your documents…</div>
      </div>
    )
  }
  if (msg.error) {
    return (
      <div className="message assistant-message">
        <div className="message-text error-text">
          {msg.error}
          <button type="button" className="secondary-btn small" onClick={() => onRetry(msg.retryQuestion)}>Retry</button>
        </div>
      </div>
    )
  }
  return (
    <div className="message assistant-message">
      <div className="message-text">
        {STATUS_NOTE[msg.status] && <div className={`status-note ${msg.status}`}>{STATUS_NOTE[msg.status]}</div>}
        <Markdown text={msg.content} onCite={(ref) => setHighlight({ ref, at: Date.now() })} />
        <Sources citations={msg.citations} highlight={highlight} onChatWithFile={onChatWithFile} fileMode={fileMode} />
      </div>
      {msg.timestamp && <div className="message-timestamp">{new Date(msg.timestamp).toLocaleTimeString()}</div>}
    </div>
  )
}

function EmptyState({ folder, session, onPickFolder, onOpenSettings }) {
  if (session?.mode === 'file') return null
  return (
    <div className="empty-state">
      <h2>Ask your documents anything</h2>
      {folder ? (
        <p>Questions are answered from the PDFs (including scanned ones), Word, text, Markdown and image files
          in <code>{folder}</code>, with citations to the exact file and page.</p>
      ) : (
        <>
          <p>Pick a folder of documents. DocLamar indexes it once (only changed files are re-read later), reads
            scanned PDFs and photos of pages with OCR, then answers questions with citations to the exact file
            and page.</p>
          {isDesktop && <button type="button" className="primary-btn" onClick={onPickFolder}>Choose a folder…</button>}
        </>
      )}
      <p className="muted small">Answers are written by your configured LLM provider; manage your API key in{' '}
        <button type="button" className="link-btn" onClick={onOpenSettings}>Settings</button>.</p>
    </div>
  )
}

const ChatWindow = ({ messages, folder, session, onChatWithFile, onRetry, onPickFolder, onOpenSettings }) => {
  const endRef = useRef(null)
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const fileMode = session?.mode === 'file'

  return (
    <div className="chat-window">
      <div className="messages-container">
        {messages.length === 0 && (
          <EmptyState folder={folder} session={session} onPickFolder={onPickFolder} onOpenSettings={onOpenSettings} />
        )}
        {messages.map((msg) => {
          if (msg.role === 'system') {
            return <div key={msg.id} className="system-message"><Markdown text={msg.content} /></div>
          }
          if (msg.role === 'user') {
            return (
              <div key={msg.id} className="message user-message">
                <div className="message-text">{msg.content}</div>
              </div>
            )
          }
          return <AssistantMessage key={msg.id} msg={msg} onRetry={onRetry} onChatWithFile={onChatWithFile} fileMode={fileMode} />
        })}
        <div ref={endRef} />
      </div>
    </div>
  )
}

export default ChatWindow
