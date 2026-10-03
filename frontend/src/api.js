// Client for the local DocLamar engine.
// In the desktop app, Electron supplies the URL and a per-launch auth token.
// In a plain browser (`npm run dev` against a backend you started yourself) they
// come from VITE_API_URL / VITE_API_TOKEN.

export const isDesktop = typeof window !== 'undefined' && Boolean(window.doclamar)

let configPromise = null

function loadConfig() {
  if (isDesktop) return window.doclamar.getBackendConfig()
  return Promise.resolve({
    url: import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000',
    token: import.meta.env.VITE_API_TOKEN || '',
  })
}

function errorDetail(data, status) {
  if (typeof data?.detail === 'string') return data.detail
  if (Array.isArray(data?.detail)) return data.detail.map((d) => d.msg).join('; ')
  return `Request failed (${status})`
}

export async function api(path, { method = 'GET', body, params } = {}) {
  configPromise ||= loadConfig()
  const { url, token } = await configPromise
  const query = params ? `?${new URLSearchParams(params)}` : ''
  const headers = {}
  if (body !== undefined) headers['Content-Type'] = 'application/json'
  if (token) headers['X-Doclamar-Token'] = token

  let res
  try {
    res = await fetch(`${url}${path}${query}`, {
      method,
      headers,
      body: body !== undefined ? JSON.stringify(body) : undefined,
    })
  } catch {
    throw new Error('Cannot reach the DocLamar engine. It may still be starting up.')
  }
  const data = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(errorDetail(data, res.status))
  return data
}

// Errors thrown in Electron's main process arrive wrapped in boilerplate.
export function ipcErrorMessage(err) {
  return String(err?.message || err).replace(/^Error invoking remote method '[^']+': (Error: )?/, '')
}

export const storage = {
  get(key) {
    try {
      return localStorage.getItem(key)
    } catch {
      return null
    }
  },
  set(key, value) {
    try {
      if (value === null || value === undefined || value === '') localStorage.removeItem(key)
      else localStorage.setItem(key, value)
    } catch {
      /* storage unavailable: nothing to persist */
    }
  },
}
