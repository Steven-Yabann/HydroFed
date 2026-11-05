// Thin wrapper over fetch that adds the session token and turns API errors
// into readable messages. A 401 clears the session so the app returns to
// the sign-in page.

export const apiBaseUrl = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')

const TOKEN_KEY = 'hydrofed.session'
let unauthorisedHandler = () => {}

export function readSession() {
  try {
    const session = JSON.parse(localStorage.getItem(TOKEN_KEY) || 'null')
    if (!session?.token) return null
    if (session.expires_at && new Date(session.expires_at) <= new Date()) return null
    return session
  } catch {
    return null
  }
}

export function saveSession(session) {
  try {
    if (session) localStorage.setItem(TOKEN_KEY, JSON.stringify(session))
    else localStorage.removeItem(TOKEN_KEY)
  } catch {
    // Private windows can refuse storage; the session then lasts for this tab only.
  }
}

export function onUnauthorised(handler) {
  unauthorisedHandler = handler
}

export class ApiError extends Error {
  constructor(message, status) {
    super(message)
    this.status = status
  }
}

export async function api(path, { method = 'GET', body, token } = {}) {
  const session = token ? { token } : readSession()
  const headers = { Accept: 'application/json' }
  if (body !== undefined) headers['Content-Type'] = 'application/json'
  if (session?.token) headers.Authorization = `Bearer ${session.token}`

  let response
  try {
    response = await fetch(`${apiBaseUrl}${path}`, {
      method,
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
    })
  } catch {
    throw new ApiError('The HydroFed server cannot be reached. Check that the API is running.', 0)
  }

  const data = await response.json().catch(() => ({}))
  if (!response.ok) {
    const detail = Array.isArray(data.detail)
      ? data.detail.map((item) => item.msg).join(', ')
      : data.detail
    if (response.status === 401 && !path.startsWith('/api/auth/login')) unauthorisedHandler(detail)
    throw new ApiError(detail || `Request failed (${response.status})`, response.status)
  }
  return data
}

export function oauthStartUrl(provider) {
  return `${apiBaseUrl}/api/auth/oauth/${provider}/start`
}
