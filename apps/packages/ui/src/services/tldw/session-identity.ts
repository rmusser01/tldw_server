// Per-install id sent at sign-in so the server can group one browser's own
// sessions, and the "signed in elsewhere" notice. Sign-in must never fail
// because of any of this, so every storage and crypto access is guarded.
// The id is never logged.
export const INSTALL_ID_HEADER = "X-TLDW-Device-Id"
export const SESSION_REPLACED_MESSAGE =
  "You were signed out because your account was signed in on another device."

const INSTALL_ID_KEY = "tldw:install-id"
const SESSION_REPLACED_KEY = "tldw:session-replaced"
const INSTALL_ID_PATTERN = /^[A-Za-z0-9_-]{8,64}$/

export const createRandomId = (): string | null => {
  try {
    const c = globalThis.crypto
    if (typeof c?.randomUUID === "function") return c.randomUUID()
    if (typeof c?.getRandomValues === "function") {
      const bytes = c.getRandomValues(new Uint8Array(16))
      return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("")
    }
  } catch {
    // fall through
  }
  return null
}

export const getInstallIdHeaders = (): Record<string, string> => {
  try {
    let id = localStorage.getItem(INSTALL_ID_KEY)
    if (!id || !INSTALL_ID_PATTERN.test(id)) {
      id = createRandomId()
      if (!id) return {}
      localStorage.setItem(INSTALL_ID_KEY, id)
    }
    return { [INSTALL_ID_HEADER]: id }
  } catch {
    return {}
  }
}

export const noteSessionReplaced = (error: unknown): void => {
  try {
    const e = error as {
      status?: number
      detail?: { code?: string }
      details?: { detail?: { code?: string } }
    } | null
    const detail = e?.details?.detail ?? e?.detail
    if (e?.status === 401 && detail?.code === "session_replaced") {
      localStorage.setItem(SESSION_REPLACED_KEY, "1")
    }
  } catch {
    // storage unavailable: the notice is best-effort
  }
}

export const hasSessionReplacedNotice = (): boolean => {
  try {
    return localStorage.getItem(SESSION_REPLACED_KEY) === "1"
  } catch {
    return false
  }
}

export const clearSessionReplacedNotice = (): void => {
  try {
    localStorage.removeItem(SESSION_REPLACED_KEY)
  } catch {
    // ignore
  }
}
