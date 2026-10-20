import { stripRequestSuffix } from './notes-save-machine'

export type NotesListErrorParts = {
  /** The message without the "(GET /api/...)" request suffix; empty when there was none. */
  summary: string
  /** The full original message, kept for a Details disclosure; null when it adds nothing. */
  details: string | null
  /** The request never got an answer (offline, DNS, CORS, timeout). */
  unreachable: boolean
}

const UNREACHABLE_PATTERN =
  /failed to fetch|networkerror|network error|load failed|network request failed|timed out|timeout/i

/**
 * Split a notes list error into plain words and technical details (NL-16).
 *
 * Request errors end with "(GET /api/v1/notes/?...)". That path is useful for
 * diagnosis but noise in the error card, so it moves behind a disclosure.
 */
export const describeNotesListError = (
  message: string | null | undefined
): NotesListErrorParts => {
  const raw = String(message || '').trim()
  const summary = stripRequestSuffix(raw)
  return {
    summary,
    details: raw && raw !== summary ? raw : null,
    unreachable: UNREACHABLE_PATTERN.test(summary)
  }
}

/**
 * True when a list error means the notes server is down or unreachable, the
 * cases in which a save keeps the draft on this device (NS-05): the request
 * never got an answer, or the server answered with a 5xx error. A rate limit
 * (429) or an exception in the page itself is not the server being down.
 */
export const isNotesServerUnavailableError = (error: unknown): boolean => {
  if (error == null) return false
  const candidate = error as {
    name?: unknown
    code?: unknown
    status?: unknown
    response?: { status?: unknown }
    message?: unknown
  }
  if (candidate.name === 'AbortError' || candidate.code === 'REQUEST_ABORTED') return false
  const status = Number(candidate.status ?? candidate.response?.status)
  if (Number.isFinite(status) && status >= 100) return status >= 500 && status <= 599
  const message = typeof error === 'string' ? error : String(candidate.message || '')
  return UNREACHABLE_PATTERN.test(stripRequestSuffix(message))
}
