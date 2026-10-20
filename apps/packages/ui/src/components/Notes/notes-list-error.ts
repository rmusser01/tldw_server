import {
  classifyNoteSaveError,
  isRetryableNoteSaveFailure,
  stripRequestSuffix
} from './notes-save-machine'

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
 * cases in which a save keeps the draft on this device (NS-05).
 */
export const isNotesServerUnavailableError = (error: unknown): boolean => {
  if (error == null) return false
  const failure = classifyNoteSaveError(error)
  return failure != null && isRetryableNoteSaveFailure(failure.kind)
}
