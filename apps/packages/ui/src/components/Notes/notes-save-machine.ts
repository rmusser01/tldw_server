/**
 * Notes save state machine (#3102: NS-01, NS-03, NS-N2, NS-05, NS-02, NS-06).
 *
 * One pure description of where the editor's save pipeline is. The autosave
 * timer, retry policy, conflict flow, offline queue hand-off, status pill and
 * leave guard all read this state instead of keeping their own flags.
 * `useNotesEditorState` performs the I/O and feeds events in.
 *
 *   idle ──edited──▶ dirty ──save-started──▶ saving ──2xx──▶ saved
 *                     ▲                          │
 *                     │                          ├─409───────────▶ conflict (no autosave; user resolves)
 *                     │                          ├─400/401/404/422▶ error-fatal (no retry until an edit)
 *                     └──edited / backoff timer──┴─network/5xx───▶ error-retryable ─(retries spent)─▶ offline-queued
 */
import { NOTE_AUTOSAVE_DELAY_MS, type SaveIndicatorState } from './notes-manager-utils'

/** Backoff between automatic retries of a transient (network or 5xx) failure. */
export const NOTE_SAVE_RETRY_DELAYS_MS = [5_000, 15_000, 60_000] as const
/** Consecutive transient failures before the offline queue owns the draft. */
export const NOTE_SAVE_MAX_AUTOMATIC_RETRIES = NOTE_SAVE_RETRY_DELAYS_MS.length
/** Continuous typing still saves at least this often. */
export const NOTE_AUTOSAVE_MAX_WAIT_MS = 15_000
/** Error code for "the signed-in owner changed or is unknown" (never retried). */
export const NOTE_SAVE_OWNER_ERROR_CODE = 'NOTES_OWNER_UNAVAILABLE'

export type NoteSaveStatus =
  | 'idle'
  | 'dirty'
  | 'saving'
  | 'saved'
  | 'error-retryable'
  | 'error-fatal'
  | 'conflict'
  | 'offline-queued'

export type NoteSaveFailureKind = 'network' | 'server' | 'validation' | 'conflict' | 'auth' | 'missing'

export type NoteSaveFailure = {
  kind: NoteSaveFailureKind
  status: number | null
  /** Server or transport message without the "(METHOD /path)" suffix. */
  message: string
}

export type NoteSaveState = {
  status: NoteSaveStatus
  /** Consecutive transient failures since the last acknowledged save. */
  retryAttempt: number
  /** Last failure; kept while retrying so the notice is not remounted per attempt. */
  failure: NoteSaveFailure | null
}

/** Who asked for a save. Only "keep-mine" may write over a conflict. */
export type NoteSaveTrigger = 'auto' | 'manual' | 'retry' | 'leave' | 'switch' | 'keep-mine'

export type NoteSaveEvent =
  | { type: 'edited' }
  | { type: 'loaded'; acknowledged: boolean }
  | { type: 'draft-restored'; conflict: boolean }
  | { type: 'save-started' }
  | { type: 'save-succeeded'; hasNewerEdits: boolean }
  | { type: 'save-failed'; failure: NoteSaveFailure; queuedOffline: boolean; hasNewerEdits: boolean }
  | { type: 'save-cancelled'; hasPendingEdits: boolean }
  | { type: 'queued-offline' }
  | { type: 'remote-changed' }

export const INITIAL_NOTE_SAVE_STATE: NoteSaveState = Object.freeze({
  status: 'idle',
  retryAttempt: 0,
  failure: null
}) as NoteSaveState

const CLEAN_FAILURE = { retryAttempt: 0, failure: null } as const

export const isRetryableNoteSaveFailure = (kind: NoteSaveFailureKind): boolean =>
  kind === 'network' || kind === 'server'

const failedSave = (
  state: NoteSaveState,
  event: Extract<NoteSaveEvent, { type: 'save-failed' }>
): NoteSaveState => {
  const { failure } = event
  if (failure.kind === 'conflict') return { status: 'conflict', retryAttempt: 0, failure }
  if (!isRetryableNoteSaveFailure(failure.kind)) {
    // A newer edit may already fix the cause (e.g. the user typed a title).
    return event.hasNewerEdits
      ? { status: 'dirty', ...CLEAN_FAILURE }
      : { status: 'error-fatal', retryAttempt: 0, failure }
  }
  const retryAttempt = state.retryAttempt + 1
  if (event.hasNewerEdits) return { status: 'dirty', retryAttempt, failure }
  if (retryAttempt >= NOTE_SAVE_MAX_AUTOMATIC_RETRIES && event.queuedOffline) {
    return { status: 'offline-queued', retryAttempt, failure }
  }
  return { status: 'error-retryable', retryAttempt, failure }
}

export function transitionNoteSave(state: NoteSaveState, event: NoteSaveEvent): NoteSaveState {
  switch (event.type) {
    case 'edited':
      // Same object when nothing changes: marking dirty again must not re-render.
      if (state.status === 'conflict' || state.status === 'saving' || state.status === 'dirty') return state
      if (state.status === 'error-fatal') return { status: 'dirty', ...CLEAN_FAILURE }
      return { status: 'dirty', retryAttempt: state.retryAttempt, failure: state.failure }
    case 'loaded':
      return { status: event.acknowledged ? 'saved' : 'idle', ...CLEAN_FAILURE }
    case 'draft-restored':
      return event.conflict
        ? { status: 'conflict', retryAttempt: 0, failure: { kind: 'conflict', status: 409, message: '' } }
        : { status: 'offline-queued', ...CLEAN_FAILURE }
    case 'save-started':
      return { ...state, status: 'saving' }
    case 'save-succeeded':
      return { status: event.hasNewerEdits ? 'dirty' : 'saved', ...CLEAN_FAILURE }
    case 'save-failed':
      return failedSave(state, event)
    case 'save-cancelled':
      if (state.status !== 'saving') return state
      return { ...state, status: event.hasPendingEdits ? 'dirty' : 'idle' }
    case 'queued-offline':
      return { ...state, status: 'offline-queued' }
    case 'remote-changed':
      if (state.status === 'dirty' || state.status === 'error-retryable' || state.status === 'error-fatal') {
        return {
          status: 'conflict',
          retryAttempt: 0,
          failure: { kind: 'conflict', status: null, message: '' }
        }
      }
      return state
  }
}

export function canStartNoteSave(state: NoteSaveState, trigger: NoteSaveTrigger): boolean {
  if (state.status === 'saving') return false
  if (state.status === 'conflict') return trigger === 'keep-mine'
  if (trigger === 'auto') return state.status === 'dirty' || state.status === 'error-retryable'
  return true
}

export function noteSaveRetryDelayMs(retryAttempt: number): number {
  const index = Math.min(Math.max(retryAttempt, 1), NOTE_SAVE_RETRY_DELAYS_MS.length) - 1
  return NOTE_SAVE_RETRY_DELAYS_MS[index]
}

export type NoteAutosaveTiming = {
  now: number
  /** When the latest edit happened. */
  lastEditAt: number
  /** When the editor last went from clean to dirty. */
  dirtySince: number
  /** When the latest transient failure was recorded. */
  lastFailureAt: number
}

/** Milliseconds until the next automatic save, or null when autosave must not run. */
export function nextAutosaveDelayMs(state: NoteSaveState, timing: NoteAutosaveTiming): number | null {
  const backoffDue =
    state.retryAttempt > 0 ? timing.lastFailureAt + noteSaveRetryDelayMs(state.retryAttempt) : -Infinity
  if (state.status === 'dirty') {
    const debounceDue = timing.lastEditAt + NOTE_AUTOSAVE_DELAY_MS
    const maxWaitDue = timing.dirtySince + NOTE_AUTOSAVE_MAX_WAIT_MS
    return Math.max(0, Math.max(Math.min(debounceDue, maxWaitDue), backoffDue) - timing.now)
  }
  if (state.status === 'error-retryable') return Math.max(0, backoffDue - timing.now)
  return null
}

export type NoteSaveIssueKind = 'conflict' | 'error' | 'retrying' | 'offline'

/** Which problem (if any) the single save-issue surface should show. */
export function noteSaveIssueKind(state: NoteSaveState): NoteSaveIssueKind | null {
  if (state.status === 'conflict') return 'conflict'
  if (state.status === 'error-fatal') return 'error'
  if (state.status === 'error-retryable') return 'retrying'
  if (state.status === 'offline-queued') return 'offline'
  if ((state.status === 'saving' || state.status === 'dirty') && state.failure) {
    if (state.failure.kind === 'conflict') return 'conflict'
    return isRetryableNoteSaveFailure(state.failure.kind) ? 'retrying' : 'error'
  }
  return null
}

/** The one status the pill shows. "saved" only follows a 2xx save or load. */
export function toSaveIndicator(state: NoteSaveState, isDirty: boolean): SaveIndicatorState {
  switch (state.status) {
    case 'saving':
      return 'saving'
    case 'conflict':
      return 'conflict'
    case 'error-fatal':
      return 'error'
    case 'error-retryable':
      return 'retrying'
    case 'dirty':
      return state.failure && isRetryableNoteSaveFailure(state.failure.kind) ? 'retrying' : 'dirty'
    case 'offline-queued':
      return isDirty ? 'dirty' : 'offline'
    case 'saved':
      return isDirty ? 'dirty' : 'saved'
    case 'idle':
      return isDirty ? 'dirty' : 'idle'
  }
}

const REQUEST_SUFFIX = /\s*\((?:GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS) \/[^)]*\)\s*$/

/** Drop the "(METHOD /api/...)" suffix that request errors carry. */
export const stripRequestSuffix = (message: string): string =>
  String(message || '').replace(REQUEST_SUFFIX, '').trim()

const readHttpStatus = (error: unknown): number | null => {
  const candidate = error as { status?: unknown; response?: { status?: unknown } } | null
  const raw = Number(candidate?.status ?? candidate?.response?.status)
  return Number.isFinite(raw) && raw >= 100 && raw <= 599 ? Math.trunc(raw) : null
}

const isAbortError = (error: unknown): boolean => {
  const candidate = error as { name?: unknown; code?: unknown } | null
  return candidate?.name === 'AbortError' || candidate?.code === 'REQUEST_ABORTED'
}

/** Map a failed save to a failure class, or null for an aborted request. */
export function classifyNoteSaveError(error: unknown): NoteSaveFailure | null {
  if (isAbortError(error)) return null
  const status = readHttpStatus(error)
  const rawMessage =
    typeof error === 'string' ? error : String((error as { message?: unknown } | null)?.message || '')
  const message = stripRequestSuffix(rawMessage)
  const lower = rawMessage.toLowerCase()
  const failure = (kind: NoteSaveFailureKind): NoteSaveFailure => ({ kind, status, message })
  if (
    status === 409 ||
    lower.includes('expected-version') ||
    lower.includes('expected_version') ||
    lower.includes('version mismatch')
  ) {
    return failure('conflict')
  }
  if (status === 401 || status === 403) return failure('auth')
  if (status === 404 || status === 410) return failure('missing')
  if (status === 408 || status === 425 || status === 429 || (status != null && status >= 500)) {
    return failure('server')
  }
  if (status != null && status >= 400) return failure('validation')
  if ((error as { code?: unknown } | null)?.code === NOTE_SAVE_OWNER_ERROR_CODE) return failure('auth')
  // No HTTP status: the request never got an answer (offline, DNS, CORS, timeout).
  return failure('network')
}
