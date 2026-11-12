import { describe, expect, it } from 'vitest'
import {
  INITIAL_NOTE_SAVE_STATE,
  NOTE_AUTOSAVE_MAX_WAIT_MS,
  NOTE_SAVE_MAX_AUTOMATIC_RETRIES,
  NOTE_SAVE_RETRY_DELAYS_MS,
  canStartNoteSave,
  classifyNoteSaveError,
  nextAutosaveDelayMs,
  noteSaveIssueKind,
  noteSaveRetryDelayMs,
  stripRequestSuffix,
  toSaveIndicator,
  transitionNoteSave,
  type NoteSaveEvent,
  type NoteSaveFailure,
  type NoteSaveState,
  type NoteSaveStatus,
  type NoteSaveTrigger
} from '../notes-save-machine'
import { NOTE_AUTOSAVE_DELAY_MS } from '../notes-manager-utils'

// The notes save state machine (#3102: NS-01, NS-03, NS-N2, NS-05, NS-02, NS-06).

const network: NoteSaveFailure = { kind: 'network', status: null, message: 'Failed to fetch' }
const server: NoteSaveFailure = { kind: 'server', status: 503, message: 'Unavailable' }
const validation: NoteSaveFailure = { kind: 'validation', status: 400, message: 'Title is required' }
const conflict: NoteSaveFailure = { kind: 'conflict', status: 409, message: 'Version conflict' }
const auth: NoteSaveFailure = { kind: 'auth', status: 401, message: 'Not authenticated' }
const missing: NoteSaveFailure = { kind: 'missing', status: 404, message: 'Not found' }

const at = (status: NoteSaveStatus, extra: Partial<NoteSaveState> = {}): NoteSaveState => ({
  status,
  retryAttempt: 0,
  failure: null,
  ...extra
})

const ALL_STATUSES: NoteSaveStatus[] = [
  'idle',
  'dirty',
  'saving',
  'saved',
  'error-retryable',
  'error-fatal',
  'conflict',
  'offline-queued'
]

// Representative state for each status, as the hook would reach it.
const STATE_BY_STATUS: Record<NoteSaveStatus, NoteSaveState> = {
  idle: at('idle'),
  dirty: at('dirty'),
  saving: at('saving'),
  saved: at('saved'),
  'error-retryable': at('error-retryable', { retryAttempt: 1, failure: network }),
  'error-fatal': at('error-fatal', { failure: validation }),
  conflict: at('conflict', { failure: conflict }),
  'offline-queued': at('offline-queued', { retryAttempt: NOTE_SAVE_MAX_AUTOMATIC_RETRIES, failure: network })
}

const statusAfter = (from: NoteSaveStatus, event: NoteSaveEvent) =>
  transitionNoteSave(STATE_BY_STATUS[from], event).status

describe('notes save machine: transitions', () => {
  it('starts idle with no failure', () => {
    expect(INITIAL_NOTE_SAVE_STATE).toEqual({ status: 'idle', retryAttempt: 0, failure: null })
  })

  it.each<[NoteSaveStatus, NoteSaveStatus]>([
    ['idle', 'dirty'],
    ['dirty', 'dirty'],
    ['saving', 'saving'],
    ['saved', 'dirty'],
    ['error-retryable', 'dirty'],
    ['error-fatal', 'dirty'],
    ['conflict', 'conflict'],
    ['offline-queued', 'dirty']
  ])('an edit from %s goes to %s', (from, to) => {
    expect(statusAfter(from, { type: 'edited' })).toBe(to)
  })

  it('an edit after a fatal error clears the failure and the retry count', () => {
    expect(transitionNoteSave(STATE_BY_STATUS['error-fatal'], { type: 'edited' })).toEqual(at('dirty'))
  })

  it('an edit after a transient failure keeps the backoff and the failure so the notice stays put', () => {
    expect(transitionNoteSave(STATE_BY_STATUS['error-retryable'], { type: 'edited' })).toEqual(
      at('dirty', { retryAttempt: 1, failure: network })
    )
  })

  it('a repeated edit returns the same state, so marking dirty again never re-renders', () => {
    const dirty = at('dirty', { retryAttempt: 2, failure: network })
    expect(transitionNoteSave(dirty, { type: 'edited' })).toBe(dirty)
  })

  it('an edit during a conflict keeps the conflict (autosave stays paused)', () => {
    expect(transitionNoteSave(STATE_BY_STATUS.conflict, { type: 'edited' })).toBe(STATE_BY_STATUS.conflict)
  })

  it.each(ALL_STATUSES)('save-started from %s goes to saving and keeps the last failure', (from) => {
    const next = transitionNoteSave(STATE_BY_STATUS[from], { type: 'save-started' })
    expect(next.status).toBe('saving')
    expect(next.failure).toBe(STATE_BY_STATUS[from].failure)
    expect(next.retryAttempt).toBe(STATE_BY_STATUS[from].retryAttempt)
  })

  it('a 2xx acknowledgement is the only way to reach saved from saving', () => {
    const saving = at('saving', { retryAttempt: 2, failure: network })
    expect(transitionNoteSave(saving, { type: 'save-succeeded', hasNewerEdits: false })).toEqual(at('saved'))
    expect(transitionNoteSave(saving, { type: 'save-succeeded', hasNewerEdits: true })).toEqual(at('dirty'))
  })

  it.each<[NoteSaveFailure, NoteSaveStatus]>([
    [conflict, 'conflict'],
    [validation, 'error-fatal'],
    [auth, 'error-fatal'],
    [missing, 'error-fatal'],
    [network, 'error-retryable'],
    [server, 'error-retryable']
  ])('a %o failure goes to %s', (failure, to) => {
    const next = transitionNoteSave(at('saving'), {
      type: 'save-failed',
      failure,
      queuedOffline: true,
      hasNewerEdits: false
    })
    expect(next.status).toBe(to)
    expect(next.failure).toEqual(failure)
  })

  it('never counts non-retryable failures towards the retry budget', () => {
    for (const failure of [conflict, validation, auth, missing]) {
      const next = transitionNoteSave(at('saving', { retryAttempt: 2 }), {
        type: 'save-failed',
        failure,
        queuedOffline: true,
        hasNewerEdits: false
      })
      expect(next.retryAttempt).toBe(0)
    }
  })

  it('backs off transient failures, then hands the draft to the offline queue', () => {
    let state = at('dirty')
    const statuses: NoteSaveStatus[] = []
    for (let attempt = 1; attempt <= NOTE_SAVE_MAX_AUTOMATIC_RETRIES; attempt += 1) {
      state = transitionNoteSave(state, { type: 'save-started' })
      state = transitionNoteSave(state, {
        type: 'save-failed',
        failure: network,
        queuedOffline: true,
        hasNewerEdits: false
      })
      statuses.push(state.status)
      expect(state.retryAttempt).toBe(attempt)
    }
    expect(statuses).toEqual([
      ...Array(NOTE_SAVE_MAX_AUTOMATIC_RETRIES - 1).fill('error-retryable'),
      'offline-queued'
    ])
  })

  it('keeps retrying when the offline queue could not store the draft', () => {
    const exhausted = at('saving', { retryAttempt: NOTE_SAVE_MAX_AUTOMATIC_RETRIES + 3 })
    const next = transitionNoteSave(exhausted, {
      type: 'save-failed',
      failure: server,
      queuedOffline: false,
      hasNewerEdits: false
    })
    expect(next.status).toBe('error-retryable')
  })

  it('goes back to dirty when the user typed while a transient failure was in flight', () => {
    const next = transitionNoteSave(at('saving'), {
      type: 'save-failed',
      failure: network,
      queuedOffline: true,
      hasNewerEdits: true
    })
    expect(next).toEqual(at('dirty', { retryAttempt: 1, failure: network }))
  })

  it('lets newer edits retry a fatal failure (the edit may fix it) but never a conflict', () => {
    expect(
      transitionNoteSave(at('saving'), {
        type: 'save-failed',
        failure: validation,
        queuedOffline: false,
        hasNewerEdits: true
      })
    ).toEqual(at('dirty'))
    expect(
      transitionNoteSave(at('saving'), {
        type: 'save-failed',
        failure: conflict,
        queuedOffline: false,
        hasNewerEdits: true
      }).status
    ).toBe('conflict')
  })

  it.each<[NoteSaveStatus, NoteSaveStatus]>([
    ['idle', 'idle'],
    ['dirty', 'conflict'],
    ['saving', 'saving'],
    ['saved', 'saved'],
    ['error-retryable', 'conflict'],
    ['error-fatal', 'conflict'],
    ['conflict', 'conflict'],
    ['offline-queued', 'offline-queued']
  ])('a newer server version seen from %s goes to %s', (from, to) => {
    expect(statusAfter(from, { type: 'remote-changed' })).toBe(to)
  })

  it.each(ALL_STATUSES)('loading the server copy from %s resets the machine', (from) => {
    expect(transitionNoteSave(STATE_BY_STATUS[from], { type: 'loaded', acknowledged: true })).toEqual(at('saved'))
    expect(transitionNoteSave(STATE_BY_STATUS[from], { type: 'loaded', acknowledged: false })).toEqual(at('idle'))
  })

  it.each(ALL_STATUSES)('queuing offline from %s goes to offline-queued', (from) => {
    expect(statusAfter(from, { type: 'queued-offline' })).toBe('offline-queued')
  })

  it('restoring a local draft reflects its sync state', () => {
    expect(transitionNoteSave(at('idle'), { type: 'draft-restored', conflict: false }).status).toBe('offline-queued')
    expect(transitionNoteSave(at('idle'), { type: 'draft-restored', conflict: true })).toEqual(
      at('conflict', { failure: { kind: 'conflict', status: 409, message: '' } })
    )
  })

  it('a cancelled save returns to dirty or idle without inventing a failure', () => {
    const saving = at('saving', { retryAttempt: 1, failure: network })
    expect(transitionNoteSave(saving, { type: 'save-cancelled', hasPendingEdits: true })).toEqual(
      at('dirty', { retryAttempt: 1, failure: network })
    )
    expect(transitionNoteSave(saving, { type: 'save-cancelled', hasPendingEdits: false }).status).toBe('idle')
    expect(transitionNoteSave(STATE_BY_STATUS.saved, { type: 'save-cancelled', hasPendingEdits: false })).toBe(
      STATE_BY_STATUS.saved
    )
  })
})

describe('notes save machine: who may start a save', () => {
  const triggers: NoteSaveTrigger[] = ['auto', 'manual', 'retry', 'leave', 'switch', 'keep-mine']

  it('never starts a second save while one is in flight', () => {
    for (const trigger of triggers) expect(canStartNoteSave(at('saving'), trigger)).toBe(false)
  })

  it('only an explicit "keep mine" may save during a conflict', () => {
    for (const trigger of triggers) {
      expect(canStartNoteSave(STATE_BY_STATUS.conflict, trigger)).toBe(trigger === 'keep-mine')
    }
  })

  it.each<[NoteSaveStatus, boolean]>([
    ['idle', false],
    ['dirty', true],
    ['saved', false],
    ['error-retryable', true],
    ['error-fatal', false],
    ['offline-queued', false]
  ])('autosave from %s: %s', (status, allowed) => {
    expect(canStartNoteSave(STATE_BY_STATUS[status], 'auto')).toBe(allowed)
  })

  it('explicit saves are allowed outside a conflict', () => {
    for (const status of ALL_STATUSES.filter((value) => value !== 'saving' && value !== 'conflict')) {
      for (const trigger of ['manual', 'retry', 'leave', 'switch'] as NoteSaveTrigger[]) {
        expect(canStartNoteSave(STATE_BY_STATUS[status], trigger)).toBe(true)
      }
    }
  })
})

describe('notes save machine: timing', () => {
  const base = { now: 100_000, lastEditAt: 100_000, dirtySince: 100_000, lastFailureAt: 0 }

  it('debounces from the last edit', () => {
    expect(nextAutosaveDelayMs(at('dirty'), base)).toBe(NOTE_AUTOSAVE_DELAY_MS)
    expect(nextAutosaveDelayMs(at('dirty'), { ...base, now: base.now + 2_000 })).toBe(NOTE_AUTOSAVE_DELAY_MS - 2_000)
  })

  it('saves at most every max-wait while the user keeps typing', () => {
    const dirtySince = base.now - (NOTE_AUTOSAVE_MAX_WAIT_MS - 1_000)
    expect(nextAutosaveDelayMs(at('dirty'), { ...base, dirtySince })).toBe(1_000)
    expect(
      nextAutosaveDelayMs(at('dirty'), { ...base, dirtySince: base.now - NOTE_AUTOSAVE_MAX_WAIT_MS - 50 })
    ).toBe(0)
  })

  it('backs off retries after transient failures', () => {
    expect(NOTE_SAVE_RETRY_DELAYS_MS).toEqual([5_000, 15_000, 60_000])
    expect([1, 2, 3, 4, 9].map(noteSaveRetryDelayMs)).toEqual([5_000, 15_000, 60_000, 60_000, 60_000])
    const failedAt = base.now - 1_000
    expect(
      nextAutosaveDelayMs(at('error-retryable', { retryAttempt: 2, failure: network }), {
        ...base,
        lastFailureAt: failedAt
      })
    ).toBe(14_000)
  })

  it('keeps the backoff when edits arrive after a transient failure', () => {
    const state = at('dirty', { retryAttempt: 3, failure: network })
    expect(nextAutosaveDelayMs(state, { ...base, lastFailureAt: base.now - 10_000 })).toBe(50_000)
  })

  it.each<NoteSaveStatus>(['idle', 'saving', 'saved', 'error-fatal', 'conflict', 'offline-queued'])(
    'never schedules an autosave from %s',
    (status) => {
      expect(nextAutosaveDelayMs(STATE_BY_STATUS[status], base)).toBeNull()
    }
  )
})

describe('notes save machine: one status for the UI', () => {
  it.each<[NoteSaveStatus, boolean, string]>([
    ['idle', false, 'idle'],
    ['idle', true, 'dirty'],
    ['dirty', true, 'dirty'],
    ['saving', true, 'saving'],
    ['saved', false, 'saved'],
    ['saved', true, 'dirty'],
    ['error-retryable', true, 'retrying'],
    ['error-fatal', true, 'error'],
    ['conflict', true, 'conflict'],
    ['offline-queued', false, 'offline'],
    ['offline-queued', true, 'dirty']
  ])('%s (dirty=%s) shows %s', (status, isDirty, indicator) => {
    expect(toSaveIndicator(STATE_BY_STATUS[status], isDirty)).toBe(indicator)
  })

  it('shows "retrying" while edits wait behind a transient failure', () => {
    expect(toSaveIndicator(at('dirty', { retryAttempt: 1, failure: network }), true)).toBe('retrying')
  })

  it('only says saved after an acknowledged save or load', () => {
    for (const status of ALL_STATUSES) {
      const indicator = toSaveIndicator(STATE_BY_STATUS[status], false)
      expect(indicator === 'saved').toBe(status === 'saved')
    }
  })

  it.each<[NoteSaveState, string | null]>([
    [at('idle'), null],
    [at('dirty'), null],
    [at('saving'), null],
    [at('saved'), null],
    [STATE_BY_STATUS.conflict, 'conflict'],
    [STATE_BY_STATUS['error-fatal'], 'error'],
    [STATE_BY_STATUS['error-retryable'], 'retrying'],
    [STATE_BY_STATUS['offline-queued'], 'offline'],
    [at('dirty', { retryAttempt: 1, failure: network }), 'retrying'],
    [at('saving', { failure: validation }), 'error'],
    [at('saving', { failure: conflict }), 'conflict'],
    [at('saving', { retryAttempt: 1, failure: network }), 'retrying']
  ])('one issue surface: %o -> %s', (state, kind) => {
    expect(noteSaveIssueKind(state)).toBe(kind)
  })
})

describe('notes save machine: error classification', () => {
  const failureFor = (error: unknown) => classifyNoteSaveError(error)

  it.each<[unknown, string]>([
    [{ status: 409, message: 'Conflict (PUT /api/v1/notes/1)' }, 'conflict'],
    [new Error('expected-version mismatch'), 'conflict'],
    [{ status: 400, message: 'Title is required unless auto_title=true (POST /api/v1/notes/)' }, 'validation'],
    [{ status: 422, message: 'Unprocessable' }, 'validation'],
    [{ status: 413, message: 'Too large' }, 'validation'],
    [{ status: 401, message: 'Not authenticated' }, 'auth'],
    [{ status: 403, message: 'Forbidden' }, 'auth'],
    [{ status: 404, message: 'Not found' }, 'missing'],
    [{ status: 500, message: 'Internal error' }, 'server'],
    [{ status: 503, message: 'Unavailable' }, 'server'],
    [{ status: 429, message: 'Slow down' }, 'server'],
    [{ status: 408, message: 'Timeout' }, 'server'],
    [new TypeError('Failed to fetch'), 'network'],
    [new Error('Network unavailable'), 'network'],
    [{ status: 0, message: 'NetworkError when attempting to fetch resource.' }, 'network'],
    [{ response: { status: 502 }, message: 'Bad gateway' }, 'server']
  ])('classifies %o as %s', (error, kind) => {
    expect(failureFor(error)?.kind).toBe(kind)
  })

  it('ignores aborted requests', () => {
    expect(failureFor(new DOMException('Note selection changed', 'AbortError'))).toBeNull()
    expect(failureFor({ code: 'REQUEST_ABORTED', message: 'Aborted' })).toBeNull()
  })

  it('keeps the status and strips the request suffix from user-facing messages', () => {
    expect(failureFor({ status: 400, message: 'Title is required unless auto_title=true (POST /api/v1/notes/)' })).toEqual({
      kind: 'validation',
      status: 400,
      message: 'Title is required unless auto_title=true'
    })
    expect(stripRequestSuffix('Version conflict (PUT /api/v1/notes/a%20b)')).toBe('Version conflict')
    expect(stripRequestSuffix('No suffix here')).toBe('No suffix here')
  })
})
