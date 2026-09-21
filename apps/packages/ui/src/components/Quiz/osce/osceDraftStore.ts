import type {
  OsceAttempt,
  OsceAttemptPatch,
  OsceChecklistSelection
} from "@/services/osce"

export const OSCE_DRAFT_TTL_MS = 24 * 60 * 60 * 1000

const OSCE_DRAFT_PREFIX = "tldw:quiz:osce:draft:v1"

export type OsceWritablePatch = Omit<OsceAttemptPatch, "expected_version">

export type OsceDraft = {
  attemptId: number
  version: number
  notes: string
  checklistSelections: Record<string, OsceChecklistSelection>
  rubricSelections: Record<string, string>
  savedAt: number
}

type DraftInput = Partial<OsceDraft> & Pick<OsceDraft, "attemptId" | "version">

type StorageOptions = {
  storage?: Storage
  now?: number
}

const browserStorage = (): Storage | null => {
  if (typeof window === "undefined") return null
  try {
    return window.localStorage
  } catch {
    return null
  }
}

const resolveStorage = (storage?: Storage): Storage | null => storage ?? browserStorage()

const normalizeScope = (userScope: string | number): string => {
  const normalized = String(userScope).trim()
  if (!normalized) throw new Error("OSCE draft storage requires a user scope.")
  return normalized
}

export const osceDraftKey = (userScope: string | number, attemptId: number): string => {
  if (!Number.isInteger(attemptId) || attemptId <= 0) {
    throw new Error("OSCE draft storage requires a positive attempt ID.")
  }
  return `${OSCE_DRAFT_PREFIX}:${encodeURIComponent(normalizeScope(userScope))}:${attemptId}`
}

const sanitizeChecklistSelections = (value: unknown): Record<string, OsceChecklistSelection> => {
  if (!value || typeof value !== "object" || Array.isArray(value)) return {}
  return Object.fromEntries(
    Object.entries(value).filter(
      (entry): entry is [string, OsceChecklistSelection] =>
        entry[0].length > 0 && (entry[1] === "met" || entry[1] === "not_met")
    )
  )
}

const sanitizeRubricSelections = (value: unknown): Record<string, string> => {
  if (!value || typeof value !== "object" || Array.isArray(value)) return {}
  return Object.fromEntries(
    Object.entries(value).filter(
      (entry): entry is [string, string] =>
        entry[0].length > 0 && typeof entry[1] === "string" && entry[1].length > 0
    )
  )
}

const sanitizeDraft = (input: DraftInput, now: number): OsceDraft | null => {
  if (!Number.isInteger(input.attemptId) || input.attemptId <= 0) return null
  if (!Number.isInteger(input.version) || input.version < 1) return null
  const savedAt = typeof input.savedAt === "number" && Number.isFinite(input.savedAt)
    ? input.savedAt
    : now
  return {
    attemptId: input.attemptId,
    version: input.version,
    notes: typeof input.notes === "string" ? input.notes.slice(0, 10_000) : "",
    checklistSelections: sanitizeChecklistSelections(input.checklistSelections),
    rubricSelections: sanitizeRubricSelections(input.rubricSelections),
    savedAt
  }
}

export const saveOsceDraft = (
  userScope: string | number,
  input: DraftInput,
  options: StorageOptions = {}
): boolean => {
  const storage = resolveStorage(options.storage)
  const draft = sanitizeDraft(input, options.now ?? Date.now())
  if (!storage || !draft) return false
  try {
    storage.setItem(osceDraftKey(userScope, draft.attemptId), JSON.stringify(draft))
    return true
  } catch {
    return false
  }
}

export const clearOsceDraft = (
  userScope: string | number,
  attemptId: number,
  options: Pick<StorageOptions, "storage"> = {}
): void => {
  const storage = resolveStorage(options.storage)
  if (!storage) return
  try {
    storage.removeItem(osceDraftKey(userScope, attemptId))
  } catch {
    // Storage failure is non-fatal; the TTL still bounds retained data.
  }
}

export const readOsceDraft = (
  userScope: string | number,
  attemptId: number,
  options: StorageOptions = {}
): OsceDraft | null => {
  const storage = resolveStorage(options.storage)
  if (!storage) return null
  const key = osceDraftKey(userScope, attemptId)
  try {
    const raw = storage.getItem(key)
    if (!raw) return null
    const parsed = JSON.parse(raw) as DraftInput
    const draft = sanitizeDraft(parsed, options.now ?? Date.now())
    if (!draft || draft.attemptId !== attemptId) {
      storage.removeItem(key)
      return null
    }
    const age = (options.now ?? Date.now()) - draft.savedAt
    if (age >= OSCE_DRAFT_TTL_MS) {
      storage.removeItem(key)
      return null
    }
    return draft
  } catch {
    try {
      storage.removeItem(key)
    } catch {
      // Ignore cleanup failure after malformed storage.
    }
    return null
  }
}

const writableFromAttempt = (attempt: OsceAttempt): Required<OsceWritablePatch> => ({
  notes: attempt.notes,
  checklist_selections: attempt.state === "in_progress" ? {} : attempt.checklist_selections,
  rubric_selections: attempt.state === "in_progress" ? {} : attempt.rubric_selections
})

const mergeWritable = (
  current: Required<OsceWritablePatch>,
  patch: OsceWritablePatch
): Required<OsceWritablePatch> => ({
  notes: patch.notes === undefined ? current.notes : patch.notes ?? "",
  checklist_selections: patch.checklist_selections === undefined
    ? current.checklist_selections
    : sanitizeChecklistSelections(patch.checklist_selections),
  rubric_selections: patch.rubric_selections === undefined
    ? current.rubric_selections
    : sanitizeRubricSelections(patch.rubric_selections)
})

export interface OsceSaveQueue {
  stage(patch: OsceWritablePatch): void
  enqueue(patch: OsceWritablePatch): Promise<OsceAttempt>
  enqueueStaged(): Promise<OsceAttempt | null>
  flush(): Promise<void>
  hasPending(): boolean
  hasConflict(): boolean
  getAcknowledgedVersion(): number
  replaceAcknowledgedAttempt(attempt: OsceAttempt): boolean
  resolveConflict(attempt: OsceAttempt, resolution: "discard" | "reapply"): boolean
}

type CreateQueueOptions = {
  attempt: OsceAttempt
  initialExpectedVersion?: number
  userScope: string | number
  update: (attemptId: number, patch: OsceAttemptPatch) => Promise<OsceAttempt>
  onAcknowledged?: (attempt: OsceAttempt, isCurrentRevision: boolean) => void
  storage?: Storage
  now?: () => number
}

export const createOsceSaveQueue = ({
  attempt,
  initialExpectedVersion,
  userScope,
  update,
  onAcknowledged,
  storage,
  now = Date.now
}: CreateQueueOptions): OsceSaveQueue => {
  const attemptId = attempt.id
  let acknowledged = attempt
  let acknowledgedVersion = initialExpectedVersion ?? attempt.version
  let writable = writableFromAttempt(attempt)
  let revision = 0
  let scheduledRevision = 0
  let pending = 0
  let dirty = false
  let conflicted = false
  let lastError: unknown = null
  let tail: Promise<void> = Promise.resolve()

  const persistCurrent = () => saveOsceDraft(userScope, {
    attemptId,
    version: acknowledgedVersion,
    notes: writable.notes ?? "",
    checklistSelections: writable.checklist_selections ?? {},
    rubricSelections: writable.rubric_selections ?? {},
    savedAt: now()
  }, { storage, now: now() })

  const stage = (patch: OsceWritablePatch) => {
    writable = mergeWritable(writable, patch)
    revision += 1
    dirty = true
    persistCurrent()
  }

  const enqueueStaged = (): Promise<OsceAttempt | null> => {
    if (conflicted) {
      return Promise.reject(lastError ?? new Error("OSCE draft conflict requires resolution."))
    }
    if (scheduledRevision === revision) return tail.then(() => null)

    const queuedRevision = revision
    const queuedWritable = { ...writable }
    scheduledRevision = queuedRevision
    pending += 1

    const operation = tail.then(async () => {
      try {
        const updated = await update(attemptId, {
          expected_version: acknowledgedVersion,
          notes: queuedWritable.notes,
          checklist_selections: queuedWritable.checklist_selections,
          rubric_selections: queuedWritable.rubric_selections
        })
        acknowledged = updated
        acknowledgedVersion = updated.version
        const isCurrentRevision = revision === queuedRevision
        if (isCurrentRevision) {
          writable = writableFromAttempt(updated)
          dirty = false
          conflicted = false
          lastError = null
          clearOsceDraft(userScope, attemptId, { storage })
        } else {
          lastError = null
          persistCurrent()
        }
        onAcknowledged?.(updated, isCurrentRevision)
        return updated
      } catch (error) {
        lastError = error
        conflicted = Number((error as { status?: unknown } | null)?.status) === 409
        dirty = true
        persistCurrent()
        throw error
      } finally {
        pending = Math.max(0, pending - 1)
      }
    })

    tail = operation.then(() => undefined, () => undefined)
    return operation
  }

  const enqueue = (patch: OsceWritablePatch): Promise<OsceAttempt> => {
    stage(patch)
    return enqueueStaged().then((updated) => updated ?? acknowledged)
  }

  return {
    stage,
    enqueue,
    enqueueStaged,
    async flush() {
      if (scheduledRevision !== revision) await enqueueStaged()
      await tail
      if (lastError) throw lastError
    },
    hasPending: () => pending > 0 || dirty,
    hasConflict: () => conflicted,
    getAcknowledgedVersion: () => acknowledgedVersion,
    replaceAcknowledgedAttempt(nextAttempt) {
      if (nextAttempt.id !== attemptId) return false
      if (nextAttempt.version < acknowledgedVersion) return false
      if (dirty || pending > 0 || conflicted) return false
      acknowledged = nextAttempt
      acknowledgedVersion = nextAttempt.version
      writable = writableFromAttempt(nextAttempt)
      return true
    },
    resolveConflict(nextAttempt, resolution) {
      if (!conflicted || nextAttempt.id !== attemptId || nextAttempt.version < acknowledgedVersion) {
        return false
      }
      acknowledged = nextAttempt
      acknowledgedVersion = nextAttempt.version
      lastError = null
      conflicted = false
      revision += 1
      if (resolution === "discard") {
        writable = writableFromAttempt(nextAttempt)
        dirty = false
        scheduledRevision = revision
        clearOsceDraft(userScope, attemptId, { storage })
      } else {
        dirty = true
        persistCurrent()
      }
      return true
    }
  }
}
