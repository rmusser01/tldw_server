import type {
  PersistedQuickIngestTracking,
  ReattachedQuickIngestSnapshot,
} from "@/components/Common/QuickIngest/types"

type SessionStatus = "running" | "completed" | "failed" | "cancelled"

export type QuickIngestSessionStartAck = {
  ok: boolean
  sessionId: string
  error?: string
}

export type QuickIngestSessionCancelResponse = {
  ok: boolean
  error?: string
}

export type QuickIngestSessionRunContext = {
  sessionId: string
  isCancelled: () => boolean
  registerAbortController: (controller: AbortController) => void
  setJobIds: (jobIds: number[]) => void
  setRunTracking: (tracking: PersistedQuickIngestTracking) => Promise<void>
  emitProgress: (payload: Record<string, unknown>) => void | Promise<void>
}

export type QuickIngestSessionRunResult = {
  results: Array<Record<string, unknown>>
  summary?: Record<string, unknown>
  reviewRequired?: Array<Record<string, unknown>>
}

type QuickIngestCompactSessionBase = {
  version: 1
  sessionId: string
  generation: string
  attemptToken: string
}

export type QuickIngestCompactStartSession =
  QuickIngestCompactSessionBase & {
  kind: "start"
  occurrenceIds: string[]
  startedAt: number
}

export type QuickIngestCompactActiveRunSession =
  QuickIngestCompactSessionBase & {
  kind: "run"
  sessionId: string
  runId: string
  submissionState?: PersistedQuickIngestTracking["submissionState"]
  occurrenceIds: string[]
  jobIdToItemId: Record<string, string>
  startedAt: number
}

export type QuickIngestCompactTerminalSession =
  QuickIngestCompactSessionBase & {
    kind: "terminal"
    runId: string
    expiresAt: number
    event: QuickIngestRuntimeEvent
  }

export type QuickIngestCompactReviewSession =
  QuickIngestCompactSessionBase & {
    kind: "review"
    expiresAt: number
    event: QuickIngestRuntimeEvent
  }

export type QuickIngestCompactRunSession =
  | QuickIngestCompactStartSession
  | QuickIngestCompactActiveRunSession
  | QuickIngestCompactTerminalSession
  | QuickIngestCompactReviewSession

type RuntimeDeps = {
  run: (
    payload: Record<string, unknown>,
    context: QuickIngestSessionRunContext
  ) => Promise<QuickIngestSessionRunResult>
  emit: (type: string, payload: Record<string, unknown>) => void | Promise<void>
  saveRunSession?: (
    record: QuickIngestCompactRunSession | null,
    sessionId?: string,
    expectedRunId?: string,
    expectedGeneration?: string
  ) => boolean | void | Promise<boolean | void>
  loadRunSessions?: () => unknown[] | Promise<unknown[]>
  reattachRun?: (
    tracking: PersistedQuickIngestTracking,
    options: { transportPreference: "poll" }
  ) => Promise<ReattachedQuickIngestSnapshot>
  cancelRun?: (
    tracking: PersistedQuickIngestTracking,
    reason: string
  ) => Promise<QuickIngestSessionCancelResponse>
  createSessionId?: () => string
}

type QuickIngestSession = {
  sessionId: string
  generation: string
  attemptToken: string
  status: SessionStatus
  cancelled: boolean
  cancelRequested: boolean
  jobIds: number[]
  runRecord: QuickIngestCompactActiveRunSession | null
  runRecordPersisted: boolean
  runPersistenceRetryAt: number
  abortControllers: Set<AbortController>
  pollTimer: ReturnType<typeof setTimeout> | null
}

const MAX_COMPACT_OCCURRENCES = 500
const MAX_COMPACT_ID_LENGTH = 255
const MAX_COMPACT_ERROR_LENGTH = 2_000
const MAX_COMPACT_TERMINAL_BYTES = 512 * 1_024
const TERMINAL_TOMBSTONE_TTL_MS = 24 * 60 * 60 * 1_000
const COMPACT_SUBMISSION_STATES = new Set<
  NonNullable<PersistedQuickIngestTracking["submissionState"]>
>([
  "creating_run",
  "run_created",
  "submitting",
  "cleanup_required",
  "acknowledged",
])

type QuickIngestRuntimeEvent = {
  type: string
  payload: Record<string, unknown>
}

const TERMINAL_EVENT_TYPES = new Set([
  "tldw:quick-ingest/completed",
  "tldw:quick-ingest/failed",
  "tldw:quick-ingest/cancelled",
])
const REVIEW_REQUIRED_REASONS = new Set([
  "duplicate_action_required",
  "duplicate_no_longer_present",
  "duplicate_target_changed",
  "invalid_duplicate_override",
  "unknown_review_override",
  "in_run_duplicate_requires_processing_or_skip",
])
const REVIEW_EVIDENCE_KINDS = new Set(["library", "in_run", "none"])
const REVIEW_ALLOWED_ACTIONS = new Set([
  "skip",
  "include_existing",
  "update_metadata_only",
  "overwrite",
])

const isCompactId = (value: unknown): value is string =>
  typeof value === "string" &&
  value.length > 0 &&
  value.length <= MAX_COMPACT_ID_LENGTH &&
  value === value.trim()

const normalizeIds = (values: unknown): string[] =>
  Array.from(
    new Set(
      (Array.isArray(values) ? values : [])
        .map((value) => String(value || "").trim())
        .filter((value) => isCompactId(value))
    )
  ).slice(0, MAX_COMPACT_OCCURRENCES)

const compactString = (value: unknown, maxLength = MAX_COMPACT_ERROR_LENGTH) =>
  String(value || "").trim().slice(0, maxLength)

const serializedByteLength = (value: unknown): number =>
  new TextEncoder().encode(JSON.stringify(value)).byteLength

const compactTerminalResult = (value: unknown) => {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null
  const result = value as Record<string, unknown>
  const id = compactString(result.id, MAX_COMPACT_ID_LENGTH)
  if (!isCompactId(id)) return null
  const status = compactString(result.status, 32)
  const type = compactString(result.type, 64)
  const data =
    result.data && typeof result.data === "object" && !Array.isArray(result.data)
      ? Object.fromEntries(
          ["media_id", "outcome", "title", "collection_item_id"]
            .flatMap((key) => {
              const entry = (result.data as Record<string, unknown>)[key]
              if (
                typeof entry !== "string" &&
                typeof entry !== "number" &&
                entry !== null
              ) {
                return []
              }
              return [[key, typeof entry === "string" ? compactString(entry) : entry]]
            })
        )
      : undefined
  return {
    id,
    ...(status ? { status } : {}),
    ...(type ? { type } : {}),
    ...(data && Object.keys(data).length > 0 ? { data } : {}),
    ...(result.error ? { error: compactString(result.error) } : {}),
  }
}

const compactTerminalEvent = (
  event: QuickIngestRuntimeEvent,
  sessionId: string,
  runId: string
): QuickIngestRuntimeEvent | null => {
  if (!TERMINAL_EVENT_TYPES.has(event.type)) return null
  const payload = event.payload || {}
  if (
    compactString(payload.sessionId, MAX_COMPACT_ID_LENGTH) !== sessionId ||
    compactString(payload.runId, MAX_COMPACT_ID_LENGTH) !== runId
  ) {
    return null
  }
  const rawResults = Array.isArray(payload.results) ? payload.results : []
  if (rawResults.length > MAX_COMPACT_OCCURRENCES) return null
  const results = rawResults.map(compactTerminalResult)
  if (results.some((result) => result === null)) return null
  const compacted = {
    type: event.type,
    payload: {
      sessionId,
      runId,
      results,
      ...(payload.error ? { error: compactString(payload.error) } : {}),
      ...(payload.reason ? { reason: compactString(payload.reason) } : {}),
    },
  }
  return serializedByteLength(compacted) <= MAX_COMPACT_TERMINAL_BYTES
    ? compacted
    : null
}

const compactEssentialTerminalEvent = (
  event: QuickIngestRuntimeEvent,
  sessionId: string,
  runId: string
): QuickIngestRuntimeEvent | null => {
  if (!TERMINAL_EVENT_TYPES.has(event.type)) return null
  const payload = event.payload || {}
  if (
    compactString(payload.sessionId, MAX_COMPACT_ID_LENGTH) !== sessionId ||
    compactString(payload.runId, MAX_COMPACT_ID_LENGTH) !== runId
  ) {
    return null
  }
  const rawResults = Array.isArray(payload.results) ? payload.results : []
  if (rawResults.length > MAX_COMPACT_OCCURRENCES) return null
  const results = rawResults.map((value) => {
    const compacted = compactTerminalResult(value)
    if (!compacted) return null
    const outcome =
      compacted.data && typeof compacted.data.outcome === "string"
        ? compactString(compacted.data.outcome, 64)
        : ""
    return {
      id: compacted.id,
      ...(compacted.status ? { status: compacted.status } : {}),
      ...(outcome ? { data: { outcome } } : {}),
    }
  })
  if (results.some((result) => result === null)) return null
  const compacted = {
    type: event.type,
    payload: { sessionId, runId, results },
  }
  return serializedByteLength(compacted) <= MAX_COMPACT_TERMINAL_BYTES
    ? compacted
    : null
}

const compactReviewRequiredItem = (value: unknown) => {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null
  const item = value as Record<string, unknown>
  const occurrenceId = compactString(item.occurrenceId, MAX_COMPACT_ID_LENGTH)
  const reason = compactString(item.reason, 64)
  const evidence =
    item.evidence &&
    typeof item.evidence === "object" &&
    !Array.isArray(item.evidence)
      ? (item.evidence as Record<string, unknown>)
      : null
  const kind = compactString(evidence?.kind, 32)
  const existingMediaId = evidence?.existingMediaId
  const duplicateOfOccurrenceId = evidence?.duplicateOfOccurrenceId
  const allowedActions = Array.isArray(item.allowedActions)
    ? item.allowedActions.map((action) => compactString(action, 32))
    : []
  if (
    !isCompactId(occurrenceId) ||
    !REVIEW_REQUIRED_REASONS.has(reason) ||
    !evidence ||
    !REVIEW_EVIDENCE_KINDS.has(kind) ||
    allowedActions.length > 4 ||
    allowedActions.some((action) => !REVIEW_ALLOWED_ACTIONS.has(action)) ||
    (existingMediaId !== null &&
      (typeof existingMediaId !== "number" ||
        !Number.isSafeInteger(existingMediaId) ||
        existingMediaId <= 0)) ||
    (duplicateOfOccurrenceId !== null &&
      !isCompactId(duplicateOfOccurrenceId)) ||
    (kind === "library" &&
      (existingMediaId === null || duplicateOfOccurrenceId !== null)) ||
    (kind === "in_run" &&
      (existingMediaId !== null || duplicateOfOccurrenceId === null)) ||
    (kind === "none" &&
      (existingMediaId !== null || duplicateOfOccurrenceId !== null))
  ) {
    return null
  }
  return {
    occurrenceId,
    reason,
    evidence: {
      kind,
      existingMediaId: existingMediaId as number | null,
      duplicateOfOccurrenceId: duplicateOfOccurrenceId as string | null,
    },
    allowedActions,
  }
}

const compactReviewEvent = (
  event: QuickIngestRuntimeEvent,
  sessionId: string
): QuickIngestRuntimeEvent | null => {
  if (event.type !== "tldw:quick-ingest/review-required") return null
  const payload = event.payload || {}
  if (compactString(payload.sessionId, MAX_COMPACT_ID_LENGTH) !== sessionId) {
    return null
  }
  const rawItems = Array.isArray(payload.reviewRequired)
    ? payload.reviewRequired
    : []
  if (rawItems.length === 0 || rawItems.length > MAX_COMPACT_OCCURRENCES) {
    return null
  }
  const reviewRequired = rawItems.map(compactReviewRequiredItem)
  if (reviewRequired.some((item) => item === null)) return null
  return {
    type: event.type,
    payload: { sessionId, reviewRequired },
  }
}

const compactRunRecord = (
  session: Pick<
    QuickIngestSession,
    "sessionId" | "generation" | "attemptToken"
  >,
  tracking: PersistedQuickIngestTracking
): QuickIngestCompactActiveRunSession | null => {
  const normalizedSessionId = String(session.sessionId || "").trim()
  if (!isCompactId(normalizedSessionId)) return null
  const runId = String(tracking.runId || "").trim()
  if (!isCompactId(runId)) return null
  const occurrenceIds = normalizeIds([
    ...(tracking.submissionOccurrenceIds || []),
    ...(tracking.submittedItemIds || []),
    ...(tracking.itemIds || []),
  ])
  const allowedOccurrences = new Set(occurrenceIds)
  const jobIdToItemId = Object.fromEntries(
    Object.entries(tracking.jobIdToItemId || {})
      .map(([jobId, itemId]) => [
        String(jobId || "").trim(),
        String(itemId || "").trim(),
      ])
      .filter(
        ([jobId, itemId]) =>
          isCompactId(jobId) &&
          Number.isSafeInteger(Number(jobId)) &&
          Number(jobId) > 0 &&
          isCompactId(itemId) &&
          allowedOccurrences.has(itemId)
      )
      .slice(0, MAX_COMPACT_OCCURRENCES)
  )
  return {
    version: 1,
    kind: "run",
    sessionId: normalizedSessionId,
    runId,
    generation: session.generation,
    attemptToken: session.attemptToken,
    ...(tracking.submissionState &&
    COMPACT_SUBMISSION_STATES.has(tracking.submissionState)
      ? { submissionState: tracking.submissionState }
      : {}),
    occurrenceIds,
    jobIdToItemId,
    startedAt:
      typeof tracking.startedAt === "number" && Number.isFinite(tracking.startedAt)
        ? tracking.startedAt
        : Date.now(),
  }
}

export const parseQuickIngestCompactRunSession = (
  value: unknown
): QuickIngestCompactRunSession | null => {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null
  const candidate = value as Record<string, unknown>
  if (candidate.version !== 1 || !isCompactId(candidate.sessionId)) return null
  const kind = candidate.kind === undefined ? "run" : candidate.kind
  const persistedAttemptToken = isCompactId(candidate.attemptToken)
    ? candidate.attemptToken
    : isCompactId(candidate.requestFingerprint)
      ? candidate.requestFingerprint
      : null
  const occurrenceIds = Array.isArray(candidate.occurrenceIds)
    ? candidate.occurrenceIds
    : []
  if (
    occurrenceIds.length > MAX_COMPACT_OCCURRENCES ||
    !occurrenceIds.every(isCompactId) ||
    new Set(occurrenceIds).size !== occurrenceIds.length
  ) {
    return null
  }

  if (kind === "review") {
    if (
      !isCompactId(candidate.generation) ||
      !persistedAttemptToken ||
      typeof candidate.expiresAt !== "number" ||
      !Number.isFinite(candidate.expiresAt) ||
      candidate.expiresAt > Date.now() + TERMINAL_TOMBSTONE_TTL_MS + 1_000 ||
      !candidate.event ||
      typeof candidate.event !== "object" ||
      Array.isArray(candidate.event)
    ) {
      return null
    }
    const event = compactReviewEvent(
      candidate.event as QuickIngestRuntimeEvent,
      candidate.sessionId
    )
    if (!event) return null
    const review: QuickIngestCompactReviewSession = {
      version: 1,
      kind: "review",
      sessionId: candidate.sessionId,
      generation: candidate.generation,
      attemptToken: persistedAttemptToken,
      expiresAt: candidate.expiresAt,
      event,
    }
    return serializedByteLength(review) <= MAX_COMPACT_TERMINAL_BYTES
      ? review
      : null
  }

  if (kind === "terminal") {
    if (
      !isCompactId(candidate.runId) ||
      !isCompactId(candidate.generation) ||
      !persistedAttemptToken ||
      typeof candidate.expiresAt !== "number" ||
      !Number.isFinite(candidate.expiresAt) ||
      candidate.expiresAt > Date.now() + TERMINAL_TOMBSTONE_TTL_MS + 1_000 ||
      !candidate.event ||
      typeof candidate.event !== "object" ||
      Array.isArray(candidate.event)
    ) {
      return null
    }
    const event = compactTerminalEvent(
      candidate.event as QuickIngestRuntimeEvent,
      candidate.sessionId,
      candidate.runId
    )
    if (!event) return null
    const terminal: QuickIngestCompactTerminalSession = {
      version: 1,
      kind: "terminal",
      sessionId: candidate.sessionId,
      runId: candidate.runId,
      generation: candidate.generation,
      attemptToken: persistedAttemptToken,
      expiresAt: candidate.expiresAt,
      event,
    }
    return serializedByteLength(terminal) <= MAX_COMPACT_TERMINAL_BYTES
      ? terminal
      : null
  }

  if (
    typeof candidate.startedAt !== "number" ||
    !Number.isFinite(candidate.startedAt)
  ) {
    return null
  }
  if (kind === "start") {
    if (
      !isCompactId(candidate.generation) ||
      !persistedAttemptToken
    ) {
      return null
    }
    return {
      version: 1,
      kind: "start",
      sessionId: candidate.sessionId,
      generation: candidate.generation,
      attemptToken: persistedAttemptToken,
      occurrenceIds: [...occurrenceIds] as string[],
      startedAt: candidate.startedAt,
    }
  }

  if (
    kind !== "run" ||
    !isCompactId(candidate.runId) ||
    (candidate.submissionState !== undefined &&
      !COMPACT_SUBMISSION_STATES.has(
        candidate.submissionState as NonNullable<
          PersistedQuickIngestTracking["submissionState"]
        >
      )) ||
    !candidate.jobIdToItemId ||
    typeof candidate.jobIdToItemId !== "object" ||
    Array.isArray(candidate.jobIdToItemId)
  ) {
    return null
  }

  const allowedOccurrences = new Set(occurrenceIds)
  const mappingEntries = Object.entries(
    candidate.jobIdToItemId as Record<string, unknown>
  )
  if (
    mappingEntries.length > MAX_COMPACT_OCCURRENCES ||
    mappingEntries.some(
      ([jobId, itemId]) =>
        !isCompactId(jobId) ||
        !Number.isSafeInteger(Number(jobId)) ||
        Number(jobId) <= 0 ||
        !isCompactId(itemId) ||
        !allowedOccurrences.has(itemId)
    )
  ) {
    return null
  }

  const generation = isCompactId(candidate.generation)
    ? candidate.generation
    : compactString(
        `legacy-${candidate.startedAt}-${candidate.runId}`,
        MAX_COMPACT_ID_LENGTH
      )
  const attemptToken =
    persistedAttemptToken ||
    compactString(`legacy-${candidate.runId}`, MAX_COMPACT_ID_LENGTH)
  return {
    version: 1,
    kind: "run",
    sessionId: candidate.sessionId,
    runId: candidate.runId,
    generation,
    attemptToken,
    ...(candidate.submissionState === undefined
      ? {}
      : {
          submissionState:
            candidate.submissionState as NonNullable<
              PersistedQuickIngestTracking["submissionState"]
            >,
        }),
    occurrenceIds: [...occurrenceIds] as string[],
    jobIdToItemId: Object.fromEntries(mappingEntries) as Record<string, string>,
    startedAt: candidate.startedAt,
  }
}

const trackingFromRecord = (
  record: QuickIngestCompactActiveRunSession
): PersistedQuickIngestTracking => ({
  mode: "extension-runtime",
  sessionId: record.sessionId,
  runId: String(record?.runId || "").trim(),
  ...(record.submissionState
    ? { submissionState: record.submissionState }
    : {}),
  submissionOccurrenceIds: Array.isArray(record?.occurrenceIds)
    ? record.occurrenceIds
    : [],
  submittedItemIds: Array.isArray(record?.occurrenceIds)
    ? record.occurrenceIds
    : [],
  itemIds: Array.isArray(record?.occurrenceIds) ? record.occurrenceIds : [],
  jobIds: Object.keys(record?.jobIdToItemId || {})
    .map(Number)
    .filter((jobId) => Number.isSafeInteger(jobId) && jobId > 0),
  jobIdToItemId:
    record?.jobIdToItemId && typeof record.jobIdToItemId === "object"
      ? record.jobIdToItemId
      : {},
  startedAt: record.startedAt,
})

const secureSessionSuffix = (): string => {
  try {
    if (typeof globalThis !== "undefined" && typeof globalThis.crypto?.randomUUID === "function") {
      return globalThis.crypto.randomUUID().replace(/-/g, "")
    }
    if (typeof globalThis !== "undefined" && typeof globalThis.crypto?.getRandomValues === "function") {
      const bytes = new Uint8Array(8)
      globalThis.crypto.getRandomValues(bytes)
      return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("")
    }
  } catch {
    // Fall back to timestamp-only suffix below.
  }
  return Date.now().toString(36)
}

const defaultSessionId = () =>
  `qi-${Date.now()}-${secureSessionSuffix().slice(0, 16)}`

const defaultGeneration = () =>
  `generation-${Date.now()}-${secureSessionSuffix().slice(0, 16)}`

const startOccurrenceIds = (payload: Record<string, unknown>): string[] => {
  const pending =
    payload.pendingRunRequest &&
    typeof payload.pendingRunRequest === "object" &&
    !Array.isArray(payload.pendingRunRequest)
      ? (payload.pendingRunRequest as Record<string, unknown>)
      : null
  return normalizeIds(
    Array.isArray(pending?.inputs)
      ? pending.inputs.map((input) =>
          input && typeof input === "object" && !Array.isArray(input)
            ? (input as Record<string, unknown>).occurrenceId
            : ""
        )
      : []
  )
}

export const createQuickIngestSessionRuntime = (deps: RuntimeDeps) => {
  const sessions = new Map<string, QuickIngestSession>()
  const reattachInFlight = new Map<string, Promise<void>>()
  const lastEvents = new Map<string, QuickIngestRuntimeEvent>()
  const replayRecords = new Map<
    string,
    QuickIngestCompactTerminalSession | QuickIngestCompactReviewSession
  >()
  const replayCleanupTimers = new Map<
    string,
    ReturnType<typeof setTimeout>
  >()
  const startInFlight = new Map<
    string,
    { attemptToken: string; promise: Promise<QuickIngestSessionStartAck> }
  >()
  const pollIntervalMs = 1_500
  let pollRunRecord: (
    record: QuickIngestCompactActiveRunSession
  ) => Promise<void>
  let restoreInFlight: Promise<void> | null = null

  const emitSessionEvent = async (
    type: string,
    payload: Record<string, unknown>
  ): Promise<void> => {
    const sessionId = String(payload.sessionId || "").trim()
    if (sessionId) lastEvents.set(sessionId, { type, payload })
    await deps.emit(type, payload)
  }

  const scheduleRunPoll = (
    session: QuickIngestSession,
    record: QuickIngestCompactActiveRunSession
  ): void => {
    if (
      session.cancelled ||
      session.status !== "running" ||
      session.pollTimer ||
      session.runRecord?.runId !== record.runId ||
      session.generation !== record.generation
    ) {
      return
    }
    session.pollTimer = setTimeout(() => {
      session.pollTimer = null
      if (
        sessions.get(record.sessionId) !== session ||
        session.status !== "running" ||
        session.runRecord?.runId !== record.runId ||
        session.generation !== record.generation
      ) {
        return
      }
      void pollRunRecord(record).catch(() => {
        scheduleRunPoll(session, record)
      })
    }, pollIntervalMs)
  }

  const cleanupExpiredReplay = async (
    record: QuickIngestCompactTerminalSession | QuickIngestCompactReviewSession
  ): Promise<void> => {
    const cleanupKey = `${record.sessionId}:${record.generation}`
    const scheduledCleanup = replayCleanupTimers.get(cleanupKey)
    if (scheduledCleanup) {
      clearTimeout(scheduledCleanup)
      replayCleanupTimers.delete(cleanupKey)
    }
    const applied = await deps.saveRunSession?.(
      null,
      record.sessionId,
      record.kind === "terminal" ? record.runId : undefined,
      record.generation
    )
    if (applied === false) return
    if (
      replayRecords.get(record.sessionId)?.generation === record.generation
    ) {
      replayRecords.delete(record.sessionId)
      lastEvents.delete(record.sessionId)
    }
  }

  const scheduleReplayCleanup = (
    record: QuickIngestCompactTerminalSession | QuickIngestCompactReviewSession
  ): void => {
    const key = `${record.sessionId}:${record.generation}`
    if (replayCleanupTimers.has(key)) return
    const timer = setTimeout(() => {
      replayCleanupTimers.delete(key)
      void cleanupExpiredReplay(record).catch(() => {
        scheduleReplayCleanup(record)
      })
    }, pollIntervalMs)
    replayCleanupTimers.set(key, timer)
  }

  const performRunPoll = async (
    record: QuickIngestCompactActiveRunSession
  ): Promise<void> => {
    if (!deps.reattachRun) return
    const existing = sessions.get(record.sessionId)
    if (
      existing?.runRecord?.runId !== undefined &&
      (existing.runRecord.runId !== record.runId ||
        existing.generation !== record.generation)
    ) {
      return
    }
    const session: QuickIngestSession =
      existing || {
        sessionId: record.sessionId,
        generation: record.generation,
        attemptToken: record.attemptToken,
        status: "running",
        cancelled: false,
        cancelRequested: false,
        jobIds: Object.keys(record.jobIdToItemId)
          .map(Number)
          .filter((jobId) => Number.isSafeInteger(jobId) && jobId > 0),
        runRecord: record,
        runRecordPersisted: true,
        runPersistenceRetryAt: 0,
        abortControllers: new Set(),
        pollTimer: null,
      }
    if (!existing) session.runRecord = record
    sessions.set(record.sessionId, session)

    if (
      !session.runRecordPersisted &&
      Date.now() >= session.runPersistenceRetryAt
    ) {
      try {
        const applied = await deps.saveRunSession?.(
          record,
          record.sessionId,
          undefined,
          record.generation
        )
        if (applied === false) {
          throw new Error("Quick ingest run tracking was superseded.")
        }
        session.runRecordPersisted = true
      } catch {
        session.runPersistenceRetryAt = Date.now() + pollIntervalMs
      }
    }

    const snapshot = await deps.reattachRun(trackingFromRecord(record), {
      transportPreference: "poll",
    })
    if (
      session.cancelled ||
      sessions.get(record.sessionId)?.runRecord?.runId !== record.runId ||
      sessions.get(record.sessionId)?.generation !== record.generation
    ) {
      return
    }
    for (const job of snapshot.jobs) {
      const terminalError =
        job.error ||
        (job.status === "cancelled"
          ? snapshot.errorMessage || "Cancelled by user."
          : job.status === "failed"
            ? snapshot.errorMessage || "Quick ingest item failed."
            : undefined)
      const result = {
        id: job.sourceItemId || String(job.jobId || ""),
        status: terminalError ? "error" : job.status,
        type: "item",
        data: job.result,
        error: terminalError,
      }
      await emitSessionEvent("tldw:quick-ingest/progress", {
        sessionId: record.sessionId,
        runId: record.runId,
        occurrenceId: job.sourceItemId,
        jobId: job.jobId,
        status: job.status,
        result,
        error: terminalError,
      })
    }
    if (snapshot.lifecycle === "processing") {
      scheduleRunPoll(session, record)
      return
    }

    if (snapshot.lifecycle === "interrupted") {
      await emitSessionEvent("tldw:quick-ingest/interrupted", {
        sessionId: record.sessionId,
        runId: record.runId,
        recoverable: true,
        error:
          snapshot.errorMessage || "Quick ingest run recovery was interrupted.",
      })
      scheduleRunPoll(session, record)
      return
    }

    if (session.pollTimer) {
      clearTimeout(session.pollTimer)
      session.pollTimer = null
    }

    const results = snapshot.jobs.map((job, index) => {
      const terminalError =
        job.error ||
        (job.status === "cancelled"
          ? snapshot.errorMessage || "Cancelled by user."
          : job.status === "failed"
            ? snapshot.errorMessage || "Quick ingest item failed."
            : undefined)
      return {
        id:
          job.sourceItemId ||
          record.occurrenceIds[index] ||
          String(job.jobId || ""),
        status: terminalError ? "error" : "ok",
        data: job.result,
        error: terminalError,
      }
    })
    const eventType =
      snapshot.lifecycle === "cancelled"
        ? "tldw:quick-ingest/cancelled"
        : snapshot.lifecycle === "completed"
          ? "tldw:quick-ingest/completed"
          : "tldw:quick-ingest/failed"
    const rawTerminalEvent: QuickIngestRuntimeEvent = {
      type: eventType,
      payload: {
        sessionId: record.sessionId,
        runId: record.runId,
        ...(eventType === "tldw:quick-ingest/completed"
          ? { results }
          : eventType === "tldw:quick-ingest/cancelled"
            ? {
                reason: snapshot.errorMessage || "Cancelled by user.",
                results,
              }
            : {
                error:
                  snapshot.errorMessage ||
                  "Quick ingest run could not be restored.",
                results,
              }),
      },
    }
    let terminalEvent =
      compactTerminalEvent(
        rawTerminalEvent,
        record.sessionId,
        record.runId
      ) ||
      compactEssentialTerminalEvent(
        rawTerminalEvent,
        record.sessionId,
        record.runId
      )
    if (!terminalEvent) {
      await emitSessionEvent("tldw:quick-ingest/interrupted", {
        sessionId: record.sessionId,
        runId: record.runId,
        recoverable: true,
        error: "Quick ingest terminal results could not be persisted safely.",
      })
      scheduleRunPoll(session, record)
      return
    }
    let terminalRecord: QuickIngestCompactTerminalSession = {
      version: 1,
      kind: "terminal",
      sessionId: record.sessionId,
      runId: record.runId,
      generation: record.generation,
      attemptToken: record.attemptToken,
      expiresAt: Date.now() + TERMINAL_TOMBSTONE_TTL_MS,
      event: terminalEvent,
    }
    if (serializedByteLength(terminalRecord) > MAX_COMPACT_TERMINAL_BYTES) {
      const essentialEvent = compactEssentialTerminalEvent(
        rawTerminalEvent,
        record.sessionId,
        record.runId
      )
      if (!essentialEvent) {
        await emitSessionEvent("tldw:quick-ingest/interrupted", {
          sessionId: record.sessionId,
          runId: record.runId,
          recoverable: true,
          error: "Quick ingest terminal results could not be compacted safely.",
        })
        scheduleRunPoll(session, record)
        return
      }
      terminalEvent = essentialEvent
      terminalRecord = { ...terminalRecord, event: terminalEvent }
      if (serializedByteLength(terminalRecord) > MAX_COMPACT_TERMINAL_BYTES) {
        await emitSessionEvent("tldw:quick-ingest/interrupted", {
          sessionId: record.sessionId,
          runId: record.runId,
          recoverable: true,
          error: "Quick ingest terminal results exceeded the recovery storage limit.",
        })
        scheduleRunPoll(session, record)
        return
      }
    }
    try {
      const applied = await deps.saveRunSession?.(
        terminalRecord,
        record.sessionId,
        record.runId,
        record.generation
      )
      if (applied === false) {
        session.status = "failed"
        if (
          sessions.get(record.sessionId) === session &&
          session.generation === record.generation
        ) {
          sessions.delete(record.sessionId)
        }
        return
      }
    } catch {
      await emitSessionEvent("tldw:quick-ingest/interrupted", {
        sessionId: record.sessionId,
        runId: record.runId,
        recoverable: true,
        error: "Quick ingest terminal replay could not be persisted. Retrying.",
      })
      scheduleRunPoll(session, record)
      return
    }
    replayRecords.set(record.sessionId, terminalRecord)
    session.status =
      eventType === "tldw:quick-ingest/completed"
        ? "completed"
        : eventType === "tldw:quick-ingest/cancelled"
          ? "cancelled"
          : "failed"
    await emitSessionEvent(terminalEvent.type, terminalEvent.payload)
    if (
      sessions.get(record.sessionId)?.runRecord?.runId === record.runId &&
      sessions.get(record.sessionId)?.generation === record.generation
    ) {
      sessions.delete(record.sessionId)
    }
  }

  pollRunRecord = (
    record: QuickIngestCompactActiveRunSession
  ): Promise<void> => {
    const key = `${record.sessionId}:${record.runId}:${record.generation}`
    const active = reattachInFlight.get(key)
    if (active) return active
    const pending = performRunPoll(record).finally(() => {
      reattachInFlight.delete(key)
    })
    reattachInFlight.set(key, pending)
    return pending
  }

  class RunTrackingPersistenceError extends Error {}

  const start = (
    payload: Record<string, unknown>,
    options: { sessionId?: string; attemptToken?: string } = {}
  ): Promise<QuickIngestSessionStartAck> => {
    const requestedSessionId = compactString(
      options.sessionId,
      MAX_COMPACT_ID_LENGTH
    )
    if (options.sessionId !== undefined && !isCompactId(requestedSessionId)) {
      return Promise.resolve({
        ok: false,
        sessionId: requestedSessionId,
        error: "Quick ingest session identity is invalid.",
      })
    }
    const sessionId =
      requestedSessionId || deps.createSessionId?.() || defaultSessionId()
    const requestedAttemptToken = compactString(
      options.attemptToken,
      MAX_COMPACT_ID_LENGTH
    )
    if (
      options.attemptToken !== undefined &&
      !isCompactId(requestedAttemptToken)
    ) {
      return Promise.resolve({
        ok: false,
        sessionId,
        error: "Quick ingest attempt identity is invalid.",
      })
    }
    const attemptToken =
      requestedAttemptToken ||
      `qia-${Date.now()}-${secureSessionSuffix().slice(0, 16)}`
    const pendingStart = startInFlight.get(sessionId)
    if (pendingStart) {
      return pendingStart.attemptToken === attemptToken
        ? pendingStart.promise
        : Promise.resolve({
            ok: false,
            sessionId,
            error: "Quick ingest session identity was reused by another attempt.",
          })
    }
    const existing = sessions.get(sessionId)
    const replayRecord = replayRecords.get(sessionId)
    const existingAttemptToken =
      existing?.attemptToken || replayRecord?.attemptToken
    if (existingAttemptToken) {
      return Promise.resolve(
        existingAttemptToken === attemptToken
          ? { ok: true, sessionId }
          : {
              ok: false,
              sessionId,
              error: "Quick ingest session identity was reused by another attempt.",
            }
      )
    }
    const generation = defaultGeneration()
    const marker: QuickIngestCompactStartSession = {
      version: 1,
      kind: "start",
      sessionId,
      generation,
      attemptToken,
      occurrenceIds: startOccurrenceIds(payload),
      startedAt: Date.now(),
    }
    const session: QuickIngestSession = {
      sessionId,
      generation,
      attemptToken,
      status: "running",
      cancelled: false,
      cancelRequested: false,
      jobIds: [],
      runRecord: null,
      runRecordPersisted: false,
      runPersistenceRetryAt: 0,
      abortControllers: new Set(),
      pollTimer: null,
    }
    sessions.set(sessionId, session)

    const execute = async (): Promise<void> => {
      try {
        const result = await deps.run(payload, {
          sessionId,
          isCancelled: () => session.cancelled || session.cancelRequested,
          registerAbortController: (controller: AbortController) => {
            session.abortControllers.add(controller)
          },
          setJobIds: (jobIds: number[]) => {
            session.jobIds = jobIds
          },
          setRunTracking: async (tracking: PersistedQuickIngestTracking) => {
            if (session.cancelled || session.cancelRequested) return
            if (!String(tracking.runId || "").trim()) return
            const record = compactRunRecord(session, tracking)
            if (!record) {
              throw new Error("Quick ingest run tracking is invalid.")
            }
            session.runRecord = record
            session.runRecordPersisted = false
            try {
              const applied = await deps.saveRunSession?.(
                record,
                sessionId,
                undefined,
                session.generation
              )
              if (applied === false) {
                throw new Error("Quick ingest run tracking was superseded.")
              }
              session.runRecordPersisted = true
            } catch (error) {
              let cancellationError = ""
              if (deps.cancelRun) {
                try {
                  const response = await deps.cancelRun(
                    trackingFromRecord(record),
                    "tracking_persistence_failed"
                  )
                  if (!response.ok) {
                    cancellationError = response.error || "Cancellation unconfirmed."
                  }
                } catch (cancelError) {
                  cancellationError =
                    cancelError instanceof Error
                      ? cancelError.message
                      : "Cancellation unconfirmed."
                }
              } else {
                cancellationError = "Cancellation is unavailable."
              }
              try {
                const applied = await deps.saveRunSession?.(
                  record,
                  sessionId,
                  undefined,
                  session.generation
                )
                session.runRecordPersisted = applied !== false
              } catch {
                // Keep the known run in memory and reconcile it below.
              }
              if (!session.runRecordPersisted) {
                session.runPersistenceRetryAt = Date.now() + pollIntervalMs
              }
              await emitSessionEvent("tldw:quick-ingest/interrupted", {
                sessionId,
                runId: record.runId,
                recoverable: true,
                error: `Quick ingest run tracking could not be persisted; recovery is active${
                  cancellationError ? ` (${cancellationError})` : ""
                }.`,
              })
              try {
                await pollRunRecord(record)
              } catch {
                scheduleRunPoll(session, record)
              }
              throw new RunTrackingPersistenceError(
                error instanceof Error ? error.message : "Storage unavailable."
              )
            }
          },
          emitProgress: async (progressPayload: Record<string, unknown>) => {
            await emitSessionEvent("tldw:quick-ingest/progress", {
              sessionId,
              ...progressPayload
            })
          }
        })
        if (session.cancelled) {
          return
        }
        if (Array.isArray(result.reviewRequired) && result.reviewRequired.length > 0) {
          const reviewEvent = compactReviewEvent(
            {
              type: "tldw:quick-ingest/review-required",
              payload: {
                sessionId,
                reviewRequired: result.reviewRequired,
              },
            },
            sessionId
          )
          const reviewRecord: QuickIngestCompactReviewSession | null = reviewEvent
            ? {
                version: 1,
                kind: "review",
                sessionId,
                generation: session.generation,
                attemptToken: session.attemptToken,
                expiresAt: Date.now() + TERMINAL_TOMBSTONE_TTL_MS,
                event: reviewEvent,
              }
            : null
          if (
            !reviewRecord ||
            serializedByteLength(reviewRecord) > MAX_COMPACT_TERMINAL_BYTES
          ) {
            await emitSessionEvent("tldw:quick-ingest/interrupted", {
              sessionId,
              recoverable: true,
              error: "Quick ingest review recovery could not be persisted safely.",
            })
            return
          }
          try {
            const applied = await deps.saveRunSession?.(
              reviewRecord,
              sessionId,
              undefined,
              session.generation
            )
            if (applied === false) {
              throw new Error("Quick ingest review persistence was superseded.")
            }
          } catch {
            await emitSessionEvent("tldw:quick-ingest/interrupted", {
              sessionId,
              recoverable: true,
              error: "Quick ingest review replay could not be persisted.",
            })
            return
          }
          replayRecords.set(sessionId, reviewRecord)
          session.status = "failed"
          await emitSessionEvent(reviewEvent.type, reviewEvent.payload)
          return
        }
        if (
          session.runRecord &&
          (session.cancelRequested || !result.results?.length)
        ) {
          await pollRunRecord(session.runRecord)
          return
        }
        session.status = "completed"
        await deps.saveRunSession?.(
          null,
          sessionId,
          undefined,
          session.generation
        )
        await emitSessionEvent("tldw:quick-ingest/completed", {
          sessionId,
          results: Array.isArray(result?.results) ? result.results : [],
          summary: result?.summary || {}
        })
      } catch (error) {
        if (session.cancelled) {
          return
        }
        if (error instanceof RunTrackingPersistenceError) {
          return
        }
        if (session.runRecord?.submissionState === "cleanup_required") {
          await emitSessionEvent("tldw:quick-ingest/interrupted", {
            sessionId,
            runId: session.runRecord.runId,
            recoverable: true,
            error:
              error instanceof Error
                ? error.message
                : "Quick ingest cleanup requires retry.",
          })
          await pollRunRecord(session.runRecord)
          return
        }
        if (session.cancelRequested && session.runRecord) {
          await pollRunRecord(session.runRecord)
          return
        }
        session.status = "failed"
        await deps.saveRunSession?.(
          null,
          sessionId,
          undefined,
          session.generation
        )
        await emitSessionEvent("tldw:quick-ingest/failed", {
          sessionId,
          error: error instanceof Error ? error.message : String(error || "Quick ingest failed.")
        })
      } finally {
        if (
          (!session.runRecord || session.status !== "running")
        ) {
          sessions.delete(sessionId)
        }
      }
    }

    const startPromise = (async (): Promise<QuickIngestSessionStartAck> => {
      try {
        const applied = await deps.saveRunSession?.(marker)
        if (applied === false) {
          throw new Error("Quick ingest start marker was not stored.")
        }
      } catch (error) {
        sessions.delete(sessionId)
        return {
          ok: false,
          sessionId,
          error:
            error instanceof Error
              ? `Quick ingest could not persist its start marker: ${error.message}`
              : "Quick ingest could not persist its start marker.",
        }
      }
      void execute()
      return { ok: true, sessionId }
    })().finally(() => {
      if (startInFlight.get(sessionId)?.promise === startPromise) {
        startInFlight.delete(sessionId)
      }
    })
    startInFlight.set(sessionId, { attemptToken, promise: startPromise })
    return startPromise
  }

  const cancel = async (
    sessionId: string,
    reason: string = "user_cancelled"
  ): Promise<QuickIngestSessionCancelResponse> => {
    const normalizedSessionId = String(sessionId || "").trim()
    if (!normalizedSessionId) {
      return { ok: false, error: "Missing sessionId." }
    }
    if (restoreInFlight) await restoreInFlight
    if (!sessions.has(normalizedSessionId) && deps.loadRunSessions) {
      await restore()
    }
    const session = sessions.get(normalizedSessionId)
    if (!session) {
      return { ok: false, error: "Session not found." }
    }
    if (session.cancelled || session.cancelRequested) {
      return { ok: true }
    }

    if (session.runRecord) {
      session.cancelRequested = true
      for (const controller of Array.from(session.abortControllers)) {
        try {
          controller.abort()
        } catch {
          // best effort
        }
      }
      if (!deps.cancelRun) {
        session.cancelRequested = false
        scheduleRunPoll(session, session.runRecord)
        return { ok: false, error: "Run cancellation is unavailable." }
      }
      try {
        const response = await deps.cancelRun(
          trackingFromRecord(session.runRecord),
          reason
        )
        if (!response.ok) {
          session.cancelRequested = false
          scheduleRunPoll(session, session.runRecord)
          return response
        }
      } catch (error) {
        session.cancelRequested = false
        scheduleRunPoll(session, session.runRecord)
        return {
          ok: false,
          error:
            error instanceof Error
              ? error.message
              : "The ingest run could not be cancelled.",
        }
      }
      if (session.pollTimer) {
        clearTimeout(session.pollTimer)
        session.pollTimer = null
      }
      scheduleRunPoll(session, session.runRecord)
      return { ok: true }
    }

    session.cancelled = true
    session.status = "cancelled"
    for (const controller of Array.from(session.abortControllers)) {
      try {
        controller.abort()
      } catch {
        // best effort
      }
    }
    try {
      await deps.saveRunSession?.(
        null,
        normalizedSessionId,
        undefined,
        session.generation
      )
    } catch {
      // Local cancellation is already authoritative in this worker.
    }
    await emitSessionEvent("tldw:quick-ingest/cancelled", {
      sessionId: normalizedSessionId,
      reason,
      jobIds: session.jobIds,
    })
    sessions.delete(normalizedSessionId)
    return { ok: true }
  }

  const performRestore = async (): Promise<void> => {
    if (!deps.loadRunSessions) return
    const storedRecords = await deps.loadRunSessions()
    const recordsBySession = new Map<string, QuickIngestCompactRunSession>()
    for (const stored of Array.isArray(storedRecords) ? storedRecords : []) {
      const candidate = parseQuickIngestCompactRunSession(stored)
      if (!candidate) continue
      const current = recordsBySession.get(candidate.sessionId)
      if (!current) {
        recordsBySession.set(candidate.sessionId, candidate)
        continue
      }
      const currentReplay =
        current.kind === "terminal" || current.kind === "review"
      const candidateReplay =
        candidate.kind === "terminal" || candidate.kind === "review"
      const currentActive = !currentReplay
      const candidateActive = !candidateReplay
      const currentTimestamp =
        current.kind === "terminal" || current.kind === "review"
          ? current.expiresAt
          : current.startedAt
      const candidateTimestamp =
        candidate.kind === "terminal" || candidate.kind === "review"
          ? candidate.expiresAt
          : candidate.startedAt
      if (
        (candidateActive && !currentActive) ||
        (candidateActive === currentActive &&
          candidateTimestamp >= currentTimestamp)
      ) {
        recordsBySession.set(candidate.sessionId, candidate)
      }
    }
    for (const record of recordsBySession.values()) {
      try {
        if (record.kind === "terminal" || record.kind === "review") {
          if (record.expiresAt <= Date.now()) {
            await cleanupExpiredReplay(record)
            continue
          }
          replayRecords.set(record.sessionId, record)
          lastEvents.set(record.sessionId, record.event)
          continue
        }
        if (record.kind === "start") {
          const existing = sessions.get(record.sessionId)
          if (existing?.generation === record.generation) continue
          const session: QuickIngestSession = {
            sessionId: record.sessionId,
            generation: record.generation,
            attemptToken: record.attemptToken,
            status: "running",
            cancelled: false,
            cancelRequested: false,
            jobIds: [],
            runRecord: null,
            runRecordPersisted: true,
            runPersistenceRetryAt: 0,
            abortControllers: new Set(),
            pollTimer: null,
          }
          sessions.set(record.sessionId, session)
          await emitSessionEvent("tldw:quick-ingest/interrupted", {
            sessionId: record.sessionId,
            recoverable: true,
            error:
              "Quick ingest was interrupted before the server run was created. Retry from this retained session.",
          })
          continue
        }
        if (deps.reattachRun) await pollRunRecord(record)
      } catch {
        if (
          (record.kind === "terminal" || record.kind === "review") &&
          record.expiresAt <= Date.now()
        ) {
          scheduleReplayCleanup(record)
          continue
        }
        if (record.kind !== "run") continue
        const session = sessions.get(record.sessionId)
        if (session) scheduleRunPoll(session, record)
      }
    }
  }

  const restore = (): Promise<void> => {
    if (restoreInFlight) return restoreInFlight
    const pending = performRestore().finally(() => {
      if (restoreInFlight === pending) restoreInFlight = null
    })
    restoreInFlight = pending
    return pending
  }

  const replay = async (sessionId: string) => {
    const normalizedSessionId = String(sessionId || "").trim()
    if (!normalizedSessionId) {
      return { ok: false, error: "Missing sessionId." }
    }
    if (deps.loadRunSessions) await restore()
    const session = sessions.get(normalizedSessionId)
    if (session?.runRecord) await pollRunRecord(session.runRecord)
    const event = lastEvents.get(normalizedSessionId) || null
    if (!session && !event) {
      return { ok: false, error: "Session not found." }
    }
    const retainedReplay = replayRecords.get(normalizedSessionId)
    return {
      ok: true,
      active: sessions.has(normalizedSessionId),
      event,
      ...(retainedReplay?.kind === "terminal"
        ? {
            replayAck: {
              runId: retainedReplay.runId,
              generation: retainedReplay.generation,
            },
          }
        : {}),
    }
  }

  const acknowledgeReplay = async (
    sessionId: string,
    runId: string,
    generation: string
  ): Promise<QuickIngestSessionCancelResponse> => {
    const normalizedSessionId = compactString(sessionId, MAX_COMPACT_ID_LENGTH)
    const normalizedRunId = compactString(runId, MAX_COMPACT_ID_LENGTH)
    const normalizedGeneration = compactString(
      generation,
      MAX_COMPACT_ID_LENGTH
    )
    if (
      !isCompactId(normalizedSessionId) ||
      !isCompactId(normalizedRunId) ||
      !isCompactId(normalizedGeneration)
    ) {
      return { ok: false, error: "Replay acknowledgement is invalid." }
    }
    if (deps.loadRunSessions) await restore()
    const record = replayRecords.get(normalizedSessionId)
    if (
      !record ||
      record.kind !== "terminal" ||
      record.runId !== normalizedRunId ||
      record.generation !== normalizedGeneration
    ) {
      return { ok: false, error: "Terminal replay was not found." }
    }
    const applied = await deps.saveRunSession?.(
      null,
      normalizedSessionId,
      normalizedRunId,
      normalizedGeneration
    )
    if (applied === false) {
      return {
        ok: false,
        error: "Terminal replay acknowledgement was superseded; replay was retained.",
      }
    }
    if (
      replayRecords.get(normalizedSessionId)?.generation ===
      normalizedGeneration
    ) {
      replayRecords.delete(normalizedSessionId)
      lastEvents.delete(normalizedSessionId)
    }
    return { ok: true }
  }

  return {
    start,
    cancel,
    restore,
    replay,
    acknowledgeReplay,
    hasSession: (sessionId: string) => sessions.has(String(sessionId || "").trim())
  }
}
