import { createSafeStorage } from "@/utils/safe-storage"

const storage = createSafeStorage({ area: "local" })

export const ONBOARDING_INGESTION_TELEMETRY_STORAGE_KEY =
  "tldw:onboarding:ingestion:telemetry"

const MAX_RECENT_EVENTS = 200

type EventDetails = Record<string, string | number | boolean | null>

type OnboardingIngestionRecentEvent = {
  type: OnboardingIngestionTelemetryEvent["type"]
  at: number
  details: EventDetails
}

type OnboardingIngestionSessionState = {
  started_at: number | null
  first_ingest_at: number | null
  first_chat_after_ingest_at: number | null
  first_media_id: string | null
  source_label: string | null
}

type OnboardingIngestionAggregates = {
  samples_time_to_first_ingest: number
  total_time_to_first_ingest_ms: number
  avg_time_to_first_ingest_ms: number | null
  first_chat_after_ingest_conversions: number
}

export type OnboardingIngestionTelemetryState = {
  version: 1
  counters: Record<string, number>
  last_event_at: number | null
  current_session: OnboardingIngestionSessionState
  aggregates: OnboardingIngestionAggregates
  recent_events: OnboardingIngestionRecentEvent[]
}

export type OnboardingIngestionTelemetryEvent =
  | {
      type: "onboarding_success_reached"
      source?: string
    }
  | {
      type: "onboarding_first_ingest_success"
      success_count?: number
      attempted_at?: number | null
      first_media_id?: string | number | null
      source_label?: string | null
    }
  | {
      type: "onboarding_chat_submit_success"
      route?: string
    }
  | {
      type: "onboarding_session_reset"
      reason?: string
    }

const DEFAULT_SESSION: OnboardingIngestionSessionState = {
  started_at: null,
  first_ingest_at: null,
  first_chat_after_ingest_at: null,
  first_media_id: null,
  source_label: null
}

const DEFAULT_AGGREGATES: OnboardingIngestionAggregates = {
  samples_time_to_first_ingest: 0,
  total_time_to_first_ingest_ms: 0,
  avg_time_to_first_ingest_ms: null,
  first_chat_after_ingest_conversions: 0
}

const DEFAULT_STATE: OnboardingIngestionTelemetryState = {
  version: 1,
  counters: {},
  last_event_at: null,
  current_session: DEFAULT_SESSION,
  aggregates: DEFAULT_AGGREGATES,
  recent_events: []
}

const normalizeOptionalString = (value: unknown): string | null => {
  if (typeof value !== "string") return null
  const trimmed = value.trim()
  return trimmed.length > 0 ? trimmed : null
}

const toEventDetails = (
  event: OnboardingIngestionTelemetryEvent
): EventDetails => {
  const details: EventDetails = {}
  for (const [key, value] of Object.entries(event)) {
    if (key === "type") continue
    if (
      value == null ||
      typeof value === "string" ||
      typeof value === "number" ||
      typeof value === "boolean"
    ) {
      details[key] = value
    }
  }
  return details
}

const incrementCounter = (counters: Record<string, number>, key: string) => {
  counters[key] = (counters[key] || 0) + 1
}

const readTelemetryState = async (): Promise<OnboardingIngestionTelemetryState> => {
  const raw = await storage.get<OnboardingIngestionTelemetryState | undefined>(
    ONBOARDING_INGESTION_TELEMETRY_STORAGE_KEY
  )
  const state = raw && typeof raw === "object" ? raw : DEFAULT_STATE

  return {
    ...DEFAULT_STATE,
    ...state,
    counters: { ...DEFAULT_STATE.counters, ...(state.counters || {}) },
    current_session: {
      ...DEFAULT_SESSION,
      ...(state.current_session || {})
    },
    aggregates: {
      ...DEFAULT_AGGREGATES,
      ...(state.aggregates || {})
    },
    recent_events: Array.isArray(state.recent_events)
      ? state.recent_events.slice(-MAX_RECENT_EVENTS)
      : []
  }
}

const writeTelemetryState = async (state: OnboardingIngestionTelemetryState) => {
  await storage.set(ONBOARDING_INGESTION_TELEMETRY_STORAGE_KEY, state)
}

const resetSession = (state: OnboardingIngestionTelemetryState, now: number) => {
  state.current_session = {
    ...DEFAULT_SESSION,
    started_at: now
  }
}

const appendRecentEvent = (
  state: OnboardingIngestionTelemetryState,
  event: OnboardingIngestionTelemetryEvent,
  now: number
) => {
  state.recent_events.push({
    type: event.type,
    at: now,
    details: toEventDetails(event)
  })

  if (state.recent_events.length > MAX_RECENT_EVENTS) {
    state.recent_events = state.recent_events.slice(-MAX_RECENT_EVENTS)
  }
}

const setAverageTimeToFirstIngest = (
  aggregates: OnboardingIngestionAggregates
) => {
  if (aggregates.samples_time_to_first_ingest <= 0) {
    aggregates.avg_time_to_first_ingest_ms = null
    return
  }
  aggregates.avg_time_to_first_ingest_ms = Math.round(
    aggregates.total_time_to_first_ingest_ms /
      aggregates.samples_time_to_first_ingest
  )
}

export const trackOnboardingIngestionTelemetry = async (
  event: OnboardingIngestionTelemetryEvent,
  isCurrent: () => boolean = () => true
) => {
  try {
    if (!isCurrent()) return
    const state = await readTelemetryState()
    if (!isCurrent()) return
    const now = Date.now()
    state.last_event_at = now

    switch (event.type) {
      case "onboarding_success_reached":
        incrementCounter(state.counters, event.type)
        resetSession(state, now)
        break
      case "onboarding_first_ingest_success": {
        if (state.current_session.started_at === null) break

        const successCount =
          typeof event.success_count === "number" && event.success_count > 0
            ? event.success_count
            : 0
        if (successCount <= 0) break

        const startedAt = state.current_session.started_at ?? now
        const attemptedAt =
          typeof event.attempted_at === "number" && Number.isFinite(event.attempted_at)
            ? event.attempted_at
            : null
        if (attemptedAt !== null && attemptedAt < startedAt) break

        if (state.current_session.first_ingest_at !== null) break
        incrementCounter(state.counters, event.type)

        state.current_session.first_ingest_at = now
        state.current_session.first_media_id =
          event.first_media_id === null || typeof event.first_media_id === "undefined"
            ? null
            : String(event.first_media_id)
        state.current_session.source_label = normalizeOptionalString(
          event.source_label
        )

        const elapsedMs = Math.max(0, now - startedAt)
        state.aggregates.samples_time_to_first_ingest += 1
        state.aggregates.total_time_to_first_ingest_ms += elapsedMs
        setAverageTimeToFirstIngest(state.aggregates)
        break
      }
      case "onboarding_chat_submit_success":
        if (state.current_session.started_at === null) break

        if (state.current_session.first_ingest_at === null) {
          incrementCounter(state.counters, "onboarding_chat_submit_before_ingest")
          break
        }

        if (state.current_session.first_chat_after_ingest_at === null) {
          incrementCounter(state.counters, "onboarding_first_chat_after_ingest")
          state.current_session.first_chat_after_ingest_at = now
          state.aggregates.first_chat_after_ingest_conversions += 1
        }
        break
      case "onboarding_session_reset":
        incrementCounter(state.counters, event.type)
        state.current_session = { ...DEFAULT_SESSION }
        break
      default:
        break
    }

    appendRecentEvent(state, event, now)
    await writeTelemetryState(state)
  } catch (error) {
    console.warn(
      "[onboarding-ingestion-telemetry] Failed to record telemetry event",
      error
    )
  }
}

export const trackOnboardingSuccessReached = async (source = "setup") =>
  trackOnboardingIngestionTelemetry({
    type: "onboarding_success_reached",
    source
  })

export const trackOnboardingFirstIngestSuccess = async (payload: {
  successCount: number
  attemptedAt?: number | null
  firstMediaId?: string | number | null
  primarySourceLabel?: string | null
}) =>
  trackOnboardingIngestionTelemetry({
    type: "onboarding_first_ingest_success",
    success_count: payload.successCount,
    attempted_at: payload.attemptedAt,
    first_media_id: payload.firstMediaId,
    source_label: payload.primarySourceLabel
  })

export const trackOnboardingChatSubmitSuccess = async (route = "/chat", isCurrent?: () => boolean) =>
  trackOnboardingIngestionTelemetry({
    type: "onboarding_chat_submit_success",
    route
  }, isCurrent)

export const getOnboardingIngestionTelemetryState =
  async (): Promise<OnboardingIngestionTelemetryState> => readTelemetryState()

export const resetOnboardingIngestionTelemetryState = async () => {
  await storage.set(ONBOARDING_INGESTION_TELEMETRY_STORAGE_KEY, DEFAULT_STATE)
}
