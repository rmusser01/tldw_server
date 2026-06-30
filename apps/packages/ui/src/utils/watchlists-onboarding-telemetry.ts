import { recordWatchlistsOnboardingTelemetry } from "@/services/watchlists"
import { createSafeStorage } from "@/utils/safe-storage"

const storage = createSafeStorage({ area: "local" })

export const WATCHLISTS_ONBOARDING_TELEMETRY_STORAGE_KEY =
  "tldw:watchlists:onboarding:telemetry"
const MAX_RECENT_EVENTS = 200
const MAX_PENDING_COMPLETION_EVENTS = 100
const MAX_DURATION_SAMPLES = 100

export type WatchlistsQuickSetupDestination = "runs" | "outputs" | "jobs"
export type WatchlistsQuickSetupPreview = "candidate" | "template"
export type WatchlistsOnboardingSuccessSource =
  | "overview"
  | "outputs"
  | "run_notifications"
  | "manual"
  | "unknown"

type QuickSetupDestinationCounters = Record<WatchlistsQuickSetupDestination, number>
type QuickSetupPreviewCounters = Record<WatchlistsQuickSetupPreview, number>

const DEFAULT_DESTINATION_COUNTERS: QuickSetupDestinationCounters = {
  runs: 0,
  outputs: 0,
  jobs: 0
}

const DEFAULT_PREVIEW_COUNTERS: QuickSetupPreviewCounters = {
  candidate: 0,
  template: 0
}

const createOnboardingSessionId = (): string => {
  try {
    if (typeof globalThis !== "undefined" && typeof globalThis.crypto?.randomUUID === "function") {
      return globalThis.crypto.randomUUID()
    }
    if (typeof globalThis !== "undefined" && typeof globalThis.crypto?.getRandomValues === "function") {
      const bytes = new Uint8Array(8)
      globalThis.crypto.getRandomValues(bytes)
      const randomSuffix = Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("")
      return `watchlists-${Date.now().toString(36)}-${randomSuffix}`
    }
  } catch {
    // Fallback below.
  }
  return `watchlists-${Date.now().toString(36)}`
}

export type WatchlistsQuickSetupStep = "feed" | "monitor" | "review"
export type WatchlistsGuidedTourStep = 1 | 2 | 3 | 4 | 5
export type WatchlistsPipelineSetupStep = "scope" | "briefing" | "review"
export type WatchlistsPipelineSetupMode = "create" | "test"
export type WatchlistsPipelinePreviewStatus =
  | "success"
  | "empty"
  | "no_run_context"
  | "template_empty"
  | "error"
export type WatchlistsPipelineFailureStage =
  | "validation"
  | "source_create"
  | "job_create"
  | "run_trigger"
  | "output_create"
  | "rollback"

type EventDetails = Record<string, string | number | boolean | null>

export type WatchlistsOnboardingTelemetryEvent =
  | { type: "quick_setup_opened" }
  | { type: "quick_setup_step_completed"; step: WatchlistsQuickSetupStep }
  | { type: "quick_setup_cancelled"; step: WatchlistsQuickSetupStep }
  | {
      type: "quick_setup_completed"
      goal: "briefing" | "triage"
      runNow: boolean
      destination: WatchlistsQuickSetupDestination
    }
  | { type: "quick_setup_failed"; step: WatchlistsQuickSetupStep }
  | {
      type: "quick_setup_preview_loaded"
      preview: WatchlistsQuickSetupPreview
      total?: number
      ingestable?: number
      filtered?: number
      goal?: "briefing" | "triage"
      audioEnabled?: boolean
    }
  | {
      type: "quick_setup_preview_failed"
      preview: WatchlistsQuickSetupPreview
      reason?: string
    }
  | {
      type: "quick_setup_test_run_triggered"
      runId: number
    }
  | { type: "quick_setup_test_run_failed" }
  | {
      type: "quick_setup_first_run_succeeded"
      source?: WatchlistsOnboardingSuccessSource
      runId?: number
    }
  | {
      type: "quick_setup_first_output_succeeded"
      source?: WatchlistsOnboardingSuccessSource
      outputId?: number
      format?: string
    }
  | { type: "guided_tour_started" }
  | { type: "guided_tour_step_viewed"; step: WatchlistsGuidedTourStep }
  | { type: "guided_tour_completed" }
  | { type: "guided_tour_dismissed"; step: WatchlistsGuidedTourStep }
  | { type: "guided_tour_resumed"; step: WatchlistsGuidedTourStep }
  | { type: "pipeline_setup_opened" }
  | { type: "pipeline_setup_step_completed"; step: WatchlistsPipelineSetupStep }
  | {
      type: "pipeline_setup_preview_generated"
      status: WatchlistsPipelinePreviewStatus
      warning_count?: number
      run_id?: number | null
    }
  | {
      type: "pipeline_setup_submitted"
      mode: WatchlistsPipelineSetupMode
      runNow: boolean
    }
  | {
      type: "pipeline_setup_completed"
      mode: WatchlistsPipelineSetupMode
      runNow: boolean
      destination: "jobs" | "outputs"
    }
  | {
      type: "pipeline_setup_failed"
      stage: WatchlistsPipelineFailureStage
      mode: WatchlistsPipelineSetupMode
      runNow: boolean
    }
  | { type: "first_run_succeeded"; runId: number }
  | { type: "first_output_succeeded"; outputId: number; format: string | null }

export type WatchlistsOnboardingRecentEvent = {
  type: WatchlistsOnboardingTelemetryEvent["type"]
  at: number
  details: EventDetails
}

type StepCounters = Record<WatchlistsQuickSetupStep, number>
type GuidedStepCounters = Record<`${WatchlistsGuidedTourStep}`, number>
type PipelineStepCounters = Record<WatchlistsPipelineSetupStep, number>
type PipelineModeCounters = Record<WatchlistsPipelineSetupMode, number>
type PipelinePreviewCounters = Record<WatchlistsPipelinePreviewStatus, number>
type PipelineFailureCounters = Record<WatchlistsPipelineFailureStage, number>

export type WatchlistsOnboardingTelemetryEventType =
  WatchlistsOnboardingTelemetryEvent["type"]

export type WatchlistsOnboardingTelemetryQuery = {
  eventTypes?: WatchlistsOnboardingTelemetryEventType[]
  sinceMs?: number
}

export type WatchlistsUc2PipelineDashboardSnapshot = {
  funnel: {
    opened: number
    stepScopeCompleted: number
    stepBriefingCompleted: number
    stepReviewCompleted: number
    submitted: number
    completed: number
    completedWithRunNow: number
    completedWithoutRunNow: number
  }
  firstSuccess: {
    firstRunSucceeded: number
    firstOutputSucceeded: number
  }
  rates: {
    completionPerOpened: number
    completionPerSubmitted: number
    firstRunPerCompleted: number
    firstOutputPerCompleted: number
  }
  dropOff: {
    openedToScope: number
    scopeToBriefing: number
    briefingToReview: number
    reviewToSubmitted: number
    submittedToCompleted: number
  }
  preview: PipelinePreviewCounters
  failures: PipelineFailureCounters
  windowed: {
    last24h: {
      opened: number
      completed: number
      completionPerOpened: number
    }
    last7d: {
      opened: number
      completed: number
      completionPerOpened: number
    }
  }
}

export type WatchlistsOnboardingTelemetryState = {
  version: 1
  session_id: string
  counters: Record<string, number>
  quick_setup: {
    step_completed: StepCounters
    cancelled_at_step: StepCounters
    failed_at_step: StepCounters
    completed_by_goal: Record<"briefing" | "triage", number>
    completed_by_destination: QuickSetupDestinationCounters
    completed_with_run_now: number
    completed_without_run_now: number
    preview_loaded: QuickSetupPreviewCounters
    preview_failed: QuickSetupPreviewCounters
    test_run_triggered: number
    test_run_failed: number
    first_run_success: number
    first_output_success: number
    active_setup_started_at: number | null
    pending_first_run_success_at: number[]
    pending_first_output_success_at: number[]
    seconds_to_setup_completion_samples: number[]
    seconds_to_first_run_success_samples: number[]
    seconds_to_first_output_success_samples: number[]
  }
  guided_tour: {
    started: number
    completed: number
    dismissed: number
    resumed: number
    step_views: GuidedStepCounters
  }
  uc2_pipeline: {
    opened: number
    step_completed: PipelineStepCounters
    submitted_by_mode: PipelineModeCounters
    completed_by_mode: PipelineModeCounters
    completed_with_run_now: number
    completed_without_run_now: number
    preview_by_status: PipelinePreviewCounters
    failed_by_stage: PipelineFailureCounters
  }
  value_milestones: {
    first_run_succeeded: number
    first_output_succeeded: number
    first_run_succeeded_at: number | null
    first_output_succeeded_at: number | null
  }
  last_event_at: number | null
  recent_events: WatchlistsOnboardingRecentEvent[]
}

const DEFAULT_STEP_COUNTERS: StepCounters = {
  feed: 0,
  monitor: 0,
  review: 0
}

const DEFAULT_GUIDED_STEP_COUNTERS: GuidedStepCounters = {
  "1": 0,
  "2": 0,
  "3": 0,
  "4": 0,
  "5": 0
}

const DEFAULT_PIPELINE_STEP_COUNTERS: PipelineStepCounters = {
  scope: 0,
  briefing: 0,
  review: 0
}

const DEFAULT_PIPELINE_MODE_COUNTERS: PipelineModeCounters = {
  create: 0,
  test: 0
}

const DEFAULT_PIPELINE_PREVIEW_COUNTERS: PipelinePreviewCounters = {
  success: 0,
  empty: 0,
  no_run_context: 0,
  template_empty: 0,
  error: 0
}

const DEFAULT_PIPELINE_FAILURE_COUNTERS: PipelineFailureCounters = {
  validation: 0,
  source_create: 0,
  job_create: 0,
  run_trigger: 0,
  output_create: 0,
  rollback: 0
}

const DEFAULT_STATE: WatchlistsOnboardingTelemetryState = {
  version: 1,
  session_id: createOnboardingSessionId(),
  counters: {},
  quick_setup: {
    step_completed: { ...DEFAULT_STEP_COUNTERS },
    cancelled_at_step: { ...DEFAULT_STEP_COUNTERS },
    failed_at_step: { ...DEFAULT_STEP_COUNTERS },
    completed_by_goal: {
      briefing: 0,
      triage: 0
    },
    completed_by_destination: { ...DEFAULT_DESTINATION_COUNTERS },
    completed_with_run_now: 0,
    completed_without_run_now: 0,
    preview_loaded: { ...DEFAULT_PREVIEW_COUNTERS },
    preview_failed: { ...DEFAULT_PREVIEW_COUNTERS },
    test_run_triggered: 0,
    test_run_failed: 0,
    first_run_success: 0,
    first_output_success: 0,
    active_setup_started_at: null,
    pending_first_run_success_at: [],
    pending_first_output_success_at: [],
    seconds_to_setup_completion_samples: [],
    seconds_to_first_run_success_samples: [],
    seconds_to_first_output_success_samples: []
  },
  guided_tour: {
    started: 0,
    completed: 0,
    dismissed: 0,
    resumed: 0,
    step_views: { ...DEFAULT_GUIDED_STEP_COUNTERS }
  },
  uc2_pipeline: {
    opened: 0,
    step_completed: { ...DEFAULT_PIPELINE_STEP_COUNTERS },
    submitted_by_mode: { ...DEFAULT_PIPELINE_MODE_COUNTERS },
    completed_by_mode: { ...DEFAULT_PIPELINE_MODE_COUNTERS },
    completed_with_run_now: 0,
    completed_without_run_now: 0,
    preview_by_status: { ...DEFAULT_PIPELINE_PREVIEW_COUNTERS },
    failed_by_stage: { ...DEFAULT_PIPELINE_FAILURE_COUNTERS }
  },
  value_milestones: {
    first_run_succeeded: 0,
    first_output_succeeded: 0,
    first_run_succeeded_at: null,
    first_output_succeeded_at: null
  },
  last_event_at: null,
  recent_events: []
}

const incrementCounter = (counters: Record<string, number>, key: string) => {
  counters[key] = (counters[key] || 0) + 1
}

const toBoundedNumberList = (value: unknown, maxSize: number): number[] => {
  if (!Array.isArray(value)) return []
  return value
    .map((entry) => Number(entry))
    .filter((entry) => Number.isFinite(entry) && entry >= 0)
    .slice(-maxSize)
}

const pushDurationSample = (samples: number[], durationSeconds: number): number[] => {
  if (!Number.isFinite(durationSeconds) || durationSeconds < 0) return samples.slice(-MAX_DURATION_SAMPLES)
  const rounded = Math.round(durationSeconds * 100) / 100
  return [...samples, rounded].slice(-MAX_DURATION_SAMPLES)
}

const consumePendingTimestamp = (pending: number[]): { next: number[]; consumedAt: number | null } => {
  if (!Array.isArray(pending) || pending.length <= 0) {
    return {
      next: [],
      consumedAt: null
    }
  }
  const [consumedAt, ...rest] = pending
  return {
    next: rest.slice(-MAX_PENDING_COMPLETION_EVENTS),
    consumedAt: Number.isFinite(consumedAt) ? Number(consumedAt) : null
  }
}

const hasPendingOnboardingMilestone = (
  state: WatchlistsOnboardingTelemetryState,
  event: WatchlistsOnboardingTelemetryEvent
): boolean => {
  if (event.type === "quick_setup_first_run_succeeded") {
    return state.quick_setup.pending_first_run_success_at.length > 0
  }
  if (event.type === "quick_setup_first_output_succeeded") {
    return state.quick_setup.pending_first_output_success_at.length > 0
  }
  return true
}

const toEventDetails = (
  event: WatchlistsOnboardingTelemetryEvent
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

const readTelemetryState =
  async (): Promise<WatchlistsOnboardingTelemetryState> => {
    const raw = await storage.get<WatchlistsOnboardingTelemetryState | undefined>(
      WATCHLISTS_ONBOARDING_TELEMETRY_STORAGE_KEY
    )
    const state = raw && typeof raw === "object" ? raw : DEFAULT_STATE
    return {
      ...DEFAULT_STATE,
      ...state,
      session_id:
        typeof state.session_id === "string" && state.session_id.trim().length > 0
          ? state.session_id
          : createOnboardingSessionId(),
      counters: { ...DEFAULT_STATE.counters, ...(state.counters || {}) },
      quick_setup: {
        ...DEFAULT_STATE.quick_setup,
        ...(state.quick_setup || {}),
        step_completed: {
          ...DEFAULT_STATE.quick_setup.step_completed,
          ...(state.quick_setup?.step_completed || {})
        },
        cancelled_at_step: {
          ...DEFAULT_STATE.quick_setup.cancelled_at_step,
          ...(state.quick_setup?.cancelled_at_step || {})
        },
        failed_at_step: {
          ...DEFAULT_STATE.quick_setup.failed_at_step,
          ...(state.quick_setup?.failed_at_step || {})
        },
        completed_by_goal: {
          ...DEFAULT_STATE.quick_setup.completed_by_goal,
          ...(state.quick_setup?.completed_by_goal || {})
        },
        completed_by_destination: {
          ...DEFAULT_STATE.quick_setup.completed_by_destination,
          ...(state.quick_setup?.completed_by_destination || {})
        },
        preview_loaded: {
          ...DEFAULT_STATE.quick_setup.preview_loaded,
          ...(state.quick_setup?.preview_loaded || {})
        },
        preview_failed: {
          ...DEFAULT_STATE.quick_setup.preview_failed,
          ...(state.quick_setup?.preview_failed || {})
        },
        pending_first_run_success_at: toBoundedNumberList(
          state.quick_setup?.pending_first_run_success_at,
          MAX_PENDING_COMPLETION_EVENTS
        ),
        pending_first_output_success_at: toBoundedNumberList(
          state.quick_setup?.pending_first_output_success_at,
          MAX_PENDING_COMPLETION_EVENTS
        ),
        seconds_to_setup_completion_samples: toBoundedNumberList(
          state.quick_setup?.seconds_to_setup_completion_samples,
          MAX_DURATION_SAMPLES
        ),
        seconds_to_first_run_success_samples: toBoundedNumberList(
          state.quick_setup?.seconds_to_first_run_success_samples,
          MAX_DURATION_SAMPLES
        ),
        seconds_to_first_output_success_samples: toBoundedNumberList(
          state.quick_setup?.seconds_to_first_output_success_samples,
          MAX_DURATION_SAMPLES
        ),
        active_setup_started_at: Number.isFinite(state.quick_setup?.active_setup_started_at)
          ? Number(state.quick_setup?.active_setup_started_at)
          : null
      },
      guided_tour: {
        ...DEFAULT_STATE.guided_tour,
        ...(state.guided_tour || {}),
        step_views: {
          ...DEFAULT_STATE.guided_tour.step_views,
          ...(state.guided_tour?.step_views || {})
        }
      },
      uc2_pipeline: {
        ...DEFAULT_STATE.uc2_pipeline,
        ...(state.uc2_pipeline || {}),
        step_completed: {
          ...DEFAULT_PIPELINE_STEP_COUNTERS,
          ...(state.uc2_pipeline?.step_completed || {})
        },
        submitted_by_mode: {
          ...DEFAULT_PIPELINE_MODE_COUNTERS,
          ...(state.uc2_pipeline?.submitted_by_mode || {})
        },
        completed_by_mode: {
          ...DEFAULT_PIPELINE_MODE_COUNTERS,
          ...(state.uc2_pipeline?.completed_by_mode || {})
        },
        preview_by_status: {
          ...DEFAULT_PIPELINE_PREVIEW_COUNTERS,
          ...(state.uc2_pipeline?.preview_by_status || {})
        },
        failed_by_stage: {
          ...DEFAULT_PIPELINE_FAILURE_COUNTERS,
          ...(state.uc2_pipeline?.failed_by_stage || {})
        }
      },
      value_milestones: {
        ...DEFAULT_STATE.value_milestones,
        ...(state.value_milestones || {})
      },
      recent_events: Array.isArray(state.recent_events)
        ? state.recent_events.slice(-MAX_RECENT_EVENTS)
        : []
    }
  }

const writeTelemetryState = async (state: WatchlistsOnboardingTelemetryState) => {
  await storage.set(WATCHLISTS_ONBOARDING_TELEMETRY_STORAGE_KEY, state)
}

export const trackWatchlistsOnboardingTelemetry = async (
  event: WatchlistsOnboardingTelemetryEvent
) => {
  try {
    const state = await readTelemetryState()
    if (!hasPendingOnboardingMilestone(state, event)) {
      return
    }
    const now = Date.now()
    let shouldRecord = true
    let includeInRecentEvents = true

    switch (event.type) {
      case "quick_setup_step_completed":
        state.quick_setup.step_completed[event.step] += 1
        break
      case "quick_setup_cancelled":
        state.quick_setup.cancelled_at_step[event.step] += 1
        state.quick_setup.active_setup_started_at = null
        break
      case "quick_setup_failed":
        state.quick_setup.failed_at_step[event.step] += 1
        state.quick_setup.active_setup_started_at = null
        break
      case "quick_setup_completed":
        state.quick_setup.completed_by_goal[event.goal] += 1
        state.quick_setup.completed_by_destination[event.destination] += 1
        if (Number.isFinite(state.quick_setup.active_setup_started_at)) {
          const elapsedSeconds =
            (now - Number(state.quick_setup.active_setup_started_at)) / 1000
          state.quick_setup.seconds_to_setup_completion_samples = pushDurationSample(
            state.quick_setup.seconds_to_setup_completion_samples,
            elapsedSeconds
          )
        }
        state.quick_setup.active_setup_started_at = null
        if (event.runNow) {
          state.quick_setup.completed_with_run_now += 1
        } else {
          state.quick_setup.completed_without_run_now += 1
        }
        if (event.goal === "briefing") {
          state.quick_setup.pending_first_run_success_at = [
            ...state.quick_setup.pending_first_run_success_at,
            now
          ].slice(-MAX_PENDING_COMPLETION_EVENTS)
          state.quick_setup.pending_first_output_success_at = [
            ...state.quick_setup.pending_first_output_success_at,
            now
          ].slice(-MAX_PENDING_COMPLETION_EVENTS)
        }
        break
      case "quick_setup_preview_loaded":
        state.quick_setup.preview_loaded[event.preview] += 1
        includeInRecentEvents = false
        break
      case "quick_setup_preview_failed":
        state.quick_setup.preview_failed[event.preview] += 1
        includeInRecentEvents = false
        break
      case "quick_setup_test_run_triggered":
        state.quick_setup.test_run_triggered += 1
        includeInRecentEvents = false
        break
      case "quick_setup_test_run_failed":
        state.quick_setup.test_run_failed += 1
        break
      case "quick_setup_first_run_succeeded": {
        const runCompletion = consumePendingTimestamp(
          state.quick_setup.pending_first_run_success_at
        )
        state.quick_setup.pending_first_run_success_at = runCompletion.next
        state.quick_setup.first_run_success += 1
        includeInRecentEvents = false
        if (runCompletion.consumedAt != null) {
          state.quick_setup.seconds_to_first_run_success_samples = pushDurationSample(
            state.quick_setup.seconds_to_first_run_success_samples,
            (now - runCompletion.consumedAt) / 1000
          )
        }
        break
      }
      case "quick_setup_first_output_succeeded": {
        const outputCompletion = consumePendingTimestamp(
          state.quick_setup.pending_first_output_success_at
        )
        state.quick_setup.pending_first_output_success_at = outputCompletion.next
        state.quick_setup.first_output_success += 1
        includeInRecentEvents = false
        if (outputCompletion.consumedAt != null) {
          state.quick_setup.seconds_to_first_output_success_samples = pushDurationSample(
            state.quick_setup.seconds_to_first_output_success_samples,
            (now - outputCompletion.consumedAt) / 1000
          )
        }
        break
      }
      case "guided_tour_started":
        state.guided_tour.started += 1
        break
      case "guided_tour_completed":
        state.guided_tour.completed += 1
        break
      case "guided_tour_dismissed":
        state.guided_tour.dismissed += 1
        break
      case "guided_tour_resumed":
        state.guided_tour.resumed += 1
        break
      case "guided_tour_step_viewed":
        state.guided_tour.step_views[String(event.step) as keyof GuidedStepCounters] += 1
        break
      case "pipeline_setup_opened":
        state.uc2_pipeline.opened += 1
        break
      case "pipeline_setup_step_completed":
        state.uc2_pipeline.step_completed[event.step] += 1
        break
      case "pipeline_setup_preview_generated":
        state.uc2_pipeline.preview_by_status[event.status] += 1
        break
      case "pipeline_setup_submitted":
        state.uc2_pipeline.submitted_by_mode[event.mode] += 1
        break
      case "pipeline_setup_completed":
        state.uc2_pipeline.completed_by_mode[event.mode] += 1
        if (event.runNow) {
          state.uc2_pipeline.completed_with_run_now += 1
        } else {
          state.uc2_pipeline.completed_without_run_now += 1
        }
        break
      case "pipeline_setup_failed":
        state.uc2_pipeline.failed_by_stage[event.stage] += 1
        break
      case "first_run_succeeded":
        if (state.value_milestones.first_run_succeeded > 0) {
          shouldRecord = false
          break
        }
        state.value_milestones.first_run_succeeded = 1
        state.value_milestones.first_run_succeeded_at = now
        break
      case "first_output_succeeded":
        if (state.value_milestones.first_output_succeeded > 0) {
          shouldRecord = false
          break
        }
        state.value_milestones.first_output_succeeded = 1
        state.value_milestones.first_output_succeeded_at = now
        break
      default:
        break
    }

    if (!shouldRecord) return

    state.last_event_at = now
    incrementCounter(state.counters, event.type)

    if (includeInRecentEvents) {
      state.recent_events.push({
        type: event.type,
        at: now,
        details: toEventDetails(event)
      })
      if (state.recent_events.length > MAX_RECENT_EVENTS) {
        state.recent_events = state.recent_events.slice(-MAX_RECENT_EVENTS)
      }
    }

    await writeTelemetryState(state)

    void recordWatchlistsOnboardingTelemetry({
      session_id: state.session_id,
      event_type: event.type,
      event_at: new Date(now).toISOString(),
      details: toEventDetails(event)
    }).catch(() => {
      // Backend telemetry sink failures are non-blocking by design.
    })
  } catch (error) {
    console.warn("[watchlists-onboarding-telemetry] Failed to record event", error)
  }
}

export const getWatchlistsOnboardingTelemetryState =
  async (): Promise<WatchlistsOnboardingTelemetryState> => readTelemetryState()

export const resetWatchlistsOnboardingTelemetryState = async () => {
  await storage.set(WATCHLISTS_ONBOARDING_TELEMETRY_STORAGE_KEY, DEFAULT_STATE)
}

export const WATCHLISTS_UC2_PIPELINE_EVENT_TYPES: WatchlistsOnboardingTelemetryEventType[] = [
  "pipeline_setup_opened",
  "pipeline_setup_step_completed",
  "pipeline_setup_preview_generated",
  "pipeline_setup_submitted",
  "pipeline_setup_completed",
  "pipeline_setup_failed"
]

export const queryWatchlistsOnboardingTelemetryEvents = (
  state: WatchlistsOnboardingTelemetryState | null | undefined,
  query: WatchlistsOnboardingTelemetryQuery = {}
): WatchlistsOnboardingRecentEvent[] => {
  if (!state || !Array.isArray(state.recent_events)) return []

  const eventTypeFilter =
    Array.isArray(query.eventTypes) && query.eventTypes.length > 0
      ? new Set(query.eventTypes)
      : null
  const sinceMs =
    typeof query.sinceMs === "number" && Number.isFinite(query.sinceMs)
      ? query.sinceMs
      : null

  return state.recent_events
    .filter((event) => {
      if (!event || typeof event !== "object") return false
      if (eventTypeFilter && !eventTypeFilter.has(event.type)) return false
      if (sinceMs !== null && event.at < sinceMs) return false
      return true
    })
    .slice()
    .sort((a, b) => b.at - a.at)
}

const toFiniteRate = (numerator: number, denominator: number): number => {
  if (!Number.isFinite(numerator) || numerator <= 0) return 0
  if (!Number.isFinite(denominator) || denominator <= 0) return 0
  return numerator / denominator
}

const sumNumberRecord = (value: Record<string, number>): number =>
  Object.values(value || {}).reduce((sum, entry) => {
    const next = Number(entry)
    return Number.isFinite(next) && next > 0 ? sum + next : sum
  }, 0)

export const buildWatchlistsUc2PipelineDashboardSnapshot = (
  state: WatchlistsOnboardingTelemetryState | null | undefined,
  now = Date.now()
): WatchlistsUc2PipelineDashboardSnapshot => {
  const pipeline = state?.uc2_pipeline || DEFAULT_STATE.uc2_pipeline
  const opened = pipeline.opened || 0
  const stepScopeCompleted = pipeline.step_completed.scope || 0
  const stepBriefingCompleted = pipeline.step_completed.briefing || 0
  const stepReviewCompleted = pipeline.step_completed.review || 0
  const submitted = sumNumberRecord(pipeline.submitted_by_mode)
  const completed = sumNumberRecord(pipeline.completed_by_mode)
  const completedWithRunNow = pipeline.completed_with_run_now || 0
  const completedWithoutRunNow = pipeline.completed_without_run_now || 0

  const firstRunSucceeded = state?.value_milestones.first_run_succeeded || 0
  const firstOutputSucceeded = state?.value_milestones.first_output_succeeded || 0

  const openedLast24h = queryWatchlistsOnboardingTelemetryEvents(state, {
    eventTypes: ["pipeline_setup_opened"],
    sinceMs: now - 24 * 60 * 60 * 1000
  }).length
  const completedLast24h = queryWatchlistsOnboardingTelemetryEvents(state, {
    eventTypes: ["pipeline_setup_completed"],
    sinceMs: now - 24 * 60 * 60 * 1000
  }).length
  const openedLast7d = queryWatchlistsOnboardingTelemetryEvents(state, {
    eventTypes: ["pipeline_setup_opened"],
    sinceMs: now - 7 * 24 * 60 * 60 * 1000
  }).length
  const completedLast7d = queryWatchlistsOnboardingTelemetryEvents(state, {
    eventTypes: ["pipeline_setup_completed"],
    sinceMs: now - 7 * 24 * 60 * 60 * 1000
  }).length

  return {
    funnel: {
      opened,
      stepScopeCompleted,
      stepBriefingCompleted,
      stepReviewCompleted,
      submitted,
      completed,
      completedWithRunNow,
      completedWithoutRunNow
    },
    firstSuccess: {
      firstRunSucceeded,
      firstOutputSucceeded
    },
    rates: {
      completionPerOpened: toFiniteRate(completed, opened),
      completionPerSubmitted: toFiniteRate(completed, submitted),
      firstRunPerCompleted: toFiniteRate(firstRunSucceeded, completed),
      firstOutputPerCompleted: toFiniteRate(firstOutputSucceeded, completed)
    },
    dropOff: {
      openedToScope: Math.max(0, opened - stepScopeCompleted),
      scopeToBriefing: Math.max(0, stepScopeCompleted - stepBriefingCompleted),
      briefingToReview: Math.max(0, stepBriefingCompleted - stepReviewCompleted),
      reviewToSubmitted: Math.max(0, stepReviewCompleted - submitted),
      submittedToCompleted: Math.max(0, submitted - completed)
    },
    preview: {
      ...DEFAULT_PIPELINE_PREVIEW_COUNTERS,
      ...(pipeline.preview_by_status || {})
    },
    failures: {
      ...DEFAULT_PIPELINE_FAILURE_COUNTERS,
      ...(pipeline.failed_by_stage || {})
    },
    windowed: {
      last24h: {
        opened: openedLast24h,
        completed: completedLast24h,
        completionPerOpened: toFiniteRate(completedLast24h, openedLast24h)
      },
      last7d: {
        opened: openedLast7d,
        completed: completedLast7d,
        completionPerOpened: toFiniteRate(completedLast7d, openedLast7d)
      }
    }
  }
}
