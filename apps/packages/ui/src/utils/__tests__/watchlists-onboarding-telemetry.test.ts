import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const STORAGE_KEY = "tldw:watchlists:onboarding:telemetry"

describe("watchlists-onboarding-telemetry", () => {
  let storageMap: Map<string, unknown>
  let recordOnboardingTelemetryMock: (
    ...args: unknown[]
  ) => Promise<{ accepted: boolean }>

  beforeEach(() => {
    storageMap = new Map<string, unknown>()
    recordOnboardingTelemetryMock = vi.fn(async () => ({ accepted: true }))
    vi.resetModules()
    vi.doMock("@/utils/safe-storage", () => ({
      createSafeStorage: () => ({
        get: async (key: string) => storageMap.get(key),
        set: async (key: string, value: unknown) => {
          storageMap.set(key, value)
        },
        remove: async (key: string) => {
          storageMap.delete(key)
        }
      })
    }))
    vi.doMock("@/services/watchlists", () => ({
      recordWatchlistsOnboardingTelemetry: (...args: unknown[]) =>
        recordOnboardingTelemetryMock(...args)
    }))
  })

  afterEach(() => {
    vi.clearAllMocks()
    vi.resetModules()
  })

  it("records quick setup and guided tour funnel counters", async () => {
    const telemetry = await import("@/utils/watchlists-onboarding-telemetry")

    await telemetry.trackWatchlistsOnboardingTelemetry({ type: "quick_setup_opened" })
    await telemetry.trackWatchlistsOnboardingTelemetry({
      type: "quick_setup_step_completed",
      step: "feed"
    })
    await telemetry.trackWatchlistsOnboardingTelemetry({
      type: "quick_setup_step_completed",
      step: "monitor"
    })
    await telemetry.trackWatchlistsOnboardingTelemetry({
      type: "quick_setup_completed",
      goal: "briefing",
      runNow: false,
      destination: "outputs"
    })
    await telemetry.trackWatchlistsOnboardingTelemetry({
      type: "quick_setup_preview_loaded",
      preview: "candidate",
      total: 6,
      ingestable: 4,
      filtered: 2
    })
    await telemetry.trackWatchlistsOnboardingTelemetry({
      type: "quick_setup_preview_loaded",
      preview: "template",
      goal: "briefing",
      audioEnabled: true
    })
    await telemetry.trackWatchlistsOnboardingTelemetry({
      type: "quick_setup_preview_failed",
      preview: "candidate",
      reason: "load_failed"
    })
    await telemetry.trackWatchlistsOnboardingTelemetry({
      type: "quick_setup_test_run_triggered",
      runId: 777
    })
    await telemetry.trackWatchlistsOnboardingTelemetry({
      type: "quick_setup_first_run_succeeded",
      source: "run_notifications",
      runId: 777
    })
    await telemetry.trackWatchlistsOnboardingTelemetry({
      type: "quick_setup_first_output_succeeded",
      source: "outputs"
    })
    await telemetry.trackWatchlistsOnboardingTelemetry({ type: "guided_tour_started" })
    await telemetry.trackWatchlistsOnboardingTelemetry({
      type: "guided_tour_step_viewed",
      step: 1
    })
    await telemetry.trackWatchlistsOnboardingTelemetry({
      type: "guided_tour_step_viewed",
      step: 2
    })
    await telemetry.trackWatchlistsOnboardingTelemetry({
      type: "guided_tour_dismissed",
      step: 2
    })
    await telemetry.trackWatchlistsOnboardingTelemetry({
      type: "first_run_succeeded",
      runId: 91
    })
    await telemetry.trackWatchlistsOnboardingTelemetry({
      type: "first_output_succeeded",
      outputId: 301,
      format: "md"
    })

    const state = storageMap.get(STORAGE_KEY) as Record<string, any>
    expect(state.counters.quick_setup_opened).toBe(1)
    expect(state.counters.quick_setup_step_completed).toBe(2)
    expect(state.counters.quick_setup_completed).toBe(1)
    expect(state.quick_setup.step_completed.feed).toBe(1)
    expect(state.quick_setup.step_completed.monitor).toBe(1)
    expect(state.quick_setup.completed_by_goal.briefing).toBe(1)
    expect(state.quick_setup.completed_by_destination.outputs).toBe(1)
    expect(state.quick_setup.completed_without_run_now).toBe(1)
    expect(state.quick_setup.preview_loaded.candidate).toBe(1)
    expect(state.quick_setup.preview_loaded.template).toBe(1)
    expect(state.quick_setup.preview_failed.candidate).toBe(1)
    expect(state.quick_setup.test_run_triggered).toBe(1)
    expect(state.quick_setup.first_run_success).toBe(1)
    expect(state.quick_setup.first_output_success).toBe(1)
    expect(state.quick_setup.pending_first_run_success_at).toHaveLength(0)
    expect(state.quick_setup.pending_first_output_success_at).toHaveLength(0)
    expect(state.guided_tour.started).toBe(1)
    expect(state.guided_tour.dismissed).toBe(1)
    expect(state.guided_tour.step_views["1"]).toBe(1)
    expect(state.guided_tour.step_views["2"]).toBe(1)
    expect(state.value_milestones.first_run_succeeded).toBe(1)
    expect(state.value_milestones.first_output_succeeded).toBe(1)
    expect(state.value_milestones.first_run_succeeded_at).toBeTypeOf("number")
    expect(state.value_milestones.first_output_succeeded_at).toBeTypeOf("number")
    expect(state.recent_events).toHaveLength(10)
  })

  it("caps recent events to the configured maximum", async () => {
    const telemetry = await import("@/utils/watchlists-onboarding-telemetry")

    for (let i = 0; i < 260; i += 1) {
      await telemetry.trackWatchlistsOnboardingTelemetry({
        type: "guided_tour_step_viewed",
        step: ((i % 5) + 1) as 1 | 2 | 3 | 4 | 5
      })
    }

    const state = storageMap.get(STORAGE_KEY) as Record<string, any>
    expect(state.recent_events.length).toBe(200)
    expect(state.counters.guided_tour_step_viewed).toBe(260)
  })

  it("records first-value milestones only once", async () => {
    const telemetry = await import("@/utils/watchlists-onboarding-telemetry")

    await telemetry.trackWatchlistsOnboardingTelemetry({
      type: "first_run_succeeded",
      runId: 1
    })
    await telemetry.trackWatchlistsOnboardingTelemetry({
      type: "first_run_succeeded",
      runId: 2
    })
    await telemetry.trackWatchlistsOnboardingTelemetry({
      type: "first_output_succeeded",
      outputId: 11,
      format: "md"
    })
    await telemetry.trackWatchlistsOnboardingTelemetry({
      type: "first_output_succeeded",
      outputId: 12,
      format: "html"
    })

    const state = storageMap.get(STORAGE_KEY) as Record<string, any>
    expect(state.counters.first_run_succeeded).toBe(1)
    expect(state.counters.first_output_succeeded).toBe(1)
    expect(state.value_milestones.first_run_succeeded).toBe(1)
    expect(state.value_milestones.first_output_succeeded).toBe(1)
    const milestoneEvents = state.recent_events.filter((event: Record<string, unknown>) =>
      event.type === "first_run_succeeded" || event.type === "first_output_succeeded"
    )
    expect(milestoneEvents).toHaveLength(2)
  })

  it("tracks UC2 pipeline funnel milestones and builds KPI snapshot", async () => {
    const telemetry = await import("@/utils/watchlists-onboarding-telemetry")

    await telemetry.trackWatchlistsOnboardingTelemetry({ type: "pipeline_setup_opened" })
    await telemetry.trackWatchlistsOnboardingTelemetry({
      type: "pipeline_setup_step_completed",
      step: "scope"
    })
    await telemetry.trackWatchlistsOnboardingTelemetry({
      type: "pipeline_setup_step_completed",
      step: "briefing"
    })
    await telemetry.trackWatchlistsOnboardingTelemetry({
      type: "pipeline_setup_step_completed",
      step: "review"
    })
    await telemetry.trackWatchlistsOnboardingTelemetry({
      type: "pipeline_setup_preview_generated",
      status: "success",
      warning_count: 1,
      run_id: 44
    })
    await telemetry.trackWatchlistsOnboardingTelemetry({
      type: "pipeline_setup_submitted",
      mode: "create",
      runNow: true
    })
    await telemetry.trackWatchlistsOnboardingTelemetry({
      type: "pipeline_setup_completed",
      mode: "create",
      runNow: true,
      destination: "outputs"
    })
    await telemetry.trackWatchlistsOnboardingTelemetry({
      type: "first_run_succeeded",
      runId: 44
    })
    await telemetry.trackWatchlistsOnboardingTelemetry({
      type: "first_output_succeeded",
      outputId: 55,
      format: "md"
    })

    const state = await telemetry.getWatchlistsOnboardingTelemetryState()
    expect(state.uc2_pipeline.opened).toBe(1)
    expect(state.uc2_pipeline.step_completed.scope).toBe(1)
    expect(state.uc2_pipeline.step_completed.briefing).toBe(1)
    expect(state.uc2_pipeline.step_completed.review).toBe(1)
    expect(state.uc2_pipeline.submitted_by_mode.create).toBe(1)
    expect(state.uc2_pipeline.completed_by_mode.create).toBe(1)
    expect(state.uc2_pipeline.completed_with_run_now).toBe(1)
    expect(state.uc2_pipeline.preview_by_status.success).toBe(1)

    const snapshot = telemetry.buildWatchlistsUc2PipelineDashboardSnapshot(state)
    expect(snapshot.funnel.opened).toBe(1)
    expect(snapshot.funnel.submitted).toBe(1)
    expect(snapshot.funnel.completed).toBe(1)
    expect(snapshot.rates.completionPerOpened).toBe(1)
    expect(snapshot.rates.completionPerSubmitted).toBe(1)
    expect(snapshot.rates.firstRunPerCompleted).toBe(1)
    expect(snapshot.rates.firstOutputPerCompleted).toBe(1)
    expect(snapshot.windowed.last24h.opened).toBe(1)
    expect(snapshot.windowed.last24h.completed).toBe(1)
    expect(snapshot.windowed.last7d.opened).toBe(1)
    expect(snapshot.windowed.last7d.completed).toBe(1)

    const pipelineEvents = telemetry.queryWatchlistsOnboardingTelemetryEvents(state, {
      eventTypes: telemetry.WATCHLISTS_UC2_PIPELINE_EVENT_TYPES
    })
    expect(pipelineEvents.length).toBeGreaterThanOrEqual(7)
  })

  it("tracks pipeline preview/failure statuses for drop-off analysis", async () => {
    const telemetry = await import("@/utils/watchlists-onboarding-telemetry")

    await telemetry.trackWatchlistsOnboardingTelemetry({ type: "pipeline_setup_opened" })
    await telemetry.trackWatchlistsOnboardingTelemetry({
      type: "pipeline_setup_preview_generated",
      status: "no_run_context"
    })
    await telemetry.trackWatchlistsOnboardingTelemetry({
      type: "pipeline_setup_submitted",
      mode: "test",
      runNow: true
    })
    await telemetry.trackWatchlistsOnboardingTelemetry({
      type: "pipeline_setup_failed",
      stage: "run_trigger",
      mode: "test",
      runNow: true
    })
    await telemetry.trackWatchlistsOnboardingTelemetry({
      type: "pipeline_setup_failed",
      stage: "source_create",
      mode: "create",
      runNow: false
    })

    const state = await telemetry.getWatchlistsOnboardingTelemetryState()
    expect(state.uc2_pipeline.preview_by_status.no_run_context).toBe(1)
    expect(state.uc2_pipeline.failed_by_stage.run_trigger).toBe(1)
    expect(state.uc2_pipeline.failed_by_stage.source_create).toBe(1)

    const snapshot = telemetry.buildWatchlistsUc2PipelineDashboardSnapshot(state)
    expect(snapshot.funnel.opened).toBe(1)
    expect(snapshot.funnel.completed).toBe(0)
    expect(snapshot.rates.completionPerOpened).toBe(0)
    expect(snapshot.failures.run_trigger).toBe(1)
    expect(snapshot.failures.source_create).toBe(1)
    expect(snapshot.preview.no_run_context).toBe(1)
  })
})
