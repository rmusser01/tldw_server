// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from "vitest"
import { useMilestoneStore } from "../milestones"

describe("milestone store", () => {
  beforeEach(() => {
    localStorage.clear()
    useMilestoneStore.getState().resetMilestones()
  })

  it("starts with no milestones completed", () => {
    expect(useMilestoneStore.getState().getCompletedCount()).toBe(0)
  })

  it("marks a milestone with timestamp", () => {
    const before = Date.now()
    useMilestoneStore.getState().markMilestone("first_connection")
    const state = useMilestoneStore.getState()
    expect(state.isMilestoneCompleted("first_connection")).toBe(true)
    expect(state.completedMilestones.first_connection).toBeGreaterThanOrEqual(before)
  })

  it("does not overwrite existing milestone timestamp", () => {
    useMilestoneStore.getState().markMilestone("first_ingest")
    const firstTs = useMilestoneStore.getState().completedMilestones.first_ingest
    useMilestoneStore.getState().markMilestone("first_ingest")
    expect(useMilestoneStore.getState().completedMilestones.first_ingest).toBe(firstTs)
  })

  it("persists to localStorage", () => {
    useMilestoneStore.getState().markMilestone("first_chat")
    const stored = JSON.parse(localStorage.getItem("tldw:milestones") ?? "{}")
    expect(stored.first_chat).toBeDefined()
  })

  it("counts completed milestones", () => {
    useMilestoneStore.getState().markMilestone("first_connection")
    useMilestoneStore.getState().markMilestone("first_ingest")
    expect(useMilestoneStore.getState().getCompletedCount()).toBe(2)
  })

  it("resets all milestones", () => {
    useMilestoneStore.getState().markMilestone("first_connection")
    useMilestoneStore.getState().resetMilestones()
    expect(useMilestoneStore.getState().getCompletedCount()).toBe(0)
  })

  it("bootstraps first_connection from first-run flag", () => {
    localStorage.setItem("__tldw_first_run_complete", "true")
    useMilestoneStore.getState().bootstrapFromExistingUsage()
    expect(useMilestoneStore.getState().isMilestoneCompleted("first_connection")).toBe(true)
  })

  it("bootstraps first_ingest from telemetry session state", () => {
    const telemetry = {
      version: 1,
      counters: {},
      last_event_at: null,
      current_session: {
        started_at: 1000,
        first_ingest_at: 2000,
        first_chat_after_ingest_at: null,
        first_media_id: null,
        source_label: null
      },
      aggregates: {},
      recent_events: []
    }
    localStorage.setItem(
      "tldw:onboarding:ingestion:telemetry",
      JSON.stringify(telemetry)
    )
    useMilestoneStore.getState().bootstrapFromExistingUsage()
    expect(useMilestoneStore.getState().isMilestoneCompleted("first_ingest")).toBe(true)
    expect(useMilestoneStore.getState().completedMilestones.first_ingest).toBe(2000)
  })

  it("bootstraps first_chat from telemetry session state", () => {
    const telemetry = {
      version: 1,
      counters: {},
      last_event_at: null,
      current_session: {
        started_at: 1000,
        first_ingest_at: 2000,
        first_chat_after_ingest_at: 3000,
        first_media_id: null,
        source_label: null
      },
      aggregates: {},
      recent_events: []
    }
    localStorage.setItem(
      "tldw:onboarding:ingestion:telemetry",
      JSON.stringify(telemetry)
    )
    useMilestoneStore.getState().bootstrapFromExistingUsage()
    expect(useMilestoneStore.getState().isMilestoneCompleted("first_chat")).toBe(true)
    expect(useMilestoneStore.getState().completedMilestones.first_chat).toBe(3000)
  })

  it("bootstraps first_ingest from counter fallback when session was reset", () => {
    const telemetry = {
      version: 1,
      counters: { onboarding_first_ingest_success: 2 },
      last_event_at: null,
      current_session: {
        started_at: null,
        first_ingest_at: null,
        first_chat_after_ingest_at: null,
        first_media_id: null,
        source_label: null
      },
      aggregates: {},
      recent_events: []
    }
    localStorage.setItem(
      "tldw:onboarding:ingestion:telemetry",
      JSON.stringify(telemetry)
    )
    const before = Date.now()
    useMilestoneStore.getState().bootstrapFromExistingUsage()
    expect(useMilestoneStore.getState().isMilestoneCompleted("first_ingest")).toBe(true)
    expect(
      useMilestoneStore.getState().completedMilestones.first_ingest
    ).toBeGreaterThanOrEqual(before)
  })

  it("bootstraps first_quiz_taken from quiz-attempt keys", () => {
    localStorage.setItem("quiz-attempt-abc123", JSON.stringify({ score: 8 }))
    useMilestoneStore.getState().bootstrapFromExistingUsage()
    expect(useMilestoneStore.getState().isMilestoneCompleted("first_quiz_taken")).toBe(true)
  })

  it("bootstraps family_profiles_created from family wizard telemetry", () => {
    localStorage.setItem(
      "tldw:family-guardrails:wizard:telemetry",
      JSON.stringify({
        version: 1,
        counters: {
          setup_started: 1,
          draft_resumed: 0,
          step_viewed: 4,
          step_completed: 4,
          step_error: 0,
          setup_completed: 1,
          drop_off: 0
        },
        by_cohort: {
          new_household: {
            setup_started: 1,
            draft_resumed: 0,
            setup_completed: 1,
            drop_off: 0
          },
          existing_household: {
            setup_started: 0,
            draft_resumed: 0,
            setup_completed: 0,
            drop_off: 0
          }
        },
        last_event_at: Date.now(),
        recent_events: []
      })
    )

    useMilestoneStore.getState().bootstrapFromExistingUsage()

    expect(useMilestoneStore.getState().isMilestoneCompleted("family_profiles_created")).toBe(true)
  })

  it("bootstraps content_rules_reviewed from moderation onboarding state", () => {
    localStorage.setItem("moderation-playground-onboarded", "true")

    useMilestoneStore.getState().bootstrapFromExistingUsage()

    expect(useMilestoneStore.getState().isMilestoneCompleted("content_rules_reviewed")).toBe(true)
  })

  it("caches the quiz-attempt bootstrap scan after the first run", () => {
    const keySpy = vi.spyOn(Storage.prototype, "key")

    useMilestoneStore.getState().bootstrapFromExistingUsage()
    expect(localStorage.getItem("tldw:milestones:quiz-attempt-scan-done")).toBe("1")

    keySpy.mockClear()
    useMilestoneStore.getState().bootstrapFromExistingUsage()

    expect(keySpy).not.toHaveBeenCalled()
  })

  it("bootstrap does not overwrite already-completed milestones", () => {
    useMilestoneStore.getState().markMilestone("first_connection")
    const originalTs = useMilestoneStore.getState().completedMilestones.first_connection

    localStorage.setItem("__tldw_first_run_complete", "true")
    useMilestoneStore.getState().bootstrapFromExistingUsage()

    expect(useMilestoneStore.getState().completedMilestones.first_connection).toBe(originalTs)
  })

  it("bootstrap is a no-op when no evidence exists", () => {
    useMilestoneStore.getState().bootstrapFromExistingUsage()
    expect(useMilestoneStore.getState().getCompletedCount()).toBe(0)
  })

  it("reset clears localStorage", () => {
    useMilestoneStore.getState().markMilestone("first_connection")
    expect(localStorage.getItem("tldw:milestones")).not.toBe("{}")
    useMilestoneStore.getState().resetMilestones()
    const stored = JSON.parse(localStorage.getItem("tldw:milestones") ?? "{}")
    expect(Object.keys(stored)).toHaveLength(0)
  })
})
