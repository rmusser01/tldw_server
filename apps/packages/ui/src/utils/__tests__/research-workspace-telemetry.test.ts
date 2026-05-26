import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const STORAGE_KEY = "tldw:research-workspace:telemetry"
const LEGACY_STORAGE_KEY = "tldw:workspace:playground:telemetry"

describe("research-workspace-telemetry", () => {
  let storageMap: Map<string, unknown>

  beforeEach(() => {
    storageMap = new Map<string, unknown>()
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
  })

  afterEach(() => {
    vi.clearAllMocks()
    vi.resetModules()
  })

  it("records counters and recent event details", async () => {
    const telemetry = await import("@/utils/research-workspace-telemetry")
    await telemetry.resetResearchWorkspaceTelemetryState()

    await telemetry.trackResearchWorkspaceTelemetry({
      type: "conflict_modal_opened",
      workspace_id: "workspace-a",
      changed_fields_count: 2
    })
    await telemetry.trackResearchWorkspaceTelemetry({
      type: "connectivity_state_changed",
      from: "connected",
      to: "disconnected"
    })
    await telemetry.trackResearchWorkspaceTelemetry({
      type: "confusion_retry_burst",
      retry_count: 3,
      window_ms: 30000
    })

    const state = storageMap.get(STORAGE_KEY) as Record<string, any>
    expect(state.counters.conflict_modal_opened).toBe(1)
    expect(state.counters.connectivity_state_changed).toBe(1)
    expect(state.counters.confusion_retry_burst).toBe(1)
    expect(state.recent_events).toHaveLength(3)
    expect(state.recent_events[0]?.details.workspace_id).toBe("workspace-a")
    expect(state.recent_events[0]?.details.changed_fields_count).toBe(2)
    expect(state.recent_events[2]?.details.retry_count).toBe(3)
  })

  it("builds confusion dashboard queries and CSV exports", async () => {
    const telemetry = await import("@/utils/research-workspace-telemetry")
    await telemetry.resetResearchWorkspaceTelemetryState()

    await telemetry.trackResearchWorkspaceTelemetry({
      type: "status_viewed",
      workspace_id: "workspace-a"
    })
    await telemetry.trackResearchWorkspaceTelemetry({
      type: "conflict_modal_opened",
      workspace_id: "workspace-a"
    })
    await telemetry.trackResearchWorkspaceTelemetry({
      type: "confusion_retry_burst",
      workspace_id: "workspace-a",
      retry_count: 3,
      window_ms: 30000
    })
    await telemetry.trackResearchWorkspaceTelemetry({
      type: "confusion_refresh_loop",
      workspace_id: "workspace-a",
      refresh_count: 3,
      window_ms: 45000
    })
    await telemetry.trackResearchWorkspaceTelemetry({
      type: "confusion_duplicate_submission",
      workspace_id: "workspace-a",
      duplicate_count: 2,
      window_ms: 12000
    })

    const state = await telemetry.getResearchWorkspaceTelemetryState()
    const confusionEvents = telemetry.queryResearchWorkspaceTelemetryEvents(
      state,
      {
        eventTypes: telemetry.RESEARCH_WORKSPACE_CONFUSION_EVENT_TYPES
      }
    )
    expect(confusionEvents).toHaveLength(3)

    const confusionSnapshot =
      telemetry.buildResearchWorkspaceConfusionDashboardSnapshot(state)
    expect(confusionSnapshot.counters.retryBurst).toBe(1)
    expect(confusionSnapshot.counters.refreshLoop).toBe(1)
    expect(confusionSnapshot.counters.duplicateSubmission).toBe(1)
    expect(confusionSnapshot.rates.retryPerStatusView).toBe(1)
    expect(confusionSnapshot.rates.refreshPerConflict).toBe(1)

    const csv = telemetry.buildResearchWorkspaceTelemetryEventsCsv(confusionEvents)
    expect(csv).toContain("event_type,timestamp_iso,timestamp_ms")
    expect(csv).toContain("confusion_retry_burst")
    expect(csv).toContain("confusion_refresh_loop")
    expect(csv).toContain("confusion_duplicate_submission")
  })

  it("imports legacy workspace playground telemetry once without writing new events to the old key", async () => {
    const legacyState = {
      version: 1,
      counters: {
        status_viewed: 2,
        confusion_retry_burst: 1
      },
      last_event_at: 1700000000000,
      recent_events: [
        {
          type: "status_viewed",
          at: 1699999999999,
          details: { workspace_id: "legacy-workspace" }
        }
      ]
    }
    storageMap.set(LEGACY_STORAGE_KEY, legacyState)

    const telemetry = await import("@/utils/research-workspace-telemetry")
    const importedState = await telemetry.getResearchWorkspaceTelemetryState()

    expect(importedState.counters.status_viewed).toBe(2)
    expect(importedState.counters.confusion_retry_burst).toBe(1)
    expect(importedState.recent_events[0]?.details.workspace_id).toBe(
      "legacy-workspace"
    )
    expect(storageMap.get(STORAGE_KEY)).toEqual(importedState)
    expect(storageMap.has(LEGACY_STORAGE_KEY)).toBe(false)

    await telemetry.trackResearchWorkspaceTelemetry({
      type: "source_status_ready",
      workspace_id: "research-workspace"
    })

    expect(storageMap.has(LEGACY_STORAGE_KEY)).toBe(false)
    const currentState = storageMap.get(STORAGE_KEY) as Record<string, any>
    expect(currentState.counters.source_status_ready).toBe(1)
  })
})
