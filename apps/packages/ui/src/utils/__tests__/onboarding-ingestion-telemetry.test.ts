import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const STORAGE_KEY = "tldw:onboarding:ingestion:telemetry"

describe("onboarding-ingestion-telemetry", () => {
  let storageMap: Map<string, unknown>
  let now = 1_000

  beforeEach(() => {
    storageMap = new Map<string, unknown>()
    now = 1_000
    vi.resetModules()
    vi.spyOn(Date, "now").mockImplementation(() => now)
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

  it("derives time-to-first-ingest from onboarding success to first ingest", async () => {
    const telemetry = await import("../onboarding-ingestion-telemetry")

    await telemetry.trackOnboardingSuccessReached("setup")
    now = 1_820
    await telemetry.trackOnboardingFirstIngestSuccess({
      successCount: 1,
      firstMediaId: "media-123",
      primarySourceLabel: "https://example.com/report"
    })

    const state = storageMap.get(STORAGE_KEY) as Record<string, any>
    expect(state.counters.onboarding_success_reached).toBe(1)
    expect(state.counters.onboarding_first_ingest_success).toBe(1)
    expect(state.aggregates.samples_time_to_first_ingest).toBe(1)
    expect(state.aggregates.total_time_to_first_ingest_ms).toBe(820)
    expect(state.aggregates.avg_time_to_first_ingest_ms).toBe(820)
    expect(state.current_session.first_media_id).toBe("media-123")
    expect(state.current_session.source_label).toBe("https://example.com/report")
  })

  it("records first-chat-after-ingest conversion exactly once per session", async () => {
    const telemetry = await import("../onboarding-ingestion-telemetry")

    await telemetry.trackOnboardingSuccessReached("setup")
    now = 1_300
    await telemetry.trackOnboardingFirstIngestSuccess({
      successCount: 1
    })

    now = 1_450
    await telemetry.trackOnboardingChatSubmitSuccess("/chat")
    now = 1_500
    await telemetry.trackOnboardingChatSubmitSuccess("/chat")

    const state = storageMap.get(STORAGE_KEY) as Record<string, any>
    expect(state.counters.onboarding_first_chat_after_ingest).toBe(1)
    expect(state.counters.onboarding_chat_submit_after_conversion).toBeUndefined()
    expect(state.aggregates.first_chat_after_ingest_conversions).toBe(1)
    expect(typeof state.current_session.first_chat_after_ingest_at).toBe("number")
  })

  it("tracks chat submissions before ingest separately from conversion", async () => {
    const telemetry = await import("../onboarding-ingestion-telemetry")

    await telemetry.trackOnboardingSuccessReached("setup")
    now = 1_080
    await telemetry.trackOnboardingChatSubmitSuccess("/chat")
    now = 1_250
    await telemetry.trackOnboardingFirstIngestSuccess({
      successCount: 2,
      firstMediaId: 7
    })
    now = 1_410
    await telemetry.trackOnboardingChatSubmitSuccess("/chat")

    const state = storageMap.get(STORAGE_KEY) as Record<string, any>
    expect(state.counters.onboarding_chat_submit_before_ingest).toBe(1)
    expect(state.counters.onboarding_first_chat_after_ingest).toBe(1)
    expect(state.aggregates.first_chat_after_ingest_conversions).toBe(1)
    expect(state.current_session.first_media_id).toBe("7")
  })

  it("ignores stale ingest runs that predate the current onboarding session", async () => {
    const telemetry = await import("../onboarding-ingestion-telemetry")

    now = 2_000
    await telemetry.trackOnboardingSuccessReached("setup")
    now = 2_120
    await telemetry.trackOnboardingFirstIngestSuccess({
      successCount: 1,
      attemptedAt: 1_500,
      firstMediaId: "legacy"
    })

    const state = storageMap.get(STORAGE_KEY) as Record<string, any>
    expect(state.counters.onboarding_first_ingest_success).toBeUndefined()
    expect(state.aggregates.samples_time_to_first_ingest).toBe(0)
    expect(state.current_session.first_ingest_at).toBeNull()
  })
})

it("does not write a chat conversion after its source retires during the storage read", async () => {
  vi.resetModules()
  let finish!: (value: undefined) => void, current = true
  const write = vi.fn()
  vi.doMock("@/utils/safe-storage", () => ({ createSafeStorage: () => ({
    get: () => new Promise<undefined>(resolve => { finish = resolve }), set: write
  }) }))
  const telemetry = await import("../onboarding-ingestion-telemetry")
  const pending = telemetry.trackOnboardingChatSubmitSuccess("/chat", () => current)
  current = false; finish(undefined); await pending
  expect(write).not.toHaveBeenCalled()
  vi.resetModules()
})
