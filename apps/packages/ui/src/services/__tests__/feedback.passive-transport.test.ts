import { afterEach, beforeEach, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  runtimeId: null as string | null,
  sendMessage: vi.fn(),
  config: {
    serverUrl: "https://feedback.invalid",
    authMode: "single-user",
    credentialSource: "manual",
    apiKey: "test-feedback-key",
    apiKeyPersistence: "device",
    apiKeyServerOrigin: "https://feedback.invalid"
  }
}))
vi.mock("wxt/browser", () => ({
  browser: {
    runtime: {
      get id() { return mocks.runtimeId },
      sendMessage: (...args: unknown[]) => mocks.sendMessage(...args)
    }
  }
}))
vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({
    get: vi.fn(async (key: string) => key === "tldwConfig" ? mocks.config : null),
    set: vi.fn(async () => undefined),
    remove: vi.fn(async () => undefined)
  })
}))
vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: { initialize: vi.fn(async () => undefined) }
}))

beforeEach(() => {
  vi.resetModules()
  vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")))
  mocks.sendMessage.mockReset().mockResolvedValue({ ok: false, status: 0, error: "Failed to fetch" })
})
afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals() })

it.each(["web", "extension"])("keeps automatic feedback failures out of the global dialog on %s while explicit feedback still reports failure", async (surface) => {
  mocks.runtimeId = surface === "extension" ? "test-extension" : null
  const { submitImplicitFeedback, submitExplicitFeedback } = await import("../feedback")
  const notify = vi.fn()
  window.addEventListener("tldw:backend-unreachable", notify)
  try {
    await expect(submitImplicitFeedback({
      event_type: "dwell_time",
      conversation_id: "saved-chat",
      message_id: "saved-answer",
      dwell_ms: 3000
    })).rejects.toMatchObject({ status: 0 })
    expect(notify).not.toHaveBeenCalled()

    await expect(submitExplicitFeedback({
      feedback_type: "helpful",
      helpful: true,
      conversation_id: "saved-chat",
      message_id: "saved-answer"
    })).rejects.toMatchObject({ status: 0 })
    expect(notify).toHaveBeenCalledOnce()
    expect(notify.mock.calls[0][0].detail).toMatchObject({
      method: "POST",
      path: "/api/v1/feedback/explicit",
      status: 0
    })
  } finally {
    window.removeEventListener("tldw:backend-unreachable", notify)
  }
})
