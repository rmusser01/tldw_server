import { afterEach, beforeEach, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  runtimeId: null as string | null,
  sendMessage: vi.fn(),
  initialize: vi.fn(),
  updateChat: vi.fn(),
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
  tldwClient: { initialize: mocks.initialize, updateChat: mocks.updateChat }
}))

beforeEach(() => {
  vi.resetModules()
  mocks.initialize.mockReset().mockResolvedValue(undefined)
  mocks.updateChat.mockReset().mockResolvedValue({ id: "owned-chat", version: 5 })
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


it("carries the captured rating scope to the shared versioned writer without ambient initialization", async () => {
  const { updateChatRating } = await import("../feedback")
  const options = { requestScope: { config: { serverUrl: "https://owned.invalid", authMode: "single-user" as const }, userId: 1 }, scope: { type: "workspace" as const, workspaceId: "W" }, signal: new AbortController().signal }
  await updateChatRating("owned-chat", 5, 4, options)
  expect(mocks.updateChat).toHaveBeenCalledWith("owned-chat", { rating: 5 }, { ...options, expectedVersion: 4 })
  expect(mocks.initialize).not.toHaveBeenCalled()
})

it("refuses an already retired rating before initialization or a write", async () => {
  const { updateChatRating } = await import("../feedback")
  const controller = new AbortController()
  controller.abort()
  await expect(updateChatRating("owned-chat", 5, 4, { signal: controller.signal })).rejects.toThrow()
  expect(mocks.initialize).not.toHaveBeenCalled()
  expect(mocks.updateChat).not.toHaveBeenCalled()
})

it("does not continue an unscoped rating after initialization fails", async () => {
  const { updateChatRating } = await import("../feedback")
  const failure = new Error("Configuration unavailable")
  mocks.initialize.mockRejectedValue(failure)
  await expect(updateChatRating("owned-chat", 5, 4)).rejects.toBe(failure)
  expect(mocks.updateChat).not.toHaveBeenCalled()
})
