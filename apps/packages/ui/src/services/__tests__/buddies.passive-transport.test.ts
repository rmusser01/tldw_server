import { afterEach, beforeEach, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  runtimeId: null as string | null,
  sendMessage: vi.fn(),
  config: {
    serverUrl: "https://buddy.invalid",
    authMode: "single-user" as const,
    authSource: "manual" as const,
    credentialSource: "manual" as const,
    apiKey: "test-buddy-key",
    apiKeyPersistence: "device" as const,
    apiKeyServerOrigin: "https://buddy.invalid"
  }
}))
vi.mock("wxt/browser", () => ({
  browser: { runtime: {
    get id() { return mocks.runtimeId },
    sendMessage: (...args: unknown[]) => mocks.sendMessage(...args)
  } }
}))
vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({
    get: vi.fn(async (key: string) => key === "tldwConfig" ? mocks.config : null),
    set: vi.fn(async () => undefined),
    remove: vi.fn(async () => undefined)
  }),
  safeStorageSerde: { serialize: (value: unknown) => value, deserialize: (value: unknown) => value }
}))

beforeEach(() => {
  vi.resetModules()
  vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")))
  mocks.sendMessage.mockReset().mockResolvedValue({ ok: false, status: 0, error: "Failed to fetch" })
})
afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals() })

it.each(["web", "extension"])("keeps passive Buddy read failures local on %s while explicit reads and writes still notify", async (surface) => {
  mocks.runtimeId = surface === "extension" ? "test-extension" : null
  const { tldwClient } = await import("@/services/tldw/TldwApiClient")
  vi.spyOn(tldwClient, "ensureConfigForRequest").mockResolvedValue(mocks.config)
  const buddy = await import("../buddies")
  const notify = vi.fn()
  window.addEventListener("tldw:backend-unreachable", notify)
  const quiet = { suppressBackendUnavailableEvent: true }
  try {
    for (const read of [
      () => buddy.getBuddyAttachment(quiet),
      () => buddy.listBuddies(undefined, quiet),
      () => buddy.getBuddy("duck", quiet),
      () => buddy.listBuddyConversations(undefined, quiet),
      () => buddy.listBuddyTurns(undefined, quiet),
      () => buddy.listBuddyActivity(undefined, quiet)
    ]) {
      await expect(read()).rejects.toThrow()
      expect(notify).not.toHaveBeenCalled()
    }
    await expect(buddy.getBuddyAttachment()).rejects.toThrow()
    expect(notify).toHaveBeenCalledOnce()
    expect(notify.mock.calls[0][0].detail).toMatchObject({ method: "GET", status: 0 })
    // Clear the existing event throttle before checking an explicit mutation.
    vi.spyOn(Date, "now").mockReturnValue(Date.now() + 10_000)
    await expect(buddy.detachBuddy(1)).rejects.toThrow()
    expect(notify).toHaveBeenCalledTimes(2)
    expect(notify.mock.calls[1][0].detail).toMatchObject({ method: "DELETE", status: 0 })
  } finally {
    window.removeEventListener("tldw:backend-unreachable", notify)
  }
})
