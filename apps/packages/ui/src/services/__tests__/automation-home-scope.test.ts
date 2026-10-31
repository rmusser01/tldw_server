import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const boundary = vi.hoisted(() => ({ get: vi.fn(), fetch: vi.fn() }))
vi.mock("wxt/browser", () => ({ browser: { runtime: { id: null } } }))
vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({ get: boundary.get, set: vi.fn(), remove: vi.fn() })
}))
vi.mock("@/services/tldw/runtime-auth-override", () => ({
  getRuntimeSingleUserApiKeyOverride: () => null,
  isCookieSessionConfigInvalidated: () => false
}))
import { listScheduledTaskResults, listScheduledTasks, type ScheduledTaskReadOptions } from "../scheduled-tasks-control-plane"
import { listNotifications } from "../notifications"
import { bgRequest } from "../background-proxy"

const config = (user = 7, serverUrl = "https://home.test") => ({
  serverUrl, authMode: "multi-user" as const,
  accessToken: `test.${btoa(JSON.stringify({ sub: String(user) }))}.signature`
})
const options: ScheduledTaskReadOptions = {
  servicePromptConfig: { serverUrl: "https://home.test", authMode: "multi-user", expectedUserId: 7 },
  headers: { "X-TLDW-Expected-User-ID": "7" },
  suppressBackendUnavailableEvent: true
}
const reads = [
  ["tasks", () => listScheduledTasks(options)],
  ["results", () => listScheduledTaskResults({ limit: 50 }, options)],
  ["notifications", () => listNotifications({ limit: 50 }, options)]
] as const

describe("Home automation protected service boundaries", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    boundary.get.mockImplementation(async key => key === "tldwConfig" ? config() : null)
    boundary.fetch.mockResolvedValue(new Response(JSON.stringify({ items: [] }), { headers: { "Content-Type": "application/json" } }))
    vi.stubGlobal("fetch", boundary.fetch)
  })
  afterEach(() => vi.unstubAllGlobals())

  it.each(reads)("rejects %s after account replacement before dispatch", async (_name, read) => {
    boundary.get.mockImplementation(async key => key === "tldwConfig" ? config(8) : null)
    await expect(read()).rejects.toMatchObject({ status: 412 })
    expect(boundary.fetch).not.toHaveBeenCalled()
  })

  it.each(reads)("does not send current credentials to a stale server for %s", async (_name, read) => {
    boundary.get.mockImplementation(async key => key === "tldwConfig" ? config(7, "https://other.test") : null)
    await expect(read()).rejects.toMatchObject({ status: 412 })
    expect(boundary.fetch).not.toHaveBeenCalled()
  })

  it.each(["/api/v1/scheduled-tasks", "/api/v1/scheduled-tasks/results", "/api/v1/notifications"])(
    "does not broaden scoped GET authorization to mutations at %s", async path => {
      await expect(bgRequest({ ...options, path: path as "/api/v1/notifications", method: "POST" as "GET" })).rejects.toThrow(/Service Prompt config/)
      expect(boundary.fetch).not.toHaveBeenCalled()
    }
  )

  it.each(["/api/v1/scheduled-tasks/other", "/api/v1/scheduled-tasks/results/other", "/api/v1/notifications/other"])(
    "keeps expanded source route %s outside the scoped allowlist", async path => {
      await expect(bgRequest({ ...options, path: path as "/api/v1/notifications", method: "GET" })).rejects.toThrow(/Service Prompt config/)
      expect(boundary.fetch).not.toHaveBeenCalled()
    }
  )
})
