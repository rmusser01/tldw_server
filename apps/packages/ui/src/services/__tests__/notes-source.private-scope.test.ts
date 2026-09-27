import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
const boundary = vi.hoisted(() => ({ get: vi.fn(), fetch: vi.fn() }))
vi.mock("wxt/browser", () => ({ browser: { runtime: { id: null } } }))
vi.mock("@/utils/safe-storage", () => ({ createSafeStorage: () => ({ get: boundary.get, set: vi.fn(), remove: vi.fn() }) }))
vi.mock("@/services/tldw/runtime-auth-override", () => ({ getRuntimeSingleUserApiKeyOverride: () => null, isCookieSessionConfigInvalidated: () => false }))
import { listNoteTasks, listTaskActivity } from "../notes-tasks"
import { isServicePromptRequestPath } from "../tldw/service-prompt-scope-error"

const config = (user = 1, serverUrl = "https://source.test") => ({ serverUrl, authMode: "multi-user", accessToken: `test.${btoa(JSON.stringify({ sub: String(user) }))}.signature` })
const options = () => ({ servicePromptConfig: { serverUrl: "https://source.test", authMode: "multi-user" as const, expectedUserId: 1 }, headers: { "X-TLDW-Expected-User-ID": "1" }, abortSignal: new AbortController().signal })
const operations = [
  ["tasks", () => listNoteTasks("owned-note", { limit: 500 }, options())],
  ["activity", () => listTaskActivity({ note_id: "owned-note", limit: 50 }, options())]
] as const

describe("linked Note task reads retain source ownership through real transport", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    boundary.get.mockImplementation(async (key: string) => key === "tldwConfig" ? config() : null)
    boundary.fetch.mockImplementation(async () => new Response("{}", { status: 200, headers: { "Content-Type": "application/json" } }))
    vi.stubGlobal("fetch", boundary.fetch)
  })
  afterEach(() => vi.unstubAllGlobals())
  it.each(operations)("sends %s with the captured owner and server", async (_name, run) => {
    await run()
    expect(boundary.fetch).toHaveBeenCalledTimes(1)
    const [url, init] = boundary.fetch.mock.calls[0]
    expect(new URL(url).origin).toBe("https://source.test")
    expect(new Headers(init.headers).get("X-TLDW-Expected-User-ID")).toBe("1")
  })
  it.each(operations)("rejects %s before dispatch under another account", async (_name, run) => {
    boundary.get.mockImplementation(async (key: string) => key === "tldwConfig" ? config(2) : null)
    await expect(run()).rejects.toMatchObject({ status: 412 })
    expect(boundary.fetch).not.toHaveBeenCalled()
  })
  it.each(operations)("rejects %s before dispatch to another server", async (_name, run) => {
    boundary.get.mockImplementation(async (key: string) => key === "tldwConfig" ? config(1, "https://other.test") : null)
    await expect(run()).rejects.toMatchObject({ status: 412 })
    expect(boundary.fetch).not.toHaveBeenCalled()
  })
  it.each([
    ["POST", "/api/v1/notes/owned-note/tasks"],
    ["POST", "/api/v1/notes/tasks/activity"],
    ["GET", "/api/v1/notes/owned-note/tasks/reconcile"],
    ["GET", "/api/v1/notes//tasks"],
    ["GET", "/api/v1/notes/a%2fb/tasks"],
    ["GET", "/api/v1/notes/../tasks/activity"],
    ["GET", "/api/v1/notes/tasks/activity/nested"],
    ["GET", "https://other.test/api/v1/notes/owned-note/tasks"]
  ])("keeps the new read allowlist bounded for %s %s", (method, path) => {
    expect(isServicePromptRequestPath(path, method)).toBe(false)
  })
})
