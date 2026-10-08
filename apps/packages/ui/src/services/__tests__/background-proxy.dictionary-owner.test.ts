import { TldwApiClient } from "@/services/tldw/TldwApiClient"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { bgRequest } from "../background-proxy"
import { requestScopeFields } from "../tldw/domains/service-prompts"

const boundary = vi.hoisted(() => ({
  get: vi.fn(),
  fetch: vi.fn(),
  sendMessage: vi.fn(),
  runtimeId: null as string | null
}))
vi.mock("wxt/browser", () => ({
  browser: {
    runtime: {
      get id() {
        return boundary.runtimeId
      },
      sendMessage: (...args: unknown[]) => boundary.sendMessage(...args)
    }
  }
}))
vi.mock("@/utils/safe-storage", () => ({
  safeStorageSerde: {
    serializer: JSON.stringify,
    deserializer: (value: unknown) => value
  },
  createSafeStorage: () => ({
    get: boundary.get,
    set: vi.fn(),
    remove: vi.fn()
  })
}))
vi.mock("@/services/tldw/runtime-auth-override", () => ({
  getRuntimeSingleUserApiKeyOverride: () => null,
  isCookieSessionConfigInvalidated: () => false
}))

const config = (user = 7, serverUrl = "https://dictionaries.example") => ({
  serverUrl,
  authMode: "multi-user" as const,
  accessToken: `fixture.${btoa(JSON.stringify({ sub: String(user) }))}.signature`
})
const requestScope = {
  config: {
    serverUrl: "https://dictionaries.example",
    authMode: "multi-user" as const
  },
  userId: 7
}
const data = { dictionaries: [{ id: 42, name: "Owned dictionary" }] }
const list = (signal = new AbortController().signal) =>
  new TldwApiClient().listDictionaries(true, true, { requestScope, signal })
beforeEach(() => {
  vi.clearAllMocks()
  boundary.runtimeId = null
  boundary.get.mockImplementation(async (key: string) =>
    key === "tldwConfig" ? config() : null
  )
  boundary.fetch.mockImplementation(
    async () =>
      new Response(JSON.stringify(data), {
        status: 200,
        headers: { "Content-Type": "application/json" }
      })
  )
  boundary.sendMessage.mockResolvedValue({ ok: true, status: 200, data })
  vi.stubGlobal("fetch", boundary.fetch)
})
afterEach(() => vi.unstubAllGlobals())
describe("dictionary catalogue through the actual SDK and proxy", () => {
  it("dispatches the exact GET for the original account", async () => {
    await expect(list()).resolves.toEqual(data)
    const [url, init] = boundary.fetch.mock.calls[0]
    expect(String(url)).toBe(
      "https://dictionaries.example/api/v1/chat/dictionaries?include_inactive=true&include_usage=true"
    )
    expect(new Headers(init.headers).get("X-TLDW-Expected-User-ID")).toBe("7")
    expect(init.method).toBe("GET")
    expect(init.signal).toBeInstanceOf(AbortSignal)
    expect(init.body).toBeUndefined()
  })
  it("carries the captured account into the extension request", async () => {
    boundary.runtimeId = "extension-fixture"
    await expect(list()).resolves.toEqual(data)
    expect(boundary.sendMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "tldw:request",
        payload: expect.objectContaining({
          method: "GET",
          servicePromptConfig: { ...requestScope.config, expectedUserId: 7 }
        })
      })
    )
    expect(boundary.fetch).not.toHaveBeenCalled()
  })
  it.each(["account", "server"])(
    "refuses a retired %s before HTTP",
    async (change) => {
      boundary.get.mockImplementation(async (key: string) =>
        key === "tldwConfig"
          ? change === "account"
            ? config(8)
            : config(7, "https://other.example")
          : null
      )
      await expect(list()).rejects.toMatchObject({ status: 412 })
      expect(boundary.fetch).not.toHaveBeenCalled()
    }
  )
  it("preserves cancellation after dispatch", async () => {
    boundary.fetch.mockImplementation(
      (_url: unknown, init: RequestInit) =>
        new Promise((_resolve, reject) =>
          init.signal?.addEventListener(
            "abort",
            () => reject(new DOMException("Canceled", "AbortError")),
            { once: true }
          )
        )
    )
    const controller = new AbortController(),
      pending = list(controller.signal)
    const rejected = expect(pending).rejects.toMatchObject({
      name: "AbortError"
    })
    await vi.waitFor(() => expect(boundary.fetch).toHaveBeenCalledOnce())
    controller.abort()
    await rejected
  })
  it.each(["POST", "PUT", "PATCH", "DELETE", "HEAD"])(
    "does not expand scoped catalogue permission to %s",
    async (method) => {
      await expect(
        bgRequest({
          ...requestScopeFields(requestScope),
          path: "/api/v1/chat/dictionaries" as never,
          method: method as never
        })
      ).rejects.toThrow(/Service Prompt config/)
      expect(boundary.fetch).not.toHaveBeenCalled()
      expect(boundary.sendMessage).not.toHaveBeenCalled()
    }
  )
  it.each([
    "/api/v1/chat/dictionaries/",
    "/api/v1/chat/dictionaries/42",
    "/api/v1/chat/dictionaries/42/entries",
    "/api/v1/chat/dictionaries-other",
    "/api/v1/chat//dictionaries",
    "/api/v1/chat/../chat/dictionaries",
    "/api/v1/chat/%2e%2e/dictionaries"
  ])("does not expand scoped catalogue permission to %s", async (path) => {
    await expect(
      bgRequest({
        ...requestScopeFields(requestScope),
        path: path as never,
        method: "GET"
      })
    ).rejects.toThrow(/Service Prompt config/)
    expect(boundary.fetch).not.toHaveBeenCalled()
    expect(boundary.sendMessage).not.toHaveBeenCalled()
  })
})
