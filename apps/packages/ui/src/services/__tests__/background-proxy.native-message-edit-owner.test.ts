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

const config = (user = 7, serverUrl = "https://messages.example") => ({
  serverUrl,
  authMode: "multi-user" as const,
  accessToken: `fixture.${btoa(JSON.stringify({ sub: String(user) }))}.signature`
})
const requestScope = {
  config: {
    serverUrl: "https://messages.example",
    authMode: "multi-user" as const
  },
  userId: 7
}
const data = {
  id: "message",
  conversation_id: "chat",
  version: 2,
  content: "Exact {{char}}  \n"
}
const options = (signal = new AbortController().signal) => ({
  requestScope,
  signal,
  scope: { type: "workspace" as const, workspaceId: "original" }
})
const read = (signal?: AbortSignal) =>
  new TldwApiClient().getMessage("message", options(signal))
const save = (signal?: AbortSignal) =>
  new TldwApiClient().editMessage(
    "message",
    data.content,
    1,
    "chat",
    options(signal)
  )
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
describe("native message Save through the actual SDK and proxy", () => {
  it("dispatches the exact GET for the original account", async () => {
    await expect(read()).resolves.toEqual(data)
    const [url, init] = boundary.fetch.mock.calls[0]
    expect(String(url)).toBe(
      "https://messages.example/api/v1/messages/message?scope_type=workspace&workspace_id=original"
    )
    expect(new Headers(init.headers).get("X-TLDW-Expected-User-ID")).toBe("7")
    expect(init.method).toBe("GET")
    expect(init.signal).toBeInstanceOf(AbortSignal)
    expect(init.body).toBeUndefined()
  })
  it("carries the captured account into the extension request", async () => {
    boundary.runtimeId = "extension-fixture"
    await expect(read()).resolves.toEqual(data)
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
      await expect(read()).rejects.toMatchObject({ status: 412 })
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
      pending = read(controller.signal)
    const rejected = expect(pending).rejects.toMatchObject({
      name: "AbortError"
    })
    await vi.waitFor(() => expect(boundary.fetch).toHaveBeenCalledOnce())
    controller.abort()
    await rejected
  })
  it.each(["POST", "PATCH", "DELETE", "HEAD"])(
    "does not expand native Save permission to %s",
    async (method) => {
      await expect(
        bgRequest({
          ...requestScopeFields(requestScope),
          path: "/api/v1/messages/message" as never,
          method: method as never
        })
      ).rejects.toThrow(/Service Prompt config/)
      expect(boundary.fetch).not.toHaveBeenCalled()
      expect(boundary.sendMessage).not.toHaveBeenCalled()
    }
  )
  it.each([
    "/api/v1/messages/",
    "/api/v1/messages/message/",
    "/api/v1/messages/message/extra",
    "/api/v1/messages-other/message",
    "/api/v1/messages//message",
    "/api/v1/messages/../message",
    "/api/v1/messages/owned%2Fforeign"
  ])("does not expand native Save permission to %s", async (path) => {
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

it("dispatches one exact content-only CAS for the original workspace", async () => {
  await expect(save()).resolves.toEqual(data)
  expect(boundary.fetch).toHaveBeenCalledOnce()
  const [url, init] = boundary.fetch.mock.calls[0]
  expect(String(url)).toBe(
    "https://messages.example/api/v1/messages/message?expected_version=1&scope_type=workspace&workspace_id=original"
  )
  expect(init.method).toBe("PUT")
  expect(JSON.parse(init.body)).toEqual({ content: data.content })
  expect(new Headers(init.headers).get("X-TLDW-Expected-User-ID")).toBe("7")
})

it("a retired principal cannot send a native edit to the replacement account", async () => {
  boundary.get.mockImplementation(async (key: string) =>
    key === "tldwConfig" ? config(8) : null
  )
  await expect(save()).rejects.toMatchObject({ status: 412 })
  expect(boundary.fetch).not.toHaveBeenCalled()
})


describe("native leaf Delete through the actual SDK and proxy", () => {
  const remove = (signal?: AbortSignal, id = "message") =>
    new TldwApiClient().deleteMessage(id, 2, "chat", {
      ...options(signal),
      requireLeaf: true,
      expectedRevision: "opaque-original"
    })

  it("dispatches one scoped numeric CAS and accepts only204", async () => {
    boundary.fetch.mockResolvedValue(new Response(null, { status: 204 }))
    await expect(remove()).resolves.toBeUndefined()
    const [url, init] = boundary.fetch.mock.calls[0]
    expect(String(url)).toBe(
      "https://messages.example/api/v1/messages/message?expected_version=2&scope_type=workspace&workspace_id=original&require_leaf=true&expected_revision=opaque-original"
    )
    expect(new Headers(init.headers).get("X-TLDW-Expected-User-ID")).toBe("7")
    expect(init.method).toBe("DELETE")
    expect(init.body).toBeUndefined()
    expect(init.signal).toBeInstanceOf(AbortSignal)
  })
  it("keeps the same account and leaf precondition through the extension", async () => {
    boundary.runtimeId = "extension-fixture"
    boundary.sendMessage.mockResolvedValue({
      ok: true,
      status: 204,
      data: null
    })
    await expect(remove()).resolves.toBeUndefined()
    expect(boundary.sendMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        payload: expect.objectContaining({
          method: "DELETE",
          path: "/api/v1/messages/message?expected_version=2&scope_type=workspace&workspace_id=original&require_leaf=true&expected_revision=opaque-original",
          servicePromptConfig: { ...requestScope.config, expectedUserId: 7 }
        })
      })
    )
    expect(boundary.fetch).not.toHaveBeenCalled()
  })
  it("refuses a replacement account before deletion", async () => {
    boundary.get.mockImplementation(async (key: string) =>
      key === "tldwConfig" ? config(8) : null
    )
    await expect(remove()).rejects.toMatchObject({ status: 412 })
    expect(boundary.fetch).not.toHaveBeenCalled()
  })
  it("encodes a literal identifier instead of changing the request path", async () => {
    boundary.fetch.mockResolvedValue(new Response(null, { status: 204 }))
    await expect(
      remove(undefined, "message?other=value")
    ).resolves.toBeUndefined()
    expect(String(boundary.fetch.mock.calls[0][0])).toContain(
      "/messages/message%3Fother%3Dvalue?expected_version=2&"
    )
  })
  it("rejects an unexpected200 without replay", async () => {
    await expect(remove()).rejects.toThrow("message_delete_unconfirmed")
    expect(boundary.fetch).toHaveBeenCalledOnce()
  })
  it.each([["false"], ["true"], [1], [{}], [[]], [undefined]])(
    "refuses nonboolean runtime acknowledgement %j",
    async (ok) => {
      boundary.runtimeId = "extension-fixture"
      boundary.sendMessage.mockResolvedValue({ ok, status: 204, data: null })
      const api = new TldwApiClient()
      const invalidate = vi.spyOn(api, "invalidateChatMessagesCache")
      await expect(
        api.deleteMessage("message", 2, "chat", {
          ...options(),
          requireLeaf: true,
          expectedRevision: "opaque-original"
        })
      ).rejects.toThrow("message_delete_unconfirmed")
      expect(boundary.sendMessage).toHaveBeenCalledOnce()
      expect(boundary.fetch).not.toHaveBeenCalled()
      expect(invalidate).not.toHaveBeenCalled()
    }
  )
  it("preserves a server409 from the serialized response", async () => {
    boundary.runtimeId = "extension-fixture"
    boundary.sendMessage.mockResolvedValue({
      ok: false,
      status: 409,
      data: { detail: "message_has_descendants" },
      error: "Conflict"
    })
    await expect(remove()).rejects.toMatchObject({ status: 409 })
    expect(boundary.sendMessage).toHaveBeenCalledOnce()
  })
  it("preserves cancellation before dispatch", async () => {
    const controller = new AbortController()
    controller.abort()
    await expect(remove(controller.signal)).rejects.toMatchObject({
      name: "AbortError"
    })
    expect(boundary.fetch).not.toHaveBeenCalled()
  })
  it.each([
    "?expected_version=2",
    "?require_leaf=true",
    "?expected_version=opaque&require_leaf=true",
    "?expected_version=2&require_leaf=false",
    "?expected_version=2&expected_version=3&require_leaf=true",
    "?expected_version=2&require_leaf=true&require_leaf=false"
  ])("refuses an unqualified native DELETE%s", async (query) => {
    await expect(
      bgRequest({
        ...requestScopeFields(requestScope),
        path: `/api/v1/messages/message${query}` as never,
        method: "DELETE"
      })
    ).rejects.toThrow(/Service Prompt config/)
    expect(boundary.fetch).not.toHaveBeenCalled()
  })
})
