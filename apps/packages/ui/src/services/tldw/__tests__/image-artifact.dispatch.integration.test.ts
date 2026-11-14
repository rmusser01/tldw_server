import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

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
    remove: vi.fn(),
    watch: vi.fn(),
    unwatch: vi.fn()
  })
}))
vi.mock("@/services/tldw/runtime-auth-override", () => ({
  getRuntimeSingleUserApiKeyOverride: () => null,
  isCookieSessionConfigInvalidated: () => false
}))

import { tldwClient } from "../TldwApiClient"
import { deriveSingleUserApiKeyCredentialScope } from "@/services/chat-surface-scope"

const config = {
  serverUrl: "https://image-dispatch.test",
  authMode: "single-user" as const,
  authSource: "manual" as const,
  credentialSource: "manual" as const,
  apiKeyPersistence: "device" as const,
  apiKeyServerOrigin: "https://image-dispatch.test",
  apiKey: "declared-original-image-key"
}
const scope = {
  config: {
    serverUrl: config.serverUrl,
    authMode: config.authMode,
    authSource: config.authSource,
    expectedSingleUserApiKeyScope: deriveSingleUserApiKeyCredentialScope(
      config.authMode,
      config.apiKey
    )
  },
  userId: null
}
const image = { backend: "comfyui", prompt: "Exact original image prompt" }
const result = {
  artifact: {
    file_id: 51,
    export: { content_b64: "declared-image-data", content_type: "image/png" }
  }
}
const deferred = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((done) => {
    resolve = done
  })
  return { promise, resolve }
}
const start = (signal: AbortSignal, onDispatch: () => void) =>
  tldwClient.createImageArtifact(image, {
    requestScope: scope,
    signal,
    onDispatch
  })

beforeEach(() => {
  vi.clearAllMocks()
  boundary.runtimeId = null
  boundary.get.mockImplementation(async (key: string) =>
    key === "tldwConfig" ? config : null
  )
  boundary.fetch.mockImplementation(async () => Response.json(result))
  boundary.sendMessage.mockResolvedValue({ ok: true, data: result })
  vi.stubGlobal("fetch", boundary.fetch)
})
afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

// Only storage, fetch and extension messages are declared boundaries. The real
// public domain, request allowlist, credential checks and request core execute.
describe.each(["direct", "extension"] as const)(
  "image dispatch through actual %s transport",
  (transport) => {
    beforeEach(() => {
      if (transport === "extension") boundary.runtimeId = "declared-extension"
    })

    it("notifies exactly once at physical handoff without changing or serializing the request", async () => {
      const events: string[] = [],
        dispatched = vi.fn(() => {
          events.push("dispatch")
        })
      boundary.fetch.mockImplementation(async () => {
        events.push("fetch")
        return Response.json(result)
      })
      boundary.sendMessage.mockImplementation(async () => {
        events.push("handoff")
        return { ok: true, data: result }
      })
      expect(await start(new AbortController().signal, dispatched)).toEqual(
        result
      )
      expect(events).toEqual([
        "dispatch",
        transport === "extension" ? "handoff" : "fetch"
      ])
      expect(dispatched).toHaveBeenCalledOnce()
      if (transport === "extension") {
        expect(boundary.fetch).not.toHaveBeenCalled()
        expect(boundary.sendMessage).toHaveBeenCalledOnce()
        const payload = boundary.sendMessage.mock.calls[0][0].payload
        expect(payload.servicePromptConfig).toEqual({
          ...scope.config,
          expectedUserId: null
        })
        expect(payload).not.toHaveProperty("onDispatch")
        expect(payload.body).toMatchObject({
          file_type: "image",
          payload: image
        })
      } else {
        expect(boundary.sendMessage).not.toHaveBeenCalled()
        expect(boundary.fetch).toHaveBeenCalledOnce()
        const [url, init] = boundary.fetch.mock.calls[0]
        expect(url).toBe(config.serverUrl + "/api/v1/files/create")
        expect(new Headers(init.headers).get("X-API-KEY")).toBe(config.apiKey)
        expect(JSON.parse(init.body)).toMatchObject({
          file_type: "image",
          payload: image
        })
      }
    })

    it("refuses a replaced credential before response dispatch or transport handoff", async () => {
      boundary.get.mockImplementation(async (key: string) =>
        key === "tldwConfig"
          ? { ...config, apiKey: "declared-later-key" }
          : null
      )
      const dispatched = vi.fn()
      await expect(
        start(new AbortController().signal, dispatched)
      ).rejects.toMatchObject({ status: 412 })
      expect(dispatched).not.toHaveBeenCalled()
      expect(boundary.fetch).not.toHaveBeenCalled()
      expect(boundary.sendMessage).not.toHaveBeenCalled()
    })

    it("honors Stop during actual scoped configuration without consuming a response", async () => {
      const held = deferred<typeof config>()
      boundary.get.mockImplementation((key: string) =>
        key === "tldwConfig" ? held.promise : Promise.resolve(null)
      )
      const controller = new AbortController(),
        dispatched = vi.fn()
      const pending = start(controller.signal, dispatched).catch(
        (error) => error
      )
      try {
        await vi.waitFor(() =>
          expect(boundary.get).toHaveBeenCalledWith("tldwConfig")
        )
        expect(dispatched).not.toHaveBeenCalled()
        controller.abort()
        held.resolve(config)
        expect(await pending).toMatchObject({ name: "AbortError" })
        expect(dispatched).not.toHaveBeenCalled()
        expect(boundary.fetch).not.toHaveBeenCalled()
        expect(boundary.sendMessage).not.toHaveBeenCalled()
      } finally {
        held.resolve(config)
        await pending
      }
    })
  }
)

it("counts the no-receiver direct fallback as one image dispatch", async () => {
  boundary.runtimeId = "declared-extension"
  boundary.sendMessage.mockRejectedValueOnce(
    new Error("Could not establish connection. Receiving end does not exist.")
  )
  const dispatched = vi.fn()
  expect(await start(new AbortController().signal, dispatched)).toEqual(result)
  expect(dispatched).toHaveBeenCalledOnce()
  expect(boundary.sendMessage).toHaveBeenCalledOnce()
  expect(boundary.fetch).toHaveBeenCalledOnce()
})
