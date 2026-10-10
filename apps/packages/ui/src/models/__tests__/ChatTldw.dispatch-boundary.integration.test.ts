import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const boundary = vi.hoisted(() => ({
  get: vi.fn(),
  fetch: vi.fn(),
  sendMessage: vi.fn(),
  connect: vi.fn(),
  runtimeId: null as string | null
}))
vi.mock("wxt/browser", () => ({
  browser: {
    runtime: {
      get id() {
        return boundary.runtimeId
      },
      sendMessage: (...args: unknown[]) => boundary.sendMessage(...args),
      connect: (...args: unknown[]) => boundary.connect(...args)
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

import { ChatTldw } from "../ChatTldw"
import { HumanMessage } from "@/types/messages"
import { tldwClient, type TldwConfig } from "@/services/tldw/TldwApiClient"
import type { ServicePromptRequestScope } from "@/services/tldw/domains/service-prompts"
import { deriveSingleUserApiKeyCredentialScope } from "@/services/chat-surface-scope"

const config = (user = 7) => ({
  serverUrl: "https://dispatch.example",
  authMode: "multi-user" as const,
  accessToken: `fixture.${btoa(JSON.stringify({ sub: String(user) }))}.signature`
})
const requestScope = {
  config: {
    serverUrl: "https://dispatch.example",
    authMode: "multi-user" as const
  },
  userId: 7
}
const deferred = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((done) => {
    resolve = done
  })
  return { promise, resolve }
}
const response = () =>
  new Response(
    'data: {"choices":[{"delta":{"content":"Answer"}}]}\n\ndata: [DONE]\n\n',
    { headers: { "Content-Type": "text/event-stream" } }
  )
const start = async (
  signal: AbortSignal,
  onDispatch: () => void,
  originalScope: ServicePromptRequestScope = requestScope
) => {
  const model = new ChatTldw({ model: "test", requestScope: originalScope })
  const stream = await model.stream(
    [new HumanMessage("Exact original question")],
    { signal, onDispatch }
  )
  let answer = ""
  for await (const token of stream) answer += token
  return answer
}

beforeEach(() => {
  vi.clearAllMocks()
  boundary.runtimeId = null
  boundary.get.mockImplementation(async (key: string) =>
    key === "tldwConfig" ? config() : null
  )
  boundary.fetch.mockImplementation(async () => response())
  boundary.sendMessage.mockResolvedValue({ ok: true })
  vi.spyOn(tldwClient, "initialize").mockResolvedValue(undefined)
  vi.spyOn(tldwClient, "getConfig").mockResolvedValue(config())
  vi.stubGlobal("fetch", boundary.fetch)
})
afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe("one-response consumption at the actual chat transport boundary", () => {
  it("retains steering when actual service initialization fails before dispatch", async () => {
    vi.mocked(tldwClient.initialize).mockRejectedValueOnce(
      new Error("Initialization failed")
    )
    const consumed = vi.fn()
    await expect(start(new AbortController().signal, consumed)).rejects.toThrow(
      "Stream completion failed"
    )
    expect(consumed).not.toHaveBeenCalled()
    expect(boundary.fetch).not.toHaveBeenCalled()
  })
  it.each(["initialize", "config"] as const)(
    "retains steering after Stop during held service %s",
    async (phase) => {
      const held = deferred<unknown>()
      if (phase === "initialize")
        vi.mocked(tldwClient.initialize).mockImplementationOnce(async () => {
          await held.promise
        })
      else
        vi.mocked(tldwClient.getConfig).mockImplementationOnce(async () => {
          await held.promise
          return config()
        })
      const controller = new AbortController(),
        consumed = vi.fn()
      const pending = start(controller.signal, consumed).catch((error) => error)
      await vi.waitFor(() =>
        expect(
          phase === "initialize" ? tldwClient.initialize : tldwClient.getConfig
        ).toHaveBeenCalled()
      )
      const consumedBeforeStop = consumed.mock.calls.length
      controller.abort()
      held.resolve(undefined)
      await pending
      expect(consumedBeforeStop).toBe(0)
      expect(consumed).not.toHaveBeenCalled()
      expect(boundary.fetch).not.toHaveBeenCalled()
    }
  )
  it("retains steering after actual captured-account transport refusal", async () => {
    boundary.get.mockImplementation(async (key: string) =>
      key === "tldwConfig" ? config(8) : null
    )
    const consumed = vi.fn()
    await expect(
      start(new AbortController().signal, consumed)
    ).rejects.toMatchObject({ status: 412 })
    expect(consumed).not.toHaveBeenCalled()
    expect(boundary.fetch).not.toHaveBeenCalled()
  })
  it("retains steering after Stop during held direct transport configuration", async () => {
    const held = deferred<ReturnType<typeof config>>()
    boundary.get.mockImplementation((key: string) =>
      key === "tldwConfig" ? held.promise : Promise.resolve(null)
    )
    const controller = new AbortController(),
      consumed = vi.fn()
    const pending = start(controller.signal, consumed).catch((error) => error)
    await vi.waitFor(() =>
      expect(boundary.get).toHaveBeenCalledWith("tldwConfig")
    )
    const consumedBeforeStop = consumed.mock.calls.length
    controller.abort()
    held.resolve(config())
    await pending
    expect(consumedBeforeStop).toBe(0)
    expect(consumed).not.toHaveBeenCalled()
    expect(boundary.fetch).not.toHaveBeenCalled()
  })
  it("consumes exactly once at direct fetch after original scope and input validation", async () => {
    const events: string[] = [],
      consumed = vi.fn(() => {
        events.push("dispatch")
      })
    boundary.fetch.mockImplementation(async () => {
      events.push("fetch")
      return response()
    })
    await expect(start(new AbortController().signal, consumed)).resolves.toBe(
      "Answer"
    )
    expect(events).toEqual(["dispatch", "fetch"])
    expect(consumed).toHaveBeenCalledOnce()
    expect(boundary.fetch).toHaveBeenCalledOnce()
    const [url, init] = boundary.fetch.mock.calls[0]
    expect(url).toBe("https://dispatch.example/api/v1/chat/completions")
    expect(JSON.parse(init.body).messages).toEqual([
      { role: "user", content: "Exact original question" }
    ])
    expect(new Headers(init.headers).get("X-TLDW-Expected-User-ID")).toBe("7")
  })
  it("retains steering after Stop during a held extension health check", async () => {
    boundary.runtimeId = "extension-fixture"
    const held = deferred<{ ok: boolean }>()
    boundary.sendMessage.mockReturnValueOnce(held.promise)
    const controller = new AbortController(),
      consumed = vi.fn()
    const pending = start(controller.signal, consumed).catch((error) => error)
    await vi.waitFor(() =>
      expect(boundary.sendMessage).toHaveBeenCalledWith({ type: "tldw:ping" })
    )
    const consumedBeforeStop = consumed.mock.calls.length
    controller.abort()
    held.resolve({ ok: true })
    await pending
    expect(consumedBeforeStop).toBe(0)
    expect(consumed).not.toHaveBeenCalled()
    expect(boundary.fetch).not.toHaveBeenCalled()
  })
  it("consumes exactly once at extension handoff without serializing callbacks", async () => {
    boundary.runtimeId = "extension-fixture"
    const listeners: Array<(value: unknown) => void> = [],
      events: string[] = []
    const postMessage = vi.fn((_payload: unknown) => {
      events.push("post")
      queueMicrotask(() => {
        for (const listener of listeners) {
          listener({
            event: "data",
            data: '{"choices":[{"delta":{"content":"Answer"}}]}'
          })
          listener({ event: "done" })
        }
      })
    })
    boundary.connect.mockReturnValue({
      postMessage,
      disconnect: vi.fn(),
      onMessage: {
        addListener: (listener: (value: unknown) => void) =>
          listeners.push(listener),
        removeListener: vi.fn()
      },
      onDisconnect: { addListener: vi.fn(), removeListener: vi.fn() }
    })
    const consumed = vi.fn(() => {
      events.push("dispatch")
    })
    await expect(start(new AbortController().signal, consumed)).resolves.toBe(
      "Answer"
    )
    expect(events).toEqual(["dispatch", "post"])
    expect(consumed).toHaveBeenCalledOnce()
    const payload = postMessage.mock.calls[0][0] as Record<string, unknown>
    expect(payload.servicePromptConfig).toEqual({
      ...requestScope.config,
      expectedUserId: 7
    })
    expect(payload).not.toHaveProperty("onDispatch")
    expect(payload).not.toHaveProperty("onOpen")
    expect(boundary.fetch).not.toHaveBeenCalled()
  })
  it.each([
    "user",
    "server",
    "refresh credential",
    "single-user credential"
  ] as const)(
    "retains steering when the original %s retires during extension configuration",
    async (change) => {
      boundary.runtimeId = "extension-fixture"
      const originalConfig: TldwConfig =
        change === "single-user credential"
          ? {
              serverUrl: requestScope.config.serverUrl,
              authMode: "single-user",
              authSource: "manual",
              credentialSource: "manual",
              apiKeyPersistence: "device",
              apiKeyServerOrigin: requestScope.config.serverUrl,
              apiKey: "original-dispatch-fixture-key"
            }
          : { ...config(), refreshToken: "original-dispatch-fixture-refresh" }
      const originalScope: ServicePromptRequestScope = {
        config: {
          serverUrl: originalConfig.serverUrl,
          authMode: originalConfig.authMode,
          ...(originalConfig.authSource
            ? { authSource: originalConfig.authSource }
            : {}),
          ...(originalConfig.authMode === "single-user"
            ? {
                expectedSingleUserApiKeyScope:
                  deriveSingleUserApiKeyCredentialScope(
                    originalConfig.authMode,
                    originalConfig.apiKey
                  )!
              }
            : { expectedRefreshToken: originalConfig.refreshToken })
        },
        userId: originalConfig.authMode === "single-user" ? null : 7
      }
      const retiredConfig: TldwConfig =
        change === "user"
          ? { ...originalConfig, accessToken: config(8).accessToken }
          : change === "server"
            ? { ...originalConfig, serverUrl: "https://later-dispatch.example" }
            : change === "refresh credential"
              ? {
                  ...originalConfig,
                  refreshToken: "later-dispatch-fixture-refresh"
                }
              : { ...originalConfig, apiKey: "later-dispatch-fixture-key" }
      let retired = false
      const postMessage = vi.fn()
      // The port is the declared worker I/O boundary: record every handoff and
      // report the refusal the worker would return after the original scope retires.
      // The actual model, active API domain and proxy guards remain unmocked.
      boundary.connect.mockImplementation(() => {
        const listeners = new Set<(value: unknown) => void>()
        return {
          postMessage: (payload: unknown) => {
            postMessage(payload)
            const frames = retired
              ? [
                  {
                    event: "error",
                    status: 412,
                    message:
                      "The server or authenticated account changed before the request was sent.",
                    details: {
                      detail: { code: "request_config_scope_changed" }
                    }
                  }
                ]
              : [
                  {
                    event: "data",
                    data: '{"choices":[{"delta":{"content":"Answer"}}]}'
                  },
                  { event: "done" }
                ]
            queueMicrotask(() => {
              for (const listener of listeners)
                for (const frame of frames) listener(frame)
            })
          },
          disconnect: vi.fn(),
          onMessage: {
            addListener: (listener: (value: unknown) => void) =>
              listeners.add(listener),
            removeListener: (listener: (value: unknown) => void) =>
              listeners.delete(listener)
          },
          onDisconnect: { addListener: vi.fn(), removeListener: vi.fn() }
        }
      })
      boundary.get.mockImplementation(async (key: string) =>
        key === "tldwConfig" ? originalConfig : null
      )
      const consumed = vi.fn()
      // Qualify that this captured credential and principal are initially accepted.
      await expect(
        start(new AbortController().signal, consumed, originalScope)
      ).resolves.toBe("Answer")
      expect(postMessage).toHaveBeenCalledOnce()
      expect(postMessage.mock.calls[0][0].servicePromptConfig).toEqual({
        ...originalScope.config,
        expectedUserId: originalScope.userId
      })
      consumed.mockClear()
      postMessage.mockClear()
      boundary.get.mockClear()
      const held = deferred<TldwConfig>()
      boundary.get.mockImplementation((key: string) =>
        key === "tldwConfig" ? held.promise : Promise.resolve(null)
      )
      const pending = start(
        new AbortController().signal,
        consumed,
        originalScope
      ).catch((error) => error)
      await vi.waitFor(() =>
        expect(boundary.get).toHaveBeenCalledWith("tldwConfig")
      )
      expect(consumed).not.toHaveBeenCalled()
      expect(postMessage).not.toHaveBeenCalled()
      retired = true
      held.resolve(retiredConfig)
      await expect(pending).resolves.toMatchObject({ status: 412 })
      expect(consumed).not.toHaveBeenCalled()
      expect(postMessage).not.toHaveBeenCalled()
      expect(boundary.fetch).not.toHaveBeenCalled()
    }
  )
})

describe("TASK-13274 reasoning through the actual model transport", () => {
  it.each(["direct", "extension"] as const)(
    "publishes %s reasoning before a held answer without duplicating mixed-frame text",
    async (transport) => {
      const { consumeStreamingChunk } = await import("@/utils/streaming-chunks")
      const heldAnswer = deferred<void>()
      const initial = [
        { choices: [{ delta: { reasoning_content: "First thought" } }] },
        { choices: [{ delta: { role: "assistant" } }] },
        { choices: [], usage: { completion_tokens: 1 } }
      ]
      const final = {
        choices: [
          { delta: { reasoning_content: " then more", content: "Answer" } }
        ]
      }
      let physicalSse: ReadableStreamDefaultController<Uint8Array> | undefined
      if (transport === "direct") {
        boundary.fetch.mockImplementation(
          async () =>
            new Response(
              new ReadableStream<Uint8Array>({
                start(controller) {
                  physicalSse = controller
                  for (const chunk of initial)
                    controller.enqueue(
                      new TextEncoder().encode(
                        "data: " + JSON.stringify(chunk) + "\n\n"
                      )
                    )
                  void heldAnswer.promise.then(() => {
                    controller.enqueue(
                      new TextEncoder().encode(
                        "data: " +
                          JSON.stringify(final) +
                          "\n\ndata: [DONE]\n\n"
                      )
                    )
                    controller.close()
                  })
                }
              }),
              { headers: { "Content-Type": "text/event-stream" } }
            )
        )
      } else {
        boundary.runtimeId = "extension-fixture"
        const listeners = new Set<(message: unknown) => void>()
        boundary.connect.mockReturnValue({
          postMessage: vi.fn(() => {
            const emit = (chunk: unknown) => {
              for (const listener of listeners) listener(chunk)
            }
            queueMicrotask(() => {
              for (const chunk of initial)
                emit({ event: "data", data: JSON.stringify(chunk) })
              void heldAnswer.promise.then(() => {
                emit({ event: "data", data: JSON.stringify(final) })
                emit({ event: "done" })
              })
            })
          }),
          disconnect: vi.fn(),
          onMessage: {
            addListener: (listener: (message: unknown) => void) =>
              listeners.add(listener),
            removeListener: (listener: (message: unknown) => void) =>
              listeners.delete(listener)
          },
          onDisconnect: { addListener: vi.fn(), removeListener: vi.fn() }
        })
      }
      const dispatched = vi.fn(),
        ended = vi.fn()
      const model = new ChatTldw({
        model: "test",
        saveToDb: false,
        requestScope
      })
      const stream = await model.stream(
        [new HumanMessage("Exact original question")],
        {
          signal: new AbortController().signal,
          onDispatch: dispatched,
          callbacks: [{ handleLLMEnd: ended }]
        }
      )
      let accumulated = { fullText: "", contentToSave: "", apiReasoning: false }
      const chunks: unknown[] = []
      const pending = (async () => {
        for await (const chunk of stream) {
          chunks.push(chunk)
          accumulated = consumeStreamingChunk(accumulated, chunk)
        }
      })()
      try {
        await vi.waitFor(() => expect(dispatched).toHaveBeenCalledOnce())
        if (transport === "direct") expect(physicalSse).toBeDefined()
        await vi.waitFor(() =>
          expect(accumulated.fullText).toBe("<think>First thought")
        )
        expect(accumulated.apiReasoning).toBe(true)
        expect(chunks).toHaveLength(1)
        heldAnswer.resolve(undefined)
        await pending
        expect(accumulated.fullText).toBe(
          "<think>First thought then more</think>Answer"
        )
        expect(accumulated.contentToSave).toBe(accumulated.fullText)
        expect(chunks).toHaveLength(2)
        expect(ended).toHaveBeenCalledWith(
          expect.objectContaining({
            generations: [[{ text: "Answer", generationInfo: undefined }]]
          })
        )
        expect(dispatched).toHaveBeenCalledOnce()
      } finally {
        heldAnswer.resolve(undefined)
        await pending
      }
    }
  )

  it("preserves the direct service's default string-only contract for sibling consumers", async () => {
    const { tldwChat } = await import("@/services/tldw/TldwChat")
    boundary.fetch.mockResolvedValue(
      new Response(
        'data: {"choices":[{"delta":{"reasoning_content":"Private thought"}}]}\n\ndata: {"choices":[{"delta":{"reasoning_content":" more","content":"Answer"}}]}\n\ndata: [DONE]\n\n',
        { headers: { "Content-Type": "text/event-stream" } }
      )
    )
    const chunks: string[] = []
    for await (const token of tldwChat.streamMessage(
      [{ role: "user", content: "Question" }],
      { model: "test", requestScope }
    ))
      chunks.push(token)
    expect(chunks).toEqual(["Answer"])
  })
})
