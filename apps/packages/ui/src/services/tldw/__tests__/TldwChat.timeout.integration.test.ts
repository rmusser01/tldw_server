import i18n from "i18next"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { chatRagMethods } from "../domains/chat-rag"
import { TldwChatService } from "../TldwChat"
import { buildAssistantErrorContent, decodeChatErrorPayload } from "@/utils/chat-error-message"

vi.mock("wxt/browser", () => ({ browser: { runtime: {} } }))
const storage = vi.hoisted(() => new Map<string, unknown>())
vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({
    get: async (key: string) => storage.get(key),
    set: async (key: string, value: unknown) => { storage.set(key, value) },
    remove: async (key: string) => { storage.delete(key) }
  })
}))
vi.mock("../TldwApiClient", () => ({
  tldwClient: {
    initialize: async () => {},
    getConfig: async () => storage.get("tldwConfig"),
    streamChatCompletion: (...args: Parameters<typeof chatRagMethods.streamChatCompletion>) =>
      chatRagMethods.streamChatCompletion.apply({} as never, args)
  }
}))

const messages = [{ role: "user" as const, content: "Describe the source accurately." }]
const options = { model: "local-model", conversationId: "saved-chat", clientMessageId: "same-user" }

const consume = async (service: TldwChatService, signal?: AbortSignal) => {
  let text = ""
  try {
    for await (const token of service.streamMessage(messages, { ...options, signal })) text += token
    return { text, error: undefined }
  } catch (error) {
    return { text, error }
  }
}

const serveStream = (firstOutputMs?: number, finish = true) => {
  const encoder = new TextEncoder()
  const fetch = vi.fn(async (_url: string, init: RequestInit) => {
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        const heartbeat = setInterval(() => controller.enqueue(encoder.encode(": processing\n\n")), 1_000)
        let output: ReturnType<typeof setTimeout> | undefined
        const cleanup = () => {
          clearInterval(heartbeat)
          clearTimeout(output)
          init.signal?.removeEventListener("abort", abort)
        }
        const abort = () => {
          cleanup()
          controller.error(new DOMException("Aborted", "AbortError"))
        }
        init.signal?.addEventListener("abort", abort, { once: true })
        if (firstOutputMs !== undefined) {
          output = setTimeout(() => {
            controller.enqueue(encoder.encode('data: {"choices":[{"delta":{"content":"Grounded answer"}}]}\n\n'))
            if (finish) {
              controller.enqueue(encoder.encode("data: [DONE]\n\n"))
              cleanup()
              controller.close()
            }
          }, firstOutputMs)
        }
      }
    })
    return new Response(stream, { status: 200, headers: { "content-type": "text/event-stream" } })
  })
  vi.stubGlobal("fetch", fetch)
  return fetch
}

describe("Chat startup policy and diagnostics through browser transport", () => {
  beforeEach(async () => {
    vi.useFakeTimers()
    await i18n.init({ lng: "en", resources: {} })
    storage.clear()
    storage.set("tldwConfig", {
      serverUrl: "http://127.0.0.1:8000", authMode: "single-user",
      apiKey: "synthetic-test-key", credentialSource: "manual",
      apiKeyPersistence: "device", apiKeyServerOrigin: "http://127.0.0.1:8000"
    })
  })
  afterEach(() => { vi.clearAllTimers(); vi.useRealTimers(); vi.unstubAllGlobals() })

  it.each([undefined, 0, -1, Number.NaN, Number.POSITIVE_INFINITY])(
    "allows first visible output after 10 seconds with missing/invalid startup setting %s",
    async chatStartupTimeoutMs => {
      Object.assign(storage.get("tldwConfig")!, { chatStartupTimeoutMs })
      const fetch = serveStream(20_000)
      const result = consume(new TldwChatService())
      await vi.advanceTimersByTimeAsync(20_000)
      expect(await result).toEqual({ text: "Grounded answer", error: undefined })
      expect(fetch).toHaveBeenCalledOnce()
    }
  )

  it("uses the documented 120-second startup deadline without an override", async () => {
    const fetch = serveStream()
    const result = consume(new TldwChatService())
    await vi.advanceTimersByTimeAsync(119_999)
    expect(fetch.mock.calls[0][1].signal?.aborted).toBe(false)
    await vi.advanceTimersByTimeAsync(1)
    const { error } = await result
    expect(error).toMatchObject({ name: "ChatStreamTimeoutError", phase: "startup", timeoutMs: 120_000 })
    expect(decodeChatErrorPayload(buildAssistantErrorContent("", error))?.detail).toMatch(/120.*seconds/)
    expect(fetch).toHaveBeenCalledOnce()
  })

  it("honors a custom startup limit and retains actionable diagnostics through serialization", async () => {
    Object.assign(storage.get("tldwConfig")!, { chatStartupTimeoutMs: 10_000 })
    const fetch = serveStream(20_000)
    const result = consume(new TldwChatService())
    await vi.advanceTimersByTimeAsync(10_000)
    const { error } = await result
    expect(error).toMatchObject({ name: "ChatStreamTimeoutError", phase: "startup", timeoutMs: 10_000 })
    const encoded = buildAssistantErrorContent("", error)
    const restored = JSON.parse(JSON.stringify({ message: encoded }))
    expect(decodeChatErrorPayload(restored.message)).toMatchObject({
      summary: "The model did not start responding in time.",
      hint: expect.stringContaining("Chat startup timeout"),
      detail: expect.stringMatching(/10.*seconds.*before any visible output/)
    })
    expect(fetch).toHaveBeenCalledOnce()
  })

  it("distinguishes a stalled visible answer from startup while keeping the configured idle limit", async () => {
    Object.assign(storage.get("tldwConfig")!, { chatStreamIdleTimeoutMs: 2_000 })
    const fetch = serveStream(1_000, false)
    const result = consume(new TldwChatService())
    await vi.advanceTimersByTimeAsync(3_000)
    const { text, error } = await result
    expect(text).toBe("Grounded answer")
    expect(error).toMatchObject({ name: "ChatStreamTimeoutError", phase: "idle", timeoutMs: 2_000 })
    expect(decodeChatErrorPayload(buildAssistantErrorContent("", error))).toMatchObject({
      summary: "The response stopped progressing.",
      hint: expect.stringContaining("Chat stream idle timeout"),
      detail: expect.stringMatching(/2.*seconds.*after visible output/)
    })
    expect(fetch).toHaveBeenCalledOnce()
  })

  it("keeps an explicit Stop distinct from a startup timeout", async () => {
    const fetch = serveStream()
    const stop = new AbortController()
    const result = consume(new TldwChatService(), stop.signal)
    await vi.advanceTimersByTimeAsync(5_000)
    stop.abort()
    expect((await result).error).toMatchObject({ name: "AbortError" })
    expect(fetch).toHaveBeenCalledOnce()
  })
})
