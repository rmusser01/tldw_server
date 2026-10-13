/**
 * CM-N1 (#3107) on the extension transport: TldwChat streams through the
 * background port, and the background's byte-level idle timer is armed from the
 * `streamIdleTimeoutMs` the page posts. The fake background below applies that
 * value as entries/background.ts does (armed at "open", reset on each frame,
 * "Stream timeout: no updates received" when it fires), so a slow first token
 * completes only if the page hands over the startup budget.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { chatRagMethods } from "../domains/chat-rag"
import { TldwChatService } from "../TldwChat"

type PortListener = (msg: Record<string, unknown>) => void

const runtime = vi.hoisted(() => ({ connect: vi.fn(), sendMessage: vi.fn() }))
vi.mock("wxt/browser", () => ({
  browser: {
    runtime: {
      id: "test-extension",
      connect: (...args: unknown[]) => runtime.connect(...args),
      sendMessage: (...args: unknown[]) => runtime.sendMessage(...args)
    }
  }
}))
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

/** A background that opens at once, stays silent, and answers after `firstOutputMs`. */
const connectBackground = (firstOutputMs: number) => {
  const listeners = new Set<PortListener>()
  const emit = (msg: Record<string, unknown>) => listeners.forEach((listener) => listener(msg))
  const postMessage = vi.fn((payload: { streamIdleTimeoutMs?: number }) => {
    const idleMs = Number(payload.streamIdleTimeoutMs)
    emit({ event: "open" })
    const idle = setTimeout(() => {
      clearTimeout(output)
      emit({ event: "error", message: "Stream timeout: no updates received" })
    }, idleMs)
    const output = setTimeout(() => {
      clearTimeout(idle)
      emit({ event: "data", data: '{"choices":[{"delta":{"content":"Grounded answer"}}]}' })
      emit({ event: "done" })
    }, firstOutputMs)
  })
  runtime.connect.mockReturnValue({
    onMessage: {
      addListener: (listener: PortListener) => listeners.add(listener),
      removeListener: (listener: PortListener) => listeners.delete(listener)
    },
    onDisconnect: { addListener: vi.fn(), removeListener: vi.fn() },
    postMessage,
    disconnect: vi.fn()
  })
  return postMessage
}

describe("CM-N1 (#3107): first-token wait over the extension background port", () => {
  beforeEach(() => {
    vi.useFakeTimers()
    runtime.sendMessage.mockResolvedValue({ ok: true })
    storage.clear()
    storage.set("tldwConfig", {
      serverUrl: "http://127.0.0.1:8000", authMode: "single-user",
      apiKey: "synthetic-test-key", credentialSource: "manual",
      apiKeyPersistence: "device", apiKeyServerOrigin: "http://127.0.0.1:8000"
    })
    vi.stubGlobal("fetch", vi.fn(async () => { throw new Error("direct fetch must not be used") }))
  })
  afterEach(() => { vi.clearAllTimers(); vi.useRealTimers(); vi.unstubAllGlobals() })

  it("completes a 60 s first token because the background is given the startup budget", async () => {
    const postMessage = connectBackground(60_000)
    let text = ""
    const result = (async () => {
      for await (const token of new TldwChatService().streamMessage(
        [{ role: "user", content: "Slow question" }],
        { model: "local-model" }
      )) text += token
    })()
    await vi.advanceTimersByTimeAsync(60_100)
    await result
    expect(text).toBe("Grounded answer")
    expect(postMessage).toHaveBeenCalledWith(
      expect.objectContaining({ path: "/api/v1/chat/completions", streamIdleTimeoutMs: 120_000 })
    )
    expect(fetch).not.toHaveBeenCalled()
  })
})
