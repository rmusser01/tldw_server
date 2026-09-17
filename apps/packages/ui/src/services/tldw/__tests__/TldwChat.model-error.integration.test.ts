import i18n from "i18next"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { chatRagMethods } from "../domains/chat-rag"
import { TldwChatService } from "../TldwChat"
import { createSafeStorage } from "@/utils/safe-storage"
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
    getConfig: async () => null,
    streamChatCompletion: (...args: Parameters<typeof chatRagMethods.streamChatCompletion>) =>
      chatRagMethods.streamChatCompletion.apply({} as never, args)
  }
}))

describe("ordinary chat unavailable-model recovery through browser transport", () => {
  beforeEach(async () => {
    storage.clear()
    await i18n.init({ lng: "en", resources: {} })
    await createSafeStorage().set("tldwConfig", {
      serverUrl: "http://127.0.0.1:8000", authMode: "single-user",
      apiKey: "synthetic-test-key", credentialSource: "manual",
      apiKeyPersistence: "device", apiKeyServerOrigin: "http://127.0.0.1:8000"
    })
  })

  afterEach(() => { vi.unstubAllGlobals() })

  it("renders actionable HTTP400 guidance and preserves request identity when retried", async () => {
    // Keep service wrapping, direct HTTP error parsing, SSE parsing and the
    // assistant-error formatter real. Only the remote server is replaced.
    const requests: Array<{ url: string; body: Record<string, unknown> }> = []
    vi.stubGlobal("fetch", vi.fn(async (url: string, init: RequestInit) => {
      requests.push({ url: String(url), body: JSON.parse(String(init.body)) })
      if (requests.length === 1) return new Response(JSON.stringify({ detail: {
        error_code: "model_not_available",
        message: "Model 'missing-model' is not available for provider 'ollama'. Select one of the server-advertised models for this provider.",
        provider: "ollama", model: "missing-model"
      } }), { status: 400, headers: { "content-type": "application/json" } })
      return new Response('data: {"choices":[{"delta":{"content":"Recovered answer"}}]}\n\ndata: [DONE]\n\n', {
        status: 200, headers: { "content-type": "text/event-stream" }
      })
    }))
    const service = new TldwChatService()
    const messages = [{ role: "user" as const, content: "Exact user question" }]
    const options = { model: "missing-model", apiProvider: "ollama", conversationId: "saved-chat", clientMessageId: "same-user" }
    let error: unknown
    try {
      for await (const _token of service.streamMessage(messages, options)) { /* consume */ }
    } catch (caught) { error = caught }
    expect(error).toMatchObject({ message: "Stream completion failed", cause: { status: 400 } })
    expect(decodeChatErrorPayload(buildAssistantErrorContent("", error))).toMatchObject({
      summary: "The selected model is not available.",
      recoveryAction: "open-model-selector"
    })
    let answer = ""
    for await (const token of service.streamMessage(messages, { ...options, retryFailedTurn: true })) answer += token
    expect(answer).toBe("Recovered answer")
    expect(requests).toHaveLength(2)
    expect(requests[0].url).toBe("http://127.0.0.1:8000/api/v1/chat/completions")
    expect(requests[1].body).toEqual({
      ...requests[0].body,
      metadata: { tldw_client_message_id: "same-user", tldw_retry_failed_turn: true }
    })
    expect(requests[1].body.conversation_id).toBe("saved-chat")
  })
})
