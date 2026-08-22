import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn(),
  bgStream: vi.fn()
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.bgRequest(...args),
  bgStream: (...args: unknown[]) => mocks.bgStream(...args),
  bgUpload: vi.fn()
}))

vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({
    get: vi.fn(async () => null),
    set: vi.fn(async () => undefined),
    remove: vi.fn(async () => undefined)
  }),
  safeStorageSerde: {
    serialize: (value: unknown) => value,
    deserialize: (value: unknown) => value
  }
}))

import {
  TldwApiClient,
  TldwApiClientBase
} from "@/services/tldw/TldwApiClient"

const requestScope = {
  config: {
    serverUrl: "https://research-one.test",
    authMode: "multi-user" as const
  },
  userId: 42
}

const expectedScopeFields = {
  headers: {
    "Content-Type": "application/json",
    "X-TLDW-Expected-User-ID": "42"
  },
  servicePromptConfig: {
    ...requestScope.config,
    expectedUserId: requestScope.userId
  }
}

describe("TldwApiClient captured request scope", () => {
  beforeEach(() => {
    mocks.bgRequest.mockReset()
    mocks.bgStream.mockReset()
  })

  it("binds non-streaming chat without serializing the scope", async () => {
    mocks.bgRequest.mockResolvedValueOnce({ choices: [] })
    const client = new TldwApiClient()
    const body = {
      model: "gpt-test",
      messages: [{ role: "user" as const, content: "hello" }]
    }

    await client.createChatCompletion(body, { requestScope })

    expect(mocks.bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/chat/completions",
      method: "POST",
      body,
      timeoutMs: undefined,
      abortSignal: undefined,
      ...expectedScopeFields
    })
    expect(mocks.bgRequest.mock.calls[0]?.[0].body).not.toHaveProperty(
      "requestScope"
    )
  })

  it("binds streaming chat without serializing the scope", async () => {
    mocks.bgStream.mockImplementation(async function* () {
      yield '{"choices":[]}'
    })
    const client = new TldwApiClient()
    const body = {
      model: "gpt-test",
      messages: [{ role: "user" as const, content: "hello" }]
    }

    for await (const _chunk of client.streamChatCompletion(
      body,
      { requestScope }
    )) {
      // consume the stream
    }

    expect(mocks.bgStream).toHaveBeenCalledWith({
      path: "/api/v1/chat/completions",
      method: "POST",
      body: { ...body, stream: true },
      abortSignal: undefined,
      streamIdleTimeoutMs: undefined,
      ...expectedScopeFields
    })
    expect(mocks.bgStream.mock.calls[0]?.[0].body).not.toHaveProperty(
      "requestScope"
    )
  })

  it("does not replay scoped RAG POSTs and keeps scope out of JSON", async () => {
    mocks.bgRequest.mockRejectedValueOnce(
      Object.assign(new Error("reranker unavailable"), { status: 500 })
    )
    const client = new TldwApiClient()

    await expect(
      client.ragSearch("what changed?", {
        enable_reranking: true,
        requestScope
      })
    ).rejects.toBeInstanceOf(Error)

    expect(mocks.bgRequest).toHaveBeenCalledTimes(1)
    const init = mocks.bgRequest.mock.calls[0]?.[0]
    expect(init).toMatchObject({
      ...expectedScopeFields,
      sanitizeRagProviderError: true
    })
    expect(init.body).toMatchObject({
      query: "what changed?",
      enable_reranking: true
    })
    expect(init.body).not.toHaveProperty("requestScope")
  })

  it("preserves structured scope errors in the exported base RAG method", async () => {
    const scopeError = Object.assign(new Error("scope changed"), {
      status: 412,
      details: {
        detail: { code: "request_config_scope_changed" }
      }
    })
    mocks.bgRequest.mockRejectedValueOnce(scopeError)
    const client = new TldwApiClientBase()

    await expect(client.ragSearch("what changed?", {
      requestScope
    })).rejects.toBe(scopeError)
  })

  it("binds web search and keeps request scope out of JSON", async () => {
    mocks.bgRequest.mockResolvedValueOnce({ results: [] })
    const client = new TldwApiClient()

    await client.webSearch({
      query: "current research",
      requestScope
    })

    expect(mocks.bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/research/websearch",
      method: "POST",
      body: { query: "current research" },
      timeoutMs: undefined,
      abortSignal: undefined,
      ...expectedScopeFields
    })
  })

  it("binds mirrored chat messages to the captured account and signal", async () => {
    mocks.bgRequest.mockResolvedValueOnce({ id: "message-1" })
    const controller = new AbortController()
    const client = new TldwApiClient()
    const body = { role: "assistant", content: "scoped answer" }

    await client.addChatMessage("chat-1", body, {
      requestScope,
      signal: controller.signal
    })

    expect(mocks.bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/chats/chat-1/messages?scope_type=global",
      method: "POST",
      body,
      abortSignal: controller.signal,
      ...expectedScopeFields
    })
  })

  it("binds chat-session creation to the captured account and signal", async () => {
    mocks.bgRequest.mockResolvedValueOnce({ id: "chat-1", title: "Scoped chat" })
    const controller = new AbortController()
    const client = new TldwApiClient()
    const body = { title: "Scoped chat" }

    await client.createChat(body, {
      requestScope,
      signal: controller.signal
    })

    expect(mocks.bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/chats/",
      method: "POST",
      body: { ...body, scope_type: "global" },
      abortSignal: controller.signal,
      ...expectedScopeFields
    })
  })
})
