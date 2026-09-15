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

  it.each([
    ["fetchWithAuth", ["/api/v1/chat/conversations/owned/messages-with-context"], "GET"],
    ["listCharacters", [{ limit: 5 }], "GET"],
    ["searchCharacters", ["Helpful AI Assistant", { limit: 5 }], "GET"],
    ["getChat", ["owned"], "GET"],
    ["updateChat", ["owned", { title: "Own renamed title" }], "PUT"],
    ["deleteChat", ["owned"], "DELETE"],
    ["createConversationShareLink", ["owned", { ttl_seconds: 300 }], "POST"],
    ["revokeConversationShareLink", ["owned", "share"], "DELETE"],
    ["exportChatbook", [{ name: "Own export", description: "Own", content_selections: { chats: ["owned"] } }], "POST"],
    ["downloadChatbookExport", ["job"], "GET"],
    ["createNote", ["Own answer", { title: "Own note" }], "POST"],
    ["ragSourceHealth", [], "GET"],
  ])("binds the QA %s request without changing its body", async (method, args, requestMethod) => {
    const client = new TldwApiClient()
    vi.spyOn(client, "ensureConfigForRequest").mockResolvedValue({ ...requestScope.config, accessToken: "token" })
    vi.spyOn(client, "resolveApiPath").mockImplementation(async (_key, paths) => paths[0])
    mocks.bgRequest.mockResolvedValue({ ok: true, data: new ArrayBuffer(0), id: "owned" })
    const signal = new AbortController().signal
    await (client[method as keyof typeof client] as (...args: unknown[]) => Promise<unknown>)(...args, { requestScope, signal })
    const outbound = mocks.bgRequest.mock.calls.at(-1)?.[0]
    expect(outbound).toMatchObject({
      method: requestMethod, abortSignal: signal,
      headers: { "X-TLDW-Expected-User-ID": "42" },
      servicePromptConfig: { ...requestScope.config, expectedUserId: 42 },
    })
    expect(outbound.body ?? {}).not.toHaveProperty("requestScope")
  })

  it("binds QA streaming scope outside the inference body", async () => {
    mocks.bgStream.mockImplementation(async function* () { yield '{"type":"contexts","contexts":[]}' })
    const signal = new AbortController().signal
    for await (const _chunk of new TldwApiClient().ragSearchStream("Own question", { requestScope, signal })) { /* consume */ }
    const outbound = mocks.bgStream.mock.calls[0][0]
    expect(outbound).toMatchObject({
      path: "/api/v1/rag/search/stream", abortSignal: signal,
      ...expectedScopeFields,
    })
    expect(outbound.body).not.toHaveProperty("requestScope")
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

  it("reads fresh promotion messages with captured scope without joining cached or pending reads", async () => {
    const client = new TldwApiClient()
    mocks.bgRequest.mockResolvedValueOnce([{ id: "cached", sender: "user", content: "old" }])
    await client.listChatMessages("chat-1")
    const controller = new AbortController()
    let finishPending!: (rows: unknown[]) => void
    mocks.bgRequest.mockImplementationOnce(() => new Promise(resolve => { finishPending = resolve }))
    const first = client.listChatMessages("chat-1", undefined, {
      fresh: true, requestScope, signal: controller.signal
    })
    mocks.bgRequest.mockResolvedValueOnce([{ id: "fresh", sender: "user", content: "new" }])
    const second = await client.listChatMessages("chat-1", undefined, {
      fresh: true, requestScope: { ...requestScope, userId: 43 }, signal: controller.signal
    })
    expect(mocks.bgRequest).toHaveBeenCalledTimes(3)
    expect(second[0].id).toBe("fresh")
    expect(mocks.bgRequest.mock.calls[2][0]).toMatchObject({
      method: "GET", abortSignal: controller.signal,
      headers: { "X-TLDW-Expected-User-ID": "43" },
      servicePromptConfig: { ...requestScope.config, expectedUserId: 43 }
    })
    finishPending([{ id: "late", sender: "user", content: "late" }])
    await first
    // Scoped reads must not poison the ordinary shared cache either.
    expect((await client.listChatMessages("chat-1"))[0].id).toBe("cached")
  })

  it("serializes raw recovery listing as explicit false while keeping the captured owner", async () => {
    const client = new TldwApiClient()
    mocks.bgRequest.mockResolvedValueOnce([])
    await client.listChatMessages("chat-1", { render_placeholders: false, limit: 200, offset: 0 }, {
      fresh: true, requestScope
    })
    expect(mocks.bgRequest.mock.calls[0][0]).toMatchObject({
      path: "/api/v1/chats/chat-1/messages?scope_type=global&render_placeholders=false&limit=200&offset=0",
      headers: { "X-TLDW-Expected-User-ID": "42" }
    })
  })
})
