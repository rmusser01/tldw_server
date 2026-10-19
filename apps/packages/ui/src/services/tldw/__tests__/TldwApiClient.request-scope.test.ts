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
import { characterMethods } from "@/services/tldw/domains/characters"
import { chatRagMethods } from "@/services/tldw/domains/chat-rag"

it.each(["stats", "media-budget"])("binds admin %s reads to the verified workspace", async (operation) => {
  const client = new TldwApiClient()
  const options = {
    configSnapshot: { serverUrl: "https://captured.test", authMode: "multi-user" as const },
    abortSignal: new AbortController().signal,
    headers: { "X-TLDW-Expected-User-ID": "42" }
  }
  mocks.bgRequest.mockClear()
  if (operation === "stats") await client.getSystemStats({ ...options, timeoutMs: 10_000 })
  else await client.getMediaIngestionBudgetDiagnostics({ userId: 2 }, options)
  expect(mocks.bgRequest).toHaveBeenCalledWith(expect.objectContaining(options))
})

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
  it("pins dictionary source reads to the captured owner before catalogue display", async () => {
    const client = new TldwApiClient()
    const signal = new AbortController().signal
    mocks.bgRequest.mockResolvedValue({ dictionaries: [] })
    await (client.listDictionaries as unknown as (inactive: boolean, usage: boolean, options: { requestScope: typeof requestScope; signal: AbortSignal }) => Promise<unknown>)(true, true, { requestScope, signal })
    expect(mocks.bgRequest.mock.calls[0][0]).toMatchObject({ path: "/api/v1/chat/dictionaries?include_inactive=true&include_usage=true", method: "GET", abortSignal: signal, headers: { "X-TLDW-Expected-User-ID": "42" }, servicePromptConfig: expectedScopeFields.servicePromptConfig })
  })

  beforeEach(() => {
    mocks.bgRequest.mockReset()
    mocks.bgStream.mockReset()
  })

  it("reads deployed OpenAPI from the captured native owner without consulting a replacement connection", async () => {
    const client = new TldwApiClient()
    const getConfig = vi
      .spyOn(client, "getConfig")
      .mockResolvedValue({
        serverUrl: "https://replacement.test",
        authMode: "single-user"
      })
    const initialize = vi.spyOn(client, "initialize").mockResolvedValue(undefined)
    const signal = new AbortController().signal
    const spec = { paths: { "/api/v1/chats/{chat_id}/messages": { post: {} } } }
    mocks.bgRequest.mockResolvedValue(spec)
    const read = client.getOpenAPISpec
    expect(await read.call(client, { requestScope, signal })).toBe(spec)
    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/openapi.json",
        method: "GET",
        abortSignal: signal,
        headers: { "X-TLDW-Expected-User-ID": "42" },
        servicePromptConfig: expectedScopeFields.servicePromptConfig
      })
    )
    expect(getConfig).not.toHaveBeenCalled()
    expect(initialize).not.toHaveBeenCalled()
  })

  it("does not begin retired scoped OpenAPI discovery", async () => {
    const client = new TldwApiClient()
    const signal = AbortSignal.abort()
    const initialize = vi.spyOn(client, "initialize").mockResolvedValue(undefined)
    const read = client.getOpenAPISpec
    expect(await read.call(client, { requestScope, signal })).toBeNull()
    expect(mocks.bgRequest).not.toHaveBeenCalled()
    expect(initialize).not.toHaveBeenCalled()
  })

  it.each([null, {}, { chats: {} }])("does not manufacture empty chats from an unknown list response %j", async response => {
    mocks.bgRequest.mockResolvedValue(response)
    await expect(new TldwApiClient().listChatsWithMeta({}, { requestScope })).rejects.toThrow(/chat.*list|list.*chat/i)
  })

  it.each([null, {}, "saved", { id: "" }, { id: {} }, { id: Number.MAX_SAFE_INTEGER + 1 }])("rejects malformed chat rows before normalization %j", async row => {
    mocks.bgRequest.mockResolvedValue({ chats: [row] })
    await expect(new TldwApiClient().listChatsWithMeta({}, { requestScope })).rejects.toThrow(/chat.*list|list.*chat/i)
  })

  it("pins default Character reads and writes to the captured account", async () => {
    const client = new TldwApiClient()
    const signal = new AbortController().signal
    mocks.bgRequest.mockResolvedValueOnce({ preferences: { "preferences.chat.default_character_id": { value: 7 } } })
    expect(await client.getDefaultCharacterPreference({ requestScope, signal })).toBe("7")
    expect(mocks.bgRequest.mock.calls[0][0]).toMatchObject({
      path: "/api/v1/users/me/profile?sections=preferences", method: "GET",
      headers: { "X-TLDW-Expected-User-ID": "42" },
      servicePromptConfig: expectedScopeFields.servicePromptConfig, abortSignal: signal
    })
    mocks.bgRequest.mockResolvedValueOnce({ applied: [], skipped: [] })
    await client.setDefaultCharacterPreference(" 7 ", { requestScope, signal })
    expect(mocks.bgRequest.mock.calls[1][0]).toMatchObject({
      path: "/api/v1/users/me/profile", method: "PATCH",
      headers: { "X-TLDW-Expected-User-ID": "42" },
      servicePromptConfig: expectedScopeFields.servicePromptConfig, abortSignal: signal,
      body: { updates: [{ key: "preferences.chat.default_character_id", value: "7" }] }
    })
  })

  it.each([undefined, [], ["data:image/png;base64,aW1hZ2U="]].map(images => ({ images })))("preserves optional complete attachment lists from the actual domain adapter: $images", ({ images }) => {
    mocks.bgRequest.mockResolvedValue({ messages: [{ id: "owned-user", sender: "user", content: "Question", timestamp: "2026-09-16T00:00:00Z", ...(images === undefined ? {} : { images }) }] })
    return new TldwApiClient().listChatMessages("owned", { include_images: true }, { requestScope }).then(rows => {
      expect(rows[0].images).toEqual(images)
      expect(mocks.bgRequest.mock.calls[0][0]).toMatchObject({ headers: { "X-TLDW-Expected-User-ID": "42" }, servicePromptConfig: expectedScopeFields.servicePromptConfig })
    })
  })

  it("prefers the canonical trailing-slash character collection route in both client implementations", async () => {
    mocks.bgRequest.mockResolvedValue([])

    const baseClient = new TldwApiClientBase()
    vi.spyOn(baseClient, "resolveApiPath").mockImplementation(async (_key, paths) => {
      if (!paths[0]?.startsWith("/")) throw new Error("Expected a relative API path fixture")
      return `/${paths[0].slice(1)}` as const
    })
    await TldwApiClientBase.prototype.listCharacters.call(baseClient, { limit: 5 })

    expect(mocks.bgRequest.mock.calls[0]?.[0]).toMatchObject({
      path: "/api/v1/characters/?limit=5",
      method: "GET"
    })

    const domainClient = new TldwApiClientBase()
    vi.spyOn(domainClient, "resolveApiPath").mockImplementation(async (_key, paths) => {
      if (!paths[0]?.startsWith("/")) throw new Error("Expected a relative API path fixture")
      return `/${paths[0].slice(1)}` as const
    })
    await characterMethods.listCharacters.call(domainClient, { limit: 5 })

    expect(mocks.bgRequest.mock.calls[1]?.[0]).toMatchObject({
      path: "/api/v1/characters/?limit=5",
      method: "GET"
    })
  })

  it("keeps the ordinary association read filtered and requests disabled links only when asked", async () => {
    mocks.bgRequest.mockResolvedValue([])

    const baseClient = new TldwApiClientBase()
    await TldwApiClientBase.prototype.listCharacterWorldBooks.call(baseClient, 7)
    await TldwApiClientBase.prototype.listCharacterWorldBooks.call(baseClient, 7, true)

    expect(mocks.bgRequest.mock.calls[0]?.[0]).toMatchObject({
      path: "/api/v1/characters/7/world-books",
      method: "GET"
    })
    expect(mocks.bgRequest.mock.calls[1]?.[0]).toMatchObject({
      path: "/api/v1/characters/7/world-books?enabled_only=false",
      method: "GET"
    })

    const chatRagClient = new TldwApiClientBase()
    await chatRagMethods.listCharacterWorldBooks.call(chatRagClient, 8, true)

    expect(mocks.bgRequest.mock.calls[2]?.[0]).toMatchObject({
      path: "/api/v1/characters/8/world-books?enabled_only=false",
      method: "GET"
    })
  })

  it.each([null, "image", ["data:image/png;base64,aW1hZ2U=", null], ["https://other/image"], ["data:image/png;base64,invalid!"]].map(images => ({ images })))("rejects incomplete or malformed attachment arrays without a subset: $images", ({ images }) => {
    mocks.bgRequest.mockResolvedValue({ messages: [{ id: "owned-user", sender: "user", content: "Question", images }] })
    return expect(new TldwApiClient().listChatMessages("owned", { include_images: true }, { requestScope })).rejects.toThrow(/image|attachment/i)
  })

  it.each([{}, { images: [] }])("rejects an opt-in response that claims an image but omits its complete bytes: %s", async fields => {
    mocks.bgRequest.mockResolvedValue({ messages: [{ id: "owned-user", sender: "user", content: "Question", has_image: true, ...fields }] })
    await expect(new TldwApiClient().listChatMessages("owned", { include_images: true }, { requestScope })).rejects.toThrow(/image|attachment/i)
  })

  it.each([true, false])("keeps captured character recovery scope optional without changing content (scoped: %s)", async scoped => {
    const body = { assistant_content: "<think>Partial reasoning</think>", assistant_message_id: "acknowledged", mood_label: "calm" }
    mocks.bgRequest.mockResolvedValue({ assistant_message_id: "acknowledged" })
    const signal = new AbortController().signal
    await new TldwApiClient().persistCharacterCompletion("owned", body, scoped ? { requestScope, signal } : undefined)
    const request = mocks.bgRequest.mock.calls[0][0]
    expect(request.body).toEqual(body)
    expect(request.path).toBe("/api/v1/chats/owned/completions/persist?scope_type=global")
    expect(request.method).toBe("POST")
    if (scoped) {
      expect(request).toMatchObject({ ...expectedScopeFields, abortSignal: signal })
    } else {
      expect(request.servicePromptConfig).toBeUndefined()
      expect(request.headers).not.toHaveProperty("X-TLDW-Expected-User-ID")
    }
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
    vi.spyOn(client, "resolveApiPath").mockImplementation(async (_key, paths) => {
      if (!paths[0]?.startsWith("/")) throw new Error("Expected a relative API path fixture")
      return `/${paths[0].slice(1)}` as const
    })
    if (method === "updateChat") {
      mocks.bgRequest.mockImplementation(async request => ({
        id: "owned", version: request.method === "GET" ? 4 : 5,
        title: request.method === "GET" ? "Own title" : "Own renamed title"
      }))
    } else mocks.bgRequest.mockResolvedValue({ ok: true, data: new ArrayBuffer(0), id: "owned" })
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

  it("binds captured Character streaming scope outside the inference body", async () => {
    mocks.bgStream.mockImplementation(async function* () { yield '{"choices":[{"delta":{"content":"Owned"}}]}' })
    const signal = new AbortController().signal
    const options = { requestScope, signal, streamIdleTimeoutMs: 12345 }
    for await (const _chunk of new TldwApiClient().streamCharacterChatCompletion("owned", { model: "owned-model" }, options)) { /* consume */ }
    expect(mocks.bgStream.mock.calls[0][0]).toMatchObject({
      path: "/api/v1/chats/owned/complete-v2?scope_type=global", abortSignal: signal,
      streamIdleTimeoutMs: 12345, ...expectedScopeFields,
      body: { model: "owned-model", stream: true }
    })
    expect(mocks.bgStream.mock.calls[0][0].body).not.toHaveProperty("requestScope")
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

  it.each(["public", "base", "domain"])("keeps %s greeting seed options out of JSON while retaining captured account, workspace and cancellation", async (implementation) => {
    mocks.bgRequest.mockResolvedValueOnce({id: "greeting-chat", title: "Opening"})
    const signal = new AbortController().signal
    const body = {character_id: 42, provider: "openai", model: "configured-model", temperature: 0, top_p: 0}
    const client = implementation === "public" ? new TldwApiClient() : new TldwApiClientBase()
    const options = {requestScope, signal, scope: {type: "workspace" as const, workspaceId: "owned-workspace"},
      seedFirstMessage: true, greetingStrategy: "alternate_index" as const, alternateIndex: 0}
    if (implementation === "domain") await chatRagMethods.createChat.call(client, body, options)
    else await client.createChat(body, options)
    expect(mocks.bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/chats/?seed_first_message=true&greeting_strategy=alternate_index&alternate_index=0",
      method: "POST", body: {...body, scope_type: "workspace", workspace_id: "owned-workspace"},
      abortSignal: signal, ...expectedScopeFields
    })
  })

  it("reads fresh promotion messages with captured scope without joining cached or pending reads", async () => {
    const client = new TldwApiClient()
    vi.spyOn(client, "ensureConfigForRequest").mockResolvedValue({ ...requestScope.config, accessToken: "token" })
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

describe('H1 native owner scope', () => {
  it('carries captured lease and independent workspace through both history routes and settlement', async () => {
    mocks.bgRequest.mockResolvedValue({})
    const client = new TldwApiClient()
    const options = { requestScope, scope: { type: 'workspace', workspaceId: '7' }, signal: new AbortController().signal }
    await client.captureHistorySelection('chat', { view: {} as any, purpose: 'send' }, options as any)
    await client.confirmHistoryProjection('chat', {} as any, options as any)
    await client.persistCharacterCompletion('chat', { assistant_message_id: 'reply' }, options as any)
    for (const [init] of mocks.bgRequest.mock.calls.slice(-3)) {
      expect(init).toMatchObject({ ...expectedScopeFields, abortSignal: options.signal })
      expect(init.path).toContain('workspace_id=7')
    }
  })
})

it.each([
  ["composed", TldwApiClient],
  ["domain-on-base", TldwApiClientBase]
] as const)(
  "%s opts in to native completion capture without replacing original scope",
  async (_name, Client) => {
    mocks.bgRequest.mockResolvedValue({})
    const client = new Client()
    const capture =
      _name === "composed"
        ? (client as TldwApiClient).captureHistorySelection.bind(client)
        : chatRagMethods.captureHistorySelection.bind(client)
    const signal = new AbortController().signal
    const request = {
      view: {
        view_session_id: "view-one",
        conversation_id: "chat/one",
        interpretation: { kind: "parent_graph_v1" as const },
        cursor: { kind: "empty" as const },
        selection_revision: 1
      },
      purpose: "send" as const
    }
    await capture("chat/one", request, {
      requestScope,
      signal,
      scope: { type: "workspace", workspaceId: "7" },
      includeNativeCompletion: true
    })
    expect(mocks.bgRequest.mock.lastCall?.[0]).toMatchObject({
      ...expectedScopeFields,
      abortSignal: signal,
      body: request,
      path: "/api/v1/chat/conversations/chat%2Fone/history/selection?scope_type=workspace&workspace_id=7&include_native_completion=true"
    })
    await capture("chat/one", request, {
      requestScope
    })
    expect(mocks.bgRequest.mock.lastCall?.[0].path).toBe(
      "/api/v1/chat/conversations/chat%2Fone/history/selection?scope_type=global"
    )
  }
)

it.each([['composed', TldwApiClient], ['base', TldwApiClientBase]] as const)(
  '%s streaming accepts frozen selected-history payload without mutating its request',
  async (_name, Client) => {
    mocks.bgStream.mockImplementation(async function* () {
      yield '{"choices":[{"delta":{"content":"saved reply"}}]}'
    })
    const body = Object.freeze({
      model: 'h1-model',
      stream: false,
      messages: [{ role: 'user' as const, content: 'selected input' }]
    })
    const chunks = []
    for await (const chunk of new Client().streamChatCompletion(body, { requestScope })) chunks.push(chunk)
    expect(chunks).toEqual([{ choices: [{ delta: { content: 'saved reply' } }] }])
    expect(body.stream).toBe(false)
    expect(mocks.bgStream.mock.calls.at(-1)?.[0].body).toEqual({ ...body, stream: true })
  }
)

describe("article capture scoped reads", () => {
  it("forwards owner and abort through status, membership, pin preview and Media version reads", async () => {
    const client = new TldwApiClient()
    vi.spyOn(client, "resolveApiPath").mockImplementation(
      async (_key, paths) => paths[0] as never
    )
    const signal = new AbortController().signal
    const options = { requestScope, signal }
    mocks.bgRequest.mockResolvedValue([])
    await client.getWebClipStatus("clip/1", options)
    await client.getWorkspaceSources("ws-1", options)
    await client.getWorkspaceSourcePreview(
      "ws-1",
      "web-clipper:uuid",
      { version_number: 7 },
      options
    )
    await client.listMediaDocumentVersions(71, options)
    await client.getMediaDocumentVersion(71, 7, options)
    const requests = mocks.bgRequest.mock.calls.slice(-5).map((call) => call[0])
    expect(requests.map((r) => r.path)).toEqual([
      "/api/v1/web-clipper/clip%2F1",
      "/api/v1/workspaces/ws-1/sources",
      "/api/v1/workspaces/ws-1/sources/web-clipper%3Auuid/preview?version_number=7",
      "/api/v1/media/71/versions?include_content=true",
      "/api/v1/media/71/versions/7?include_content=true"
    ])
    for (const request of requests)
      expect(request).toMatchObject({
        abortSignal: signal,
        servicePromptConfig: expectedScopeFields.servicePromptConfig,
        headers: { "X-TLDW-Expected-User-ID": "42" }
      })
  })
})

it("preserves default version URLs and forwards native page parameters and rows", async () => {
  const client = new TldwApiClient()
  const signal = new AbortController().signal
  const options = { requestScope, signal }
  const rows = [
    { media_id: 71, version_number: 9, uuid: "accepted", content: "full text" }
  ]
  mocks.bgRequest.mockResolvedValue(rows)
  expect(await client.listMediaDocumentVersions(71, options)).toBe(rows)
  expect(mocks.bgRequest.mock.lastCall?.[0].path).toBe(
    "/api/v1/media/71/versions?include_content=true"
  )
  expect(
    await client.listMediaDocumentVersions(71, options, { limit: 10, page: 2 })
  ).toBe(rows)
  expect(mocks.bgRequest.mock.lastCall?.[0]).toMatchObject({
    path: "/api/v1/media/71/versions?include_content=true&limit=10&page=2",
    method: "GET",
    abortSignal: signal,
    servicePromptConfig: expectedScopeFields.servicePromptConfig,
    headers: { "X-TLDW-Expected-User-ID": "42" }
  })
})


it.each([[true, false], [false, false], [true, true], [false, true]])("preserves native greeting seed query and captured scope (enabled=%s, base=%s)", async (enabled, base) => {
  const client = base ? new TldwApiClientBase() : new TldwApiClient()
  const signal = new AbortController().signal
  mocks.bgRequest.mockReset()
  mocks.bgRequest.mockResolvedValueOnce({ id: "seeded-chat" })
  await client.createChat({ character_id: 12 }, { requestScope, signal, seedFirstMessage: enabled, greetingStrategy: "alternate_index", alternateIndex: 2 } as Parameters<typeof client.createChat>[1])
  expect(mocks.bgRequest).toHaveBeenCalledWith(expect.objectContaining({
    path: `/api/v1/chats/?seed_first_message=${enabled}&greeting_strategy=alternate_index&alternate_index=2`,
    method: "POST", body: { character_id: 12, scope_type: "global" }, abortSignal: signal, ...expectedScopeFields
  }))
})

it.each([TldwApiClientBase, TldwApiClient])(
  "native message reads and edits preserve captured workspace, principal and cancellation through %s",
  async (Client) => {
    mocks.bgRequest.mockReset()
    const client = new Client()
    const signal = new AbortController().signal
    const options = { requestScope, signal, scope: { type: "workspace" as const, workspaceId: "original" } }
    mocks.bgRequest.mockResolvedValue({ id: "m1", version: 2 })
    await (client.getMessage as any)("m1", options)
    await (client.editMessage as any)("m1", "Exact  \n", 1, "chat", options)
    expect(mocks.bgRequest.mock.calls.map(([request]) => request)).toEqual([
      expect.objectContaining({
        path: "/api/v1/messages/m1?scope_type=workspace&workspace_id=original", method: "GET",
        abortSignal: signal, headers: expect.objectContaining({ "X-TLDW-Expected-User-ID": "42" }),
        servicePromptConfig: expectedScopeFields.servicePromptConfig
      }),
      expect.objectContaining({
        path: "/api/v1/messages/m1?expected_version=1&scope_type=workspace&workspace_id=original", method: "PUT",
        body: { content: "Exact  \n" }, abortSignal: signal,
        headers: expectedScopeFields.headers, servicePromptConfig: expectedScopeFields.servicePromptConfig
      })
    ])
  }
)

it.each(["getMessage", "editMessage"])("native %s refuses pre-aborted owner requests before transport", async method => {
  mocks.bgRequest.mockReset()
  const client = new TldwApiClient()
  const controller = new AbortController()
  controller.abort()
  const options = { requestScope, signal: controller.signal }
  const pending = method === "getMessage" ? (client.getMessage as any)("m1", options)
    : (client.editMessage as any)("m1", "Exact", 1, "chat", options)
  await expect(pending).rejects.toMatchObject({ name: "AbortError" })
  expect(mocks.bgRequest).not.toHaveBeenCalled()
})
