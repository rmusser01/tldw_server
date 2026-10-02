import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const boundary = vi.hoisted(() => ({ get: vi.fn(), fetch: vi.fn() }))
vi.mock("wxt/browser", () => ({ browser: { runtime: { id: null } } }))
vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({ get: boundary.get, set: vi.fn(), remove: vi.fn(), watch: vi.fn(), unwatch: vi.fn() }),
  safeStorageSerde: { serialize: (value: unknown) => value, deserialize: (value: unknown) => value },
}))
vi.mock("@/services/tldw/runtime-auth-override", () => ({ getRuntimeSingleUserApiKeyOverride: () => null, isCookieSessionConfigInvalidated: () => false }))
import { TldwApiClient, tldwClient } from "../tldw/TldwApiClient"
import { submitExplicitFeedback } from "../feedback"
import { createKnowledgeQaClient } from "@/components/Option/KnowledgeQA/knowledgeQaClient"
import type { ServicePromptSnapshot } from "../service-prompts"

const config = (user = 1, serverUrl = "https://qa.test") => ({
  serverUrl, authMode: "multi-user" as const,
  accessToken: `test.${btoa(JSON.stringify({ sub: String(user) }))}.signature`,
})
const options = { requestScope: { config: { serverUrl: "https://qa.test", authMode: "multi-user" as const }, userId: 1 } }
let client: TldwApiClient
const operations = [
  ["history", () => client.fetchWithAuth("/api/v1/chat/conversations?keywords=__knowledge_QA__", options)],
  ["restore", () => client.fetchWithAuth("/api/v1/chat/conversations/owned/messages-with-context", options)],
  ["tag-read", () => client.fetchWithAuth("/api/v1/chat/conversations/owned", options)],
  ["tag-write", () => client.fetchWithAuth("/api/v1/chat/conversations/owned", { ...options, method: "PATCH", body: { version: 1, keywords: ["__knowledge_QA__"] } })],
  ["context", () => client.fetchWithAuth("/api/v1/chat/messages/answer/rag-context", { ...options, method: "POST", body: { message_id: "answer", rag_context: {} } })],
  ["characters", () => client.listCharacters({ limit: 5 }, options)],
  ["character-search", () => client.searchCharacters("Helpful AI Assistant", { limit: 5 }, options)],
  ["chat-read", () => client.getChat("owned", options)],
  ["delete", () => client.deleteChat("owned", options)],
  ["share", () => client.createConversationShareLink("owned", { ttl_seconds: 300 }, options)],
  ["revoke", () => client.revokeConversationShareLink("owned", "share", options)],
  ["export", () => client.exportChatbook({ name: "Own export", description: "Own" }, options)],
  ["download", () => client.downloadChatbookExport("owned", options)],
  ["note", () => client.createNote("Own answer", { title: "Own" }, options)],
  ["source-health", () => client.ragSourceHealth(options)],
  ["rag", () => client.ragSearch("Own question", options)],
  ["rag-stream", async () => { for await (const _event of client.ragSearchStream("Own question", options)) { /* consume */ } }],
  ["source-feedback", () => submitExplicitFeedback({ feedback_type: "relevance", query: "Own question", conversation_id: "owned", chunk_ids: ["own-chunk"] }, options)],
] as const

describe("Knowledge QA real outbound owner checks", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    client = new TldwApiClient()
    vi.spyOn(tldwClient, "initialize").mockResolvedValue(undefined)
    vi.spyOn(client, "ensureConfigForRequest").mockResolvedValue(config())
    vi.spyOn(client, "resolveApiPath").mockImplementation(async (_key, paths) => paths[0])
    boundary.get.mockImplementation(async (key: string) => key === "tldwConfig" ? config() : null)
    boundary.fetch.mockImplementation(async () => new Response('{"type":"contexts","contexts":[],"id":"owned"}\n', {
      status: 200, headers: { "Content-Type": "application/json" },
    }))
    vi.stubGlobal("fetch", boundary.fetch)
  })
  afterEach(() => vi.unstubAllGlobals())

  it.each([
    ["tag-read", "/api/v1/chat/conversations/owned", "GET"],
    ["context", "/api/v1/chat/messages/answer/rag-context", "POST"],
  ] as const)("normalizes server owner denial for QA %s", async (_name, path, method) => {
    vi.spyOn(tldwClient, "ensureConfigForRequest").mockResolvedValue(config())
    boundary.fetch.mockResolvedValue(new Response(JSON.stringify({ detail: { code: "request_config_scope_changed", message: "Account changed." } }), { status: 412, headers: { "Content-Type": "application/json" } }))
    const controller = new AbortController()
    const qa = createKnowledgeQaClient({ requestScope: options.requestScope, scopeSignal: controller.signal } as ServicePromptSnapshot, () => true)
    await expect(qa.fetchWithAuth(path, { method })).rejects.toMatchObject({ status: 412, details: { detail: { code: "request_config_scope_changed" } } })
    expect(boundary.fetch).toHaveBeenCalledTimes(1)
  })

  it.each([412, 503])("preserves an unrelated server %s response for QA optional metadata", async status => {
    vi.spyOn(tldwClient, "ensureConfigForRequest").mockResolvedValue(config())
    const data = { detail: { code: "metadata_unavailable", message: "Metadata unavailable." } }
    boundary.fetch.mockResolvedValue(new Response(JSON.stringify(data), { status, headers: { "Content-Type": "application/json" } }))
    const controller = new AbortController()
    const qa = createKnowledgeQaClient({ requestScope: options.requestScope, scopeSignal: controller.signal } as ServicePromptSnapshot, () => true)
    const result = await qa.fetchWithAuth("/api/v1/chat/conversations/owned")
    expect(result).toMatchObject({ ok: false, status })
    expect(await result.json()).toEqual(data)
  })

  it("does not send prior QA feedback if the account changes during client initialization", async () => {
    let release!: () => void
    vi.mocked(tldwClient.initialize).mockReturnValueOnce(new Promise<void>((resolve) => { release = resolve }))
    const feedback = submitExplicitFeedback({ feedback_type: "relevance", query: "Alice private question", chunk_ids: ["alice-chunk"] }, options)
    boundary.get.mockImplementation(async (key: string) => key === "tldwConfig" ? config(2) : null)
    release()
    await expect(feedback).rejects.toMatchObject({ status: 412 })
    expect(boundary.fetch).not.toHaveBeenCalled()
  })

  it.each(operations)("sends %s only to its captured account and target", async (_name, run) => {
    await run()
    expect(boundary.fetch).toHaveBeenCalledTimes(1)
    const [url, init] = boundary.fetch.mock.calls[0]
    expect(new URL(url).origin).toBe("https://qa.test")
    expect(new Headers(init.headers).get("X-TLDW-Expected-User-ID")).toBe("1")
  })
  it.each(operations)("blocks %s when credentials resolve to another account", async (_name, run) => {
    boundary.get.mockImplementation(async (key: string) => key === "tldwConfig" ? config(2) : null)
    await expect(run()).rejects.toMatchObject({ status: 412 })
    expect(boundary.fetch).not.toHaveBeenCalled()
  })
  it.each(operations)("blocks %s when the server changes", async (_name, run) => {
    boundary.get.mockImplementation(async (key: string) => key === "tldwConfig" ? config(1, "https://other.test") : null)
    await expect(run()).rejects.toMatchObject({ status: 412 })
    expect(boundary.fetch).not.toHaveBeenCalled()
  })
})
