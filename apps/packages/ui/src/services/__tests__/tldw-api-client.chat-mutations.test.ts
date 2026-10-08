import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn(),
  bgUpload: vi.fn()
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.bgRequest(...args),
  bgUpload: (...args: unknown[]) => mocks.bgUpload(...args),
  bgStream: vi.fn()
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

import { TldwApiClient } from "@/services/tldw/TldwApiClient"

describe("TldwApiClient chat mutations", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.bgRequest.mockReset()
  })

  it("creates chats through the backend-canonical trailing-slash endpoint", async () => {
    mocks.bgRequest.mockResolvedValue({
      id: "chat-1",
      title: "Role-play",
      created_at: "2026-02-18T00:00:00Z",
      updated_at: "2026-02-18T00:00:00Z",
      character_id: 42
    })

    const client = new TldwApiClient()
    await client.createChat({ character_id: 42 })

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/chats/",
        method: "POST",
        body: expect.objectContaining({ character_id: 42 })
      })
    )
  })

  it.each([
    { options: {seedFirstMessage: true, greetingStrategy: "default" as const}, query: "?seed_first_message=true&greeting_strategy=default" },
    { options: {seedFirstMessage: true, greetingStrategy: "alternate_index" as const, alternateIndex: 0}, query: "?seed_first_message=true&greeting_strategy=alternate_index&alternate_index=0" },
    { options: {seedFirstMessage: false}, query: "?seed_first_message=false" }
  ])("uses the existing greeting seed query contract: $query", async ({options, query}) => {
    mocks.bgRequest.mockResolvedValue({id: "greeting-chat", character_id: 42})
    const result = await new TldwApiClient().createChat({character_id: 42}, options)
    expect(mocks.bgRequest).toHaveBeenCalledWith(expect.objectContaining({
      path: "/api/v1/chats/" + query, method: "POST", body: {character_id: 42, scope_type: "global"}
    }))
    expect(result.id).toBe("greeting-chat")
  })

  it.each([-1, 0.5, Number.NaN, Number.POSITIVE_INFINITY, Number.MAX_SAFE_INTEGER + 1])(
    "rejects invalid alternate index %s before transport", async (alternateIndex) => {
      await expect(new TldwApiClient().createChat({character_id: 42}, {
        seedFirstMessage: true, greetingStrategy: "alternate_index", alternateIndex
      })).rejects.toThrow("alternateIndex")
      expect(mocks.bgRequest).not.toHaveBeenCalled()
    }
  )

  it.each([409, 412])("preserves explicit update version on conflict %s without replay", async (status) => {
    const conflict = Object.assign(new Error("Version conflict"), { status })
    mocks.bgRequest.mockImplementation(async (request) => {
      if (request.method === "GET") return { id: "abc", version: 7, title: "Changed elsewhere" }
      if (request.path.endsWith("expected_version=4")) throw conflict
      return { id: "abc", version: 8, title: "Renamed" }
    })
    await expect(new TldwApiClient().updateChat("abc", { title: "Renamed" }, { expectedVersion: 4 })).rejects.toBe(conflict)
    expect(mocks.bgRequest.mock.calls.map(([request]) => [request.method, request.path])).toEqual([
      ["PUT", "/api/v1/chats/abc?scope_type=global&expected_version=4"]
    ])
  })

  it("does not silently rebase an implicitly discovered version after conflict", async () => {
    const conflict = Object.assign(new Error("Version conflict"), { status: 409 })
    mocks.bgRequest.mockImplementation(async (request) => {
      if (request.method === "GET") return { id: "abc", version: 4, title: "Before" }
      throw conflict
    })
    await expect(new TldwApiClient().updateChat("abc", { title: "Renamed" })).rejects.toBe(conflict)
    expect(mocks.bgRequest.mock.calls.map(([request]) => request.method)).toEqual(["GET", "PUT"])
  })

  it("does not PUT when metadata version discovery fails", async () => {
    const failure = new Error("Details unavailable")
    mocks.bgRequest.mockImplementation(async (request) => {
      if (request.method === "GET") throw failure
      return { id: "abc", version: 5, title: "Renamed" }
    })
    await expect(new TldwApiClient().updateChat("abc", { title: "Renamed" })).rejects.toBe(failure)
    expect(mocks.bgRequest.mock.calls.map(([request]) => request.method)).toEqual(["GET"])
  })

  it.each([
    { id: "other", version: 4 }, { id: "abc" }, { id: "abc", expected_version: 4 },
    { id: "abc", version: 0 }, { id: "abc", version: -1 }, { id: "abc", version: 1.5 },
    { id: "abc", version: Number.NaN }, { id: "abc", version: Number.POSITIVE_INFINITY },
    { id: "abc", version: Number.MAX_SAFE_INTEGER + 1 }
  ])("does not PUT after unqualified metadata discovery %j", async (details) => {
    mocks.bgRequest.mockImplementation(async (request) => request.method === "GET" ? details : { id: "abc", version: 5, title: "Renamed" })
    await expect(new TldwApiClient().updateChat("abc", { title: "Renamed" })).rejects.toThrow()
    expect(mocks.bgRequest.mock.calls.map(([request]) => request.method)).toEqual(["GET"])
  })

  it.each([0, -1, 1.5, Number.NaN, Number.POSITIVE_INFINITY, Number.MAX_SAFE_INTEGER + 1])(
    "rejects invalid explicit metadata version %s before transport", async (expectedVersion) => {
      await expect(new TldwApiClient().updateChat("abc", { title: "Renamed" }, { expectedVersion })).rejects.toThrow()
      expect(mocks.bgRequest).not.toHaveBeenCalled()
    }
  )

  it.each([
    null, [], {}, { version: 5, title: "Renamed" }, { id: "other", version: 5, title: "Renamed" },
    { id: "abc", expected_version: 5, title: "Renamed" }, { id: "abc", version: 4, title: "Renamed" },
    { id: "abc", version: 5.5, title: "Renamed" }, { id: "abc", version: 5 },
    { id: "abc", version: 5, title: "Unchanged" }
  ].map(reply => ({ reply })))("rejects unacknowledged metadata reply $reply", async ({ reply }) => {
    mocks.bgRequest.mockResolvedValue(reply)
    await expect(new TldwApiClient().updateChat("abc", { title: "Renamed" }, { expectedVersion: 4 })).rejects.toThrow()
    expect(mocks.bgRequest).toHaveBeenCalledTimes(1)
  })

  it("requires an explicit topic clear in the raw receipt", async () => {
    mocks.bgRequest.mockResolvedValue({ id: "abc", version: 5 })
    await expect(new TldwApiClient().updateChat("abc", { topic_label: null }, { expectedVersion: 4 })).rejects.toThrow()
  })

  it.each([{ persona_memory_mode: "read_write" }, { topic_label: undefined }, {}])(
    "rejects unsupported or empty metadata intent before transport %j", async (patch) => {
      await expect(new TldwApiClient().updateChat("abc", patch, { expectedVersion: 4 })).rejects.toThrow()
      expect(mocks.bgRequest).not.toHaveBeenCalled()
    }
  )

  it.each([{ title: "Renamed" }, { topic_label: null }, { rating: 1 }])(
    "returns matching scoped acknowledged metadata %j", async (patch) => {
      const signal = new AbortController().signal
      const requestScope = { config: { serverUrl: "https://owned.test", authMode: "single-user" as const }, userId: 1 }
      mocks.bgRequest.mockResolvedValue({ id: "abc", version: 5, ...patch })
      const result = await new TldwApiClient().updateChat("abc", patch, { expectedVersion: 4, requestScope, signal, scope: { type: "workspace", workspaceId: "W" } })
      expect(result).toMatchObject({ id: "abc", version: 5 })
      if ("title" in patch) expect(result.title).toBe(patch.title)
      if ("topic_label" in patch) expect(result.topic_label).toBeNull()
      expect(mocks.bgRequest).toHaveBeenCalledWith(expect.objectContaining({
        path: "/api/v1/chats/abc?scope_type=workspace&workspace_id=W&expected_version=4",
        method: "PUT", body: patch, abortSignal: signal,
        servicePromptConfig: expect.objectContaining({ serverUrl: "https://owned.test" })
      }))
    }
  )

  it.each([{ rating: 0 }, { rating: 6 }, { rating: 1.5 }, { title: {} }, { state: "invalid" }])(
    "rejects invalid supported metadata %j before transport", async (patch) => {
      await expect(new TldwApiClient().updateChat("abc", patch, { expectedVersion: 4 })).rejects.toThrow()
      expect(mocks.bgRequest).not.toHaveBeenCalled()
    }
  )

  it("uses the deployed canonical lifecycle state in the intent and receipt", async () => {
    mocks.bgRequest.mockResolvedValue({ id: "abc", version: 5, state: "resolved" })
    const saved = await new TldwApiClient().updateChat("abc", { state: " Resolved " }, { expectedVersion: 4 })
    expect(saved.state).toBe("resolved")
    expect(mocks.bgRequest.mock.calls[0][0].body).toEqual({ state: "resolved" })
  })

  it("captures metadata intent before a held version read", async () => {
    let release!: () => void
    const gate = new Promise<void>(resolve => { release = resolve })
    mocks.bgRequest.mockImplementation(async (request) => {
      if (request.method === "GET") { await gate; return { id: "abc", version: 4 } }
      return { id: "abc", version: 5, title: "Original edit" }
    })
    const patch = { title: "Original edit" }
    const pending = new TldwApiClient().updateChat("abc", patch)
    patch.title = "Newer edit"
    release()
    await pending
    expect(mocks.bgRequest.mock.calls[1][0].body).toEqual({ title: "Original edit" })
  })

  it.each(["GET", "PUT"])("refuses a retired metadata %s continuation", async (phase) => {
    let release!: () => void
    const gate = new Promise<void>(resolve => { release = resolve })
    let enter!: () => void
    const entered = new Promise<void>(resolve => { enter = resolve })
    const controller = new AbortController()
    mocks.bgRequest.mockImplementation(async (request) => {
      if (request.method === phase) { enter(); await gate }
      return { id: "abc", version: request.method === "GET" ? 4 : 5, title: "Original edit" }
    })
    const pending = new TldwApiClient().updateChat("abc", { title: "Original edit" }, { signal: controller.signal })
    await entered
    controller.abort()
    release()
    await expect(pending).rejects.toThrow()
    expect(mocks.bgRequest.mock.calls.map(([request]) => request.method)).toEqual(phase === "GET" ? ["GET"] : ["GET", "PUT"])
  })

  it("forwards nested routing overrides when creating a chat completion", async () => {
    mocks.bgRequest.mockResolvedValue({ id: "resp-1", object: "chat.completion" })

    const client = new TldwApiClient()
    await client.createChatCompletion({
      model: "auto",
      messages: [{ role: "user", content: "hello" }],
      routing: { mode: "per_turn", cross_provider: false }
    })

    const request = mocks.bgRequest.mock.calls.at(-1)?.[0] as {
      path?: string
      method?: string
      body?: {
        routing?: {
          mode?: string
          cross_provider?: boolean
        }
      }
    }

    expect(request.path).toBe("/api/v1/chat/completions")
    expect(request.method).toBe("POST")
    expect(request.body?.routing).toEqual({
      mode: "per_turn",
      cross_provider: false
    })
  })

  it("forwards explicit chat completion timeouts to the request transport", async () => {
    mocks.bgRequest.mockResolvedValue({ id: "resp-1", object: "chat.completion" })

    const client = new TldwApiClient()
    await client.createChatCompletion(
      {
        model: "auto",
        messages: [{ role: "user", content: "hello" }]
      },
      { timeoutMs: 180_000 }
    )

    const request = mocks.bgRequest.mock.calls.at(-1)?.[0] as {
      path?: string
      method?: string
      timeoutMs?: number
    }

    expect(request.path).toBe("/api/v1/chat/completions")
    expect(request.method).toBe("POST")
    expect(request.timeoutMs).toBe(180_000)
  })

  it("preserves successful background payloads even when they discuss errors and paths", async () => {
    const completion = {
      id: "resp-1",
      object: "chat.completion",
      error: "trace=/Users/private/stack.txt",
      details: "/Users/private/stack.txt",
      nested: {
        traceback: "Traceback: /Users/private/stack.txt",
        items: [{ exception: "boom" }]
      }
    }
    mocks.bgRequest.mockResolvedValue(completion)

    const client = new TldwApiClient()
    const response = await client.createChatCompletion({
      model: "auto",
      messages: [{ role: "user", content: "hello" }]
    })

    const payload = await response.json()

    expect(payload).toEqual(completion)
  })
})


describe("qualified metadata reads for an unresolved proposal", () => {
  beforeEach(() => mocks.bgRequest.mockReset())
  it.each([
    { id: "chat-1", version: 5 },
    { id: "chat-1", version: 5, topicLabel: null },
    { id: "other", version: 5, topic_label: null },
    { id: "chat-1", expected_version: 5, topic_label: null },
    { id: "chat-1", version: 5, topic_label: 123 },
    { id: "chat-1", version: 0, topic_label: null }
  ])("does not certify topic Clear from an unqualified raw read %j", async response => {
    mocks.bgRequest.mockResolvedValue(response)
    await expect(new TldwApiClient().getChat("chat-1", { requiredMetadata: ["topic_label"] })).rejects.toThrow("conversation metadata")
  })
  it("preserves canonical null despite contradictory metadata aliases", async () => {
    mocks.bgRequest.mockResolvedValue({ id: "chat-1", version: 5, topic_label: null, topicLabel: "Old topic", cluster_id: null, clusterId: "Old cluster", external_ref: null, externalRef: "Old ref" })
    const result = await new TldwApiClient().getChat("chat-1", { requiredMetadata: ["topic_label", "cluster_id", "external_ref"] })
    expect(result).toMatchObject({ topic_label: null, cluster_id: null, external_ref: null })
  })
  it("rejects an unsupported canonical state in a qualified metadata read", async () => {
    mocks.bgRequest.mockResolvedValue({ id: "chat-1", version: 5, state: "imaginary-state" })
    await expect(new TldwApiClient().getChat("chat-1", { requiredMetadata: ["state"] })).rejects.toThrow("conversation metadata")
  })
  it.each([null, "Saved tag"])("accepts actual owned metadata values without exposing normalized absence: %j", async topic => {
    mocks.bgRequest.mockResolvedValue({ id: "chat-1", version: 5, topic_label: topic })
    const signal = new AbortController().signal
    const result = await new TldwApiClient().getChat("chat-1", { requiredMetadata: ["topic_label"], scope: { type: "workspace", workspaceId: "W" }, signal })
    expect(result).toMatchObject({ id: "chat-1", version: 5, topic_label: topic })
    expect(mocks.bgRequest).toHaveBeenCalledExactlyOnceWith(expect.objectContaining({ path: "/api/v1/chats/chat-1?scope_type=workspace&workspace_id=W", abortSignal: signal }))
  })
})
