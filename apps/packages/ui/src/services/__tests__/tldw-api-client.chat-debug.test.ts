import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn(),
  bgUpload: vi.fn(),
  bgStream: vi.fn()
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.bgRequest(...args),
  bgUpload: (...args: unknown[]) => mocks.bgUpload(...args),
  bgStream: (...args: unknown[]) => mocks.bgStream(...args)
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
import { getLastChatRequestDebugSnapshot } from "@/services/tldw/chat-request-debug"

describe("TldwApiClient chat request debug snapshot", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("captures stream complete-v2 payloads used by character chat", async () => {
    mocks.bgStream.mockImplementation(async function* () {
      yield JSON.stringify({ delta: "hello" })
    })

    const client = new TldwApiClient()
    for await (const _ of client.streamCharacterChatCompletion("42", {
      model: "kimi-k2",
      include_character_context: true
    })) {
      break
    }

    const snapshot = getLastChatRequestDebugSnapshot()
    expect(snapshot).toMatchObject({
      method: "POST",
      mode: "stream"
    })
    expect(snapshot?.endpoint).toMatch(
      /^\/api\/v1\/chats\/42\/complete-v2(?:\?|$)/
    )
    expect((snapshot?.body as any)?.model).toBe("kimi-k2")
    expect((snapshot?.body as any)?.include_character_context).toBe(true)
    expect((snapshot?.body as any)?.stream).toBe(true)
  })

  it("adds chat scope query params to streamed character completions", async () => {
    mocks.bgStream.mockImplementation(async function* () {
      yield JSON.stringify({ delta: "scoped" })
    })

    const client = new TldwApiClient()
    const scope = { type: "workspace", workspaceId: "workspace-7" } as const

    for await (const _ of client.streamCharacterChatCompletion(
      "42",
      { include_character_context: true },
      { scope }
    )) {
      break
    }

    expect(mocks.bgStream).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/chats/42/complete-v2?scope_type=workspace&workspace_id=workspace-7"
      })
    )
    expect(getLastChatRequestDebugSnapshot()?.endpoint).toBe(
      "/api/v1/chats/42/complete-v2?scope_type=workspace&workspace_id=workspace-7"
    )
  })

  it("captures complete endpoint payloads for non-stream character completion", async () => {
    mocks.bgRequest.mockResolvedValue({ ok: true })

    const client = new TldwApiClient()
    await client.completeChat("55", { foo: "bar" })

    const snapshot = getLastChatRequestDebugSnapshot()
    expect(snapshot).toMatchObject({
      endpoint: "/api/v1/chats/55/complete",
      method: "POST",
      mode: "non-stream"
    })
    expect((snapshot?.body as any)?.foo).toBe("bar")
  })
})

it("native versioned completion keeps workspace routing separate from the captured request lease", async () => {
  const requestScope: any = {
    config: {
      serverUrl: "https://origin.test",
      authMode: "multi-user",
      accessToken: "captured"
    },
    userId: "alice"
  }
  const selection: any = { version: 1, selection_digest: "bound" }
  mocks.bgStream.mockImplementation(async function* () {
    yield JSON.stringify({
      tldw_history_admission_v1: { input_message_id: "input" }
    })
    yield JSON.stringify({
      tldw_message_id: "result",
      tldw_conversation_id: "chat"
    })
  })
  const request = {
    model: "m",
    api_provider: "provider",
    messages: [{ role: "user" as const, content: "new" }],
    save_to_db: true,
    conversation_id: "chat",
    tldw_history_selection_v1: selection
  }
  const frames = []
  for await (const frame of new TldwApiClient().streamChatCompletion(request, {
    scope: { type: "workspace", workspaceId: "work" },
    requestScope
  }))
    frames.push(frame)
  expect(mocks.bgStream).toHaveBeenCalledWith(
    expect.objectContaining({
      path: "/api/v1/chat/completions?scope_type=workspace&workspace_id=work",
      servicePromptConfig: { ...requestScope.config, expectedUserId: "alice" },
      body: expect.objectContaining({ tldw_history_selection_v1: selection })
    })
  )
  expect(frames).toEqual([
    { tldw_history_admission_v1: { input_message_id: "input" } },
    { tldw_message_id: "result", tldw_conversation_id: "chat" }
  ])
})

it("unparseable native stream is uncertainty instead of a silently successful truncated stream", async () => {
  mocks.bgStream.mockImplementation(async function* () {
    yield "broken frame"
  })
  const consume = async () => {
    for await (const _frame of new TldwApiClient().streamChatCompletion({
      model: "m",
      messages: [{ role: "user", content: "next" }],
      tldw_history_selection_v1: {} as any
    })) {
      /* consume */
    }
  }
  await expect(consume()).rejects.toThrow(
    "unparseable_native_completion_stream"
  )
})
