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

  it("retries updateChat once with the latest version after conflict", async () => {
    mocks.bgRequest.mockImplementation(
      async (request: { path?: string; method?: string; body?: unknown }) => {
        if (
          request.path === "/api/v1/chats/abc?expected_version=4" &&
          request.method === "PUT"
        ) {
          throw Object.assign(new Error("Version conflict"), { status: 409 })
        }
        if (
          request.path === "/api/v1/chats/abc?scope_type=global" &&
          request.method === "GET"
        ) {
          return {
            id: "abc",
            title: "Latest",
            created_at: "2026-02-18T00:00:00Z",
            updated_at: "2026-02-18T00:01:00Z",
            version: 7
          }
        }
        if (
          request.path ===
            "/api/v1/chats/abc?scope_type=global&expected_version=7" &&
          request.method === "PUT"
        ) {
          return {
            id: "abc",
            title: "Renamed",
            created_at: "2026-02-18T00:00:00Z",
            updated_at: "2026-02-18T00:02:00Z",
            version: 8
          }
        }
        throw new Error(`Unexpected request: ${request.method} ${request.path}`)
      }
    )

    const client = new TldwApiClient()
    const result = await client.updateChat(
      "abc",
      { title: "Renamed" },
      { expectedVersion: 4 }
    )

    const calls = mocks.bgRequest.mock.calls.map(([request]) => request as {
      path?: string
      method?: string
    })
    expect(calls).toEqual([
      expect.objectContaining({
        path: "/api/v1/chats/abc?scope_type=global&expected_version=4",
        method: "PUT"
      }),
      expect.objectContaining({
        path: "/api/v1/chats/abc?scope_type=global",
        method: "GET"
      }),
      expect.objectContaining({
        path: "/api/v1/chats/abc?scope_type=global&expected_version=7",
        method: "PUT"
      })
    ])
    expect(result.version).toBe(8)
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

  it("sanitizes leaky background payload fields when creating a chat completion", async () => {
    mocks.bgRequest.mockResolvedValue({
      id: "resp-1",
      object: "chat.completion",
      error: "trace=/Users/private/stack.txt",
      details: "/Users/private/stack.txt",
      nested: {
        traceback: "Traceback: /Users/private/stack.txt",
        items: [{ exception: "boom" }]
      }
    })

    const client = new TldwApiClient()
    const response = await client.createChatCompletion({
      model: "auto",
      messages: [{ role: "user", content: "hello" }]
    })

    const payload = await response.json()

    expect(payload.error).not.toContain("/Users/private")
    expect(payload.details).toBeUndefined()
    expect(payload.nested.traceback).toBeUndefined()
    expect(payload.nested.items[0].exception).toBeUndefined()
  })
})
