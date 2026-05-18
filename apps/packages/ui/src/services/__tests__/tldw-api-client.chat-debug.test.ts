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
    expect(snapshot?.endpoint).toMatch(/^\/api\/v1\/chats\/42\/complete-v2(?:\?|$)/)
    expect((snapshot?.body as any)?.model).toBe("kimi-k2")
    expect((snapshot?.body as any)?.include_character_context).toBe(true)
    expect((snapshot?.body as any)?.stream).toBe(true)
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
