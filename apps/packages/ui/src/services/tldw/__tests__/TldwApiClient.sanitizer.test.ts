import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn(),
  tldwRequest: vi.fn(),
  storage: new Map<string, unknown>()
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.bgRequest(...args),
  bgUpload: vi.fn(),
  bgStream: vi.fn()
}))

vi.mock("@/services/tldw/request-core", async importOriginal => ({
  ...await importOriginal<typeof import("@/services/tldw/request-core")>(),
  tldwRequest: (...args: unknown[]) => mocks.tldwRequest(...args)
}))

vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({
    get: vi.fn(async (key: string) => mocks.storage.get(key) ?? null),
    set: vi.fn(async (key: string, value: unknown) => { mocks.storage.set(key, value) }),
    remove: vi.fn(async (key: string) => { mocks.storage.delete(key) })
  }),
  safeStorageSerde: {
    serialize: (value: unknown) => value,
    deserialize: (value: unknown) => value
  }
}))

import { TldwApiClient } from "@/services/tldw/TldwApiClient"

describe("TldwApiClient.createChatCompletion (non-streaming sanitizer)", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.storage.clear()
  })

  it("returns successful completion content verbatim even when it looks like an error", async () => {
    const content =
      "To handle this exception, wrap it in try/catch — see /Users/foo/bar.py:12"
    const completion = {
      id: "chatcmpl-1",
      object: "chat.completion",
      choices: [
        {
          index: 0,
          message: { role: "assistant", content },
          finish_reason: "stop"
        }
      ]
    }
    // bgRequest throws on non-2xx, so a resolved value is always a success body.
    mocks.bgRequest.mockResolvedValueOnce(completion)

    const client = new TldwApiClient()
    const res = await client.createChatCompletion({
      model: "gpt-test",
      messages: [{ role: "user", content: "How do I handle errors?" }]
    } as any)

    const body = await res.json()
    expect(body.choices[0].message.content).toBe(content)
    // The suspicious substrings must survive untouched.
    expect(body.choices[0].message.content).toContain("exception")
    expect(body.choices[0].message.content).toContain("/Users/foo/bar.py:12")
    expect(JSON.stringify(body)).not.toContain("Chat completion failed.")
  })

  it("preserves error-shaped keys inside successful assistant content", async () => {
    const completion = {
      choices: [
        {
          index: 0,
          message: {
            role: "assistant",
            content: "Here is a stack trace and a traceback for you."
          },
          finish_reason: "stop"
        }
      ]
    }
    mocks.bgRequest.mockResolvedValueOnce(completion)

    const client = new TldwApiClient()
    const res = await client.createChatCompletion({
      model: "gpt-test",
      messages: [{ role: "user", content: "show me a trace" }]
    } as any)

    const body = await res.json()
    expect(body.choices[0].message.content).toBe(
      "Here is a stack trace and a traceback for you."
    )
  })
})

describe("TldwApiClient.synthesizeSpeech (timeout)", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.storage.clear()
  })

  const configureClient = async (client: TldwApiClient) => {
    await client.updateConfig({
      serverUrl: "http://127.0.0.1:8000",
      apiKey: "test-api-key-123",
      authMode: "single-user"
    })
  }

  const findSpeechCall = () =>
    mocks.tldwRequest.mock.calls
      .map((call) => call[0] as any)
      .find((init) => init?.path === "/api/v1/audio/speech")

  it("uses a generous default timeout so long synthesis is not aborted", async () => {
    mocks.tldwRequest.mockResolvedValue({ ok: true, status: 200, data: new ArrayBuffer(8) })

    const client = new TldwApiClient()
    await configureClient(client)
    await client.synthesizeSpeech("Some long passage to render.")

    const speechCall = findSpeechCall()
    expect(speechCall).toBeTruthy()
    expect(speechCall.timeoutMs).toBeGreaterThanOrEqual(120000)
  })

  it("lets callers override the timeout", async () => {
    mocks.tldwRequest.mockResolvedValue({ ok: true, status: 200, data: new ArrayBuffer(8) })

    const client = new TldwApiClient()
    await configureClient(client)
    await client.synthesizeSpeech("hi", { timeoutMs: 5000 } as any)

    const speechCall = findSpeechCall()
    expect(speechCall).toBeTruthy()
    expect(speechCall.timeoutMs).toBe(5000)
  })
})
