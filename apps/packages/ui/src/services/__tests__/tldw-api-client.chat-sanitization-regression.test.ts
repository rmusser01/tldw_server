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

import {
  TldwApiClient,
  type ChatCompletionRequest
} from "@/services/tldw/TldwApiClient"
import { chatRagMethods } from "@/services/tldw/domains/chat-rag"
import type { ChatCompletionRequest } from "@/services/tldw/TldwApiClient"

const request: ChatCompletionRequest = {
  model: "auto",
  messages: [{ role: "user", content: "hello" }]
}

describe("chat completion response sanitization regressions", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("wraps plain text completions as JSON content for TldwApiClient", async () => {
    mocks.bgRequest.mockResolvedValue("hello world")

    const client = new TldwApiClient()
    const response = await client.createChatCompletion(request)

    expect(response.headers.get("content-type")).toContain("application/json")
    await expect(response.json()).resolves.toEqual({ content: "hello world" })
  })

  it("sanitizes suspicious plain text responses for TldwApiClient", async () => {
    mocks.bgRequest.mockResolvedValue(
      "Traceback: /Users/private/stack.txt\nRuntimeError: boom"
    )

    const client = new TldwApiClient()
    const response = await client.createChatCompletion(request)
    const payload = await response.json()

    expect(payload).toEqual({
      error: "Chat completion failed.",
      errors: ["One or more internal errors were suppressed."]
    })
  })

  it("sanitizes nested stack-bearing strings for TldwApiClient", async () => {
    mocks.bgRequest.mockResolvedValue({
      id: "resp-1",
      nested: {
        message: "Traceback: /Users/private/stack.txt\nRuntimeError: boom",
        note: "/Users/private/app.py:77"
      },
      choices: [{ message: { content: "safe assistant response" } }]
    })

    const client = new TldwApiClient()
    const response = await client.createChatCompletion(request)
    const payload = await response.json()

    expect(payload.nested).toEqual({
      message: "Chat completion failed.",
      note: "Chat completion failed."
    })
    expect(payload.choices[0].message.content).toBe("safe assistant response")
  })

  it("sanitizes suspicious plain text responses for chatRagMethods", async () => {
    mocks.bgRequest.mockResolvedValue(
      "Traceback: /Users/private/stack.txt\nRuntimeError: boom"
    )

    const response = await chatRagMethods.createChatCompletion.call(
      {} as never,
      request
    )
    const payload = await response.json()

    expect(payload).toEqual({
      error: "Chat completion failed.",
      errors: ["One or more internal errors were suppressed."]
    })
  })

  it("sanitizes nested stack-bearing strings for chatRagMethods", async () => {
    mocks.bgRequest.mockResolvedValue({
      id: "resp-2",
      nested: {
        message: "Traceback: /Users/private/stack.txt\nRuntimeError: boom",
        note: "/Users/private/app.py:77"
      },
      choices: [{ message: { content: "safe assistant response" } }]
    })

    const response = await chatRagMethods.createChatCompletion.call(
      {} as never,
      request
    )
    const payload = await response.json()

    expect(payload.nested).toEqual({
      message: "Chat completion failed.",
      note: "Chat completion failed."
    })
    expect(payload.choices[0].message.content).toBe("safe assistant response")
  })
})
