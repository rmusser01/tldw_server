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

const request: ChatCompletionRequest = {
  model: "auto",
  messages: [{ role: "user", content: "hello" }]
}

describe("successful chat completion response preservation", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("preserves plain text completions for TldwApiClient", async () => {
    mocks.bgRequest.mockResolvedValue("hello world")

    const client = new TldwApiClient()
    const response = await client.createChatCompletion(request)

    expect(response.headers.get("content-type")).toContain("application/json")
    await expect(response.json()).resolves.toBe("hello world")
  })

  it("preserves successful error-like text for TldwApiClient", async () => {
    const content = "Traceback: /Users/private/stack.txt\nRuntimeError: boom"
    mocks.bgRequest.mockResolvedValue(content)

    const client = new TldwApiClient()
    const response = await client.createChatCompletion(request)
    const payload = await response.json()

    expect(payload).toBe(content)
  })

  it("preserves nested successful content for TldwApiClient", async () => {
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
      message: "Traceback: /Users/private/stack.txt\nRuntimeError: boom",
      note: "/Users/private/app.py:77"
    })
    expect(payload.choices[0].message.content).toBe("safe assistant response")
  })

  it("preserves successful error-like text for chatRagMethods", async () => {
    const content = "Traceback: /Users/private/stack.txt\nRuntimeError: boom"
    mocks.bgRequest.mockResolvedValue(content)

    const response = await chatRagMethods.createChatCompletion.call(
      {} as never,
      request
    )
    const payload = await response.json()

    expect(payload).toBe(content)
  })

  it("preserves nested successful content for chatRagMethods", async () => {
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
      message: "Traceback: /Users/private/stack.txt\nRuntimeError: boom",
      note: "/Users/private/app.py:77"
    })
    expect(payload.choices[0].message.content).toBe("safe assistant response")
  })
})
