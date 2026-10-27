/**
 * CC-01 (#3111): the model chip's readiness must reflect every chat send, not
 * only the plain-chat path through TldwChatService. Character chats stream
 * through /chats/{id}/complete-v2, and native-history character sends call
 * streamChatCompletion directly, so the outcome is recorded at the transport.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

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

import { TldwApiClient } from "@/services/tldw/TldwApiClient"
import {
  getChatModelSendOutcome,
  resetChatModelSendOutcomes
} from "@/utils/chat-model-readiness"

const httpError = (status: number) =>
  Object.assign(new Error(`Request failed (${status})`), { status })

const consume = async (stream: AsyncIterable<unknown>) => {
  const chunks: unknown[] = []
  for await (const chunk of stream) chunks.push(chunk)
  return chunks
}

const characterPayload = { model: "llama3.2:3b", provider: "ollama" }
const chatRequest = {
  model: "llama3.2:3b",
  api_provider: "ollama",
  messages: [{ role: "user" as const, content: "hello" }]
}

beforeEach(() => {
  mocks.bgRequest.mockReset()
  mocks.bgStream.mockReset()
  resetChatModelSendOutcomes()
})

afterEach(() => {
  resetChatModelSendOutcomes()
})

describe("chat model send outcomes recorded at the transport (CC-01)", () => {
  it("marks the model unreachable when a character-chat stream gets a 502", async () => {
    mocks.bgStream.mockImplementation(async function* () {
      throw httpError(502)
    })

    await expect(
      consume(new TldwApiClient().streamCharacterChatCompletion("chat-1", characterPayload))
    ).rejects.toThrow()

    expect(getChatModelSendOutcome("ollama", "llama3.2:3b")).toBe("unreachable")
  })

  it("marks the model working once a character-chat stream answers", async () => {
    mocks.bgStream.mockImplementation(async function* () {
      yield '{"choices":[{"delta":{"content":"Hi there"}}]}'
    })

    await consume(new TldwApiClient().streamCharacterChatCompletion("chat-1", characterPayload))

    expect(getChatModelSendOutcome("ollama", "llama3.2:3b")).toBe("working")
  })

  it("marks the model unreachable from an in-stream provider error frame", async () => {
    mocks.bgStream.mockImplementation(async function* () {
      yield '{"error":{"code":"provider_unavailable","message":"down"}}'
    })

    await consume(new TldwApiClient().streamCharacterChatCompletion("chat-1", characterPayload))

    expect(getChatModelSendOutcome("ollama", "llama3.2:3b")).toBe("unreachable")
  })

  it("records a non-streaming character turn", async () => {
    mocks.bgRequest.mockRejectedValueOnce(httpError(503))
    await expect(
      new TldwApiClient().completeCharacterChatTurn("chat-1", characterPayload)
    ).rejects.toThrow()
    expect(getChatModelSendOutcome("ollama", "llama3.2:3b")).toBe("unreachable")
  })

  it("records a direct streamChatCompletion send (native-history path)", async () => {
    mocks.bgStream.mockImplementationOnce(async function* () {
      throw httpError(502)
    })
    await expect(consume(new TldwApiClient().streamChatCompletion(chatRequest))).rejects.toThrow()
    expect(getChatModelSendOutcome("ollama", "llama3.2:3b")).toBe("unreachable")

    mocks.bgStream.mockImplementationOnce(async function* () {
      yield '{"choices":[{"delta":{"content":"Back up"}}]}'
    })
    await consume(new TldwApiClient().streamChatCompletion(chatRequest))
    expect(getChatModelSendOutcome("ollama", "llama3.2:3b")).toBe("working")
  })

  it("records a non-streaming chat completion", async () => {
    mocks.bgRequest.mockResolvedValueOnce({
      choices: [{ message: { role: "assistant", content: "ok" } }]
    })
    await new TldwApiClient().createChatCompletion(chatRequest)
    expect(getChatModelSendOutcome("ollama", "llama3.2:3b")).toBe("working")
  })

  it("does not treat a validation error or an abort as evidence either way", async () => {
    mocks.bgStream.mockImplementationOnce(async function* () {
      throw httpError(422)
    })
    await expect(
      consume(new TldwApiClient().streamCharacterChatCompletion("chat-1", characterPayload))
    ).rejects.toThrow()
    mocks.bgStream.mockImplementationOnce(async function* () {
      throw new DOMException("The operation was aborted.", "AbortError")
    })
    await expect(
      consume(new TldwApiClient().streamCharacterChatCompletion("chat-1", characterPayload))
    ).rejects.toThrow()

    expect(getChatModelSendOutcome("ollama", "llama3.2:3b")).toBeNull()
  })
})
