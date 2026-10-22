import { afterEach, describe, expect, it } from "vitest"
import {
  chatModelReadinessKey,
  getChatModelSendOutcome,
  isChatModelUnreachableError,
  recordChatModelSendOutcome,
  resetChatModelSendOutcomes,
  resolveChatModelReadinessState
} from "../chat-model-readiness"
import { toProviderHealthSignal } from "@/services/tldw/model-provider-availability"

afterEach(() => {
  resetChatModelSendOutcomes()
})

describe("chatModelReadinessKey", () => {
  it("treats provider aliases and the internal tldw prefix as the same model", () => {
    const key = chatModelReadinessKey("custom-openai-api", "local-uat-chat")
    expect(chatModelReadinessKey("custom_openai_api", "tldw:local-uat-chat")).toBe(key)
    expect(chatModelReadinessKey("tabbyapi", "m")).toBe(chatModelReadinessKey("tabby", "m"))
  })

  it("reads the provider from a provider-qualified model id", () => {
    expect(chatModelReadinessKey(undefined, "ollama:llama3")).toBe(
      chatModelReadinessKey("ollama", "llama3")
    )
  })

  it("keeps the same model id on different providers apart", () => {
    expect(chatModelReadinessKey("ollama", "shared")).not.toBe(
      chatModelReadinessKey("openai", "shared")
    )
  })

  it("returns null without a model id", () => {
    expect(chatModelReadinessKey("ollama", "  ")).toBeNull()
  })

  it("keeps Ollama models whose family is also a provider apart (CC-06)", () => {
    expect(chatModelReadinessKey("ollama", "mistral:7b")).not.toBe(
      chatModelReadinessKey("ollama", "qwen:7b")
    )
    expect(chatModelReadinessKey("ollama", "ollama:mistral:7b")).toBe(
      chatModelReadinessKey("ollama", "mistral:7b")
    )
    expect(chatModelReadinessKey(undefined, "ollama:mistral:7b")).toBe(
      chatModelReadinessKey("ollama", "mistral:7b")
    )
  })

  it("does not mark one Ollama model unreachable after another one fails", () => {
    recordChatModelSendOutcome("ollama", "mistral:7b", "unreachable")

    expect(getChatModelSendOutcome("ollama", "mistral:7b")).toBe("unreachable")
    expect(getChatModelSendOutcome("ollama", "qwen:7b")).toBeNull()
  })
})

describe("isChatModelUnreachableError", () => {
  it.each([
    ["a 502 response", { status: 502 }],
    ["a 5xx wrapped as the cause", new Error("Stream completion failed", { cause: { status: 503 } })],
    [
      "a provider_unavailable error code",
      { status: 400, details: { detail: { error_code: "provider_unavailable" } } }
    ],
    ["an in-stream provider error frame", { error: { code: "provider_unavailable", message: "down" } }],
    ["an HTTP status in the message", new Error("Request failed: HTTP 504 Gateway Timeout")]
  ])("is true for %s", (_label, error) => {
    expect(isChatModelUnreachableError(error)).toBe(true)
  })

  it.each([
    ["a validation error", { status: 422 }],
    ["an abort", new DOMException("The operation was aborted.", "AbortError")],
    ["a policy refusal", { error: { code: "model_not_allowed" } }],
    ["nothing", undefined]
  ])("is false for %s", (_label, error) => {
    expect(isChatModelUnreachableError(error)).toBe(false)
  })
})

describe("toProviderHealthSignal", () => {
  it("does not treat a never-used provider as working", () => {
    expect(toProviderHealthSignal({ status: "healthy", success_count: 0, failure_count: 0 })).toBe(
      "unknown"
    )
    expect(toProviderHealthSignal({ status: "unknown", success_count: 0 })).toBe("unknown")
    expect(toProviderHealthSignal(undefined)).toBe("unknown")
  })

  it("is working once the provider has answered", () => {
    expect(toProviderHealthSignal({ status: "healthy", success_count: 3 })).toBe("working")
  })

  it("is unreachable when the circuit is open or the latest calls failed", () => {
    expect(toProviderHealthSignal({ status: "circuit_open", success_count: 9 })).toBe("unreachable")
    expect(
      toProviderHealthSignal({ status: "healthy", success_count: 2, consecutive_failures: 1 })
    ).toBe("unreachable")
  })
})

describe("resolveChatModelReadinessState", () => {
  it("is not checked without any evidence", () => {
    expect(
      resolveChatModelReadinessState({ model: {}, provider: "ollama", modelId: "llama3" })
    ).toBe("not_checked")
  })

  it("uses the server's provider health", () => {
    expect(
      resolveChatModelReadinessState({
        model: { provider_health: "working" },
        provider: "ollama",
        modelId: "llama3"
      })
    ).toBe("working")
  })

  it("lets this session's latest send outcome win over the provider report", () => {
    recordChatModelSendOutcome("ollama", "tldw:llama3", "unreachable")

    expect(getChatModelSendOutcome("ollama", "llama3")).toBe("unreachable")
    expect(
      resolveChatModelReadinessState({
        model: { provider_health: "working" },
        provider: "ollama",
        modelId: "llama3"
      })
    ).toBe("unreachable")
  })
})
