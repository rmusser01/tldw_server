import { afterEach, describe, expect, it } from "vitest"
import {
  chatModelReadinessKey,
  getChatModelSendOutcome,
  isChatModelUnreachableError,
  isChatProviderFailureError,
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

describe("isChatProviderFailureError (CC-N2)", () => {
  it.each([
    [
      "a provider_unavailable code wrapped as the cause",
      new Error("Stream completion failed", {
        cause: { status: 502, details: { detail: { error_code: "provider_unavailable" } } }
      })
    ],
    ["a provider_error code", { status: 500, detail: { error_code: "provider_error" } }],
    ["an in-stream provider error frame", { error: { code: "provider_unavailable", message: "down" } }],
    ["a 502 with no error code", { status: 502 }],
    ["a 504 in the message", new Error("Request failed: HTTP 504 Gateway Timeout")]
  ])("is true for %s", (_label, error) => {
    expect(isChatProviderFailureError(error)).toBe(true)
  })

  // The server also answers 503 for missing credentials, routing failures
  // and while it is busy; none of those mean the provider didn't answer.
  it.each([
    ["a 503 for missing credentials", { status: 503, detail: { error_code: "missing_provider_credentials" } }],
    ["a bare 503", { status: 503 }],
    ["a 500 from the server", new Error("HTTP 500: database unavailable")],
    ["an unavailable model", { status: 400, detail: { error_code: "model_not_available" } }],
    ["an abort", new DOMException("The operation was aborted.", "AbortError")],
    ["nothing", undefined]
  ])("is false for %s", (_label, error) => {
    expect(isChatProviderFailureError(error)).toBe(false)
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

  it("is unreachable when the circuit is open or the server's failure streak is reached", () => {
    expect(toProviderHealthSignal({ status: "circuit_open", success_count: 9 })).toBe("unreachable")
    expect(toProviderHealthSignal({ status: "unhealthy", success_count: 9 })).toBe("unreachable")
    expect(
      toProviderHealthSignal({ status: "healthy", success_count: 2, consecutive_failures: 3 })
    ).toBe("unreachable")
  })

  // The chat path records a failure for client-side errors too (a 400 for an
  // over-long prompt, a content filter); one or two of those must not mark
  // every model of the provider unreachable for every user.
  it("makes no claim after one or two failures below the server's threshold", () => {
    expect(
      toProviderHealthSignal({ status: "healthy", success_count: 2, consecutive_failures: 1 })
    ).toBe("unknown")
    expect(
      toProviderHealthSignal({ status: "healthy", success_count: 2, consecutive_failures: 2 })
    ).toBe("unknown")
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
