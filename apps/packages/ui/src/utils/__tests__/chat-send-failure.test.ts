import i18n from "i18next"
import { beforeAll, describe, expect, it } from "vitest"

import { ChatStreamTimeoutError } from "@/services/tldw/chat-timeouts"
import {
  buildAssistantErrorContent,
  buildSendFailureContent,
  decodeChatErrorPayload
} from "../chat-error-message"
import {
  describeSendFailure,
  findHttpFailure,
  isDefinitiveSendFailure,
  isProviderUnavailableFailure,
  providerDisplayLabel,
  sendFailureError,
  sendFailureTarget
} from "../chat-send-failure"

const providerUnavailable = () =>
  Object.assign(new Error("The chat service provider is currently unavailable."), {
    status: 502,
    details: {
      detail: {
        error_code: "provider_unavailable",
        message: "The chat service provider is currently unavailable."
      }
    }
  })

describe("chat send failures (CC-02, CC-N2, CM-06)", () => {
  beforeAll(async () => {
    await i18n.init({ lng: "en", resources: {} })
  })

  it("reads status, code and message through wrapped errors", () => {
    const wrapped = new Error("Stream completion failed", { cause: providerUnavailable() })
    expect(findHttpFailure(wrapped)).toEqual({
      status: 502,
      code: "provider_unavailable",
      message: "The chat service provider is currently unavailable."
    })
    expect(findHttpFailure(new Error("network error"))).toEqual({})
  })

  it("names the provider and model a selected model id points at", () => {
    expect(sendFailureTarget("custom-openai-api:local-uat-chat")).toEqual({
      provider: "custom-openai-api",
      model: "local-uat-chat"
    })
    expect(providerDisplayLabel("custom-openai-api")).toBe("Custom OpenAI API")
    expect(providerDisplayLabel("not-a-provider")).toBe("not-a-provider")
  })

  it("treats HTTP errors and client startup timeouts as definitive, but not conflicts", () => {
    const failure = describeSendFailure(providerUnavailable(), { model: "gemma3:1b" })
    expect(failure).toEqual({
      status: 502,
      code: "provider_unavailable",
      message: "The chat service provider is currently unavailable.",
      model: "gemma3:1b"
    })
    expect(isDefinitiveSendFailure(failure)).toBe(true)
    expect(isDefinitiveSendFailure({ status: 409, code: "history_branch_changed" })).toBe(false)
    expect(describeSendFailure(new TypeError("network error"))).toBeUndefined()

    const timeout = describeSendFailure(new ChatStreamTimeoutError("startup", 120_000), { model: "m" })
    expect(timeout).toEqual({ model: "m", timeout_ms: 120_000 })
    expect(isDefinitiveSendFailure(timeout, "client")).toBe(true)
    // A server-settled reply may still be saved after the client stops waiting.
    expect(isDefinitiveSendFailure(timeout, "server")).toBe(false)
  })

  it("recognises provider outages by code, status or message, but not bad credentials", () => {
    expect(isProviderUnavailableFailure({ code: "provider_error" })).toBe(true)
    expect(isProviderUnavailableFailure({ status: 503 })).toBe(true)
    expect(isProviderUnavailableFailure({ message: "The chat service provider is currently unavailable." })).toBe(true)
    expect(isProviderUnavailableFailure({ status: 502, code: "provider_authentication_failed" })).toBe(false)
    expect(isProviderUnavailableFailure({ status: 500 })).toBe(false)
  })

  it("words a provider outage as the provider's, not the tldw server's, and offers Switch model", () => {
    const payload = decodeChatErrorPayload(
      buildAssistantErrorContent("", providerUnavailable(), {
        provider: "custom-openai-api",
        model: "local-uat-chat"
      })
    )
    expect(payload).toMatchObject({
      summary: "Couldn't reach Custom OpenAI API (local-uat-chat)",
      hint: "The tldw server is fine, but this model's provider didn't respond. Start the provider or switch models.",
      recoveryAction: "open-model-selector",
      recoveryLabel: "Switch model"
    })
    expect(payload?.detail).toContain("HTTP 502")
    expect(payload?.summary).not.toMatch(/Something went wrong/)
    expect(
      decodeChatErrorPayload(buildAssistantErrorContent("", providerUnavailable()))?.summary
    ).toBe("Couldn't reach the model's provider")
  })

  it("rebuilds the same message from a retained failure", () => {
    const failure = describeSendFailure(providerUnavailable(), sendFailureTarget("custom-openai-api:local-uat-chat"))!
    expect(sendFailureError(failure)).toMatchObject({ status: 502 })
    expect(decodeChatErrorPayload(buildSendFailureContent(failure))?.summary).toBe(
      "Couldn't reach Custom OpenAI API (local-uat-chat)"
    )
    expect(
      decodeChatErrorPayload(buildSendFailureContent({ model: "gemma3:1b", timeout_ms: 30_000 }))?.summary
    ).toBe("No response from gemma3:1b after 30s.")
    expect(
      decodeChatErrorPayload(buildSendFailureContent({ model: "gemma3:1b", timeout_ms: 120_000 }))?.summary
    ).toBe("No response from gemma3:1b after 2:00.")
  })
})
