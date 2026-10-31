import i18n from "i18next"
import { beforeAll, describe, expect, it } from "vitest"

import { ChatStreamTimeoutError } from "@/services/tldw/chat-timeouts"
import {
  buildAssistantErrorContent,
  buildFriendlyErrorMessage,
  decodeChatErrorPayload,
  encodeChatErrorPayload,
  ImageSupportUnconfirmedError
} from "../chat-error-message"

describe("chat error message recovery actions", () => {
  beforeAll(async () => { await i18n.init({ lng: "en", resources: {} }) })
  it.each([
    new Error('HTTP 400: {"detail":{"code":"model_not_available","message":"Selected model is unavailable"}}'),
    { detail: { error_code: "model_not_available", message: "Selected model is unavailable" } },
    Object.assign(new Error("Selected model is unavailable"), { details: { detail: { error_code: "model_not_available", message: "Selected model is unavailable" } }, status: 400 })
  ])("offers model recovery for the unavailable-model wire contract: %s", error => {
    expect(decodeChatErrorPayload(buildFriendlyErrorMessage(error))).toMatchObject({
      summary: "The selected model is not available.",
      hint: "Choose a different model or refresh the model list, then try again.",
      recoveryAction: "open-model-selector",
      recoveryLabel: "Choose another model"
    })
  })

  it("sanitizes unavailable-model details without including provider secrets or local paths", () => {
    const error = Object.assign(new Error("Model /Users/example/private.gguf unavailable at https://provider.test?api_key=synthetic-sensitive-value"), {
      details: { detail: { error_code: "model_not_available" } }, status: 400
    })
    const encoded = buildFriendlyErrorMessage(error)
    expect(decodeChatErrorPayload(encoded)?.recoveryAction).toBe("open-model-selector")
    expect(encoded).not.toContain("synthetic-sensitive-value")
    expect(encoded).not.toContain("/Users/example")
    expect(encoded).not.toContain("provider.test")
  })

  it("does not classify unrelated server failures as unavailable models", () => {
    expect(decodeChatErrorPayload(buildFriendlyErrorMessage(new Error("HTTP 500: database unavailable")))?.recoveryAction).toBeUndefined()
  })

  it("finds unavailable-model guidance through ordinary-chat error causes", () => {
    const serverError = Object.assign(new Error("Model /Users/example/private.gguf unavailable at https://provider.test?api_key=synthetic-sensitive-value"), {
      details: { detail: { error_code: "model_not_available" } }, status: 400
    })
    const wrapped = new Error("Stream completion failed", { cause: serverError })
    const payload = decodeChatErrorPayload(buildFriendlyErrorMessage(wrapped))
    expect(payload).toMatchObject({
      summary: "The selected model is not available.",
      recoveryAction: "open-model-selector"
    })
    expect(payload?.detail).toContain("Model")
    expect(payload?.detail).not.toMatch(/synthetic-sensitive-value|\/Users\/example|provider\.test/)
  })

  it("keeps unrelated wrapped errors generic and terminates on circular causes", () => {
    const error = new Error("Stream completion failed", { cause: new Error("HTTP 500: database unavailable") })
    Object.assign(error.cause as Error, { cause: error })
    expect(decodeChatErrorPayload(buildFriendlyErrorMessage(error))).toMatchObject({
      detail: "Stream completion failed"
    })
    expect(decodeChatErrorPayload(buildFriendlyErrorMessage(error))?.recoveryAction).toBeUndefined()
  })
  it("offers model selection for unconfirmed image support without blaming the server", () => {
    const error = new ImageSupportUnconfirmedError(false)
    expect(decodeChatErrorPayload(buildFriendlyErrorMessage(error))).toMatchObject({
      summary: "Image support is not confirmed for this model.",
      hint: "Choose a model that supports images, or start a new text-only conversation.",
      recoveryAction: "open-model-selector",
      recoveryLabel: "Choose another model",
      serverRetryRequired: false
    })
  })

  it.each([false, true])("round-trips trusted local retry provenance: %s", serverRetryRequired => {
    const result = decodeChatErrorPayload(buildFriendlyErrorMessage(new ImageSupportUnconfirmedError(serverRetryRequired)))
    expect(result?.serverRetryRequired).toBe(serverRetryRequired)
  })

  it.each([undefined, null, "false", 0])("does not infer retry provenance from a missing or malformed field: %s", serverRetryRequired => {
    const encoded = encodeChatErrorPayload({ summary: "Failure", hint: "Retry", detail: "", serverRetryRequired } as never)
    expect(decodeChatErrorPayload(encoded)?.serverRetryRequired).toBeUndefined()
  })

  it.each([
    "Image support is not confirmed for this model.",
    { message: "Image support is not confirmed for this model.", serverRetryRequired: false },
    Object.assign(new Error("Image support is not confirmed for this model."), { name: "ImageSupportUnconfirmedError", serverRetryRequired: false })
  ])("does not trust provider text or error-object lookalikes", error => {
    expect(decodeChatErrorPayload(buildFriendlyErrorMessage(error))?.serverRetryRequired).toBeUndefined()
  })

  it("routes unavailable-model errors to the compact model selector", () => {
    const encoded = buildFriendlyErrorMessage(
      new Error("model_not_found: no such model")
    )

    expect(decodeChatErrorPayload(encoded)).toMatchObject({
      summary: "The selected model is not available.",
      recoveryAction: "open-model-selector",
      recoveryLabel: "Choose another model"
    })
  })

  it("routes empty model responses to the compact model selector", () => {
    const encoded = buildFriendlyErrorMessage(
      new Error("No response text was returned")
    )

    expect(decodeChatErrorPayload(encoded)).toMatchObject({
      summary: "No response was returned.",
      recoveryAction: "open-model-selector",
      recoveryLabel: "Choose another model"
    })
  })

  it.each([false, true])("recognizes numeric HTTP413 through direct and wrapped errors: %s", wrapped => {
    const serverError = Object.assign(new Error("413 /Users/example/private.png https://provider.test?api_key=synthetic-sensitive-value"), { status: 413 })
    const error = wrapped ? new Error("Stream completion failed", { cause: serverError }) : serverError
    const payload = decodeChatErrorPayload(buildFriendlyErrorMessage(error))
    expect(payload).toMatchObject({
      summary: "Your message is too large.",
      hint: "Use a smaller image or shorten your message, then try again.",
      detail: "413 [redacted-path] [server-url]"
    })
    expect(JSON.stringify(payload)).not.toMatch(/synthetic-sensitive-value|\/Users\/example|provider\.test/)
  })

  it("keeps unrelated status and quoted size text generic through circular causes", () => {
    const inner = Object.assign(new Error("Model description mentions HTTP413 and too large"), { status: 500 })
    const wrapped = new Error("Stream completion failed", { cause: inner })
    Object.assign(inner, { cause: wrapped })
    expect(decodeChatErrorPayload(buildFriendlyErrorMessage(wrapped))?.summary)
      .toBe("Something went wrong while talking to your tldw server.")
  })

})

describe("provider failures name the model (CC-N2)", () => {
  beforeAll(async () => { await i18n.init({ lng: "en", resources: {} }) })

  const providerDown = () =>
    new Error("Stream completion failed", {
      cause: {
        status: 502,
        details: {
          detail: {
            error_code: "provider_unavailable",
            message: "The chat service provider is currently unavailable."
          }
        }
      }
    })

  it("says which model didn't answer and offers Switch model", () => {
    expect(
      decodeChatErrorPayload(
        buildFriendlyErrorMessage(providerDown(), { modelLabel: "gemma3:1b (Ollama)" })
      )
    ).toMatchObject({
      summary: "Couldn't reach gemma3:1b (Ollama).",
      hint: "Your tldw server is fine, but this model's provider didn't respond. Start the provider or switch models.",
      recoveryAction: "open-model-selector",
      recoveryLabel: "Switch model"
    })
  })

  it("doesn't claim the server is fine for a bare gateway error, which may be a proxy", () => {
    const bare = new Error("Stream completion failed", { cause: { status: 502 } })
    expect(
      decodeChatErrorPayload(buildFriendlyErrorMessage(bare, { modelLabel: "gemma3:1b (Ollama)" }))
    ).toMatchObject({
      summary: "Couldn't reach gemma3:1b (Ollama).",
      hint: "The model's provider, or a proxy in front of your tldw server, didn't respond. Check that both are running, or switch models.",
      recoveryAction: "open-model-selector"
    })
  })

  it("names the provider generically when the model is unknown", () => {
    expect(decodeChatErrorPayload(buildFriendlyErrorMessage(providerDown()))?.summary).toBe(
      "Couldn't reach this model's provider."
    )
  })

  it("passes the model label through buildAssistantErrorContent", () => {
    expect(
      decodeChatErrorPayload(
        buildAssistantErrorContent("", providerDown(), { modelLabel: "mock-gpt-4o (Custom OpenAI API)" })
      )?.summary
    ).toBe("Couldn't reach mock-gpt-4o (Custom OpenAI API).")
  })

  it("keeps a server 503 for missing credentials out of the provider wording", () => {
    const error = new Error("Stream completion failed", {
      cause: { status: 503, detail: { error_code: "missing_provider_credentials" } }
    })
    expect(decodeChatErrorPayload(buildFriendlyErrorMessage(error))?.summary).not.toMatch(/Couldn't reach/)
  })
})

describe("a reply that never starts names the model and the wait (CM-06)", () => {
  beforeAll(async () => { await i18n.init({ lng: "en", resources: {} }) })

  it("says which model didn't respond, and after how long", () => {
    expect(
      decodeChatErrorPayload(
        buildFriendlyErrorMessage(new ChatStreamTimeoutError("startup", 30_000), {
          modelLabel: "mock-gpt-4o (Custom OpenAI API)"
        })
      )?.summary
    ).toBe("No response from mock-gpt-4o (Custom OpenAI API) after 30 seconds.")
  })

  it("keeps the general wording without a model", () => {
    expect(
      decodeChatErrorPayload(buildFriendlyErrorMessage(new ChatStreamTimeoutError("startup", 30_000)))
        ?.summary
    ).toBe("The model did not start responding in time.")
  })
})
