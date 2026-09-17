import { describe, expect, it } from "vitest"

import {
  buildFriendlyErrorMessage,
  decodeChatErrorPayload,
  encodeChatErrorPayload,
  ImageSupportUnconfirmedError
} from "../chat-error-message"

describe("chat error message recovery actions", () => {
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
})
