import { describe, expect, it } from "vitest"

import {
  buildFriendlyErrorMessage,
  decodeChatErrorPayload,
  encodeChatErrorPayload,
  ImageSupportUnconfirmedError
} from "../chat-error-message"

describe("chat error message recovery actions", () => {
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
