import { describe, expect, it } from "vitest"

import {
  buildFriendlyErrorMessage,
  decodeChatErrorPayload
} from "../chat-error-message"

describe("chat error message recovery actions", () => {
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
