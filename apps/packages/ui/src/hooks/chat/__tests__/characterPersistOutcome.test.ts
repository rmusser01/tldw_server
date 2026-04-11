import { describe, expect, it } from "vitest"

import { resolveSavedDegradedCharacterPersist } from "../characterPersistOutcome"

describe("resolveSavedDegradedCharacterPersist", () => {
  it("extracts a saved degraded outcome from a 503 error payload", () => {
    const error = Object.assign(new Error("degraded"), {
      status: 503,
      details: {
        detail: {
          code: "persist_validation_degraded",
          saved: true,
          assistant_message_id: "assistant-server-1"
        }
      }
    })

    expect(resolveSavedDegradedCharacterPersist(error)).toEqual({
      saved: true,
      assistantMessageId: "assistant-server-1",
      version: undefined
    })
  })

  it("extracts a saved degraded outcome from a top-level FastAPI detail payload", () => {
    const error = Object.assign(new Error("degraded"), {
      status: "503",
      detail: {
        code: "persist_validation_degraded",
        saved: true,
        assistant_message_id: "assistant-server-2",
        version: 3
      }
    })

    expect(resolveSavedDegradedCharacterPersist(error)).toEqual({
      saved: true,
      assistantMessageId: "assistant-server-2",
      version: 3
    })
  })

  it("returns null for ordinary persistence errors", () => {
    const error = Object.assign(new Error("boom"), {
      status: 500,
      details: { detail: "boom" }
    })

    expect(resolveSavedDegradedCharacterPersist(error)).toBeNull()
  })
})
