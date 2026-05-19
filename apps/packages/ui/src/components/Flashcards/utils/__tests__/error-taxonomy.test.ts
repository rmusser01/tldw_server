import { describe, expect, it } from "vitest"
import {
  extractFlashcardsErrorStatus,
  formatFlashcardsUiErrorMessage,
  mapFlashcardsUiError
} from "../error-taxonomy"

const buildOptions = () => ({
  operation: "saving this card",
  fallback: "Unable to save flashcard."
})

describe("flashcards error taxonomy", () => {
  it("maps version conflicts to explicit reload guidance", () => {
    const mapped = mapFlashcardsUiError(
      {
        status: 409,
        message: "Version mismatch: expected 3 got 4"
      },
      buildOptions()
    )

    expect(mapped.code).toBe("FLASHCARDS_VERSION_CONFLICT")
    expect(mapped.actionLabel).toBe("Refresh")
    expect(formatFlashcardsUiErrorMessage(mapped)).toContain("[FLASHCARDS_VERSION_CONFLICT]")
  })

  it("maps network failures to retry guidance", () => {
    const mapped = mapFlashcardsUiError(
      new Error("Failed to fetch (POST /api/v1/flashcards/review)"),
      {
        operation: "submitting your review",
        fallback: "Unable to submit review."
      }
    )

    expect(mapped.code).toBe("FLASHCARDS_NETWORK")
    expect(mapped.message).toContain("submitting your review")
    expect(mapped.actionLabel).toBe("Retry")
  })

  it("maps timeout-style transport failures to network guidance", () => {
    const mapped = mapFlashcardsUiError(
      new Error("Extension messaging timeout after 20000ms"),
      {
        operation: "submitting your review",
        fallback: "Unable to submit review."
      }
    )

    expect(mapped.code).toBe("FLASHCARDS_NETWORK")
    expect(mapped.actionLabel).toBe("Retry")
  })

  it("maps validation failures to fix-input guidance", () => {
    const mapped = mapFlashcardsUiError(
      {
        status: 400,
        message: "Invalid cloze"
      },
      buildOptions()
    )

    expect(mapped.code).toBe("FLASHCARDS_VALIDATION")
    expect(mapped.message).toContain("Fix the input and retry.")
  })

  it("maps server failures to retry guidance", () => {
    const mapped = mapFlashcardsUiError(
      {
        status: 500,
        message: "Failed to update flashcard"
      },
      buildOptions()
    )

    expect(mapped.code).toBe("FLASHCARDS_SERVER")
    expect(mapped.message).toContain("Server error")
  })

  it("extracts HTTP status from nested or message-based errors", () => {
    expect(
      extractFlashcardsErrorStatus({
        response: { statusCode: 422 }
      })
    ).toBe(422)
    expect(extractFlashcardsErrorStatus(new Error("Request failed: 503"))).toBe(503)
  })

  it("falls back to unknown when no recognizable error shape is present", () => {
    const mapped = mapFlashcardsUiError(
      {
        detail: "Something odd happened"
      },
      buildOptions()
    )

    expect(mapped.code).toBe("FLASHCARDS_UNKNOWN")
    expect(mapped.message).toContain("Please retry")
  })
})
