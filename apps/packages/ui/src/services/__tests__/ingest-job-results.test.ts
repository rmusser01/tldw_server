import { describe, expect, it } from "vitest"

import {
  completedIngestJobIndicatesFailure,
  completedIngestJobIndicatesSkipped,
  extractCompletedIngestJobError,
  extractCompletedIngestJobMediaId,
  extractCompletedIngestJobWarning,
} from "@/services/tldw/ingest-job-results"

describe("ingest job result helpers", () => {
  it("surfaces saved-source processing warnings instead of reporting full success", () => {
    const payload = {
      status: "completed",
      result: { status: "Warning", db_id: 1, warnings: ["Analysis failed: model is required"] }
    }
    expect(completedIngestJobIndicatesFailure(payload)).toBe(false)
    expect(extractCompletedIngestJobError(payload)).toBeUndefined()
    expect(extractCompletedIngestJobWarning(payload)).toBe("Analysis failed: model is required")
    expect(extractCompletedIngestJobMediaId(payload)).toBe(1)
  })

  it.each([undefined, null, "", " ", 0, -1, Number.NaN])("does not promote an unsaved Warning with media ID %s", (mediaId) => {
    const payload = { status: "Warning", media_id: mediaId, warnings: ["Analysis failed"] }
    expect(completedIngestJobIndicatesFailure(payload)).toBe(true)
    expect(extractCompletedIngestJobWarning(payload)).toBeUndefined()
  })

  it.each([
    { error: "Explicit failure" },
    { detail: "Explicit failure" },
    { errors: ["Explicit failure"] },
  ])("keeps explicit failures ahead of saved-source warnings: %j", (failure) => {
    const payload = { status: "Warning", media_id: 1, warnings: ["Analysis warning"], ...failure }
    expect(completedIngestJobIndicatesFailure(payload)).toBe(true)
    expect(extractCompletedIngestJobWarning(payload)).toBeUndefined()
  })

  it.each(["Error", "cancelled", "canceled", "unknown"])("does not infer a saved warning or clean success from %s", (status) => {
    const payload = { status, media_id: 1, warnings: ["Analysis warning"] }
    expect(completedIngestJobIndicatesFailure(payload)).toBe(true)
    expect(extractCompletedIngestJobWarning(payload)).toBeUndefined()
  })

  it("deduplicates warning details and keeps an empty warning list distinct from clean success", () => {
    expect(extractCompletedIngestJobWarning({ status: "Warning", media_id: 1, warnings: ["Analysis failed", "Analysis failed", "", "Chunk omitted"] })).toBe("Analysis failed\nChunk omitted")
    expect(extractCompletedIngestJobWarning({ status: "Warning", media_id: 1 })).toBeTruthy()
  })

  it("does not borrow a media ID from a mixed aggregate result to promote a Warning", () => {
    const payload = { status: "Warning", warnings: ["Some items failed"], results: [{ status: "Success", media_id: 1 }, { status: "Error", error: "Failed to save" }] }
    expect(completedIngestJobIndicatesFailure(payload)).toBe(true)
    expect(extractCompletedIngestJobWarning(payload)).toBeUndefined()
  })

  it.each(["error_message", "cancellation_reason"])("retains terminal %s over a nested saved Warning", (field) => {
    const payload = { status: "completed", [field]: "Terminal failure", result: { status: "Warning", media_id: 1, warnings: ["Analysis warning"] } }
    expect(completedIngestJobIndicatesFailure(payload)).toBe(true)
    expect(extractCompletedIngestJobWarning(payload)).toBeUndefined()
  })
  it("treats completed jobs with nested error payloads as failures", () => {
    const payload = {
      status: "completed",
      result: {
        status: "Error",
        error: "Downloader failed"
      }
    }

    expect(completedIngestJobIndicatesFailure(payload)).toBe(true)
    expect(extractCompletedIngestJobError(payload)).toBe("Downloader failed")
  })

  it("treats duplicate-complete payloads as skipped instead of failed", () => {
    const payload = {
      status: "completed",
      result: {
        status: "duplicate",
        message: "Item already exists in the database.",
        db_id: 321
      }
    }

    expect(completedIngestJobIndicatesSkipped(payload)).toBe(true)
    expect(completedIngestJobIndicatesFailure(payload)).toBe(false)
    expect(extractCompletedIngestJobMediaId(payload)).toBe(321)
  })

  it.each(["failed", "skipped"])("finds persisted media after a %s first batch item", (status) => {
    expect(extractCompletedIngestJobMediaId({
      result: { results: [{ status }, { status: "Success", mediaId: "saved-2" }] }
    })).toBe("saved-2")
  })

  it("scans all identifier aliases and skips invalid identifiers in order", () => {
    expect(extractCompletedIngestJobMediaId([
      null, { media_id: false }, { media_id: " ", mediaId: 0, db_id: -1 },
      { media_id: Number.NaN }, { media_id: {}, db_id: 321 }, { media_id: 456 }
    ])).toBe(321)
  })

  it("prefers a valid top-level persistence identifier over batch results", () => {
    expect(extractCompletedIngestJobMediaId({ media_id: "", db_id: 123, results: [{ db_id: 321 }] })).toBe(123)
  })

  it("extracts the persisted media id from a media/add result list", () => {
    expect(
      extractCompletedIngestJobMediaId({
        results: [{ status: "Success", db_id: 321 }]
      })
    ).toBe(321)
  })

  it("extracts the persisted media id when media/add returns the list directly", () => {
    expect(
      extractCompletedIngestJobMediaId([
        { status: "Success", db_id: 321 }
      ])
    ).toBe(321)
  })

  it("surfaces root-level completed error metadata", () => {
    const payload = {
      status: "completed",
      error_message: "Quota exceeded"
    }

    expect(completedIngestJobIndicatesFailure(payload)).toBe(true)
    expect(extractCompletedIngestJobError(payload)).toBe("Quota exceeded")
  })

  it("surfaces backend errors arrays from persist-style responses", () => {
    const payload = {
      status: "persist-ok",
      total_articles: 1,
      stored_articles: 0,
      errors: ["Failed to extract: https://example.com/article"]
    }

    expect(completedIngestJobIndicatesFailure(payload)).toBe(true)
    expect(extractCompletedIngestJobError(payload)).toBe(
      "Failed to extract: https://example.com/article"
    )
  })
})
