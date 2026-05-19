import { describe, expect, it } from "vitest"

import {
  completedIngestJobIndicatesFailure,
  completedIngestJobIndicatesSkipped,
  extractCompletedIngestJobError,
  extractCompletedIngestJobMediaId,
} from "@/services/tldw/ingest-job-results"

describe("ingest job result helpers", () => {
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

  it("surfaces root-level completed error metadata", () => {
    const payload = {
      status: "completed",
      error_message: "Quota exceeded"
    }

    expect(completedIngestJobIndicatesFailure(payload)).toBe(true)
    expect(extractCompletedIngestJobError(payload)).toBe("Quota exceeded")
  })
})
