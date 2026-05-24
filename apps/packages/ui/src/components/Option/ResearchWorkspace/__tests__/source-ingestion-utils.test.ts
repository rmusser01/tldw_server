import { describe, expect, it } from "vitest"
import {
  DEFAULT_SOURCE_UPLOAD_MAX_SIZE_MB,
  mapSourceIngestionError,
  parseSourceCreatedAt,
  resolveSourceUploadMaxSizeMb,
  validateSourceUploadFile
} from "../SourcesPane/source-ingestion-utils"

describe("source-ingestion-utils", () => {
  it("falls back to default upload size limit when override is invalid", () => {
    expect(resolveSourceUploadMaxSizeMb("")).toBe(DEFAULT_SOURCE_UPLOAD_MAX_SIZE_MB)
    expect(resolveSourceUploadMaxSizeMb("not-a-number")).toBe(
      DEFAULT_SOURCE_UPLOAD_MAX_SIZE_MB
    )
    expect(resolveSourceUploadMaxSizeMb(-1)).toBe(
      DEFAULT_SOURCE_UPLOAD_MAX_SIZE_MB
    )
  })

  it("accepts valid upload size limit overrides", () => {
    expect(resolveSourceUploadMaxSizeMb("750")).toBe(750)
    expect(resolveSourceUploadMaxSizeMb(1024)).toBe(1024)
  })

  it("rejects unsupported file types with explicit code", () => {
    const result = validateSourceUploadFile(
      {
        name: "archive.zip",
        type: "application/zip",
        size: 1024
      },
      10 * 1024 * 1024
    )

    expect(result).toEqual({
      valid: false,
      code: "unsupported_file_type",
      fileName: "archive.zip"
    })
  })

  it("rejects files larger than the configured max size", () => {
    const result = validateSourceUploadFile(
      {
        name: "large.pdf",
        type: "application/pdf",
        size: 2_000
      },
      1_000
    )

    expect(result).toEqual({
      valid: false,
      code: "file_too_large",
      fileName: "large.pdf",
      maxSizeBytes: 1_000
    })
  })

  it("maps common status/network errors to actionable ingestion messages", () => {
    expect(
      mapSourceIngestionError({ status: 413, message: "Payload too large" })
    ).toContain("too large")
    expect(
      mapSourceIngestionError({ status: 415, message: "Unsupported Media Type" })
    ).toContain("not supported")
    expect(
      mapSourceIngestionError(new Error("NetworkError when attempting to fetch resource."))
    ).toContain("Unable to reach the server")
    expect(mapSourceIngestionError({ status: 429 })).toContain("Too many requests")
  })

  it("normalizes source created-at timestamps from iso, seconds, and millis values", () => {
    expect(parseSourceCreatedAt("2026-02-18T09:00:00.000Z")?.toISOString()).toBe(
      "2026-02-18T09:00:00.000Z"
    )
    expect(parseSourceCreatedAt(1_739_869_200)?.toISOString()).toBe(
      "2025-02-18T09:00:00.000Z"
    )
    expect(parseSourceCreatedAt("1739869200000")?.toISOString()).toBe(
      "2025-02-18T09:00:00.000Z"
    )
    expect(parseSourceCreatedAt("not-a-date")).toBeUndefined()
  })
})
