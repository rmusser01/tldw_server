import { describe, expect, it, vi } from "vitest"
import {
  DUPLICATE_SKIP_MESSAGE,
  QUICK_INGEST_ACCEPT_STRING,
  isDbMessageDuplicate,
} from "../constants"

describe("isDbMessageDuplicate", () => {
  it.each(["already exists", "O'Brian already exists", "line\nalready exists"])("does not misclassify successfully saved titles: %s", (title) => {
    expect(isDbMessageDuplicate({ db_message: `Media '${title}' added.` })).toBe(false)
    expect(isDbMessageDuplicate({ db_message: `Media '${title}' updated to new version.` })).toBe(false)
  })

  it("retains legacy and concurrent skip messages after a quoted title", () => {
    expect(isDbMessageDuplicate({ db_message: "Source already exists." })).toBe(true)
    expect(isDbMessageDuplicate({ db_message: "Media 'O'Brian already exists' already exists (concurrent insert). Overwrite not enabled." })).toBe(true)
  })

  it("returns true when db_message contains 'already exists'", () => {
    expect(
      isDbMessageDuplicate({ db_message: "Media 'test.pdf' already exists. Overwrite not enabled." })
    ).toBe(true)
  })

  it("returns true for concurrent insert variant", () => {
    expect(
      isDbMessageDuplicate({ db_message: "Media 'test.pdf' already exists (concurrent insert). Overwrite not enabled." })
    ).toBe(true)
  })

  it("is case-insensitive", () => {
    expect(
      isDbMessageDuplicate({ db_message: "Media 'test.pdf' ALREADY EXISTS." })
    ).toBe(true)
  })

  it("returns false when db_message is absent", () => {
    expect(isDbMessageDuplicate({})).toBe(false)
  })

  it("returns false for null input", () => {
    expect(isDbMessageDuplicate(null)).toBe(false)
  })

  it("returns false for undefined input", () => {
    expect(isDbMessageDuplicate(undefined)).toBe(false)
  })

  it("returns false when db_message is a non-duplicate string", () => {
    expect(
      isDbMessageDuplicate({ db_message: "Successfully persisted media." })
    ).toBe(false)
  })

  it("returns false when db_message is not a string", () => {
    expect(isDbMessageDuplicate({ db_message: 42 })).toBe(false)
  })
})

describe("DUPLICATE_SKIP_MESSAGE", () => {
  it("is a non-empty string", () => {
    expect(typeof DUPLICATE_SKIP_MESSAGE).toBe("string")
    expect(DUPLICATE_SKIP_MESSAGE.length).toBeGreaterThan(0)
  })

  it("mentions the Deep preset", () => {
    expect(DUPLICATE_SKIP_MESSAGE.toLowerCase()).toContain("deep")
  })
})

describe("QUICK_INGEST_ACCEPT_STRING", () => {
  const accepted = QUICK_INGEST_ACCEPT_STRING.split(",")

  it("advertises only document formats supported by backend processing", () => {
    expect(accepted).toEqual(
      expect.arrayContaining([
        ".pdf",
        ".txt",
        ".rtf",
        ".docx",
        ".md",
        ".markdown",
        ".html",
        ".htm",
        ".xhtml",
        ".xml",
        ".json"
      ])
    )
    expect(accepted).not.toContain(".doc")
    expect(accepted).not.toContain("application/msword")
  })
})

describe("Quick Ingest upload configuration", () => {
  it.each([
    [undefined, 250, "250 MB"],
    ["384", 384, "384 MB"],
    ["2048", 1024, "1 GB"],
    ["0", 250, "250 MB"],
    ["25.5", 250, "250 MB"],
    ["invalid", 250, "250 MB"],
  ] as const)("uses document limit %s within the API ceiling", async (value, limit, label) => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_QUICK_INGEST_DOCUMENT_MAX_MIB", value)
    vi.resetModules()
    try {
      const { QUICK_INGEST_DOCUMENT_LIMIT, QUICK_INGEST_AV_LIMIT } = await import("../constants")
      expect(QUICK_INGEST_DOCUMENT_LIMIT).toEqual({ bytes: limit * 1024 ** 2, label })
      expect(QUICK_INGEST_AV_LIMIT).toEqual({ bytes: 1024 ** 3, label: "1 GB" })
    } finally {
      vi.unstubAllEnvs()
    }
  })
})
