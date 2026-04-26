import { describe, expect, it } from "vitest"
import {
  DUPLICATE_SKIP_MESSAGE,
  isDbMessageDuplicate,
} from "../constants"

describe("isDbMessageDuplicate", () => {
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
