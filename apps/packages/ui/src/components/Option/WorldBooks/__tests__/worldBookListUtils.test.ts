import { readFileSync } from "node:fs"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"

import { describe, expect, it } from "vitest"
import {
  formatWorldBookLastModified,
  parseWorldBookTimestamp,
  UNKNOWN_LAST_MODIFIED_LABEL
} from "../worldBookListUtils"

describe("worldBookListUtils", () => {
  it.each([
    ["2026-02-18T09:00:00Z", 1771405200000],
    [1771405200, 1771405200000],
  ])("parses %s as a millisecond timestamp", (value, expected) => {
    expect(parseWorldBookTimestamp(value)).toBe(expected)
  })

  it("returns unknown-safe display values for null/invalid timestamps", () => {
    expect(parseWorldBookTimestamp("invalid-date")).toBeNull()
    expect(formatWorldBookLastModified(null)).toEqual({
      relative: UNKNOWN_LAST_MODIFIED_LABEL,
      absolute: null,
      timestamp: null
    })
  })

  it.each([
    ["2026-02-18T11:59:30Z", "a few seconds ago"],
    ["2026-02-18T11:59:00Z", "a minute ago"],
    ["2026-02-18T11:58:00Z", "2 minutes ago"],
    ["2026-02-18T09:00:00Z", "3 hours ago"],
    ["2026-02-17T12:00:00Z", "a day ago"],
    ["2026-02-19T12:00:00Z", "in a day"],
  ])("formats %s with dayjs-compatible relative text", (value, relative) => {
    const nowMs = Date.parse("2026-02-18T12:00:00Z")

    expect(
      formatWorldBookLastModified(value, { nowMs }).relative
    ).toBe(relative)
  })

  it("formats absolute timestamps from a stable now", () => {
    const nowMs = Date.parse("2026-02-18T12:00:00Z")

    expect(
      formatWorldBookLastModified("2026-02-18T09:00:00Z", { nowMs })
    ).toMatchObject({
      absolute: "2026-02-18 09:00:00 UTC",
      timestamp: 1771405200000
    })
  })

  it("does not depend on dayjs for display-only relative timestamps", () => {
    const testDir = dirname(fileURLToPath(import.meta.url))
    const source = readFileSync(resolve(testDir, "../worldBookListUtils.ts"), "utf8")

    expect(source).not.toContain("dayjs")
  })
})
