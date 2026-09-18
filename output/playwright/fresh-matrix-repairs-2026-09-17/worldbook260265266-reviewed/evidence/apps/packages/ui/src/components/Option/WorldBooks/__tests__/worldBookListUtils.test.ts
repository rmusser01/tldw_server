import { readFileSync } from "node:fs"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"

import { describe, expect, it } from "vitest"
import {
  formatWorldBookLastModified,
  parseWorldBookTimestamp,
  UNKNOWN_LAST_MODIFIED_LABEL
} from "../worldBookListUtils"

const STABLE_NOW_MS = Date.parse("2026-02-18T12:00:00Z")
const beforeStableNow = (deltaMs: number): string =>
  new Date(STABLE_NOW_MS - deltaMs).toISOString()

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
    expect(
      formatWorldBookLastModified(value, { nowMs: STABLE_NOW_MS }).relative
    ).toBe(relative)
  })

  it.each([
    [beforeStableNow(89_500), "a minute ago"],
    [beforeStableNow(90_000), "2 minutes ago"],
    [beforeStableNow(89.5 * 60_000), "an hour ago"],
    [beforeStableNow(90 * 60_000), "2 hours ago"],
    [beforeStableNow(35.5 * 60 * 60_000), "a day ago"],
    [beforeStableNow(36 * 60 * 60_000), "2 days ago"],
    [beforeStableNow(540 * 24 * 60 * 60_000), "a year ago"],
    [beforeStableNow(550 * 24 * 60 * 60_000), "2 years ago"],
  ])("formats rounding boundary %s as %s", (value, relative) => {
    expect(
      formatWorldBookLastModified(value, { nowMs: STABLE_NOW_MS }).relative
    ).toBe(relative)
  })

  it("formats absolute timestamps from a stable now", () => {
    expect(
      formatWorldBookLastModified("2026-02-18T09:00:00Z", { nowMs: STABLE_NOW_MS })
    ).toMatchObject({
      absolute: "2026-02-18 09:00:00 UTC",
      timestamp: 1771405200000
    })
  })

  it.each(["UTC", "America/Los_Angeles", "Pacific/Kiritimati"])(
    "renders an explicit World Book API instant consistently in %s",
    browserTimezone => {
      const originalTimezone = process.env.TZ
      process.env.TZ = browserTimezone
      try {
        expect(
          formatWorldBookLastModified("2026-09-17T23:51:02.389993Z", {
            nowMs: Date.parse("2026-09-17T23:51:12Z")
          })
        ).toMatchObject({
          relative: "a few seconds ago",
          absolute: "2026-09-17 23:51:02 UTC",
          timestamp: Date.parse("2026-09-17T23:51:02.389Z")
        })
      } finally {
        if (originalTimezone === undefined) delete process.env.TZ
        else process.env.TZ = originalTimezone
      }
    }
  )

  it.each([
    "2026-09-17T23:51:02.389993Z",
    "2026-09-17T23:51:02.389993+00:00",
    "2026-09-17T16:51:02.389993-07:00",
    "2026-09-18T05:36:02.389993+05:45"
  ])("preserves the instant from an explicit API offset: %s", value => {
    expect(parseWorldBookTimestamp(value)).toBe(Date.parse(value))
  })

  it("does not depend on dayjs for display-only relative timestamps", () => {
    const testDir = dirname(fileURLToPath(import.meta.url))
    const source = readFileSync(resolve(testDir, "../worldBookListUtils.ts"), "utf8")
    const dayjsImportPattern = /^\s*import\s+(?:type\s+)?(?:.+?\s+from\s+)?["']dayjs(?:\/[^"']*)?["']/m

    expect(source).not.toMatch(dayjsImportPattern)
  })
})
