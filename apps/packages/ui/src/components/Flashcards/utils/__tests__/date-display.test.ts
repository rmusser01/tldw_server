import { describe, expect, it } from "vitest"
import {
  formatFlashcardAbsoluteDateTime,
  formatFlashcardRelativeTime,
  formatFlashcardTimestampWithRelative,
  parseFlashcardTimestamp
} from "../date-display"

const STABLE_NOW_MS = new Date(2026, 1, 18, 12, 0).getTime()
const beforeStableNow = (deltaMs: number): string =>
  new Date(STABLE_NOW_MS - deltaMs).toISOString()

describe("flashcard date display utilities", () => {
  it("formats local absolute timestamps as YYYY-MM-DD HH:mm", () => {
    const value = new Date(2026, 1, 20, 9, 5).toISOString()

    expect(formatFlashcardAbsoluteDateTime(value)).toBe("2026-02-20 09:05")
  })

  it.each([
    ["invalid-date", null],
    [null, null],
    ["2026-02-18T09:00:00Z", Date.parse("2026-02-18T09:00:00Z")],
  ])("parses %s to %s", (value, expected) => {
    expect(parseFlashcardTimestamp(value)).toBe(expected)
  })

  it.each([
    ["2026-02-18T11:59:30", "a few seconds ago"],
    ["2026-02-18T11:59:00", "a minute ago"],
    ["2026-02-18T11:58:00", "2 minutes ago"],
    ["2026-02-18T09:00:00", "3 hours ago"],
    ["2026-02-17T12:00:00", "a day ago"],
    ["2026-02-19T12:00:00", "in a day"],
  ])("formats %s with dayjs-compatible relative text", (value, expected) => {
    expect(formatFlashcardRelativeTime(value, { nowMs: STABLE_NOW_MS })).toBe(
      expected
    )
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
  ])("formats rounding boundary %s as %s", (value, expected) => {
    expect(formatFlashcardRelativeTime(value, { nowMs: STABLE_NOW_MS })).toBe(
      expected
    )
  })

  it("combines absolute and relative labels for valid timestamps", () => {
    expect(
      formatFlashcardTimestampWithRelative("2026-02-18T09:00:00", {
        nowMs: STABLE_NOW_MS
      })
    ).toEqual({
      absolute: "2026-02-18 09:00",
      relative: "3 hours ago",
      timestamp: new Date(2026, 1, 18, 9, 0).getTime()
    })
  })
})
