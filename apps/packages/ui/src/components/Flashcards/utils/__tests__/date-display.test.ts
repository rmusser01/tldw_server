import { describe, expect, it } from "vitest"
import {
  formatFlashcardAbsoluteDateTime,
  formatFlashcardLongDateTime,
  formatFlashcardRelativeTime,
  formatFlashcardTimestampWithRelative,
  isFlashcardTimestampBefore,
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

  it("formats long local absolute timestamps for next-due labels", () => {
    const value = new Date(2026, 1, 18, 9, 5).toISOString()

    expect(formatFlashcardLongDateTime(value, { locale: "en-US" })).toBe(
      "Wednesday, February 18 at 9:05 AM"
    )
  })

  it.each([
    ["invalid-date", null],
    [null, null],
    ["2026-02-18T09:00:00Z", Date.parse("2026-02-18T09:00:00Z")],
    [
      Date.parse("2026-02-18T09:00:00Z") / 1000,
      Date.parse("2026-02-18T09:00:00Z")
    ],
    [
      new Date(2000, 0, 1, 0, 0).getTime(),
      new Date(2000, 0, 1, 0, 0).getTime()
    ],
    [
      new Date(1960, 0, 1, 0, 0).getTime() / 1000,
      new Date(1960, 0, 1, 0, 0).getTime()
    ],
    [
      new Date(1960, 0, 1, 0, 0).getTime(),
      new Date(1960, 0, 1, 0, 0).getTime()
    ]
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

  it("combines pre-2001 millisecond timestamps without reinterpreting them as seconds", () => {
    const timestamp = new Date(2000, 0, 1, 0, 0).getTime()
    const nowMs = new Date(2000, 0, 2, 0, 0).getTime()

    expect(formatFlashcardTimestampWithRelative(timestamp, { nowMs })).toEqual({
      absolute: "2000-01-01 00:00",
      relative: "a day ago",
      timestamp
    })
  })

  it.each([
    [new Date(2026, 1, 18, 11, 59).toISOString(), true],
    [new Date(2026, 1, 18, 12, 0).toISOString(), false],
    [new Date(2026, 1, 18, 12, 1).toISOString(), false],
    ["invalid-date", false]
  ])("checks whether %s is before the reference time", (value, expected) => {
    expect(isFlashcardTimestampBefore(value, STABLE_NOW_MS)).toBe(expected)
  })
})
