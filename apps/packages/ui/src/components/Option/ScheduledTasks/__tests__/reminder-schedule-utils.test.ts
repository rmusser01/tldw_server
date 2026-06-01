import { describe, expect, it } from "vitest"

import {
  buildDailyCron,
  buildWeeklyCron,
  datetimeLocalToIsoString,
  getDefaultReminderTimezone,
  validateCronExpression
} from "../reminder-schedule-utils"

describe("reminder schedule utilities", () => {
  it("returns a browser IANA timezone or UTC fallback", () => {
    expect(getDefaultReminderTimezone()).toMatch(/^(UTC|[A-Za-z_]+\/[A-Za-z0-9_+\-]+(?:\/[A-Za-z0-9_+\-]+)?)$/)
  })

  it("converts a one-time datetime-local string to an ISO string", () => {
    expect(datetimeLocalToIsoString("2026-03-21T10:00")).toBe(
      new Date(2026, 2, 21, 10, 0).toISOString()
    )
  })

  it("builds a daily 09:00 cron expression", () => {
    expect(buildDailyCron(9, 0)).toBe("0 9 * * *")
  })

  it("builds a weekly Monday 09:00 cron expression", () => {
    expect(buildWeeklyCron("MON", 9, 0)).toBe("0 9 * * MON")
  })

  it("returns a field-count validation error for invalid custom cron", () => {
    expect(validateCronExpression("0 9 * *")).toEqual({
      valid: false,
      error: "Cron must have exactly 5 fields"
    })
  })
})
