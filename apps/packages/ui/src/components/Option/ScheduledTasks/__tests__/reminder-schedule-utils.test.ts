import { describe, expect, it } from "vitest"

import {
  buildDailyCron,
  buildWeeklyCron,
  datetimeLocalToIsoString,
  getDefaultReminderTimezone,
  getRecurringPreviewCopy,
  parseReminderCron,
  validateCronExpression,
  validateReminderTimezone
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

  it("allows APScheduler nth-weekday cron tokens", () => {
    expect(validateCronExpression("0 9 * * mon#2")).toEqual({
      valid: true,
      error: null
    })
  })

  it("rejects unsupported question-mark cron tokens", () => {
    expect(validateCronExpression("0 9 * * ?")).toEqual({
      valid: false,
      error: "Cron field '?' is not supported by the scheduler."
    })
  })

  it("rejects APScheduler-invalid numeric weekday 7", () => {
    expect(validateCronExpression("0 9 * * 7")).toEqual({
      valid: false,
      error: "Cron day of week must be between 0 and 6."
    })
  })

  it("parses numeric weekdays using APScheduler weekday numbering", () => {
    expect(parseReminderCron("0 9 * * 0")).toMatchObject({
      preset: "weekly",
      weekday: "MON"
    })
    expect(parseReminderCron("0 9 * * 6")).toMatchObject({
      preset: "weekly",
      weekday: "SUN"
    })
  })

  it("rejects APScheduler-invalid reversed named weekday ranges", () => {
    expect(validateCronExpression("0 9 * * fri-mon")).toEqual({
      valid: false,
      error: "Cron day of week range start must be less than or equal to the end."
    })
  })

  it("rejects APScheduler-invalid reversed named month ranges", () => {
    expect(validateCronExpression("0 9 * dec-jan mon")).toEqual({
      valid: false,
      error: "Cron month range start must be less than or equal to the end."
    })
  })

  it("rejects APScheduler-invalid numeric-to-named ranges", () => {
    expect(validateCronExpression("0 9 * 1-dec mon")).toEqual({
      valid: false,
      error: "Cron month range cannot start with a number and end with a name."
    })
    expect(validateCronExpression("0 9 * * 0-sun")).toEqual({
      valid: false,
      error: "Cron day of week range cannot start with a number and end with a name."
    })
  })

  it("allows APScheduler-valid name-to-numeric and open named ranges", () => {
    expect(validateCronExpression("0 9 * dec-1 mon")).toEqual({
      valid: true,
      error: null
    })
    expect(validateCronExpression("0 9 * * sun-0")).toEqual({
      valid: true,
      error: null
    })
    expect(validateCronExpression("0 9 * jan- mon")).toEqual({
      valid: true,
      error: null
    })
    expect(validateCronExpression("0 9 * * mon-")).toEqual({
      valid: true,
      error: null
    })
  })

  it("allows APScheduler-valid nth weekday bounds", () => {
    expect(validateCronExpression("0 9 * * mon#0")).toEqual({
      valid: true,
      error: null
    })
    expect(validateCronExpression("0 9 * * mon#6")).toEqual({
      valid: true,
      error: null
    })
  })

  it("rejects scheduler-invalid cron ranges", () => {
    expect(validateCronExpression("99 99 * * *")).toEqual({
      valid: false,
      error: "Cron minute must be between 0 and 59."
    })
  })

  it("rejects non-cron words in numeric scheduler fields", () => {
    expect(validateCronExpression("banana banana * * *")).toEqual({
      valid: false,
      error: "Cron minute must be a number, range, step, list, or wildcard."
    })
  })

  it("validates reminder timezones with browser Intl support", () => {
    expect(validateReminderTimezone("America/Los_Angeles")).toEqual({
      valid: true,
      error: null
    })
    expect(validateReminderTimezone("Mars/Olympus")).toEqual({
      valid: false,
      error: "Timezone must be a valid IANA timezone."
    })
  })

  it("describes daily recurring preview as a next-run oriented cadence instead of raw cron output", () => {
    const preview = getRecurringPreviewCopy("daily", "0 9 * * *", "America/Los_Angeles")

    expect(preview).toContain("Next run:")
    expect(preview).toContain("daily 09:00")
    expect(preview).toContain("America/Los_Angeles")
    expect(preview).not.toBe("Daily schedule: 0 9 * * * (America/Los_Angeles)")
  })
})
