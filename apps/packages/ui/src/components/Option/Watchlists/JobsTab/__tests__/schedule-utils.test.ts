import { describe, expect, it } from "vitest"
import {
  buildCronFromPreset,
  createDefaultPresetState,
  formatScheduleTimeValue,
  parseScheduleTime,
  parsePresetFromCron,
  validateCronFormat,
  validateCronSchedule
} from "../schedule-utils"

describe("schedule-utils", () => {
  it("builds cron expressions from preset state", () => {
    expect(
      buildCronFromPreset({
        preset: "interval",
        intervalValue: 15,
        intervalUnit: "minutes",
        hour: 9,
        minute: 0,
        weekday: "MON"
      })
    ).toBe("*/15 * * * *")
    expect(
      buildCronFromPreset({
        preset: "interval",
        intervalValue: 5,
        intervalUnit: "hours",
        hour: 9,
        minute: 0,
        weekday: "MON"
      })
    ).toBe("0 */5 * * *")
    expect(
      buildCronFromPreset({
        preset: "daily",
        intervalValue: 1,
        intervalUnit: "hours",
        hour: 8,
        minute: 30,
        weekday: "MON"
      })
    ).toBe("30 8 * * *")
    expect(
      buildCronFromPreset({
        preset: "weekdays",
        intervalValue: 1,
        intervalUnit: "hours",
        hour: 8,
        minute: 0,
        weekday: "MON"
      })
    ).toBe("0 8 * * MON-FRI")
    expect(
      buildCronFromPreset({
        preset: "weekly",
        intervalValue: 1,
        intervalUnit: "hours",
        hour: 7,
        minute: 45,
        weekday: "FRI"
      })
    ).toBe("45 7 * * FRI")
  })

  it("parses supported cron patterns into preset state", () => {
    expect(parsePresetFromCron("*/15 * * * *")).toMatchObject({
      preset: "interval",
      intervalUnit: "minutes",
      intervalValue: 15
    })
    expect(parsePresetFromCron("0 * * * *")).toMatchObject({
      preset: "interval",
      intervalUnit: "hours",
      intervalValue: 1,
      minute: 0
    })
    expect(parsePresetFromCron("15 */6 * * *")).toMatchObject({
      preset: "interval",
      intervalUnit: "hours",
      intervalValue: 6,
      minute: 15
    })
    expect(parsePresetFromCron("30 9 * * *")).toEqual({
      preset: "daily",
      intervalValue: 1,
      intervalUnit: "hours",
      hour: 9,
      minute: 30,
      weekday: "MON"
    })
    expect(parsePresetFromCron("0 8 * * MON-FRI")).toEqual({
      preset: "weekdays",
      intervalValue: 1,
      intervalUnit: "hours",
      hour: 8,
      minute: 0,
      weekday: "MON"
    })
    expect(parsePresetFromCron("5 14 * * TUE")).toEqual({
      preset: "weekly",
      intervalValue: 1,
      intervalUnit: "hours",
      hour: 14,
      minute: 5,
      weekday: "TUE"
    })
  })

  it("returns null for unsupported cron patterns", () => {
    expect(parsePresetFromCron("0 8 1 * *")).toBeNull()
    expect(parsePresetFromCron("*/1 * * * *")).toBeNull()
    expect(parsePresetFromCron("*/4 * * * *")).toBeNull()
    expect(parsePresetFromCron("")).toBeNull()
  })

  it("creates a stable default state", () => {
    expect(createDefaultPresetState()).toEqual({
      preset: "daily",
      intervalValue: 1,
      intervalUnit: "hours",
      hour: 9,
      minute: 0,
      weekday: "MON"
    })
  })

  it("normalizes shared cadence time values", () => {
    expect(parseScheduleTime("7:05")).toEqual({ hour: 7, minute: 5 })
    expect(parseScheduleTime("24:00")).toEqual({ hour: 8, minute: 0 })
    expect(formatScheduleTimeValue("7:05")).toBe("07:05")
    expect(formatScheduleTimeValue("bad")).toBe("08:00")
  })

  it("validates cron format and minimum frequency consistently", () => {
    expect(validateCronFormat("15 6 * * WED")).toBeNull()
    expect(validateCronFormat("15 6 *")).toBe("field_count")
    expect(validateCronFormat("15 6 * * WED;rm")).toBe("invalid_token")
    expect(validateCronSchedule("*/1 * * * *")).toBe("too_frequent")
    expect(validateCronSchedule("*/5 * * * *")).toBeNull()
  })
})
