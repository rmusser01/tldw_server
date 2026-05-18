import { describe, expect, it } from "vitest"
import {
  buildCronFromPreset,
  createDefaultPresetState,
  parsePresetFromCron
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
})
