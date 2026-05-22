import { describe, expect, it, vi } from "vitest"
import {
  getLocalTimezone,
  parseQuickSetupExtraSourceUrls,
  QUICK_SETUP_DEFAULT_VALUES,
  resolveQuickSetupSchedule,
  toQuickSetupJobPayload,
  toQuickSetupSourcePayload
} from "../quick-setup"

describe("watchlists overview quick setup helpers", () => {
  it("provides sensible defaults", () => {
    expect(QUICK_SETUP_DEFAULT_VALUES).toEqual({
      sourceName: "",
      sourceUrl: "",
      extraSourceUrls: "",
      sourceType: "rss",
      monitorName: "",
      schedulePreset: "daily",
      runNow: true,
      setupGoal: "briefing",
      includeAudioBriefing: true
    })
  })

  it("resolves preset schedules into cron/timezone", () => {
    const timezoneSpy = vi
      .spyOn(Intl, "DateTimeFormat")
      .mockImplementation(() => ({
        resolvedOptions: () => ({ timeZone: "America/New_York" })
      }) as Intl.DateTimeFormat)

    expect(resolveQuickSetupSchedule("none")).toEqual({})
    expect(resolveQuickSetupSchedule("hourly")).toEqual({
      schedule_expr: "0 * * * *",
      timezone: "America/New_York"
    })
    expect(resolveQuickSetupSchedule("daily")).toEqual({
      schedule_expr: "0 8 * * *",
      timezone: "America/New_York"
    })
    expect(resolveQuickSetupSchedule("weekdays")).toEqual({
      schedule_expr: "0 8 * * MON-FRI",
      timezone: "America/New_York"
    })

    timezoneSpy.mockRestore()
  })

  it("resolves variable cadence drafts into existing schedule fields", () => {
    const timezoneSpy = vi
      .spyOn(Intl, "DateTimeFormat")
      .mockImplementation(() => ({
        resolvedOptions: () => ({ timeZone: "America/Los_Angeles" })
      }) as Intl.DateTimeFormat)

    expect(resolveQuickSetupSchedule({ kind: "manual" })).toEqual({})
    expect(resolveQuickSetupSchedule({ kind: "interval", every: 5, unit: "hour" })).toEqual({
      schedule_expr: "0 */5 * * *",
      timezone: "America/Los_Angeles"
    })
    expect(resolveQuickSetupSchedule({ kind: "interval", every: 30, unit: "minute" })).toEqual({
      schedule_expr: "*/30 * * * *",
      timezone: "America/Los_Angeles"
    })
    expect(resolveQuickSetupSchedule({ kind: "daily", time: "07:45" })).toEqual({
      schedule_expr: "45 7 * * *",
      timezone: "America/Los_Angeles"
    })
    expect(resolveQuickSetupSchedule({ kind: "weekdays", time: "08:15" })).toEqual({
      schedule_expr: "15 8 * * MON-FRI",
      timezone: "America/Los_Angeles"
    })
    expect(resolveQuickSetupSchedule({ kind: "weekly", weekday: "mon", time: "08:00" })).toEqual({
      schedule_expr: "0 8 * * MON",
      timezone: "America/Los_Angeles"
    })
    expect(resolveQuickSetupSchedule({ kind: "advanced", cron: "15 6 * * WED" })).toEqual({
      schedule_expr: "15 6 * * WED",
      timezone: "America/Los_Angeles"
    })

    timezoneSpy.mockRestore()
  })

  it("does not serialize invalid advanced cron drafts", () => {
    const timezoneSpy = vi
      .spyOn(Intl, "DateTimeFormat")
      .mockImplementation(() => ({
        resolvedOptions: () => ({ timeZone: "America/Los_Angeles" })
      }) as Intl.DateTimeFormat)

    expect(resolveQuickSetupSchedule({ kind: "advanced", cron: "15 6 *" })).toEqual({})
    expect(resolveQuickSetupSchedule({ kind: "advanced", cron: "15 6 * * WED;rm" })).toEqual({})
    expect(resolveQuickSetupSchedule({ kind: "advanced", cron: "61 6 * * WED" })).toEqual({})
    expect(resolveQuickSetupSchedule({ kind: "advanced", cron: "? 6 * * WED" })).toEqual({})
    expect(resolveQuickSetupSchedule({ kind: "advanced", cron: "*/1 * * * *" })).toEqual({})

    timezoneSpy.mockRestore()
  })

  it("falls back to UTC when timezone is unavailable", () => {
    const timezoneSpy = vi
      .spyOn(Intl, "DateTimeFormat")
      .mockImplementation(() => ({
        resolvedOptions: () => ({ timeZone: "" })
      }) as Intl.DateTimeFormat)

    expect(getLocalTimezone()).toBe("UTC")
    timezoneSpy.mockRestore()
  })

  it("builds trimmed source and monitor payloads", () => {
    expect(
      toQuickSetupSourcePayload({
        sourceName: " My Feed ",
        sourceUrl: " https://example.com/rss.xml ",
        sourceType: "rss"
      })
    ).toEqual({
      name: "My Feed",
      url: "https://example.com/rss.xml",
      source_type: "rss",
      active: true
    })

    const timezoneSpy = vi
      .spyOn(Intl, "DateTimeFormat")
      .mockImplementation(() => ({
        resolvedOptions: () => ({ timeZone: "UTC" })
      }) as Intl.DateTimeFormat)

    expect(
      toQuickSetupJobPayload(
        {
          monitorName: " Morning Monitor ",
          schedulePreset: "daily",
          setupGoal: "triage",
          includeAudioBriefing: false
        },
        [42]
      )
    ).toEqual({
      name: "Morning Monitor",
      scope: { sources: [42] },
      active: true,
      schedule_expr: "0 8 * * *",
      timezone: "UTC"
    })

    expect(
      toQuickSetupJobPayload(
        {
          monitorName: " Morning Monitor ",
          schedulePreset: "daily",
          setupGoal: "briefing",
          includeAudioBriefing: true
        },
        [42, 42, 77]
      )
    ).toEqual({
      name: "Morning Monitor",
      scope: { sources: [42, 77] },
      active: true,
      schedule_expr: "0 8 * * *",
      timezone: "UTC",
      output_prefs: {
        template_name: "briefing_markdown",
        template: {
          default_name: "briefing_markdown"
        },
        generate_audio: true
      }
    })

    timezoneSpy.mockRestore()
  })

  it("attaches selected Watchlist id when building collection payloads", () => {
    const timezoneSpy = vi
      .spyOn(Intl, "DateTimeFormat")
      .mockImplementation(() => ({
        resolvedOptions: () => ({ timeZone: "UTC" })
      }) as Intl.DateTimeFormat)

    expect(
      toQuickSetupSourcePayload(
        {
          sourceName: " Security Feed ",
          sourceUrl: " https://example.com/security.xml ",
          sourceType: "rss"
        },
        42
      )
    ).toEqual({
      name: "Security Feed",
      url: "https://example.com/security.xml",
      source_type: "rss",
      active: true,
      watchlist_id: 42
    })

    expect(
      toQuickSetupJobPayload(
        {
          monitorName: " Security Monitor ",
          schedulePreset: "daily",
          setupGoal: "briefing",
          includeAudioBriefing: false
        },
        [101],
        42
      )
    ).toEqual({
      name: "Security Monitor",
      scope: { sources: [101] },
      active: true,
      schedule_expr: "0 8 * * *",
      timezone: "UTC",
      watchlist_id: 42,
      output_prefs: {
        template_name: "briefing_markdown",
        template: {
          default_name: "briefing_markdown"
        },
        generate_audio: false
      }
    })

    timezoneSpy.mockRestore()
  })

  it("parses extra source URLs from newline/comma-delimited values", () => {
    expect(
      parseQuickSetupExtraSourceUrls(
        "https://example.com/a.xml\ninvalid\nhttps://example.com/b.xml, https://example.com/a.xml"
      )
    ).toEqual(["https://example.com/a.xml", "https://example.com/b.xml"])
  })
})
