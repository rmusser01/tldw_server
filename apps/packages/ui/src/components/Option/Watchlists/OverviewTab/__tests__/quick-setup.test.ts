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
        template_name: "briefing_md",
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
