import { describe, expect, it } from "vitest"
import {
  applyWatchlistSetupPreset,
  buildWatchlistSetupJobPayload,
  buildWatchlistSetupPlan,
  parseSetupSourceUrls,
  WATCHLIST_SETUP_PRESETS
} from "../watchlist-setup-model"

describe("watchlist setup model", () => {
  it("provides concrete presets without creating content alert rules", () => {
    expect(Object.keys(WATCHLIST_SETUP_PRESETS)).toEqual([
      "cti_osint",
      "news",
      "general",
      "blank"
    ])

    const ctiDefaults = applyWatchlistSetupPreset("cti_osint")
    expect(ctiDefaults).toMatchObject({
      domain: "cti_osint",
      priority: "high"
    })
    expect(ctiDefaults.tags).toEqual(expect.arrayContaining(["cti", "osint"]))
    expect(ctiDefaults).not.toHaveProperty("alertRules")

    const newsDefaults = applyWatchlistSetupPreset("news")
    expect(newsDefaults).toMatchObject({
      domain: "news",
      priority: "medium"
    })
    expect(newsDefaults.tags).toContain("news")
  })

  it("builds a topic-only Watchlist payload without sources or monitor payloads", () => {
    const result = buildWatchlistSetupPlan({
      preset: "news",
      startMode: "topic",
      name: "Election integrity",
      objective: "Track source diversity and recency",
      trackedScopeText: "election officials, state courts",
      sourceUrlsText: "",
      monitorName: "",
      reportGoal: ""
    })

    expect(result.watchlist).toMatchObject({
      name: "Election integrity",
      domain: "news",
      objective: "Track source diversity and recency",
      status: "active",
      priority: "medium"
    })
    expect(result.watchlist.description).toBe("Tracked scope: election officials, state courts")
    expect(result.watchlist.tags).toEqual(
      expect.arrayContaining(["news", "election officials", "state courts"])
    )
    expect(result.sources).toEqual([])
    expect(result.canCreateMonitor).toBe(false)
    expect(result.destination).toBe("sources")
  })

  it("builds source-backed Watchlist and feed payloads", () => {
    const result = buildWatchlistSetupPlan({
      preset: "cti_osint",
      startMode: "sources",
      name: "Healthcare ransomware",
      objective: "Find ransomware reports affecting hospitals",
      trackedScopeText: "hospitals, Germany",
      sourceName: "Ransomware feed",
      sourceUrlsText: "https://example.com/feed.xml\nhttps://advisories.example.org/rss",
      sourceType: "rss",
      monitorName: "Healthcare ransomware monitor",
      reportGoal: "Daily situational brief"
    })

    expect(result.watchlist).toMatchObject({
      name: "Healthcare ransomware",
      domain: "cti_osint",
      priority: "high"
    })
    expect(result.sources).toEqual([
      {
        name: "Ransomware feed",
        url: "https://example.com/feed.xml",
        source_type: "rss",
        active: true
      },
      {
        name: "advisories.example.org",
        url: "https://advisories.example.org/rss",
        source_type: "rss",
        active: true
      }
    ])
    expect(result.canCreateMonitor).toBe(true)
    expect(result.destination).toBe("jobs")
  })

  it("builds monitor payloads after sources have IDs", () => {
    const payload = buildWatchlistSetupJobPayload(
      {
        preset: "cti_osint",
        startMode: "report_goal",
        name: "Healthcare ransomware",
        objective: "Find ransomware reports affecting hospitals",
        trackedScopeText: "hospitals, Germany",
        sourceUrlsText: "https://example.com/feed.xml",
        monitorName: "Healthcare ransomware monitor",
        reportGoal: "Daily situational brief",
        includeAudioBriefing: true,
        schedulePreset: "weekdays"
      },
      [42, 77]
    )

    expect(payload).toEqual({
      name: "Healthcare ransomware monitor",
      description: "Report goal: Daily situational brief",
      scope: { sources: [42, 77] },
      active: true,
      schedule_expr: "0 8 * * MON-FRI",
      timezone: expect.any(String),
      output_prefs: {
        template_name: "briefing_md",
        generate_audio: true
      }
    })
  })

  it("normalizes setup source URLs and rejects unsupported schemes", () => {
    expect(
      parseSetupSourceUrls(
        "https://example.com/feed.xml, http://example.org/rss\nftp://bad.example/feed\nnot a url\nhttps://example.com/feed.xml"
      )
    ).toEqual(["https://example.com/feed.xml", "http://example.org/rss"])
  })
})
