import watchlistsLocale from "../../../../assets/locale/en/watchlists.json"
import { describe, expect, it } from "vitest"

type JsonObject = Record<string, unknown>

const getNestedValue = (source: JsonObject, keyPath: string): unknown =>
  keyPath.split(".").reduce<unknown>((acc, segment) => {
    if (!acc || typeof acc !== "object") return undefined
    return (acc as JsonObject)[segment]
  }, source)

describe("Watchlists terminology contract", () => {
  it("keeps a canonical terminology map aligned with tab and section labels", () => {
    const labels = watchlistsLocale as JsonObject
    const aliases = getNestedValue(labels, "terminology.aliases") as JsonObject
    const canonical = getNestedValue(labels, "terminology.canonical") as JsonObject
    const pairs: Array<[string, string, string]> = [
      ["sources", "tabs.sources", "sources.title"],
      ["jobs", "tabs.jobs", "jobs.title"],
      ["runs", "tabs.runs", "runs.title"],
      ["items", "tabs.items", "overview.cards.items.title"],
      ["outputs", "tabs.outputs", "outputs.title"]
    ]

    expect(canonical.feeds).toBe("Feeds")
    expect(canonical.monitors).toBe("Monitors")
    expect(canonical.activity).toBe("Activity")
    expect(canonical.articles).toBe("Updates")
    expect(canonical.reports).toBe("Reports")

    for (const [aliasKey, tabKeyPath, sectionKeyPath] of pairs) {
      const expected = aliases[aliasKey]
      expect(typeof expected).toBe("string")
      expect(getNestedValue(labels, tabKeyPath)).toBe(expected)
      expect(getNestedValue(labels, sectionKeyPath)).toBe(expected)
    }

    expect(getNestedValue(labels, "overview.cards.runs.title")).toBe(aliases.runs)
  })

  it("uses canonical nouns in quick actions and help labels", () => {
    const labels = watchlistsLocale as JsonObject
    expect(getNestedValue(labels, "quickActions.sources")).toBe("Set up feeds")
    expect(getNestedValue(labels, "quickActions.jobs")).toBe("Configure monitors")
    expect(getNestedValue(labels, "quickActions.runs")).toBe("Check activity")
    expect(getNestedValue(labels, "quickActions.items")).toBe("Review updates")
    expect(getNestedValue(labels, "quickActions.outputs")).toBe("View reports")

    expect(getNestedValue(labels, "help.tabs.sources")).toBe("Feeds setup")
    expect(getNestedValue(labels, "help.tabs.jobs")).toBe("Monitor scheduling")
    expect(getNestedValue(labels, "help.tabs.outputs")).toBe("Reports guidance")
    expect(getNestedValue(labels, "help.tabs.runs")).toBe("Activity guidance")
    expect(getNestedValue(labels, "help.tabs.items")).toBe("Updates review")
  })
})
