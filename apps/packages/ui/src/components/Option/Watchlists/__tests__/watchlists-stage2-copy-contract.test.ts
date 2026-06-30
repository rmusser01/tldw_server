import watchlistsLocale from "../../../../assets/locale/en/watchlists.json"
import { describe, expect, it } from "vitest"

type JsonObject = Record<string, unknown>

const getNestedValue = (source: JsonObject, keyPath: string): unknown =>
  keyPath.split(".").reduce<unknown>((acc, segment) => {
    if (!acc || typeof acc !== "object") return undefined
    return (acc as JsonObject)[segment]
  }, source)

describe("Watchlists Stage 2 setup copy contract", () => {
  it("keeps Watchlist-first preset and start-mode labels available", () => {
    const labels = watchlistsLocale as JsonObject

    expect(getNestedValue(labels, "setupWizard.title")).toBe("Create Watchlist")
    expect(getNestedValue(labels, "setupWizard.presets.cti_osint.label")).toBe("CTI / OSINT")
    expect(getNestedValue(labels, "setupWizard.presets.news.label")).toBe("News")
    expect(getNestedValue(labels, "setupWizard.presets.general.label")).toBe("General")
    expect(getNestedValue(labels, "setupWizard.presets.blank.label")).toBe("Blank")
    expect(getNestedValue(labels, "setupWizard.startModes.sources.label")).toBe("Start from sources")
    expect(getNestedValue(labels, "setupWizard.startModes.topic.label")).toBe("Start from topic")
    expect(getNestedValue(labels, "setupWizard.startModes.report_goal.label")).toBe(
      "Start from report goal"
    )
  })

  it("states the Stage 3 alert boundary directly", () => {
    const labels = watchlistsLocale as JsonObject
    expect(getNestedValue(labels, "setupWizard.boundaries.alerts")).toBe(
      "Content-match alerts come later. This setup defines the Watchlist and its initial collection scope."
    )
  })
})
