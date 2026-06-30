import watchlistsLocale from "../../../../assets/locale/en/watchlists.json"
import { describe, expect, it } from "vitest"

type JsonObject = Record<string, unknown>

const getNestedValue = (source: JsonObject, keyPath: string): unknown =>
  keyPath.split(".").reduce<unknown>((acc, segment) => {
    if (!acc || typeof acc !== "object") return undefined
    return (acc as JsonObject)[segment]
  }, source)

const collectStringValues = (value: unknown): string[] => {
  if (typeof value === "string") return [value]
  if (Array.isArray(value)) return value.flatMap((item) => collectStringValues(item))
  if (value && typeof value === "object") {
    return Object.values(value as JsonObject).flatMap((item) => collectStringValues(item))
  }
  return []
}

const FIRST_RUN_KEY_PATHS = [
  "overview.onboarding.pipeline",
  "overview.onboarding.steps.addFeed.description",
  "overview.onboarding.steps.createMonitor.description",
  "overview.onboarding.steps.reviewResults.description",
  "overview.onboarding.cta.addFeed",
  "overview.onboarding.cta.createMonitor",
  "overview.onboarding.cta.reviewArticles",
  "overview.onboarding.quickSetup.help.feed",
  "overview.onboarding.quickSetup.help.monitor",
  "overview.onboarding.quickSetup.help.review",
  "overview.onboarding.quickSetup.reviewDescription",
  "overview.onboarding.quickSetup.destinationHint.runNow",
  "overview.onboarding.quickSetup.destinationHint.briefing",
  "overview.onboarding.quickSetup.destinationHint.triage",
  "guide.completedDescription",
  "guide.steps.sources.description",
  "guide.steps.jobs.description",
  "guide.steps.runs.description",
  "guide.steps.items.description",
  "guide.steps.outputs.description"
] as const

describe("Watchlists first-run copy contract", () => {
  it("keeps first-run terminology anchored on Feeds, Monitors, Activity, and Reports", () => {
    const labels = watchlistsLocale as JsonObject
    expect(getNestedValue(labels, "overview.onboarding.cta.addFeed")).toBe("Add first feed")
    expect(getNestedValue(labels, "overview.onboarding.cta.createMonitor")).toBe(
      "Create first monitor"
    )
    expect(getNestedValue(labels, "overview.onboarding.steps.reviewResults.description")).toBe(
      "Open Activity for monitor health, Updates for captured content, and Reports for generated briefings."
    )
    expect(getNestedValue(labels, "guide.steps.outputs.title")).toBe("5. Deliver reports")
  })

  it("avoids backend model nouns in first-run copy blocks", () => {
    const labels = watchlistsLocale as JsonObject
    const values = FIRST_RUN_KEY_PATHS
      .map((keyPath) => getNestedValue(labels, keyPath))
      .flatMap((value) => collectStringValues(value))
      .join(" ")
      .toLowerCase()

    expect(values).not.toMatch(/\bsource\b/)
    expect(values).not.toMatch(/\bsources\b/)
    expect(values).not.toMatch(/\bjob\b/)
    expect(values).not.toMatch(/\bjobs\b/)
    expect(values).not.toMatch(/\boutput\b/)
    expect(values).not.toMatch(/\boutputs\b/)
  })
})
