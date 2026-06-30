import watchlistsLocale from "../../../../../assets/locale/en/watchlists.json"
import { describe, expect, it } from "vitest"

type JsonObject = Record<string, unknown>

const pick = <T = unknown>(source: JsonObject, keyPath: string): T | undefined =>
  keyPath.split(".").reduce<unknown>((acc, segment) => {
    if (!acc || typeof acc !== "object") return undefined
    return (acc as JsonObject)[segment]
  }, source) as T | undefined

describe("Overview onboarding copy contract", () => {
  it("keeps quick setup and guided-tour onboarding copy stable", () => {
    const labels = watchlistsLocale as JsonObject

    expect(pick(labels, "overview.onboarding.title")).toBe("Add initial collection")
    expect(pick(labels, "overview.onboarding.pipeline")).toBe(
      "Add feeds -> Configure monitor -> Check Activity -> Review Updates -> Generate Reports"
    )
    expect(pick(labels, "overview.onboarding.cta")).toEqual({
      addFeed: "Add first feed",
      createMonitor: "Create first monitor",
      guidedSetup: "Add initial collection",
      reviewArticles: "Open Updates"
    })
    expect(pick(labels, "overview.onboarding.quickSetup.title")).toBe("Add initial collection")
    expect(pick(labels, "overview.onboarding.quickSetup.fields")).toEqual(
      expect.objectContaining({
        sourceUrl: "Feed URL",
        monitorName: "Monitor name",
        schedule: "Schedule",
        setupGoal: "Setup goal",
        runNow: "Run test generation immediately",
        audioBriefing: "Audio briefing"
      })
    )
    expect(pick(labels, "guide.steps")).toEqual(
      expect.objectContaining({
        sources: expect.objectContaining({ title: "1. Add feeds" }),
        jobs: expect.objectContaining({ title: "2. Create monitors" }),
        runs: expect.objectContaining({ title: "3. Check activity" }),
        items: expect.objectContaining({ title: "4. Review updates" }),
        outputs: expect.objectContaining({ title: "5. Deliver reports" })
      })
    )
    expect(pick(labels, "teachPoints")).toEqual(
      expect.objectContaining({
        jobs: expect.objectContaining({
          title: "Monitor setup tip"
        }),
        templates: expect.objectContaining({
          title: "Template setup tip"
        })
      })
    )
  })
})
