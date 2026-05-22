import { describe, expect, it } from "vitest"
import {
  STUDY_SAFETY_SPECIALIZED_ROUTE_FINDINGS,
  STUDY_SAFETY_SPECIALIZED_ROUTE_JOBS
} from "../study-safety-specialized-route-jobs"
import { getRouteMetadata } from "../route-metadata"

const routes = [
  "/evaluations",
  "/flashcards",
  "/quiz",
  "/moderation-playground",
  "/content-review",
  "/claims-review",
  "/data-tables",
  "/chunking-playground",
  "/kanban",
  "/vn-assets",
  "/vn-play"
] as const

describe("study, safety, and specialized route jobs", () => {
  it("covers every Task 11B route once", () => {
    expect(STUDY_SAFETY_SPECIALIZED_ROUTE_JOBS.map((job) => job.route).sort()).toEqual(
      Array.from(routes).sort()
    )
  })

  it("keeps labels aligned with route metadata", () => {
    for (const job of STUDY_SAFETY_SPECIALIZED_ROUTE_JOBS) {
      expect(job.label).toBe(getRouteMetadata(job.route)?.label)
    }
  })

  it("keeps labels, jobs, and classifications usable", () => {
    for (const job of STUDY_SAFETY_SPECIALIZED_ROUTE_JOBS) {
      expect(job.label).not.toHaveLength(0)
      expect(job.primaryJob).not.toHaveLength(0)
      expect(job.primaryActionLabel).not.toHaveLength(0)
      expect(job.classification).not.toHaveLength(0)
      expect(job.canonicalComponent).not.toHaveLength(0)
      expect(job.visibilityDecision).not.toHaveLength(0)
    }
  })

  it("maps all Task 11B audit findings", () => {
    const covered = new Set(
      STUDY_SAFETY_SPECIALIZED_ROUTE_JOBS.flatMap((job) => job.findings)
    )

    for (const finding of STUDY_SAFETY_SPECIALIZED_ROUTE_FINDINGS) {
      expect(covered.has(finding)).toBe(true)
    }
  })

  it("preserves canonical ownership for aliases, labs, and shared routes", () => {
    expect(STUDY_SAFETY_SPECIALIZED_ROUTE_JOBS).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          route: "/moderation-playground",
          routeOwner: "shared_alias",
          canonicalComponent: "Navigate:/moderation/rules",
          visibilityDecision: "alias_only"
        }),
        expect.objectContaining({
          route: "/claims-review",
          routeOwner: "next_alias",
          canonicalComponent: "RouteRedirect:/content-review",
          visibilityDecision: "alias_only"
        }),
        expect.objectContaining({
          route: "/vn-assets",
          routeOwner: "next_page",
          canonicalComponent: "VNAssetsWorkbench",
          visibilityDecision: "labs_nav"
        }),
        expect.objectContaining({
          route: "/vn-play",
          routeOwner: "next_page",
          canonicalComponent: "VNPlayWorkspace",
          visibilityDecision: "labs_nav"
        })
      ])
    )
  })

  it("keeps alias routes hidden and recoverable in route metadata", () => {
    const aliases = [
      ["/moderation-playground", "/moderation/rules"],
      ["/claims-review", "/content-review"]
    ] as const

    for (const [route, destination] of aliases) {
      const metadata = getRouteMetadata(route)

      expect(metadata?.canonicalPath).toBe(destination)
      expect(metadata?.redirectsTo).toBe(destination)
      expect(metadata?.surface).toBe("redirect")
      expect(metadata?.commandPalette).toBe("alias_only")
      expect(metadata?.nav).toBe("hidden")
      expect(metadata?.smoke).toBe("exclude")
    }
  })
})
