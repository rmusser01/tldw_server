import { existsSync, readFileSync } from "node:fs"
import { describe, expect, it } from "vitest"
import { getRouteMetadata } from "../route-metadata"
import { STUDY_SAFETY_SPECIALIZED_ROUTE_JOBS } from "../study-safety-specialized-route-jobs"

const registryPathCandidates = [
  "src/routes/sidepanel-route-registry.tsx",
  "../packages/ui/src/routes/sidepanel-route-registry.tsx",
  "apps/packages/ui/src/routes/sidepanel-route-registry.tsx"
]

const registryPath = registryPathCandidates.find((candidate) =>
  existsSync(candidate)
)

if (!registryPath) {
  throw new Error(
    "Unable to locate sidepanel-route-registry.tsx for sidepanel flashcards route test"
  )
}

const registrySource = readFileSync(registryPath, "utf8")

const flashcardsComponentPathCandidates = [
  "src/routes/sidepanel-flashcards.tsx",
  "../packages/ui/src/routes/sidepanel-flashcards.tsx",
  "apps/packages/ui/src/routes/sidepanel-flashcards.tsx"
]

const flashcardsComponentPath = flashcardsComponentPathCandidates.find(
  (candidate) => existsSync(candidate)
)

describe("sidepanel flashcards route registration", () => {
  it("registers a /flashcards route in the sidepanel registry", () => {
    expect(registrySource).toMatch(/path:\s*"\/flashcards"/)
    expect(registrySource).toContain("SidepanelFlashcards")
    expect(getRouteMetadata("/flashcards")?.availability).toContain(
      "extension_sidepanel"
    )
  })

  it("keeps flashcards as the only Task 11B sidepanel route", () => {
    const sidepanelOnlyRoutes = STUDY_SAFETY_SPECIALIZED_ROUTE_JOBS.filter(
      (job) => job.route !== "/flashcards"
    )

    for (const job of sidepanelOnlyRoutes) {
      expect(registrySource).not.toMatch(
        new RegExp(`path:\\s*"${job.route.replace("/", "\\/")}"`)
      )
    }
  })

  it("lazy-imports the SidepanelFlashcards component", () => {
    expect(registrySource).toMatch(
      /const\s+SidepanelFlashcards\s*=\s*lazy\(/
    )
    expect(registrySource).toContain("sidepanel-flashcards")
  })

  it("sidepanel-flashcards.tsx component file exists", () => {
    expect(flashcardsComponentPath).toBeDefined()
  })

  it("sidepanel-flashcards.tsx imports browser from wxt/browser", () => {
    expect(flashcardsComponentPath).toBeDefined()
    const source = readFileSync(flashcardsComponentPath!, "utf8")
    expect(source).toContain('from "wxt/browser"')
    expect(source).not.toContain("browser-polyfill")
  })

  it("sidepanel-flashcards.tsx opens options page at /flashcards", () => {
    expect(flashcardsComponentPath).toBeDefined()
    const source = readFileSync(flashcardsComponentPath!, "utf8")
    expect(source).toContain("options.html#")
    expect(source).toContain('openOptionsHashRoute("/flashcards")')
  })
})
