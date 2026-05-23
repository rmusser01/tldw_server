import { existsSync, readFileSync } from "node:fs"
import { describe, expect, it } from "vitest"
import { getRouteMetadata } from "../route-metadata"

const routeRegistryPathCandidates = [
  "src/routes/sidepanel-route-registry.tsx",
  "../packages/ui/src/routes/sidepanel-route-registry.tsx",
  "apps/packages/ui/src/routes/sidepanel-route-registry.tsx"
]

const routeRegistryPath = routeRegistryPathCandidates.find((candidate) =>
  existsSync(candidate)
)

if (!routeRegistryPath) {
  throw new Error(
    "Unable to locate sidepanel-route-registry.tsx for sidepanel chat route test"
  )
}

const routeRegistrySource = readFileSync(routeRegistryPath, "utf8")

describe("sidepanel route registry chat parity", () => {
  it("registers a dedicated sidepanel chat route", () => {
    expect(routeRegistrySource).toMatch(/path:\s*"\/chat"/)
    expect(routeRegistrySource).toContain("SidepanelChat")
    expect(getRouteMetadata("/chat")?.availability).toContain(
      "extension_sidepanel"
    )
  })

  it("keeps the sidepanel home resolver on the root route", () => {
    expect(routeRegistrySource).toMatch(/path:\s*"\/"/)
    expect(routeRegistrySource).toContain("SidepanelHomeResolver")
    expect(getRouteMetadata("/")?.availability).toContain("extension_sidepanel")
  })
})
