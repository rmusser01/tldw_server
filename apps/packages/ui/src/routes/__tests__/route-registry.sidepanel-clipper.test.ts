import { existsSync, readFileSync } from "node:fs"
import { describe, expect, it } from "vitest"

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
    "Unable to locate sidepanel-route-registry.tsx for sidepanel clipper route test"
  )
}

const routeRegistrySource = readFileSync(routeRegistryPath, "utf8")

describe("sidepanel route registry clipper parity", () => {
  it("registers a dedicated sidepanel clipper route", () => {
    expect(routeRegistrySource).toMatch(/path:\s*"\/clipper"/)
    expect(routeRegistrySource).toContain("SidepanelClipper")
  })
})
