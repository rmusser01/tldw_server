import { existsSync, readFileSync } from "node:fs"
import { describe, expect, it } from "vitest"
import { getRouteMetadata } from "../route-metadata"

const optionRouteRegistryPathCandidates = [
  "src/routes/route-registry.tsx",
  "../packages/ui/src/routes/route-registry.tsx",
  "apps/packages/ui/src/routes/route-registry.tsx"
]
const sidepanelRouteRegistryPathCandidates = [
  "src/routes/sidepanel-route-registry.tsx",
  "../packages/ui/src/routes/sidepanel-route-registry.tsx",
  "apps/packages/ui/src/routes/sidepanel-route-registry.tsx"
]

const optionRouteRegistryPath = optionRouteRegistryPathCandidates.find(
  (candidate) => existsSync(candidate)
)
const sidepanelRouteRegistryPath = sidepanelRouteRegistryPathCandidates.find(
  (candidate) => existsSync(candidate)
)

if (!optionRouteRegistryPath || !sidepanelRouteRegistryPath) {
  throw new Error(
    "Unable to locate both option and sidepanel route registry sources for companion parity test"
  )
}

const optionRouteRegistrySource = readFileSync(optionRouteRegistryPath, "utf8")
const sidepanelRouteRegistrySource = readFileSync(sidepanelRouteRegistryPath, "utf8")

describe("sidepanel route registry companion parity", () => {
  it("registers a dedicated companion sidepanel route", () => {
    expect(sidepanelRouteRegistrySource).toMatch(/path:\s*"\/companion"/)
    expect(getRouteMetadata("/companion")?.availability).toContain(
      "extension_sidepanel"
    )
  })

  it("registers the companion conversation route in both option and sidepanel shells", () => {
    const optionMatches =
      optionRouteRegistrySource.match(/path:\s*"\/companion\/conversation"/g) ?? []
    const sidepanelMatches =
      sidepanelRouteRegistrySource.match(/path:\s*"\/companion\/conversation"/g) ?? []
    expect(optionMatches).toHaveLength(1)
    expect(sidepanelMatches).toHaveLength(1)
    expect(getRouteMetadata("/companion/conversation")?.availability).toContain(
      "extension_sidepanel"
    )
  })
})
