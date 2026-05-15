import { existsSync, readFileSync } from "node:fs"
import { describe, expect, it } from "vitest"

const extensionRouteRegistryPathCandidates = [
  "extension/routes/route-registry.tsx",
  "apps/tldw-frontend/extension/routes/route-registry.tsx"
]

const extensionRouteRegistryPath = extensionRouteRegistryPathCandidates.find(
  (candidate) => existsSync(candidate)
)

if (!extensionRouteRegistryPath) {
  throw new Error(
    "Unable to locate extension route-registry.tsx for workspace-playground parity test"
  )
}

const extensionRouteRegistrySource = readFileSync(
  extensionRouteRegistryPath,
  "utf8"
)

describe("extension route registry research studio parity", () => {
  it("registers /research-studio as the navigation route", () => {
    expect(extensionRouteRegistrySource).toMatch(/path:\s*"\/research-studio"/)
  })

  it("exposes research studio navigation metadata", () => {
    expect(extensionRouteRegistrySource).toMatch(
      /labelToken:\s*"settings:researchStudioNav"/
    )
  })
})
