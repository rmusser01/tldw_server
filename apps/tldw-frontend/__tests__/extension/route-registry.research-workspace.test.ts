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
    "Unable to locate extension route-registry.tsx for research-workspace parity test"
  )
}

const extensionRouteRegistrySource = readFileSync(
  extensionRouteRegistryPath,
  "utf8"
)

describe("extension route registry research-workspace parity", () => {
  it("registers /research-workspace options route without a legacy /workspace-playground route", () => {
    expect(extensionRouteRegistrySource).toMatch(/path:\s*"\/research-workspace"/)
    expect(extensionRouteRegistrySource).not.toMatch(
      /path:\s*"\/workspace-playground"/
    )
    expect(extensionRouteRegistrySource).not.toMatch(
      /path:\s*"\/workspace-studio"/
    )
    expect(extensionRouteRegistrySource).not.toMatch(
      /path:\s*"\/research-studio"/
    )
  })

  it("exposes research workspace navigation metadata", () => {
    expect(extensionRouteRegistrySource).toMatch(
      /labelToken:\s*"settings:researchWorkspaceNav"/
    )
  })
})
