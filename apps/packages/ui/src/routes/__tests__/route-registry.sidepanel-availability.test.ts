import { existsSync, readFileSync } from "node:fs"
import { describe, expect, it } from "vitest"

import { sidepanelRoutes } from "../sidepanel-route-registry"
import {
  getRouteMetadata,
  isRouteAvailableForSurface
} from "../route-metadata"

const extensionSidepanelRegistryCandidates = [
  "apps/tldw-frontend/extension/routes/sidepanel-route-registry.tsx",
  "../../tldw-frontend/extension/routes/sidepanel-route-registry.tsx",
  "../tldw-frontend/extension/routes/sidepanel-route-registry.tsx"
]

const extensionSidepanelRegistryPath = extensionSidepanelRegistryCandidates.find(
  (candidate) => existsSync(candidate)
)

if (!extensionSidepanelRegistryPath) {
  throw new Error(
    "Unable to locate extension sidepanel-route-registry.tsx for metadata validation"
  )
}

const extensionSidepanelRegistrySource = readFileSync(
  extensionSidepanelRegistryPath,
  "utf8"
)

const extractLiteralPaths = (source: string): string[] =>
  [...source.matchAll(/path:\s*"([^"]+)"/g)].map((match) => match[1])

describe("sidepanel route metadata availability", () => {
  it("declares explicit sidepanel availability for shared sidepanel routes", () => {
    const missingSidepanelMetadata = sidepanelRoutes
      .map((route) => route.path)
      .filter((routePath) => !isRouteAvailableForSurface(routePath, "extension_sidepanel"))

    expect(missingSidepanelMetadata).toEqual([])
  })

  it("declares explicit sidepanel availability for extension sidepanel routes", () => {
    const extensionSidepanelPaths = extractLiteralPaths(
      extensionSidepanelRegistrySource
    )
    const missingSidepanelMetadata = extensionSidepanelPaths.filter(
      (routePath) => !isRouteAvailableForSurface(routePath, "extension_sidepanel")
    )

    expect(missingSidepanelMetadata).toEqual([])
  })

  it("keeps sidepanel debug routes classified as internal QA", () => {
    const errorBoundaryRoute = getRouteMetadata("/error-boundary-test")

    expect(errorBoundaryRoute?.surface).toBe("internal_qa_debug")
    expect(errorBoundaryRoute?.nav).toBe("hidden")
    expect(errorBoundaryRoute?.smoke).toBe("exclude")
  })
})
