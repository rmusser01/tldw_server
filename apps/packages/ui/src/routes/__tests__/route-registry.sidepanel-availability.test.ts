import path from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"

import {
  getRouteMetadata,
  isRouteVisibleForSurface
} from "../route-metadata"
import {
  extractRoutePathsFromRouteObjects,
  readFirstExistingSource,
  uniqueSorted
} from "./route-registry-ast-helpers"

const testDir = path.dirname(fileURLToPath(import.meta.url))

const sharedSidepanelRegistry = readFirstExistingSource(
  [path.resolve(testDir, "../sidepanel-route-registry.tsx")],
  "shared sidepanel-route-registry.tsx"
)

const extensionSidepanelRegistry = readFirstExistingSource(
  [
    path.resolve(
      testDir,
      "../../../../../tldw-frontend/extension/routes/sidepanel-route-registry.tsx"
    )
  ],
  "extension sidepanel-route-registry.tsx"
)

const extensionOptionRegistry = readFirstExistingSource(
  [
    path.resolve(
      testDir,
      "../../../../../tldw-frontend/extension/routes/route-registry.tsx"
    )
  ],
  "extension route-registry.tsx"
)

const sidepanelRoutePaths = uniqueSorted([
  ...extractRoutePathsFromRouteObjects(
    sharedSidepanelRegistry.source,
    sharedSidepanelRegistry.path,
    { kind: "sidepanel" }
  ),
  ...extractRoutePathsFromRouteObjects(
    extensionSidepanelRegistry.source,
    extensionSidepanelRegistry.path,
    { kind: "sidepanel" }
  )
])

const extensionOptionNavPaths = uniqueSorted(
  extractRoutePathsFromRouteObjects(
    extensionOptionRegistry.source,
    extensionOptionRegistry.path,
    { kind: "options", requireNav: true }
  ).filter(
    (routePath) =>
      !routePath.includes(":") && !sidepanelRoutePaths.includes(routePath)
  )
)

describe("sidepanel route availability metadata", () => {
  it("keeps extension sidepanel chat reachable at both root and /chat", () => {
    expect(extensionSidepanelRegistry.source).toMatch(/path\s*:\s*["']\/["']/)
    expect(extensionSidepanelRegistry.source).toMatch(
      /path\s*:\s*["']\/chat["']/
    )
    expect(extensionOptionRegistry.source).toMatch(/path\s*:\s*["']\/["']/)
    expect(extensionOptionRegistry.source).toMatch(
      /path\s*:\s*["']\/chat["']/
    )
  })

  it("declares sidepanel availability for every shared or extension sidepanel route", () => {
    const routesMissingSidepanelAvailability = sidepanelRoutePaths.filter(
      (routePath) =>
        !getRouteMetadata(routePath)?.availability.includes("extension_sidepanel")
    )

    expect(routesMissingSidepanelAvailability).toEqual([])
  })

  it("defines metadata labels and groups for extension option routes that appear in nav", () => {
    const routesMissingNavMetadata = extensionOptionNavPaths.filter((routePath) => {
      const metadata = getRouteMetadata(routePath)

      return !metadata?.label || !metadata.group
    })

    expect(routesMissingNavMetadata).toEqual([])
  })

  it("marks sidepanel debug routes as internal QA/debug routes", () => {
    const debugRoutes = [
      "/error-boundary-test",
      "/__debug__/sidepanel-chat",
      "/__debug__/sidepanel-error-boundary"
    ]

    for (const routePath of debugRoutes) {
      const metadata = getRouteMetadata(routePath)

      expect(metadata?.surface, routePath).toBe("internal_qa_debug")
      expect(metadata?.nav, routePath).toBe("hidden")
      expect(metadata?.commandPalette, routePath).toBe("hide")
    }
  })

  it("does not infer sidepanel availability from web or extension options availability", () => {
    expect(isRouteVisibleForSurface("/chat", "extension_sidepanel")).toBe(true)
    expect(isRouteVisibleForSurface("/media", "extension_sidepanel")).toBe(false)
    expect(isRouteVisibleForSurface("/settings/model", "extension_sidepanel")).toBe(
      false
    )
  })
})
