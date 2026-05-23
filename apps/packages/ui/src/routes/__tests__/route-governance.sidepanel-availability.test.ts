import path from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"

import {
  getRouteMetadata,
  normalizeRoutePath,
  ROUTE_METADATA
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

const extensionUnifiedRegistry = readFirstExistingSource(
  [
    path.resolve(
      testDir,
      "../../../../../tldw-frontend/extension/routes/route-registry.tsx"
    )
  ],
  "extension route-registry.tsx"
)

const extractSidepanelPaths = (source: string, fileName: string) =>
  extractRoutePathsFromRouteObjects(source, fileName, { kind: "sidepanel" }).map(
    normalizeRoutePath
  )

const sharedSidepanelPaths = extractSidepanelPaths(
  sharedSidepanelRegistry.source,
  sharedSidepanelRegistry.path
)

const extensionSidepanelPaths = uniqueSorted([
  ...extractSidepanelPaths(
    extensionSidepanelRegistry.source,
    extensionSidepanelRegistry.path
  ),
  ...extractSidepanelPaths(
    extensionUnifiedRegistry.source,
    extensionUnifiedRegistry.path
  )
])

const sidepanelRegistryPaths = uniqueSorted([
  ...sharedSidepanelPaths,
  ...extensionSidepanelPaths
])

const metadataSidepanelPaths = uniqueSorted(
  ROUTE_METADATA.filter((metadata) =>
    metadata.availability.includes("extension_sidepanel")
  ).map((metadata) => normalizeRoutePath(metadata.path))
)

describe("route governance sidepanel availability", () => {
  it("registers every metadata-declared sidepanel route in a sidepanel registry", () => {
    const missingRegistryRoutes = metadataSidepanelPaths.filter(
      (routePath) => !sidepanelRegistryPaths.includes(routePath)
    )

    expect(missingRegistryRoutes).toEqual([])
  })

  it("marks every shared or extension sidepanel registry route as sidepanel-available metadata", () => {
    const missingMetadataAvailability = sidepanelRegistryPaths.filter(
      (routePath) =>
        !getRouteMetadata(routePath)?.availability.includes("extension_sidepanel")
    )

    expect(missingMetadataAvailability).toEqual([])
  })

  it("keeps core handoff routes available in the sidepanel registry union", () => {
    const expectedHandoffRoutes = [
      "/chat",
      "/clipper",
      "/companion",
      "/flashcards",
      "/persona"
    ]

    for (const routePath of expectedHandoffRoutes) {
      expect(sidepanelRegistryPaths, routePath).toContain(routePath)
      expect(getRouteMetadata(routePath)?.availability, routePath).toContain(
        "extension_sidepanel"
      )
    }
  })

  it("keeps internal sidepanel QA routes out of default navigation", () => {
    const exposedDebugRoutes = sidepanelRegistryPaths.filter((routePath) => {
      const metadata = getRouteMetadata(routePath)

      return (
        metadata?.surface === "internal_qa_debug" &&
        (metadata.nav !== "hidden" ||
          metadata.commandPalette !== "hide" ||
          metadata.smoke === "include")
      )
    })

    expect(exposedDebugRoutes).toEqual([])
  })
})
