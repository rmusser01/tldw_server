import path from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"

import { PAGES } from "../../../../../tldw-frontend/e2e/smoke/page-inventory"
import {
  getRouteMetadata,
  normalizeRoutePath,
  ROUTE_METADATA
} from "../route-metadata"
import {
  extractRoutePathsFromRouteObjects,
  readFirstExistingSource
} from "./route-registry-ast-helpers"

const sorted = (values: string[]): string[] => [...values].sort()

const pageEntryByPath = new Map(
  PAGES.map((entry) => [normalizeRoutePath(entry.path), entry])
)
const testDir = path.dirname(fileURLToPath(import.meta.url))
const routeRegistry = readFirstExistingSource(
  [path.resolve(testDir, "../route-registry.tsx")],
  "shared route-registry.tsx"
)

const optionRegistryPaths = extractRoutePathsFromRouteObjects(
  routeRegistry.source,
  routeRegistry.path,
  { kind: "options" }
)

describe("route governance metadata coverage", () => {
  it("does not define duplicate smoke inventory paths", () => {
    const paths = PAGES.map((entry) => normalizeRoutePath(entry.path))
    const duplicatePaths = paths.filter(
      (path, index) => paths.indexOf(path) !== index
    )

    expect(sorted(duplicatePaths)).toEqual([])
  })

  it("covers every shared option route", () => {
    const missingMetadata = optionRegistryPaths.filter(
      (path) => !getRouteMetadata(path)
    )

    expect(sorted(missingMetadata)).toEqual([])
  })

  it("requires every active smoke inventory route to have metadata", () => {
    const missingMetadata = PAGES
      .filter((entry) => !entry.skip)
      .filter((entry) => !getRouteMetadata(entry.path))
      .map((entry) => entry.path)

    expect(sorted(missingMetadata)).toEqual([])
  })

  it("requires skipped smoke inventory routes to have metadata and reasons", () => {
    const invalidSkippedRoutes = PAGES
      .filter((entry) => entry.skip)
      .filter((entry) => {
        const metadata = getRouteMetadata(entry.path)

        return !metadata || !entry.skip?.trim() || metadata.smoke === "include"
      })
      .map((entry) => entry.path)

    expect(sorted(invalidSkippedRoutes)).toEqual([])
  })

  it("keeps included web smoke routes active in the page inventory", () => {
    const missingIncludedRoutes = ROUTE_METADATA
      .filter((metadata) => metadata.availability.includes("web"))
      .filter((metadata) => metadata.smoke === "include")
      .filter((metadata) => {
        const pageEntry = pageEntryByPath.get(normalizeRoutePath(metadata.path))

        return !pageEntry || Boolean(pageEntry.skip)
      })
      .map((metadata) => metadata.path)

    expect(sorted(missingIncludedRoutes)).toEqual([])
  })

  it("does not run smoke-excluded routes as active page inventory entries", () => {
    const activeExcludedRoutes = PAGES
      .filter((entry) => !entry.skip)
      .filter((entry) => getRouteMetadata(entry.path)?.smoke === "exclude")
      .map((entry) => entry.path)

    expect(sorted(activeExcludedRoutes)).toEqual([])
  })
})
