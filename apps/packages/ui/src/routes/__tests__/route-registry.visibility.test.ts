import { existsSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"

import { getRouteMetadata, ROUTE_METADATA } from "../route-metadata"
import {
  extractRoutePathsFromRouteObjects,
  readFirstExistingSource,
  resolveFirstExistingPath
} from "./route-registry-ast-helpers"

const isDynamicRoutePath = (routePath: string): boolean =>
  routePath.includes(":") || routePath.includes("*")

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

const nonDynamicOptionRegistryPaths = optionRegistryPaths.filter(
  (routePath) => !isDynamicRoutePath(routePath)
)

const frontendPagesRoot = resolveFirstExistingPath(
  [path.resolve(testDir, "../../../../../tldw-frontend/pages")],
  "tldw-frontend/pages for visibility test"
)

const routePathToPageCandidates = (routePath: string): string[] => {
  const normalizedPath = routePath === "/" ? "/index" : routePath
  const pagePath = normalizedPath.replace(/^\//, "")

  return [
    path.join(frontendPagesRoot, `${pagePath}.tsx`),
    path.join(frontendPagesRoot, pagePath, "index.tsx")
  ]
}

const hasNextPageFile = (routePath: string): boolean =>
  routePathToPageCandidates(routePath).some((candidate) => existsSync(candidate))

const isRegistryBackedRoute = (routePath: string): boolean =>
  optionRegistryPaths.includes(routePath)

describe("route registry visibility metadata", () => {
  it("defines metadata for every non-dynamic option registry route", () => {
    const missingMetadata = nonDynamicOptionRegistryPaths.filter(
      (routePath) => !getRouteMetadata(routePath)
    )

    expect(missingMetadata).toEqual([])
  })

  it("does not claim web availability for unknown routes", () => {
    for (const metadata of ROUTE_METADATA) {
      if (!metadata.availability.includes("web")) {
        continue
      }

      const hasRouteOwner =
        isRegistryBackedRoute(metadata.path) ||
        hasNextPageFile(metadata.path) ||
        Boolean(metadata.redirectsTo)

      expect(hasRouteOwner, metadata.path).toBe(true)
    }
  })

  it("registers the legacy audio redirect alias in the shared WebUI router", () => {
    expect(optionRegistryPaths).toContain("/audio")
    expect(getRouteMetadata("/audio")).toMatchObject({
      canonicalPath: "/speech",
      redirectsTo: "/speech",
      surface: "redirect"
    })
  })

  it("keeps internal QA and debug routes out of primary navigation", () => {
    for (const metadata of ROUTE_METADATA) {
      if (metadata.surface !== "internal_qa_debug") {
        continue
      }

      expect(metadata.nav, metadata.path).not.toBe("primary")
      expect(metadata.commandPalette, metadata.path).toBe("hide")
    }
  })

  it("keeps hosted-only routes out of default self-hosted navigation", () => {
    for (const metadata of ROUTE_METADATA) {
      if (metadata.surface !== "hosted_only") {
        continue
      }

      expect(metadata.nav, metadata.path).not.toBe("primary")
    }
  })
})
