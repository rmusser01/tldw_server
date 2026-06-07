import { existsSync, readFileSync } from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

import {
  AUDITED_ROOT_ROUTE_PATHS,
  getRouteMetadata,
  isAuditedRootRoute,
  ROUTE_METADATA
} from "../route-metadata"

const routeRegistryPathCandidates = [
  path.resolve(process.cwd(), "src/routes/route-registry.tsx"),
  path.resolve(process.cwd(), "../packages/ui/src/routes/route-registry.tsx"),
  path.resolve(process.cwd(), "apps/packages/ui/src/routes/route-registry.tsx")
]

const routeRegistryPath = routeRegistryPathCandidates.find((candidate) =>
  existsSync(candidate)
)

if (!routeRegistryPath) {
  throw new Error("Unable to locate route-registry.tsx for metadata validation")
}

const routeRegistrySource = readFileSync(routeRegistryPath, "utf8")

const pageRootCandidates = [
  path.resolve(process.cwd(), "../../tldw-frontend/pages"),
  path.resolve(process.cwd(), "apps/tldw-frontend/pages"),
  path.resolve(process.cwd(), "pages")
]

const pageRoot = pageRootCandidates.find((candidate) => existsSync(candidate))

if (!pageRoot) {
  throw new Error("Unable to locate tldw-frontend pages directory")
}

const routeToPageCandidates = (routePath: string): string[] => {
  if (routePath === "/") {
    return [path.join(pageRoot, "index.tsx")]
  }
  const relativePath = routePath.replace(/^\//, "")
  return [
    path.join(pageRoot, `${relativePath}.tsx`),
    path.join(pageRoot, relativePath, "index.tsx")
  ]
}

const hasNextPageForRoute = (routePath: string): boolean =>
  routeToPageCandidates(routePath).some((candidate) => existsSync(candidate))

const extractLiteralPaths = (source: string): string[] =>
  [...source.matchAll(/path:\s*"([^"]+)"/g)].map((match) => match[1])

const optionRegistryPaths = new Set([
  ...extractLiteralPaths(routeRegistrySource),
  "/repo2txt",
  "/chat-workspace",
  "/workspaces",
  "/prototype-workspaces",
  "/document-workspace"
])

describe("route registry visibility metadata", () => {
  it("covers every audited option-registry route with metadata", () => {
    const missingMetadata = [...optionRegistryPaths]
      .filter((routePath) => isAuditedRootRoute(routePath))
      .filter((routePath) => !getRouteMetadata(routePath))

    expect(missingMetadata).toEqual([])
  })

  it("anchors every audited web route in either shared options routes or Next pages", () => {
    const unownedRoutes = AUDITED_ROOT_ROUTE_PATHS.filter((routePath) => {
      const metadata = getRouteMetadata(routePath)
      if (!metadata?.availability.includes("web")) {
        return false
      }
      return !optionRegistryPaths.has(routePath) && !hasNextPageForRoute(routePath)
    })

    expect(unownedRoutes).toEqual([])
  })

  it("does not promote hosted, debug, or legacy routes as primary self-hosted navigation", () => {
    const promotedRoutes = ROUTE_METADATA.filter(
      (metadata) =>
        metadata.nav === "primary" &&
        (metadata.surface === "hosted_only" ||
          metadata.surface === "internal_qa_debug" ||
          metadata.surface === "legacy_alias")
    ).map((metadata) => metadata.path)

    expect(promotedRoutes).toEqual([])
  })
})
