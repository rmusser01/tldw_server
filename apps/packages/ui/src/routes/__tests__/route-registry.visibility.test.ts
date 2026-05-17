import { existsSync, readFileSync } from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

import { getRouteMetadata, ROUTE_METADATA } from "../route-metadata"
import {
  CHAT_WORKSPACE_PATH,
  DOCUMENT_WORKSPACE_PATH,
  MODERATION_PLAYGROUND_LEGACY_PATH,
  MODERATION_REVIEW_PATH,
  MODERATION_RULES_PATH,
  PROTOTYPE_WORKSPACES_PATH,
  RESEARCH_STUDIO_PATH,
  REPO2TXT_PATH,
  WORKSPACE_PLAYGROUND_PATH,
  WORKSPACE_STUDIO_PATH
} from "../route-paths"

const isDynamicRoutePath = (routePath: string): boolean =>
  routePath.includes(":") || routePath.includes("*")

const routeRegistryPathCandidates = [
  "src/routes/route-registry.tsx",
  "../packages/ui/src/routes/route-registry.tsx",
  "apps/packages/ui/src/routes/route-registry.tsx"
]

const routeRegistryPath = routeRegistryPathCandidates.find((candidate) =>
  existsSync(candidate)
)

if (!routeRegistryPath) {
  throw new Error("Unable to locate route-registry.tsx for visibility test")
}

const routeRegistrySource = readFileSync(routeRegistryPath, "utf8")

const literalRoutePaths = Array.from(
  routeRegistrySource.matchAll(/path:\s*"([^"]+)"/g),
  (match) => match[1]
)

const constantRoutePaths = [
  CHAT_WORKSPACE_PATH,
  DOCUMENT_WORKSPACE_PATH,
  MODERATION_PLAYGROUND_LEGACY_PATH,
  MODERATION_REVIEW_PATH,
  MODERATION_RULES_PATH,
  PROTOTYPE_WORKSPACES_PATH,
  RESEARCH_STUDIO_PATH,
  REPO2TXT_PATH,
  WORKSPACE_PLAYGROUND_PATH,
  WORKSPACE_STUDIO_PATH
]

const optionRegistryPaths = Array.from(
  new Set([...literalRoutePaths, ...constantRoutePaths])
)

const nonDynamicOptionRegistryPaths = optionRegistryPaths.filter(
  (routePath) => !isDynamicRoutePath(routePath)
)

const frontendPagesRoot = path.resolve(
  process.cwd(),
  "../../tldw-frontend/pages"
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
