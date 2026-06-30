import { readFileSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"

import {
  HOSTED_VISIBLE_OPTION_PATHS,
  isHostedVisibleOptionPath
} from "../option-route-visibility"
import {
  getRouteMetadata,
  normalizeRoutePath,
  ROUTE_METADATA
} from "../route-metadata"
import {
  extractRoutePathsFromRouteObjects,
  readFirstExistingSource
} from "./route-registry-ast-helpers"

const testDir = path.dirname(fileURLToPath(import.meta.url))
const optionRouteVisibilitySourcePath = path.resolve(
  testDir,
  "../option-route-visibility.ts"
)
const optionRouteVisibilitySource = readFileSync(
  optionRouteVisibilitySourcePath,
  "utf8"
)
const routeRegistry = readFirstExistingSource(
  [path.resolve(testDir, "../route-registry.tsx")],
  "shared route-registry.tsx"
)

const optionRegistryPaths = extractRoutePathsFromRouteObjects(
  routeRegistry.source,
  routeRegistry.path,
  { kind: "options" }
)

const sorted = (values: string[]) => [...values].sort()

describe("hosted option route visibility", () => {
  it("derives hosted-visible option paths from route metadata", () => {
    const metadataVisiblePaths = ROUTE_METADATA.filter(
      (metadata) => metadata.hostedOptionVisibility === "visible"
    ).map((metadata) => normalizeRoutePath(metadata.path))

    expect(sorted(metadataVisiblePaths)).toEqual(
      sorted(Array.from(HOSTED_VISIBLE_OPTION_PATHS))
    )
  })

  it("keeps production hosted visibility independent from the full route metadata registry", () => {
    expect(optionRouteVisibilitySource).not.toMatch(/route-metadata/)
    expect(optionRouteVisibilitySource).not.toMatch(/ROUTE_METADATA/)
  })

  it("does not expose internal, redirect, or deprecated routes in hosted mode", () => {
    const invalidHostedRoutes = Array.from(HOSTED_VISIBLE_OPTION_PATHS).filter(
      (routePath) => {
        const metadata = getRouteMetadata(routePath)

        return (
          !metadata ||
          [
            "internal_qa_debug",
            "legacy_alias",
            "redirect",
            "deprecated"
          ].includes(metadata.surface)
        )
      }
    )

    expect(invalidHostedRoutes).toEqual([])
  })

  it("keeps hosted-hidden option routes backed by an explicit metadata reason", () => {
    const hiddenRoutesWithoutReason = optionRegistryPaths.filter((routePath) => {
      if (isHostedVisibleOptionPath(routePath)) {
        return false
      }

      const metadata = getRouteMetadata(routePath)

      return !metadata?.rationale?.trim()
    })

    expect(hiddenRoutesWithoutReason).toEqual([])
  })

  it("keeps audio explainer routes visible in hosted mode", () => {
    expect(isHostedVisibleOptionPath("/tts")).toBe(true)
    expect(isHostedVisibleOptionPath("/stt")).toBe(true)
    expect(getRouteMetadata("/tts")?.hostedOptionVisibility).toBe("visible")
    expect(getRouteMetadata("/stt")?.hostedOptionVisibility).toBe("visible")
  })
})
