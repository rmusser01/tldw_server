import { describe, expect, it } from "vitest"

import { PAGES } from "../../e2e/smoke/page-inventory"
import {
  getRouteHeadingPolicy,
  getRouteMetadata,
  normalizeRoutePath,
  ROUTE_METADATA
} from "../../../packages/ui/src/routes/route-metadata"

const sorted = (values: string[]): string[] => [...values].sort()

describe("route heading governance metadata", () => {
  it("requires active smoke inventory routes to have an h1 policy or metadata exception", () => {
    const routesWithoutHeadingPolicy = PAGES
      .filter((entry) => !entry.skip)
      .map((entry) => ({
        path: normalizeRoutePath(entry.path),
        metadata: getRouteMetadata(entry.path)
      }))
      .filter(({ metadata }) => {
        if (!metadata) return true

        const policy = getRouteHeadingPolicy(metadata)

        return !policy.requiresH1 && !policy.exceptionReason?.trim()
      })
      .map(({ path }) => path)

    expect(sorted(routesWithoutHeadingPolicy)).toEqual([])
  })

  it("requires explicit h1 opt-outs to carry a recovery-friendly reason", () => {
    const routesWithoutExceptionReasons = ROUTE_METADATA
      .filter((metadata) => getRouteHeadingPolicy(metadata).requiresH1 === false)
      .filter((metadata) => !getRouteHeadingPolicy(metadata).exceptionReason?.trim())
      .map((metadata) => metadata.path)

    expect(sorted(routesWithoutExceptionReasons)).toEqual([])
  })
})
