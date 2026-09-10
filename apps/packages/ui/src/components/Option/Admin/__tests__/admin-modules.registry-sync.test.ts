import { describe, expect, it } from "vitest"

import { getRouteMetadata, ROUTE_METADATA } from "@/routes/route-metadata"
import { ADMIN_MODULES } from "../admin-modules"

/**
 * The admin registry (admin-modules.ts) and route metadata are parallel
 * declarations by design — the registry carries operator descriptions and
 * groups that metadata lacks. These invariants keep the two from drifting
 * (PR #2879 review recommendation).
 */
describe("admin module registry <-> route metadata sync", () => {
  it("registers route metadata for every admin module", () => {
    for (const module of ADMIN_MODULES) {
      const metadata = getRouteMetadata(module.route)
      expect(metadata, module.route).toBeDefined()
      expect(metadata?.surface, module.route).toBe("admin_operator")
    }
  })

  it("declares every admin_operator drill-down route in the admin registry", () => {
    const registryRoutes = new Set(ADMIN_MODULES.map((module) => module.route))
    const adminMetadataRoutes = ROUTE_METADATA.filter(
      (metadata) =>
        metadata.surface === "admin_operator" && metadata.path !== "/admin"
    ).map((metadata) => metadata.path)

    expect(adminMetadataRoutes.length).toBeGreaterThan(0)
    for (const path of adminMetadataRoutes) {
      expect(registryRoutes.has(path), path).toBe(true)
    }
  })

  it("keeps registry routes unique and shaped like admin drill-downs", () => {
    const routes = ADMIN_MODULES.map((module) => module.route)
    expect(new Set(routes).size).toBe(routes.length)
    for (const module of ADMIN_MODULES) {
      expect(module.route, module.label).toMatch(/^\/admin\/[a-z0-9-]+$/)
      expect(module.label.trim().length, module.route).toBeGreaterThan(0)
      expect(module.description.trim().length, module.route).toBeGreaterThan(0)
    }
  })
})
