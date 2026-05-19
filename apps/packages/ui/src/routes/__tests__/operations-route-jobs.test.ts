import { describe, expect, it } from "vitest"

import {
  getOperationsRouteJob,
  OPERATIONS_ROUTE_JOBS
} from "../operations-route-jobs"
import { getRouteMetadata } from "../route-metadata"

const requiredRoutes = [
  "/admin",
  "/admin/server",
  "/admin/integrations",
  "/admin/sources",
  "/admin/monitoring",
  "/mcp-hub",
  "/sources",
  "/connectors",
  "/connectors/browse",
  "/connectors/jobs",
  "/connectors/sources",
  "/integrations",
  "/scheduled-tasks",
  "/watchlists",
  "/workflow-editor",
  "/skills"
] as const

describe("operations route jobs", () => {
  it("defines every WP10 operations route job exactly once", () => {
    const routes = OPERATIONS_ROUTE_JOBS.map((job) => job.route)

    expect(routes).toEqual(requiredRoutes)
    expect(new Set(routes).size).toBe(routes.length)
  })

  it("aligns route-job labels with root route metadata where metadata exists", () => {
    for (const job of OPERATIONS_ROUTE_JOBS) {
      const metadata = getRouteMetadata(job.route)

      if (metadata) {
        expect(job.label, job.route).toBe(metadata.label)
      }
    }
  })

  it("distinguishes frontend state cleanup from backend-gated work", () => {
    expect(getOperationsRouteJob("/connectors")).toMatchObject({
      capabilityMode: "placeholder",
      diagnosticsPolicy: "not_applicable",
      implementationOwner: "next_page"
    })
    expect(getOperationsRouteJob("/scheduled-tasks")).toMatchObject({
      capabilityMode: "existing_probe",
      diagnosticsPolicy: "disclosed",
      implementationOwner: "shared_route"
    })
    expect(getOperationsRouteJob("/integrations")).toMatchObject({
      capabilityMode: "existing_probe",
      diagnosticsPolicy: "disclosed",
      implementationOwner: "shared_route"
    })
  })

  it("treats admin root as an overview with module drill-down routes", () => {
    expect(getOperationsRouteJob("/admin")).toMatchObject({
      capabilityMode: "frontend_state",
      implementationOwner: "next_page",
      relatedRoutes: [
        "/admin/server",
        "/admin/integrations",
        "/admin/sources",
        "/admin/monitoring"
      ]
    })
  })

  it("keeps connector placeholder child routes explicit and tied to supported alternatives", () => {
    for (const route of [
      "/connectors",
      "/connectors/browse",
      "/connectors/jobs",
      "/connectors/sources"
    ] as const) {
      expect(getOperationsRouteJob(route)).toMatchObject({
        concept: "connector",
        capabilityMode: "placeholder",
        diagnosticsPolicy: "not_applicable",
        implementationOwner: "next_page"
      })
    }

    expect(getOperationsRouteJob("/connectors/sources")?.relatedRoutes).toEqual(
      expect.arrayContaining(["/sources"])
    )
    expect(getOperationsRouteJob("/connectors/jobs")?.relatedRoutes).toEqual(
      expect.arrayContaining(["/scheduled-tasks", "/watchlists"])
    )
  })
})
