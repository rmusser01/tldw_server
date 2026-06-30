import { describe, expect, it } from "vitest"

import {
  DEFAULT_MCP_HUB_ROUTE_STATE,
  MCP_HUB_VIEW_KEYS,
  MCP_HUB_WORKFLOWS,
  getDefaultMcpHubView,
  resolveMcpHubRouteState,
  workflowForMcpHubView
} from "../mcpHubWorkflowConfig"

describe("mcpHubWorkflowConfig", () => {
  it("maps every current MCP Hub view key to exactly one workflow", () => {
    const configuredViews = Object.values(MCP_HUB_WORKFLOWS).flatMap(
      (workflow) => workflow.views
    )

    expect(new Set(configuredViews)).toEqual(new Set(MCP_HUB_VIEW_KEYS))
    expect(configuredViews).toHaveLength(MCP_HUB_VIEW_KEYS.length)

    for (const view of MCP_HUB_VIEW_KEYS) {
      expect(workflowForMcpHubView(view)).toBeTruthy()
    }
  })

  it("uses Setup / Servers & Credentials as the default route state", () => {
    expect(DEFAULT_MCP_HUB_ROUTE_STATE).toEqual({
      workflow: "setup",
      view: "credentials"
    })
    expect(getDefaultMcpHubView("setup")).toBe("credentials")
  })

  it("derives workflow from a valid view when query workflow disagrees", () => {
    expect(
      resolveMcpHubRouteState({
        workflow: "setup",
        view: "assignments"
      })
    ).toEqual({
      workflow: "access",
      view: "assignments"
    })
  })

  it("keeps a valid workflow deep link when the view query is invalid", () => {
    expect(
      resolveMcpHubRouteState({
        workflow: "workspaces",
        view: "not-real"
      })
    ).toEqual({
      workflow: "workspaces",
      view: "path-scopes"
    })
  })

  it("falls back to Setup / Servers & Credentials for invalid query values", () => {
    expect(
      resolveMcpHubRouteState({
        workflow: "missing",
        view: "not-real"
      })
    ).toEqual(DEFAULT_MCP_HUB_ROUTE_STATE)
  })
})
