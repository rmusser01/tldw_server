import { describe, expect, it } from "vitest"

import {
  ROUTE_DEFINITIONS,
  resolveRouteAliasDestination
} from "../../extension/routes/route-registry"

describe("Research Studio extension route aliases", () => {
  it("registers /research-studio as the canonical extension route", () => {
    expect(
      ROUTE_DEFINITIONS.some((route) => route.path === "/research-studio")
    ).toBe(true)
  })

  it("keeps both legacy extension routes as aliases", () => {
    expect(
      ROUTE_DEFINITIONS.some((route) => route.path === "/workspace-playground")
    ).toBe(true)
    expect(
      ROUTE_DEFINITIONS.some((route) => route.path === "/workspace-studio")
    ).toBe(true)
  })

  it("resolves alias redirects as string destinations for the Next router shim", () => {
    expect(
      resolveRouteAliasDestination("/research-studio", {
        search: "?tab=studio&shared=abc",
        hash: "#workspace-studio-panel"
      })
    ).toBe("/research-studio?tab=studio&shared=abc#workspace-studio-panel")
  })

  it("does not append legacy route state when the target already has route state", () => {
    expect(
      resolveRouteAliasDestination("/research-studio?tab=chat", {
        search: "?tab=studio",
        hash: "#workspace-studio-panel"
      })
    ).toBe("/research-studio?tab=chat")
  })
})
