// @vitest-environment jsdom
import React, { Suspense } from "react"
import { render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import {
  MemoryRouter,
  Route,
  Routes,
  useLocation
} from "react-router-dom"

const routeMocks = vi.hoisted(() => ({
  standaloneMcpHub: vi.fn(() =>
    React.createElement(
      "div",
      { "data-testid": "standalone-mcp-hub" },
      "Standalone MCP Hub Wrapper"
    )
  ),
  settingsMcpHub: vi.fn(() =>
    React.createElement(
      "div",
      { "data-testid": "settings-mcp-hub" },
      "Settings MCP Hub Wrapper"
    )
  )
}))

vi.mock("../../extension/routes/option-mcp-hub", () => ({
  __esModule: true,
  default: routeMocks.standaloneMcpHub
}))

vi.mock("../../extension/routes/option-settings-mcp-hub", () => ({
  __esModule: true,
  default: routeMocks.settingsMcpHub
}))

const renderRoute = (element: React.ReactElement) =>
  render(
    React.createElement(
      Suspense,
      { fallback: React.createElement("div", { "data-testid": "route-fallback" }) },
      element
    )
  )

const RedirectLocationProbe = () => {
  const location = useLocation()

  return React.createElement(
    "div",
    { "data-testid": "redirect-location" },
    `${location.pathname}${location.search}`
  )
}

describe("extension route registry MCP Hub parity", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("renders the standalone MCP Hub route from the extension registry", async () => {
    const { ROUTE_DEFINITIONS } = await import(
      "../../extension/routes/route-registry"
    )
    const route = ROUTE_DEFINITIONS.find((candidate) => candidate.path === "/mcp-hub")

    expect(route).toBeDefined()
    renderRoute(route!.element)

    expect(await screen.findByTestId("standalone-mcp-hub")).toBeVisible()
    expect(screen.queryByTestId("settings-mcp-hub")).not.toBeInTheDocument()
  }, 15_000)

  it("renders the settings MCP Hub route from the extension registry", async () => {
    const { ROUTE_DEFINITIONS } = await import(
      "../../extension/routes/route-registry"
    )
    const route = ROUTE_DEFINITIONS.find(
      (candidate) => candidate.path === "/settings/mcp-hub"
    )

    expect(route).toBeDefined()
    expect(route?.nav).toBeDefined()
    expect(route?.nav?.group).toBe("server")
    expect(route?.nav?.labelToken).toBe("settings:mcpHubNav")
    renderRoute(route!.element)

    expect(await screen.findByTestId("settings-mcp-hub")).toBeVisible()
    expect(routeMocks.settingsMcpHub).toHaveBeenCalledTimes(1)
    expect(screen.queryByTestId("standalone-mcp-hub")).not.toBeInTheDocument()
  }, 15_000)

  it("uses react-router redirects for legacy extension routes", async () => {
    const { ROUTE_DEFINITIONS } = await import(
      "../../extension/routes/route-registry"
    )
    const route = ROUTE_DEFINITIONS.find(
      (candidate) => candidate.path === "/prompt-studio"
    )

    expect(route).toBeDefined()

    render(
      React.createElement(
        MemoryRouter,
        { initialEntries: ["/prompt-studio"] },
        React.createElement(
          Routes,
          null,
          React.createElement(Route, {
            path: "/prompt-studio",
            element: route!.element
          }),
          React.createElement(Route, {
            path: "/prompts",
            element: React.createElement(RedirectLocationProbe)
          })
        )
      )
    )

    expect(await screen.findByTestId("redirect-location")).toHaveTextContent(
      "/prompts?tab=studio"
    )
  })
})
