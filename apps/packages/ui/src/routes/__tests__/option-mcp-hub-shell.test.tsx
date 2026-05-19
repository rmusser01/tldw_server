// @vitest-environment jsdom
import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

vi.mock("~/components/Layouts/Layout", () => ({
  default: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="option-layout">{children}</div>
  )
}))

vi.mock("@/components/Common/PageShell", () => ({
  PageShell: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="page-shell">{children}</div>
  )
}))

vi.mock("@/components/Common/RouteErrorBoundary", () => ({
  RouteErrorBoundary: ({
    children,
    routeId,
    routeLabel
  }: {
    children: React.ReactNode
    routeId: string
    routeLabel: string
  }) => (
    <section
      data-route-id={routeId}
      data-route-label={routeLabel}
      data-testid="route-error-boundary"
    >
      {children}
    </section>
  )
}))

vi.mock("@/components/Option/MCPHub", () => ({
  McpHubPage: () => <div data-testid="mcp-hub-page">MCP Hub page</div>
}))

import OptionMcpHub from "../option-mcp-hub"

describe("option MCP Hub route shell", () => {
  it("wraps the standalone MCP Hub route in a route error boundary", () => {
    render(<OptionMcpHub />)

    const boundary = screen.getByTestId("route-error-boundary")
    expect(boundary).toHaveAttribute("data-route-id", "mcp-hub")
    expect(boundary).toHaveAttribute("data-route-label", "MCP Hub")
    expect(screen.getByTestId("mcp-hub-page")).toBeVisible()
  })
})
