import React from "react"
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { MemoryRouter, useLocation } from "react-router-dom"
import { ROUTE_ERROR_FIXTURE_QUERY_KEY, RouteErrorBoundary } from "../RouteErrorBoundary"

const LocationProbe = () => {
  const location = useLocation()
  return <div data-testid="route-path">{location.pathname}</div>
}

const StableChild = () => <div data-testid="stable-child">Stable content</div>

let shouldThrow = true
let allowRecovery = false
const ThrowOnceChild = () => {
  if (shouldThrow && !allowRecovery) {
    throw new Error("Transient render error")
  }
  return <div data-testid="recovered-child">Recovered content</div>
}

const AlwaysThrowChild = () => {
  throw new Error("Route failure")
}

describe("RouteErrorBoundary", () => {
  beforeEach(() => {
    shouldThrow = true
    allowRecovery = false
    vi.spyOn(console, "error").mockImplementation(() => {})
  })

  afterEach(() => {
    vi.unstubAllEnvs()
    vi.restoreAllMocks()
  })

  it("renders route content when no error is thrown", () => {
    render(
      <MemoryRouter initialEntries={["/chat"]}>
        <RouteErrorBoundary routeId="chat" routeLabel="Chat">
          <StableChild />
        </RouteErrorBoundary>
      </MemoryRouter>
    )

    expect(screen.getByTestId("stable-child")).toBeInTheDocument()
    expect(screen.queryByTestId("error-boundary")).not.toBeInTheDocument()
  })

  it("renders contract fallback UI when route content throws", () => {
    render(
      <MemoryRouter initialEntries={["/media"]}>
        <RouteErrorBoundary routeId="media" routeLabel="Media">
          <AlwaysThrowChild />
        </RouteErrorBoundary>
      </MemoryRouter>
    )

    expect(screen.getByTestId("error-boundary")).toBeInTheDocument()
    expect(screen.getByTestId("route-error-boundary-media")).toBeInTheDocument()
    expect(screen.getByTestId("route-error-title")).toHaveTextContent(
      "This page hit an unexpected error"
    )
    expect(screen.getByTestId("route-error-route-label")).toHaveTextContent("Media")
    expect(screen.getByTestId("route-error-retry")).toBeInTheDocument()
    expect(screen.getByTestId("route-error-go-chat")).toBeInTheDocument()
    expect(screen.getByTestId("route-error-open-settings")).toBeInTheDocument()
    expect(screen.getByTestId("route-error-reload")).toBeInTheDocument()
  })

  it("applies focus-visible classes to route recovery actions", () => {
    render(
      <MemoryRouter initialEntries={["/media"]}>
        <RouteErrorBoundary routeId="media" routeLabel="Media">
          <AlwaysThrowChild />
        </RouteErrorBoundary>
      </MemoryRouter>
    )

    const controls = [
      screen.getByTestId("route-error-retry"),
      screen.getByTestId("route-error-go-chat"),
      screen.getByTestId("route-error-open-settings"),
      screen.getByTestId("route-error-reload"),
    ]

    for (const control of controls) {
      expect(control.className).toContain("focus-visible:ring-2")
      expect(control.className).toContain("focus-visible:ring-focus")
      expect(control.className).toContain("focus-visible:ring-offset-2")
      expect(control.className).toContain("focus-visible:ring-offset-bg")
    }
  })

  it("retries and remounts route content after a transient error", async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter initialEntries={["/prompts"]}>
        <RouteErrorBoundary routeId="prompts" routeLabel="Prompts">
          <ThrowOnceChild />
        </RouteErrorBoundary>
      </MemoryRouter>
    )

    expect(screen.getByTestId("error-boundary")).toBeInTheDocument()
    allowRecovery = true
    shouldThrow = false
    await user.click(screen.getByTestId("route-error-retry"))

    expect(await screen.findByTestId("recovered-child")).toBeInTheDocument()
    expect(screen.queryByTestId("error-boundary")).not.toBeInTheDocument()
  })

  it("uses route recovery navigation actions", async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter initialEntries={["/knowledge"]}>
        <LocationProbe />
        <RouteErrorBoundary routeId="knowledge" routeLabel="Knowledge QA">
          <AlwaysThrowChild />
        </RouteErrorBoundary>
      </MemoryRouter>
    )

    expect(screen.getByTestId("route-path")).toHaveTextContent("/knowledge")
    await user.click(screen.getByTestId("route-error-open-settings"))
    await waitFor(() => {
      expect(screen.getByTestId("route-path")).toHaveTextContent("/settings/tldw")
    })

    await user.click(screen.getByTestId("route-error-go-chat"))
    await waitFor(() => {
      expect(screen.getByTestId("route-path")).toHaveTextContent("/")
    })
  })

  it("supports forced-error fixture via query param in non-production mode", () => {
    render(
      <MemoryRouter initialEntries={[`/admin/server?${ROUTE_ERROR_FIXTURE_QUERY_KEY}=admin-server`]}>
        <RouteErrorBoundary routeId="admin-server" routeLabel="Server Admin">
          <StableChild />
        </RouteErrorBoundary>
      </MemoryRouter>
    )

    expect(screen.getByTestId("error-boundary")).toBeInTheDocument()
    expect(screen.getByTestId("route-error-boundary-admin-server")).toBeInTheDocument()
    expect(screen.getByTestId("route-error-route-label")).toHaveTextContent("Server Admin")
  })

  it("ignores the forced-error fixture in production mode", () => {
    vi.stubEnv("NODE_ENV", "production")

    render(
      <MemoryRouter initialEntries={[`/kanban?${ROUTE_ERROR_FIXTURE_QUERY_KEY}=kanban`]}>
        <RouteErrorBoundary routeId="kanban" routeLabel="Kanban">
          <StableChild />
        </RouteErrorBoundary>
      </MemoryRouter>
    )

    expect(screen.getByTestId("stable-child")).toBeInTheDocument()
    expect(screen.queryByTestId("error-boundary")).not.toBeInTheDocument()
  })
})
