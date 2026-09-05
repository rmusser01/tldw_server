// @vitest-environment jsdom
import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { render, screen, waitFor, within } from "@testing-library/react"

const apiMock = vi.hoisted(() => ({
  getSystemStats: vi.fn(),
  getSecurityAlertStatus: vi.fn(),
  listBackups: vi.fn(),
  getLlamacppStatus: vi.fn(),
  getMlxStatus: vi.fn(),
  getGovernorCoverage: vi.fn()
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: apiMock
}))

const connectionMock = vi.hoisted(() => ({
  serverUrl: "http://127.0.0.1:8000" as string | null
}))

vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionState: () => ({ serverUrl: connectionMock.serverUrl })
}))

import { AdminOperationsOverviewPage } from "../AdminOperationsOverviewPage"
import { ADMIN_MODULES } from "../admin-modules"

describe("AdminOperationsOverviewPage", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    connectionMock.serverUrl = "http://127.0.0.1:8000"
    apiMock.getSystemStats.mockResolvedValue({ users: { total: 1 } })
    apiMock.getSecurityAlertStatus.mockResolvedValue({ health: "ok" })
    apiMock.listBackups.mockResolvedValue({ backups: [] })
    apiMock.getLlamacppStatus.mockRejectedValue(new Error("Request failed: 503"))
    apiMock.getMlxStatus.mockResolvedValue({ active: false })
    apiMock.getGovernorCoverage.mockResolvedValue({ coverage_pct: 78.9 })
  })

  it("links every registered admin module (the overview is the complete map)", () => {
    render(<AdminOperationsOverviewPage />)

    expect(
      screen.getByRole("heading", { name: "Admin Operations" })
    ).toBeInTheDocument()

    const modules = screen.getByTestId("admin-operations-modules")
    for (const module of ADMIN_MODULES) {
      const link = within(modules).getByRole("link", { name: module.label })
      expect(link).toHaveAttribute("href", module.route)
    }
    // Guard against regressing to a partial list (2026-09 audit finding S1).
    expect(ADMIN_MODULES.length).toBeGreaterThanOrEqual(17)
  })

  it("speaks operator language, not implementation status", () => {
    render(<AdminOperationsOverviewPage />)

    expect(screen.queryByText("Route ready")).not.toBeInTheDocument()
    expect(screen.queryByText("Diagnostics")).not.toBeInTheDocument()
    expect(
      screen.queryByText("frontend_state", { exact: false })
    ).not.toBeInTheDocument()
    expect(
      screen.queryByText("Needs module configuration")
    ).not.toBeInTheDocument()
  })

  it("shows live module signals and degrades to the static map on failure", async () => {
    render(<AdminOperationsOverviewPage />)

    // Healthy / attention signals from resolved fetchers.
    expect(await screen.findByText("1 user")).toBeInTheDocument()
    expect(screen.getByText("Alerting healthy")).toBeInTheDocument()
    expect(screen.getByText("No backups yet")).toBeInTheDocument()
    expect(screen.getByText("No model loaded")).toBeInTheDocument()
    expect(screen.getByText("78.9% endpoint coverage")).toBeInTheDocument()

    // A failed fetcher renders as unavailable, not an error wall.
    expect(screen.getByText("Status unavailable")).toBeInTheDocument()

    // Modules without a signal fetcher render no badge at all.
    await waitFor(() => {
      const badges = screen.getAllByTestId("admin-module-signal")
      expect(badges).toHaveLength(6)
    })
  })

  it("renders a deliberately-disabled backend as 'Not configured', not an outage (#2894)", async () => {
    apiMock.getLlamacppStatus.mockRejectedValue(
      new Error(
        "Managed llama.cpp backend is not configured. Enable [LlamaCpp] enabled=true."
      )
    )

    render(<AdminOperationsOverviewPage />)

    expect(await screen.findByText("Not configured")).toBeInTheDocument()
    expect(screen.queryByText("Status unavailable")).not.toBeInTheDocument()
  })

  it("links each signal badge to its module (#2899)", async () => {
    render(<AdminOperationsOverviewPage />)

    const badge = await screen.findByText("1 user")
    expect(badge.closest("a")).toHaveAttribute("href", "/admin/server")
  })

  it("shows a connect-first banner when no server is configured (#2893)", async () => {
    connectionMock.serverUrl = null

    render(<AdminOperationsOverviewPage />)

    const banner = screen.getByTestId("admin-not-connected-banner")
    expect(banner).toHaveTextContent("Not connected to a tldw server")
    expect(within(banner).getByRole("link", { name: "Connect" })).toHaveAttribute(
      "href",
      "/setup"
    )
    // The module map stays visible beneath the banner.
    expect(screen.getByTestId("admin-operations-modules")).toBeInTheDocument()
  })

  it("omits the connect banner when a server is configured (#2893)", () => {
    render(<AdminOperationsOverviewPage />)

    expect(
      screen.queryByTestId("admin-not-connected-banner")
    ).not.toBeInTheDocument()
  })
})
