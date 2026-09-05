// @vitest-environment jsdom
import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { render, screen, waitFor, within } from "@testing-library/react"

const apiMock = vi.hoisted(() => ({
  getSystemStats: vi.fn(),
  listAlertHistory: vi.fn(),
  listBackups: vi.fn(),
  getLlamacppStatus: vi.fn(),
  getMlxStatus: vi.fn(),
  getGovernorCoverage: vi.fn()
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: apiMock
}))

import { AdminOperationsOverviewPage } from "../AdminOperationsOverviewPage"
import { ADMIN_MODULES } from "../admin-modules"

describe("AdminOperationsOverviewPage", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    apiMock.getSystemStats.mockResolvedValue({ users: { total: 1 } })
    apiMock.listAlertHistory.mockResolvedValue([])
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
    expect(screen.getByText("No open alerts")).toBeInTheDocument()
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
})
