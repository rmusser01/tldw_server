// @vitest-environment jsdom
import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { render, screen, waitFor, within } from "@testing-library/react"

const apiMock = vi.hoisted(() => ({
  getSystemStats: vi.fn(),
  getSecurityAlertStatus: vi.fn(),
  listBackups: vi.fn(),
  listBackupSchedules: vi.fn(),
  listAlertRules: vi.fn(),
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
    window.localStorage.clear()
    apiMock.getSystemStats.mockResolvedValue({ users: { total: 1 } })
    apiMock.getSecurityAlertStatus.mockResolvedValue({ health: "ok" })
    apiMock.listBackups.mockResolvedValue({ backups: [] })
    // Checklist probes default to "all done" so the first-steps card stays
    // out of the way of the unrelated tests.
    apiMock.listBackupSchedules.mockResolvedValue({ schedules: [{ id: 1 }] })
    apiMock.listAlertRules.mockResolvedValue([{ id: 1 }])
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

  it("shows the first-steps checklist while setup tasks remain (#2899 I6)", async () => {
    // Default mocks: schedules + rules exist, coverage 78.9% (< 80) remains.
    render(<AdminOperationsOverviewPage />)

    const card = await screen.findByTestId("admin-first-steps")
    expect(
      within(card).getByRole("link", { name: "Review unprotected endpoints" })
    ).toHaveAttribute("href", "/admin/rate-limiting")
    // Done items render struck-through, not as links.
    expect(
      within(card).queryByRole("link", { name: "Create a backup schedule" })
    ).not.toBeInTheDocument()
    expect(within(card).getByText("Create a backup schedule")).toBeInTheDocument()
  })

  it("hides the checklist once every step is done (#2899 I6)", async () => {
    apiMock.getGovernorCoverage.mockResolvedValue({ coverage_pct: 92 })

    render(<AdminOperationsOverviewPage />)

    // Signals settle (badge appears) without the checklist ever rendering.
    expect(await screen.findByText("1 user")).toBeInTheDocument()
    expect(screen.queryByTestId("admin-first-steps")).not.toBeInTheDocument()
  })

  it("dismissal hides the checklist and persists per server (#2899 I6)", async () => {
    const { unmount } = render(<AdminOperationsOverviewPage />)

    const card = await screen.findByTestId("admin-first-steps")
    within(card).getByRole("button", { name: "Dismiss" }).click()
    await waitFor(() => {
      expect(screen.queryByTestId("admin-first-steps")).not.toBeInTheDocument()
    })

    // A fresh mount of the same server keeps it dismissed...
    unmount()
    render(<AdminOperationsOverviewPage />)
    expect(await screen.findByText("1 user")).toBeInTheDocument()
    expect(screen.queryByTestId("admin-first-steps")).not.toBeInTheDocument()

    // ...but a different server gets its own checklist.
    expect(
      window.localStorage.getItem(
        "__tldw_admin_first_steps_dismissed::http://127.0.0.1:8000"
      )
    ).toBe("1")
  })

  it("omits the connect banner when a server is configured (#2893)", () => {
    render(<AdminOperationsOverviewPage />)

    expect(
      screen.queryByTestId("admin-not-connected-banner")
    ).not.toBeInTheDocument()
  })
})
