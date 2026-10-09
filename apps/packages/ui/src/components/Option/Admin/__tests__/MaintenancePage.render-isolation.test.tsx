// @vitest-environment jsdom

import React from "react"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"
import MaintenancePage from "../MaintenancePage"

// Count every Table render: keystrokes in the maintenance banner must not
// re-render the flags/incidents/rotation tables (C-S5 render isolation).
const tableRenders = vi.hoisted(() => ({ count: 0 }))

vi.mock("antd", async () => {
  const actual = await vi.importActual<typeof import("antd")>("antd")
  const { createElement } = await import("react")
  return {
    ...actual,
    Table: (props: unknown) => {
      tableRenders.count += 1
      return createElement(actual.Table, props as never)
    }
  }
})

const apiMock = vi.hoisted(() => ({
  getMaintenanceState: vi.fn(),
  listFeatureFlags: vi.fn(),
  listIncidents: vi.fn(),
  listRotationRuns: vi.fn(),
  updateMaintenanceState: vi.fn(),
  updateFeatureFlag: vi.fn(),
  deleteFeatureFlag: vi.fn(),
  createIncident: vi.fn(),
  updateIncident: vi.fn(),
  deleteIncident: vi.fn(),
  createRotationRun: vi.fn()
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: apiMock
}))

describe("MaintenancePage banner keystroke isolation (C-S5)", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    tableRenders.count = 0
    apiMock.getMaintenanceState.mockResolvedValue({
      enabled: false,
      message: "old",
      allowlist: ["1.2.3.4"]
    })
    apiMock.listFeatureFlags.mockResolvedValue([
      { key: "f1", enabled: true }
    ])
    apiMock.listIncidents.mockResolvedValue([
      { id: 3, title: "Incident one", status: "investigating", severity: "low" }
    ])
    apiMock.listRotationRuns.mockResolvedValue([
      { id: 9, status: "completed", started_at: "2026-01-01T00:00:00Z", completed_at: null }
    ])
    apiMock.updateMaintenanceState.mockResolvedValue({ enabled: true })
  })

  it("typing in the banner re-renders the form only, not the three tables", async () => {
    render(<MaintenancePage />)

    // Flags + incidents tables mount with their data.
    await screen.findByText("f1")
    await screen.findByText("Incident one")

    // Rotation runs live in a collapsed panel - expand so its table mounts.
    const user = userEvent.setup()
    await user.click(screen.getByText("Rotation Runs"))
    await screen.findByText("9")

    tableRenders.count = 0

    const messageArea = screen.getByDisplayValue("old")
    await user.clear(messageArea)
    await user.type(messageArea, "Scheduled downtime")
    expect(messageArea).toHaveValue("Scheduled downtime")

    // Allowlist typing is isolated too.
    const allowlistInput = screen.getByDisplayValue("1.2.3.4")
    await user.type(allowlistInput, ", 10.0.0.1")
    expect(allowlistInput).toHaveValue("1.2.3.4, 10.0.0.1")

    expect(tableRenders.count).toBe(0)

    // Explicit Apply: the existing Save Changes button commits the banner's
    // draft values (enabled + message + allowlist) in one call.
    await user.click(screen.getAllByRole("switch")[0])
    await user.click(screen.getByRole("button", { name: "Save Changes" }))

    expect(apiMock.updateMaintenanceState).toHaveBeenCalledWith({
      enabled: true,
      message: "Scheduled downtime",
      allowlist: ["1.2.3.4", "10.0.0.1"]
    })
  })
})
