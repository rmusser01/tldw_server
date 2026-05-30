// @vitest-environment jsdom

import React from "react"
import { render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import MaintenancePage from "../MaintenancePage"

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

const expectDesignSystemAlertForTitle = async (titleText: string) => {
  const title = await screen.findByText(titleText)
  const alert = title.closest('[data-ds-component="Alert"]')

  expect(alert).not.toBeNull()
  const alertEl = alert as HTMLElement
  expect(alertEl).toHaveAttribute("role", "alert")
  return alertEl
}

describe("MaintenancePage design-system states", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    apiMock.getMaintenanceState.mockResolvedValue({
      enabled: false,
      message: "",
      allowlist: []
    })
    apiMock.listFeatureFlags.mockResolvedValue([])
    apiMock.listIncidents.mockResolvedValue([])
    apiMock.listRotationRuns.mockResolvedValue([])
  })

  it("renders forbidden guard feedback through the design-system Alert primitive", async () => {
    apiMock.getMaintenanceState.mockRejectedValueOnce({ status: 403 })

    render(<MaintenancePage />)

    const alert = await expectDesignSystemAlertForTitle("Access Denied")
    expect(alert).toHaveTextContent(
      "You don't have permission to access the maintenance console."
    )
  })

  it("renders missing-endpoint guard feedback through the design-system Alert primitive", async () => {
    apiMock.getMaintenanceState.mockRejectedValueOnce({ status: 404 })

    render(<MaintenancePage />)

    const alert = await expectDesignSystemAlertForTitle("Not Available")
    expect(alert).toHaveTextContent(
      "The maintenance console is not available on this server."
    )
  })
})
