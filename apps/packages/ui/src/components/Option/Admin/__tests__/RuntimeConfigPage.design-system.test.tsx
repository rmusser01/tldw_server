// @vitest-environment jsdom

import React from "react"
import { render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import RuntimeConfigPage from "../RuntimeConfigPage"

const apiMock = vi.hoisted(() => ({
  getCleanupSettings: vi.fn(),
  getRegistrationSettings: vi.fn(),
  updateCleanupSettings: vi.fn(),
  updateRegistrationSettings: vi.fn()
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

describe("RuntimeConfigPage design-system states", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    apiMock.getCleanupSettings.mockResolvedValue({
      soft_delete_retention_days: 30,
      orphan_cleanup_enabled: true,
      orphan_cleanup_interval_hours: 24,
      log_retention_days: 14
    })
    apiMock.getRegistrationSettings.mockResolvedValue({
      open_registration: false,
      require_email_verification: true,
      default_role: "user",
      max_users: 0
    })
  })

  it("renders forbidden guard feedback through the design-system Alert primitive", async () => {
    apiMock.getCleanupSettings.mockRejectedValueOnce({ status: 403 })

    render(<RuntimeConfigPage />)

    const alert = await expectDesignSystemAlertForTitle("Forbidden")
    expect(alert).toHaveTextContent(
      "You do not have permission to view runtime configuration."
    )
  })

  it("renders missing-endpoint guard feedback through the design-system Alert primitive", async () => {
    apiMock.getCleanupSettings.mockRejectedValueOnce({ status: 404 })

    render(<RuntimeConfigPage />)

    const alert = await expectDesignSystemAlertForTitle("Not Available")
    expect(alert).toHaveTextContent(
      "The runtime configuration endpoints are not available on this server."
    )
  })
})
