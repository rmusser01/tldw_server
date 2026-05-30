// @vitest-environment jsdom

import React from "react"
import { render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import UsageAnalyticsPage from "../UsageAnalyticsPage"

const apiMock = vi.hoisted(() => ({
  getDailyUsage: vi.fn(),
  getTopUsage: vi.fn(),
  getLlmUsage: vi.fn(),
  getLlmUsageSummary: vi.fn(),
  getLlmTopSpenders: vi.fn(),
  getRouterAnalyticsProviders: vi.fn(),
  exportDailyUsageCsv: vi.fn(),
  exportTopUsageCsv: vi.fn()
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

describe("UsageAnalyticsPage design-system states", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    apiMock.getDailyUsage.mockResolvedValue([])
    apiMock.getTopUsage.mockResolvedValue([])
    apiMock.getLlmUsage.mockResolvedValue([])
    apiMock.getLlmUsageSummary.mockResolvedValue({})
    apiMock.getLlmTopSpenders.mockResolvedValue([])
    apiMock.getRouterAnalyticsProviders.mockResolvedValue([])
    apiMock.exportDailyUsageCsv.mockResolvedValue("")
    apiMock.exportTopUsageCsv.mockResolvedValue("")
  })

  it("renders forbidden guard feedback through the design-system Alert primitive", async () => {
    apiMock.getDailyUsage.mockRejectedValueOnce({ status: 403 })

    render(<UsageAnalyticsPage />)

    const alert = await expectDesignSystemAlertForTitle("Access Denied")
    expect(alert).toHaveTextContent(
      "You don't have permission to access usage analytics."
    )
  })

  it("renders missing-endpoint guard feedback through the design-system Alert primitive", async () => {
    apiMock.getDailyUsage.mockRejectedValueOnce({ status: 404 })

    render(<UsageAnalyticsPage />)

    const alert = await expectDesignSystemAlertForTitle("Not Available")
    expect(alert).toHaveTextContent(
      "Usage analytics is not available on this server."
    )
  })
})
