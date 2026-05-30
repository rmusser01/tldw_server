// @vitest-environment jsdom

import React from "react"
import { render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  useCanonicalConnectionConfig: vi.fn(),
  getBillingOverview: vi.fn(),
  getStorageQuotaSummary: vi.fn(),
  listAllSubscriptions: vi.fn(),
  listBillingEvents: vi.fn()
}))

vi.mock("@/hooks/useCanonicalConnectionConfig", () => ({
  useCanonicalConnectionConfig: (...args: unknown[]) =>
    mocks.useCanonicalConnectionConfig(...args)
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getBillingOverview: (...args: unknown[]) => mocks.getBillingOverview(...args),
    getStorageQuotaSummary: (...args: unknown[]) =>
      mocks.getStorageQuotaSummary(...args),
    listAllSubscriptions: (...args: unknown[]) =>
      mocks.listAllSubscriptions(...args),
    listBillingEvents: (...args: unknown[]) => mocks.listBillingEvents(...args)
  }
}))

import BillingDashboardPage from "../BillingDashboardPage"

const fetchMock = vi.fn()
vi.stubGlobal("fetch", fetchMock)

const expectDesignSystemAlertForTitle = async (titleText: string) => {
  const title = await screen.findByText(titleText)
  const alert = title.closest('[data-ds-component="Alert"]')

  expect(alert).not.toBeNull()
  const alertEl = alert as HTMLElement
  expect(alertEl).toHaveAttribute("role", "alert")
  return alertEl
}

describe("BillingDashboardPage", () => {
  beforeEach(() => {
    vi.clearAllMocks()

    mocks.useCanonicalConnectionConfig.mockReturnValue({
      config: {
        serverUrl: "http://127.0.0.1:8000",
        authMode: "single-user",
        apiKey: "test-key"
      },
      loading: false
    })
  })

  it("shows an unsupported-state message without calling billing endpoints when the route is absent", async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({
        paths: {}
      })
    })

    render(<BillingDashboardPage />)

    const alert = await expectDesignSystemAlertForTitle("Not Available")
    expect(
      alert
    ).toHaveTextContent("Billing endpoints are not available on this server.")
    expect(mocks.getBillingOverview).not.toHaveBeenCalled()
    expect(mocks.listAllSubscriptions).not.toHaveBeenCalled()
    expect(mocks.listBillingEvents).not.toHaveBeenCalled()
  })

  it("renders forbidden guard feedback through the design-system Alert primitive", async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({
        paths: {
          "/api/v1/admin/billing/overview": {}
        }
      })
    })
    mocks.getBillingOverview.mockRejectedValueOnce({ status: 403 })
    mocks.getStorageQuotaSummary.mockResolvedValueOnce({})

    render(<BillingDashboardPage />)

    const alert = await expectDesignSystemAlertForTitle("Access Denied")
    expect(alert).toHaveTextContent(
      "You do not have permission to view the billing dashboard."
    )
  })
})
