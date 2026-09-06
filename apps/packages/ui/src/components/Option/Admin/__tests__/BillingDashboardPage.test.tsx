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

    const alert = await expectDesignSystemAlertForTitle(
      "Not available on this server"
    )
    expect(alert).toHaveTextContent("Billing endpoints are not enabled here.")
    expect(mocks.getBillingOverview).not.toHaveBeenCalled()
    expect(mocks.listAllSubscriptions).not.toHaveBeenCalled()
    expect(mocks.listBillingEvents).not.toHaveBeenCalled()
  })

  it("aggregates the storage summary from the real {total_quotas, items} envelope", async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({
        paths: {
          "/api/v1/admin/billing/overview": {}
        }
      })
    })
    mocks.getBillingOverview.mockResolvedValueOnce({
      mrr: 0,
      active_subscriptions: 0,
      canceled_subscriptions: 0,
      past_due_subscriptions: 0
    })
    // Actual StorageQuotaSummaryResponse shape: no flat total_used_mb /
    // avg_utilization_pct fields exist - the page must aggregate items.
    mocks.getStorageQuotaSummary.mockResolvedValueOnce({
      total_quotas: 2,
      items: [
        { id: 1, org_id: 1, quota_mb: 1000, used_mb: 250 },
        { id: 2, org_id: 2, quota_mb: 1000, used_mb: 250 }
      ]
    })

    render(<BillingDashboardPage />)

    expect(await screen.findByText("Quota Records")).toBeInTheDocument()
    expect(screen.getByText("Total Used (MB)")).toBeInTheDocument()
    // 500 used of 2000 total -> 25.0% (antd splits digits across spans, so
    // assert on the statistic container's text)
    const utilizationStat = screen.getByText("Utilization").closest(".ant-statistic")
    expect(utilizationStat?.textContent).toContain("25.0")
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
