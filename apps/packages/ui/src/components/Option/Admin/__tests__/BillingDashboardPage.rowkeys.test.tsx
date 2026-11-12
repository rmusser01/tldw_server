// @vitest-environment jsdom

import React from "react"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
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
    listAllSubscriptions: (...args: unknown[]) => mocks.listAllSubscriptions(...args),
    listBillingEvents: (...args: unknown[]) => mocks.listBillingEvents(...args)
  }
}))

import { clearCapabilityProbeCacheForTests } from "@/services/tldw/capability-probe"
import BillingDashboardPage from "../BillingDashboardPage"
import { formatAdminDateTime } from "../admin-format"

const fetchMock = vi.fn()
vi.stubGlobal("fetch", fetchMock)

const tableRowKeys = () =>
  Array.from(document.querySelectorAll("tr.ant-table-row")).map((row) =>
    row.getAttribute("data-row-key")
  )

describe("BillingDashboardPage stable row keys (C-S5)", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    clearCapabilityProbeCacheForTests()
    mocks.useCanonicalConnectionConfig.mockReturnValue({
      config: {
        serverUrl: "http://127.0.0.1:8000",
        authMode: "single-user",
        apiKey: "test-key"
      },
      loading: false
    })
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({
        paths: {
          "/api/v1/admin/billing/overview": {}
        }
      })
    })
    mocks.getBillingOverview.mockResolvedValue({
      mrr: 0,
      active_subscriptions: 0,
      canceled_subscriptions: 0,
      past_due_subscriptions: 0
    })
    mocks.getStorageQuotaSummary.mockResolvedValue({
      total_quotas: 0,
      items: []
    })
  })

  it("keys subscriptions by the user/created composite, never Math.random", async () => {
    // No user_id and no id on either row: the old fallback returned
    // Math.random() numbers that churned on every render.
    mocks.listAllSubscriptions.mockResolvedValue({
      items: [
        { plan_id: "pro", status: "active", created_at: "2026-01-01T00:00:00Z" },
        { plan_id: "pro", status: "active", created_at: "2026-01-01T00:00:00Z" }
      ],
      total: 2
    })
    mocks.listBillingEvents.mockResolvedValue({ items: [], total: 0 })

    const view = render(<BillingDashboardPage />)

    await screen.findByText("Quota Records")
    const user = userEvent.setup()
    await user.click(screen.getByRole("tab", { name: "Subscriptions" }))

    await screen.findAllByText("pro")
    const keys = tableRowKeys()
    expect(new Set(keys).size).toBe(keys.length)
    expect(keys).toEqual([
      "u|2026-01-01T00:00:00Z",
      "u|2026-01-01T00:00:00Z#2"
    ])

    view.rerender(<BillingDashboardPage />)
    expect(tableRowKeys()).toEqual(keys)
    for (const key of keys) {
      expect(key).not.toMatch(/^[0-9.]+$/) // no bare Math.random() numbers
    }

    // The subscriptions table's created cell renders through the shared
    // formatter (was toLocaleDateString before C-S5).
    expect(
      screen.getAllByText(formatAdminDateTime("2026-01-01T00:00:00Z")).length
    ).toBeGreaterThan(0)
  })

  it("keys billing events by the user/created composite", async () => {
    mocks.listAllSubscriptions.mockResolvedValue({ items: [], total: 0 })
    mocks.listBillingEvents.mockResolvedValue({
      items: [
        { event_type: "subscription.created", user_id: 7, amount: 5, description: "pro plan", created_at: "2026-02-01T09:00:00Z" },
        { event_type: "credits.granted", user_id: 7, amount: 1, description: "grant", created_at: "2026-02-01T09:00:00Z" }
      ],
      total: 2
    })

    render(<BillingDashboardPage />)

    await screen.findByText("Quota Records")
    const user = userEvent.setup()
    await user.click(screen.getByRole("tab", { name: "Billing Events" }))

    await screen.findByText("subscription.created")
    const keys = tableRowKeys()
    expect(new Set(keys).size).toBe(keys.length)
    expect(keys).toEqual([
      "7|2026-02-01T09:00:00Z",
      "7|2026-02-01T09:00:00Z#2"
    ])

    // The events table's created cell renders through the shared formatter.
    expect(
      screen.getAllByText(formatAdminDateTime("2026-02-01T09:00:00Z")).length
    ).toBeGreaterThan(0)
  })
})
