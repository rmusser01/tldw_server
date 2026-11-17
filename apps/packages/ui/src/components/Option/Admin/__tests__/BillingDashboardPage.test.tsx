// @vitest-environment jsdom

import React from "react"
import { render, screen, act } from "@testing-library/react"
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
    listAllSubscriptions: (...args: unknown[]) =>
      mocks.listAllSubscriptions(...args),
    listBillingEvents: (...args: unknown[]) => mocks.listBillingEvents(...args)
  }
}))

import { clearCapabilityProbeCacheForTests } from "@/services/tldw/capability-probe"
import BillingDashboardPage, { aggregateStorageSummary } from "../BillingDashboardPage"

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
    // The capability probe caches the openapi spec per server URL at module
    // scope; reset it so each test's fetch mock owns its own spec answer.
    clearCapabilityProbeCacheForTests()

    mocks.useCanonicalConnectionConfig.mockReturnValue({
      config: {
        serverUrl: "http://127.0.0.1:8000",
        authMode: "single-user",
        apiKey: "test-key"
      },
      loading: false
    })
  })

  it("downgrades the billing content in place when the route is absent, without reaching the lazy tabs", async () => {
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
    // The page chrome (tabs) survives the downgrade - only the tab content
    // is swapped for the notice.
    expect(screen.getAllByRole("tab")).toHaveLength(3)
    // Tabs render immediately, so the Overview tab fires exactly one
    // speculative load before the probe downgrades it...
    expect(mocks.getBillingOverview).toHaveBeenCalledTimes(1)
    // ...while the lazy Subscriptions/Events tabs never mount at all.
    expect(mocks.listAllSubscriptions).not.toHaveBeenCalled()
    expect(mocks.listBillingEvents).not.toHaveBeenCalled()
  })

  it("keeps the in-place downgrade when the speculative overview 404 lands after the probe rules the routes absent", async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({
        paths: {}
      })
    })
    // Sequence the race deterministically: the probe downgrades first, the
    // speculative overview rejects 404 afterwards.
    let rejectOverview!: (reason: unknown) => void
    mocks.getBillingOverview.mockImplementation(
      () => new Promise((_, reject) => { rejectOverview = reject })
    )
    mocks.getStorageQuotaSummary.mockResolvedValue({})

    render(<BillingDashboardPage />)

    // Probe outcome: tabs + inline unavailable notice.
    expect(await screen.findAllByRole("tab")).toHaveLength(3)
    await expectDesignSystemAlertForTitle("Not available on this server")

    // The late 404 must not flip the page into the full-page notFound guard:
    // the inline downgrade is the durable end state.
    await act(async () => {
      rejectOverview({ status: 404 })
    })

    expect(screen.getAllByRole("tab")).toHaveLength(3)
    await expectDesignSystemAlertForTitle("Not available on this server")
  })

  it("recovers the in-place downgrade when the speculative 404 lands before the probe rules the routes absent", async () => {
    let resolveProbe!: (spec: { ok: boolean; json: () => Promise<unknown> }) => void
    fetchMock.mockImplementation(() => new Promise((resolve) => { resolveProbe = resolve }))
    mocks.getBillingOverview.mockRejectedValue({ status: 404 })
    mocks.getStorageQuotaSummary.mockResolvedValue({})

    render(<BillingDashboardPage />)

    // The 404 trips the page-level notFound guard while the probe is pending
    // (no tabs in that state)...
    await expectDesignSystemAlertForTitle("Not available on this server")
    expect(screen.queryAllByRole("tab")).toHaveLength(0)

    // ...then the probe definitively rules billing absent: the guard must
    // give way to the in-place downgrade (tabs + inline notice).
    await act(async () => {
      resolveProbe({ ok: true, json: async () => ({ paths: {} }) })
    })

    expect(await screen.findAllByRole("tab")).toHaveLength(3)
    await expectDesignSystemAlertForTitle("Not available on this server")
  })

  it("renders the tabs before the capability probe resolves", async () => {
    let resolveProbe!: (spec: { ok: boolean; json: () => Promise<unknown> }) => void
    fetchMock.mockImplementation(() => new Promise((resolve) => { resolveProbe = resolve }))
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

    render(<BillingDashboardPage />)

    // Tabs (and their data surface) render immediately - the openapi.json
    // probe must never hold the first render in a full-page skeleton.
    // Skeletons inside the tab content (Statistic loading) are fine; a
    // page-level skeleton outside .ant-tabs is the old render gate.
    expect(screen.getAllByRole("tab")).toHaveLength(3)
    expect(screen.getByRole("tab", { name: "Overview" })).toBeInTheDocument()
    const skeletonsOutsideTabs = Array.from(
      document.querySelectorAll(".ant-skeleton")
    ).filter((el) => !el.closest(".ant-tabs"))
    expect(skeletonsOutsideTabs).toHaveLength(0)

    resolveProbe({
      ok: true,
      json: async () => ({
        paths: {
          "/api/v1/admin/billing/overview": {}
        }
      })
    })
    expect(await screen.findByText("Quota Records")).toBeInTheDocument()
  })

  it("reads the {items, total} envelope for subscriptions and events (legacy fallbacks kept)", async () => {
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
    mocks.getStorageQuotaSummary.mockResolvedValue({ total_quotas: 0, items: [] })
    mocks.listAllSubscriptions.mockResolvedValue({
      items: [{ user_id: 7, username: "ada", plan_id: "pro", status: "active", created_at: "2026-01-01T00:00:00Z" }],
      total: 1
    })
    mocks.listBillingEvents.mockResolvedValue({
      items: [{ id: 42, event_type: "subscription.created", user_id: 7, amount: 5, description: "pro plan", created_at: "2026-01-01T00:00:00Z" }],
      total: 1
    })

    render(<BillingDashboardPage />)

    // Overview tab is default-active; switch to Subscriptions.
    await screen.findByText("Quota Records")
    const user = userEvent.setup()
    await user.click(screen.getByRole("tab", { name: "Subscriptions" }))
    expect(await screen.findByText("ada")).toBeInTheDocument()

    await user.click(screen.getByRole("tab", { name: "Billing Events" }))
    expect(await screen.findByText("subscription.created")).toBeInTheDocument()
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

  it("flags truncated storage summaries so paginated servers aren't understated silently", () => {
    const truncated = aggregateStorageSummary({
      total_quotas: 200,
      items: [{ quota_mb: 100, used_mb: 50 }],
      pagination: { has_more: true }
    })
    expect(truncated.hasMore).toBe(true)
    expect(truncated.utilizationPct).toBe(50)

    const complete = aggregateStorageSummary({ total_quotas: 1, items: [], has_more: false })
    expect(complete.hasMore).toBe(false)
    expect(complete.utilizationPct).toBe(0)
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
