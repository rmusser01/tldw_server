// @vitest-environment jsdom

import React from "react"
import { act, fireEvent, render, screen, within } from "@testing-library/react"
import { QueryClientProvider, type QueryClient } from "@tanstack/react-query"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  getSystemStats: vi.fn(),
  getSecurityAlertStatus: vi.fn(),
  listAlertRules: vi.fn(),
  createAlertRule: vi.fn(),
  deleteAlertRule: vi.fn(),
  listAlertHistory: vi.fn(),
  assignAlert: vi.fn(),
  snoozeAlert: vi.fn(),
  escalateAlert: vi.fn(),
  getDashboardActivity: vi.fn(),
  getSandboxRuntimeDiagnostics: vi.fn(),
  getCurrentUserProfile: vi.fn()
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getSystemStats: (...args: unknown[]) => mocks.getSystemStats(...args),
    getSecurityAlertStatus: (...args: unknown[]) =>
      mocks.getSecurityAlertStatus(...args),
    listAlertRules: (...args: unknown[]) => mocks.listAlertRules(...args),
    createAlertRule: (...args: unknown[]) => mocks.createAlertRule(...args),
    deleteAlertRule: (...args: unknown[]) => mocks.deleteAlertRule(...args),
    listAlertHistory: (...args: unknown[]) => mocks.listAlertHistory(...args),
    assignAlert: (...args: unknown[]) => mocks.assignAlert(...args),
    snoozeAlert: (...args: unknown[]) => mocks.snoozeAlert(...args),
    escalateAlert: (...args: unknown[]) => mocks.escalateAlert(...args),
    getDashboardActivity: (...args: unknown[]) =>
      mocks.getDashboardActivity(...args),
    getSandboxRuntimeDiagnostics: (...args: unknown[]) =>
      mocks.getSandboxRuntimeDiagnostics(...args),
    getCurrentUserProfile: (...args: unknown[]) =>
      mocks.getCurrentUserProfile(...args)
  }
}))

/**
 * Row-render counters attributed by stable row-key prefixes: "alert-N" rows
 * belong to the Alert History table, "act-N"/"activity-N" rows to the
 * Activity table. Counting rows (not the whole Table) matters because
 * rc-table memoizes its Body subtree: a counted row re-render means the
 * table body genuinely reconciled new data or columns, not merely that the
 * page above it re-rendered.
 */
const rowRenderCounts = vi.hoisted(() => ({ history: 0, activity: 0 }))

vi.mock("antd", async (importOriginal) => {
  const actual = await importOriginal<typeof import("antd")>()

  const CountingRow = (
    props: React.HTMLAttributes<HTMLTableRowElement> & {
      "data-row-key"?: string | number
    }
  ) => {
    const key = String(props["data-row-key"] ?? "")
    if (key.startsWith("alert-")) rowRenderCounts.history += 1
    else if (key.startsWith("act-")) rowRenderCounts.activity += 1
    return <tr {...props} />
  }

  /**
   * Wraps antd Table solely to inject the counting row through `components`.
   * The injected object must keep a stable reference across renders — an
   * unstable `components` prop would itself force rc-table Body re-renders
   * and defeat the measurement.
   */
  const CountingTable = (props: React.ComponentProps<typeof actual.Table>) => {
    const components = React.useMemo(
      () =>
        ({ body: { row: CountingRow } }) as React.ComponentProps<
          typeof actual.Table
        >["components"],
      []
    )
    return <actual.Table {...props} components={components} />
  }

  return { ...actual, Table: CountingTable }
})

import MonitoringDashboardPage from "../MonitoringDashboardPage"
import { createAdminQueryClient } from "../AdminQueryProvider"

/** Render the page under a fresh admin query client (B-S4 pattern). */
const adminQueryClients: QueryClient[] = []
const renderPage = (ui: React.ReactElement) => {
  const client = createAdminQueryClient()
  adminQueryClients.push(client)
  return render(
    <QueryClientProvider client={client}>{ui}</QueryClientProvider>
  )
}

/** Flush pending microtasks (mocked loaders resolve immediately). */
const flushAsync = async () => {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(0)
  })
}

/** Switch the auto-refresh Select in the System Overview header. */
const enableAutoRefresh = async (optionLabel: string) => {
  const selectRoot = screen.getByText("Off").closest(".ant-select")
  expect(selectRoot).not.toBeNull()
  fireEvent.mouseDown(
    selectRoot?.querySelector(".ant-select-selector") ?? selectRoot
  )
  fireEvent.click(screen.getByText(optionLabel))
  await flushAsync()
}

/** The System Overview card's header Refresh button (runs refreshAll). */
const systemOverviewRefreshButton = (): HTMLElement => {
  const card = screen.getByText("System Overview").closest(".ant-card")
  expect(card).not.toBeNull()
  return within(card as HTMLElement).getByText("Refresh")
}

const historyRow = (id: string) => ({
  id,
  alert: `Alert ${id}`,
  severity: "high",
  status: "active",
  triggered_at: "2026-01-01T00:00:00Z"
})

const activityEntry = (id: string, seq: number) => ({
  id,
  action: `Event ${seq}`,
  user: "admin",
  details: `details ${seq}`,
  timestamp: `2026-10-01T10:00:${String(seq).padStart(2, "0")}Z`
})

describe("MonitoringDashboardPage render isolation (admin perf C-S1)", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.useFakeTimers()
    rowRenderCounts.history = 0
    rowRenderCounts.activity = 0

    // Default mocks: minimal data so the counted tables each have one row.
    mocks.getSystemStats.mockResolvedValue({ cpu_usage: 45, memory_percent: 62 })
    mocks.getSecurityAlertStatus.mockResolvedValue({})
    mocks.listAlertRules.mockResolvedValue([])
    mocks.listAlertHistory.mockResolvedValue([historyRow("alert-1")])
    mocks.getDashboardActivity.mockResolvedValue({
      entries: [activityEntry("act-1", 1)]
    })
    mocks.getSandboxRuntimeDiagnostics.mockResolvedValue({
      source: "feature_discovery",
      summary: {
        total: 0,
        ready: 0,
        unavailable: 0,
        host_gated: 0,
        scaffold: 0,
        host_local_warning_runtimes: [],
        repair_supported_runtimes: []
      },
      runtimes: [],
      startup_warning_summary: null
    })
    mocks.getCurrentUserProfile.mockResolvedValue({ id: 42, username: "admin" })
  })

  afterEach(() => {
    adminQueryClients.splice(0).forEach((client) => client.clear())
    vi.useRealTimers()
  })

  it("test_tick_does_not_rerender_tables", async () => {
    renderPage(<MonitoringDashboardPage />)
    await flushAsync()
    await flushAsync()

    // Baseline after everything settled: the counted history row rendered at
    // least once (the history table always keys rows by real ids).
    expect(rowRenderCounts.history).toBeGreaterThan(0)
    const historyBefore = rowRenderCounts.history
    const activityBefore = rowRenderCounts.activity

    // The label shows a fresh timestamp as "just now".
    expect(screen.getByText(/just now/)).toBeTruthy()

    act(() => {
      vi.advanceTimersByTime(10_000)
    })

    // The "time ago" text re-bucketed to "10s ago"...
    expect(screen.getByText(/10s ago/)).toBeTruthy()
    // ...without re-rendering any table rows.
    expect(rowRenderCounts.history).toBe(historyBefore)
    expect(rowRenderCounts.activity).toBe(activityBefore)
  })

  it("test_identical_poll_data_skips_setstate", async () => {
    renderPage(<MonitoringDashboardPage />)
    await flushAsync()
    await flushAsync()
    await enableAutoRefresh("30s")

    // Baseline after the Select interaction settled.
    const historyBefore = rowRenderCounts.history

    // Poll cycle resolves with byte-identical payloads.
    act(() => {
      vi.advanceTimersByTime(30_000)
    })
    await flushAsync()
    expect(mocks.getSecurityAlertStatus.mock.calls.length).toBeGreaterThanOrEqual(2)
    expect(rowRenderCounts.history).toBe(historyBefore)

    // Control: the counter is not vacuous — changed history data must
    // re-render history rows on the next full refresh.
    mocks.listAlertHistory.mockResolvedValue([
      historyRow("alert-1"),
      historyRow("alert-2")
    ])
    fireEvent.click(systemOverviewRefreshButton())
    await flushAsync()
    await flushAsync()
    expect(mocks.listAlertHistory.mock.calls.length).toBeGreaterThanOrEqual(2)
    expect(rowRenderCounts.history).toBeGreaterThan(historyBefore)
  })

  it("test_activity_table_paginated", async () => {
    mocks.getDashboardActivity.mockResolvedValue({
      entries: Array.from({ length: 45 }, (_, i) => activityEntry(`act-${i}`, i))
    })

    renderPage(<MonitoringDashboardPage />)
    await flushAsync()
    await flushAsync()

    const wrapper = screen
      .getByText("Event 7")
      .closest(".ant-table-wrapper") as HTMLElement
    expect(wrapper).not.toBeNull()

    // pageSize 20 — one page of rows rendered, not all 45.
    const renderedRows = wrapper.querySelectorAll(
      ".ant-table-tbody .ant-table-row"
    )
    expect(renderedRows.length).toBe(20)

    // Pagination control present and knows about the third page.
    const pagination = wrapper.querySelector(".ant-pagination")
    expect(pagination).not.toBeNull()
    expect(
      pagination?.querySelector('.ant-pagination-item[title="3"]')
    ).not.toBeNull()
  })

  it("activity rows fall back to fetch-time indexes when entries lack ids", async () => {
    mocks.getDashboardActivity.mockResolvedValue({
      entries: Array.from({ length: 3 }, (_, i) => ({
        action: `NoKey ${i}`,
        user: "admin"
      }))
    })

    renderPage(<MonitoringDashboardPage />)
    await flushAsync()
    await flushAsync()

    const wrapper = screen
      .getByText("NoKey 0")
      .closest(".ant-table-wrapper") as HTMLElement
    expect(wrapper).not.toBeNull()
    expect(wrapper.querySelector('[data-row-key="activity-0"]')).not.toBeNull()
    expect(wrapper.querySelector('[data-row-key="activity-1"]')).not.toBeNull()
    expect(wrapper.querySelector('[data-row-key="activity-2"]')).not.toBeNull()
  })
})
