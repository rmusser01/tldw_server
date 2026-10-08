// @vitest-environment jsdom

import React from "react"
import { act, fireEvent, render, screen } from "@testing-library/react"
import { QueryClientProvider } from "@tanstack/react-query"
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
    getSecurityAlertStatus: (...args: unknown[]) => mocks.getSecurityAlertStatus(...args),
    listAlertRules: (...args: unknown[]) => mocks.listAlertRules(...args),
    createAlertRule: (...args: unknown[]) => mocks.createAlertRule(...args),
    deleteAlertRule: (...args: unknown[]) => mocks.deleteAlertRule(...args),
    listAlertHistory: (...args: unknown[]) => mocks.listAlertHistory(...args),
    assignAlert: (...args: unknown[]) => mocks.assignAlert(...args),
    snoozeAlert: (...args: unknown[]) => mocks.snoozeAlert(...args),
    escalateAlert: (...args: unknown[]) => mocks.escalateAlert(...args),
    getDashboardActivity: (...args: unknown[]) => mocks.getDashboardActivity(...args),
    getSandboxRuntimeDiagnostics: (...args: unknown[]) => mocks.getSandboxRuntimeDiagnostics(...args),
    getCurrentUserProfile: (...args: unknown[]) => mocks.getCurrentUserProfile(...args)
  }
}))

import MonitoringDashboardPage from "../MonitoringDashboardPage"
import { createAdminQueryClient } from "../AdminQueryProvider"
import type { QueryClient } from "@tanstack/react-query"

/**
 * Render the page under a fresh admin query client so each test owns its
 * cache (B-S4): reference-data queries now flow through react-query.
 */
const adminQueryClients: QueryClient[] = []
const renderPage = (ui: React.ReactElement) => {
  const client = createAdminQueryClient()
  adminQueryClients.push(client)
  return render(
    <QueryClientProvider client={client}>{ui}</QueryClientProvider>
  )
}

type LoaderCounts = {
  stats: number
  security: number
  diagnostics: number
  rules: number
  history: number
  activity: number
}

const loaderCounts = (): LoaderCounts => ({
  stats: mocks.getSystemStats.mock.calls.length,
  security: mocks.getSecurityAlertStatus.mock.calls.length,
  diagnostics: mocks.getSandboxRuntimeDiagnostics.mock.calls.length,
  rules: mocks.listAlertRules.mock.calls.length,
  history: mocks.listAlertHistory.mock.calls.length,
  activity: mocks.getDashboardActivity.mock.calls.length
})

/** Set document visibility the jsdom way and notify listeners. */
const setDocumentHidden = (hidden: boolean) => {
  Object.defineProperty(document, "hidden", {
    configurable: true,
    value: hidden
  })
  Object.defineProperty(document, "visibilityState", {
    configurable: true,
    value: hidden ? "hidden" : "visible"
  })
  fireEvent(document, new Event("visibilitychange"))
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

describe("MonitoringDashboardPage polling discipline (admin perf B-S2)", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.useFakeTimers()

    // Default mocks: empty data
    mocks.getSystemStats.mockResolvedValue({ cpu_usage: 45, memory_percent: 62 })
    mocks.getSecurityAlertStatus.mockResolvedValue({})
    mocks.listAlertRules.mockResolvedValue([])
    mocks.listAlertHistory.mockResolvedValue([])
    mocks.getDashboardActivity.mockResolvedValue({ entries: [] })
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
    // Drop instance-level visibility overrides so later tests see jsdom defaults.
    delete (document as Partial<Document> & { hidden?: boolean }).hidden
    delete (document as Partial<Document> & {
      visibilityState?: string
    }).visibilityState
  })

  it("auto refresh polls only live datasets (stats + security status)", async () => {
    renderPage(<MonitoringDashboardPage />)
    await flushAsync()
    expect(mocks.getSystemStats).toHaveBeenCalledTimes(1)
    expect(mocks.listAlertHistory).toHaveBeenCalledTimes(1)

    await enableAutoRefresh("30s")
    const before = loaderCounts()

    act(() => {
      vi.advanceTimersByTime(30_000)
    })
    await flushAsync()

    expect(mocks.getSystemStats.mock.calls.length).toBe(before.stats + 1)
    expect(mocks.getSecurityAlertStatus.mock.calls.length).toBe(
      before.security + 1
    )
    expect(mocks.getSandboxRuntimeDiagnostics.mock.calls.length).toBe(
      before.diagnostics
    )
    expect(mocks.listAlertRules.mock.calls.length).toBe(before.rules)
    expect(mocks.listAlertHistory.mock.calls.length).toBe(before.history)
    expect(mocks.getDashboardActivity.mock.calls.length).toBe(before.activity)
  })

  it("does not poll while the tab is hidden and catches up live data when visible again", async () => {
    renderPage(<MonitoringDashboardPage />)
    await flushAsync()
    await enableAutoRefresh("30s")

    setDocumentHidden(true)
    const before = loaderCounts()

    act(() => {
      vi.advanceTimersByTime(90_000)
    })
    await flushAsync()
    expect(loaderCounts()).toEqual(before)

    setDocumentHidden(false)
    await flushAsync()
    expect(mocks.getSystemStats.mock.calls.length).toBe(before.stats + 1)
    expect(mocks.getSecurityAlertStatus.mock.calls.length).toBe(
      before.security + 1
    )
    expect(mocks.getSandboxRuntimeDiagnostics.mock.calls.length).toBe(
      before.diagnostics
    )
    expect(mocks.listAlertRules.mock.calls.length).toBe(before.rules)
    expect(mocks.listAlertHistory.mock.calls.length).toBe(before.history)
    expect(mocks.getDashboardActivity.mock.calls.length).toBe(before.activity)
  })

  it("requests alert history with an explicit limit of 200", async () => {
    renderPage(<MonitoringDashboardPage />)
    await flushAsync()

    expect(mocks.listAlertHistory).toHaveBeenCalledWith({ limit: 200 })
  })
})
