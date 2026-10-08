// @vitest-environment jsdom

import React from "react"
import { act, cleanup, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const apiMock = vi.hoisted(() => ({
  getSystemStats: vi.fn(),
  getSecurityAlertStatus: vi.fn(),
  listBackups: vi.fn(),
  getLlamacppStatus: vi.fn(),
  getMlxStatus: vi.fn(),
  getGovernorCoverage: vi.fn()
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: apiMock
}))

import {
  adminKeys,
  useSystemStats
} from "@/services/tldw/adminQueries"
import {
  AdminQueryProvider,
  getAdminQueryClient
} from "@/components/Option/Admin/AdminQueryProvider"
import {
  loadAdminModuleSignals,
  type AdminModuleSignal
} from "@/components/Option/Admin/admin-module-signals"

const StatsProbe: React.FC<{ label: string }> = ({ label }) => {
  const stats = useSystemStats()
  return (
    <p data-testid={`probe-${label}`}>
      {stats.data
        ? `users:${(stats.data as { users?: { total?: number } }).users?.total}`
        : stats.isFetching
          ? "loading"
          : "empty"}
    </p>
  )
}

const SignalsProbe: React.FC = () => {
  const [signals, setSignals] = React.useState<Record<
    string,
    AdminModuleSignal
  > | null>(null)
  React.useEffect(() => {
    let cancelled = false
    void loadAdminModuleSignals().then((loaded) => {
      if (!cancelled) setSignals(loaded)
    })
    return () => {
      cancelled = true
    }
  }, [])
  if (!signals) {
    return <p data-testid="signals-loading">loading</p>
  }
  return (
    <ul data-testid="signals-loaded">
      {Object.entries(signals).map(([route, signal]) => (
        <li key={route}>{`${route}:${signal.state}:${signal.detail}`}</li>
      ))}
    </ul>
  )
}

const resolveAllSignalMocks = () => {
  apiMock.getSystemStats.mockResolvedValue({ users: { total: 3 } })
  apiMock.getSecurityAlertStatus.mockResolvedValue({ health: "ok" })
  apiMock.listBackups.mockResolvedValue({ backups: [{ id: 1 }] })
  apiMock.getLlamacppStatus.mockResolvedValue({ state: "running" })
  apiMock.getMlxStatus.mockResolvedValue({ active: true })
  apiMock.getGovernorCoverage.mockResolvedValue({ coverage_pct: 90 })
}

describe("adminQueries shared reference-data cache (admin perf B-S4)", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    resolveAllSignalMocks()
    // The admin surfaces share one QueryClient; drop any cache left over from
    // the previous test so every test starts with a cold window.
    getAdminQueryClient().clear()
  })

  afterEach(() => {
    cleanup()
    vi.useRealTimers()
  })

  it("test_two_mounts_one_fetch — two consumers under one provider share a single fetch", async () => {
    render(
      <AdminQueryProvider>
        <StatsProbe label="one" />
        <StatsProbe label="two" />
      </AdminQueryProvider>
    )

    await waitFor(() => {
      expect(screen.getByTestId("probe-one").textContent).toBe("users:3")
      expect(screen.getByTestId("probe-two").textContent).toBe("users:3")
    })

    expect(apiMock.getSystemStats).toHaveBeenCalledTimes(1)
  })

  it("test_stale_window_refetches — a remount past the 30s stale window refetches", async () => {
    vi.useFakeTimers()

    const first = render(
      <AdminQueryProvider>
        <StatsProbe label="first" />
      </AdminQueryProvider>
    )
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })
    expect(screen.getByTestId("probe-first").textContent).toBe("users:3")
    expect(apiMock.getSystemStats).toHaveBeenCalledTimes(1)

    first.unmount()
    // Advance past the shared 30s stale window (staleTime is set on the
    // admin QueryClient, not per hook call).
    await act(async () => {
      await vi.advanceTimersByTimeAsync(30_001)
    })

    render(
      <AdminQueryProvider>
        <StatsProbe label="second" />
      </AdminQueryProvider>
    )
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })

    expect(screen.getByTestId("probe-second").textContent).toBe("users:3")
    expect(apiMock.getSystemStats).toHaveBeenCalledTimes(2)
  })

  it("test_module_signals_use_query_cache — remounting the signals loader probes each endpoint once per stale window", async () => {
    const first = render(
      <AdminQueryProvider>
        <SignalsProbe />
      </AdminQueryProvider>
    )
    expect(await screen.findByTestId("signals-loaded")).toBeTruthy()
    expect(apiMock.getSystemStats).toHaveBeenCalledTimes(1)
    expect(apiMock.listBackups).toHaveBeenCalledTimes(1)

    first.unmount()
    render(
      <AdminQueryProvider>
        <SignalsProbe />
      </AdminQueryProvider>
    )
    expect(await screen.findByTestId("signals-loaded")).toBeTruthy()

    // The second mount is served from the shared query cache within the
    // stale window: every probe endpoint still fired exactly once.
    expect(apiMock.getSystemStats).toHaveBeenCalledTimes(1)
    expect(apiMock.listBackups).toHaveBeenCalledTimes(1)
    expect(apiMock.getSecurityAlertStatus).toHaveBeenCalledTimes(1)
    expect(apiMock.getLlamacppStatus).toHaveBeenCalledTimes(1)
    expect(apiMock.getMlxStatus).toHaveBeenCalledTimes(1)
    expect(apiMock.getGovernorCoverage).toHaveBeenCalledTimes(1)
  })

  it("adminKeys keep the documented shape", () => {
    expect(adminKeys.systemStats).toEqual(["admin", "system-stats"])
    expect(adminKeys.usersPage(2, 20)).toEqual(["admin", "users", 2, 20, ""])
    expect(adminKeys.usersPage(2, 20, "ada")).toEqual([
      "admin",
      "users",
      2,
      20,
      "ada"
    ])
    expect(adminKeys.roles).toEqual(["admin", "roles"])
    expect(adminKeys.permissions).toEqual(["admin", "permissions"])
  })
})
