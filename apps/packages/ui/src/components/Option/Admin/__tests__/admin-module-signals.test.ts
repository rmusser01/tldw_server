// @vitest-environment jsdom
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

import { loadAdminModuleSignals } from "../admin-module-signals"

const resolveAllHealthy = () => {
  apiMock.getSystemStats.mockResolvedValue({ users: { total: 3 } })
  apiMock.getSecurityAlertStatus.mockResolvedValue({ health: "ok" })
  apiMock.listBackups.mockResolvedValue({ backups: [{ id: 1 }] })
  apiMock.getLlamacppStatus.mockResolvedValue({ state: "running" })
  apiMock.getMlxStatus.mockResolvedValue({ active: true })
  apiMock.getGovernorCoverage.mockResolvedValue({ coverage_pct: 90 })
}

describe("loadAdminModuleSignals", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    resolveAllHealthy()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it("maps healthy responses onto per-route signals", async () => {
    const signals = await loadAdminModuleSignals()

    expect(signals["/admin/server"]).toEqual({
      state: "healthy",
      detail: "3 users"
    })
    expect(signals["/admin/monitoring"]).toEqual({
      state: "healthy",
      detail: "Alerting healthy"
    })
    expect(signals["/admin/data-ops"]).toEqual({
      state: "healthy",
      detail: "1 backup"
    })
    expect(signals["/admin/llamacpp"]).toEqual({
      state: "healthy",
      detail: "Runtime running"
    })
    expect(signals["/admin/mlx"]).toEqual({
      state: "healthy",
      detail: "Model loaded"
    })
    expect(signals["/admin/rate-limiting"]).toEqual({
      state: "healthy",
      detail: "90% endpoint coverage"
    })
  })

  it("flags degraded alerting, missing backups, and low coverage for attention", async () => {
    apiMock.getSecurityAlertStatus.mockResolvedValue({ health: "degraded" })
    apiMock.listBackups.mockResolvedValue({ backups: [] })
    apiMock.getLlamacppStatus.mockResolvedValue({ status: "stopped" })
    apiMock.getMlxStatus.mockResolvedValue({ active: false })
    apiMock.getGovernorCoverage.mockResolvedValue({ coverage_pct: 12 })

    const signals = await loadAdminModuleSignals()

    expect(signals["/admin/monitoring"]).toEqual({
      state: "attention",
      detail: "Alerting degraded"
    })
    expect(signals["/admin/data-ops"]).toEqual({
      state: "attention",
      detail: "No backups yet"
    })
    expect(signals["/admin/llamacpp"]).toEqual({
      state: "attention",
      detail: "Runtime stopped"
    })
    expect(signals["/admin/mlx"]).toEqual({
      state: "attention",
      detail: "No model loaded"
    })
    expect(signals["/admin/rate-limiting"]).toEqual({
      state: "attention",
      detail: "12% endpoint coverage"
    })
  })

  it("accepts alternative response envelopes without misreporting", async () => {
    // Backups may arrive as a bare array or an items envelope.
    apiMock.listBackups.mockResolvedValue([{ id: 1 }, { id: 2 }])
    // Stats without a numeric user count still count as reachable.
    apiMock.getSystemStats.mockResolvedValue({})
    // An unknown alerting health value must not claim degradation.
    apiMock.getSecurityAlertStatus.mockResolvedValue({})

    const signals = await loadAdminModuleSignals()

    expect(signals["/admin/data-ops"]).toEqual({
      state: "healthy",
      detail: "2 backups"
    })
    expect(signals["/admin/server"]).toEqual({
      state: "healthy",
      detail: "Server reachable"
    })
    expect(signals["/admin/monitoring"]).toEqual({
      state: "healthy",
      detail: "Monitoring reachable"
    })
  })

  it("degrades a rejected fetcher to unavailable and logs the cause", async () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {})
    apiMock.getLlamacppStatus.mockRejectedValue(new Error("Request failed: 503"))

    const signals = await loadAdminModuleSignals()

    expect(signals["/admin/llamacpp"]).toEqual({
      state: "unavailable",
      detail: "Status unavailable"
    })
    // Every other signal still resolves.
    expect(signals["/admin/server"].state).toBe("healthy")
    expect(warnSpy).toHaveBeenCalledWith(
      expect.stringContaining("/admin/llamacpp"),
      expect.any(Error)
    )
    warnSpy.mockRestore()
  })

  it("times out a hung fetcher instead of blocking the overview", async () => {
    vi.useFakeTimers()
    apiMock.getMlxStatus.mockImplementation(() => new Promise(() => {}))

    const pending = loadAdminModuleSignals()
    await vi.advanceTimersByTimeAsync(4100)
    const signals = await pending

    expect(signals["/admin/mlx"]).toEqual({
      state: "unavailable",
      detail: "Status unavailable"
    })
    expect(signals["/admin/server"].state).toBe("healthy")
  })
})
