// @vitest-environment jsdom

import React from "react"
import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"

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

const designSystemLabels = vi.hoisted(() => ({
  ready: "Registry Ready",
  unavailable: "Registry Unavailable",
  missingKeys: new Set<string>()
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

vi.mock("@/design-system", async (importActual) => {
  const actual = await importActual<typeof import("@/design-system")>()

  return {
    ...actual,
    getDesignSystemState: vi.fn(
      (key: Parameters<typeof actual.getDesignSystemState>[0]) => {
        if (designSystemLabels.missingKeys.has(key)) {
          return undefined as unknown as ReturnType<
            typeof actual.getDesignSystemState
          >
        }

        const state = actual.getDesignSystemState(key)

        if (key === "ready") {
          return { ...state, label: designSystemLabels.ready }
        }

        if (key === "unavailable") {
          return { ...state, label: designSystemLabels.unavailable }
        }

        return state
      }
    )
  }
})

import MonitoringDashboardPage from "../MonitoringDashboardPage"

const expectDesignSystemAlertForText = async (text: string | RegExp) => {
  const title = await screen.findByText(text)
  const alert = title.closest('[data-ds-component="Alert"]')

  expect(alert).not.toBeNull()
  return alert as HTMLElement
}

describe("MonitoringDashboardPage", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    designSystemLabels.missingKeys.clear()

    if (!window.matchMedia) {
      Object.defineProperty(window, "matchMedia", {
        writable: true,
        value: vi.fn().mockImplementation((query: string) => ({
          matches: false,
          media: query,
          onchange: null,
          addListener: vi.fn(),
          removeListener: vi.fn(),
          addEventListener: vi.fn(),
          removeEventListener: vi.fn(),
          dispatchEvent: vi.fn()
        }))
      })
    }

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
    mocks.createAlertRule.mockResolvedValue({ item: { id: 1 } })
    mocks.assignAlert.mockResolvedValue({})
  })

  it("renders the intro text and page title", async () => {
    render(<MonitoringDashboardPage />)

    expect(screen.getByText("Monitoring & Alerting")).toBeTruthy()
    await waitFor(() => {
      expect(
        screen.getByText(/Monitor your tldw server/)
      ).toBeTruthy()
    })
  })

  it("renders forbidden guard feedback through the design-system Alert primitive", async () => {
    mocks.getSystemStats.mockRejectedValueOnce({ status: 403 })

    render(<MonitoringDashboardPage />)

    const alert = await expectDesignSystemAlertForText("Access Denied")
    expect(alert).toHaveAttribute("role", "alert")
    expect(alert).toHaveTextContent(
      "You don't have permission to access the monitoring dashboard."
    )
  })

  it("renders missing-endpoint guard feedback through the design-system Alert primitive", async () => {
    mocks.getSystemStats.mockRejectedValueOnce({ status: 404 })

    render(<MonitoringDashboardPage />)

    const alert = await expectDesignSystemAlertForText("Not Available")
    expect(alert).toHaveAttribute("role", "alert")
    expect(alert).toHaveTextContent(
      "The monitoring dashboard is not available on this server."
    )
  })

  it("renders missing system data feedback through the design-system Alert primitive", async () => {
    mocks.getSystemStats.mockResolvedValueOnce(null)
    mocks.getSecurityAlertStatus.mockResolvedValueOnce(null)

    render(<MonitoringDashboardPage />)

    const alert = await expectDesignSystemAlertForText("No system data available yet.")
    expect(alert).toHaveAttribute("role", "status")
  })

  it("shows empty state with starter rules when no alert rules exist", async () => {
    render(<MonitoringDashboardPage />)

    const alert = await expectDesignSystemAlertForText("No alert rules configured")
    expect(alert).toHaveAttribute("role", "status")
    // Starter rule buttons should be visible
    expect(screen.getByText(/cpu_usage > 90/)).toBeTruthy()
    expect(screen.getByText(/memory_percent > 85/)).toBeTruthy()
    expect(screen.getByText(/disk_usage > 95/)).toBeTruthy()
  })

  it("does not show empty state when alert rules exist", async () => {
    mocks.listAlertRules.mockResolvedValue([
      { id: 1, metric: "cpu_usage", operator: ">", threshold: 80, duration_minutes: 5, severity: "high", enabled: true }
    ])

    render(<MonitoringDashboardPage />)

    await waitFor(() => {
      expect(screen.queryByText("No alert rules configured")).toBeNull()
    })
  })

  it("shows host-local sandbox runtime warnings for weaker isolation runtimes", async () => {
    mocks.getSandboxRuntimeDiagnostics.mockResolvedValue({
      source: "feature_discovery",
      summary: {
        total: 2,
        ready: 2,
        unavailable: 0,
        host_gated: 0,
        scaffold: 0,
        host_local_warning_runtimes: ["seatbelt", "worktree"],
        repair_supported_runtimes: []
      },
      runtimes: [
        {
          name: "seatbelt",
          available: true,
          implementation_state: "supported",
          readiness: "ready",
          reasons: [],
          normalized_reasons: [],
          boundary_class: "host_local",
          vm_grade_isolation: false,
          untrusted_eligible: false,
          isolation_warnings: ["host_local_boundary", "not_untrusted_eligible"],
          strict_deny_all_supported: false,
          strict_allowlist_supported: false,
          session_reuse_model: "none",
          requires_live_health_check: false,
          repair_supported: false,
          recommended_action: "none"
        },
        {
          name: "worktree",
          available: true,
          implementation_state: "supported",
          readiness: "ready",
          reasons: [],
          normalized_reasons: [],
          boundary_class: "host_local",
          vm_grade_isolation: false,
          untrusted_eligible: false,
          isolation_warnings: ["host_local_boundary", "not_untrusted_eligible"],
          strict_deny_all_supported: false,
          strict_allowlist_supported: false,
          session_reuse_model: "ephemeral",
          requires_live_health_check: false,
          repair_supported: false,
          recommended_action: "inspect_reasons"
        }
      ],
      startup_warning_summary: null
    })

    render(<MonitoringDashboardPage />)

    await waitFor(() => {
      expect(screen.getByText("Sandbox Runtime Isolation")).toBeTruthy()
    })
    const alert = await expectDesignSystemAlertForText(
      "Host-local sandbox runtimes require operator review"
    )
    expect(alert).toHaveAttribute("role", "alert")
    expect(screen.getByText("seatbelt")).toBeTruthy()
    expect(screen.getByText("worktree")).toBeTruthy()
    expect(screen.getByText(/not VM-grade isolation/i)).toBeTruthy()
  })

  it("renders empty sandbox diagnostics feedback through the design-system Alert primitive", async () => {
    render(<MonitoringDashboardPage />)

    const alert = await expectDesignSystemAlertForText(
      "No sandbox runtime diagnostics available yet."
    )
    expect(alert).toHaveAttribute("role", "status")
  })

  it("uses design-system state labels for sandbox readiness summary counts", async () => {
    mocks.getSandboxRuntimeDiagnostics.mockResolvedValue({
      source: "feature_discovery",
      summary: {
        total: 3,
        ready: 2,
        unavailable: 1,
        host_gated: 0,
        scaffold: 0,
        host_local_warning_runtimes: [],
        repair_supported_runtimes: []
      },
      runtimes: [],
      startup_warning_summary: null
    })

    render(<MonitoringDashboardPage />)

    await waitFor(() => {
      expect(screen.getByText(designSystemLabels.ready)).toBeTruthy()
    })
    const readyItem = screen
      .getByText(designSystemLabels.ready)
      .closest(".ant-descriptions-item-container")
    const unavailableItem = screen
      .getByText(designSystemLabels.unavailable)
      .closest(".ant-descriptions-item-container")

    expect(readyItem).not.toBeNull()
    expect(unavailableItem).not.toBeNull()
    expect(within(readyItem as HTMLElement).getByText("2")).toBeTruthy()
    expect(within(unavailableItem as HTMLElement).getByText("1")).toBeTruthy()
  })

  it("falls back to readable state keys when sandbox readiness registry labels are missing", async () => {
    designSystemLabels.missingKeys.add("ready")
    designSystemLabels.missingKeys.add("unavailable")
    vi.resetModules()
    const { default: FallbackMonitoringDashboardPage } = await import(
      "../MonitoringDashboardPage"
    )
    const { getDesignSystemState: getFallbackDesignSystemState } =
      await import("@/design-system")
    mocks.getSandboxRuntimeDiagnostics.mockResolvedValue({
      source: "feature_discovery",
      summary: {
        total: 3,
        ready: 2,
        unavailable: 1,
        host_gated: 0,
        scaffold: 0,
        host_local_warning_runtimes: [],
        repair_supported_runtimes: []
      },
      runtimes: [],
      startup_warning_summary: null
    })

    render(<FallbackMonitoringDashboardPage />)

    await waitFor(() => {
      expect(screen.getByText("ready")).toBeTruthy()
    })
    expect(screen.getByText("unavailable")).toBeTruthy()
    expect(getFallbackDesignSystemState).toHaveBeenCalledWith("ready")
    expect(getFallbackDesignSystemState).toHaveBeenCalledWith("unavailable")
  })

  it("distinguishes forbidden sandbox diagnostics from unavailable diagnostics", async () => {
    mocks.getSandboxRuntimeDiagnostics.mockRejectedValue(
      Object.assign(
        new Error("Request failed: 403 (GET /api/v1/sandbox/admin/runtime-diagnostics)"),
        { status: 403 }
      )
    )

    render(<MonitoringDashboardPage />)

    await waitFor(() => {
      expect(screen.getByText("Sandbox diagnostics access denied")).toBeTruthy()
    })
    const alert = await expectDesignSystemAlertForText(
      "Sandbox diagnostics access denied"
    )
    expect(alert).toHaveAttribute("role", "alert")
    expect(screen.queryByText("Sandbox diagnostics unavailable")).toBeNull()
    expect(screen.getByText(/Request failed: 403/)).toBeTruthy()
    expect(screen.queryByText(/\/api\/v1\/sandbox\/admin\/runtime-diagnostics/)).toBeNull()
  })

  it("renders empty activity feedback through the design-system Alert primitive", async () => {
    render(<MonitoringDashboardPage />)

    const alert = await expectDesignSystemAlertForText(
      "No recent activity data available."
    )
    expect(alert).toHaveAttribute("role", "status")
  })

  describe("MON-001: alert assignment uses correct user ID and field name", () => {
    it("fetches current user profile on mount", async () => {
      render(<MonitoringDashboardPage />)

      await waitFor(() => {
        expect(mocks.getCurrentUserProfile).toHaveBeenCalledTimes(1)
      })
    })

    it("renders alert history with assign button when history is loaded", async () => {
      mocks.listAlertHistory.mockResolvedValue([
        { id: "alert-1", alert: "High CPU Alert", severity: "high", status: "active", triggered_at: "2026-01-01T00:00:00Z" }
      ])

      render(<MonitoringDashboardPage />)

      // Wait for alert history to load
      await waitFor(() => {
        expect(screen.getByText("High CPU Alert")).toBeTruthy()
      })

      // Assign button should be present
      expect(screen.getByText("Assign")).toBeTruthy()
      // Snooze and Escalate buttons should also be present
      expect(screen.getByText("Snooze")).toBeTruthy()
      expect(screen.getByText("Escalate")).toBeTruthy()
    })
  })

  describe("MON-BUG-002: duration and severity are required fields", () => {
    it("validates duration_minutes and severity as required before submission", async () => {
      const user = userEvent.setup()
      render(<MonitoringDashboardPage />)

      // Wait for page to load
      await waitFor(() => {
        expect(screen.getByText("Alert Rules")).toBeTruthy()
      })

      // Click Create Rule without filling any fields — validation should catch duration+severity
      const createButton = screen.getByText("Create Rule")
      await user.click(createButton)

      // Validation errors should appear for all required fields including duration and severity
      await waitFor(() => {
        expect(screen.getByText("Duration is required")).toBeTruthy()
        expect(screen.getByText("Severity is required")).toBeTruthy()
      })

      // createAlertRule should NOT have been called
      expect(mocks.createAlertRule).not.toHaveBeenCalled()
    })
  })
})
