// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from "vitest"

const apiMock = vi.hoisted(() => ({
  listBackupSchedules: vi.fn(),
  listAlertRules: vi.fn(),
  getGovernorCoverage: vi.fn()
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: apiMock
}))

import { loadAdminFirstSteps } from "../admin-first-steps"

describe("loadAdminFirstSteps", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    apiMock.listBackupSchedules.mockResolvedValue({ schedules: [] })
    apiMock.listAlertRules.mockResolvedValue([])
    apiMock.getGovernorCoverage.mockResolvedValue({ coverage_pct: 78.9 })
  })

  it("marks every step undone on a fresh server", async () => {
    const steps = await loadAdminFirstSteps()

    expect(steps).toEqual([
      {
        key: "backup-schedule",
        label: "Create a backup schedule",
        done: false,
        route: "/admin/data-ops"
      },
      {
        key: "alert-rule",
        label: "Add an alert rule",
        done: false,
        route: "/admin/monitoring"
      },
      {
        key: "coverage-review",
        label: "Review unprotected endpoints",
        done: false,
        route: "/admin/rate-limiting"
      }
    ])
  })

  it("marks steps done once schedules, rules, and coverage exist", async () => {
    apiMock.listBackupSchedules.mockResolvedValue({
      schedules: [{ id: 1 }]
    })
    apiMock.listAlertRules.mockResolvedValue([{ id: 1 }])
    apiMock.getGovernorCoverage.mockResolvedValue({ coverage_pct: 92.5 })

    const steps = await loadAdminFirstSteps()

    expect(steps.map((step) => step.done)).toEqual([true, true, true])
  })

  it("accepts bare-array and items-envelope responses", async () => {
    apiMock.listBackupSchedules.mockResolvedValue([{ id: 1 }])
    apiMock.listAlertRules.mockResolvedValue({ items: [{ id: 1 }] })

    const steps = await loadAdminFirstSteps()

    expect(steps.find((s) => s.key === "backup-schedule")?.done).toBe(true)
    expect(steps.find((s) => s.key === "alert-rule")?.done).toBe(true)
  })

  it("treats missing coverage data as done rather than nagging", async () => {
    apiMock.getGovernorCoverage.mockResolvedValue({})

    const steps = await loadAdminFirstSteps()

    expect(steps.find((s) => s.key === "coverage-review")?.done).toBe(true)
  })

  it("drops a failed probe instead of failing or misreporting the rest", async () => {
    apiMock.listAlertRules.mockRejectedValue(new Error("Request failed: 503"))

    const steps = await loadAdminFirstSteps()

    expect(steps.map((step) => step.key)).toEqual([
      "backup-schedule",
      "coverage-review"
    ])
  })
})
