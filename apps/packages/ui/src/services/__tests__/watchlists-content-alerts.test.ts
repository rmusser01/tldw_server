import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn(),
  bgUpload: vi.fn(),
  getTldwTTSModel: vi.fn(),
  getTldwTTSVoice: vi.fn()
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.bgRequest(...args),
  bgUpload: (...args: unknown[]) => mocks.bgUpload(...args)
}))

vi.mock("@/services/tts", () => ({
  getTldwTTSModel: (...args: unknown[]) => mocks.getTldwTTSModel(...args),
  getTldwTTSVoice: (...args: unknown[]) => mocks.getTldwTTSVoice(...args)
}))

import {
  createWatchlistContentAlertRule,
  deleteWatchlistContentAlertRule,
  fetchWatchlistContentAlertRules,
  fetchWatchlistContentAlerts,
  getWatchlistContentAlert,
  updateWatchlistContentAlert,
  updateWatchlistContentAlertRule
} from "../watchlists"

describe("watchlist content alert service contract", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.bgRequest.mockResolvedValue({ items: [], total: 0 })
  })

  it("uses selected-Watchlist nested content alert rule routes", async () => {
    await fetchWatchlistContentAlertRules(42, { enabled: true, page: 2, size: 25 })
    expect(mocks.bgRequest).toHaveBeenLastCalledWith({
      path: "/api/v1/watchlists/42/content-alert-rules?enabled=true&page=2&size=25",
      method: "GET"
    })

    await createWatchlistContentAlertRule(42, {
      name: "Active exploitation",
      rule_kind: "descriptor",
      match_mode: "contains",
      pattern: "active exploitation",
      severity: "critical",
      source_constraints: { source_tags: ["advisory"] }
    })
    expect(mocks.bgRequest).toHaveBeenLastCalledWith({
      path: "/api/v1/watchlists/42/content-alert-rules",
      method: "POST",
      body: {
        name: "Active exploitation",
        rule_kind: "descriptor",
        match_mode: "contains",
        pattern: "active exploitation",
        severity: "critical",
        source_constraints: { source_tags: ["advisory"] }
      }
    })

    await updateWatchlistContentAlertRule(42, 7, { enabled: false })
    expect(mocks.bgRequest).toHaveBeenLastCalledWith({
      path: "/api/v1/watchlists/42/content-alert-rules/7",
      method: "PATCH",
      body: { enabled: false }
    })

    await deleteWatchlistContentAlertRule(42, 7)
    expect(mocks.bgRequest).toHaveBeenLastCalledWith({
      path: "/api/v1/watchlists/42/content-alert-rules/7",
      method: "DELETE"
    })
  })

  it("uses selected-Watchlist nested content alert inbox routes", async () => {
    await fetchWatchlistContentAlerts(42, {
      status: "unread",
      severity: "critical",
      rule_id: 7,
      source_id: 11,
      page: 1,
      size: 50
    })
    expect(mocks.bgRequest).toHaveBeenLastCalledWith({
      path: "/api/v1/watchlists/42/alerts?status=unread&severity=critical&rule_id=7&source_id=11&page=1&size=50",
      method: "GET"
    })

    await getWatchlistContentAlert(42, 99)
    expect(mocks.bgRequest).toHaveBeenLastCalledWith({
      path: "/api/v1/watchlists/42/alerts/99",
      method: "GET"
    })

    await updateWatchlistContentAlert(42, 99, { status: "dismissed" })
    expect(mocks.bgRequest).toHaveBeenLastCalledWith({
      path: "/api/v1/watchlists/42/alerts/99",
      method: "PATCH",
      body: { status: "dismissed" }
    })
  })
})
