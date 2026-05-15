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
  batchUpdateScrapedItems,
  createWatchlistItemView,
  deleteWatchlistItemView,
  fetchScrapedItems,
  fetchScrapedItemSmartCounts,
  fetchWatchlistItemViews,
  updateWatchlistItemView
} from "../watchlists"

describe("watchlist item triage service contract", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.bgRequest.mockResolvedValue({})
  })

  it("serializes Stage 4 item list and smart-count filters", async () => {
    await fetchScrapedItems({
      watchlist_id: 42,
      source_id: 7,
      sort: "alert_severity_desc",
      has_alert: true,
      alert_status: "unread",
      alert_severity: "critical",
      alert_rule_id: 99,
      include_alert_summary: true,
      page: 2,
      size: 50
    })

    expect(mocks.bgRequest).toHaveBeenLastCalledWith({
      path: "/api/v1/watchlists/items?watchlist_id=42&source_id=7&sort=alert_severity_desc&has_alert=true&alert_status=unread&alert_severity=critical&alert_rule_id=99&include_alert_summary=true&page=2&size=50",
      method: "GET"
    })

    await fetchScrapedItemSmartCounts({
      watchlist_id: 42,
      has_alert: true,
      alert_status: "unread",
      alert_severity: "critical",
      alert_rule_id: 99
    })

    expect(mocks.bgRequest).toHaveBeenLastCalledWith({
      path: "/api/v1/watchlists/items/smart-counts?watchlist_id=42&has_alert=true&alert_status=unread&alert_severity=critical&alert_rule_id=99",
      method: "GET"
    })
  })

  it("posts batch triage requests to the static item batch route", async () => {
    await batchUpdateScrapedItems({
      watchlist_id: 42,
      scope: {
        reviewed: false,
        status: "ingested",
        has_alert: true,
        alert_severity: "critical"
      },
      reviewed: true,
      queued_for_briefing: true,
      limit: 500
    })

    expect(mocks.bgRequest).toHaveBeenLastCalledWith({
      path: "/api/v1/watchlists/items/batch-update",
      method: "POST",
      body: {
        watchlist_id: 42,
        scope: {
          reviewed: false,
          status: "ingested",
          has_alert: true,
          alert_severity: "critical"
        },
        reviewed: true,
        queued_for_briefing: true,
        limit: 500
      }
    })
  })

  it("uses selected-Watchlist saved view CRUD routes", async () => {
    await fetchWatchlistItemViews(42)
    expect(mocks.bgRequest).toHaveBeenLastCalledWith({
      path: "/api/v1/watchlists/42/item-views",
      method: "GET"
    })

    await createWatchlistItemView(42, {
      name: "Critical unread",
      filters: { source_id: 7, smart_filter: "unread", alert_severity: "critical" },
      sort: "unread_first",
      is_default: true
    })
    expect(mocks.bgRequest).toHaveBeenLastCalledWith({
      path: "/api/v1/watchlists/42/item-views",
      method: "POST",
      body: {
        name: "Critical unread",
        filters: { source_id: 7, smart_filter: "unread", alert_severity: "critical" },
        sort: "unread_first",
        is_default: true
      }
    })

    await updateWatchlistItemView(42, 5, { name: "Critical queue" })
    expect(mocks.bgRequest).toHaveBeenLastCalledWith({
      path: "/api/v1/watchlists/42/item-views/5",
      method: "PATCH",
      body: { name: "Critical queue" }
    })

    await deleteWatchlistItemView(42, 5)
    expect(mocks.bgRequest).toHaveBeenLastCalledWith({
      path: "/api/v1/watchlists/42/item-views/5",
      method: "DELETE"
    })
  })
})
