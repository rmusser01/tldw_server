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

vi.mock("zustand/traditional", () => ({
  createWithEqualityFn:
    () =>
      (initializer: (set: (value: unknown) => void) => Record<string, unknown>) => {
        let state: Record<string, unknown> = {}
        const set = (value: unknown) => {
          const next = typeof value === "function"
            ? (value as (current: Record<string, unknown>) => Record<string, unknown>)(state)
            : value
          state = { ...state, ...(next as Record<string, unknown>) }
        }
        state = initializer(set)
        const store = ((selector?: (current: Record<string, unknown>) => unknown) =>
          selector ? selector(state) : state) as {
            (selector?: (current: Record<string, unknown>) => unknown): unknown
            getState: () => Record<string, unknown>
            setState: (value: unknown, replace?: boolean) => void
          }
        store.getState = () => state
        store.setState = (value: unknown, replace?: boolean) => {
          const next = typeof value === "function"
            ? (value as (current: Record<string, unknown>) => Record<string, unknown>)(state)
            : value
          state = replace
            ? { ...(next as Record<string, unknown>) }
            : { ...state, ...(next as Record<string, unknown>) }
        }
        return store
      }
}))

import {
  createWatchlist,
  deleteWatchlist,
  fetchScrapedItems,
  fetchScrapedItemSmartCounts,
  fetchWatchlistJobs,
  fetchWatchlistOutputs,
  fetchWatchlistRuns,
  fetchWatchlists,
  fetchWatchlistSources,
  getWatchlist,
  restoreWatchlist,
  updateWatchlist
} from "../watchlists"
import { useWatchlistsStore } from "../../store/watchlists"
import type {
  WatchlistContainer,
  WatchlistJobCreate,
  WatchlistSourceCreate
} from "@/types/watchlists"

const container: WatchlistContainer = {
  id: 42,
  name: "Healthcare ransomware",
  description: "Track hospital impact",
  objective: "Find new ransomware reports affecting hospitals",
  domain: "cti_osint",
  status: "active",
  priority: "high",
  tags: ["ransomware"],
  created_at: "2026-05-15T00:00:00Z",
  updated_at: "2026-05-15T00:00:00Z"
}

describe("first-class watchlists client contract", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.bgRequest.mockResolvedValue(container)
    useWatchlistsStore.getState().resetStore()
  })

  it("uses root Watchlist CRUD API paths", async () => {
    mocks.bgRequest.mockResolvedValueOnce({ items: [container], total: 1 })
    await fetchWatchlists({ page: 2, size: 25 })
    expect(mocks.bgRequest).toHaveBeenLastCalledWith({
      path: "/api/v1/watchlists?page=2&size=25",
      method: "GET"
    })

    await createWatchlist({
      name: "Healthcare ransomware",
      domain: "cti_osint",
      objective: "Track hospitals",
      priority: "high"
    })
    expect(mocks.bgRequest).toHaveBeenLastCalledWith({
      path: "/api/v1/watchlists",
      method: "POST",
      body: {
        name: "Healthcare ransomware",
        domain: "cti_osint",
        objective: "Track hospitals",
        priority: "high"
      }
    })

    await getWatchlist(42)
    expect(mocks.bgRequest).toHaveBeenLastCalledWith({
      path: "/api/v1/watchlists/42",
      method: "GET"
    })

    await updateWatchlist(42, { status: "paused", tags: ["cti"] })
    expect(mocks.bgRequest).toHaveBeenLastCalledWith({
      path: "/api/v1/watchlists/42",
      method: "PATCH",
      body: { status: "paused", tags: ["cti"] }
    })

    await deleteWatchlist(42)
    expect(mocks.bgRequest).toHaveBeenLastCalledWith({
      path: "/api/v1/watchlists/42",
      method: "DELETE"
    })

    await restoreWatchlist(42)
    expect(mocks.bgRequest).toHaveBeenLastCalledWith({
      path: "/api/v1/watchlists/42/restore",
      method: "POST"
    })
  })

  it("threads watchlist_id through existing child fetchers and create payloads", async () => {
    const sourcePayload: WatchlistSourceCreate = {
      name: "CTI feed",
      url: "https://example.com/rss.xml",
      source_type: "rss",
      watchlist_id: 42
    }
    expect(sourcePayload.watchlist_id).toBe(42)

    const jobPayload: WatchlistJobCreate = {
      name: "CTI monitor",
      scope: { sources: [1] },
      watchlist_id: 42
    }
    expect(jobPayload.watchlist_id).toBe(42)

    mocks.bgRequest.mockResolvedValue({ items: [], total: 0 })
    await fetchWatchlistSources({ watchlist_id: 42, page: 1, size: 20 })
    await fetchWatchlistJobs({ watchlist_id: 42, page: 1, size: 20 })
    await fetchWatchlistRuns({ watchlist_id: 42, page: 1, size: 20 })
    await fetchScrapedItems({ watchlist_id: 42, page: 1, size: 20 })
    await fetchScrapedItemSmartCounts({ watchlist_id: 42 })
    await fetchWatchlistOutputs({ watchlist_id: 42, page: 1, size: 20 })

    const paths = mocks.bgRequest.mock.calls.map((call) => String(call[0]?.path))
    expect(paths).toEqual(
      expect.arrayContaining([
        expect.stringMatching(/^\/api\/v1\/watchlists\/sources\?.*watchlist_id=42/),
        expect.stringMatching(/^\/api\/v1\/watchlists\/jobs\?.*watchlist_id=42/),
        expect.stringMatching(/^\/api\/v1\/watchlists\/runs\?.*watchlist_id=42/),
        expect.stringMatching(/^\/api\/v1\/watchlists\/items\?.*watchlist_id=42/),
        expect.stringMatching(/^\/api\/v1\/watchlists\/items\/smart-counts\?.*watchlist_id=42/),
        expect.stringMatching(/^\/api\/v1\/watchlists\/outputs\?.*watchlist_id=42/)
      ])
    )
  })

  it("stores and updates selected Watchlist container state", () => {
    const store = useWatchlistsStore.getState()
    expect(store.watchlists).toEqual([])
    expect(store.selectedWatchlistId).toBeNull()

    store.setWatchlists([container])
    store.setSelectedWatchlistId(42)
    expect(useWatchlistsStore.getState().watchlists).toEqual([container])
    expect(useWatchlistsStore.getState().selectedWatchlistId).toBe(42)

    store.updateWatchlistInList(42, { status: "paused", priority: "critical" })
    expect(useWatchlistsStore.getState().watchlists[0]).toMatchObject({
      id: 42,
      status: "paused",
      priority: "critical"
    })

    store.removeWatchlist(42)
    expect(useWatchlistsStore.getState().watchlists).toEqual([])
    expect(useWatchlistsStore.getState().selectedWatchlistId).toBeNull()
  })
})
