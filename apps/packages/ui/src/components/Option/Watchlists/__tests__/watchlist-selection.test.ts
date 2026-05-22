import { describe, expect, it } from "vitest"
import type { WatchlistContainer } from "@/types/watchlists"
import { resolvePreferredWatchlistId } from "../watchlist-selection"

const buildWatchlist = (overrides: Partial<WatchlistContainer>): WatchlistContainer => ({
  id: 1,
  name: "Watchlist",
  description: null,
  objective: null,
  domain: "general",
  status: "paused",
  priority: "medium",
  tags: [],
  archived_at: null,
  deleted_at: null,
  created_at: "2026-05-01T00:00:00Z",
  updated_at: "2026-05-01T00:00:00Z",
  ...overrides
})

describe("resolvePreferredWatchlistId", () => {
  it("prefers an active non-imported watchlist over an inactive imported placeholder", () => {
    const importedInactive = buildWatchlist({
      id: 1,
      name: "Imported Watchlist",
      status: "paused",
      domain: "general"
    })
    const activeNewsWatchlist = buildWatchlist({
      id: 2,
      name: "Morning News",
      status: "active",
      domain: "news"
    })

    expect(resolvePreferredWatchlistId([importedInactive, activeNewsWatchlist], null)).toBe(2)
  })

  it("keeps the current selected id when it still exists", () => {
    const existing = buildWatchlist({ id: 7, name: "Existing", status: "paused" })
    const active = buildWatchlist({ id: 8, name: "Active", status: "active", domain: "news" })

    expect(resolvePreferredWatchlistId([active, existing], existing.id)).toBe(existing.id)
  })

  it("returns null when no watchlists exist", () => {
    expect(resolvePreferredWatchlistId([], null)).toBeNull()
  })

  it("breaks ties by most recently updated watchlist", () => {
    const older = buildWatchlist({
      id: 4,
      name: "Older News",
      status: "active",
      domain: "news",
      updated_at: "2026-05-01T00:00:00Z",
      created_at: "2026-05-01T00:00:00Z"
    })
    const newer = buildWatchlist({
      id: 5,
      name: "Newer News",
      status: "active",
      domain: "news",
      updated_at: "2026-05-02T00:00:00Z",
      created_at: "2026-05-01T00:00:00Z"
    })

    expect(resolvePreferredWatchlistId([older, newer], null)).toBe(newer.id)
  })
})
