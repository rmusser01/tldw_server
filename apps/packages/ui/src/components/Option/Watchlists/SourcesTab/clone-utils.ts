import type { WatchlistSource, WatchlistSourceCreate } from "@/types/watchlists"

const cloneValue = <T>(value: T): T => {
  if (value == null) return value
  if (typeof structuredClone === "function") {
    return structuredClone(value)
  }
  return JSON.parse(JSON.stringify(value)) as T
}

export const buildClonedWatchlistSourcePayload = (
  source: WatchlistSource,
  watchlistId?: number | null
): WatchlistSourceCreate => ({
  name: `${source.name} copy`,
  url: source.url,
  source_type: source.source_type,
  active: false,
  tags: cloneValue(source.tags || []),
  settings: source.settings ? cloneValue(source.settings) : source.settings ?? null,
  group_ids: cloneValue(source.group_ids || []),
  watchlist_id: watchlistId ?? source.watchlist_ids?.[0]
})
