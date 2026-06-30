import type { WatchlistContainer } from "@/types/watchlists"

const isImportedPlaceholder = (watchlist: WatchlistContainer): boolean =>
  watchlist.name.trim().toLowerCase() === "imported watchlist"

const timestampScore = (value: string | null | undefined): number => {
  if (!value) return 0
  const parsed = Date.parse(value)
  return Number.isFinite(parsed) ? parsed : 0
}

const watchlistScore = (watchlist: WatchlistContainer): number => {
  let score = 0
  if (watchlist.status === "active") score += 100
  if (!watchlist.deleted_at && !watchlist.archived_at) score += 40
  if (!isImportedPlaceholder(watchlist)) score += 20
  if (watchlist.domain === "news" || watchlist.domain === "cti_osint") score += 5
  return score
}

export const resolvePreferredWatchlistId = (
  items: WatchlistContainer[],
  selectedWatchlistId: number | null
): number | null => {
  if (selectedWatchlistId != null && items.some((watchlist) => watchlist.id === selectedWatchlistId)) {
    return selectedWatchlistId
  }

  if (items.length === 0) return null

  const [preferred] = items
    .map((watchlist, index) => ({
      watchlist,
      index,
      score: watchlistScore(watchlist),
      updatedAt: timestampScore(watchlist.updated_at),
      createdAt: timestampScore(watchlist.created_at)
    }))
    .sort((left, right) => {
      if (right.score !== left.score) return right.score - left.score
      if (right.updatedAt !== left.updatedAt) return right.updatedAt - left.updatedAt
      if (right.createdAt !== left.createdAt) return right.createdAt - left.createdAt
      return left.index - right.index
    })

  return preferred?.watchlist.id ?? null
}
