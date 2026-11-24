import { useMemo } from "react"
import { useWatchlistsQuery } from "@/hooks/useWatchlistsQuery"
import { fetchWatchlistSources } from "@/services/watchlists"
import type { WatchlistSource } from "@/types/watchlists"

const PAGE_SIZE = 200
const SOURCE_LIMIT = 500
const EMPTY_SOURCES: WatchlistSource[] = []

export function useScopeSources(watchlistId?: number | null, enabled = true) {
  const params = { watchlist_id: watchlistId ?? undefined, page: 1, size: PAGE_SIZE }
  const firstPage = useWatchlistsQuery("sources", params, ({ signal }) => fetchWatchlistSources(params, signal), { enabled })
  const firstItems = firstPage.data?.items ?? EMPTY_SOURCES
  const hasMore = firstItems.length === PAGE_SIZE && (firstPage.data?.total ?? Infinity) > firstItems.length
  const remainingParams = { ...params, page: 2, maxItems: SOURCE_LIMIT }
  const remaining = useWatchlistsQuery("sources", remainingParams, async ({ signal }) => {
    const items: WatchlistSource[] = []
    for (let page = remainingParams.page; items.length < SOURCE_LIMIT - PAGE_SIZE; page += 1) {
      const result = await fetchWatchlistSources({ ...params, page }, signal)
      const batch = Array.isArray(result.items) ? result.items : []
      items.push(...batch)
      if (batch.length < PAGE_SIZE || PAGE_SIZE + items.length >= result.total) break
    }
    return items.slice(0, SOURCE_LIMIT - PAGE_SIZE)
  }, { enabled: enabled && hasMore })

  const sources = useMemo(
    () => hasMore && remaining.data ? firstItems.concat(remaining.data) : firstItems,
    [firstItems, hasMore, remaining.data]
  )
  return { sources, loading: firstPage.isFetching || remaining.isFetching }
}
