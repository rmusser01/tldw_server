import { useCallback } from "react"
import { type QueryKey, useQueryClient } from "@tanstack/react-query"
import {
  useIsCurrentWatchlistsAccount,
  useSetWatchlistsQueryData,
  useWatchlistsAccountGeneration
} from "@/hooks/useWatchlistsQuery"
import type { PaginatedResponse, WatchlistRun } from "@/types/watchlists"

const isFilteredRunList = (key: QueryKey): boolean => {
  const params = key[3] as { q?: string; status?: string } | undefined
  return Boolean(params?.q || params?.status)
}

export const usePatchRunLists = () => {
  const client = useQueryClient()
  const generation = useWatchlistsAccountGeneration()
  const isCurrentAccount = useIsCurrentWatchlistsAccount()
  const setQueryData = useSetWatchlistsQueryData()

  return useCallback((
    runId: number,
    patch: Partial<WatchlistRun>,
    { statusChanged = false } = {}
  ) => {
    if (!isCurrentAccount()) return
    const queryKey = ["watchlists", generation, "runs"]
    const pages = client.getQueriesData<PaginatedResponse<WatchlistRun>>({ queryKey })
    let refreshFilteredLists = statusChanged

    for (const [key, page] of pages) {
      const run = page?.items?.find((item) => item.id === runId)
      if (!page || !run) continue
      // A status change can affect membership and pagination across filtered pages.
      if (patch.status !== undefined && run.status !== patch.status && isFilteredRunList(key)) {
        refreshFilteredLists = true
        continue
      }
      setQueryData<PaginatedResponse<WatchlistRun>>("runs", key[3], () => ({
        ...page,
        items: page.items.map((run) => run.id === runId ? { ...run, ...patch } : run)
      }))
    }

    if (refreshFilteredLists) {
      void client.invalidateQueries({
        queryKey,
        predicate: (query) => isFilteredRunList(query.queryKey)
      })
    }
  }, [client, generation, isCurrentAccount, setQueryData])
}
