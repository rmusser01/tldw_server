import { useCallback } from "react"
import { useQueryClient } from "@tanstack/react-query"
import { useSetWatchlistsQueryData, useWatchlistsAccountGeneration } from "@/hooks/useWatchlistsQuery"
import type { PaginatedResponse, WatchlistRun } from "@/types/watchlists"

export const usePatchRunLists = () => {
  const client = useQueryClient()
  const generation = useWatchlistsAccountGeneration()
  const setQueryData = useSetWatchlistsQueryData()

  return useCallback((runId: number, patch: Partial<WatchlistRun>) => {
    const pages = client.getQueriesData<PaginatedResponse<WatchlistRun>>({
      queryKey: ["watchlists", generation, "runs"]
    })
    for (const [key, page] of pages) {
      if (!page?.items?.some((run) => run.id === runId)) continue
      setQueryData<PaginatedResponse<WatchlistRun>>("runs", key[3], () => ({
        ...page,
        items: page.items.map((run) => run.id === runId ? { ...run, ...patch } : run)
      }))
    }
  }, [client, generation, setQueryData])
}
