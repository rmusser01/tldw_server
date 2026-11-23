import { useCallback, useSyncExternalStore } from "react"
import { useQuery, useQueryClient, type QueryFunction, type UseQueryOptions } from "@tanstack/react-query"
import {
  getWatchlistsAccountGeneration,
  watchWatchlistsAccountChanges
} from "@/services/watchlists-account-boundary"

export const useWatchlistsAccountGeneration = () =>
  useSyncExternalStore(watchWatchlistsAccountChanges, getWatchlistsAccountGeneration, () => 0)

export const useIsCurrentWatchlistsAccount = () => {
  const generation = useWatchlistsAccountGeneration()
  return useCallback(() => generation === getWatchlistsAccountGeneration(), [generation])
}

export const useAssertCurrentWatchlistsAccount = () => {
  const isCurrent = useIsCurrentWatchlistsAccount()
  return useCallback((signal?: AbortSignal) => {
    signal?.throwIfAborted()
    if (!isCurrent()) throw new DOMException("Watchlists account changed", "AbortError")
  }, [isCurrent])
}

const queryKey = (generation: number, resource: string, params: unknown) =>
  ["watchlists", generation, resource, params] as const

export const useWatchlistsQuery = <T>(
  resource: string,
  params: unknown,
  queryFn: QueryFunction<T>,
  options?: Pick<UseQueryOptions<T>, "enabled" | "refetchInterval" | "refetchIntervalInBackground" | "gcTime">
) => {
  const generation = useWatchlistsAccountGeneration()
  return useQuery({
    queryKey: queryKey(generation, resource, params),
    queryFn,
    staleTime: 15_000,
    retry: false,
    ...options
  })
}

export const useSetWatchlistsQueryData = () => {
  const client = useQueryClient()
  const generation = useWatchlistsAccountGeneration()
  return useCallback(<T>(resource: string, params: unknown, update: (data: T | undefined) => T | undefined) => {
    if (generation !== getWatchlistsAccountGeneration()) return
    client.setQueryData<T>(queryKey(generation, resource, params), update)
  }, [client, generation])
}

export const useCancelWatchlistsQuery = () => {
  const client = useQueryClient()
  const generation = useWatchlistsAccountGeneration()
  return useCallback((resource: string, params: unknown) => {
    if (generation !== getWatchlistsAccountGeneration()) return
    return client.cancelQueries({ queryKey: queryKey(generation, resource, params), exact: true })
  }, [client, generation])
}

export const useFetchWatchlistsQuery = () => {
  const client = useQueryClient()
  const generation = useWatchlistsAccountGeneration()
  const assertCurrent = useAssertCurrentWatchlistsAccount()
  return useCallback(async <T>(resource: string, params: unknown, queryFn: QueryFunction<T>) => {
    assertCurrent()
    const data = await client.fetchQuery({
      queryKey: queryKey(generation, resource, params),
      queryFn,
      retry: false
    })
    assertCurrent()
    return data
  }, [assertCurrent, client, generation])
}

export const useInvalidateWatchlists = () => {
  const client = useQueryClient()
  const generation = useWatchlistsAccountGeneration()
  return useCallback(async (...resources: string[]) => {
    if (generation !== getWatchlistsAccountGeneration()) return
    const prefix = ["watchlists", generation]
    const filters = resources.length === 0
      ? [{ queryKey: prefix }]
      : resources.map(resource => ({ queryKey: [...prefix, resource] }))
    await Promise.all(filters.map(filter => client.cancelQueries(filter, { revert: false })))
    if (generation !== getWatchlistsAccountGeneration()) return
    await Promise.all(filters.map(filter => client.invalidateQueries(filter)))
  }, [client, generation])
}
