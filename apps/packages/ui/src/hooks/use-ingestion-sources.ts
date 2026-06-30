import {
  useMutation,
  useQuery,
  useQueryClient,
  type UseMutationResult,
  type UseQueryResult
} from "@tanstack/react-query"

import { useTldwApiClient } from "@/hooks/useTldwApiClient"
import type { TldwApiClient } from "@/services/tldw/TldwApiClient"
import type {
  CreateIngestionSourceRequest,
  IngestionSourceDirectoryBrowseResponse,
  IngestionSourceItem,
  IngestionSourceItemFilters,
  IngestionSourceItemsListResponse,
  IngestionSourceListResponse,
  IngestionSourceSummary,
  IngestionSourceSyncTriggerResponse,
  UpdateIngestionSourceRequest
} from "@/types/ingestion-sources"

type IngestionSourceApiMethodName =
  | "listIngestionSources"
  | "getIngestionSource"
  | "listIngestionSourceItems"
  | "browseIngestionSourceDirectories"
  | "createIngestionSource"
  | "updateIngestionSource"
  | "syncIngestionSource"
  | "uploadIngestionSourceArchive"
  | "reattachIngestionSourceItem"

type BoundApiMethod<T> = T extends (this: any, ...args: infer Args) => infer Result
  ? (...args: Args) => Result
  : T

type IngestionSourcesApiClient = {
  [Key in IngestionSourceApiMethodName]: BoundApiMethod<TldwApiClient[Key]>
}

export const ingestionSourceKeys = {
  all: () => ["ingestion-sources"] as const,
  list: () => [...ingestionSourceKeys.all(), "list"] as const,
  detail: (sourceId: string) =>
    [...ingestionSourceKeys.all(), "detail", String(sourceId)] as const,
  itemsRoot: (sourceId: string) =>
    [...ingestionSourceKeys.all(), "items", String(sourceId)] as const,
  items: (sourceId: string, filters?: IngestionSourceItemFilters) =>
    [...ingestionSourceKeys.itemsRoot(sourceId), filters ?? {}] as const,
  directoryBrowse: (path: string | null) =>
    [...ingestionSourceKeys.all(), "directory-browse", path ?? "roots"] as const
}

const useResolvedApiClient = (api?: IngestionSourcesApiClient): IngestionSourcesApiClient => {
  const defaultClient = useTldwApiClient()
  return api ?? defaultClient
}

export const useIngestionSourcesQuery = (
  api?: IngestionSourcesApiClient,
  options?: { enabled?: boolean }
): UseQueryResult<IngestionSourceListResponse, Error> => {
  const client = useResolvedApiClient(api)
  return useQuery({
    queryKey: ingestionSourceKeys.list(),
    queryFn: () => client.listIngestionSources(),
    enabled: options?.enabled ?? true
  })
}

export const useIngestionSourceDetailQuery = (
  sourceId: string | null | undefined,
  api?: IngestionSourcesApiClient,
  options?: { enabled?: boolean }
): UseQueryResult<IngestionSourceSummary, Error> => {
  const client = useResolvedApiClient(api)
  return useQuery({
    queryKey: ingestionSourceKeys.detail(sourceId ?? "unknown"),
    queryFn: () => client.getIngestionSource(String(sourceId)),
    enabled: Boolean(sourceId) && (options?.enabled ?? true)
  })
}

export const useIngestionSourceItemsQuery = (
  sourceId: string | null | undefined,
  filters?: IngestionSourceItemFilters,
  api?: IngestionSourcesApiClient,
  options?: { enabled?: boolean }
): UseQueryResult<IngestionSourceItemsListResponse, Error> => {
  const client = useResolvedApiClient(api)
  return useQuery({
    queryKey: ingestionSourceKeys.items(sourceId ?? "unknown", filters),
    queryFn: () => client.listIngestionSourceItems(String(sourceId), filters),
    enabled: Boolean(sourceId) && (options?.enabled ?? true)
  })
}

export const useIngestionSourceDirectoryBrowseQuery = (
  path?: string | null,
  api?: IngestionSourcesApiClient,
  options?: { enabled?: boolean }
): UseQueryResult<IngestionSourceDirectoryBrowseResponse, Error> => {
  const client = useResolvedApiClient(api)
  const normalizedPath = typeof path === "string" && path.trim().length > 0 ? path.trim() : null
  return useQuery({
    queryKey: ingestionSourceKeys.directoryBrowse(normalizedPath),
    queryFn: () => client.browseIngestionSourceDirectories(normalizedPath),
    enabled: options?.enabled ?? true
  })
}

export const useCreateIngestionSourceMutation = (
  api?: IngestionSourcesApiClient
): UseMutationResult<IngestionSourceSummary, Error, CreateIngestionSourceRequest> => {
  const client = useResolvedApiClient(api)
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (payload) => client.createIngestionSource(payload),
    onSuccess: async () => {
      await queryClient.invalidateQueries({
        queryKey: ingestionSourceKeys.list()
      })
    }
  })
}

export const useUpdateIngestionSourceMutation = (
  sourceId: string,
  api?: IngestionSourcesApiClient
): UseMutationResult<IngestionSourceSummary, Error, UpdateIngestionSourceRequest> => {
  const client = useResolvedApiClient(api)
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (payload) => client.updateIngestionSource(sourceId, payload),
    onSuccess: async (data) => {
      queryClient.setQueryData(ingestionSourceKeys.detail(sourceId), data)
      await Promise.all([
        queryClient.invalidateQueries({
          queryKey: ingestionSourceKeys.list()
        }),
        queryClient.invalidateQueries({
          queryKey: ingestionSourceKeys.detail(sourceId)
        })
      ])
    }
  })
}

export const useSyncIngestionSourceMutation = (
  api?: IngestionSourcesApiClient
): UseMutationResult<IngestionSourceSyncTriggerResponse, Error, string> => {
  const client = useResolvedApiClient(api)
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (sourceId) => client.syncIngestionSource(sourceId),
    onSuccess: async (_data, sourceId) => {
      await Promise.all([
        queryClient.invalidateQueries({
          queryKey: ingestionSourceKeys.list()
        }),
        queryClient.invalidateQueries({
          queryKey: ingestionSourceKeys.detail(sourceId)
        })
      ])
    }
  })
}

export const useUploadIngestionSourceArchiveMutation = (
  sourceId: string,
  api?: IngestionSourcesApiClient
): UseMutationResult<IngestionSourceSyncTriggerResponse, Error, File> => {
  const client = useResolvedApiClient(api)
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (file) => client.uploadIngestionSourceArchive(sourceId, file),
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({
          queryKey: ingestionSourceKeys.list()
        }),
        queryClient.invalidateQueries({
          queryKey: ingestionSourceKeys.detail(sourceId)
        }),
        queryClient.invalidateQueries({
          queryKey: ingestionSourceKeys.itemsRoot(sourceId)
        })
      ])
    }
  })
}

export const useReattachIngestionSourceItemMutation = (
  sourceId: string,
  api?: IngestionSourcesApiClient
): UseMutationResult<IngestionSourceItem, Error, string> => {
  const client = useResolvedApiClient(api)
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (itemId) => client.reattachIngestionSourceItem(sourceId, itemId),
    onSuccess: async () => {
      await queryClient.invalidateQueries({
        queryKey: ingestionSourceKeys.itemsRoot(sourceId)
      })
    }
  })
}
