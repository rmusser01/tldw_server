import { useQuery } from "@tanstack/react-query"
import { useMemo } from "react"
import { useConnectionState } from "@/hooks/useConnectionState"
import { useConnectionStore } from "@/store/connection"
import { isRecoverableAuthConfigError } from "@/services/auth-errors"
import { tldwClient, type ServerChatSummary } from "@/services/tldw/TldwApiClient"
import type { ChatScope } from "@/types/chat-scope"
import type { ServicePromptRequestScope } from "@/services/tldw/domains/service-prompts"
import { createServicePromptScopeChangedError } from "@/services/tldw/service-prompt-scope-error"

export type ServerChatHistoryItem = ServerChatSummary & {
  createdAtMs: number
  updatedAtMs?: number | null
}

const SERVER_CHAT_FETCH_LIMIT = 200
const SERVER_CHAT_FETCH_MAX_PAGES = 50
const SERVER_CHAT_SEARCH_LIMIT = 50
export const SERVER_CHAT_HISTORY_OVERVIEW_PAGE_SIZE = 25

type FetchServerChatsPage = (params: {
  limit: number
  offset: number
  signal?: AbortSignal
}) => Promise<{
  chats: ServerChatSummary[]
  total: number
}>

const isAbortLikeError = (error: unknown): boolean => {
  const message =
    error instanceof Error
      ? error.message
      : typeof error === "string"
        ? error
        : String((error as { message?: unknown } | null)?.message || "")
  const normalizedMessage = message.toLowerCase()
  const name =
    error instanceof Error
      ? error.name
      : String((error as { name?: unknown } | null)?.name || "")
  return name === "AbortError" || normalizedMessage.includes("abort")
}

const isRateLimitedError = (error: unknown): boolean => {
  const status = (error as { status?: unknown } | null)?.status
  if (typeof status === "number" && Number.isFinite(status) && status === 429) {
    return true
  }

  const message =
    error instanceof Error
      ? error.message
      : typeof error === "string"
        ? error
        : String((error as { message?: unknown } | null)?.message || "")
  const normalized = message.toLowerCase()
  if (!normalized) return false

  return (
    normalized.includes("rate_limited") ||
    normalized.includes("rate limit") ||
    normalized.includes("too many requests")
  )
}

export const isRecoverableServerChatHistoryError = (error: unknown): boolean => {
  return (
    isRecoverableAuthConfigError(error) ||
    isAbortLikeError(error) ||
    isRateLimitedError(error)
  )
}

export const mapServerChatHistoryItems = (
  chats: ServerChatSummary[]
): ServerChatHistoryItem[] =>
  chats.map((chat) => ({
    ...chat,
    createdAtMs: Date.parse(chat.created_at || ""),
    updatedAtMs: chat.updated_at ? Date.parse(chat.updated_at) : null
  }))

export const filterServerChatHistoryItems = (
  items: ServerChatHistoryItem[],
  query: string
): ServerChatHistoryItem[] => {
  const normalizedQuery = query.trim().toLowerCase()
  if (!normalizedQuery) {
    return items
  }

  return items.filter((item) => {
    const haystack = `${item.title || ""} ${item.topic_label || ""} ${item.state || ""}`.toLowerCase()
    return haystack.includes(normalizedQuery)
  })
}

export const deriveServerChatHistoryViewState = ({
  previousData,
  error
}: {
  previousData: ServerChatHistoryItem[]
  error: unknown
}): {
  data: ServerChatHistoryItem[]
  total: number
  sidebarRefreshState: "recoverable-error"
  hasUsableData: boolean
  isShowingStaleData: boolean
} => ({
  data: previousData,
  total: previousData.length,
  sidebarRefreshState: "recoverable-error",
  hasUsableData: previousData.length > 0,
  isShowingStaleData: previousData.length > 0
})

export const fetchAllServerChatPages = async (
  fetchPage: FetchServerChatsPage,
  options?: {
    limit?: number
    maxPages?: number
    signal?: AbortSignal
  }
): Promise<ServerChatSummary[]> => {
  const limit = Math.max(1, options?.limit ?? SERVER_CHAT_FETCH_LIMIT)
  const maxPages = Math.max(1, options?.maxPages ?? SERVER_CHAT_FETCH_MAX_PAGES)
  const allChats: ServerChatSummary[] = []

  let offset = 0
  let total: number | null = null

  for (let page = 0; page < maxPages; page += 1) {
    let response: { chats: ServerChatSummary[]; total: number }
    try {
      response = await fetchPage({
        limit,
        offset,
        signal: options?.signal
      })
    } catch (error) {
      // Keep already-fetched chat history visible if a later page gets rate-limited.
      if (allChats.length > 0 && isRateLimitedError(error)) {
        break
      }
      throw error
    }
    const { chats, total: pageTotal } = response
    const batch = Array.isArray(chats) ? chats : []

    if (typeof pageTotal === "number" && Number.isFinite(pageTotal) && pageTotal >= 0) {
      total = pageTotal
    }

    if (batch.length === 0) {
      break
    }

    allChats.push(...batch)
    offset += batch.length

    const reachedTotal = total !== null && offset >= total
    const reachedLastPage = batch.length < limit

    if (reachedTotal || reachedLastPage) {
      break
    }
  }

  return allChats
}

type UseServerChatHistoryOptions = {
  owner?: {
    key: string
    requestScope: ServicePromptRequestScope
    isCurrent: () => boolean
  }
  enabled?: boolean
  includeDeleted?: boolean
  deletedOnly?: boolean
  scope?: ChatScope
  mode?: "overview" | "search"
  page?: number
  limit?: number
  filterMode?: "all" | "character" | "non_character" | "trash"
}

type ServerChatHistoryQueryData = {
  items: ServerChatHistoryItem[]
  total: number
}

const getCharacterScopeForFilterMode = (
  filterMode: NonNullable<UseServerChatHistoryOptions["filterMode"]>
): "character" | "non_character" | undefined => {
  if (filterMode === "character") {
    return "character"
  }
  if (filterMode === "non_character") {
    return "non_character"
  }
  return undefined
}

const supportsServerPagedOverview = (
  filterMode: NonNullable<UseServerChatHistoryOptions["filterMode"]>
): boolean =>
  filterMode === "all" ||
  filterMode === "trash" ||
  filterMode === "character" ||
  filterMode === "non_character"

export const useServerChatHistory = (
  searchQuery: string,
  options?: UseServerChatHistoryOptions
) => {
  const { isConnected } = useConnectionState()
  const checkConnection = useConnectionStore((state) => state.checkOnce)
  const normalizedQuery = searchQuery.trim().toLowerCase()
  const mode = options?.mode ?? "overview"
  const includeDeleted = options?.includeDeleted ?? false
  const deletedOnly = options?.deletedOnly ?? false
  const scope = options?.scope
  const owner = options?.owner
  const ownerCurrent = !owner || owner.isCurrent()
  const filterMode = options?.filterMode ?? (deletedOnly ? "trash" : "all")
  const characterScope = getCharacterScopeForFilterMode(filterMode)
  const overviewPage = Math.max(1, Math.trunc(options?.page ?? 1))
  const overviewLimit = Math.max(
    1,
    Math.min(
      SERVER_CHAT_FETCH_LIMIT,
      Math.trunc(options?.limit ?? SERVER_CHAT_HISTORY_OVERVIEW_PAGE_SIZE)
    )
  )
  const searchPage = Math.max(1, Math.trunc(options?.page ?? 1))
  const searchLimit = Math.max(
    1,
    Math.min(
      SERVER_CHAT_FETCH_LIMIT,
      Math.trunc(options?.limit ?? SERVER_CHAT_SEARCH_LIMIT)
    )
  )
  const canUsePagedOverview =
    mode === "overview" && supportsServerPagedOverview(filterMode)
  const canUseConversationSearch =
    mode === "search" && normalizedQuery.length > 0
  const isServerPagedResult = canUseConversationSearch || canUsePagedOverview
  const queryStrategy = canUseConversationSearch
    ? "search-server"
    : canUsePagedOverview
      ? "overview-page"
      : "overview-full"
  const isEnabled =
    ownerCurrent &&
    isConnected &&
    (options?.enabled ?? true) &&
    (mode !== "search" || normalizedQuery.length > 0)

  const query = useQuery({
    queryKey: [
      "serverChatHistory",
      {
        includeDeleted,
        deletedOnly,
        mode,
        q: mode === "search" ? normalizedQuery : "",
        strategy: queryStrategy,
        page: isServerPagedResult
          ? canUseConversationSearch
            ? searchPage
            : overviewPage
          : 1,
        limit: isServerPagedResult
          ? canUseConversationSearch
            ? searchLimit
            : overviewLimit
          : null,
        filterMode,
        ...(owner ? { ownerKey: owner.key } : {}),
        scope
      }
    ],
    enabled: isEnabled,
    queryFn: async ({ signal }): Promise<ServerChatHistoryQueryData> => {
      const assertOwner = () => {
        if (owner && !owner.isCurrent()) throw createServicePromptScopeChangedError()
      }
      assertOwner()
      await tldwClient.initialize().catch(() => null)
      assertOwner()
      try {
        if (canUseConversationSearch) {
          const response = await tldwClient.searchConversationsWithMeta(
            {
              query: normalizedQuery,
              limit: searchLimit,
              offset: (searchPage - 1) * searchLimit,
              order_by: "recency",
              ...(includeDeleted || deletedOnly ? { include_deleted: true } : {}),
              ...(deletedOnly ? { deleted_only: true } : {}),
              ...(characterScope ? { character_scope: characterScope } : {})
            },
            { signal, scope, ...(owner ? { requestScope: owner.requestScope } : {}) }
          )
          assertOwner()
          return {
            items: mapServerChatHistoryItems(response.chats),
            total:
              typeof response.total === "number" ? response.total : response.chats.length
          }
        }

        if (canUsePagedOverview) {
          const response = await tldwClient.listChatsWithMeta(
            {
              limit: overviewLimit,
              offset: (overviewPage - 1) * overviewLimit,
              ordering: "-updated_at",
              include_message_counts: false,
              ...(characterScope ? { character_scope: characterScope } : {}),
              ...(includeDeleted ? { include_deleted: true } : {}),
              ...(deletedOnly ? { deleted_only: true } : {})
            },
            { signal, scope, ...(owner ? { requestScope: owner.requestScope } : {}) }
          )
          assertOwner()
          return {
            items: mapServerChatHistoryItems(response.chats),
            total: response.total
          }
        }

        const chats = await fetchAllServerChatPages(
          ({ limit, offset, signal: pageSignal }) =>
            tldwClient.listChatsWithMeta(
              {
                limit,
                offset,
                ordering: "-updated_at",
                include_message_counts: false,
                ...(characterScope ? { character_scope: characterScope } : {}),
                ...(includeDeleted ? { include_deleted: true } : {}),
                ...(deletedOnly ? { deleted_only: true } : {})
              },
              { signal: pageSignal, scope, ...(owner ? { requestScope: owner.requestScope } : {}) }
            ),
          {
            signal
          }
        )

        assertOwner()
        const items = mapServerChatHistoryItems(chats)
        return {
          items,
          total: items.length
        }
      } catch (e) {
        if (isRecoverableServerChatHistoryError(e)) {
          // Keep sidebar/chat shell usable while connection state catches up.
          void checkConnection().catch(() => null)
          throw e
        }
        // eslint-disable-next-line no-console
        console.error(
          "[serverChatHistory] Failed to fetch server chats",
          e
        )
        throw e
      }
    },
    staleTime: 60_000,
    gcTime: 5 * 60_000,
    refetchOnMount: false,
    retry: (failureCount, error) =>
      !isRecoverableServerChatHistoryError(error) && failureCount < 1
  })

  const filteredData = useMemo(
    () =>
      !ownerCurrent ? [] : canUseConversationSearch
        ? query.data?.items || []
        : filterServerChatHistoryItems(query.data?.items || [], normalizedQuery),
    [ownerCurrent, canUseConversationSearch, query.data, normalizedQuery]
  )
  const resolvedTotal = useMemo(() => {
    if (!ownerCurrent) return 0
    if (isServerPagedResult) {
      return query.data?.total ?? filteredData.length
    }
    return filteredData.length
  }, [ownerCurrent, filteredData.length, isServerPagedResult, query.data])

  const sidebarState = useMemo(() => {
    if (query.status === "success") {
      return {
        data: filteredData,
        total: resolvedTotal,
        sidebarRefreshState: "ready" as const,
        hasUsableData: filteredData.length > 0,
        isShowingStaleData: false
      }
    }

    if (query.status === "error") {
      if (isRecoverableServerChatHistoryError(query.error)) {
        return deriveServerChatHistoryViewState({
          previousData: filteredData,
          error: query.error
        })
      }

      return {
        data: filteredData,
        total: resolvedTotal,
        sidebarRefreshState: "hard-error" as const,
        hasUsableData: filteredData.length > 0,
        isShowingStaleData: false
      }
    }

    return {
      data: filteredData,
      total: resolvedTotal,
      sidebarRefreshState: "idle" as const,
      hasUsableData: filteredData.length > 0,
      isShowingStaleData: false
    }
  }, [filteredData, query.error, query.status, resolvedTotal])

  return {
    ...query,
    data: sidebarState.data,
    total: sidebarState.total,
    sidebarRefreshState: sidebarState.sidebarRefreshState,
    hasUsableData: sidebarState.hasUsableData,
    isShowingStaleData: sidebarState.isShowingStaleData,
    isServerPagedResult
  }
}
