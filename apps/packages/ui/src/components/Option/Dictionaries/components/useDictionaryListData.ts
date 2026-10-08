import type { useChatDraftOwner } from "@/hooks/useChatDraftOwner"
import { loadServicePromptSnapshot } from "@/services/service-prompts"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { useQuery, useQueryClient } from "@tanstack/react-query"
import React from "react"

import {
  buildDuplicateDictionaryName,
  filterDictionariesBySearch,
  normalizeDictionaryTags
} from "../listUtils"

type UseDictionaryListDataParams = {
  owner: ReturnType<typeof useChatDraftOwner>
  isOnline: boolean
  dictionarySearch: string
  dictionaryCategoryFilter: string
  dictionaryTagFilters: string[]
  openEntries: number | null
  notification: {
    success: (config: { message: string; description?: string }) => void
    error: (config: { message: string; description?: string }) => void
  }
  queryClient: {
    invalidateQueries: (input: {
      queryKey: readonly unknown[]
    }) => Promise<unknown>
  }
}

type UseDictionaryListDataResult = {
  data: any[] | undefined
  status: "pending" | "error" | "success"
  error: unknown
  refetch: () => Promise<unknown>
  filteredDictionaries: any[]
  categoryFilterOptions: string[]
  tagFilterOptions: string[]
  activeEntriesDictionary: any | null
  dictionariesById: Map<number, any>
  isDictionaryCurrent: (dictionary: unknown) => boolean
  duplicateDictionary: (dictionary: any) => Promise<void>
}

export function useDictionaryListData({
  owner,
  isOnline,
  dictionarySearch,
  dictionaryCategoryFilter,
  dictionaryTagFilters,
  openEntries,
  notification,
  queryClient
}: UseDictionaryListDataParams): UseDictionaryListDataResult {
  const cache = useQueryClient()
  const queryKey = React.useMemo(
    () => ["tldw:listDictionaries", owner.serverScopeKey],
    [owner.serverScopeKey]
  )
  const query = useQuery({
    queryKey,
    enabled: isOnline && owner.isCurrent(),
    retry: false,
    queryFn: async ({ signal }) => {
      if (!owner.isCurrent() || !owner.requestScope)
        throw new Error("Verify the dictionary account before loading.")
      const snapshot = await loadServicePromptSnapshot([], {
        signal,
        requestScope: owner.requestScope
      })
      try {
        if (!owner.isCurrent() || snapshot.scopeSignal.aborted)
          throw new Error("The dictionary account changed. Reload the library.")
        await tldwClient.initialize()
        if (!owner.isCurrent() || snapshot.scopeSignal.aborted)
          throw new Error("The dictionary account changed. Reload the library.")
        const response = await tldwClient.listDictionaries(true, true, {
          signal: snapshot.scopeSignal,
          requestScope: owner.requestScope
        })
        if (!owner.isCurrent() || snapshot.scopeSignal.aborted)
          throw new Error("The dictionary account changed. Reload the library.")
        if (
          !response ||
          !Array.isArray(response.dictionaries) ||
          response.dictionaries.some(
            (row: { id?: unknown } | null) =>
              !row ||
              typeof row.id !== "number" ||
              !Number.isSafeInteger(row.id) ||
              row.id <= 0
          )
        )
          throw new Error(
            "The dictionary list could not be verified. Retry loading it."
          )
        return {
          ownerIsCurrent: owner.isCurrent,
          dictionaries: response.dictionaries as NonNullable<
            UseDictionaryListDataResult["data"]
          >
        }
      } finally {
        snapshot.release()
      }
    }
  })
  const ownerIsCurrent = owner.isCurrent
  const ownerScopeKey = owner.serverScopeKey
  const previousOwner = React.useRef({
    key: ownerScopeKey,
    isCurrent: ownerIsCurrent
  })
  const refetchQuery = query.refetch
  React.useEffect(() => {
    const previous = previousOwner.current
    previousOwner.current = { key: ownerScopeKey, isCurrent: ownerIsCurrent }
    if (previous.key !== ownerScopeKey || previous.isCurrent === ownerIsCurrent)
      return
    void cache.cancelQueries({ queryKey, exact: true }).then(() => {
      if (ownerIsCurrent() && isOnline) void refetchQuery()
    })
  }, [cache, queryKey, refetchQuery, ownerScopeKey, ownerIsCurrent, isOnline])
  const data =
    query.status === "success" &&
    query.fetchStatus === "idle" &&
    query.data?.ownerIsCurrent === owner.isCurrent &&
    owner.isCurrent()
      ? query.data.dictionaries
      : undefined
  const ownerFailed = owner.verificationState === "error" && !owner.isCurrent()
  const paused = query.fetchStatus === "paused" || !isOnline
  const status =
    ownerFailed || paused
      ? "error"
      : query.status === "success" && !data
        ? "pending"
        : query.status
  const error = ownerFailed
    ? new Error(
        "The server account could not be verified. Retry account verification."
      )
    : paused
      ? new Error("You are offline. Reconnect and retry loading dictionaries.")
      : query.error
  const currentOwnerPredicate = React.useRef(owner.isCurrent)
  React.useLayoutEffect(() => {
    currentOwnerPredicate.current = owner.isCurrent
  }, [owner.isCurrent])
  const refetch = async () => {
    if (
      currentOwnerPredicate.current !== owner.isCurrent ||
      (owner.verificationState === "ready" && !owner.isCurrent())
    )
      return
    if (!owner.isCurrent()) {
      owner.retryVerification()
      return
    }
    if (isOnline) return await refetchQuery()
  }
  const isDictionaryCurrent = (dictionary: unknown) => {
    const state = cache.getQueryState<typeof query.data>(queryKey)
    return Boolean(
      isOnline &&
      owner.isCurrent() &&
      state?.status === "success" &&
      state.fetchStatus === "idle" &&
      state.data === query.data &&
      state.data?.ownerIsCurrent === owner.isCurrent &&
      state.data.dictionaries.includes(dictionary)
    )
  }

  const categoryFilterOptions = React.useMemo(() => {
    const categories = new Set<string>()
    for (const dictionary of Array.isArray(data) ? data : []) {
      const category =
        typeof dictionary?.category === "string"
          ? dictionary.category.trim()
          : ""
      if (category) categories.add(category)
    }
    return Array.from(categories).sort((a, b) => a.localeCompare(b))
  }, [data])

  const tagFilterOptions = React.useMemo(() => {
    const tags = new Set<string>()
    for (const dictionary of Array.isArray(data) ? data : []) {
      for (const tag of normalizeDictionaryTags(dictionary?.tags)) {
        tags.add(tag)
      }
    }
    return Array.from(tags).sort((a, b) => a.localeCompare(b))
  }, [data])

  const filteredDictionaries = React.useMemo(() => {
    const normalizedCategoryFilter = dictionaryCategoryFilter
      .trim()
      .toLowerCase()
    const normalizedTagFilters = dictionaryTagFilters
      .map((tag) => tag.trim().toLowerCase())
      .filter(Boolean)

    return filterDictionariesBySearch(
      Array.isArray(data) ? data : [],
      dictionarySearch
    )
      .filter((dictionary: any) => {
        if (!normalizedCategoryFilter) return true
        const dictionaryCategory = String(dictionary?.category || "")
          .trim()
          .toLowerCase()
        return dictionaryCategory === normalizedCategoryFilter
      })
      .filter((dictionary: any) => {
        if (normalizedTagFilters.length === 0) return true
        const dictionaryTags = normalizeDictionaryTags(dictionary?.tags).map(
          (tag) => tag.toLowerCase()
        )
        return normalizedTagFilters.some((tag) => dictionaryTags.includes(tag))
      })
  }, [data, dictionaryCategoryFilter, dictionarySearch, dictionaryTagFilters])

  const activeEntriesDictionary = React.useMemo(() => {
    if (openEntries == null) return null
    return (
      (Array.isArray(data) ? data : []).find(
        (dictionary: any) => Number(dictionary?.id) === Number(openEntries)
      ) || null
    )
  }, [data, openEntries])

  const dictionariesById = React.useMemo(() => {
    const next = new Map<number, any>()
    for (const item of Array.isArray(data) ? data : []) {
      const id = Number(item?.id)
      if (!Number.isNaN(id) && id > 0) {
        next.set(id, item)
      }
    }
    return next
  }, [data])

  const duplicateDictionary = React.useCallback(
    async (dictionary: any) => {
      try {
        const exported = await tldwClient.exportDictionaryJSON(dictionary.id)
        const existingNames = Array.isArray(data)
          ? data.map((item: any) => item?.name)
          : []
        const duplicateName = buildDuplicateDictionaryName(
          exported?.name || dictionary?.name || "Dictionary",
          existingNames
        )
        const duplicatePayload = {
          ...exported,
          name: duplicateName,
          description: exported?.description ?? dictionary?.description
        }
        await tldwClient.importDictionaryJSON(
          duplicatePayload,
          Boolean(dictionary?.is_active)
        )
        notification.success({
          message: "Dictionary duplicated",
          description: `"${duplicateName}" created.`
        })
        await queryClient.invalidateQueries({
          queryKey: ["tldw:listDictionaries"]
        })
      } catch (error: any) {
        notification.error({
          message: "Duplicate failed",
          description: error?.message || "Unable to duplicate dictionary"
        })
      }
    },
    [data, notification, queryClient]
  )

  return {
    data: Array.isArray(data) ? data : undefined,
    status,
    error,
    refetch,
    filteredDictionaries,
    categoryFilterOptions,
    tagFilterOptions,
    activeEntriesDictionary,
    dictionariesById,
    isDictionaryCurrent,
    duplicateDictionary
  }
}
