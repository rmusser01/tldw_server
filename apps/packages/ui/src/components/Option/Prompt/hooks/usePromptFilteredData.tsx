import React, { useMemo } from "react"
import { useQuery } from "@tanstack/react-query"
import {
  matchesPromptSearchText,
  matchesTagFilter,
  mapServerSearchItemsToLocalPrompts,
  PROMPT_SEARCH_FIELDS,
  type TagMatchMode
} from "../custom-prompts-utils"
import {
  searchPromptsServer
} from "@/services/prompts-api"
import {
  isPromptInCollection
} from "../prompt-collections-utils"
import {
  buildSyncBatchPlan
} from "../sync-batch-utils"
import type {
  PromptListQueryState,
  PromptRowVM,
  PromptSavedView
} from "../prompt-workspace-types"
import { classifyPromptRecipe } from "../prompt-recipe-library"

export interface UsePromptFilteredDataDeps {
  data: any[] | undefined
  isOnline: boolean
  normalizedSearchText: string
  shouldUseServerSearch: boolean
  projectFilter: string | null
  typeFilter: PromptListQueryState["typeFilter"]
  syncFilter: string
  usageFilter: "all" | "used" | "unused"
  tagFilter: string[]
  tagMatchMode: TagMatchMode
  savedView: PromptSavedView
  selectedCollection: { prompt_ids?: number[] } | null
  currentPage: number
  resultsPerPage: number
  promptSort: { key: "title" | "modifiedAt" | null; order: "ascend" | "descend" | null }
  getPromptKeywords: (prompt: any) => string[]
  getPromptTexts: (prompt: any) => { systemText: string | undefined; userText: string | undefined }
  getPromptType: (prompt: any) => string
  getPromptModifiedAt: (prompt: any) => number
  getPromptUsageCount: (prompt: any) => number
  getPromptLastUsedAt: (prompt: any) => number | null
  t: (key: string, opts?: Record<string, any>) => string
}

export function usePromptFilteredData(deps: UsePromptFilteredDataDeps) {
  const {
    data,
    isOnline,
    normalizedSearchText,
    shouldUseServerSearch,
    projectFilter,
    typeFilter,
    syncFilter,
    usageFilter,
    tagFilter,
    tagMatchMode,
    savedView,
    selectedCollection,
    currentPage,
    resultsPerPage,
    promptSort,
    getPromptKeywords,
    getPromptTexts,
    getPromptType,
    getPromptModifiedAt,
    getPromptUsageCount,
    getPromptLastUsedAt,
    t
  } = deps

  const {
    data: serverSearchData,
    status: serverSearchStatus
  } = useQuery({
    queryKey: [
      "searchPrompts",
      normalizedSearchText,
      currentPage,
      resultsPerPage
    ],
    queryFn: () =>
      searchPromptsServer({
        searchQuery: normalizedSearchText,
        searchFields: PROMPT_SEARCH_FIELDS,
        page: currentPage,
        resultsPerPage,
        includeDeleted: false
      }),
    enabled: shouldUseServerSearch
  })

  const allTags = useMemo(() => {
    const set = new Set<string>()
    ;(data || []).forEach((p: any) =>
      (getPromptKeywords(p) || []).forEach((t: string) => set.add(t))
    )
    return Array.from(set.values())
  }, [data, getPromptKeywords])

  const pendingSyncCount = useMemo(() => {
    const prompts = Array.isArray(data) ? data : []
    return prompts.filter((prompt: any) => prompt?.syncStatus === "pending").length
  }, [data])

  const localSyncBatchPlan = useMemo(() => {
    const prompts = Array.isArray(data) ? data : []
    return buildSyncBatchPlan(
      prompts.map((prompt: any) => ({
        prompt,
        syncStatus: prompt?.syncStatus
      }))
    )
  }, [data])

  const baseFilteredData = useMemo(() => {
    let items = ((data || []) as any[]).filter(
      (prompt) => classifyPromptRecipe(prompt).kind !== "quarantined_recipe"
    )
    if (projectFilter) {
      const projectId = parseInt(projectFilter, 10)
      if (!isNaN(projectId)) {
        items = items.filter((p) => p.studioProjectId === projectId)
      }
    }
    if (typeFilter !== "all") {
      items = items.filter((p) => {
        const promptType = getPromptType(p)
        if (typeFilter === "recipe") return promptType.startsWith("recipe_")
        if (typeFilter === "system") return promptType === "system" || promptType === "mixed"
        if (typeFilter === "quick") return promptType === "quick" || promptType === "mixed"
        return promptType === typeFilter
      })
    }
    if (syncFilter !== "all") {
      items = items.filter((p) => (p.syncStatus || "local") === syncFilter)
    }
    if (usageFilter !== "all") {
      items = items.filter((p) =>
        usageFilter === "used"
          ? getPromptUsageCount(p) > 0
          : getPromptUsageCount(p) === 0
      )
    }
    if (selectedCollection) {
      const selectedPromptIds = new Set(selectedCollection.prompt_ids || [])
      items = items.filter((prompt) =>
        isPromptInCollection(prompt, selectedPromptIds)
      )
    }
    if (tagFilter.length > 0) {
      items = items.filter((p) =>
        matchesTagFilter(getPromptKeywords(p), tagFilter, tagMatchMode)
      )
    }
    if (savedView === "favorites") {
      items = items.filter((p) => !!p.favorite)
    } else if (savedView === "recent") {
      items = [...items]
        .filter((p) => typeof p.lastUsedAt === "number" && p.lastUsedAt > 0)
        .sort((a, b) => (b.lastUsedAt || 0) - (a.lastUsedAt || 0))
        .slice(0, 20)
    } else if (savedView === "most_used") {
      items = [...items]
        .filter((p) => getPromptUsageCount(p) > 0)
        .sort((a, b) => getPromptUsageCount(b) - getPromptUsageCount(a))
        .slice(0, 20)
    } else if (savedView === "untagged") {
      items = items.filter((p) => {
        const kw = getPromptKeywords(p)
        return !kw || kw.length === 0
      })
    }
    if (savedView === "all" || savedView === "favorites" || savedView === "untagged") {
      items = items.sort(
        (a, b) =>
          Number(!!b.favorite) - Number(!!a.favorite) ||
          (b.createdAt || 0) - (a.createdAt || 0)
      )
    }
    return items
  }, [
    data,
    projectFilter,
    typeFilter,
    syncFilter,
    usageFilter,
    selectedCollection,
    tagFilter,
    tagMatchMode,
    savedView,
    getPromptUsageCount,
    getPromptKeywords,
    getPromptType
  ])

  const sidebarCounts = useMemo(() => {
    const all = ((data || []) as any[]).filter(
      (prompt) => classifyPromptRecipe(prompt).kind !== "quarantined_recipe"
    )
    const typeCounts: Record<string, number> = { all: all.length }
    const syncCounts: Record<string, number> = { all: all.length }
    const tagCounts: Record<string, number> = {}
    let favCount = 0
    let recentCount = 0
    let mostUsedCount = 0
    let untaggedCount = 0

    for (const p of all) {
      const pt = getPromptType(p)
      typeCounts[pt] = (typeCounts[pt] || 0) + 1
      if (pt.startsWith("recipe_")) {
        typeCounts.recipe = (typeCounts.recipe || 0) + 1
      }
      const ss = p.syncStatus || "local"
      syncCounts[ss] = (syncCounts[ss] || 0) + 1
      const kw = getPromptKeywords(p)
      if (!kw || kw.length === 0) {
        untaggedCount++
      } else {
        for (const tag of kw) {
          tagCounts[tag] = (tagCounts[tag] || 0) + 1
        }
      }
      if (p.favorite) favCount++
      if (typeof p.lastUsedAt === "number" && p.lastUsedAt > 0) recentCount++
      if (getPromptUsageCount(p) > 0) mostUsedCount++
    }

    return {
      typeCounts,
      syncCounts,
      tagCounts,
      smartCounts: {
        all: all.length,
        favorites: favCount,
        recent: Math.min(recentCount, 20),
        most_used: Math.min(mostUsedCount, 20),
        untagged: untaggedCount,
      } as Partial<Record<PromptSavedView, number>>,
    }
  }, [data, getPromptType, getPromptKeywords, getPromptUsageCount])

  const localSearchFilteredData = useMemo(() => {
    if (normalizedSearchText.length === 0) {
      return baseFilteredData
    }
    const queryLower = normalizedSearchText.toLowerCase()
    return baseFilteredData.filter((prompt) =>
      matchesPromptSearchText(prompt, queryLower, getPromptKeywords)
    )
  }, [baseFilteredData, normalizedSearchText, getPromptKeywords])

  const serverSearchMappedData = useMemo(() => {
    if (!shouldUseServerSearch || serverSearchStatus !== "success" || !serverSearchData) {
      return []
    }
    return mapServerSearchItemsToLocalPrompts(serverSearchData.items, baseFilteredData)
  }, [baseFilteredData, serverSearchData, serverSearchStatus, shouldUseServerSearch])

  const hasLocalRecipeSearchOverlay = useMemo(
    () =>
      shouldUseServerSearch &&
      localSearchFilteredData.some((prompt: any) => {
        const syncStatus = prompt?.syncStatus ?? "local"
        return (
          classifyPromptRecipe(prompt).kind === "recipe" &&
          (syncStatus === "local" || syncStatus === "pending")
        )
      }),
    [localSearchFilteredData, shouldUseServerSearch]
  )

  const useServerSearchResults =
    shouldUseServerSearch &&
    serverSearchStatus === "success" &&
    !hasLocalRecipeSearchOverlay

  const filteredData = useMemo(() => {
    if (useServerSearchResults) {
      return serverSearchMappedData
    }
    return localSearchFilteredData
  }, [
    localSearchFilteredData,
    serverSearchMappedData,
    useServerSearchResults
  ])

  const sortedFilteredData = useMemo(() => {
    if (!promptSort.key || !promptSort.order) {
      return filteredData
    }

    const direction = promptSort.order === "ascend" ? 1 : -1
    const items = [...filteredData]

    items.sort((a, b) => {
      let compare = 0
      if (promptSort.key === "title") {
        compare = String(a?.name || a?.title || "").localeCompare(
          String(b?.name || b?.title || "")
        )
      } else if (promptSort.key === "modifiedAt") {
        compare = getPromptModifiedAt(a) - getPromptModifiedAt(b)
      }
      if (compare === 0) {
        compare = getPromptModifiedAt(a) - getPromptModifiedAt(b)
      }
      return compare * direction
    })

    return items
  }, [
    filteredData,
    getPromptModifiedAt,
    promptSort.key,
    promptSort.order
  ])

  const paginatedData = useMemo(() => {
    if (useServerSearchResults) {
      return sortedFilteredData
    }
    const start = (currentPage - 1) * resultsPerPage
    return sortedFilteredData.slice(start, start + resultsPerPage)
  }, [currentPage, resultsPerPage, sortedFilteredData, useServerSearchResults])

  const tableTotal = useMemo(() => {
    if (useServerSearchResults) {
      return Math.max(
        serverSearchData?.total_matches ?? 0,
        sortedFilteredData.length
      )
    }
    return sortedFilteredData.length
  }, [serverSearchData?.total_matches, sortedFilteredData.length, useServerSearchResults])

  const customPromptRows = useMemo<PromptRowVM[]>(() => {
    return paginatedData.map((prompt: any) => {
      const { systemText, userText } = getPromptTexts(prompt)
      const recipe = classifyPromptRecipe(prompt)
      return {
        id: String(prompt?.id || ""),
        title:
          prompt?.name ||
          prompt?.title ||
          t("common:untitled", { defaultValue: "Untitled" }),
        author: prompt?.author,
        details: prompt?.details,
        previewSystem: systemText || undefined,
        previewUser: userText || undefined,
        keywords: getPromptKeywords(prompt) || [],
        favorite: !!prompt?.favorite,
        syncStatus: prompt?.syncStatus || "local",
        sourceSystem: prompt?.sourceSystem || "workspace",
        serverId: prompt?.serverId,
        updatedAt: getPromptModifiedAt(prompt),
        createdAt:
          typeof prompt?.createdAt === "number" ? prompt.createdAt : Date.now(),
        usageCount: getPromptUsageCount(prompt),
        lastUsedAt: getPromptLastUsedAt(prompt),
        kind:
          recipe.kind === "recipe"
            ? "recipe"
            : recipe.kind === "quarantined_recipe"
              ? "quarantined_recipe"
              : "prompt",
        recipeTarget: recipe.kind === "recipe" ? recipe.target : undefined
      }
    })
  }, [
    paginatedData,
    getPromptTexts,
    t,
    getPromptKeywords,
    getPromptModifiedAt,
    getPromptUsageCount,
    getPromptLastUsedAt
  ])

  const hiddenServerResultsOnPage = useMemo(() => {
    if (!useServerSearchResults || !serverSearchData) {
      return 0
    }
    return Math.max(0, serverSearchData.items.length - serverSearchMappedData.length)
  }, [serverSearchData, serverSearchMappedData.length, useServerSearchResults])

  return {
    serverSearchStatus,
    allTags,
    pendingSyncCount,
    localSyncBatchPlan,
    baseFilteredData,
    sidebarCounts,
    sortedFilteredData,
    customPromptRows,
    tableTotal,
    hiddenServerResultsOnPage,
    useServerSearchResults
  }
}
