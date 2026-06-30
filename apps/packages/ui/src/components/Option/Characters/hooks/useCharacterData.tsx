import React from "react"
import { useQuery, type QueryClient } from "@tanstack/react-query"
import {
  tldwClient,
  type ServerChatSummary
} from "@/services/tldw/TldwApiClient"
import { fetchFolders } from "@/services/folder-api"
import {
  buildTagUsage
} from "../tag-manager-utils"
import {
  filterCharactersForWorkspace,
  hasInlineConversationCount,
  paginateCharactersForWorkspace,
  sortCharactersForWorkspace
} from "../search-utils"
import {
  toCharactersSortBy,
  toCharactersSortOrder,
  toIsoBoundaryFromDateInput,
  isCharacterQueryRouteConflictError,
  buildCharacterFolderToken,
  getCharacterVisibleTags,
  normalizeWorldBookIds,
  toCharacterWorldBookOption,
  readFavoriteFromRecord,
  resolveCharacterSelectionId,
  buildCharacterSelectionPayload,
  EMPTY_CHARACTER_WORLD_BOOK_DATA,
  SERVER_QUERY_ROLLOUT_FLAG_KEY,
  type CharacterTableSortOrder,
  type CharacterWorldBookOption,
  type CharacterFolderOption,
  type DefaultCharacterPreferenceQueryResult
} from "../utils"
import { useStorage } from "@plasmohq/storage/hook"

export interface UseCharacterDataDeps {
  t: (key: string, opts?: Record<string, any>) => string
  notification: {
    error: (args: { message: string; description?: any }) => void
    warning: (args: { message: string; description?: any }) => void
    success: (args: { message: string; description?: any }) => void
    info: (args: { message: string; description?: any }) => void
  }
  qc: QueryClient
  /** Filter state from useCharacterFiltering */
  searchTerm: string
  debouncedSearchTerm: string
  filterTags: string[]
  folderFilterId: string | undefined
  matchAllTags: boolean
  creatorFilter: string | undefined
  createdFromDate: string
  createdToDate: string
  updatedFromDate: string
  updatedToDate: string
  hasConversationsOnly: boolean
  favoritesOnly: boolean
  characterListScope: "active" | "deleted"
  sortColumn: string | null
  sortOrder: CharacterTableSortOrder
  currentPage: number
  pageSize: number
  setCurrentPage: (page: number) => void
  /** From useCharacterModalState */
  previewCharacter: any
  setPreviewCharacter: React.Dispatch<React.SetStateAction<any>>
  editCharacterNumericId: number | null
  editWorldBooksInitializedRef: React.MutableRefObject<boolean>
  open: boolean
  openEdit: boolean
  editForm: any
  /** Default character storage */
  defaultCharacterSelection: any
  setDefaultCharacterSelection: (value: any) => Promise<void> | void
  defaultCharacterId: string | undefined
}

export function useCharacterData(deps: UseCharacterDataDeps) {
  const {
    t,
    notification,
    qc,
    debouncedSearchTerm,
    filterTags,
    folderFilterId,
    matchAllTags,
    creatorFilter,
    createdFromDate,
    createdToDate,
    updatedFromDate,
    updatedToDate,
    hasConversationsOnly,
    favoritesOnly,
    characterListScope,
    sortColumn,
    sortOrder,
    currentPage,
    pageSize,
    setCurrentPage,
    previewCharacter,
    setPreviewCharacter,
    editCharacterNumericId,
    editWorldBooksInitializedRef,
    open,
    openEdit,
    editForm,
    defaultCharacterId,
    setDefaultCharacterSelection
  } = deps

  const [serverQueryRolloutFlag] = useStorage<boolean | null>(
    SERVER_QUERY_ROLLOUT_FLAG_KEY,
    true
  )
  const isServerQueryRolloutEnabled = serverQueryRolloutFlag !== false

  const previewCharacterId = React.useMemo(() => {
    const parsed = Number(previewCharacter?.id)
    return Number.isFinite(parsed) && parsed > 0 ? parsed : null
  }, [previewCharacter?.id])

  // --- Server sort/filter memos ---
  const serverSortBy = React.useMemo(
    () => toCharactersSortBy(sortColumn),
    [sortColumn]
  )
  const serverSortOrder = React.useMemo(
    () => toCharactersSortOrder(sortOrder),
    [sortOrder]
  )
  const createdFromIso = React.useMemo(
    () => toIsoBoundaryFromDateInput(createdFromDate, "start"),
    [createdFromDate]
  )
  const createdToIso = React.useMemo(
    () => toIsoBoundaryFromDateInput(createdToDate, "end"),
    [createdToDate]
  )
  const updatedFromIso = React.useMemo(
    () => toIsoBoundaryFromDateInput(updatedFromDate, "start"),
    [updatedFromDate]
  )
  const updatedToIso = React.useMemo(
    () => toIsoBoundaryFromDateInput(updatedToDate, "end"),
    [updatedToDate]
  )
  const folderFilterToken = React.useMemo(
    () => buildCharacterFolderToken(folderFilterId),
    [folderFilterId]
  )
  const effectiveFilterTags = React.useMemo(() => {
    const merged = [...filterTags]
    if (folderFilterToken) {
      merged.push(folderFilterToken)
    }
    return Array.from(
      new Set(
        merged
          .map((tag) => String(tag).trim())
          .filter((tag) => tag.length > 0)
      )
    )
  }, [filterTags, folderFilterToken])
  const effectiveMatchAllTags = matchAllTags || Boolean(folderFilterToken)
  const useServerQuery =
    isServerQueryRolloutEnabled ||
    characterListScope === "deleted" ||
    Boolean(createdFromIso || createdToIso || updatedFromIso || updatedToIso)

  const characterQueryParams = React.useMemo(
    () => ({
      page: currentPage,
      page_size: pageSize,
      query: debouncedSearchTerm.trim() || undefined,
      tags: effectiveFilterTags.length > 0 ? effectiveFilterTags : undefined,
      match_all_tags:
        effectiveFilterTags.length > 0 ? effectiveMatchAllTags : undefined,
      creator: creatorFilter || undefined,
      has_conversations: hasConversationsOnly ? true : undefined,
      favorite_only: favoritesOnly ? true : undefined,
      created_from: createdFromIso,
      created_to: createdToIso,
      updated_from: updatedFromIso,
      updated_to: updatedToIso,
      include_deleted: characterListScope === "deleted" ? true : undefined,
      deleted_only: characterListScope === "deleted" ? true : undefined,
      sort_by: serverSortBy,
      sort_order: serverSortOrder,
      include_image_base64: true
    }),
    [
      characterListScope,
      createdFromIso,
      createdToIso,
      creatorFilter,
      currentPage,
      debouncedSearchTerm,
      effectiveFilterTags,
      effectiveMatchAllTags,
      favoritesOnly,
      hasConversationsOnly,
      pageSize,
      serverSortBy,
      serverSortOrder,
      updatedFromIso,
      updatedToIso
    ]
  )

  // --- Main character list query ---
  const {
    data: characterListResponse,
    status,
    error,
    refetch
  } = useQuery({
    queryKey: [
      "tldw:listCharacters",
      characterQueryParams,
      useServerQuery ? "server" : "legacy"
    ],
    queryFn: async () => {
      const hasLegacyClientFilters =
        Boolean(debouncedSearchTerm.trim()) ||
        effectiveFilterTags.length > 0 ||
        Boolean(creatorFilter) ||
        hasConversationsOnly ||
        favoritesOnly
      const loadLegacyCharacterPage = async () => {
        const offset = Math.max(0, (currentPage - 1) * pageSize)
        const response = await tldwClient.listCharacters({
          limit: pageSize,
          offset,
          query: debouncedSearchTerm.trim() || undefined,
          tags: effectiveFilterTags.length > 0 ? effectiveFilterTags : undefined,
          match_all_tags:
            effectiveFilterTags.length > 0 ? effectiveMatchAllTags : undefined,
          creator: creatorFilter || undefined,
          has_conversations: hasConversationsOnly ? true : undefined,
          favorite_only: favoritesOnly ? true : undefined,
          created_from: createdFromIso,
          created_to: createdToIso,
          updated_from: updatedFromIso,
          updated_to: updatedToIso,
          include_deleted: characterListScope === "deleted" ? true : undefined,
          deleted_only: characterListScope === "deleted" ? true : undefined,
          sort_by: serverSortBy,
          sort_order: serverSortOrder,
          include_image_base64: true
        })

        const candidate = response as
          | {
              items?: unknown
              total?: unknown
              page?: unknown
              page_size?: unknown
              has_more?: unknown
            }
          | undefined
          | null
        const items = Array.isArray(candidate?.items)
          ? candidate.items
          : Array.isArray(response)
            ? response
            : []
        const totalFromResponse =
          typeof candidate?.total === "number" && Number.isFinite(candidate.total)
            ? candidate.total
            : null
        const hasMoreFromResponse =
          typeof candidate?.has_more === "boolean"
            ? candidate.has_more
            : items.length >= pageSize
        const total =
          totalFromResponse ??
          (hasMoreFromResponse ? offset + items.length + 1 : offset + items.length)
        return {
          items,
          total,
          page: currentPage,
          page_size: pageSize,
          has_more: hasMoreFromResponse
        }
      }
      const loadLegacyCharacterList = async () => {
        const allCharacters = await tldwClient.listAllCharacters({
          pageSize: 250,
          maxPages: 50
        })
        const filtered = filterCharactersForWorkspace(allCharacters, {
          query: debouncedSearchTerm.trim() || undefined,
          tags: effectiveFilterTags,
          matchAllTags: effectiveMatchAllTags,
          creator: creatorFilter
        })
        const withConversationFilter = hasConversationsOnly
          ? filtered.filter((character) => hasInlineConversationCount(character))
          : filtered
        const withFavoritesFilter = favoritesOnly
          ? withConversationFilter.filter((character) =>
              readFavoriteFromRecord(character)
            )
          : withConversationFilter
        const sorted = sortCharactersForWorkspace(withFavoritesFilter, {
          sortBy: serverSortBy,
          sortOrder: serverSortOrder
        })
        const paged = paginateCharactersForWorkspace(sorted, {
          page: currentPage,
          pageSize
        })

        return {
          items: paged.items,
          total: paged.total,
          page: paged.page,
          page_size: paged.pageSize,
          has_more: paged.hasMore
        }
      }
      const loadLegacyFallbackAfterRouteConflict = async () => {
        if (characterListScope === "deleted") {
          return {
            items: [],
            total: 0,
            page: currentPage,
            page_size: pageSize,
            has_more: false
          }
        }
        if (!hasLegacyClientFilters) {
          try {
            return await loadLegacyCharacterPage()
          } catch {
            // Fall back to full-list pagination if server list page path is unavailable.
          }
        }
        return await loadLegacyCharacterList()
      }
      const buildEmptyCharacterQueryResponse = () => ({
        items: [],
        total: 0,
        page: currentPage,
        page_size: pageSize,
        has_more: false
      })

      try {
        await tldwClient.initialize()
        if (useServerQuery) {
          try {
            return await tldwClient.listCharactersPage(characterQueryParams)
          } catch (serverQueryError) {
            if (isCharacterQueryRouteConflictError(serverQueryError)) {
              return await loadLegacyFallbackAfterRouteConflict()
            }
            throw serverQueryError
          }
        }

        return await loadLegacyCharacterList()
      } catch (e: any) {
        if (useServerQuery && isCharacterQueryRouteConflictError(e)) {
          try {
            return await loadLegacyFallbackAfterRouteConflict()
          } catch {
            // fall through to existing notification + error state
          }
        }
        notification.error({
          message: t("settings:manageCharacters.notification.error", {
            defaultValue: "Error"
          }),
          description:
            e?.message ||
            t("settings:manageCharacters.notification.someError", {
              defaultValue: "Something went wrong. Please try again later"
            })
        })
        return buildEmptyCharacterQueryResponse()
      }
    },
    staleTime: 5 * 60 * 1000,
    throwOnError: false
  })

  // --- Default character preference ---
  const { data: defaultCharacterPreference } = useQuery<DefaultCharacterPreferenceQueryResult>({
    queryKey: ["tldw:defaultCharacterPreference"],
    queryFn: async () => {
      await tldwClient.initialize()
      const defaultCharacterId = await tldwClient.getDefaultCharacterPreference()
      return { defaultCharacterId }
    },
    staleTime: 60 * 1000,
    throwOnError: false
  })

  const serverDefaultCharacterId = defaultCharacterPreference?.defaultCharacterId
  const effectiveDefaultCharacterId =
    typeof serverDefaultCharacterId === "undefined"
      ? defaultCharacterId
      : serverDefaultCharacterId

  // --- Derived data from character list ---
  const isLegacyCharacterListResponse = Array.isArray(characterListResponse)
  const rawData = React.useMemo(
    () => {
      if (Array.isArray(characterListResponse)) return characterListResponse
      if (Array.isArray(characterListResponse?.items)) {
        return characterListResponse.items
      }
      return []
    },
    [characterListResponse]
  )
  const rawTotalCharacters = React.useMemo(
    () => {
      if (Array.isArray(characterListResponse)) {
        return characterListResponse.length
      }
      if (typeof characterListResponse?.total === "number") {
        return characterListResponse.total
      }
      return rawData.length
    },
    [characterListResponse, rawData.length]
  )

  // --- Conversation counts ---
  const characterIds = React.useMemo(() => {
    if (!Array.isArray(rawData)) return []
    return rawData
      .map((c: any) => String(c.id || c.slug || c.name))
      .filter(Boolean)
  }, [rawData])

  const { data: conversationCounts } = useQuery<Record<string, number>>({
    queryKey: ["tldw:characterConversationCounts", characterIds],
    queryFn: async () => {
      if (characterIds.length === 0) return {}
      await tldwClient.initialize()
      const chats: ServerChatSummary[] = []
      const chatPageSize = 200
      const maxPages = 5
      for (let pageIndex = 0; pageIndex < maxPages; pageIndex += 1) {
        const offset = pageIndex * chatPageSize
        const page = await tldwClient.listChats({
          limit: chatPageSize,
          offset,
          ordering: "-updated_at"
        })
        if (!Array.isArray(page) || page.length === 0) {
          break
        }
        chats.push(...page)
        if (page.length < chatPageSize) {
          break
        }
      }
      const counts: Record<string, number> = {}
      for (const chat of chats) {
        const charId = String(chat.character_id ?? "")
        if (charId && characterIds.includes(charId)) {
          counts[charId] = (counts[charId] || 0) + 1
        }
      }
      return counts
    },
    enabled: characterIds.length > 0,
    staleTime: 60 * 1000
  })

  // --- Preview world books ---
  const {
    data: previewCharacterWorldBooks = [],
    isFetching: previewCharacterWorldBooksLoading
  } = useQuery<Array<{ id: number; name: string }>>({
    queryKey: ["tldw:characterPreviewWorldBooks", previewCharacterId],
    queryFn: async () => {
      if (previewCharacterId == null) return []
      await tldwClient.initialize()
      const linkedBooks = await tldwClient.listCharacterWorldBooks(previewCharacterId)
      const parsed = Array.isArray(linkedBooks) ? linkedBooks : []
      return parsed
        .map((book: any) => {
          const worldBookId = Number(book?.world_book_id ?? book?.id)
          if (!Number.isFinite(worldBookId) || worldBookId <= 0) return null
          const rawName = book?.world_book_name ?? book?.name
          const worldBookName =
            typeof rawName === "string" && rawName.trim().length > 0
              ? rawName
              : `World Book ${worldBookId}`
          return { id: worldBookId, name: worldBookName }
        })
        .filter((book): book is { id: number; name: string } => book !== null)
        .sort((a, b) => a.name.localeCompare(b.name))
    },
    enabled: previewCharacterId != null,
    staleTime: 30 * 1000
  })

  // --- Edit world books ---
  const {
    data: characterWorldBookData = EMPTY_CHARACTER_WORLD_BOOK_DATA,
    isFetching: worldBookOptionsLoading
  } = useQuery<{ options: CharacterWorldBookOption[]; attachedIds: number[] }>({
    queryKey: ["tldw:characterEditWorldBooks", editCharacterNumericId],
    queryFn: async () => {
      await tldwClient.initialize()
      const allBooksResponse = await tldwClient.listWorldBooks(true)
      const attachedBooksResponse =
        editCharacterNumericId == null
          ? []
          : await tldwClient.listCharacterWorldBooks(editCharacterNumericId)

      const allBooks = Array.isArray(allBooksResponse?.world_books)
        ? allBooksResponse.world_books
        : Array.isArray(allBooksResponse)
          ? allBooksResponse
          : []
      const attachedBooks = Array.isArray(attachedBooksResponse)
        ? attachedBooksResponse
        : []

      const optionMap = new Map<number, CharacterWorldBookOption>()
      for (const rawBook of allBooks) {
        const option = toCharacterWorldBookOption(rawBook)
        if (!option) continue
        optionMap.set(option.id, option)
      }
      for (const rawBook of attachedBooks) {
        const option = toCharacterWorldBookOption(rawBook)
        if (!option) continue
        if (!optionMap.has(option.id)) {
          optionMap.set(option.id, option)
        }
      }

      const options = Array.from(optionMap.values()).sort((a, b) =>
        a.name.localeCompare(b.name)
      )
      const attachedIds = normalizeWorldBookIds(
        attachedBooks.map((book: any) => book?.world_book_id ?? book?.id)
      )

      return { options, attachedIds }
    },
    enabled: open || openEdit,
    staleTime: 30 * 1000
  })

  const worldBookOptions = characterWorldBookData.options

  // --- World book initialization effect ---
  React.useEffect(() => {
    if (!openEdit) {
      editWorldBooksInitializedRef.current = false
      editForm.setFieldValue("world_book_ids", [])
      return
    }
    if (editCharacterNumericId == null) return
    if (worldBookOptionsLoading) return
    if (editWorldBooksInitializedRef.current) return
    const attachedIds = normalizeWorldBookIds(characterWorldBookData.attachedIds)
    editWorldBooksInitializedRef.current = true
    editForm.setFieldValue("world_book_ids", attachedIds)
  }, [
    characterWorldBookData.attachedIds,
    editCharacterNumericId,
    editForm,
    openEdit,
    worldBookOptionsLoading
  ])

  // --- Filtered/merged data ---
  const data = React.useMemo(
    () => {
      const withConversationFilter =
        !isLegacyCharacterListResponse || !hasConversationsOnly
          ? rawData
          : rawData.filter((record: any) => {
              const charId = String(record?.id || record?.slug || record?.name || "")
              const mappedCount =
                typeof conversationCounts?.[charId] === "number"
                  ? conversationCounts[charId]
                  : undefined
              const inlineCountCandidates = [
                record?.conversation_count,
                record?.conversationCount,
                record?.chat_count,
                record?.chatCount
              ]
              const inlineCount = inlineCountCandidates.find(
                (value) => typeof value === "number" && Number.isFinite(value)
              ) as number | undefined
              return (mappedCount ?? inlineCount ?? 0) > 0
            })

      if (!favoritesOnly) {
        return withConversationFilter
      }
      return withConversationFilter.filter((record: any) =>
        readFavoriteFromRecord(record)
      )
    },
    [
      conversationCounts,
      favoritesOnly,
      hasConversationsOnly,
      isLegacyCharacterListResponse,
      rawData
    ]
  )

  // --- Sync default character from server ---
  React.useEffect(() => {
    if (typeof serverDefaultCharacterId === "undefined") return

    if (!serverDefaultCharacterId) {
      if (defaultCharacterId) {
        void setDefaultCharacterSelection(null)
      }
      return
    }

    if (serverDefaultCharacterId === defaultCharacterId) return

    const matchingCharacter = (data || []).find((record: any) => {
      const candidateId = resolveCharacterSelectionId({
        id: record?.id || record?.slug || record?.name
      } as any)
      return candidateId === serverDefaultCharacterId
    })

    if (matchingCharacter) {
      void setDefaultCharacterSelection(
        buildCharacterSelectionPayload(matchingCharacter)
      )
      return
    }

    void setDefaultCharacterSelection({ id: serverDefaultCharacterId } as any)
  }, [
    data,
    defaultCharacterId,
    serverDefaultCharacterId,
    setDefaultCharacterSelection
  ])

  // --- Focus character from cross-navigation ---
  const hasHandledFocusCharacterRef = React.useRef(false)
  const crossNavigationContext = React.useMemo(
    () => {
      if (typeof window === "undefined") {
        return {
          launchedFromWorldBooks: false,
          focusCharacterId: "",
          focusWorldBookId: null as number | null
        }
      }
      const params = new URLSearchParams(window.location.search)
      const focusWorldBookIdRaw = params.get("focusWorldBookId")
      const parsedFocusWorldBookId = Number(focusWorldBookIdRaw)
      return {
        launchedFromWorldBooks: params.get("from") === "world-books",
        focusCharacterId: params.get("focusCharacterId") || "",
        focusWorldBookId:
          Number.isFinite(parsedFocusWorldBookId) && parsedFocusWorldBookId > 0
            ? parsedFocusWorldBookId
            : null
      }
    },
    []
  )

  React.useEffect(() => {
    if (hasHandledFocusCharacterRef.current) return
    const focusCharacterId = crossNavigationContext.focusCharacterId.trim()
    if (!focusCharacterId) {
      hasHandledFocusCharacterRef.current = true
      return
    }
    if (status !== "success") return

    const matchingCharacter = (data || []).find((character: any) => {
      const candidates = [
        character?.id,
        character?.slug,
        character?.name
      ]
      return candidates.some(
        (candidate) => String(candidate || "").trim() === focusCharacterId
      )
    })

    hasHandledFocusCharacterRef.current = true
    if (matchingCharacter) {
      setPreviewCharacter(matchingCharacter)
    }
  }, [crossNavigationContext.focusCharacterId, data, setPreviewCharacter, status])

  // --- Total and page reset ---
  const totalCharacters = React.useMemo(
    () => (isLegacyCharacterListResponse ? data.length : rawTotalCharacters),
    [data.length, isLegacyCharacterListResponse, rawTotalCharacters]
  )

  React.useEffect(() => {
    if (status !== "success") return
    const maxPage = Math.max(1, Math.ceil(totalCharacters / pageSize))
    if (currentPage > maxPage) {
      setCurrentPage(maxPage)
    }
  }, [status, totalCharacters, currentPage, pageSize, setCurrentPage])

  const pagedGalleryData = data

  // --- Tag usage and filter options ---
  const tagUsageData = React.useMemo(() => {
    const source = Array.isArray(data) ? data : []
    return buildTagUsage(
      source.map((character: any) => ({
        ...character,
        tags: getCharacterVisibleTags(character?.tags)
      }))
    )
  }, [data])

  const allTags = React.useMemo(() => {
    return tagUsageData.map(({ tag }) => tag)
  }, [tagUsageData])

  const popularTags = React.useMemo(() => {
    return tagUsageData.slice(0, 5)
  }, [tagUsageData])

  const tagOptionsWithCounts = React.useMemo(() => {
    return tagUsageData.map(({ tag, count }) => ({
      label: (
        <span className="flex items-center justify-between w-full">
          <span>{tag}</span>
          <span className="text-xs text-text-subtle ml-2">({count})</span>
        </span>
      ),
      value: tag
    }))
  }, [tagUsageData])

  const tagFilterOptions = React.useMemo(
    () =>
      Array.from(
        new Set([...(allTags || []), ...(filterTags || [])].filter(Boolean))
      ).map((tag) => ({ label: tag, value: tag })),
    [allTags, filterTags]
  )

  const creatorFilterOptions = React.useMemo(() => {
    const source = Array.isArray(data) ? data : []
    const creators = Array.from(
      new Set(
        source
          .map((character: any) =>
            String(
              character?.creator ?? character?.created_by ?? character?.createdBy ?? ""
            ).trim()
          )
          .filter((creator) => creator.length > 0)
      )
    ).sort((a, b) => a.localeCompare(b))
    return creators.map((creator) => ({ label: creator, value: creator }))
  }, [data])

  // --- Folder options ---
  const { data: characterFolderOptions = [], isFetching: characterFolderOptionsLoading } =
    useQuery<CharacterFolderOption[]>({
      queryKey: ["tldw:characterFolders"],
      queryFn: async () => {
        const response = await fetchFolders({ timeoutMs: 5000 })
        if (!response.ok || !Array.isArray(response.data)) {
          return []
        }
        return response.data
          .map((folder: any) => {
            const folderId = Number(folder?.id)
            const folderName = String(folder?.name || "").trim()
            if (!Number.isFinite(folderId) || folderId <= 0 || !folderName) {
              return null
            }
            if (folder?.deleted) {
              return null
            }
            return {
              id: Math.trunc(folderId),
              name: folderName
            }
          })
          .filter((folder): folder is CharacterFolderOption => folder !== null)
          .sort((left, right) => left.name.localeCompare(right.name))
      },
      staleTime: 60 * 1000,
      throwOnError: false
    })

  const characterFolderOptionsById = React.useMemo(
    () =>
      new Map(
        characterFolderOptions.map((folder) => [String(folder.id), folder.name])
      ),
    [characterFolderOptions]
  )

  const selectedFolderFilterLabel = React.useMemo(
    () =>
      folderFilterId
        ? characterFolderOptionsById.get(folderFilterId) || folderFilterId
        : undefined,
    [characterFolderOptionsById, folderFilterId]
  )

  return {
    // Query state
    characterListResponse,
    status,
    error,
    refetch,
    data,
    rawData,
    totalCharacters,
    pagedGalleryData,
    isLegacyCharacterListResponse,

    // Default character
    defaultCharacterPreference,
    serverDefaultCharacterId,
    effectiveDefaultCharacterId,

    // Conversation counts
    conversationCounts,
    characterIds,

    // World books
    previewCharacterWorldBooks,
    previewCharacterWorldBooksLoading,
    characterWorldBookData,
    worldBookOptionsLoading,
    worldBookOptions,

    // Tags & filters
    tagUsageData,
    allTags,
    popularTags,
    tagOptionsWithCounts,
    tagFilterOptions,
    creatorFilterOptions,

    // Folder options
    characterFolderOptions,
    characterFolderOptionsLoading,
    characterFolderOptionsById,
    selectedFolderFilterLabel,

    // Cross-navigation
    crossNavigationContext,
    previewCharacterId,

    // Server query params (needed by CRUD for world book sync)
    serverSortBy,
    serverSortOrder
  }
}
