import { useInfiniteQuery } from "@tanstack/react-query"

import {
  listFlashcards,
  type Flashcard,
  type FlashcardListResponse
} from "@/services/flashcards"
import {
  applyManageClientSort,
  cardHasAllTags,
  getManageServerOrderBy,
  normalizeManageQuery,
  normalizeManageTags,
  type DueStatus,
  type UseFlashcardQueriesOptions,
  useFlashcardsEnabled
} from "./useFlashcardQueries"

export const DOCUMENT_VIEW_SUPPORTED_SORTS = ["due", "created"] as const

export type DocumentManageSortBy = (typeof DOCUMENT_VIEW_SUPPORTED_SORTS)[number]

export interface FlashcardDocumentQueryParams {
  deckId?: number | null
  query?: string
  tag?: string
  tags?: string[]
  dueStatus?: DueStatus
  sortBy?: DocumentManageSortBy
  pageSize?: number
  includeWorkspaceItems?: boolean
  workspaceId?: string | null
}

export interface FlashcardDocumentPage {
  items: Flashcard[]
  nextPageParam?: number
  isTruncated: boolean
  total: number
}

const DOCUMENT_PAGE_SIZE = 100
const DOCUMENT_SCAN_PAGE_SIZE = 500
const DOCUMENT_MAX_SCAN = 10000

export const getFlashcardDocumentQueryKey = (
  params: FlashcardDocumentQueryParams,
  resolved?: {
    normalizedTags?: string[]
    dueStatus?: DueStatus
    sortBy?: DocumentManageSortBy
    pageSize?: number
    includeWorkspaceItems?: boolean
    workspaceId?: string | null
  }
) => [
  "flashcards:document",
  params.deckId ?? null,
  normalizeManageQuery(params.query) ?? "",
  (resolved?.normalizedTags ?? normalizeManageTags(params.tags, params.tag)).join("|"),
  resolved?.dueStatus ?? params.dueStatus ?? "all",
  resolved?.sortBy ?? params.sortBy ?? "due",
  resolved?.pageSize ?? params.pageSize ?? DOCUMENT_PAGE_SIZE,
  resolved?.includeWorkspaceItems ?? params.includeWorkspaceItems ?? false,
  resolved?.workspaceId ?? params.workspaceId ?? null
] as const

const getListTotal = (response: FlashcardListResponse) =>
  Number(response.total ?? response.count ?? 0)

async function fetchSingleTagDocumentPage(
  params: Required<Pick<FlashcardDocumentQueryParams, "dueStatus" | "sortBy" | "pageSize">> &
    Pick<FlashcardDocumentQueryParams, "deckId" | "query"> & {
      primaryTag?: string
      includeWorkspaceItems?: boolean
      workspaceId?: string | null
    },
  pageIndex: number
): Promise<FlashcardDocumentPage> {
  const response = await listFlashcards({
    deck_id: params.deckId ?? undefined,
    q: normalizeManageQuery(params.query),
    tag: params.primaryTag || undefined,
    due_status: params.dueStatus,
    limit: params.pageSize,
    offset: pageIndex * params.pageSize,
    order_by: getManageServerOrderBy(params.sortBy),
    include_workspace_items: params.includeWorkspaceItems ?? false,
    workspace_id: params.workspaceId ?? undefined
  })
  const items = applyManageClientSort(response.items || [], params.sortBy)

  return {
    items,
    nextPageParam: items.length < params.pageSize ? undefined : pageIndex + 1,
    isTruncated: false,
    total: getListTotal(response)
  }
}

async function fetchMultiTagDocumentPage(
  params: Required<Pick<FlashcardDocumentQueryParams, "dueStatus" | "sortBy" | "pageSize">> &
    Pick<FlashcardDocumentQueryParams, "deckId" | "query"> & {
      normalizedTags: string[]
      primaryTag: string
      includeWorkspaceItems?: boolean
      workspaceId?: string | null
    },
  pageIndex: number
): Promise<FlashcardDocumentPage> {
  const targetCount = (pageIndex + 1) * params.pageSize
  const matched: Flashcard[] = []
  let offset = 0
  let total = 0
  let reachedEnd = false

  while (offset < DOCUMENT_MAX_SCAN && matched.length < targetCount) {
    const response = await listFlashcards({
      deck_id: params.deckId ?? undefined,
      q: normalizeManageQuery(params.query),
      tag: params.primaryTag,
      due_status: params.dueStatus,
      limit: DOCUMENT_SCAN_PAGE_SIZE,
      offset,
      order_by: getManageServerOrderBy(params.sortBy),
      include_workspace_items: params.includeWorkspaceItems ?? false,
      workspace_id: params.workspaceId ?? undefined
    })
    total = getListTotal(response)
    const items = response.items || []

    if (items.length === 0) {
      reachedEnd = true
      break
    }

    matched.push(...items.filter((card) => cardHasAllTags(card, params.normalizedTags)))

    if (items.length < DOCUMENT_SCAN_PAGE_SIZE) {
      reachedEnd = true
      break
    }

    offset += DOCUMENT_SCAN_PAGE_SIZE
  }

  const sorted = applyManageClientSort(matched, params.sortBy)
  const pageStart = pageIndex * params.pageSize
  const pageItems = sorted.slice(pageStart, pageStart + params.pageSize)
  const isTruncated = total > DOCUMENT_MAX_SCAN || offset >= DOCUMENT_MAX_SCAN
  const hasMoreMatchesLoaded = sorted.length > pageStart + params.pageSize

  return {
    items: pageItems,
    nextPageParam:
      pageItems.length < params.pageSize && !hasMoreMatchesLoaded && reachedEnd && !isTruncated
        ? undefined
        : pageItems.length > 0
          ? pageIndex + 1
          : undefined,
    isTruncated,
    total
  }
}

export function useFlashcardDocumentQuery(
  params: FlashcardDocumentQueryParams,
  options?: UseFlashcardQueriesOptions
) {
  const { flashcardsEnabled } = useFlashcardsEnabled()
  const normalizedTags = normalizeManageTags(params.tags, params.tag)
  const normalizedQuery = normalizeManageQuery(params.query)
  const primaryTag = normalizedTags[0]
  const dueStatus = params.dueStatus ?? "all"
  const sortBy = params.sortBy ?? "due"
  const pageSize = params.pageSize ?? DOCUMENT_PAGE_SIZE
  const includeWorkspaceItems = params.includeWorkspaceItems ?? false
  const workspaceId = params.workspaceId ?? null

  const query = useInfiniteQuery({
    queryKey: getFlashcardDocumentQueryKey(params, {
      normalizedTags,
      dueStatus,
      sortBy,
      pageSize,
      includeWorkspaceItems,
      workspaceId
    }),
    initialPageParam: 0,
    queryFn: async ({ pageParam }) => {
      const pageIndex =
        typeof pageParam === "number" && Number.isFinite(pageParam) ? pageParam : 0
      if (normalizedTags.length > 1 && primaryTag) {
        return fetchMultiTagDocumentPage(
          {
            deckId: params.deckId,
            query: normalizedQuery,
            dueStatus,
            sortBy,
            pageSize,
            normalizedTags,
            primaryTag,
            includeWorkspaceItems,
            workspaceId
          },
          pageIndex
        )
      }
      return fetchSingleTagDocumentPage(
        {
          deckId: params.deckId,
          query: normalizedQuery,
          dueStatus,
          sortBy,
          pageSize,
          primaryTag,
          includeWorkspaceItems,
          workspaceId
        },
        pageIndex
      )
    },
    getNextPageParam: (lastPage) => lastPage.nextPageParam,
    enabled: options?.enabled ?? flashcardsEnabled
  })

  const pages = query.data?.pages || []
  const items = pages.flatMap((page) => page.items)

  return {
    ...query,
    items,
    isTruncated: pages.some((page) => page.isTruncated),
    supportedSorts: [...DOCUMENT_VIEW_SUPPORTED_SORTS]
  }
}
