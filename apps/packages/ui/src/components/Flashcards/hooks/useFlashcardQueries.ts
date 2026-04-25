import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import {
  listDecks,
  listFlashcards,
  listFlashcardTagSuggestions,
  listRecentFlashcardReviewSessions,
  endFlashcardReviewSession,
  createFlashcard,
  createFlashcardsBulk,
  updateFlashcardsBulk,
  createDeck,
  updateDeck,
  createFlashcardTemplate,
  deleteFlashcardTemplate,
  getFlashcardTemplate,
  updateFlashcard,
  deleteFlashcard,
  resetFlashcardScheduling,
  reviewFlashcard,
  getNextReviewCard,
  getFlashcardAssistant,
  listFlashcardTemplates,
  updateFlashcardTemplate,
  respondFlashcardAssistant,
  generateFlashcards,
  getFlashcard,
  importFlashcards,
  previewStructuredQaImport,
  importFlashcardsJson,
  importFlashcardsApkg,
  getFlashcardsAnalyticsSummary,
  exportFlashcards,
  exportFlashcardsFile,
  getFlashcardsImportLimits,
  type Deck,
  type DeckUpdate,
  type Flashcard,
  type FlashcardTemplateCreate,
  type FlashcardTemplateUpdate,
  type StudyAssistantContextResponse,
  type StudyAssistantRespondRequest,
  type FlashcardBulkUpdateItem,
  type FlashcardBulkUpdateResponse,
  type FlashcardCreate,
  type FlashcardUpdate
} from "@/services/flashcards"
import { useServerOnline } from "@/hooks/useServerOnline"
import { useServerCapabilities } from "@/hooks/useServerCapabilities"
import { isTutorialResidueCard } from "../utils/review-card-hygiene"

export type DueStatus = "new" | "learning" | "due" | "all"

export interface DueCounts {
  due: number
  new: number
  learning: number
  total: number
}

export interface UseFlashcardQueriesOptions {
  enabled?: boolean
  includeWorkspaceItems?: boolean
  workspaceId?: string | null
}

export interface UseFlashcardDeckRecentCardsQueryOptions extends UseFlashcardQueriesOptions {
}
export interface UseGlobalFlashcardTagSuggestionsQueryOptions {
  enabled?: boolean
  limit?: number
}

export interface UseRecentFlashcardReviewSessionsQueryOptions extends UseFlashcardQueriesOptions {
  limit?: number
  scopeKey?: string | null
  status?: string | null
}

const invalidateFlashcardsQueries = (qc: ReturnType<typeof useQueryClient>) =>
  qc.invalidateQueries({
    predicate: (query) =>
      Array.isArray(query.queryKey) &&
      typeof query.queryKey[0] === "string" &&
      query.queryKey[0].startsWith("flashcards:")
  })

const getListTotal = (res: { total?: number | null; count?: number }) => (res.total ?? res.count ?? 0)
const STUDY_ASSISTANT_ACTIONS = ["explain", "mnemonic", "follow_up", "fact_check", "freeform"] as const

const buildWorkspaceVisibilityParams = (options?: UseFlashcardQueriesOptions) => ({
  workspace_id: options?.workspaceId ?? undefined,
  include_workspace_items: options?.includeWorkspaceItems ?? false
})

const deckMatchesVisibility = (deck: Deck, options?: UseFlashcardQueriesOptions): boolean => {
  const rawOptions = options as UseFlashcardQueriesOptions & {
    workspace_id?: string | null
    include_workspace_items?: boolean | null
  } | undefined
  const workspaceId = (rawOptions?.workspaceId ?? rawOptions?.workspace_id)?.trim() || null
  const includeWorkspaceItems =
    rawOptions?.includeWorkspaceItems ?? rawOptions?.include_workspace_items ?? false
  const deckWorkspaceId = deck.workspace_id?.trim() || null
  const isWorkspaceOwned = deckWorkspaceId != null

  if (workspaceId != null) {
    if (includeWorkspaceItems) {
      return !isWorkspaceOwned || deckWorkspaceId === workspaceId
    }
    return deckWorkspaceId === workspaceId
  }

  if (includeWorkspaceItems) {
    return true
  }

  return !isWorkspaceOwned
}

async function fetchDueCounts(
  deckId?: number | null,
  options?: UseFlashcardQueriesOptions
): Promise<DueCounts> {
  const visibilityParams = buildWorkspaceVisibilityParams(options)
  const [due, newCards, learning] = await Promise.all([
    listFlashcards({
      deck_id: deckId ?? undefined,
      due_status: "due",
      limit: 1,
      offset: 0,
      ...visibilityParams
    }),
    listFlashcards({
      deck_id: deckId ?? undefined,
      due_status: "new",
      limit: 1,
      offset: 0,
      ...visibilityParams
    }),
    listFlashcards({
      deck_id: deckId ?? undefined,
      due_status: "learning",
      limit: 1,
      offset: 0,
      ...visibilityParams
    })
  ])

  const dueTotal = getListTotal(due)
  const newTotal = getListTotal(newCards)
  const learningTotal = getListTotal(learning)
  return {
    due: dueTotal,
    new: newTotal,
    learning: learningTotal,
    total: dueTotal + newTotal + learningTotal
  }
}

/**
 * Hook for fetching flashcard decks
 */
export function useDecksQuery(options?: UseFlashcardQueriesOptions) {
  const { flashcardsEnabled } = useFlashcardsEnabled()
  const visibilityParams = buildWorkspaceVisibilityParams(options)

  return useQuery({
    queryKey: ["flashcards:decks", visibilityParams],
    queryFn: () => listDecks(visibilityParams),
    enabled: options?.enabled ?? flashcardsEnabled
  })
}

/**
 * Hook for fetching next due card for review
 */
export function useReviewQuery(deckId: number | null | undefined, options?: UseFlashcardQueriesOptions) {
  const { flashcardsEnabled } = useFlashcardsEnabled()
  const visibilityParams = buildWorkspaceVisibilityParams(options)

  return useQuery({
    queryKey: ["flashcards:review:next", deckId, visibilityParams],
    queryFn: async (): Promise<Flashcard | null> => {
      const response = await getNextReviewCard(deckId ?? undefined, visibilityParams)
      return response.card ?? null
    },
    enabled: options?.enabled ?? flashcardsEnabled
  })
}

export function useFlashcardAssistantQuery(
  cardUuid: string | null | undefined,
  options?: UseFlashcardQueriesOptions
) {
  const { flashcardsEnabled } = useFlashcardsEnabled()

  return useQuery({
    queryKey: ["flashcards:assistant", cardUuid ?? null],
    queryFn: ({ signal }) => getFlashcardAssistant(cardUuid!, { signal }),
    enabled: (options?.enabled ?? flashcardsEnabled) && !!cardUuid
  })
}

/**
 * Hook for fetching recent cards for a deck reference view.
 */
export function useFlashcardDeckRecentCardsQuery(
  deckId: number | null | undefined,
  options?: UseFlashcardDeckRecentCardsQueryOptions
) {
  const { flashcardsEnabled } = useFlashcardsEnabled()
  const visibilityParams = buildWorkspaceVisibilityParams(options)
  const limit = options?.limit ?? 6

  return useQuery({
    queryKey: ["flashcards:deck:recent", deckId ?? null, limit, visibilityParams],
    queryFn: async (): Promise<Flashcard[]> => {
      if (deckId == null) {
        return []
      }
      const response = await listFlashcards({
        deck_id: deckId,
        due_status: "all",
        limit,
        offset: 0,
        order_by: "created_at",
        ...visibilityParams
      })
      return response.items || []
    },
    enabled: (options?.enabled ?? flashcardsEnabled) && !!deckId
  })
}

/**
 * Hook for searching cards in a deck reference view.
 */
export function useFlashcardDeckSearchQuery(
  params: {
    deckId: number | null | undefined
    query: string
    limit?: number
  },
  options?: UseFlashcardQueriesOptions
) {
  const { flashcardsEnabled } = useFlashcardsEnabled()
  const visibilityParams = buildWorkspaceVisibilityParams(options)
  const trimmedQuery = params.query.trim()
  const limit = params.limit ?? 20

  return useQuery({
    queryKey: [
      "flashcards:deck:search",
      params.deckId ?? null,
      trimmedQuery,
      limit,
      visibilityParams
    ],
    queryFn: async (): Promise<Flashcard[]> => {
      if (params.deckId == null || trimmedQuery.length === 0) {
        return []
      }
      const response = await listFlashcards({
        deck_id: params.deckId,
        q: trimmedQuery,
        due_status: "all",
        limit,
        offset: 0,
        order_by: "created_at",
        ...visibilityParams
      })
      return response.items || []
    },
    enabled: (options?.enabled ?? flashcardsEnabled) && !!params.deckId && trimmedQuery.length > 0
  })
}

export function useRecentFlashcardReviewSessionsQuery(
  params: {
    deckId?: number | null
    scopeKey?: string | null
    status?: string | null
    limit?: number
  } = {},
  options?: UseRecentFlashcardReviewSessionsQueryOptions
) {
  const { flashcardsEnabled } = useFlashcardsEnabled()
  const effectiveLimit = params.limit ?? options?.limit ?? 20

  return useQuery({
    queryKey: [
      "flashcards:review-sessions:recent",
      params.deckId ?? null,
      params.scopeKey ?? null,
      params.status ?? null,
      effectiveLimit
    ],
    queryFn: () =>
      listRecentFlashcardReviewSessions({
        deck_id: params.deckId ?? undefined,
        scope_key: params.scopeKey ?? undefined,
        status: params.status ?? undefined,
        limit: effectiveLimit
      }),
    enabled: options?.enabled ?? flashcardsEnabled,
    refetchOnWindowFocus: false
  })
}

export function useEndFlashcardReviewSessionMutation() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationKey: ["flashcards:review-sessions:end"],
    mutationFn: (reviewSessionId: number) => endFlashcardReviewSession(reviewSessionId),
    onSuccess: async () => {
      await invalidateFlashcardsQueries(queryClient)
    }
  })
}

/**
 * Hook for fetching a cram-mode queue (cards regardless of due state), optionally filtered by tag.
 */
export function useCramQueueQuery(
  deckId: number | null | undefined,
  tag?: string | null,
  options?: UseFlashcardQueriesOptions
) {
  const { flashcardsEnabled } = useFlashcardsEnabled()
  const MAX_QUEUE_SIZE = 1000
  const PAGE_SIZE = 200
  const visibilityParams = buildWorkspaceVisibilityParams(options)

  return useQuery({
    queryKey: ["flashcards:review:cram-queue", deckId ?? null, tag ?? null, visibilityParams],
    queryFn: async (): Promise<Flashcard[]> => {
      const queue: Flashcard[] = []
      let offset = 0

      while (queue.length < MAX_QUEUE_SIZE) {
        const res = await listFlashcards({
          deck_id: deckId ?? undefined,
          tag: tag || undefined,
          due_status: "all",
          order_by: "due_at",
          limit: PAGE_SIZE,
          offset,
          ...visibilityParams
        })
        const items = res.items || []
        if (items.length === 0) break
        queue.push(...items.filter((card) => !isTutorialResidueCard(card)))
        if (items.length < PAGE_SIZE) break
        offset += PAGE_SIZE
      }

      return queue.slice(0, MAX_QUEUE_SIZE)
    },
    enabled: options?.enabled ?? flashcardsEnabled
  })
}

/**
 * Hook for fetching flashcard list with filters
 */
export interface ManageQueryParams {
  deckId?: number | null
  query?: string
  tag?: string
  tags?: string[]
  dueStatus?: DueStatus
  sortBy?: ManageSortBy
  page?: number
  pageSize?: number
}

export type ManageSortBy =
  | "due"
  | "created"
  | "ease"
  | "last_reviewed"
  | "front_alpha"

const parseTimestamp = (value?: string | null): number => {
  if (!value) return Number.POSITIVE_INFINITY
  const parsed = new Date(value).getTime()
  if (Number.isNaN(parsed)) return Number.POSITIVE_INFINITY
  return parsed
}

export const getManageServerOrderBy = (
  sortBy: ManageSortBy
): "due_at" | "created_at" => (sortBy === "created" ? "created_at" : "due_at")

export const applyManageClientSort = (
  items: Flashcard[],
  sortBy: ManageSortBy
): Flashcard[] => {
  const next = [...items]
  switch (sortBy) {
    case "created":
      return next.sort((a, b) => parseTimestamp(a.created_at) - parseTimestamp(b.created_at))
    case "ease":
      return next.sort((a, b) => a.ef - b.ef)
    case "last_reviewed":
      return next.sort((a, b) => {
        const left = a.last_reviewed_at
          ? new Date(a.last_reviewed_at).getTime()
          : null
        const right = b.last_reviewed_at
          ? new Date(b.last_reviewed_at).getTime()
          : null
        if (left === null && right === null) return 0
        if (left === null) return 1
        if (right === null) return -1
        if (left === right) return 0
        return right - left
      })
    case "front_alpha":
      return next.sort((a, b) =>
        (a.front || "").localeCompare(b.front || "", undefined, {
          sensitivity: "base"
        })
      )
    case "due":
    default:
      return next.sort((a, b) => parseTimestamp(a.due_at) - parseTimestamp(b.due_at))
  }
}

export const normalizeManageTags = (
  tags?: string[] | null,
  singleTag?: string | null
): string[] => {
  const seen = new Set<string>()
  const normalized: string[] = []
  const input = [...(tags || []), singleTag || ""]
  for (const raw of input) {
    const tag = String(raw || "").trim().toLowerCase()
    if (!tag || seen.has(tag)) continue
    seen.add(tag)
    normalized.push(tag)
  }
  return normalized
}

export const cardHasAllTags = (card: Flashcard, normalizedTags: string[]): boolean => {
  if (normalizedTags.length === 0) return true
  const cardTags = new Set((card.tags || []).map((tag) => String(tag || "").trim().toLowerCase()))
  return normalizedTags.every((tag) => cardTags.has(tag))
}

export function useManageQuery(params: ManageQueryParams, options?: UseFlashcardQueriesOptions) {
  const { flashcardsEnabled } = useFlashcardsEnabled()
  const visibilityParams = buildWorkspaceVisibilityParams(options)

  const {
    deckId,
    query,
    tag,
    tags,
    dueStatus = "all",
    sortBy = "due",
    page = 1,
    pageSize = 20
  } = params
  const normalizedTags = normalizeManageTags(tags, tag)
  const primaryTag = normalizedTags[0]

  return useQuery({
    queryKey: [
      "flashcards:list",
      deckId,
      query,
      normalizedTags.join("|"),
      dueStatus,
      sortBy,
      page,
      pageSize,
      visibilityParams
    ],
    queryFn: async () => {
      if (normalizedTags.length > 1) {
        const bulk: Flashcard[] = []
        const PAGE_SCAN_SIZE = 500
        const MAX_SCAN = 10000
        let offset = 0

        while (offset < MAX_SCAN) {
          const chunk = await listFlashcards({
            deck_id: deckId ?? undefined,
            q: query || undefined,
            tag: primaryTag,
            due_status: dueStatus,
            limit: PAGE_SCAN_SIZE,
            offset,
            order_by: getManageServerOrderBy(sortBy),
            ...visibilityParams
          })
          const items = chunk.items || []
          if (items.length === 0) break
          bulk.push(...items.filter((card) => cardHasAllTags(card, normalizedTags)))
          if (items.length < PAGE_SCAN_SIZE) break
          offset += PAGE_SCAN_SIZE
        }

        const sorted = applyManageClientSort(bulk, sortBy)
        const start = (page - 1) * pageSize
        const pageItems = sorted.slice(start, start + pageSize)
        return {
          items: pageItems,
          count: pageItems.length,
          total: sorted.length
        }
      }

      const response = await listFlashcards({
        deck_id: deckId ?? undefined,
        q: query || undefined,
        tag: primaryTag,
        due_status: dueStatus,
        limit: pageSize,
        offset: (page - 1) * pageSize,
        order_by: getManageServerOrderBy(sortBy),
        ...visibilityParams
      })
      return {
        ...response,
        items: applyManageClientSort(response.items || [], sortBy)
      }
    },
    enabled: options?.enabled ?? flashcardsEnabled
  })
}

/**
 * Hook for fetching tag suggestions for autocomplete/multi-tag filter chips.
 */
export function useTagSuggestionsQuery(
  deckId?: number | null,
  options?: UseFlashcardQueriesOptions
) {
  const { flashcardsEnabled } = useFlashcardsEnabled()
  const visibilityParams = buildWorkspaceVisibilityParams(options)

  return useQuery({
    queryKey: ["flashcards:tags:suggestions", deckId ?? null, visibilityParams],
    queryFn: async () => {
      const PAGE_SCAN_SIZE = 500
      const MAX_SCAN = 10000
      const tagSet = new Set<string>()
      let offset = 0

      while (offset < MAX_SCAN) {
        const response = await listFlashcards({
          deck_id: deckId ?? undefined,
          due_status: "all",
          limit: PAGE_SCAN_SIZE,
          offset,
          order_by: "created_at",
          ...visibilityParams
        })
        const items = response.items || []
        if (items.length === 0) break
        for (const card of items) {
          for (const rawTag of card.tags || []) {
            const tag = String(rawTag || "").trim()
            if (!tag) continue
            tagSet.add(tag)
          }
        }
        if (items.length < PAGE_SCAN_SIZE) break
        offset += PAGE_SCAN_SIZE
      }

      return Array.from(tagSet).sort((left, right) =>
        left.localeCompare(right, undefined, { sensitivity: "base" })
      )
    },
    enabled: options?.enabled ?? flashcardsEnabled
  })
}

/**
 * Hook for fetching global flashcard tag suggestions for create/edit tag autocompletion.
 */
export function useGlobalFlashcardTagSuggestionsQuery(
  query: string | null | undefined,
  options?: UseGlobalFlashcardTagSuggestionsQueryOptions
) {
  const { flashcardsEnabled } = useFlashcardsEnabled()
  const limit = options?.limit ?? 50
  const normalizedQuery = query?.trim() || undefined

  return useQuery({
    queryKey: ["flashcards:tags:suggestions:global", normalizedQuery ?? null, limit],
    queryFn: ({ signal }) =>
      listFlashcardTagSuggestions({
        q: normalizedQuery,
        limit,
        signal
      }),
    enabled: options?.enabled ?? flashcardsEnabled
  })
}

/**
 * Hook for fetching import limits
 */
export function useImportLimitsQuery(options?: UseFlashcardQueriesOptions) {
  const { flashcardsEnabled } = useFlashcardsEnabled()

  return useQuery({
    queryKey: ["flashcards:import:limits"],
    queryFn: getFlashcardsImportLimits,
    enabled: options?.enabled ?? flashcardsEnabled
  })
}

/**
 * Hook for fetching flashcard analytics summary
 */
export function useReviewAnalyticsSummaryQuery(
  deckId?: number | null,
  options?: UseFlashcardQueriesOptions
) {
  const { flashcardsEnabled } = useFlashcardsEnabled()
  const visibilityParams = buildWorkspaceVisibilityParams(options)

  return useQuery({
    queryKey: ["flashcards:analytics:summary", deckId ?? null, visibilityParams],
    queryFn: ({ signal }) =>
      getFlashcardsAnalyticsSummary({
        deck_id: deckId ?? undefined,
        ...visibilityParams,
        signal
      }),
    enabled: options?.enabled ?? flashcardsEnabled
  })
}

/**
 * Hook for creating a flashcard
 */
export function useCreateFlashcardMutation() {
  const qc = useQueryClient()

  return useMutation({
    mutationKey: ["flashcards:create"],
    mutationFn: (payload: FlashcardCreate) => createFlashcard(payload),
    onSuccess: () => {
      invalidateFlashcardsQueries(qc)
    },
    onError: (error) => {
      console.error("Failed to create flashcard:", error)
    }
  })
}

/**
 * Hook for creating multiple flashcards in a single batch.
 */
export function useCreateFlashcardsBulkMutation() {
  const qc = useQueryClient()

  return useMutation({
    mutationKey: ["flashcards:create:bulk"],
    mutationFn: (payload: FlashcardCreate[]) => createFlashcardsBulk(payload),
    onSuccess: () => {
      invalidateFlashcardsQueries(qc)
    },
    onError: (error) => {
      console.error("Failed to bulk create flashcards:", error)
    }
  })
}

/**
 * Hook for creating a deck
 */
export function useCreateDeckMutation() {
  const qc = useQueryClient()

  return useMutation({
    mutationKey: ["flashcards:deck:create"],
    mutationFn: (params: {
      name: string
      description?: string
      review_prompt_side?: Deck["review_prompt_side"]
      scheduler_type?: Deck["scheduler_type"]
      scheduler_settings?: Deck["scheduler_settings"]
    }) =>
      createDeck({
        name: params.name.trim(),
        description: params.description?.trim() || undefined,
        review_prompt_side: params.review_prompt_side,
        scheduler_type: params.scheduler_type,
        scheduler_settings: params.scheduler_settings
      }),
    onSuccess: () => {
      invalidateFlashcardsQueries(qc)
    },
    onError: (error) => {
      console.error("Failed to create deck:", error)
    }
  })
}

/**
 * Hook for updating a deck, including scheduler settings.
 */
export function useUpdateDeckMutation() {
  const qc = useQueryClient()

  return useMutation({
    mutationKey: ["flashcards:deck:update"],
    mutationFn: (params: { deckId: number; update: DeckUpdate }) =>
      updateDeck(params.deckId, params.update),
    onSuccess: (deck) => {
      qc.getQueriesData<Deck[]>({ queryKey: ["flashcards:decks"] }).forEach(([queryKey, current]) => {
        if (!current) return
        const params = queryKey[1] as UseFlashcardQueriesOptions | undefined
        const nextItems = current
          .map((item) => (item.id === deck.id ? deck : item))
          .filter((item) => deckMatchesVisibility(item, params))

        qc.setQueryData(queryKey, nextItems)
      })
      invalidateFlashcardsQueries(qc)
    },
    onError: (error) => {
      console.error("Failed to update flashcard deck:", error)
    }
  })
}

export function useFlashcardTemplatesQuery(options?: UseFlashcardQueriesOptions) {
  const { flashcardsEnabled } = useFlashcardsEnabled()

  return useQuery({
    queryKey: ["flashcards:templates"],
    queryFn: () => listFlashcardTemplates({}),
    enabled: options?.enabled ?? flashcardsEnabled
  })
}

export function useFlashcardTemplateQuery(
  templateId: number | null | undefined,
  options?: UseFlashcardQueriesOptions
) {
  const { flashcardsEnabled } = useFlashcardsEnabled()

  return useQuery({
    queryKey: ["flashcards:templates", templateId ?? null],
    queryFn: () => getFlashcardTemplate(templateId!),
    enabled: (options?.enabled ?? flashcardsEnabled) && templateId != null
  })
}

export function useCreateFlashcardTemplateMutation() {
  const qc = useQueryClient()

  return useMutation({
    mutationKey: ["flashcards:template:create"],
    mutationFn: (payload: FlashcardTemplateCreate) => createFlashcardTemplate(payload),
    onSuccess: async (template) => {
      qc.setQueryData(["flashcards:templates", template.id], template)
      await qc.invalidateQueries({ queryKey: ["flashcards:templates"] })
    },
    onError: (error) => {
      console.error("Failed to create flashcard template:", error)
    }
  })
}

export function useUpdateFlashcardTemplateMutation() {
  const qc = useQueryClient()

  return useMutation({
    mutationKey: ["flashcards:template:update"],
    mutationFn: (params: { templateId: number; update: FlashcardTemplateUpdate }) =>
      updateFlashcardTemplate(params.templateId, params.update),
    onSuccess: async (template) => {
      qc.setQueryData(["flashcards:templates", template.id], template)
      await qc.invalidateQueries({ queryKey: ["flashcards:templates"] })
    },
    onError: (error) => {
      console.error("Failed to update flashcard template:", error)
    }
  })
}

export function useDeleteFlashcardTemplateMutation() {
  const qc = useQueryClient()

  return useMutation({
    mutationKey: ["flashcards:template:delete"],
    mutationFn: (params: { templateId: number; expectedVersion: number }) =>
      deleteFlashcardTemplate(params.templateId, params.expectedVersion),
    onSuccess: async (_result, variables) => {
      qc.removeQueries({ queryKey: ["flashcards:templates", variables.templateId] })
      await qc.invalidateQueries({ queryKey: ["flashcards:templates"] })
    },
    onError: (error) => {
      console.error("Failed to delete flashcard template:", error)
    }
  })
}

/**
 * Hook for updating a flashcard
 */
export function useUpdateFlashcardMutation() {
  const qc = useQueryClient()

  return useMutation({
    mutationKey: ["flashcards:update"],
    mutationFn: (params: { uuid: string; update: FlashcardUpdate }) =>
      updateFlashcard(params.uuid, params.update),
    onSuccess: () => {
      invalidateFlashcardsQueries(qc)
    },
    onError: (error) => {
      console.error("Failed to update flashcard:", error)
    }
  })
}

/**
 * Hook for updating multiple flashcards in one request without automatic global invalidation.
 */
export function useUpdateFlashcardsBulkMutation() {
  return useMutation<FlashcardBulkUpdateResponse, Error, FlashcardBulkUpdateItem[]>({
    mutationKey: ["flashcards:update:bulk"],
    mutationFn: (payload) => updateFlashcardsBulk(payload),
    onError: (error) => {
      console.error("Failed to bulk update flashcards:", error)
    }
  })
}

/**
 * Hook for deleting a flashcard
 */
export function useDeleteFlashcardMutation() {
  const qc = useQueryClient()

  return useMutation({
    mutationKey: ["flashcards:delete"],
    mutationFn: (params: { uuid: string; version: number }) =>
      deleteFlashcard(params.uuid, params.version),
    onSuccess: () => {
      invalidateFlashcardsQueries(qc)
    },
    onError: (error) => {
      console.error("Failed to delete flashcard:", error)
    }
  })
}

/**
 * Hook for resetting flashcard scheduling metadata back to new-card defaults
 */
export function useResetFlashcardSchedulingMutation() {
  const qc = useQueryClient()

  return useMutation({
    mutationKey: ["flashcards:reset-scheduling"],
    mutationFn: (params: { uuid: string; expectedVersion: number }) =>
      resetFlashcardScheduling(params.uuid, {
        expected_version: params.expectedVersion
      }),
    onSuccess: () => {
      invalidateFlashcardsQueries(qc)
    },
    onError: (error) => {
      console.error("Failed to reset flashcard scheduling:", error)
    }
  })
}

/**
 * Hook for submitting a review
 */
export function useReviewFlashcardMutation() {
  const qc = useQueryClient()

  return useMutation({
    mutationKey: ["flashcards:review"],
    mutationFn: (params: { cardUuid: string; rating: number; answerTimeMs?: number }) =>
      reviewFlashcard({
        card_uuid: params.cardUuid,
        rating: params.rating,
        answer_time_ms: params.answerTimeMs
      }),
    onSuccess: () => {
      invalidateFlashcardsQueries(qc)
    },
    onError: (error) => {
      console.error("Failed to submit flashcard review:", error)
    }
  })
}

export function useFlashcardAssistantRespondMutation() {
  const qc = useQueryClient()

  return useMutation({
    mutationKey: ["flashcards:assistant:respond"],
    mutationFn: (params: {
      cardUuid: string
      request: StudyAssistantRespondRequest
      signal?: AbortSignal
    }) => {
      const cached = qc.getQueryData<StudyAssistantContextResponse>([
        "flashcards:assistant",
        params.cardUuid
      ])
      const request = params.request.expected_thread_version != null
        ? params.request
        : cached?.thread?.version != null
          ? {
              ...params.request,
              expected_thread_version: cached.thread.version
            }
          : params.request

      return respondFlashcardAssistant(
        params.cardUuid,
        request,
        params.signal ? { signal: params.signal } : undefined
      )
    },
    onSuccess: (response, variables) => {
      qc.setQueryData<StudyAssistantContextResponse>(
        ["flashcards:assistant", variables.cardUuid],
        (current) => ({
          thread: response.thread,
          messages: current
            ? [...current.messages, response.user_message, response.assistant_message]
            : [response.user_message, response.assistant_message],
          context_snapshot: response.context_snapshot,
          available_actions: current?.available_actions ?? [...STUDY_ASSISTANT_ACTIONS]
        })
      )
    },
    onError: (error) => {
      console.error("Failed to respond with flashcard assistant:", error)
    }
  })
}

/**
 * Hook for generating flashcards from free text via LLM adapter.
 */
export function useGenerateFlashcardsMutation() {
  return useMutation({
    mutationKey: ["flashcards:generate"],
    mutationFn: (params: {
      text: string
      numCards?: number
      cardType?: "basic" | "basic_reverse" | "cloze"
      difficulty?: "easy" | "medium" | "hard" | "mixed"
      focusTopics?: string[]
      provider?: string
      model?: string
    }) =>
      generateFlashcards({
        text: params.text,
        num_cards: params.numCards,
        card_type: params.cardType,
        difficulty: params.difficulty,
        focus_topics: params.focusTopics,
        provider: params.provider,
        model: params.model
      }),
    onError: (error) => {
      console.error("Failed to generate flashcards:", error)
    }
  })
}

/**
 * Hook for importing flashcards
 */
export function useImportFlashcardsMutation() {
  const qc = useQueryClient()

  return useMutation({
    mutationKey: ["flashcards:import"],
    mutationFn: (params: { content: string; delimiter: string; hasHeader: boolean }) =>
      importFlashcards({
        content: params.content,
        delimiter: params.delimiter,
        has_header: params.hasHeader
      }),
    onSuccess: () => {
      invalidateFlashcardsQueries(qc)
    },
    onError: (error) => {
      console.error("Failed to import flashcards:", error)
    }
  })
}

/**
 * Hook for previewing deterministic structured Q&A imports without saving.
 */
export function usePreviewStructuredQaImportMutation() {
  return useMutation({
    mutationKey: ["flashcards:import:structured:preview"],
    mutationFn: (params: {
      content: string
      maxLines?: number
      maxLineLength?: number
      maxFieldLength?: number
    }) =>
      previewStructuredQaImport(
        {
          content: params.content
        },
        {
          max_lines: params.maxLines,
          max_line_length: params.maxLineLength,
          max_field_length: params.maxFieldLength
        }
      ),
    onError: (error) => {
      console.error("Failed to preview structured Q&A import:", error)
    }
  })
}

/**
 * Hook for importing flashcards from JSON/JSONL content via upload endpoint.
 */
export function useImportFlashcardsJsonMutation() {
  const qc = useQueryClient()

  return useMutation({
    mutationKey: ["flashcards:import-json"],
    mutationFn: (params: { content: string; filename?: string }) =>
      importFlashcardsJson({
        content: params.content,
        filename: params.filename
      }),
    onSuccess: () => {
      invalidateFlashcardsQueries(qc)
    },
    onError: (error) => {
      console.error("Failed to import JSON flashcards:", error)
    }
  })
}

/**
 * Hook for importing flashcards from APKG upload endpoint.
 */
export function useImportFlashcardsApkgMutation() {
  const qc = useQueryClient()

  return useMutation({
    mutationKey: ["flashcards:import-apkg"],
    mutationFn: (params: { bytes: Uint8Array; filename?: string }) =>
      importFlashcardsApkg({
        bytes: params.bytes,
        filename: params.filename
      }),
    onSuccess: () => {
      invalidateFlashcardsQueries(qc)
    },
    onError: (error) => {
      console.error("Failed to import APKG flashcards:", error)
    }
  })
}

/**
 * Helper to check if flashcards feature is available
 */
export function useFlashcardsEnabled() {
  const isOnline = useServerOnline()
  const { capabilities, loading: capsLoading } = useServerCapabilities()
  const flashcardsUnsupported = !capsLoading && !!capabilities && !capabilities.hasFlashcards

  return {
    isOnline,
    capsLoading,
    flashcardsUnsupported,
    flashcardsEnabled: isOnline && !flashcardsUnsupported
  }
}

/**
 * Hook for fetching due counts across statuses
 */
export function useDueCountsQuery(deckId?: number | null, options?: UseFlashcardQueriesOptions) {
  const { flashcardsEnabled } = useFlashcardsEnabled()
  const visibilityParams = buildWorkspaceVisibilityParams(options)

  return useQuery({
    queryKey: ["flashcards:due-counts", deckId, visibilityParams],
    queryFn: () => fetchDueCounts(deckId, options),
    enabled: options?.enabled ?? flashcardsEnabled
  })
}

/**
 * Hook for fetching due counts for all decks (for selector labels and overviews)
 */
export function useDeckDueCountsQuery(options?: UseFlashcardQueriesOptions) {
  const { flashcardsEnabled } = useFlashcardsEnabled()
  const visibilityParams = buildWorkspaceVisibilityParams(options)

  return useQuery({
    queryKey: ["flashcards:due-counts:by-deck", visibilityParams],
    queryFn: async () => {
      const decks = await listDecks(visibilityParams)
      const entries = await Promise.all(
        decks.map(async (deck) => [deck.id, await fetchDueCounts(deck.id, options)] as const)
      )
      return Object.fromEntries(entries) as Record<number, DueCounts>
    },
    enabled: options?.enabled ?? flashcardsEnabled
  })
}

/**
 * Hook to check if user has any flashcards
 */
export function useHasCardsQuery(options?: UseFlashcardQueriesOptions) {
  const { flashcardsEnabled } = useFlashcardsEnabled()
  const visibilityParams = buildWorkspaceVisibilityParams(options)

  return useQuery({
    queryKey: ["flashcards:has-cards", visibilityParams],
    queryFn: async () => {
      const res = await listFlashcards({
        limit: 1,
        offset: 0,
        ...visibilityParams
      })
      return (res.total ?? res.count ?? 0) > 0
    },
    enabled: options?.enabled ?? flashcardsEnabled
  })
}

/**
 * Hook to get the next due card info (for showing when the next review is due)
 */
export function useNextDueQuery(deckId?: number | null, options?: UseFlashcardQueriesOptions) {
  const { flashcardsEnabled } = useFlashcardsEnabled()
  const visibilityParams = buildWorkspaceVisibilityParams(options)

  return useQuery({
    queryKey: ["flashcards:next-due", deckId, visibilityParams],
    queryFn: async () => {
      const PAGE_SIZE = 200
      const MAX_PAGES = 10
      const oneHour = 60 * 60 * 1000
      const nowMs = Date.now()

      let offset = 0
      let pagesChecked = 0
      let scanned = 0
      let nextDueAt: string | null = null
      let nextDueMs = 0
      let cardsDue = 0

      while (true) {
        const res = await listFlashcards({
          deck_id: deckId ?? undefined,
          due_status: "all",
          order_by: "due_at",
          limit: PAGE_SIZE,
          offset,
          ...visibilityParams
        })
        const items = res.items || []
        if (items.length === 0) break
        pagesChecked += 1
        scanned += items.length

        for (const card of items) {
          if (!card.due_at) continue
          const dueMs = new Date(card.due_at).getTime()
          if (Number.isNaN(dueMs)) continue

          if (!nextDueAt) {
            if (dueMs > nowMs) {
              nextDueAt = card.due_at
              nextDueMs = dueMs
              cardsDue = 1
            }
            continue
          }

          if (dueMs <= nextDueMs + oneHour) {
            cardsDue += 1
          } else {
            return { nextDueAt, cardsDue, isCapped: false, scanned }
          }
        }

        if (pagesChecked >= MAX_PAGES) {
          return {
            nextDueAt,
            cardsDue,
            isCapped: true,
            scanned
          }
        }

        if (items.length < PAGE_SIZE) break
        offset += PAGE_SIZE
      }

      if (!nextDueAt) return null

      return {
        nextDueAt,
        cardsDue,
        isCapped: false,
        scanned
      }
    },
    enabled: options?.enabled ?? flashcardsEnabled
  })
}

// Re-export service functions that are used directly
export { getFlashcard, exportFlashcards, exportFlashcardsFile }
