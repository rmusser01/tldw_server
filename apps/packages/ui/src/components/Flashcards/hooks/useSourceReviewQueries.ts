import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"

import {
  completeSourceReviewOccurrence,
  createSourceReviewPlan,
  deleteSourceReviewPlan,
  listDueSourceReviewOccurrences,
  listSourceReviewPlans,
  skipSourceReviewOccurrence,
  startSourceReviewOccurrence,
  type SourceReviewPlanCreateRequest
} from "@/services/flashcards"
import { useFlashcardsEnabled } from "./useFlashcardQueries"

const SOURCE_REVIEW_QUERY_PREFIX = "flashcards:source-review"

type SourceReviewQueryOptions = {
  limit?: number
  offset?: number
  enabled?: boolean
}

const invalidateSourceReviewQueries = async (
  queryClient: ReturnType<typeof useQueryClient>
) => {
  await queryClient.invalidateQueries({
    predicate: (query) =>
      Array.isArray(query.queryKey) &&
      typeof query.queryKey[0] === "string" &&
      query.queryKey[0].startsWith(SOURCE_REVIEW_QUERY_PREFIX)
  })
}

export function useSourceReviewPlansQuery(
  options: SourceReviewQueryOptions = {}
) {
  const { flashcardsEnabled } = useFlashcardsEnabled()
  const { limit, offset } = options
  return useQuery({
    queryKey: [`${SOURCE_REVIEW_QUERY_PREFIX}:plans`, limit, offset],
    queryFn: () => listSourceReviewPlans({ limit, offset }),
    enabled: (options.enabled ?? true) && flashcardsEnabled,
    retry: false
  })
}

export function useDueSourceReviewOccurrencesQuery(
  options: SourceReviewQueryOptions = {}
) {
  const { flashcardsEnabled } = useFlashcardsEnabled()
  const { limit, offset } = options
  const enabled = (options.enabled ?? true) && flashcardsEnabled
  return useQuery({
    queryKey: [`${SOURCE_REVIEW_QUERY_PREFIX}:due`, limit, offset],
    queryFn: () => listDueSourceReviewOccurrences({ limit, offset }),
    enabled,
    refetchInterval: enabled ? 60_000 : false,
    retry: false
  })
}

export function useCreateSourceReviewPlanMutation() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationKey: [`${SOURCE_REVIEW_QUERY_PREFIX}:create`],
    mutationFn: (request: SourceReviewPlanCreateRequest) =>
      createSourceReviewPlan(request),
    onSuccess: () => invalidateSourceReviewQueries(queryClient)
  })
}

const useOccurrenceMutation = (
  action: "start" | "complete" | "skip"
) => {
  const queryClient = useQueryClient()
  const mutation =
    action === "start"
      ? startSourceReviewOccurrence
      : action === "complete"
        ? completeSourceReviewOccurrence
        : skipSourceReviewOccurrence
  return useMutation({
    mutationKey: [`${SOURCE_REVIEW_QUERY_PREFIX}:${action}`],
    mutationFn: (occurrenceId: number) => mutation(occurrenceId),
    onSuccess: () => invalidateSourceReviewQueries(queryClient)
  })
}

export const useStartSourceReviewOccurrenceMutation = () =>
  useOccurrenceMutation("start")

export const useCompleteSourceReviewOccurrenceMutation = () =>
  useOccurrenceMutation("complete")

export const useSkipSourceReviewOccurrenceMutation = () =>
  useOccurrenceMutation("skip")

export function useDeleteSourceReviewPlanMutation() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationKey: [`${SOURCE_REVIEW_QUERY_PREFIX}:delete`],
    mutationFn: (planId: number) => deleteSourceReviewPlan(planId),
    onSuccess: () => invalidateSourceReviewQueries(queryClient)
  })
}
