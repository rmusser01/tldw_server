/**
 * useSyntheticEval hook
 * Shared review-queue queries and mutations for synthetic eval drafts.
 */

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { useTranslation } from "react-i18next"
import { useAntdNotification } from "@/hooks/useAntdNotification"
import { useEvaluationsStore } from "@/store/evaluations"
import type { ApiSendResponse } from "@/services/api-send"
import {
  generateSyntheticEvalDrafts,
  listSyntheticEvalQueue,
  promoteSyntheticEvalSamples,
  reviewSyntheticEvalSample
} from "@/services/evaluations"

const ensureOk = <T,>(resp: ApiSendResponse<T>): ApiSendResponse<T> => {
  if (!resp?.ok) {
    const err = new Error(resp?.error || `HTTP ${resp?.status}`)
    ;(err as any).resp = resp
    throw err
  }
  return resp
}

export function useSyntheticEvalQueue(params: {
  recipeKind?: string | null
  reviewState?: string | null
  sourceKind?: string | null
  generationBatchId?: string | null
  limit?: number
  offset?: number
}) {
  return useQuery({
    queryKey: [
      "evaluations",
      "synthetic",
      "queue",
      params.recipeKind || null,
      params.reviewState || null,
      params.sourceKind || null,
      params.generationBatchId || null,
      params.limit || 50,
      params.offset || 0
    ],
    queryFn: async () =>
      ensureOk(
        await listSyntheticEvalQueue({
          recipe_kind: params.recipeKind || undefined,
          review_state: params.reviewState || undefined,
          source_kind: params.sourceKind || undefined,
          generation_batch_id: params.generationBatchId || undefined,
          limit: params.limit,
          offset: params.offset
        })
      )
  })
}

export function useGenerateSyntheticEvalDrafts() {
  const queryClient = useQueryClient()
  const { t } = useTranslation(["evaluations", "common"])
  const notification = useAntdNotification()
  const setSyntheticReviewRecipeKind = useEvaluationsStore(
    (s) => s.setSyntheticReviewRecipeKind
  )
  const setSyntheticReviewBatchId = useEvaluationsStore(
    (s) => s.setSyntheticReviewBatchId
  )
  const setSyntheticReviewSampleIds = useEvaluationsStore(
    (s) => s.setSyntheticReviewSampleIds
  )

  return useMutation({
    mutationFn: async (params: {
      recipe_kind: string
      corpus_scope?: Record<string, any> | string[]
      generation_metadata?: Record<string, any>
      context_snapshot_ref?: string
      retrieval_baseline_ref?: string
      reference_answer?: string
      real_examples?: Record<string, any>[]
      seed_examples?: Record<string, any>[]
      target_sample_count: number
    }) => ensureOk(await generateSyntheticEvalDrafts(params)),
    onSuccess: (response, variables) => {
      const sampleIds = Array.isArray(response.data?.samples)
        ? response.data.samples
            .map((sample) => sample.sample_id)
            .filter((sampleId): sampleId is string => Boolean(sampleId))
        : []
      setSyntheticReviewRecipeKind(variables.recipe_kind)
      setSyntheticReviewBatchId(response.data?.generation_batch_id || null)
      setSyntheticReviewSampleIds(sampleIds)
      notification.success({
        message: t("evaluations:syntheticGenerationSuccessTitle", {
          defaultValue: "Synthetic drafts created"
        })
      })
      void queryClient.invalidateQueries({
        queryKey: ["evaluations", "synthetic", "queue"]
      })
    },
    onError: (error: any) => {
      notification.error({
        message: t("evaluations:syntheticGenerationErrorTitle", {
          defaultValue: "Failed to generate synthetic drafts"
        }),
        description: error?.message
      })
    }
  })
}

export function useReviewSyntheticEvalSample() {
  const queryClient = useQueryClient()
  const { t } = useTranslation(["evaluations", "common"])
  const notification = useAntdNotification()

  return useMutation({
    mutationFn: async (params: {
      sampleId: string
      action: string
      reviewer_id?: string
      notes?: string
      action_payload?: Record<string, any>
      resulting_review_state?: string
    }) =>
      ensureOk(
        await reviewSyntheticEvalSample(params.sampleId, {
          action: params.action,
          reviewer_id: params.reviewer_id,
          notes: params.notes,
          action_payload: params.action_payload,
          resulting_review_state: params.resulting_review_state
        })
      ),
    onSuccess: () => {
      notification.success({
        message: t("evaluations:syntheticReviewActionSuccessTitle", {
          defaultValue: "Review updated"
        })
      })
      void queryClient.invalidateQueries({
        queryKey: ["evaluations", "synthetic", "queue"]
      })
    },
    onError: (error: any) => {
      notification.error({
        message: t("evaluations:syntheticReviewActionErrorTitle", {
          defaultValue: "Failed to update review"
        }),
        description: error?.message
      })
    }
  })
}

export function usePromoteSyntheticEvalSamples() {
  const queryClient = useQueryClient()
  const { t } = useTranslation(["evaluations", "common"])
  const notification = useAntdNotification()

  return useMutation({
    mutationFn: async (params: {
      sample_ids: string[]
      dataset_name: string
      dataset_description?: string
      dataset_metadata?: Record<string, any>
      promoted_by?: string
      promotion_reason?: string
    }) =>
      ensureOk(
        await promoteSyntheticEvalSamples(params)
      ),
    onSuccess: () => {
      notification.success({
        message: t("evaluations:syntheticPromoteSuccessTitle", {
          defaultValue: "Synthetic dataset promoted"
        })
      })
      void queryClient.invalidateQueries({
        queryKey: ["evaluations", "synthetic", "queue"]
      })
      void queryClient.invalidateQueries({
        queryKey: ["evaluations", "datasets"]
      })
    },
    onError: (error: any) => {
      notification.error({
        message: t("evaluations:syntheticPromoteErrorTitle", {
          defaultValue: "Failed to promote synthetic dataset"
        }),
        description: error?.message
      })
    }
  })
}
