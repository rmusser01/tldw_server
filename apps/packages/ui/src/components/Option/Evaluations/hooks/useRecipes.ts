/**
 * useRecipes hook
 * Handles recipe registry, validation, launch, and report queries.
 */

import { useMutation, useQuery } from "@tanstack/react-query"
import { useTranslation } from "react-i18next"
import { useAntdNotification } from "@/hooks/useAntdNotification"
import type { ApiSendResponse } from "@/services/api-send"
import {
  createRecipeRun,
  getEmbeddingRecipeCandidates,
  getRecipeLaunchReadiness,
  getRecipeRunReport,
  listRecipeManifests,
  previewRecipeRecommendationApply,
  validateRecipeDataset,
  type DatasetSample
} from "@/services/evaluations"

const ensureOk = <T,>(resp: ApiSendResponse<T>): ApiSendResponse<T> => {
  if (!resp?.ok) {
    const err = new Error(resp?.error || `HTTP ${resp?.status}`)
    ;(err as any).resp = resp
    throw err
  }
  return resp
}

export const getRecipeRunUserErrorMessage = (error: unknown): string => {
  const rawMessage =
    error instanceof Error
      ? error.message
      : typeof error === "string"
        ? error
        : (error as any)?.resp?.error || (error as any)?.message || ""

  if (String(rawMessage).includes("recipe_run_worker_disabled")) {
    return "Recipe runs are unavailable because the recipe worker is not running on this server. Enable the evaluations recipe worker and try again."
  }
  if (String(rawMessage).includes("recipe_run_enqueue_failed")) {
    return "The recipe run could not be queued on this server. Try again."
  }

  return rawMessage || "Failed to start recipe run."
}

export function useRecipeManifests() {
  return useQuery({
    queryKey: ["evaluations", "recipes", "manifests"],
    queryFn: async () => ensureOk(await listRecipeManifests())
  })
}

export function useValidateRecipeDataset() {
  const { t } = useTranslation(["evaluations", "common"])
  const notification = useAntdNotification()

  return useMutation({
    mutationFn: async (params: {
      recipeId: string
      datasetId?: string
      dataset?: DatasetSample[]
      runConfig?: Record<string, any>
    }) =>
      ensureOk(
        await validateRecipeDataset(params.recipeId, {
          dataset_id: params.datasetId,
          dataset: params.dataset,
          run_config: params.runConfig
        })
      ),
    onError: (error: any) => {
      notification.error({
        message: t("evaluations:recipeValidateErrorTitle", {
          defaultValue: "Failed to validate dataset"
        }),
        description: error?.message
      })
    }
  })
}

export function useRecipeLaunchReadiness(recipeId: string | null) {
  return useQuery({
    queryKey: ["evaluations", "recipes", "launch-readiness", recipeId],
    queryFn: async () => ensureOk(await getRecipeLaunchReadiness(recipeId as string)),
    enabled: !!recipeId
  })
}

export function useCreateRecipeRun() {
  const { t } = useTranslation(["evaluations", "common"])
  const notification = useAntdNotification()

  return useMutation({
    mutationFn: async (params: {
      recipeId: string
      datasetId?: string
      dataset?: DatasetSample[]
      runConfig: Record<string, any>
      forceRerun?: boolean
    }) =>
      ensureOk(
        await createRecipeRun(params.recipeId, {
          dataset_id: params.datasetId,
          dataset: params.dataset,
          run_config: params.runConfig,
          force_rerun: params.forceRerun
        })
      ),
    onSuccess: (resp: any) => {
      const runStatus = String(resp?.data?.status || "").toLowerCase()
      notification.success({
        message: t(
          runStatus === "completed"
            ? "evaluations:recipeRunCreateReuseTitle"
            : "evaluations:recipeRunCreateSuccessTitle",
          {
            defaultValue:
              runStatus === "completed"
                ? "Reused matching recipe run"
                : "Recipe run started"
          }
        )
      })
    },
    onError: (error: any) => {
      notification.error({
        message: t("evaluations:recipeRunCreateErrorTitle", {
          defaultValue: "Failed to start recipe run"
        }),
        description: getRecipeRunUserErrorMessage(error)
      })
    }
  })
}

export function useRecipeRunReport(runId: string | null) {
  return useQuery({
    queryKey: ["evaluations", "recipes", "report", runId],
    queryFn: async () => ensureOk(await getRecipeRunReport(runId as string)),
    enabled: !!runId,
    refetchInterval: (query) => {
      const status = String((query.state.data as any)?.data?.run?.status || "").toLowerCase()
      return ["pending", "running"].includes(status) ? 3000 : false
    }
  })
}

export function useEmbeddingRecipeCandidates(enabled: boolean) {
  return useQuery({
    queryKey: ["evaluations", "recipes", "embeddings_model_selection", "candidates"],
    queryFn: getEmbeddingRecipeCandidates,
    enabled,
    staleTime: 60 * 1000
  })
}

export function usePreviewRecipeRecommendationApply() {
  return useMutation({
    mutationFn: (params: {
      runId: string
      slotName: string
      candidateRunId?: string | null
    }) =>
      previewRecipeRecommendationApply(params.runId, {
        slot_name: params.slotName,
        candidate_run_id: params.candidateRunId ?? null
      })
  })
}
