import React from "react"
import { act, renderHook, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { beforeEach, describe, expect, it, vi } from "vitest"

import {
  getRecipeRunUserErrorMessage,
  useCreateRecipeRun,
  useEmbeddingRecipeCandidates,
  usePreviewRecipeRecommendationApply
} from "../useRecipes"
import {
  createRecipeRun,
  getEmbeddingRecipeCandidates,
  previewRecipeRecommendationApply
} from "@/services/evaluations"

const { successNotificationSpy, errorNotificationSpy } = vi.hoisted(() => ({
  successNotificationSpy: vi.fn(),
  errorNotificationSpy: vi.fn()
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?:
        | string
        | {
            defaultValue?: string
          }
    ) => {
      if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
      return defaultValueOrOptions?.defaultValue || key
    }
  })
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({
    success: successNotificationSpy,
    error: errorNotificationSpy
  })
}))

vi.mock("@/services/evaluations", async () => {
  const actual = await vi.importActual<typeof import("@/services/evaluations")>(
    "@/services/evaluations"
  )
  const noopAsync = vi.fn()
  return {
    ...actual,
    createRecipeRun: vi.fn(),
    getEmbeddingRecipeCandidates: vi.fn(),
    getRecipeLaunchReadiness: noopAsync,
    getRecipeRunReport: noopAsync,
    listRecipeManifests: noopAsync,
    previewRecipeRecommendationApply: vi.fn(),
    validateRecipeDataset: noopAsync
  }
})

const buildWrapper = (queryClient: QueryClient) => {
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  )
}

describe("useRecipes", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("maps worker-disabled failures to the explicit worker guidance", () => {
    expect(
      getRecipeRunUserErrorMessage(new Error("recipe_run_worker_disabled"))
    ).toBe(
      "Recipe runs are unavailable because the recipe worker is not running on this server. Enable the evaluations recipe worker and try again."
    )
  })

  it("keeps generic enqueue failures distinct from worker-disabled guidance", () => {
    expect(
      getRecipeRunUserErrorMessage(new Error("recipe_run_enqueue_failed"))
    ).toBe("The recipe run could not be queued on this server. Try again.")
  })

  it("announces when a matching completed recipe run was reused", async () => {
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false }
      }
    })

    vi.mocked(createRecipeRun).mockResolvedValue({
      ok: true,
      data: {
        run_id: "recipe-run-1",
        recipe_id: "summarization_quality",
        recipe_version: "1",
        status: "completed",
        created_at: "2026-03-31T08:00:00Z"
      }
    } as any)

    const { result } = renderHook(() => useCreateRecipeRun(), {
      wrapper: buildWrapper(queryClient)
    })

    await act(async () => {
      await result.current.mutateAsync({
        recipeId: "summarization_quality",
        runConfig: {}
      })
    })

    expect(successNotificationSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        message: "Reused matching recipe run"
      })
    )
  })

  it("keeps the started title for newly queued recipe runs", async () => {
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false }
      }
    })

    vi.mocked(createRecipeRun).mockResolvedValue({
      ok: true,
      data: {
        run_id: "recipe-run-2",
        recipe_id: "summarization_quality",
        recipe_version: "1",
        status: "pending",
        created_at: "2026-03-31T08:05:00Z"
      }
    } as any)

    const { result } = renderHook(() => useCreateRecipeRun(), {
      wrapper: buildWrapper(queryClient)
    })

    await act(async () => {
      await result.current.mutateAsync({
        recipeId: "summarization_quality",
        runConfig: {}
      })
    })

    expect(successNotificationSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        message: "Recipe run started"
      })
    )
  })

  it("loads embeddings recipe candidates with a stable query key", async () => {
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false }
      }
    })

    vi.mocked(getEmbeddingRecipeCandidates).mockResolvedValue({
      ok: true,
      data: {
        recipe_id: "embeddings_model_selection",
        current: {
          provider: "openai",
          model: "text-embedding-3-small",
          status: "ready",
          default: true,
          is_local: false
        },
        candidates: [],
        warnings: []
      }
    } as any)

    const { result } = renderHook(() => useEmbeddingRecipeCandidates(true), {
      wrapper: buildWrapper(queryClient)
    })

    await waitFor(() =>
      expect(result.current.data?.data?.current?.provider).toBe("openai")
    )

    expect(
      queryClient.getQueryData([
        "evaluations",
        "recipes",
        "embeddings_model_selection",
        "candidates"
      ])
    ).toEqual(result.current.data)
  })

  it("posts apply preview requests through the hook", async () => {
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false }
      }
    })

    vi.mocked(previewRecipeRecommendationApply).mockResolvedValue({
      ok: true,
      data: {
        run_id: "recipe-run-1",
        recipe_id: "embeddings_model_selection",
        slot_name: "best_overall",
        apply_eligible: true,
        apply_available: false,
        current: {},
        proposed: {},
        affected_config: {},
        copy_config: {},
        warnings: []
      }
    } as any)

    const { result } = renderHook(() => usePreviewRecipeRecommendationApply(), {
      wrapper: buildWrapper(queryClient)
    })

    await act(async () => {
      await result.current.mutateAsync({
        runId: "recipe-run-1",
        slotName: "best_overall"
      })
    })

    expect(previewRecipeRecommendationApply).toHaveBeenCalledWith("recipe-run-1", {
      slot_name: "best_overall",
      candidate_run_id: null
    })
  })
})
