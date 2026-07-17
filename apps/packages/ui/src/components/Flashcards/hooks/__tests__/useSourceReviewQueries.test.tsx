import React from "react"
import { act, renderHook, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { beforeEach, describe, expect, it, vi } from "vitest"

import {
  useCompleteSourceReviewOccurrenceMutation,
  useCreateSourceReviewPlanMutation,
  useDeleteSourceReviewPlanMutation,
  useDueSourceReviewOccurrencesQuery,
  useSkipSourceReviewOccurrenceMutation,
  useSourceReviewPlansQuery,
  useStartSourceReviewOccurrenceMutation
} from "../useSourceReviewQueries"
import {
  completeSourceReviewOccurrence,
  createSourceReviewPlan,
  deleteSourceReviewPlan,
  listDueSourceReviewOccurrences,
  listSourceReviewPlans,
  skipSourceReviewOccurrence,
  startSourceReviewOccurrence,
  type SourceReviewOccurrenceActionResponse,
  type SourceReviewPlanResponse,
  type SourceReviewPlanCreateRequest
} from "@/services/flashcards"

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({
    capabilities: { hasFlashcards: true },
    loading: false
  })
}))

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => true
}))

vi.mock("@/services/flashcards", async () => {
  const actual = await vi.importActual<typeof import("@/services/flashcards")>(
    "@/services/flashcards"
  )
  return {
    ...actual,
    createSourceReviewPlan: vi.fn(),
    listSourceReviewPlans: vi.fn(),
    listDueSourceReviewOccurrences: vi.fn(),
    startSourceReviewOccurrence: vi.fn(),
    completeSourceReviewOccurrence: vi.fn(),
    skipSourceReviewOccurrence: vi.fn(),
    deleteSourceReviewPlan: vi.fn()
  }
})

const wrapper = (queryClient: QueryClient) =>
  function QueryWrapper({ children }: { children: React.ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        {children}
      </QueryClientProvider>
    )
  }

const request: SourceReviewPlanCreateRequest = {
  title: "Cardiac physiology",
  starts_on: "2026-07-09",
  timezone: "UTC",
  source_items: [
    {
      source_type: "note",
      source_id: "note-42",
      label: "Cardiac physiology"
    }
  ],
  schedule: [
    {
      offset_value: 1,
      offset_unit: "day",
      activity_type: "reread"
    }
  ]
}

const queryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false }
    }
  })

const occurrence: SourceReviewOccurrenceActionResponse = {
  id: 31,
  plan_id: 7,
  offset_value: 1,
  offset_unit: "day",
  activity_type: "reread",
  due_at: "2026-07-10T00:00:00Z",
  status: "pending",
  started_at: null,
  completed_at: null,
  completion_source: null,
  created_at: "2026-07-09T12:00:00Z",
  last_modified: "2026-07-09T12:00:00Z",
  client_id: "source-review-tests",
  version: 1,
  plan_title: "Cardiac physiology",
  launch_state: null
}

const plan: SourceReviewPlanResponse = {
  id: 7,
  title: "Cardiac physiology",
  starts_on: "2026-07-09",
  timezone: "UTC",
  source_bundle: { items: request.source_items },
  occurrences: [occurrence],
  created_at: "2026-07-09T12:00:00Z",
  last_modified: "2026-07-09T12:00:00Z",
  client_id: "source-review-tests",
  version: 1
}

describe("source review query hooks", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(listSourceReviewPlans).mockResolvedValue({ items: [], total: 0 })
    vi.mocked(listDueSourceReviewOccurrences).mockResolvedValue({
      items: [],
      total: 0,
      now: "2026-07-09T12:00:00Z"
    })
    vi.mocked(createSourceReviewPlan).mockResolvedValue(plan)
    vi.mocked(startSourceReviewOccurrence).mockResolvedValue(occurrence)
    vi.mocked(completeSourceReviewOccurrence).mockResolvedValue({
      ...occurrence,
      status: "completed"
    })
    vi.mocked(skipSourceReviewOccurrence).mockResolvedValue({
      ...occurrence,
      status: "skipped"
    })
    vi.mocked(deleteSourceReviewPlan).mockResolvedValue({ deleted: true })
  })

  it("fetches plan and due lists with the requested pagination", async () => {
    const client = queryClient()
    renderHook(() => useSourceReviewPlansQuery({ limit: 20, offset: 5 }), {
      wrapper: wrapper(client)
    })
    renderHook(
      () =>
        useDueSourceReviewOccurrencesQuery({
          limit: 10,
          offset: 2,
          enabled: true
        }),
      { wrapper: wrapper(client) }
    )

    await waitFor(() => {
      expect(listSourceReviewPlans).toHaveBeenCalledWith({
        limit: 20,
        offset: 5
      })
      expect(listDueSourceReviewOccurrences).toHaveBeenCalledWith({
        limit: 10,
        offset: 2
      })
    })
  })

  it("refetches enabled due reviews every minute", async () => {
    vi.useFakeTimers()
    const client = queryClient()
    try {
      renderHook(() => useDueSourceReviewOccurrencesQuery({ enabled: true }), {
        wrapper: wrapper(client)
      })
      await act(async () => {
        await vi.advanceTimersByTimeAsync(0)
      })
      expect(listDueSourceReviewOccurrences).toHaveBeenCalledTimes(1)

      await act(async () => {
        await vi.advanceTimersByTimeAsync(60_000)
      })
      expect(listDueSourceReviewOccurrences).toHaveBeenCalledTimes(2)
    } finally {
      client.clear()
      vi.useRealTimers()
    }
  })

  it("does not fetch due reviews when disabled", async () => {
    const client = queryClient()
    renderHook(
      () => useDueSourceReviewOccurrencesQuery({ enabled: false }),
      { wrapper: wrapper(client) }
    )

    await act(async () => {
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(listDueSourceReviewOccurrences).not.toHaveBeenCalled()
  })

  it("calls every mutation service and invalidates source-review queries", async () => {
    const client = queryClient()
    const invalidate = vi.spyOn(client, "invalidateQueries")
    const create = renderHook(() => useCreateSourceReviewPlanMutation(), {
      wrapper: wrapper(client)
    })
    const start = renderHook(() => useStartSourceReviewOccurrenceMutation(), {
      wrapper: wrapper(client)
    })
    const complete = renderHook(
      () => useCompleteSourceReviewOccurrenceMutation(),
      { wrapper: wrapper(client) }
    )
    const skip = renderHook(() => useSkipSourceReviewOccurrenceMutation(), {
      wrapper: wrapper(client)
    })
    const remove = renderHook(() => useDeleteSourceReviewPlanMutation(), {
      wrapper: wrapper(client)
    })

    await act(async () => {
      await create.result.current.mutateAsync(request)
      await start.result.current.mutateAsync(31)
      await complete.result.current.mutateAsync(31)
      await skip.result.current.mutateAsync(31)
      await remove.result.current.mutateAsync(7)
    })

    expect(createSourceReviewPlan).toHaveBeenCalledWith(request)
    expect(startSourceReviewOccurrence).toHaveBeenCalledWith(31)
    expect(completeSourceReviewOccurrence).toHaveBeenCalledWith(31)
    expect(skipSourceReviewOccurrence).toHaveBeenCalledWith(31)
    expect(deleteSourceReviewPlan).toHaveBeenCalledWith(7)
    expect(invalidate).toHaveBeenCalledTimes(5)
    for (const call of invalidate.mock.calls) {
      const predicate = call[0]?.predicate
      expect(
        predicate?.({
          queryKey: ["flashcards:source-review:due"]
        } as never)
      ).toBe(true)
      expect(
        predicate?.({
          queryKey: ["flashcards:other"]
        } as never)
      ).toBe(false)
    }
  })
})
