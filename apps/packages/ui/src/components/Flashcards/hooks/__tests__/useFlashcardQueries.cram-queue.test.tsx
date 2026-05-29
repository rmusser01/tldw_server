import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { renderHook, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { useCramQueueQuery } from "../useFlashcardQueries"
import { listFlashcards, type Flashcard } from "@/services/flashcards"

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
    listFlashcards: vi.fn()
  }
})

const buildWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false }
    }
  })

  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  )
}

const buildFlashcard = (uuid: string): Flashcard => ({
  uuid,
  deck_id: 42,
  front: "Front",
  back: "Back",
  notes: null,
  extra: null,
  is_cloze: false,
  tags: [],
  ef: 2.5,
  interval_days: 1,
  repetitions: 0,
  lapses: 0,
  due_at: null,
  created_at: "2026-02-18T12:00:00.000Z",
  last_reviewed_at: null,
  queue_state: "review",
  step_index: 0,
  suspended_reason: null,
  last_modified: "2026-02-18T12:00:00.000Z",
  deleted: false,
  client_id: "test",
  version: 1,
  model_type: "basic",
  reverse: false,
  source_ref_type: "manual",
  source_ref_id: null,
  conversation_id: null,
  message_id: null,
  next_intervals: null
})

describe("useCramQueueQuery", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("can probe cram availability with a single-card limit", async () => {
    vi.mocked(listFlashcards).mockResolvedValue({
      items: [buildFlashcard("card-1")],
      count: 1,
      total: 500
    })

    const { result } = renderHook(() => useCramQueueQuery(42, null, { limit: 1 }), {
      wrapper: buildWrapper()
    })

    await waitFor(() => {
      expect(result.current.data).toHaveLength(1)
    })

    expect(listFlashcards).toHaveBeenCalledTimes(1)
    expect(listFlashcards).toHaveBeenCalledWith(
      expect.objectContaining({
        deck_id: 42,
        due_status: "all",
        include_workspace_items: false,
        limit: 1,
        offset: 0,
        order_by: "due_at"
      })
    )
  })

  it("caps the availability probe by fetched cards before filtering residue", async () => {
    vi.mocked(listFlashcards)
      .mockResolvedValueOnce({
        items: [
          {
            ...buildFlashcard("tutorial-residue"),
            front: "Tips for effective flashcard use"
          }
        ],
        count: 1,
        total: 500
      })
      .mockResolvedValueOnce({
        items: [buildFlashcard("card-2")],
        count: 1,
        total: 500
      })

    const { result } = renderHook(() => useCramQueueQuery(42, null, { limit: 1 }), {
      wrapper: buildWrapper()
    })

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true)
    })

    expect(result.current.data).toEqual([])
    expect(listFlashcards).toHaveBeenCalledTimes(1)
  })
})
