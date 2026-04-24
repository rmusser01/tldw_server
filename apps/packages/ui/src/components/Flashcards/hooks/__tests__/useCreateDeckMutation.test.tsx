import React from "react"
import { act, renderHook } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { useCreateDeckMutation } from "../useFlashcardQueries"
import { createDeck, type DeckSchedulerSettings } from "@/services/flashcards"
import { DEFAULT_SCHEDULER_SETTINGS_ENVELOPE } from "../../utils/scheduler-settings"

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
  const noopAsync = vi.fn()
  return {
    ...actual,
    listDecks: noopAsync,
    listFlashcards: noopAsync,
    createFlashcard: noopAsync,
    createFlashcardsBulk: noopAsync,
    updateFlashcardsBulk: noopAsync,
    createDeck: vi.fn(),
    updateDeck: noopAsync,
    updateFlashcard: noopAsync,
    deleteFlashcard: noopAsync,
    resetFlashcardScheduling: noopAsync,
    reviewFlashcard: noopAsync,
    getNextReviewCard: noopAsync,
    getFlashcardAssistant: noopAsync,
    respondFlashcardAssistant: noopAsync,
    generateFlashcards: noopAsync,
    getFlashcard: noopAsync,
    importFlashcards: noopAsync,
    previewStructuredQaImport: noopAsync,
    importFlashcardsJson: noopAsync,
    importFlashcardsApkg: noopAsync,
    getFlashcardsAnalyticsSummary: noopAsync,
    exportFlashcards: noopAsync,
    exportFlashcardsFile: noopAsync,
    getFlashcardsImportLimits: noopAsync
  }
})

const schedulerSettings: DeckSchedulerSettings = {
  new_steps_minutes: [1, 5, 15],
  relearn_steps_minutes: [10],
  graduating_interval_days: 1,
  easy_interval_days: 3,
  easy_bonus: 1.15,
  interval_modifier: 0.9,
  max_interval_days: 3650,
  leech_threshold: 10,
  enable_fuzz: false
}

const schedulerEnvelope = {
  ...DEFAULT_SCHEDULER_SETTINGS_ENVELOPE,
  sm2_plus: schedulerSettings
}

const buildWrapper = (queryClient: QueryClient) => {
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  )
}

describe("useCreateDeckMutation", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("forwards full scheduler settings when creating a deck", async () => {
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false }
      }
    })

    vi.mocked(createDeck).mockResolvedValue({
      id: 9,
      name: "Biology Basics",
      description: null,
      review_prompt_side: "back",
      deleted: false,
      client_id: "test-client",
      version: 1,
      created_at: "2026-03-13T08:00:00Z",
      last_modified: "2026-03-13T08:00:00Z",
      scheduler_type: "sm2_plus",
      scheduler_settings_json: JSON.stringify(schedulerEnvelope),
      scheduler_settings: schedulerEnvelope
    })

    const { result } = renderHook(() => useCreateDeckMutation(), {
      wrapper: buildWrapper(queryClient)
    })

    await act(async () => {
      await result.current.mutateAsync({
        name: "  Biology Basics  ",
        description: "  Intro deck  ",
        review_prompt_side: "back",
        scheduler_settings: schedulerEnvelope
      })
    })

    expect(createDeck).toHaveBeenCalledWith({
      name: "Biology Basics",
      description: "Intro deck",
      review_prompt_side: "back",
      scheduler_settings: schedulerEnvelope
    })
  })
})
