import React from "react"
import { act, renderHook } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { useUpdateDeckMutation } from "../useFlashcardQueries"
import {
  updateDeck,
  type Deck,
  type DeckSchedulerSettings
} from "@/services/flashcards"
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
    createDeck: noopAsync,
    updateDeck: vi.fn(),
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

const baseSettings: DeckSchedulerSettings = {
  new_steps_minutes: [1, 10],
  relearn_steps_minutes: [10],
  graduating_interval_days: 1,
  easy_interval_days: 4,
  easy_bonus: 1.3,
  interval_modifier: 1,
  max_interval_days: 365,
  leech_threshold: 8,
  enable_fuzz: true
}

const baseEnvelope = {
  ...DEFAULT_SCHEDULER_SETTINGS_ENVELOPE,
  sm2_plus: baseSettings
}

const makeDeck = (overrides: Partial<Deck> = {}): Deck => ({
  id: 7,
  name: "Biology",
  description: "Study deck",
  review_prompt_side: "front",
  deleted: false,
  client_id: "test-client",
  version: 2,
  created_at: "2026-03-13T08:00:00Z",
  last_modified: "2026-03-13T08:00:00Z",
  scheduler_type: "sm2_plus",
  scheduler_settings_json: JSON.stringify(baseEnvelope),
  scheduler_settings: baseEnvelope,
  ...overrides
})

const buildWrapper = (queryClient: QueryClient) => {
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  )
}

describe("useUpdateDeckMutation", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("updates the deck cache with the returned scheduler settings", async () => {
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false }
      }
    })

    const updatedDeck = makeDeck({
      version: 3,
      review_prompt_side: "back",
      scheduler_settings: {
        ...baseEnvelope,
        sm2_plus: {
          ...baseSettings,
          new_steps_minutes: [2, 20],
          enable_fuzz: false
        }
      }
    })
    vi.mocked(updateDeck).mockResolvedValue(updatedDeck)
    const invalidateSpy = vi.spyOn(queryClient, "invalidateQueries")

    queryClient.setQueryData(["flashcards:decks"], [makeDeck()])

    const { result } = renderHook(() => useUpdateDeckMutation(), {
      wrapper: buildWrapper(queryClient)
    })

    await act(async () => {
      await result.current.mutateAsync({
        deckId: 7,
        update: {
          review_prompt_side: "back",
          scheduler_settings: {
            sm2_plus: {
              new_steps_minutes: [2, 20],
              enable_fuzz: false
            }
          },
          expected_version: 2
        }
      })
    })

    expect(updateDeck).toHaveBeenCalledWith(
      7,
      {
        review_prompt_side: "back",
        scheduler_settings: {
          sm2_plus: {
            new_steps_minutes: [2, 20],
            enable_fuzz: false
          }
        },
        expected_version: 2
      }
    )
    expect(queryClient.getQueryData<Deck[]>(["flashcards:decks"])).toEqual([updatedDeck])
    expect(invalidateSpy).toHaveBeenCalled()
  })

  it("propagates deck update conflicts without corrupting deck cache", async () => {
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false }
      }
    })
    const existingDeck = makeDeck()
    const conflictError = Object.assign(new Error("Version mismatch"), {
      response: { status: 409 }
    })
    vi.mocked(updateDeck).mockRejectedValue(conflictError)

    queryClient.setQueryData(["flashcards:decks"], [existingDeck])

    const { result } = renderHook(() => useUpdateDeckMutation(), {
      wrapper: buildWrapper(queryClient)
    })

    await expect(
      result.current.mutateAsync({
        deckId: 7,
        update: {
          scheduler_settings: {
            sm2_plus: {
              leech_threshold: 10
            }
          },
          expected_version: 2
        }
      })
    ).rejects.toBe(conflictError)

    expect(queryClient.getQueryData<Deck[]>(["flashcards:decks"])).toEqual([existingDeck])
  })

  it("forwards review_prompt_side when updating a deck", async () => {
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false }
      }
    })
    const updatedDeck = makeDeck({
      version: 3,
      review_prompt_side: "back"
    })
    vi.mocked(updateDeck).mockResolvedValue(updatedDeck)

    queryClient.setQueryData(["flashcards:decks"], [makeDeck()])

    const { result } = renderHook(() => useUpdateDeckMutation(), {
      wrapper: buildWrapper(queryClient)
    })

    await act(async () => {
      await result.current.mutateAsync({
        deckId: 7,
        update: {
          review_prompt_side: "back",
          expected_version: 2
        }
      })
    })

    expect(updateDeck).toHaveBeenCalledWith(7, {
      review_prompt_side: "back",
      expected_version: 2
    })
  })
})
