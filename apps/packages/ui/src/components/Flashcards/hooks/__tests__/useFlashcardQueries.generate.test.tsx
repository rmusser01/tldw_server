import React from "react"
import { act, renderHook } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { useGenerateFlashcardsMutation } from "../useFlashcardQueries"

const generateFlashcardsSpy = vi.hoisted(() => vi.fn())

vi.mock("@/services/flashcards", async () => {
  const actual = await vi.importActual<typeof import("@/services/flashcards")>(
    "@/services/flashcards"
  )
  return {
    ...actual,
    generateFlashcards: generateFlashcardsSpy
  }
})

const buildWrapper = (queryClient: QueryClient) => {
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  )
}

describe("useGenerateFlashcardsMutation", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    generateFlashcardsSpy.mockResolvedValue({ flashcards: [], count: 0 })
  })

  it("falls back to card_type when cardPlan is empty", async () => {
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false }, mutations: { retry: false } }
    })
    const { result } = renderHook(() => useGenerateFlashcardsMutation(), {
      wrapper: buildWrapper(queryClient)
    })

    await act(async () => {
      await result.current.mutateAsync({
        text: "ATP powers the cell.",
        numCards: 5,
        cardType: "cloze",
        cardPlan: []
      })
    })

    expect(generateFlashcardsSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        text: "ATP powers the cell.",
        num_cards: 5,
        card_type: "cloze",
        card_plan: undefined
      })
    )
  })
})
