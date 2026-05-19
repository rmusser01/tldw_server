import React from "react"
import { renderHook, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { useGlobalFlashcardTagSuggestionsQuery } from "../useFlashcardQueries"
import {
  listFlashcardTagSuggestions,
  listFlashcards,
  type FlashcardTagSuggestionsResponse
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
    listFlashcardTagSuggestions: vi.fn(),
    listFlashcards: vi.fn()
  }
})

const buildWrapper = (queryClient: QueryClient) => {
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  )
}

describe("useGlobalFlashcardTagSuggestionsQuery", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("uses the dedicated tag suggestions service and forwards the react query signal", async () => {
    const response: FlashcardTagSuggestionsResponse = {
      items: [
        { tag: "biology", count: 4 },
        { tag: "bioinformatics", count: 2 }
      ],
      count: 2
    }
    vi.mocked(listFlashcardTagSuggestions).mockResolvedValue(response)
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false }
      }
    })

    const { result } = renderHook(
      () => useGlobalFlashcardTagSuggestionsQuery("  bio  ", { limit: 12 }),
      { wrapper: buildWrapper(queryClient) }
    )

    await waitFor(() => {
      expect(result.current.data?.items[0]?.tag).toBe("biology")
    })

    expect(listFlashcardTagSuggestions).toHaveBeenCalledWith(
      expect.objectContaining({
        q: "bio",
        limit: 12,
        signal: expect.any(AbortSignal)
      })
    )
    expect(listFlashcards).not.toHaveBeenCalled()
  })

  it("stays disabled when explicitly disabled", async () => {
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false }
      }
    })

    const { result } = renderHook(
      () => useGlobalFlashcardTagSuggestionsQuery("bio", { enabled: false }),
      { wrapper: buildWrapper(queryClient) }
    )

    await waitFor(() => {
      expect(result.current.fetchStatus).toBe("idle")
    })

    expect(listFlashcardTagSuggestions).not.toHaveBeenCalled()
    expect(listFlashcards).not.toHaveBeenCalled()
  })
})
