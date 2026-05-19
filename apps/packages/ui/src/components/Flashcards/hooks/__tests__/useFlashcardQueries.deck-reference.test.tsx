import React from "react"
import { act, renderHook, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { beforeEach, describe, expect, it, vi } from "vitest"

import {
  useFlashcardDeckRecentCardsQuery,
  useFlashcardDeckSearchQuery
} from "../useFlashcardQueries"
import { listFlashcards, type FlashcardListResponse } from "@/services/flashcards"

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

const buildWrapper = (queryClient: QueryClient) => {
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  )
}

describe("useFlashcardDeckRecentCardsQuery + useFlashcardDeckSearchQuery", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("calls listFlashcards with the recent-card deck reference parameters", async () => {
    const response: FlashcardListResponse = {
      items: [
        { uuid: "card-b", created_at: "2026-03-13T08:02:00Z" } as never,
        { uuid: "card-a", created_at: "2026-03-13T08:01:00Z" } as never
      ],
      count: 2,
      total: 2
    }
    vi.mocked(listFlashcards).mockResolvedValue(response)
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false }
      }
    })

    const { result } = renderHook(() => useFlashcardDeckRecentCardsQuery(42), {
      wrapper: buildWrapper(queryClient)
    })

    await waitFor(() => {
      expect(result.current.data).toBeDefined()
    })

    expect(listFlashcards).toHaveBeenCalledWith(
      expect.objectContaining({
        deck_id: 42,
        due_status: "all",
        order_by: "created_at",
        limit: 6,
        offset: 0
      })
    )
  })

  it("uses the explicit recent-card limit in the request and query key", async () => {
    vi.mocked(listFlashcards).mockResolvedValue({
      items: [
        { uuid: "card-b", created_at: "2026-03-13T08:02:00Z" } as never,
        { uuid: "card-a", created_at: "2026-03-13T08:01:00Z" } as never
      ],
      count: 2,
      total: 2
    })
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false }
      }
    })

    renderHook(() => useFlashcardDeckRecentCardsQuery(42, { limit: 12 }), {
      wrapper: buildWrapper(queryClient)
    })

    await waitFor(() => {
      expect(listFlashcards).toHaveBeenCalledWith(
        expect.objectContaining({
          deck_id: 42,
          due_status: "all",
          order_by: "created_at",
          limit: 12,
          offset: 0
        })
      )
    })

    const recentQuery = queryClient
      .getQueryCache()
      .getAll()
      .find((query) => query.queryKey[0] === "flashcards:deck:recent")

    expect(recentQuery?.queryKey).toEqual([
      "flashcards:deck:recent",
      42,
      12,
      {
        workspace_id: undefined,
        include_workspace_items: false
      }
    ])
    expect(recentQuery?.queryKey[2]).toBe(12)
  })

  it("forwards workspace visibility parameters for recent cards", async () => {
    vi.mocked(listFlashcards).mockResolvedValue({
      items: [],
      count: 0,
      total: 0
    })
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false }
      }
    })

    renderHook(
      () =>
        useFlashcardDeckRecentCardsQuery(42, {
          limit: 9,
          workspaceId: "workspace-1",
          includeWorkspaceItems: true
        }),
      {
        wrapper: buildWrapper(queryClient)
      }
    )

    await waitFor(() => {
      expect(listFlashcards).toHaveBeenCalledWith(
        expect.objectContaining({
          deck_id: 42,
          workspace_id: "workspace-1",
          include_workspace_items: true,
          limit: 9,
          offset: 0
        })
      )
    })
  })

  it("preserves backend order for recent cards", async () => {
    vi.mocked(listFlashcards).mockResolvedValue({
      items: [
        { uuid: "card-b", created_at: "2026-03-13T08:02:00Z" } as never,
        { uuid: "card-a", created_at: "2026-03-13T08:01:00Z" } as never
      ],
      count: 2,
      total: 2
    })
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false }
      }
    })

    const { result } = renderHook(() => useFlashcardDeckRecentCardsQuery(42), {
      wrapper: buildWrapper(queryClient)
    })

    await waitFor(() => {
      expect(result.current.data?.map((card) => card.uuid)).toEqual(["card-b", "card-a"])
    })
  })

  it("does not call listFlashcards when the trimmed search term is empty", async () => {
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false }
      }
    })

    const { result } = renderHook(() => useFlashcardDeckSearchQuery({ deckId: 42, query: "   " }), {
      wrapper: buildWrapper(queryClient)
    })

    await act(async () => {
      const response = await result.current.refetch()
      expect(response.data).toEqual([])
    })

    expect(listFlashcards).toHaveBeenCalledTimes(0)
  })

  it("does not call listFlashcards when the search deckId is missing", async () => {
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false }
      }
    })

    const { result } = renderHook(() => useFlashcardDeckSearchQuery({ deckId: undefined, query: "term" }), {
      wrapper: buildWrapper(queryClient)
    })

    await act(async () => {
      const response = await result.current.refetch()
      expect(response.data).toEqual([])
    })

    expect(listFlashcards).toHaveBeenCalledTimes(0)
  })

  it("calls listFlashcards with the trimmed deck search parameters", async () => {
    vi.mocked(listFlashcards).mockResolvedValue({
      items: [
        { uuid: "card-b", created_at: "2026-03-13T08:02:00Z" } as never,
        { uuid: "card-a", created_at: "2026-03-13T08:01:00Z" } as never
      ],
      count: 2,
      total: 2
    })
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false }
      }
    })

    const { result } = renderHook(
      () => useFlashcardDeckSearchQuery({ deckId: 42, query: "  spaced term  " }),
      {
        wrapper: buildWrapper(queryClient)
      }
    )

    await waitFor(() => {
      expect(result.current.data).toBeDefined()
    })

    expect(listFlashcards).toHaveBeenCalledWith(
      expect.objectContaining({
        deck_id: 42,
        due_status: "all",
        order_by: "created_at",
        q: "spaced term",
        limit: 20,
        offset: 0
      })
    )

    const searchQuery = queryClient
      .getQueryCache()
      .getAll()
      .find((query) => query.queryKey[0] === "flashcards:deck:search")

    expect(searchQuery?.queryKey).toEqual([
      "flashcards:deck:search",
      42,
      "spaced term",
      20,
      {
        workspace_id: undefined,
        include_workspace_items: false
      }
    ])
  })

  it("forwards workspace visibility parameters for deck search", async () => {
    vi.mocked(listFlashcards).mockResolvedValue({
      items: [],
      count: 0,
      total: 0
    })
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false }
      }
    })

    renderHook(
      () =>
        useFlashcardDeckSearchQuery(
          { deckId: 42, query: "term", limit: 11 },
          {
            workspaceId: "workspace-1",
            includeWorkspaceItems: true
          }
        ),
      {
        wrapper: buildWrapper(queryClient)
      }
    )

    await waitFor(() => {
      expect(listFlashcards).toHaveBeenCalledWith(
        expect.objectContaining({
          deck_id: 42,
          q: "term",
          workspace_id: "workspace-1",
          include_workspace_items: true,
          limit: 11,
          offset: 0
        })
      )
    })
  })

  it("uses the explicit search limit in the request and query key", async () => {
    vi.mocked(listFlashcards).mockResolvedValue({
      items: [
        { uuid: "card-b", created_at: "2026-03-13T08:02:00Z" } as never,
        { uuid: "card-a", created_at: "2026-03-13T08:01:00Z" } as never
      ],
      count: 2,
      total: 2
    })
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false }
      }
    })

    renderHook(
      () => useFlashcardDeckSearchQuery({ deckId: 42, query: "term", limit: 7 }),
      {
        wrapper: buildWrapper(queryClient)
      }
    )

    await waitFor(() => {
      expect(listFlashcards).toHaveBeenCalledWith(
        expect.objectContaining({
          deck_id: 42,
          due_status: "all",
          order_by: "created_at",
          q: "term",
          limit: 7,
          offset: 0
        })
      )
    })

    const searchQuery = queryClient
      .getQueryCache()
      .getAll()
      .find((query) => query.queryKey[0] === "flashcards:deck:search")

    expect(searchQuery?.queryKey).toEqual([
      "flashcards:deck:search",
      42,
      "term",
      7,
      {
        workspace_id: undefined,
        include_workspace_items: false
      }
    ])
  })

  it("preserves backend order for deck search results", async () => {
    vi.mocked(listFlashcards).mockResolvedValue({
      items: [
        { uuid: "card-b", created_at: "2026-03-13T08:02:00Z" } as never,
        { uuid: "card-a", created_at: "2026-03-13T08:01:00Z" } as never
      ],
      count: 2,
      total: 2
    })
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false }
      }
    })

    const { result } = renderHook(
      () => useFlashcardDeckSearchQuery({ deckId: 42, query: "term" }),
      {
        wrapper: buildWrapper(queryClient)
      }
    )

    await waitFor(() => {
      expect(result.current.data?.map((card) => card.uuid)).toEqual(["card-b", "card-a"])
    })
  })

  it("does not call listFlashcards when the recent deckId is missing", async () => {
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false }
      }
    })

    const { result } = renderHook(() => useFlashcardDeckRecentCardsQuery(undefined, { limit: 8 }), {
      wrapper: buildWrapper(queryClient)
    })

    await act(async () => {
      const response = await result.current.refetch()
      expect(response.data).toEqual([])
    })

    expect(listFlashcards).toHaveBeenCalledTimes(0)
    expect(result.current.data).toBeUndefined()
  })

  it("keeps both query keys under the flashcards namespace", async () => {
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false }
      }
    })

    renderHook(() => useFlashcardDeckRecentCardsQuery(42), {
      wrapper: buildWrapper(queryClient)
    })
    renderHook(() => useFlashcardDeckSearchQuery({ deckId: 42, query: "term" }), {
      wrapper: buildWrapper(queryClient)
    })

    const keys = queryClient
      .getQueryCache()
      .getAll()
      .map((query) => query.queryKey)

    expect(keys.every((key) => typeof key[0] === "string" && key[0].startsWith("flashcards:"))).toBe(
      true
    )
  })
})
