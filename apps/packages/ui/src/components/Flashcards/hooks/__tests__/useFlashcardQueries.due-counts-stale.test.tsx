import React from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { renderHook, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { listDecks, listFlashcards } from "@/services/flashcards"
import {
  useDeckDueCountsQuery,
  useDueCountsQuery
} from "../useFlashcardQueries"

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({
    capabilities: { hasFlashcards: true },
    loading: false
  })
}))
vi.mock("@/hooks/useServerOnline", () => ({ useServerOnline: () => true }))
vi.mock("@/services/flashcards", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/services/flashcards")>(),
  listFlashcards: vi.fn(),
  listDecks: vi.fn()
}))

const flashcardsMock = vi.mocked(listFlashcards)
const decksMock = vi.mocked(listDecks)

const emptyDueCounts = { items: [], count: 0, total: 0 }

const makeClient = () =>
  new QueryClient({ defaultOptions: { queries: { retry: false } } })

const mountWith = (client: QueryClient, hook: () => unknown) =>
  renderHook(hook, {
    wrapper: ({ children }: { children: React.ReactNode }) => (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    )
  })

describe("due-count queries 60s staleTime (WP-46)", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    flashcardsMock.mockResolvedValue(emptyDueCounts as never)
    decksMock.mockResolvedValue([
      { id: 7, name: "Deck A" },
      { id: 8, name: "Deck B" }
    ] as never)
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("issues the per-deck N+1 once per minute, not per mount", async () => {
    const client = makeClient()
    const first = mountWith(client, () => useDueCountsQuery(7))
    await waitFor(() => expect(first.result.current.isSuccess).toBe(true))
    // Three parallel due-status counts (due / new / learning).
    expect(flashcardsMock).toHaveBeenCalledTimes(3)
    first.unmount()

    const second = mountWith(client, () => useDueCountsQuery(7))
    await waitFor(() => expect(second.result.current.isSuccess).toBe(true))
    expect(second.result.current.isStale).toBe(false)
    // No refetch inside the 60s stale window.
    expect(flashcardsMock).toHaveBeenCalledTimes(3)
    second.unmount()
  })

  it("refetches after the 60s stale window elapses", async () => {
    const realNow = Date.now()
    const client = makeClient()
    const first = mountWith(client, () => useDueCountsQuery(7))
    await waitFor(() => expect(first.result.current.isSuccess).toBe(true))
    first.unmount()

    vi.spyOn(Date, "now").mockReturnValue(realNow + 61_000)
    const second = mountWith(client, () => useDueCountsQuery(7))
    await waitFor(() => expect(flashcardsMock).toHaveBeenCalledTimes(6))
    second.unmount()
  })

  it("caches the all-deck due-count fan-out with the same 60s window", async () => {
    const client = makeClient()
    const first = mountWith(client, () => useDeckDueCountsQuery())
    await waitFor(() => expect(first.result.current.isSuccess).toBe(true))
    // 1 deck list + 3 due-status counts per deck (2 decks).
    expect(decksMock).toHaveBeenCalledTimes(1)
    expect(flashcardsMock).toHaveBeenCalledTimes(6)
    first.unmount()

    const second = mountWith(client, () => useDeckDueCountsQuery())
    await waitFor(() => expect(second.result.current.isSuccess).toBe(true))
    expect(decksMock).toHaveBeenCalledTimes(1)
    expect(flashcardsMock).toHaveBeenCalledTimes(6)
    second.unmount()
  })
})
