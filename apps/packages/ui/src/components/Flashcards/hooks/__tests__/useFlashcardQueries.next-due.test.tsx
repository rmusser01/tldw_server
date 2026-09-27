import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { renderHook, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { listFlashcards, type Flashcard } from "@/services/flashcards"
import { useNextDueQuery } from "../useFlashcardQueries"

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({ capabilities: { hasFlashcards: true }, loading: false })
}))
vi.mock("@/hooks/useServerOnline", () => ({ useServerOnline: () => true }))
vi.mock("@/services/flashcards", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/services/flashcards")>(),
  listFlashcards: vi.fn()
}))

const now = Date.parse("2026-09-15T10:00:00Z")
const due = (minutes: number) => new Date(now + minutes * 60_000).toISOString()
const card = (dueAt: string | null, index: number): Flashcard => ({
  uuid: `card-${index}`, deck_id: 7, front: "Question", back: "Answer",
  notes: null, extra: null, is_cloze: false, tags: [], ef: 2.5,
  interval_days: 1, repetitions: 1, lapses: 0, due_at: dueAt,
  last_reviewed_at: null, deleted: false, client_id: "test", version: 1,
  model_type: "basic", reverse: false
})

const setup = () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return renderHook(() => useNextDueQuery(7), {
    wrapper: ({ children }: { children: React.ReactNode }) =>
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
  })
}

describe("next review hour window", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.spyOn(Date, "now").mockReturnValue(now)
  })

  it("counts the inclusive hour starting at the earliest future card, even when it is tomorrow", async () => {
    const times = [null, "invalid", due(-1), due(0), due(1440), due(1470), due(1500), due(1501)]
    const items = times.map(card)
    vi.mocked(listFlashcards).mockResolvedValue({ items, count: items.length, total: items.length })
    const { result } = setup()
    await waitFor(() => expect(result.current.isSuccess).toBe(true))
    expect(result.current.data).toEqual({ nextDueAt: due(1440), cardsDue: 3, isCapped: false, scanned: 8 })
  })

  it("reports a single future card", async () => {
    vi.mocked(listFlashcards).mockResolvedValue({ items: [card(due(30), 0)], count: 1, total: 1 })
    const { result } = setup()
    await waitFor(() => expect(result.current.isSuccess).toBe(true))
    expect(result.current.data).toEqual({ nextDueAt: due(30), cardsDue: 1, isCapped: false, scanned: 1 })
  })

  it("continues the same hour window across pages", async () => {
    const first = Array.from({ length: 200 }, (_, index) => card(index < 199 ? due(-1) : due(120), index))
    vi.mocked(listFlashcards)
      .mockResolvedValueOnce({ items: first, count: 200, total: 203 })
      .mockResolvedValueOnce({ items: [card(due(150), 200), card(due(180), 201), card(due(181), 202)], count: 3, total: 203 })
    const { result } = setup()
    await waitFor(() => expect(result.current.isSuccess).toBe(true))
    expect(result.current.data).toEqual({ nextDueAt: due(120), cardsDue: 3, isCapped: false, scanned: 203 })
    expect(vi.mocked(listFlashcards).mock.calls.map(([params]) => params.offset)).toEqual([0, 200])
  })

  it.each([false, true])("retains the scan-cap uncertainty when a future card is %s", async (hasFuture) => {
    vi.mocked(listFlashcards).mockImplementation(async ({ offset = 0 }) => ({
      items: Array.from({ length: 200 }, (_, index) => card(hasFuture ? due(120) : due(-1), offset + index)),
      count: 200, total: 2001
    }))
    const { result } = setup()
    await waitFor(() => expect(result.current.isSuccess).toBe(true))
    expect(result.current.data).toEqual({
      nextDueAt: hasFuture ? due(120) : null, cardsDue: hasFuture ? 2000 : 0, isCapped: true, scanned: 2000
    })
    expect(listFlashcards).toHaveBeenCalledTimes(10)
  })

  it("returns no future estimate when every card is unscheduled or already due", async () => {
    const items = [null, "invalid", due(-1), due(0)].map(card)
    vi.mocked(listFlashcards).mockResolvedValue({ items, count: items.length, total: items.length })
    const { result } = setup()
    await waitFor(() => expect(result.current.isSuccess).toBe(true))
    expect(result.current.data).toBeNull()
  })
})
