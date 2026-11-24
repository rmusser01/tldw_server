import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { beforeEach, describe, expect, it, vi } from "vitest"

import * as backgroundProxy from "@/services/background-proxy"
import { ReviewTab } from "../ReviewTab"
import {
  useDecksQuery,
  useCramQueueQuery,
  useReviewQuery,
  useReviewFlashcardMutation,
  useRecentFlashcardReviewSessionsQuery,
  useGlobalFlashcardTagSuggestionsQuery,
  useFlashcardAssistantQuery,
  useFlashcardAssistantRespondMutation,
  useUpdateFlashcardMutation,
  useResetFlashcardSchedulingMutation,
  useDeleteFlashcardMutation,
  useFlashcardShortcuts,
  useDueCountsQuery,
  useDeckDueCountsQuery,
  useReviewAnalyticsSummaryQuery,
  useHasCardsQuery,
  useNextDueQuery
} from "../../hooks"

const messageSpies = {
  success: vi.fn(),
  error: vi.fn(),
  info: vi.fn(),
  warning: vi.fn(),
  loading: vi.fn(),
  open: vi.fn(),
  destroy: vi.fn()
}

vi.mock("@/services/service-prompts", async importOriginal => ({
  ...await importOriginal<typeof import("@/services/service-prompts")>(),
  ...await import("./review-scope-fixture")
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
      if (defaultValueOrOptions?.defaultValue) {
        return defaultValueOrOptions.defaultValue.replace(
          /\{\{(\w+)\}\}/g,
          (_match, token: string) =>
            String((defaultValueOrOptions as Record<string, unknown>)[token] ?? `{{${token}}}`)
        )
      }
      return key
    }
  })
}))

vi.mock("react-router-dom", async (importOriginal) => {
  const actual = await importOriginal<typeof import("react-router-dom")>()
  return {
    ...actual,
    useNavigate: () => vi.fn()
  }
})

vi.mock("@/hooks/useAntdMessage", () => ({
  useAntdMessage: () => messageSpies
}))

vi.mock("@/hooks/useTTS", () => ({
  useTTS: () => ({
    speak: vi.fn(),
    cancel: vi.fn(),
    isSpeaking: false
  })
}))

vi.mock("@/hooks/useSpeechRecognition", () => ({
  useSpeechRecognition: () => ({
    supported: false,
    isListening: false,
    transcript: "",
    start: vi.fn(),
    stop: vi.fn(),
    resetTranscript: vi.fn()
  })
}))

vi.mock("../../hooks", () => ({
  useDecksQuery: vi.fn(),
  useCramQueueQuery: vi.fn(),
  useReviewQuery: vi.fn(),
  useReviewFlashcardMutation: vi.fn(),
  useEndFlashcardReviewSessionMutation: vi.fn(() => ({ mutateAsync: vi.fn().mockResolvedValue({ id: 77 }), isPending: false })),
  useRecentFlashcardReviewSessionsQuery: vi.fn(() => ({
    data: [],
    isLoading: false,
    isError: false,
    error: null,
    refetch: vi.fn()
  })),
  useGlobalFlashcardTagSuggestionsQuery: vi.fn(),
  useFlashcardAssistantQuery: vi.fn(),
  useFlashcardAssistantRespondMutation: vi.fn(),
  useUpdateFlashcardMutation: vi.fn(),
  useResetFlashcardSchedulingMutation: vi.fn(),
  useDeleteFlashcardMutation: vi.fn(),
  useFlashcardShortcuts: vi.fn(),
  useDebouncedFormField: vi.fn(() => undefined),
  useDueCountsQuery: vi.fn(),
  useDeckDueCountsQuery: vi.fn(),
  useReviewAnalyticsSummaryQuery: vi.fn(),
  useHasCardsQuery: vi.fn(),
  useNextDueQuery: vi.fn()
}))

if (!(globalThis as any).ResizeObserver) {
  ;(globalThis as any).ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
}

if (typeof window !== "undefined" && typeof window.matchMedia !== "function") {
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    value: vi.fn().mockImplementation((query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn()
    }))
  })
}

const makeDeck = (id: number, name: string, reviewPromptSide: "front" | "back" = "front") => ({
  id,
  name,
  review_prompt_side: reviewPromptSide,
  deleted: false,
  client_id: "test",
  version: 1
})

const makeCard = (overrides: Partial<Record<string, unknown>> = {}) => ({
  uuid: "review-card-1",
  deck_id: 1,
  front: "ATP",
  back: "Energy currency",
  notes: null,
  extra: null,
  is_cloze: false,
  tags: [],
  ef: 2.5,
  interval_days: 2,
  repetitions: 1,
  lapses: 0,
  due_at: null,
  last_reviewed_at: null,
  last_modified: null,
  deleted: false,
  client_id: "test",
  version: 2,
  model_type: "basic",
  reverse: false,
  ...overrides
})

const revision = "a".repeat(64)
const firstChapter = `1:0:${revision}`

describe("ReviewTab orientation", () => {
  const reviewMutateAsync = vi.fn()
  let currentDecks: Array<ReturnType<typeof makeDeck>>
  let currentCard: ReturnType<typeof makeCard>

  beforeEach(() => {
    vi.clearAllMocks()
    currentDecks = [makeDeck(1, "Biology", "front"), makeDeck(2, "Physics", "front")]
    currentCard = makeCard()

    vi.mocked(useDecksQuery).mockImplementation(() => ({
      data: currentDecks,
      isLoading: false
    } as any))
    vi.mocked(useReviewQuery).mockImplementation(() => ({
      data: currentCard
    } as any))
    vi.mocked(useCramQueueQuery).mockReturnValue({ data: [] } as any)
    vi.mocked(useReviewFlashcardMutation).mockReturnValue({
      mutateAsync: reviewMutateAsync
    } as any)
    vi.mocked(useRecentFlashcardReviewSessionsQuery).mockReturnValue({
      data: [],
      isLoading: false
    } as any)
    vi.mocked(useFlashcardAssistantQuery).mockReturnValue({
      data: null,
      isLoading: false,
      isError: false
    } as any)
    vi.mocked(useFlashcardAssistantRespondMutation).mockReturnValue({
      mutateAsync: vi.fn(),
      isPending: false
    } as any)
    vi.mocked(useUpdateFlashcardMutation).mockReturnValue({
      mutateAsync: vi.fn(),
      isPending: false
    } as any)
    vi.mocked(useResetFlashcardSchedulingMutation).mockReturnValue({
      mutateAsync: vi.fn(),
      isPending: false
    } as any)
    vi.mocked(useDeleteFlashcardMutation).mockReturnValue({
      mutateAsync: vi.fn(),
      isPending: false
    } as any)
    vi.mocked(useGlobalFlashcardTagSuggestionsQuery).mockReturnValue({
      data: { items: [] },
      isLoading: false,
      isFetching: false,
      isError: false
    } as any)
    vi.mocked(useFlashcardShortcuts).mockImplementation(() => undefined)
    vi.mocked(useDueCountsQuery).mockReturnValue({
      data: { due: 1, new: 0, learning: 0, total: 1 }
    } as any)
    vi.mocked(useDeckDueCountsQuery).mockReturnValue({ data: {} } as any)
    vi.mocked(useReviewAnalyticsSummaryQuery).mockReturnValue({
      data: null,
      isLoading: false
    } as any)
    vi.mocked(useHasCardsQuery).mockReturnValue({ data: true } as any)
    vi.mocked(useNextDueQuery).mockReturnValue({ data: null } as any)
  })

  const renderReviewTab = (props?: Partial<React.ComponentProps<typeof ReviewTab>>, queryClient?: QueryClient) =>
    render(
      <ReviewTab
        onNavigateToCreate={() => {}}
        onNavigateToImport={() => {}}
        reviewDeckId={1}
        onReviewDeckChange={() => {}}
        isActive
        {...props}
      />,
      { wrapper: queryClient ? ({ children }) => (
          <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
        ) : undefined }
    )

  const renderCardBrowser = () => {
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    return renderReviewTab({ reviewDeckId: 1 }, queryClient)
  }

  it("browses All cards and pages a chapter's cards without starting review", async () => {
    const chapters = [
      { id: firstChapter, title: "Cell structure", source_id: "1", source_title: "Biology", card_count: 21 },
      { id: `1:1:${revision}`, title: "Metabolism", source_id: "1", source_title: "Biology", card_count: 1 },
      { id: `2:0:${revision}`, title: "Digestion", source_id: "2", source_title: "Physiology", card_count: 1 }
    ]
    const request = vi.spyOn(backgroundProxy, "bgRequest")
      .mockResolvedValueOnce({
        chapters, all_cards_count: 23, count: 20, total: 23,
        items: [makeCard({ uuid: "all", front: "Unsorted question", back: "Unsorted answer" })]
      })
      .mockResolvedValueOnce({
        chapters, all_cards_count: 23, count: 20, total: 21,
        items: [makeCard({ uuid: "first", front: "First chapter question", back: "First chapter answer" })]
      })
      .mockResolvedValueOnce({
        chapters, all_cards_count: 23, count: 1, total: 21,
        items: [makeCard({ uuid: "later", front: "Later chapter question", back: "Later chapter answer" })]
      })
    renderCardBrowser()

    fireEvent.click(screen.getByRole("button", { name: "Browse cards" }))
    const drawer = within(await screen.findByRole("dialog", { name: "Browse cards" }))
    const filter = await drawer.findByRole("combobox", { name: "Card set" })
    expect(within(filter).getAllByRole("option").map((option) => option.textContent)).toEqual([
      "All cards (23)", "Biology · Cell structure (21)", "Biology · Metabolism (1)", "Physiology · Digestion (1)"
    ])
    expect(filter).toHaveValue("")
    await waitFor(() => expect(drawer.getByText("Unsorted question")).toBeVisible())
    expect(drawer.getByText("Unsorted answer")).toBeVisible()
    expect(request).toHaveBeenNthCalledWith(1, {
      method: "GET",
      path: "/api/v1/flashcards/study-packs/decks/1/cards?limit=20&offset=0",
      abortSignal: expect.any(AbortSignal)
    })

    fireEvent.change(filter, { target: { value: firstChapter } })
    await waitFor(() => expect(drawer.getByText("First chapter question")).toBeVisible())
    expect(drawer.queryByText("Unsorted question")).not.toBeInTheDocument()
    fireEvent.click(drawer.getByRole("button", { name: "Next" }))
    await waitFor(() => expect(drawer.getByText("Later chapter question")).toBeVisible())
    expect(request).toHaveBeenLastCalledWith({
      method: "GET",
      path: `/api/v1/flashcards/study-packs/decks/1/cards?chapter_id=1%3A0%3A${revision}&limit=20&offset=20`,
      abortSignal: expect.any(AbortSignal)
    })
    expect(drawer.getByRole("button", { name: "Next" })).toBeDisabled()
    expect(reviewMutateAsync).not.toHaveBeenCalled()
    expect(useFlashcardShortcuts).toHaveBeenLastCalledWith(expect.objectContaining({ enabled: false }))
  })

  it("keeps All cards available when the selected chapter disappears", async () => {
    vi.spyOn(backgroundProxy, "bgRequest")
      .mockResolvedValueOnce({
        chapters: [{ id: firstChapter, title: "Cell structure", source_id: "1", card_count: 1 }],
        all_cards_count: 2, count: 2, total: 2, items: [makeCard()]
      })
      .mockResolvedValueOnce({ chapters: [], all_cards_count: 1, count: 0, total: 0, items: [] })
      .mockResolvedValueOnce({
        chapters: [], all_cards_count: 1, count: 1, total: 1,
        items: [makeCard({ uuid: "remaining", front: "Remaining question" })]
      })
    renderCardBrowser()

    fireEvent.click(screen.getByRole("button", { name: "Browse cards" }))
    const drawer = within(await screen.findByRole("dialog", { name: "Browse cards" }))
    const filter = await drawer.findByRole("combobox", { name: "Card set" })
    fireEvent.change(filter, { target: { value: firstChapter } })
    expect(await drawer.findByRole("option", { name: "Chapter unavailable" })).toBeInTheDocument()
    expect(filter).toHaveValue(firstChapter)
    fireEvent.change(filter, { target: { value: "" } })
    await waitFor(() => expect(drawer.getByText("Remaining question")).toBeVisible())
  })

  it("aborts browsing and removes deck content when the account changes", async () => {
    const request = vi.spyOn(backgroundProxy, "bgRequest")
      .mockImplementationOnce(() => new Promise(() => {}))
    renderCardBrowser()

    fireEvent.click(screen.getByRole("button", { name: "Browse cards" }))
    await screen.findByRole("dialog", { name: "Browse cards" })
    await waitFor(() => expect(request).toHaveBeenCalledOnce())
    const signal = request.mock.calls[0][0].abortSignal
    expect(signal?.aborted).toBe(false)
    fireEvent(window, new Event("tldw:auth-principal-changed"))

    expect(signal?.aborted).toBe(true)
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "Browse cards" })).not.toBeInTheDocument())
  })

  it("aborts the old deck page and starts with All cards in the new deck", async () => {
    const request = vi.spyOn(backgroundProxy, "bgRequest")
      .mockImplementationOnce(() => new Promise(() => {}))
      .mockResolvedValueOnce({
        chapters: [], all_cards_count: 1, count: 1, total: 1,
        items: [makeCard({ uuid: "new-deck", deck_id: 2, front: "Physics question" })]
      })
    const { rerender } = renderCardBrowser()

    fireEvent.click(screen.getByRole("button", { name: "Browse cards" }))
    await waitFor(() => expect(request).toHaveBeenCalledOnce())
    const signal = request.mock.calls[0][0].abortSignal
    rerender(
      <ReviewTab
        onNavigateToCreate={() => {}}
        onNavigateToImport={() => {}}
        reviewDeckId={2}
        onReviewDeckChange={() => {}}
        isActive
      />
    )
    expect(signal?.aborted).toBe(true)
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "Browse cards" })).not.toBeInTheDocument())
    fireEvent.click(screen.getByRole("button", { name: "Browse cards" }))
    const drawer = within(await screen.findByRole("dialog", { name: "Browse cards" }))
    await waitFor(() => expect(drawer.getByText("Physics question")).toBeVisible())
    expect(drawer.getByRole("combobox", { name: "Card set" })).toHaveValue("")
    expect(request).toHaveBeenLastCalledWith(expect.objectContaining({
      path: "/api/v1/flashcards/study-packs/decks/2/cards?limit=20&offset=0"
    }))
  })

  it("shows the front as the prompt by default", async () => {
    renderReviewTab()

    await waitFor(() => {
      expect(screen.getByText("Front")).toBeInTheDocument()
    })
    expect(screen.getByText("ATP")).toBeInTheDocument()
    expect(screen.queryByText("Back")).not.toBeInTheDocument()
  })

  it("shows the back as the prompt when the deck default is back-first", async () => {
    currentDecks = [makeDeck(1, "Biology", "back")]

    renderReviewTab()

    await waitFor(() => {
      expect(screen.getByText("Back")).toBeInTheDocument()
    })
    expect(screen.getByText("Energy currency")).toBeInTheDocument()
    expect(screen.queryByText("ATP")).not.toBeInTheDocument()

    fireEvent.click(screen.getByTestId("flashcards-review-show-answer"))

    await waitFor(() => {
      expect(screen.getAllByText("Front").length).toBeGreaterThan(0)
    })
    expect(screen.getByText("ATP")).toBeInTheDocument()
  })

  it("ignores back-first orientation for cloze cards", async () => {
    currentDecks = [makeDeck(1, "Biology", "back")]
    currentCard = makeCard({
      front: "The {{c1::ATP}} powers the cell.",
      back: "ATP",
      is_cloze: true,
      model_type: "cloze"
    })

    renderReviewTab()

    await waitFor(() => {
      expect(screen.getByText("Front")).toBeInTheDocument()
    })
    expect(screen.getByText("The {{c1::ATP}} powers the cell.")).toBeInTheDocument()
    expect(screen.queryByText("Back")).not.toBeInTheDocument()
  })

  it("locks the prompt-side control to front-first for cloze cards", async () => {
    const { rerender } = renderReviewTab()

    fireEvent.click(screen.getByText("Back first"))

    await waitFor(() => {
      expect(screen.getByText("Back")).toBeInTheDocument()
    })

    currentCard = makeCard({
      front: "The {{c1::ATP}} powers the cell.",
      back: "ATP",
      is_cloze: true,
      model_type: "cloze"
    })

    rerender(
      <ReviewTab
        onNavigateToCreate={() => {}}
        onNavigateToImport={() => {}}
        reviewDeckId={1}
        onReviewDeckChange={() => {}}
        isActive
      />
    )

    await waitFor(() => {
      expect(screen.getByText("Front")).toBeInTheDocument()
    })

    const promptSideToggle = screen.getByTestId("flashcards-review-prompt-side-toggle")
    const frontOption = within(promptSideToggle).getByText("Front first").closest(
      ".ant-segmented-item"
    )
    const backOption = within(promptSideToggle).getByText("Back first").closest(
      ".ant-segmented-item"
    )

    expect(promptSideToggle).toHaveClass("ant-segmented-disabled")
    expect(frontOption).toHaveClass("ant-segmented-item-selected")
    expect(backOption).not.toHaveClass("ant-segmented-item-selected")
  })

  it("uses the session override instead of the deck default", async () => {
    renderReviewTab()

    fireEvent.click(screen.getByText("Back first"))

    await waitFor(() => {
      expect(screen.getByText("Back")).toBeInTheDocument()
    })
    expect(screen.getByText("Energy currency")).toBeInTheDocument()
  })

  it("resets the session override when the review scope changes", async () => {
    const { rerender } = renderReviewTab()

    fireEvent.click(screen.getByText("Back first"))

    await waitFor(() => {
      expect(screen.getByText("Back")).toBeInTheDocument()
    })

    rerender(
      <ReviewTab
        onNavigateToCreate={() => {}}
        onNavigateToImport={() => {}}
        reviewDeckId={2}
        onReviewDeckChange={() => {}}
        isActive
      />
    )

    await waitFor(() => {
      expect(screen.getByText("Front")).toBeInTheDocument()
    })
    expect(screen.getByText("ATP")).toBeInTheDocument()
  })
})
