import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

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
  useEndFlashcardReviewSessionMutation: vi.fn(),
  useRecentFlashcardReviewSessionsQuery: vi.fn(),
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

  const renderReviewTab = (props?: Partial<React.ComponentProps<typeof ReviewTab>>) =>
    render(
      <ReviewTab
        onNavigateToCreate={() => {}}
        onNavigateToImport={() => {}}
        reviewDeckId={1}
        onReviewDeckChange={() => {}}
        isActive
        {...props}
      />
    )

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
