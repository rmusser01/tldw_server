import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { ReviewTab } from "../ReviewTab"
import { clearSetting } from "@/services/settings/registry"
import type { Flashcard } from "@/services/flashcards"
import {
  FLASHCARDS_REVIEW_ONBOARDING_DISMISSED_SETTING,
  FLASHCARDS_SHORTCUT_HINT_DENSITY_SETTING
} from "@/services/settings/ui-settings"
import {
  useDecksQuery,
  useCramQueueQuery,
  useReviewQuery,
  useReviewFlashcardMutation,
  useEndFlashcardReviewSessionMutation,
  useGlobalFlashcardTagSuggestionsQuery,
  useFlashcardAssistantQuery,
  useFlashcardAssistantRespondMutation,
  useUpdateFlashcardMutation,
  useResetFlashcardSchedulingMutation,
  useDeleteFlashcardMutation,
  useRecentFlashcardReviewSessionsQuery,
  useFlashcardShortcuts,
  useDebouncedFormField,
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

vi.mock("@/components/StudySuggestions/StudySuggestionsPanel", () => ({ StudySuggestionsPanel: () => null }))

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

describe("ReviewTab cram mode", () => {
  const reviewMutateAsync = vi.fn()
  const endMutateAsync = vi.fn()

  beforeEach(async () => {
    vi.clearAllMocks()
    reviewMutateAsync.mockResolvedValue({ review_session_id: 77, interval_days: 1 })
    endMutateAsync.mockResolvedValue({ id: 77 })
    vi.mocked(useEndFlashcardReviewSessionMutation).mockReturnValue({ mutateAsync: endMutateAsync } as ReturnType<typeof useEndFlashcardReviewSessionMutation>)
    await clearSetting(FLASHCARDS_SHORTCUT_HINT_DENSITY_SETTING)
    await clearSetting(FLASHCARDS_REVIEW_ONBOARDING_DISMISSED_SETTING)
    vi.mocked(useDecksQuery).mockReturnValue({
      data: [{ id: 1, name: "Biology" }],
      isLoading: false
    } as any)
    vi.mocked(useReviewQuery).mockReturnValue({
      data: {
        uuid: "due-card-1",
        deck_id: 1,
        front: "Due front",
        back: "Due back",
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
        reverse: false
      }
    } as any)
    vi.mocked(useCramQueueQuery).mockReturnValue({
      data: [
        {
          uuid: "cram-card-1",
          deck_id: 1,
          front: "Cram front",
          back: "Cram back",
          notes: null,
          extra: null,
          is_cloze: false,
          tags: ["biology"],
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
          reverse: false
        }
      ]
    } as any)
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
    vi.mocked(useDebouncedFormField).mockReturnValue(undefined as any)
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

  it("shows cram controls and tag filter when cram mode is selected", () => {
    render(
      <ReviewTab
        onNavigateToCreate={() => {}}
        onNavigateToImport={() => {}}
        reviewDeckId={1}
        onReviewDeckChange={() => {}}
        isActive
      />
    )

    fireEvent.click(screen.getByText("Cram"))
    expect(screen.getByTestId("flashcards-review-cram-tag")).toBeInTheDocument()
    expect(screen.getByTestId("flashcards-review-cram-update-schedule")).toBeInTheDocument()
    expect(screen.getByText("Cram front")).toBeInTheDocument()
  })

  it("does not call review mutation when practicing in cram mode without schedule updates", async () => {
    render(
      <ReviewTab
        onNavigateToCreate={() => {}}
        onNavigateToImport={() => {}}
        reviewDeckId={1}
        onReviewDeckChange={() => {}}
        isActive
      />
    )

    fireEvent.click(screen.getByText("Cram"))
    fireEvent.click(screen.getByTestId("flashcards-review-show-answer"))
    fireEvent.click(screen.getByTestId("flashcards-review-rate-3"))

    expect(reviewMutateAsync).not.toHaveBeenCalled()
    expect(messageSpies.success).toHaveBeenCalledWith(
      "Practice saved. Scheduling unchanged."
    )
    await waitFor(() => {
      expect(
        screen.getByText("1 cards practiced in this cram session")
      ).toBeInTheDocument()
    })
  })

  it("keeps cram queue progression unchanged when the prompt side flips", async () => {
    render(
      <ReviewTab
        onNavigateToCreate={() => {}}
        onNavigateToImport={() => {}}
        reviewDeckId={1}
        onReviewDeckChange={() => {}}
        isActive
      />
    )

    fireEvent.click(screen.getByText("Cram"))
    fireEvent.click(screen.getByText("Back first"))

    await waitFor(() => {
      expect(screen.getByText("Back")).toBeInTheDocument()
    })
    expect(screen.getByText("Cram back")).toBeInTheDocument()

    fireEvent.click(screen.getByTestId("flashcards-review-show-answer"))
    fireEvent.click(screen.getByTestId("flashcards-review-rate-3"))

    expect(reviewMutateAsync).not.toHaveBeenCalled()
    expect(messageSpies.success).toHaveBeenCalledWith(
      "Practice saved. Scheduling unchanged."
    )
    await waitFor(() => {
      expect(
        screen.getByText("1 cards practiced in this cram session")
      ).toBeInTheDocument()
    })
    expect(screen.getByTestId("flashcards-review-empty-card")).toBeInTheDocument()
  })

  const mountQueue = () => {
    const template = vi.mocked(useCramQueueQuery)().data![0]
    const cards = ["Alpha", "Bravo", "Charlie"].map((front, index) => ({
      ...template, uuid: `card-${index}`, front,
    })) as Flashcard[]
    let queue = cards
    vi.mocked(useCramQueueQuery).mockImplementation(() => ({
      data: queue, isSuccess: true, isFetching: false, isLoading: false,
    } as ReturnType<typeof useCramQueueQuery>))
    const props = { onNavigateToCreate: vi.fn(), onNavigateToImport: vi.fn(), reviewDeckId: 1,
      onReviewDeckChange: vi.fn(), isActive: true }
    const view = render(<ReviewTab {...props} />)
    fireEvent.click(screen.getByText("Cram"))
    return { cards, view, props, refresh: (next: Flashcard[], changedProps: Partial<typeof props> = {}) => {
      queue = next
      Object.assign(props, changedProps)
      view.rerender(<ReviewTab {...props} />)
    } }
  }

  const rate = async (front: string, rating = 3) => {
    await screen.findByText(front)
    if (!screen.queryByTestId(`flashcards-review-rate-${rating}`)) {
      fireEvent.click(screen.getByTestId("flashcards-review-show-answer"))
    }
    await act(async () => { fireEvent.click(screen.getByTestId(`flashcards-review-rate-${rating}`)) })
  }

  it.each([false, true])("reaches every intended card after scheduled queue refetch (reordered=%s)", async reordered => {
    const { cards, refresh } = mountQueue()
    fireEvent.click(screen.getByTestId("flashcards-review-cram-update-schedule"))
    await rate("Alpha")
    await screen.findByText("Bravo")
    refresh(reordered ? [cards[1], cards[2], { ...cards[0], due_at: "2030-01-01" }] : cards)
    expect(screen.getByText("Bravo")).toBeInTheDocument()
    expect(screen.getByTestId("flashcards-review-progress")).toHaveTextContent("2 cards remaining, 1 reviewed")
    await rate("Bravo")
    refresh(reordered ? [cards[2], cards[0], cards[1]] : cards)
    await rate("Charlie")
    expect(reviewMutateAsync.mock.calls.map(([input]) => input.cardUuid)).toEqual(cards.map(card => card.uuid))
    await waitFor(() => expect(endMutateAsync).toHaveBeenCalledTimes(1))
    expect(screen.getByText("Cram session complete!")).toBeInTheDocument()
    expect(screen.queryByTestId("flashcards-review-progress")).not.toBeInTheDocument()
  })

  it.each([false, true])("re-rating preserves the remaining queue after refetch (reordered=%s)", async reordered => {
    const { cards, refresh } = mountQueue()
    fireEvent.click(screen.getByTestId("flashcards-review-cram-update-schedule"))
    await rate("Alpha")
    refresh(reordered ? [cards[1], cards[2], cards[0]] : cards)
    fireEvent.click(screen.getByTestId("flashcards-review-undo-rating"))
    await rate("Alpha", 2)
    expect(screen.getByText("Bravo")).toBeInTheDocument()
    expect(screen.getByTestId("flashcards-review-progress")).toHaveTextContent("2 cards remaining, 1 reviewed")
    expect(screen.queryByText("Cram session complete!")).not.toBeInTheDocument()
    expect(endMutateAsync).not.toHaveBeenCalled()
    expect(reviewMutateAsync.mock.calls.map(([input]) => [input.cardUuid, input.rating])).toEqual([
      [cards[0].uuid, 3], [cards[0].uuid, 2],
    ])
  })

  it("retains practice-only progress when scheduling starts and reads fresh pending cards", async () => {
    const { cards, refresh } = mountQueue()
    await rate("Alpha")
    expect(reviewMutateAsync).not.toHaveBeenCalled()
    fireEvent.click(screen.getByTestId("flashcards-review-cram-update-schedule"))
    refresh([cards[2], { ...cards[1], front: "Bravo edited" }, cards[0]])
    await rate("Charlie")
    await rate("Bravo edited")
    expect(reviewMutateAsync.mock.calls.map(([input]) => input.cardUuid)).toEqual([cards[2].uuid, cards[1].uuid])
    await waitFor(() => expect(endMutateAsync).toHaveBeenCalledTimes(1))
    expect(screen.getByText("3 cards practiced in this cram session")).toBeInTheDocument()
  })

  it("uses the same pending cards for progress and completion when a card disappears", async () => {
    const { cards, refresh } = mountQueue()
    fireEvent.click(screen.getByTestId("flashcards-review-cram-update-schedule"))
    await rate("Alpha")
    refresh([cards[2], cards[0]])
    expect(screen.getByTestId("flashcards-review-progress")).toHaveTextContent("1 cards remaining, 1 reviewed")
    await rate("Charlie")
    expect(screen.queryByTestId("flashcards-review-progress")).not.toBeInTheDocument()
    await waitFor(() => expect(endMutateAsync).toHaveBeenCalledTimes(1))
  })

  it("starts practice again with every card pending and zero reviewed", async () => {
    const { cards, refresh } = mountQueue()
    await rate("Alpha")
    refresh([cards[1], cards[2], cards[0]])
    await rate("Bravo")
    await rate("Charlie")
    fireEvent.click(screen.getByTestId("flashcards-review-practice-again"))
    expect(screen.getByTestId("flashcards-review-progress")).toHaveTextContent("3 cards remaining, 0 reviewed")
    expect(screen.getByText("Bravo")).toBeInTheDocument()
    expect(reviewMutateAsync).not.toHaveBeenCalled()
    expect(endMutateAsync).not.toHaveBeenCalled()
  })

  it.each(["deck", "tag", "account"])("clears practiced identities when the %s scope changes", async scope => {
    const { cards, refresh } = mountQueue()
    await rate("Alpha")
    if (scope === "deck") refresh(cards, { reviewDeckId: 2 })
    if (scope === "tag") fireEvent.change(screen.getByTestId("flashcards-review-cram-tag"), { target: { value: "new-tag" } })
    if (scope === "account") await act(async () => { window.dispatchEvent(new Event("tldw:auth-principal-changed")) })
    expect(screen.getByText("Alpha")).toBeInTheDocument()
    expect(screen.getByTestId("flashcards-review-progress")).toHaveTextContent("3 cards remaining, 0 reviewed")
    expect(reviewMutateAsync).not.toHaveBeenCalled()
  })

  it("manual End keeps the remaining cards available for the next scheduled session", async () => {
    mountQueue()
    fireEvent.click(screen.getByTestId("flashcards-review-cram-update-schedule"))
    await rate("Alpha")
    fireEvent.click(screen.getByRole("button", { name: "End Session", exact: true }))
    await waitFor(() => expect(endMutateAsync).toHaveBeenCalledTimes(1))
    expect(screen.getByText("Bravo")).toBeInTheDocument()
    await rate("Bravo")
    expect(screen.getByText("Charlie")).toBeInTheDocument()
    expect(reviewMutateAsync.mock.calls[1][0].reviewSessionId).toBeUndefined()
    expect(screen.getByTestId("flashcards-review-progress")).toHaveTextContent("1 cards remaining, 2 reviewed")
  })
})
