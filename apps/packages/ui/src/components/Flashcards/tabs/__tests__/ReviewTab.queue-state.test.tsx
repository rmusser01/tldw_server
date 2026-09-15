import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { createInstance } from "i18next"
import ICU from "i18next-icu"
import englishOptions from "@/assets/locale/en/option.json"

import { ReviewTab } from "../ReviewTab"
import {
  useCramQueueQuery,
  useDeckDueCountsQuery,
  useDecksQuery,
  useDeleteFlashcardMutation,
  useDueCountsQuery,
  useGlobalFlashcardTagSuggestionsQuery,
  useFlashcardAssistantQuery,
  useFlashcardAssistantRespondMutation,
  useFlashcardShortcuts,
  useHasCardsQuery,
  useNextDueQuery,
  useRecentFlashcardReviewSessionsQuery,
  useResetFlashcardSchedulingMutation,
  useReviewAnalyticsSummaryQuery,
  useReviewFlashcardMutation,
  useReviewQuery,
  useUpdateFlashcardMutation
} from "../../hooks"

const translations = vi.hoisted(() => ({
  actual: null as null | ((key: string, options?: string | Record<string, unknown>) => string)
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?:
        | string
        | {
            defaultValue?: string
            [key: string]: unknown
          }
    ) => {
      if (translations.actual) return translations.actual(key, defaultValueOrOptions)
      if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
      if (defaultValueOrOptions?.defaultValue) {
        return defaultValueOrOptions.defaultValue.replace(/\{\{(\w+)\}\}/g, (_match, token: string) =>
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
  useAntdMessage: () => ({
    success: vi.fn(),
    error: vi.fn(),
    info: vi.fn(),
    warning: vi.fn(),
    loading: vi.fn(),
    open: vi.fn(),
    destroy: vi.fn()
  })
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
  useEndFlashcardReviewSessionMutation: vi.fn(() => ({ mutateAsync: vi.fn(), isPending: false })),
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

const makeCard = (
  queueState: "new" | "learning" | "review" | "relearning" | "suspended",
  suspendedReason: "manual" | "leech" | null = null,
  schedulerType: "sm2_plus" | "fsrs" = "sm2_plus"
) => ({
  uuid: `card-${queueState}`,
  deck_id: 1,
  front: "Question",
  back: "Answer",
  notes: null,
  extra: null,
  is_cloze: false,
  tags: ["biology"],
  ef: 2.5,
  interval_days: 3,
  repetitions: 2,
  lapses: suspendedReason === "leech" ? 8 : 0,
  due_at: null,
  last_reviewed_at: null,
  queue_state: queueState,
  step_index: queueState === "learning" || queueState === "relearning" ? 1 : null,
  suspended_reason: suspendedReason,
  deleted: false,
  client_id: "test",
  version: 2,
  model_type: "basic" as const,
  reverse: false,
  scheduler_type: schedulerType,
  next_intervals: {
    again: "1 min",
    hard: "10 min",
    good: "1 day",
    easy: "4 days"
  }
})

describe("ReviewTab queue state visibility", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    translations.actual = null

    vi.mocked(useDecksQuery).mockReturnValue({
      data: [{ id: 1, name: "Biology", description: null, deleted: false, client_id: "test", version: 1 }],
      isLoading: false
    } as any)
    vi.mocked(useCramQueueQuery).mockReturnValue({ data: [] } as any)
    vi.mocked(useReviewFlashcardMutation).mockReturnValue({ mutateAsync: vi.fn(), isPending: false } as any)
    vi.mocked(useFlashcardAssistantQuery).mockReturnValue({ data: null, isLoading: false, isError: false } as any)
    vi.mocked(useFlashcardAssistantRespondMutation).mockReturnValue({ mutateAsync: vi.fn(), isPending: false } as any)
    vi.mocked(useUpdateFlashcardMutation).mockReturnValue({ mutateAsync: vi.fn(), isPending: false } as any)
    vi.mocked(useResetFlashcardSchedulingMutation).mockReturnValue({ mutateAsync: vi.fn(), isPending: false } as any)
    vi.mocked(useDeleteFlashcardMutation).mockReturnValue({ mutateAsync: vi.fn(), isPending: false } as any)
    vi.mocked(useGlobalFlashcardTagSuggestionsQuery).mockReturnValue({
      data: { items: [] },
      isLoading: false,
      isFetching: false,
      isError: false
    } as any)
    vi.mocked(useFlashcardShortcuts).mockImplementation(() => undefined)
    vi.mocked(useDueCountsQuery).mockReturnValue({
      data: { due: 1, new: 0, learning: 0, total: 1 },
      refetch: vi.fn().mockResolvedValue(undefined)
    } as any)
    vi.mocked(useDeckDueCountsQuery).mockReturnValue({ data: { 1: { due: 1, new: 0, learning: 0, total: 1 } } } as any)
    vi.mocked(useReviewAnalyticsSummaryQuery).mockReturnValue({ data: null, isLoading: false } as any)
    vi.mocked(useHasCardsQuery).mockReturnValue({ data: true } as any)
    vi.mocked(useNextDueQuery).mockReturnValue({ data: null } as any)
    vi.mocked(useRecentFlashcardReviewSessionsQuery).mockReturnValue({
      data: [],
      isLoading: false,
      isError: false,
      error: null,
      refetch: vi.fn().mockResolvedValue(undefined)
    } as any)
  })

  it.each([
    ["new", null, "New"],
    ["learning", null, "Learning"],
    ["review", null, "Review"],
    ["relearning", null, "Relearning"],
    ["suspended", "leech", "Suspended (Leech)"]
  ] as const)("renders the %s queue state badge on the active card", (queueState, suspendedReason, expectedLabel) => {
    vi.mocked(useReviewQuery).mockReturnValue({
      data: makeCard(queueState, suspendedReason),
      refetch: vi.fn().mockResolvedValue(undefined)
    } as any)

    render(
      <ReviewTab
        onNavigateToCreate={() => {}}
        onNavigateToImport={() => {}}
        reviewDeckId={1}
        onReviewDeckChange={() => {}}
        isActive
      />
    )

    expect(screen.getByTestId("flashcards-review-queue-state")).toHaveTextContent(expectedLabel)
    expect(screen.getByTestId("flashcards-review-scheduler-type")).toHaveTextContent("SM-2+")
  })

  it("renders an FSRS scheduler badge when the active card uses the FSRS deck scheduler", () => {
    vi.mocked(useReviewQuery).mockReturnValue({
      data: makeCard("review", null, "fsrs"),
      refetch: vi.fn().mockResolvedValue(undefined)
    } as any)

    render(
      <ReviewTab
        onNavigateToCreate={() => {}}
        onNavigateToImport={() => {}}
        reviewDeckId={1}
        onReviewDeckChange={() => {}}
        isActive
      />
    )

    expect(screen.getByTestId("flashcards-review-scheduler-type")).toHaveTextContent("FSRS")
  })

  it("offers the ready queue without claiming it is complete before review starts", () => {
    vi.mocked(useReviewQuery).mockReturnValue({ data: makeCard("new"), refetch: vi.fn() } as unknown as ReturnType<typeof useReviewQuery>)
    render(<ReviewTab onNavigateToCreate={() => {}} onNavigateToImport={() => {}}
      reviewDeckId={null} onReviewDeckChange={() => {}} isActive />)

    expect(screen.getByRole("button", { name: "Review all due" })).toBeEnabled()
    expect(screen.queryByText("You're all caught up!")).not.toBeInTheDocument()
    expect(screen.queryByText("No cards are due for review. Great job!")).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Review all due" }))
    expect(screen.getByTestId("flashcards-review-active-card")).toHaveTextContent("Question")
  })

  it("does not announce completion while the review queue is loading", () => {
    vi.mocked(useReviewQuery).mockReturnValue({ data: undefined, isLoading: true, refetch: vi.fn() } as unknown as ReturnType<typeof useReviewQuery>)
    render(<ReviewTab onNavigateToCreate={() => {}} onNavigateToImport={() => {}}
      reviewDeckId={1} onReviewDeckChange={() => {}} isActive />)
    expect(screen.queryByText("You're all caught up!")).not.toBeInTheDocument()
  })

  it("announces the refreshed due queue through successive ratings and newly due cards", async () => {
    let remaining = 5
    const mutation = vi.fn(async () => {
      remaining -= 1
      return { review_session_id: 1 }
    })
    vi.mocked(useReviewFlashcardMutation).mockReturnValue({ mutateAsync: mutation, isPending: false } as unknown as ReturnType<typeof useReviewFlashcardMutation>)
    vi.mocked(useReviewQuery).mockImplementation(() => ({
      data: { ...makeCard("review"), uuid: `card-${remaining}` }, refetch: vi.fn()
    } as unknown as ReturnType<typeof useReviewQuery>))
    vi.mocked(useDueCountsQuery).mockImplementation(() => ({
      data: { due: remaining, new: 0, learning: 0, total: remaining }, refetch: vi.fn()
    } as unknown as ReturnType<typeof useDueCountsQuery>))
    const view = render(<ReviewTab onNavigateToCreate={() => {}} onNavigateToImport={() => {}}
      reviewDeckId={1} onReviewDeckChange={() => {}} isActive />)

    for (let reviewed = 1; reviewed <= 3; reviewed += 1) {
      fireEvent.click(screen.getByTestId("flashcards-review-show-answer"))
      fireEvent.click(screen.getByTestId("flashcards-review-rate-3"))
      await waitFor(() => expect(screen.getByTestId("flashcards-review-progress"))
        .toHaveTextContent(`${5 - reviewed} cards remaining, ${reviewed} reviewed`))
    }
    remaining = 4
    view.rerender(<ReviewTab onNavigateToCreate={() => {}} onNavigateToImport={() => {}}
      reviewDeckId={1} onReviewDeckChange={() => {}} isActive />)
    expect(screen.getByTestId("flashcards-review-progress")).toHaveTextContent("4 cards remaining, 3 reviewed")
    expect(mutation).toHaveBeenCalledTimes(3)
  })

  it.each([1, 3])("describes %s future cards as an hour window using the actual English resource", async (count) => {
    const i18n = createInstance()
    await i18n.use(ICU).init({
      lng: "en", fallbackLng: "en", defaultNS: "option",
      resources: { en: { option: englishOptions } }
    })
    translations.actual = (key, options) => String(typeof options === "string" ? i18n.t(key, options) : i18n.t(key, options))
    vi.mocked(useReviewQuery).mockReturnValue({ data: null, refetch: vi.fn() } as unknown as ReturnType<typeof useReviewQuery>)
    vi.mocked(useDueCountsQuery).mockReturnValue({ data: { due: 0, new: 0, learning: 0, total: 0 }, refetch: vi.fn() } as unknown as ReturnType<typeof useDueCountsQuery>)
    vi.mocked(useNextDueQuery).mockReturnValue({ data: {
      nextDueAt: "2026-09-16T16:00:00Z", cardsDue: count, isCapped: false, scanned: count
    } } as unknown as ReturnType<typeof useNextDueQuery>)

    render(<ReviewTab onNavigateToCreate={() => {}} onNavigateToImport={() => {}}
      reviewDeckId={1} onReviewDeckChange={() => {}} isActive />)

    expect(screen.getByText(new RegExp(`${count} ${count === 1 ? "card" : "cards"} due within the following hour`)))
      .toBeInTheDocument()
  })

  it.each([false, true])(
    "describes the limited scan truthfully when a future review was found: %s",
    async (hasFuture) => {
      const i18n = createInstance()
      await i18n.use(ICU).init({
        lng: "en",
        fallbackLng: "en",
        defaultNS: "option",
        resources: { en: { option: englishOptions } }
      })
      translations.actual = (key, options) => String(typeof options === "string" ? i18n.t(key, options) : i18n.t(key, options))
      vi.mocked(useReviewQuery).mockReturnValue({
        data: null,
        refetch: vi.fn()
      } as unknown as ReturnType<typeof useReviewQuery>)
      vi.mocked(useDueCountsQuery).mockReturnValue({
        data: { due: 0, new: 0, learning: 0, total: 0 },
        refetch: vi.fn()
      } as unknown as ReturnType<typeof useDueCountsQuery>)
      vi.mocked(useNextDueQuery).mockReturnValue({
        data: {
          nextDueAt: hasFuture ? "2026-09-16T16:00:00Z" : null,
          cardsDue: hasFuture ? 2000 : 0,
          isCapped: true,
          scanned: 2000
        }
      } as unknown as ReturnType<typeof useNextDueQuery>)

      render(
        <ReviewTab
          onNavigateToCreate={() => {}}
          onNavigateToImport={() => {}}
          reviewDeckId={1}
          onReviewDeckChange={() => {}}
          isActive
        />
      )

      if (hasFuture) {
        expect(
          screen.getByText(/2,000 cards due within the following hour/)
        ).toBeInTheDocument()
        expect(
          screen.queryByText("Next review estimate unavailable")
        ).not.toBeInTheDocument()
      } else {
        expect(
          screen.getByText("Next review estimate unavailable")
        ).toBeInTheDocument()
        expect(
          screen.queryByText(/cards due within the following hour/)
        ).not.toBeInTheDocument()
      }
      expect(
        screen.getByText(
          "Estimate based on the first 2,000 cards. The count may be incomplete. Narrow filters to improve the estimate."
        )
      ).toBeInTheDocument()
      expect(
        screen.queryByText(/Next review is beyond the first/)
      ).not.toBeInTheDocument()
    }
  )
})
