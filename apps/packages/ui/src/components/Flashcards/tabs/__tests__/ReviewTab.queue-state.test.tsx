import { render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

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
  useResetFlashcardSchedulingMutation,
  useReviewAnalyticsSummaryQuery,
  useReviewFlashcardMutation,
  useReviewQuery,
  useUpdateFlashcardMutation
} from "../../hooks"

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
})
