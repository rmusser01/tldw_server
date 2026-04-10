import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { ReviewTab } from "../ReviewTab"
import { clearSetting } from "@/services/settings/registry"
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
  useRecentFlashcardReviewSessionsQuery,
  useFlashcardAssistantQuery,
  useFlashcardAssistantRespondMutation,
  useUpdateFlashcardMutation,
  useResetFlashcardSchedulingMutation,
  useDeleteFlashcardMutation,
  useGlobalFlashcardTagSuggestionsQuery,
  useFlashcardShortcuts,
  useDebouncedFormField,
  useDueCountsQuery,
  useDeckDueCountsQuery,
  useReviewAnalyticsSummaryQuery,
  useHasCardsQuery,
  useNextDueQuery
} from "../../hooks"

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

describe("ReviewTab analytics summary", () => {
  beforeEach(async () => {
    vi.clearAllMocks()
    await clearSetting(FLASHCARDS_SHORTCUT_HINT_DENSITY_SETTING)
    await clearSetting(FLASHCARDS_REVIEW_ONBOARDING_DISMISSED_SETTING)
    vi.mocked(useDecksQuery).mockReturnValue({
      data: [{ id: 9, name: "Biology" }],
      isLoading: false
    } as any)
    vi.mocked(useCramQueueQuery).mockReturnValue({ data: [] } as any)
    vi.mocked(useReviewQuery).mockReturnValue({ data: null } as any)
    vi.mocked(useReviewFlashcardMutation).mockReturnValue({
      mutateAsync: vi.fn()
    } as any)
    vi.mocked(useEndFlashcardReviewSessionMutation).mockReturnValue({
      mutateAsync: vi.fn(),
      isPending: false
    } as any)
    vi.mocked(useRecentFlashcardReviewSessionsQuery).mockReturnValue({
      data: [],
      isLoading: false,
      isError: false,
      error: null,
      refetch: vi.fn()
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
      data: { due: 0, new: 0, learning: 0, total: 0 }
    } as any)
    vi.mocked(useDeckDueCountsQuery).mockReturnValue({ data: {} } as any)
    vi.mocked(useHasCardsQuery).mockReturnValue({ data: false } as any)
    vi.mocked(useNextDueQuery).mockReturnValue({ data: null } as any)
    vi.mocked(useReviewAnalyticsSummaryQuery).mockReturnValue({
      data: {
        reviewed_today: 12,
        retention_rate_today: 87.5,
        lapse_rate_today: 12.5,
        avg_answer_time_ms_today: 1850,
        study_streak_days: 6,
        generated_at: "2026-02-18T12:00:00.000Z",
        decks: [
          {
            deck_id: 9,
            deck_name: "Biology",
            total: 40,
            new: 8,
            learning: 10,
            due: 6,
            mature: 22
          }
        ]
      },
      isLoading: false
    } as any)
  })

  it("renders analytics metrics and deck progress cards", () => {
    render(
      <ReviewTab
        onNavigateToCreate={() => {}}
        onNavigateToImport={() => {}}
        reviewDeckId={9}
        onReviewDeckChange={() => {}}
        isActive
      />
    )

    expect(useReviewAnalyticsSummaryQuery).toHaveBeenCalledWith(9, undefined)
    expect(screen.getByTestId("flashcards-review-analytics-summary")).toBeInTheDocument()
    expect(screen.getByText("Reviewed today")).toBeInTheDocument()
    expect(screen.getByText("12")).toBeInTheDocument()
    expect(screen.getByText("87.5%")).toBeInTheDocument()
    expect(screen.getByText("Lapse rate")).toBeInTheDocument()
    expect(screen.getByText("12.5%")).toBeInTheDocument()
    expect(screen.getByText("1.9s")).toBeInTheDocument()
    expect(screen.getByText("6 days")).toBeInTheDocument()
    expect(screen.getByText("Deck progress")).toBeInTheDocument()
    expect(screen.getAllByText("Biology").length).toBeGreaterThan(0)
    expect(screen.getByText("Due: 6")).toBeInTheDocument()
    expect(screen.getByText("Mature: 22")).toBeInTheDocument()
  })

  it("renders a fallback for missing lapse rate", () => {
    vi.mocked(useReviewAnalyticsSummaryQuery).mockReturnValue({
      data: {
        reviewed_today: 4,
        retention_rate_today: 75,
        lapse_rate_today: null,
        avg_answer_time_ms_today: null,
        study_streak_days: 2,
        generated_at: "2026-02-18T12:00:00.000Z",
        decks: [
          {
            deck_id: 9,
            deck_name: "Biology",
            total: 10,
            new: 2,
            learning: 2,
            due: 2,
            mature: 4
          }
        ]
      },
      isLoading: false
    } as any)

    render(
      <ReviewTab
        onNavigateToCreate={() => {}}
        onNavigateToImport={() => {}}
        reviewDeckId={9}
        onReviewDeckChange={() => {}}
        isActive
      />
    )

    const lapseRateLabel = screen.getByText("Lapse rate")
    expect(lapseRateLabel.parentElement).not.toBeNull()
    expect(lapseRateLabel.parentElement).toHaveTextContent("—")
  })

  it("shows plain-language rating interval guidance in the review action area", () => {
    vi.mocked(useReviewQuery).mockReturnValue({
      data: {
        uuid: "card-1",
        deck_id: 9,
        front: "Front text",
        back: "Back text",
        notes: null,
        extra: null,
        is_cloze: false,
        tags: [],
        ef: 2.5,
        interval_days: 3,
        repetitions: 2,
        lapses: 0,
        due_at: null,
        last_reviewed_at: null,
        last_modified: null,
        deleted: false,
        client_id: "test",
        version: 1,
        model_type: "basic",
        reverse: false
      }
    } as any)
    vi.mocked(useDueCountsQuery).mockReturnValue({
      data: { due: 1, new: 0, learning: 0, total: 1 }
    } as any)

    render(
      <ReviewTab
        onNavigateToCreate={() => {}}
        onNavigateToImport={() => {}}
        reviewDeckId={9}
        onReviewDeckChange={() => {}}
        isActive
      />
    )

    fireEvent.click(screen.getByTestId("flashcards-review-show-answer"))
    expect(
      screen.getByText(
        "Again = shortest gap, Hard = short gap, Good = medium gap, Easy = longest gap."
      )
    ).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Why these ratings?" }))
    expect(screen.getByText("Again = 0: forgot it, repeat very soon.")).toBeInTheDocument()
    expect(screen.getByText("Hard = 2: remembered with strain, keep gap short.")).toBeInTheDocument()
    expect(
      screen.getByText("Good = 3: normal recall, use the default schedule step.")
    ).toBeInTheDocument()
    expect(
      screen.getByText("Easy = 5: effortless recall, jump to a longer gap.")
    ).toBeInTheDocument()
  })
})
