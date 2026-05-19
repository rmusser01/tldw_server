import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { ReviewTab } from "../ReviewTab"
import {
  useCramQueueQuery,
  useDeckDueCountsQuery,
  useDecksQuery,
  useDeleteFlashcardMutation,
  useDueCountsQuery,
  useEndFlashcardReviewSessionMutation,
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

const reviewMutationMock = vi.hoisted(() => vi.fn())
const assistantRefetchMock = vi.hoisted(() => vi.fn())

const buildAssistantContext = (overrides: Record<string, unknown> = {}) => ({
  thread: {
    id: 9,
    context_type: "flashcard",
    flashcard_uuid: "card-1",
    quiz_attempt_id: null,
    question_id: null,
    last_message_at: "2026-03-13T08:00:00Z",
    message_count: 1,
    deleted: false,
    client_id: "test",
    version: 1,
    created_at: "2026-03-13T08:00:00Z",
    last_modified: "2026-03-13T08:00:00Z"
  },
  messages: [
    {
      id: 12,
      thread_id: 9,
      role: "assistant",
      action_type: "explain",
      input_modality: "text",
      content: "Earlier explanation",
      structured_payload: {},
      context_snapshot: {},
      provider: "openai",
      model: "gpt-5",
      created_at: "2026-03-13T08:00:01Z",
      client_id: "test"
    }
  ],
  context_snapshot: {},
  available_actions: ["explain", "mnemonic", "follow_up", "fact_check", "freeform"],
  ...overrides
})

describe("ReviewTab study-pack remediation", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    reviewMutationMock.mockResolvedValue({
      due_at: null,
      interval_days: 1
    })

    vi.mocked(useDecksQuery).mockReturnValue({
      data: [{ id: 1, name: "Biology" }],
      isLoading: false
    } as any)
    vi.mocked(useCramQueueQuery).mockReturnValue({ data: [] } as any)
    vi.mocked(useReviewQuery).mockReturnValue({
      data: {
        uuid: "card-1",
        deck_id: 1,
        front: "What filters blood?",
        back: "The glomerulus.",
        notes: null,
        extra: null,
        is_cloze: false,
        tags: ["renal"],
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
        source_ref_type: "note",
        source_ref_id: "88",
        next_intervals: {
          again: "1 min",
          hard: "10 min",
          good: "1 day",
          easy: "4 days"
        }
      }
    } as any)
    vi.mocked(useReviewFlashcardMutation).mockReturnValue({
      mutateAsync: reviewMutationMock,
      isPending: false
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
    vi.mocked(useEndFlashcardReviewSessionMutation).mockReturnValue({
      mutateAsync: vi.fn(),
      isPending: false
    } as any)
    vi.mocked(useRecentFlashcardReviewSessionsQuery).mockReturnValue({
      data: [],
      isLoading: false
    } as any)
  })

  it("renders citation-backed remediation, deep-dive access, and keeps review controls usable", async () => {
    vi.mocked(useFlashcardAssistantQuery).mockReturnValue({
      data: buildAssistantContext({
        citations: [
          {
            id: 1,
            flashcard_uuid: "card-1",
            source_type: "note",
            source_id: "88",
            citation_text: "Cells rely on ATP to power active transport.",
            locator: "{\"section\":\"membrane-transport\"}",
            ordinal: 0,
            deleted: false,
            client_id: "test",
            version: 1
          }
        ],
        primary_citation: {
          id: 1,
          flashcard_uuid: "card-1",
          source_type: "note",
          source_id: "88",
          citation_text: "Cells rely on ATP to power active transport.",
          locator: "{\"section\":\"membrane-transport\"}",
          ordinal: 0,
          deleted: false,
          client_id: "test",
          version: 1
        },
        deep_dive_target: {
          source_type: "note",
          source_id: "88",
          citation_ordinal: 0,
          route_kind: "exact_locator",
          route: "/notes/88?section=membrane-transport",
          available: true,
          fallback_reason: null
        }
      }),
      isLoading: false,
      isError: false,
      refetch: assistantRefetchMock
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

    expect(
      screen.getByText(/Cells rely on ATP to power active transport\./)
    ).toBeInTheDocument()
    expect(
      screen.getByRole("link", { name: "Deep dive to source" })
    ).toHaveAttribute("href", "/notes/88?section=membrane-transport")

    fireEvent.click(screen.getByRole("button", { name: "Show answer (Space)" }))
    const goodButton = screen.getByTestId("flashcards-review-rate-3")
    expect(goodButton).toBeEnabled()
    fireEvent.click(goodButton)

    await waitFor(() => {
      expect(reviewMutationMock).toHaveBeenCalledWith({
        cardUuid: "card-1",
        rating: 3,
        answerTimeMs: expect.any(Number)
      })
    })
  })

  it("falls back to legacy source metadata when citation remediation is unavailable", () => {
    vi.mocked(useFlashcardAssistantQuery).mockReturnValue({
      data: buildAssistantContext({
        citations: [],
        primary_citation: null,
        deep_dive_target: null
      }),
      isLoading: false,
      isError: false,
      refetch: assistantRefetchMock
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

    expect(screen.getByRole("link", { name: "Note #88" })).toHaveAttribute(
      "href",
      "/notes?source_ref_id=88"
    )
  })
})
