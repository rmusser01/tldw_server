// @vitest-environment jsdom
import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { ReviewTab } from "../ReviewTab"
import { StudySuggestionsPanel } from "@/components/StudySuggestions/StudySuggestionsPanel"
import { useStudySuggestions } from "@/components/StudySuggestions/hooks/useStudySuggestions"
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

const reviewMutationMock = vi.hoisted(() => vi.fn())
const endSessionMutationMock = vi.hoisted(() => vi.fn())
const assistantRefetchMock = vi.hoisted(() => vi.fn())
const navigateMock = vi.hoisted(() => vi.fn())
const studySuggestionActionMock = vi.hoisted(() => vi.fn())
const studySuggestionRefreshMock = vi.hoisted(() => vi.fn())

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
    useNavigate: () => navigateMock
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
  useNextDueQuery: vi.fn(),
  useEndFlashcardReviewSessionMutation: vi.fn(),
  useRecentFlashcardReviewSessionsQuery: vi.fn(),
  useGlobalFlashcardTagSuggestionsQuery: vi.fn()
}))

vi.mock("@/components/StudySuggestions/hooks/useStudySuggestions", () => ({
  useStudySuggestions: vi.fn()
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

const buildCard = () => ({
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
  scheduler_type: "sm2_plus",
  next_intervals: {
    again: "1 min",
    hard: "10 min",
    good: "1 day",
    easy: "4 days"
  }
})

const buildSessionSnapshot = (overrides: Partial<Record<string, unknown>> = {}) => ({
  snapshot: {
    id: 88,
    service: "flashcards",
    activity_type: "flashcard_review_session",
    anchor_type: "flashcard_review_session",
    anchor_id: 77,
    suggestion_type: "study_suggestions",
    status: "active",
    payload: {
      summary: {
        cards_reviewed: 4,
        correct_count: 3
      },
      topics: [
        {
          id: "topic-1",
          display_label: "Dialysis overview",
          type: "derived",
          status: "adjacent",
          selected: true,
          source_type: "note"
        },
        {
          id: "topic-2",
          display_label: "Kidney basics",
          type: "derived",
          status: "exploratory",
          selected: true,
          source_type: "note"
        }
      ]
    },
    user_selection: {
      selected_topic_ids: ["topic-1", "topic-2"]
    },
    refreshed_from_snapshot_id: null,
    created_at: "2026-04-05T18:00:00Z",
    last_modified: "2026-04-05T18:00:00Z",
    ...overrides
  },
  live_evidence: {
    "topic-1": {
      source_available: true,
      source_type: "note",
      source_id: "note-7"
    },
    "topic-2": {
      source_available: true,
      source_type: "note",
      source_id: "note-8"
    }
  }
})

const buildExploratorySessionSnapshot = () => ({
  snapshot: {
    id: 99,
    service: "flashcards",
    activity_type: "flashcard_review_session",
    anchor_type: "flashcard_review_session",
    anchor_id: 77,
    suggestion_type: "study_suggestions",
    status: "active",
    payload: {
      summary: {
        cards_reviewed: 4,
        correct_count: 2
      },
      topics: [
        {
          id: "topic-1",
          display_label: "Dialysis overview",
          type: "derived",
          status: "adjacent",
          selected: true,
          source_type: "note"
        },
        {
          id: "topic-2",
          display_label: "Self-directed follow-up",
          type: "derived",
          status: "exploratory",
          selected: true,
          source_type: "note"
        }
      ]
    },
    user_selection: {
      selected_topic_ids: ["topic-1", "topic-2"]
    },
    refreshed_from_snapshot_id: null,
    created_at: "2026-04-05T19:00:00Z",
    last_modified: "2026-04-05T19:00:00Z"
  },
  live_evidence: {
    "topic-1": {
      source_available: false,
      source_type: "note",
      source_id: "note-7"
    },
    "topic-2": {
      source_available: false,
      source_type: "note",
      source_id: "note-8"
    }
  }
})

const buildStudySuggestionsResult = (
  snapshot: ReturnType<typeof buildSessionSnapshot> | ReturnType<typeof buildExploratorySessionSnapshot>
) => ({
  status: "ready",
  snapshot,
  activeSnapshotId: snapshot.snapshot.id,
  isLoading: false,
  isRefreshing: false,
  refresh: studySuggestionRefreshMock,
  performAction: studySuggestionActionMock
})

describe("ReviewTab study suggestions", () => {
  const studySuggestionsByAnchorId = new Map<number, ReturnType<typeof buildStudySuggestionsResult>>()

  beforeEach(() => {
    vi.clearAllMocks()
    studySuggestionsByAnchorId.clear()

    reviewMutationMock.mockResolvedValue({
      review_session_id: 77,
      due_at: null,
      interval_days: 1,
      next_intervals: {
        again: "1 min",
        hard: "10 min",
        good: "1 day",
        easy: "4 days"
      }
    })
    endSessionMutationMock.mockResolvedValue({
      id: 77,
      deck_id: 1,
      review_mode: "due",
      tag_filter: null,
      scope_key: "due:deck:1",
      status: "completed",
      started_at: "2026-04-05T18:00:00Z",
      last_activity_at: "2026-04-05T18:10:00Z",
      completed_at: "2026-04-05T18:12:00Z",
      client_id: "test"
    })

    vi.mocked(useDecksQuery).mockReturnValue({
      data: [{ id: 1, name: "Biology" }],
      isLoading: false
    } as any)
    vi.mocked(useCramQueueQuery).mockReturnValue({ data: [] } as any)
    vi.mocked(useReviewQuery).mockReturnValue({
      data: buildCard(),
      refetch: vi.fn().mockResolvedValue(undefined)
    } as any)
    vi.mocked(useReviewFlashcardMutation).mockReturnValue({
      mutateAsync: reviewMutationMock,
      isPending: false
    } as any)
    vi.mocked(useEndFlashcardReviewSessionMutation).mockReturnValue({
      mutateAsync: endSessionMutationMock,
      isPending: false
    } as any)
    vi.mocked(useFlashcardAssistantQuery).mockReturnValue({
      data: {
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
        messages: [],
        context_snapshot: {},
        available_actions: ["explain", "mnemonic", "follow_up", "fact_check", "freeform"]
      },
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
    vi.mocked(useRecentFlashcardReviewSessionsQuery).mockReturnValue({
      data: [],
      isLoading: false,
      isFetching: false
    } as any)
    vi.mocked(useStudySuggestions).mockImplementation((anchorType, anchorId) => {
      if (anchorType === "flashcard_review_session" && anchorId != null) {
        return (
          studySuggestionsByAnchorId.get(anchorId) ??
          buildStudySuggestionsResult(buildSessionSnapshot())
        ) as never
      }
      return {
        status: "none",
        statusQuery: { data: null },
        snapshot: null,
        activeSnapshotId: null,
        isLoading: false,
        isRefreshing: false,
        refresh: studySuggestionRefreshMock,
        performAction: studySuggestionActionMock
      } as never
    })
    studySuggestionsByAnchorId.set(77, buildStudySuggestionsResult(buildSessionSnapshot()))
    studySuggestionsByAnchorId.set(
      81,
      buildStudySuggestionsResult(
        buildSessionSnapshot({
          id: 181,
          anchor_id: 81
        })
      )
    )
    studySuggestionsByAnchorId.set(99, buildStudySuggestionsResult(buildExploratorySessionSnapshot()))
  })

  it("stores the active review_session_id returned by review submissions", async () => {
    render(
      <ReviewTab
        onNavigateToCreate={() => {}}
        onNavigateToImport={() => {}}
        reviewDeckId={1}
        onReviewDeckChange={() => {}}
        isActive
      />
    )

    fireEvent.click(screen.getByTestId("flashcards-review-show-answer"))
    fireEvent.click(screen.getByTestId("flashcards-review-rate-3"))

    await waitFor(() => {
      expect(reviewMutationMock).toHaveBeenCalledWith(
        expect.objectContaining({
          cardUuid: "card-1",
          rating: 3
        })
      )
    })

    expect(screen.getByRole("button", { name: /End Session/i })).toBeInTheDocument()
  })

  it("clicking End Session completes the session and reveals the panel", async () => {
    render(
      <ReviewTab
        onNavigateToCreate={() => {}}
        onNavigateToImport={() => {}}
        reviewDeckId={1}
        onReviewDeckChange={() => {}}
        isActive
      />
    )

    fireEvent.click(screen.getByTestId("flashcards-review-show-answer"))
    fireEvent.click(screen.getByTestId("flashcards-review-rate-3"))

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /End Session/i })).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole("button", { name: /End Session/i }))

    await waitFor(() => {
      expect(endSessionMutationMock).toHaveBeenCalledWith(77)
    })

    expect(await screen.findByText("Flashcard follow-up")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /Create flashcards/i })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /Create quiz/i })).toBeInTheDocument()
  })

  it("queue exhaustion auto-calls the end-session path once", async () => {
    const reviewQueryState = {
      data: buildCard(),
      refetch: vi.fn().mockResolvedValue(undefined)
    }
    vi.mocked(useReviewQuery).mockImplementation(() => reviewQueryState as any)

    const { rerender } = render(
      <ReviewTab
        onNavigateToCreate={() => {}}
        onNavigateToImport={() => {}}
        reviewDeckId={1}
        onReviewDeckChange={() => {}}
        isActive
      />
    )

    fireEvent.click(screen.getByTestId("flashcards-review-show-answer"))
    fireEvent.click(screen.getByTestId("flashcards-review-rate-3"))

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /End Session/i })).toBeInTheDocument()
    })

    reviewQueryState.data = null
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
      expect(endSessionMutationMock).toHaveBeenCalledTimes(1)
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
      expect(endSessionMutationMock).toHaveBeenCalledTimes(1)
    })
  })

  it("recent study sessions reopen the linked snapshot from ReviewTab", async () => {
    vi.mocked(useRecentFlashcardReviewSessionsQuery).mockReturnValue({
      data: [
        {
          id: 81,
          deck_id: 1,
          review_mode: "due",
          tag_filter: null,
          scope_key: "due:deck:1",
          status: "completed",
          started_at: "2026-04-05T17:00:00Z",
          last_activity_at: "2026-04-05T17:10:00Z",
          completed_at: "2026-04-05T17:12:00Z",
          client_id: "test"
        }
      ],
      isLoading: false,
      isFetching: false
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

    fireEvent.click(screen.getByRole("button", { name: /Reopen snapshot for session 81/i }))

    await waitFor(() => {
      expect(useStudySuggestions).toHaveBeenCalledWith("flashcard_review_session", 81)
    })

    expect(screen.getByText("Flashcard follow-up")).toBeInTheDocument()
  })

  it("ends the active session when the review scope changes", async () => {
    const { rerender } = render(
      <ReviewTab
        onNavigateToCreate={() => {}}
        onNavigateToImport={() => {}}
        reviewDeckId={1}
        onReviewDeckChange={() => {}}
        isActive
      />
    )

    fireEvent.click(screen.getByTestId("flashcards-review-show-answer"))
    fireEvent.click(screen.getByTestId("flashcards-review-rate-3"))

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /End Session/i })).toBeInTheDocument()
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
      expect(endSessionMutationMock).toHaveBeenCalledWith(77)
    })
  })

  it("reopens generated flashcard follow-ups in the review surface", async () => {
    const onReviewDeckChange = vi.fn()
    studySuggestionActionMock.mockResolvedValueOnce({
      disposition: "generated",
      snapshot_id: 88,
      selection_fingerprint: "fingerprint-flashcards",
      target_service: "flashcards",
      target_type: "deck",
      target_id: "deck-44"
    })

    render(
      <ReviewTab
        onNavigateToCreate={() => {}}
        onNavigateToImport={() => {}}
        reviewDeckId={1}
        onReviewDeckChange={onReviewDeckChange}
        isActive
      />
    )

    fireEvent.click(screen.getByTestId("flashcards-review-show-answer"))
    fireEvent.click(screen.getByTestId("flashcards-review-rate-3"))

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /End Session/i })).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole("button", { name: /End Session/i }))
    expect(await screen.findByText("Flashcard follow-up")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Create flashcards" }))

    await waitFor(() => {
      expect(studySuggestionActionMock).toHaveBeenCalledWith(
        expect.objectContaining({
          targetService: "flashcards",
          targetType: "deck",
          actionKind: "follow_up_flashcards"
        })
      )
      expect(onReviewDeckChange).toHaveBeenCalledWith(44)
    })

    expect(navigateMock).not.toHaveBeenCalled()
  })

  it("routes generated quiz follow-ups through the flashcards handoff", async () => {
    studySuggestionActionMock.mockResolvedValueOnce({
      disposition: "generated",
      snapshot_id: 88,
      selection_fingerprint: "fingerprint-quiz",
      target_service: "quiz",
      target_type: "quiz",
      target_id: "quiz-19"
    })

    render(
      <ReviewTab
        onNavigateToCreate={() => {}}
        onNavigateToImport={() => {}}
        reviewDeckId={1}
        onReviewDeckChange={() => {}}
        isActive
      />
    )

    fireEvent.click(screen.getByTestId("flashcards-review-show-answer"))
    fireEvent.click(screen.getByTestId("flashcards-review-rate-3"))

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /End Session/i })).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole("button", { name: /End Session/i }))
    expect(await screen.findByText("Flashcard follow-up")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Create quiz" }))

    await waitFor(() => {
      expect(studySuggestionActionMock).toHaveBeenCalledWith(
        expect.objectContaining({
          targetService: "quiz",
          targetType: "quiz",
          actionKind: "follow_up_quiz"
        })
      )
      expect(navigateMock).toHaveBeenCalledWith(
        "/quiz?tab=take&source=flashcards&start_quiz_id=19&highlight_quiz_id=19&deck_id=1&deck_name=Biology"
      )
    })
  })

  it("exploratory-only sessions use weaker copy and suppress source-aware adjacency claims", () => {
    render(<StudySuggestionsPanel anchorType="flashcard_review_session" anchorId={99} />)

    expect(screen.getByText("Exploratory follow-up")).toBeInTheDocument()
    expect(screen.queryByText("Adjacent")).not.toBeInTheDocument()
    expect(screen.getAllByText("Evidence: Exploratory").length).toBeGreaterThan(0)
    expect(screen.getByRole("button", { name: /Create flashcards/i })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /Create quiz/i })).toBeInTheDocument()
  })
})
