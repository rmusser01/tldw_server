import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { ResultsTab } from "../ResultsTab"
import {
  useAllAttemptsQuery,
  useAttemptQuery,
  useAttemptRemediationConversionsQuery,
  useConvertAttemptRemediationQuestionsMutation,
  useGenerateRemediationQuizMutation,
  useQuizAttemptQuestionAssistantQuery,
  useQuizAttemptQuestionAssistantRespondMutation,
  useQuizzesQuery
} from "../../hooks"
import { useDecksQuery } from "@/components/Flashcards/hooks/useFlashcardQueries"
import { useStudySuggestions } from "@/components/StudySuggestions/hooks/useStudySuggestions"

const mocks = vi.hoisted(() => ({
  convertRemediationQuestions: vi.fn(),
  navigate: vi.fn(),
  remediationQuiz: vi.fn(),
  refreshStudySuggestions: vi.fn(),
  performStudySuggestionAction: vi.fn()
}))

const interpolate = (template: string, values: Record<string, unknown> | undefined) => {
  return template.replace(/\{\{\s*([^\s}]+)\s*\}\}/g, (_, key: string) => {
    const value = values?.[key]
    return value == null ? "" : String(value)
  })
}

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
      const defaultValue = defaultValueOrOptions?.defaultValue
      if (typeof defaultValue === "string") {
        return interpolate(defaultValue, defaultValueOrOptions)
      }
      return key
    }
  })
}))

vi.mock("react-router-dom", async (importOriginal) => {
  const actual = await importOriginal<typeof import("react-router-dom")>()
  return {
    ...actual,
    useNavigate: () => mocks.navigate
  }
})

vi.mock("../../hooks", () => ({
  useAllAttemptsQuery: vi.fn(),
  useQuizzesQuery: vi.fn(),
  useAttemptQuery: vi.fn(),
  useAttemptRemediationConversionsQuery: vi.fn(),
  useConvertAttemptRemediationQuestionsMutation: vi.fn(),
  useGenerateRemediationQuizMutation: vi.fn(),
  useQuizAttemptQuestionAssistantQuery: vi.fn(),
  useQuizAttemptQuestionAssistantRespondMutation: vi.fn()
}))

vi.mock("@/components/Flashcards/hooks/useFlashcardQueries", () => ({
  useDecksQuery: vi.fn()
}))

vi.mock("@/components/StudySuggestions/hooks/useStudySuggestions", () => ({
  useStudySuggestions: vi.fn()
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

if (!(globalThis as any).ResizeObserver) {
  ;(globalThis as any).ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
}

const readySnapshot = {
  snapshot: {
    id: 88,
    service: "quiz",
    activity_type: "quiz_attempt",
    anchor_type: "quiz_attempt",
    anchor_id: 101,
    suggestion_type: "study_suggestions",
    status: "active",
    payload: {
      summary: {
        score: 1,
        correct_count: 1,
        total_count: 3
      },
      topics: [
        {
          id: "topic-1",
          display_label: "Renal basics",
          type: "grounded",
          status: "weakness",
          selected: true
        }
      ]
    },
    user_selection: {
      selected_topic_ids: ["topic-1"]
    },
    refreshed_from_snapshot_id: null,
    created_at: "2026-04-05T18:00:00Z",
    last_modified: "2026-04-05T18:00:00Z"
  },
  live_evidence: {
    "topic-1": {
      source_available: true,
      source_type: "note",
      source_id: "note-7"
    }
  }
}

const configureBaseQueries = () => {
  vi.mocked(useAllAttemptsQuery).mockReturnValue({
    data: {
      items: [
        {
          id: 101,
          quiz_id: 7,
          started_at: "2026-03-13T09:00:00Z",
          completed_at: "2026-03-13T09:12:00Z",
          score: 1,
          total_possible: 3,
          time_spent_seconds: 720,
          answers: []
        }
      ],
      count: 1
    },
    isLoading: false
  } as any)

  vi.mocked(useQuizzesQuery).mockReturnValue({
    data: {
      items: [
        {
          id: 7,
          name: "Renal Physiology",
          total_questions: 3,
          media_id: 42
        }
      ],
      count: 1
    },
    isLoading: false
  } as any)

  vi.mocked(useAttemptQuery).mockReturnValue({
    data: {
      id: 101,
      quiz_id: 7,
      started_at: "2026-03-13T09:00:00Z",
      completed_at: "2026-03-13T09:12:00Z",
      score: 1,
      total_possible: 3,
      time_spent_seconds: 720,
      answers: [
        {
          question_id: 12,
          user_answer: "Bowman capsule",
          is_correct: false,
          correct_answer: "Glomerulus",
          explanation: "Filtration happens at the glomerulus."
        },
        {
          question_id: 13,
          user_answer: "ADH",
          is_correct: true,
          correct_answer: "ADH",
          explanation: "Correct."
        }
      ],
      questions: [
        {
          id: 12,
          quiz_id: 7,
          question_type: "multiple_choice",
          question_text: "Which structure filters blood?",
          options: ["Bowman capsule", "Glomerulus", "Loop of Henle"],
          points: 1,
          order_index: 0,
          deleted: false,
          client_id: "test",
          version: 1
        },
        {
          id: 13,
          quiz_id: 7,
          question_type: "multiple_choice",
          question_text: "Which hormone raises water reabsorption?",
          options: ["ADH", "Insulin", "Thyroxine"],
          points: 1,
          order_index: 1,
          deleted: false,
          client_id: "test",
          version: 1
        }
      ]
    },
    isLoading: false,
    isFetching: false
  } as any)

  vi.mocked(useAttemptRemediationConversionsQuery).mockReturnValue({
    data: {
      attempt_id: 101,
      items: [],
      count: 0,
      superseded_count: 0
    },
    isLoading: false
  } as any)

  vi.mocked(useConvertAttemptRemediationQuestionsMutation).mockReturnValue({
    mutateAsync: mocks.convertRemediationQuestions,
    isPending: false
  } as any)

  vi.mocked(useGenerateRemediationQuizMutation).mockReturnValue({
    mutateAsync: mocks.remediationQuiz,
    isPending: false
  } as any)

  vi.mocked(useQuizAttemptQuestionAssistantQuery).mockReturnValue({
    data: null,
    isLoading: false,
    isError: false
  } as any)

  vi.mocked(useQuizAttemptQuestionAssistantRespondMutation).mockReturnValue({
    mutateAsync: vi.fn(),
    isPending: false
  } as any)

  vi.mocked(useDecksQuery).mockReturnValue({
    data: [],
    isLoading: false
  } as any)
}

describe("ResultsTab study suggestions", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    window.sessionStorage.clear()
    configureBaseQueries()

    mocks.convertRemediationQuestions.mockResolvedValue({
      attempt_id: 101,
      quiz_id: 7,
      target_deck: { id: 3, name: "Renal Recovery Deck" },
      results: [],
      created_flashcard_uuids: []
    })
    mocks.remediationQuiz.mockResolvedValue({
      quiz: { id: 55, name: "Quiz: Remediation" },
      questions: []
    })
    mocks.refreshStudySuggestions.mockResolvedValue(undefined)
    mocks.performStudySuggestionAction.mockResolvedValue({
      disposition: "generated",
      snapshot_id: 88,
      selection_fingerprint: "fingerprint",
      target_service: "quiz",
      target_type: "quiz",
      target_id: "quiz-19"
    })

    vi.mocked(useStudySuggestions).mockReturnValue({
      status: "ready",
      statusQuery: {} as any,
      snapshot: readySnapshot,
      activeSnapshotId: 88,
      isLoading: false,
      isRefreshing: false,
      refresh: mocks.refreshStudySuggestions,
      performAction: mocks.performStudySuggestionAction
    } as any)
  })

  it("shows the study-suggestions loading state when an attempt is selected", async () => {
    vi.mocked(useAttemptQuery).mockReturnValue({
      data: null,
      isLoading: true,
      isFetching: false
    } as any)

    vi.mocked(useStudySuggestions).mockReturnValue({
      status: "pending",
      statusQuery: {} as any,
      snapshot: null,
      activeSnapshotId: null,
      isLoading: true,
      isRefreshing: false,
      refresh: mocks.refreshStudySuggestions,
      performAction: mocks.performStudySuggestionAction
    } as any)

    render(<ResultsTab />)

    fireEvent.click(screen.getByRole("button", { name: /View Details/i }))

    expect(useStudySuggestions).toHaveBeenCalledWith("quiz_attempt", 101)
    expect(await screen.findByText("Loading study suggestions...")).toBeInTheDocument()
    expect(screen.getByTestId("results-detail-loading-skeleton")).toBeInTheDocument()
  })

  it("renders quiz study suggestions with summary, topic builder, and both actions, then routes action results", async () => {
    mocks.performStudySuggestionAction
      .mockResolvedValueOnce({
        disposition: "generated",
        snapshot_id: 88,
        selection_fingerprint: "fingerprint-quiz",
        target_service: "quiz",
        target_type: "quiz",
        target_id: "quiz-19"
      })
      .mockResolvedValueOnce({
        disposition: "opened_existing",
        snapshot_id: 88,
        selection_fingerprint: "fingerprint-flashcards",
        target_service: "flashcards",
        target_type: "deck",
        target_id: "deck-44"
      })

    render(<ResultsTab />)

    fireEvent.click(screen.getByRole("button", { name: /View Details/i }))

    expect(await screen.findByText("Study suggestions")).toBeInTheDocument()
    expect(screen.getByText("Score")).toBeInTheDocument()
    expect(screen.getByDisplayValue("Renal basics")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Create quiz" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Create flashcards" })).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Create quiz" }))

    await waitFor(() => {
      expect(mocks.performStudySuggestionAction).toHaveBeenCalledWith(
        expect.objectContaining({
          targetService: "quiz",
          targetType: "quiz",
          actionKind: "follow_up_quiz",
          selectedTopicIds: ["topic-1"],
          selectedTopicEdits: [{ id: "topic-1", label: "Renal basics" }]
        })
      )
      expect(mocks.navigate).toHaveBeenCalledWith("/quiz?tab=take&start_quiz_id=19&highlight_quiz_id=19")
    })

    fireEvent.click(screen.getByRole("button", { name: "Create flashcards" }))

    await waitFor(() => {
      expect(mocks.performStudySuggestionAction).toHaveBeenLastCalledWith(
        expect.objectContaining({
          targetService: "flashcards",
          targetType: "deck",
          actionKind: "follow_up_flashcards",
          selectedTopicIds: ["topic-1"],
          selectedTopicEdits: [{ id: "topic-1", label: "Renal basics" }]
        })
      )
      expect(mocks.navigate).toHaveBeenLastCalledWith(
        "/flashcards?tab=review&study_source=quiz&quiz_id=7&attempt_id=101&deck_id=44&include_workspace_items=1"
      )
    })
  })

  it("shows retry for failed study suggestions without hiding remediation controls", async () => {
    vi.mocked(useStudySuggestions).mockReturnValue({
      status: "failed",
      statusQuery: {} as any,
      snapshot: null,
      activeSnapshotId: null,
      isLoading: false,
      isRefreshing: false,
      refresh: mocks.refreshStudySuggestions,
      performAction: mocks.performStudySuggestionAction
    } as any)

    render(<ResultsTab />)

    fireEvent.click(screen.getByRole("button", { name: /View Details/i }))

    expect(await screen.findByRole("button", { name: /Retry/i })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /Create Remediation Quiz/i })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /Create Flashcards from Missed Questions/i })).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: /Retry/i }))

    await waitFor(() => {
      expect(mocks.refreshStudySuggestions).toHaveBeenCalledTimes(1)
    })
  })
})
