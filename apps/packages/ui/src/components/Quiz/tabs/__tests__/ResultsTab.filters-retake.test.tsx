import { fireEvent, render, screen } from "@testing-library/react"
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
import {
  useCreateDeckMutation,
  useCreateFlashcardMutation,
  useDecksQuery
} from "@/components/Flashcards/hooks/useFlashcardQueries"

const navigationMocks = vi.hoisted(() => ({
  navigate: vi.fn()
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

vi.mock("react-router-dom", () => ({
  useNavigate: () => navigationMocks.navigate
}))

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
  useDecksQuery: vi.fn(),
  useCreateDeckMutation: vi.fn(),
  useCreateFlashcardMutation: vi.fn()
}))

if (!(globalThis as any).ResizeObserver) {
  ;(globalThis as any).ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
}

describe("ResultsTab filters and retake workflow", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    window.sessionStorage.clear()
    navigationMocks.navigate.mockReset()

    vi.mocked(useAllAttemptsQuery).mockReturnValue({
      data: {
        items: [
          {
            id: 301,
            quiz_id: 7,
            started_at: "2026-02-18T10:00:00Z",
            completed_at: "2026-02-18T10:03:00Z",
            score: 2,
            total_possible: 3,
            time_spent_seconds: 180,
            answers: []
          },
          {
            id: 302,
            quiz_id: 8,
            started_at: "2026-02-17T10:00:00Z",
            completed_at: "2026-02-17T10:05:00Z",
            score: 1,
            total_possible: 5,
            time_spent_seconds: 300,
            answers: []
          }
        ],
        count: 2
      },
      isLoading: false
    } as any)

    vi.mocked(useQuizzesQuery).mockReturnValue({
      data: {
        items: [
          { id: 7, name: "Biology Basics", total_questions: 3 },
          { id: 8, name: "Chemistry Basics", total_questions: 5 }
        ],
        count: 2
      },
      isLoading: false
    } as any)

    vi.mocked(useAttemptQuery).mockReturnValue({
      data: null,
      isLoading: false,
      isFetching: false
    } as any)
    vi.mocked(useGenerateRemediationQuizMutation).mockReturnValue({
      mutateAsync: vi.fn(),
      isPending: false
    } as any)
    vi.mocked(useAttemptRemediationConversionsQuery).mockReturnValue({
      data: {
        attempt_id: 301,
        items: [],
        count: 0,
        superseded_count: 0
      },
      isLoading: false
    } as any)
    vi.mocked(useConvertAttemptRemediationQuestionsMutation).mockReturnValue({
      mutateAsync: vi.fn(),
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
    vi.mocked(useCreateDeckMutation).mockReturnValue({
      mutateAsync: vi.fn(),
      isPending: false
    } as any)
    vi.mocked(useCreateFlashcardMutation).mockReturnValue({
      mutateAsync: vi.fn(),
      isPending: false
    } as any)
  })

  it("applies persisted quiz filter to attempts query and supports retake action", () => {
    window.sessionStorage.setItem("quiz-results-filters-v1", JSON.stringify({
      page: 1,
      pageSize: 10,
      quizFilterId: 7,
      passFilter: "all",
      dateRangeFilter: "all"
    }))

    const onRetakeQuiz = vi.fn()
    render(<ResultsTab onRetakeQuiz={onRetakeQuiz} />)

    expect(useAllAttemptsQuery).toHaveBeenCalledWith(expect.objectContaining({ quiz_id: 7 }))
    expect(useQuizzesQuery).toHaveBeenCalledWith(
      expect.objectContaining({ limit: 20, offset: 0 }),
      expect.objectContaining({ enabled: true })
    )

    fireEvent.click(screen.getByRole("button", { name: /Retake/i }))
    expect(onRetakeQuiz).toHaveBeenCalledWith({
      startQuizId: 7,
      highlightQuizId: 7,
      sourceTab: "results",
      attemptId: 301
    })
  })

  it("routes row-level Study with Flashcards action with quiz and attempt context", () => {
    render(<ResultsTab />)

    fireEvent.click(screen.getAllByRole("button", { name: /Study with Flashcards/i })[0])

    expect(navigationMocks.navigate).toHaveBeenCalledWith(
      expect.stringContaining(
        "/flashcards?tab=review&study_source=quiz&quiz_id=7&attempt_id=301"
      )
    )
  })

  it("shows no-match empty state when persisted pass filter excludes all attempts", () => {
    window.sessionStorage.setItem("quiz-results-filters-v1", JSON.stringify({
      page: 1,
      pageSize: 10,
      quizFilterId: null,
      passFilter: "pass",
      dateRangeFilter: "all"
    }))

    vi.mocked(useAllAttemptsQuery).mockReturnValue({
      data: {
        items: [
          {
            id: 401,
            quiz_id: 7,
            started_at: "2026-02-18T10:00:00Z",
            completed_at: "2026-02-18T10:03:00Z",
            score: 1,
            total_possible: 5,
            time_spent_seconds: 180,
            answers: []
          }
        ],
        count: 1
      },
      isLoading: false
    } as any)

    render(<ResultsTab />)

    expect(screen.getByText("No attempts match the selected filters.")).toBeInTheDocument()
  })

  it("restores persisted pagination state on remount", () => {
    window.sessionStorage.setItem("quiz-results-filters-v1", JSON.stringify({
      page: 2,
      pageSize: 1,
      quizFilterId: null,
      passFilter: "all",
      dateRangeFilter: "all"
    }))

    render(<ResultsTab />)

    expect(screen.getByText("Chemistry Basics")).toBeInTheDocument()
    expect(screen.queryByText("Biology Basics")).not.toBeInTheDocument()
  })

  it("uses quiz-specific passing_score when applying pass/fail filters", () => {
    window.sessionStorage.setItem("quiz-results-filters-v1", JSON.stringify({
      page: 1,
      pageSize: 10,
      quizFilterId: null,
      passFilter: "pass",
      dateRangeFilter: "all"
    }))

    vi.mocked(useQuizzesQuery).mockReturnValue({
      data: {
        items: [
          { id: 7, name: "Biology Basics", total_questions: 3, passing_score: 90 }
        ],
        count: 1
      },
      isLoading: false
    } as any)

    vi.mocked(useAllAttemptsQuery).mockReturnValue({
      data: {
        items: [
          {
            id: 501,
            quiz_id: 7,
            started_at: "2026-02-18T10:00:00Z",
            completed_at: "2026-02-18T10:04:00Z",
            score: 4,
            total_possible: 5,
            time_spent_seconds: 240,
            answers: []
          }
        ],
        count: 1
      },
      isLoading: false
    } as any)

    render(<ResultsTab />)

    expect(screen.getByText("No attempts match the selected filters.")).toBeInTheDocument()
  })

  it("renders score trend visualization when at least two attempts exist", () => {
    render(<ResultsTab />)

    expect(screen.getByText("Score Trend")).toBeInTheDocument()
    const chart = screen.getByLabelText("Score percentage trend over recent attempts")
    expect(chart.querySelector("polyline")).not.toBeNull()
  })

  it("renders results skeleton placeholders while loading", () => {
    vi.mocked(useAllAttemptsQuery).mockReturnValue({
      data: undefined,
      isLoading: true
    } as any)
    vi.mocked(useQuizzesQuery).mockReturnValue({
      data: undefined,
      isLoading: true
    } as any)

    render(<ResultsTab />)

    expect(screen.getByTestId("results-loading-skeleton")).toBeInTheDocument()
  })
})
