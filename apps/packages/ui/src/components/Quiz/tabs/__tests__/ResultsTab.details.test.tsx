import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { afterAll, beforeAll, beforeEach, describe, expect, it, vi } from "vitest"

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

const flashcardMocks = vi.hoisted(() => ({
  convertRemediationQuestions: vi.fn(),
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

vi.mock("react-router-dom", async (importOriginal) => {
  const actual = await importOriginal<typeof import("react-router-dom")>()
  return {
    ...actual,
    useNavigate: () => flashcardMocks.navigate
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

if (!(globalThis as any).ResizeObserver) {
  ;(globalThis as any).ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
}

describe("ResultsTab drill-down details", () => {
  const originalMatchMedia = window.matchMedia

  beforeAll(() => {
    if (typeof window.matchMedia !== "function") {
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
  })

  afterAll(() => {
    Object.defineProperty(window, "matchMedia", {
      writable: true,
      value: originalMatchMedia
    })
  })

  beforeEach(() => {
    vi.clearAllMocks()
    window.sessionStorage.clear()
    flashcardMocks.convertRemediationQuestions.mockReset()
    flashcardMocks.navigate.mockReset()
    vi.mocked(useStudySuggestions).mockReturnValue({
      status: "none",
      statusQuery: {} as any,
      snapshot: null,
      activeSnapshotId: null,
      isLoading: false,
      isRefreshing: false,
      refresh: vi.fn(),
      performAction: vi.fn()
    } as any)
    flashcardMocks.convertRemediationQuestions.mockResolvedValue({
      attempt_id: 101,
      quiz_id: 7,
      target_deck: { id: 3, name: "Biology Recovery Deck" },
      results: [],
      created_flashcard_uuids: []
    })

    vi.mocked(useAllAttemptsQuery).mockReturnValue({
      data: {
        items: [
          {
            id: 101,
            quiz_id: 7,
            started_at: "2026-02-18T10:00:00Z",
            completed_at: "2026-02-18T10:08:00Z",
            score: 2,
            total_possible: 3,
            time_spent_seconds: 480,
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
            name: "Biology Basics",
            total_questions: 3,
            media_id: 42
          }
        ],
        count: 1
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
        attempt_id: 101,
        items: [],
        count: 0,
        superseded_count: 0
      },
      isLoading: false
    } as any)
    vi.mocked(useConvertAttemptRemediationQuestionsMutation).mockReturnValue({
      mutateAsync: flashcardMocks.convertRemediationQuestions,
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
      data: [{ id: 3, name: "Biology Recovery Deck" }],
      isLoading: false
    } as any)
  })

  it("opens attempt detail modal and renders per-question breakdown", async () => {
    vi.mocked(useAttemptQuery).mockReturnValue({
      data: {
        id: 101,
        quiz_id: 7,
        started_at: "2026-02-18T10:00:00Z",
        completed_at: "2026-02-18T10:08:00Z",
        score: 2,
        total_possible: 3,
        time_spent_seconds: 480,
        answers: [
          {
            question_id: 1,
            user_answer: 1,
            is_correct: true,
            correct_answer: 1,
            explanation: "Mitochondria produce ATP.",
            source_citations: [
              {
                label: "Bio Text",
                quote: "ATP is generated by mitochondria.",
                chunk_id: "bio-1"
              }
            ]
          },
          {
            question_id: 2,
            user_answer: "false",
            is_correct: false,
            correct_answer: "true",
            explanation: "Cells are living units."
          }
        ],
        questions: [
          {
            id: 1,
            quiz_id: 7,
            question_type: "multiple_choice",
            question_text: "Powerhouse of the cell?",
            options: ["Nucleus", "Mitochondria", "Ribosome"],
            points: 1,
            order_index: 0,
            deleted: false,
            client_id: "test",
            version: 1
          },
          {
            id: 2,
            quiz_id: 7,
            question_type: "true_false",
            question_text: "Cells are not alive.",
            options: null,
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

    render(<ResultsTab />)

    fireEvent.click(screen.getByRole("button", { name: /View Details/i }))

    await waitFor(() => {
      expect(screen.getByText("Attempt Details")).toBeInTheDocument()
    })
    expect(await screen.findByText(/Powerhouse of the cell\?/i)).toBeInTheDocument()
    expect(screen.getByText("Mitochondria produce ATP.")).toBeInTheDocument()
    expect(screen.getByText("Source citations")).toBeInTheDocument()
    const sourceLink = screen.getByRole("link", { name: "Bio Text" })
    expect(sourceLink.getAttribute("href")).toContain("/media?id=42&chunk_id=bio-1")
    expect(screen.getByText("Time Spent")).toBeInTheDocument()
    const dialog = screen.getByRole("dialog")
    expect(within(dialog).getByText("8:00")).toBeInTheDocument()
  }, 15000)

  it("falls back to question-id labels for historical attempts without question snapshots", async () => {
    vi.mocked(useAllAttemptsQuery).mockReturnValue({
      data: {
        items: [
          {
            id: 202,
            quiz_id: 7,
            started_at: "2026-02-18T09:00:00Z",
            completed_at: "2026-02-18T09:02:00Z",
            score: 0,
            total_possible: 1,
            time_spent_seconds: 120,
            answers: []
          }
        ],
        count: 1
      },
      isLoading: false
    } as any)

    vi.mocked(useAttemptQuery).mockReturnValue({
      data: {
        id: 202,
        quiz_id: 7,
        started_at: "2026-02-18T09:00:00Z",
        completed_at: "2026-02-18T09:02:00Z",
        score: 0,
        total_possible: 1,
        time_spent_seconds: 120,
        answers: [
          {
            question_id: 99,
            user_answer: "foo",
            is_correct: false,
            correct_answer: "bar"
          }
        ]
      },
      isLoading: false,
      isFetching: false
    } as any)

    render(<ResultsTab />)

    fireEvent.click(screen.getByRole("button", { name: /View Details/i }))

    expect(await screen.findByText(/Question #99/i)).toBeInTheDocument()
    expect(screen.getByText("foo")).toBeInTheDocument()
    expect(screen.getByText("bar")).toBeInTheDocument()
  }, 15000)

  it("creates flashcards from missed questions through server-backed conversion state", async () => {
    vi.mocked(useAttemptQuery).mockReturnValue({
      data: {
        id: 101,
        quiz_id: 7,
        started_at: "2026-02-18T10:00:00Z",
        completed_at: "2026-02-18T10:08:00Z",
        score: 2,
        total_possible: 3,
        time_spent_seconds: 480,
        answers: [
          {
            question_id: 2,
            user_answer: "false",
            is_correct: false,
            correct_answer: "true",
            explanation: "Cells are living units."
          }
        ],
        questions: [
          {
            id: 2,
            quiz_id: 7,
            question_type: "true_false",
            question_text: "Cells are not alive.",
            options: null,
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

    render(<ResultsTab />)

    fireEvent.click(screen.getByRole("button", { name: /View Details/i }))
    await waitFor(() => {
      expect(screen.getByText("Attempt Details")).toBeInTheDocument()
    })

    fireEvent.click(
      screen.getByRole("button", { name: "Create Flashcards from Missed Questions" })
    )
    await waitFor(() => {
      expect(screen.getByText("Destination deck")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole("button", { name: "Create Flashcards" }))

    await waitFor(() => {
      expect(flashcardMocks.convertRemediationQuestions).toHaveBeenCalledTimes(1)
    })
    expect(flashcardMocks.convertRemediationQuestions).toHaveBeenCalledWith({
      attemptId: 101,
      request: {
        question_ids: [2],
        target_deck_id: 3,
        replace_active: false
      }
    })
  }, 15000)

  it("navigates to flashcards study route with attempt context from details modal", async () => {
    vi.mocked(useAttemptQuery).mockReturnValue({
      data: {
        id: 101,
        quiz_id: 7,
        started_at: "2026-02-18T10:00:00Z",
        completed_at: "2026-02-18T10:08:00Z",
        score: 2,
        total_possible: 3,
        time_spent_seconds: 480,
        answers: [],
        questions: []
      },
      isLoading: false,
      isFetching: false
    } as any)

    render(<ResultsTab />)

    fireEvent.click(screen.getByRole("button", { name: /View Details/i }))
    await waitFor(() => {
      expect(screen.getByText("Attempt Details")).toBeInTheDocument()
    })

    fireEvent.click(
      screen.getAllByRole("button", { name: /Study with Flashcards/i }).at(-1) as HTMLButtonElement
    )

    expect(flashcardMocks.navigate).toHaveBeenCalledWith(
      expect.stringContaining(
        "/flashcards?tab=review&study_source=quiz&quiz_id=7&attempt_id=101"
      )
    )
  })
})
