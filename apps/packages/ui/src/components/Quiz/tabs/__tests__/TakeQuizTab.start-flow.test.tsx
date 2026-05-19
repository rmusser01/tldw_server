import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterAll, beforeAll, beforeEach, describe, expect, it, vi } from "vitest"
import { MemoryRouter } from "react-router-dom"
import { TakeQuizTab } from "../TakeQuizTab"
import {
  useAttemptsQuery,
  useQuizzesQuery,
  useQuizQuery,
  useStartAttemptMutation,
  useSubmitAttemptMutation
} from "../../hooks"
import { useQuizAutoSave } from "../../hooks/useQuizAutoSave"
import { useQuizTimer } from "../../hooks/useQuizTimer"
import { buildShuffledOptionEntries } from "../../utils/optionShuffle"

vi.mock("react-router-dom", () => ({
  Link: ({ to, children, ...props }: Record<string, unknown>) => <a href={to as string} {...props}>{children as React.ReactNode}</a>
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

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => true
}))

vi.mock("../../hooks", () => ({
  useAttemptsQuery: vi.fn(),
  useQuizzesQuery: vi.fn(),
  useQuizQuery: vi.fn(),
  useStartAttemptMutation: vi.fn(),
  useSubmitAttemptMutation: vi.fn()
}))

vi.mock("../../hooks/useQuizTimer", () => ({
  useQuizTimer: vi.fn(() => null)
}))

vi.mock("../../hooks/useQuizAutoSave", () => ({
  useQuizAutoSave: vi.fn(() => ({
    storageUnavailable: false,
    restoreSavedAnswers: vi.fn(async () => false),
    clearSavedProgress: vi.fn(async () => {}),
    hasSavedProgress: vi.fn(async () => false),
    getSavedProgress: vi.fn(async () => null),
    forceSave: vi.fn(async () => {})
  }))
}))

if (!(globalThis as any).ResizeObserver) {
  ;(globalThis as any).ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
}

describe("TakeQuizTab start flow", () => {
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
    window.localStorage.clear()

    vi.mocked(useAttemptsQuery).mockReturnValue({
      data: {
        items: [
          {
            id: 99,
            quiz_id: 7,
            started_at: "2026-02-17T12:00:00Z",
            completed_at: "2026-02-17T12:10:00Z",
            score: 8,
            total_possible: 10,
            answers: []
          }
        ],
        count: 1
      }
    } as any)

    vi.mocked(useQuizzesQuery).mockReturnValue({
      data: {
        items: [
          {
            id: 7,
            name: "Biology Basics",
            description: "Cell structures and functions",
            total_questions: 12,
            time_limit_seconds: 900,
            passing_score: 75,
            media_id: 42,
            created_at: "2026-02-16T12:00:00Z"
          }
        ],
        count: 1
      },
      isLoading: false
    } as any)

    vi.mocked(useQuizQuery).mockReturnValue({
      data: null
    } as any)

    vi.mocked(useStartAttemptMutation).mockReturnValue({
      mutateAsync: vi.fn(async () => ({
        id: 123,
        quiz_id: 7,
        started_at: "2026-02-18T10:00:00Z",
        total_possible: 12,
        answers: [],
        questions: [
          {
            id: 1,
            quiz_id: 7,
            question_type: "true_false",
            question_text: "Cells are alive.",
            options: null,
            points: 1,
            order_index: 0,
            tags: null,
            deleted: false,
            client_id: "test",
            version: 1
          }
        ]
      }))
    } as any)

    vi.mocked(useSubmitAttemptMutation).mockReturnValue({
      mutateAsync: vi.fn(),
      isPending: false
    } as any)

    vi.mocked(useQuizTimer).mockReturnValue(null)
  })

  it("requires pre-quiz confirmation before creating an attempt", async () => {
    const mutateAsync = vi.fn(async () => ({
      id: 123,
      quiz_id: 7,
      started_at: "2026-02-18T10:00:00Z",
      total_possible: 12,
      answers: [],
      questions: []
    }))
    vi.mocked(useStartAttemptMutation).mockReturnValue({ mutateAsync } as any)

    render(
      <MemoryRouter>
        <TakeQuizTab
          onNavigateToGenerate={() => {}}
          onNavigateToCreate={() => {}}
        />
      </MemoryRouter>
    )

    fireEvent.click(screen.getByRole("button", { name: /Start Quiz/i }))

    expect(screen.getByText("Ready to begin?")).toBeInTheDocument()
    expect(mutateAsync).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole("button", { name: "Begin Quiz" }))

    await waitFor(() => {
      expect(mutateAsync).toHaveBeenCalledWith(7)
    })
  }, 15000)

  it("renders expanded quiz metadata on list cards", () => {
    render(
      <MemoryRouter>
        <TakeQuizTab
          onNavigateToGenerate={() => {}}
          onNavigateToCreate={() => {}}
        />
      </MemoryRouter>
    )

    expect(screen.getByText("Pass: 75%")).toBeInTheDocument()
    expect(screen.getByText("Last score: 80%")).toBeInTheDocument()
    expect(screen.getByText(/Created:/)).toBeInTheDocument()

    const sourceLink = screen.getByRole("link", { name: /Source media #42/i })
    expect(sourceLink).toHaveAttribute("href", "/media?id=42")
  }, 15000)

  it("shows autosave warning when local storage is unavailable", () => {
    vi.mocked(useQuizAutoSave).mockReturnValue({
      storageUnavailable: true,
      restoreSavedAnswers: vi.fn(async () => false),
      clearSavedProgress: vi.fn(async () => {}),
      hasSavedProgress: vi.fn(async () => false),
      getSavedProgress: vi.fn(async () => null),
      forceSave: vi.fn(async () => {})
    } as any)

    render(
      <MemoryRouter>
        <TakeQuizTab
          onNavigateToGenerate={() => {}}
          onNavigateToCreate={() => {}}
        />
      </MemoryRouter>
    )

    expect(
      screen.getByText(
        "Auto-save unavailable — your progress won't be preserved if you navigate away."
      )
    ).toBeInTheDocument()
  }, 15000)

  it("renders shared-assignment context with due date and note", () => {
    render(
      <MemoryRouter>
        <TakeQuizTab
          onNavigateToGenerate={() => {}}
          onNavigateToCreate={() => {}}
          startQuizId={7}
          highlightQuizId={7}
          navigationSource="assignment"
          assignmentMode="shared"
          assignmentDueAt="2026-03-01T14:30:00.000Z"
          assignmentNote="Complete before the lab session."
          assignedByRole="lead"
        />
      </MemoryRouter>
    )

    expect(
      screen.getByText("This quiz was opened from a shared assignment link.")
    ).toBeInTheDocument()
    expect(screen.getAllByText("This shared assignment is past due.").length).toBeGreaterThan(0)
    expect(screen.getAllByText("Note: Complete before the lab session.").length).toBeGreaterThan(0)
    expect(screen.getAllByText("Assigned by role: lead").length).toBeGreaterThan(0)
    expect(
      screen.getByText("Shared assignment ready: Biology Basics.")
    ).toBeInTheDocument()
  }, 15000)

  it("does not enter attempt mode when the quiz has zero questions", async () => {
    vi.mocked(useStartAttemptMutation).mockReturnValue({
      mutateAsync: vi.fn(async () => ({
        id: 777,
        quiz_id: 7,
        started_at: "2026-02-18T10:00:00Z",
        total_possible: 0,
        answers: [],
        questions: []
      }))
    } as any)

    render(
      <MemoryRouter>
        <TakeQuizTab
          onNavigateToGenerate={() => {}}
          onNavigateToCreate={() => {}}
        />
      </MemoryRouter>
    )

    fireEvent.click(screen.getByRole("button", { name: /Start Quiz/i }))
    fireEvent.click(screen.getByRole("button", { name: "Begin Quiz" }))

    await waitFor(() => {
      expect(screen.getByText("Select a quiz to begin")).toBeInTheDocument()
    })
    expect(screen.queryByText("Question navigator")).not.toBeInTheDocument()
  }, 15000)

  it("adds semantic grouping for question radios and labels progress", async () => {
    render(
      <MemoryRouter>
        <TakeQuizTab
          onNavigateToGenerate={() => {}}
          onNavigateToCreate={() => {}}
        />
      </MemoryRouter>
    )

    fireEvent.click(screen.getByRole("button", { name: /Start Quiz/i }))
    fireEvent.click(screen.getByRole("button", { name: "Begin Quiz" }))

    expect(await screen.findByText("True or false for: Cells are alive.")).toBeInTheDocument()
    const completionProgress = screen.getByRole("progressbar", { name: "Quiz completion progress" })
    expect(completionProgress).toHaveAttribute("aria-valuemin", "0")
    expect(completionProgress).toHaveAttribute("aria-valuemax", "100")
  }, 15000)

  it("announces danger-zone timer updates in assertive live region", async () => {
    vi.mocked(useQuizTimer).mockReturnValue({
      minutes: 0,
      seconds: 58,
      totalSeconds: 58,
      isWarning: false,
      isDanger: true,
      isExpired: false,
      formattedTime: "0:58"
    })

    render(
      <MemoryRouter>
        <TakeQuizTab
          onNavigateToGenerate={() => {}}
          onNavigateToCreate={() => {}}
        />
      </MemoryRouter>
    )

    fireEvent.click(screen.getByRole("button", { name: /Start Quiz/i }))
    fireEvent.click(screen.getByRole("button", { name: "Begin Quiz" }))

    const liveRegion = await screen.findByText("58 seconds remaining")
    expect(liveRegion).toHaveAttribute("aria-live", "assertive")
  }, 15000)

  it("shuffles multiple-choice option order per graded attempt while preserving answer mapping", async () => {
    const optionLabels = ["Alpha", "Beta", "Gamma", "Delta"]
    const questionId = 11

    const firstAttemptId = 200
    let secondAttemptId = 201
    const firstOrder = buildShuffledOptionEntries(optionLabels, questionId, firstAttemptId).map((entry) => entry.originalIndex)
    let secondOrder = buildShuffledOptionEntries(optionLabels, questionId, secondAttemptId).map((entry) => entry.originalIndex)
    while (secondOrder.join(",") === firstOrder.join(",")) {
      secondAttemptId += 1
      secondOrder = buildShuffledOptionEntries(optionLabels, questionId, secondAttemptId).map((entry) => entry.originalIndex)
    }

    const startMutate = vi
      .fn()
      .mockResolvedValueOnce({
        id: firstAttemptId,
        quiz_id: 7,
        started_at: "2026-02-18T10:00:00Z",
        total_possible: 1,
        answers: [],
        questions: [
          {
            id: questionId,
            quiz_id: 7,
            question_type: "multiple_choice",
            question_text: "Pick the second Greek letter.",
            options: optionLabels,
            points: 1,
            order_index: 0,
            tags: null,
            deleted: false,
            client_id: "test",
            version: 1
          }
        ]
      })
      .mockResolvedValueOnce({
        id: secondAttemptId,
        quiz_id: 7,
        started_at: "2026-02-18T10:02:00Z",
        total_possible: 1,
        answers: [],
        questions: [
          {
            id: questionId,
            quiz_id: 7,
            question_type: "multiple_choice",
            question_text: "Pick the second Greek letter.",
            options: optionLabels,
            points: 1,
            order_index: 0,
            tags: null,
            deleted: false,
            client_id: "test",
            version: 1
          }
        ]
      })

    const submitMutate = vi.fn(async ({ attemptId, answers }: {
      attemptId: number
      answers: Array<{ question_id: number; user_answer: number; hint_used?: boolean }>
    }) => ({
      id: attemptId,
      quiz_id: 7,
      started_at: "2026-02-18T10:00:00Z",
      completed_at: "2026-02-18T10:03:00Z",
      score: 1,
      total_possible: 1,
      answers: [
        {
          question_id: questionId,
          user_answer: answers[0]?.user_answer,
          is_correct: true,
          correct_answer: 1
        }
      ]
    }))

    vi.mocked(useStartAttemptMutation).mockReturnValue({
      mutateAsync: startMutate
    } as any)
    vi.mocked(useSubmitAttemptMutation).mockReturnValue({
      mutateAsync: submitMutate,
      isPending: false
    } as any)

    render(
      <MemoryRouter>
        <TakeQuizTab
          onNavigateToGenerate={() => {}}
          onNavigateToCreate={() => {}}
        />
      </MemoryRouter>
    )

    fireEvent.click(screen.getByRole("button", { name: /Start Quiz/i }))
    fireEvent.click(screen.getByRole("button", { name: "Begin Quiz" }))

    await screen.findByTestId(`quiz-question-${questionId}`)

    const firstRenderOrder = screen.getAllByRole("radio").map((node) => Number((node as HTMLInputElement).value))
    expect(firstRenderOrder).toEqual(firstOrder)

    fireEvent.click(screen.getAllByRole("radio")[0])
    fireEvent.click(screen.getByRole("button", { name: "Submit" }))

    await waitFor(() => {
      expect(submitMutate).toHaveBeenCalledWith(
        expect.objectContaining({
          attemptId: firstAttemptId,
          answers: [{ question_id: questionId, user_answer: firstOrder[0], hint_used: false }]
        })
      )
    })

    fireEvent.click(screen.getByRole("button", { name: /Retake Quiz/i }))
    fireEvent.click(screen.getByRole("button", { name: "Begin Quiz" }))

    await screen.findByTestId(`quiz-question-${questionId}`)
    const secondRenderOrder = screen.getAllByRole("radio").map((node) => Number((node as HTMLInputElement).value))
    expect(secondRenderOrder).toEqual(secondOrder)
  }, 15000)

  it("tracks hint usage in submit payload and reflects penalty in results", async () => {
    const questionId = 19
    const startMutate = vi.fn(async () => ({
      id: 919,
      quiz_id: 7,
      started_at: "2026-02-18T10:00:00Z",
      total_possible: 1,
      answers: [],
      questions: [
        {
          id: questionId,
          quiz_id: 7,
          question_type: "multiple_choice",
          question_text: "Which city has the Eiffel Tower?",
          options: ["Berlin", "Paris"],
          hint: "It is known as the city of lights.",
          hint_penalty_points: 1,
          points: 1,
          order_index: 0,
          tags: null,
          deleted: false,
          client_id: "test",
          version: 1
        }
      ]
    }))

    const submitMutate = vi.fn(async () => ({
      id: 919,
      quiz_id: 7,
      started_at: "2026-02-18T10:00:00Z",
      completed_at: "2026-02-18T10:01:00Z",
      score: 0,
      total_possible: 1,
      answers: [
        {
          question_id: questionId,
          user_answer: 1,
          is_correct: true,
          correct_answer: 1,
          hint_used: true,
          hint_penalty_points: 1,
          points_awarded: 0
        }
      ]
    }))

    vi.mocked(useStartAttemptMutation).mockReturnValue({
      mutateAsync: startMutate
    } as any)
    vi.mocked(useSubmitAttemptMutation).mockReturnValue({
      mutateAsync: submitMutate,
      isPending: false
    } as any)

    render(
      <MemoryRouter>
        <TakeQuizTab
          onNavigateToGenerate={() => {}}
          onNavigateToCreate={() => {}}
        />
      </MemoryRouter>
    )

    fireEvent.click(screen.getByRole("button", { name: /Start Quiz/i }))
    fireEvent.click(screen.getByRole("button", { name: "Begin Quiz" }))

    await screen.findByTestId(`quiz-question-${questionId}`)
    fireEvent.click(screen.getByRole("button", { name: "Show hint for question 19" }))
    expect(screen.getByText("It is known as the city of lights.")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("radio", { name: "Paris" }))
    fireEvent.click(screen.getByRole("button", { name: "Submit" }))

    await waitFor(() => {
      expect(submitMutate).toHaveBeenCalledWith(
        expect.objectContaining({
          attemptId: 919,
          answers: [{ question_id: questionId, user_answer: 1, hint_used: true }]
        })
      )
    })

    expect(screen.getByText("Hint used (-1 point(s)).")).toBeInTheDocument()
    expect(screen.getByText("Points:")).toBeInTheDocument()
  }, 15000)
})
