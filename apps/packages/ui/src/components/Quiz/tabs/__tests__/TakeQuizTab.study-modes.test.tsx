import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
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
import { listQuestions } from "@/services/quizzes"
import { TAKE_QUIZ_LIST_PREFS_KEY } from "../../stateKeys"
import { drawDeterministicQuestionPool } from "../../utils/optionShuffle"

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

vi.mock("@/services/quizzes", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/services/quizzes")>()
  return {
    ...actual,
    listQuestions: vi.fn()
  }
})

if (!(globalThis as any).ResizeObserver) {
  ;(globalThis as any).ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
}

describe("TakeQuizTab study modes", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    window.localStorage.clear()
    window.sessionStorage.clear()

    vi.mocked(useAttemptsQuery).mockReturnValue({
      data: {
        items: [],
        count: 0
      }
    } as any)

    vi.mocked(useQuizzesQuery).mockReturnValue({
      data: {
        items: [
          {
            id: 7,
            name: "Biology Basics",
            description: "Cell structures and functions",
            total_questions: 3,
            time_limit_seconds: 900,
            passing_score: 75
          }
        ],
        count: 1
      },
      isLoading: false
    } as any)

    vi.mocked(useQuizQuery).mockReturnValue({
      data: {
        id: 7,
        name: "Biology Basics",
        total_questions: 3,
        time_limit_seconds: 900,
        passing_score: 75
      }
    } as any)

    vi.mocked(useStartAttemptMutation).mockReturnValue({
      mutateAsync: vi.fn(async () => ({
        id: 101,
        quiz_id: 7,
        started_at: "2026-02-18T10:00:00Z",
        total_possible: 3,
        answers: [],
        questions: []
      })),
      isPending: false
    } as any)

    vi.mocked(useSubmitAttemptMutation).mockReturnValue({
      mutateAsync: vi.fn(),
      isPending: false
    } as any)

    vi.mocked(useQuizTimer).mockReturnValue(null)
    vi.mocked(useQuizAutoSave).mockReturnValue({
      storageUnavailable: false,
      restoreSavedAnswers: vi.fn(async () => false),
      clearSavedProgress: vi.fn(async () => {}),
      hasSavedProgress: vi.fn(async () => false),
      getSavedProgress: vi.fn(async () => null),
      forceSave: vi.fn(async () => {})
    } as any)
  })

  it("starts in practice mode and shows immediate correctness feedback", async () => {
    window.sessionStorage.setItem(
      TAKE_QUIZ_LIST_PREFS_KEY,
      JSON.stringify({ modePreference: "practice" })
    )

    vi.mocked(listQuestions).mockResolvedValue({
      items: [
        {
          id: 11,
          quiz_id: 7,
          question_type: "multiple_choice",
          question_text: "Which city is the capital of France?",
          options: ["Berlin", "Paris"],
          correct_answer: 1,
          explanation: "Paris is the capital city of France."
        }
      ],
      count: 1
    } as any)

    const startAttempt = vi.fn()
    vi.mocked(useStartAttemptMutation).mockReturnValue({
      mutateAsync: startAttempt,
      isPending: false
    } as any)

    render(<MemoryRouter><TakeQuizTab onNavigateToGenerate={() => {}} onNavigateToCreate={() => {}} /></MemoryRouter>)

    fireEvent.click(screen.getByRole("button", { name: /Start Practice/i }))

    await waitFor(() => {
      expect(listQuestions).toHaveBeenCalledWith(
        7,
        expect.objectContaining({ include_answers: true, limit: 500, offset: 0 })
      )
    })
    expect(startAttempt).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole("radio", { name: "Berlin" }))
    expect(await screen.findByText("Incorrect")).toBeInTheDocument()
    expect(screen.getByText(/Correct answer:/)).toHaveTextContent("Paris")

    fireEvent.click(screen.getByRole("radio", { name: "Paris" }))
    expect(await screen.findByText("Correct")).toBeInTheDocument()
  }, 15000)

  it("grades multi-select answers in practice mode using full-set matching", async () => {
    window.sessionStorage.setItem(
      TAKE_QUIZ_LIST_PREFS_KEY,
      JSON.stringify({ modePreference: "practice" })
    )

    vi.mocked(listQuestions).mockResolvedValue({
      items: [
        {
          id: 19,
          quiz_id: 7,
          question_type: "multi_select",
          question_text: "Select all prime numbers.",
          options: ["2", "4", "5", "6"],
          correct_answer: [0, 2],
          explanation: "2 and 5 are prime."
        }
      ],
      count: 1
    } as any)

    render(<MemoryRouter><TakeQuizTab onNavigateToGenerate={() => {}} onNavigateToCreate={() => {}} /></MemoryRouter>)

    fireEvent.click(screen.getByRole("button", { name: /Start Practice/i }))

    expect(await screen.findByRole("checkbox", { name: "2" })).toBeInTheDocument()

    fireEvent.click(screen.getByRole("checkbox", { name: "2" }))
    expect(await screen.findByText("Incorrect")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("checkbox", { name: "5" }))
    expect(await screen.findByText("Correct")).toBeInTheDocument()
  }, 15000)

  it("opens review mode as read-only with answers and explanations", async () => {
    window.sessionStorage.setItem(
      TAKE_QUIZ_LIST_PREFS_KEY,
      JSON.stringify({ modePreference: "review" })
    )

    vi.mocked(listQuestions).mockResolvedValue({
      items: [
        {
          id: 31,
          quiz_id: 7,
          question_type: "true_false",
          question_text: "Cells are living organisms.",
          options: null,
          correct_answer: "true",
          explanation: "Cells are the basic structural and functional unit of life."
        }
      ],
      count: 1
    } as any)

    const startAttempt = vi.fn()
    vi.mocked(useStartAttemptMutation).mockReturnValue({
      mutateAsync: startAttempt,
      isPending: false
    } as any)

    render(<MemoryRouter><TakeQuizTab onNavigateToGenerate={() => {}} onNavigateToCreate={() => {}} /></MemoryRouter>)

    fireEvent.click(screen.getByRole("button", { name: /Open Review/i }))

    expect(await screen.findByText("Review mode is read-only. No graded attempt is created and no score is recorded.")).toBeInTheDocument()
    expect(screen.getByText(/Correct answer:/)).toHaveTextContent("true")
    expect(screen.getByText("Cells are the basic structural and functional unit of life.")).toBeInTheDocument()
    expect(startAttempt).not.toHaveBeenCalled()
    expect(screen.queryByRole("button", { name: "Submit" })).not.toBeInTheDocument()
  }, 15000)

  it("persists mode preference updates to session storage", async () => {
    render(<MemoryRouter><TakeQuizTab onNavigateToGenerate={() => {}} onNavigateToCreate={() => {}} /></MemoryRouter>)

    expect(screen.getByRole("button", { name: /Start Quiz/i })).toBeInTheDocument()

    const modeSelect = screen.getAllByRole("combobox")[0]
    fireEvent.mouseDown(modeSelect)
    fireEvent.click(await screen.findByText("Mode: Practice"))

    expect(screen.getByRole("button", { name: /Start Practice/i })).toBeInTheDocument()

    const stored = window.sessionStorage.getItem(TAKE_QUIZ_LIST_PREFS_KEY)
    expect(stored).toBeTruthy()
    expect(JSON.parse(stored as string)).toEqual(
      expect.objectContaining({
        modePreference: "practice"
      })
    )
  }, 15000)

  it("draws a deterministic question pool in practice mode when configured", async () => {
    const questionBank = [
      {
        id: 101,
        quiz_id: 7,
        question_type: "true_false",
        question_text: "Q1",
        options: null,
        correct_answer: "true"
      },
      {
        id: 102,
        quiz_id: 7,
        question_type: "true_false",
        question_text: "Q2",
        options: null,
        correct_answer: "true"
      },
      {
        id: 103,
        quiz_id: 7,
        question_type: "true_false",
        question_text: "Q3",
        options: null,
        correct_answer: "true"
      },
      {
        id: 104,
        quiz_id: 7,
        question_type: "true_false",
        question_text: "Q4",
        options: null,
        correct_answer: "true"
      }
    ]
    window.sessionStorage.setItem(
      TAKE_QUIZ_LIST_PREFS_KEY,
      JSON.stringify({ modePreference: "practice", studyPoolSize: 2, studyPoolSeedOverride: 777 })
    )
    vi.mocked(listQuestions).mockResolvedValue({
      items: questionBank,
      count: questionBank.length
    } as any)

    const expectedPool = drawDeterministicQuestionPool(questionBank, 2, 777)

    render(<MemoryRouter><TakeQuizTab onNavigateToGenerate={() => {}} onNavigateToCreate={() => {}} /></MemoryRouter>)

    fireEvent.click(screen.getByRole("button", { name: /Start Practice/i }))
    await waitFor(() => {
      expect(screen.getAllByTestId(/quiz-question-/)).toHaveLength(2)
    })
    expectedPool.forEach((question) => {
      expect(screen.getByText(question.question_text)).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole("button", { name: "Back to list" }))
    fireEvent.click(screen.getByRole("button", { name: /Start Practice/i }))

    await waitFor(() => {
      expect(screen.getAllByTestId(/quiz-question-/)).toHaveLength(2)
    })
    expectedPool.forEach((question) => {
      expect(screen.getByText(question.question_text)).toBeInTheDocument()
    })
  }, 15000)

  it("auto-advances focus when the per-question timer expires in practice mode", async () => {
    window.sessionStorage.setItem(
      TAKE_QUIZ_LIST_PREFS_KEY,
      JSON.stringify({
        modePreference: "practice",
        practiceQuestionTimerSeconds: 1
      })
    )
    vi.mocked(listQuestions).mockResolvedValue({
      items: [
        {
          id: 501,
          quiz_id: 7,
          question_type: "true_false",
          question_text: "Timer Q1",
          options: null,
          correct_answer: "true"
        },
        {
          id: 502,
          quiz_id: 7,
          question_type: "true_false",
          question_text: "Timer Q2",
          options: null,
          correct_answer: "false"
        }
      ],
      count: 2
    } as any)

    render(<MemoryRouter><TakeQuizTab onNavigateToGenerate={() => {}} onNavigateToCreate={() => {}} /></MemoryRouter>)

    fireEvent.click(screen.getByRole("button", { name: /Start Practice/i }))

    const firstQuestion = await screen.findByTestId("quiz-question-501")
    expect(firstQuestion).toHaveAttribute("data-highlighted", "true")

    await waitFor(
      () => {
        expect(screen.getByTestId("quiz-question-502")).toHaveAttribute("data-highlighted", "true")
      },
      { timeout: 3000 }
    )
  }, 15000)

  it("grades fuzzy fill-blank alternates in practice mode", async () => {
    window.sessionStorage.setItem(
      TAKE_QUIZ_LIST_PREFS_KEY,
      JSON.stringify({ modePreference: "practice" })
    )
    vi.mocked(listQuestions).mockResolvedValue({
      items: [
        {
          id: 701,
          quiz_id: 7,
          question_type: "fill_blank",
          question_text: "Spell this close match",
          options: null,
          correct_answer: "~mitochondrion || nucleus"
        }
      ],
      count: 1
    } as any)

    render(<MemoryRouter><TakeQuizTab onNavigateToGenerate={() => {}} onNavigateToCreate={() => {}} /></MemoryRouter>)

    fireEvent.click(screen.getByRole("button", { name: /Start Practice/i }))
    const input = await screen.findByRole("textbox", { name: /Answer for question/i })
    fireEvent.change(input, { target: { value: "mitocondrion" } })

    expect(await screen.findByText("Correct")).toBeInTheDocument()
  }, 15000)
})
