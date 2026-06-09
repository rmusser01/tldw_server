import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { afterAll, beforeEach, describe, expect, it, vi } from "vitest"
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
import {
  expectInsideDesignSystemAlert,
  expectInsideDesignSystemAlertAsync
} from "@/test-utils/designSystemAlert"

vi.mock("react-router-dom", async (importOriginal) => {
  const actual = await importOriginal<typeof import("react-router-dom")>()
  return {
    ...actual,
    Link: ({ to, children, ...props }: Record<string, unknown>) => (
      <a href={to as string} {...props}>
        {children as React.ReactNode}
      </a>
    )
  }
})

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
  useQuizAutoSave: vi.fn()
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

const defaultQuiz = {
  id: 7,
  name: "Biology Basics",
  description: "Cell structures and functions",
  total_questions: 2,
  time_limit_seconds: 900,
  passing_score: 75,
  media_id: 42,
  created_at: "2026-02-16T12:00:00Z"
}

const defaultQuestions = [
  {
    id: 11,
    quiz_id: 7,
    question_type: "multiple_choice",
    question_text: "Which city has the Eiffel Tower?",
    options: ["Berlin", "Paris"],
    correct_answer: 1,
    hint: "It is known as the city of lights.",
    hint_penalty_points: 1,
    points: 1,
    order_index: 0,
    tags: null,
    deleted: false,
    client_id: "test",
    version: 1
  },
  {
    id: 12,
    quiz_id: 7,
    question_type: "true_false",
    question_text: "Cells are alive.",
    options: null,
    correct_answer: "true",
    points: 1,
    order_index: 1,
    tags: null,
    deleted: false,
    client_id: "test",
    version: 1
  }
]

const renderTakeQuizTab = (props: Partial<React.ComponentProps<typeof TakeQuizTab>> = {}) => {
  return render(
    <MemoryRouter>
      <TakeQuizTab
        onNavigateToGenerate={() => {}}
        onNavigateToCreate={() => {}}
        {...props}
      />
    </MemoryRouter>
  )
}

const expectInsideDesignSystemBadge = (text: string | RegExp) => {
  const badge = screen
    .getAllByText(text)
    .map((node) => node.closest('[data-ds-component="Badge"]'))
    .find((candidate): candidate is HTMLElement => candidate != null)
  expect(badge).not.toBeNull()
  expect(badge).toHaveAttribute("data-ds-component", "Badge")
  return badge
}

describe("TakeQuizTab design-system product states", () => {
  const originalScrollIntoView = HTMLElement.prototype.scrollIntoView

  afterAll(() => {
    HTMLElement.prototype.scrollIntoView = originalScrollIntoView
  })

  beforeEach(() => {
    vi.clearAllMocks()
    window.localStorage.clear()
    window.sessionStorage.clear()
    HTMLElement.prototype.scrollIntoView = vi.fn()

    vi.mocked(useAttemptsQuery).mockReturnValue({
      data: { items: [], count: 0 }
    } as any)

    vi.mocked(useQuizzesQuery).mockReturnValue({
      data: {
        items: [defaultQuiz],
        count: 1
      },
      isLoading: false
    } as any)

    vi.mocked(useQuizQuery).mockReturnValue({
      data: defaultQuiz
    } as any)

    vi.mocked(useStartAttemptMutation).mockReturnValue({
      mutateAsync: vi.fn(async () => ({
        id: 123,
        quiz_id: 7,
        started_at: "2026-02-18T10:00:00Z",
        total_possible: 2,
        answers: [],
        questions: defaultQuestions
      })),
      isPending: false
    } as any)

    vi.mocked(useSubmitAttemptMutation).mockReturnValue({
      mutateAsync: vi.fn(async () => ({
        id: 123,
        quiz_id: 7,
        started_at: "2026-02-18T10:00:00Z",
        completed_at: "2026-02-18T10:03:00Z",
        score: 1,
        total_possible: 2,
        answers: [
          {
            question_id: 11,
            user_answer: 1,
            is_correct: true,
            correct_answer: 1,
            hint_used: true,
            hint_penalty_points: 1,
            points_awarded: 1
          },
          {
            question_id: 12,
            user_answer: "true",
            is_correct: true,
            correct_answer: "true",
            points_awarded: 1
          }
        ]
      })),
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

  it("renders list-level assignment, autosave, and highlight notices through design-system Alerts", () => {
    vi.mocked(useQuizAutoSave).mockReturnValue({
      storageUnavailable: true,
      restoreSavedAnswers: vi.fn(async () => false),
      clearSavedProgress: vi.fn(async () => {}),
      hasSavedProgress: vi.fn(async () => false),
      getSavedProgress: vi.fn(async () => null),
      forceSave: vi.fn(async () => {})
    } as any)

    renderTakeQuizTab({
      startQuizId: 7,
      highlightQuizId: 7,
      navigationSource: "assignment",
      assignmentMode: "shared",
      assignmentDueAt: "2099-03-01T14:30:00.000Z",
      assignmentNote: "Complete before the lab session.",
      assignedByRole: "lead"
    })

    const assignmentAlert = expectInsideDesignSystemAlert(
      "This quiz was opened from a shared assignment link."
    )
    expect(assignmentAlert).toHaveAttribute("data-testid", "quiz-assignment-alert")
    expectInsideDesignSystemAlert(
      "Auto-save unavailable — your progress won't be preserved if you navigate away."
    )
    expectInsideDesignSystemAlert("Shared assignment ready: Biology Basics.")
  })

  it("renders start-confirmation notices through design-system Alerts", () => {
    renderTakeQuizTab({
      assignmentMode: "shared",
      assignmentDueAt: "2099-03-01T14:30:00.000Z",
      assignmentNote: "Complete before the lab session."
    })

    fireEvent.click(screen.getByRole("button", { name: /Start Quiz/i }))

    expectInsideDesignSystemAlert("Shared assignment details")
    expectInsideDesignSystemAlert(
      "Retake uses the same questions. Answer options may be reshuffled."
    )
  })

  it("renders review and practice guidance through design-system Alerts", async () => {
    vi.mocked(listQuestions).mockResolvedValue({
      items: defaultQuestions,
      count: defaultQuestions.length
    } as any)

    window.sessionStorage.setItem(
      TAKE_QUIZ_LIST_PREFS_KEY,
      JSON.stringify({ modePreference: "review" })
    )
    const { unmount } = renderTakeQuizTab()

    fireEvent.click(screen.getByRole("button", { name: /Open Review/i }))

    await expectInsideDesignSystemAlertAsync(
      "Review mode is read-only. No graded attempt is created and no score is recorded."
    )
    expect(expectInsideDesignSystemBadge("Review Mode")).toHaveAttribute("data-ds-variant", "info")

    unmount()
    vi.clearAllMocks()
    vi.mocked(useAttemptsQuery).mockReturnValue({ data: { items: [], count: 0 } } as any)
    vi.mocked(useQuizzesQuery).mockReturnValue({
      data: { items: [defaultQuiz], count: 1 },
      isLoading: false
    } as any)
    vi.mocked(useQuizQuery).mockReturnValue({ data: defaultQuiz } as any)
    vi.mocked(useStartAttemptMutation).mockReturnValue({
      mutateAsync: vi.fn(),
      isPending: false
    } as any)
    vi.mocked(useSubmitAttemptMutation).mockReturnValue({
      mutateAsync: vi.fn(),
      isPending: false
    } as any)
    vi.mocked(useQuizAutoSave).mockReturnValue({
      storageUnavailable: false,
      restoreSavedAnswers: vi.fn(async () => false),
      clearSavedProgress: vi.fn(async () => {}),
      hasSavedProgress: vi.fn(async () => false),
      getSavedProgress: vi.fn(async () => null),
      forceSave: vi.fn(async () => {})
    } as any)
    vi.mocked(listQuestions).mockResolvedValue({
      items: defaultQuestions,
      count: defaultQuestions.length
    } as any)

    window.sessionStorage.setItem(
      TAKE_QUIZ_LIST_PREFS_KEY,
      JSON.stringify({ modePreference: "practice" })
    )
    renderTakeQuizTab()

    fireEvent.click(screen.getByRole("button", { name: /Start Practice/i }))

    await expectInsideDesignSystemAlertAsync(
      "Practice mode gives immediate feedback after each answer and does not create a graded attempt."
    )
    expect(expectInsideDesignSystemBadge("Practice Mode")).toHaveAttribute("data-ds-variant", "primary")

    fireEvent.click(await screen.findByRole("radio", { name: "Berlin" }))
    await expectInsideDesignSystemAlertAsync("Incorrect")
  }, 15000)

  it("renders graded hint, unanswered, results, and correctness states through design-system primitives", async () => {
    renderTakeQuizTab()

    fireEvent.click(screen.getByRole("button", { name: /Start Quiz/i }))
    fireEvent.click(screen.getByRole("button", { name: "Begin Quiz" }))

    await screen.findByTestId("quiz-question-11")
    fireEvent.click(screen.getByRole("button", { name: "Show hint for question 11" }))
    expectInsideDesignSystemAlert("It is known as the city of lights.")

    fireEvent.click(screen.getByRole("radio", { name: "Paris" }))
    fireEvent.click(screen.getByRole("button", { name: "Submit" }))
    expectInsideDesignSystemAlert("Unanswered questions: 2")

    fireEvent.click(screen.getByRole("radio", { name: "True" }))
    fireEvent.click(screen.getByRole("button", { name: "Submit" }))

    await expectInsideDesignSystemAlertAsync("Score: 1 / 2 (50%)")
    const correctBadge = expectInsideDesignSystemBadge("Correct")
    expect(correctBadge).toHaveAttribute("data-ds-variant", "success")
  }, 15000)

  it("renders queued submission recovery through a design-system Alert", async () => {
    const mutateAsync = vi
      .fn()
      .mockRejectedValueOnce(new Error("Network Error"))
      .mockResolvedValueOnce({
        id: 123,
        quiz_id: 7,
        started_at: "2026-02-18T10:00:00Z",
        completed_at: "2026-02-18T10:03:00Z",
        score: 2,
        total_possible: 2,
        answers: []
      })
    vi.mocked(useSubmitAttemptMutation).mockReturnValue({
      mutateAsync,
      isPending: false
    } as any)

    renderTakeQuizTab()

    fireEvent.click(screen.getByRole("button", { name: /Start Quiz/i }))
    fireEvent.click(screen.getByRole("button", { name: "Begin Quiz" }))

    await screen.findByTestId("quiz-question-11")
    fireEvent.click(screen.getByRole("radio", { name: "Paris" }))
    fireEvent.click(screen.getByRole("radio", { name: "True" }))
    fireEvent.click(screen.getByRole("button", { name: "Submit" }))

    await expectInsideDesignSystemAlertAsync("Submission failed. Answers queued locally.")

    fireEvent.click(screen.getByRole("button", { name: "Retry submission" }))

    await waitFor(() => {
      expect(mutateAsync).toHaveBeenCalledTimes(2)
    })
  }, 15000)
})
