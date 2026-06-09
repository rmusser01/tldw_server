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

const assignmentProps = {
  assignmentMode: "shared",
  assignmentDueAt: "2099-03-01T14:30:00.000Z",
  assignmentNote: "Complete before the lab session.",
  assignedByRole: "lead"
} satisfies Partial<React.ComponentProps<typeof TakeQuizTab>>

const reviewModeTitle =
  "Review mode is read-only. No graded attempt is created and no score is recorded."
const practiceModeTitle =
  "Practice mode gives immediate feedback after each answer and does not create a graded attempt."
const scoreSummaryText = "Score: 1 / 2 (50%)"

const createAutoSaveMock = (storageUnavailable = false) => ({
  storageUnavailable,
  restoreSavedAnswers: vi.fn(async () => false),
  clearSavedProgress: vi.fn(async () => {}),
  hasSavedProgress: vi.fn(async () => false),
  getSavedProgress: vi.fn(async () => null),
  forceSave: vi.fn(async () => {})
})

const mockQuestionList = () => {
  vi.mocked(listQuestions).mockResolvedValue({
    items: defaultQuestions,
    count: defaultQuestions.length
  } as any)
}

const openStartConfirmation = (
  props: Partial<React.ComponentProps<typeof TakeQuizTab>> = {}
) => {
  renderTakeQuizTab(props)
  fireEvent.click(screen.getByRole("button", { name: /Start Quiz/i }))
}

const openStudyMode = async (mode: "review" | "practice") => {
  mockQuestionList()
  window.sessionStorage.setItem(TAKE_QUIZ_LIST_PREFS_KEY, JSON.stringify({ modePreference: mode }))
  renderTakeQuizTab()

  fireEvent.click(
    screen.getByRole("button", {
      name: mode === "review" ? /Open Review/i : /Start Practice/i
    })
  )

  await screen.findByText(mode === "review" ? reviewModeTitle : practiceModeTitle)
}

const startGradedAttempt = async () => {
  renderTakeQuizTab()

  fireEvent.click(screen.getByRole("button", { name: /Start Quiz/i }))
  fireEvent.click(screen.getByRole("button", { name: "Begin Quiz" }))

  await screen.findByTestId("quiz-question-11")
}

const answerFirstQuestionAndSubmit = async () => {
  await startGradedAttempt()

  fireEvent.click(screen.getByRole("radio", { name: "Paris" }))
  fireEvent.click(screen.getByRole("button", { name: "Submit" }))
}

const submitCompletedAttempt = async () => {
  await startGradedAttempt()

  fireEvent.click(screen.getByRole("radio", { name: "Paris" }))
  fireEvent.click(screen.getByRole("radio", { name: "True" }))
  fireEvent.click(screen.getByRole("button", { name: "Submit" }))

  await screen.findByText(scoreSummaryText)
}

const mockQueuedSubmitFailure = () => {
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
  return mutateAsync
}

const submitQueuedAttempt = async () => {
  await startGradedAttempt()

  fireEvent.click(screen.getByRole("radio", { name: "Paris" }))
  fireEvent.click(screen.getByRole("radio", { name: "True" }))
  fireEvent.click(screen.getByRole("button", { name: "Submit" }))

  await screen.findByText("Submission failed. Answers queued locally.")
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
    vi.mocked(useQuizAutoSave).mockReturnValue(createAutoSaveMock() as any)
  })

  it("renders the assignment notice through a design-system Alert", () => {
    renderTakeQuizTab({
      navigationSource: "assignment",
      ...assignmentProps
    })

    const assignmentAlert = expectInsideDesignSystemAlert(
      "This quiz was opened from a shared assignment link."
    )
    expect(assignmentAlert).toHaveAttribute("data-testid", "quiz-assignment-alert")
  })

  it("renders the auto-save unavailable notice through a design-system Alert", () => {
    vi.mocked(useQuizAutoSave).mockReturnValue(createAutoSaveMock(true) as any)

    renderTakeQuizTab()

    expectInsideDesignSystemAlert(
      "Auto-save unavailable — your progress won't be preserved if you navigate away."
    )
  })

  it("renders the shared assignment ready highlight through a design-system Alert", () => {
    renderTakeQuizTab({
      startQuizId: 7,
      highlightQuizId: 7,
      navigationSource: "assignment",
      ...assignmentProps
    })

    expectInsideDesignSystemAlert("Shared assignment ready: Biology Basics.")
  })

  it("renders start-confirmation assignment details through a design-system Alert", () => {
    openStartConfirmation(assignmentProps)

    expectInsideDesignSystemAlert("Shared assignment details")
  })

  it("renders the start-confirmation retake notice through a design-system Alert", () => {
    openStartConfirmation(assignmentProps)

    expectInsideDesignSystemAlert(
      "Retake uses the same questions. Answer options may be reshuffled."
    )
  })

  it("renders review guidance through a design-system Alert", async () => {
    await openStudyMode("review")

    expectInsideDesignSystemAlert(reviewModeTitle)
  }, 15000)

  it("renders the review mode label through a design-system Badge", async () => {
    await openStudyMode("review")

    expect(expectInsideDesignSystemBadge("Review Mode")).toHaveAttribute("data-ds-variant", "info")
  }, 15000)

  it("renders practice guidance through a design-system Alert", async () => {
    await openStudyMode("practice")

    expectInsideDesignSystemAlert(practiceModeTitle)
  }, 15000)

  it("renders the practice mode label through a design-system Badge", async () => {
    await openStudyMode("practice")

    expect(expectInsideDesignSystemBadge("Practice Mode")).toHaveAttribute(
      "data-ds-variant",
      "primary"
    )
  }, 15000)

  it("renders practice incorrect feedback through a design-system Alert", async () => {
    await openStudyMode("practice")

    fireEvent.click(await screen.findByRole("radio", { name: "Berlin" }))
    await expectInsideDesignSystemAlertAsync("Incorrect")
  }, 15000)

  it("renders graded hints through a design-system Alert", async () => {
    await startGradedAttempt()
    fireEvent.click(screen.getByRole("button", { name: "Show hint for question 11" }))

    expectInsideDesignSystemAlert("It is known as the city of lights.")
  }, 15000)

  it("renders unanswered warnings through a design-system Alert", async () => {
    await answerFirstQuestionAndSubmit()

    expectInsideDesignSystemAlert("Unanswered questions: 2")
  }, 15000)

  it("renders graded score summaries through a design-system Alert", async () => {
    await submitCompletedAttempt()

    expectInsideDesignSystemAlert(scoreSummaryText)
  }, 15000)

  it("renders result correctness through a design-system Badge", async () => {
    await submitCompletedAttempt()

    const correctBadge = expectInsideDesignSystemBadge("Correct")
    expect(correctBadge).toHaveAttribute("data-ds-variant", "success")
  }, 15000)

  it("renders queued submission recovery through a design-system Alert", async () => {
    mockQueuedSubmitFailure()

    await submitQueuedAttempt()

    expectInsideDesignSystemAlert("Submission failed. Answers queued locally.")
  }, 15000)

  it("retries queued submission from the design-system Alert action", async () => {
    const mutateAsync = mockQueuedSubmitFailure()

    await submitQueuedAttempt()

    fireEvent.click(screen.getByRole("button", { name: "Retry submission" }))

    await waitFor(() => {
      expect(mutateAsync).toHaveBeenCalledTimes(2)
    })
  }, 15000)
})
