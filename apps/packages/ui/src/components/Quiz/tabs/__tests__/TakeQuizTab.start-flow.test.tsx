import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
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

const ASSERTION_REASONING_OPTIONS = [
  "Both the assertion and reason are true, and the reason correctly explains the assertion.",
  "Both the assertion and reason are true, but the reason does not explain the assertion.",
  "The assertion is true, but the reason is false.",
  "The assertion is false, but the reason is true.",
  "Both the assertion and reason are false."
]

vi.mock("react-router-dom", async (importOriginal) => {
  const actual = await importOriginal<typeof import("react-router-dom")>()
  return {
    ...actual,
    Link: ({ to, children, ...props }: Record<string, unknown>) => <a href={to as string} {...props}>{children as React.ReactNode}</a>
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
    Element.prototype.scrollIntoView = vi.fn()

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

  it("labels and shuffles Best of Five questions while preserving answer mapping", async () => {
    const questionId = 41
    let attemptId = 141
    let shuffledEntries = buildShuffledOptionEntries(
      ["A", "B", "C", "D", "E"],
      questionId,
      attemptId
    )
    for (
      let attempts = 0;
      shuffledEntries.every((entry, index) => entry.originalIndex === index) && attempts < 128;
      attempts += 1
    ) {
      attemptId += 1
      shuffledEntries = buildShuffledOptionEntries(
        ["A", "B", "C", "D", "E"],
        questionId,
        attemptId
      )
    }
    expect(shuffledEntries.some((entry, index) => entry.originalIndex !== index)).toBe(true)
    vi.mocked(useStartAttemptMutation).mockReturnValue({
      mutateAsync: vi.fn(async () => ({
        id: attemptId,
        quiz_id: 7,
        started_at: "2026-02-18T10:00:00Z",
        total_possible: 1,
        answers: [],
        questions: [
          {
            id: questionId,
            quiz_id: 7,
            question_type: "multiple_choice",
            question_text: "Which option is the best supported answer?",
            options: ["A", "B", "C", "D", "E"],
            points: 1,
            order_index: 0,
            tags: ["best_of_five"],
            deleted: false,
            client_id: "test",
            version: 1
          }
        ]
      }))
    } as any)
    const submitMutate = vi.fn(async ({ answers }: {
      answers: Array<{ question_id: number; user_answer: number; hint_used?: boolean }>
    }) => ({
      id: attemptId,
      quiz_id: 7,
      started_at: "2026-02-18T10:00:00Z",
      completed_at: "2026-02-18T10:03:00Z",
      score: 1,
      total_possible: 1,
      answers: [{
        question_id: questionId,
        user_answer: answers[0]?.user_answer,
        is_correct: true,
        correct_answer: answers[0]?.user_answer,
        points_awarded: 1
      }]
    }))
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
    expect(screen.getByText("Best of Five")).toBeInTheDocument()
    const radioOrder = screen.getAllByRole("radio").map((node) => Number((node as HTMLInputElement).value))
    expect(radioOrder).toEqual(shuffledEntries.map((entry) => entry.originalIndex))

    fireEvent.click(screen.getAllByRole("radio")[0])
    fireEvent.click(screen.getByRole("button", { name: "Submit" }))

    await waitFor(() => {
      expect(submitMutate).toHaveBeenCalledWith({
        attemptId,
        answers: [{
          question_id: questionId,
          user_answer: shuffledEntries[0]?.originalIndex,
          hint_used: false
        }]
      })
    })
  }, 15000)

  it("keeps Assertion / Reasoning order fixed and renders one cited guide in taking and results", async () => {
    const questionId = 51
    let attemptId = 251
    let shuffledEntries = buildShuffledOptionEntries(
      ASSERTION_REASONING_OPTIONS,
      questionId,
      attemptId
    )
    for (
      let attempts = 0;
      shuffledEntries.every((entry, index) => entry.originalIndex === index) && attempts < 128;
      attempts += 1
    ) {
      attemptId += 1
      shuffledEntries = buildShuffledOptionEntries(
        ASSERTION_REASONING_OPTIONS,
        questionId,
        attemptId
      )
    }
    expect(shuffledEntries.some((entry, index) => entry.originalIndex !== index)).toBe(true)

    vi.mocked(useStartAttemptMutation).mockReturnValue({
      mutateAsync: vi.fn(async () => ({
        id: attemptId,
        quiz_id: 7,
        started_at: "2026-02-18T10:00:00Z",
        total_possible: 1,
        answers: [],
        questions: [
          {
            id: questionId,
            quiz_id: 7,
            question_type: "multiple_choice",
            question_text: "**Assertion:** The drug lowers blood pressure.\n\n**Reason:** It reduces vascular resistance.",
            options: ASSERTION_REASONING_OPTIONS,
            points: 1,
            order_index: 0,
            tags: ["Assertion / Reasoning"],
            deleted: false,
            client_id: "test",
            version: 1
          }
        ]
      }))
    } as any)

    const submitMutate = vi.fn(async ({ answers }: {
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
          correct_answer: 3,
          points_awarded: 1,
          explanation: "The assertion is false, while reduced vascular resistance is true.",
          source_citations: [{ label: "Pharmacology source" }]
        }
      ]
    }))
    vi.mocked(useSubmitAttemptMutation).mockReturnValue({
      mutateAsync: submitMutate,
      isPending: false
    } as any)

    render(
      <MemoryRouter>
        <TakeQuizTab onNavigateToGenerate={() => {}} onNavigateToCreate={() => {}} />
      </MemoryRouter>
    )

    fireEvent.click(screen.getByRole("button", { name: /Start Quiz/i }))
    fireEvent.click(screen.getByRole("button", { name: "Begin Quiz" }))

    await screen.findByTestId(`quiz-question-${questionId}`)
    expect(screen.getAllByTestId("assertion-reasoning-scale")).toHaveLength(1)
    expect(screen.getByText("Assertion / Reasoning")).toBeInTheDocument()
    expect(within(screen.getByTestId("assertion-reasoning-scale")).getByText(
      `A. ${ASSERTION_REASONING_OPTIONS[0]}`
    )).toBeInTheDocument()
    expect(screen.getAllByRole("radio").map((node) => Number((node as HTMLInputElement).value))).toEqual([0, 1, 2, 3, 4])
    expect(shuffledEntries.map((entry) => entry.originalIndex)).not.toEqual([0, 1, 2, 3, 4])

    fireEvent.click(screen.getByRole("radio", { name: ASSERTION_REASONING_OPTIONS[3] }))
    fireEvent.click(screen.getByRole("button", { name: "Submit" }))

    await waitFor(() => {
      expect(submitMutate).toHaveBeenCalledWith({
        attemptId,
        answers: [{ question_id: questionId, user_answer: 3, hint_used: false }]
      })
    })
    expect(await screen.findByText("The assertion is false, while reduced vascular resistance is true.")).toBeInTheDocument()
    expect(screen.getByText("Pharmacology source")).toBeInTheDocument()
    expect(screen.getAllByTestId("assertion-reasoning-scale")).toHaveLength(1)
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
    const optionLabels = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]
    const questionId = 11

    const firstAttemptId = 200
    let secondAttemptId = 201
    const firstOrder = buildShuffledOptionEntries(optionLabels, questionId, firstAttemptId).map((entry) => entry.originalIndex)
    let secondOrder = buildShuffledOptionEntries(optionLabels, questionId, secondAttemptId).map((entry) => entry.originalIndex)
    for (
      let attempts = 0;
      secondOrder.join(",") === firstOrder.join(",") && attempts < 128;
      attempts += 1
    ) {
      secondAttemptId += 1
      secondOrder = buildShuffledOptionEntries(optionLabels, questionId, secondAttemptId).map((entry) => entry.originalIndex)
    }
    expect(secondOrder).not.toEqual(firstOrder)

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
            tags: ["assertion/reasoning-extra"],
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
            tags: ["assertion/reasoning-extra"],
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
    expect(screen.queryByRole("combobox", { name: "Pick the second Greek letter." })).not.toBeInTheDocument()

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

  it("presents and submits grouped EMQ stems independently through one shared bank", async () => {
    const groupPrompt = "For each patient, choose the most likely diagnosis from the shared option bank."
    const optionBank = ["Acute appendicitis", "Renal colic", "Gastroenteritis"]
    const firstStem = "A 24-year-old has migrating right lower quadrant pain."
    const secondStem = "A 45-year-old has colicky flank pain radiating to the groin."
    const questions = [
      {
        id: 81,
        quiz_id: 7,
        question_type: "multiple_choice",
        question_text: firstStem,
        options: optionBank,
        group_id: "abdominal-pain",
        group_prompt: groupPrompt,
        points: 1,
        order_index: 0,
        tags: null,
        deleted: false,
        client_id: "test",
        version: 1
      },
      {
        id: 82,
        quiz_id: 7,
        question_type: "multiple_choice",
        question_text: secondStem,
        options: optionBank,
        group_id: "abdominal-pain",
        group_prompt: groupPrompt,
        points: 1,
        order_index: 1,
        tags: null,
        deleted: false,
        client_id: "test",
        version: 1
      }
    ]
    vi.mocked(useStartAttemptMutation).mockReturnValue({
      mutateAsync: vi.fn(async () => ({
        id: 808,
        quiz_id: 7,
        started_at: "2026-02-18T10:00:00Z",
        total_possible: 2,
        answers: [],
        questions
      }))
    } as any)

    const submitMutate = vi.fn(async ({ answers }: {
      answers: Array<{ question_id: number; user_answer: number; hint_used?: boolean }>
    }) => ({
      id: 808,
      quiz_id: 7,
      started_at: "2026-02-18T10:00:00Z",
      completed_at: "2026-02-18T10:03:00Z",
      score: 2,
      total_possible: 2,
      answers: [
        {
          question_id: 81,
          user_answer: answers.find((answer) => answer.question_id === 81)?.user_answer,
          is_correct: true,
          correct_answer: 0,
          points_awarded: 1,
          explanation: "The pain pattern supports appendicitis.",
          source_citations: [{ label: "Appendicitis source" }]
        },
        {
          question_id: 82,
          user_answer: answers.find((answer) => answer.question_id === 82)?.user_answer,
          is_correct: true,
          correct_answer: 1,
          points_awarded: 1,
          explanation: "The radiation pattern supports renal colic.",
          source_citations: [{ label: "Renal colic source" }]
        }
      ]
    }))
    vi.mocked(useSubmitAttemptMutation).mockReturnValue({
      mutateAsync: submitMutate,
      isPending: false
    } as any)

    render(
      <MemoryRouter>
        <TakeQuizTab onNavigateToGenerate={() => {}} onNavigateToCreate={() => {}} />
      </MemoryRouter>
    )

    fireEvent.click(screen.getByRole("button", { name: /Start Quiz/i }))
    fireEvent.click(screen.getByRole("button", { name: "Begin Quiz" }))

    const firstSelect = await screen.findByRole("combobox", { name: firstStem })
    expect(screen.getByRole("combobox", { name: secondStem })).toBeInTheDocument()
    expect(screen.getAllByText(groupPrompt)).toHaveLength(1)
    expect(screen.getAllByText(firstStem)).toHaveLength(1)
    expect(screen.getAllByText(secondStem)).toHaveLength(1)
    expect(screen.getAllByTestId("emq-group-bank")).toHaveLength(1)
    expect(within(screen.getByTestId("emq-group-bank")).getByText("A. Acute appendicitis")).toBeInTheDocument()
    expect(within(screen.getByTestId("emq-group-bank")).getByText("B. Renal colic")).toBeInTheDocument()
    expect(screen.queryAllByRole("radio")).toHaveLength(0)

    fireEvent.mouseDown(firstSelect)
    const firstVisibleOption = (await screen.findAllByText("A. Acute appendicitis")).find(
      (element) => element.closest(".ant-select-item-option") && !element.closest(".ant-select-dropdown-hidden")
    )
    expect(firstVisibleOption).toBeDefined()
    fireEvent.click(firstVisibleOption as HTMLElement)
    const refreshedSecondSelect = screen.getByRole("combobox", { name: secondStem })
    fireEvent.mouseDown(refreshedSecondSelect)
    fireEvent.keyDown(refreshedSecondSelect, {
      key: "ArrowDown",
      code: "ArrowDown",
      keyCode: 40,
      which: 40
    })
    fireEvent.keyDown(refreshedSecondSelect, {
      key: "Enter",
      code: "Enter",
      keyCode: 13,
      which: 13
    })
    fireEvent.click(screen.getByRole("button", { name: "Submit" }))

    await waitFor(() => {
      expect(submitMutate).toHaveBeenCalledWith({
        attemptId: 808,
        answers: [
          { question_id: 81, user_answer: 0, hint_used: false },
          { question_id: 82, user_answer: 1, hint_used: false }
        ]
      })
    })

    expect(await screen.findByText("The pain pattern supports appendicitis.")).toBeInTheDocument()
    expect(screen.getByText("The radiation pattern supports renal colic.")).toBeInTheDocument()
    expect(screen.getByText("Appendicitis source")).toBeInTheDocument()
    expect(screen.getByText("Renal colic source")).toBeInTheDocument()
    expect(screen.getAllByText(groupPrompt)).toHaveLength(1)
    expect(screen.getAllByTestId("emq-group-bank")).toHaveLength(1)
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
