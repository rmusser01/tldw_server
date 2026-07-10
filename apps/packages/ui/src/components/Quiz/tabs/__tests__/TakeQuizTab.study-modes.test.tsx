import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { MemoryRouter } from "react-router-dom"
import { ASSERTION_REASONING_OPTIONS, TakeQuizTab } from "../TakeQuizTab"
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
    Element.prototype.scrollIntoView = vi.fn()

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

  it("uses the true-false control for a malformed grouped true-false question", async () => {
    window.sessionStorage.setItem(
      TAKE_QUIZ_LIST_PREFS_KEY,
      JSON.stringify({ modePreference: "practice" })
    )

    vi.mocked(listQuestions).mockResolvedValue({
      items: [
        {
          id: 20,
          quiz_id: 7,
          question_type: "true_false",
          question_text: "This malformed legacy record has group metadata.",
          options: ["Distractor A", "Distractor B"],
          correct_answer: "true",
          group_id: "legacy-group"
        }
      ],
      count: 1
    } as any)

    render(<MemoryRouter><TakeQuizTab onNavigateToGenerate={() => {}} onNavigateToCreate={() => {}} /></MemoryRouter>)

    fireEvent.click(screen.getByRole("button", { name: /Start Practice/i }))

    expect(await screen.findByRole("radio", { name: "True" })).toBeInTheDocument()
    expect(screen.getByRole("radio", { name: "False" })).toBeInTheDocument()
    expect(screen.queryByRole("combobox")).not.toBeInTheDocument()
  }, 15000)

  it("keeps Assertion / Reasoning practice order fixed and shows evidence after incorrect and correct answers", async () => {
    window.sessionStorage.setItem(
      TAKE_QUIZ_LIST_PREFS_KEY,
      JSON.stringify({ modePreference: "practice" })
    )

    vi.mocked(listQuestions).mockResolvedValue({
      items: [
        {
          id: 21,
          quiz_id: 7,
          question_type: "multiple_choice",
          question_text: "**Assertion:** Insulin lowers blood glucose.\n\n**Reason:** Insulin promotes cellular glucose uptake.",
          options: ASSERTION_REASONING_OPTIONS,
          correct_answer: 0,
          explanation: "Both statements are true, and increased glucose uptake explains the effect.",
          source_citations: [{ label: "Endocrinology source" }],
          tags: ["assertion_reasoning"]
        }
      ],
      count: 1
    } as any)

    render(<MemoryRouter><TakeQuizTab onNavigateToGenerate={() => {}} onNavigateToCreate={() => {}} /></MemoryRouter>)

    fireEvent.click(screen.getByRole("button", { name: /Start Practice/i }))

    expect(await screen.findAllByTestId("assertion-reasoning-scale")).toHaveLength(1)
    expect(screen.getByText("Assertion / Reasoning")).toBeInTheDocument()
    expect(screen.getAllByRole("radio").map((node) => Number((node as HTMLInputElement).value))).toEqual([0, 1, 2, 3, 4])

    fireEvent.click(screen.getByRole("radio", { name: ASSERTION_REASONING_OPTIONS[2] }))
    expect(await screen.findByText("Incorrect")).toBeInTheDocument()
    expect(screen.getByText("Both statements are true, and increased glucose uptake explains the effect.")).toBeInTheDocument()
    expect(screen.getByText("Endocrinology source")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("radio", { name: ASSERTION_REASONING_OPTIONS[0] }))
    expect(await screen.findByText("Correct")).toBeInTheDocument()
    expect(screen.getByText("Both statements are true, and increased glucose uptake explains the effect.")).toBeInTheDocument()
    expect(screen.getByText("Endocrinology source")).toBeInTheDocument()
    expect(screen.getAllByTestId("assertion-reasoning-scale")).toHaveLength(1)
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

  it("renders one Assertion / Reasoning guide with per-question evidence in review mode", async () => {
    window.sessionStorage.setItem(
      TAKE_QUIZ_LIST_PREFS_KEY,
      JSON.stringify({ modePreference: "review" })
    )

    vi.mocked(listQuestions).mockResolvedValue({
      items: [
        {
          id: 41,
          quiz_id: 7,
          question_type: "multiple_choice",
          question_text: "**Assertion:** Vaccines stimulate adaptive immunity.\n\n**Reason:** They expose the immune system to antigen.",
          options: ASSERTION_REASONING_OPTIONS,
          correct_answer: 0,
          explanation: "Antigen exposure explains the adaptive immune response.",
          source_citations: [{ label: "Immunology source" }],
          tags: ["assertion_reasoning"]
        },
        {
          id: 42,
          quiz_id: 7,
          question_type: "multiple_choice",
          question_text: "**Assertion:** Antibiotics treat influenza.\n\n**Reason:** Influenza is caused by bacteria.",
          options: ASSERTION_REASONING_OPTIONS,
          correct_answer: 4,
          explanation: "Both statements are false because influenza is viral.",
          source_citations: [{ label: "Virology source" }],
          tags: ["assertion_reasoning"]
        }
      ],
      count: 2
    } as any)

    render(<MemoryRouter><TakeQuizTab onNavigateToGenerate={() => {}} onNavigateToCreate={() => {}} /></MemoryRouter>)

    fireEvent.click(screen.getByRole("button", { name: /Open Review/i }))

    expect(await screen.findAllByTestId("assertion-reasoning-scale")).toHaveLength(1)
    expect(screen.getByText("Antigen exposure explains the adaptive immune response.")).toBeInTheDocument()
    expect(screen.getByText("Both statements are false because influenza is viral.")).toBeInTheDocument()
    expect(screen.getByText("Immunology source")).toBeInTheDocument()
    expect(screen.getByText("Virology source")).toBeInTheDocument()
    expect(screen.getAllByText("Assertion / Reasoning")).toHaveLength(2)
  }, 15000)

  it.each([
    {
      label: "non-MCQ type",
      question: {
        id: 51,
        quiz_id: 7,
        question_type: "true_false",
        question_text: "**Assertion:** A tagged true/false question.\n\n**Reason:** This isolates the type guard.",
        options: ASSERTION_REASONING_OPTIONS,
        correct_answer: "true",
        tags: ["assertion_reasoning"]
      }
    },
    {
      label: "grouped question",
      question: {
        id: 52,
        quiz_id: 7,
        question_type: "multiple_choice",
        question_text: "**Assertion:** A tagged EMQ stem.\n\n**Reason:** This isolates the grouping guard.",
        options: ASSERTION_REASONING_OPTIONS,
        correct_answer: 0,
        group_id: "manual-emq",
        group_prompt: "Choose from the shared bank.",
        tags: ["assertion_reasoning"]
      }
    },
    {
      label: "non-five-option scale",
      question: {
        id: 53,
        quiz_id: 7,
        question_type: "multiple_choice",
        question_text: "**Assertion:** A tagged question has four options.\n\n**Reason:** This isolates the scale-size guard.",
        options: ASSERTION_REASONING_OPTIONS.slice(0, 4),
        correct_answer: 0,
        tags: ["assertion_reasoning"]
      }
    }
  ])("does not apply Assertion / Reasoning UI to a $label", async ({ question }) => {
    window.sessionStorage.setItem(
      TAKE_QUIZ_LIST_PREFS_KEY,
      JSON.stringify({ modePreference: "review" })
    )

    vi.mocked(listQuestions).mockResolvedValue({
      items: [
        {
          id: 50,
          quiz_id: 7,
          question_type: "multiple_choice",
          question_text: "**Assertion:** A valid tagged question is present.\n\n**Reason:** Its body and scale are canonical.",
          options: ASSERTION_REASONING_OPTIONS,
          correct_answer: 0,
          tags: ["assertion_reasoning"]
        },
        question
      ],
      count: 2
    } as any)

    render(<MemoryRouter><TakeQuizTab onNavigateToGenerate={() => {}} onNavigateToCreate={() => {}} /></MemoryRouter>)

    fireEvent.click(screen.getByRole("button", { name: /Open Review/i }))

    expect(await screen.findByText(/Review mode is read-only/)).toBeInTheDocument()
    expect(screen.queryByTestId("assertion-reasoning-scale")).not.toBeInTheDocument()
    expect(screen.queryByText("Assertion / Reasoning")).not.toBeInTheDocument()
  }, 15000)

  it.each([
    {
      label: "noncanonical five-option scale",
      questionText: "**Assertion:** A tagged question has five options.\n\n**Reason:** The options are not the canonical scale.",
      options: ["One", "Two", "Three", "Four", "Five"]
    },
    {
      label: "unlabeled question body",
      questionText: "A tagged question without explicit assertion and reason labels.",
      options: ASSERTION_REASONING_OPTIONS
    }
  ])("does not apply Assertion / Reasoning UI to a $label", async ({ questionText, options }) => {
    window.sessionStorage.setItem(
      TAKE_QUIZ_LIST_PREFS_KEY,
      JSON.stringify({ modePreference: "review" })
    )

    vi.mocked(listQuestions).mockResolvedValue({
      items: [
        {
          id: 54,
          quiz_id: 7,
          question_type: "multiple_choice",
          question_text: questionText,
          options,
          correct_answer: 0,
          tags: ["assertion_reasoning"]
        }
      ],
      count: 1
    } as any)

    render(<MemoryRouter><TakeQuizTab onNavigateToGenerate={() => {}} onNavigateToCreate={() => {}} /></MemoryRouter>)

    fireEvent.click(screen.getByRole("button", { name: /Open Review/i }))

    expect(await screen.findByText(/Review mode is read-only/)).toBeInTheDocument()
    expect(screen.queryByTestId("assertion-reasoning-scale")).not.toBeInTheDocument()
    expect(screen.queryByText("Assertion / Reasoning")).not.toBeInTheDocument()
  }, 15000)

  it("fails closed when tagged questions disagree on the answer scale", async () => {
    window.sessionStorage.setItem(
      TAKE_QUIZ_LIST_PREFS_KEY,
      JSON.stringify({ modePreference: "review" })
    )

    vi.mocked(listQuestions).mockResolvedValue({
      items: [
        {
          id: 61,
          quiz_id: 7,
          question_type: "multiple_choice",
          question_text: "**Assertion:** The first tagged question is valid.\n\n**Reason:** Its labels are explicit.",
          options: ASSERTION_REASONING_OPTIONS,
          correct_answer: 0,
          tags: ["assertion_reasoning"]
        },
        {
          id: 62,
          quiz_id: 7,
          question_type: "multiple_choice",
          question_text: "**Assertion:** The second tagged question is valid.\n\n**Reason:** Its labels are explicit.",
          options: [...ASSERTION_REASONING_OPTIONS].reverse(),
          correct_answer: 4,
          tags: ["assertion_reasoning"]
        }
      ],
      count: 2
    } as any)

    render(<MemoryRouter><TakeQuizTab onNavigateToGenerate={() => {}} onNavigateToCreate={() => {}} /></MemoryRouter>)

    fireEvent.click(screen.getByRole("button", { name: /Open Review/i }))

    expect(await screen.findByText(/Review mode is read-only/)).toBeInTheDocument()
    expect(screen.queryByTestId("assertion-reasoning-scale")).not.toBeInTheDocument()
    expect(screen.queryByText("Assertion / Reasoning")).not.toBeInTheDocument()
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

  it("keeps an EMQ group atomic in a practice pool and grades each stem immediately", async () => {
    const groupPrompt = "For each presentation, choose one condition from the shared bank."
    const optionBank = ["Asthma", "Pneumonia", "Pulmonary embolism"]
    const stems = [
      "Episodic wheeze improves after a bronchodilator.",
      "Fever, productive cough, and focal crackles are present.",
      "Sudden pleuritic pain follows a long-haul flight."
    ]
    window.sessionStorage.setItem(
      TAKE_QUIZ_LIST_PREFS_KEY,
      JSON.stringify({ modePreference: "practice", studyPoolSize: 1, studyPoolSeedOverride: 77 })
    )
    vi.mocked(listQuestions).mockResolvedValue({
      items: stems.map((questionText, index) => ({
        id: 601 + index,
        quiz_id: 7,
        question_type: "multiple_choice",
        question_text: questionText,
        options: optionBank,
        correct_answer: index,
        explanation: `Explanation ${index + 1}`,
        group_id: "respiratory-emq",
        group_prompt: groupPrompt
      })),
      count: stems.length
    } as any)

    render(<MemoryRouter><TakeQuizTab onNavigateToGenerate={() => {}} onNavigateToCreate={() => {}} /></MemoryRouter>)

    fireEvent.click(screen.getByRole("button", { name: /Start Practice/i }))

    await waitFor(() => {
      expect(screen.getAllByTestId(/quiz-question-/)).toHaveLength(3)
    })
    expect(screen.getAllByTestId("emq-group-bank")).toHaveLength(1)
    expect(screen.getAllByText(groupPrompt)).toHaveLength(1)
    stems.forEach((stem) => {
      expect(screen.getByRole("combobox", { name: stem })).toBeInTheDocument()
    })
    expect(within(screen.getByTestId("emq-group-bank")).getByText("A. Asthma")).toBeInTheDocument()

    fireEvent.mouseDown(screen.getByRole("combobox", { name: stems[0] }))
    const visibleOption = (await screen.findAllByText("A. Asthma")).find(
      (element) => element.closest(".ant-select-item-option") && !element.closest(".ant-select-dropdown-hidden")
    )
    expect(visibleOption).toBeDefined()
    fireEvent.click(visibleOption as HTMLElement)

    expect(await screen.findByText("Correct")).toBeInTheDocument()
  }, 15000)

  it("keeps one shared EMQ bank with per-stem evidence in review mode", async () => {
    const groupPrompt = "Choose the best investigation for each presentation."
    const optionBank = ["ECG", "Chest X-ray", "CT pulmonary angiogram"]
    window.sessionStorage.setItem(
      TAKE_QUIZ_LIST_PREFS_KEY,
      JSON.stringify({ modePreference: "review" })
    )
    vi.mocked(listQuestions).mockResolvedValue({
      items: [
        {
          id: 711,
          quiz_id: 7,
          question_type: "multiple_choice",
          question_text: "Exertional chest pressure with diaphoresis.",
          options: optionBank,
          correct_answer: 0,
          explanation: "An ECG is the immediate investigation.",
          source_citations: [{ label: "Cardiology source" }],
          group_id: "investigation-emq",
          group_prompt: groupPrompt
        },
        {
          id: 712,
          quiz_id: 7,
          question_type: "true_false",
          question_text: "This ungrouped question separates the stems.",
          options: null,
          correct_answer: "true"
        },
        {
          id: 713,
          quiz_id: 7,
          question_type: "multiple_choice",
          question_text: "Sudden hypoxia after recent surgery.",
          options: optionBank,
          correct_answer: 2,
          explanation: "CT pulmonary angiography evaluates suspected embolism.",
          source_citations: [{ label: "Respiratory source" }],
          group_id: "investigation-emq",
          group_prompt: groupPrompt
        }
      ],
      count: 3
    } as any)

    render(<MemoryRouter><TakeQuizTab onNavigateToGenerate={() => {}} onNavigateToCreate={() => {}} /></MemoryRouter>)

    fireEvent.click(screen.getByRole("button", { name: /Open Review/i }))

    expect(await screen.findByText("An ECG is the immediate investigation.")).toBeInTheDocument()
    expect(screen.getByText("CT pulmonary angiography evaluates suspected embolism.")).toBeInTheDocument()
    expect(screen.getByText("Cardiology source")).toBeInTheDocument()
    expect(screen.getByText("Respiratory source")).toBeInTheDocument()
    expect(screen.getAllByText(groupPrompt)).toHaveLength(1)
    expect(screen.getAllByTestId("emq-group-bank")).toHaveLength(1)
    expect(screen.getAllByText("EMQ")).toHaveLength(2)
    expect(within(screen.getByTestId("emq-group-bank")).getByText("C. CT pulmonary angiogram")).toBeInTheDocument()
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
