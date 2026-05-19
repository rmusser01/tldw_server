import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
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
import { useQuizTimer } from "../../hooks/useQuizTimer"

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

describe("TakeQuizTab navigation and submit guardrails", () => {
  const originalMatchMedia = window.matchMedia
  const originalScrollIntoView = HTMLElement.prototype.scrollIntoView

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
    HTMLElement.prototype.scrollIntoView = originalScrollIntoView
  })

  beforeEach(() => {
    vi.clearAllMocks()
    window.localStorage.clear()

    HTMLElement.prototype.scrollIntoView = vi.fn()
    vi.mocked(useQuizTimer).mockReturnValue(null as any)

    vi.mocked(useAttemptsQuery).mockReturnValue({
      data: { items: [], count: 0 }
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
        id: 123,
        quiz_id: 7,
        started_at: "2026-02-18T10:00:00Z",
        total_possible: 3,
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
          },
          {
            id: 2,
            quiz_id: 7,
            question_type: "multiple_choice",
            question_text: "Powerhouse of the cell?",
            options: ["Nucleus", "Mitochondria", "Ribosome"],
            points: 1,
            order_index: 1,
            tags: null,
            deleted: false,
            client_id: "test",
            version: 1
          },
          {
            id: 3,
            quiz_id: 7,
            question_type: "fill_blank",
            question_text: "DNA stands for ____.",
            options: null,
            points: 1,
            order_index: 2,
            tags: null,
            deleted: false,
            client_id: "test",
            version: 1
          }
        ]
      }))
    } as any)

    vi.mocked(useSubmitAttemptMutation).mockReturnValue({
      mutateAsync: vi.fn(async () => ({
        id: 123,
        quiz_id: 7,
        started_at: "2026-02-18T10:00:00Z",
        completed_at: "2026-02-18T10:04:00Z",
        score: 2,
        total_possible: 3,
        answers: []
      })),
      isPending: false
    } as any)
  })

  const startQuizFlow = async () => {
    fireEvent.click(screen.getByRole("button", { name: /Start Quiz/i }))
    fireEvent.click(screen.getByRole("button", { name: "Begin Quiz" }))
    await screen.findByTestId("quiz-question-1")
  }

  it("shows fill-blank guidance and supports navigator jump-to-question", async () => {
    render(
      <MemoryRouter>
        <TakeQuizTab
          onNavigateToGenerate={() => {}}
          onNavigateToCreate={() => {}}
        />
      </MemoryRouter>
    )

    await startQuizFlow()

    expect(
      screen.getByText(/Case-insensitive match\./)
    ).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Go to question 3" }))
    expect(HTMLElement.prototype.scrollIntoView).toHaveBeenCalled()
  }, 15000)

  it("blocks submit when unanswered and highlights first missing question", async () => {
    const submitSpy = vi.fn(async () => ({
      id: 123,
      quiz_id: 7,
      started_at: "2026-02-18T10:00:00Z",
      completed_at: "2026-02-18T10:04:00Z",
      score: 1,
      total_possible: 3,
      answers: []
    }))
    vi.mocked(useSubmitAttemptMutation).mockReturnValue({
      mutateAsync: submitSpy,
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

    await startQuizFlow()

    fireEvent.click(screen.getByRole("radio", { name: "True" }))
    fireEvent.click(screen.getByRole("button", { name: "Submit" }))

    expect(submitSpy).not.toHaveBeenCalled()

    await waitFor(() => {
      expect(screen.getByText("Unanswered questions: 2, 3")).toBeInTheDocument()
    })

    expect(screen.getByTestId("quiz-question-2")).toHaveAttribute("data-highlighted", "true")
  }, 30000)

  it("queues failed submissions and allows retry from inline status", async () => {
    const submitSpy = vi
      .fn()
      .mockRejectedValueOnce(new Error("Network Error"))
      .mockResolvedValueOnce({
        id: 123,
        quiz_id: 7,
        started_at: "2026-02-18T10:00:00Z",
        completed_at: "2026-02-18T10:04:00Z",
        score: 2,
        total_possible: 3,
        answers: []
      })
    vi.mocked(useSubmitAttemptMutation).mockReturnValue({
      mutateAsync: submitSpy,
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

    await startQuizFlow()

    fireEvent.click(screen.getByRole("radio", { name: "True" }))
    fireEvent.click(screen.getByRole("radio", { name: "Mitochondria" }))
    fireEvent.change(screen.getByPlaceholderText("Enter the correct answer..."), {
      target: { value: "deoxyribonucleic acid" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Submit" }))

    await waitFor(() => {
      expect(screen.getByText("Submission failed. Answers queued locally.")).toBeInTheDocument()
    })

    const retryButtons = screen.getAllByRole("button", { name: "Retry submission" })
    const enabledRetryButton = retryButtons.find((button) => !button.hasAttribute("disabled"))
    fireEvent.click(enabledRetryButton ?? retryButtons[0])

    await waitFor(() => {
      expect(submitSpy).toHaveBeenCalledTimes(2)
    })
    await waitFor(() => {
      expect(screen.getByText("Score: 2 / 3 (67%)")).toBeInTheDocument()
    })
  }, 15000)

  it("queues timer-expiry submissions when submission fails", async () => {
    let expireHandler: (() => void) | null = null
    vi.mocked(useQuizTimer).mockImplementation((params: any) => {
      expireHandler = params.onExpire
      return null as any
    })

    const submitSpy = vi.fn().mockRejectedValue(new Error("offline"))
    vi.mocked(useSubmitAttemptMutation).mockReturnValue({
      mutateAsync: submitSpy,
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

    await startQuizFlow()

    fireEvent.click(screen.getByRole("radio", { name: "True" }))
    expect(expireHandler).toBeTypeOf("function")

    await act(async () => {
      expireHandler?.()
    })

    await waitFor(() => {
      expect(submitSpy).toHaveBeenCalledTimes(1)
    })
    await waitFor(() => {
      expect(screen.getByText("Time expired. Submission pending retry.")).toBeInTheDocument()
    })
    expect(submitSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        attemptId: 123,
        answers: [
          {
            question_id: 1,
            user_answer: "true",
            hint_used: false
          }
        ]
      })
    )
  }, 15000)

  it("shows sticky mobile timer bar while taking a timed quiz", async () => {
    vi.mocked(useQuizTimer).mockReturnValue({
      totalSeconds: 599,
      formattedTime: "09:59",
      isDanger: false,
      isWarning: true,
      isExpired: false
    } as any)

    render(
      <MemoryRouter>
        <TakeQuizTab
          onNavigateToGenerate={() => {}}
          onNavigateToCreate={() => {}}
        />
      </MemoryRouter>
    )

    await startQuizFlow()

    const timerBar = screen.getByTestId("quiz-mobile-timer-bar")
    expect(timerBar).toHaveTextContent("Time Remaining")
    expect(timerBar).toHaveTextContent("09:59")
    expect(timerBar.className).toContain("md:hidden")
  }, 15000)

  it("applies 44px touch-target sizing to key quiz actions", async () => {
    render(
      <MemoryRouter>
        <TakeQuizTab
          onNavigateToGenerate={() => {}}
          onNavigateToCreate={() => {}}
        />
      </MemoryRouter>
    )

    expect(screen.getByRole("button", { name: /Start Quiz/i }).className).toContain("min-h-11")

    await startQuizFlow()

    expect(screen.getByRole("button", { name: "Back to list" }).className).toContain("min-h-11")
    expect(screen.getByRole("button", { name: "Submit" }).className).toContain("min-h-11")
  }, 15000)
})
