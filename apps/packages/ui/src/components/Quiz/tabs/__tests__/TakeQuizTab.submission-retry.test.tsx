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

const connectivity = vi.hoisted(() => ({ online: true }))
const timerControls = vi.hoisted(() => ({ onExpire: null as null | (() => void) }))

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
  useServerOnline: () => connectivity.online
}))

vi.mock("../../hooks", () => ({
  useAttemptsQuery: vi.fn(),
  useQuizzesQuery: vi.fn(),
  useQuizQuery: vi.fn(),
  useStartAttemptMutation: vi.fn(),
  useSubmitAttemptMutation: vi.fn()
}))

vi.mock("../../hooks/useQuizTimer", () => ({
  useQuizTimer: vi.fn((options?: { onExpire?: () => void }) => {
    timerControls.onExpire = options?.onExpire ?? null
    return null
  })
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

describe("TakeQuizTab submission retry recovery", () => {
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
    connectivity.online = true
    timerControls.onExpire = null
    window.localStorage.clear()
    window.sessionStorage.clear()

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
            total_questions: 1,
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
        total_questions: 1,
        time_limit_seconds: 900,
        passing_score: 75
      }
    } as any)

    vi.mocked(useStartAttemptMutation).mockReturnValue({
      mutateAsync: vi.fn(async () => ({
        id: 123,
        quiz_id: 7,
        started_at: "2026-02-18T10:00:00Z",
        total_possible: 1,
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
      mutateAsync: vi.fn(async () => ({
        id: 123,
        quiz_id: 7,
        started_at: "2026-02-18T10:00:00Z",
        completed_at: "2026-02-18T10:02:00Z",
        score: 1,
        total_possible: 1,
        answers: [
          {
            question_id: 1,
            user_answer: "true",
            is_correct: true,
            correct_answer: "true"
          }
        ]
      })),
      isPending: false
    } as any)
  })

  const startAttempt = async () => {
    fireEvent.click(screen.getByRole("button", { name: /Start Quiz/i }))
    fireEvent.click(screen.getByRole("button", { name: "Begin Quiz" }))
    await screen.findByTestId("quiz-question-1")
  }

  it("queues failed submissions and allows manual retry", async () => {
    let calls = 0
    const mutateAsync = vi.fn(async () => {
      calls += 1
      if (calls === 1) {
        throw new Error("Network Error")
      }
      return {
        id: 123,
        quiz_id: 7,
        started_at: "2026-02-18T10:00:00Z",
        completed_at: "2026-02-18T10:02:00Z",
        score: 1,
        total_possible: 1,
        answers: [
          {
            question_id: 1,
            user_answer: "true",
            is_correct: true,
            correct_answer: "true"
          }
        ]
      }
    })
    vi.mocked(useSubmitAttemptMutation).mockReturnValue({
      mutateAsync,
      isPending: false
    } as any)

    render(<MemoryRouter><TakeQuizTab onNavigateToGenerate={() => {}} onNavigateToCreate={() => {}} /></MemoryRouter>)
    await startAttempt()

    fireEvent.click(screen.getByRole("radio", { name: "True" }))
    fireEvent.click(screen.getByRole("button", { name: "Submit" }))

    await screen.findByText("Submission failed. Answers queued locally.")
    fireEvent.click(screen.getByRole("button", { name: "Retry submission" }))

    await waitFor(() => {
      expect(mutateAsync).toHaveBeenCalledTimes(2)
    })
    expect(screen.getByText("Score: 1 / 1 (100%)")).toBeInTheDocument()
  }, 15000)

  it("auto-retries queued submission when connectivity is restored", async () => {
    connectivity.online = false
    let calls = 0
    const mutateAsync = vi.fn(async () => {
      calls += 1
      if (calls === 1) {
        throw new Error("Network Error")
      }
      return {
        id: 123,
        quiz_id: 7,
        started_at: "2026-02-18T10:00:00Z",
        completed_at: "2026-02-18T10:02:00Z",
        score: 1,
        total_possible: 1,
        answers: [
          {
            question_id: 1,
            user_answer: "true",
            is_correct: true,
            correct_answer: "true"
          }
        ]
      }
    })
    vi.mocked(useSubmitAttemptMutation).mockReturnValue({
      mutateAsync,
      isPending: false
    } as any)

    const view = render(<MemoryRouter><TakeQuizTab onNavigateToGenerate={() => {}} onNavigateToCreate={() => {}} /></MemoryRouter>)
    await startAttempt()

    fireEvent.click(screen.getByRole("radio", { name: "True" }))
    fireEvent.click(screen.getByRole("button", { name: "Submit" }))
    await screen.findByText("Submission failed. Answers queued locally.")
    expect(screen.getByText("You're offline. We'll retry automatically when your connection returns.")).toBeInTheDocument()

    connectivity.online = true
    view.rerender(<MemoryRouter><TakeQuizTab onNavigateToGenerate={() => {}} onNavigateToCreate={() => {}} /></MemoryRouter>)

    await waitFor(() => {
      expect(mutateAsync).toHaveBeenCalledTimes(2)
    })
    expect(screen.getByText("Score: 1 / 1 (100%)")).toBeInTheDocument()
  }, 15000)

  it("queues timer-expiry submission failures instead of losing answers", async () => {
    const mutateAsync = vi.fn(async () => {
      throw new Error("Network Error")
    })
    vi.mocked(useSubmitAttemptMutation).mockReturnValue({
      mutateAsync,
      isPending: false
    } as any)
    vi.mocked(useQuizTimer).mockImplementation((options?: { onExpire?: () => void }) => {
      timerControls.onExpire = options?.onExpire ?? null
      return null
    })

    render(<MemoryRouter><TakeQuizTab onNavigateToGenerate={() => {}} onNavigateToCreate={() => {}} /></MemoryRouter>)
    await startAttempt()

    await act(async () => {
      timerControls.onExpire?.()
      await Promise.resolve()
    })

    await waitFor(() => {
      expect(mutateAsync).toHaveBeenCalledTimes(1)
    })
    expect(screen.getByText("Time expired. Submission pending retry.")).toBeInTheDocument()
  }, 15000)
})
