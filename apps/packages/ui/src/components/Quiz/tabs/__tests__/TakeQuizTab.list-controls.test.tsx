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

describe("TakeQuizTab list controls and default passing policy", () => {
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
    window.sessionStorage.clear()

    vi.mocked(useAttemptsQuery).mockReturnValue({
      data: { items: [], count: 0 }
    } as any)

    vi.mocked(useQuizzesQuery).mockImplementation((params: any) => ({
      data: {
        items: [
          {
            id: 7,
            name: "Biology Basics",
            description: "Cell structures and functions",
            total_questions: 1,
            time_limit_seconds: 900,
            passing_score: null,
            media_id: 42,
            source_bundle_json: [
              { source_type: "media", source_id: "42" },
              { source_type: "note", source_id: "note-1" },
              { source_type: "flashcard_deck", source_id: "20" },
              { source_type: "flashcard_card", source_id: "card-1" }
            ],
            created_at: "2026-02-16T12:00:00Z"
          }
        ],
        count: 1
      },
      isLoading: false,
      _params: params
    } as any))

    vi.mocked(useQuizQuery).mockImplementation((quizId: any) => ({
      data:
        quizId === 42
          ? {
              id: 42,
              name: "Workspace Biology",
              description: "Scoped quiz",
              workspace_id: "workspace-1",
              total_questions: 2,
              time_limit_seconds: 600,
              passing_score: 80
            }
          : {
              id: 7,
              name: "Biology Basics",
              total_questions: 1,
              time_limit_seconds: 900,
              passing_score: null
            }
    } as any))

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
        completed_at: "2026-02-18T10:03:00Z",
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

  it("forwards search query to quiz list hook", async () => {
    render(
      <MemoryRouter>
        <TakeQuizTab
          onNavigateToGenerate={() => {}}
          onNavigateToCreate={() => {}}
        />
      </MemoryRouter>
    )

    const input = screen.getByPlaceholderText("Search quizzes...")
    fireEvent.change(input, { target: { value: "biology" } })

    await waitFor(() => {
      expect(vi.mocked(useQuizzesQuery)).toHaveBeenLastCalledWith(
        expect.objectContaining({ q: "biology" })
      )
    })
  }, 15000)

  it("shows explicit default passing score policy when quiz has no passing score", async () => {
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

    await screen.findByTestId("quiz-question-1")
    fireEvent.click(screen.getByRole("radio", { name: "True" }))
    fireEvent.click(screen.getByRole("button", { name: "Submit" }))

    await waitFor(() => {
      expect(
        screen.getByText("No passing score set. Using default: 70%.")
      ).toBeInTheDocument()
    })
  }, 15000)

  it("renders loading skeleton placeholders while quiz list is loading", () => {
    vi.mocked(useQuizzesQuery).mockReturnValue({
      data: undefined,
      isLoading: true
    } as any)

    render(
      <MemoryRouter>
        <TakeQuizTab
          onNavigateToGenerate={() => {}}
          onNavigateToCreate={() => {}}
        />
      </MemoryRouter>
    )

    expect(screen.getByTestId("take-loading-skeleton")).toBeInTheDocument()
  })

  it("shows source badges for mixed-source quizzes", async () => {
    render(
      <MemoryRouter>
        <TakeQuizTab
          onNavigateToGenerate={() => {}}
          onNavigateToCreate={() => {}}
        />
      </MemoryRouter>
    )

    await waitFor(() => {
      expect(screen.getByTestId("take-quiz-source-media-7")).toBeInTheDocument()
    })
    expect(screen.getByTestId("take-quiz-source-notes-7")).toBeInTheDocument()
    expect(screen.getByTestId("take-quiz-source-flashcards-7")).toBeInTheDocument()
    expect(screen.getByText("Media 1")).toBeInTheDocument()
    expect(screen.getByText("Notes 1")).toBeInTheDocument()
    expect(screen.getByText("Flashcards 2")).toBeInTheDocument()
  })

  it("force-shows a direct workspace quiz without changing the general list default", async () => {
    vi.mocked(useQuizzesQuery).mockImplementation((params: any) => ({
      data: {
        items: params?.include_workspace_items
          ? [
              {
                id: 42,
                name: "Workspace Biology",
                description: "Scoped quiz",
                workspace_id: "workspace-1",
                total_questions: 2,
                time_limit_seconds: 600,
                passing_score: 80,
                media_id: null,
                created_at: "2026-02-16T12:00:00Z"
              }
            ]
          : [],
        count: 0
      },
      isLoading: false,
      _params: params
    } as any))

    render(
      <MemoryRouter>
        <TakeQuizTab
          onNavigateToGenerate={() => {}}
          onNavigateToCreate={() => {}}
          startQuizId={42}
          highlightQuizId={42}
        />
      </MemoryRouter>
    )

    const dialog = await screen.findByRole("dialog")
    expect(within(dialog).getByText("Workspace Biology")).toBeInTheDocument()
    expect(within(dialog).getByText("Questions")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Begin Quiz" })).toBeInTheDocument()

    expect(vi.mocked(useQuizzesQuery)).toHaveBeenCalledWith(
      expect.objectContaining({
        q: undefined
      })
    )
    expect(vi.mocked(useQuizQuery)).toHaveBeenCalledWith(
      42,
      expect.objectContaining({
        enabled: true
      })
    )
  })
})
