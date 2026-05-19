import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { QuizPlayground } from "../QuizPlayground"
import { RESULTS_FILTER_PREFS_KEY, TAKE_QUIZ_LIST_PREFS_KEY } from "../stateKeys"
import { useAttemptsQuery, useQuizzesQuery } from "../hooks"

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
      if (typeof defaultValueOrOptions?.defaultValue === "string") {
        return defaultValueOrOptions.defaultValue
      }
      return key
    }
  })
}))

vi.mock("../hooks", () => ({
  useQuizzesQuery: vi.fn(() => ({ data: { items: [], count: 5 }, isLoading: false })),
  useAttemptsQuery: vi.fn(() => ({ data: { items: [], count: 12 }, isLoading: false }))
}))

vi.mock("antd", () => ({
  Button: ({ children, onClick, ...rest }: any) => (
    <button type="button" onClick={onClick} {...rest}>
      {children}
    </button>
  ),
  Tabs: ({ items, activeKey, onChange, destroyInactiveTabPane, className }: any) => {
    const activeItem = Array.isArray(items)
      ? items.find((item: any) => item.key === activeKey)
      : null

    return (
      <div data-testid="quiz-tabs" className={className}>
        <div data-testid="destroy-inactive-tab-pane">
          {String(destroyInactiveTabPane)}
        </div>
        <div>
          {Array.isArray(items)
            ? items.map((item: any) => (
                <button
                  key={item.key}
                  type="button"
                  onClick={() => onChange?.(item.key)}
                >
                  {item.label}
                </button>
              ))
            : null}
        </div>
        <div data-testid="active-tab">{activeKey}</div>
        <div>{activeItem?.children}</div>
      </div>
    )
  }
}))

vi.mock("../tabs/TakeQuizTab", () => ({
  TakeQuizTab: ({
    startQuizId,
    highlightQuizId,
    navigationSource,
    assignmentMode,
    assignmentDueAt,
    assignmentNote,
    assignedByRole,
    externalSearchQuery
  }: any) => (
    <div data-testid="take-intent">
      {JSON.stringify({
        startQuizId,
        highlightQuizId,
        navigationSource,
        assignmentMode,
        assignmentDueAt,
        assignmentNote,
        assignedByRole,
        externalSearchQuery
      })}
    </div>
  )
}))

vi.mock("../tabs/GenerateTab", () => ({
  GenerateTab: ({ onNavigateToTake, onNavigateToManage }: any) => (
    <div>
      <button
        type="button"
        onClick={() =>
          onNavigateToTake({
            highlightQuizId: 77,
            sourceTab: "generate"
          })
        }
      >
        Mock Generate Navigate
      </button>
      <button type="button" onClick={() => onNavigateToManage?.()}>
        Mock Generate Review
      </button>
    </div>
  )
}))

vi.mock("../tabs/CreateTab", () => ({
  CreateTab: ({ onNavigateToTake, onDirtyStateChange }: any) => (
    <div>
      <button
        type="button"
        onClick={() =>
          onNavigateToTake({
            highlightQuizId: 88,
            sourceTab: "create"
          })
        }
      >
        Mock Create Navigate
      </button>
      <button type="button" onClick={() => onDirtyStateChange?.(true)}>
        Mock Mark Create Dirty
      </button>
    </div>
  )
}))

vi.mock("../tabs/ManageTab", () => ({
  ManageTab: ({ onStartQuiz, externalSearchQuery }: any) => (
    <div>
      <button type="button" onClick={() => onStartQuiz(99)}>
        Mock Manage Start
      </button>
      <div data-testid="manage-search-intent">{externalSearchQuery ?? ""}</div>
    </div>
  )
}))

vi.mock("../tabs/ResultsTab", () => ({
  ResultsTab: ({ onRetakeQuiz }: any) => (
    <button
      type="button"
      onClick={() =>
        onRetakeQuiz?.({
          startQuizId: 7,
          highlightQuizId: 7,
          sourceTab: "results",
          attemptId: 301
        })
      }
    >
      Mock Results Retake
    </button>
  )
}))

describe("QuizPlayground navigation intents", () => {
  it("prompts before leaving Create tab when unsaved changes are present", async () => {
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(false)

    render(<QuizPlayground />)
    fireEvent.click(screen.getByRole("button", { name: "Create" }))
    fireEvent.click(await screen.findByRole("button", { name: "Mock Mark Create Dirty" }))
    fireEvent.click(screen.getByRole("button", { name: "Take Quiz" }))

    expect(confirmSpy).toHaveBeenCalledWith("You have unsaved quiz changes. Leave Create tab?")
    expect(screen.getByTestId("active-tab")).toHaveTextContent("create")

    confirmSpy.mockRestore()
  })

  it("allows leaving Create tab after confirmation", async () => {
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(true)

    render(<QuizPlayground />)
    fireEvent.click(screen.getByRole("button", { name: "Create" }))
    fireEvent.click(await screen.findByRole("button", { name: "Mock Mark Create Dirty" }))
    fireEvent.click(screen.getByRole("button", { name: "Take Quiz" }))

    expect(screen.getByTestId("active-tab")).toHaveTextContent("take")

    confirmSpy.mockRestore()
  })

  it("applies global search to Take tab by default and navigates there", () => {
    render(<QuizPlayground />)

    fireEvent.change(screen.getByTestId("quiz-global-search-input"), {
      target: { value: "bio" }
    })
    fireEvent.click(screen.getByTestId("quiz-global-search-apply"))

    expect(screen.getByTestId("active-tab")).toHaveTextContent("take")
    expect(screen.getByTestId("take-intent")).toHaveTextContent("\"externalSearchQuery\":\"bio\"")
  })

  it("applies global search to Manage tab when Manage is active", async () => {
    render(<QuizPlayground />)

    fireEvent.click(screen.getByRole("button", { name: "Manage" }))
    await screen.findByTestId("manage-search-intent")
    fireEvent.change(screen.getByTestId("quiz-global-search-input"), {
      target: { value: "chem" }
    })
    fireEvent.click(screen.getByTestId("quiz-global-search-apply"))

    expect(screen.getByTestId("active-tab")).toHaveTextContent("manage")
    expect(screen.getByTestId("manage-search-intent")).toHaveTextContent("chem")
  })

  it("resets take-tab intent and clears take-tab persisted state on explicit reset", async () => {
    window.sessionStorage.setItem(TAKE_QUIZ_LIST_PREFS_KEY, JSON.stringify({ page: 2 }))
    render(<QuizPlayground />)

    fireEvent.click(screen.getByRole("button", { name: "Generate" }))
    fireEvent.click(await screen.findByRole("button", { name: "Mock Generate Navigate" }))

    expect(screen.getByTestId("take-intent")).toHaveTextContent(
      JSON.stringify({
        startQuizId: null,
        highlightQuizId: 77,
        navigationSource: "generate",
        assignmentMode: null,
        assignmentDueAt: null,
        assignmentNote: null,
        assignedByRole: null,
        externalSearchQuery: null
      })
    )

    fireEvent.click(screen.getByTestId("quiz-reset-current-tab"))

    expect(window.sessionStorage.getItem(TAKE_QUIZ_LIST_PREFS_KEY)).toBeNull()
    expect(screen.getByTestId("take-intent")).toHaveTextContent(
      JSON.stringify({
        startQuizId: null,
        highlightQuizId: null,
        navigationSource: null,
        assignmentMode: null,
        assignmentDueAt: null,
        assignmentNote: null,
        assignedByRole: null,
        externalSearchQuery: null
      })
    )
  })

  it("clears results-tab persisted state on explicit reset", () => {
    window.sessionStorage.setItem(RESULTS_FILTER_PREFS_KEY, JSON.stringify({ page: 3 }))
    render(<QuizPlayground />)

    fireEvent.click(screen.getByRole("button", { name: "Results" }))
    fireEvent.click(screen.getByTestId("quiz-reset-current-tab"))

    expect(window.sessionStorage.getItem(RESULTS_FILTER_PREFS_KEY)).toBeNull()
  })

  it("keeps inactive tab panes mounted for per-tab state preservation", () => {
    render(<QuizPlayground />)
    expect(screen.getByTestId("destroy-inactive-tab-pane")).toHaveTextContent("false")
  })

  it("configures mobile-friendly tab overflow and short labels", () => {
    render(<QuizPlayground />)
    const tabsRoot = screen.getByTestId("quiz-tabs")
    expect(tabsRoot.className).toContain("quiz-tabs")
    expect(tabsRoot.className).toContain("ant-tabs-nav-wrap")
    expect(screen.getAllByText("Generate").length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText("Create").length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText("Results").length).toBeGreaterThanOrEqual(1)
  })

  it("routes generate navigation payload into Take tab intent", async () => {
    render(<QuizPlayground />)

    fireEvent.click(screen.getByRole("button", { name: "Generate" }))
    fireEvent.click(await screen.findByRole("button", { name: "Mock Generate Navigate" }))

    expect(screen.getByTestId("active-tab")).toHaveTextContent("take")
    expect(screen.getByTestId("take-intent")).toHaveTextContent(
      JSON.stringify({
        startQuizId: null,
        highlightQuizId: 77,
        navigationSource: "generate",
        assignmentMode: null,
        assignmentDueAt: null,
        assignmentNote: null,
        assignedByRole: null,
        externalSearchQuery: null
      })
    )
  })

  it("routes generate preview-review action into Manage tab", async () => {
    render(<QuizPlayground />)

    fireEvent.click(screen.getByRole("button", { name: "Generate" }))
    fireEvent.click(await screen.findByRole("button", { name: "Mock Generate Review" }))

    expect(screen.getByTestId("active-tab")).toHaveTextContent("manage")
  })

  it("routes results retake payload into Take tab intent", async () => {
    render(<QuizPlayground />)

    fireEvent.click(screen.getByRole("button", { name: "Results" }))
    fireEvent.click(await screen.findByRole("button", { name: "Mock Results Retake" }))

    expect(screen.getByTestId("active-tab")).toHaveTextContent("take")
    expect(screen.getByTestId("take-intent")).toHaveTextContent(
      JSON.stringify({
        startQuizId: 7,
        highlightQuizId: 7,
        navigationSource: "results",
        assignmentMode: null,
        assignmentDueAt: null,
        assignmentNote: null,
        assignedByRole: null,
        externalSearchQuery: null
      })
    )
  })

  it("hydrates shared-assignment URL params into Take tab intent", () => {
    const originalUrl = window.location.href
    window.history.replaceState(
      {},
      "",
      "/quiz?tab=take&start_quiz_id=7&highlight_quiz_id=7&assignment_mode=shared&assignment_due_at=2026-03-01T14%3A30%3A00.000Z&assignment_note=Complete%20before%20lab&assigned_by_role=lead"
    )

    render(<QuizPlayground />)

    expect(screen.getByTestId("take-intent")).toHaveTextContent(
      JSON.stringify({
        startQuizId: 7,
        highlightQuizId: 7,
        navigationSource: "assignment",
        assignmentMode: "shared",
        assignmentDueAt: "2026-03-01T14:30:00.000Z",
        assignmentNote: "Complete before lab",
        assignedByRole: "lead",
        externalSearchQuery: null
      })
    )

    window.history.replaceState({}, "", originalUrl)
  })

  it("renders count badges on take, manage, and results tab labels", () => {
    vi.mocked(useQuizzesQuery).mockReturnValue({
      data: { items: [], count: 9 },
      isLoading: false
    } as any)
    vi.mocked(useAttemptsQuery).mockReturnValue({
      data: { items: [], count: 4 },
      isLoading: false
    } as any)

    render(<QuizPlayground />)

    expect(screen.getByRole("button", { name: "Take Quiz" })).toHaveTextContent("9")
    expect(screen.getByRole("button", { name: "Manage" })).toHaveTextContent("9")
    expect(screen.getByRole("button", { name: "Results" })).toHaveTextContent("4")
  })
})
