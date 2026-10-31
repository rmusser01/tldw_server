import React from "react"
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { ManageTab } from "../ManageTab"
import {
  useCreateQuizMutation,
  useCreateQuestionMutation,
  useDeleteQuestionMutation,
  useDeleteQuizMutation,
  useQuestionsQuery,
  useQuizzesQuery,
  useUpdateQuestionMutation,
  useUpdateQuizMutation
} from "../../hooks"

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

vi.mock("../../hooks", () => ({
  useQuizzesQuery: vi.fn(),
  useQuestionsQuery: vi.fn(),
  useCreateQuizMutation: vi.fn(),
  useDeleteQuizMutation: vi.fn(),
  useUpdateQuizMutation: vi.fn(),
  useCreateQuestionMutation: vi.fn(),
  useUpdateQuestionMutation: vi.fn(),
  useDeleteQuestionMutation: vi.fn(),
  useAllOsceStationsQuery: vi.fn(() => ({ data: [], isLoading: false })),
  useDeleteOsceStationMutation: vi.fn(() => ({ mutateAsync: vi.fn(), isPending: false })),
  useOsceStationQuery: vi.fn(() => ({ data: undefined, isLoading: false }))
}))

if (!(globalThis as any).ResizeObserver) {
  ;(globalThis as any).ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
}

describe("ManageTab undo accessibility", () => {
  const deleteQuizMutateAsync = vi.fn(async () => undefined)

  beforeEach(() => {
    vi.clearAllMocks()

    vi.mocked(useQuizzesQuery).mockReturnValue({
      data: {
        items: [
          {
            id: 7,
            name: "Biology Basics",
            description: "Cell structures and functions",
            total_questions: 3,
            passing_score: 75,
            media_id: null,
            time_limit_seconds: 600,
            deleted: false,
            client_id: "test",
            version: 1
          }
        ],
        count: 1
      },
      isLoading: false,
      refetch: vi.fn()
    } as any)

    vi.mocked(useQuestionsQuery).mockReturnValue({
      data: { items: [], count: 0 },
      isLoading: false,
      refetch: vi.fn()
    } as any)

    vi.mocked(useDeleteQuizMutation).mockReturnValue({
      mutateAsync: deleteQuizMutateAsync
    } as any)
    vi.mocked(useCreateQuizMutation).mockReturnValue({
      mutateAsync: vi.fn(async () => ({ id: 999 }))
    } as any)

    vi.mocked(useUpdateQuizMutation).mockReturnValue({ mutateAsync: vi.fn() } as any)
    vi.mocked(useCreateQuestionMutation).mockReturnValue({ mutateAsync: vi.fn() } as any)
    vi.mocked(useUpdateQuestionMutation).mockReturnValue({ mutateAsync: vi.fn() } as any)
    vi.mocked(useDeleteQuestionMutation).mockReturnValue({ mutateAsync: vi.fn() } as any)
  })

  it("shows a focusable inline undo banner after delete", async () => {
    render(
      <ManageTab
        onNavigateToCreate={() => {}}
        onNavigateToGenerate={() => {}}
        onStartQuiz={() => {}}
      />
    )

    expect(screen.getByText("Biology Basics")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: /delete/i }))
    await waitFor(() => {
      expect(screen.queryByText("Biology Basics")).not.toBeInTheDocument()
    })

    const undoButton = screen.getByRole("button", { name: /undo/i })
    undoButton.focus()
    expect(undoButton).toHaveFocus()

    fireEvent.click(undoButton)
    await waitFor(() => {
      expect(screen.getByText("Biology Basics")).toBeInTheDocument()
    })
  })

  it("returns focus to the edit trigger after the edit modal closes", async () => {
    render(
      <ManageTab
        onNavigateToCreate={() => {}}
        onNavigateToGenerate={() => {}}
        onStartQuiz={() => {}}
      />
    )

    const editTrigger = screen.getByTestId("quiz-edit-7")
    editTrigger.focus()
    expect(editTrigger).toHaveFocus()

    fireEvent.click(editTrigger)
    fireEvent.click(screen.getByRole("button", { name: /cancel/i }))

    await waitFor(() => {
      expect(editTrigger).toHaveFocus()
    })
  }, 30000)

  it("links inline validation errors and restores focus when question modal closes", async () => {
    render(
      <ManageTab
        onNavigateToCreate={() => {}}
        onNavigateToGenerate={() => {}}
        onStartQuiz={() => {}}
      />
    )

    fireEvent.click(screen.getByTestId("quiz-edit-7"))

    const addQuestionButton = await screen.findByTestId("manage-add-question")
    addQuestionButton.focus()
    expect(addQuestionButton).toHaveFocus()

    fireEvent.click(addQuestionButton)

    const questionInput = await screen.findByPlaceholderText("Enter your question...")
    const questionModal = questionInput.closest(".ant-modal")
    const questionModalQueries = questionModal ? within(questionModal as HTMLElement) : screen

    fireEvent.click(questionModalQueries.getByRole("button", { name: /^save$/i }))

    await waitFor(() => {
      expect(screen.getByText("Question text is required.")).toBeInTheDocument()
    })
    expect(screen.getByText("Please provide at least two options.")).toBeInTheDocument()

    expect(questionInput).toHaveAttribute("aria-describedby", "manage-question-text-error")

    const optionOneInput = questionModalQueries.getByPlaceholderText("Option 1")
    expect(optionOneInput).toHaveAttribute("aria-describedby", "manage-question-options-error")

    fireEvent.click(questionModalQueries.getByRole("button", { name: /cancel/i }))

    await waitFor(() => {
      expect(addQuestionButton).toHaveFocus()
    })
  })

  it("keeps undo available after temporary tab-style visibility switches", async () => {
    const VisibilityHarness = () => {
      const [manageVisible, setManageVisible] = React.useState(true)
      return (
        <div>
          <button type="button" onClick={() => setManageVisible(false)}>
            Show Other Tab
          </button>
          <button type="button" onClick={() => setManageVisible(true)}>
            Show Manage Tab
          </button>
          <div style={{ display: manageVisible ? "block" : "none" }}>
            <ManageTab
              onNavigateToCreate={() => {}}
              onNavigateToGenerate={() => {}}
              onStartQuiz={() => {}}
            />
          </div>
        </div>
      )
    }

    render(<VisibilityHarness />)

    fireEvent.click(screen.getAllByRole("button", { name: /delete/i })[0])
    await waitFor(() => {
      expect(screen.queryByText("Biology Basics")).not.toBeInTheDocument()
    })
    expect(screen.getByRole("button", { name: /undo/i })).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: /show other tab/i }))
    fireEvent.click(screen.getByRole("button", { name: /show manage tab/i }))

    const undoButton = screen.getByRole("button", { name: /undo/i })
    expect(undoButton).toBeInTheDocument()
    fireEvent.click(undoButton)

    await waitFor(() => {
      expect(screen.getByText("Biology Basics")).toBeInTheDocument()
    })
  }, 20000)

  it("commits deletion only after grace period and cancels commit on undo", async () => {
    vi.useFakeTimers()
    try {
      render(
        <ManageTab
          onNavigateToCreate={() => {}}
          onNavigateToGenerate={() => {}}
          onStartQuiz={() => {}}
        />
      )

      fireEvent.click(screen.getByRole("button", { name: /delete/i }))
      expect(screen.queryByText("Biology Basics")).not.toBeInTheDocument()

      await act(async () => {
        vi.advanceTimersByTime(7999)
        await Promise.resolve()
      })
      expect(deleteQuizMutateAsync).not.toHaveBeenCalled()

      fireEvent.click(screen.getByRole("button", { name: /undo/i }))
      await act(async () => {
        vi.advanceTimersByTime(1)
        await Promise.resolve()
      })
      expect(deleteQuizMutateAsync).not.toHaveBeenCalled()
      expect(screen.getByText("Biology Basics")).toBeInTheDocument()

      fireEvent.click(screen.getByRole("button", { name: /delete/i }))
      await act(async () => {
        vi.advanceTimersByTime(8000)
        await Promise.resolve()
      })
      expect(deleteQuizMutateAsync).toHaveBeenCalledTimes(1)
    } finally {
      vi.useRealTimers()
    }
  }, 20000)

  it("renders loading skeleton placeholders while quiz list is loading", () => {
    vi.mocked(useQuizzesQuery).mockReturnValue({
      data: undefined,
      isLoading: true,
      refetch: vi.fn()
    } as any)

    render(
      <ManageTab
        onNavigateToCreate={() => {}}
        onNavigateToGenerate={() => {}}
        onStartQuiz={() => {}}
      />
    )

    expect(screen.getByTestId("manage-loading-skeleton")).toBeInTheDocument()
  })
})
