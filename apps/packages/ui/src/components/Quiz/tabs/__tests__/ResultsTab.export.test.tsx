import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { ResultsTab } from "../ResultsTab"
import {
  useAllAttemptsQuery,
  useAttemptQuery,
  useAttemptRemediationConversionsQuery,
  useConvertAttemptRemediationQuestionsMutation,
  useGenerateRemediationQuizMutation,
  useQuizAttemptQuestionAssistantQuery,
  useQuizAttemptQuestionAssistantRespondMutation,
  useQuizzesQuery
} from "../../hooks"
import {
  useCreateDeckMutation,
  useCreateFlashcardMutation,
  useDecksQuery
} from "@/components/Flashcards/hooks/useFlashcardQueries"

const navigationMocks = vi.hoisted(() => ({
  navigate: vi.fn()
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

vi.mock("react-router-dom", () => ({
  useNavigate: () => navigationMocks.navigate
}))

vi.mock("../../hooks", () => ({
  useAllAttemptsQuery: vi.fn(),
  useQuizzesQuery: vi.fn(),
  useAttemptQuery: vi.fn(),
  useAttemptRemediationConversionsQuery: vi.fn(),
  useConvertAttemptRemediationQuestionsMutation: vi.fn(),
  useGenerateRemediationQuizMutation: vi.fn(),
  useQuizAttemptQuestionAssistantQuery: vi.fn(),
  useQuizAttemptQuestionAssistantRespondMutation: vi.fn()
}))

vi.mock("@/components/Flashcards/hooks/useFlashcardQueries", () => ({
  useDecksQuery: vi.fn(),
  useCreateDeckMutation: vi.fn(),
  useCreateFlashcardMutation: vi.fn()
}))

if (!(globalThis as any).ResizeObserver) {
  ;(globalThis as any).ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
}

describe("ResultsTab CSV export", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    window.sessionStorage.clear()
    navigationMocks.navigate.mockReset()
    window.sessionStorage.setItem("quiz-results-filters-v1", JSON.stringify({
      page: 1,
      pageSize: 10,
      quizFilterId: null,
      passFilter: "pass",
      dateRangeFilter: "all"
    }))

    vi.mocked(useAllAttemptsQuery).mockReturnValue({
      data: {
        items: [
          {
            id: 601,
            quiz_id: 7,
            started_at: "2026-02-18T10:00:00Z",
            completed_at: "2026-02-18T10:03:00Z",
            score: 4,
            total_possible: 5,
            time_spent_seconds: 180,
            answers: []
          },
          {
            id: 602,
            quiz_id: 7,
            started_at: "2026-02-18T11:00:00Z",
            completed_at: "2026-02-18T11:03:00Z",
            score: 2,
            total_possible: 5,
            time_spent_seconds: 180,
            answers: []
          }
        ],
        count: 2
      },
      isLoading: false
    } as any)

    vi.mocked(useQuizzesQuery).mockReturnValue({
      data: {
        items: [
          { id: 7, name: "Biology Basics", total_questions: 5, passing_score: 70 }
        ],
        count: 1
      },
      isLoading: false
    } as any)

    vi.mocked(useAttemptQuery).mockReturnValue({
      data: null,
      isLoading: false,
      isFetching: false
    } as any)
    vi.mocked(useAttemptRemediationConversionsQuery).mockReturnValue({
      data: {
        attempt_id: 0,
        items: [],
        count: 0,
        superseded_count: 0
      },
      isLoading: false,
      isFetching: false
    } as ReturnType<typeof useAttemptRemediationConversionsQuery>)
    vi.mocked(useConvertAttemptRemediationQuestionsMutation).mockReturnValue({
      mutateAsync: vi.fn(),
      isPending: false
    } as ReturnType<typeof useConvertAttemptRemediationQuestionsMutation>)
    vi.mocked(useGenerateRemediationQuizMutation).mockReturnValue({
      mutateAsync: vi.fn(),
      isPending: false
    } as any)
    vi.mocked(useQuizAttemptQuestionAssistantQuery).mockReturnValue({
      data: null,
      isLoading: false,
      isError: false
    } as any)
    vi.mocked(useQuizAttemptQuestionAssistantRespondMutation).mockReturnValue({
      mutateAsync: vi.fn(),
      isPending: false
    } as any)

    vi.mocked(useDecksQuery).mockReturnValue({
      data: [],
      isLoading: false
    } as any)
    vi.mocked(useCreateDeckMutation).mockReturnValue({
      mutateAsync: vi.fn(),
      isPending: false
    } as any)
    vi.mocked(useCreateFlashcardMutation).mockReturnValue({
      mutateAsync: vi.fn(),
      isPending: false
    } as any)
  })

  it("exports a CSV that respects active filters and includes filter metadata columns", async () => {
    const createObjectURLSpy = vi
      .spyOn(window.URL, "createObjectURL")
      .mockReturnValue("blob:quiz-export")
    const revokeObjectURLSpy = vi
      .spyOn(window.URL, "revokeObjectURL")
      .mockImplementation(() => {})
    const clickSpy = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => {})

    render(<ResultsTab />)
    fireEvent.click(screen.getByRole("button", { name: "Export CSV" }))

    expect(createObjectURLSpy).toHaveBeenCalledTimes(1)
    const blob = createObjectURLSpy.mock.calls[0]?.[0] as Blob
    const csv = await blob.text()

    expect(csv).toContain("attempt_id,quiz_id,quiz_name")
    expect(csv).toContain("\"601\"")
    expect(csv).not.toContain("\"602\"")
    expect(csv).toContain("\"pass\"")
    expect(csv).toContain("filter_date_start_iso")

    expect(clickSpy).toHaveBeenCalledTimes(1)
    expect(revokeObjectURLSpy).toHaveBeenCalledTimes(1)

    clickSpy.mockRestore()
    revokeObjectURLSpy.mockRestore()
    createObjectURLSpy.mockRestore()
  })
})
