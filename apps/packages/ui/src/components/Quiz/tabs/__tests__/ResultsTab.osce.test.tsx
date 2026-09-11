import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { MemoryRouter } from "react-router-dom"

import { ResultsTab } from "../ResultsTab"
import {
  useAllAttemptsQuery,
  useAttemptQuery,
  useAttemptRemediationConversionsQuery,
  useConvertAttemptRemediationQuestionsMutation,
  useGenerateRemediationQuizMutation,
  useQuizzesQuery
} from "../../hooks"
import { useDecksQuery } from "@/components/Flashcards/hooks/useFlashcardQueries"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, value?: string | { defaultValue?: string }) =>
      typeof value === "string" ? value : value?.defaultValue ?? _key
  })
}))

vi.mock("../../hooks", () => ({
  useAllAttemptsQuery: vi.fn(),
  useAttemptQuery: vi.fn(),
  useAttemptRemediationConversionsQuery: vi.fn(),
  useConvertAttemptRemediationQuestionsMutation: vi.fn(),
  useGenerateRemediationQuizMutation: vi.fn(),
  useQuizzesQuery: vi.fn()
}))

vi.mock("@/components/Flashcards/hooks/useFlashcardQueries", () => ({ useDecksQuery: vi.fn() }))
vi.mock("@/components/StudySuggestions/StudySuggestionsPanel", () => ({ StudySuggestionsPanel: () => null }))
vi.mock("../components/QuizRemediationPanel", () => ({ QuizRemediationPanel: () => null }))
vi.mock("../../osce/OsceResultsPanel", () => ({
  OsceResultsPanel: () => <section aria-label="OSCE completed practice">OSCE result body</section>
}))

if (!(globalThis as any).ResizeObserver) {
  ;(globalThis as any).ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
}

describe("ResultsTab OSCE segment", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    window.sessionStorage.clear()
    vi.mocked(useAllAttemptsQuery).mockReturnValue({
      data: {
        items: [{
          id: 101,
          quiz_id: 7,
          started_at: "2026-09-11T10:00:00Z",
          completed_at: "2026-09-11T10:02:00Z",
          score: 1,
          total_possible: 1,
          time_spent_seconds: 120,
          answers: []
        }],
        count: 1
      },
      isLoading: false
    } as any)
    vi.mocked(useQuizzesQuery).mockReturnValue({
      data: { items: [{ id: 7, name: "Biology", passing_score: 70 }], count: 1 },
      isLoading: false
    } as any)
    vi.mocked(useAttemptQuery).mockReturnValue({ data: null, isLoading: false, isFetching: false } as any)
    vi.mocked(useAttemptRemediationConversionsQuery).mockReturnValue({
      data: { attempt_id: 0, items: [], count: 0, superseded_count: 0 },
      isLoading: false
    } as any)
    vi.mocked(useConvertAttemptRemediationQuestionsMutation).mockReturnValue({ mutateAsync: vi.fn(), isPending: false } as any)
    vi.mocked(useGenerateRemediationQuizMutation).mockReturnValue({ mutateAsync: vi.fn(), isPending: false } as any)
    vi.mocked(useDecksQuery).mockReturnValue({ data: [], isLoading: false } as any)
  })

  it("keeps ordinary question results unchanged and isolates OSCE results", () => {
    render(<MemoryRouter><ResultsTab /></MemoryRouter>)

    expect(screen.getByText("Quiz attempts")).toBeInTheDocument()
    expect(screen.getByText("OSCE practice")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Export CSV" })).toBeInTheDocument()
    expect(screen.getByText("1/1 (100%)")).toBeInTheDocument()

    fireEvent.click(screen.getByText("OSCE practice"))

    expect(screen.getByRole("region", { name: "OSCE completed practice" })).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Export CSV" })).not.toBeInTheDocument()
    expect(screen.queryByText("1/1 (100%)")).not.toBeInTheDocument()
  })

  it("keeps complete labels available at narrow and desktop container widths", () => {
    const view = render(
      <div style={{ width: 320 }}><MemoryRouter><ResultsTab /></MemoryRouter></div>
    )
    expect(screen.getByText("Quiz attempts")).toBeVisible()
    expect(screen.getByText("OSCE practice")).toBeVisible()

    view.rerender(
      <div style={{ width: 1024 }}><MemoryRouter><ResultsTab /></MemoryRouter></div>
    )
    expect(screen.getByText("Quiz attempts")).toBeVisible()
    expect(screen.getByText("OSCE practice")).toBeVisible()
  })
})
