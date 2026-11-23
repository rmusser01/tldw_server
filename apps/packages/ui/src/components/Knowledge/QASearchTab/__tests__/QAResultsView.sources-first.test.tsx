// CC-10 (#3101), owner decision [2]: in Search & Context the source chunks
// come first and the generated answer is opt-in.
import React from "react"
import { fireEvent, render, screen, within } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { QAResultsView } from "../QAResultsView"
import type { QASearchResponse } from "../../hooks/useQASearch"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string, vars?: Record<string, unknown>) => {
      if (fallback && typeof vars?.count === "number") {
        return fallback.replace("{{count}}", String(vars.count))
      }
      return fallback ?? _key
    }
  })
}))

vi.mock("react-markdown", () => ({
  default: ({ children }: { children: React.ReactNode }) => <p>{children}</p>
}))
vi.mock("remark-gfm", () => ({ default: () => null }))

const response = (
  overrides: Partial<QASearchResponse> = {}
): QASearchResponse => ({
  generatedAnswer: null,
  documents: [
    {
      content: "Embedding models map text to vectors.",
      score: 0.91,
      metadata: { title: "Embeddings primer" }
    }
  ],
  citations: [],
  academicCitations: [],
  timings: {},
  totalTime: 0,
  cacheHit: false,
  feedbackId: null,
  errors: [],
  query: "embedding model evaluation",
  expandedQueries: [],
  ...overrides
})

const handlers = () => ({
  onCopyAnswer: vi.fn(),
  onInsertAnswer: vi.fn(),
  onCopyChunk: vi.fn(),
  onInsertChunk: vi.fn(),
  onPinChunk: vi.fn(),
  onPreviewChunk: vi.fn()
})

describe("QAResultsView: sources first, answer on request", () => {
  it("lists the source chunks before the generated answer", () => {
    render(
      <QAResultsView
        response={response({ generatedAnswer: "Use a held-out set." })}
        loading={false}
        hasAttemptedSearch
        timedOut={false}
        pinnedResults={[]}
        {...handlers()}
      />
    )

    const sources = screen.getByRole("list", {
      name: "Retrieved source chunks"
    })
    const answer = screen.getByText("Use a held-out set.")
    expect(
      sources.compareDocumentPosition(answer) &
        Node.DOCUMENT_POSITION_FOLLOWING
    ).toBeTruthy()
  })

  // CC-10 review: Generate answer sits in the results header, so it is one
  // tab stop away instead of after every chunk's actions.
  it("offers Generate answer in the results header, before the chunks' actions", () => {
    const onGenerateAnswer = vi.fn()
    render(
      <QAResultsView
        response={response()}
        loading={false}
        hasAttemptedSearch
        timedOut={false}
        pinnedResults={[]}
        onGenerateAnswer={onGenerateAnswer}
        {...handlers()}
      />
    )

    const header = screen.getByTestId("qa-results-header")
    expect(header).toHaveTextContent("Source Chunks (1)")
    const generate = within(header).getByRole("button", {
      name: "Generate answer"
    })
    const firstChunkAction = within(
      screen.getByRole("list", { name: "Retrieved source chunks" })
    ).getAllByRole("button")[0]
    expect(
      generate.compareDocumentPosition(firstChunkAction) &
        Node.DOCUMENT_POSITION_FOLLOWING
    ).toBeTruthy()
    fireEvent.click(generate)
    expect(onGenerateAnswer).toHaveBeenCalledTimes(1)
  })

  it("keeps the sources and the focused button while the answer is generated", () => {
    const onGenerateAnswer = vi.fn()
    render(
      <QAResultsView
        response={response()}
        loading={false}
        answerLoading
        hasAttemptedSearch
        timedOut={false}
        pinnedResults={[]}
        onGenerateAnswer={onGenerateAnswer}
        {...handlers()}
      />
    )

    expect(
      screen.getByRole("list", { name: "Retrieved source chunks" })
    ).toBeInTheDocument()
    const generating = screen.getByRole("button", {
      name: "Generating answer…"
    })
    expect(generating).toHaveAttribute("aria-disabled", "true")
    expect(generating).not.toBeDisabled()
    fireEvent.click(generating)
    expect(onGenerateAnswer).not.toHaveBeenCalled()
  })

  it("moves focus to the answer and announces it when it arrives", () => {
    const props = {
      loading: false,
      hasAttemptedSearch: true,
      timedOut: false,
      pinnedResults: [],
      onGenerateAnswer: vi.fn(),
      ...handlers()
    }
    const { rerender } = render(
      <QAResultsView {...props} response={response()} />
    )
    screen.getByRole("button", { name: "Generate answer" }).focus()
    fireEvent.click(screen.getByRole("button", { name: "Generate answer" }))
    rerender(<QAResultsView {...props} answerLoading response={response()} />)
    expect(screen.getByRole("button", { name: "Generating answer…" })).toHaveFocus()

    rerender(
      <QAResultsView
        {...props}
        response={response({ generatedAnswer: "Use a held-out set." })}
      />
    )

    const heading = screen.getByRole("heading", { name: "Generated Answer" })
    expect(heading).toHaveAttribute("tabindex", "-1")
    expect(heading).toHaveFocus()
    expect(screen.getByTestId("qa-answer-status")).toHaveTextContent(
      "Answer ready."
    )
    expect(screen.getByTestId("qa-answer-status")).toHaveAttribute(
      "aria-live",
      "polite"
    )
  })

  it("says why there is no answer when generation fails", () => {
    render(
      <QAResultsView
        response={response()}
        loading={false}
        answerError="The model did not respond."
        hasAttemptedSearch
        timedOut={false}
        pinnedResults={[]}
        onGenerateAnswer={vi.fn()}
        {...handlers()}
      />
    )

    expect(screen.getByTestId("qa-answer-status")).toHaveTextContent(
      "The model did not respond."
    )
    expect(screen.getByText("The model did not respond.", { selector: "p:not([data-testid])" })).toBeVisible()
    expect(
      screen.getByRole("button", { name: "Generate answer" })
    ).not.toHaveAttribute("aria-disabled", "true")
  })
})
