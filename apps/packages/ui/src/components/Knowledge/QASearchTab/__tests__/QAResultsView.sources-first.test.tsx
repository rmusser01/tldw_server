// CC-10 (#3101), owner decision [2]: in Search & Context the source chunks
// come first and the generated answer is opt-in.
import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
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

  it("offers Generate answer after the sources instead of generating one", () => {
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

    const sources = screen.getByRole("list", {
      name: "Retrieved source chunks"
    })
    const generate = screen.getByRole("button", { name: "Generate answer" })
    expect(
      sources.compareDocumentPosition(generate) &
        Node.DOCUMENT_POSITION_FOLLOWING
    ).toBeTruthy()
    fireEvent.click(generate)
    expect(onGenerateAnswer).toHaveBeenCalledTimes(1)
  })

  it("keeps the sources on screen while the answer is generated", () => {
    render(
      <QAResultsView
        response={response()}
        loading={false}
        answerLoading
        hasAttemptedSearch
        timedOut={false}
        pinnedResults={[]}
        onGenerateAnswer={vi.fn()}
        {...handlers()}
      />
    )

    expect(
      screen.getByRole("list", { name: "Retrieved source chunks" })
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Generating answer…" })
    ).toBeDisabled()
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

    expect(screen.getByRole("alert")).toHaveTextContent(
      "The model did not respond."
    )
    expect(
      screen.getByRole("button", { name: "Generate answer" })
    ).toBeEnabled()
  })
})
