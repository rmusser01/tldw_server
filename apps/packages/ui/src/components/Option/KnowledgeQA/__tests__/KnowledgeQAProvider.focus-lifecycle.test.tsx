import "./knowledgeQaAuthorityFixture"
import React from "react"
import { act, render, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { KnowledgeQAProvider, useKnowledgeQA } from "../KnowledgeQAProvider"

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: () => [undefined],
}))

vi.mock("@/hooks/useAntdMessage", () => ({
  useAntdMessage: () => ({
    open: vi.fn(),
  }),
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: vi.fn().mockResolvedValue(undefined),
    fetchWithAuth: vi.fn().mockResolvedValue({
      ok: false,
      json: async () => [],
      text: async () => "",
    }),
    ragSearch: vi.fn(),
    ragSearchStream: vi.fn(),
  },
}))

let latestContext: ReturnType<typeof useKnowledgeQA> | null = null

function ContextProbe() {
  latestContext = useKnowledgeQA()
  return null
}

describe("KnowledgeQAProvider focused source lifecycle", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    latestContext = null
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it("auto-clears focused source after timeout following citation scroll", async () => {
    const sourceElement = document.createElement("div")
    sourceElement.id = "source-card-0"
    sourceElement.scrollIntoView = vi.fn()
    document.body.appendChild(sourceElement)

    render(
      <KnowledgeQAProvider>
        <ContextProbe />
      </KnowledgeQAProvider>
    )

    await waitFor(() => expect(latestContext).not.toBeNull())
    vi.useFakeTimers()

    act(() => {
      latestContext!.scrollToSource(0)
    })

    expect(sourceElement.scrollIntoView).toHaveBeenCalledWith({
      behavior: "smooth",
      block: "center",
    })
    expect(latestContext!.focusedSourceIndex).toBe(0)

    act(() => {
      vi.advanceTimersByTime(5000)
    })
    expect(latestContext!.focusedSourceIndex).toBeNull()

    document.body.removeChild(sourceElement)
  })

  it("keeps manual clear behavior while timeout is active", async () => {
    render(
      <KnowledgeQAProvider>
        <ContextProbe />
      </KnowledgeQAProvider>
    )

    await waitFor(() => expect(latestContext).not.toBeNull())
    vi.useFakeTimers()

    act(() => {
      latestContext!.focusSource(2)
    })
    expect(latestContext!.focusedSourceIndex).toBe(2)

    act(() => {
      latestContext!.focusSource(null)
      vi.advanceTimersByTime(5000)
    })
    expect(latestContext!.focusedSourceIndex).toBeNull()
  })

  it("scrolls to first matching answer citation marker when requested", async () => {
    const answerContent = document.createElement("div")
    answerContent.id = "knowledge-answer-content"
    const citationButton = document.createElement("button")
    citationButton.setAttribute("data-knowledge-citation-index", "2")
    citationButton.scrollIntoView = vi.fn()
    citationButton.focus = vi.fn()
    answerContent.appendChild(citationButton)
    document.body.appendChild(answerContent)

    render(
      <KnowledgeQAProvider>
        <ContextProbe />
      </KnowledgeQAProvider>
    )

    await waitFor(() => expect(latestContext).not.toBeNull())

    act(() => {
      latestContext!.scrollToCitation(2)
    })

    expect(citationButton.scrollIntoView).toHaveBeenCalledWith({
      behavior: "smooth",
      block: "center",
    })
    expect(citationButton.focus).toHaveBeenCalled()

    document.body.removeChild(answerContent)
  })

  it("targets a specific citation occurrence when provided", async () => {
    const answerContent = document.createElement("div")
    answerContent.id = "knowledge-answer-content"

    const firstCitation = document.createElement("button")
    firstCitation.setAttribute("data-knowledge-citation-index", "2")
    firstCitation.setAttribute("data-knowledge-citation-occurrence", "1")
    firstCitation.scrollIntoView = vi.fn()
    firstCitation.focus = vi.fn()
    answerContent.appendChild(firstCitation)

    const secondCitation = document.createElement("button")
    secondCitation.setAttribute("data-knowledge-citation-index", "2")
    secondCitation.setAttribute("data-knowledge-citation-occurrence", "2")
    secondCitation.scrollIntoView = vi.fn()
    secondCitation.focus = vi.fn()
    answerContent.appendChild(secondCitation)

    document.body.appendChild(answerContent)

    render(
      <KnowledgeQAProvider>
        <ContextProbe />
      </KnowledgeQAProvider>
    )

    await waitFor(() => expect(latestContext).not.toBeNull())

    act(() => {
      latestContext!.scrollToCitation(2, 2)
    })

    expect(secondCitation.scrollIntoView).toHaveBeenCalledWith({
      behavior: "smooth",
      block: "center",
    })
    expect(secondCitation.focus).toHaveBeenCalled()
    expect(firstCitation.scrollIntoView).not.toHaveBeenCalled()

    document.body.removeChild(answerContent)
  })
})
