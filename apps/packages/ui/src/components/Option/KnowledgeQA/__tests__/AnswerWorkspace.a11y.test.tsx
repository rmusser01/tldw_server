import { render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { AnswerWorkspace } from "../panels/AnswerWorkspace"
import type { KnowledgeSourceStatus } from "../types"

const state = {
  answer: null as string | null,
  searchDetails: null as { sourceStatus: Record<string, KnowledgeSourceStatus> } | null,
  results: [] as Array<{ id: string; score?: number }>,
  error: null as string | null,
  queryWarning: null as string | null,
  messages: [] as Array<{
    id: string
    role: "user" | "assistant" | "system"
    content?: string
  }>,
  currentThreadId: null as string | null,
  citations: [] as Array<{ index: number; documentId: string }>,
  settings: { strip_min_relevance: 0.3 } as Record<string, unknown>,
}

vi.mock("../KnowledgeQAProvider", () => ({
  useKnowledgeQA: () => ({
    answer: state.answer,
    searchDetails: state.searchDetails,
    results: state.results,
    error: state.error,
    queryWarning: state.queryWarning,
    messages: state.messages,
    currentThreadId: state.currentThreadId,
    citations: state.citations,
    settings: state.settings,
    setSettingsPanelOpen: vi.fn(),
    updateSetting: vi.fn(),
  }),
}))

vi.mock("../ConversationThread", () => ({
  ConversationThread: () => <div data-testid="knowledge-conversation-thread" />,
}))

vi.mock("../AnswerPanel", () => ({
  AnswerPanel: () => <div data-testid="knowledge-answer-panel" />,
}))

vi.mock("../FollowUpInput", () => ({
  FollowUpInput: () => <div data-testid="knowledge-followup-input" />,
}))

describe("AnswerWorkspace accessibility announcements", () => {
  beforeEach(() => {
    state.answer = null
    state.searchDetails = null
    state.results = []
    state.error = null
    state.queryWarning = null
    state.messages = []
    state.currentThreadId = null
    state.citations = []
    state.settings = { strip_min_relevance: 0.3 }
  })

  it.each([true, false])("discloses failed sources without an answer when generation is %s", (generationEnabled) => {
    state.results = [{ id: "owned-media" }]
    state.settings = { enable_generation: generationEnabled, sources: ["media_db", "chats"] }
    state.searchDetails = { sourceStatus: {
      media_db: { status: "searched", count: 1 },
      chats: { status: "error", count: 0, reason: "retrieval_failed" },
    } }

    const { rerender } = render(<AnswerWorkspace queryStage="complete" />)
    expect(screen.getByRole("status")).toHaveTextContent("Could not search Chats.")
    expect(screen.getByRole("status")).not.toHaveTextContent("Could not search Documents & Media")

    state.searchDetails = { sourceStatus: {
      media_db: { status: "searched", count: 1 },
      chats: { status: "searched", count: 2 },
    } }
    rerender(<AnswerWorkspace queryStage="complete" />)
    expect(screen.queryByRole("status")).not.toBeInTheDocument()
  })

  it("uses completed source diagnostics after the next-search scope changes", () => {
    state.results = [{ id: "owned-media" }, { id: "retained-character" }]
    state.settings = { sources: ["media_db"] }
    state.searchDetails = { sourceStatus: {
      media_db: { status: "searched", count: 1 },
      chats: { status: "unavailable", count: 0, reason: "retrieval_failed" },
      characters: { status: "error", count: 1, reason: "retrieval_failed" },
    } }

    render(<AnswerWorkspace queryStage="complete" />)
    expect(screen.getByRole("status")).toHaveTextContent("Could not search Chats.")
    expect(screen.getByRole("status")).toHaveTextContent("Some searches failed for Characters.")
  })

  it("leaves generated-answer and zero-result diagnostics to their existing panels", () => {
    state.searchDetails = { sourceStatus: { chats: { status: "error", count: 0 } } }
    const { rerender } = render(<AnswerWorkspace queryStage="complete" />)
    expect(screen.queryByRole("status")).not.toBeInTheDocument()
    state.results = [{ id: "owned-media" }]
    state.answer = "An answer [1]."
    rerender(<AnswerWorkspace queryStage="complete" />)
    expect(screen.queryByRole("status")).not.toBeInTheDocument()
    state.answer = null
    rerender(<AnswerWorkspace queryStage="searching" />)
    expect(screen.queryByRole("status")).not.toBeInTheDocument()
  })

  it("retains evidence from a single partially failed source without claiming other sources", () => {
    state.results = [{ id: "retained-chat" }]
    state.searchDetails = { sourceStatus: { chats: { status: "error", count: 1 } } }
    render(<AnswerWorkspace queryStage="complete" />)
    expect(screen.getByRole("status")).toHaveTextContent("Some searches failed for Chats.")
    expect(screen.getByRole("status")).toHaveTextContent("Retrieved sources are still available.")
    expect(screen.getByRole("status")).not.toHaveTextContent("other sources")
  })

  it("announces active and completed search stages through live regions", () => {
    const { rerender } = render(<AnswerWorkspace queryStage="searching" />)

    expect(screen.getByText("Searching your selected sources.")).toBeInTheDocument()

    state.results = [{ id: "r1" }, { id: "r2" }]
    rerender(<AnswerWorkspace queryStage="complete" />)

    expect(screen.getByText("Search complete. 2 sources found.")).toBeInTheDocument()
  })

  it("announces search errors through assertive live region", () => {
    state.error = "Search timed out"

    render(<AnswerWorkspace queryStage="error" />)

    expect(screen.getByText("Search error. Search timed out")).toBeInTheDocument()
  })

  it("announces cancellation neutrally and replaces the stale active message", () => {
    const { container, rerender } = render(<AnswerWorkspace queryStage="searching" />)
    rerender(<AnswerWorkspace queryStage="cancelled" />)
    expect(screen.getByRole("status")).toHaveTextContent("Search cancelled")
    expect(container.querySelector('[aria-live="polite"]')).toHaveTextContent("Search cancelled.")
    expect(container.querySelector('[aria-live="assertive"]')).toBeEmptyDOMElement()
    expect(screen.queryByText("Searching your selected sources.")).not.toBeInTheDocument()
    expect(screen.queryByText("Searching selected sources")).not.toBeInTheDocument()
    rerender(<AnswerWorkspace queryStage="searching" />)
    expect(screen.queryByRole("status")).not.toBeInTheDocument()
    expect(container.querySelector('[aria-live="polite"]')).toHaveTextContent("Searching your selected sources.")
  })

  it("clears the assertive timeout when a retry starts and completes", () => {
    state.error = "Search timed out"
    const { rerender } = render(<AnswerWorkspace queryStage="error" />)
    expect(screen.getByText("Search error. Search timed out")).toBeInTheDocument()
    state.error = null
    rerender(<AnswerWorkspace queryStage="searching" />)
    expect(screen.queryByText("Search error. Search timed out")).not.toBeInTheDocument()
    state.results = [{ id: "cedar" }]
    rerender(<AnswerWorkspace queryStage="complete" />)
    expect(screen.queryByText("Search error. Search timed out")).not.toBeInTheDocument()
    expect(screen.getByText("Search complete. 1 source found.")).toBeInTheDocument()
  })

  it("announces the security outcome instead of claiming no sources were found", () => {
    state.queryWarning = "Security settings excluded all retrieved sources."
    render(<AnswerWorkspace queryStage="complete" />)
    expect(screen.getByText(state.queryWarning)).toBeInTheDocument()
    expect(screen.queryByText("Search complete. 0 sources found.")).not.toBeInTheDocument()
  })

  it("clears the active announcement on failure and does not reuse failed turns", () => {
    const { rerender } = render(<AnswerWorkspace queryStage="ranking" />)
    state.error = "Provider unavailable"
    state.messages = [
      { id: "failed", role: "user", content: "Failed question" },
      { id: "retry", role: "user", content: "Retry question" },
    ]
    rerender(<AnswerWorkspace queryStage="error" />)
    expect(screen.queryByText("Ranking retrieved sources.")).not.toBeInTheDocument()
    expect(screen.queryByText("Using context from turn 1.")).not.toBeInTheDocument()
    expect(screen.queryByText("No answer recorded.")).not.toBeInTheDocument()
  })

  it("shows persistent thread context summary", () => {
    state.currentThreadId = "thread-1"
    state.messages = [
      { id: "m1", role: "user", content: "What changed in this release?" },
      {
        id: "m2",
        role: "assistant",
        content: "The release improves indexing speed and citation quality.",
      },
      { id: "m3", role: "user", content: "What should I test first?" },
    ]

    render(<AnswerWorkspace queryStage="idle" />)

    expect(screen.getByText("Conversation • 2 turns")).toBeInTheDocument()
    expect(screen.getByText("Using context from turn 1.")).toBeInTheDocument()
    expect(screen.getByText("Previous turn")).toBeInTheDocument()
    expect(
      screen.getByText("What changed in this release?")
    ).toBeInTheDocument()
    expect(
      screen.getByText("The release improves indexing speed and citation quality.")
    ).toBeInTheDocument()
    expect(screen.queryByText("Context previews (1)")).not.toBeInTheDocument()
  })
})
