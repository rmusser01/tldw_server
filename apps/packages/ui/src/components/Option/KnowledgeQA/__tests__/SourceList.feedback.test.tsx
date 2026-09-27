import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { SourceList } from "../SourceList"

const submitExplicitFeedbackMock = vi.fn()
const messageOpenMock = vi.fn()
const trackMetricMock = vi.fn()
const isAuthorityCurrent = () => true
const qaClient = { submitSourceFeedback: (...args: unknown[]) => submitExplicitFeedbackMock(...args) }

const state = {
  results: [
    {
      id: "doc-1",
      content: "Relevant chunk content",
      metadata: {
        title: "Source A",
        chunk_id: "chunk-1",
        source_type: "media_db",
      },
      score: 0.91,
    },
  ] as Array<Record<string, any>>,
  citations: [{ index: 1 }] as Array<{ index: number }>,
  focusedSourceIndex: null as number | null,
  focusSource: vi.fn(),
  setQuery: vi.fn(),
  query: "How is revenue trending?",
  currentThreadId: "thread-feedback",
  messages: [{ id: "assistant-2", role: "assistant" }] as Array<{
    id: string
    role: string
  }>,
}

vi.mock("../KnowledgeQAProvider", () => ({
  useKnowledgeQA: () => ({
    storageScopeKey: "test-owner",
    isAuthorityCurrent,
    client: qaClient,
    results: state.results,
    citations: state.citations,
    focusedSourceIndex: state.focusedSourceIndex,
    focusSource: state.focusSource,
    setQuery: state.setQuery,
    query: state.query,
    currentThreadId: state.currentThreadId,
    messages: state.messages,
  }),
}))

vi.mock("@/services/feedback", () => ({
  getFeedbackSessionId: () => "session-qa",
  submitExplicitFeedback: (...args: unknown[]) => submitExplicitFeedbackMock(...args),
}))

vi.mock("@/hooks/useAntdMessage", () => ({
  useAntdMessage: () => ({
    open: messageOpenMock,
  }),
}))

vi.mock("@/utils/knowledge-qa-search-metrics", () => ({
  trackKnowledgeQaSearchMetric: (...args: unknown[]) => trackMetricMock(...args),
}))

describe("SourceList source feedback", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    submitExplicitFeedbackMock.mockResolvedValue({ ok: true })
    trackMetricMock.mockResolvedValue(undefined)
    state.currentThreadId = "thread-feedback"
    state.messages = [{ id: "assistant-2", role: "assistant" }]
  })

  it("submits per-source relevance feedback with document and chunk ids", async () => {
    render(<SourceList />)

    fireEvent.click(screen.getByRole("button", { name: "Yes" }))

    await waitFor(() => expect(submitExplicitFeedbackMock).toHaveBeenCalledTimes(1))
    expect(submitExplicitFeedbackMock).toHaveBeenCalledWith(
      expect.objectContaining({
        conversation_id: "thread-feedback",
        message_id: "assistant-2",
        query: "How is revenue trending?",
        feedback_type: "relevance",
        relevance_score: 5,
        document_ids: ["doc-1"],
        chunk_ids: ["chunk-1"],
      })
    )
    expect(trackMetricMock).toHaveBeenCalledWith({
      type: "source_feedback_submit",
      relevant: true,
    })
  })

  it("shows retry action when source feedback submission fails", async () => {
    submitExplicitFeedbackMock.mockRejectedValueOnce(new Error("offline"))
    submitExplicitFeedbackMock.mockResolvedValueOnce({ ok: true })

    render(<SourceList />)

    fireEvent.click(screen.getByRole("button", { name: "No" }))

    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Retry feedback" })).toBeInTheDocument()
    )

    fireEvent.click(screen.getByRole("button", { name: "Retry feedback" }))

    await waitFor(() => expect(submitExplicitFeedbackMock).toHaveBeenCalledTimes(2))
  })

  it("resets source feedback state when the active answer session changes with the same results", async () => {
    const { rerender } = render(<SourceList />)

    fireEvent.click(screen.getByRole("button", { name: "Yes" }))

    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Yes" })).toHaveAttribute(
        "aria-pressed",
        "true"
      )
    )

    state.currentThreadId = "thread-feedback-2"
    state.messages = [{ id: "assistant-3", role: "assistant" }]
    rerender(<SourceList />)

    expect(screen.getByRole("button", { name: "Yes" })).toHaveAttribute(
      "aria-pressed",
      "false"
    )
  })

  it("ignores stale source feedback completion after the user switches to a different answer session", async () => {
    let resolveFeedback: ((value: { ok: boolean }) => void) | null = null
    submitExplicitFeedbackMock.mockImplementation(
      () =>
        new Promise<{ ok: boolean }>((resolve) => {
          resolveFeedback = resolve
        })
    )

    const { rerender } = render(<SourceList />)

    fireEvent.click(screen.getByRole("button", { name: "Yes" }))

    await waitFor(() =>
      expect(submitExplicitFeedbackMock).toHaveBeenCalledWith(
        expect.objectContaining({
          conversation_id: "thread-feedback",
          message_id: "assistant-2",
          relevance_score: 5,
        })
      )
    )

    state.currentThreadId = "thread-feedback-2"
    state.messages = [{ id: "assistant-3", role: "assistant" }]
    rerender(<SourceList />)

    await act(async () => {
      resolveFeedback?.({ ok: true })
      await Promise.resolve()
    })

    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Yes" })).toHaveAttribute(
        "aria-pressed",
        "false"
      )
    )
    expect(trackMetricMock).not.toHaveBeenCalledWith({
      type: "source_feedback_submit",
      relevant: true,
    })
  })
})
