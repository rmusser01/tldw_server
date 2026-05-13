import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { AnswerPanel } from "../AnswerPanel"

const navigateMock = vi.fn()
const queueWorkspacePlaygroundPrefillMock = vi.fn()
const messageOpenMock = vi.fn()
const trackMetricMock = vi.fn()

const state = {
  answer: "Final answer with citation [1].",
  citations: [{ index: 1 }],
  isSearching: false,
  error: null as string | null,
  results: [
    {
      id: "123",
      metadata: {
        title: "Report A",
        source_type: "pdf",
      },
    },
  ],
  searchDetails: null,
  query: "Compare reports",
  currentThreadId: "thread-xyz",
  messages: [] as Array<{ id: string; role: string }>,
  scrollToSource: vi.fn(),
}

vi.mock("@/services/feedback", () => ({
  getFeedbackSessionId: () => "session-1",
  submitExplicitFeedback: vi.fn(),
}))

vi.mock("@/hooks/useAntdMessage", () => ({
  useAntdMessage: () => ({
    open: messageOpenMock,
  }),
}))

vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual("react-router-dom")
  return {
    ...actual,
    useNavigate: () => navigateMock,
  }
})

vi.mock("@/utils/workspace-playground-prefill", () => ({
  buildKnowledgeQaWorkspacePrefill: vi.fn((payload) => payload),
  queueWorkspacePlaygroundPrefill: (...args: unknown[]) =>
    queueWorkspacePlaygroundPrefillMock(...args),
}))

vi.mock("@/utils/knowledge-qa-search-metrics", () => ({
  trackKnowledgeQaSearchMetric: (...args: unknown[]) => trackMetricMock(...args),
}))

vi.mock("../KnowledgeQAProvider", () => ({
  useKnowledgeQA: () => state,
}))

describe("AnswerPanel workspace handoff", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    queueWorkspacePlaygroundPrefillMock.mockResolvedValue(undefined)
    trackMetricMock.mockResolvedValue(undefined)
  })

  it("queues workspace prefill payload and navigates to workspace route", async () => {
    render(<AnswerPanel />)

    fireEvent.click(screen.getByRole("button", { name: "Continue in editor" }))

    await waitFor(() =>
      expect(queueWorkspacePlaygroundPrefillMock).toHaveBeenCalledWith(
        expect.objectContaining({
          threadId: "thread-xyz",
          query: "Compare reports",
          answer: "Final answer with citation [1].",
        })
      )
    )
    expect(trackMetricMock).toHaveBeenCalledWith({
      type: "workspace_handoff",
      source_count: 1,
    })
    expect(navigateMock).toHaveBeenCalledWith("/research-studio")
  })
})
