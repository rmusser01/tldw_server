import React from "react"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { useDocumentWorkspaceStore } from "@/store/document-workspace"
import type { QuickIngestOperation } from "@/services/tldw/quick-ingest-authority"
import { DocumentChat } from "../DocumentChat"

const io = vi.hoisted(() => ({ submit: vi.fn(), stop: vi.fn(), retry: vi.fn(), status: "ready" }))
vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string, fallback?: string) => fallback ?? key })
}))
vi.mock("@/hooks/useMessageOption", () => ({
  useMessageOption: () => ({ selectedModel: "test-model", onSubmit: io.submit, stopStreamingRequest: io.stop })
}))
vi.mock("@/hooks/document-workspace/useDocumentMetadata", () => ({
  useDocumentMetadata: () => ({ data: null })
}))
vi.mock("../SuggestedQuestions", () => ({ SuggestedQuestions: () => null }))
vi.mock("@/hooks/document-workspace/useDocumentChat", () => ({
  useDocumentChat: () => {
    const [conversations, setConversations] = React.useState([
      { key: "first", title: "First topic" }, { key: "second", title: "Second topic" }
    ])
    const [activeConversationKey, selectConversation] = React.useState("first")
    const [inputValue, setInputValue] = React.useState("")
    return {
      conversations, activeConversationKey, selectConversation, inputValue, setInputValue,
      newConversation: () => {
        setConversations([...conversations, { key: "new", title: "New topic" }])
        selectConversation("new")
      },
      retryConversation: io.retry,
      conversationStatus: io.status,
      isChatReady: io.status === "ready",
      isServerAvailable: true,
      messages: [],
      trackTurn: (send: () => Promise<unknown>) => send()
    }
  }
}))

const operation: QuickIngestOperation = {
  authorityKey: "test",
  authorityRevision: 1,
  requestScope: { config: { serverUrl: "http://chat.test", authMode: "multi-user" }, userId: "alice" },
  signal: new AbortController().signal,
  isCurrent: () => true,
  assertCurrent: () => {}
}

describe("Document Chat conversation sidebar", () => {
  beforeEach(() => {
    vi.resetAllMocks()
    io.status = "ready"
    useDocumentWorkspaceStore.setState({ activeDocumentId: 19 })
  })

  it("selects tabs from the keyboard, stops the previous turn and opens a new chat", () => {
    io.submit.mockImplementation(() => new Promise(() => {}))
    render(<DocumentChat operation={operation} />)
    const list = screen.getByRole("tablist", { name: "Conversations" })
    const first = within(list).getByRole("tab", { name: "First topic" })
    const second = within(list).getByRole("tab", { name: "Second topic" })
    expect(first).toHaveAttribute("aria-selected", "true")
    first.focus()
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "A question" } })
    fireEvent.keyDown(screen.getByRole("textbox"), { key: "Enter" })
    const { controller } = io.submit.mock.calls[0]![0] as { controller: AbortController }

    fireEvent.keyDown(first, { key: "ArrowRight" })
    expect(second).toHaveFocus()
    expect(second).toHaveAttribute("aria-selected", "true")
    expect(controller.signal.aborted).toBe(true)
    expect(screen.getByRole("tabpanel", { name: "Second topic" })).toContainElement(screen.getByRole("textbox"))

    fireEvent.click(screen.getByRole("button", { name: "New chat" }))
    expect(within(list).getByRole("tab", { name: "New topic" })).toHaveAttribute("aria-selected", "true")
  })

  it("keeps switching available while an unavailable conversation can be retried", () => {
    io.status = "error"
    render(<DocumentChat operation={operation} />)
    expect(screen.getByRole("tab", { name: "Second topic" })).toBeEnabled()
    expect(screen.getByRole("textbox")).toBeDisabled()
    fireEvent.click(screen.getByRole("button", { name: "Retry" }))
    expect(io.retry).toHaveBeenCalledTimes(1)
  })

  it("restores the typed draft when shared chat refuses the send before admission", async () => {
    io.submit.mockResolvedValue({ status: "failed", errorMessage: "Chat details are loading", restoreDraft: true })
    render(<DocumentChat operation={operation} />)
    const input = screen.getByRole("textbox")
    fireEvent.change(input, { target: { value: "Retry this draft" } })
    fireEvent.keyDown(input, { key: "Enter" })
    await waitFor(() => expect(input).toHaveValue("Retry this draft"))
  })

  it("uses a highlighted passage queued while Chat was hidden without sending it", () => {
    useDocumentWorkspaceStore.setState({
      pendingSelectionPrompt: { documentId: 19, prompt: 'Explain this passage: "Selected passage"' }
    })
    render(<DocumentChat operation={operation} />)
    const input = screen.getByRole("textbox")
    expect(input).toHaveValue('Explain this passage: "Selected passage"')
    expect(input).toHaveFocus()
    expect(useDocumentWorkspaceStore.getState().pendingSelectionPrompt).toBeNull()
    expect(io.submit).not.toHaveBeenCalled()
  })
})
