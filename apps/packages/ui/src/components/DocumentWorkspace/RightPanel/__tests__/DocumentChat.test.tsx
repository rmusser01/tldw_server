import React from "react"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { useDocumentWorkspaceStore } from "@/store/document-workspace"
import { tldwClient } from "@/services/tldw"
import type { QuickIngestOperation } from "@/services/tldw/quick-ingest-authority"
import { DocumentChat } from "../DocumentChat"

const io = vi.hoisted(() => ({
  submit: vi.fn(), stop: vi.fn(), retry: vi.fn(), status: "ready",
  connected: true, cachedMessage: "", citation: null as null | Record<string, unknown>
}))
vi.mock("@/components/Common/Playground/Message", () => ({
  PlaygroundMessage: ({ message }: { message: string }) => <p>{message}</p>
}))
vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string, fallback?: string | { defaultValue?: string; page?: number }) =>
    typeof fallback === "string" ? fallback : fallback?.defaultValue?.replace("{{page}}", String(fallback.page)) ?? key })
}))
vi.mock("@/hooks/useMessageOption", () => ({
  useMessageOption: () => ({ selectedModel: "test-model", onSubmit: io.submit, stopStreamingRequest: io.stop })
}))
vi.mock("@/hooks/document-workspace/useDocumentMetadata", () => ({
  useDocumentMetadata: () => ({ data: io.citation ? { uuid: io.citation.media_uuid } : null })
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
      isServerAvailable: io.connected,
      canShowTranscript: Boolean(io.cachedMessage),
      messages: io.cachedMessage ? [{
        id: "cached", role: "assistant", isBot: true, message: io.cachedMessage,
        sources: io.citation ? [{ documentCitation: io.citation }] : []
      }] : [],
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
  afterEach(() => vi.restoreAllMocks())

  beforeEach(() => {
    vi.resetAllMocks()
    io.status = "ready"
    io.connected = true
    io.cachedMessage = ""
    io.citation = null
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
    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled()

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

  it("shows a verified cached transcript and keeps its draft during a connection loss", async () => {
    render(<DocumentChat operation={operation} />)
    const input = screen.getByRole("textbox")
    fireEvent.change(input, { target: { value: "Draft through disconnect" } })
    io.status = "error"
    io.connected = false
    io.cachedMessage = "Previously verified answer"
    fireEvent.click(screen.getByRole("tab", { name: "Second topic" }))
    expect(await screen.findByText(io.cachedMessage)).toBeVisible()
    expect(screen.getByRole("textbox")).toHaveValue("Draft through disconnect")
    expect(screen.getByRole("textbox")).toBeDisabled()
  })

  it("opens a citation only after rechecking the live document UUID", async () => {
    const uuid = "b679f16d-b053-4f9d-9f39-a6153d2d3cab"
    useDocumentWorkspaceStore.setState({ activeDocumentType: "pdf", currentPage: 1, totalPages: 5 })
    io.cachedMessage = "Read this passage"
    io.citation = { media_uuid: uuid, title: "PDF fixture", excerpt: "Example passage", page: 3 }
    const getMediaDetails = vi.spyOn(tldwClient, "getMediaDetails")
      .mockResolvedValueOnce({ uuid: "replacement" }).mockResolvedValueOnce({ uuid })
    render(<DocumentChat operation={operation} />)
    const open = screen.getByRole("button", { name: "Open page 3" })
    fireEvent.click(open)
    await expect(screen.findByText("Could not open this source.")).resolves.toBeVisible()
    expect(useDocumentWorkspaceStore.getState().currentPage).toBe(1)
    fireEvent.click(open)
    await waitFor(() => expect(useDocumentWorkspaceStore.getState().currentPage).toBe(3))
    expect(getMediaDetails).toHaveBeenLastCalledWith(19, expect.objectContaining({ requestScope: operation.requestScope }))
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
