import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { ExportDialog } from "../ExportDialog"

const {
  messageOpenMock,
  createNoteMock,
  exportChatbookMock,
  downloadChatbookExportMock,
  createShareLinkMock,
  revokeShareLinkMock,
} = vi.hoisted(() => ({
  messageOpenMock: vi.fn(),
  createNoteMock: vi.fn(),
  exportChatbookMock: vi.fn(),
  downloadChatbookExportMock: vi.fn(),
  createShareLinkMock: vi.fn(),
  revokeShareLinkMock: vi.fn(),
}))
const state = {
  messages: [] as Array<{ role: string; content: string }>,
  currentThreadId: "thread-1" as string | null,
  results: [] as Array<{ id: string }>,
  citations: [] as Array<{ index: number }>,
  answer: "Test answer" as string | null,
  query: "What does this source say?",
  settings: {
    sources: ["media_db", "notes"],
    include_media_ids: [42],
    include_note_ids: ["note-a"],
    top_k: 12,
    generation_provider: "openai",
    generation_model: "gpt-4o-mini",
    enable_web_fallback: false,
  },
  preset: "balanced",
  searchDetails: null as null | {
    expandedQueries?: string[]
    rerankingEnabled?: boolean
    rerankingStrategy?: string
    averageRelevance?: number | null
    webFallbackTriggered?: boolean
    webFallbackEngine?: string | null
  },
}

vi.mock("../KnowledgeQAProvider", () => ({
  useKnowledgeQA: () => ({
    messages: state.messages,
    currentThreadId: state.currentThreadId,
    results: state.results,
    citations: state.citations,
    answer: state.answer,
    query: state.query,
    settings: state.settings,
    preset: state.preset,
    searchDetails: state.searchDetails,
  })
}))

vi.mock("@/hooks/useAntdMessage", () => ({
  useAntdMessage: () => ({
    open: messageOpenMock,
  }),
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    createNote: createNoteMock,
    exportChatbook: exportChatbookMock,
    downloadChatbookExport: downloadChatbookExportMock,
    createConversationShareLink: createShareLinkMock,
    revokeConversationShareLink: revokeShareLinkMock,
  },
}))

describe("ExportDialog accessibility", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    createNoteMock.mockResolvedValue({ id: 1 })
    exportChatbookMock.mockResolvedValue({
      success: true,
      job_id: "job-1",
      download_url: "/api/v1/chatbooks/download/job-1",
    })
    downloadChatbookExportMock.mockResolvedValue({
      blob: new Blob(["chatbook-content"], { type: "application/zip" }),
      filename: "knowledge.chatbook.zip",
    })
    createShareLinkMock.mockResolvedValue({
      share_id: "share-1",
      token: "token-1",
      share_path: "/knowledge/shared/token-1",
      created_at: "2026-02-19T10:00:00.000Z",
      expires_at: "2026-02-20T10:00:00.000Z",
      permission: "view",
    })
    revokeShareLinkMock.mockResolvedValue({ success: true, share_id: "share-1" })
    state.messages = []
    state.currentThreadId = "thread-1"
    state.results = []
    state.citations = []
    state.answer = "Test answer"
    state.query = "What does this source say?"
    state.settings = {
      sources: ["media_db", "notes"],
      include_media_ids: [42],
      include_note_ids: ["note-a"],
      top_k: 12,
      generation_provider: "openai",
      generation_model: "gpt-4o-mini",
      enable_web_fallback: false,
    }
    state.preset = "balanced"
    state.searchDetails = null
  })

  it("exposes modal dialog semantics", () => {
    render(<ExportDialog open onClose={vi.fn()} />)

    const dialog = screen.getByRole("dialog", { name: "Export Conversation" })
    expect(dialog).toHaveAttribute("aria-modal", "true")
    expect(dialog).toHaveAttribute("aria-labelledby", "export-dialog-title")
    expect(screen.getByText("Export Conversation")).toHaveAttribute(
      "id",
      "export-dialog-title"
    )
  })

  it("stacks export format cards on small screens", () => {
    render(<ExportDialog open onClose={vi.fn()} />)

    const markdownButton = screen.getByRole("button", { name: /Markdown/i })
    const formatGrid = markdownButton.closest("div.grid")
    expect(formatGrid).not.toBeNull()
    expect(formatGrid!.className).toContain("grid-cols-1")
    expect(formatGrid!.className).toContain("sm:grid-cols-3")
  })

  it("traps keyboard focus and closes on Escape", async () => {
    const onClose = vi.fn()
    render(<ExportDialog open onClose={onClose} />)

    const closeButton = screen.getByRole("button", {
      name: "Close export dialog"
    })
    const exportButton = screen.getByRole("button", { name: "Export" })

    await waitFor(() => expect(closeButton).toHaveFocus())

    exportButton.focus()
    fireEvent.keyDown(document, { key: "Tab" })
    expect(closeButton).toHaveFocus()

    closeButton.focus()
    fireEvent.keyDown(document, { key: "Tab", shiftKey: true })
    expect(exportButton).toHaveFocus()

    fireEvent.keyDown(document, { key: "Escape" })
    expect(onClose).toHaveBeenCalledTimes(1)
  })

  it("shows actionable error feedback when chatbook export fails", async () => {
    exportChatbookMock.mockRejectedValueOnce(new Error("thread not found"))

    render(<ExportDialog open onClose={vi.fn()} />)

    fireEvent.click(screen.getByRole("button", { name: /Chatbook/i }))
    fireEvent.click(screen.getByRole("button", { name: "Export" }))

    await waitFor(() =>
      expect(messageOpenMock).toHaveBeenCalledWith(
        expect.objectContaining({
          type: "error",
        })
      )
    )

    expect(screen.getByText(/Chatbook export failed/i)).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Retry export" })).toBeInTheDocument()
  })

  it.each([
    {
      error: "HTTP 401 unauthorized",
      expected:
        "Chatbook export failed. You are not authorized to export this thread.",
    },
    {
      error: "HTTP 422 validation failed: content_selections is required",
      expected:
        "Chatbook export failed. Export request is invalid. Check the selected thread and try again.",
    },
    {
      error: "network unreachable",
      expected: "Chatbook export failed. Cannot reach server.",
    },
  ])("maps chatbook export failure copy for '$error'", async ({ error, expected }) => {
    exportChatbookMock.mockRejectedValueOnce(new Error(error))

    render(<ExportDialog open onClose={vi.fn()} />)

    fireEvent.click(screen.getByRole("button", { name: /Chatbook/i }))
    fireEvent.click(screen.getByRole("button", { name: "Export" }))

    await waitFor(() =>
      expect(messageOpenMock).toHaveBeenCalledWith(
        expect.objectContaining({
          type: "error",
          content: expected,
        })
      )
    )
    expect(screen.getByText(expected)).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Retry export" })).toBeInTheDocument()
  })

  it("uses chatbook export contract and downloads by returned job id", async () => {
    const onClose = vi.fn()
    const originalCreateObjectURL = URL.createObjectURL
    const originalRevokeObjectURL = URL.revokeObjectURL
    const createObjectURLMock = vi.fn(() => "blob:test-download")
    const revokeObjectURLMock = vi.fn(() => undefined)

    Object.defineProperty(URL, "createObjectURL", {
      configurable: true,
      writable: true,
      value: createObjectURLMock,
    })
    Object.defineProperty(URL, "revokeObjectURL", {
      configurable: true,
      writable: true,
      value: revokeObjectURLMock,
    })

    try {
      render(<ExportDialog open onClose={onClose} />)

      fireEvent.click(screen.getByRole("button", { name: /Chatbook/i }))
      fireEvent.click(screen.getByRole("button", { name: "Export" }))

      await waitFor(() =>
        expect(exportChatbookMock).toHaveBeenCalledWith(
          expect.objectContaining({
            content_selections: { conversation: ["thread-1"] },
            async_mode: false,
          })
        )
      )
      await waitFor(() => expect(downloadChatbookExportMock).toHaveBeenCalledWith("job-1"))
      await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1))
    } finally {
      Object.defineProperty(URL, "createObjectURL", {
        configurable: true,
        writable: true,
        value: originalCreateObjectURL,
      })
      Object.defineProperty(URL, "revokeObjectURL", {
        configurable: true,
        writable: true,
        value: originalRevokeObjectURL,
      })
    }
  })

  it("ignores stale chatbook export completions after the dialog closes", async () => {
    let resolveExport: ((value: Record<string, unknown>) => void) | null = null
    exportChatbookMock.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveExport = resolve
        })
    )
    const onClose = vi.fn()
    const { rerender } = render(<ExportDialog open onClose={onClose} />)

    fireEvent.click(screen.getByRole("button", { name: /Chatbook/i }))
    fireEvent.click(screen.getByRole("button", { name: "Export" }))

    rerender(<ExportDialog open={false} onClose={onClose} />)

    resolveExport?.({
      success: true,
      job_id: "job-stale",
      download_url: "/api/v1/chatbooks/download/job-stale",
    })

    await act(async () => {
      await Promise.resolve()
    })

    expect(downloadChatbookExportMock).not.toHaveBeenCalled()
    expect(onClose).toHaveBeenCalledTimes(0)
    expect(messageOpenMock).not.toHaveBeenCalledWith(
      expect.objectContaining({
        type: "error",
      })
    )
  })

  it("ignores stale chatbook export completions after reopening the same thread", async () => {
    let resolveExport: ((value: Record<string, unknown>) => void) | null = null
    exportChatbookMock.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveExport = resolve
        })
    )
    const onClose = vi.fn()
    const { rerender } = render(<ExportDialog open onClose={onClose} />)

    fireEvent.click(screen.getByRole("button", { name: /Chatbook/i }))
    fireEvent.click(screen.getByRole("button", { name: "Export" }))

    rerender(<ExportDialog open={false} onClose={onClose} />)
    rerender(<ExportDialog open onClose={onClose} />)

    resolveExport?.({
      success: true,
      job_id: "job-reopened",
      download_url: "/api/v1/chatbooks/download/job-reopened",
    })

    await act(async () => {
      await Promise.resolve()
    })

    expect(downloadChatbookExportMock).not.toHaveBeenCalled()
    expect(onClose).not.toHaveBeenCalled()
  })

  it("uses browser print fallback for PDF exports", async () => {
    vi.useFakeTimers()
    const printSpy = vi.spyOn(window, "print").mockImplementation(() => {})
    try {
      render(<ExportDialog open onClose={vi.fn()} />)

      fireEvent.click(screen.getByRole("button", { name: /PDF/i }))
      fireEvent.click(screen.getByRole("button", { name: "Export" }))

      await Promise.resolve()
      expect(printSpy).not.toHaveBeenCalled()

      vi.advanceTimersByTime(500)
      expect(printSpy).toHaveBeenCalledTimes(1)
    } finally {
      printSpy.mockRestore()
      vi.useRealTimers()
    }
  })

  it("cancels pending PDF print when the dialog closes before the timeout fires", async () => {
    vi.useFakeTimers()
    const printSpy = vi.spyOn(window, "print").mockImplementation(() => {})
    try {
      const { rerender } = render(<ExportDialog open onClose={vi.fn()} />)

      fireEvent.click(screen.getByRole("button", { name: /PDF/i }))
      fireEvent.click(screen.getByRole("button", { name: "Export" }))

      await Promise.resolve()
      rerender(<ExportDialog open={false} onClose={vi.fn()} />)

      vi.advanceTimersByTime(500)
      expect(printSpy).not.toHaveBeenCalled()
    } finally {
      printSpy.mockRestore()
      vi.useRealTimers()
    }
  })

  it("shows citation transparency guidance and active share-link control", async () => {
    const writeTextMock = vi.fn().mockResolvedValue(undefined)
    Object.defineProperty(globalThis.navigator, "clipboard", {
      value: { writeText: writeTextMock },
      configurable: true,
    })

    render(<ExportDialog open onClose={vi.fn()} />)

    expect(
      screen.getByText(/Citation formatting is approximate/i)
    ).toBeInTheDocument()

    const shareButton = screen.getByRole("button", { name: "Create share link" })
    expect(shareButton).toBeEnabled()
    fireEvent.click(shareButton)

    await waitFor(() =>
      expect(writeTextMock).toHaveBeenCalledWith(
        expect.stringContaining("/knowledge/shared/")
      )
    )
    expect(
      screen.getByText(/dedicated token with read-only access/i)
    ).toBeInTheDocument()
  })

  it("keeps the active share link visible even when clipboard copy fails", async () => {
    const writeTextMock = vi.fn().mockRejectedValue(new Error("clipboard denied"))
    Object.defineProperty(globalThis.navigator, "clipboard", {
      value: { writeText: writeTextMock },
      configurable: true,
    })

    render(<ExportDialog open onClose={vi.fn()} />)

    fireEvent.click(screen.getByRole("button", { name: "Create share link" }))

    await waitFor(() => expect(createShareLinkMock).toHaveBeenCalledTimes(1))
    await waitFor(() =>
      expect(screen.getByText(/Active link expires/i)).toBeInTheDocument()
    )
    expect(screen.getByRole("button", { name: "Revoke link" })).toBeEnabled()
    expect(messageOpenMock).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "error",
        content: "Unable to copy share link, but the link remains active.",
      })
    )
  })

  it("saves the active conversation to Notes from workflow actions", async () => {
    state.results = [
      {
        id: "source-1",
        content: "Important excerpt content",
        metadata: {
          title: "Source A",
          url: "https://example.com/source-a",
        },
      } as any,
    ]

    render(<ExportDialog open onClose={vi.fn()} />)

    fireEvent.click(screen.getByRole("button", { name: "Save to Notes" }))

    await waitFor(() => expect(createNoteMock).toHaveBeenCalledTimes(1))

    const [noteContent, noteMetadata] = createNoteMock.mock.calls[0]
    expect(noteContent).toContain("# Knowledge QA Export")
    expect(noteContent).toContain("## Bibliography")
    expect(noteMetadata).toEqual(
      expect.objectContaining({
        title: expect.stringContaining("Knowledge QA:"),
        metadata: expect.objectContaining({
          origin: "knowledge_qa",
          source: "knowledge_export",
          thread_id: "thread-1",
        }),
      })
    )
    expect(messageOpenMock).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "success",
        content: "Saved to Notes.",
      })
    )
  })

  it("exports citation mappings and optional settings snapshot for grounded review", async () => {
    state.answer = "The planning document recommends staged rollout [1]."
    state.citations = [{ index: 1 }]
    state.results = [
      {
        id: "source-1",
        content: "Staged rollout recommendation and supporting evidence",
        metadata: {
          title: "Planning Memo",
          source: "planning-memo.pdf",
          url: "https://example.com/planning-memo",
          page_number: 4,
        },
        score: 0.87,
      } as any,
    ]
    state.messages = [
      { role: "user", content: "What does the planning memo recommend?" },
      { role: "assistant", content: "It recommends staged rollout [1]." },
    ]
    state.searchDetails = {
      expandedQueries: ["rollout plan", "deployment stages"],
      rerankingEnabled: true,
      rerankingStrategy: "hybrid",
      averageRelevance: 0.87,
      webFallbackTriggered: false,
      webFallbackEngine: null,
    }

    render(<ExportDialog open onClose={vi.fn()} />)

    fireEvent.click(screen.getByLabelText("Settings snapshot"))
    fireEvent.click(screen.getByRole("button", { name: "Export" }))

    await waitFor(() => expect(screen.getByText("Preview")).toBeInTheDocument())

    const preview = screen.getByText((_, element) => {
      if (!element || element.tagName.toLowerCase() !== "pre") return false
      const text = element.textContent || ""
      return (
        text.includes("## Citations") &&
        text.includes("[1] Planning Memo") &&
        text.includes("maps to Source 1") &&
        text.includes('"preset": "balanced"') &&
        text.includes('"sources": [') &&
        text.includes('"include_media_ids": [') &&
        text.includes('"expandedQueries": [')
      )
    })

    expect(preview).toBeInTheDocument()
  })

  it("shows a user-visible error when Save to Notes fails", async () => {
    createNoteMock.mockRejectedValueOnce(new Error("notes backend unavailable"))

    render(<ExportDialog open onClose={vi.fn()} />)

    fireEvent.click(screen.getByRole("button", { name: "Save to Notes" }))

    await waitFor(() =>
      expect(messageOpenMock).toHaveBeenCalledWith(
        expect.objectContaining({
          type: "error",
          content: expect.stringContaining("Failed to save to Notes."),
        })
      )
    )
  })

  it("ignores stale Save to Notes completions after the dialog closes", async () => {
    let resolveSave: ((value: { id: number }) => void) | null = null
    createNoteMock.mockImplementation(
      () =>
        new Promise<{ id: number }>((resolve) => {
          resolveSave = resolve
        })
    )

    const { rerender } = render(<ExportDialog open onClose={vi.fn()} />)

    fireEvent.click(screen.getByRole("button", { name: "Save to Notes" }))
    expect(screen.getByRole("button", { name: "Saving..." })).toBeDisabled()

    rerender(<ExportDialog open={false} onClose={vi.fn()} />)

    resolveSave?.({ id: 42 })
    await act(async () => {
      await Promise.resolve()
    })

    rerender(<ExportDialog open onClose={vi.fn()} />)

    expect(screen.getByRole("button", { name: "Save to Notes" })).toBeEnabled()
    expect(messageOpenMock).not.toHaveBeenCalledWith(
      expect.objectContaining({
        type: "success",
        content: "Saved to Notes.",
      })
    )
  })

  it("disables share-link action for local-only threads", () => {
    state.currentThreadId = "local-thread-123"

    render(<ExportDialog open onClose={vi.fn()} />)

    expect(screen.getByRole("button", { name: "Create share link" })).toBeDisabled()
  })

  it("clears stale share-link state when the active thread changes", async () => {
    const writeTextMock = vi.fn().mockResolvedValue(undefined)
    Object.defineProperty(globalThis.navigator, "clipboard", {
      value: { writeText: writeTextMock },
      configurable: true,
    })

    const { rerender } = render(<ExportDialog open onClose={vi.fn()} />)

    fireEvent.click(screen.getByRole("button", { name: "Create share link" }))

    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Revoke link" })).toBeEnabled()
    )
    expect(screen.getByText(/Active link expires/i)).toBeInTheDocument()

    state.currentThreadId = "thread-2"
    rerender(<ExportDialog open onClose={vi.fn()} />)

    expect(screen.getByRole("button", { name: "Create share link" })).toBeEnabled()
    expect(screen.getByRole("button", { name: "Revoke link" })).toBeDisabled()
    expect(screen.queryByText(/Active link expires/i)).not.toBeInTheDocument()
  })

  it("clears export preview state when the active thread changes", async () => {
    const { rerender } = render(<ExportDialog open onClose={vi.fn()} />)

    fireEvent.click(screen.getByRole("button", { name: "Export" }))
    await waitFor(() => expect(screen.getByText("Preview")).toBeInTheDocument())

    state.currentThreadId = "thread-2"
    rerender(<ExportDialog open onClose={vi.fn()} />)

    expect(screen.queryByText("Preview")).not.toBeInTheDocument()
    expect(screen.queryByRole("button", { name: /^Copy$/ })).not.toBeInTheDocument()
  })

  it("ignores stale share-link completions after the active thread changes", async () => {
    let resolveShareLink: ((value: Record<string, unknown>) => void) | null = null
    createShareLinkMock.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveShareLink = resolve
        })
    )
    const writeTextMock = vi.fn().mockResolvedValue(undefined)
    Object.defineProperty(globalThis.navigator, "clipboard", {
      value: { writeText: writeTextMock },
      configurable: true,
    })

    const { rerender } = render(<ExportDialog open onClose={vi.fn()} />)

    fireEvent.click(screen.getByRole("button", { name: "Create share link" }))

    state.currentThreadId = "thread-2"
    rerender(<ExportDialog open onClose={vi.fn()} />)

    resolveShareLink?.({
      share_id: "share-1",
      token: "token-1",
      share_path: "/knowledge/shared/token-1",
      created_at: "2026-02-19T10:00:00.000Z",
      expires_at: "2026-02-20T10:00:00.000Z",
      permission: "view",
    })

    await act(async () => {
      await Promise.resolve()
    })

    expect(writeTextMock).not.toHaveBeenCalled()
    expect(screen.getByRole("button", { name: "Create share link" })).toBeEnabled()
    expect(screen.getByRole("button", { name: "Revoke link" })).toBeDisabled()
    expect(screen.queryByText(/Active link expires/i)).not.toBeInTheDocument()
  })

  it("keeps the revoke handle when clipboard copy fails after share-link creation", async () => {
    const writeTextMock = vi.fn().mockRejectedValue(new Error("clipboard denied"))
    Object.defineProperty(globalThis.navigator, "clipboard", {
      value: { writeText: writeTextMock },
      configurable: true,
    })

    render(<ExportDialog open onClose={vi.fn()} />)

    fireEvent.click(screen.getByRole("button", { name: "Create share link" }))

    await waitFor(() =>
      expect(messageOpenMock).toHaveBeenCalledWith(
        expect.objectContaining({
          type: "error",
          content: "Unable to copy share link, but the link remains active.",
        })
      )
    )

    expect(screen.getByText(/Active link expires/i)).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Revoke link" })).toBeEnabled()
  })

  it("preserves format defaults and preview copy feedback behavior", async () => {
    state.answer = "A".repeat(2205)
    Object.defineProperty(globalThis.navigator, "clipboard", {
      value: {
        writeText: vi.fn().mockResolvedValue(undefined),
      },
      configurable: true,
    })

    render(<ExportDialog open onClose={vi.fn()} />)

    expect(
      screen.getByRole("button", { name: /Markdown/i })
    ).toHaveAttribute("aria-pressed", "true")
    expect(screen.getByLabelText("Source excerpts")).toBeChecked()
    expect(screen.getByLabelText("Settings snapshot")).not.toBeChecked()

    fireEvent.click(screen.getByRole("button", { name: "Export" }))

    await waitFor(() => expect(screen.getByText("Preview")).toBeInTheDocument())
    expect(screen.getByText(/\.\.\. \(truncated\)/i)).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: /^Copy$/ }))

    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Copied" })).toBeInTheDocument()
    )
  })

  it("keeps the latest export copy confirmation visible until the latest timeout completes", async () => {
    const writeTextMock = vi.fn().mockResolvedValue(undefined)
    Object.defineProperty(globalThis.navigator, "clipboard", {
      value: {
        writeText: writeTextMock,
      },
      configurable: true,
    })

    try {
      render(<ExportDialog open onClose={vi.fn()} />)

      fireEvent.click(screen.getByRole("button", { name: "Export" }))

      await waitFor(() => expect(screen.getByText("Preview")).toBeInTheDocument())
      vi.useFakeTimers()

      fireEvent.click(screen.getByRole("button", { name: /^Copy$/ }))
      await act(async () => {
        await Promise.resolve()
      })
      expect(screen.getByRole("button", { name: "Copied" })).toBeInTheDocument()

      act(() => {
        vi.advanceTimersByTime(1000)
      })

      fireEvent.click(screen.getByRole("button", { name: "Copied" }))
      await act(async () => {
        await Promise.resolve()
      })
      expect(writeTextMock).toHaveBeenCalledTimes(2)

      act(() => {
        vi.advanceTimersByTime(1500)
      })
      expect(screen.getByRole("button", { name: "Copied" })).toBeInTheDocument()

      act(() => {
        vi.advanceTimersByTime(500)
      })
      expect(screen.getByRole("button", { name: /^Copy$/ })).toBeInTheDocument()
    } finally {
      vi.useRealTimers()
    }
  })

  it("resets transient export state when the dialog is closed and reopened", async () => {
    const { rerender } = render(<ExportDialog open onClose={vi.fn()} />)

    fireEvent.click(screen.getByRole("button", { name: "Export" }))

    await waitFor(() => expect(screen.getByText("Preview")).toBeInTheDocument())

    rerender(<ExportDialog open={false} onClose={vi.fn()} />)
    rerender(<ExportDialog open onClose={vi.fn()} />)

    expect(screen.queryByText("Preview")).not.toBeInTheDocument()
    expect(screen.queryByRole("button", { name: /^Copy$/ })).not.toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Download" })).not.toBeInTheDocument()
  })

  it("does not let a stale preview-copy completion leak into the next export session", async () => {
    let resolveCopy: (() => void) | null = null
    Object.defineProperty(globalThis.navigator, "clipboard", {
      value: {
        writeText: vi.fn().mockImplementation(
          () =>
            new Promise<void>((resolve) => {
              resolveCopy = resolve
            })
        ),
      },
      configurable: true,
    })

    const { rerender } = render(<ExportDialog open onClose={vi.fn()} />)

    fireEvent.click(screen.getByRole("button", { name: "Export" }))
    await waitFor(() => expect(screen.getByText("Preview")).toBeInTheDocument())

    fireEvent.click(screen.getByRole("button", { name: /^Copy$/ }))
    rerender(<ExportDialog open={false} onClose={vi.fn()} />)

    resolveCopy?.()
    await act(async () => {
      await Promise.resolve()
    })

    rerender(<ExportDialog open onClose={vi.fn()} />)
    fireEvent.click(screen.getByRole("button", { name: "Export" }))
    await waitFor(() => expect(screen.getByText("Preview")).toBeInTheDocument())

    expect(screen.getByRole("button", { name: /^Copy$/ })).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Copied" })).not.toBeInTheDocument()
  })
})
