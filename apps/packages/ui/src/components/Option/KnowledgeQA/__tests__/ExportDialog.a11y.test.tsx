import { readSurfaceOfflineDraftQueue, retainSurfaceOfflineDraft, type OfflineDraftEntry } from "@/components/Notes/notes-manager-utils"
vi.mock("@plasmohq/storage", () => import("../../../../../../../tldw-frontend/extension/shims/plasmo-storage"))
import React from "react"
import { MemoryRouter } from "react-router-dom"
import {
  act,
  fireEvent,
  render as renderBare,
  screen,
  waitFor,
} from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { ExportDialog } from "../ExportDialog"
import type { RagResult, ScopeSnapshot } from "../types"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, options?: { defaultValue?: string }) =>
      options?.defaultValue ?? key,
  }),
}))

const render = (element: React.ReactElement) =>
  renderBare(element, { wrapper: MemoryRouter })

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
  notesAuthorityScope: "export-alice",
  messages: [] as Array<{ role: string; content: string }>,
  currentThreadId: "thread-1" as string | null,
  results: [] as RagResult[],
  citations: [] as Array<{ index: number }>,
  answer: "Test answer" as string | null,
  answerTrustState: "cited_answer" as
    | "cited_answer"
    | "uncited_degraded_answer"
    | "no_answer_insufficient_evidence"
    | "no_results"
    | "failed_search"
    | "unsynced_local_result"
    | "unknown_trust",
  answerEvidenceOrigin: "local_library" as
    | "local_library"
    | "web_fallback"
    | "mixed"
    | "unknown_origin"
    | null,
  query: "What does this source say?",
  resultQuery: undefined as string | null | undefined,
  settings: {
    sources: ["media_db", "notes"],
    include_media_ids: [42],
    include_note_ids: ["note-a"],
    top_k: 12,
    generation_provider: "openai",
    generation_model: "gpt-4o-mini",
    enable_web_fallback: false,
  },
  lastSearchScope: null as ScopeSnapshot | null,
  preset: "balanced",
  searchDetails: null as null | {
    expandedQueries?: string[];
    rerankingEnabled?: boolean;
    rerankingStrategy?: string;
    averageRelevance?: number | null;
    webFallbackTriggered?: boolean;
    webFallbackEngine?: string | null;
  },
};

vi.mock("../KnowledgeQAProvider", () => {
  const client = {
    createNote: createNoteMock,
    exportChatbook: exportChatbookMock,
    downloadChatbookExport: downloadChatbookExportMock,
    createConversationShareLink: createShareLinkMock,
    revokeConversationShareLink: revokeShareLinkMock,
  };
  return {
    useKnowledgeQA: () => ({
      isAuthorityCurrent: () => true,
      notesAuthorityScope: state.notesAuthorityScope,
      client,
      messages: state.messages,
      currentThreadId: state.currentThreadId,
      results: state.results,
      citations: state.citations,
      answer: state.answer,
      answerTrustState: state.answerTrustState,
      answerEvidenceOrigin: state.answerEvidenceOrigin,
      query: state.query,
      resultQuery: state.resultQuery,
      settings: state.settings,
      lastSearchScope: state.lastSearchScope,
      preset: state.preset,
      searchDetails: state.searchDetails,
    }),
  };
});

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

beforeEach(() => {
  vi.clearAllMocks();
  window.localStorage.clear();
  state.notesAuthorityScope = "export-alice";
  createNoteMock.mockResolvedValue({ id: 1 });
  exportChatbookMock.mockResolvedValue({
    success: true,
    job_id: "job-1",
    download_url: "/api/v1/chatbooks/download/job-1",
  });
  downloadChatbookExportMock.mockResolvedValue({
    blob: new Blob(["chatbook-content"], { type: "application/zip" }),
    filename: "knowledge.chatbook.zip",
  });
  createShareLinkMock.mockResolvedValue({
    share_id: "share-1",
    token: "token-1",
    share_path: "/knowledge/shared/token-1",
    created_at: "2026-02-19T10:00:00.000Z",
    expires_at: "2026-02-20T10:00:00.000Z",
    permission: "view",
  });
  revokeShareLinkMock.mockResolvedValue({ success: true, share_id: "share-1" });
  state.messages = [];
  state.currentThreadId = "thread-1";
  state.results = [];
  state.citations = [];
  state.answer = "Test answer";
  state.answerTrustState = "cited_answer";
  state.answerEvidenceOrigin = "local_library";
  state.query = "What does this source say?";
  state.resultQuery = undefined;
  state.settings = {
    sources: ["media_db", "notes"],
    include_media_ids: [42],
    include_note_ids: ["note-a"],
    top_k: 12,
    generation_provider: "openai",
    generation_model: "gpt-4o-mini",
    enable_web_fallback: false,
  };
  state.preset = "balanced";
  state.searchDetails = null;
  state.lastSearchScope = null;
});

describe("ExportDialog accessibility", () => {
  it("persists provenance in canonical content when NoteResponse drops metadata", async () => {
    createNoteMock.mockImplementation(async (content, fields) => ({ id: "canonical-export", title: fields.title, content, conversation_id: fields.conversation_id, version: 1 }))
    render(<ExportDialog open onClose={vi.fn()} />)
    fireEvent.click(screen.getByRole("button", { name: "Save to Notes" }))
    await waitFor(() => expect(createNoteMock).toHaveBeenCalled())
    expect(createNoteMock.mock.calls[0][0]).toContain("<!-- tldw-knowledge:v1:")
    expect(createNoteMock.mock.calls[0][1].conversation_id).toBe("thread-1")
    expect(createNoteMock.mock.calls[0][1]).toMatchObject({
      expected_provenance_version: 0,
      knowledge_provenance: { origin: "knowledge_qa", question: state.query, scope: { include_media_ids: [42], include_note_ids: ["note-a"] } },
    })
    expect(await screen.findByRole("link", { name: "Open saved note" })).toHaveAttribute("href", "/notes?source_ref_id=canonical-export")
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
        type: "warning",
        content: "Share link created, but copying it to the clipboard failed.",
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

    expect(
      await screen.findByRole("link", { name: "Open saved note" }),
    ).toHaveAttribute("href", "/notes?source_ref_id=1")
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

  it("exports an uncalibrated RRF score without inventing a relevance percentage", async () => {
    state.results = [{ id: "late_chunk:2:0", score: 0.004838709677419355 }]
    render(<ExportDialog open onClose={vi.fn()} />)
    fireEvent.click(screen.getByRole("button", { name: "Export" }))
    await waitFor(() => expect(screen.getByText("Preview")).toBeInTheDocument())
    const preview = screen.getByText((_, element) => element?.tagName.toLowerCase() === "pre" && Boolean(element.textContent?.includes("## Sources")))
    expect(preview).not.toHaveTextContent("Relevance: 0%")
    expect(preview).toHaveTextContent("Relevance: not measured")
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

  it("requires acknowledgement before exporting unsupported draft answers", async () => {
    state.answer = "Answer without citations."
    state.answerTrustState = "uncited_degraded_answer"
    state.answerEvidenceOrigin = "local_library"

    render(<ExportDialog open onClose={vi.fn()} />)

    expect(
      screen.getByText(/This answer is an unsupported draft/i)
    ).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Export" })).toBeDisabled()

    fireEvent.click(
      screen.getByRole("checkbox", {
        name: /I understand this unsupported draft/i,
      })
    )
    expect(screen.getByRole("button", { name: "Export" })).toBeEnabled()

    fireEvent.click(screen.getByRole("button", { name: "Export" }))

    await waitFor(() => expect(screen.getByText("Preview")).toBeInTheDocument())
    const preview = screen.getByText((_, element) => {
      if (!element || element.tagName.toLowerCase() !== "pre") return false
      const text = element.textContent || ""
      return (
        text.includes("Trust: unsupported draft") &&
        text.includes("Answer status: Uncited answer") &&
        text.includes("Evidence origin: local library")
      )
    })
    expect(preview).toBeInTheDocument()
  })

  it("blocks answer-content export for failed and no-result searches", () => {
    state.answer = null
    state.answerTrustState = "failed_search"
    state.answerEvidenceOrigin = null

    const { rerender } = render(<ExportDialog open onClose={vi.fn()} />)

    expect(
      screen.getByText(/This search state cannot be exported as answer content/i)
    ).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Export" })).toBeDisabled()

    state.answerTrustState = "no_results"
    rerender(<ExportDialog open onClose={vi.fn()} />)

    expect(screen.getByRole("button", { name: "Export" })).toBeDisabled()
    expect(
      screen.getByText(/This search state cannot be exported as answer content/i)
    ).toBeInTheDocument()
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
          type: "warning",
          content: "Share link created, but copying it to the clipboard failed.",
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

it("retries Save to Notes with the exact portable body and request identity", async () => {
  createNoteMock.mockRejectedValueOnce(new Error("Lost response")).mockResolvedValueOnce({ id: "saved" })
  render(<ExportDialog open onClose={vi.fn()} />)
  fireEvent.click(screen.getByRole("button", { name: "Save to Notes" }))
  await waitFor(() => expect(messageOpenMock).toHaveBeenCalledWith(expect.objectContaining({ type: "error" })))
  fireEvent.click(screen.getByRole("button", { name: "Save to Notes" }))
  await screen.findByRole("link", { name: "Open saved note" })
  expect(createNoteMock.mock.calls[1]).toEqual(createNoteMock.mock.calls[0])
  expect(createNoteMock.mock.calls[0][2].idempotencyKey).toBeTruthy()
})

it("retains the searched scope after the controls change", async () => {
  state.lastSearchScope = { preset: "balanced", sources: ["notes"], includeNoteIds: ["original"], includeMediaIds: [], collectionId: 7, keywordFilter: "original-topic", webFallback: false }
  render(<ExportDialog open onClose={vi.fn()} />)
  fireEvent.click(screen.getByRole("button", { name: "Save to Notes" }))
  await waitFor(() => expect(createNoteMock).toHaveBeenCalled())
  expect(createNoteMock.mock.calls.at(-1)![1].knowledge_provenance.scope).toEqual({ sources: ["notes"], include_note_ids: ["original"], include_media_ids: [], collection_id: 7, keyword_filter: "original-topic", enable_web_fallback: false })
  state.lastSearchScope = null
})

it.each(['Question one', null])('saves the answered question after editing the search input without searching (question=%s)', async resultQuery => {
  createNoteMock.mockClear().mockResolvedValue({ id: 'saved' })
  state.resultQuery = resultQuery
  state.query = 'Question two, not searched'
  render(<ExportDialog open onClose={vi.fn()} />)
  fireEvent.click(screen.getByRole('button', { name: 'Save to Notes' }))
  await waitFor(() => expect(createNoteMock).toHaveBeenCalled())
  expect(createNoteMock.mock.calls.at(-1)?.[1]).toMatchObject({ knowledge_provenance: { question: resultQuery ?? '' } })
  state.resultQuery = undefined
})

it.each([422, 409])('handles a rejected direct save without losing uncertain identity (status=%s)', async status => {
  messageOpenMock.mockClear()
  createNoteMock.mockReset()
  state.query = 'Original question'
  state.resultQuery = 'Original question'
  state.answer = 'Original answer'
  state.answerTrustState = 'cited_answer'
  const policy = { error_code: 'notes_provenance_encryption_unsupported', message: 'Knowledge provenance could not be saved; refresh its state and retry.' }
  createNoteMock.mockRejectedValueOnce(Object.assign(new Error('rejected'), { status, details: { detail: status === 409 ? policy : 'invalid input' } })).mockResolvedValue({ id: 'saved' })
  const view = render(<ExportDialog open onClose={vi.fn()} />)
  fireEvent.click(screen.getByRole('button', { name: 'Save to Notes' }))
  await waitFor(() => expect(messageOpenMock).toHaveBeenCalledWith(expect.objectContaining({ type: 'error' })))
  state.answer = 'Corrected answer'
  view.rerender(<ExportDialog open onClose={vi.fn()} />)
  fireEvent.click(screen.getByRole('button', { name: 'Save to Notes' }))
  await screen.findByRole('link', { name: 'Open saved note' })
  const [first, second] = createNoteMock.mock.calls
  if (status === 422) {
    expect(second[0]).toContain('Corrected answer')
    expect(second[2].idempotencyKey).not.toBe(first[2].idempotencyKey)
  } else expect(second).toEqual(first)
  state.resultQuery = undefined
})


it.each([429, 401])('retains a lost-ack direct save through pre-receipt HTTP %s', async status => {
  messageOpenMock.mockClear()
  createNoteMock.mockReset()
  state.resultQuery = 'Original question'
  state.answer = 'Original answer'
  state.answerTrustState = 'cited_answer'
  createNoteMock.mockRejectedValueOnce(new Error('Lost acknowledgment'))
    .mockRejectedValueOnce(Object.assign(new Error('Temporarily unavailable'), { status }))
    .mockResolvedValue({ id: 'saved' })
  const view = render(<ExportDialog open onClose={vi.fn()} />)
  fireEvent.click(screen.getByRole('button', { name: 'Save to Notes' }))
  await waitFor(() => expect(createNoteMock).toHaveBeenCalledTimes(1))
  await waitFor(() => expect(screen.getByRole('button', { name: 'Save to Notes' })).not.toBeDisabled())
  fireEvent.click(screen.getByRole('button', { name: 'Save to Notes' }))
  await waitFor(() => expect(createNoteMock).toHaveBeenCalledTimes(2))
  await waitFor(() => expect(screen.getByRole('button', { name: 'Save to Notes' })).not.toBeDisabled())
  state.answer = 'Later unsaved answer'
  view.rerender(<ExportDialog open onClose={vi.fn()} />)
  fireEvent.click(screen.getByRole('button', { name: 'Save to Notes' }))
  await screen.findByRole('link', { name: 'Open saved note' })
  expect(createNoteMock.mock.calls[2]).toEqual(createNoteMock.mock.calls[0])
  expect(createNoteMock.mock.calls[0][1].knowledge_provenance.question).toBe('Original question')
  state.resultQuery = undefined
})

function CloseableExport() {
  const [open, setOpen] = React.useState(true)
  return open ? <ExportDialog open onClose={() => setOpen(false)} /> :
    <button onClick={() => setOpen(true)}>Reopen export</button>
}

it("reconciles a committed export after real Close and allows a later deliberate export", async () => {
  const notes = new Map<string, Record<string, unknown>>()
  const receipts = new Map<string, Record<string, unknown>>()
  createNoteMock.mockImplementation(async (content, fields, options) => {
    if (receipts.has(options.idempotencyKey)) return receipts.get(options.idempotencyKey)
    const id = fields.id || crypto.randomUUID()
    const saved = { ...fields, content, id, version: 1 }
    notes.set(id, saved)
    receipts.set(options.idempotencyKey, saved)
    if (notes.size === 1) throw new Error("Committed, response dropped")
    return saved
  })
  render(<CloseableExport />)
  fireEvent.click(screen.getByRole("button", { name: "Save to Notes" }))
  await waitFor(() => expect(messageOpenMock).toHaveBeenCalled())
  fireEvent.click(screen.getByRole("button", { name: "Close export dialog" }))
  state.answer = "Newer displayed answer"
  fireEvent.click(screen.getByRole("button", { name: "Reopen export" }))
  fireEvent.click(screen.getByRole("button", { name: "Save to Notes" }))
  await screen.findByRole("link", { name: "Open saved note" })
  expect(notes.size).toBe(1)
  expect(createNoteMock.mock.calls[1]).toEqual(createNoteMock.mock.calls[0])
  expect(createNoteMock.mock.calls[0][1].id).toMatch(/^[a-f0-9-]{36}$/)
  fireEvent.click(screen.getByRole("button", { name: "Save to Notes" }))
  await waitFor(() => expect(notes.size).toBe(2))
  expect(createNoteMock.mock.calls[2][0]).toContain("Newer displayed answer")
})

it.each(["write", "readback"])("does not dispatch export when persistent retention fails at %s", async (failure) => {
  render(<ExportDialog open onClose={vi.fn()} />)
  const prototype = Object.getPrototypeOf(window.localStorage) as Storage
  const originalSet = prototype.setItem
  const originalGet = prototype.getItem
  let written = false
  const write = vi.spyOn(prototype, "setItem").mockImplementation(function (key, value) {
    if (key.startsWith("tldw:notesOfflineDraftQueue:v1")) {
      if (failure === "write") throw new Error("Storage full")
      written = true
    }
    return originalSet.call(this, key, value)
  })
  const read = vi.spyOn(prototype, "getItem").mockImplementation(function (key) {
    if (key.startsWith("tldw:notesOfflineDraftQueue:v1") && written) return null
    return originalGet.call(this, key)
  })
  fireEvent.click(screen.getByRole("button", { name: "Save to Notes" }))
  await waitFor(() => expect(messageOpenMock).toHaveBeenCalled())
  expect(createNoteMock).not.toHaveBeenCalled()
  read.mockRestore()
  write.mockRestore()
})

it("does not move an uncertain export into another owner after Close", async () => {
  createNoteMock.mockRejectedValueOnce(new Error("Response lost")).mockResolvedValue({ id: "bob-note" })
  render(<CloseableExport />)
  fireEvent.click(screen.getByRole("button", { name: "Save to Notes" }))
  await waitFor(() => expect(messageOpenMock).toHaveBeenCalled())
  const oldQueue = await readSurfaceOfflineDraftQueue("export-alice")
  fireEvent.click(screen.getByRole("button", { name: "Close export dialog" }))
  state.notesAuthorityScope = "export-bob"
  state.answer = "Bob answer"
  fireEvent.click(screen.getByRole("button", { name: "Reopen export" }))
  fireEvent.click(screen.getByRole("button", { name: "Save to Notes" }))
  await screen.findByRole("link", { name: "Open saved note" })
  expect(createNoteMock.mock.calls[1][2].idempotencyKey).not.toBe(createNoteMock.mock.calls[0][2].idempotencyKey)
  expect(createNoteMock.mock.calls[1][0]).toContain("Bob answer")
  expect(await readSurfaceOfflineDraftQueue("export-alice")).toEqual(oldQueue)
})

it("refuses to select another operation when two unresolved exports belong to the same owned thread", async () => {
  const entry: OfflineDraftEntry = { key: 'surface:knowledge-export:"thread-1":0463b785-f3ba-4b66-a656-a77986e0fe9e',
    noteId: null, baseVersion: null, title: 'Original', content: 'Original body', keywords: [], metadata: null,
    backlinkConversationId: 'thread-1', backlinkMessageId: null, updatedAt: '2026-10-08T00:00:00Z', syncState: 'queued', lastError: null,
    pendingWrite: { key: 'original-key', expectedVersion: null, body: { content: 'Original body', id: '80d02e84-b04c-45be-9c65-9bf54cfc9d3a' } } }
  await retainSurfaceOfflineDraft('export-alice', entry)
  await retainSurfaceOfflineDraft('export-alice', { ...entry, key: 'surface:knowledge-export:"thread-1":ecfd16eb-9091-4aad-92d2-697894654fed',
    pendingWrite: { ...entry.pendingWrite!, key: 'different-key', body: { content: 'Different body', id: '8dd8608b-8470-4d1b-b2fc-8be311c68f62' } } })
  render(<ExportDialog open onClose={vi.fn()} />)
  fireEvent.click(screen.getByRole('button', { name: 'Save to Notes' }))
  await waitFor(() => expect(messageOpenMock).toHaveBeenCalled())
  expect(createNoteMock).not.toHaveBeenCalled()
  expect(messageOpenMock.mock.calls[0][0].content).toContain('Multiple unresolved exports')
})


it("retries the acknowledged export after retirement fails and accepts a new deliberate export only after cleanup", async () => {
  const notes = new Map<string, Record<string, unknown>>()
  const receipts = new Map<string, Record<string, unknown>>()
  createNoteMock.mockImplementation(async (content, fields, options) => {
    if (receipts.has(options.idempotencyKey)) return receipts.get(options.idempotencyKey)
    const id = fields.id || crypto.randomUUID()
    const saved = { ...fields, content, id, version: 1 }
    notes.set(id, saved)
    receipts.set(options.idempotencyKey, saved)
    return saved
  })
  const prototype = Object.getPrototypeOf(window.localStorage) as Storage
  const originalRemove = prototype.removeItem
  const remove = vi.spyOn(prototype, "removeItem").mockImplementation(function (key) {
    if (key.startsWith("tldw:notesOfflineDraftQueue:v1")) throw new Error("Storage blocked")
    return originalRemove.call(this, key)
  })
  render(<CloseableExport />)
  fireEvent.click(screen.getByRole("button", { name: "Save to Notes" }))
  await screen.findByRole("link", { name: "Open saved note" })
  await waitFor(() => expect(screen.getByRole("button", { name: "Save to Notes" }).hasAttribute("disabled")).toBe(false))
  expect(Object.keys(await readSurfaceOfflineDraftQueue("export-alice"))).toHaveLength(1)
  fireEvent.click(screen.getByRole("button", { name: "Close export dialog" }))
  remove.mockRestore()
  state.answer = "Newer answer after acknowledgment"
  fireEvent.click(screen.getByRole("button", { name: "Reopen export" }))
  fireEvent.click(screen.getByRole("button", { name: "Save to Notes" }))
  await screen.findByRole("link", { name: "Open saved note" })
  await waitFor(() => expect(screen.getByRole("button", { name: "Save to Notes" }).hasAttribute("disabled")).toBe(false))
  expect(createNoteMock.mock.calls[1]).toEqual(createNoteMock.mock.calls[0])
  expect(notes.size).toBe(1)
  expect(await readSurfaceOfflineDraftQueue("export-alice")).toEqual({})
  fireEvent.click(screen.getByRole("button", { name: "Save to Notes" }))
  await waitFor(() => expect(notes.size).toBe(2))
  expect(createNoteMock.mock.calls[2][0]).toContain("Newer answer after acknowledgment")
})
