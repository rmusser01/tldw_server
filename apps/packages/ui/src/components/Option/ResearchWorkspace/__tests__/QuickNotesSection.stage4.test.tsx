import { readKnowledgeNoteProvenance } from "@/utils/knowledge-note-provenance"
import React from "react"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { QuickNotesSection } from "../StudioPane/QuickNotesSection"

const {
  mockBgRequest,
  mockGetNoteKeywords,
  mockMessageSuccess,
  mockMessageError,
  mockMessageWarning,
  workspaceStoreState
} = vi.hoisted(() => {
  const bgRequest = vi.fn()
  const getNoteKeywords = vi.fn()
  const messageSuccess = vi.fn()
  const messageError = vi.fn()
  const messageWarning = vi.fn()

  const storeState = {
    currentNote: {
      id: 7 as string | number,
      title: "My Study Note",
      content: "Line one\nLine two",
      keywords: ["analysis", "physics"],
      version: 1,
      isDirty: false
    },
    workspaceTag: "workspace:test",
    updateNoteTitle: vi.fn(),
    updateNoteContent: vi.fn(),
    updateNoteKeywords: vi.fn(),
    setCurrentNote: vi.fn(),
    clearCurrentNote: vi.fn(),
    loadNote: vi.fn(),
    noteFocusTarget: null as { field: "title" | "content"; token: number } | null,
    clearNoteFocusTarget: vi.fn()
  }

  return {
    mockBgRequest: bgRequest,
    mockGetNoteKeywords: getNoteKeywords,
    mockMessageSuccess: messageSuccess,
    mockMessageError: messageError,
    mockMessageWarning: messageWarning,
    workspaceStoreState: storeState
  }
})

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?:
        | string
        | {
            defaultValue?: string
          }
    ) => {
      if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
      if (defaultValueOrOptions?.defaultValue) return defaultValueOrOptions.defaultValue
      return key
    }
  })
}))

vi.mock("@/store/workspace", () => ({
  useWorkspaceStore: Object.assign(
    (selector: (state: typeof workspaceStoreState) => unknown) =>
      selector(workspaceStoreState),
    { getState: () => workspaceStoreState, subscribe: () => () => {} }
  )
}))

vi.mock("@/services/service-prompts", () => ({
  loadServicePromptSnapshot: async () => ({
    requestScope: {
      config: {
        serverUrl: "https://research.example",
        authMode: "single-user"
      },
      userId: null
    },
    scopeSignal: new AbortController().signal,
    scopeInvalidatedSignal: new AbortController().signal,
    release: () => {}
  })
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: mockBgRequest
}))

vi.mock("@/services/note-keywords", () => ({
  getNoteKeywords: mockGetNoteKeywords
}))

vi.mock("@/components/Common/MarkdownPreview", () => ({
  MarkdownPreview: ({ content }: { content: string }) => <div>{content}</div>
}))

vi.mock("antd", async () => {
  const actual = await vi.importActual<typeof import("antd")>("antd")
  return {
    ...actual,
    message: {
      useMessage: () => [
        {
          success: mockMessageSuccess,
          error: mockMessageError,
          warning: mockMessageWarning
        },
        <></>
      ]
    }
  }
})

describe("QuickNotesSection Stage 4 layout and export", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    workspaceStoreState.currentNote = {
      id: 7,
      title: "My Study Note",
      content: "Line one\nLine two",
      keywords: ["analysis", "physics"],
      version: 1,
      isDirty: false,
    }
    workspaceStoreState.noteFocusTarget = null
    mockGetNoteKeywords.mockResolvedValue([])
    mockBgRequest.mockImplementation(async (request: { path: string }) => {
      const path = String(request.path)
      if (path.includes("/api/v1/notes/search/") && path.includes("limit=8")) {
        return []
      }
      if (path.includes("/api/v1/notes/?")) {
        return { notes: [] }
      }
      return { notes: [] }
    })
  })

  it("exports the current note as markdown with a sanitized filename", async () => {
    const originalBlob = Blob
    const capturedBlobPayloads: string[] = []
    class CapturingBlob extends originalBlob {
      constructor(parts?: BlobPart[], options?: BlobPropertyBag) {
        super(parts, options)
        const payload = Array.isArray(parts)
          ? parts.map((part) => String(part)).join("")
          : ""
        capturedBlobPayloads.push(payload)
      }
    }
    vi.stubGlobal("Blob", CapturingBlob as unknown as typeof Blob)

    const createObjectUrlSpy = vi
      .spyOn(URL, "createObjectURL")
      .mockReturnValue("blob:quick-note")
    const revokeObjectUrlSpy = vi
      .spyOn(URL, "revokeObjectURL")
      .mockImplementation(() => {})
    const anchorClickSpy = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => {})

    const originalCreateElement = document.createElement.bind(document)
    const anchor = originalCreateElement("a")
    const createElementSpy = vi
      .spyOn(document, "createElement")
      .mockImplementation(((tagName: string) => {
        if (tagName.toLowerCase() === "a") {
          return anchor
        }
        return originalCreateElement(tagName)
      }) as typeof document.createElement)

    render(<QuickNotesSection />)

    fireEvent.click(screen.getByRole("button", { name: "Download .md" }))

    await waitFor(() => {
      expect(createObjectUrlSpy).toHaveBeenCalledTimes(1)
    })
    expect(anchorClickSpy).toHaveBeenCalledTimes(1)
    expect(revokeObjectUrlSpy).toHaveBeenCalledTimes(1)
    expect(anchor.download).toBe("my-study-note.md")

    const exportedText = capturedBlobPayloads[0] || ""
    expect(exportedText).toContain("# My Study Note")
    expect(exportedText).toContain("Tags: #analysis #physics")
    expect(exportedText).toContain("Line one")
    expect(exportedText).toContain("Line two")

    createElementSpy.mockRestore()
    createObjectUrlSpy.mockRestore()
    revokeObjectUrlSpy.mockRestore()
    anchorClickSpy.mockRestore()
    vi.stubGlobal("Blob", originalBlob)
  })

  it("keeps quick notes controls visible in constrained container heights", async () => {
    render(
      <div style={{ height: "260px" }}>
        <QuickNotesSection />
      </div>,
    )

    expect(screen.getByText("Quick Notes")).toBeInTheDocument()
    expect(
      screen.getByPlaceholderText("Jot down notes, ideas, or observations..."),
    ).toBeInTheDocument()
  })

  it("exposes durable labels for note fields and compact note actions", async () => {
    render(<QuickNotesSection />)

    expect(screen.getByLabelText("Note title")).toHaveValue("My Study Note")
    expect(screen.getByLabelText("Note keywords")).toBeInTheDocument()
    expect(screen.getByLabelText("Note content")).toHaveValue(
      "Line one\nLine two",
    )
    expect(
      screen.getByRole("button", { name: "Clear current note" }),
    ).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Load note" }))

    expect(await screen.findByLabelText("Search notes")).toBeInTheDocument()
  })

  it("preserves canonical provenance when Quick Notes replaces the entire body", async () => {
    const original = {
      origin: "knowledge_qa",
      trust_state: "uncited_degraded_answer",
      thread_id: "owned-thread",
      research: {
        workspace_id: "workspace-a",
        import_id: "import-a",
        sources: [],
      },
    }
    const marker = `<!-- tldw-knowledge:v1:${encodeURIComponent(JSON.stringify(original))} -->`
    workspaceStoreState.currentNote.id = "e3b16146-9e38-42e0-bd15-549e60bd31a3"
    workspaceStoreState.currentNote.content = `Original excerpt\n\n${marker}`
    const view = render(<QuickNotesSection />)
    expect(screen.getByLabelText("Note content")).toHaveValue(
      "Original excerpt",
    )
    fireEvent.change(screen.getByLabelText("Note content"), {
      target: { value: "Edited body" },
    })
    const edited = workspaceStoreState.updateNoteContent.mock.lastCall![0]
    expect(edited).toContain("Edited body")
    expect(readKnowledgeNoteProvenance(edited)).toEqual(original)
    workspaceStoreState.currentNote.content = edited
    view.rerender(<QuickNotesSection />)
    fireEvent.click(screen.getByRole("button", { name: "Update" }))
    await waitFor(() =>
      expect(mockBgRequest).toHaveBeenCalledWith(
        expect.objectContaining({
          path: "/api/v1/notes/e3b16146-9e38-42e0-bd15-549e60bd31a3",
          headers: expect.objectContaining({ "expected-version": "1" }),
          method: "PUT",
          body: expect.objectContaining({
            content: edited,
            keywords: expect.arrayContaining([
              "workspace:workspace-a",
              "workspace:test",
            ]),
          }),
        }),
      ),
    )
  })
  it.each(["Hi", "", "<!-- tldw-knowledge:v1:invalid -->"])(
    "hides validated provenance in Load-note preview for %s",
    async (body) => {
      const marker = `<!-- tldw-knowledge:v1:${encodeURIComponent(JSON.stringify({ origin: "knowledge_qa" }))} -->`
      const note = {
        id: "preview-note",
        title: "Saved source",
        content: `${body}\n\n${marker}`,
        keywords: [],
      }
      mockBgRequest.mockResolvedValue({ notes: [note] })
      render(<QuickNotesSection />)
      fireEvent.click(screen.getByRole("button", { name: "Load note" }))
      const title = await within(screen.getByRole("dialog")).findByText(
        "Saved source",
      )
      const preview = title.closest("button")!
      expect(preview.textContent).not.toContain("%7B")
      expect(preview.textContent).toContain(body || "No content")
    },
  )

})
