import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import {
  QuickNotesSection,
  rankKeywordSuggestions
} from "../StudioPane/QuickNotesSection"

const {
  mockBgRequest,
  mockGetNoteKeywords,
  mockMessageSuccess,
  mockMessageError,
  mockMessageWarning,
  mockMessageOpen,
  mockMessageDestroy,
  mockSetCurrentNote,
  workspaceStoreState
} = vi.hoisted(() => {
  const bgRequest = vi.fn()
  const getNoteKeywords = vi.fn()
  const messageSuccess = vi.fn()
  const messageError = vi.fn()
  const messageWarning = vi.fn()
  const messageOpen = vi.fn()
  const messageDestroy = vi.fn()
  const setCurrentNote = vi.fn()

  const storeState = {
    currentNote: {
      id: undefined as number | undefined,
      title: "",
      content: "",
      keywords: [] as string[],
      version: undefined as number | undefined,
      isDirty: false
    },
    workspaceTag: "workspace:test",
    updateNoteTitle: vi.fn(),
    updateNoteContent: vi.fn(),
    updateNoteKeywords: vi.fn(),
    setCurrentNote,
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
    mockMessageOpen: messageOpen,
    mockMessageDestroy: messageDestroy,
    mockSetCurrentNote: setCurrentNote,
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
  useWorkspaceStore: (
    selector: (state: typeof workspaceStoreState) => unknown
  ) => selector(workspaceStoreState)
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: mockBgRequest
}))

vi.mock("@/services/note-keywords", () => ({
  getNoteKeywords: mockGetNoteKeywords
}))

vi.mock("@/components/Common/MarkdownPreview", () => ({
  MarkdownPreview: ({ content }: { content: string }) => (
    <div data-testid="markdown-preview-content">{content}</div>
  )
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
          warning: mockMessageWarning,
          open: mockMessageOpen,
          destroy: mockMessageDestroy
        },
        <></>
      ]
    }
  }
})

describe("QuickNotesSection Stage 3 authoring and conflict recovery", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    workspaceStoreState.currentNote = {
      id: undefined,
      title: "",
      content: "",
      keywords: [],
      version: undefined,
      isDirty: false
    }
    workspaceStoreState.workspaceTag = "workspace:test"
    workspaceStoreState.noteFocusTarget = null
    mockGetNoteKeywords.mockResolvedValue(["analysis", "summary", "research"])

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

  it("toggles between markdown edit and preview modes", async () => {
    workspaceStoreState.currentNote = {
      id: 5,
      title: "Preview Note",
      content: "## Heading\n\n- point",
      keywords: ["analysis"],
      version: 1,
      isDirty: true
    }

    render(<QuickNotesSection />)

    fireEvent.click(screen.getByRole("button", { name: "Preview" }))

    expect(await screen.findByTestId("quick-notes-markdown-preview")).toBeInTheDocument()
    expect(screen.getByTestId("markdown-preview-content")).toHaveTextContent(
      "## Heading"
    )
    expect(screen.getByTestId("markdown-preview-content")).toHaveTextContent(
      "- point"
    )

    fireEvent.click(screen.getByRole("button", { name: "Edit" }))
    expect(screen.getByPlaceholderText("Jot down notes, ideas, or observations...")).toBeInTheDocument()
  })

  it("ranks keyword suggestions by prefix before contains with dedupe", () => {
    const ranked = rankKeywordSuggestions("re", [
      "Research",
      "report",
      "core",
      "reference",
      "research"
    ])

    expect(ranked).toEqual(["Research", "report", "reference", "core"])
  })

  it("reloads latest note version and preserves unsaved local draft on conflict", async () => {
    workspaceStoreState.currentNote = {
      id: 42,
      title: "Local draft title",
      content: "Local unsaved paragraph",
      keywords: ["local-tag"],
      version: 1,
      isDirty: true
    }

    mockBgRequest.mockImplementation(async (request: { path: string; method?: string }) => {
      const path = String(request.path)
      if (path.includes("/api/v1/notes/search/") && path.includes("limit=8")) {
        return []
      }
      if (path.includes("/api/v1/notes/42?expected_version=1")) {
        throw { status: 409, message: "version conflict" }
      }
      if (path.endsWith("/api/v1/notes/42")) {
        return {
          id: 42,
          title: "Server title",
          content: "Server copy",
          keywords: [{ keyword: "workspace:test" }, { keyword: "server-tag" }],
          version: 2
        }
      }
      if (path.includes("/api/v1/notes/?")) {
        return { notes: [] }
      }
      return { notes: [] }
    })

    render(<QuickNotesSection />)

    fireEvent.click(screen.getByRole("button", { name: "Update" }))

    await waitFor(() => {
      expect(mockMessageOpen).toHaveBeenCalledTimes(1)
    })

    const conflictConfig = mockMessageOpen.mock.calls[0]?.[0]
    const renderedConflict = render(<>{conflictConfig?.content}</>)
    expect(renderedConflict.getByText("Reload latest")).toBeInTheDocument()
    fireEvent.click(renderedConflict.getByRole("button", { name: "Reload latest" }))

    await waitFor(() => {
      expect(mockSetCurrentNote).toHaveBeenCalledWith(
        expect.objectContaining({
          id: 42,
          title: "Local draft title",
          keywords: ["server-tag", "local-tag"],
          isDirty: true
        })
      )
    })

    const merged = mockSetCurrentNote.mock.calls[0]?.[0]
    expect(merged.content).toContain("Server copy")
    expect(merged.content).toContain("## Local Draft (Unsaved)")
    expect(merged.content).toContain("Local unsaved paragraph")
  })

  it("hides the saved indicator as soon as the note becomes dirty again", async () => {
    workspaceStoreState.currentNote = {
      id: 7,
      title: "Saved Note",
      content: "Stable content",
      keywords: [],
      version: 1,
      isDirty: false
    }
    workspaceStoreState.loadNote = vi.fn((nextNote) => {
      workspaceStoreState.currentNote = nextNote
    })

    mockBgRequest.mockImplementation(async (request: { path: string; method?: string }) => {
      const path = String(request.path)
      if (path.includes("/api/v1/notes/search/") && path.includes("limit=8")) {
        return []
      }
      if (path.includes("/api/v1/notes/7?expected_version=1")) {
        return {
          id: 7,
          title: "Saved Note",
          content: "Stable content",
          keywords: [{ keyword: "workspace:test" }],
          version: 2
        }
      }
      if (path.includes("/api/v1/notes/?")) {
        return { notes: [] }
      }
      return { notes: [] }
    })

    const { rerender } = render(<QuickNotesSection />)

    fireEvent.click(screen.getByRole("button", { name: "Update" }))

    expect(await screen.findByTestId("quick-notes-saved-indicator")).toBeInTheDocument()
    expect(screen.queryByText("Unsaved")).not.toBeInTheDocument()

    workspaceStoreState.currentNote = {
      ...workspaceStoreState.currentNote,
      content: "Stable content with draft changes",
      isDirty: true
    }
    rerender(<QuickNotesSection />)

    await waitFor(() => {
      expect(screen.queryByTestId("quick-notes-saved-indicator")).not.toBeInTheDocument()
    })
    expect(screen.getByText("Unsaved")).toBeInTheDocument()
  })
})
