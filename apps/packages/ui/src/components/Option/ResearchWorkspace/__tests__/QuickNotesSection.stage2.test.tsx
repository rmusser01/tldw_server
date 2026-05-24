import React from "react"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { QuickNotesSection } from "../StudioPane/QuickNotesSection"

const {
  mockBgRequest,
  mockMessageSuccess,
  mockMessageError,
  mockMessageWarning,
  mockLoadNote,
  workspaceStoreState
} = vi.hoisted(() => {
  const bgRequest = vi.fn()
  const messageSuccess = vi.fn()
  const messageError = vi.fn()
  const messageWarning = vi.fn()
  const loadNote = vi.fn()

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
    setCurrentNote: vi.fn(),
    clearCurrentNote: vi.fn(),
    loadNote,
    noteFocusTarget: null as { field: "title" | "content"; token: number } | null,
    clearNoteFocusTarget: vi.fn()
  }

  return {
    mockBgRequest: bgRequest,
    mockMessageSuccess: messageSuccess,
    mockMessageError: messageError,
    mockMessageWarning: messageWarning,
    mockLoadNote: loadNote,
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

const createDefaultBgRequestHandler = () => {
  mockBgRequest.mockImplementation(async (request: { path: string }) => {
    const path = String(request.path)

    if (path.includes("/api/v1/notes/search/") && path.includes("limit=8")) {
      return [
        {
          id: 11,
          title: "Workspace Seed Note",
          content: "Workspace seed content",
          keywords: ["workspace:test", "seed"],
          last_modified: "2026-02-18T09:00:00.000Z"
        }
      ]
    }

    if (path.includes("/api/v1/notes/?")) {
      return {
        notes: [
          {
            id: 101,
            title: "Global result note",
            content: "Global content",
            keywords: ["general"],
            last_modified: "2026-02-17T10:00:00.000Z"
          }
        ]
      }
    }

    if (
      path.includes("/api/v1/notes/search/") &&
      path.includes("limit=20") &&
      path.includes("tokens=workspace%3Atest")
    ) {
      return [
        {
          id: 202,
          title: "Workspace result note",
          content: "Workspace content",
          keywords: ["workspace:test", "focused"],
          last_modified: "2026-02-18T11:00:00.000Z"
        }
      ]
    }

    if (path.endsWith("/api/v1/notes/11")) {
      return {
        id: 11,
        title: "Workspace Seed Note",
        content: "Loaded detail",
        keywords: [{ keyword: "workspace:test" }, { keyword: "analysis" }],
        version: 4
      }
    }

    if (path.endsWith("/api/v1/notes/202")) {
      return {
        id: 202,
        title: "Workspace result note",
        content: "Loaded workspace result",
        keywords: [{ keyword: "workspace:test" }, { keyword: "focused" }],
        version: 2
      }
    }

    return { notes: [] }
  })
}

describe("QuickNotesSection Stage 2 workspace navigation", () => {
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
    createDefaultBgRequestHandler()
  })

  it("renders workspace notes list and loads a selected note", async () => {
    render(<QuickNotesSection />)

    const list = await screen.findByTestId("workspace-notes-list")
    expect(within(list).getByRole("button", { name: "Workspace Seed Note" })).toBeInTheDocument()

    fireEvent.click(within(list).getByRole("button", { name: "Workspace Seed Note" }))

    await waitFor(() => {
      expect(mockLoadNote).toHaveBeenCalledWith(
        expect.objectContaining({
          id: 11,
          title: "Workspace Seed Note",
          keywords: ["analysis"]
        })
      )
    })
  })

  it("prioritizes workspace-tagged notes in the load modal results", async () => {
    mockBgRequest.mockImplementation(async (request: { path: string }) => {
      const path = String(request.path)
      if (path.includes("/api/v1/notes/search/") && path.includes("limit=8")) {
        return []
      }
      if (path.includes("/api/v1/notes/?")) {
        return {
          notes: [
            {
              id: 1,
              title: "Global result note",
              content: "Global note body",
              keywords: ["general"],
              last_modified: "2026-02-18T08:00:00.000Z"
            }
          ]
        }
      }
      if (
        path.includes("/api/v1/notes/search/") &&
        path.includes("limit=20") &&
        path.includes("tokens=workspace%3Atest")
      ) {
        return [
          {
            id: 2,
            title: "Workspace result note",
            content: "Workspace note body",
            keywords: ["workspace:test", "focused"],
            last_modified: "2026-02-18T12:00:00.000Z"
          }
        ]
      }
      return { notes: [] }
    })

    render(<QuickNotesSection />)

    fireEvent.click(screen.getByRole("button", { name: "Load note" }))

    const dialog = await screen.findByRole("dialog")
    await waitFor(() => {
      expect(within(dialog).getByText("Workspace result note")).toBeInTheDocument()
      expect(within(dialog).getByText("Global result note")).toBeInTheDocument()
    })

    const titleNodes = within(dialog)
      .getAllByText(/result note/)
      .filter((node) => node.tagName.toLowerCase() === "p")

    expect(titleNodes[0]).toHaveTextContent("Workspace result note")
    expect(titleNodes[1]).toHaveTextContent("Global result note")
  })

  it("includes workspace token filter in note search requests", async () => {
    render(<QuickNotesSection />)

    fireEvent.click(screen.getByRole("button", { name: "Load note" }))

    await waitFor(() => {
      const tokenCalls = mockBgRequest.mock.calls.filter(
        ([request]: [{ path: string }]) =>
          String(request.path).includes("/api/v1/notes/search/") &&
          String(request.path).includes("tokens=workspace%3Atest")
      )
      expect(tokenCalls.length).toBeGreaterThanOrEqual(2)
    })
  })
})
