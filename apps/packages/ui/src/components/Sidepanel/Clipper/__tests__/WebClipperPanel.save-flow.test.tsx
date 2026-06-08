import React from "react"
import userEvent from "@testing-library/user-event"
import { render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { buildClipDraft } from "@/services/web-clipper/draft-builder"
import {
  clearPendingClipDraft,
  readPendingClipDraft,
  writePendingClipDraft
} from "@/services/web-clipper/pending-draft"
import WebClipperPanel from "../WebClipperPanel"

const apiMocks = vi.hoisted(() => ({
  initialize: vi.fn(),
  listNoteFolders: vi.fn(),
  listWorkspaces: vi.fn(),
  saveWebClip: vi.fn(),
  persistWebClipEnrichment: vi.fn(),
  createChatCompletion: vi.fn()
}))

const openTabMock = vi.hoisted(() => vi.fn())
const navigateMock = vi.hoisted(() => vi.fn())

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: (...args: unknown[]) =>
      apiMocks.initialize(...args),
    listNoteFolders: (...args: unknown[]) =>
      apiMocks.listNoteFolders(...args),
    listWorkspaces: (...args: unknown[]) =>
      apiMocks.listWorkspaces(...args),
    saveWebClip: (...args: unknown[]) =>
      apiMocks.saveWebClip(...args),
    persistWebClipEnrichment: (...args: unknown[]) =>
      apiMocks.persistWebClipEnrichment(...args),
    createChatCompletion: (...args: unknown[]) =>
      apiMocks.createChatCompletion(...args)
  }
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?: string | { defaultValue?: string }
    ) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      if (fallbackOrOptions && typeof fallbackOrOptions === "object") {
        return fallbackOrOptions.defaultValue || key
      }
      return key
    }
  })
}))

const createDraft = () =>
  buildClipDraft({
    clipId: "clip-123",
    requestedType: "article",
    pageUrl: "https://example.com/story",
    pageTitle: "Example Story",
    extracted: {
      articleText: "Alpha body copy",
      fullPageText: "Alpha body copy"
    }
  })

const createScreenshotDraft = () =>
  buildClipDraft({
    clipId: "clip-shot-123",
    requestedType: "screenshot",
    pageUrl: "https://example.com/screenshot",
    pageTitle: "Screenshot Story",
    extracted: {
      screenshotDataUrl: "data:image/png;base64,QUJDRA=="
    }
  })

const createRichExtractDraft = () =>
  buildClipDraft({
    clipId: "clip-rich-123",
    requestedType: "article",
    pageUrl: "https://example.com/rich-story",
    pageTitle: "Rich Story",
    extracted: {
      articleText: "Visible article summary",
      fullPageText: "Full article body with more detail"
    }
  })

const createChatCompletionResponse = (content: string) => ({
  json: vi.fn().mockResolvedValue({
    choices: [{ message: { content } }]
  })
})

const createDeferred = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((nextResolve) => {
    resolve = nextResolve
  })
  return { promise, resolve }
}

const chooseWorkspaceDestination = async (
  user: ReturnType<typeof userEvent.setup>,
  destinationLabel: "Workspace" | "Both" = "Workspace",
  workspaceId = "workspace-alpha"
) => {
  await user.click(screen.getByRole("radio", { name: destinationLabel }))
  const picker = await screen.findByRole("combobox", { name: "Workspace" })
  await user.selectOptions(picker, workspaceId)
}

describe("WebClipperPanel save flow", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    window.sessionStorage.clear()
    clearPendingClipDraft()
    apiMocks.initialize.mockResolvedValue(undefined)
    apiMocks.listNoteFolders.mockResolvedValue({
      items: [
        {
          id: 1,
          name: "Inbox",
          path: "Inbox",
          parent_id: null
        },
        {
          id: 17,
          name: "Captured Articles",
          path: "Inbox/Captured Articles",
          parent_id: 1
        }
      ],
      count: 2
    })
    apiMocks.listWorkspaces.mockResolvedValue({
      items: [
        {
          id: "workspace-alpha",
          name: "Alpha Workspace",
          archived: false,
          study_materials_policy: "general",
          deleted: false,
          created_at: "2026-05-27T00:00:00Z",
          last_modified: "2026-05-27T00:00:00Z",
          version: 1
        },
        {
          id: "workspace-beta",
          name: "Beta Workspace",
          archived: false,
          study_materials_policy: "workspace",
          deleted: false,
          created_at: "2026-05-27T00:00:00Z",
          last_modified: "2026-05-27T00:00:00Z",
          version: 1
        }
      ],
      total: 2
    })
    apiMocks.saveWebClip.mockResolvedValue({
      clip_id: "clip-123",
      note_id: "note-123",
      note: { id: "note-123", title: "Example Story", version: 1 },
      workspace_placement: null,
      attachments: [],
      status: "saved",
      warnings: [],
      workspace_placement_saved: false,
      workspace_placement_count: 0
    })
    apiMocks.persistWebClipEnrichment.mockResolvedValue({
      clip_id: "clip-123",
      enrichment_type: "ocr",
      status: "complete",
      source_note_version: 2,
      inline_applied: true,
      inline_summary: "Captured text summary.",
      conflict_reason: null,
      warnings: []
    })
    apiMocks.createChatCompletion.mockResolvedValue(
      createChatCompletionResponse(
        JSON.stringify({
          inline_summary: "Captured text summary.",
          structured_payload: {
            transcript: "Captured text summary."
          }
        })
      )
    )

    vi.stubGlobal("chrome", {
      tabs: {
        create: openTabMock
      },
      runtime: {
        getURL: (path: string) => `chrome-extension://unit-test/${path}`
      }
    })
    ;(window as Window & { __tldwNavigate?: (path: string) => void }).__tldwNavigate =
      navigateMock
  })

  it("shows the review-sheet filing controls", async () => {
    render(<WebClipperPanel draft={createDraft()} onCancel={vi.fn()} />)

    expect(screen.getByLabelText("Title")).toHaveValue("Example Story")
    expect(screen.getByLabelText("Comment")).toHaveValue("")
    expect(screen.getByLabelText("Tags")).toHaveValue("")
    expect(
      await screen.findByRole("combobox", { name: "Folder" })
    ).toBeInTheDocument()
    expect(screen.getByRole("radio", { name: "Note" })).toBeChecked()
    expect(screen.getByRole("radio", { name: "Workspace" })).not.toBeChecked()
    expect(screen.getByRole("radio", { name: "Both" })).not.toBeChecked()
    expect(apiMocks.listNoteFolders).toHaveBeenCalledTimes(1)
    expect(apiMocks.listWorkspaces).not.toHaveBeenCalled()
    expect(screen.getByLabelText("Run OCR")).not.toBeChecked()
    expect(screen.getByLabelText("Run visual analysis")).not.toBeChecked()
    expect(screen.getByRole("button", { name: "Cancel" })).toBeEnabled()
  })

  it("loads workspace choices after choosing Workspace and submits the selected workspace", async () => {
    const user = userEvent.setup()

    render(<WebClipperPanel draft={createDraft()} onCancel={vi.fn()} />)

    await user.click(screen.getByRole("radio", { name: "Workspace" }))

    const picker = await screen.findByRole("combobox", { name: "Workspace" })
    await user.selectOptions(picker, "workspace-beta")
    await user.click(screen.getByRole("button", { name: "Save clip" }))

    await waitFor(() => {
      expect(apiMocks.saveWebClip).toHaveBeenCalledTimes(1)
    })
    expect(apiMocks.listWorkspaces).toHaveBeenCalledTimes(1)
    expect(apiMocks.saveWebClip).toHaveBeenCalledWith(
      expect.objectContaining({
        destination_mode: "workspace",
        workspace: { workspace_id: "workspace-beta" }
      })
    )
    expect(screen.getByText("Advanced: enter workspace ID manually")).toBeInTheDocument()
  })

  it("loads note folder choices and submits the selected folder", async () => {
    const user = userEvent.setup()

    render(<WebClipperPanel draft={createDraft()} onCancel={vi.fn()} />)

    const picker = await screen.findByRole("combobox", { name: "Folder" })
    await user.selectOptions(picker, "17")
    await user.click(screen.getByRole("button", { name: "Save clip" }))

    await waitFor(() => {
      expect(apiMocks.saveWebClip).toHaveBeenCalledTimes(1)
    })
    expect(apiMocks.saveWebClip).toHaveBeenCalledWith(
      expect.objectContaining({
        note: expect.objectContaining({
          folder_id: 17
        })
      })
    )
    expect(screen.getByText("Advanced: enter folder ID manually")).toBeInTheDocument()
  })

  it("keeps raw folder ID entry available when the folder picker cannot load", async () => {
    const user = userEvent.setup()
    apiMocks.listNoteFolders.mockRejectedValueOnce(new Error("offline"))

    render(<WebClipperPanel draft={createDraft()} onCancel={vi.fn()} />)

    expect(
      await screen.findByText("Folder picker could not load. Enter a folder ID manually.")
    ).toBeInTheDocument()
    await user.type(screen.getByLabelText("Folder ID"), "17")
    await user.click(screen.getByRole("button", { name: "Save clip" }))

    await waitFor(() => {
      expect(apiMocks.saveWebClip).toHaveBeenCalledTimes(1)
    })
    expect(apiMocks.saveWebClip).toHaveBeenCalledWith(
      expect.objectContaining({
        note: expect.objectContaining({
          folder_id: 17
        })
      })
    )
  })

  it("retries folder picker loading after a transient failure when returning to note mode", async () => {
    const user = userEvent.setup()
    apiMocks.listNoteFolders
      .mockRejectedValueOnce(new Error("offline"))
      .mockResolvedValueOnce({
        items: [
          {
            id: 17,
            name: "Captured Articles",
            path: "Inbox/Captured Articles",
            parent_id: 1
          }
        ],
        count: 1
      })

    render(<WebClipperPanel draft={createDraft()} onCancel={vi.fn()} />)

    expect(
      await screen.findByText("Folder picker could not load. Enter a folder ID manually.")
    ).toBeInTheDocument()

    await user.click(screen.getByRole("radio", { name: "Workspace" }))
    await user.click(screen.getByRole("radio", { name: "Note" }))

    expect(
      await screen.findByRole("combobox", { name: "Folder" })
    ).toBeInTheDocument()
    expect(apiMocks.listNoteFolders).toHaveBeenCalledTimes(2)
  })

  it("keeps raw workspace ID entry hidden while workspace choices are loading", async () => {
    const user = userEvent.setup()
    const workspaceList = createDeferred<{
      items: unknown[]
      total: number
    }>()
    apiMocks.listWorkspaces.mockReturnValueOnce(workspaceList.promise)

    render(<WebClipperPanel draft={createDraft()} onCancel={vi.fn()} />)

    await user.click(screen.getByRole("radio", { name: "Workspace" }))

    expect(screen.getByText("Loading workspaces...")).toBeInTheDocument()
    expect(screen.queryByLabelText("Workspace ID")).not.toBeInTheDocument()

    workspaceList.resolve({
      items: [
        {
          id: "workspace-alpha",
          name: "Alpha Workspace",
          archived: false,
          study_materials_policy: "general",
          deleted: false,
          created_at: "2026-05-27T00:00:00Z",
          last_modified: "2026-05-27T00:00:00Z",
          version: 1
        }
      ],
      total: 1
    })

    expect(
      await screen.findByRole("combobox", { name: "Workspace" })
    ).toBeInTheDocument()
  })

  it("keeps raw workspace ID entry available when the workspace picker cannot load", async () => {
    const user = userEvent.setup()
    apiMocks.listWorkspaces.mockRejectedValueOnce(new Error("offline"))

    render(<WebClipperPanel draft={createDraft()} onCancel={vi.fn()} />)

    await user.click(screen.getByRole("radio", { name: "Workspace" }))

    expect(
      await screen.findByText("Workspace picker could not load. Enter a workspace ID manually.")
    ).toBeInTheDocument()
    await user.type(screen.getByLabelText("Workspace ID"), "workspace-fallback")
    await user.click(screen.getByRole("button", { name: "Save clip" }))

    await waitFor(() => {
      expect(apiMocks.saveWebClip).toHaveBeenCalledTimes(1)
    })
    expect(apiMocks.saveWebClip).toHaveBeenCalledWith(
      expect.objectContaining({
        workspace: { workspace_id: "workspace-fallback" }
      })
    )
  })

  it("clears stale manual workspace IDs when picker options load", async () => {
    const user = userEvent.setup()
    apiMocks.listWorkspaces
      .mockRejectedValueOnce(new Error("offline"))
      .mockResolvedValueOnce({
        items: [
          {
            id: "workspace-alpha",
            name: "Alpha Workspace",
            archived: false,
            study_materials_policy: "general",
            deleted: false,
            created_at: "2026-05-27T00:00:00Z",
            last_modified: "2026-05-27T00:00:00Z",
            version: 1
          }
        ],
        total: 1
      })

    render(<WebClipperPanel draft={createDraft()} onCancel={vi.fn()} />)

    await user.click(screen.getByRole("radio", { name: "Workspace" }))
    expect(
      await screen.findByText("Workspace picker could not load. Enter a workspace ID manually.")
    ).toBeInTheDocument()
    await user.type(screen.getByLabelText("Workspace ID"), "workspace-stale")

    await user.click(screen.getByRole("radio", { name: "Note" }))
    await user.click(screen.getByRole("radio", { name: "Workspace" }))
    expect(await screen.findByRole("combobox", { name: "Workspace" })).toHaveValue("")

    await user.click(screen.getByRole("button", { name: "Save clip" }))

    expect(
      await screen.findByText("Choose a workspace before saving to Workspace or Both.")
    ).toBeInTheDocument()
    expect(apiMocks.saveWebClip).not.toHaveBeenCalled()
  })

  it("retries workspace picker loading after a transient failure when returning to workspace mode", async () => {
    const user = userEvent.setup()
    apiMocks.listWorkspaces
      .mockRejectedValueOnce(new Error("offline"))
      .mockResolvedValueOnce({
        items: [
          {
            id: "workspace-alpha",
            name: "Alpha Workspace",
            archived: false,
            study_materials_policy: "general",
            deleted: false,
            created_at: "2026-05-27T00:00:00Z",
            last_modified: "2026-05-27T00:00:00Z",
            version: 1
          }
        ],
        total: 1
      })

    render(<WebClipperPanel draft={createDraft()} onCancel={vi.fn()} />)

    await user.click(screen.getByRole("radio", { name: "Workspace" }))
    expect(
      await screen.findByText("Workspace picker could not load. Enter a workspace ID manually.")
    ).toBeInTheDocument()

    await user.click(screen.getByRole("radio", { name: "Note" }))
    await user.click(screen.getByRole("radio", { name: "Workspace" }))

    expect(
      await screen.findByRole("combobox", { name: "Workspace" })
    ).toBeInTheDocument()
    expect(apiMocks.listWorkspaces).toHaveBeenCalledTimes(2)
  })

  it("does not load workspace choices for note-only saves", async () => {
    const user = userEvent.setup()

    render(<WebClipperPanel draft={createDraft()} onCancel={vi.fn()} />)

    await user.click(screen.getByRole("button", { name: "Save clip" }))

    await waitFor(() => {
      expect(apiMocks.saveWebClip).toHaveBeenCalledTimes(1)
    })
    expect(apiMocks.listWorkspaces).not.toHaveBeenCalled()
  })

  it.each([
    ["Workspace"],
    ["Both"]
  ])("requires a selected workspace before saving to %s", async (destinationLabel) => {
    const user = userEvent.setup()

    render(<WebClipperPanel draft={createDraft()} onCancel={vi.fn()} />)

    await user.click(screen.getByRole("radio", { name: destinationLabel }))
    await user.click(screen.getByRole("button", { name: "Save clip" }))

    expect(
      screen.getByText("Choose a workspace before saving to Workspace or Both.")
    ).toBeInTheDocument()
    expect(apiMocks.saveWebClip).not.toHaveBeenCalled()
  })

  it("loads workspace choices and submits the selected workspace id", async () => {
    const user = userEvent.setup()
    apiMocks.listWorkspaces.mockResolvedValueOnce({
      items: [
        {
          id: "workspace-alpha",
          name: "Research Workspace",
          archived: false,
          deleted: false
        }
      ],
      total: 1
    })

    render(<WebClipperPanel draft={createDraft()} onCancel={vi.fn()} />)

    await user.click(screen.getByRole("radio", { name: "Workspace" }))
    await waitFor(() => {
      expect(apiMocks.listWorkspaces).toHaveBeenCalledTimes(1)
    })
    const workspaceSelect = await screen.findByRole("combobox", { name: "Workspace" })
    await user.selectOptions(workspaceSelect, "workspace-alpha")
    await user.click(screen.getByRole("button", { name: "Save clip" }))

    await waitFor(() => {
      expect(apiMocks.saveWebClip).toHaveBeenCalledTimes(1)
    })

    expect(apiMocks.saveWebClip).toHaveBeenCalledWith(
      expect.objectContaining({
        destination_mode: "workspace",
        workspace: { workspace_id: "workspace-alpha" }
      })
    )
  })

  it.each([
    ["saved", "Clip saved"],
    ["saved_with_warnings", "Clip saved with warnings"],
    ["partially_saved", "Clip partially saved"]
  ] as const)(
    "renders the %s banner after save",
    async (status, bannerTitle) => {
      const user = userEvent.setup()
      apiMocks.saveWebClip.mockResolvedValueOnce({
        clip_id: "clip-123",
        note_id: "note-123",
        note: { id: "note-123", title: "Example Story", version: 1 },
        workspace_placement: status === "partially_saved"
          ? null
          : { workspace_id: "workspace-alpha", workspace_note_id: 42, source_note_id: "note-123" },
        attachments: [],
        status,
        warnings: ["Attachment upload skipped"],
        workspace_placement_saved: status !== "partially_saved",
        workspace_placement_count: status === "partially_saved" ? 0 : 1
      })

      render(<WebClipperPanel draft={createDraft()} onCancel={vi.fn()} />)

      await user.click(screen.getByRole("button", { name: "Save clip" }))

      const banner = await screen.findByRole("status", { name: "Clip save result" })
      expect(banner).toHaveTextContent(bannerTitle)
      expect(screen.getByText("Attachment upload skipped")).toBeInTheDocument()
    }
  )

  it.each([
    ["Note", "#/notes"],
    ["Both", "#/notes"],
    ["Workspace", "#/research-workspace"]
  ])(
    "save and open routes %s clips to %s",
    async (destinationLabel, expectedPath) => {
      const user = userEvent.setup()
      apiMocks.saveWebClip.mockResolvedValueOnce({
        clip_id: "clip-123",
        note_id: "note-123",
        note: { id: "note-123", title: "Example Story", version: 1 },
        workspace_placement: destinationLabel === "Workspace"
          ? {
              workspace_id: "workspace-alpha",
              workspace_note_id: 42,
              source_note_id: "note-123"
            }
          : null,
        attachments: [],
        status: "saved",
        warnings: [],
        workspace_placement_saved: destinationLabel === "Workspace",
        workspace_placement_count: destinationLabel === "Workspace" ? 1 : 0
      })

      render(<WebClipperPanel draft={createDraft()} onCancel={vi.fn()} />)

      if (destinationLabel !== "Note") {
        await chooseWorkspaceDestination(
          user,
          destinationLabel as "Workspace" | "Both"
        )
      }

      await user.click(screen.getByRole("button", { name: "Save and open" }))

      await waitFor(() => {
        expect(apiMocks.saveWebClip).toHaveBeenCalledTimes(1)
      })

      expect(openTabMock).toHaveBeenCalledWith(
        expect.objectContaining({
          url: expect.stringContaining(expectedPath)
        })
      )
    }
  )

  it("does not open a destination when save-and-open returns a failed outcome", async () => {
    const user = userEvent.setup()
    apiMocks.saveWebClip.mockResolvedValueOnce({
      clip_id: "clip-123",
      note_id: "clip-123",
      note: null,
      workspace_placement: null,
      attachments: [],
      status: "failed",
      warnings: [],
      workspace_placement_saved: false,
      workspace_placement_count: 0
    })

    render(<WebClipperPanel draft={createDraft()} onCancel={vi.fn()} />)

    await user.click(screen.getByRole("button", { name: "Save and open" }))

    expect(await screen.findByText("Clip save failed")).toBeInTheDocument()
    expect(openTabMock).not.toHaveBeenCalled()
  })

  it("announces submission errors and keeps action buttons usable in narrow sidepanels", async () => {
    const user = userEvent.setup()
    apiMocks.saveWebClip.mockRejectedValueOnce(new Error("Server unavailable"))

    render(<WebClipperPanel draft={createDraft()} onCancel={vi.fn()} />)

    await user.click(screen.getByRole("button", { name: "Save clip" }))

    const alert = await screen.findByRole("alert", { name: "Clip save error" })
    expect(alert).toHaveTextContent("Server unavailable")
    expect(screen.getByTestId("web-clipper-actions").className).toContain("flex-wrap")
  })

  it("save and open falls back to notes when only the canonical note was created", async () => {
    const user = userEvent.setup()
    apiMocks.saveWebClip.mockResolvedValueOnce({
      clip_id: "clip-123",
      note_id: "note-123",
      note: { id: "note-123", title: "Example Story", version: 1 },
      workspace_placement: null,
      attachments: [],
      status: "partially_saved",
      warnings: ["Workspace placement failed."],
      workspace_placement_saved: false,
      workspace_placement_count: 0
    })

    render(<WebClipperPanel draft={createDraft()} onCancel={vi.fn()} />)

    await chooseWorkspaceDestination(user)
    await user.click(screen.getByRole("button", { name: "Save and open" }))

    await waitFor(() => {
      expect(apiMocks.saveWebClip).toHaveBeenCalledTimes(1)
    })

    expect(openTabMock).toHaveBeenCalledWith(
      expect.objectContaining({
        url: expect.stringContaining("#/notes")
      })
    )
  })

  it("clears a previous success banner before showing a later validation error", async () => {
    const user = userEvent.setup()

    render(<WebClipperPanel draft={createDraft()} onCancel={vi.fn()} />)

    await user.click(screen.getByRole("button", { name: "Save clip" }))

    expect(await screen.findByText("Clip saved")).toBeInTheDocument()

    await user.click(screen.getByRole("radio", { name: "Workspace" }))
    await user.click(screen.getByRole("button", { name: "Save clip" }))

    expect(
      screen.getByText("Choose a workspace before saving to Workspace or Both.")
    ).toBeInTheDocument()
    expect(screen.queryByText("Clip saved")).not.toBeInTheDocument()
  })

  it("blocks invalid folder ids before submitting the save request", async () => {
    const user = userEvent.setup()
    apiMocks.listNoteFolders.mockResolvedValueOnce({ items: [], count: 0 })

    render(<WebClipperPanel draft={createDraft()} onCancel={vi.fn()} />)

    await user.type(await screen.findByLabelText("Folder ID"), "0")
    await user.click(screen.getByRole("button", { name: "Save clip" }))

    expect(
      screen.getByText("Choose a positive whole-number folder ID or leave it blank.")
    ).toBeInTheDocument()
    expect(apiMocks.saveWebClip).not.toHaveBeenCalled()
  })

  it("adds the durable captured tag while preserving user-entered tags", async () => {
    const user = userEvent.setup()

    render(<WebClipperPanel draft={createDraft()} onCancel={vi.fn()} />)

    await user.type(screen.getByLabelText("Tags"), "research, source")
    await user.click(screen.getByRole("button", { name: "Save clip" }))

    await waitFor(() => {
      expect(apiMocks.saveWebClip).toHaveBeenCalledTimes(1)
    })

    expect(apiMocks.saveWebClip).toHaveBeenCalledWith(
      expect.objectContaining({
        note: expect.objectContaining({
          keywords: expect.arrayContaining(["research", "source", "captured"])
        })
      })
    )
  })

  it("includes screenshot attachments when the draft carries a captured image", async () => {
    const user = userEvent.setup()

    render(<WebClipperPanel draft={createScreenshotDraft()} onCancel={vi.fn()} />)

    await user.click(screen.getByRole("button", { name: "Save clip" }))

    await waitFor(() => {
      expect(apiMocks.saveWebClip).toHaveBeenCalledTimes(1)
    })

    expect(apiMocks.saveWebClip).toHaveBeenCalledWith(
      expect.objectContaining({
        clip_id: "clip-shot-123",
        attachments: [
          expect.objectContaining({
            slot: "page-screenshot",
            file_name: "page-screenshot.png",
            media_type: "image/png",
            content_base64: "QUJDRA==",
            source_url: "https://example.com/screenshot"
          })
        ]
      })
    )
  })

  it("clears the pending clip draft after a successful save", async () => {
    const user = userEvent.setup()
    const draft = createDraft()
    writePendingClipDraft(draft)

    render(<WebClipperPanel draft={draft} onCancel={vi.fn()} />)

    await user.click(screen.getByRole("button", { name: "Save clip" }))

    await waitFor(() => {
      expect(apiMocks.saveWebClip).toHaveBeenCalledTimes(1)
    })
    expect(readPendingClipDraft()).toBeNull()
  })

  it("preserves the richer full extract when saving article clips", async () => {
    const user = userEvent.setup()

    render(<WebClipperPanel draft={createRichExtractDraft()} onCancel={vi.fn()} />)

    await user.click(screen.getByRole("button", { name: "Save clip" }))

    await waitFor(() => {
      expect(apiMocks.saveWebClip).toHaveBeenCalledTimes(1)
    })

    expect(apiMocks.saveWebClip).toHaveBeenCalledWith(
      expect.objectContaining({
        clip_id: "clip-rich-123",
        content: expect.objectContaining({
          visible_body: "Visible article summary",
          full_extract: "Full article body with more detail"
        })
      })
    )
  })

  it("marks note saves as captured while preserving user tags and provenance", async () => {
    const user = userEvent.setup()

    render(<WebClipperPanel draft={createDraft()} onCancel={vi.fn()} />)

    await user.type(screen.getByLabelText("Tags"), "research, planning")
    await user.click(screen.getByRole("button", { name: "Save clip" }))

    await waitFor(() => {
      expect(apiMocks.saveWebClip).toHaveBeenCalledTimes(1)
    })

    expect(apiMocks.saveWebClip).toHaveBeenCalledWith(
      expect.objectContaining({
        source_url: "https://example.com/story",
        capture_metadata: expect.objectContaining({
          requested_type: "article",
          actual_type: "article"
        }),
        note: expect.objectContaining({
          keywords: ["research", "planning", "captured"]
        })
      })
    )
  })

  it("runs requested enrichments after save and surfaces completion and conflict states", async () => {
    const user = userEvent.setup()
    const ocrCompletion = createDeferred<Record<string, unknown>>()
    const vlmCompletion = createDeferred<Record<string, unknown>>()

    apiMocks.createChatCompletion
      .mockResolvedValueOnce({
        json: vi.fn().mockReturnValue(ocrCompletion.promise)
      })
      .mockResolvedValueOnce({
        json: vi.fn().mockReturnValue(vlmCompletion.promise)
      })
    apiMocks.persistWebClipEnrichment
      .mockResolvedValueOnce({
        clip_id: "clip-123",
        enrichment_type: "ocr",
        status: "complete",
        source_note_version: 2,
        inline_applied: true,
        inline_summary: "Captured text summary.",
        conflict_reason: null,
        warnings: []
      })
      .mockResolvedValueOnce({
        clip_id: "clip-123",
        enrichment_type: "vlm",
        status: "complete",
        source_note_version: 2,
        inline_applied: false,
        inline_summary: "Visual summary.",
        conflict_reason: "source_note_version_mismatch",
        warnings: []
      })

    render(<WebClipperPanel draft={createScreenshotDraft()} onCancel={vi.fn()} />)

    await user.click(screen.getByLabelText("Run OCR"))
    await user.click(screen.getByLabelText("Run visual analysis"))
    await user.click(screen.getByRole("button", { name: "Save clip" }))

    expect(await screen.findByText("OCR pending")).toBeInTheDocument()
    expect(screen.getByText("Visual analysis pending")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Save clip" })).toBeEnabled()

    ocrCompletion.resolve({
      choices: [
        {
          message: {
            content: JSON.stringify({
              inline_summary: "Captured text summary.",
              structured_payload: { transcript: "OCR raw text" }
            })
          }
        }
      ]
    })
    vlmCompletion.resolve({
      choices: [
        {
          message: {
            content: JSON.stringify({
              inline_summary: "Visual summary.",
              structured_payload: {
                description: "Interface summary",
                notable_elements: ["Header", "Button"]
              }
            })
          }
        }
      ]
    })

    await waitFor(() => {
      expect(apiMocks.createChatCompletion).toHaveBeenCalledTimes(2)
      expect(apiMocks.persistWebClipEnrichment).toHaveBeenCalledTimes(2)
    })

    expect(screen.getByText("OCR complete")).toBeInTheDocument()
    expect(screen.getByText("Visual analysis needs refresh")).toBeInTheDocument()
  })

  it("analyze now queues a chat handoff with clip context after save", async () => {
    const user = userEvent.setup()

    render(<WebClipperPanel draft={createScreenshotDraft()} onCancel={vi.fn()} />)

    await user.click(screen.getByRole("button", { name: "Analyze now" }))

    await waitFor(() => {
      expect(apiMocks.saveWebClip).toHaveBeenCalledTimes(1)
    })

    const stored = window.sessionStorage.getItem("tldw:web-clipper:pendingAnalyze")
    expect(stored).not.toBeNull()
    expect(JSON.parse(String(stored))).toMatchObject({
      clipId: "clip-123",
      pageUrl: "https://example.com/screenshot",
      pageTitle: "Screenshot Story",
      image: "data:image/png;base64,QUJDRA==",
      requestOverrides: {
        chatMode: "vision"
      }
    })
    expect(navigateMock).toHaveBeenCalledWith("/chat")
  })

  it("ignores late enrichment results after the panel switches to a different clip", async () => {
    const user = userEvent.setup()
    const ocrCompletion = createDeferred<Record<string, unknown>>()

    apiMocks.createChatCompletion.mockResolvedValueOnce({
      json: vi.fn().mockReturnValue(ocrCompletion.promise)
    })
    apiMocks.persistWebClipEnrichment.mockResolvedValueOnce({
      clip_id: "clip-shot-123",
      enrichment_type: "ocr",
      status: "complete",
      source_note_version: 2,
      inline_applied: true,
      inline_summary: "Captured text summary.",
      conflict_reason: null,
      warnings: []
    })

    const { rerender } = render(
      <WebClipperPanel draft={createScreenshotDraft()} onCancel={vi.fn()} />
    )

    await user.click(screen.getByLabelText("Run OCR"))
    await user.click(screen.getByRole("button", { name: "Save clip" }))

    expect(await screen.findByText("OCR pending")).toBeInTheDocument()

    rerender(<WebClipperPanel draft={createDraft()} onCancel={vi.fn()} />)

    expect(screen.queryByText("OCR pending")).not.toBeInTheDocument()

    ocrCompletion.resolve({
      choices: [
        {
          message: {
            content: JSON.stringify({
              inline_summary: "Captured text summary.",
              structured_payload: {
                transcript: "OCR raw text"
              }
            })
          }
        }
      ]
    })

    await waitFor(() => {
      expect(apiMocks.persistWebClipEnrichment).toHaveBeenCalledTimes(1)
    })

    expect(screen.queryByText("OCR complete")).not.toBeInTheDocument()
    expect(screen.getByLabelText("Title")).toHaveValue("Example Story")
  })

  it("ignores late save completions after the panel switches to a different clip", async () => {
    const user = userEvent.setup()
    const saveCompletion = createDeferred<Record<string, unknown>>()

    apiMocks.saveWebClip.mockReturnValueOnce(saveCompletion.promise)

    const { rerender } = render(
      <WebClipperPanel draft={createScreenshotDraft()} onCancel={vi.fn()} />
    )

    await user.click(screen.getByRole("button", { name: "Analyze now" }))

    rerender(<WebClipperPanel draft={createDraft()} onCancel={vi.fn()} />)

    saveCompletion.resolve({
      clip_id: "clip-shot-123",
      note_id: "note-shot-123",
      note: { id: "note-shot-123", title: "Screenshot Story", version: 1 },
      workspace_placement: null,
      attachments: [],
      status: "saved",
      warnings: [],
      workspace_placement_saved: false,
      workspace_placement_count: 0
    })

    await waitFor(() => {
      expect(apiMocks.saveWebClip).toHaveBeenCalledTimes(1)
    })

    expect(screen.queryByText("Clip saved")).not.toBeInTheDocument()
    expect(window.sessionStorage.getItem("tldw:web-clipper:pendingAnalyze")).toBeNull()
    expect(navigateMock).not.toHaveBeenCalled()
    expect(screen.getByLabelText("Title")).toHaveValue("Example Story")
  })

  it("does not let a hidden invalid folder id block workspace-only saves", async () => {
    const user = userEvent.setup()
    apiMocks.listNoteFolders.mockResolvedValueOnce({ items: [], count: 0 })

    render(<WebClipperPanel draft={createDraft()} onCancel={vi.fn()} />)

    await user.type(await screen.findByLabelText("Folder ID"), "0")
    await chooseWorkspaceDestination(user)
    await user.click(screen.getByRole("button", { name: "Save clip" }))

    await waitFor(() => {
      expect(apiMocks.saveWebClip).toHaveBeenCalledTimes(1)
    })
    expect(
      screen.queryByText("Choose a positive whole-number folder ID or leave it blank.")
    ).not.toBeInTheDocument()
  })

  it("invokes the cancel action without attempting a save", async () => {
    const user = userEvent.setup()
    const onCancel = vi.fn()

    render(<WebClipperPanel draft={createDraft()} onCancel={onCancel} />)

    await user.click(screen.getByRole("button", { name: "Cancel" }))

    expect(onCancel).toHaveBeenCalledTimes(1)
    expect(apiMocks.saveWebClip).not.toHaveBeenCalled()
  })
})
