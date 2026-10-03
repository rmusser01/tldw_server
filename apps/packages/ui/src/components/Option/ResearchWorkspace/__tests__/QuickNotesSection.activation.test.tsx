import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { useWorkspaceStore } from "@/store/workspace"
import { hydrateWorkspaceFromServer } from "@/store/workspace-api"
import { serverWorkspacePayload } from "@/store/__tests__/workspace-activation.fixtures"
import { QuickNotesSection } from "../StudioPane/QuickNotesSection"
import { restoreMigratedResearchWorkspace } from "../workspace-server-restore"

// Unit regression: real notes/store behavior, external requests and messages doubled.
const boundary = vi.hoisted(() => ({ request: vi.fn(), keywords: vi.fn(), resolve: vi.fn() }))
vi.mock("@/services/background-proxy", () => ({ bgRequest: boundary.request }))
vi.mock("@/services/note-keywords", () => ({ getNoteKeywords: boundary.keywords }))
vi.mock("@/services/service-prompts", () => ({
  resolveServicePromptScope: boundary.resolve,
  loadServicePromptSnapshot: async () => ({
    scopeKey: "owner-a",
    requestScope: { config: { serverUrl: "https://workspace.test", authMode: "multi-user" }, userId: "alice" },
    scopeSignal: new AbortController().signal,
    scopeInvalidatedSignal: new AbortController().signal,
    release: () => {},
  }),
}))
vi.mock("@/components/Common/MarkdownPreview", () => ({ MarkdownPreview: ({ content }: { content: string }) => <div>{content}</div> }))
vi.mock("react-i18next", () => ({ useTranslation: () => ({ t: (_key: string, fallback: string) => fallback }) }))
vi.mock("antd", async () => ({
  ...await vi.importActual<typeof import("antd")>("antd"),
  message: { useMessage: () => [{ success: vi.fn(), error: vi.fn(), warning: vi.fn(), open: vi.fn(), destroy: vi.fn() }, null] }
}))

const install = async () => {
  const payload = await hydrateWorkspaceFromServer("server-research", { fetch: async () => serverWorkspacePayload() })
  useWorkspaceStore.getState().installServerWorkspace(payload, { scopeKey: "owner-a", expectedWorkspaceId: useWorkspaceStore.getState().workspaceId })
}

describe("canonical QuickNotes adapter (unit boundary doubles)", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
    useWorkspaceStore.getState().reset()
    useWorkspaceStore.setState({ storeHydrated: true, savedWorkspaces: [], workspaceSnapshots: {} })
    boundary.request.mockResolvedValue([])
    boundary.keywords.mockResolvedValue([])
    boundary.resolve.mockResolvedValue({ scopeKey: "owner-a" })
  })

  it("reads staged full canonical notes without using colliding legacy IDs", async () => {
    await install()
    render(<QuickNotesSection />)
    const note = await screen.findByRole("button", { name: "Canonical Note" })
    fireEvent.click(note)
    expect(useWorkspaceStore.getState().currentNote).toMatchObject({
      id: 7, content: "Canonical body", keywords: ["canonical"],
      serverWorkspaceId: "server-research", serverScopeKey: "owner-a"
    })
    expect(boundary.request).not.toHaveBeenCalled()
    expect(boundary.keywords).not.toHaveBeenCalled()
    expect(screen.getByRole("button", { name: /Update/ })).toBeDisabled()
  })

  it("keeps migrated canonical notes view-only and never dispatches colliding legacy note IDs", async () => {
    const retained = serverWorkspacePayload()
    localStorage.setItem("tldw:research-workspace:migration:tombstone:migrated", JSON.stringify({
      serverWorkspaceId: "server-research", migrationId: "migrated", contentRetained: false,
      serverScopeKey: "owner-a", deletedAt: "2026-10-03T00:00:00Z",
    }))
    boundary.request.mockImplementation(async ({ path }: { path: string }) => {
      if (path.endsWith("/context")) return {
        workspace_id: retained.id, workspace: retained.metadata,
        sources: { items: retained.sources }, partial_errors: [],
      }
      if (path.endsWith("/artifacts")) return retained.artifacts
      if (path.endsWith("/notes")) return retained.notes
      throw new Error(`Unexpected legacy request ${path}`)
    })
    const origin = useWorkspaceStore.getState().workspaceId
    await restoreMigratedResearchWorkspace({
      signal: new AbortController().signal,
      apply: (workspace, scopeKey) => useWorkspaceStore.getState().installServerWorkspace(
        workspace, { scopeKey, expectedWorkspaceId: origin },
      ),
    })
    render(<QuickNotesSection />)
    fireEvent.click(await screen.findByRole("button", { name: "Canonical Note" }))
    expect(useWorkspaceStore.getState().currentNote).toMatchObject({
      id: 7, content: "Canonical body", serverWorkspaceId: "server-research", serverScopeKey: "owner-a",
    })
    expect(screen.getByRole("button", { name: /Update/ })).toBeDisabled()
    expect(screen.getByRole("textbox", { name: "Note content" })).toHaveAttribute("readonly")
    expect(boundary.request.mock.calls.map(([request]) => request.path)).toEqual([
      "/api/v1/workspaces/server-research/context", "/api/v1/workspaces/server-research/artifacts", "/api/v1/workspaces/server-research/notes",
    ])
    expect(boundary.keywords).not.toHaveBeenCalled()
  })

  it("does not strip a canonical keyword merely because it matches a legacy workspace tag", async () => {
    await install()
    const state = useWorkspaceStore.getState()
    useWorkspaceStore.setState({ serverWorkspace: { ...state.serverWorkspace,
      notes: state.serverWorkspace.notes.map(note => ({ ...note, keywords_json: JSON.stringify([state.workspaceTag]) })) } })
    render(<QuickNotesSection />)
    fireEvent.click(await screen.findByRole("button", { name: "Canonical Note" }))
    expect(useWorkspaceStore.getState().currentNote.keywords).toEqual([state.workspaceTag])
  })

  it("keeps canonical title, content and keywords view-only without offering tag removal", async () => {
    await install()
    render(<QuickNotesSection />)
    fireEvent.click(await screen.findByRole("button", { name: "Canonical Note" }))
    expect(screen.getByRole("textbox", { name: "Note title" })).toHaveAttribute("readonly")
    expect(screen.getByRole("textbox", { name: "Note content" })).toHaveAttribute("readonly")
    expect(screen.getByRole("combobox", { name: "Note keywords" })).toBeDisabled()
    expect(screen.queryByRole("img", { name: "close" })).toBeNull()
    expect(screen.getByRole("button", { name: "Edit" })).toBeDisabled()
    expect(screen.getByRole("button", { name: "Clear current note" })).toBeDisabled()
    expect(screen.getByText("View only")).toBeVisible()
    fireEvent.click(screen.getByRole("button", { name: "Preview" }))
    expect(screen.getByTestId("quick-notes-markdown-preview")).toHaveTextContent("Canonical body")
    expect(screen.getByRole("button", { name: "Download .md" })).toBeEnabled()
    expect(boundary.request).not.toHaveBeenCalled()
  })

  it.each(["workspace-aba", "draft-edit", "unmount", "account-aba"])("does not commit a legacy note load after %s", async change => {
    useWorkspaceStore.getState().initializeWorkspace("Legacy")
    const origin = useWorkspaceStore.getState().workspaceId
    let resolve!: (value: unknown) => void
    const pending = new Promise(yes => { resolve = yes })
    boundary.request.mockImplementation(async (request: { path: string }) => {
      if (request.path === "/api/v1/notes/7") return pending
      return [{ id: 7, title: "Legacy note" }]
    })
    const view = render(<QuickNotesSection />)
    fireEvent.click(await screen.findByRole("button", { name: "Legacy note" }))
    await waitFor(() => expect(boundary.request).toHaveBeenCalledWith(expect.objectContaining({ path: "/api/v1/notes/7" })))
    act(() => {
      if (change === "workspace-aba") {
        useWorkspaceStore.getState().createNewWorkspace("Another")
        useWorkspaceStore.getState().switchWorkspace(origin)
      }
      if (change === "draft-edit") useWorkspaceStore.getState().updateNoteContent("New local draft")
      if (change === "unmount") view.unmount()
      if (change === "account-aba") window.dispatchEvent(new Event("tldw:auth-credentials-changed"))
    })
    await act(async () => resolve({ id: 7, title: "Late remote", content: "Late response", version: 1 }))
    expect(useWorkspaceStore.getState().currentNote.content).not.toBe("Late response")
  })

  it("ignores an outgoing legacy notes list that settles after canonical activation", async () => {
    useWorkspaceStore.getState().initializeWorkspace("Legacy")
    let resolve!: (value: unknown) => void
    boundary.request.mockReturnValue(new Promise(yes => { resolve = yes }))
    render(<QuickNotesSection />)
    await waitFor(() => expect(boundary.request).toHaveBeenCalled())
    await act(async () => { await install() })
    await screen.findByRole("button", { name: "Canonical Note" })
    await act(async () => resolve([{ id: 7, title: "Late legacy list" }]))
    expect(screen.queryByRole("button", { name: "Late legacy list" })).toBeNull()
    expect(screen.getByRole("button", { name: "Canonical Note" })).toBeVisible()
  })

  it("retains a draft edited while an ordinary save is in flight", async () => {
    useWorkspaceStore.getState().initializeWorkspace("Legacy")
    useWorkspaceStore.getState().setCurrentNote({ id: 7, title: "Draft", content: "First", keywords: [], version: 1, isDirty: true })
    let resolve!: (value: unknown) => void
    const pending = new Promise(yes => { resolve = yes })
    boundary.request.mockImplementation(async (request: { method: string }) => request.method === "PUT" ? pending : [])
    render(<QuickNotesSection />)
    fireEvent.click(screen.getByRole("button", { name: /Update/ }))
    await waitFor(() => expect(boundary.request).toHaveBeenCalledWith(expect.objectContaining({ method: "PUT" })))
    act(() => { useWorkspaceStore.getState().updateNoteContent("New unsaved edit") })
    await act(async () => resolve({ id: 7, title: "Draft", content: "First", version: 2 }))
    expect(useWorkspaceStore.getState().currentNote).toMatchObject({ content: "New unsaved edit", isDirty: true })
  })
})
