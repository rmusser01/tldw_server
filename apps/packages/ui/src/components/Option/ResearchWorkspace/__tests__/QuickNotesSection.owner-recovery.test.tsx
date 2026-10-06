import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { ConfigProvider } from "antd"
import { MemoryRouter } from "react-router-dom"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { useWorkspaceStore } from "@/store/workspace"
import { hydrateWorkspaceFromServer } from "@/store/workspace-api"
import { serverWorkspacePayload } from "@/store/__tests__/workspace-activation.fixtures"
import { retainKnowledgeNoteProvenance } from "@/utils/knowledge-note-provenance"
import { QuickNotesSection } from "../StudioPane/QuickNotesSection"
import { ActivatedLocalWorkspace } from "../ResearchWorkspaceRouteGate"

const boundary = vi.hoisted(() => ({
  request: vi.fn(), resolve: vi.fn(),
  changed: new Set<(invalidated: boolean) => void>(),
  scope: { scopeKey: "owner-a", config: { serverUrl: "https://workspace.test", authMode: "multi-user" }, userId: "alice" },
  getWorkspace: vi.fn(), getWorkspaceSources: vi.fn(), getWorkspaceArtifacts: vi.fn(), getWorkspaceNotes: vi.fn()
}))
vi.mock("@/services/background-proxy", () => ({ bgRequest: boundary.request }))
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: boundary }))
vi.mock("@/services/service-prompts", () => ({
  resolveServicePromptScope: boundary.resolve,
  loadServicePromptSnapshot: vi.fn()
}))
vi.mock("@/services/chat-account-boundary", () => ({
  watchChatAccountChanges: (callback: (invalidated: boolean) => void) => {
    boundary.changed.add(callback)
    return () => { boundary.changed.delete(callback) }
  }
}))
vi.mock("@/services/note-keywords", async original => ({
  ...await original<typeof import("@/services/note-keywords")>(), getNoteKeywords: vi.fn(async () => [])
}))
vi.mock("@/components/Common/MarkdownPreview", () => ({ MarkdownPreview: ({ content }: { content: string }) => <div>{content}</div> }))
vi.mock("react-i18next", () => ({ useTranslation: () => ({ t: (_key: string, fallback: string) => fallback }) }))
vi.mock("antd", async () => ({
  ...await vi.importActual<typeof import("antd")>("antd"),
  message: { useMessage: () => [{ success: vi.fn(), error: vi.fn(), warning: vi.fn(), open: vi.fn(), destroy: vi.fn() }, null] }
}))

const uuid = "12345678-1234-4123-8123-123456789abc"
const content = retainKnowledgeNoteProvenance("Alice's private note", {
  origin: "knowledge_qa", research: { workspace_id: "server-research", import_id: "knowledge-import", sources: [] }
})

const deferred = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(yes => { resolve = yes })
  return { promise, resolve }
}

describe("UUID Notes owner and canonical reopen boundaries", () => {
  beforeEach(async () => {
    vi.clearAllMocks()
    localStorage.clear()
    boundary.changed.clear()
    boundary.scope = { scopeKey: "owner-a", config: { serverUrl: "https://workspace.test", authMode: "multi-user" }, userId: "alice" }
    boundary.resolve.mockImplementation(async () => boundary.scope)
    useWorkspaceStore.getState().reset()
    useWorkspaceStore.setState({ storeHydrated: true, savedWorkspaces: [], workspaceSnapshots: {} })
    const server = serverWorkspacePayload()
    boundary.getWorkspace.mockResolvedValue(server.metadata)
    boundary.getWorkspaceSources.mockResolvedValue(server.sources)
    boundary.getWorkspaceArtifacts.mockResolvedValue(server.artifacts)
    boundary.getWorkspaceNotes.mockResolvedValue(server.notes)
    boundary.request.mockImplementation(async ({ path }: { path: string }) =>
      path.startsWith("/api/v1/notes/search/")
        ? { notes: [{ id: uuid, title: "Alice title", content, version: 4, keywords: ["private", "workspace:server-research"] }] }
        : []
    )
    const local = await hydrateWorkspaceFromServer(server.id, { fetch: async () => server })
    local.currentNote = { id: uuid, title: "Alice title", content, keywords: ["private"], version: 4, isDirty: false,
      serverWorkspaceId: server.id, serverScopeKey: "owner-a" }
    expect(useWorkspaceStore.getState().installServerWorkspace(local, { scopeKey: "owner-a", expectedWorkspaceId: "" })).toBe(true)
  })

  it("hides cached canonical content and export when its verified owner is invalidated", async () => {
    render(<ConfigProvider theme={{ token: { motion: false } }}><QuickNotesSection /></ConfigProvider>)
    await waitFor(() => expect(screen.getByRole("textbox", { name: "Note content" })).toHaveValue("Alice's private note"))
    fireEvent.click(screen.getByRole("button", { name: "Preview" }))
    expect(screen.getByTestId("quick-notes-markdown-preview")).toHaveTextContent("Alice's private note")
    fireEvent.click(screen.getByRole("button", { name: "Load note" }))
    const dialog = await screen.findByRole("dialog")
    await waitFor(() => {
      expect(dialog).toBeVisible()
      expect(within(dialog).getByText("Load Note")).toBeVisible()
    })
    act(() => {
      boundary.scope = { ...boundary.scope, scopeKey: "owner-b", userId: "bob" }
      for (const changed of boundary.changed) changed(true)
    })
    expect(screen.queryByText("Alice's private note")).toBeNull()
    expect(screen.queryByDisplayValue("Alice title")).toBeNull()
    expect(screen.queryByText("private")).toBeNull()
    expect(screen.queryByRole("dialog")).toBeNull()
    expect(screen.getByRole("button", { name: "Download .md" })).toBeDisabled()
    expect(useWorkspaceStore.getState().currentNote.content).toBe(content)
  })

  it.each(["pending", "failed", "foreign-owner", "foreign-note", "foreign-workspace", "missing-workspace"])(
    "does not expose cached notes before qualified ownership (%s)", async state => {
      if (state === "pending") boundary.resolve.mockReturnValue(new Promise(() => {}))
      if (state === "failed") boundary.resolve.mockRejectedValue(new Error("Owner unavailable"))
      if (state === "foreign-owner") boundary.scope = { ...boundary.scope, scopeKey: "owner-b", userId: "bob" }
      if (state === "foreign-note" || state === "foreign-workspace") {
        const note = useWorkspaceStore.getState().currentNote
        useWorkspaceStore.setState({ currentNote: { ...note,
          ...(state === "foreign-note" ? { serverScopeKey: "owner-b" } : { serverWorkspaceId: "other" }) } })
      }
      if (state === "missing-workspace") useWorkspaceStore.setState({ serverWorkspace: null })
      const before = useWorkspaceStore.getState().currentNote
      render(<QuickNotesSection />)
      await act(async () => { await Promise.resolve() })
      expect(screen.queryByRole("textbox", { name: "Note content" })).toBeNull()
      expect(screen.queryByDisplayValue("Alice title")).toBeNull()
      expect(screen.queryByText("private")).toBeNull()
      expect(screen.getByRole("button", { name: "Download .md" })).toBeDisabled()
      expect(useWorkspaceStore.getState().currentNote).toBe(before)
    }
  )

  it("keeps an unbound local draft editable and retained", () => {
    useWorkspaceStore.getState().reset()
    useWorkspaceStore.getState().initializeWorkspace("Local")
    useWorkspaceStore.getState().updateNoteContent("Unsent local draft")
    render(<QuickNotesSection />)
    expect(screen.getByRole("textbox", { name: "Note content" })).toHaveValue("Unsent local draft")
    expect(screen.getByRole("textbox", { name: "Note content" })).not.toHaveAttribute("readonly")
    expect(screen.getByRole("button", { name: "Download .md" })).toBeEnabled()
  })

  it("rehydrates UUID Notes on actual canonical activation without adding provenance membership", async () => {
    render(<MemoryRouter><ActivatedLocalWorkspace workspaceId="server-research" webClip={false}>
      <div data-testid="activated-note" />
    </ActivatedLocalWorkspace></MemoryRouter>)
    expect(await screen.findByTestId("activated-note")).toBeVisible()
    expect(useWorkspaceStore.getState().currentNote).toMatchObject({
      id: uuid, content, version: 4, serverWorkspaceId: "server-research", serverScopeKey: "owner-a", isDirty: false
    })
    expect(useWorkspaceStore.getState().sources.map(source => source.id)).toEqual(serverWorkspacePayload().sources.map(source => source.id))
    expect(boundary.request).toHaveBeenCalledWith(expect.objectContaining({
      path: expect.stringContaining("/api/v1/notes/search/"), method: "GET", abortSignal: expect.any(AbortSignal),
      servicePromptConfig: expect.objectContaining({ expectedUserId: "alice" }), headers: { "X-TLDW-Expected-User-ID": "alice" }
    }))
  })

  it("fails closed on a UUID Notes read failure without replacing cached content", async () => {
    boundary.request.mockRejectedValue(new Error("Notes unavailable"))
    const before = useWorkspaceStore.getState()
    render(<MemoryRouter><ActivatedLocalWorkspace workspaceId="server-research" webClip={false}>
      <div data-testid="activated-note" />
    </ActivatedLocalWorkspace></MemoryRouter>)
    expect(await screen.findByRole("alert")).toBeVisible()
    expect(screen.queryByTestId("activated-note")).toBeNull()
    expect(useWorkspaceStore.getState().currentNote).toBe(before.currentNote)
    expect(useWorkspaceStore.getState().sources).toBe(before.sources)
  })

  it("does not revive a deleted clean UUID note from the cached snapshot", async () => {
    boundary.request.mockResolvedValue({ notes: [] })
    render(<MemoryRouter><ActivatedLocalWorkspace workspaceId="server-research" webClip={false}>
      <div data-testid="activated-note" />
    </ActivatedLocalWorkspace></MemoryRouter>)
    expect(await screen.findByTestId("activated-note")).toBeVisible()
    expect(useWorkspaceStore.getState().currentNote.id).toBeUndefined()
    expect(useWorkspaceStore.getState().currentNote.content).toBe("")
  })

  it.each(["unmount", "account-aba", "workspace-aba"])("rejects a late UUID read after %s", async change => {
    const read = deferred<{ notes: Array<{ id: string; title: string; content: string; version: number }> }>()
    boundary.request.mockReturnValue(read.promise)
    const original = useWorkspaceStore.getState().currentNote
    const view = render(<MemoryRouter><ActivatedLocalWorkspace workspaceId="server-research" webClip={false}>
      <div data-testid="activated-note" />
    </ActivatedLocalWorkspace></MemoryRouter>)
    await waitFor(() => expect(boundary.request).toHaveBeenCalled())
    const signal = boundary.request.mock.calls[0][0].abortSignal
    act(() => {
      if (change === "unmount") view.unmount()
      if (change === "account-aba") for (const changed of boundary.changed) changed(true)
      if (change === "workspace-aba") {
        useWorkspaceStore.getState().createNewWorkspace("Intervening")
        useWorkspaceStore.getState().switchWorkspace("server-research")
      }
    })
    await act(async () => read.resolve({ notes: [{ id: uuid, title: "Late title", content, version: 9 }] }))
    expect(signal.aborted).toBe(true)
    expect(useWorkspaceStore.getState().currentNote).toEqual(original)
    expect(screen.queryByTestId("activated-note")).toBeNull()
    if (change !== "unmount") expect(screen.getByRole("alert")).toBeVisible()
  })
})
