import {
  readKnowledgeNoteProvenance,
  stripKnowledgeNoteProvenance,
} from "../knowledge-note-provenance"
import {
  act,
  fireEvent,
  render,
  renderHook,
  screen,
  waitFor,
} from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { ResearchWorkspace } from "@/components/Option/ResearchWorkspace"
import { useWorkspaceStore } from "@/store/workspace"
import {
  buildKnowledgeQaWorkspacePrefill,
  consumeResearchWorkspacePrefill,
  queueResearchWorkspacePrefill,
} from "../research-workspace-prefill"
import { useResearchWorkspacePrefill } from "../use-research-workspace-prefill"

const mocks = vi.hoisted(() => ({
  owner: "alice",
  multiUser: false,
  values: new Map<string, unknown>(),
  upload: vi.fn(),
  details: vi.fn(),
  request: vi.fn(),
  persistent: true,
  writeError: false,
  readGate: null as Promise<void> | null,
}))
vi.mock("@/services/chat-surface-scope", () => ({
  buildChatSurfaceScopeKeyFromConfig: (
    config: any,
    options?: { userId?: string | number | null },
  ) =>
    config.serverUrl +
    (config.authMode === "multi-user"
      ? `:${options?.userId ?? (config.accessToken ? "user-a" : "anonymous")}`
      : ""),
}))
vi.mock("@/utils/media-navigation-scope", () => ({
  deriveScopedUserId: () => "user:single",
}))
vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getConfig: async () => ({
      serverUrl: mocks.owner,
      authMode: mocks.multiUser ? "multi-user" : "single-user",
      accessToken: mocks.multiUser ? "synthetic-user-a" : undefined,
    }),
    uploadMedia: (...args: unknown[]) => mocks.upload(...args),
    getMediaDetails: (...args: unknown[]) => mocks.details(...args),
  },
}))
vi.mock("@/services/service-prompts", () => ({
  loadServicePromptSnapshot: async () => ({
    scopeKey: mocks.owner,
    requestScope: {
      config: { serverUrl: mocks.owner, authMode: "single-user" },
      userId: null,
    },
    scopeSignal: new AbortController().signal,
    scopeInvalidatedSignal: new AbortController().signal,
    release: () => {},
  }),
  resolveServicePromptScope: async () => ({
    config: {
      serverUrl: mocks.owner,
      authMode: mocks.multiUser ? "multi-user" : "single-user",
    },
    userId: mocks.multiUser ? "user-a" : null,
  }),
}))
vi.mock("@/utils/safe-storage", () => ({
  safeStorageSerde: { deserializer: (value: unknown) => value },
  createSafeStorage: () => ({
    get hasPersistentBackend() {
      return mocks.persistent
    },
    get: async (key: string) => {
      await mocks.readGate
      return structuredClone(mocks.values.get(key))
    },
    set: async (key: string, value: unknown) => {
      if (mocks.writeError) throw new Error("storage full")
      mocks.values.set(key, structuredClone(value))
    },
  }),
}))
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, options?: any) =>
      typeof options === "string" ? options : options?.defaultValue || key,
  }),
}))
vi.mock("@/hooks/useMediaQuery", () => ({ useMobile: () => false }))
vi.mock("@/hooks/useFeatureFlags", () => ({
  FEATURE_FLAGS: {},
  useFeatureFlag: () => [true],
}))
vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.request(...args),
}))
vi.mock("@/components/Option/ResearchWorkspace/WorkspaceHeader", () => ({
  WorkspaceHeader: () => null,
}))
vi.mock("@/components/Option/ResearchWorkspace/SourcesPane", () => ({
  SourcesPane: () => null,
}))
vi.mock("@/components/Option/ResearchWorkspace/ChatPane", () => ({
  ChatPane: () => null,
}))
vi.mock("@/components/Option/ResearchWorkspace/StudioPane", () => ({
  StudioPane: () => null,
}))
vi.mock("@/components/Option/ResearchWorkspace/WorkspaceStatusBar", () => ({
  WorkspaceStatusBar: () => null,
}))
const payload = () =>
  buildKnowledgeQaWorkspacePrefill({
    threadId: "thread-1",
    query: "Evidence?",
    answer: "Unsupported draft",
    citations: [1, 2, 3],
    answerTrustState: "uncited_degraded_answer",
    scope: { sources: ["notes"] },
    results: [
      {
        id: "chunk-1",
        content: "First excerpt",
        metadata: {
          source_type: "notes",
          note_id: "note-uuid",
          title: "Field note",
        },
      },
      {
        id: "chunk-2",
        content: "Second excerpt",
        metadata: {
          source_type: "notes",
          note_id: "note-uuid",
          title: "Field note",
        },
      },
      {
        id: "web-1",
        content: "Web excerpt",
        metadata: { source_type: "web", url: "https://example.com/evidence" },
      },
      { id: "7", content: "Native excerpt", metadata: { source_type: "pdf" } },
    ],
  })
beforeEach(() => {
  mocks.persistent = true
  mocks.multiUser = false
  mocks.owner = "alice"
  mocks.values.clear()
  mocks.upload.mockReset()
  mocks.details.mockReset()
  mocks.request.mockReset().mockResolvedValue({})
  mocks.writeError = false
  mocks.readGate = null
  localStorage.clear()
  useWorkspaceStore.setState({
    workspaceId: "workspace-a",
    workspaceTag: "workspace:research",
    sources: [],
    selectedSourceIds: [],
    selectedSourceFolderIds: [],
    sourceFolders: [],
    sourceFolderMemberships: [],
    savedWorkspaces: [],
    workspaceSnapshots: {},
    currentNote: { title: "", content: "", keywords: [], isDirty: false },
    storeHydrated: true,
  })
})
const selectUnrelatedFolder = (sourceId: string) => {
  const state = useWorkspaceStore.getState()
  const folder = state.createSourceFolder("Unrelated folder")
  state.assignSourceToFolders(sourceId, [folder.id])
  state.toggleSourceFolderSelection(folder.id)
  return folder.id
}
describe("mounted Knowledge research import", () => {
  it.each([false, true, "folder"])(
    "selects a note-only snapshot through actual workspace readiness polling unless selection changes (%s)",
    async (manuallySelected) => {
      useWorkspaceStore
        .getState()
        .addSources([{ mediaId: 50, title: "Unrelated", type: "text" }])
      const unrelatedId = useWorkspaceStore.getState().sources[0].id
      useWorkspaceStore.getState().setSelectedSourceIds([unrelatedId])
      const handoff = payload()
      handoff.sources = handoff.sources.filter(
        (source) => source.sourceType === "notes",
      )
      await queueResearchWorkspacePrefill(handoff, "alice")
      mocks.upload.mockResolvedValue({ media_id: 101 })
      let finish!: (value: unknown) => void
      mocks.details.mockImplementation(
        () =>
          new Promise((resolve) => {
            finish = resolve
          }),
      )
      const view = render(<ResearchWorkspace />)
      await waitFor(() =>
        expect(mocks.details).toHaveBeenCalledWith(
          101,
          expect.objectContaining({ include_content: true }),
        ),
      )
      await waitFor(async () =>
        expect(await consumeResearchWorkspacePrefill("alice")).toBeNull(),
      )
      expect(useWorkspaceStore.getState().selectedSourceIds).toEqual([])
      if (manuallySelected)
        await act(async () => {
          if (manuallySelected === "folder") selectUnrelatedFolder(unrelatedId)
          else useWorkspaceStore.getState().setSelectedSourceIds([unrelatedId])
        })
      await act(async () => {
        finish({
          content: { text: "First excerpt" },
          vector_processing_status: "completed",
        })
      })
      await waitFor(() =>
        expect(
          useWorkspaceStore
            .getState()
            .sources.find((source) => source.mediaId === 101)?.status,
        ).toBe("ready"),
      )
      expect(
        useWorkspaceStore.getState().getEffectiveSelectedMediaIds(),
      ).toEqual(manuallySelected ? [50] : [101])
      view.unmount()
    },
  )
  it.each(["none", "direct", "folder"])(
    "resumes durable readiness after full workspace remount, preserving %s choices",
    async (choice) => {
      useWorkspaceStore
        .getState()
        .addSources([{ mediaId: 50, title: "Unrelated", type: "text" }])
      const unrelatedId = useWorkspaceStore.getState().sources[0].id
      const handoff = payload()
      handoff.sources = handoff.sources.filter(
        (source) => source.sourceType === "notes",
      )
      await queueResearchWorkspacePrefill(handoff, "alice")
      mocks.upload.mockResolvedValue({ media_id: 101 })
      mocks.details.mockImplementation(() => new Promise(() => {}))
      const first = render(<ResearchWorkspace />)
      await waitFor(async () =>
        expect(await consumeResearchWorkspacePrefill("alice")).toBeNull(),
      )
      await waitFor(() => expect(mocks.details).toHaveBeenCalled())
      const draft = useWorkspaceStore.getState().currentNote.content
      first.unmount()
      await useWorkspaceStore.persist.rehydrate()
      if (choice === "direct")
        useWorkspaceStore.getState().setSelectedSourceIds([unrelatedId])
      if (choice === "folder") selectUnrelatedFolder(unrelatedId)
      let finish!: (value: unknown) => void
      mocks.details.mockImplementation(
        () =>
          new Promise((resolve) => {
            finish = resolve
          }),
      )
      const second = render(<ResearchWorkspace />)
      await waitFor(() => expect(finish).toBeTypeOf("function"))
      await act(async () => {
        finish({
          content: { text: "First excerpt" },
          vector_processing_status: "completed",
        })
      })
      await waitFor(() =>
        expect(
          useWorkspaceStore
            .getState()
            .sources.find((source) => source.mediaId === 101)?.status,
        ).toBe("ready"),
      )
      expect(
        useWorkspaceStore.getState().getEffectiveSelectedMediaIds(),
      ).toEqual(choice === "none" ? [101] : [50])
      expect(mocks.upload).toHaveBeenCalledTimes(1)
      expect(useWorkspaceStore.getState().currentNote.content).toBe(draft)
      second.unmount()
    },
  )
  it("does not replay completed cached imports after same-hook workspace switching", async () => {
    useWorkspaceStore
      .getState()
      .addSources([{ mediaId: 50, title: "Unrelated", type: "text" }])
    const unrelatedId = useWorkspaceStore.getState().sources[0].id
    const handoff = payload()
    handoff.sources = handoff.sources.filter((source) => source.mediaId === 7)
    await queueResearchWorkspacePrefill(handoff, "alice")
    const view = renderHook(() =>
      useResearchWorkspacePrefill(
        useWorkspaceStore((state) => state.workspaceId),
        true,
      ),
    )
    await waitFor(async () =>
      expect(await consumeResearchWorkspacePrefill("alice")).toBeNull(),
    )
    const draft = useWorkspaceStore.getState().currentNote.content
    act(() => {
      const state = useWorkspaceStore.getState()
      state.removeSource(
        state.sources.find((source) => source.mediaId === 7)!.id,
      )
      state.setSelectedSourceIds([unrelatedId])
      state.createNewWorkspace("Workspace B")
    })
    await act(async () => {
      await Promise.resolve()
    })
    act(() => useWorkspaceStore.getState().switchWorkspace("workspace-a"))
    await act(async () => {
      await Promise.resolve()
    })
    expect(
      useWorkspaceStore.getState().sources.map((source) => source.mediaId),
    ).toEqual([50])
    expect(useWorkspaceStore.getState().getEffectiveSelectedMediaIds()).toEqual(
      [50],
    )
    expect(useWorkspaceStore.getState().currentNote.content).toBe(draft)
    view.unmount()
  })
  it("replaces unrelated active sources with the transferred set through partial retry", async () => {
    const state = useWorkspaceStore.getState()
    state.addSources([
      { mediaId: 50, title: "Unrelated library source", type: "text" },
    ])
    const unrelatedId = useWorkspaceStore.getState().sources[0].id
    state.setSelectedSourceIds([unrelatedId])
    selectUnrelatedFolder(unrelatedId)
    state.captureToCurrentNote({ content: "Existing draft", mode: "append" })
    const selectedMediaIds = () =>
      useWorkspaceStore.getState().getEffectiveSelectedMediaIds()
    await queueResearchWorkspacePrefill(payload(), "alice")
    let finish!: (value: unknown) => void
    mocks.upload
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            finish = resolve
          }),
      )
      .mockRejectedValueOnce(new Error("offline"))
      .mockResolvedValueOnce({ media_id: 102 })
    const view = renderHook(() =>
      useResearchWorkspacePrefill("workspace-a", true),
    )
    await waitFor(() => expect(mocks.upload).toHaveBeenCalledTimes(1))
    expect(selectedMediaIds()).toEqual([])
    await act(async () => {
      finish({ media_id: 101 })
    })
    await waitFor(() => expect(view.result.current.importing).toBe(false))
    expect(view.result.current.failed).toBe(1)
    // Processing snapshots stay attached but cannot enter active chat scope yet.
    expect(selectedMediaIds()).toEqual([7])
    expect(
      useWorkspaceStore.getState().sources.map((source) => source.mediaId),
    ).toEqual([50, 101, 7])
    const retainedDraft = useWorkspaceStore.getState().currentNote.content
    expect(retainedDraft).toContain("Existing draft")
    expect(retainedDraft).toContain("Unsupported draft")
    const snapshotId = useWorkspaceStore
      .getState()
      .sources.find((source) => source.mediaId === 101)!.id
    useWorkspaceStore.getState().setSourceStatusById(snapshotId, "ready")
    await act(async () => {
      await view.result.current.retry()
    })
    await waitFor(() => expect(view.result.current.failed).toBe(0))
    await waitFor(() => expect(view.result.current.importing).toBe(false))
    expect(selectedMediaIds()).toEqual([101, 7])
    expect(
      useWorkspaceStore.getState().sources.map((source) => source.mediaId),
    ).toEqual([50, 101, 7, 102])
    expect(useWorkspaceStore.getState().currentNote.content).toBe(retainedDraft)
    expect(mocks.upload).toHaveBeenCalledTimes(3)
    view.unmount()
    useWorkspaceStore.getState().setSelectedSourceIds([unrelatedId])
    const reopened = renderHook(() =>
      useResearchWorkspacePrefill("workspace-a", true),
    )
    await act(async () => {
      await Promise.resolve()
    })
    expect(selectedMediaIds()).toEqual([50])
    reopened.unmount()
  })
  it("leaves no unrelated active source when every transferred snapshot fails", async () => {
    useWorkspaceStore
      .getState()
      .addSources([{ mediaId: 50, title: "Unrelated", type: "text" }])
    useWorkspaceStore
      .getState()
      .setSelectedSourceIds([useWorkspaceStore.getState().sources[0].id])
    selectUnrelatedFolder(useWorkspaceStore.getState().sources[0].id)
    const handoff = payload()
    handoff.sources = handoff.sources.filter((source) => source.mediaId == null)
    await queueResearchWorkspacePrefill(handoff, "alice")
    mocks.upload.mockRejectedValue(new Error("offline"))
    const view = renderHook(() =>
      useResearchWorkspacePrefill("workspace-a", true),
    )
    await waitFor(() => expect(view.result.current.failed).toBe(2))
    expect(useWorkspaceStore.getState().getEffectiveSelectedMediaIds()).toEqual(
      [],
    )
    expect(
      useWorkspaceStore.getState().sources.map((source) => source.mediaId),
    ).toEqual([50])
    view.unmount()
  })
  it("retains two note chunks, retries only unfinished snapshots, and reopens source provenance", async () => {
    await queueResearchWorkspacePrefill(payload(), "alice")
    mocks.upload
      .mockResolvedValueOnce({ results: [{ media_id: 101 }] })
      .mockRejectedValueOnce(new Error("offline"))
      .mockResolvedValueOnce({ results: [{ db_id: 102 }] })
    const view = renderHook(() =>
      useResearchWorkspacePrefill("workspace-a", true),
    )
    await waitFor(() => expect(view.result.current.failed).toBe(1))
    expect(useWorkspaceStore.getState().sources.map((s) => s.mediaId)).toEqual([
      101, 7,
    ])
    const firstNote = useWorkspaceStore.getState().currentNote.content
    expect(firstNote).toContain("First excerpt")
    expect(firstNote).toContain("Second excerpt")
    expect(firstNote).toContain("uncited_degraded_answer")
    expect(
      (await consumeResearchWorkspacePrefill("alice"))?.sources[0]
        .snapshotMediaId,
    ).toBe(101)
    await act(async () => {
      await view.result.current.retry()
    })
    expect(useWorkspaceStore.getState().sources.map((s) => s.mediaId)).toEqual([
      101, 7, 102,
    ])
    expect(useWorkspaceStore.getState().currentNote.content).toBe(firstNote)
    expect(mocks.upload.mock.calls.map(([file]) => file.name)).toEqual([
      "Field note - retrieved excerpts.txt",
      "Source 3 - retrieved excerpts.txt",
      "Source 3 - retrieved excerpts.txt",
    ])
    const source = useWorkspaceStore.getState().sources[0]
    expect(source.knowledgeQaEvidence?.sources.map((s) => s.excerpt)).toEqual([
      "First excerpt",
      "Second excerpt",
    ])
    useWorkspaceStore.getState().loadNote({
      id: "e3b16146-9e38-42e0-bd15-549e60bd31a3",
      title: "Canonical draft",
      content: firstNote,
    })
    view.unmount()
    const persisted = new Map(
      Array.from(
        { length: localStorage.length },
        (_, index) => localStorage.key(index)!,
      ).map((key) => [key, localStorage.getItem(key)!]),
    )
    expect(
      [...persisted.keys()].some(
        (key) => key.includes(":workspace:") && key.endsWith(":snapshot"),
      ),
    ).toBe(true)
    useWorkspaceStore.setState({
      sources: [],
      currentNote: { title: "", content: "", keywords: [], isDirty: false },
    })
    for (const [key, value] of persisted) localStorage.setItem(key, value)
    await useWorkspaceStore.persist.rehydrate()
    expect(useWorkspaceStore.getState().currentNote.id).toBe(
      "e3b16146-9e38-42e0-bd15-549e60bd31a3",
    )
    expect(useWorkspaceStore.getState().sources).toHaveLength(3)
    expect(
      useWorkspaceStore.getState().sources[0].knowledgeQaEvidence?.sources[0]
        .originalId,
    ).toBe("note-uuid")
    expect(
      useWorkspaceStore.getState().sources[0].knowledgeQaEvidence?.trustState,
    ).toBe("uncited_degraded_answer")
    expect(useWorkspaceStore.getState().currentNote.content).toBe(firstNote)
    // Saving/clearing the useful draft must not append it a second time on return.
    useWorkspaceStore.getState().clearCurrentNote()
    const reopened = renderHook(() =>
      useResearchWorkspacePrefill("workspace-a", true),
    )
    await act(async () => {
      await Promise.resolve()
    })
    expect(useWorkspaceStore.getState().currentNote.content).toBe("")
    reopened.unmount()
  })
  it.each(["owner", "workspace"])(
    "never publishes late uploads after %s changes",
    async (change) => {
      await queueResearchWorkspacePrefill(payload(), "alice")
      let finish!: (value: unknown) => void
      mocks.upload.mockImplementation(
        () =>
          new Promise((resolve) => {
            finish = resolve
          }),
      )
      const view = renderHook(
        ({ workspace }) => useResearchWorkspacePrefill(workspace, true),
        { initialProps: { workspace: "workspace-a" } },
      )
      await waitFor(() => expect(mocks.upload).toHaveBeenCalled())
      if (change === "owner")
        act(() => {
          mocks.owner = "bob"
          window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed"))
        })
      else {
        act(() => {
          useWorkspaceStore.setState({ workspaceId: "workspace-b" })
        })
        view.rerender({ workspace: "workspace-b" })
      }
      await act(async () => {
        finish({ results: [{ media_id: 101 }] })
      })
      expect(useWorkspaceStore.getState().sources).toEqual([])
      expect(useWorkspaceStore.getState().currentNote.content).toBe("")
      expect(await consumeResearchWorkspacePrefill("bob")).toBeNull()
      view.unmount()
    },
  )
  it("waits for hydration and rejects an owner change during storage read", async () => {
    await queueResearchWorkspacePrefill(payload(), "alice")
    let finish!: () => void
    mocks.readGate = new Promise((resolve) => {
      finish = resolve
    })
    const view = renderHook(
      ({ hydrated }) => useResearchWorkspacePrefill("workspace-a", hydrated),
      { initialProps: { hydrated: false } },
    )
    expect(mocks.upload).not.toHaveBeenCalled()
    view.rerender({ hydrated: true })
    await act(async () => {
      mocks.owner = "bob"
      window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed"))
      finish()
    })
    expect(useWorkspaceStore.getState().sources).toEqual([])
    expect(useWorkspaceStore.getState().currentNote.content).toBe("")
    view.unmount()
  })
  it("rejects memory-only storage instead of navigating to an empty workspace", async () => {
    mocks.persistent = false
    await expect(
      queueResearchWorkspacePrefill(payload(), "alice"),
    ).rejects.toThrow("persistent storage")
    expect(mocks.values.size).toBe(0)
  })
  it("retains a successful upload ID when a progress write fails", async () => {
    await queueResearchWorkspacePrefill(payload(), "alice")
    mocks.upload
      .mockImplementationOnce(async () => {
        mocks.writeError = true
        return { results: [{ media_id: 101 }] }
      })
      .mockResolvedValue({ media_id: 102 })
    const view = renderHook(() =>
      useResearchWorkspacePrefill("workspace-a", true),
    )
    await waitFor(() =>
      expect(view.result.current.error).toContain("could not be saved"),
    )
    mocks.writeError = false
    await act(async () => {
      await view.result.current.retry()
    })
    await waitFor(() => expect(view.result.current.attached).toBe(3))
    expect(
      useWorkspaceStore.getState().sources.map((source) => source.mediaId),
    ).toEqual([101, 102, 7])
    expect(
      mocks.upload.mock.calls.filter(([file]) =>
        file.name.startsWith("Field note"),
      ),
    ).toHaveLength(1)
    view.unmount()
  })
  it("surfaces storage failures to the sender", async () => {
    mocks.writeError = true
    await expect(
      queueResearchWorkspacePrefill(payload(), "alice"),
    ).rejects.toThrow("storage full")
  })
  it("does not reattach removed sources after a completed import", async () => {
    await queueResearchWorkspacePrefill(payload(), "alice")
    mocks.upload
      .mockResolvedValueOnce({ media_id: 101 })
      .mockResolvedValueOnce({ media_id: 102 })
    const view = renderHook(() =>
      useResearchWorkspacePrefill("workspace-a", true),
    )
    await waitFor(() => expect(view.result.current.attached).toBe(3))
    await waitFor(() => expect(view.result.current.importing).toBe(false))
    view.unmount()
    const source = useWorkspaceStore.getState().sources[0]
    useWorkspaceStore.getState().removeSource(source.id)
    const reopened = renderHook(() =>
      useResearchWorkspacePrefill("workspace-a", true),
    )
    await act(async () => {
      await Promise.resolve()
    })
    expect(
      useWorkspaceStore.getState().sources.map((item) => item.mediaId),
    ).not.toContain(101)
    expect(await consumeResearchWorkspacePrefill("alice")).toBeNull()
    reopened.unmount()
  })
  it("includes the verified principal when matching credential-free request scopes", async () => {
    mocks.multiUser = true
    await queueResearchWorkspacePrefill(payload(), "alice:user-a")
    mocks.upload
      .mockResolvedValueOnce({ media_id: 101 })
      .mockResolvedValueOnce({ media_id: 102 })
    const view = renderHook(() =>
      useResearchWorkspacePrefill("workspace-a", true),
    )
    await waitFor(() => expect(view.result.current.attached).toBe(3))
    expect(useWorkspaceStore.getState().sources[0].mediaId).toBe(101)
    view.unmount()
  })
})

it.each([false, true])(
  "retains canonical Quick Notes and matching source evidence after a tombstoned workspace reload (lost response=%s)",
  async (loseResponse) => {
    const { restoreMigratedResearchWorkspace } =
      await import("@/components/Option/ResearchWorkspace/workspace-server-restore")
    localStorage.setItem(
      "tldw:research-workspace:migration:tombstone:workspace-a",
      JSON.stringify({
        legacyWorkspaceId: "workspace-a",
        serverWorkspaceId: "workspace-a",
        migrationId: "migration-a",
        serverScopeKey: "alice",
        contentRetained: false,
        deletedAt: "2026-10-03T00:00:00Z",
      }),
    )
    let canonical: any = null
    let droppedResponse = false
    mocks.request.mockImplementation(async ({ path, method, body }: any) => {
      if (path === "/api/v1/notes/" && method === "POST") {
        canonical = {
          id: body.id,
          title: body.title,
          content: body.content,
          keywords: body.keywords.map((keyword: string) => ({ keyword })),
          conversation_id: body.conversation_id,
          version: 1,
        }
        if (loseResponse && !droppedResponse) {
          droppedResponse = true
          throw new Error("response lost")
        }
        return canonical
      }
      if (path.startsWith("/api/v1/notes/search/"))
        return { notes: canonical ? [canonical] : [] }
      if (path.startsWith("/api/v1/notes/")) {
        if (!canonical)
          throw Object.assign(new Error("missing"), { status: 404 })
        return canonical
      }
      if (path.endsWith("/context"))
        return {
          workspace_id: "workspace-a",
          workspace: {
            id: "workspace-a",
            name: "Research",
            created_at: "2026-10-01T00:00:00Z",
            version: 1,
          },
          sources: {
            items: [
              {
                id: "saved-source",
                workspace_id: "workspace-a",
                media_id: 101,
                title: "Field note — retrieved excerpts",
                source_type: "text",
                selected: false,
                state: "queryable",
                added_at: "2026-10-01T00:00:00Z",
              },
            ],
          },
          partial_errors: [],
        }
      return []
    })
    const transfer = payload()
    transfer.sources = transfer.sources.slice(0, 2)
    mocks.upload.mockResolvedValue({ id: 101 })
    await queueResearchWorkspacePrefill(transfer)
    const receiver = renderHook(() =>
      useResearchWorkspacePrefill("workspace-a", true),
    )
    await waitFor(() =>
      expect(useWorkspaceStore.getState().currentNote.content).toContain(
        "Import reference:",
      ),
    )
    await waitFor(() => expect(receiver.result.current.importing).toBe(false))
    if (loseResponse) {
      expect(receiver.result.current.error).toContain("could not be saved")
      await act(async () => {
        await receiver.result.current.retry()
      })
      await waitFor(() => expect(receiver.result.current.error).toBeNull())
      await waitFor(() => expect(receiver.result.current.importing).toBe(false))
    }
    expect(receiver.result.current.error).toBeNull()
    expect(canonical.id).toMatch(/^[0-9a-f-]{36}$/)
    expect(useWorkspaceStore.getState().currentNote.id).toBe(canonical.id)
    await act(async () => {
      useWorkspaceStore.setState({ selectedSourceFolderIds: ["manual-folder"] })
    })
    await waitFor(() =>
      expect(
        [...mocks.values.values()].some(
          (value: any) => value.selectionIntent === null,
        ),
      ).toBe(true),
    )
    receiver.unmount()
    useWorkspaceStore.setState({
      workspaceId: null,
      selectedSourceFolderIds: [],
      sources: [],
      currentNote: { title: "", content: "", keywords: [], isDirty: false },
      workspaceSnapshots: {},
      selectedSourceIds: [],
    })
    await restoreMigratedResearchWorkspace({
      signal: new AbortController().signal,
      apply: useWorkspaceStore.getState().restoreServerWorkspace,
    })
    const reopened = renderHook(() =>
      useResearchWorkspacePrefill("workspace-a", true),
    )
    await waitFor(() =>
      expect(useWorkspaceStore.getState().currentNote.content).toContain(
        "Unsupported draft",
      ),
    )
    const state = useWorkspaceStore.getState()
    expect(state.currentNote.id).toBe(canonical.id)
    expect(state.sources.map((source) => source.mediaId)).toEqual([101])
    expect(state.sources[0].knowledgeQaEvidence).toMatchObject({
      trustState: "uncited_degraded_answer",
      sources: [
        { originalId: "note-uuid", excerpt: "First excerpt" },
        { originalId: "note-uuid", excerpt: "Second excerpt" },
      ],
    })
    expect(state.selectedSourceIds).toEqual([])
    expect(mocks.upload).toHaveBeenCalledTimes(1)
    expect(
      mocks.request.mock.calls.filter(([request]) => request.method === "POST"),
    ).toHaveLength(1)
    reopened.unmount()
  },
)

it.each(["edit", "clear", "owner", "workspace"])(
  "fences canonical save acknowledgment after %s",
  async (change) => {
    localStorage.setItem(
      "tldw:research-workspace:migration:tombstone:workspace-a",
      JSON.stringify({
        legacyWorkspaceId: "workspace-a",
        serverWorkspaceId: "workspace-a",
        migrationId: "migration-a",
        contentRetained: false,
      }),
    )
    let finish!: () => void
    let canonical: any = null
    mocks.request.mockImplementation(async ({ method, body }: any) => {
      if (method === "POST") {
        canonical = {
          id: body.id,
          title: body.title,
          content: body.content,
          keywords: body.keywords.map((keyword: string) => ({ keyword })),
          version: 1,
        }
        await new Promise<void>((resolve) => {
          finish = resolve
        })
        return canonical
      }
      if (!canonical) throw Object.assign(new Error("missing"), { status: 404 })
      return canonical
    })
    const transfer = payload()
    transfer.sources = transfer.sources.slice(3)
    await queueResearchWorkspacePrefill(transfer)
    const receiver = renderHook(() =>
      useResearchWorkspacePrefill("workspace-a", true),
    )
    await waitFor(() => expect(canonical).not.toBeNull())
    await act(async () => {
      if (change === "clear") useWorkspaceStore.getState().clearCurrentNote()
      else useWorkspaceStore.getState().updateNoteContent("Later unsaved body")
      if (change === "owner") {
        mocks.owner = "bob"
        window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed"))
      }
      if (change === "workspace")
        useWorkspaceStore.setState({ workspaceId: "workspace-b" })
      finish()
    })
    if (change === "edit") {
      await waitFor(() =>
        expect(useWorkspaceStore.getState().currentNote.id).toBe(canonical.id),
      )
      expect(useWorkspaceStore.getState().currentNote.content).toContain(
        "Later unsaved body",
      )
      expect(useWorkspaceStore.getState().currentNote.content).toContain(
        "tldw-knowledge:v1:",
      )
      expect(useWorkspaceStore.getState().currentNote.isDirty).toBe(true)
    } else {
      expect(useWorkspaceStore.getState().currentNote.id).toBeUndefined()
      expect(useWorkspaceStore.getState().currentNote.content).toBe(
        change === "clear" ? "" : "Later unsaved body",
      )
    }
    receiver.unmount()
  },
)

it("retains an existing canonical note's unsaved title and body when appending the handoff", async () => {
  localStorage.setItem(
    "tldw:research-workspace:migration:tombstone:workspace-a",
    JSON.stringify({
      legacyWorkspaceId: "workspace-a",
      serverWorkspaceId: "workspace-a",
      migrationId: "migration-a",
      contentRetained: false,
    }),
  )
  let canonical = {
    id: "e3b16146-9e38-42e0-bd15-549e60bd31a3",
    title: "Server title",
    content: "Server body",
    keywords: [{ keyword: "old" }],
    version: 1,
  }
  useWorkspaceStore.getState().loadNote({ ...canonical, keywords: ["new"] })
  useWorkspaceStore.getState().updateNoteTitle("Unsaved title")
  useWorkspaceStore.getState().updateNoteContent("Unsaved body")
  mocks.request.mockImplementation(async ({ method, body }: any) => {
    if (method === "PUT")
      canonical = {
        ...canonical,
        title: body.title,
        content: body.content,
        keywords: body.keywords.map((keyword: string) => ({ keyword })),
        version: 2,
      }
    return canonical
  })
  const transfer = payload()
  transfer.sources = transfer.sources.slice(3)
  await queueResearchWorkspacePrefill(transfer)
  const receiver = renderHook(() =>
    useResearchWorkspacePrefill("workspace-a", true),
  )
  await waitFor(() =>
    expect(useWorkspaceStore.getState().currentNote.version).toBe(2),
  )
  expect(canonical.title).toBe("Unsaved title")
  expect(canonical.content).toContain("Unsaved body")
  expect(canonical.content).toContain("Unsupported draft")
  expect(canonical.keywords).toContainEqual({ keyword: "new" })
  expect(receiver.result.current.error).toBeNull()
  receiver.unmount()
})


it.each([
  ["lost response", false],
  ["lost response", true],
  ["partial import", false],
  ["partial import", true],
] as const)(
  "preserves dirty edits before canonical %s retry and subsequent edits=%s",
  async (failure, editDuringRetry) => {
    localStorage.setItem(
      "tldw:research-workspace:migration:tombstone:workspace-a",
      JSON.stringify({
        legacyWorkspaceId: "workspace-a",
        serverWorkspaceId: "workspace-a",
        migrationId: "migration-a",
        contentRetained: false,
      }),
    )
    let canonical: any = null
    let finishRetry!: () => void
    mocks.request.mockImplementation(async ({ method, body }: any) => {
      if (method === "POST") {
        canonical = {
          id: body.id,
          title: body.title,
          content: body.content,
          keywords: body.keywords.map((keyword: string) => ({ keyword })),
          version: 1,
        }
        if (failure === "lost response")
          throw new Error("response lost after commit")
      }
      if (method === "PUT") {
        canonical = {
          ...canonical,
          title: body.title,
          content: body.content,
          keywords: body.keywords.map((keyword: string) => ({ keyword })),
          version: 2,
        }
        await new Promise<void>((resolve) => {
          finishRetry = resolve
        })
      }
      if (!canonical) throw Object.assign(new Error("missing"), { status: 404 })
      return canonical
    })
    mocks.upload
      .mockResolvedValueOnce({ id: 101 })
      .mockRejectedValueOnce(new Error("offline"))
      .mockResolvedValueOnce({ id: 102 })
    const transfer = payload()
    transfer.sources = transfer.sources.slice(
      0,
      failure === "partial import" ? 3 : 2,
    )
    await queueResearchWorkspacePrefill(transfer)
    const receiver = renderHook(() =>
      useResearchWorkspacePrefill("workspace-a", true),
    )
    await waitFor(() => expect(canonical).not.toBeNull())
    await waitFor(() => expect(receiver.result.current.importing).toBe(false))
    if (failure === "lost response")
      expect(receiver.result.current.error).toContain("could not be saved")
    else expect(receiver.result.current.failed).toBe(1)
    const canonicalId = canonical.id
    await act(async () => {
      useWorkspaceStore.getState().updateNoteTitle("Title edited before retry")
      useWorkspaceStore
        .getState()
        .updateNoteContent("Body replaced before retry")
      useWorkspaceStore.getState().updateNoteKeywords(["before-retry"])
      await receiver.result.current.retry()
    })
    await waitFor(() => expect(finishRetry).toBeTypeOf("function"))
    await act(async () => {
      if (editDuringRetry) {
        useWorkspaceStore.getState().updateNoteTitle("Later title")
        useWorkspaceStore.getState().updateNoteContent("Later body")
        useWorkspaceStore.getState().updateNoteKeywords(["later"])
      }
      finishRetry()
    })
    await waitFor(() => expect(receiver.result.current.importing).toBe(false))
    expect(receiver.result.current.error).toBeNull()
    expect(canonical.title).toBe("Title edited before retry")
    expect(stripKnowledgeNoteProvenance(canonical.content)).toBe(
      "Body replaced before retry",
    )
    expect(canonical.keywords).toContainEqual({ keyword: "before-retry" })
    const note = useWorkspaceStore.getState().currentNote
    expect(note.id).toBe(canonicalId)
    expect(note.title).toBe(
      editDuringRetry ? "Later title" : "Title edited before retry",
    )
    expect(stripKnowledgeNoteProvenance(note.content)).toBe(
      editDuringRetry ? "Later body" : "Body replaced before retry",
    )
    expect(note.keywords).toEqual([editDuringRetry ? "later" : "before-retry"])
    expect(note.isDirty).toBe(editDuringRetry)
    expect(
      readKnowledgeNoteProvenance(note.content)?.research?.sources.map(
        (source) => source.mediaId,
      ),
    ).toEqual(failure === "partial import" ? [101, 102] : [101])
    expect(mocks.upload).toHaveBeenCalledTimes(
      failure === "partial import" ? 3 : 1,
    )
    expect(
      mocks.request.mock.calls.filter(([request]) => request.method === "POST"),
    ).toHaveLength(1)
    expect(
      useWorkspaceStore.getState().sources.map((source) => source.mediaId),
    ).toEqual(failure === "partial import" ? [101, 102] : [101])
    receiver.unmount()
  },
)

it.each(["unchanged", "edited", "replaced", "cleared"] as const)(
  "acknowledges a restored legacy numeric note only while owned (%s) and saves its canonical UUID",
  async (change) => {
    const { restoreMigratedResearchWorkspace } =
      await import("@/components/Option/ResearchWorkspace/workspace-server-restore")
    const { QuickNotesSection } =
      await import("@/components/Option/ResearchWorkspace/StudioPane/QuickNotesSection")
    localStorage.setItem(
      "tldw:research-workspace:migration:tombstone:workspace-a",
      JSON.stringify({
        legacyWorkspaceId: "workspace-a",
        serverWorkspaceId: "workspace-a",
        migrationId: "migration-a",
        serverScopeKey: "alice",
        contentRetained: false,
        deletedAt: "2026-10-03T00:00:00Z",
      }),
    )
    let canonical: any = null
    let finishCreate!: () => void
    mocks.request.mockImplementation(async ({ path, method, body }: any) => {
      if (path === "/api/v1/workspaces/workspace-a/context")
        return {
          workspace_id: "workspace-a",
          workspace: {
            id: "workspace-a",
            name: "Research",
            created_at: "2026-10-01T00:00:00Z",
            version: 1,
          },
          sources: { items: [] },
          partial_errors: [],
        }
      if (path === "/api/v1/workspaces/workspace-a/notes")
        return [
          {
            id: 7,
            workspace_id: "workspace-a",
            title: "Legacy note",
            content: "Legacy body",
            keywords_json: '["legacy"]',
            version: 3,
            created_at: "2026-10-01T00:00:00Z",
            last_modified: "2026-10-01T00:00:00Z",
          },
        ]
      if (path === "/api/v1/notes/" && method === "POST") {
        canonical = {
          id: body.id,
          title: body.title,
          content: body.content,
          keywords: body.keywords.map((keyword: string) => ({ keyword })),
          version: 1,
        }
        await new Promise<void>((resolve) => {
          finishCreate = resolve
        })
        return canonical
      }
      if (path.startsWith("/api/v1/notes/search/")) return { notes: [] }
      if (path.startsWith("/api/v1/notes/keywords/")) return []
      if (path.startsWith("/api/v1/notes/")) {
        if (
          !canonical ||
          path.split("?")[0] !== `/api/v1/notes/${canonical.id}`
        )
          throw Object.assign(new Error("missing"), { status: 404 })
        if (method === "PUT")
          canonical = {
            ...canonical,
            title: body.title,
            content: body.content,
            keywords: body.keywords.map((keyword: string) => ({ keyword })),
            version: canonical.version + 1,
          }
        return canonical
      }
      return []
    })
    await restoreMigratedResearchWorkspace({
      signal: new AbortController().signal,
      apply: useWorkspaceStore.getState().restoreServerWorkspace,
    })
    expect(useWorkspaceStore.getState().currentNote).toMatchObject({
      id: 7,
      title: "Legacy note",
      content: "Legacy body",
      version: 3,
    })
    const transfer = payload()
    transfer.sources = transfer.sources.slice(0, 2)
    mocks.upload.mockResolvedValue({ id: 101 })
    await queueResearchWorkspacePrefill(transfer)
    const receiver = renderHook(() =>
      useResearchWorkspacePrefill("workspace-a", true),
    )
    await waitFor(() => expect(finishCreate).toBeTypeOf("function"))
    await act(async () => {
      const state = useWorkspaceStore.getState()
      if (change === "edited") state.updateNoteContent("Edited during import")
      if (change === "replaced")
        state.loadNote({
          id: 8,
          title: "Different legacy note",
          content: "Keep replacement",
        })
      if (change === "cleared") state.clearCurrentNote()
      finishCreate()
    })
    await waitFor(() => expect(receiver.result.current.importing).toBe(false))
    expect(receiver.result.current.error).toBeNull()
    const note = useWorkspaceStore.getState().currentNote
    if (change === "replaced" || change === "cleared") {
      expect(note.id).toBe(change === "replaced" ? 8 : undefined)
      expect(note.content).toBe(change === "replaced" ? "Keep replacement" : "")
      receiver.unmount()
      return
    }
    expect(canonical.id).toMatch(/^[0-9a-f-]{36}$/)
    expect(note.id).toBe(canonical.id)
    expect(note.title).toBe("Legacy note")
    expect(note.keywords).toEqual(["legacy"])
    expect(note.version).toBe(1)
    expect(note.isDirty).toBe(change === "edited")
    expect(note.content).toContain(
      change === "edited" ? "Edited during import" : "Legacy body",
    )
    expect(readKnowledgeNoteProvenance(note.content)).toMatchObject({
      origin: "knowledge_qa",
      trust_state: "uncited_degraded_answer",
      research: {
        workspace_id: "workspace-a",
        sources: [
          {
            mediaId: 101,
            evidence: {
              sources: [
                { originalId: "note-uuid", excerpt: "First excerpt" },
                { originalId: "note-uuid", excerpt: "Second excerpt" },
              ],
            },
          },
        ],
      },
    })
    receiver.unmount()
    const editor = render(<QuickNotesSection />)
    fireEvent.change(screen.getByRole("textbox", { name: "Note title" }), {
      target: { value: "Canonical title edited" },
    })
    fireEvent.click(screen.getByRole("button", { name: "Update" }))
    await waitFor(() =>
      expect(useWorkspaceStore.getState().currentNote.isDirty).toBe(false),
    )
    expect(canonical.title).toBe("Canonical title edited")
    expect(canonical.version).toBe(2)
    expect(mocks.request).toHaveBeenCalledWith(
      expect.objectContaining({
        method: "PUT",
        path: `/api/v1/notes/${note.id}?expected_version=1`,
      }),
    )
    expect(readKnowledgeNoteProvenance(canonical.content)).toEqual(
      readKnowledgeNoteProvenance(note.content),
    )
    expect(canonical.keywords).toContainEqual({
      keyword: "workspace:workspace-a",
    })
    expect(
      mocks.request.mock.calls.filter(([request]) => request.method === "POST"),
    ).toHaveLength(1)
    expect(mocks.upload).toHaveBeenCalledTimes(1)
    editor.unmount()
  },
)

it.each([
  [7, false],
  [7, true],
  [8, false],
  [8, true],
  ["6a1ba7ea-7384-413a-90be-62c5e7f56b41", false],
  ["6a1ba7ea-7384-413a-90be-62c5e7f56b41", true],
] as const)(
  "retries an unacknowledged legacy note without adopting replacement %s (remount=%s)",
  async (currentId, remount) => {
    localStorage.setItem(
      "tldw:research-workspace:migration:tombstone:workspace-a",
      JSON.stringify({
        legacyWorkspaceId: "workspace-a",
        serverWorkspaceId: "workspace-a",
        migrationId: "migration-a",
        contentRetained: false,
      }),
    )
    useWorkspaceStore.getState().loadNote({
      id: 7,
      title: "Original legacy note",
      content: "Original legacy body",
    })
    let canonical: any = null
    mocks.request.mockImplementation(async ({ method, body }: any) => {
      if (method === "POST") {
        canonical = { ...body, version: 1 }
        throw new Error("response lost after commit")
      }
      if (method === "PUT") canonical = { ...canonical, ...body, version: 2 }
      if (!canonical) throw Object.assign(new Error("missing"), { status: 404 })
      return canonical
    })
    const transfer = payload()
    transfer.sources = transfer.sources.slice(3)
    await queueResearchWorkspacePrefill(transfer)
    let receiver = renderHook(() =>
      useResearchWorkspacePrefill("workspace-a", true),
    )
    await waitFor(() =>
      expect(receiver.result.current.error).toContain("could not be saved"),
    )
    const originalBody = canonical.content
    await act(async () => {
      const state = useWorkspaceStore.getState()
      if (currentId !== 7)
        state.loadNote({
          id: currentId,
          title: "Replacement note",
          content: "Replacement body",
        })
      state.updateNoteContent("Edits before retry")
      if (!remount) await receiver.result.current.retry()
    })
    if (remount) {
      receiver.unmount()
      receiver = renderHook(() =>
        useResearchWorkspacePrefill("workspace-a", true),
      )
    }
    await waitFor(() => expect(canonical.version).toBe(2))
    await waitFor(() => expect(receiver.result.current.importing).toBe(false))
    expect(receiver.result.current.error).toBeNull()
    const note = useWorkspaceStore.getState().currentNote
    expect(note.id).toBe(currentId === 7 ? canonical.id : currentId)
    expect(stripKnowledgeNoteProvenance(note.content)).toBe(
      "Edits before retry",
    )
    expect(note.isDirty).toBe(currentId !== 7)
    expect(canonical.content).toBe(
      currentId === 7 ? note.content : originalBody,
    )
    expect(
      mocks.request.mock.calls.filter(([request]) => request.method === "POST"),
    ).toHaveLength(1)
    receiver.unmount()
  },
)
