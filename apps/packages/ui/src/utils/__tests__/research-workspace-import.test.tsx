import type { BgRequestInit } from "@/services/background-proxy"
import type { CitationRef, KnowledgeQAContextValue } from "@/components/Option/KnowledgeQA/types"
import type { TldwConfig } from "@/services/tldw/TldwApiClient"
import type { WorkspaceNote } from "@/types/workspace"
import type { KnowledgeNoteProvenance } from "../knowledge-note-provenance"
import {
  readKnowledgeNoteProvenance,
  retainKnowledgeNoteProvenance,
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
import { AnswerPanel } from "@/components/Option/KnowledgeQA/AnswerPanel"
import { MediaKnowledgeActions } from "@/components/Review/MediaKnowledgeActions"
import { DEFAULT_RAG_SETTINGS } from "@/services/rag/unified-rag"
import { ResearchWorkspace } from "@/components/Option/ResearchWorkspace"
import { useWorkspaceStore } from "@/store/workspace"
import {
  buildKnowledgeQaWorkspacePrefill,
  consumeResearchWorkspacePrefill,
  queueResearchWorkspacePrefill,
} from "../research-workspace-prefill"
import { useResearchWorkspacePrefill } from "../use-research-workspace-prefill"

type AnswerPanelStateFixture = Partial<Omit<KnowledgeQAContextValue, "citations">> & {
  citations?: Pick<CitationRef, "index">[]
}
type CanonicalImportNoteFixture<Keyword = string | { keyword: string }> =
  Omit<WorkspaceNote, "id" | "isDirty" | "keywords"> & {
    id: string
    version: number
    keywords?: Keyword[]
    conversation_id?: string | null
  }
type CanonicalNoteWriteFixture = Pick<WorkspaceNote, "title" | "content" | "keywords"> & {
  id?: string
  conversation_id?: string | null
  knowledge_provenance?: KnowledgeNoteProvenance
}
type CanonicalRequestFixture = Omit<BgRequestInit, "method" | "body"> & (
  | { method: "POST"; body: CanonicalNoteWriteFixture & { id: string } }
  | { method: Exclude<BgRequestInit["method"], "POST">; body?: CanonicalNoteWriteFixture }
)
type PrefillCheckpointFixture = NonNullable<Awaited<ReturnType<typeof consumeResearchWorkspacePrefill>>> & {
  confirmedSeedlessWrite?: NonNullable<NonNullable<Awaited<ReturnType<typeof consumeResearchWorkspacePrefill>>>["pendingNoteWrite"]>
}

const mocks = vi.hoisted(() => ({
  owner: "alice",
  answerState: {} as AnswerPanelStateFixture,
  navigate: vi.fn(),
  multiUser: false,
  values: new Map<string, unknown>(),
  upload: vi.fn(),
  details: vi.fn(),
  request: vi.fn(),
  sourceNoteRequest: vi.fn(),
  sourceNoteIds: new Set<string>(),
  persistent: true,
  writeError: false,
  readGate: null as Promise<void> | null,
}))
vi.mock("@/services/chat-surface-scope", () => ({
  buildChatSurfaceScopeKeyFromConfig: (
    config: Pick<TldwConfig, "serverUrl" | "authMode" | "accessToken">,
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
    scopeKey: mocks.multiUser ? `${mocks.owner}:user-a` : mocks.owner,
    clientPrincipalVerified: true,
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
    t: (key: string, options?: string | { defaultValue?: string }) =>
      typeof options === "string" ? options : options?.defaultValue || key,
  }),
}))
vi.mock("@/components/Option/KnowledgeQA/KnowledgeQAProvider", () => ({
  useKnowledgeQA: () => mocks.answerState,
}))
vi.mock("@/hooks/useHomeMilestoneScope", () => ({
  useHomeMilestoneScope: () => mocks.owner,
}))
vi.mock("@/hooks/useAntdMessage", () => ({
  useAntdMessage: () => ({ open: vi.fn(), error: vi.fn() }),
}))
vi.mock("@/services/feedback", () => ({
  getFeedbackSessionId: () => "session-a",
  submitExplicitFeedback: vi.fn(),
}))
vi.mock("@/utils/knowledge-qa-search-metrics", () => ({
  trackKnowledgeQaSearchMetric: async () => {},
}))
vi.mock("react-router-dom", async () => ({
  ...(await vi.importActual("react-router-dom")),
  useNavigate: () => mocks.navigate,
}))

vi.mock("@/hooks/useMediaQuery", () => ({ useMobile: () => false }))
vi.mock("@/hooks/useFeatureFlags", () => ({
  FEATURE_FLAGS: {},
  useFeatureFlag: () => [true],
}))
vi.mock("@/services/background-proxy", () => ({
  bgRequest: (request: BgRequestInit) =>
    request.method === "GET" &&
    mocks.sourceNoteIds.has(decodeURIComponent(request.path.split("/").at(-1)))
      ? mocks.sourceNoteRequest(request)
      : mocks.request(request),
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
  mocks.navigate.mockReset()
  mocks.persistent = true
  mocks.multiUser = false
  mocks.owner = "alice"
  mocks.values.clear()
  mocks.upload.mockReset()
  mocks.details.mockReset()
  mocks.request.mockReset().mockResolvedValue({})
  mocks.sourceNoteIds.clear()
  mocks.sourceNoteIds.add("note-uuid")
  mocks.sourceNoteRequest
    .mockReset()
    .mockImplementation(async (request: BgRequestInit) => ({
      id: decodeURIComponent(request.path.split("/").at(-1)),
      title: "Field note",
      version: 4,
      deleted: false,
      content:
        "First excerpt\nSecond excerpt\nComplete note ending outside retrieved chunks.",
    }))
  mocks.writeError = false
  mocks.readGate = null
  localStorage.clear()
  useWorkspaceStore.getState().reset()
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
      "Field note - full note snapshot (v4).txt",
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
        {
          initialProps: { workspace: "workspace-a" },
        },
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
  "retains canonical Quick Notes and reused source evidence through reconcile and tombstoned reload (lost response=%s)",
  async (loseResponse) => {
    const { restoreMigratedResearchWorkspace } =
      await import("@/components/Option/ResearchWorkspace/workspace-server-restore")
    const { reconcileResearchWorkspaceServerState } =
      await import("@/components/Option/ResearchWorkspace/workspace-server-reconcile")
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
    const workspace = {
      id: "workspace-a",
      name: "Research",
      workspace_profile: "research",
      study_materials_policy: "workspace",
      deleted: false,
      archived: false,
      created_at: "2026-10-01T00:00:00Z",
      version: 1,
    }
    let serverSources = [7, 101, 50].map((mediaId, position) => ({
      id: `server-source-${mediaId}`,
      workspace_id: "workspace-a",
      media_id: mediaId,
      title: `Existing ${mediaId}`,
      source_type: "text",
      url: null,
      position,
      selected: mediaId === 50,
      state: "queryable",
      added_at: "2026-10-01T00:00:00Z",
      version: 1,
    }))
    let canonical: CanonicalImportNoteFixture | null = null
    let droppedResponse = false
    mocks.request.mockImplementation(async (request: CanonicalRequestFixture) => {
      const { path, method, body, headers } = request
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
        if (method === "PUT") {
          if (headers?.["expected-version"] !== String(canonical.version))
            throw Object.assign(new Error("expected-version header required"), {
              status: 422,
            })
          canonical = {
            ...canonical,
            ...body,
            version: canonical.version + 1,
          }
        }
        return canonical
      }
      if (path.endsWith("/context"))
        return {
          workspace_id: "workspace-a",
          workspace,
          sources: { items: serverSources },
          partial_errors: [],
        }
      return []
    })
    const restore = () => {
      const origin = useWorkspaceStore.getState().workspaceId
      return restoreMigratedResearchWorkspace({
        signal: new AbortController().signal,
        apply: (workspace, scopeKey) =>
          useWorkspaceStore.getState().installServerWorkspace(workspace, {
            scopeKey,
            expectedWorkspaceId: origin,
          }),
      })
    }
    useWorkspaceStore.getState().reset()
    useWorkspaceStore.setState({ workspaceId: null, storeHydrated: true })
    await restore()
    const transfer = payload()
    mocks.sourceNoteIds.add("b905bb24-0657-45de-af47-6c8a4d5db498")
    transfer.sources = [
      transfer.sources[3],
      ...transfer.sources.slice(0, 2).map((source) => ({
        ...source,
        originalId: "b905bb24-0657-45de-af47-6c8a4d5db498",
      })),
    ]
    mocks.upload.mockResolvedValue({ id: 101 })
    await queueResearchWorkspacePrefill(transfer)
    const receiver = renderHook(() =>
      useResearchWorkspacePrefill("workspace-a", true),
    )
    await waitFor(() => expect(canonical).not.toBeNull())
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
    const imported = useWorkspaceStore.getState()
    expect(imported.sources.map((source) => source.id)).toEqual([
      "server-source-7",
      "server-source-101",
      "server-source-50",
    ])
    expect(imported.getEffectiveSelectedMediaIds()).toEqual([7, 101])
    expect(imported.sources[0].knowledgeQaEvidence).toMatchObject({
      snapshot: false,
      trustState: "uncited_degraded_answer",
      sources: [{ originalId: "7", excerpt: "Native excerpt" }],
    })
    expect(imported.sources[1].knowledgeQaEvidence).toMatchObject({
      snapshot: true,
      trustState: "uncited_degraded_answer",
      sources: [
        {
          originalId: "b905bb24-0657-45de-af47-6c8a4d5db498",
          excerpt: "First excerpt",
        },
        {
          originalId: "b905bb24-0657-45de-af47-6c8a4d5db498",
          excerpt: "Second excerpt",
        },
      ],
    })
    expect(imported.sources[2].knowledgeQaEvidence).toBeUndefined()
    expect(imported.currentNote.id).toBe(canonical.id)
    expect(
      readKnowledgeNoteProvenance(canonical.content)?.research?.sources.map(
        (source) => source.mediaId,
      ),
    ).toEqual([7, 101])
    await act(async () => {
      imported.setSelectedSourceIds(["server-source-50"])
    })
    const client = {
      upsertWorkspace: vi.fn().mockResolvedValue(workspace),
      getWorkspaceSources: vi.fn(async () => serverSources),
      addWorkspaceSource: vi.fn(),
      updateWorkspaceSourceSelection: vi.fn(
        async (_id: string, selected: string[]) => {
          serverSources = serverSources.map((source) => ({
            ...source,
            selected: selected.includes(source.id),
          }))
        },
      ),
    }
    const synced = await reconcileResearchWorkspaceServerState({
      client,
      workspaceId: "workspace-a",
      workspaceName: "Research",
      sources: useWorkspaceStore.getState().sources,
      selectedSourceIds: useWorkspaceStore.getState().selectedSourceIds,
    })
    expect(synced.errors).toEqual([])
    expect(client.addWorkspaceSource).not.toHaveBeenCalled()
    expect(useWorkspaceStore.getState().sources[1].knowledgeQaEvidence).toEqual(
      imported.sources[1].knowledgeQaEvidence,
    )
    receiver.unmount()
    // A later server deletion must stay deleted even though the canonical note retains its evidence.
    serverSources = serverSources.filter((source) => source.media_id !== 7)
    useWorkspaceStore.getState().reset()
    useWorkspaceStore.setState({ workspaceId: null, storeHydrated: true })
    await restore()
    const reopened = renderHook(() =>
      useResearchWorkspacePrefill("workspace-a", true),
    )
    await waitFor(() =>
      expect(useWorkspaceStore.getState().currentNote.id).toBe(canonical.id),
    )
    const state = useWorkspaceStore.getState()
    expect(state.sources.map((source) => source.mediaId)).toEqual([101, 50])
    expect(state.sources[0].knowledgeQaEvidence).toEqual(
      imported.sources[1].knowledgeQaEvidence,
    )
    expect(state.selectedSourceIds).toEqual(["server-source-50"])
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
        deletedAt: "2026-10-03T00:00:00Z",
      }),
    )
    let finish!: () => void
    let canonical: CanonicalImportNoteFixture | null = null
    mocks.request.mockImplementation(async ({ method, body }: CanonicalRequestFixture) => {
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
      deletedAt: "2026-10-03T00:00:00Z",
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
  mocks.request.mockImplementation(async ({ method, body }: CanonicalRequestFixture) => {
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
        deletedAt: "2026-10-03T00:00:00Z",
      }),
    )
    let canonical: CanonicalImportNoteFixture | null = null
    let finishRetry!: () => void
    mocks.request.mockImplementation(async ({ method, body }: CanonicalRequestFixture) => {
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

it.each(["unchanged", "edited", "replaced", "cleared", "unversioned"] as const)(
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
    let canonical: CanonicalImportNoteFixture | null = null
    let finishCreate!: () => void
    mocks.request.mockImplementation(async (request: CanonicalRequestFixture) => {
      const { path, method, body, headers } = request
      if (path === "/api/v1/workspaces/workspace-a/context")
        return {
          workspace_id: "workspace-a",
          workspace: {
            id: "workspace-a",
            name: "Research",
            workspace_profile: "research",
            study_materials_policy: "workspace",
            deleted: false,
            archived: false,
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
            title: "Workspace-table note",
            content: "Separate workspace-table body",
            keywords_json: '["workspace-table"]',
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
        if (method === "PUT") {
          if (headers?.["expected-version"] !== String(canonical.version))
            throw Object.assign(new Error("expected-version header required"), {
              status: 422,
            })
          canonical = {
            ...canonical,
            title: body.title,
            content: body.content,
            keywords: body.keywords.map((keyword: string) => ({ keyword })),
            version: canonical.version + 1,
          }
        }
        return canonical
      }
      return []
    })
    useWorkspaceStore.getState().reset()
    useWorkspaceStore.setState({ workspaceId: null, storeHydrated: true })
    const origin = useWorkspaceStore.getState().workspaceId
    await restoreMigratedResearchWorkspace({
      signal: new AbortController().signal,
      apply: (workspace, scopeKey) =>
        useWorkspaceStore.getState().installServerWorkspace(workspace, {
          scopeKey,
          expectedWorkspaceId: origin,
        }),
    })
    // Legacy Notes and workspace-table Notes have independent numeric ID namespaces.
    useWorkspaceStore.getState().loadNote({
      id: 7,
      title: "Legacy note",
      content: "Legacy body",
      keywords: ["legacy"],
      version: 3,
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
    const canonicalPreview = render(<QuickNotesSection />)
    expect(await screen.findByRole("button", { name: "Update" })).toBeDisabled()
    canonicalPreview.unmount()
    useWorkspaceStore.getState().createNewWorkspace("Local UUID Notes")
    useWorkspaceStore.getState().setCurrentNote(note)
    if (change === "unversioned")
      useWorkspaceStore
        .getState()
        .setCurrentNote({ ...note, version: undefined })
    let editor!: ReturnType<typeof render>
    await act(async () => {
      editor = render(<QuickNotesSection />)
    })
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
        path: `/api/v1/notes/${note.id}`,
        headers: expect.objectContaining({ "expected-version": "1" }),
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
        deletedAt: "2026-10-03T00:00:00Z",
      }),
    )
    useWorkspaceStore.getState().loadNote({
      id: 7,
      title: "Original legacy note",
      content: "Original legacy body",
    })
    let canonical: CanonicalImportNoteFixture | null = null
    mocks.request.mockImplementation(async ({ method, body, headers }: CanonicalRequestFixture) => {
      if (method === "POST") {
        canonical = { ...body, version: 1 }
        throw new Error("response lost after commit")
      }
      if (method === "PUT") {
        if (headers?.["expected-version"] !== String(canonical.version))
          throw Object.assign(new Error("expected-version header required"), {
            status: 422,
          })
        canonical = { ...canonical, ...body, version: 2 }
      }
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

const canonicalNoteId = "1467b10d-1d40-46eb-9b55-d6c7a413a43c"
const originalNoteId = "b905bb24-0657-45de-af47-6c8a4d5db498"
const previousProvenance: KnowledgeNoteProvenance = {
  origin: "reviewed_sources",
  research: {
    workspace_id: "workspace-a",
    import_id: "previous-review-import",
    sources: [],
  },
}
const markMigrated = () =>
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
const checkpoint = (id: string) =>
  [...mocks.values.values()].find((value: PrefillCheckpointFixture) => value.id === id) as PrefillCheckpointFixture

it.each([false, true])(
  "persists the actual AnswerPanel default scope and current import over old marker=%s through canonical restore",
  async (withOldMarker) => {
    markMigrated()
    let canonical: CanonicalImportNoteFixture<string> | null = withOldMarker
      ? {
          id: canonicalNoteId,
          title: "Review draft",
          content: retainKnowledgeNoteProvenance(
            "Earlier reviewed source body",
            previousProvenance,
          ),
          keywords: ["workspace:workspace-a"],
          version: 1,
        }
      : null
    if (canonical) useWorkspaceStore.getState().loadNote(canonical)
    const sources = [7, 101].map((mediaId, position) => ({
      id: `server-source-${mediaId}`,
      workspace_id: "workspace-a",
      media_id: mediaId,
      title: `Source ${mediaId}`,
      source_type: "text",
      url: null,
      position,
      selected: true,
      state: "queryable",
      added_at: "2026-10-01T00:00:00Z",
      version: 1,
    }))
    mocks.request.mockImplementation(async (request: CanonicalRequestFixture) => {
      const { path, method, body, headers } = request
      if (path.endsWith("/context"))
        return {
          workspace_id: "workspace-a",
          workspace: {
            id: "workspace-a",
            name: "Research",
            workspace_profile: "research",
            study_materials_policy: "workspace",
            deleted: false,
            archived: false,
            created_at: "2026-10-01T00:00:00Z",
            version: 1,
          },
          sources: { items: sources },
          partial_errors: [],
        }
      if (path.startsWith("/api/v1/notes/search/"))
        return { notes: canonical ? [canonical] : [] }
      if (!path.startsWith("/api/v1/notes/")) return []
      if (method === "POST") canonical = { ...body, version: 1 }
      if (method === "PUT") {
        if (headers?.["expected-version"] !== String(canonical.version))
          throw Object.assign(new Error("version required"), { status: 422 })
        canonical = { ...canonical, ...body, version: canonical.version + 1 }
      }
      if (!canonical) throw Object.assign(new Error("missing"), { status: 404 })
      return canonical
    })
    mocks.upload.mockResolvedValue({ media_id: 101 })
    mocks.answerState = {
      isAuthorityCurrent: () => true,
      answerTrustState: "uncited_degraded_answer",
      answerEvidenceOrigin: "local_library",
      answerTrustReasonCodes: ["missing_citations"],
      settings: { ...DEFAULT_RAG_SETTINGS },
      answer: "Qualified answer [1].",
      citations: [{ index: 1 }, { index: 2 }],
      isSearching: false,
      error: null,
      searchDetails: null,
      query: "What is supported?",
      currentThreadId: "thread-current",
      messages: [],
      scrollToSource: vi.fn(),
      results: [
        {
          id: "7",
          content: "Native source excerpt",
          metadata: { title: "Native source", source_type: "pdf" },
        },
        {
          id: "chunk-note",
          content: "Original note excerpt",
          metadata: {
            title: "Original note",
            source_type: "notes",
            note_id: originalNoteId,
          },
        },
      ],
    }
    mocks.sourceNoteIds.add(originalNoteId)
    const caller = render(<AnswerPanel />)
    fireEvent.click(
      screen.getByRole("button", { name: "Continue in Research Workspace" }),
    )
    await waitFor(() =>
      expect(mocks.navigate).toHaveBeenCalledWith("/research-workspace"),
    )
    const transfer = (await consumeResearchWorkspacePrefill("alice"))!
    expect(transfer.scope).toMatchObject({
      keyword_filter: "",
      collection_id: null,
    })
    caller.unmount()
    const receiver = renderHook(() =>
      useResearchWorkspacePrefill("workspace-a", true),
    )
    await waitFor(() => expect(checkpoint(transfer.id)?.completed).toBe(true))
    expect(receiver.result.current.error).toBeNull()
    expect(checkpoint(transfer.id).draftRetained).toBe(true)
    const provenance = readKnowledgeNoteProvenance(canonical.content)!
    expect(provenance.research?.import_id).toBe(transfer.id)
    expect(provenance.research?.sources).toMatchObject([
      {
        mediaId: 7,
        evidence: {
          importId: transfer.id,
          snapshot: false,
          scope: transfer.scope,
          sources: [
            {
              originalId: "7",
              sourceType: "pdf",
              excerpt: "Native source excerpt",
            },
          ],
        },
      },
      {
        mediaId: 101,
        evidence: {
          importId: transfer.id,
          snapshot: true,
          scope: transfer.scope,
          trustState: "uncited_degraded_answer",
          sources: [
            {
              originalId: originalNoteId,
              sourceType: "notes",
              excerpt: "Original note excerpt",
            },
          ],
        },
      },
    ])
    expect(canonical.content.match(/<!-- tldw-knowledge:v1:/g)).toHaveLength(1)
    if (withOldMarker)
      expect(canonical.content).toContain("Earlier reviewed source body")
    expect(useWorkspaceStore.getState().currentNote.content).toBe(
      canonical.content,
    )
    receiver.unmount()
    useWorkspaceStore.getState().reset()
    useWorkspaceStore.setState({ workspaceId: null, storeHydrated: true })
    const { restoreMigratedResearchWorkspace } =
      await import("@/components/Option/ResearchWorkspace/workspace-server-restore")
    const origin = useWorkspaceStore.getState().workspaceId
    await restoreMigratedResearchWorkspace({
      signal: new AbortController().signal,
      apply: (workspace, scopeKey) =>
        useWorkspaceStore.getState().installServerWorkspace(workspace, {
          scopeKey,
          expectedWorkspaceId: origin,
        }),
    })
    expect(useWorkspaceStore.getState().currentNote.content).toBe(
      canonical.content,
    )
    expect(
      useWorkspaceStore
        .getState()
        .sources.map((source) => source.knowledgeQaEvidence),
    ).toEqual(provenance.research!.sources.map((source) => source.evidence))
    const reopened = renderHook(() =>
      useResearchWorkspacePrefill("workspace-a", true),
    )
    await act(async () => {
      await Promise.resolve()
    })
    expect(mocks.upload).toHaveBeenCalledTimes(1)
    expect(checkpoint(transfer.id).completed).toBe(true)
    reopened.unmount()
  },
)

it("rejects invalid required current-import provenance before writing an older marker", async () => {
  markMigrated()
  const oldContent = retainKnowledgeNoteProvenance(
    "Human draft",
    previousProvenance,
  )
  const canonical = {
    id: canonicalNoteId,
    title: "Human title",
    content: oldContent,
    version: 2,
    keywords: [],
  }
  useWorkspaceStore.getState().loadNote(canonical)
  const transfer = payload()
  transfer.sources = transfer.sources.slice(0, 2)
  transfer.scope = { keyword_filter: "x".repeat(513) }
  mocks.upload.mockResolvedValue({ media_id: 101 })
  mocks.request.mockResolvedValue(canonical)
  await queueResearchWorkspacePrefill(transfer)
  const receiver = renderHook(() =>
    useResearchWorkspacePrefill("workspace-a", true),
  )
  await waitFor(() =>
    expect(receiver.result.current.error).toContain("could not be saved"),
  )
  expect(
    mocks.request.mock.calls.filter(([request]) =>
      ["POST", "PUT"].includes(request.method),
    ),
  ).toEqual([])
  expect(checkpoint(transfer.id)?.completed).not.toBe(true)
  expect(useWorkspaceStore.getState().currentNote.content).toContain(
    "Human draft",
  )
  expect(useWorkspaceStore.getState().currentNote.content).toContain(
    "Unsupported draft",
  )
  await act(async () => {
    await receiver.result.current.retry()
  })
  await waitFor(() =>
    expect(receiver.result.current.error).toContain("could not be saved"),
  )
  expect(mocks.upload).toHaveBeenCalledTimes(1)
  expect(
    mocks.request.mock.calls.filter(([request]) =>
      ["POST", "PUT"].includes(request.method),
    ),
  ).toEqual([])
  receiver.unmount()
})

it.each(["web_document", "web"])(
  "imports the actual Review caller's stored %s directly without snapshots",
  async (type) => {
    const caller = render(
      <MediaKnowledgeActions
        items={[{ id: 7, title: "Stored article", type }]}
        navigate={mocks.navigate}
      />,
    )
    fireEvent.click(
      screen.getByRole("button", { name: "Research with this source" }),
    )
    await waitFor(() =>
      expect(mocks.navigate).toHaveBeenCalledWith("/research-workspace"),
    )
    const transfer = (await consumeResearchWorkspacePrefill("alice"))!
    caller.unmount()
    const receiver = renderHook(() =>
      useResearchWorkspacePrefill("workspace-a", true),
    )
    await waitFor(() => expect(receiver.result.current.attached).toBe(1))
    expect(transfer.sources[0]).toMatchObject({
      mediaId: 7,
      originalId: 7,
      sourceType: type,
      type: "website",
    })
    expect(useWorkspaceStore.getState().sources[0]).toMatchObject({
      mediaId: 7,
      knowledgeQaEvidence: {
        snapshot: false,
        sources: [{ mediaId: 7, sourceType: type }],
      },
    })
    expect(mocks.upload).not.toHaveBeenCalled()
    receiver.unmount()
  },
)

it.each([false, true])(
  "retains the scoped principal through canonical GET/PUT/readback with changed principal=%s",
  async (changePrincipal) => {
    markMigrated()
    mocks.multiUser = true
    let principal = "user-a"
    let canonical = {
      id: canonicalNoteId,
      title: "Human title",
      content: "Human draft",
      keywords: ["workspace:workspace-a"],
      version: 2,
    }
    useWorkspaceStore.getState().loadNote(canonical)
    const transfer = payload()
    transfer.sources = [transfer.sources[3]]
    let mutations = 0
    let capturedDraft: unknown
    const seen: BgRequestInit[] = []
    mocks.request.mockImplementation(async (request: CanonicalRequestFixture) => {
      seen.push(request)
      const expectedUser = request.headers?.["X-TLDW-Expected-User-ID"]
      if (expectedUser && expectedUser !== principal)
        throw Object.assign(new Error("request_config_scope_changed"), {
          status: 412,
        })
      if (request.method === "PUT") {
        if (request.headers?.["expected-version"] !== "2")
          throw Object.assign(new Error("version required"), { status: 422 })
        mutations += 1
        canonical = { ...canonical, ...request.body, version: 3 }
      } else if (changePrincipal) {
        capturedDraft = useWorkspaceStore.getState().currentNote
        principal = "user-b"
      }
      return canonical
    })
    await queueResearchWorkspacePrefill(transfer, "alice:user-a")
    const receiver = renderHook(() =>
      useResearchWorkspacePrefill("workspace-a", true),
    )
    if (changePrincipal) {
      await waitFor(() =>
        expect(receiver.result.current.error).toContain("could not be saved"),
      )
      expect(mutations).toBe(0)
      expect(canonical.content).toBe("Human draft")
      expect(useWorkspaceStore.getState().currentNote).toBe(capturedDraft)
      expect(checkpoint(transfer.id)?.completed).not.toBe(true)
      expect(checkpoint(transfer.id)?.draftRetained).not.toBe(true)
    } else {
      await waitFor(() => expect(checkpoint(transfer.id)?.completed).toBe(true))
      expect(seen.map((request) => request.method)).toEqual([
        "GET",
        "PUT",
        "GET",
      ])
      for (const request of seen)
        expect(request.headers?.["X-TLDW-Expected-User-ID"]).toBe("user-a")
      expect(seen[1].headers["expected-version"]).toBe("2")
      expect(
        readKnowledgeNoteProvenance(canonical.content)?.research?.import_id,
      ).toBe(transfer.id)
    }
    receiver.unmount()
  },
)

it("imports an external web result's arbitrary numeric ID as a snapshot, not stored media", async () => {
  useWorkspaceStore
    .getState()
    .addSources([{ mediaId: 7, title: "Unrelated stored item", type: "text" }])
  const transfer = buildKnowledgeQaWorkspacePrefill({
    threadId: "thread-web",
    query: "External evidence",
    answer: "Qualified web answer",
    citations: [1],
    results: [
      {
        id: "7",
        content: "External web excerpt",
        metadata: {
          source_type: "web",
          title: "External article",
          url: "https://example.com/article",
        },
      },
    ],
  })
  mocks.upload.mockResolvedValue({ media_id: 101 })
  await queueResearchWorkspacePrefill(transfer)
  const receiver = renderHook(() =>
    useResearchWorkspacePrefill("workspace-a", true),
  )
  await waitFor(() => expect(checkpoint(transfer.id)?.completed).toBe(true))
  expect(
    useWorkspaceStore.getState().sources.map((source) => source.mediaId),
  ).toEqual([7, 101])
  expect(useWorkspaceStore.getState().sources[1]).toMatchObject({
    mediaId: 101,
    knowledgeQaEvidence: {
      snapshot: true,
      sources: [
        {
          originalId: "7",
          mediaId: null,
          sourceType: "web",
          excerpt: "External web excerpt",
          url: "https://example.com/article",
        },
      ],
    },
  })
  expect(
    useWorkspaceStore.getState().getEffectiveSelectedMediaIds(),
  ).not.toContain(7)
  expect(mocks.upload).toHaveBeenCalledTimes(1)
  receiver.unmount()
})

it("imports complete canonical note content once and retains the original evidence and revision", async () => {
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
  let canonical: CanonicalImportNoteFixture | null = null
  mocks.request.mockImplementation(async ({ method, body }: CanonicalRequestFixture) => {
    if (method === "POST") canonical = { ...body, version: 1 }
    if (!canonical) throw Object.assign(new Error("missing"), { status: 404 })
    return canonical
  })
  const handoff = payload()
  handoff.sources = handoff.sources.filter(
    (source) => source.sourceType === "notes",
  )
  await queueResearchWorkspacePrefill(handoff, "alice")
  mocks.upload.mockResolvedValue({ media_id: 101 })
  const view = renderHook(() =>
    useResearchWorkspacePrefill("workspace-a", true),
  )
  await waitFor(() => expect(view.result.current.attached).toBe(1))
  expect(mocks.sourceNoteRequest).toHaveBeenCalledTimes(1)
  expect(mocks.sourceNoteRequest).toHaveBeenCalledWith(
    expect.objectContaining({
      method: "GET",
      path: "/api/v1/notes/note-uuid",
      abortSignal: expect.any(AbortSignal),
    }),
  )
  const file: File = mocks.upload.mock.calls[0][0]
  const content = await new Promise<string>((resolve) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result))
    reader.readAsText(file)
  })
  expect(content).toContain("Complete note ending outside retrieved chunks.")
  expect(content).toContain("Original reference: notes / note-uuid")
  const source = useWorkspaceStore.getState().sources[0]
  expect(source.title).toBe("Field note — full note snapshot (v4)")
  expect(source.knowledgeQaEvidence?.sources).toEqual(
    expect.arrayContaining([
      expect.objectContaining({
        originalId: "note-uuid",
        originalVersion: 4,
        excerpt: "First excerpt",
      }),
      expect.objectContaining({
        originalId: "note-uuid",
        originalVersion: 4,
        excerpt: "Second excerpt",
      }),
    ]),
  )
  expect(mocks.upload).toHaveBeenCalledTimes(1)
  await waitFor(() => expect(view.result.current.importing).toBe(false))
  expect(view.result.current.error).toBeNull()
  expect(
    readKnowledgeNoteProvenance(canonical.content)?.research?.sources[0]
      .evidence.sources[0],
  ).toEqual(expect.objectContaining({ originalVersion: 4 }))
  view.unmount()
})

it.each([
  ["denied read", null],
  [
    "wrong identity",
    {
      id: "other-note",
      content: "Private content",
      title: "Other",
      version: 4,
    },
  ],
  [
    "deleted note",
    {
      id: "note-uuid",
      content: "Deleted content",
      title: "Field",
      version: 4,
      deleted: true,
    },
  ],
  [
    "invalid revision",
    { id: "note-uuid", content: "Content", title: "Field", version: 0 },
  ],
])(
  "retains a failed import instead of substituting old excerpts after a %s",
  async (_label, note) => {
    const handoff = payload()
    handoff.sources = handoff.sources.filter(
      (source) => source.sourceType === "notes",
    )
    await queueResearchWorkspacePrefill(handoff, "alice")
    if (note) mocks.sourceNoteRequest.mockResolvedValue(note)
    else
      mocks.sourceNoteRequest.mockRejectedValue(
        Object.assign(new Error("Denied"), { status: 403 }),
      )
    mocks.upload.mockResolvedValue({ media_id: 101 })
    const view = renderHook(() =>
      useResearchWorkspacePrefill("workspace-a", true),
    )
    await waitFor(() => expect(view.result.current.failed).toBe(1))
    expect(view.result.current.importing).toBe(false)
    expect(mocks.upload).not.toHaveBeenCalled()
    expect(
      (await consumeResearchWorkspacePrefill("alice"))?.sources[0].importError,
    ).toContain("full note")
    view.unmount()
  },
)

it("retires a late full-note read before uploading into a changed workspace", async () => {
  const handoff = payload()
  handoff.sources = handoff.sources.filter(
    (source) => source.sourceType === "notes",
  )
  await queueResearchWorkspacePrefill(handoff, "alice")
  let finish!: (value: unknown) => void
  mocks.sourceNoteRequest.mockImplementation(
    () =>
      new Promise((resolve) => {
        finish = resolve
      }),
  )
  mocks.upload.mockResolvedValue({ media_id: 101 })
  const view = renderHook(() =>
    useResearchWorkspacePrefill("workspace-a", true),
  )
  await waitFor(() => expect(mocks.sourceNoteRequest).toHaveBeenCalled())
  await act(async () => {
    useWorkspaceStore.setState({ workspaceId: "workspace-b" })
    finish({
      id: "note-uuid",
      title: "Field note",
      content: "Complete note",
      version: 4,
    })
  })
  expect(mocks.upload).not.toHaveBeenCalled()
  expect(useWorkspaceStore.getState().sources).toEqual([])
  view.unmount()
})

it.each(["streamed", "restored"])(
  "imports %s note chunks using their canonical source identity",
  async (shape) => {
    const handoff = buildKnowledgeQaWorkspacePrefill({
      threadId: "thread-a",
      query: "Field evidence",
      answer: null,
      citations: [],
      results: ["First excerpt", "Second excerpt"].map((content, index) => ({
        id: `note_chunk_${index}`,
        content,
        ...(shape === "streamed" ? { sourceId: "note-uuid" } : {}),
        metadata: {
          source_type: "notes",
          title: "Field note",
          ...(shape === "restored" ? { source_id: "note-uuid" } : {}),
        },
      })),
    })
    expect(handoff.sources.map((source) => source.originalId)).toEqual([
      "note-uuid",
      "note-uuid",
    ])
    await queueResearchWorkspacePrefill(handoff, "alice")
    mocks.upload.mockResolvedValue({ media_id: 101 })
    const view = renderHook(() =>
      useResearchWorkspacePrefill("workspace-a", true),
    )
    await waitFor(() => expect(view.result.current.attached).toBe(1))
    expect(mocks.sourceNoteRequest).toHaveBeenCalledWith(
      expect.objectContaining({ path: "/api/v1/notes/note-uuid" }),
    )
    expect(mocks.upload).toHaveBeenCalledTimes(1)
    expect(
      useWorkspaceStore
        .getState()
        .sources[0].knowledgeQaEvidence?.sources.map(
          (source) => source.excerpt,
        ),
    ).toEqual(["First excerpt", "Second excerpt"])
    view.unmount()
  },
)

it("retains canonical import provenance in a fresh server workspace", async () => {
  let canonical: CanonicalImportNoteFixture | null = null
  mocks.request.mockImplementation(async ({ method, body }: CanonicalRequestFixture) => {
    if (method === "POST") canonical = { ...body, version: 1,
      content: stripKnowledgeNoteProvenance(body.content),
      knowledge_provenance_state: "active", knowledge_provenance_version: 1,
      knowledge_provenance_hash: `sha256:${"a".repeat(64)}`,
    }
    if (!canonical) throw Object.assign(new Error("missing"), { status: 404 })
    return canonical
  })
  const handoff = payload()
  handoff.sources = handoff.sources.filter(
    (source) => source.sourceType === "notes",
  )
  await queueResearchWorkspacePrefill(handoff, "alice")
  mocks.upload.mockResolvedValue({ media_id: 101 })
  const view = renderHook(() =>
    useResearchWorkspacePrefill("workspace-a", true, true),
  )
  await waitFor(() => expect(canonical).not.toBeNull())
  expect(
    canonical.knowledge_provenance?.research?.import_id,
  ).toBe(handoff.id)
  expect(useWorkspaceStore.getState().currentNote.id).toBe(canonical.id)
  expect(mocks.upload).toHaveBeenCalledTimes(1)
  expect(view.result.current.error).toBeNull()
  view.unmount()
})

it("does not revive a discarded completed local import when the server workspace is confirmed", async () => {
  const handoff = payload()
  handoff.sources = handoff.sources.filter(
    (source) => source.sourceType === "notes",
  )
  await queueResearchWorkspacePrefill(handoff, "alice")
  mocks.upload.mockResolvedValue({ media_id: 101 })
  const view = renderHook(
    ({ serverBacked }) =>
      useResearchWorkspacePrefill("workspace-a", true, serverBacked),
    { initialProps: { serverBacked: false } },
  )
  await waitFor(() =>
    expect((mocks.values.values().next().value as PrefillCheckpointFixture)?.completed).toBe(true),
  )
  act(() => {
    useWorkspaceStore.getState().clearCurrentNote()
  })
  view.rerender({ serverBacked: true })
  await act(async () => {
    await Promise.resolve()
  })
  expect(mocks.request).not.toHaveBeenCalled()
  expect(useWorkspaceStore.getState().currentNote.content).toBe("")
  view.unmount()
})

it("waits for server workspace confirmation before starting a fresh canonical import", async () => {
  const { tldwClient } = await import("@/services/tldw/TldwApiClient")
  const client = tldwClient as Pick<typeof tldwClient, "getConfig"> & {
    upsertWorkspace?: (...args: Parameters<typeof tldwClient.upsertWorkspace>) => Promise<Partial<Awaited<ReturnType<typeof tldwClient.upsertWorkspace>>>>
    getWorkspaceSources?: (...args: Parameters<typeof tldwClient.getWorkspaceSources>) => ReturnType<typeof tldwClient.getWorkspaceSources>
    addWorkspaceSource?: (...args: Parameters<typeof tldwClient.addWorkspaceSource>) => Promise<Partial<Awaited<ReturnType<typeof tldwClient.addWorkspaceSource>>>>
  }
  let confirm!: (value: unknown) => void
  client.upsertWorkspace = vi.fn(
    () =>
      new Promise((resolve) => {
        confirm = resolve
      }),
  )
  client.getWorkspaceSources = vi.fn().mockResolvedValue([])
  client.addWorkspaceSource = vi.fn().mockResolvedValue({})
  let canonical: CanonicalImportNoteFixture | null = null
  mocks.request.mockImplementation(async ({ method, body }: CanonicalRequestFixture) => {
    if (method === "POST") canonical = { ...body, version: 1 }
    if (!canonical) throw Object.assign(new Error("missing"), { status: 404 })
    return canonical
  })
  const handoff = payload()
  handoff.sources = handoff.sources.filter(
    (source) => source.sourceType === "notes",
  )
  await queueResearchWorkspacePrefill(handoff, "alice")
  mocks.upload.mockResolvedValue({ media_id: 101 })
  const view = render(<ResearchWorkspace />)
  try {
    await waitFor(() => expect(client.upsertWorkspace).toHaveBeenCalled())
    await act(async () => {
      await Promise.resolve()
    })
    expect(mocks.upload).not.toHaveBeenCalled()
    await act(async () => {
      confirm({})
    })
    await waitFor(() => expect(canonical).not.toBeNull())
    expect(
      readKnowledgeNoteProvenance(canonical.content)?.research?.import_id,
    ).toBe(handoff.id)
  } finally {
    view.unmount()
    delete client.upsertWorkspace
    delete client.getWorkspaceSources
    delete client.addWorkspaceSource
  }
})

it.each(["streamed", "restored"])(
  "reuses %s media chunks through their canonical media identity",
  async (form) => {
    const handoff = buildKnowledgeQaWorkspacePrefill({
      threadId: "thread-a",
      query: "Evidence?",
      answer: "A cited answer",
      citations: [1],
      results: [
        {
          id: "media-chunk-7-0",
          content: "Original media excerpt",
          ...(form === "streamed" ? { sourceId: "7" } : {}),
          metadata: {
            source_type: "media_db",
            title: "Original media",
            ...(form === "restored" ? { source_id: "7" } : {}),
          },
        },
      ],
    })
    await queueResearchWorkspacePrefill(handoff, "alice")
    mocks.upload.mockResolvedValue({ media_id: 101 })
    mocks.details.mockResolvedValue({
      content: { text: "Original media excerpt" },
      vector_processing_status: "completed",
    })
    const view = renderHook(() =>
      useResearchWorkspacePrefill("workspace-a", true),
    )
    await waitFor(() => expect(view.result.current.attached).toBe(1))
    expect(useWorkspaceStore.getState().sources[0].mediaId).toBe(7)
    expect(mocks.upload).not.toHaveBeenCalled()
    view.unmount()
  },
)

it.each(["note-first", "media-first"])(
  "retains both original references when a note snapshot reuses retrieved media: %s",
  async (order) => {
    const handoff = payload()
    const noteSources = handoff.sources.filter(
      (source) => source.sourceType === "notes",
    )
    const mediaSource = buildKnowledgeQaWorkspacePrefill({
      threadId: "thread-1",
      query: "Evidence?",
      answer: null,
      citations: [],
      results: [
        {
          id: "media-chunk-7-0",
          sourceId: "7",
          content: "Existing snapshot excerpt",
          metadata: {
            source_type: "media_db",
            title: "Existing note snapshot",
          },
        },
      ],
    }).sources[0]
    handoff.sources =
      order === "note-first"
        ? [...noteSources, mediaSource]
        : [mediaSource, ...noteSources]
    await queueResearchWorkspacePrefill(handoff, "alice")
    mocks.upload.mockResolvedValue({ media_id: 7 })
    let canonical: CanonicalImportNoteFixture | null = null
    mocks.request.mockImplementation(async ({ method, body }: CanonicalRequestFixture) => {
      if (method === "POST") canonical = { ...body, version: 1 }
      if (!canonical) throw Object.assign(new Error("missing"), { status: 404 })
      return canonical
    })
    const view = renderHook(() =>
      useResearchWorkspacePrefill("workspace-a", true, true),
    )
    await waitFor(() => expect(view.result.current.importing).toBe(false))
    await waitFor(() => expect(canonical).not.toBeNull())
    const retained = readKnowledgeNoteProvenance(canonical.content)?.research
      ?.sources
    expect(retained).toEqual([
      expect.objectContaining({
        mediaId: 7,
        evidence: expect.objectContaining({
          snapshot: true,
          sources: expect.arrayContaining([
            expect.objectContaining({
              originalId: "note-uuid",
              originalVersion: 4,
              excerpt: "First excerpt",
            }),
            expect.objectContaining({
              originalId: "note-uuid",
              originalVersion: 4,
              excerpt: "Second excerpt",
            }),
            expect.objectContaining({
              originalId: "7",
              excerpt: "Existing snapshot excerpt",
            }),
          ]),
        }),
      }),
    ])
    expect(useWorkspaceStore.getState().sources).toHaveLength(1)
    expect(mocks.upload).toHaveBeenCalledTimes(1)
    view.unmount()
  },
)

it("replays canonical import receipts without replacing edits made before retry", async () => {
  const transfer = payload()
  transfer.sources = transfer.sources.filter(source => source.sourceType === "notes")
  await queueResearchWorkspacePrefill(transfer, "alice")
  mocks.upload.mockResolvedValue({ media_id: 101 })
  let canonical: { id: string; content: string; knowledge_provenance: KnowledgeNoteProvenance } | null = null
  const writes: BgRequestInit[] = []
  mocks.request.mockImplementation(async request => {
    if (request.method === "POST") {
      writes.push(request)
      if (!canonical) canonical = { ...request.body, version: 1, knowledge_provenance_state: "active", knowledge_provenance_version: 1, knowledge_provenance_hash: `sha256:${"a".repeat(64)}` }
      if (writes.length === 1) throw new Error("Lost acknowledgment")
    }
    if (!canonical) throw Object.assign(new Error("missing"), { status: 404 })
    return canonical
  })
  const view = renderHook(() => useResearchWorkspacePrefill("workspace-a", true, true))
  await waitFor(() => expect(view.result.current.error).not.toBeNull())
  await act(async () => {
    useWorkspaceStore.getState().updateNoteContent("Edited before receipt recovery")
    await view.result.current.retry()
  })
  await waitFor(() => expect(view.result.current.error).toBeNull())
  await waitFor(() => expect(view.result.current.importing).toBe(false))
  expect(writes).toHaveLength(2)
  expect(writes[1].body).toEqual(writes[0].body)
  expect(writes[1].headers).toEqual(writes[0].headers)
  expect(writes[0].headers["Idempotency-Key"]).toBeTruthy()
  expect(stripKnowledgeNoteProvenance(useWorkspaceStore.getState().currentNote.content)).toBe("Edited before receipt recovery")
  expect(useWorkspaceStore.getState().currentNote).toMatchObject({ id: canonical.id, isDirty: true, knowledge_provenance_version: 1 })
  view.unmount()
})

it("does not finish a partial import when its recovered receipt lacks newly attached sources", async () => {
  const transfer = payload()
  transfer.sources = transfer.sources.slice(0, 3)
  await queueResearchWorkspacePrefill(transfer, "alice")
  mocks.upload.mockResolvedValueOnce({ media_id: 101 })
    .mockRejectedValueOnce(new Error("source unavailable"))
    .mockResolvedValueOnce({ media_id: 102 })
  let canonical: { id: string; content: string; knowledge_provenance: KnowledgeNoteProvenance } | null = null
  const writes: BgRequestInit[] = []
  mocks.request.mockImplementation(async request => {
    if (request.method === "POST" || request.method === "PUT") {
      writes.push(request)
      if (!canonical || request.method === "PUT") canonical = {
        ...request.body, id: transfer.id, version: writes.length,
        knowledge_provenance_state: "active", knowledge_provenance_version: writes.length,
        knowledge_provenance_hash: `sha256:${"a".repeat(64)}`,
      }
      if (writes.length === 1) throw new Error("Lost acknowledgment")
    }
    if (!canonical) throw Object.assign(new Error("missing"), { status: 404 })
    return canonical
  })
  const view = renderHook(() => useResearchWorkspacePrefill("workspace-a", true, true))
  await waitFor(() => expect(view.result.current.error).not.toBeNull())
  await act(async () => {
    useWorkspaceStore.getState().updateNoteContent("Later authored draft")
    await view.result.current.retry()
  })
  await waitFor(() => expect(view.result.current.importing).toBe(false))
  expect(view.result.current.error).not.toBeNull()
  expect((await consumeResearchWorkspacePrefill("alice"))?.completed).not.toBe(true)
  expect(writes[1].body).toEqual(writes[0].body)
  expect(writes[1].headers).toEqual(writes[0].headers)
  await act(async () => { await view.result.current.retry() })
  await waitFor(() => expect(view.result.current.importing).toBe(false))
  expect(view.result.current.error).toBeNull()
  expect(writes[2].method).toBe("PUT")
  expect(writes[2].headers["Idempotency-Key"]).not.toBe(writes[0].headers["Idempotency-Key"])
  expect(canonical.knowledge_provenance.research.sources.map(source => source.mediaId)).toEqual([101, 102])
  expect(stripKnowledgeNoteProvenance(canonical.content)).toBe("Later authored draft")
  view.unmount()
})

it.each([
    { status: 409, detail: { error_code: 'notes_provenance_encryption_unsupported' }, message: 'Policy unavailable' },
    { status: 429, detail: 'Rate limit exceeded for notes.create', message: 'Rate limit exceeded for notes.create' },
    { status: 409, detail: { error_code: 'notes_organization_sync_not_ready' }, message: 'Notes organization Sync is not ready for writes.' },
  ])('retains the pending canonical import through $status $message receipt replay', async ({ status, detail, message }) => {
  const transfer = payload()
  transfer.sources = transfer.sources.filter(source => source.sourceType === 'notes')
  await queueResearchWorkspacePrefill(transfer, 'alice')
  mocks.upload.mockResolvedValue({ media_id: 101 })
  const writes: BgRequestInit[] = []
  let canonical: { id: string; content: string; knowledge_provenance: KnowledgeNoteProvenance } | null = null
  mocks.request.mockImplementation(async request => {
    if (request.method === 'POST') {
      writes.push(request)
      canonical ||= { ...request.body, version: 1, knowledge_provenance_state: 'active', knowledge_provenance_version: 1 }
      if (writes.length === 1) throw new Error('Lost response')
      if (writes.length === 2) throw Object.assign(new Error(message), { status, details: { detail } })
    }
    if (!canonical) throw Object.assign(new Error('missing'), { status: 404 })
    return canonical
  })
  const view = renderHook(() => useResearchWorkspacePrefill('workspace-a', true, true))
  await waitFor(() => expect(view.result.current.error).not.toBeNull())
  await act(async () => { await view.result.current.retry() })
  await waitFor(() => expect(view.result.current.importing).toBe(false))
  expect(view.result.current.error).not.toBeNull()
  await act(async () => { await view.result.current.retry() })
  await waitFor(() => expect(view.result.current.error).toBeNull())
  expect(writes).toHaveLength(3)
  expect(writes[2].body).toEqual(writes[0].body)
  expect(writes[2].headers).toEqual(writes[0].headers)
  view.unmount()
})

describe("captured-owner legacy import seed", () => {
  const seedId = "cda40f93-c58a-43e4-ad39-a6445a981702"
  const gate = () => {
    let release!: () => void
    const promise = new Promise<void>(resolve => { release = resolve })
    return { promise, release }
  }
  const installDestination = async (scopeKey = "alice") => {
    const { hydrateWorkspaceFromServer } = await import("@/store/workspace-api")
    const { serverWorkspacePayload } = await import("@/store/__tests__/workspace-activation.fixtures")
    const server = serverWorkspacePayload()
    server.id = "workspace-a"
    server.metadata.id = "workspace-a"
    server.sources = []
    server.artifacts = []
    server.notes = []
    useWorkspaceStore.getState().reset()
    useWorkspaceStore.setState({ workspaceId: null, storeHydrated: true })
    const workspace = await hydrateWorkspaceFromServer("workspace-a", {
      fetch: async () => server, requireComplete: true,
    })
    expect(useWorkspaceStore.getState().installServerWorkspace(workspace, {
      scopeKey, expectedWorkspaceId: null,
    })).toBe(true)
    markMigrated()
    expect(useWorkspaceStore.getState().serverWorkspace?.scopeKey).toBe(scopeKey)
  }
  const legacyDraft = (id: number | undefined = 7) => {
    const state = useWorkspaceStore.getState()
    state.setCurrentNote({
      ...state.currentNote, id, version: 3, title: "Unsaved legacy title",
      content: "Unsaved legacy body", keywords: ["legacy"], isDirty: true,
      pendingKnowledgeProvenance: previousProvenance,
    })
    return useWorkspaceStore.getState().currentNote
  }
  const queueSeed = async () => {
    const transfer = payload()
    transfer.id = seedId
    transfer.query = "Legacy seed question"
    transfer.answer = "Fresh incoming answer"
    transfer.sources = transfer.sources.slice(3)
    await queueResearchWorkspacePrefill(transfer, "alice")
    return transfer
  }
  const canonicalDouble = (options: { readGate?: Promise<void>; writeGate?: Promise<void>; lostAck?: boolean } = {}) => {
    let canonical: CanonicalImportNoteFixture | null = null
    const writes: BgRequestInit[] = []
    let reads = 0
    mocks.request.mockImplementation(async (request: BgRequestInit) => {
      if (request.method === "GET") {
        expect(request.path).toBe(`/api/v1/notes/${seedId}`)
        if (++reads === 1) await options.readGate
      } else {
        expect(request.method).toBe("POST")
        expect(request.path).toBe("/api/v1/notes/")
        writes.push(request)
        canonical ||= {
          ...request.body, version: 1, knowledge_provenance_state: "active",
          knowledge_provenance_version: 1, knowledge_provenance_hash: `sha256:${"a".repeat(64)}`,
        }
        await options.writeGate
        if (options.lostAck && writes.length === 1) throw new Error("Lost acknowledgment")
      }
      if (!canonical) throw Object.assign(new Error("missing"), { status: 404 })
      return canonical
    })
    return writes
  }

  it.each([false, true])("retains numeric legacy identity, authored draft and seed before GET/ACK (lost ACK=%s)", async lostAck => {
    await installDestination()
    const original = legacyDraft()
    const transfer = await queueSeed()
    const read = gate()
    const write = gate()
    const writes = canonicalDouble({ readGate: read.promise, writeGate: write.promise, lostAck })
    const view = renderHook(() => useResearchWorkspacePrefill("workspace-a", true, true))
    try {
      await waitFor(() => expect(mocks.request).toHaveBeenCalledTimes(1))
      const seeded = useWorkspaceStore.getState().currentNote
      expect(seeded).toMatchObject({ id: 7, version: 3, title: original.title, keywords: original.keywords, isDirty: true })
      expect(seeded.pendingKnowledgeProvenance).toBe(original.pendingKnowledgeProvenance)
      expect(seeded.content).toContain("Unsaved legacy body")
      expect(seeded.content).toContain("Fresh incoming answer")
      expect(seeded.content).toContain(`Import reference: ${transfer.id}`)
      expect(checkpoint(seedId)).toMatchObject({ legacyNoteId: 7, canonicalNoteId: seedId })
      expect(useWorkspaceStore.getState().serverWorkspace?.scopeKey).toBe("alice")
      await act(async () => { read.release() })
      await waitFor(() => expect(writes).toHaveLength(1))
      expect(writes[0].body).toMatchObject({ id: seedId, title: original.title })
      expect(writes[0].body.content).toContain("Unsaved legacy body")
      expect(writes[0].body.content).toContain("Fresh incoming answer")
      expect(writes[0].body.content).toContain(`Import reference: ${seedId}`)
      expect(writes[0].servicePromptConfig?.serverUrl).toBe("alice")
      expect(writes[0].abortSignal?.aborted).toBe(false)
      expect(writes[0].headers?.["Idempotency-Key"]).toBeTruthy()
      expect(useWorkspaceStore.getState().currentNote.id).toBe(7)
      await act(async () => { write.release() })
      if (lostAck) {
        await waitFor(() => expect(view.result.current.error).not.toBeNull())
        expect(checkpoint(seedId).pendingNoteWrite.body).toEqual(writes[0].body)
        expect(useWorkspaceStore.getState().currentNote.id).toBe(7)
        expect(useWorkspaceStore.getState().currentNote.pendingKnowledgeProvenance).toBe(original.pendingKnowledgeProvenance)
        await act(async () => { await view.result.current.retry() })
        await waitFor(() => expect(writes).toHaveLength(2))
        expect(writes[1].body).toEqual(writes[0].body)
        expect(writes[1].headers).toEqual(writes[0].headers)
      }
      await waitFor(() => expect(checkpoint(seedId)?.completed).toBe(true))
      expect(view.result.current.error).toBeNull()
      expect(useWorkspaceStore.getState().currentNote.id).toBe(seedId)
      expect(useWorkspaceStore.getState().currentNote.content).toContain("Fresh incoming answer")
      expect(useWorkspaceStore.getState().currentNote.content.split(`Import reference: ${seedId}`)).toHaveLength(2)
    } finally {
      read.release()
      write.release()
      view.unmount()
    }
  })

  it.each(["append", "replace"] as const)("still refuses canonical manual %s capture", async mode => {
    await installDestination()
    const original = legacyDraft()
    useWorkspaceStore.getState().captureToCurrentNote({ title: "Manual", content: "Forbidden capture", mode })
    expect(useWorkspaceStore.getState().currentNote).toBe(original)
  })

  it.each(["local", "local-only", "unbound"])("retains the %s draft import path", async path => {
    if (path === "unbound") await installDestination()
    else if (path === "local") markMigrated()
    const original = legacyDraft()
    if (path === "unbound") useWorkspaceStore.getState().setCurrentNote({ ...original, id: undefined })
    await queueSeed()
    const writes = canonicalDouble()
    const view = renderHook(() => useResearchWorkspacePrefill("workspace-a", true, path !== "local-only"))
    await waitFor(() => expect(checkpoint(seedId)?.completed).toBe(true))
    expect(view.result.current.error).toBeNull()
    if (path === "local-only") {
      expect(mocks.request).not.toHaveBeenCalled()
      expect(useWorkspaceStore.getState().currentNote.id).toBe(7)
    } else {
      expect(writes).toHaveLength(1)
      expect(writes[0].body.content).toContain("Fresh incoming answer")
      expect(writes[0].body.content).toContain("Unsaved legacy body")
    }
    expect(useWorkspaceStore.getState().currentNote.content).toContain("Fresh incoming answer")
    view.unmount()
  })

  it.each(["workspace account", "note account", "workspace-table note"])("refuses a seed into a retired or read-only %s destination", async owner => {
    await installDestination(owner === "workspace account" ? "bob" : "alice")
    legacyDraft()
    if (owner !== "workspace account") {
      const state = useWorkspaceStore.getState()
      state.setCurrentNote({ ...state.currentNote, serverScopeKey: owner === "note account" ? "bob" : "alice",
        ...(owner === "workspace-table note" ? { serverWorkspaceId: "workspace-a" } : {}) })
    }
    const original = useWorkspaceStore.getState().currentNote
    await queueSeed()
    canonicalDouble()
    const view = renderHook(() => useResearchWorkspacePrefill("workspace-a", true, true))
    await waitFor(() => expect(view.result.current.error).not.toBeNull())
    expect(mocks.request).not.toHaveBeenCalled()
    expect(useWorkspaceStore.getState().currentNote).toBe(original)
    view.unmount()
  })

  it.each(["edited", "replaced", "cleared"])("preserves %s draft during ACK and later authored edits", async change => {
    await installDestination()
    legacyDraft()
    await queueSeed()
    const write = gate()
    const writes = canonicalDouble({ writeGate: write.promise })
    const view = renderHook(() => useResearchWorkspacePrefill("workspace-a", true, true))
    try {
      await waitFor(() => expect(writes).toHaveLength(1))
      await act(async () => {
        const state = useWorkspaceStore.getState()
        if (change === "edited") state.setCurrentNote({ ...state.currentNote, content: "Live authored edit", pendingKnowledgeProvenance: { ...previousProvenance, question: "Later capture" } })
        if (change === "replaced") state.loadNote({ id: 8, title: "Replacement", content: "Replacement body", version: 2 })
        if (change === "cleared") state.clearCurrentNote()
        write.release()
      })
      await waitFor(() => expect(checkpoint(seedId)?.completed).toBe(true))
      const latest = useWorkspaceStore.getState().currentNote
      if (change === "edited") {
        expect(latest.id).toBe(seedId)
        expect(stripKnowledgeNoteProvenance(latest.content)).toBe("Live authored edit")
        expect(latest.pendingKnowledgeProvenance?.question).toBe("Later capture")
      } else if (change === "replaced") expect(latest).toMatchObject({ id: 8, content: "Replacement body", version: 2 })
      else expect(latest).toMatchObject({ id: undefined, title: "", content: "" })
      await act(async () => {
        useWorkspaceStore.getState().updateNoteContent("Authored after ACK")
        await view.result.current.retry()
      })
      expect(writes).toHaveLength(1)
      expect(useWorkspaceStore.getState().currentNote.content).toBe("Authored after ACK")
    } finally {
      write.release()
      view.unmount()
    }
  })

  it.each(["account ABA", "workspace ABA", "unmount"])("retires an old GET across %s without dropping the local draft", async boundary => {
    await installDestination()
    legacyDraft()
    await queueSeed()
    const read = gate()
    const laterRead = gate()
    const writes: BgRequestInit[] = []
    let reads = 0
    mocks.request.mockImplementation(async (request: BgRequestInit) => {
      if (request.method !== "GET") { writes.push(request); return {} }
      await (++reads === 1 ? read.promise : laterRead.promise)
      throw Object.assign(new Error("missing"), { status: 404 })
    })
    const view = renderHook(({ id }) => useResearchWorkspacePrefill(id, true, true), { initialProps: { id: "workspace-a" } })
    try {
      await waitFor(() => expect(reads).toBe(1))
      const old = mocks.request.mock.calls[0][0] as BgRequestInit
      const retainedDraft = useWorkspaceStore.getState().currentNote
      if (boundary === "account ABA") {
        await act(async () => { mocks.owner = "bob"; window.dispatchEvent(new Event("tldw:auth-principal-changed")) })
        await act(async () => { mocks.owner = "alice"; window.dispatchEvent(new Event("tldw:auth-principal-changed")) })
      } else if (boundary === "workspace ABA") {
        await act(async () => { useWorkspaceStore.setState({ workspaceId: "workspace-b" }); view.rerender({ id: "workspace-b" }) })
        await act(async () => { useWorkspaceStore.setState({ workspaceId: "workspace-a" }); view.rerender({ id: "workspace-a" }) })
      } else view.unmount()
      expect(old.abortSignal?.aborted).toBe(true)
      await act(async () => { read.release() })
      expect(writes).toHaveLength(0)
      expect(useWorkspaceStore.getState().currentNote).toBe(retainedDraft)
    } finally {
      view.unmount()
      read.release()
      laterRead.release()
    }
  })

  describe("legacy ACK corrections", () => {
    type WorkspaceNote = import("@/types/workspace").WorkspaceNote
    type CanonicalNoteFixture = Omit<WorkspaceNote, "isDirty"> & {
      id: string
      version: number
      knowledge_provenance_version: number
      knowledge_provenance: KnowledgeNoteProvenance & {
        research: NonNullable<KnowledgeNoteProvenance["research"]>
      }
    }
    const capture = (id: string): NonNullable<KnowledgeNoteProvenance["sources"]>[number] => ({ originalId: id, excerpt: "Captured excerpt", mediaId: 71,
      title: id, type: "website", sourceType: "web_capture", originalVersion: 9 })
    const withPendingCapture = () => {
      legacyDraft()
      const state = useWorkspaceStore.getState()
      const old = { origin: "reviewed_sources" as const, sources: [capture("old-clip")] }
      state.setCurrentNote({ ...state.currentNote, content: retainKnowledgeNoteProvenance(state.currentNote.content, old),
        knowledge_provenance_state: "active", knowledge_provenance: old, knowledge_provenance_version: 4,
        pendingKnowledgeProvenance: { ...old, sources: [...old.sources, capture("pending-clip")] } })
    }

    it.each([false, true])("retains distinct unconfirmed capture additions after conversion ACK (lost ACK=%s)", async lostAck => {
      await installDestination()
      withPendingCapture()
      await queueSeed()
      const writes = canonicalDouble({ lostAck })
      const view = renderHook(() => useResearchWorkspacePrefill("workspace-a", true, true))
      try {
        await waitFor(() => expect(writes).toHaveLength(1))
        if (lostAck) {
          await waitFor(() => expect(view.result.current.error).not.toBeNull())
          await act(async () => { await view.result.current.retry() })
          await waitFor(() => expect(writes).toHaveLength(2))
          expect(writes[1].body).toEqual(writes[0].body)
          expect(writes[1].headers).toEqual(writes[0].headers)
        }
        await waitFor(() => expect(checkpoint(seedId)?.completed).toBe(true))
        const latest = useWorkspaceStore.getState().currentNote
        expect(latest.pendingKnowledgeProvenance?.sources).toContainEqual(capture("pending-clip"))
        expect(latest.pendingKnowledgeProvenance?.sources).not.toContainEqual(capture("old-clip"))
        expect(latest.knowledge_provenance?.sources).not.toContainEqual(capture("old-clip"))
        expect(latest.knowledge_provenance?.sources).not.toContainEqual(capture("pending-clip"))
        expect(latest).toMatchObject({ id: seedId, isDirty: true })
        expect(latest.content).toContain("Unsaved legacy body")
        expect(latest.content).toContain("Fresh incoming answer")
      } finally { view.unmount() }
    })

    it.each(["live edit", "account retirement", "deleted readback"])("keeps pending capture ownership across %s at ACK", async boundary => {
      await installDestination()
      withPendingCapture()
      await queueSeed()
      const write = gate()
      let canonical: CanonicalNoteFixture | null = null
      const writes: BgRequestInit[] = []
      mocks.request.mockImplementation(async (request: BgRequestInit) => {
        if (request.method === "POST") {
          writes.push(request)
          canonical = { ...request.body, version: 1, knowledge_provenance_state: "active",
            knowledge_provenance: request.body.knowledge_provenance, knowledge_provenance_version: 1 }
          await write.promise
          return canonical
        }
        if (!canonical) throw Object.assign(new Error("missing"), { status: 404 })
        return boundary === "deleted readback" ? { ...canonical, knowledge_provenance_state: "deleted", knowledge_provenance: null } : canonical
      })
      const view = renderHook(() => useResearchWorkspacePrefill("workspace-a", true, true))
      try {
        await waitFor(() => expect(writes).toHaveLength(1))
        await act(async () => {
          if (boundary === "live edit") {
            const state = useWorkspaceStore.getState()
            state.setCurrentNote({ ...state.currentNote, content: "Live authored capture body",
              pendingKnowledgeProvenance: { ...state.currentNote.pendingKnowledgeProvenance!,
                sources: [...state.currentNote.pendingKnowledgeProvenance!.sources!, capture("later-clip")] } })
          }
          if (boundary === "account retirement") { mocks.owner = "bob"; window.dispatchEvent(new Event("tldw:auth-principal-changed")) }
          write.release()
        })
        if (boundary === "live edit") await waitFor(() => expect(checkpoint(seedId)?.completed).toBe(true))
        else if (boundary === "deleted readback") await waitFor(() => expect(view.result.current.error).not.toBeNull())
        const latest = useWorkspaceStore.getState().currentNote
        expect(latest.pendingKnowledgeProvenance?.sources).toContainEqual(capture("pending-clip"))
        expect(latest.isDirty).toBe(true)
        if (boundary === "live edit") {
          expect(stripKnowledgeNoteProvenance(latest.content)).toBe("Live authored capture body")
          expect(latest.pendingKnowledgeProvenance?.sources).toContainEqual(capture("later-clip"))
          expect(latest.pendingKnowledgeProvenance?.sources).not.toContainEqual(capture("old-clip"))
          expect(latest.knowledge_provenance?.sources).not.toContainEqual(capture("old-clip"))
        } else {
          expect(latest.id).toBe(7)
          expect(checkpoint(seedId)?.completed).not.toBe(true)
        }
      } finally { write.release(); view.unmount() }
    })

    const queueSeedlessReceipt = async () => {
      const { validateKnowledgeNoteProvenance } = await import("../knowledge-note-provenance")
      const transfer = payload()
      transfer.id = seedId
      transfer.query = "Legacy seed question"
      transfer.answer = "Fresh incoming answer"
      transfer.sources = []
      transfer.workspaceId = "workspace-a"
      transfer.canonicalNoteId = seedId
      transfer.legacyNoteId = 7
      transfer.draftRetained = false
      transfer.completed = false
      const history = validateKnowledgeNoteProvenance({ origin: "knowledge_qa", trust_state: transfer.answerTrustState,
        evidence_origin: transfer.answerEvidenceOrigin, thread_id: transfer.threadId, question: transfer.query,
        scope: transfer.scope, trust_reason_codes: transfer.answerTrustReasonCodes, sources: [],
        research: { workspace_id: "workspace-a", import_id: seedId, sources: [] } })!
      transfer.pendingNoteWrite = { idempotencyKey: "b0e0a1ce-3ef7-4b7a-9a93-03725fdc4a25", method: "POST",
        body: { id: seedId, title: "Unsaved legacy title", content: retainKnowledgeNoteProvenance("Unsaved legacy body", history),
          keywords: ["legacy", "workspace:workspace-a"], conversation_id: transfer.threadId,
          knowledge_provenance: history, expected_provenance_version: 0 } }
      await queueResearchWorkspacePrefill(transfer, "alice")
      return structuredClone(transfer.pendingNoteWrite)
    }
    const receiptDouble = (options: { ackGate?: Promise<void>; loseRepairAck?: boolean } = {}) => {
      const writes: BgRequestInit[] = []
      const receipts = new Map<string, CanonicalNoteFixture>()
      let canonical: CanonicalNoteFixture | null = null
      mocks.request.mockImplementation(async (request: BgRequestInit) => {
        if (request.method === "GET") {
          if (!canonical) throw Object.assign(new Error("missing"), { status: 404 })
          return canonical
        }
        writes.push(structuredClone({ ...request, abortSignal: undefined }))
        const key = request.headers?.["Idempotency-Key"] as string
        if (!receipts.has(key)) {
          if (request.method === "PUT") expect(request.headers?.["expected-version"]).toBe(String(canonical!.version))
          canonical = { ...canonical, ...request.body, version: (canonical?.version || 0) + 1,
            knowledge_provenance_state: "active", knowledge_provenance_version: 1 }
          receipts.set(key, canonical)
          if (options.loseRepairAck && request.method === "PUT") throw new Error("Lost repair acknowledgment")
        }
        if (writes.length === 1) await options.ackGate
        return receipts.get(key)
      })
      return writes
    }

    it.each([false, true])("repairs a prior seedless receipt only after immutable confirmation (lost repair ACK=%s)", async loseRepairAck => {
      await installDestination()
      legacyDraft()
      const original = useWorkspaceStore.getState().currentNote
      if (original) useWorkspaceStore.getState().captureToCurrentNote({ content: "Previously refused seed", mode: "append" })
      expect(useWorkspaceStore.getState().currentNote).toBe(original)
      const pending = await queueSeedlessReceipt()
      const ack = gate()
      const writes = receiptDouble({ ackGate: ack.promise, loseRepairAck })
      const view = renderHook(() => useResearchWorkspacePrefill("workspace-a", true, true))
      try {
        await waitFor(() => expect(writes).toHaveLength(1))
        expect(writes[0].body).toEqual(pending.body)
        expect(writes[0].headers?.["Idempotency-Key"]).toBe(pending.idempotencyKey)
        expect(checkpoint(seedId).pendingNoteWrite).toEqual(pending)
        expect(useWorkspaceStore.getState().currentNote.content).not.toContain("Fresh incoming answer")
        await act(async () => { ack.release() })
        await waitFor(() => expect(view.result.current.importing).toBe(false))
        expect(checkpoint(seedId)?.completed).not.toBe(true)
        expect(checkpoint(seedId)?.pendingNoteWrite).toBeUndefined()
        expect(useWorkspaceStore.getState().currentNote).toMatchObject({ id: seedId, isDirty: true })
        expect(useWorkspaceStore.getState().currentNote.content).toContain("Fresh incoming answer")
        expect(useWorkspaceStore.getState().currentNote.content).toContain(`Import reference: ${seedId}`)
        await act(async () => { await view.result.current.retry() })
        await waitFor(() => expect(writes).toHaveLength(2))
        expect(writes[1].method).toBe("PUT")
        expect(writes[1].headers?.["Idempotency-Key"]).not.toBe(pending.idempotencyKey)
        expect(writes[1].body.content).toContain("Unsaved legacy body")
        expect(writes[1].body.content).toContain("Fresh incoming answer")
        if (loseRepairAck) {
          await waitFor(() => expect(view.result.current.error).not.toBeNull())
          expect(checkpoint(seedId).pendingNoteWrite.body).toEqual(writes[1].body)
          await act(async () => { await view.result.current.retry() })
          await waitFor(() => expect(writes).toHaveLength(3))
          expect(writes[2].body).toEqual(writes[1].body)
          expect(writes[2].headers).toEqual(writes[1].headers)
        }
        await waitFor(() => expect(checkpoint(seedId)?.completed).toBe(true))
        expect(useWorkspaceStore.getState().currentNote.content.split(`Import reference: ${seedId}`)).toHaveLength(2)
        expect(writes[0].body).toEqual(pending.body)
      } finally { ack.release(); view.unmount() }
    })

    it.each(["edited before mount", "cleared before mount", "replaced before mount", "live edit", "clear", "replace"])("does not repair a seedless receipt over an intentional %s", async intent => {
      await installDestination()
      legacyDraft()
      const pending = await queueSeedlessReceipt()
      const change = () => {
        const state = useWorkspaceStore.getState()
        if (intent.includes("edited") || intent === "live edit") state.updateNoteContent("Intentional authored body")
        else if (intent.includes("clear")) state.clearCurrentNote()
        else state.loadNote({ id: 8, title: "Replacement", content: "Replacement body", version: 3 })
      }
      if (intent.includes("before mount")) change()
      const ack = gate()
      const writes = receiptDouble({ ackGate: ack.promise })
      const view = renderHook(() => useResearchWorkspacePrefill("workspace-a", true, true))
      try {
        await waitFor(() => expect(writes).toHaveLength(1))
        let retained!: WorkspaceNote
        await act(async () => { if (!intent.includes("before mount")) change(); retained = useWorkspaceStore.getState().currentNote; ack.release() })
        await waitFor(() => expect(view.result.current.importing).toBe(false))
        expect(useWorkspaceStore.getState().currentNote).toBe(retained)
        expect(useWorkspaceStore.getState().currentNote.content).not.toContain("Fresh incoming answer")
        expect(checkpoint(seedId)?.completed).not.toBe(true)
        expect(writes).toHaveLength(1)
        expect(writes[0].body).toEqual(pending.body)
        expect(writes[0].headers?.["Idempotency-Key"]).toBe(pending.idempotencyKey)
        await act(async () => { await view.result.current.retry() })
        await waitFor(() => expect(view.result.current.importing).toBe(false))
        expect(useWorkspaceStore.getState().currentNote).toBe(retained)
        expect(checkpoint(seedId)?.completed).not.toBe(true)
        expect(writes).toHaveLength(1)
      } finally { ack.release(); view.unmount() }
    })

    describe("round3 connected receipt corrections", () => {
      it.each([false, true])("repairs a seedless lost receipt after failed source attachment recovers (lost repair ACK=%s)", async loseRepairAck => {
        await installDestination()
        legacyDraft()
        await queueSeedlessReceipt()
        const transfer = (await consumeResearchWorkspacePrefill("alice"))!
        transfer.sources = payload().sources.slice(0, 2)
        for (const source of transfer.sources) source.importError = "Could not import the full note. Retry when the server is available."
        const oldHistory = {
          ...(transfer.pendingNoteWrite!.body.knowledge_provenance as KnowledgeNoteProvenance),
          sources: transfer.sources.map(({ importError: _error, ...source }) => source),
        }
        transfer.pendingNoteWrite!.body.knowledge_provenance = oldHistory
        transfer.pendingNoteWrite!.body.content = retainKnowledgeNoteProvenance("Unsaved legacy body", oldHistory)
        await queueResearchWorkspacePrefill(transfer, "alice")
        const pending = structuredClone(transfer.pendingNoteWrite!)
        let canonical: CanonicalNoteFixture = { ...pending.body as CanonicalNoteFixture, version: 1, knowledge_provenance_state: "active",
          knowledge_provenance_version: 1, knowledge_provenance_hash: `sha256:${"a".repeat(64)}` }
        const receipts = new Map<string, CanonicalNoteFixture>([[pending.idempotencyKey, canonical]])
        const writes: BgRequestInit[] = []
        mocks.upload.mockRejectedValueOnce(new Error("Snapshot unavailable"))
          .mockResolvedValue({ media_id: 101 })
        mocks.request.mockImplementation(async (request: BgRequestInit) => {
          expect(request.servicePromptConfig?.serverUrl).toBe("alice")
          if (request.method === "GET") {
            expect(request.path).toBe(`/api/v1/notes/${seedId}`)
            return canonical
          }
          writes.push(structuredClone({ ...request, abortSignal: undefined }))
          const key = request.headers?.["Idempotency-Key"] as string
          if (!receipts.has(key)) {
            expect(request.method).toBe("PUT")
            expect(request.headers?.["expected-version"]).toBe(String(canonical.version))
            canonical = { ...canonical, ...request.body, version: canonical.version + 1,
              knowledge_provenance_version: canonical.knowledge_provenance_version + 1 }
            receipts.set(key, canonical)
            if (loseRepairAck) throw new Error("Lost repair acknowledgment")
          }
          if (writes.length === 1) throw new Error("Lost original acknowledgment")
          return receipts.get(key)
        })
        const view = renderHook(() => useResearchWorkspacePrefill("workspace-a", true, true))
        try {
          await waitFor(() => expect(view.result.current.error).not.toBeNull())
          expect(checkpoint(seedId).sources[0].importError).toBeTruthy()
          expect(checkpoint(seedId).pendingNoteWrite).toEqual(pending)
          expect(writes).toHaveLength(1)
          expect(useWorkspaceStore.getState().currentNote).toMatchObject({ id: 7, content: "Unsaved legacy body", isDirty: true })
          await act(async () => { await view.result.current.retry() })
          await waitFor(() => expect(view.result.current.importing).toBe(false))
          expect(writes).toHaveLength(2)
          expect(writes[0].body).toEqual(pending.body)
          expect(writes[1].body).toEqual(pending.body)
          expect(writes[1].headers).toEqual(writes[0].headers)
          expect(writes[1].headers?.["Idempotency-Key"]).toBe(pending.idempotencyKey)
          expect(checkpoint(seedId).sources[0]).toMatchObject({ snapshotMediaId: 101, originalVersion: 4 })
          expect(checkpoint(seedId).sources[0].importError).toBeUndefined()
          expect(checkpoint(seedId).pendingNoteWrite).toBeUndefined()
          expect(checkpoint(seedId).completed).not.toBe(true)
          const staged = useWorkspaceStore.getState().currentNote
          expect(staged).toMatchObject({ id: seedId, isDirty: true })
          expect(staged.content).toContain("Fresh incoming answer")
          expect(staged.content).toContain(`Import reference: ${seedId}`)
          expect(staged.knowledge_provenance?.research?.sources).toEqual([])
          await act(async () => { await view.result.current.retry() })
          await waitFor(() => expect(writes).toHaveLength(3))
          expect(writes[2].method).toBe("PUT")
          expect(writes[2].headers?.["Idempotency-Key"]).not.toBe(pending.idempotencyKey)
          expect(writes[2].body.content).toContain("Unsaved legacy body")
          expect(writes[2].body.content).toContain("Fresh incoming answer")
          expect(writes[2].body.content).toContain(`Import reference: ${seedId}`)
          expect(writes[2].body.knowledge_provenance.sources).toMatchObject([
            { originalId: "note-uuid", snapshotMediaId: 101, originalVersion: 4 },
            { originalId: "note-uuid", snapshotMediaId: 101, originalVersion: 4 },
          ])
          expect(writes[2].body.knowledge_provenance.research.sources).toMatchObject([
            { mediaId: 101, evidence: { snapshot: true, importId: seedId } },
          ])
          if (loseRepairAck) {
            await waitFor(() => expect(view.result.current.error).not.toBeNull())
            expect(checkpoint(seedId).completed).not.toBe(true)
            await act(async () => { await view.result.current.retry() })
            await waitFor(() => expect(writes).toHaveLength(4))
            expect(writes[3].body).toEqual(writes[2].body)
            expect(writes[3].headers).toEqual(writes[2].headers)
          }
          await waitFor(() => expect(checkpoint(seedId).completed).toBe(true))
          expect(view.result.current.error).toBeNull()
          expect(useWorkspaceStore.getState().currentNote.content.split(`Import reference: ${seedId}`)).toHaveLength(2)
          expect(canonical.knowledge_provenance.research.sources).toHaveLength(1)
          expect(mocks.upload).toHaveBeenCalledTimes(2)
          expect(writes[0].body).toEqual(pending.body)
        } finally { view.unmount() }
      })

      it.each([
        { label: "keyword-only edit", pendingCapture: false, edit: "changed" },
        { label: "keyword-only edit with pending capture", pendingCapture: true, edit: "changed" },
        { label: "retained nondirty receipt", pendingCapture: false, edit: "nondirty" },
        { label: "normalized equivalent authored keywords", pendingCapture: false, edit: "normalized" },
      ])("settles lost ACK without overwriting $label", async ({ pendingCapture, edit }) => {
        await installDestination()
        if (pendingCapture) withPendingCapture()
        else {
          const draft = legacyDraft()
          useWorkspaceStore.getState().setCurrentNote({ ...draft, pendingKnowledgeProvenance: undefined })
        }
        await queueSeed()
        const writes = canonicalDouble({ lostAck: true })
        const view = renderHook(() => useResearchWorkspacePrefill("workspace-a", true, true))
        try {
          await waitFor(() => expect(view.result.current.error).not.toBeNull())
          const beforeRetry = useWorkspaceStore.getState().currentNote
          await act(async () => {
            const state = useWorkspaceStore.getState()
            if (edit === "changed") state.updateNoteKeywords(["new-authored-tag"])
            else if (edit === "normalized") state.updateNoteKeywords([" legacy ", "legacy", "", state.workspaceTag, "workspace:workspace-a"])
            else state.setCurrentNote({ ...state.currentNote, isDirty: false })
            await view.result.current.retry()
          })
          await waitFor(() => expect(checkpoint(seedId)?.completed).toBe(true))
          expect(writes).toHaveLength(2)
          expect(writes[1].body).toEqual(writes[0].body)
          expect(writes[1].headers).toEqual(writes[0].headers)
          const latest = useWorkspaceStore.getState().currentNote
          expect(latest.title).toBe(beforeRetry.title)
          expect(stripKnowledgeNoteProvenance(latest.content)).toBe(stripKnowledgeNoteProvenance(beforeRetry.content))
          expect(latest).toMatchObject({ id: seedId, knowledge_provenance_state: "active", knowledge_provenance_version: 1 })
          expect(latest.keywords).toEqual(edit === "changed" ? ["new-authored-tag"] : ["legacy"])
          expect(latest.isDirty).toBe(edit === "changed")
          if (pendingCapture) {
            expect(latest.pendingKnowledgeProvenance?.sources).toContainEqual(capture("pending-clip"))
            expect(latest.pendingKnowledgeProvenance?.sources).not.toContainEqual(capture("old-clip"))
            expect(latest.knowledge_provenance?.sources).not.toContainEqual(capture("pending-clip"))
          } else expect(latest.pendingKnowledgeProvenance).toBeUndefined()
          await act(async () => { await view.result.current.retry() })
          expect(writes).toHaveLength(2)
          expect(useWorkspaceStore.getState().currentNote).toBe(latest)
        } finally { view.unmount() }
      })
    })

    describe("recovery journal checkpoint windows", () => {
      const pausedCheckpoint = () => {
        const checkpointGate = gate()
        const writes = receiptDouble()
        const send = mocks.request.getMockImplementation()!
        let entered = false
        mocks.request.mockImplementation(async (request: BgRequestInit) => {
          const result = await send(request)
          if (!entered && request.method === "GET" && writes.length === 1) {
            entered = true
            mocks.readGate = checkpointGate.promise
          }
          return result
        })
        return { checkpointGate, writes, hasEntered: () => entered }
      }

      it.each(["storage rejection", "account retirement", "unmount"] as const)(
        "recovers the immutable seedless receipt across $0 at checkpoint",
        async boundary => {
          await installDestination()
          const original = legacyDraft()
          const pending = await queueSeedlessReceipt()
          const paused = pausedCheckpoint()
          let view = renderHook(() => useResearchWorkspacePrefill("workspace-a", true, true))
          try {
            await waitFor(() => expect(paused.hasEntered()).toBe(true))
            expect(paused.writes).toHaveLength(1)
            expect(checkpoint(seedId).pendingNoteWrite).toEqual(pending)
            expect(useWorkspaceStore.getState().currentNote).toBe(original)
            if (boundary === "storage rejection") {
              mocks.writeError = true
              await act(async () => { paused.checkpointGate.release() })
              await waitFor(() => expect(view.result.current.error).not.toBeNull())
              expect(checkpoint(seedId).pendingNoteWrite).toEqual(pending)
              mocks.writeError = false
              mocks.readGate = null
              await act(async () => { await view.result.current.retry() })
            } else {
              if (boundary === "account retirement") {
                await act(async () => {
                  mocks.owner = "bob"
                  window.dispatchEvent(new Event("tldw:auth-principal-changed"))
                })
              }
              view.unmount()
              mocks.readGate = null
              await act(async () => { paused.checkpointGate.release() })
              await waitFor(() => expect(checkpoint(seedId).pendingNoteWrite).toBeUndefined())
              expect(useWorkspaceStore.getState().currentNote).toBe(original)
              expect(original.content).not.toContain("Fresh incoming answer")
              if (boundary === "account retirement") {
                await act(async () => {
                  mocks.owner = "alice"
                  window.dispatchEvent(new Event("tldw:auth-principal-changed"))
                })
              }
              view = renderHook(() => useResearchWorkspacePrefill("workspace-a", true, true))
            }
            await waitFor(() => expect(view.result.current.error).not.toBeNull())
            expect(paused.writes).toHaveLength(2)
            expect(paused.writes[1].body).toEqual(pending.body)
            expect(paused.writes[1].headers).toEqual(paused.writes[0].headers)
            expect(paused.writes[1].headers?.["Idempotency-Key"]).toBe(pending.idempotencyKey)
            expect(checkpoint(seedId).completed).not.toBe(true)
            expect(checkpoint(seedId).pendingNoteWrite).toBeUndefined()
            const staged = useWorkspaceStore.getState().currentNote
            expect(staged).toMatchObject({ id: seedId, isDirty: true })
            expect(staged.content).toContain("Unsaved legacy body")
            expect(staged.content).toContain("Fresh incoming answer")
            expect(staged.content).toContain(`Import reference: ${seedId}`)
            await act(async () => { await view.result.current.retry() })
            await waitFor(() => expect(checkpoint(seedId).completed).toBe(true))
            expect(view.result.current.error).toBeNull()
            expect(paused.writes).toHaveLength(3)
            expect(paused.writes[2].method).toBe("PUT")
            expect(paused.writes[2].headers?.["Idempotency-Key"]).not.toBe(pending.idempotencyKey)
            expect(paused.writes[2].headers?.["expected-version"]).toBe("1")
            expect(paused.writes[2].body.content).toContain("Fresh incoming answer")
            expect(useWorkspaceStore.getState().currentNote.content.split(`Import reference: ${seedId}`)).toHaveLength(2)
            expect(useWorkspaceStore.getState().serverWorkspace?.scopeKey).toBe("alice")
            const settled = useWorkspaceStore.getState().currentNote
            useWorkspaceStore.getState().captureToCurrentNote({ content: "Forbidden manual capture", mode: "append" })
            expect(useWorkspaceStore.getState().currentNote).toBe(settled)
            expect(paused.writes[0].body).toEqual(pending.body)
          } finally {
            mocks.writeError = false
            mocks.readGate = null
            paused.checkpointGate.release()
            view.unmount()
          }
        },
      )

      it.each(["edit", "clear", "replace"] as const)(
        "does not stage or replay over intentional $0 during checkpoint",
        async intent => {
          await installDestination()
          legacyDraft()
          const pending = await queueSeedlessReceipt()
          const paused = pausedCheckpoint()
          const view = renderHook(() => useResearchWorkspacePrefill("workspace-a", true, true))
          try {
            await waitFor(() => expect(paused.hasEntered()).toBe(true))
            let authored!: WorkspaceNote
            await act(async () => {
              const state = useWorkspaceStore.getState()
              if (intent === "edit") state.updateNoteContent("Intentional checkpoint edit")
              else if (intent === "clear") state.clearCurrentNote()
              else state.loadNote({ id: 8, title: "Replacement", content: "Replacement body", version: 3 })
              authored = useWorkspaceStore.getState().currentNote
              mocks.readGate = null
              paused.checkpointGate.release()
            })
            await waitFor(() => expect(view.result.current.error).not.toBeNull())
            expect(useWorkspaceStore.getState().currentNote).toBe(authored)
            expect(authored.content).not.toContain("Fresh incoming answer")
            expect(checkpoint(seedId).completed).not.toBe(true)
            await act(async () => { await view.result.current.retry() })
            await waitFor(() => expect(view.result.current.error).not.toBeNull())
            expect(useWorkspaceStore.getState().currentNote).toBe(authored)
            expect(paused.writes).toHaveLength(1)
            expect(paused.writes[0].body).toEqual(pending.body)
          } finally {
            mocks.readGate = null
            paused.checkpointGate.release()
            view.unmount()
          }
        },
      )
    })

    describe("repair checkpoint selection race", () => {
      type Checkpoint = NonNullable<Awaited<ReturnType<typeof consumeResearchWorkspacePrefill>>> & {
        confirmedSeedlessWrite?: NonNullable<NonNullable<Awaited<ReturnType<typeof consumeResearchWorkspacePrefill>>>["pendingNoteWrite"]>
      }

      it.each([false, true])("replays the exact lost repair receipt after selection during checkpoint=%s", async changeSelection => {
        await installDestination()
        legacyDraft()
        useWorkspaceStore.getState().addSources([
          { mediaId: 99, title: "Ready selection source", type: "text", status: "ready" },
        ])
        const source = useWorkspaceStore.getState().sources.find(item => item.mediaId === 99)!
        expect(source.status).toBe("ready")
        const originalReceipt = await queueSeedlessReceipt()
        const checkpoints: Checkpoint[] = []
        const set = mocks.values.set.bind(mocks.values)
        const observeSet = vi.spyOn(mocks.values, "set").mockImplementation((key, value) => {
          if (value && typeof value === "object" && "id" in value && value.id === seedId)
            checkpoints.push(structuredClone(value) as Checkpoint)
          return set(key, value)
        })
        const writes = receiptDouble({ loseRepairAck: true })
        const send = mocks.request.getMockImplementation()!
        const repairGate = gate()
        let holdRepair = false
        let checkpointHeld = false
        mocks.request.mockImplementation(async (request: BgRequestInit) => {
          const result = await send(request)
          if (holdRepair && !checkpointHeld && request.method === "GET") {
            checkpointHeld = true
            mocks.readGate = repairGate.promise
          }
          return result
        })
        let view = renderHook(() => useResearchWorkspacePrefill("workspace-a", true, true))
        try {
          await waitFor(() => expect(view.result.current.error).not.toBeNull())
          expect(writes).toHaveLength(1)
          expect(writes[0].body).toEqual(originalReceipt.body)
          const staged = useWorkspaceStore.getState().currentNote
          expect(staged.content).toContain("Fresh incoming answer")
          expect(staged.content).toContain(`Import reference: ${seedId}`)
          expect(checkpoint(seedId).pendingNoteWrite).toBeUndefined()
          expect(checkpoint(seedId).completed).not.toBe(true)
          holdRepair = true
          await act(async () => { await view.result.current.retry() })
          await waitFor(() => expect(checkpointHeld).toBe(true))
          expect(writes).toHaveLength(1)
          expect(useWorkspaceStore.getState().selectedSourceIds).toEqual([])
          if (changeSelection) {
            await act(async () => { useWorkspaceStore.getState().toggleSourceSelection(source.id) })
            expect(useWorkspaceStore.getState().selectedSourceIds).toEqual([source.id])
          }
          mocks.readGate = null
          await act(async () => { repairGate.release() })
          await waitFor(() => expect(writes).toHaveLength(2))
          await waitFor(() => expect(view.result.current.error).not.toBeNull())
          const repair = writes[1]
          expect(repair.method).toBe("PUT")
          expect(repair.headers?.["expected-version"]).toBe("1")
          expect(repair.headers?.["Idempotency-Key"]).not.toBe(originalReceipt.idempotencyKey)
          expect(repair.body.content).toContain("Fresh incoming answer")
          const persistedBeforeRemount = await consumeResearchWorkspacePrefill("alice") as Checkpoint
          expect(persistedBeforeRemount).toMatchObject({ ownerScope: "alice", workspaceId: "workspace-a", completed: false })
          expect(checkpoints.some(item => item.pendingNoteWrite?.idempotencyKey === repair.headers?.["Idempotency-Key"])).toBe(true)
          view.unmount()
          view = renderHook(() => useResearchWorkspacePrefill("workspace-a", true, true))
          await waitFor(() => expect(writes).toHaveLength(3))
          const replay = writes[2]
          console.info("repair-selection-race trace", JSON.stringify({
            changeSelection,
            originalKey: originalReceipt.idempotencyKey,
            repairKey: repair.headers?.["Idempotency-Key"],
            replayKey: replay.headers?.["Idempotency-Key"],
            repairExpectedVersion: repair.headers?.["expected-version"],
            replayExpectedVersion: replay.headers?.["expected-version"],
            repairBody: repair.body,
            replayBody: replay.body,
            persistedBeforeRemount,
            checkpoints: checkpoints.map(item => ({
              pendingKey: item.pendingNoteWrite?.idempotencyKey ?? null,
              confirmedKey: item.confirmedSeedlessWrite?.idempotencyKey ?? null,
              selectionIntent: item.selectionIntent,
            })),
          }))
          expect(replay.headers?.["Idempotency-Key"]).toBe(repair.headers?.["Idempotency-Key"])
          expect(replay.body).toEqual(repair.body)
          expect(replay.headers).toEqual(repair.headers)
          await waitFor(() => expect(checkpoint(seedId).completed).toBe(true))
          expect(view.result.current.error).toBeNull()
          expect(useWorkspaceStore.getState().currentNote.content.split(`Import reference: ${seedId}`)).toHaveLength(2)
        } finally {
          mocks.readGate = null
          repairGate.release()
          view.unmount()
          observeSet.mockRestore()
        }
      })
    })

    describe("field-scoped shared prefill writer", () => {
      it.each(["missing", "replaced", "foreign-owner"] as const)(
        "refuses a selection-only overwrite of a $0 handoff",
        async condition => {
          const { saveResearchWorkspacePrefill } = await import("../research-workspace-prefill")
          await queueSeedlessReceipt()
          const original = (await consumeResearchWorkspacePrefill("alice"))!
          const [key] = [...mocks.values.entries()].find(([, value]) =>
            value && typeof value === "object" && "id" in value && value.id === seedId,
          )!
          if (condition === "missing") mocks.values.delete(key)
          else mocks.values.set(key, {
            ...original,
            ...(condition === "replaced" ? { id: "newer-handoff" } : { ownerScope: "bob" }),
          })
          const before = structuredClone([...mocks.values.entries()])
          await expect(saveResearchWorkspacePrefill({ ...original, selectionIntent: null }, "selection")).rejects.toThrow()
          expect([...mocks.values.entries()]).toEqual(before)
        },
      )

      it("changes only selection intent while preserving immutable pending and journal records", async () => {
        const { saveResearchWorkspacePrefill } = await import("../research-workspace-prefill")
        const journal = await queueSeedlessReceipt()
        const stale = (await consumeResearchWorkspacePrefill("alice"))!
        const current = {
          ...stale,
          query: "Current required question",
          sources: payload().sources.slice(3),
          selectionIntent: { mediaIds: [7], selectedSourceIds: ["ready-7"] },
          pendingNoteWrite: {
            ...journal, idempotencyKey: "immutable-repair-key", method: "PUT" as const,
            expectedVersion: 1, body: { title: "Repair title", content: "Full repair body" },
          },
          confirmedSeedlessWrite: journal,
        }
        await saveResearchWorkspacePrefill(current)
        expect(await consumeResearchWorkspacePrefill("alice")).toEqual(current)
        await saveResearchWorkspacePrefill({ ...stale, selectionIntent: null }, "selection")
        expect(await consumeResearchWorkspacePrefill("alice")).toEqual({ ...current, selectionIntent: null })
        await saveResearchWorkspacePrefill(current, "all")
        expect(await consumeResearchWorkspacePrefill("alice")).toEqual(current)
      })
    })
  })
})
