import type { TldwConfig } from "@/services/tldw/TldwApiClient"
import type { WorkspaceNote } from "@/types/workspace"
import type { ResearchWorkspacePrefill } from "../research-workspace-prefill"
import type { BgRequestInit } from "@/services/background-proxy"
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

type CanonicalNote = Omit<WorkspaceNote, "isDirty"> & {
  id: string | number;
  conversation_id?: string | null;
};

const mocks = vi.hoisted(() => ({
  owner: "alice",
  answerState: {} as Record<string, unknown>,
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
}));
vi.mock("@/services/chat-surface-scope", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/services/chat-surface-scope")>()),
  buildChatSurfaceScopeKeyFromConfig: (
    config: TldwConfig,
    options?: { userId?: string | number | null },
  ) =>
    config.serverUrl +
    (config.authMode === "multi-user"
      ? `:${options?.userId ?? (config.accessToken ? "user-a" : "anonymous")}`
      : ""),
}));
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
vi.mock("@/services/tldw/TldwAuth", () => ({
  tldwAuth: {
    getCurrentUser: async () => ({
      id: mocks.multiUser ? "user-a" : "single-owner",
      is_active: true,
    }),
  },
}));
vi.mock("@/utils/safe-storage", () => ({
  safeStorageSerde: { deserializer: (value: unknown) => value },
  createSafeStorage: () => ({
    get hasPersistentBackend() {
      return mocks.persistent;
    },
    get: async (key: string) => {
      await mocks.readGate;
      return structuredClone(mocks.values.get(key));
    },
    getAll: async () => Object.fromEntries(mocks.values),
    remove: async (key: string) => {
      mocks.values.delete(key);
    },
    set: async (key: string, value: unknown) => {
      if (mocks.writeError) throw new Error("storage full");
      mocks.values.set(key, structuredClone(value));
    },
  }),
}));
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, options?: string | { defaultValue?: string }) =>
      typeof options === "string" ? options : options?.defaultValue || key,
  }),
}));
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
}));
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
    }));
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
    let canonical: CanonicalNote = null;
    let droppedResponse = false
    mocks.request.mockImplementation(async (request: BgRequestInit) => {
      const { path, method, body, headers } = request;
      if (path === "/api/v1/notes/" && method === "POST") {
        canonical = {
          id: body.id,
          title: body.title,
          content: body.content,
          keywords: body.keywords.map((keyword: string) => ({ keyword })),
          conversation_id: body.conversation_id,
          version: 1,
        };
        if (loseResponse && !droppedResponse) {
          droppedResponse = true;
          throw new Error("response lost");
        }
        return canonical;
      }
      if (path.startsWith("/api/v1/notes/search/"))
        return { notes: canonical ? [canonical] : [] };
      if (path.startsWith("/api/v1/notes/")) {
        if (!canonical)
          throw Object.assign(new Error("missing"), { status: 404 });
        if (method === "PUT") {
          if (headers?.["expected-version"] !== String(canonical.version))
            throw Object.assign(new Error("expected-version header required"), {
              status: 422,
            });
          canonical = {
            ...canonical,
            ...body,
            version: canonical.version + 1,
          };
        }
        return canonical;
      }
      if (path.endsWith("/context"))
        return {
          workspace_id: "workspace-a",
          workspace,
          sources: { items: serverSources },
          partial_errors: [],
        };
      return [];
    });
    const restore = () =>
      restoreMigratedResearchWorkspace({
        signal: new AbortController().signal,
        apply: useWorkspaceStore.getState().restoreServerWorkspace,
      })
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
    useWorkspaceStore.setState({
      workspaceId: null,
      sources: [],
      currentNote: { title: "", content: "", keywords: [], isDirty: false },
      workspaceSnapshots: {},
      selectedSourceIds: [],
    })
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
      }),
    )
    let finish!: () => void
    let canonical: CanonicalNote = null;
    mocks.request.mockImplementation(
      async ({ method, body }: BgRequestInit) => {
        if (method === "POST") {
          canonical = {
            id: body.id,
            title: body.title,
            content: body.content,
            keywords: body.keywords.map((keyword: string) => ({ keyword })),
            version: 1,
          };
          await new Promise<void>((resolve) => {
            finish = resolve;
          });
          return canonical;
        }
        if (!canonical)
          throw Object.assign(new Error("missing"), { status: 404 });
        return canonical;
      },
    );
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
  mocks.request.mockImplementation(async ({ method, body }: BgRequestInit) => {
    if (method === "PUT")
      canonical = {
        ...canonical,
        title: body.title,
        content: body.content,
        keywords: body.keywords.map((keyword: string) => ({ keyword })),
        version: 2,
      };
    return canonical;
  });
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
    let canonical: CanonicalNote = null;
    let finishRetry!: () => void
    mocks.request.mockImplementation(
      async ({ method, body }: BgRequestInit) => {
        if (method === "POST") {
          canonical = {
            id: body.id,
            title: body.title,
            content: body.content,
            keywords: body.keywords.map((keyword: string) => ({ keyword })),
            version: 1,
          };
          if (failure === "lost response")
            throw new Error("response lost after commit");
        }
        if (method === "PUT") {
          canonical = {
            ...canonical,
            title: body.title,
            content: body.content,
            keywords: body.keywords.map((keyword: string) => ({ keyword })),
            version: 2,
          };
          await new Promise<void>((resolve) => {
            finishRetry = resolve;
          });
        }
        if (!canonical)
          throw Object.assign(new Error("missing"), { status: 404 });
        return canonical;
      },
    );
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
    let canonical: CanonicalNote = null;
    let finishCreate!: () => void
    mocks.request.mockImplementation(async (request: BgRequestInit) => {
      const { path, method, body, headers } = request;
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
        };
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
        ];
      if (path === "/api/v1/notes/" && method === "POST") {
        canonical = {
          id: body.id,
          title: body.title,
          content: body.content,
          keywords: body.keywords.map((keyword: string) => ({ keyword })),
          version: 1,
        };
        await new Promise<void>((resolve) => {
          finishCreate = resolve;
        });
        return canonical;
      }
      if (path.startsWith("/api/v1/notes/search/")) return { notes: [] };
      if (path.startsWith("/api/v1/notes/keywords/")) return [];
      if (path.startsWith("/api/v1/notes/")) {
        if (
          !canonical ||
          path.split("?")[0] !== `/api/v1/notes/${canonical.id}`
        )
          throw Object.assign(new Error("missing"), { status: 404 });
        if (method === "PUT") {
          if (headers?.["expected-version"] !== String(canonical.version))
            throw Object.assign(new Error("expected-version header required"), {
              status: 422,
            });
          canonical = {
            ...canonical,
            title: body.title,
            content: body.content,
            keywords: body.keywords.map((keyword: string) => ({ keyword })),
            version: canonical.version + 1,
          };
        }
        return canonical;
      }
      return [];
    });
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
    if (change === "unversioned")
      useWorkspaceStore
        .getState()
        .setCurrentNote({ ...note, version: undefined })
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
      }),
    )
    useWorkspaceStore.getState().loadNote({
      id: 7,
      title: "Original legacy note",
      content: "Original legacy body",
    })
    let canonical: CanonicalNote = null;
    mocks.request.mockImplementation(
      async ({ method, body, headers }: BgRequestInit) => {
        if (method === "POST") {
          canonical = { ...body, version: 1 };
          throw new Error("response lost after commit");
        }
        if (method === "PUT") {
          if (headers?.["expected-version"] !== String(canonical.version))
            throw Object.assign(new Error("expected-version header required"), {
              status: 422,
            });
          canonical = { ...canonical, ...body, version: 2 };
        }
        if (!canonical)
          throw Object.assign(new Error("missing"), { status: 404 });
        return canonical;
      },
    );
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
const previousProvenance = {
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
  ([...mocks.values.values()] as ResearchWorkspacePrefill[]).find(
    (value) => value.id === id,
  );

it.each([false, true])(
  "persists the actual AnswerPanel default scope and current import over old marker=%s through canonical restore",
  async (withOldMarker) => {
    markMigrated()
    let canonical: CanonicalNote = withOldMarker
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
      : null;
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
    mocks.request.mockImplementation(async (request: BgRequestInit) => {
      const { path, method, body, headers } = request;
      if (path.endsWith("/context"))
        return {
          workspace_id: "workspace-a",
          workspace: {
            id: "workspace-a",
            name: "Research",
            created_at: "2026-10-01T00:00:00Z",
            version: 1,
          },
          sources: { items: sources },
          partial_errors: [],
        };
      if (path.startsWith("/api/v1/notes/search/"))
        return { notes: canonical ? [canonical] : [] };
      if (!path.startsWith("/api/v1/notes/")) return [];
      if (method === "POST") canonical = { ...body, version: 1 };
      if (method === "PUT") {
        if (headers?.["expected-version"] !== String(canonical.version))
          throw Object.assign(new Error("version required"), { status: 422 });
        canonical = { ...canonical, ...body, version: canonical.version + 1 };
      }
      if (!canonical)
        throw Object.assign(new Error("missing"), { status: 404 });
      return canonical;
    });
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
    useWorkspaceStore.setState({
      workspaceId: null,
      sources: [],
      currentNote: { title: "", content: "", keywords: [], isDirty: false },
      workspaceSnapshots: {},
      selectedSourceIds: [],
    })
    const { restoreMigratedResearchWorkspace } =
      await import("@/components/Option/ResearchWorkspace/workspace-server-restore")
    await restoreMigratedResearchWorkspace({
      signal: new AbortController().signal,
      apply: useWorkspaceStore.getState().restoreServerWorkspace,
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
    const seen: BgRequestInit[] = [];
    mocks.request.mockImplementation(async (request: BgRequestInit) => {
      seen.push(request);
      const expectedUser = request.headers?.["X-TLDW-Expected-User-ID"];
      if (expectedUser && expectedUser !== principal)
        throw Object.assign(new Error("request_config_scope_changed"), {
          status: 412,
        });
      if (request.method === "PUT") {
        if (request.headers?.["expected-version"] !== "2")
          throw Object.assign(new Error("version required"), { status: 422 });
        mutations += 1;
        canonical = { ...canonical, ...request.body, version: 3 };
      } else if (changePrincipal) {
        capturedDraft = useWorkspaceStore.getState().currentNote;
        principal = "user-b";
      }
      return canonical;
    });
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
  let canonical: CanonicalNote = null;
  mocks.request.mockImplementation(async ({ method, body }: BgRequestInit) => {
    if (method === "POST") canonical = { ...body, version: 1 };
    if (!canonical) throw Object.assign(new Error("missing"), { status: 404 });
    return canonical;
  });
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
  let canonical: CanonicalNote = null;
  mocks.request.mockImplementation(async ({ method, body }: BgRequestInit) => {
    if (method === "POST")
      canonical = {
        ...body,
        version: 1,
        content: stripKnowledgeNoteProvenance(body.content),
        knowledge_provenance_state: "active",
        knowledge_provenance_version: 1,
        knowledge_provenance_hash: `sha256:${"a".repeat(64)}`,
      };
    if (!canonical) throw Object.assign(new Error("missing"), { status: 404 });
    return canonical;
  });
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
    expect(
      (mocks.values.values().next().value as ResearchWorkspacePrefill)
        ?.completed,
    ).toBe(true),
  );
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
  const client = tldwClient as unknown as Pick<
    typeof tldwClient,
    "getWorkspaceSources" | "addWorkspaceSource"
  > & { upsertWorkspace: ReturnType<typeof vi.fn> };
  let confirm!: (value: unknown) => void
  client.upsertWorkspace = vi.fn(
    () =>
      new Promise((resolve) => {
        confirm = resolve
      }),
  )
  client.getWorkspaceSources = vi.fn().mockResolvedValue([])
  client.addWorkspaceSource = vi.fn().mockResolvedValue({})
  let canonical: CanonicalNote = null;
  mocks.request.mockImplementation(async ({ method, body }: BgRequestInit) => {
    if (method === "POST") canonical = { ...body, version: 1 };
    if (!canonical) throw Object.assign(new Error("missing"), { status: 404 });
    return canonical;
  });
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
    let canonical: CanonicalNote = null;
    mocks.request.mockImplementation(
      async ({ method, body }: BgRequestInit) => {
        if (method === "POST") canonical = { ...body, version: 1 };
        if (!canonical)
          throw Object.assign(new Error("missing"), { status: 404 });
        return canonical;
      },
    );
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
