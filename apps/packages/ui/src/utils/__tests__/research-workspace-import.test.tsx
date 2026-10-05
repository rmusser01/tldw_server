import { act, render, renderHook, waitFor } from "@testing-library/react"
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
vi.mock("@/services/background-proxy", () => ({ bgRequest: async () => ({}) }))
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
  mocks.writeError = false
  mocks.readGate = null
  localStorage.clear()
  useWorkspaceStore.setState({
    workspaceId: "workspace-a",
    sources: [],
    selectedSourceIds: [],
    currentNote: { title: "", content: "", keywords: [], isDirty: false },
    storeHydrated: true,
  })
})
describe("mounted Knowledge research import", () => {
  it.each([false, true])(
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
          useWorkspaceStore.getState().setSelectedSourceIds([unrelatedId])
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
      const state = useWorkspaceStore.getState()
      expect(
        state.sources
          .filter((source) => state.selectedSourceIds.includes(source.id))
          .map((source) => source.mediaId),
      ).toEqual(manuallySelected ? [50] : [101])
      view.unmount()
    },
  )
  it("replaces unrelated active sources with the transferred set through partial retry", async () => {
    const state = useWorkspaceStore.getState()
    state.addSources([
      { mediaId: 50, title: "Unrelated library source", type: "text" },
    ])
    const unrelatedId = useWorkspaceStore.getState().sources[0].id
    state.setSelectedSourceIds([unrelatedId])
    state.captureToCurrentNote({ content: "Existing draft", mode: "append" })
    const selectedMediaIds = () => {
      const current = useWorkspaceStore.getState()
      return current.sources
        .filter((source) => current.selectedSourceIds.includes(source.id))
        .map((source) => source.mediaId)
    }
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
    const handoff = payload()
    handoff.sources = handoff.sources.filter((source) => source.mediaId == null)
    await queueResearchWorkspacePrefill(handoff, "alice")
    mocks.upload.mockRejectedValue(new Error("offline"))
    const view = renderHook(() =>
      useResearchWorkspacePrefill("workspace-a", true),
    )
    await waitFor(() => expect(view.result.current.failed).toBe(2))
    expect(useWorkspaceStore.getState().selectedSourceIds).toEqual([])
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
