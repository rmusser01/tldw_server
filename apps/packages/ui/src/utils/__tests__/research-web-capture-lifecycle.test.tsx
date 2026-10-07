import { act, renderHook, waitFor } from "@testing-library/react"
import { beforeEach, expect, it, vi } from "vitest"
import { webcrypto } from "node:crypto"
import { useResearchWebCapture } from "../use-research-web-capture"
import {
  readResearchWebCaptures,
  queueResearchWorkspacePrefill
} from "../research-workspace-prefill"
import { useWorkspaceStore } from "@/store/workspace"
const mocks = vi.hoisted(() => ({
  values: new Map<string, unknown>(),
  owner: "alice",
  extract: vi.fn(),
  save: vi.fn(),
  confirm: vi.fn(),
  failStorage: false,
  listeners: new Set<(changed: boolean) => void>()
}))
vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({
    hasPersistentBackend: true,
    get: async (key: string) => structuredClone(mocks.values.get(key)),
    set: async (key: string, value: unknown) => {
      if (mocks.failStorage) throw Error("storage full")
      mocks.values.set(key, structuredClone(value))
    }
  })
}))
vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getConfig: async () => ({ serverUrl: mocks.owner }),
    saveWebClip: (...args: unknown[]) => mocks.save(...args)
  }
}))
vi.mock("@/services/tldw/TldwMedia", () => ({
  tldwMedia: {
    extractPublicArticle: (...args: unknown[]) => mocks.extract(...args)
  }
}))
vi.mock("@/services/chat-surface-scope", () => ({
  buildChatSurfaceScopeKeyFromConfig: (config: { serverUrl: string }) =>
    config.serverUrl
}))
vi.mock("@/utils/media-navigation-scope", () => ({
  deriveScopedUserId: () => "user:single"
}))
vi.mock("@/services/chat-account-boundary", () => ({
  watchChatAccountChanges: (fn: (changed: boolean) => void) => {
    mocks.listeners.add(fn)
    return () => mocks.listeners.delete(fn)
  }
}))
vi.mock("@/services/service-prompts", () => ({
  loadServicePromptSnapshot: async (
    _ids: unknown,
    options: { signal: AbortSignal }
  ) => ({
    scopeKey: mocks.owner,
    requestScope: { config: { serverUrl: mocks.owner } },
    scopeSignal: options.signal,
    scopeInvalidatedSignal: options.signal,
    release: () => {}
  })
}))
vi.mock("../research-web-capture", async (original) => ({
  ...(await original<typeof import("../research-web-capture")>()),
  confirmWebCaptureAcceptance: (...args: unknown[]) => mocks.confirm(...args)
}))
vi.mock("@/store/workspace", async () => {
  const { create } = await import("zustand")
  return {
    useWorkspaceStore: create(() => ({
      workspaceId: "workspace",
      sources: [],
      selectedSourceIds: [],
      selectedSourceFolderIds: [],
      setSelectedSourceIds: (ids: string[]) =>
        useWorkspaceStore.setState({ selectedSourceIds: ids })
    }))
  }
})
const original = {
  id: "original",
  mediaId: 1,
  title: "Original result",
  type: "website" as const,
  url: "https://example.org/story",
  addedAt: new Date(),
  knowledgeQaEvidence: {
    importId: "import",
    threadId: null,
    snapshot: true,
    sources: [
      {
        originalId: "result",
        mediaId: null,
        title: "Original result",
        type: "website" as const,
        excerpt: "Retrieved excerpt",
        sourceType: "web"
      }
    ]
  }
}
const deferred = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((r) => {
    resolve = r
  })
  return { promise, resolve }
}
beforeEach(() => {
  vi.clearAllMocks()
  Object.defineProperty(globalThis, "crypto", {
    value: webcrypto,
    configurable: true
  })
  mocks.values.clear()
  mocks.failStorage = false
  mocks.owner = "alice"
  useWorkspaceStore.setState({
    workspaceId: "workspace",
    sources: [original],
    selectedSourceIds: [original.id],
    selectedSourceFolderIds: []
  })
  mocks.extract.mockResolvedValue({
    results: [
      {
        url: original.url,
        title: "Full article",
        content: " complete accepted article ",
        extraction_successful: true
      }
    ]
  })
  mocks.save.mockResolvedValue({ status: "saved" })
  mocks.confirm.mockImplementation(async (body) => ({
    source: {
      id: `web-clipper:${body.clip_id}`,
      media_id: mocks.save.mock.calls.length + 1,
      title: body.source_title,
      url: body.source_url,
      added_at: "2026-10-07T00:00:00Z",
      state: "queryable",
      review_state: "needs_review"
    },
    pin: {
      clipId: body.clip_id,
      requestedUrl: body.source_url,
      capturedAt: body.capture_metadata.web_capture_v1.captured_at,
      contentSha256: body.capture_metadata.web_capture_v1.content_sha256,
      refreshOf: body.capture_metadata.web_capture_v1.refresh_of,
      mediaId: mocks.save.mock.calls.length + 1,
      versionNumber: 9,
      versionUuid: "version-nine"
    }
  }))
})
it("mount and preview cancel never acquire or save", async () => {
  const { result } = renderHook(() => useResearchWebCapture("workspace"))
  await act(async () => {})
  expect(mocks.extract).not.toHaveBeenCalled()
  act(() => result.current.open(original))
  expect(mocks.extract).not.toHaveBeenCalled()
  await act(async () => result.current.extract())
  expect(result.current.preview?.text).toBe(" complete accepted article ")
  act(() => result.current.cancel())
  expect(mocks.save).not.toHaveBeenCalled()
})
it("checkpoints before mutation, confirms before adding, preserves evidence and manual selection", async () => {
  const gate = deferred<Awaited<ReturnType<typeof mocks.confirm>>>()
  const confirm = mocks.confirm.getMockImplementation()!
  mocks.confirm.mockImplementation(async (body) => {
    await gate.promise
    return confirm(body)
  })
  mocks.save.mockImplementation(async (body) => {
    expect(
      (await readResearchWebCaptures("alice", "workspace"))[0].body
    ).toEqual(body)
  })
  const { result } = renderHook(() => useResearchWebCapture("workspace"))
  act(() => result.current.open(original))
  await act(async () => result.current.extract())
  let save!: Promise<void>
  act(() => {
    save = result.current.save()
  })
  await waitFor(() => expect(mocks.confirm).toHaveBeenCalled())
  expect(useWorkspaceStore.getState().sources).toEqual([original])
  act(() => useWorkspaceStore.setState({ selectedSourceIds: [] }))
  await act(async () => {
    gate.resolve(undefined)
    await save
  })
  expect(useWorkspaceStore.getState().sources).toHaveLength(2)
  expect(useWorkspaceStore.getState().sources[0]).toEqual(original)
  expect(
    useWorkspaceStore.getState().sources[1].webCapture?.versionNumber
  ).toBe(9)
  expect(useWorkspaceStore.getState().selectedSourceIds).toEqual([])
})
it("partial failure retries the frozen body without acquiring again, including after remount", async () => {
  mocks.confirm.mockRejectedValueOnce(Error("readback failed"))
  const first = renderHook(() => useResearchWebCapture("workspace"))
  act(() => first.result.current.open(original))
  await act(async () => first.result.current.extract())
  await act(async () => first.result.current.save())
  const body = mocks.save.mock.calls[0][0]
  first.unmount()
  const second = renderHook(() => useResearchWebCapture("workspace"))
  act(() => second.result.current.open(original))
  await waitFor(() => expect(second.result.current.pending).toBeTruthy())
  await act(async () => second.result.current.save())
  expect(mocks.save.mock.calls[1][0]).toEqual(body)
  expect(mocks.extract).toHaveBeenCalledTimes(1)
})
it("storage failure prevents mutation", async () => {
  const { result } = renderHook(() => useResearchWebCapture("workspace"))
  act(() => result.current.open(original))
  await act(async () => result.current.extract())
  mocks.failStorage = true
  await act(async () => result.current.save())
  expect(mocks.save).not.toHaveBeenCalled()
})
it("rapid duplicate save is one mutation", async () => {
  const { result } = renderHook(() => useResearchWebCapture("workspace"))
  act(() => result.current.open(original))
  await act(async () => result.current.extract())
  await act(async () => {
    await Promise.all([result.current.save(), result.current.save()])
  })
  expect(mocks.save).toHaveBeenCalledTimes(1)
})
it.each(["workspace", "account", "source", "unmount"])(
  "retires on %s change and retains the original owner pending body",
  async (kind) => {
    const gate = deferred<void>()
    mocks.save.mockImplementation(() => gate.promise)
    const { result, unmount } = renderHook(() =>
      useResearchWebCapture("workspace")
    )
    act(() => result.current.open(original))
    await act(async () => result.current.extract())
    let saving!: Promise<void>
    act(() => {
      saving = result.current.save()
    })
    await waitFor(() => expect(mocks.save).toHaveBeenCalled())
    act(() => {
      if (kind === "workspace")
        useWorkspaceStore.setState({ workspaceId: "other" })
      if (kind === "account") {
        mocks.owner = "bob"
        mocks.listeners.forEach((fn) => fn(true))
      }
      if (kind === "source") useWorkspaceStore.setState({ sources: [] })
      if (kind === "unmount") unmount()
    })
    await act(async () => {
      gate.resolve()
      await saving
    })
    expect(useWorkspaceStore.getState().sources.some((s) => s.webCapture)).toBe(
      false
    )
    expect(
      (await readResearchWebCaptures("alice", "workspace"))[0].body.content
        .full_extract
    ).toBe("complete accepted article")
    expect(await readResearchWebCaptures("bob", "workspace")).toEqual([])
    expect(await readResearchWebCaptures("alice", "other")).toEqual([])
  }
)
it("a newer handoff cannot replace a capture checkpoint created without any prefill", async () => {
  const { result } = renderHook(() => useResearchWebCapture("workspace"))
  act(() => result.current.open(original))
  await act(async () => result.current.extract())
  await act(async () => result.current.save())
  const before = await readResearchWebCaptures("alice", "workspace")
  await queueResearchWorkspacePrefill({
    kind: "knowledge_qa_thread",
    id: "new-handoff",
    createdAt: new Date().toISOString(),
    threadId: null,
    query: "new",
    answer: null,
    citations: [],
    sources: []
  })
  expect(await readResearchWebCaptures("alice", "workspace")).toEqual(before)
})
it.each([true, false])(
  "refresh keeps the prior capture identity (unchanged=%s)",
  async (unchanged) => {
    const first = renderHook(() => useResearchWebCapture("workspace"))
    act(() => first.result.current.open(original))
    await act(async () => first.result.current.extract())
    await act(async () => first.result.current.save())
    const previous = useWorkspaceStore.getState().sources[1]
    const body = mocks.save.mock.calls[0][0]
    if (!unchanged)
      mocks.extract.mockResolvedValue({
        results: [
          { content: "changed complete text", extraction_successful: true }
        ]
      })
    act(() => first.result.current.open(previous))
    await act(async () => first.result.current.extract())
    if (unchanged) expect(first.result.current.notice).toBe("Text unchanged")
    await act(async () => first.result.current.save())
    expect(useWorkspaceStore.getState().sources[1]).toEqual(previous)
    expect(mocks.save).toHaveBeenCalledTimes(2)
    expect(mocks.save.mock.calls[1][0].clip_id).not.toBe(body.clip_id)
    expect(
      mocks.save.mock.calls[1][0].capture_metadata.web_capture_v1.refresh_of
    ).toBe(body.clip_id)
    expect(useWorkspaceStore.getState().sources).toHaveLength(3)
    expect(useWorkspaceStore.getState().sources[2].mediaId).not.toBe(
      previous.mediaId
    )
    expect(useWorkspaceStore.getState().sources[0]).toEqual(original)
    expect(useWorkspaceStore.getState().sources[2].knowledgeQaEvidence).toEqual(
      original.knowledgeQaEvidence
    )
    if (unchanged)
      expect(
        mocks.save.mock.calls[1][0].capture_metadata.web_capture_v1
          .content_sha256
      ).toBe(body.capture_metadata.web_capture_v1.content_sha256)
  }
)
