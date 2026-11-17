import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, cleanup, renderHook, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import {
  useReadingProgress,
  useReadingProgressSaveOnClose
} from "../useReadingProgress"
import type { QuickIngestOperation } from "@/services/tldw/quick-ingest-authority"
import { useAnnotations } from "../useAnnotations"
import {
  useAnnotationSync,
  useAnnotationSyncOnClose
} from "../useAnnotationSync"
import { useDocumentWorkspaceStore as store } from "@/store/document-workspace"
import { useConnectionStore } from "@/store/connection"

const api = vi.hoisted(() => ({
  getReadingProgress: vi.fn(),
  updateReadingProgress: vi.fn(),
  listAnnotations: vi.fn(),
  syncAnnotations: vi.fn(),
  updateAnnotation: vi.fn(),
  deleteAnnotation: vi.fn(),
  createAnnotation: vi.fn()
}))
const authority = vi.hoisted(() => ({
  current: null as QuickIngestOperation | null
}))
vi.mock("@/services/tldw", () => ({
  tldwClient: api,
  tldwModels: { subscribeInvalidation: vi.fn() }
}))
vi.mock("@/services/tldw/quick-ingest-authority", () => ({
  useQuickIngestAuthority: () => authority.current?.authorityKey ?? null,
  quickIngestAuthority: { capture: () => authority.current }
}))
const row = (documentId: number, id = `saved-${documentId}`) => ({
  id,
  media_id: documentId,
  location: "1",
  text: `Document ${documentId}`,
  color: "yellow" as const,
  note: "Saved",
  annotation_type: "highlight" as const,
  created_at: new Date(0).toISOString(),
  updated_at: new Date(0).toISOString()
})
const deferred = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((r) => {
    resolve = r
  })
  return { promise, resolve }
}
let client: QueryClient
let initialConnection: ReturnType<typeof useConnectionStore.getState>
const wrapper = ({ children }: { children: React.ReactNode }) => (
  <QueryClientProvider client={client}>{children}</QueryClientProvider>
)
beforeEach(() => {
  vi.clearAllMocks()
  store.getState().reset()
  client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } }
  })
  initialConnection = useConnectionStore.getState()
  useConnectionStore.setState({
    state: { ...initialConnection.state, isConnected: true, mode: "normal" }
  })
  const controller = new AbortController()
  const owner: QuickIngestOperation = {
    authorityKey: "alice",
    authorityRevision: 1,
    requestScope: {
      userId: "1",
      config: { serverUrl: "https://a.test", authMode: "multi-user" }
    },
    signal: controller.signal,
    isCurrent: () => authority.current === owner && !controller.signal.aborted,
    assertCurrent: () => {
      if (authority.current !== owner || controller.signal.aborted)
        throw new Error("Retired owner")
    }
  }
  authority.current = owner
})
afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  client.clear()
  store.getState().reset()
  useConnectionStore.setState(initialConnection, true)
  vi.restoreAllMocks()
})

describe("annotation request ownership", () => {
  it("does not let a held document A GET erase active document B", async () => {
    const first = deferred<unknown>()
    api.listAnnotations.mockImplementation((id: number) =>
      id === 1
        ? first.promise
        : Promise.resolve({
            media_id: 2,
            annotations: [row(2)],
            total_count: 1
          })
    )
    store.setState({ activeDocumentId: 1 })
    const hook = renderHook(({ id }) => useAnnotations(id), {
      initialProps: { id: 1 },
      wrapper
    })
    await waitFor(() => expect(api.listAnnotations).toHaveBeenCalledTimes(1))
    store.setState({ activeDocumentId: 2 })
    hook.rerender({ id: 2 })
    await waitFor(() =>
      expect(store.getState().annotations.some((a) => a.documentId === 2)).toBe(
        true
      )
    )
    await act(async () => {
      first.resolve({ media_id: 1, annotations: [row(1)], total_count: 1 })
      await first.promise
    })
    expect(
      store
        .getState()
        .annotations.filter((a) => a.documentId === 2)
        .map((a) => a.text)
    ).toEqual(["Document 2"])
  })
  it("uses cached document data when returning from B to A", async () => {
    api.listAnnotations.mockImplementation((id: number) =>
      Promise.resolve({ media_id: id, annotations: [row(id)], total_count: 1 })
    )
    store.setState({ activeDocumentId: 1 })
    const hook = renderHook(({ id }) => useAnnotations(id), {
      initialProps: { id: 1 },
      wrapper
    })
    await waitFor(() =>
      expect(store.getState().annotations.some((a) => a.documentId === 1)).toBe(
        true
      )
    )
    store.setState({ activeDocumentId: 2 })
    hook.rerender({ id: 2 })
    await waitFor(() =>
      expect(store.getState().annotations.some((a) => a.documentId === 2)).toBe(
        true
      )
    )
    store.setState({ activeDocumentId: 1 })
    hook.rerender({ id: 1 })
    await waitFor(() =>
      expect(
        store
          .getState()
          .annotations.filter((a) => a.documentId === 1)
          .map((a) => a.text)
      ).toEqual(["Document 1"])
    )
    expect(api.listAnnotations).toHaveBeenCalledTimes(2)
  })
  it("never sends another document's pending creation to the current document", async () => {
    api.syncAnnotations.mockResolvedValue({
      media_id: 2,
      annotations: [],
      id_mapping: {},
      synced_count: 0
    })
    store.setState({ activeDocumentId: 2 })
    store
      .getState()
      .addAnnotation({
        documentId: 1,
        location: 1,
        text: "A pending",
        color: "blue"
      })
    const hook = renderHook(() => useAnnotationSync(2, 60000), { wrapper })
    await act(async () => {
      await hook.result.current.forceSync()
    })
    expect(api.syncAnnotations).not.toHaveBeenCalled()
  })
  it("acknowledges only submitted creations while retaining a later one", async () => {
    const reply = deferred<unknown>()
    api.syncAnnotations.mockReturnValue(reply.promise)
    store.setState({ activeDocumentId: 1 })
    store
      .getState()
      .addAnnotation({
        documentId: 1,
        location: 1,
        text: "First pending",
        color: "blue"
      })
    const first = store.getState().pendingAnnotations[0]
    const hook = renderHook(() => useAnnotationSync(1, 60000), { wrapper })
    let flush: unknown
    act(() => {
      flush = hook.result.current.forceSync()
    })
    await waitFor(() => expect(api.syncAnnotations).toHaveBeenCalledTimes(1))
    act(() =>
      store
        .getState()
        .addAnnotation({
          documentId: 1,
          location: 2,
          text: "Later pending",
          color: "blue"
        })
    )
    const later = store.getState().pendingAnnotations[1]
    await act(async () => {
      reply.resolve({
        media_id: 1,
        annotations: [{ ...row(1, "server-first"), text: first.text }],
        id_mapping: { [first.id]: "server-first" },
        synced_count: 1
      })
      await reply.promise
      await flush
    })
    await waitFor(() =>
      expect(store.getState().pendingAnnotations).toEqual([later])
    )
    expect(store.getState().annotations.map((a) => a.id)).toEqual([
      "server-first",
      later.id
    ])
  })
  it("does not publish a held GET after the captured account retires", async () => {
    const reply = deferred<unknown>()
    api.listAnnotations.mockReturnValue(reply.promise)
    store.setState({ activeDocumentId: 1 })
    renderHook(() => useAnnotations(1), { wrapper })
    await waitFor(() => expect(api.listAnnotations).toHaveBeenCalledTimes(1))
    authority.current = null
    await act(async () => {
      reply.resolve({ media_id: 1, annotations: [row(1)], total_count: 1 })
      await reply.promise
    })
    expect(store.getState().annotations).toEqual([])
  })
  it("warns about unfinished writes instead of claiming beacon delivery", () => {
    const sendBeacon = vi.fn().mockReturnValue(true)
    vi.stubGlobal("navigator", { sendBeacon })
    store.setState({ activeDocumentId: 1 })
    store
      .getState()
      .addAnnotation({
        documentId: 1,
        location: 1,
        text: "Unfinished",
        color: "blue"
      })
    renderHook(() => useAnnotationSyncOnClose(1, vi.fn()), { wrapper })
    const event = new Event("beforeunload", { cancelable: true })
    act(() => window.dispatchEvent(event))
    expect({
      warned: event.defaultPrevented,
      beacons: sendBeacon.mock.calls.length
    }).toEqual({ warned: true, beacons: 0 })
    vi.unstubAllGlobals()
  })
  it("does not retry a permanent creation conflict", async () => {
    vi.useFakeTimers()
    try {
      api.syncAnnotations.mockRejectedValue(
        Object.assign(new Error("Creation conflict"), { status: 409 })
      )
      store.setState({ activeDocumentId: 1 })
      store
        .getState()
        .addAnnotation({
          documentId: 1,
          location: 1,
          text: "Conflicting pending",
          color: "blue"
        })
      const hook = renderHook(() => useAnnotationSync(1, 60000), { wrapper })
      let result: unknown
      act(() => {
        result = hook.result.current.forceSync().catch((error) => error)
      })
      await act(async () => {
        await vi.advanceTimersByTimeAsync(7000)
      })
      expect(api.syncAnnotations).toHaveBeenCalledTimes(1)
      expect(((await result) as Error).message).toBe("Creation conflict")
      expect(store.getState().pendingAnnotations).toHaveLength(1)
    } finally {
      vi.useRealTimers()
    }
  })

  it("does not acknowledge two creation intents with one server identity", async () => {
    store.setState({ activeDocumentId: 1 })
    for (const text of ["First", "Second"])
      store
        .getState()
        .addAnnotation({ documentId: 1, location: 1, text, color: "blue" })
    const pending = store.getState().pendingAnnotations
    api.syncAnnotations.mockResolvedValue({
      media_id: 1,
      synced_count: 2,
      annotations: [row(1, "same"), row(1, "same")],
      id_mapping: Object.fromEntries(pending.map((a) => [a.id, "same"]))
    })
    const hook = renderHook(() => useAnnotationSync(1, 60000), { wrapper })
    await act(async () => {
      await hook.result.current.forceSync().catch(() => {})
    })
    expect(store.getState().pendingAnnotations).toEqual(pending)
  })
  it("does not apply a held reading position to a replacement document", async () => {
    const reply = deferred<unknown>()
    api.getReadingProgress.mockReturnValue(reply.promise)
    store.setState({ activeDocumentId: 1, currentPage: 1 })
    const hook = renderHook(({ id }) => useReadingProgress(id), {
      initialProps: { id: 1 },
      wrapper
    })
    await waitFor(() => expect(api.getReadingProgress).toHaveBeenCalledTimes(1))
    store.setState({ activeDocumentId: 2, currentPage: 1 })
    hook.rerender({ id: 2 })
    await act(async () => {
      reply.resolve({
        media_id: 1,
        current_page: 9,
        total_pages: 10,
        zoom_level: 100,
        view_mode: "single",
        percent_complete: 90,
        last_read_at: new Date(0).toISOString()
      })
      await reply.promise
    })
    expect(store.getState().currentPage).toBe(1)
  })
  it("flushes the previous reading callback on a document switch", async () => {
    const first = vi.fn(),
      second = vi.fn()
    const hook = renderHook(
      ({ id, save }) => useReadingProgressSaveOnClose(id, save),
      { initialProps: { id: 1, save: first }, wrapper }
    )
    hook.rerender({ id: 2, save: second })
    await act(async () => {
      await Promise.resolve()
    })
    expect([first.mock.calls.length, second.mock.calls.length]).toEqual([1, 0])
  })

  it("flushes the explicitly closing document even while another tab is active", async () => {
    store.setState({ activeDocumentId: 2 })
    store
      .getState()
      .addAnnotation({
        documentId: 1,
        location: 1,
        text: "Closing A",
        color: "blue"
      })
    const pending = store.getState().pendingAnnotations[0]
    api.syncAnnotations.mockResolvedValue({
      media_id: 1,
      synced_count: 1,
      annotations: [row(1, "saved-a")],
      id_mapping: { [pending.id]: "saved-a" }
    })
    const hook = renderHook(() => useAnnotationSync(2, 60000), { wrapper })
    await act(async () => {
      await hook.result.current.forceSync(1)
    })
    expect(store.getState().pendingAnnotations).toEqual([])
    expect(api.syncAnnotations.mock.calls[0][0]).toBe(1)
  })
  it("rejects an explicit save after its account retires", async () => {
    const hook = renderHook(() => useAnnotationSync(1, 60000), { wrapper })
    authority.current = null
    await expect(hook.result.current.forceSync()).rejects.toThrow()
    expect(api.syncAnnotations).not.toHaveBeenCalled()
  })

  it("restores saved reading progress after a real cold document open", async () => {
    store
      .getState()
      .openDocument({
        id: 1,
        title: "Cold document",
        type: "pdf",
        url: "blob:cold"
      })
    api.getReadingProgress.mockResolvedValue({
      media_id: 1,
      current_page: 9,
      total_pages: 10,
      zoom_level: 125,
      view_mode: "continuous",
      percent_complete: 90,
      last_read_at: new Date(0).toISOString()
    })
    renderHook(() => useReadingProgress(1), { wrapper })
    await waitFor(() => expect(store.getState().currentPage).toBe(9))
    expect([store.getState().zoomLevel, store.getState().viewMode]).toEqual([
      125,
      "continuous"
    ])
  })
  it("preserves a local page choice made while the cold reading request is pending", async () => {
    const reply = deferred<unknown>()
    api.getReadingProgress.mockReturnValue(reply.promise)
    store
      .getState()
      .openDocument({
        id: 1,
        title: "Cold document",
        type: "pdf",
        url: "blob:cold"
      })
    renderHook(() => useReadingProgress(1), { wrapper })
    await waitFor(() => expect(api.getReadingProgress).toHaveBeenCalled())
    act(() => store.getState().setCurrentPage(3))
    await act(async () => {
      reply.resolve({
        media_id: 1,
        current_page: 9,
        total_pages: 10,
        zoom_level: 100,
        view_mode: "single",
        percent_complete: 90,
        last_read_at: new Date(0).toISOString()
      })
      await reply.promise
    })
    expect(store.getState().currentPage).toBe(3)
  })
})
