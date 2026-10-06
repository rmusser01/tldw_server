import { beforeEach, describe, expect, it, vi } from "vitest"

import { createQuickIngestSessionStore } from "../quick-ingest-session"

const key = "tldw-quick-ingest-session"
const owner = "verified-A"
const source = {
  id: "source",
  kind: "url",
  url: "https://user:secret@example.com/report?token=private#secret",
  detectedType: "web",
  icon: "Globe",
  fileSize: 0,
  validation: { valid: true }
} as const
const start = (
  store: ReturnType<typeof createQuickIngestSessionStore>,
  id = "run"
) => {
  store.getState().createDraftSession({ id, queueItems: [{ ...source }] })
  store.getState().markProcessingTracking({
    mode: "webui-direct",
    batchId: `batch-${id}`,
    jobIds: [3]
  })
}
describe("recent import metadata", () => {
  beforeEach(() => {
    sessionStorage.clear()
    vi.restoreAllMocks()
  })
  it("appears at submission and resumes the recognizable current import after reload", () => {
    const store = createQuickIngestSessionStore()
    store.getState().setAuthority(owner)
    start(store)
    expect(store.getState().recentImports).toMatchObject([
      {
        id: "run",
        sourceLabel: "example.com",
        sourceCount: 1,
        lifecycle: "processing",
        batchIds: ["batch-run"]
      }
    ])
    const reloaded = createQuickIngestSessionStore()
    expect(reloaded.getState().recentImports).toEqual([])
    reloaded.getState().setAuthority(owner)
    reloaded.getState().showSession()
    expect(reloaded.getState().session?.id).toBe("run")
    expect(reloaded.getState().recentImports[0].id).toBe("run")
  })
  it("keeps known batches across retry attempts while live tracking follows only the new executor", () => {
    const store = createQuickIngestSessionStore()
    store.getState().setAuthority(owner)
    start(store)
    store.getState().markProcessingTracking({
      mode: "webui-direct",
      sessionId: "attempt-one"
    })
    store.getState().markProcessingTracking({
      mode: "webui-direct",
      sessionId: "attempt-two"
    })
    expect(store.getState().session?.tracking?.jobIds).toBeUndefined()
    store.getState().upsertSession({ lifecycle: "completed" })
    const reloaded = createQuickIngestSessionStore()
    reloaded.getState().setAuthority(owner)
    expect(reloaded.getState().recentImports[0]).toMatchObject({
      batchIds: ["batch-run"],
      jobIds: [3]
    })
  })
  it("archives replacements in newest-first order with a maximum of ten", () => {
    const store = createQuickIngestSessionStore()
    store.getState().setAuthority(owner)
    for (let i = 0; i < 12; i++) {
      start(store, `run-${i}`)
      store.getState().upsertSession({
        lifecycle: "completed",
        results: [{ id: "ok", status: "ok", type: "web", mediaId: i + 1 }]
      })
    }
    store.getState().clearSession()
    expect(store.getState().recentImports.map((item) => item.id)).toEqual(
      Array.from({ length: 10 }, (_, i) => `run-${11 - i}`)
    )
    expect(
      JSON.parse(sessionStorage.getItem(key)!).state.recentImports
    ).toHaveLength(10)
  })
  it("never archives credentials, pasted content, File data or unsaved/error IDs", () => {
    const store = createQuickIngestSessionStore()
    store.getState().setAuthority(owner)
    start(store)
    store.getState().upsertSession({
      results: [
        {
          id: "ok",
          status: "ok",
          type: "web",
          mediaId: 5,
          content: "extracted-secret"
        },
        {
          id: "unsaved",
          status: "ok",
          persisted: false,
          type: "web",
          mediaId: null
        },
        { id: "failed", status: "error", type: "web", mediaId: 6 },
        { id: "duplicate", status: "ok", type: "web", mediaId: 5 }
      ],
      lifecycle: "completed"
    })
    store.getState().replaceWithNewDraft({
      id: "text",
      queueItems: [
        {
          ...source,
          kind: "text",
          url: "pasted-private-body",
          name: "pasted-private-body"
        }
      ]
    })
    store.getState().markProcessingTracking({ mode: "unknown" })
    store.getState().clearSession()
    const persisted = sessionStorage.getItem(key)!
    expect(persisted).not.toMatch(/secret|private|token=|extracted/)
    expect(
      store.getState().recentImports.find((item) => item.id === "run")
        ?.savedMediaIds
    ).toEqual([5])
  })
  it.each(["verified-B", "other-server-A"])(
    "hides old imports and rejects captured history writers under %s",
    (next) => {
      const store = createQuickIngestSessionStore()
      store.getState().setAuthority(owner)
      start(store)
      const update = store.getState().updateRecentImport
      store.getState().setAuthority(next)
      update("run", { lifecycle: "completed", savedMediaIds: [99] })
      expect(store.getState().recentImports).toEqual([])
    }
  )
  it("does not erase unread prior history when the real sessionStorage read fails", () => {
    const original = createQuickIngestSessionStore()
    original.getState().setAuthority(owner)
    start(original)
    original.getState().clearSession()
    const raw = sessionStorage.getItem(key)
    const spy = vi
      .spyOn(Object.getPrototypeOf(window.sessionStorage), "getItem")
      .mockImplementationOnce(() => {
        throw new Error("temporary read failure")
      })
    const reloaded = createQuickIngestSessionStore()
    reloaded.getState().setAuthority(owner)
    reloaded.getState().createDraftSession()
    expect(sessionStorage.getItem(key)).toBe(raw)
    spy.mockRestore()
    reloaded.persist.rehydrate()
    expect(reloaded.getState().recentImports[0].id).toBe("run")
  })
  it.each(["completed", "partial_failure", "cancelled"] as const)(
    "retains authoritative terminal %s outcomes after reload despite superseded job refresh",
    (lifecycle) => {
      const store = createQuickIngestSessionStore()
      store.getState().setAuthority(owner)
      start(store)
      store
        .getState()
        .upsertSession({
          lifecycle,
          completedAt: 2,
          results: [{ id: "source", type: "web", status: "ok", mediaId: 41 }]
        })
      store.getState().replaceWithNewDraft()
      const reloaded = createQuickIngestSessionStore()
      reloaded.getState().setAuthority(owner)
      reloaded
        .getState()
        .updateRecentImport("run", {
          lifecycle: "processing",
          savedMediaIds: [99]
        })
      expect(reloaded.getState().recentImports[0]).toMatchObject({
        lifecycle,
        completedAt: 2,
        savedMediaIds: [41]
      })
    }
  )

  it("allows recovery refresh when the session was interrupted rather than authoritatively completed", () => {
    const store = createQuickIngestSessionStore()
    store.getState().setAuthority(owner)
    start(store)
    store.getState().upsertSession({ lifecycle: "interrupted", completedAt: 2 })
    store
      .getState()
      .updateRecentImport("run", { lifecycle: "completed", savedMediaIds: [41] })
    expect(store.getState().recentImports[0]).toMatchObject({
      lifecycle: "completed",
      savedMediaIds: [41]
    })
  })

})
