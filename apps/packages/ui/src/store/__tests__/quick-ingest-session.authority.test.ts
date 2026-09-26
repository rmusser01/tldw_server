import { beforeEach, describe, expect, it } from "vitest"
import { createEmptyQuickIngestSession, createQuickIngestSessionStore } from "../quick-ingest-session"

describe("private Quick Ingest session authority", () => {
  beforeEach(() => sessionStorage.clear())

  it("quarantines an unowned legacy completed result during real persisted hydration", () => {
    sessionStorage.setItem("tldw-quick-ingest-session", JSON.stringify({
      version: 1,
      state: { session: {
        ...createEmptyQuickIngestSession(), lifecycle: "completed", currentStep: 5,
        results: [{ id: "bob-result", fileName: "Bob-private.pdf", mediaId: 7, status: "ok", type: "pdf" }]
      } }
    }))
    const reloaded = createQuickIngestSessionStore()
    expect(reloaded.getState().session).toBeNull()
    expect(reloaded.getState().triggerSummary.count).toBe(0)
  })

  it("does not recreate a cleared session when a previously captured completion arrives", () => {
    const store = createQuickIngestSessionStore()
    store.getState().setAuthority("server-A/user-A")
    store.getState().createDraftSession()
    const completeLater = store.getState().upsertSession
    store.getState().clearSession()
    completeLater({ lifecycle: "completed", results: [{ id: "late-Bob", status: "ok", type: "pdf", mediaId: 7 }] })
    expect(store.getState().session).toBeNull()
  })

  it("keeps owned hydration masked until the same authority is verified", () => {
    const original = createQuickIngestSessionStore()
    original.getState().setAuthority("server-A/user-A")
    original.getState().createDraftSession({ results: [{ id: "private", status: "ok", type: "pdf", mediaId: 7 }] })
    const reloaded = createQuickIngestSessionStore()
    expect(reloaded.getState().session).toBeNull()
    reloaded.getState().setAuthority("server-A/user-A")
    expect(reloaded.getState().session?.results[0].id).toBe("private")
  })

  it.each(["server-A/user-B", "server-B/user-A"])("drops owned persisted numeric IDs under a different authority (%s)", authority => {
    const original = createQuickIngestSessionStore()
    original.getState().setAuthority("server-A/user-A")
    original.getState().createDraftSession({ results: [{ id: "private", status: "ok", type: "pdf", mediaId: 7 }] })
    const reloaded = createQuickIngestSessionStore()
    reloaded.getState().setAuthority(authority)
    expect(reloaded.getState().session).toBeNull()
    expect(sessionStorage.getItem("tldw-quick-ingest-session")).toBeNull()
  })

  it("rejects late completion and tracking callbacks even after A to B to A creates a new session", () => {
    const store = createQuickIngestSessionStore()
    store.getState().setAuthority("server-A/user-A")
    store.getState().createDraftSession()
    const oldCompletion = store.getState().upsertSession
    const oldTracking = store.getState().markProcessingTracking
    store.getState().setAuthority("server-A/user-B")
    store.getState().setAuthority("server-A/user-A")
    const replacement = store.getState().createDraftSession()
    oldCompletion({ results: [{ id: "old-private", status: "ok", type: "pdf", mediaId: 7 }] })
    oldTracking({ mode: "webui-direct", jobIds: [7] })
    expect(store.getState().session).toMatchObject({ id: replacement.id, results: [], lifecycle: "draft" })
    expect(store.getState().session?.tracking).toBeUndefined()
  })

  it("does not rehydrate a delayed stored session across A to B to A", async () => {
    let resolve!: (value: string) => void
    const storage = { getItem: () => new Promise<string>(done => { resolve = done }), setItem: () => {}, removeItem: () => {} }
    const store = createQuickIngestSessionStore(storage)
    store.getState().setAuthority("server-A/user-A")
    store.getState().setAuthority("server-A/user-B")
    store.getState().setAuthority("server-A/user-A")
    resolve(JSON.stringify({ version: 2, state: { session: { ...createEmptyQuickIngestSession(), authorityKey: "server-A/user-A", results: [{ id: "delayed-private" }] } } }))
    await Promise.resolve()
    await Promise.resolve()
    expect(store.getState().session).toBeNull()
  })
})

import { useQuickIngestStore } from "../quick-ingest"
import { useQuickIngestSessionStore } from "../quick-ingest-session"
it("clears recent filenames and rejects captured run-summary writers across A to B to A", () => {
  const store = useQuickIngestSessionStore
  store.getState().setAuthority("a")
  store.getState().createDraftSession()
  useQuickIngestStore.getState().addRecentlyIngestedDoc({ id: 7, type: "pdf", title: "Bob-private.pdf" })
  const lateSummary = useQuickIngestStore.getState().recordRunSuccess
  const lateRecent = useQuickIngestStore.getState().addRecentlyIngestedDoc
  store.getState().setAuthority("b")
  expect(useQuickIngestStore.getState().recentlyIngestedDocs).toEqual([])
  store.getState().setAuthority("a")
  store.getState().createDraftSession()
  lateSummary({ totalCount: 1, successCount: 1, failedCount: 0, firstMediaId: 7, primarySourceLabel: "Bob-private.pdf" })
  lateRecent({ id: 7, type: "pdf", title: "Bob-private.pdf" })
  expect(useQuickIngestStore.getState().lastRunSummary.status).toBe("idle")
  expect(useQuickIngestStore.getState().recentlyIngestedDocs).toEqual([])
})
