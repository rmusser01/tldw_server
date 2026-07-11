// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from "vitest"

import {
  createEmptyQuickIngestSession,
  createQuickIngestSessionStore,
} from "../quick-ingest-session"

const STORAGE_KEY = "tldw-quick-ingest-session"

describe("quick ingest session store", () => {
  beforeEach(() => {
    sessionStorage.clear()
  })

  it("persists a hidden completed session and rehydrates it in the same tab", () => {
    const store = createQuickIngestSessionStore()

    store.getState().upsertSession({
      ...createEmptyQuickIngestSession(),
      lifecycle: "completed",
      visibility: "hidden",
      currentStep: 5,
      resultSummary: {
        status: "success",
        attemptedAt: 1700000000000,
        completedAt: 1700000005000,
        totalCount: 1,
        successCount: 1,
        failedCount: 0,
        cancelledCount: 0,
        firstMediaId: "media-1",
        primarySourceLabel: "Example Source",
        errorMessage: null,
      },
      results: [{ id: "result-1", status: "ok", type: "html" }],
      completedAt: 1700000005000,
    })

    const persistedRaw = sessionStorage.getItem(STORAGE_KEY)
    expect(persistedRaw).toContain('"lifecycle":"completed"')
    expect(persistedRaw).toContain('"visibility":"hidden"')

    const rehydratedStore = createQuickIngestSessionStore()
    const rehydrated = rehydratedStore.getState().session

    expect(rehydrated?.lifecycle).toBe("completed")
    expect(rehydrated?.visibility).toBe("hidden")
    expect(rehydrated?.resultSummary.status).toBe("success")
    expect(rehydratedStore.getState().triggerSummary.label).toMatch(/completed/i)
  })

  it("removes completed sessions only when clearSession is called", () => {
    const store = createQuickIngestSessionStore()

    store.getState().upsertSession({
      ...createEmptyQuickIngestSession(),
      lifecycle: "completed",
      visibility: "hidden",
      currentStep: 5,
      completedAt: 1700000005000,
    })

    expect(store.getState().session).not.toBeNull()
    expect(sessionStorage.getItem(STORAGE_KEY)).toContain('"visibility":"hidden"')

    store.getState().clearSession()

    expect(store.getState().session).toBeNull()
    expect(sessionStorage.getItem(STORAGE_KEY)).toBeNull()
  })

  it("stores queue file stubs without raw File instances", () => {
    const file = new File(["sample"], "sample.txt", {
      type: "text/plain",
      lastModified: 1700000000000,
    })
    const store = createQuickIngestSessionStore()

    store.getState().upsertSession({
      queueItems: [
        {
          kind: "file",
          id: "file-1",
          key: "sample.txt::6::1700000000000",
          file,
          name: file.name,
          size: file.size,
          type: file.type,
          lastModified: file.lastModified,
          transientPayload: { should: "not-persist" },
        } as any,
      ] as any,
    })

    const persistedRaw = sessionStorage.getItem(STORAGE_KEY)
    const persisted = persistedRaw ? JSON.parse(persistedRaw) : null
    const persistedItem = persisted?.state?.session?.queueItems?.[0]

    expect(persistedItem?.kind).toBe("file")
    expect(persistedItem?.name).toBe("sample.txt")
    expect(persistedItem?.file).toBeUndefined()
    expect(persistedItem?.transientPayload).toBeUndefined()
  })

  it("merges persisted tracking metadata across direct-session updates", () => {
    const store = createQuickIngestSessionStore()

    store.getState().markProcessingTracking({
      mode: "webui-direct",
      sessionId: "qi-direct-merge",
      itemIds: ["url-1", "file-1"],
      startedAt: 1700000000000,
    } as any)

    store.getState().markProcessingTracking({
      mode: "webui-direct",
      sessionId: "qi-direct-merge",
      batchId: "batch-1",
      batchIds: ["batch-1"],
      jobIds: [77],
    } as any)

    store.getState().markProcessingTracking({
      mode: "webui-direct",
      sessionId: "qi-direct-merge",
      batchId: "batch-2",
      batchIds: ["batch-2"],
      jobIds: [88],
    } as any)

    expect(store.getState().session?.tracking).toMatchObject({
      mode: "webui-direct",
      sessionId: "qi-direct-merge",
      batchId: "batch-2",
      batchIds: ["batch-1", "batch-2"],
      jobIds: [77, 88],
      itemIds: ["url-1", "file-1"],
      startedAt: 1700000000000,
    })
  })

  it("clears completed run tracking when the session returns to draft", () => {
    const store = createQuickIngestSessionStore()

    store.getState().markProcessingTracking({
      mode: "webui-direct",
      sessionId: "qi-direct-completed",
      batchId: "batch-1",
      jobIds: [77],
    })

    store.getState().upsertSession({
      lifecycle: "draft",
      currentStep: 1,
    })

    expect(store.getState().session?.tracking).toBeUndefined()
  })
})
