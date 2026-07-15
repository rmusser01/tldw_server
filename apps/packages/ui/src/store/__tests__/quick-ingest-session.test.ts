// @vitest-environment jsdom
import { describe, expect, it, vi } from "vitest"
import type { StateStorage } from "zustand/middleware"

import {
  createEmptyQuickIngestSession,
  createQuickIngestSessionStore,
} from "../quick-ingest-session"

type TestPersistenceStatus =
  | "ready"
  | "migrating"
  | "unavailable"
  | "quota_error"

const createControlledPersistence = () => {
  let value: string | null = null
  let status: TestPersistenceStatus = "migrating"
  let nextWriteFailure: unknown = null
  let nextWriteGate: Promise<void> | null = null
  let delayNextRead = false
  let acquireResult = true
  let writeTail: Promise<void> = Promise.resolve()
  let pendingWriteFailure: unknown = null
  const writeAttempts: string[] = []
  const listeners = new Set<(next: TestPersistenceStatus) => void>()

  const publishStatus = (next: TestPersistenceStatus) => {
    status = next
    for (const listener of listeners) listener(next)
  }

  const awaitNextWrite = async () => {
    const gate = nextWriteGate
    const failure = nextWriteFailure
    nextWriteGate = null
    nextWriteFailure = null
    if (gate) await gate
    if (failure) {
      publishStatus(
        failure instanceof DOMException && failure.name === "QuotaExceededError"
          ? "quota_error"
          : "unavailable"
      )
      throw failure
    }
  }

  const storage: StateStorage = {
    getItem: async () => {
      if (delayNextRead) {
        delayNextRead = false
        await new Promise<void>((resolve) => setTimeout(resolve, 0))
      }
      return value
    },
    setItem: (_key, next) => {
      const write = (async () => {
        writeAttempts.push(next)
        await awaitNextWrite()
        value = next
      })()
      writeTail = write.then(
        () => {
          pendingWriteFailure = null
        },
        (error) => {
          pendingWriteFailure = error
        }
      )
      return write
    },
    removeItem: async () => {
      value = null
    },
  }

  return {
    storage,
    initialize: vi.fn(async () => {
      await Promise.resolve()
      publishStatus("ready")
    }),
    cleanupExpired: vi.fn(async () => {}),
    flush: vi.fn(async () => {
      await writeTail
      const failure = pendingWriteFailure
      pendingWriteFailure = null
      if (failure) throw failure
    }),
    commitReviewHandoff: vi.fn(async (expected: string, next: string) => {
      writeAttempts.push(next)
      await awaitNextWrite()
      if (value !== expected) return false
      value = next
      return true
    }),
    commitProcessingHandoff: vi.fn(async (expected: string, next: string) => {
      writeAttempts.push(next)
      await awaitNextWrite()
      if (value !== expected) return false
      value = next
      return true
    }),
    clearAuthoritativeSession: vi.fn(
      async (expected: { id: string; lifecycle: string; updatedAt: number }) => {
        const session = value ? JSON.parse(value)?.state?.session : null
        if (
          session?.id !== expected.id ||
          session?.lifecycle !== expected.lifecycle ||
          session?.updatedAt !== expected.updatedAt
        ) {
          return false
        }
        value = null
        return true
      }
    ),
    getStatus: () => status,
    subscribeStatus: (listener: (next: TestPersistenceStatus) => void) => {
      listeners.add(listener)
      listener(status)
      return () => listeners.delete(listener)
    },
    acquireSubmissionLease: vi.fn(async () => acquireResult),
    renewSubmissionLease: vi.fn(async () => true),
    releaseSubmissionLease: vi.fn(async () => {}),
    publishStatus,
    failNextWrite: (failure: unknown) => {
      nextWriteFailure = failure
    },
    blockNextWrite: (gate: Promise<void>) => {
      nextWriteGate = gate
    },
    delayNextRead: () => {
      delayNextRead = true
    },
    seedValue: (next: string | null) => {
      value = next
    },
    setAcquireResult: (next: boolean) => {
      acquireResult = next
    },
    get value() {
      return value
    },
    get writeAttempts() {
      return [...writeAttempts]
    },
  }
}

const createStoreWithPersistence = (
  persistence: ReturnType<typeof createControlledPersistence>
) =>
  (createQuickIngestSessionStore as unknown as (options: {
    persistence: ReturnType<typeof createControlledPersistence>
  }) => ReturnType<typeof createQuickIngestSessionStore>)({ persistence })

const flushPersistence = async () => {
  await Promise.resolve()
  await new Promise<void>((resolve) => setTimeout(resolve, 0))
}

describe("quick ingest session store", () => {
  it("persists a hidden completed session and rehydrates it in the same origin", async () => {
    const persistence = createControlledPersistence()
    const store = createStoreWithPersistence(persistence)

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

    await flushPersistence()
    const persistedRaw = persistence.value
    expect(persistedRaw).toContain('"lifecycle":"completed"')
    expect(persistedRaw).toContain('"visibility":"hidden"')

    const rehydratedStore = createStoreWithPersistence(persistence)
    await rehydratedStore.persist.rehydrate()
    const rehydrated = rehydratedStore.getState().session

    expect(rehydrated?.lifecycle).toBe("completed")
    expect(rehydrated?.visibility).toBe("hidden")
    expect(rehydrated?.resultSummary.status).toBe("success")
    expect(rehydratedStore.getState().triggerSummary.label).toMatch(/completed/i)
  })

  it("keeps durable non-draft authority when an early draft intent is newer than hydration", async () => {
    const persistence = createControlledPersistence()
    const durable = {
      ...createEmptyQuickIngestSession(),
      id: "durable-before-open",
      lifecycle: "completed" as const,
      visibility: "hidden" as const,
      currentStep: 5 as const,
      createdAt: 1,
      updatedAt: 2,
      completedAt: 2,
    }
    persistence.seedValue(
      JSON.stringify({ state: { session: durable }, version: 0 })
    )
    persistence.delayNextRead()
    let releaseDraftWrite!: () => void
    persistence.blockNextWrite(
      new Promise<void>((resolve) => {
        releaseDraftWrite = resolve
      })
    )
    const store = createStoreWithPersistence(persistence)
    const hydrationFinished = new Promise<void>((resolve) => {
      const unsubscribe = store.persist.onFinishHydration(() => {
        unsubscribe()
        resolve()
      })
    })

    const earlyDraft = store.getState().createDraftSession({
      id: "early-open-draft",
      visibility: "visible",
      updatedAt: Date.now(),
    })
    expect(earlyDraft.updatedAt).toBeGreaterThan(durable.updatedAt)
    await hydrationFinished

    expect(store.getState().session).toMatchObject({
      id: "durable-before-open",
      lifecycle: "completed",
      currentStep: 5,
    })
    releaseDraftWrite()
  })

  it("removes completed sessions only when clearSession is called", async () => {
    const persistence = createControlledPersistence()
    const store = createStoreWithPersistence(persistence)

    store.getState().upsertSession({
      ...createEmptyQuickIngestSession(),
      lifecycle: "completed",
      visibility: "hidden",
      currentStep: 5,
      completedAt: 1700000005000,
    })

    await flushPersistence()
    expect(store.getState().session).not.toBeNull()
    expect(persistence.value).toContain('"visibility":"hidden"')

    store.getState().clearSession()

    await flushPersistence()
    expect(store.getState().session).toBeNull()
    expect(persistence.value).toBeNull()
  })

  it("keeps the prior replay identity when a Review handoff cannot be confirmed", async () => {
    const persistence = createControlledPersistence()
    const store = createStoreWithPersistence(persistence)
    store.getState().upsertSession({
      ...createEmptyQuickIngestSession(),
      lifecycle: "processing",
      currentStep: 4,
      tracking: {
        mode: "extension-runtime",
        sessionId: "qi-store-review-write-failure",
        itemIds: ["occ-store-review-write-failure"],
        startedAt: Date.now(),
      },
    })
    await flushPersistence()
    const before = store.getState().session
    const durableBefore = persistence.value
    persistence.failNextWrite(new Error("Review persistence unavailable"))

    expect(
      await store.getState().commitReviewHandoff({
        lifecycle: "draft",
        currentStep: 3,
        processingState: {
          status: "idle",
          perItemProgress: [],
          elapsed: 0,
          estimatedRemaining: 0,
        },
      })
    ).toBe(false)
    expect(store.getState().session).toEqual(before)
    expect(persistence.value).toBe(durableBefore)
    expect(store.getState().persistenceStatus).toBe("unavailable")
  })

  it("retries an identical background value after its first write fails", async () => {
    const persistence = createControlledPersistence()
    const store = createStoreWithPersistence(persistence)
    store.getState().createDraftSession()
    await flushPersistence()
    const durableBefore = persistence.value

    persistence.failNextWrite(new Error("background persistence unavailable"))
    store.getState().hideSession()
    await flushPersistence()

    expect(persistence.value).toBe(durableBefore)
    expect(store.getState().persistenceStatus).toBe("unavailable")

    persistence.publishStatus("ready")
    await flushPersistence()

    expect(JSON.parse(persistence.value || "null")?.state?.session).toMatchObject({
      visibility: "hidden",
    })
  })

  it("does not let an older failed write clear a newer successful dedupe marker", async () => {
    const persistence = createControlledPersistence()
    const store = createStoreWithPersistence(persistence)
    store.getState().createDraftSession()
    await flushPersistence()
    let releaseOlderWrite!: () => void
    persistence.failNextWrite(new Error("older background write failed"))
    persistence.blockNextWrite(
      new Promise<void>((resolve) => {
        releaseOlderWrite = resolve
      })
    )

    store.getState().hideSession()
    store.getState().showSession()
    await flushPersistence()
    releaseOlderWrite()
    await flushPersistence()
    const attemptsAfterOlderFailure = persistence.writeAttempts.length

    persistence.publishStatus("ready")
    await flushPersistence()

    expect(persistence.writeAttempts).toHaveLength(attemptsAfterOlderFailure)
  })

  it("writes a Review handoff in the envelope used by normal store rehydration", async () => {
    const persistence = createControlledPersistence()
    const store = createStoreWithPersistence(persistence)
    store.getState().upsertSession({
      ...createEmptyQuickIngestSession(),
      lifecycle: "processing",
      currentStep: 4,
      tracking: {
        mode: "extension-runtime",
        sessionId: "qi-store-review-envelope",
        itemIds: ["occ-store-review-envelope"],
        startedAt: Date.now(),
      },
    })

    expect(
      await store.getState().commitReviewHandoff({
        lifecycle: "draft",
        currentStep: 3,
        processingState: {
          status: "idle",
          perItemProgress: [],
          elapsed: 0,
          estimatedRemaining: 0,
        },
      })
    ).toBe(true)

    const persisted = JSON.parse(persistence.value || "null")
    expect(persisted?.version).toBe(0)
    expect(persisted?.state?.session?.currentStep).toBe(3)
    expect(persisted?.state?.session?.tracking).toBeUndefined()

    const rehydratedStore = createStoreWithPersistence(persistence)
    await rehydratedStore.persist.rehydrate()
    const rehydrated = rehydratedStore.getState().session
    expect(rehydrated?.currentStep).toBe(3)
    expect(rehydrated?.tracking).toBeUndefined()
  })

  it("stores queue file stubs without raw File instances", async () => {
    const file = new File(["sample"], "sample.txt", {
      type: "text/plain",
      lastModified: 1700000000000,
    })
    const persistence = createControlledPersistence()
    const store = createStoreWithPersistence(persistence)

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

    await flushPersistence()
    const persistedRaw = persistence.value
    const persisted = persistedRaw ? JSON.parse(persistedRaw) : null
    const persistedItem = persisted?.state?.session?.queueItems?.[0]

    expect(persistedItem?.kind).toBe("file")
    expect(persistedItem?.name).toBe("sample.txt")
    expect(persistedItem?.file).toBeUndefined()
    expect(persistedItem?.transientPayload).toBeUndefined()
  })

  it("persists a compact sanitized 500-item record without file or raw display bytes", async () => {
    const persistence = createControlledPersistence()
    const store = createStoreWithPersistence(persistence)
    const thumbnail = `data:image/png;base64,${"x".repeat(10_000)}`
    const queueItems = Array.from({ length: 500 }, (_, index) => ({
      id: `bounded-${index + 1}`,
      sourceRef: {
        kind: "direct_url",
        occurrenceId: `bounded-${index + 1}`,
        url: `https://example.com/watch/${index + 1}`,
      },
      kind: "url",
      url: `https://example.com/watch/${index + 1}`,
      detectedType: "video",
      icon: "Film",
      fileSize: 0,
      validation: { valid: true },
      file: new File(["not durable"], `source-${index + 1}.mp4`, {
        type: "video/mp4",
      }),
      thumbnail,
      rawBytes: new Uint8Array(10_000),
      base64: thumbnail,
      transientPayload: { thumbnail },
    }))

    store.getState().upsertSession({ queueItems } as never)
    await flushPersistence()

    expect(persistence.value).not.toBeNull()
    expect(persistence.value?.length).toBeLessThan(500_000)
    const persisted = JSON.parse(persistence.value || "null")
    expect(persisted?.state?.session?.queueItems).toHaveLength(500)
    expect(persistence.value).not.toContain("data:image/png;base64")
    expect(persistence.value).not.toContain("transientPayload")
    expect(persisted?.state?.session?.queueItems?.[0]?.file).toBeUndefined()
    expect(persisted?.state?.session?.queueItems?.[0]?.rawBytes).toBeUndefined()
  })

  it("sanitizes and bounds every persisted full-envelope payload surface", async () => {
    const persistence = createControlledPersistence()
    const store = createStoreWithPersistence(persistence)
    const marker = `data:image/png;base64,${"unsafe".repeat(700)}`
    const rows = Array.from({ length: 500 }, (_, index) => {
      const id = `full-envelope-${index + 1}`
      return {
        queueItem: {
          id,
          sourceRef: {
            kind: "direct_url",
            occurrenceId: id,
            url: `https://example.com/watch/${index + 1}`,
          },
          kind: "url",
          url: `https://example.com/watch/${index + 1}`,
          detectedType: "video",
          icon: "Film",
          fileSize: 0,
          validation: { valid: true },
          thumbnail: marker,
          transientPayload: { marker },
        },
        result: {
          id,
          status: "ok",
          outcome: "ingested",
          type: "video",
          title: `Restored title ${index + 1}`,
          message: "Restored result",
          mediaId: index + 1,
          retryAttempt: 2,
          data: {
            file: new File(["unsafe"], `${id}.mp4`, { type: "video/mp4" }),
            blob: new Blob(["unsafe"], { type: "application/octet-stream" }),
            base64: marker,
            thumbnail: marker,
            transient: marker,
            bytes: new Uint8Array(256),
          },
        },
        progress: {
          id,
          status: "processing",
          progressPercent: 50,
          currentStage: "Restored processing",
          estimatedRemaining: 10,
          lifecycleState: "processing",
          retryable: true,
          attempt: 2,
          thumbnail: marker,
          transient: { marker },
          bytes: new Uint8Array(256),
        },
      }
    })

    store.getState().upsertSession({
      queueItems: rows.map(({ queueItem }) => queueItem),
      results: rows.map(({ result }) => result),
      processingState: {
        status: "running",
        perItemProgress: rows.map(({ progress }) => progress),
        elapsed: 12,
        estimatedRemaining: 10,
        thumbnail: marker,
      },
      openDetail: {
        source: "extension_active_tab",
        action: "playlist_preflight",
        url: "https://example.com/playlist",
        thumbnail: marker,
        transient: { marker },
      },
      conferenceBatchMetadata: {
        collectionName: "Restored conference",
        sharedTags: ["conference"],
        thumbnail: marker,
        transient: { marker },
      },
      presetConfig: {
        ...createEmptyQuickIngestSession().presetConfig,
        thumbnail: marker,
        transient: { marker },
      },
      customOptions: {
        thumbnail: marker,
        transient: { marker },
      },
    } as never)
    await flushPersistence()

    const raw = persistence.value || ""
    const session = JSON.parse(raw || "null")?.state?.session
    expect(raw.length).toBeLessThan(1_000_000)
    expect(raw).not.toContain("data:image/png;base64")
    expect(session.results).toHaveLength(500)
    expect(session.processingState.perItemProgress).toHaveLength(500)
    expect(session.results[0]).toMatchObject({
      id: "full-envelope-1",
      status: "ok",
      outcome: "ingested",
      type: "video",
      title: "Restored title 1",
      message: "Restored result",
      mediaId: 1,
      retryAttempt: 2,
    })
    expect(session.results[0].data).toBeUndefined()
    expect(session.processingState.perItemProgress[0]).toMatchObject({
      id: "full-envelope-1",
      status: "processing",
      progressPercent: 50,
      currentStage: "Restored processing",
      lifecycleState: "processing",
      retryable: true,
      attempt: 2,
    })
    expect(session.processingState.perItemProgress[0].transient).toBeUndefined()
    expect(session.openDetail).toEqual({
      source: "extension_active_tab",
      action: "playlist_preflight",
      url: "https://example.com/playlist",
    })
    expect(session.conferenceBatchMetadata).toEqual({
      collectionName: "Restored conference",
      sharedTags: ["conference"],
    })
    expect(session.presetConfig.transient).toBeUndefined()
    expect(session.customOptions.transient).toBeUndefined()
  })

  it("bounds every tracking collection and mapping while retaining recovery identity", async () => {
    const persistence = createControlledPersistence()
    const store = createStoreWithPersistence(persistence)
    const ids = Array.from({ length: 650 }, (_, index) => `occurrence-${index + 1}`)
    const jobIdToItemId = Object.fromEntries(
      ids.map((id, index) => [String(index + 1), id])
    )
    const jobIdToCollectionItemId = Object.fromEntries(
      ids.map((_id, index) => [String(index + 1), `collection-item-${index + 1}`])
    )
    jobIdToItemId["x".repeat(300)] = "y".repeat(300)

    store.getState().upsertSession({
      tracking: {
        mode: "webui-direct",
        sessionId: "session-bounded",
        runId: "run-bounded",
        submissionOccurrenceIds: ids,
        plannedItemIds: ids,
        submittedItemIds: ids,
        itemIds: ids,
        batchIds: ids.map((_, index) => `batch-${index + 1}`),
        jobIds: ids.map((_, index) => index + 1),
        jobIdToItemId,
        jobIdToCollectionItemId,
      },
    } as never)
    await flushPersistence()

    const raw = persistence.value || ""
    const tracking = JSON.parse(raw || "null")?.state?.session?.tracking
    expect(raw.length).toBeLessThan(500_000)
    expect(tracking).toMatchObject({
      mode: "webui-direct",
      sessionId: "session-bounded",
      runId: "run-bounded",
    })
    for (const key of [
      "submissionOccurrenceIds",
      "plannedItemIds",
      "submittedItemIds",
      "itemIds",
      "batchIds",
      "jobIds",
    ]) {
      expect(tracking[key].length).toBeLessThanOrEqual(500)
    }
    for (const key of ["jobIdToItemId", "jobIdToCollectionItemId"]) {
      const entries = Object.entries(tracking[key])
      expect(entries.length).toBeLessThanOrEqual(500)
      expect(
        entries.every(
          ([entryKey, entryValue]) =>
            entryKey.length <= 255 && String(entryValue).length <= 255
        )
      ).toBe(true)
    }
    expect(tracking.plannedItemIds).toContain("occurrence-1")
    expect(tracking.jobIdToItemId["1"]).toBe("occurrence-1")
    expect(tracking.jobIdToCollectionItemId["1"]).toBe("collection-item-1")
  })

  it("awaits durable Review persistence before changing in-memory replay authority", async () => {
    const persistence = createControlledPersistence()
    const store = createStoreWithPersistence(persistence)
    store.getState().upsertSession({
      ...createEmptyQuickIngestSession(),
      lifecycle: "processing",
      currentStep: 4,
      tracking: {
        mode: "extension-runtime",
        sessionId: "qi-awaited-review",
        itemIds: ["occ-awaited-review"],
      },
    })
    await flushPersistence()
    let releaseWrite!: () => void
    persistence.blockNextWrite(
      new Promise<void>((resolve) => {
        releaseWrite = resolve
      })
    )

    const handoff = store.getState().commitReviewHandoff({
      lifecycle: "draft",
      currentStep: 3,
      processingState: {
        status: "idle",
        perItemProgress: [],
        elapsed: 0,
        estimatedRemaining: 0,
      },
    })

    expect(handoff).toBeInstanceOf(Promise)
    expect(store.getState().session).toMatchObject({
      lifecycle: "processing",
      currentStep: 4,
      tracking: { sessionId: "qi-awaited-review" },
    })
    releaseWrite()
    await expect(handoff).resolves.toBe(true)
    expect(store.getState().session).toMatchObject({
      lifecycle: "draft",
      currentStep: 3,
      tracking: undefined,
    })
  })

  it("reports a failed durable Review handoff without replacing prior replay state", async () => {
    const persistence = createControlledPersistence()
    const store = createStoreWithPersistence(persistence)
    store.getState().upsertSession({
      ...createEmptyQuickIngestSession(),
      lifecycle: "processing",
      currentStep: 4,
      tracking: {
        mode: "extension-runtime",
        sessionId: "qi-failed-review",
        itemIds: ["occ-failed-review"],
      },
    })
    await flushPersistence()
    const before = store.getState().session
    const durableBefore = persistence.value
    persistence.failNextWrite(
      new DOMException("Review persistence full", "QuotaExceededError")
    )

    await expect(
      store.getState().commitReviewHandoff({
        lifecycle: "draft",
        currentStep: 3,
      })
    ).resolves.toBe(false)
    expect(store.getState().session).toEqual(before)
    expect(persistence.value).toBe(durableBefore)
    expect(store.getState().persistenceStatus).toBe("quota_error")
  })

  it("rejects a captured Review handoff when the exact durable authority changed", async () => {
    const persistence = createControlledPersistence()
    const store = createStoreWithPersistence(persistence)
    const processing = {
      ...createEmptyQuickIngestSession(),
      lifecycle: "processing" as const,
      currentStep: 4 as const,
      tracking: {
        mode: "extension-runtime" as const,
        sessionId: "qi-captured-review",
        generation: "generation-old",
      },
    }
    store.setState({ session: processing, persistenceStatus: "ready" })
    await flushPersistence()
    let releaseWrite!: () => void
    persistence.blockNextWrite(
      new Promise<void>((resolve) => {
        releaseWrite = resolve
      })
    )

    const handoff = store.getState().commitReviewHandoff({
      lifecycle: "draft",
      currentStep: 3,
    })
    await Promise.resolve()
    const newerAuthority = {
      ...processing,
      updatedAt: processing.updatedAt + 100,
      tracking: {
        ...processing.tracking,
        generation: "generation-new",
      },
    }
    persistence.seedValue(
      JSON.stringify({ state: { session: newerAuthority }, version: 0 })
    )
    releaseWrite()

    await expect(handoff).resolves.toBe(false)
    expect(store.getState().session).toEqual(processing)
    expect(JSON.parse(persistence.value || "null")?.state?.session).toMatchObject({
      lifecycle: "processing",
      tracking: { generation: "generation-new" },
    })
  })

  it("awaits durable processing authority before changing in-memory replay state", async () => {
    const persistence = createControlledPersistence()
    const store = createStoreWithPersistence(persistence)
    const draft = createEmptyQuickIngestSession()
    store.getState().upsertSession({
      ...draft,
      queueItems: [
        {
          id: "occ-awaited-processing",
          kind: "url",
          url: "https://example.com/awaited-processing",
          status: "pending",
        },
      ],
    })
    await flushPersistence()
    store.setState({ isSubmissionOwner: true, persistenceStatus: "ready" })
    let releaseWrite!: () => void
    persistence.blockNextWrite(
      new Promise<void>((resolve) => {
        releaseWrite = resolve
      })
    )

    const handoff = store.getState().commitProcessingHandoff(
      {
        currentStep: 4,
        queueItems: store.getState().session?.queueItems || [],
      },
      {
        mode: "unknown",
        submissionState: "creating_run",
        submissionOccurrenceIds: ["occ-awaited-processing"],
        startedAt: 1700000000000,
      }
    )

    expect(handoff).toBeInstanceOf(Promise)
    expect(store.getState().session).toMatchObject({
      lifecycle: "draft",
      currentStep: 1,
      tracking: undefined,
    })
    releaseWrite()
    await expect(handoff).resolves.toBe(true)
    expect(store.getState().session).toMatchObject({
      lifecycle: "processing",
      currentStep: 4,
      tracking: {
        mode: "unknown",
        submissionState: "creating_run",
        submissionOccurrenceIds: ["occ-awaited-processing"],
        startedAt: 1700000000000,
      },
    })
    expect(JSON.parse(persistence.value || "null")?.state?.session).toMatchObject({
      lifecycle: "processing",
      currentStep: 4,
      tracking: { submissionState: "creating_run" },
    })
  })

  it("keeps prior authority when the durable processing handoff fails", async () => {
    const persistence = createControlledPersistence()
    const store = createStoreWithPersistence(persistence)
    store.getState().createDraftSession()
    await flushPersistence()
    store.setState({ isSubmissionOwner: true, persistenceStatus: "ready" })
    const before = store.getState().session
    const durableBefore = persistence.value
    persistence.failNextWrite(
      new DOMException("Processing persistence full", "QuotaExceededError")
    )

    await expect(
      store.getState().commitProcessingHandoff(
        { currentStep: 4 },
        {
          mode: "unknown",
          submissionState: "creating_run",
          submissionOccurrenceIds: ["occ-failed-processing"],
        }
      )
    ).resolves.toBe(false)
    expect(store.getState().session).toEqual(before)
    expect(persistence.value).toBe(durableBefore)
    expect(store.getState().persistenceStatus).toBe("quota_error")
  })

  it("surfaces persistence status and submission ownership in store state", async () => {
    const persistence = createControlledPersistence()
    const store = createStoreWithPersistence(persistence)
    store.getState().createDraftSession()

    expect(store.getState().persistenceStatus).toBe("migrating")
    persistence.publishStatus("unavailable")
    expect(store.getState().persistenceStatus).toBe("unavailable")
    persistence.publishStatus("ready")

    await expect(store.getState().acquireSubmissionLease()).resolves.toBe(true)
    expect(store.getState().isSubmissionOwner).toBe(true)
    await expect(store.getState().renewSubmissionLease()).resolves.toBe(true)
    await store.getState().releaseSubmissionLease()
    expect(store.getState().isSubmissionOwner).toBe(false)
  })

  it("merges newer durable processing authority when lease acquisition is rejected", async () => {
    const persistence = createControlledPersistence()
    const store = createStoreWithPersistence(persistence)
    const draft = createEmptyQuickIngestSession()
    store.getState().upsertSession({
      ...draft,
      lifecycle: "draft",
      currentStep: 3,
      updatedAt: 100,
    })
    await flushPersistence()
    persistence.seedValue(
      JSON.stringify({
        state: {
          session: {
            ...draft,
            lifecycle: "processing",
            currentStep: 4,
            updatedAt: 200,
            tracking: {
              mode: "webui-direct",
              sessionId: "authoritative-processing-session",
              runId: "authoritative-processing-run",
            },
          },
        },
        version: 0,
      })
    )
    persistence.setAcquireResult(false)

    await expect(store.getState().acquireSubmissionLease()).resolves.toBe(false)

    expect(store.getState().isSubmissionOwner).toBe(false)
    expect(store.getState().session).toMatchObject({
      id: draft.id,
      lifecycle: "processing",
      currentStep: 4,
      tracking: {
        sessionId: "authoritative-processing-session",
        runId: "authoritative-processing-run",
      },
    })
  })

  it("reconciles a newer full durable draft after successful acquisition without rolling back to an older draft", async () => {
    const persistence = createControlledPersistence()
    const store = createStoreWithPersistence(persistence)
    const draft = {
      ...createEmptyQuickIngestSession(),
      updatedAt: 100,
      queueItems: [
        {
          id: "local-stale-row",
          kind: "url" as const,
          url: "https://example.com/local-stale-row",
          sourceRef: {
            kind: "direct_url" as const,
            occurrenceId: "local-stale-row",
            url: "https://example.com/local-stale-row",
          },
          detectedType: "web" as const,
          icon: "Globe",
          fileSize: 0,
          validation: { valid: true },
        },
      ],
    }
    store.setState({ session: draft, persistenceStatus: "ready" })
    await flushPersistence()
    const durable = {
      ...draft,
      updatedAt: 200,
      queueItems: [
        {
          ...draft.queueItems[0],
          id: "durable-full-row",
          url: "https://example.com/durable-full-row",
          sourceRef: {
            kind: "direct_url" as const,
            occurrenceId: "durable-full-row",
            url: "https://example.com/durable-full-row",
          },
        },
      ],
      selectedPreset: "custom" as const,
      customBasePreset: "deep" as const,
      presetConfig: {
        ...draft.presetConfig,
        common: { ...draft.presetConfig.common, perform_analysis: false },
      },
      customOptions: { common: { perform_analysis: false } },
      conferenceBatchMetadata: {
        collectionName: "Durable conference",
        sharedTags: ["durable"],
      },
      openDetail: {
        source: "extension_active_tab" as const,
        action: "playlist_preflight" as const,
        url: "https://youtube.com/playlist?list=durable",
        sourceKind: "youtube_playlist" as const,
      },
    }
    persistence.seedValue(
      JSON.stringify({ state: { session: durable }, version: 0 })
    )

    await expect(store.getState().acquireSubmissionLease()).resolves.toBe(true)
    expect(store.getState()).toMatchObject({
      isSubmissionOwner: true,
      externalAuthorityRevision: 1,
      session: {
        queueItems: [{ id: "durable-full-row" }],
        selectedPreset: "custom",
        customOptions: { common: { perform_analysis: false } },
        conferenceBatchMetadata: { collectionName: "Durable conference" },
        openDetail: {
          action: "playlist_preflight",
          url: "https://youtube.com/playlist?list=durable",
        },
      },
    })

    await flushPersistence()
    persistence.seedValue(
      JSON.stringify({
        state: {
          session: {
            ...durable,
            updatedAt: 150,
            queueItems: [{ ...durable.queueItems[0], id: "older-durable-row" }],
          },
        },
        version: 0,
      })
    )
    await expect(store.getState().acquireSubmissionLease()).resolves.toBe(true)
    expect(store.getState().session?.queueItems[0]?.id).toBe("durable-full-row")
    expect((store.getState() as any).externalAuthorityRevision).toBe(1)
  })

  it("reads durable authority before publishing a rejected lease result after fresh hydration", async () => {
    const persistence = createControlledPersistence()
    const draft = {
      ...createEmptyQuickIngestSession(),
      lifecycle: "draft" as const,
      currentStep: 3 as const,
    }
    persistence.seedValue(
      JSON.stringify({ state: { session: draft }, version: 0 })
    )
    const store = createStoreWithPersistence(persistence)
    await store.persist.rehydrate()
    expect(store.getState().persistenceStatus).toBe("ready")
    expect(store.getState().session).toMatchObject({
      id: draft.id,
      lifecycle: "draft",
      currentStep: 3,
    })
    const attemptsBeforeAcquire = persistence.writeAttempts.length
    const authoritativeEnvelope = JSON.stringify({
      state: {
        session: {
          ...draft,
          lifecycle: "processing",
          currentStep: 4,
          updatedAt: draft.updatedAt + 100,
          tracking: {
            mode: "webui-direct",
            sessionId: "fresh-hydration-authority",
            runId: "fresh-hydration-run",
          },
        },
      },
      version: 0,
    })
    persistence.seedValue(authoritativeEnvelope)
    persistence.setAcquireResult(false)
    persistence.delayNextRead()

    await expect(store.getState().acquireSubmissionLease()).resolves.toBe(false)
    await flushPersistence()

    const leaseAttempts = persistence.writeAttempts.slice(attemptsBeforeAcquire)
    expect(
      leaseAttempts.every((value) =>
        value.includes('"lifecycle":"processing"')
      )
    ).toBe(true)
    expect(JSON.parse(persistence.value || "null")?.state?.session).toMatchObject({
      id: draft.id,
      lifecycle: "processing",
      currentStep: 4,
      tracking: {
        sessionId: "fresh-hydration-authority",
        runId: "fresh-hydration-run",
      },
    })
    expect(store.getState().session).toMatchObject({
      id: draft.id,
      lifecycle: "processing",
      currentStep: 4,
    })
    await expect(store.getState().acquireSubmissionLease()).resolves.toBe(false)
    expect(store.getState().isSubmissionOwner).toBe(false)
    expect(persistence.acquireSubmissionLease).toHaveBeenCalledTimes(2)
  })

  it("merges persisted tracking metadata across direct-session updates", async () => {
    const persistence = createControlledPersistence()
    const store = createStoreWithPersistence(persistence)

    await store.getState().markProcessingTracking({
      mode: "webui-direct",
      sessionId: "qi-direct-merge",
      itemIds: ["url-1", "file-1"],
      startedAt: 1700000000000,
    } as never)

    await store.getState().markProcessingTracking({
      mode: "webui-direct",
      sessionId: "qi-direct-merge",
      batchId: "batch-1",
      batchIds: ["batch-1"],
      jobIds: [77],
    } as any)

    await store.getState().markProcessingTracking({
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

  it("clears completed run tracking when the session returns to draft", async () => {
    const persistence = createControlledPersistence()
    const store = createStoreWithPersistence(persistence)

    await store.getState().markProcessingTracking({
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

  it("deduplicates cumulative job IDs before applying the 500-ID persistence bound", async () => {
    const persistence = createControlledPersistence()
    const store = createStoreWithPersistence(persistence)

    await store.getState().markProcessingTracking({
      mode: "webui-direct",
      sessionId: "qi-cumulative-jobs",
      jobIds: Array.from({ length: 250 }, (_, index) => index + 1),
    })
    await store.getState().markProcessingTracking({
      mode: "webui-direct",
      sessionId: "qi-cumulative-jobs",
      jobIds: Array.from({ length: 500 }, (_, index) => index + 1),
    })

    expect(store.getState().session?.tracking?.jobIds).toEqual(
      Array.from({ length: 500 }, (_, index) => index + 1)
    )
  })

  it("bounds persisted run identity to the backend identifier limit", async () => {
    const persistence = createControlledPersistence()
    const store = createStoreWithPersistence(persistence)
    const oversizedRunId = "r".repeat(256)

    store.getState().markProcessingTracking({
      mode: "webui-direct",
      sessionId: "qi-direct-bounded-run",
      runId: oversizedRunId,
    })

    expect(store.getState().session?.tracking?.runId).toBeUndefined()
    await flushPersistence()
    expect(persistence.value).not.toContain(oversizedRunId)

    const maximumRunId = ` ${"r".repeat(255)} `
    store.getState().markProcessingTracking({
      mode: "webui-direct",
      sessionId: "qi-direct-bounded-run",
      runId: maximumRunId,
    })

    expect(store.getState().session?.tracking?.runId).toBe("r".repeat(255))
  })

  it("persists version-2 submission state before a run id exists", async () => {
    const persistence = createControlledPersistence()
    const store = createStoreWithPersistence(persistence)

    store.getState().markProcessingTracking({
      mode: "webui-direct",
      sessionId: "qi-direct-submission-intent",
      submissionState: "creating_run",
      submissionOccurrenceIds: ["occ-submission-intent"],
      startedAt: 1700000000000,
    } as any)

    expect(store.getState().session?.tracking).toMatchObject({
      sessionId: "qi-direct-submission-intent",
      submissionState: "creating_run",
      submissionOccurrenceIds: ["occ-submission-intent"],
    })
    await flushPersistence()
    expect(persistence.value).toContain(
      '"submissionState":"creating_run"'
    )

    const rehydratedStore = createStoreWithPersistence(persistence)
    await rehydratedStore.persist.rehydrate()
    const rehydrated = rehydratedStore.getState().session
    expect(rehydrated?.tracking).toMatchObject({
      submissionState: "creating_run",
      runId: undefined,
    })
  })

  it("does not resolve run tracking publication before its durable write", async () => {
    const persistence = createControlledPersistence()
    const store = createStoreWithPersistence(persistence)
    store.getState().createDraftSession()
    await flushPersistence()

    let releaseWrite!: () => void
    persistence.blockNextWrite(
      new Promise<void>((resolve) => {
        releaseWrite = resolve
      })
    )

    let resolved = false
    const publication = Promise.resolve(
      store.getState().markProcessingTracking({
        mode: "webui-direct",
        sessionId: "qi-direct-durable-run-marker",
        submissionState: "run_created",
        submissionOccurrenceIds: ["occ-durable-run-marker"],
        runId: "run-durable-marker",
      })
    ).then(() => {
      resolved = true
    })

    await Promise.resolve()
    const resolvedBeforeWrite = resolved
    releaseWrite()
    await publication
    await persistence.flush()

    expect(resolvedBeforeWrite).toBe(false)
    expect(JSON.parse(persistence.value || "null")?.state?.session).toMatchObject({
      lifecycle: "processing",
      tracking: {
        submissionState: "run_created",
        runId: "run-durable-marker",
      },
    })
  })

  it("rejects run tracking publication when its durable write fails", async () => {
    const persistence = createControlledPersistence()
    const store = createStoreWithPersistence(persistence)
    store.getState().createDraftSession()
    await flushPersistence()

    const failure = new DOMException("blocked", "SecurityError")
    persistence.failNextWrite(failure)

    await expect(
      store.getState().markProcessingTracking({
        mode: "webui-direct",
        sessionId: "qi-direct-failed-run-marker",
        submissionState: "run_created",
        submissionOccurrenceIds: ["occ-failed-run-marker"],
        runId: "run-failed-marker",
      })
    ).rejects.toBe(failure)
    expect(store.getState().persistenceStatus).toBe("unavailable")
  })

  it("bounds dedicated submission occurrence recovery identities", async () => {
    const persistence = createControlledPersistence()
    const store = createStoreWithPersistence(persistence)

    store.getState().markProcessingTracking({
      mode: "webui-direct",
      sessionId: "qi-direct-bounded-submission-occurrences",
      submissionState: "creating_run",
      submissionOccurrenceIds: [
        ...Array.from({ length: 501 }, (_, index) => `occ-${index + 1}`),
        "x".repeat(256),
      ],
    } as any)

    const occurrenceIds = store.getState().session?.tracking
      ?.submissionOccurrenceIds
    expect(occurrenceIds).toHaveLength(500)
    expect(occurrenceIds?.at(-1)).toBe("occ-500")
    await flushPersistence()
    expect(persistence.value).not.toContain("x".repeat(256))
  })

  it("reconstructs bounded playlist records and fails closed on mismatched authority", () => {
    const store = createQuickIngestSessionStore()

    store.getState().upsertSession({
      queueItems: [
        {
          id: "occ-sanitized",
          sourceRef: {
            kind: "materialized_playlist_item",
            materializationId: "materialization-sanitized",
            occurrenceId: "occ-sanitized",
            injected: true,
          },
          detectedType: "video",
          icon: "Film",
          fileSize: 0,
          validation: { valid: true, errors: ["x".repeat(2001), "kept error"] },
          playlist: {
            ordinal: 7,
            title: "Sanitized title",
            duplicateStatus: "not-a-status",
            materializationExpiresAt: "2099-01-01T00:00:00Z",
            injected: true,
          },
          playlistReview: {
            selected: true,
            duplicatePolicy: "not-a-policy",
            duplicateEvidence: { kind: "unknown", existingMediaId: "42" },
            allowedDuplicatePolicies: ["skip", "skip", "not-a-policy"],
            reviewReason: "Needs review",
            metadataPatch: {
              title: "Edited title",
              author: "x".repeat(501),
              keywordsAdd: ["Research", "research", "video"],
              injected: true,
            },
            editedFields: ["title", "author", "unknown"],
            injected: true,
          },
          injected: true,
        },
        {
          id: "occ-mismatched",
          sourceRef: {
            kind: "direct_url",
            occurrenceId: "different-id",
            url: "https://example.com/mismatched",
          },
          url: "https://example.com/mismatched",
          detectedType: "web",
          icon: "Globe",
          fileSize: 0,
          validation: { valid: true },
        },
      ],
    } as any)

    const [sanitized, mismatched] = store.getState().session?.queueItems || []
    expect(sanitized).toEqual({
      id: "occ-sanitized",
      sourceRef: {
        kind: "materialized_playlist_item",
        materializationId: "materialization-sanitized",
        occurrenceId: "occ-sanitized",
      },
      kind: "url",
      detectedType: "video",
      icon: "Film",
      fileSize: 0,
      validation: { valid: true, errors: ["kept error"] },
      playlist: {
        ordinal: 7,
        title: "Sanitized title",
        materializationExpiresAt: "2099-01-01T00:00:00Z",
      },
      playlistReview: {
        selected: true,
        allowedDuplicatePolicies: ["skip"],
        reviewReason: "Needs review",
        metadataPatch: {
          title: "Edited title",
          keywordsAdd: ["Research", "video"],
        },
        editedFields: ["title"],
      },
    })
    expect(mismatched?.sourceRef).toBeUndefined()
    expect(mismatched?.validation.valid).toBe(false)
    expect(mismatched?.validation.errors).toContain(
      "Reattach this source before processing."
    )
  })

  it("preserves 501 queued items while dropping non-canonical item identifiers", () => {
    const store = createQuickIngestSessionStore()
    const queueItems = [
      {
        id: " padded-id ",
        detectedType: "web",
        icon: "Globe",
        fileSize: 0,
        validation: { valid: true },
      },
      ...Array.from({ length: 501 }, (_, index) => ({
        id: `bounded-${index + 1}`,
        detectedType: "web",
        icon: "Globe",
        fileSize: 0,
        validation: { valid: true },
      })),
    ]

    store.getState().upsertSession({ queueItems } as never)

    expect(store.getState().session?.queueItems).toHaveLength(501)
    expect(store.getState().session?.queueItems[0]?.id).toBe("bounded-1")
    expect(store.getState().session?.queueItems.at(-1)?.id).toBe("bounded-501")
  })

  it("marks a draft incomplete when it exceeds the persisted source safety limit", () => {
    const store = createQuickIngestSessionStore()
    const queueItems = Array.from({ length: 1001 }, (_, index) => ({
      id: `overflow-${index + 1}`,
      detectedType: "web",
      icon: "Globe",
      fileSize: 0,
      validation: { valid: true },
    }))

    store.getState().upsertSession({ queueItems } as never)

    const restored = store.getState().session?.queueItems ?? []
    expect(restored).toHaveLength(1001)
    expect(restored.at(-1)).toMatchObject({
      detectedType: "unknown",
      validation: {
        valid: false,
        errors: [
          "This draft exceeded the 1000-source persistence safety limit. Start a new batch for the omitted sources.",
        ],
      },
    })
  })

  it("canonicalizes a persisted direct URL display cache from its source authority", () => {
    const store = createQuickIngestSessionStore()

    store.getState().upsertSession({
      queueItems: [
        {
          id: "direct-authority",
          sourceRef: {
            kind: "direct_url",
            occurrenceId: "direct-authority",
            url: "https://example.com/authoritative",
          },
          url: "https://cached.example.invalid/stale",
          detectedType: "web",
          icon: "Globe",
          fileSize: 0,
          validation: { valid: true },
        },
      ],
    } as never)

    expect(store.getState().session?.queueItems[0]).toMatchObject({
      url: "https://example.com/authoritative",
      sourceRef: {
        kind: "direct_url",
        occurrenceId: "direct-authority",
        url: "https://example.com/authoritative",
      },
    })
  })

  it("canonicalizes kind and compatible display fields from each source authority", () => {
    const store = createQuickIngestSessionStore()

    store.getState().upsertSession({
      queueItems: [
        {
          id: "corrupt-direct",
          sourceRef: {
            kind: "direct_url",
            occurrenceId: "corrupt-direct",
            url: "https://example.com/direct-authority",
          },
          kind: "file",
          url: "https://cached.example.invalid/direct",
          fileName: "not-a-file.txt",
          fileStub: { key: "not-a-file" },
          detectedType: "web",
          icon: "Globe",
          fileSize: 0,
          validation: { valid: true },
        },
        {
          id: "corrupt-file",
          sourceRef: {
            kind: "file_stub",
            occurrenceId: "corrupt-file",
          },
          kind: "url",
          url: "https://example.com/not-file-authority",
          name: "reattach-me.pdf",
          detectedType: "pdf",
          icon: "FileText",
          fileSize: 128,
          validation: { valid: true },
        },
        {
          id: "corrupt-materialized",
          sourceRef: {
            kind: "materialized_playlist_item",
            materializationId: "canonical-materialization",
            occurrenceId: "corrupt-materialized",
          },
          kind: "file",
          url: "https://cached.example.invalid/materialized-display",
          fileName: "not-materialized.txt",
          detectedType: "video",
          icon: "Film",
          fileSize: 0,
          validation: { valid: true },
          playlist: {
            materializationExpiresAt: "2099-01-01T00:00:00Z",
          },
        },
      ],
    } as never)

    const [direct, file, materialized] = store.getState().session?.queueItems ?? []
    expect(direct).toMatchObject({
      kind: "url",
      url: "https://example.com/direct-authority",
      validation: { valid: true },
    })
    expect(direct?.fileName).toBeUndefined()
    expect(direct?.fileStub).toBeUndefined()

    expect(file).toMatchObject({
      kind: "file",
      fileName: "reattach-me.pdf",
      validation: {
        valid: false,
        errors: ["Reattach this source before processing."],
      },
    })
    expect(file?.url).toBeUndefined()

    expect(materialized).toMatchObject({
      kind: "url",
      url: "https://cached.example.invalid/materialized-display",
      validation: { valid: true },
    })
    expect(materialized?.fileName).toBeUndefined()
  })

  it("rehydrates corrupt materialized drafts without throwing or URL fallback authority", async () => {
    const session = {
      ...createEmptyQuickIngestSession(),
      queueItems: [
        {
          id: "old-materialized-row",
          url: "https://cached.example.invalid/display-only",
          detectedType: "video",
          icon: "Film",
          fileSize: 0,
          validation: { valid: true },
          playlist: {
            title: "Old row",
            materializationExpiresAt: "2099-01-01T00:00:00Z",
          },
          playlistReview: {
            selected: true,
            duplicatePolicy: "unknown-old-policy",
          },
        },
      ],
    }
    const persistence = createControlledPersistence()
    persistence.seedValue(JSON.stringify({ state: { session }, version: 0 }))

    let restored: ReturnType<typeof createQuickIngestSessionStore> | undefined
    expect(() => {
      restored = createStoreWithPersistence(persistence)
    }).not.toThrow()
    await restored?.persist.rehydrate()
    const row = restored?.getState().session?.queueItems[0]
    expect(row?.sourceRef).toBeUndefined()
    expect(row?.url).toBe("https://cached.example.invalid/display-only")
    expect(row?.validation.valid).toBe(false)
    expect(row?.playlistReview?.duplicatePolicy).toBeUndefined()
  })

  it("fails closed when persisted materialization cues lose source authority and expiry", () => {
    const store = createQuickIngestSessionStore()

    store.getState().upsertSession({
      queueItems: [
        {
          id: "orphaned-materialized-cues",
          kind: "url",
          url: "https://cached.example.invalid/display-only-cues",
          detectedType: "video",
          icon: "Film",
          fileSize: 0,
          validation: { valid: true },
          playlist: {
            playlistId: "playlist-with-lost-authority",
            ordinal: 4,
          },
        },
      ],
    } as never)

    const row = store.getState().session?.queueItems[0]
    expect(row?.sourceRef).toBeUndefined()
    expect(row).toMatchObject({
      url: "https://cached.example.invalid/display-only-cues",
      validation: {
        valid: false,
        errors: ["Reattach this source before processing."],
      },
    })
  })

  it("retains server-valid source URLs and display metadata at contract bounds", () => {
    const sourceUrl = "u".repeat(8192)
    const title = "t".repeat(2000)
    const store = createQuickIngestSessionStore()

    store.getState().upsertSession({
      queueItems: [
        {
          id: "bounded-materialized",
          sourceRef: {
            kind: "materialized_playlist_item",
            materializationId: "bounded-materialization",
            occurrenceId: "bounded-materialized",
          },
          detectedType: "video",
          icon: "Film",
          fileSize: 0,
          validation: { valid: true },
          playlist: {
            title,
            sourceUrl,
            materializationExpiresAt: "2099-01-01T00:00:00Z",
          },
        },
      ],
    })

    expect(store.getState().session?.queueItems[0]?.playlist).toMatchObject({
      title,
      sourceUrl,
    })
  })

  it("clears disallowed policies and unresolved in-run duplicate evidence", () => {
    const materializedItem = (id: string, playlistReview?: Record<string, unknown>) => ({
      id,
      sourceRef: {
        kind: "materialized_playlist_item" as const,
        materializationId: "consistent-review-materialization",
        occurrenceId: id,
      },
      detectedType: "video" as const,
      icon: "Film",
      fileSize: 0,
      validation: { valid: true },
      playlist: {
        duplicateStatus: "duplicate_in_batch" as const,
        materializationExpiresAt: "2099-01-01T00:00:00Z",
      },
      playlistReview: { selected: true, ...playlistReview },
    })
    const store = createQuickIngestSessionStore()
    store.getState().upsertSession({
      queueItems: [
        materializedItem("duplicate-target"),
        materializedItem("disallowed-policy", {
          duplicatePolicy: "overwrite",
          allowedDuplicatePolicies: ["skip"],
          duplicateEvidence: {
            kind: "in_run",
            existingMediaId: null,
            duplicateOfOccurrenceId: "duplicate-target",
          },
        }),
        materializedItem("missing-target", {
          duplicatePolicy: "overwrite",
          allowedDuplicatePolicies: ["overwrite"],
          duplicateEvidence: {
            kind: "in_run",
            existingMediaId: null,
            duplicateOfOccurrenceId: "not-queued",
          },
        }),
      ],
    })

    const [, disallowed, unresolved] = store.getState().session?.queueItems || []
    expect(disallowed?.playlistReview).toMatchObject({
      allowedDuplicatePolicies: ["skip"],
      duplicateEvidence: {
        kind: "in_run",
        duplicateOfOccurrenceId: "duplicate-target",
      },
    })
    expect(disallowed?.playlistReview?.duplicatePolicy).toBeUndefined()
    expect(unresolved?.playlistReview?.duplicatePolicy).toBeUndefined()
    expect(unresolved?.playlistReview?.duplicateEvidence).toBeUndefined()
  })
})
