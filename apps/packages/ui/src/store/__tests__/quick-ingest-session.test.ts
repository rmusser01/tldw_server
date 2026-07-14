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
    } as never)

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

  it("bounds persisted run identity to the backend identifier limit", () => {
    const store = createQuickIngestSessionStore()
    const oversizedRunId = "r".repeat(256)

    store.getState().markProcessingTracking({
      mode: "webui-direct",
      sessionId: "qi-direct-bounded-run",
      runId: oversizedRunId,
    })

    expect(store.getState().session?.tracking?.runId).toBeUndefined()
    expect(sessionStorage.getItem(STORAGE_KEY)).not.toContain(oversizedRunId)

    const maximumRunId = ` ${"r".repeat(255)} `
    store.getState().markProcessingTracking({
      mode: "webui-direct",
      sessionId: "qi-direct-bounded-run",
      runId: maximumRunId,
    })

    expect(store.getState().session?.tracking?.runId).toBe("r".repeat(255))
  })

  it("persists version-2 submission state before a run id exists", () => {
    const store = createQuickIngestSessionStore()

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
    expect(sessionStorage.getItem(STORAGE_KEY)).toContain(
      '"submissionState":"creating_run"'
    )

    const rehydrated = createQuickIngestSessionStore().getState().session
    expect(rehydrated?.tracking).toMatchObject({
      submissionState: "creating_run",
      runId: undefined,
    })
  })

  it("bounds dedicated submission occurrence recovery identities", () => {
    const store = createQuickIngestSessionStore()

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
    expect(sessionStorage.getItem(STORAGE_KEY)).not.toContain("x".repeat(256))
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

  it("rehydrates corrupt materialized drafts without throwing or URL fallback authority", () => {
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
    sessionStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ state: { session }, version: 0 })
    )

    let restored: ReturnType<typeof createQuickIngestSessionStore> | undefined
    expect(() => {
      restored = createQuickIngestSessionStore()
    }).not.toThrow()
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
