import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  runtimeId: undefined as string | undefined,
  manifestVersion: 3,
  sendMessage: vi.fn(),
  bgRequest: vi.fn(),
  bgUpload: vi.fn()
}))

vi.mock("wxt/browser", () => ({
  browser: {
    runtime: {
      get id() {
        return mocks.runtimeId
      },
      getManifest: () => ({
        manifest_version: mocks.manifestVersion
      }),
      sendMessage: (...args: unknown[]) => mocks.sendMessage(...args)
    }
  }
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.bgRequest(...args),
  bgUpload: (...args: unknown[]) => mocks.bgUpload(...args)
}))

import {
  __resetQuickIngestRuntimeHealthForTests,
  cancelQuickIngestSession,
  startQuickIngestSession,
  submitQuickIngestBatch
} from "@/services/tldw/quick-ingest-batch"
import { DUPLICATE_SKIP_MESSAGE } from "@/components/Common/QuickIngest/constants"

describe("submitQuickIngestBatch", () => {
  beforeEach(() => {
    __resetQuickIngestRuntimeHealthForTests()
    vi.useRealTimers()
    mocks.runtimeId = undefined
    mocks.manifestVersion = 3
    mocks.sendMessage.mockReset()
    mocks.bgRequest.mockReset()
    mocks.bgUpload.mockReset()
  })

  it("uses direct upload path when extension runtime id is unavailable", async () => {
    mocks.bgUpload.mockResolvedValue({
      batch_id: "batch-1",
      jobs: [{ id: 101 }]
    })
    mocks.bgRequest.mockResolvedValue({
      ok: true,
      data: {
        status: "completed",
        result: { media_id: "m1" }
      }
    })

    const result = await submitQuickIngestBatch({
      entries: [
        {
          id: "entry-1",
          url: "https://example.com/article",
          type: "document"
        }
      ],
      files: [],
      storeRemote: true,
      processOnly: false,
      common: {
        perform_analysis: true,
        perform_chunking: false,
        overwrite_existing: false
      },
      advancedValues: {}
    })

    expect(mocks.sendMessage).not.toHaveBeenCalled()
    expect(mocks.bgUpload).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/ingest/jobs",
        method: "POST",
        fields: expect.objectContaining({
          media_type: "document",
          urls: ["https://example.com/article"]
        })
      })
    )
    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/ingest/jobs/101",
        method: "GET"
      })
    )
    expect(result.ok).toBe(true)
    expect(result.results?.[0]).toMatchObject({
      id: "entry-1",
      status: "ok"
    })
  })

  it("marks duplicate remote file uploads as skipped with guidance", async () => {
    mocks.bgUpload.mockResolvedValue({
      batch_id: "batch-duplicate-file",
      jobs: [{ id: 303 }]
    })
    mocks.bgRequest.mockResolvedValue({
      ok: true,
      data: {
        status: "completed",
        result: {
          media_id: "m-duplicate-file",
          db_message: "Media 'existing.pdf' already exists. Overwrite not enabled."
        }
      }
    })

    const result = await submitQuickIngestBatch({
      entries: [],
      files: [
        {
          id: "file-duplicate-1",
          name: "existing.pdf",
          type: "application/pdf",
          data: [1, 2, 3]
        }
      ],
      storeRemote: true,
      processOnly: false,
      common: {
        perform_analysis: true,
        perform_chunking: false,
        overwrite_existing: false
      },
      advancedValues: {}
    })

    expect(result.ok).toBe(true)
    expect(result.results?.[0]).toMatchObject({
      id: "file-duplicate-1",
      status: "ok",
      outcome: "skipped",
      fileName: "existing.pdf",
      message: DUPLICATE_SKIP_MESSAGE
    })
  })

  it("surfaces completed ingest jobs with backend error payloads as failed results", async () => {
    mocks.bgUpload.mockResolvedValue({
      batch_id: "batch-completed-error",
      jobs: [{ id: 909 }]
    })
    mocks.bgRequest.mockResolvedValue({
      ok: true,
      data: {
        status: "completed",
        result: {
          status: "Error",
          error: "File preparation/download failed: Port not allowed: 3000"
        }
      }
    })

    const result = await submitQuickIngestBatch({
      entries: [
        {
          id: "entry-completed-error",
          url: "http://127.0.0.1:3000/e2e/quick-ingest-source.html",
          type: "document"
        }
      ],
      files: [],
      storeRemote: true,
      processOnly: false,
      common: {
        perform_analysis: true,
        perform_chunking: false,
        overwrite_existing: false
      },
      advancedValues: {}
    })

    expect(result.ok).toBe(true)
    expect(result.results?.[0]).toMatchObject({
      id: "entry-completed-error",
      status: "error",
      error: "File preparation/download failed: Port not allowed: 3000"
    })
  })

  it("defaults perform_chunking to true when common options are omitted", async () => {
    mocks.bgUpload.mockResolvedValue({
      batch_id: "batch-default-chunking",
      jobs: [{ id: 202 }]
    })
    mocks.bgRequest.mockResolvedValue({
      ok: true,
      data: {
        status: "completed",
        result: { media_id: "m-default-chunking" }
      }
    })

    await submitQuickIngestBatch({
      entries: [
        {
          id: "entry-default-chunking",
          url: "https://example.com/default-chunking",
          type: "document"
        }
      ],
      files: [],
      storeRemote: true,
      processOnly: false
    } as any)

    expect(mocks.bgUpload).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/ingest/jobs",
        method: "POST",
        fields: expect.objectContaining({
          perform_chunking: true
        })
      })
    )
  })

  it("captures direct batch tracking metadata before polling completes", async () => {
    const onTrackingMetadata = vi.fn()

    mocks.bgUpload.mockResolvedValue({
      batch_id: "batch-1",
      jobs: [{ id: 1234 }]
    })
    mocks.bgRequest.mockResolvedValue({
      ok: true,
      data: {
        status: "completed",
        result: { media_id: "m-track" }
      }
    })

    await submitQuickIngestBatch({
      entries: [
        {
          id: "entry-track-1",
          url: "https://example.com/tracked",
          type: "document"
        }
      ],
      files: [],
      storeRemote: true,
      processOnly: false,
      __quickIngestSessionId: "qi-direct-1",
      onTrackingMetadata
    } as any)

    expect(onTrackingMetadata).toHaveBeenCalledWith(
      expect.objectContaining({
        mode: "webui-direct",
        sessionId: "qi-direct-1",
        batchId: "batch-1",
        batchIds: ["batch-1"],
        jobIds: [1234],
        jobIdToItemId: {
          "1234": "entry-track-1"
        },
        startedAt: expect.any(Number)
      })
    )
  })

  it("emits per-item direct tracking metadata for both url and file submissions", async () => {
    const onTrackingMetadata = vi.fn()

    mocks.bgUpload
      .mockResolvedValueOnce({
        batch_id: "batch-url-1",
        jobs: [{ id: 501 }]
      })
      .mockResolvedValueOnce({
        batch_id: "batch-file-1",
        jobs: [{ id: 601 }]
      })
    mocks.bgRequest
      .mockResolvedValueOnce({
        ok: true,
        data: {
          status: "completed",
          result: { media_id: "m-url-1" }
        }
      })
      .mockResolvedValueOnce({
        ok: true,
        data: {
          status: "completed",
          result: { media_id: "m-file-1" }
        }
      })

    await submitQuickIngestBatch({
      entries: [
        {
          id: "entry-501",
          url: "https://example.com/per-item-url",
          type: "document"
        }
      ],
      files: [
        {
          id: "file-601",
          name: "session-restore.mkv",
          type: "video/x-matroska",
          data: [1, 2, 3]
        }
      ],
      storeRemote: true,
      processOnly: false,
      __quickIngestSessionId: "qi-direct-per-item",
      onTrackingMetadata
    } as any)

    expect(onTrackingMetadata).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({
        sessionId: "qi-direct-per-item",
        batchId: "batch-url-1",
        batchIds: ["batch-url-1"],
        jobIds: [501],
        jobIdToItemId: {
          "501": "entry-501"
        }
      })
    )
    expect(onTrackingMetadata).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        sessionId: "qi-direct-per-item",
        batchId: "batch-file-1",
        batchIds: ["batch-file-1"],
        jobIds: [601],
        jobIdToItemId: {
          "601": "file-601"
        }
      })
    )
  })

  it("forces direct transport for direct-session submits and polls when runtime messaging exists", async () => {
    mocks.runtimeId = "ext-runtime-1"

    mocks.bgUpload.mockResolvedValue({
      batch_id: "batch-direct-transport",
      jobs: [{ id: 818 }]
    })
    mocks.bgRequest.mockResolvedValue({
      ok: true,
      data: {
        status: "completed",
        result: { media_id: "m-direct-transport" }
      }
    })

    await submitQuickIngestBatch({
      entries: [
        {
          id: "entry-direct-transport",
          url: "https://example.com/direct-transport",
          type: "document"
        }
      ],
      files: [],
      storeRemote: true,
      processOnly: false,
      __quickIngestSessionId: "qi-direct-transport",
      common: {
        perform_analysis: true,
        perform_chunking: false,
        overwrite_existing: false
      },
      advancedValues: {}
    })

    expect(mocks.sendMessage).not.toHaveBeenCalledWith(
      expect.objectContaining({
        type: "tldw:quick-ingest-batch"
      })
    )
    expect(mocks.bgUpload).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/ingest/jobs",
        preferDirect: true
      })
    )
    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/ingest/jobs/818",
        method: "GET",
        returnResponse: true,
        preferDirect: true
      })
    )
  })

  it("tracks only submitted direct items when later queue items fail before job creation", async () => {
    const onTrackingMetadata = vi.fn()

    mocks.bgUpload
      .mockResolvedValueOnce({
        batch_id: "batch-first-only",
        jobs: [{ id: 701 }]
      })
      .mockRejectedValueOnce(new Error("submit failed for second item"))
    mocks.bgRequest.mockResolvedValue({
      ok: true,
      data: {
        status: "completed",
        result: { media_id: "m-first-only" }
      }
    })

    const response = await submitQuickIngestBatch({
      entries: [
        {
          id: "entry-first-submitted",
          url: "https://example.com/first-submitted",
          type: "document"
        },
        {
          id: "entry-never-submitted",
          url: "https://example.com/never-submitted",
          type: "document"
        }
      ],
      files: [],
      storeRemote: true,
      processOnly: false,
      __quickIngestSessionId: "qi-direct-partial-submit",
      onTrackingMetadata
    } as any)

    expect(onTrackingMetadata).toHaveBeenCalledTimes(1)
    expect(onTrackingMetadata).toHaveBeenCalledWith(
      expect.objectContaining({
        sessionId: "qi-direct-partial-submit",
        submittedItemIds: ["entry-first-submitted"],
        jobIdToItemId: {
          "701": "entry-first-submitted"
        }
      })
    )
    expect(response.results).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          id: "entry-first-submitted",
          status: "ok"
        }),
        expect.objectContaining({
          id: "entry-never-submitted",
          status: "error"
        })
      ])
    )
  })

  it("cancels direct-session tracked batches through backend cancel endpoint", async () => {
    mocks.bgUpload.mockResolvedValue({
      batch_id: "batch-direct-cancel",
      jobs: [{ id: 777 }]
    })

    let statusPollCount = 0
    mocks.bgRequest.mockImplementation(async (request: { path?: string }) => {
      const path = String(request?.path || "")
      if (path.includes("/api/v1/media/ingest/jobs/cancel?batch_id=batch-direct-cancel")) {
        return { ok: true, data: { success: true } }
      }
      if (path === "/api/v1/media/ingest/jobs/777") {
        statusPollCount += 1
        return { ok: true, data: { status: statusPollCount > 1 ? "cancelled" : "processing" } }
      }
      return { ok: false, error: "unexpected path" }
    })

    const runPromise = submitQuickIngestBatch({
      entries: [
        {
          id: "entry-cancel-1",
          url: "https://example.com/cancel-me",
          type: "document"
        }
      ],
      files: [],
      storeRemote: true,
      processOnly: false,
      __quickIngestSessionId: "direct-session-1",
      common: {
        perform_analysis: true,
        perform_chunking: false,
        overwrite_existing: false
      },
      advancedValues: {}
    })

    await vi.waitFor(() => {
      expect(mocks.bgRequest).toHaveBeenCalledWith(
        expect.objectContaining({
          path: "/api/v1/media/ingest/jobs/777",
          method: "GET",
          preferDirect: true
        })
      )
    })

    const cancelResponse = await cancelQuickIngestSession({
      sessionId: "direct-session-1",
      reason: "user_cancelled"
    })
    const runResult = await runPromise

    expect(cancelResponse).toEqual({ ok: true })
    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: expect.stringContaining(
          "/api/v1/media/ingest/jobs/cancel?batch_id=batch-direct-cancel"
        ),
        method: "POST",
        preferDirect: true
      })
    )
    expect(runResult.results?.[0]).toMatchObject({
      id: "entry-cancel-1",
      status: "error"
    })
  })

  it("stops submitting later direct items once the session is cancelled", async () => {
    vi.useFakeTimers()
    mocks.bgUpload.mockResolvedValue({
      batch_id: "batch-stop-1",
      jobs: [{ id: 901 }]
    })

    let statusPollCount = 0
    mocks.bgRequest.mockImplementation(async (request: { path?: string }) => {
      const path = String(request?.path || "")
      if (path.includes("/api/v1/media/ingest/jobs/cancel?batch_id=batch-stop-1")) {
        return { ok: true, data: { success: true } }
      }
      if (path === "/api/v1/media/ingest/jobs/901") {
        statusPollCount += 1
        return {
          ok: true,
          data: { status: statusPollCount > 1 ? "cancelled" : "processing" }
        }
      }
      return { ok: false, error: "unexpected path" }
    })

    const runPromise = submitQuickIngestBatch({
      entries: [
        {
          id: "entry-stop-1",
          url: "https://example.com/stop-first",
          type: "document"
        },
        {
          id: "entry-stop-2",
          url: "https://example.com/stop-second",
          type: "document"
        }
      ],
      files: [],
      storeRemote: true,
      processOnly: false,
      __quickIngestSessionId: "direct-session-stop",
      common: {
        perform_analysis: true,
        perform_chunking: false,
        overwrite_existing: false
      },
      advancedValues: {}
    })

    await vi.waitFor(() => {
      expect(mocks.bgRequest).toHaveBeenCalledWith(
        expect.objectContaining({
          path: "/api/v1/media/ingest/jobs/901",
          method: "GET"
        })
      )
    })

    await cancelQuickIngestSession({
      sessionId: "direct-session-stop",
      reason: "user_cancelled"
    })
    await vi.advanceTimersByTimeAsync(2_000)
    const runResult = await runPromise

    expect(mocks.bgUpload).toHaveBeenCalledTimes(1)
    expect(runResult.results?.map((item) => item.id)).not.toContain("entry-stop-2")
  })

  it("uses extension message transport when extension runtime is available", async () => {
    mocks.runtimeId = "ext-1"
    mocks.sendMessage
      .mockResolvedValueOnce({ ok: true })
      .mockResolvedValueOnce({
      ok: true,
      results: [{ id: "entry-1", status: "ok", type: "document" }]
      })

    const result = await submitQuickIngestBatch({
      entries: [
        {
          id: "entry-1",
          url: "https://example.com/article",
          type: "document"
        }
      ],
      files: [],
      storeRemote: true,
      processOnly: false,
      common: {
        perform_analysis: true,
        perform_chunking: false,
        overwrite_existing: false
      },
      advancedValues: {}
    })

    expect(mocks.sendMessage).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({
        type: "tldw:ping"
      })
    )
    expect(mocks.sendMessage).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        type: "tldw:quick-ingest-batch",
        payload: expect.objectContaining({
          entries: expect.any(Array)
        })
      })
    )
    expect(mocks.bgUpload).not.toHaveBeenCalled()
    expect(mocks.bgRequest).not.toHaveBeenCalled()
    expect(result.ok).toBe(true)
  })

  it("falls back to direct mode when runtime ping preflight times out", async () => {
    vi.useFakeTimers()
    mocks.runtimeId = "ext-1"
    mocks.sendMessage.mockImplementation(() => new Promise(() => undefined))
    mocks.bgUpload.mockResolvedValue({
      batch_id: "batch-direct-fallback",
      jobs: [{ id: 808 }]
    })
    mocks.bgRequest.mockResolvedValue({
      ok: true,
      data: {
        status: "completed",
        result: { media_id: "m-fallback" }
      }
    })

    const resultPromise = submitQuickIngestBatch({
      entries: [
        {
          id: "entry-fallback",
          url: "https://example.com/runtime-fallback",
          type: "document"
        }
      ],
      files: [],
      storeRemote: true,
      processOnly: false,
      common: {
        perform_analysis: true,
        perform_chunking: false,
        overwrite_existing: false
      },
      advancedValues: {}
    })

    await vi.advanceTimersByTimeAsync(401)
    const result = await resultPromise

    expect(mocks.sendMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "tldw:ping"
      })
    )
    expect(mocks.bgUpload).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/ingest/jobs",
        method: "POST",
        preferDirect: true
      })
    )
    expect(result.results?.[0]).toMatchObject({
      id: "entry-fallback",
      status: "ok"
    })
  })

  it("routes html process-only entries through process-web-scraping", async () => {
    mocks.bgRequest.mockResolvedValue({ content: "processed" })

    const result = await submitQuickIngestBatch({
      entries: [
        {
          id: "entry-html",
          url: "https://example.com/page",
          type: "html"
        }
      ],
      files: [],
      storeRemote: false,
      processOnly: true,
      common: {
        perform_analysis: true,
        perform_chunking: false,
        overwrite_existing: false
      },
      advancedValues: {
        custom_headers: '{"x-test":"1"}'
      }
    })

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/process-web-scraping",
        method: "POST",
        body: expect.objectContaining({
          url_input: "https://example.com/page",
          scrape_method: "Individual URLs"
        })
      })
    )
    expect(result.results?.[0]).toMatchObject({
      id: "entry-html",
      status: "ok",
      type: "html"
    })
  })

  it("routes local files through direct process endpoints in web runtime", async () => {
    mocks.bgUpload.mockResolvedValue({ result: "ok" })

    const result = await submitQuickIngestBatch({
      entries: [],
      files: [
        {
          id: "file-1",
          name: "notes.txt",
          type: "text/plain",
          data: [1, 2, 3]
        }
      ],
      storeRemote: false,
      processOnly: true,
      common: {
        perform_analysis: true,
        perform_chunking: true,
        overwrite_existing: false
      },
      advancedValues: {}
    })

    expect(mocks.bgUpload).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/process-documents",
        method: "POST",
        file: expect.objectContaining({
          name: "notes.txt"
        })
      })
    )
    expect(result.results?.[0]).toMatchObject({
      id: "file-1",
      status: "ok"
    })
  })

  it("falls back to persistent /media/add when ingest-job submission is rejected by the concurrent-job limit", async () => {
    const queueLimitError = new Error(
      "User 1 has reached the maximum concurrent job limit (5)"
    ) as Error & { status?: number }
    queueLimitError.status = 429

    mocks.bgUpload
      .mockRejectedValueOnce(queueLimitError)
      .mockResolvedValueOnce({
        results: [
          {
            status: "Success",
            db_id: 321,
            metadata: { title: "Queued article fallback" }
          }
        ]
      })

    const result = await submitQuickIngestBatch({
      entries: [
        {
          id: "entry-queue-limit",
          url: "https://example.com/article",
          type: "auto"
        }
      ],
      files: [],
      storeRemote: true,
      processOnly: false,
      common: {
        perform_analysis: true,
        perform_chunking: false,
        overwrite_existing: false
      },
      advancedValues: {}
    })

    expect(mocks.bgUpload).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({
        path: "/api/v1/media/ingest/jobs",
        method: "POST"
      })
    )
    expect(mocks.bgUpload).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        path: "/api/v1/media/add",
        method: "POST",
        fields: expect.objectContaining({
          urls: ["https://example.com/article"]
        })
      })
    )
    expect(result.results?.[0]).toMatchObject({
      id: "entry-queue-limit",
      status: "ok",
      data: {
        results: [
          expect.objectContaining({
            media_id: 321
          })
        ]
      }
    })
  })

  it("returns a direct session ack for mv3 extension pages", async () => {
    mocks.runtimeId = "ext-1"

    const ack = await startQuickIngestSession({
      entries: [
        {
          id: "entry-1",
          url: "https://example.com/article",
          type: "document"
        }
      ],
      files: [],
      storeRemote: true,
      processOnly: false,
      common: {
        perform_analysis: true,
        perform_chunking: false,
        overwrite_existing: false
      },
      advancedValues: {}
    })

    expect(mocks.sendMessage).not.toHaveBeenCalled()
    expect(ack.ok).toBe(true)
    expect(ack.sessionId).toMatch(/^qi-direct-/)
  })

  it("returns a direct session ack when runtime ping preflight times out", async () => {
    vi.useFakeTimers()
    mocks.runtimeId = "ext-1"
    mocks.manifestVersion = 2
    mocks.sendMessage.mockImplementation(() => new Promise(() => undefined))

    const ackPromise = startQuickIngestSession({
      entries: [
        {
          id: "entry-1",
          url: "https://example.com/article",
          type: "document"
        }
      ],
      files: [],
      storeRemote: true,
      processOnly: false,
      common: {
        perform_analysis: true,
        perform_chunking: false,
        overwrite_existing: false
      },
      advancedValues: {}
    })

    await vi.advanceTimersByTimeAsync(401)
    const ack = await ackPromise

    expect(mocks.sendMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "tldw:ping"
      })
    )
    expect(ack.ok).toBe(true)
    expect(ack.sessionId).toMatch(/^qi-direct-/)
  })

  it("bypasses background batch orchestration for explicit direct sessions", async () => {
    mocks.runtimeId = "ext-1"
    mocks.bgUpload.mockResolvedValue({
      batch_id: "batch-direct-explicit",
      jobs: [{ id: 818 }]
    })
    mocks.bgRequest.mockResolvedValue({
      ok: true,
      data: {
        status: "completed",
        result: { media_id: "m-direct-explicit" }
      }
    })

    const result = await submitQuickIngestBatch({
      entries: [
        {
          id: "entry-direct-explicit",
          url: "https://example.com/direct-explicit",
          type: "document"
        }
      ],
      files: [],
      storeRemote: true,
      processOnly: false,
      __quickIngestSessionId: "qi-direct-explicit",
      common: {
        perform_analysis: true,
        perform_chunking: false,
        overwrite_existing: false
      },
      advancedValues: {}
    })

    expect(mocks.sendMessage).not.toHaveBeenCalled()
    expect(mocks.bgUpload).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/ingest/jobs",
        method: "POST"
      })
    )
    expect(result.results?.[0]).toMatchObject({
      id: "entry-direct-explicit",
      status: "ok"
    })
  })

  it("cancels direct sessions without routing through background session runtime", async () => {
    mocks.runtimeId = "ext-1"

    const response = await cancelQuickIngestSession({
      sessionId: "qi-direct-active",
      reason: "user_cancelled",
      tracking: {
        mode: "webui-direct",
        batchIds: ["batch-direct-cancel"],
        batchId: "batch-direct-cancel"
      }
    } as any)

    expect(mocks.sendMessage).not.toHaveBeenCalled()
    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: expect.stringContaining("batch_id=batch-direct-cancel"),
        method: "POST",
        preferDirect: true
      })
    )
    expect(response).toEqual({ ok: true })
  })

  it("sends explicit cancel message with session id", async () => {
    mocks.runtimeId = "ext-1"
    mocks.manifestVersion = 2
    mocks.sendMessage.mockResolvedValueOnce({ ok: true }).mockResolvedValueOnce({
      ok: true
    })

    const response = await cancelQuickIngestSession({
      sessionId: "qi-session-123",
      reason: "user_cancelled"
    })

    expect(mocks.sendMessage).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({
        type: "tldw:ping"
      })
    )
    expect(mocks.sendMessage).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        type: "tldw:quick-ingest/cancel",
        payload: {
          sessionId: "qi-session-123",
          reason: "user_cancelled"
        }
      })
    )
    expect(response).toEqual({ ok: true })
  })

  it("cancels persisted direct batches after refresh using tracking metadata", async () => {
    const response = await cancelQuickIngestSession({
      sessionId: "qi-direct-restored",
      reason: "user_cancelled",
      tracking: {
        mode: "webui-direct",
        batchIds: ["batch-restore-1", "batch-restore-2"],
        batchId: "batch-restore-2"
      }
    } as any)

    expect(response).toEqual({ ok: true })
    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: expect.stringContaining("batch_id=batch-restore-1"),
        method: "POST",
        preferDirect: true
      })
    )
    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: expect.stringContaining("batch_id=batch-restore-2"),
        method: "POST",
        preferDirect: true
      })
    )
  })
})
