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

  it("sends auto chunking fields and suppresses stale manual fields", async () => {
    mocks.bgUpload.mockResolvedValue({
      batch_id: "batch-auto-chunking",
      jobs: [{ id: 204 }]
    })
    mocks.bgRequest.mockResolvedValue({
      ok: true,
      data: {
        status: "completed",
        result: { media_id: "m-auto-chunking" }
      }
    })

    await submitQuickIngestBatch({
      entries: [
        {
          id: "entry-auto-chunking",
          url: "https://example.com/auto-chunking",
          type: "document"
        }
      ],
      files: [],
      storeRemote: true,
      processOnly: false,
      common: {
        perform_analysis: true,
        perform_chunking: true,
        overwrite_existing: false,
        chunking_mode: "auto",
        auto_chunking_goal: "qa_search",
        auto_chunking_use_llm: true
      },
      advancedValues: {
        perform_analysis: false,
        overwrite_existing: true,
        chunk_method: "tokens",
        chunk_size: 1200,
        chunk_overlap: 200,
        use_adaptive_chunking: true,
        hierarchical_chunking: true,
        hierarchical_template: { boundaries: [{ kind: "heading" }] },
        transcription_model: "parakeet-standard"
      },
      chunkingTemplateName: "manual-template",
      autoApplyTemplate: true
    } as any)

    expect(mocks.bgUpload).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/ingest/jobs",
        method: "POST",
        fields: expect.objectContaining({
          perform_chunking: true,
          perform_analysis: true,
          overwrite_existing: false,
          chunking_mode: "auto",
          auto_chunking_goal: "qa_search",
          auto_chunking_use_llm: true,
          transcription_model: "parakeet-standard"
        })
      })
    )
    const fields = mocks.bgUpload.mock.calls[0][0].fields
    expect(fields).not.toHaveProperty("chunk_method")
    expect(fields).not.toHaveProperty("chunk_size")
    expect(fields).not.toHaveProperty("chunk_overlap")
    expect(fields).not.toHaveProperty("use_adaptive_chunking")
    expect(fields).not.toHaveProperty("hierarchical_chunking")
    expect(fields).not.toHaveProperty("hierarchical_template")
    expect(fields).not.toHaveProperty("chunking_template_name")
    expect(fields).not.toHaveProperty("auto_apply_template")
  })

  it("sends manual chunking fields and templates in manual mode", async () => {
    mocks.bgUpload.mockResolvedValue({
      batch_id: "batch-manual-chunking",
      jobs: [{ id: 205 }]
    })
    mocks.bgRequest.mockResolvedValue({
      ok: true,
      data: {
        status: "completed",
        result: { media_id: "m-manual-chunking" }
      }
    })

    await submitQuickIngestBatch({
      entries: [
        {
          id: "entry-manual-chunking",
          url: "https://example.com/manual-chunking",
          type: "document"
        }
      ],
      files: [],
      storeRemote: true,
      processOnly: false,
      common: {
        perform_analysis: true,
        perform_chunking: true,
        overwrite_existing: false,
        chunking_mode: "manual",
        auto_chunking_goal: "qa_search",
        auto_chunking_use_llm: true
      },
      advancedValues: {
        chunk_method: "tokens",
        chunk_size: 900,
        chunk_overlap: 100
      },
      chunkingTemplateName: "manual-template",
      autoApplyTemplate: true
    } as any)

    expect(mocks.bgUpload).toHaveBeenCalledWith(
      expect.objectContaining({
        fields: expect.objectContaining({
          perform_chunking: true,
          chunking_mode: "manual",
          chunk_method: "tokens",
          chunk_size: 900,
          chunk_overlap: 100,
          chunking_template_name: "manual-template",
          auto_apply_template: true
        })
      })
    )
    const fields = mocks.bgUpload.mock.calls[0][0].fields
    expect(fields).not.toHaveProperty("auto_chunking_goal")
    expect(fields).not.toHaveProperty("auto_chunking_use_llm")
  })

  it("omits auto and manual chunking controls when chunking is disabled", async () => {
    mocks.bgUpload.mockResolvedValue({
      batch_id: "batch-disabled-chunking",
      jobs: [{ id: 206 }]
    })
    mocks.bgRequest.mockResolvedValue({
      ok: true,
      data: {
        status: "completed",
        result: { media_id: "m-disabled-chunking" }
      }
    })

    await submitQuickIngestBatch({
      entries: [
        {
          id: "entry-disabled-chunking",
          url: "https://example.com/disabled-chunking",
          type: "document"
        }
      ],
      files: [],
      storeRemote: true,
      processOnly: false,
      common: {
        perform_analysis: true,
        perform_chunking: false,
        overwrite_existing: false,
        chunking_mode: "auto",
        auto_chunking_goal: "navigation_summary",
        auto_chunking_use_llm: true
      },
      advancedValues: {
        chunk_size: 900,
        chunk_overlap: 100
      },
      chunkingTemplateName: "manual-template",
      autoApplyTemplate: true
    } as any)

    expect(mocks.bgUpload).toHaveBeenCalledWith(
      expect.objectContaining({
        fields: expect.objectContaining({
          perform_chunking: false
        })
      })
    )
    const fields = mocks.bgUpload.mock.calls[0][0].fields
    expect(fields).not.toHaveProperty("chunking_mode")
    expect(fields).not.toHaveProperty("auto_chunking_goal")
    expect(fields).not.toHaveProperty("auto_chunking_use_llm")
    expect(fields).not.toHaveProperty("chunk_size")
    expect(fields).not.toHaveProperty("chunk_overlap")
    expect(fields).not.toHaveProperty("chunking_template_name")
    expect(fields).not.toHaveProperty("auto_apply_template")
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

  it("routes persisted ordinary web URLs through process-web-scraping", async () => {
    mocks.bgRequest.mockResolvedValue({
      status: "persist-ok",
      media_ids: [123],
      total_articles: 1
    })

    const result = await submitQuickIngestBatch({
      entries: [
        {
          id: "entry-web-persist",
          url: "https://example.com/article",
          type: "auto"
        }
      ],
      files: [],
      storeRemote: true,
      processOnly: false,
      common: {
        perform_analysis: false,
        perform_chunking: true,
        overwrite_existing: false
      },
      advancedValues: {}
    })

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/process-web-scraping",
        method: "POST",
        body: expect.objectContaining({
          url_input: "https://example.com/article",
          scrape_method: "Individual URLs",
          mode: "persist"
        })
      })
    )
    expect(
      mocks.bgUpload.mock.calls.some(
        ([request]) => request?.path === "/api/v1/media/ingest/jobs"
      )
    ).toBe(false)
    expect(result.results?.[0]).toMatchObject({
      id: "entry-web-persist",
      status: "ok",
      type: "html",
      mediaId: 123,
      persisted: true
    })
  })

  it("keeps direct Markdown URLs on the document ingest job route", async () => {
    mocks.bgUpload.mockResolvedValue({
      batch_id: "batch-markdown-url",
      jobs: [{ id: 778 }]
    })
    mocks.bgRequest.mockResolvedValue({
      ok: true,
      data: {
        status: "completed",
        result: { media_id: "m-markdown-url" }
      }
    })

    const result = await submitQuickIngestBatch({
      entries: [
        {
          id: "entry-markdown-url",
          url: "https://example.com/source.md",
          type: "auto"
        }
      ],
      files: [],
      storeRemote: true,
      processOnly: false,
      common: {
        perform_analysis: false,
        perform_chunking: true,
        overwrite_existing: false
      },
      advancedValues: {}
    })

    expect(mocks.bgUpload).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/ingest/jobs",
        method: "POST",
        fields: expect.objectContaining({
          media_type: "document",
          urls: ["https://example.com/source.md"]
        })
      })
    )
    expect(
      mocks.bgRequest.mock.calls.some(
        ([request]) => request?.path === "/api/v1/media/process-web-scraping"
      )
    ).toBe(false)
    expect(result.results?.[0]).toMatchObject({
      id: "entry-markdown-url",
      status: "ok",
      type: "document"
    })
  })

  it("passes auto chunking fields to process-web-scraping JSON requests", async () => {
    mocks.bgRequest.mockResolvedValue({ content: "processed" })

    await submitQuickIngestBatch({
      entries: [
        {
          id: "entry-html-auto",
          url: "https://example.com/page",
          type: "html"
        }
      ],
      files: [],
      storeRemote: false,
      processOnly: true,
      common: {
        perform_analysis: true,
        perform_chunking: true,
        overwrite_existing: false,
        chunking_mode: "auto",
        auto_chunking_goal: "navigation_summary",
        auto_chunking_use_llm: false
      },
      advancedValues: {
        custom_headers: '{"x-test":"1"}',
        chunk_size: 1200,
        chunk_overlap: 200
      }
    } as any)

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/process-web-scraping",
        method: "POST",
        body: expect.objectContaining({
          url_input: "https://example.com/page",
          scrape_method: "Individual URLs",
          chunking_mode: "auto",
          auto_chunking_goal: "navigation_summary",
          custom_headers: { "x-test": "1" }
        })
      })
    )
    const body = mocks.bgRequest.mock.calls[0][0].body
    expect(body).not.toHaveProperty("auto_chunking_use_llm")
    expect(body).not.toHaveProperty("chunk_size")
    expect(body).not.toHaveProperty("chunk_overlap")
  })

  it("passes manual chunking templates to process-web-scraping JSON requests", async () => {
    mocks.bgRequest.mockResolvedValue({ content: "processed" })

    await submitQuickIngestBatch({
      entries: [
        {
          id: "entry-html-manual",
          url: "https://example.com/manual-page",
          type: "html"
        }
      ],
      files: [],
      storeRemote: false,
      processOnly: true,
      common: {
        perform_analysis: true,
        perform_chunking: true,
        overwrite_existing: false,
        chunking_mode: "manual",
        auto_chunking_goal: "balanced",
        auto_chunking_use_llm: false
      },
      advancedValues: {
        chunk_method: "sentences",
        chunk_size: 900
      },
      chunkingTemplateName: "article-template",
      autoApplyTemplate: true
    } as any)

    const body = mocks.bgRequest.mock.calls[0][0].body
    expect(body).toMatchObject({
      url_input: "https://example.com/manual-page",
      chunking_mode: "manual",
      chunk_method: "sentences",
      chunk_size: 900,
      chunking_template_name: "article-template",
      auto_apply_template: true
    })
    expect(body).not.toHaveProperty("auto_chunking_goal")
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
          url: "https://example.com/article.md",
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
          urls: ["https://example.com/article.md"]
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

  it("creates planned conference collection items before direct job submission", async () => {
    const onTrackingMetadata = vi.fn()

    mocks.bgUpload
      .mockResolvedValueOnce({
        batch_id: "batch-talk-1",
        jobs: [{ id: 501 }]
      })
      .mockResolvedValueOnce({
        batch_id: "batch-talk-2",
        jobs: [{ id: 502 }]
      })
    mocks.bgRequest.mockImplementation(async (request: { path?: string; body?: any }) => {
      const path = String(request?.path || "")
      if (path === "/api/v1/media/collections") {
        return {
          id: 7,
          name: "Strange Loop 2012",
          kind: "conference",
          source_url: "https://youtube.com/playlist?list=PL-conf",
          metadata: request.body?.metadata || {},
          default_tags: request.body?.default_tags || [],
          created_at: "2026-05-01T00:00:00Z",
          updated_at: "2026-05-01T00:00:00Z",
          items: []
        }
      }
      if (path === "/api/v1/media/collections/7/items") {
        const ordinal = Number(request.body?.ordinal || 0)
        return {
          id: ordinal === 1 ? 11 : 12,
          collection_id: 7,
          ordinal,
          source_url: request.body?.source_url,
          normalized_source_id: request.body?.normalized_source_id,
          source_kind: request.body?.source_kind,
          title: request.body?.title,
          speaker: request.body?.speaker,
          duplicate_status: request.body?.duplicate_status || "new",
          status: "planned",
          retry_count: 0,
          idempotency_key: `conference-7-${ordinal}`,
          warnings: [],
          metadata: request.body?.metadata || {},
          tags: request.body?.tags || [],
          created_at: "2026-05-01T00:00:00Z",
          updated_at: "2026-05-01T00:00:00Z"
        }
      }
      if (path === "/api/v1/media/ingest/jobs/501" || path === "/api/v1/media/ingest/jobs/502") {
        return {
          ok: true,
          data: {
            status: "completed",
            result: { media_id: path.endsWith("501") ? 901 : 902 }
          }
        }
      }
      if (path === "/api/v1/media/collections/7/items/11" || path === "/api/v1/media/collections/7/items/12") {
        return {
          id: path.endsWith("/11") ? 11 : 12,
          collection_id: 7,
          ordinal: path.endsWith("/11") ? 1 : 2,
          source_url: request.body?.source_url || "https://youtube.com/watch?v=talk",
          duplicate_status: "new",
          status: request.body?.status || "processing",
          retry_count: 0,
          warnings: [],
          metadata: {},
          tags: [],
          created_at: "2026-05-01T00:00:00Z",
          updated_at: "2026-05-01T00:00:00Z"
        }
      }
      throw new Error(`Unexpected bgRequest path: ${path}`)
    })

    await submitQuickIngestBatch({
      entries: [
        {
          id: "talk-1",
          url: "https://youtube.com/watch?v=talk-1",
          type: "video",
          playlist: {
            playlistId: "PL-conf",
            playlistTitle: "Strange Loop 2012",
            ordinal: 1,
            normalizedSourceId: "youtube:video:talk-1",
            duplicateStatus: "new"
          },
          conferenceOverride: {
            selected: true,
            title: "Simplicity Matters",
            speaker: "Rich Hickey",
            tags: ["keynote"]
          }
        },
        {
          id: "talk-2",
          url: "https://youtube.com/watch?v=talk-2",
          type: "video",
          playlist: {
            playlistId: "PL-conf",
            playlistTitle: "Strange Loop 2012",
            ordinal: 2,
            normalizedSourceId: "youtube:video:talk-2",
            duplicateStatus: "new"
          },
          conferenceOverride: {
            selected: true,
            speaker: "Alex Miller"
          }
        }
      ],
      files: [],
      storeRemote: true,
      processOnly: false,
      conferenceBatchMetadata: {
        collectionName: "Strange Loop 2012",
        conferenceName: "Strange Loop",
        eventYear: "2012",
        sharedTags: ["conference", "clojure"],
        sourcePlaylistUrl: "https://youtube.com/playlist?list=PL-conf"
      },
      common: {
        perform_analysis: true,
        perform_chunking: false,
        overwrite_existing: false
      },
      advancedValues: {},
      __quickIngestSessionId: "qi-direct-conference-run",
      onTrackingMetadata
    } as any)

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/collections",
        method: "POST",
        body: expect.objectContaining({
          name: "Strange Loop 2012",
          kind: "conference",
          source_url: "https://youtube.com/playlist?list=PL-conf",
          metadata: expect.objectContaining({
            conference_name: "Strange Loop",
            event_year: "2012",
            source_playlist_url: "https://youtube.com/playlist?list=PL-conf"
          }),
          default_tags: ["conference", "clojure"]
        })
      })
    )
    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/collections/7/items",
        method: "POST",
        body: expect.objectContaining({
          ordinal: 1,
          source_url: "https://youtube.com/watch?v=talk-1",
          normalized_source_id: "youtube:video:talk-1",
          title: "Simplicity Matters",
          speaker: "Rich Hickey",
          tags: ["conference", "clojure", "keynote"]
        })
      })
    )
    expect(mocks.bgUpload).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({
        fields: expect.objectContaining({
          media_collection_id: 7,
          media_collection_item_id: 11,
          idempotency_key: "conference-7-1"
        })
      })
    )
    expect(onTrackingMetadata).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({
        mode: "webui-direct",
        sessionId: "qi-direct-conference-run",
        batchId: "batch-talk-1",
        collectionId: "7",
        plannedItemIds: ["11"],
        jobIdToCollectionItemId: {
          "501": "11"
        },
        durableMode: "durable_collection"
      })
    )
  })

  it("skips direct ingest submission for existing conference items when policy includes existing", async () => {
    mocks.bgRequest.mockImplementation(async (request: { path?: string; body?: any }) => {
      const path = String(request?.path || "")
      if (path === "/api/v1/media/collections") {
        return {
          id: 9,
          name: "Conference Batch",
          kind: "conference",
          metadata: {},
          default_tags: [],
          created_at: "2026-05-01T00:00:00Z",
          updated_at: "2026-05-01T00:00:00Z",
          items: []
        }
      }
      if (path === "/api/v1/media/collections/9/items") {
        return {
          id: 91,
          collection_id: 9,
          ordinal: 3,
          source_url: request.body?.source_url,
          duplicate_status: request.body?.duplicate_status,
          status: request.body?.status,
          retry_count: 0,
          idempotency_key: "conference-9-3",
          warnings: [],
          metadata: request.body?.metadata || {},
          tags: [],
          created_at: "2026-05-01T00:00:00Z",
          updated_at: "2026-05-01T00:00:00Z"
        }
      }
      if (path === "/api/v1/media/collections/9/items/91") {
        return {
          id: 91,
          collection_id: 9,
          ordinal: 3,
          source_url: "https://youtube.com/watch?v=existing",
          duplicate_status: "duplicate_existing",
          status: request.body?.status,
          retry_count: request.body?.retry_count || 0,
          warnings: [],
          metadata: {},
          tags: [],
          created_at: "2026-05-01T00:00:00Z",
          updated_at: "2026-05-01T00:00:00Z"
        }
      }
      throw new Error(`Unexpected bgRequest path: ${path}`)
    })

    const result = await submitQuickIngestBatch({
      entries: [
        {
          id: "existing-talk",
          url: "https://youtube.com/watch?v=existing",
          type: "video",
          playlist: {
            playlistId: "PL-conf",
            playlistTitle: "Conference Batch",
            ordinal: 3,
            normalizedSourceId: "youtube:video:existing",
            duplicateStatus: "duplicate_existing"
          },
          conferenceOverride: {
            selected: true,
            duplicatePolicy: "include_existing",
            title: "Existing Talk"
          }
        }
      ],
      files: [],
      storeRemote: true,
      processOnly: false,
      conferenceBatchMetadata: {
        collectionName: "Conference Batch",
        sharedTags: []
      },
      common: {
        perform_analysis: true,
        perform_chunking: false,
        overwrite_existing: false
      },
      advancedValues: {},
      __quickIngestSessionId: "qi-direct-duplicate-policy"
    } as any)

    expect(mocks.bgUpload).not.toHaveBeenCalled()
    expect(result.results?.[0]).toMatchObject({
      id: "existing-talk",
      status: "ok",
      outcome: "skipped",
      collectionItemId: 91,
      idempotencyKey: "conference-9-3"
    })
    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/collections/9/items",
        method: "POST",
        body: expect.objectContaining({
          status: "skipped_existing",
          metadata: expect.objectContaining({
            duplicate_policy: "include_existing"
          })
        })
      })
    )
  })

  it("marks planned conference items as submit_failed when direct job submission fails", async () => {
    mocks.bgUpload.mockRejectedValueOnce(new Error("job submit failed"))
    mocks.bgRequest.mockImplementation(async (request: { path?: string; body?: any }) => {
      const path = String(request?.path || "")
      if (path === "/api/v1/media/collections") {
        return {
          id: 8,
          name: "Conference Batch",
          kind: "conference",
          metadata: {},
          default_tags: [],
          created_at: "2026-05-01T00:00:00Z",
          updated_at: "2026-05-01T00:00:00Z",
          items: []
        }
      }
      if (path === "/api/v1/media/collections/8/items") {
        return {
          id: 81,
          collection_id: 8,
          ordinal: 1,
          source_url: request.body?.source_url,
          duplicate_status: "new",
          status: "planned",
          retry_count: 0,
          warnings: [],
          metadata: {},
          tags: [],
          created_at: "2026-05-01T00:00:00Z",
          updated_at: "2026-05-01T00:00:00Z"
        }
      }
      if (path === "/api/v1/media/collections/8/items/81") {
        return {
          id: 81,
          collection_id: 8,
          ordinal: 1,
          source_url: "https://youtube.com/watch?v=failed",
          duplicate_status: "new",
          status: request.body?.status,
          error_summary: request.body?.error_summary,
          retry_count: 0,
          warnings: [],
          metadata: {},
          tags: [],
          created_at: "2026-05-01T00:00:00Z",
          updated_at: "2026-05-01T00:00:00Z"
        }
      }
      throw new Error(`Unexpected bgRequest path: ${path}`)
    })

    const result = await submitQuickIngestBatch({
      entries: [
        {
          id: "failed-talk",
          url: "https://youtube.com/watch?v=failed",
          type: "video",
          conferenceOverride: {
            selected: true,
            title: "Failed Talk"
          }
        }
      ],
      files: [],
      storeRemote: true,
      processOnly: false,
      conferenceBatchMetadata: {
        collectionName: "Conference Batch",
        sharedTags: []
      },
      common: {
        perform_analysis: true,
        perform_chunking: false,
        overwrite_existing: false
      },
      advancedValues: {}
    } as any)

    expect(result.results?.[0]).toMatchObject({
      id: "failed-talk",
      status: "error",
      error: "job submit failed"
    })
    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/collections/8/items/81",
        method: "PATCH",
        body: expect.objectContaining({
          status: "submit_failed",
          error_summary: "job submit failed"
        })
      })
    )
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
