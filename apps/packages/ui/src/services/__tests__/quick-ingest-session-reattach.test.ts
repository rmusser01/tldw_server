import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn(),
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.bgRequest(...args),
}))

import { reattachQuickIngestSession } from "@/services/tldw/quick-ingest-session-reattach"

describe("reattachQuickIngestSession", () => {
  beforeEach(() => {
    mocks.bgRequest.mockReset()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it("reattaches active direct jobs into a processing snapshot", async () => {
    mocks.bgRequest.mockResolvedValue({
      ok: true,
      data: {
        status: "processing",
      },
    })

    const snapshot = await reattachQuickIngestSession({
      mode: "webui-direct",
      batchId: "batch-1",
      jobIds: [77],
      startedAt: Date.now(),
    })

    expect(snapshot.lifecycle).toBe("processing")
    expect(snapshot.jobs).toEqual([
      expect.objectContaining({
        jobId: 77,
        status: "processing",
      }),
    ])
    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/ingest/jobs/77",
        method: "GET",
        returnResponse: true,
        preferDirect: true,
      })
    )
  })

  it("retries a thrown status read and preserves direct transport", async () => {
    vi.useFakeTimers()
    mocks.bgRequest
      .mockRejectedValueOnce(new Error("network timeout"))
      .mockResolvedValueOnce({
        ok: true,
        data: {
          status: "processing",
        },
      })

    const pendingSnapshot = reattachQuickIngestSession({
      mode: "webui-direct",
      batchId: "batch-1",
      jobIds: [77],
      startedAt: Date.now(),
    })

    await vi.runAllTimersAsync()
    const snapshot = await pendingSnapshot

    expect(snapshot.lifecycle).toBe("processing")
    expect(mocks.bgRequest).toHaveBeenCalledTimes(2)
    for (const [request] of mocks.bgRequest.mock.calls) {
      expect(request).toEqual(expect.objectContaining({ preferDirect: true }))
    }
  })

  it("retries a thrown transient numeric status", async () => {
    vi.useFakeTimers()
    mocks.bgRequest
      .mockRejectedValueOnce(
        Object.assign(new Error("service unavailable"), { status: 503 })
      )
      .mockResolvedValueOnce({
        ok: true,
        data: {
          status: "processing",
        },
      })

    const pendingSnapshot = reattachQuickIngestSession({
      mode: "webui-direct",
      jobIds: [77],
      startedAt: Date.now(),
    })

    await vi.runAllTimersAsync()
    const snapshot = await pendingSnapshot

    expect(snapshot.lifecycle).toBe("processing")
    expect(mocks.bgRequest).toHaveBeenCalledTimes(2)
  })

  it("retries a thrown status-zero transport failure", async () => {
    vi.useFakeTimers()
    mocks.bgRequest
      .mockRejectedValueOnce(
        Object.assign(new Error("network unavailable"), { status: 0 })
      )
      .mockResolvedValueOnce({
        ok: true,
        data: {
          status: "processing",
        },
      })

    const pendingSnapshot = reattachQuickIngestSession({
      mode: "webui-direct",
      jobIds: [77],
      startedAt: Date.now(),
    })

    await vi.runAllTimersAsync()
    const snapshot = await pendingSnapshot

    expect(snapshot.lifecycle).toBe("processing")
    expect(mocks.bgRequest).toHaveBeenCalledTimes(2)
  })

  it("does not retry a thrown permanent numeric status", async () => {
    vi.useFakeTimers()
    mocks.bgRequest
      .mockRejectedValueOnce(
        Object.assign(new Error("unauthorized"), { status: 401 })
      )
      .mockResolvedValueOnce({
        ok: true,
        data: {
          status: "processing",
        },
      })

    const pendingSnapshot = reattachQuickIngestSession({
      mode: "webui-direct",
      jobIds: [77],
      startedAt: Date.now(),
    })

    await vi.runAllTimersAsync()
    const snapshot = await pendingSnapshot

    expect(snapshot.lifecycle).toBe("interrupted")
    expect(mocks.bgRequest).toHaveBeenCalledTimes(1)
  })

  it("retries an HTTP 503 status read and returns the completed job", async () => {
    vi.useFakeTimers()
    mocks.bgRequest
      .mockResolvedValueOnce({
        ok: false,
        status: 503,
        error: "service unavailable",
      })
      .mockResolvedValueOnce({
        ok: true,
        data: {
          status: "completed",
          result: { media_id: "media-77" },
        },
      })

    const pendingSnapshot = reattachQuickIngestSession({
      mode: "webui-direct",
      batchId: "batch-1",
      jobIds: [77],
      startedAt: Date.now(),
    })

    await vi.runAllTimersAsync()
    const snapshot = await pendingSnapshot

    expect(snapshot.lifecycle).toBe("completed")
    expect(mocks.bgRequest).toHaveBeenCalledTimes(2)
    for (const [request] of mocks.bgRequest.mock.calls) {
      expect(request).toEqual(expect.objectContaining({ preferDirect: true }))
    }
  })

  it("retries a resolved status-zero transport failure", async () => {
    vi.useFakeTimers()
    mocks.bgRequest
      .mockResolvedValueOnce({
        ok: false,
        status: 0,
        error: "network unavailable",
      })
      .mockResolvedValueOnce({
        ok: true,
        data: {
          status: "processing",
        },
      })

    const pendingSnapshot = reattachQuickIngestSession({
      mode: "webui-direct",
      jobIds: [77],
      startedAt: Date.now(),
    })

    await vi.runAllTimersAsync()
    const snapshot = await pendingSnapshot

    expect(snapshot.lifecycle).toBe("processing")
    expect(mocks.bgRequest).toHaveBeenCalledTimes(2)
  })

  it("does not retry a string HTTP status", async () => {
    vi.useFakeTimers()
    mocks.bgRequest
      .mockResolvedValueOnce({
        ok: false,
        status: "503",
        error: "service unavailable",
      })
      .mockResolvedValueOnce({
        ok: true,
        data: {
          status: "completed",
        },
      })

    const pendingSnapshot = reattachQuickIngestSession({
      mode: "webui-direct",
      jobIds: [77],
      startedAt: Date.now(),
    })

    await vi.runAllTimersAsync()
    const snapshot = await pendingSnapshot

    expect(snapshot.lifecycle).toBe("interrupted")
    expect(mocks.bgRequest).toHaveBeenCalledTimes(1)
  })

  it("does not retry an ok response carrying a transient status code", async () => {
    vi.useFakeTimers()
    mocks.bgRequest
      .mockResolvedValueOnce({
        ok: true,
        status: 503,
        data: {
          status: "processing",
        },
      })
      .mockResolvedValueOnce({
        ok: true,
        data: {
          status: "completed",
        },
      })

    const pendingSnapshot = reattachQuickIngestSession({
      mode: "webui-direct",
      jobIds: [77],
      startedAt: Date.now(),
    })

    await vi.runAllTimersAsync()
    const snapshot = await pendingSnapshot

    expect(snapshot.lifecycle).toBe("processing")
    expect(mocks.bgRequest).toHaveBeenCalledTimes(1)
  })

  it("marks a persisted processing session as interrupted when reattachment cannot prove live progress", async () => {
    mocks.bgRequest.mockResolvedValue({
      ok: false,
      status: 404,
      error: "not found",
    })

    const result = await reattachQuickIngestSession({
      mode: "webui-direct",
      batchId: "missing",
      jobIds: [77],
      startedAt: Date.now()
    })

    expect(result.lifecycle).toBe("interrupted")
    expect(result.errorMessage).toMatch(/could not reconnect/i)
    expect(mocks.bgRequest).toHaveBeenCalledTimes(1)
  })

  it.each([401, 403])(
    "does not retry permanent HTTP %i status reads",
    async (status) => {
      mocks.bgRequest.mockResolvedValue({
        ok: false,
        status,
        error: "permanent failure",
      })

      const result = await reattachQuickIngestSession({
        mode: "webui-direct",
        jobIds: [77],
        startedAt: Date.now(),
      })

      expect(result.lifecycle).toBe("interrupted")
      expect(mocks.bgRequest).toHaveBeenCalledTimes(1)
    }
  )

  it("does not retry a successful response with no job status", async () => {
    mocks.bgRequest.mockResolvedValue({
      ok: true,
      data: {},
    })

    const result = await reattachQuickIngestSession({
      mode: "webui-direct",
      jobIds: [77],
      startedAt: Date.now(),
    })

    expect(result.lifecycle).toBe("interrupted")
    expect(mocks.bgRequest).toHaveBeenCalledTimes(1)
  })

  it.each([408, 429])(
    "stops after three transient HTTP %i status-read attempts",
    async (status) => {
      vi.useFakeTimers()
      mocks.bgRequest.mockResolvedValue({
        ok: false,
        status,
        error: "transient failure",
      })

      const pendingSnapshot = reattachQuickIngestSession({
        mode: "webui-direct",
        jobIds: [77],
        startedAt: Date.now(),
      })

      await vi.runAllTimersAsync()
      const result = await pendingSnapshot

      expect(result.lifecycle).toBe("interrupted")
      expect(mocks.bgRequest).toHaveBeenCalledTimes(3)
    }
  )

  it("maps reattached jobs back to submitted queue item identities", async () => {
    mocks.bgRequest
      .mockResolvedValueOnce({
        ok: true,
        data: {
          status: "completed",
          result: { media_id: "media-88" },
        },
      })
      .mockResolvedValueOnce({
        ok: true,
        data: {
          status: "completed",
          result: { media_id: "media-99" },
        },
      })

    const snapshot = await reattachQuickIngestSession({
      mode: "webui-direct",
      jobIds: [88, 99],
      submittedItemIds: ["queued-url-1", "queued-file-1"],
      jobIdToItemId: {
        "99": "queued-file-1",
      },
      startedAt: Date.now(),
    })

    expect(snapshot.lifecycle).toBe("completed")
    expect(snapshot.jobs).toEqual([
      expect.objectContaining({
        jobId: 88,
        sourceItemId: "queued-url-1",
      }),
      expect.objectContaining({
        jobId: 99,
        sourceItemId: "queued-file-1",
      }),
    ])
  })

  it("treats completed jobs with error payloads as partial failures during reattach", async () => {
    mocks.bgRequest.mockResolvedValue({
      ok: true,
      data: {
        status: "completed",
        result: {
          status: "Error",
          error: "File preparation/download failed: Port not allowed: 3000"
        }
      },
    })

    const snapshot = await reattachQuickIngestSession({
      mode: "webui-direct",
      jobIds: [77],
      submittedItemIds: ["queued-url-1"],
      startedAt: Date.now(),
    })

    expect(snapshot.lifecycle).toBe("partial_failure")
    expect(snapshot.jobs).toEqual([
      expect.objectContaining({
        jobId: 77,
        status: "completed",
        error: "File preparation/download failed: Port not allowed: 3000",
        sourceItemId: "queued-url-1",
      }),
    ])
  })
})
