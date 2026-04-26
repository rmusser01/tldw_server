import { beforeEach, describe, expect, it, vi } from "vitest"

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
  })

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
