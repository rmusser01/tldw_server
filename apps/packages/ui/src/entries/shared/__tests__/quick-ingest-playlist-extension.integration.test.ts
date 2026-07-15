import {
  buildQuickIngestOpenDetailFromUrl,
  createQuickIngestSessionSeedFromOpenDetail
} from "@/utils/quick-ingest-open"
import { describe, expect, it, vi } from "vitest"

import { createQuickIngestSessionRuntime } from "../quick-ingest-session-runtime"

describe("extension playlist Quick Ingest integration", () => {
  it("keeps the typed active-tab seed and reattaches all occurrences after worker recreation", async () => {
    const playlistUrl =
      "https://www.youtube.com/watch?v=PrNmmN6qBiw&list=PL0065D9B288E6804B"
    const detail = buildQuickIngestOpenDetailFromUrl(playlistUrl)
    const occurrenceIds = Array.from(
      { length: 34 },
      (_, index) => `conference-occurrence-${index + 1}`
    )
    const marker = {
      version: 1 as const,
      kind: "run" as const,
      sessionId: "extension-playlist-session",
      runId: "extension-playlist-run",
      generation: "extension-playlist-generation",
      attemptToken: "extension-playlist-attempt",
      submissionState: "acknowledged" as const,
      occurrenceIds,
      jobIdToItemId: { "901": occurrenceIds[0], "934": occurrenceIds[33] },
      startedAt: Date.now()
    }
    const run = vi.fn()
    const emit = vi.fn()
    const reattachRun = vi.fn().mockResolvedValue({
      lifecycle: "processing",
      jobs: [
        {
          jobId: 901,
          attempt: 1,
          status: "running",
          sourceItemId: occurrenceIds[0]
        },
        {
          jobId: 934,
          attempt: 1,
          status: "queued",
          sourceItemId: occurrenceIds[33]
        }
      ],
      errorMessage: null
    })

    expect(detail).toEqual({
      source: "extension_active_tab",
      url: playlistUrl,
      sourceKind: "youtube_watch_playlist",
      action: "playlist_preflight"
    })
    expect(createQuickIngestSessionSeedFromOpenDetail(detail)).toEqual({
      openDetail: detail,
      firstSourceAddMode: null
    })

    const runtime = createQuickIngestSessionRuntime({
      run,
      emit,
      loadRunSessions: vi.fn().mockResolvedValue([marker]),
      saveRunSession: vi.fn(),
      reattachRun
    } satisfies Parameters<typeof createQuickIngestSessionRuntime>[0])

    await runtime.restore()

    expect(run).not.toHaveBeenCalled()
    expect(reattachRun).toHaveBeenCalledWith(
      expect.objectContaining({
        runId: marker.runId,
        submissionOccurrenceIds: occurrenceIds,
        jobIdToItemId: marker.jobIdToItemId
      }),
      { transportPreference: "poll" }
    )
    expect(emit).toHaveBeenCalledWith(
      "tldw:quick-ingest/progress",
      expect.objectContaining({
        sessionId: marker.sessionId,
        occurrenceId: occurrenceIds[0],
        jobId: 901,
        status: "running"
      })
    )
    expect(emit).toHaveBeenCalledWith(
      "tldw:quick-ingest/progress",
      expect.objectContaining({
        sessionId: marker.sessionId,
        occurrenceId: occurrenceIds[33],
        jobId: 934,
        status: "queued"
      })
    )
  })
})
