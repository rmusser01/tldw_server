import { beforeEach, describe, expect, it, vi } from "vitest"
import type { WatchlistOutputCreate } from "@/types/watchlists"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn(),
  bgUpload: vi.fn()
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.bgRequest(...args),
  bgUpload: (...args: unknown[]) => mocks.bgUpload(...args)
}))

import {
  getWatchlistRunAudio,
  getWatchlistRunDiagnostics,
  retryWatchlistRunAudio,
  retryWatchlistRunDelivery
} from "@/services/watchlists"

describe("watchlists audio services", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("fetches run audio briefing status from the watchlists run endpoint", async () => {
    mocks.bgRequest.mockResolvedValueOnce({
      run_id: 123,
      task_id: "task-abc",
      status: "completed",
      queue_name: "workflows",
      audio_uri: "file:///tmp/briefing.mp3",
      download_url: "/api/v1/workflows/artifacts/art-1/download",
      artifact_id: "art-1",
      size_bytes: 1024,
      mime_type: "audio/mpeg"
    })

    const result = await getWatchlistRunAudio(123)

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/watchlists/runs/123/audio",
        method: "GET"
      })
    )
    expect(result).toEqual({
      run_id: 123,
      task_id: "task-abc",
      status: "completed",
      queue_name: "workflows",
      audio_uri: "file:///tmp/briefing.mp3",
      download_url: "/api/v1/workflows/artifacts/art-1/download",
      artifact_id: "art-1",
      size_bytes: 1024,
      mime_type: "audio/mpeg"
    })
  })

  it("retries only the audio briefing stage for a run", async () => {
    mocks.bgRequest.mockResolvedValueOnce({
      run_id: 123,
      stage: "audio",
      retried: true,
      task_id: "task-retry"
    })

    const result = await retryWatchlistRunAudio(123)

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/watchlists/runs/123/retry-audio",
        method: "POST"
      })
    )
    expect(result.task_id).toBe("task-retry")
  })

  it("retries only the output delivery stage for a run", async () => {
    mocks.bgRequest.mockResolvedValueOnce({
      run_id: 123,
      stage: "delivery",
      retried: true,
      output_id: 55,
      delivery_results: [{ channel: "email", status: "sent" }]
    })

    const result = await retryWatchlistRunDelivery(123)

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/watchlists/runs/123/retry-delivery",
        method: "POST"
      })
    )
    expect(result.delivery_results).toEqual([{ channel: "email", status: "sent" }])
  })

  it("fetches a run diagnostic bundle without rerunning ingestion", async () => {
    mocks.bgRequest.mockResolvedValueOnce({
      run_id: 123,
      generated_at: "2026-05-19T04:00:00Z",
      run: { id: 123, status: "failed" },
      outputs: [{ id: 55, delivery_status: "failed" }]
    })

    const result = await getWatchlistRunDiagnostics(123)

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/watchlists/runs/123/diagnostics",
        method: "GET"
      })
    )
    expect(result.run_id).toBe(123)
  })

  it("accepts backend-supported audio fields on output creation payloads", () => {
    const output = {
      run_id: 10,
      generate_audio: true,
      target_audio_minutes: 12,
      audio_model: "kokoro",
      audio_voice: "af_heart",
      audio_speed: 1.05,
      background_audio_uri: "file:///tmp/bed.mp3",
      background_volume: 0.15,
      background_delay_ms: 250,
      background_fade_seconds: 2,
      audio_language: "en",
      llm_provider: "openai",
      llm_model: "gpt-4.1-mini",
      persona_summarize: true,
      persona_id: "daily-anchor",
      persona_provider: "openai",
      persona_model: "gpt-4.1-mini",
      voice_map: { HOST: "af_bella", ANALYST: "am_adam" },
      audio_cast: {
        speaker_count: 2,
        speakers: [
          { id: "host", label: "Host", role: "anchor", voice: "af_bella" },
          { id: "analyst", label: "Analyst", role: "analysis", voice: "am_adam" }
        ]
      }
    } satisfies WatchlistOutputCreate

    expect(output.generate_audio).toBe(true)
    expect(output.voice_map?.HOST).toBe("af_bella")
    expect(output.audio_cast?.speaker_count).toBe(2)
  })
})
