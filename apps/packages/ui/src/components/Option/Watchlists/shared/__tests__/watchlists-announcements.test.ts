import { describe, expect, it, vi } from "vitest"
import type { WatchlistBriefingProjection } from "@/types/watchlists"
import {
  blockingFailureAnnouncement,
  transitionAnnouncement
} from "../watchlists-announcements"

const t = vi.fn((_key: string, fallback: string, values?: Record<string, unknown>) =>
  fallback.replace(/{{(\w+)}}/g, (_match, token) => String(values?.[token] ?? ""))
)

const projection = (
  overrides: Partial<WatchlistBriefingProjection> = {}
): WatchlistBriefingProjection => ({
  occurrence_id: 4,
  run_id: 8,
  job_id: 2,
  artifact_status: "running",
  delivery_status: "waiting_for_artifacts",
  stages: {
    persist_text: { status: "ready" },
    generate_audio: { status: "running" }
  },
  output: { id: 12, title: "Purple and Gold Weekly" },
  audio: null,
  editorial: {
    outcome_noun: "episode",
    show_name: "Purple and Gold Weekly"
  },
  selection: { candidate_count: 9, included_count: 7, omitted_count: 2 },
  next_run_at: "2026-07-12T18:00:00-07:00",
  timezone: "America/Los_Angeles",
  recovery: { can_open_report: true },
  ...overrides
})

describe("watchlists briefing announcements", () => {
  it("announces a semantic ready transition once", () => {
    const running = projection()
    const ready = projection({
      artifact_status: "ready",
      delivery_status: "delivered",
      stages: {
        persist_text: { status: "ready" },
        generate_audio: { status: "ready" }
      },
      audio: { run_id: 8, status: "completed", download_url: "/audio/8" }
    })

    expect(transitionAnnouncement(running, ready, t)).toBe(
      "Purple and Gold Weekly is ready. Audio and show notes are available."
    )
    expect(transitionAnnouncement(ready, ready, t)).toBeNull()
  })

  it("suppresses unchanged polling refreshes and initial hydration", () => {
    const running = projection()
    expect(transitionAnnouncement(null, running, t)).toBeNull()
    expect(transitionAnnouncement(running, { ...running, occurrence_id: 4 }, t)).toBeNull()
  })

  it("treats a new occurrence as a new identity and localizes stage progress", () => {
    const previous = projection()
    const next = projection({ occurrence_id: 5, run_id: 9 })
    const localized = vi.fn((key: string, fallback: string, values?: Record<string, unknown>) => {
      const translations: Record<string, string> = {
        "watchlists:overview.latest.stages.generateAudio": "Localized audio",
        "watchlists:overview.latest.status.running": "Localized running"
      }
      return (translations[key] || fallback).replace(
        /{{(\w+)}}/g,
        (_match, token) => String(values?.[token] ?? "")
      )
    })

    expect(transitionAnnouncement(previous, next, localized)).toBe(
      "Purple and Gold Weekly: Localized audio is Localized running."
    )
  })

  it("uses assertive copy only for a newly blocking artifact failure", () => {
    const running = projection({ output: null, recovery: {} })
    const blocked = projection({
      output: null,
      artifact_status: "failed",
      stages: { persist_text: { status: "failed", code: "report_persist_failed" } },
      recovery: { can_retry_text: true }
    })

    expect(blockingFailureAnnouncement(running, blocked, t)).toBe(
      "Purple and Gold Weekly failed before an artifact was ready. Inspect run 8 to recover it."
    )
    expect(blockingFailureAnnouncement(blocked, blocked, t)).toBeNull()
  })

  it("keeps a partial audio failure polite when the report remains usable", () => {
    const partial = projection({
      artifact_status: "failed",
      stages: {
        persist_text: { status: "ready" },
        generate_audio: { status: "failed", retryable: true }
      }
    })

    expect(blockingFailureAnnouncement(projection(), partial, t)).toBeNull()
    expect(transitionAnnouncement(projection(), partial, t)).toBe(
      "Purple and Gold Weekly show notes are ready, but audio failed."
    )
  })
})
