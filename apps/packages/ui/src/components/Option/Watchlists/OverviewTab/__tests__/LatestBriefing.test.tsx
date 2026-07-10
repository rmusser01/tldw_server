import React from "react"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"
import type {
  WatchlistBriefingProjection,
  WatchlistBriefingRetryStage
} from "@/types/watchlists"
import { formatWatchlistOccurrenceDate, LatestBriefing } from "../LatestBriefing"

const artifactMocks = vi.hoisted(() => ({
  fetchBlob: vi.fn(),
  fetchText: vi.fn(),
  createUrl: vi.fn(),
  revokeUrl: vi.fn()
}))

vi.mock("@/services/watchlists-artifacts", () => ({
  fetchWatchlistArtifactBlob: (...args: unknown[]) => artifactMocks.fetchBlob(...args),
  fetchWatchlistArtifactText: (...args: unknown[]) => artifactMocks.fetchText(...args),
  createWatchlistArtifactObjectUrl: (...args: unknown[]) => artifactMocks.createUrl(...args),
  revokeWatchlistArtifactObjectUrl: (...args: unknown[]) => artifactMocks.revokeUrl(...args)
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    i18n: { resolvedLanguage: "en-US", language: "en-US", dir: () => "ltr" },
    t: (
      _key: string,
      fallbackOrOptions?: string | { defaultValue?: string; count?: number },
      values?: Record<string, unknown>
    ) => {
      const fallback = typeof fallbackOrOptions === "string"
        ? fallbackOrOptions
        : fallbackOrOptions?.defaultValue || ""
      const allValues = typeof fallbackOrOptions === "object"
        ? { ...fallbackOrOptions, ...values }
        : values
      return fallback.replace(/{{(\w+)}}/g, (_match, token) => String(allValues?.[token] ?? ""))
    }
  })
}))

const readyEpisode = (
  overrides: Partial<WatchlistBriefingProjection> = {}
): WatchlistBriefingProjection => ({
  occurrence_id: 31,
  run_id: 123,
  job_id: 7,
  artifact_status: "ready",
  delivery_status: "delivered",
  stages: {
    collect: { status: "ready" },
    select: { status: "ready" },
    render_text: { status: "ready" },
    persist_text: { status: "ready" },
    compose_audio_script: { status: "ready" },
    persist_audio_script: { status: "ready" },
    generate_audio: { status: "ready" },
    persist_audio: { status: "ready" },
    "deliver:email": { status: "ready", outcome: "successful" },
    "deliver:chatbook": { status: "ready", outcome: "successful" }
  },
  output: {
    id: 71,
    title: "Week 28: Summer League",
    created_at: "2026-07-10T19:20:00Z",
    metadata: {
      no_material_updates: false,
      provenance: [1, 2, 3, 4, 5].map((source_id) => ({ source_id }))
    }
  },
  audio: {
    run_id: 123,
    task_id: "audio-task-123",
    status: "completed",
    download_url: "/api/v1/watchlists/runs/123/audio/download",
    script_artifact: {
      artifact_id: "script-123",
      title: "Week 28 script",
      download_url: "/api/v1/watchlists/runs/123/audio/script/download"
    }
  },
  editorial: {
    outcome_noun: "episode",
    program_format: "host_discussion",
    show_name: "Purple and Gold Weekly",
    show_notes: true,
    target_minutes: 20,
    cast: {
      speaker_count: 2,
      speakers: [
        { label: "Morgan", role: "host", voice: "alloy", synthetic: true },
        { label: "Avery", role: "analyst", voice: "nova", synthetic: true }
      ]
    }
  },
  delivery: {
    email: { adapter: "email", recipient_count: 2, masked_label: "2 recipients" },
    chatbook: { adapter: "chatbook", recipient_count: 1, masked_label: "Chatbook" }
  },
  selection: { candidate_count: 9, included_count: 7, omitted_count: 2 },
  next_run_at: "2026-07-12T18:00:00-07:00",
  timezone: "America/Los_Angeles",
  recovery: {
    can_open_report: true,
    can_regenerate_audio: true
  },
  ...overrides
})

const actions = () => ({
  onPlay: vi.fn(),
  onOpenReport: vi.fn(),
  onInspectRun: vi.fn(),
  onRetryStage: vi.fn(),
  onRegenerate: vi.fn(),
  onTestNow: vi.fn(),
  onViewReports: vi.fn(),
  onReviewDeliverySettings: vi.fn()
})

describe("LatestBriefing", () => {
  beforeEach(() => {
    vi.restoreAllMocks()
    vi.clearAllMocks()
    Object.defineProperty(HTMLMediaElement.prototype, "play", {
      configurable: true,
      value: vi.fn().mockResolvedValue(undefined)
    })
    Object.defineProperty(HTMLMediaElement.prototype, "pause", {
      configurable: true,
      value: vi.fn()
    })
    Object.defineProperty(HTMLMediaElement.prototype, "load", {
      configurable: true,
      value: vi.fn()
    })
    artifactMocks.fetchBlob.mockResolvedValue(new Blob(["audio"], { type: "audio/mpeg" }))
    artifactMocks.fetchText.mockResolvedValue("# Week 28 script")
    artifactMocks.createUrl.mockReturnValue("blob:latest-audio")
    artifactMocks.revokeUrl.mockImplementation(() => undefined)
  })

  it("keeps ready show notes usable when audio fails", async () => {
    const user = userEvent.setup()
    const callbacks = actions()
    const partialProjection = readyEpisode({
      artifact_status: "failed",
      audio: { run_id: 123, status: "failed", error: "provider unavailable" },
      stages: {
        ...readyEpisode().stages,
        generate_audio: { status: "failed", retryable: true, code: "provider_unavailable" },
        persist_audio: { status: "not_started" }
      },
      delivery_status: "waiting_for_artifacts",
      recovery: { can_open_report: true, can_retry_audio: true }
    })

    render(<LatestBriefing projection={partialProjection} {...callbacks} />)

    expect(screen.getByRole("button", { name: "Open show notes for Purple and Gold Weekly" })).toBeEnabled()
    expect(screen.getByText("Audio failed")).toBeVisible()
    await user.click(screen.getByRole("button", { name: "Retry generating audio for Purple and Gold Weekly" }))
    expect(callbacks.onRetryStage).toHaveBeenCalledWith(123, "generate_audio")
  })

  it("shows playback, deliveries, provenance, counts, and the exact next run", async () => {
    render(<LatestBriefing projection={readyEpisode()} unreadCount={3} newCount={4} {...actions()} />)

    expect(screen.getByRole("heading", { name: "Latest episode" })).toBeVisible()
    expect(await screen.findByRole("button", { name: "Play Purple and Gold Weekly" })).toBeEnabled()
    expect(screen.getByText("Email delivered")).toBeVisible()
    expect(screen.getByText("Chatbook delivered")).toBeVisible()
    expect(screen.getByText("Next run: Sunday, July 12 at 6:00 PM GMT-7")).toBeVisible()
    expect(screen.getByText("Included 7")).toBeVisible()
    expect(screen.getByText("Unread 3")).toBeVisible()
    expect(screen.getByText("New 4")).toBeVisible()
    expect(screen.getByText("5 tracked sources")).toBeVisible()
    expect(screen.getByText("Morgan, Avery")).toBeVisible()
    expect(screen.getByText("targeting 20 minutes")).toBeVisible()
  })

  it("exposes Play, Pause, Resume, seek, elapsed duration, loading, and error states", async () => {
    const user = userEvent.setup()
    const callbacks = actions()
    const { container } = render(<LatestBriefing projection={readyEpisode()} {...callbacks} />)
    await waitFor(() => expect(container.querySelector("audio")).not.toBeNull())
    const audio = container.querySelector("audio") as HTMLAudioElement
    Object.defineProperty(audio, "duration", { configurable: true, value: 600 })
    Object.defineProperty(audio, "currentTime", { configurable: true, writable: true, value: 90 })

    fireEvent.loadedMetadata(audio)
    expect(screen.getByText("1:30 / 10:00")).toBeVisible()
    await user.click(screen.getByRole("button", { name: "Play Purple and Gold Weekly" }))
    expect(callbacks.onPlay).toHaveBeenCalledWith(readyEpisode().audio, readyEpisode())
    fireEvent.play(audio)
    expect(screen.getByRole("button", { name: "Pause Purple and Gold Weekly" })).toBeVisible()
    fireEvent.waiting(audio)
    expect(screen.getByText("Loading audio")).toBeVisible()
    fireEvent.canPlay(audio)
    fireEvent.pause(audio)
    expect(screen.getByRole("button", { name: "Resume Purple and Gold Weekly" })).toBeVisible()

    fireEvent.change(screen.getByRole("slider", { name: "Seek Purple and Gold Weekly" }), {
      target: { value: "300" }
    })
    expect(audio.currentTime).toBe(300)
    fireEvent.error(audio)
    expect(screen.getByText("Audio could not be played. Open the report or regenerate audio.")).toBeVisible()
  })

  it("shows pending progress and preserves zero-update output", () => {
    const pending = readyEpisode({
      artifact_status: "running",
      delivery_status: "waiting_for_artifacts",
      output: {
        ...readyEpisode().output,
        metadata: { no_material_updates: true, provenance: { source_count: 0 } }
      },
      audio: { run_id: 123, status: "running" },
      stages: {
        persist_text: { status: "ready" },
        compose_audio_script: { status: "running" },
        persist_audio_script: { status: "queued" },
        generate_audio: { status: "not_started" },
        persist_audio: { status: "not_started" }
      },
      selection: { candidate_count: 0, included_count: 0, omitted_count: 0 }
    })

    render(<LatestBriefing projection={pending} {...actions()} />)
    expect(screen.getByText("No qualifying updates were found. A status episode was saved.")).toBeVisible()
    expect(screen.getByText("Audio script running")).toBeVisible()
    expect(screen.getByRole("button", { name: "Open show notes for Purple and Gold Weekly" })).toBeEnabled()
  })

  it("requires reviewed acknowledgement in an accessible dialog before retrying unknown delivery", async () => {
    const user = userEvent.setup()
    const callbacks = actions()
    const unknown = readyEpisode({
      delivery_status: "unknown",
      stages: {
        ...readyEpisode().stages,
        "deliver:email": { status: "failed", outcome: "unknown", code: "delivery_outcome_unknown" }
      },
      recovery: {
        ...readyEpisode().recovery,
        can_retry_delivery: true,
        requires_unknown_delivery_confirmation: true
      }
    })

    render(<LatestBriefing projection={unknown} {...callbacks} />)
    const trigger = screen.getByRole("button", { name: "Review and retry email delivery for Purple and Gold Weekly" })
    await user.click(trigger)

    const dialog = screen.getByRole("dialog", { name: "Review email delivery retry" })
    expect(dialog).toHaveTextContent("2 recipients")
    expect(dialog).toHaveTextContent(/duplicate/i)
    const confirmRetry = screen.getByRole("button", { name: "Retry email delivery" })
    expect(confirmRetry).toBeDisabled()
    await user.keyboard("{Escape}")
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument())
    expect(trigger).toHaveFocus()

    await user.click(trigger)
    await user.click(screen.getByRole("button", { name: "Review delivery settings" }))
    expect(callbacks.onReviewDeliverySettings).toHaveBeenCalledWith(7)

    await user.click(trigger)
    await user.click(screen.getByRole("checkbox", { name: /reviewed the destination/i }))
    await user.click(screen.getByRole("button", { name: "Retry email delivery" }))
    expect(callbacks.onRetryStage).toHaveBeenCalledWith(123, "deliver:email", {
      confirm_unknown_delivery_retry: true
    })
  })

  it("uses record-specific names for multiple briefing records", () => {
    const callbacks = actions()
    render(
      <>
        <LatestBriefing projection={readyEpisode()} {...callbacks} />
        <LatestBriefing
          projection={readyEpisode({
            occurrence_id: 32,
            run_id: 124,
            output: { ...readyEpisode().output, id: 72, title: "City Hall AM" },
            editorial: { outcome_noun: "briefing", show_name: "City Hall AM" }
          })}
          {...callbacks}
        />
      </>
    )

    expect(screen.getByRole("button", { name: "Open show notes for Purple and Gold Weekly" })).toBeVisible()
    expect(screen.getByRole("button", { name: "Open report for City Hall AM" })).toBeVisible()
    expect(screen.getByRole("button", { name: "Inspect run 123 for Purple and Gold Weekly" })).toBeVisible()
    expect(screen.getByRole("button", { name: "Inspect run 124 for City Hall AM" })).toBeVisible()
  })

  it("uses a container-driven one-to-two-column layout with RTL, long copy, and coarse targets", () => {
    const { container } = render(
      <div dir="rtl" style={{ width: 320 }}>
        <LatestBriefing
          projection={readyEpisode({
            editorial: {
              ...readyEpisode().editorial,
              show_name: "برنامج طويل للغاية لمراجعة المصادر الموثوقة والتحديثات اليومية المهمة"
            }
          })}
          {...actions()}
        />
      </div>
    )

    const section = container.querySelector("section")
    expect(section).toHaveClass("@container")
    expect(section?.querySelector("[data-testid='latest-briefing-layout']")).toHaveClass("@3xl:grid-cols-[minmax(0,1fr)_minmax(13rem,0.35fr)]")
    expect(screen.getAllByRole("button")[0]).toHaveClass("min-h-11")
    expect(section).not.toHaveAttribute("style")
  })

  it("offers Test now for the earliest active monitor and its exact schedule when no occurrence exists", async () => {
    const user = userEvent.setup()
    const callbacks = actions()
    render(<LatestBriefing projection={null} emptyJobId={7} nextRunAt="2026-07-12T18:00:00-07:00" timezone="America/Los_Angeles" {...callbacks} />)

    expect(screen.getByRole("heading", { name: "Latest briefing" })).toBeVisible()
    await user.click(screen.getByRole("button", { name: "Test now" }))
    await user.click(screen.getByRole("button", { name: "View all reports" }))
    expect(screen.getByText(/Sunday, July 12 at 6:00 PM GMT-7/)).toBeVisible()
    expect(callbacks.onTestNow).toHaveBeenCalledWith(7)
    expect(callbacks.onViewReports).toHaveBeenCalled()
  })

  it("routes a failed stage to its exact recovery callback", async () => {
    const user = userEvent.setup()
    const callbacks = actions()
    const failed = readyEpisode({
      artifact_status: "failed",
      output: null,
      stages: { persist_text: { status: "failed", retryable: true } },
      recovery: { can_retry_text: true }
    })
    render(<LatestBriefing projection={failed} {...callbacks} />)

    await user.click(screen.getByRole("button", { name: "Retry saving report for Purple and Gold Weekly" }))
    expect(callbacks.onRetryStage).toHaveBeenCalledWith(123, "persist_text" satisfies WatchlistBriefingRetryStage)
  })

  it("starts a new test occurrence for the same monitor", async () => {
    const user = userEvent.setup()
    const callbacks = actions()
    render(<LatestBriefing projection={readyEpisode()} {...callbacks} />)

    await user.click(screen.getByRole("button", { name: "Test now: Purple and Gold Weekly" }))
    expect(callbacks.onTestNow).toHaveBeenCalledWith(7)
  })

  it("uses authenticated Blob URLs and resets the player when artifact identity changes", async () => {
    artifactMocks.createUrl
      .mockReturnValueOnce("blob:first-audio")
      .mockReturnValueOnce("blob:second-audio")
    const callbacks = actions()
    const { container, rerender } = render(<LatestBriefing projection={readyEpisode()} {...callbacks} />)
    await waitFor(() => expect(container.querySelector("audio")).toHaveAttribute("src", "blob:first-audio"))
    const firstAudio = container.querySelector("audio") as HTMLAudioElement
    fireEvent.play(firstAudio)
    fireEvent.timeUpdate(firstAudio)

    rerender(
      <LatestBriefing
        projection={readyEpisode({
          occurrence_id: 32,
          run_id: 124,
          audio: {
            ...readyEpisode().audio!,
            run_id: 124,
            task_id: "audio-task-124",
            download_url: "/api/v1/watchlists/runs/124/audio/download"
          }
        })}
        {...callbacks}
      />
    )

    await waitFor(() => expect(container.querySelector("audio")).toHaveAttribute("src", "blob:second-audio"))
    expect(HTMLMediaElement.prototype.pause).toHaveBeenCalled()
    expect(HTMLMediaElement.prototype.load).toHaveBeenCalled()
    expect(artifactMocks.revokeUrl).toHaveBeenCalledWith("blob:first-audio")
    expect(screen.getByRole("button", { name: "Play Purple and Gold Weekly" })).toBeVisible()
    expect(screen.getByText("0:00 / 0:00")).toBeVisible()
  })

  it.each([
    { audioState: { stale: true, superseded_by: null }, expectedStatus: "Stale" },
    { audioState: { stale: false, superseded_by: "audio-task-new" }, expectedStatus: "Superseded" },
    { audioState: { stale: false, superseded_by: null, download_url: null }, expectedStatus: "Unavailable" }
  ])("reports completed noncurrent audio as $expectedStatus", ({ audioState, expectedStatus }) => {
    render(<LatestBriefing projection={readyEpisode({
      audio: { ...readyEpisode().audio!, ...audioState }
    })} {...actions()} />)

    const audioRow = screen.getByText("Audio", { selector: "span" }).closest("div")
    expect(audioRow).not.toBeNull()
    expect(within(audioRow!).getByText(expectedStatus)).toBeVisible()
    expect(within(audioRow!).queryByText("Ready")).not.toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Play Purple and Gold Weekly" })).not.toBeInTheDocument()
    expect(artifactMocks.fetchBlob).not.toHaveBeenCalled()
  })

  it("reports completed current audio as ready", async () => {
    render(<LatestBriefing projection={readyEpisode()} {...actions()} />)

    const audioRow = screen.getByText("Audio", { selector: "span" }).closest("div")
    expect(audioRow).not.toBeNull()
    expect(within(audioRow!).getByText("Ready")).toBeVisible()
    expect(await screen.findByRole("button", { name: "Play Purple and Gold Weekly" })).toBeEnabled()
  })

  it("downloads audio and reviews a script through authenticated artifact requests", async () => {
    const user = userEvent.setup()
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => undefined)
    artifactMocks.createUrl.mockReturnValueOnce("blob:player").mockReturnValueOnce("blob:download")
    render(<LatestBriefing projection={readyEpisode()} {...actions()} />)

    await user.click(await screen.findByRole("button", { name: "Download audio for Purple and Gold Weekly" }))
    expect(artifactMocks.fetchBlob).toHaveBeenCalledWith(
      "/api/v1/watchlists/runs/123/audio/download",
      expect.objectContaining({ mimeType: expect.any(String) })
    )
    expect(click).toHaveBeenCalled()
    expect(artifactMocks.revokeUrl).toHaveBeenCalledWith("blob:download")

    await user.click(screen.getByRole("button", { name: "Review script for Purple and Gold Weekly" }))
    expect(artifactMocks.fetchText).toHaveBeenCalledWith(
      "/api/v1/watchlists/runs/123/audio/script/download",
      expect.any(Object)
    )
    expect(await screen.findByRole("dialog", { name: "Week 28 script" })).toHaveTextContent("# Week 28 script")
  })

  it("distinguishes missing artifacts from authorization or network errors", async () => {
    const user = userEvent.setup()
    artifactMocks.fetchText
      .mockRejectedValueOnce({ kind: "missing", status: 404 })
      .mockRejectedValueOnce({ kind: "auth", status: 403 })
    render(<LatestBriefing projection={readyEpisode()} {...actions()} />)

    const review = screen.getByRole("button", { name: "Review script for Purple and Gold Weekly" })
    await user.click(review)
    expect(await screen.findByText("This script artifact is no longer available.")).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Close" }))
    await user.click(review)
    expect(await screen.findByText("Script access failed. Check your sign-in and server connection.")).toBeInTheDocument()
  })

  it("uses authoritative show_notes instead of the outcome noun", () => {
    const { rerender } = render(<LatestBriefing projection={readyEpisode({
      editorial: { ...readyEpisode().editorial, outcome_noun: "episode", show_notes: false }
    })} {...actions()} />)
    expect(screen.getByRole("button", { name: "Open report for Purple and Gold Weekly" })).toBeVisible()

    rerender(<LatestBriefing projection={readyEpisode({
      editorial: { ...readyEpisode().editorial, outcome_noun: "briefing", show_notes: true }
    })} {...actions()} />)
    expect(screen.getByRole("button", { name: "Open show notes for Purple and Gold Weekly" })).toBeVisible()
  })

  it("renders every retryable failed or cancelled stage with its exact recovery label", () => {
    render(<LatestBriefing projection={readyEpisode({
      artifact_status: "failed",
      stages: {
        compose_audio_script: { status: "cancelled", retryable: true },
        persist_audio_script: { status: "failed", retryable: true },
        generate_audio: { status: "failed", retryable: true },
        persist_audio: { status: "cancelled", retryable: true },
        "deliver:email": { status: "cancelled", retryable: true, outcome: "failed" }
      }
    })} {...actions()} />)

    expect(screen.getByRole("button", { name: "Retry composing audio script for Purple and Gold Weekly" })).toBeVisible()
    expect(screen.getByRole("button", { name: "Retry saving audio script for Purple and Gold Weekly" })).toBeVisible()
    expect(screen.getByRole("button", { name: "Retry generating audio for Purple and Gold Weekly" })).toBeVisible()
    expect(screen.getByRole("button", { name: "Retry saving audio for Purple and Gold Weekly" })).toBeVisible()
    expect(screen.getByRole("button", { name: "Retry email delivery for Purple and Gold Weekly" })).toBeVisible()
  })
})

describe("formatWatchlistOccurrenceDate", () => {
  it("distinguishes both sides of a repeated DST fold with numeric offsets", () => {
    const first = formatWatchlistOccurrenceDate("2026-11-01T08:30:00Z", "America/Los_Angeles", "en-US")
    const second = formatWatchlistOccurrenceDate("2026-11-01T09:30:00Z", "America/Los_Angeles", "en-US")
    expect(first).toContain("1:30 AM GMT-7")
    expect(second).toContain("1:30 AM GMT-8")
    expect(first).not.toBe(second)
  })

  it("renders the post-gap time and cross-year dates in the active locale", () => {
    expect(formatWatchlistOccurrenceDate(
      "2026-03-08T10:30:00Z",
      "America/Los_Angeles",
      "en-US"
    )).toContain("3:30 AM GMT-7")
    expect(formatWatchlistOccurrenceDate(
      "2027-01-01T00:30:00Z",
      "Europe/Paris",
      "fr-FR",
      true
    )).toMatch(/2027.*UTC\+1|UTC\+1.*2027/)
  })
})
