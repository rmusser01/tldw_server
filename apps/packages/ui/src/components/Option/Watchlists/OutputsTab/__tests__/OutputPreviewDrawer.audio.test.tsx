// @vitest-environment jsdom

import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { render, screen, waitFor } from "@testing-library/react"
import { OutputPreviewDrawer } from "../OutputPreviewDrawer"
import type { WatchlistOutput } from "@/types/watchlists"

const serviceMocks = vi.hoisted(() => ({
  downloadWatchlistOutput: vi.fn(),
  downloadWatchlistOutputBinary: vi.fn()
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?: string | { defaultValue?: string },
      values?: Record<string, unknown>
    ) => {
      if (typeof fallbackOrOptions === "string") {
        if (!values) return fallbackOrOptions
        return fallbackOrOptions.replace(/\{\{(\w+)\}\}/g, (_match, token) => {
          const value = values[token]
          return value == null ? "" : String(value)
        })
      }
      if (
        fallbackOrOptions &&
        typeof fallbackOrOptions === "object" &&
        typeof fallbackOrOptions.defaultValue === "string"
      ) {
        return fallbackOrOptions.defaultValue
      }
      return key
    }
  })
}))

vi.mock("@/services/watchlists", () => ({
  downloadWatchlistOutput: (...args: unknown[]) =>
    serviceMocks.downloadWatchlistOutput(...args),
  downloadWatchlistOutputBinary: (...args: unknown[]) =>
    serviceMocks.downloadWatchlistOutputBinary(...args)
}))

const buildOutput = (overrides: Partial<WatchlistOutput> = {}): WatchlistOutput => ({
  id: 42,
  run_id: 9,
  job_id: 7,
  type: "briefing",
  format: "md",
  title: "Daily Brief",
  content: null,
  storage_path: "watchlists/brief-42.md",
  metadata: {},
  media_item_id: null,
  chatbook_path: null,
  version: 1,
  expires_at: null,
  expired: false,
  created_at: "2026-02-20T00:00:00Z",
  ...overrides
})

describe("OutputPreviewDrawer audio support", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    serviceMocks.downloadWatchlistOutput.mockResolvedValue("# Briefing")
    serviceMocks.downloadWatchlistOutputBinary.mockResolvedValue(new Uint8Array([1, 2, 3]).buffer)
    vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:audio-output")
    vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => undefined)
  })

  it("uses binary download and renders audio player for audio outputs", async () => {
    render(
      <OutputPreviewDrawer
        open
        onClose={vi.fn()}
        output={buildOutput({
          type: "tts_audio",
          format: "mp3",
          storage_path: "watchlists/audio-42.mp3"
        })}
      />
    )

    await waitFor(() => {
      expect(serviceMocks.downloadWatchlistOutputBinary).toHaveBeenCalledWith(42)
    })

    expect(serviceMocks.downloadWatchlistOutput).not.toHaveBeenCalled()
    expect(screen.getByText("Audio playback")).toBeInTheDocument()
    expect(screen.getByTestId("output-preview-provenance")).toHaveTextContent(
      "Monitor #7 • Run #9 • Artifact: Audio briefing"
    )
    const audioElement = document.querySelector("audio")
    expect(audioElement).not.toBeNull()
    expect(audioElement?.getAttribute("src")).toBe("blob:audio-output")
  })

  it("surfaces audio artifact graph and fallback metadata", async () => {
    render(
      <OutputPreviewDrawer
        open
        onClose={vi.fn()}
        output={buildOutput({
          type: "tts_audio",
          format: "mp3",
          storage_path: "watchlists/audio-42.mp3",
          metadata: {
            audio: {
              status: "completed",
              fallback_reason: "Speaker B voice failed; used fallback single voice.",
              script_artifact: {
                title: "Briefing script",
                uri: "file:///srv/tldw/watchlists/runs/9/script.md",
                download_url: "/api/v1/watchlists/runs/9/audio/script/download"
              },
              speaker_artifacts: [
                {
                  speaker_id: "host",
                  label: "Host",
                  uri: "file:///srv/tldw/watchlists/runs/9/host.mp3"
                },
                {
                  speaker_id: "analyst",
                  label: "Analyst",
                  uri: "file:///srv/tldw/watchlists/runs/9/analyst.mp3",
                  download_url: "/api/v1/watchlists/runs/9/audio/speakers/analyst/download"
                }
              ],
              final_artifact: {
                title: "Final mix",
                uri: "file:///srv/tldw/watchlists/runs/9/final.mp3",
                download_url: "/api/v1/watchlists/runs/9/audio/final/download"
              }
            }
          }
        })}
      />
    )

    await waitFor(() => {
      expect(serviceMocks.downloadWatchlistOutputBinary).toHaveBeenCalledWith(42)
    })

    expect(screen.getByText("Audio artifacts")).toBeInTheDocument()
    expect(screen.getByText("Completed")).toBeInTheDocument()
    expect(screen.getByText("Briefing script")).toBeInTheDocument()
    expect(screen.getByText("Host")).toBeInTheDocument()
    expect(screen.getByText("Analyst")).toBeInTheDocument()
    expect(screen.getByText("Final mix")).toBeInTheDocument()
    expect(screen.getByText("script.md")).toBeInTheDocument()
    expect(screen.getByText("host.mp3")).toBeInTheDocument()
    expect(screen.getByRole("link", { name: "Open Briefing script" })).toHaveAttribute(
      "href",
      "/api/v1/watchlists/runs/9/audio/script/download"
    )
    expect(screen.getByRole("link", { name: "Open Analyst" })).toHaveAttribute(
      "href",
      "/api/v1/watchlists/runs/9/audio/speakers/analyst/download"
    )
    expect(screen.getByRole("link", { name: "Open Final mix" })).toHaveAttribute(
      "href",
      "/api/v1/watchlists/runs/9/audio/final/download"
    )
    expect(screen.queryByText(/file:\/\//)).not.toBeInTheDocument()
    expect(screen.queryByText(/srv\/tldw/)).not.toBeInTheDocument()
    expect(screen.getByText(/Speaker B voice failed/)).toBeInTheDocument()
  })

  it("keeps text-preview flow for non-audio outputs", async () => {
    render(
      <OutputPreviewDrawer
        open
        onClose={vi.fn()}
        output={buildOutput({
          type: "brief",
          format: "md",
          metadata: {
            audio_briefing_requested: true,
            audio_briefing_status: "pending",
            audio_briefing_task_id: "task_audio_pending"
          }
        })}
      />
    )

    await waitFor(() => {
      expect(serviceMocks.downloadWatchlistOutput).toHaveBeenCalledWith(42)
    })

    expect(serviceMocks.downloadWatchlistOutputBinary).not.toHaveBeenCalled()
    expect(await screen.findByTestId("output-preview-provenance")).toHaveTextContent(
      "Monitor #7 • Run #9 • Artifact: Markdown"
    )
    expect(screen.getByText("Audio artifacts")).toBeInTheDocument()
    expect(screen.getByText("Queued")).toBeInTheDocument()
    expect(screen.getByText(/task_audio_pending/)).toBeInTheDocument()
    expect(await screen.findByText("# Briefing")).toBeInTheDocument()
  })

  it("restores focus to the launch control when the drawer closes", async () => {
    const trigger = document.createElement("button")
    trigger.type = "button"
    trigger.textContent = "Open output preview"
    document.body.appendChild(trigger)
    trigger.focus()

    const { rerender } = render(
      <OutputPreviewDrawer
        open
        onClose={vi.fn()}
        output={buildOutput({ type: "brief", format: "md" })}
      />
    )

    await waitFor(() => {
      expect(serviceMocks.downloadWatchlistOutput).toHaveBeenCalledWith(42)
    })

    const drawerButton = document.querySelector(".ant-drawer button")
    expect(drawerButton).not.toBeNull()
    ;(drawerButton as HTMLButtonElement).focus()

    rerender(
      <OutputPreviewDrawer
        open={false}
        onClose={vi.fn()}
        output={buildOutput({ type: "brief", format: "md" })}
      />
    )

    await waitFor(() => {
      expect(trigger).toHaveFocus()
    })

    trigger.remove()
  })
})
