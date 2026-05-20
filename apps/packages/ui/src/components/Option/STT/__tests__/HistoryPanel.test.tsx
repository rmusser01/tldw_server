import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string) => fallback ?? key
  })
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({
    error: vi.fn(),
    success: vi.fn(),
    info: vi.fn()
  })
}))

import type { SttHistoryEntry } from "../HistoryPanel"
import { HistoryPanel } from "../HistoryPanel"

const sampleEntries: SttHistoryEntry[] = [
  {
    id: "rec-1",
    recordingId: "dexie-1",
    createdAt: new Date("2026-03-06T14:00:00Z").toISOString(),
    durationMs: 5000,
    results: [
      {
        model: "whisper-1",
        text: "Hello world from whisper",
        latencyMs: 1200,
        wordCount: 4,
        config: {
          model: "whisper-1",
          language: "en",
          task: "transcribe",
          responseFormat: "verbose_json",
          timestampGranularities: ["word"],
          segmentationEnabled: true
        },
        metadata: {
          createdAt: "2026-03-06T14:05:09.000Z",
          audioSourceLabel: "Recorded audio",
          audioSizeBytes: 1536,
          clientLatencyMs: 1200,
          language: "en",
          durationSeconds: 4.2,
          segmentCount: 3,
          wordCount: 4
        }
      },
      {
        model: "distil-v3",
        text: "Hello world from distil",
        latencyMs: 800,
        wordCount: 4
      }
    ]
  }
]

describe("HistoryPanel", () => {
  const defaultProps = {
    entries: [] as SttHistoryEntry[],
    onRecompare: vi.fn(),
    onExport: vi.fn(),
    onDelete: vi.fn(),
    onClearAll: vi.fn()
  }

  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("shows empty state when no entries", () => {
    render(<HistoryPanel {...defaultProps} />)

    expect(
      screen.getByText(
        "Completed transcriptions will appear here after you run a comparison."
      )
    ).toBeInTheDocument()
  })

  it("renders entries with model count and timestamp", () => {
    render(<HistoryPanel {...defaultProps} entries={sampleEntries} />)

    expect(screen.getByText(/2 models compared/)).toBeInTheDocument()
    // Duration tag should be present
    expect(screen.getByText(/5\.0s/)).toBeInTheDocument()
  })

  it("shows clear all button when entries exist", () => {
    render(<HistoryPanel {...defaultProps} entries={sampleEntries} />)

    const clearBtn = screen.getByRole("button", { name: /clear all/i })
    expect(clearBtn).toBeInTheDocument()
  })

  it("does not show clear all button when no entries", () => {
    render(<HistoryPanel {...defaultProps} />)

    expect(
      screen.queryByRole("button", { name: /clear all/i })
    ).not.toBeInTheDocument()
  })

  it("calls onDelete when delete clicked", () => {
    render(<HistoryPanel {...defaultProps} entries={sampleEntries} />)

    // Expand the collapse panel first
    const header = screen.getByText(/2 models compared/)
    fireEvent.click(header)

    const deleteBtn = screen.getByRole("button", { name: /delete/i })
    fireEvent.click(deleteBtn)

    expect(defaultProps.onDelete).toHaveBeenCalledWith("rec-1")
  })

  it("renders persisted STT provenance metadata", () => {
    render(<HistoryPanel {...defaultProps} entries={sampleEntries} />)

    fireEvent.click(screen.getByText(/2 models compared/))

    expect(screen.getByText("2026-03-06 14:05:09 UTC")).toBeInTheDocument()
    expect(screen.getByText("Recorded audio")).toBeInTheDocument()
    expect(screen.getByText("1.5 KB")).toBeInTheDocument()
    expect(screen.getByText("Client measured 1.2s")).toBeInTheDocument()
    expect(screen.getByText("Language en")).toBeInTheDocument()
    expect(screen.getByText("Task transcribe")).toBeInTheDocument()
    expect(screen.getByText("Format verbose_json")).toBeInTheDocument()
    expect(screen.getByText("Timestamps word")).toBeInTheDocument()
    expect(screen.getByText("Segmentation on")).toBeInTheDocument()
    expect(screen.getByText("Duration 4.2s")).toBeInTheDocument()
    expect(screen.getByText("3 segments")).toBeInTheDocument()
  })
})
