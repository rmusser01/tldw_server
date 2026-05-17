import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { PlaylistPreflightPanel } from "../PlaylistPreflightPanel"
import type { PlaylistPreflightResult } from "@/services/tldw/playlist-preflight"

const buildResult = (): PlaylistPreflightResult => ({
  sourceUrl: "https://www.youtube.com/playlist?list=PLtest",
  sourceKind: "youtube_playlist",
  playlistId: "PLtest",
  playlistTitle: "Conference 2010",
  videoId: null,
  itemCount: 6,
  selectedCount: 6,
  duplicateCount: 0,
  warnings: [],
  items: Array.from({ length: 6 }, (_, index) => ({
    id: `youtube:video:item-${index + 1}`,
    ordinal: index + 1,
    sourceUrl: `https://www.youtube.com/watch?v=item-${index + 1}`,
    normalizedSourceId: `youtube:video:item-${index + 1}`,
    sourceKind: "youtube_video",
    title: `Talk ${index + 1}`,
    speaker: null,
    durationSeconds: null,
    publishedAt: null,
    thumbnailUrl: null,
    duplicateStatus: "new",
    duplicateOfOrdinal: null,
    selected: true
  }))
})

describe("PlaylistPreflightPanel", () => {
  it("shows the full preflight item list and emits item selection changes", () => {
    const onItemSelectionChange = vi.fn()
    render(
      <PlaylistPreflightPanel
        candidateUrl="https://www.youtube.com/playlist?list=PLtest"
        result={buildResult()}
        onPreview={vi.fn()}
        onAddItems={vi.fn()}
        onItemSelectionChange={onItemSelectionChange}
      />
    )

    expect(screen.getByText("6. Talk 6")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("checkbox", { name: "Select Talk 2" }))

    expect(onItemSelectionChange).toHaveBeenCalledWith(2, false)
  })

  it("shows duplicate policy controls only when duplicates are present", () => {
    const onDuplicatePolicyChange = vi.fn()
    const result = buildResult()
    result.duplicateCount = 1
    result.items[1] = {
      ...result.items[1],
      duplicateStatus: "duplicate_existing",
      selected: false
    }

    render(
      <PlaylistPreflightPanel
        candidateUrl="https://www.youtube.com/playlist?list=PLtest"
        result={result}
        duplicatePolicy="skip"
        onDuplicatePolicyChange={onDuplicatePolicyChange}
        onPreview={vi.fn()}
        onAddItems={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("radio", { name: "Include existing" }))

    expect(onDuplicatePolicyChange).toHaveBeenCalledWith("include_existing")
  })

  it("renders playlist preflight state indicators through design-system primitives", () => {
    const result = buildResult()
    result.duplicateCount = 1
    result.items[1] = {
      ...result.items[1],
      duplicateStatus: "duplicate_existing",
      selected: false
    }

    render(
      <PlaylistPreflightPanel
        candidateUrl="https://www.youtube.com/playlist?list=PLtest"
        error="Unable to preview playlist metadata"
        result={result}
        duplicatePolicy="skip"
        onPreview={vi.fn()}
        onAddItems={vi.fn()}
      />
    )

    const warning = screen.getByText("Unable to preview playlist metadata")
    expect(warning.closest('[data-ds-component="Alert"]')).toBeInTheDocument()

    const duplicateSummary = screen.getByText("1 duplicates")
    expect(
      duplicateSummary.closest('[data-ds-component="Badge"]')
    ).toBeInTheDocument()

    const duplicateItem = screen.getByText("duplicate")
    expect(duplicateItem.closest('[data-ds-component="Badge"]')).toBeInTheDocument()
  })
})
