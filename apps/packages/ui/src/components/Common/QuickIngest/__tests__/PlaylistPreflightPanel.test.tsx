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
})
