import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { PlaylistPreflightPanel } from "../PlaylistPreflightPanel";
import type { PlaylistInspectionCandidate } from "../usePlaylistInspection";

const buildCandidate = (): PlaylistInspectionCandidate => ({
  key: "https://www.youtube.com/playlist?list=PLtest",
  url: "https://www.youtube.com/playlist?list=PLtest",
  status: "ready",
  preflightId: "preflight-1",
  summary: {
    contractVersion: 2,
    preflightId: "preflight-1",
    status: "ready",
    sourceUrl: "https://www.youtube.com/playlist?list=PLtest",
    sourceKind: "youtube_playlist",
    playlistId: "PLtest",
    summary: {
      playlistTitle: "Conference 2010",
      totalCount: 1,
      loadedCount: 1,
      ingestibleCount: 1,
      unavailableCount: 0,
      duplicateCount: 1,
      selectedCount: 0,
      warnings: [],
    },
    error: null,
    createdAt: "2026-07-13T00:00:00Z",
    updatedAt: "2026-07-13T00:00:01Z",
    expiresAt: "2026-07-14T00:00:00Z",
  },
  items: [
    {
      occurrenceId: "occ-1",
      ordinal: 1,
      occurrenceIndexForSource: 1,
      sourceUrl: "https://www.youtube.com/watch?v=item-1",
      normalizedSourceId: "youtube:video:item-1",
      sourceKind: "youtube_video",
      availability: "available",
      duplicateStatus: "duplicate_existing",
      duplicateOfOccurrenceId: "library:1",
      selectedByDefault: false,
      displayMetadata: { title: "Talk 1" },
    },
  ],
  nextCursor: null,
  error: null,
  selectedOccurrenceIds: new Set(),
  sessionDuplicateOccurrenceIds: new Set(),
  selectionWarning: null,
});

describe("PlaylistPreflightPanel", () => {
  it("renders v2 inspection indicators through design-system badges", () => {
    render(
      <PlaylistPreflightPanel
        candidate={buildCandidate()}
        onCancel={vi.fn()}
        onRetry={vi.fn()}
        onRemove={vi.fn()}
        onRefresh={vi.fn()}
        onSelectionChange={vi.fn()}
        onSelectionBatchChange={vi.fn()}
      />,
    );

    const ready = screen.getByText("Inspection ready");
    const duplicateSummary = screen.getByText("1 duplicates");
    expect(ready.closest('[data-ds-component="Badge"]')).toBeInTheDocument();
    expect(
      duplicateSummary.closest('[data-ds-component="Badge"]'),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Add 0 videos" })).toBeDisabled();
  });
});
