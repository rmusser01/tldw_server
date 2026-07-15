import React from "react";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { PlaylistPreflightPanel } from "../PlaylistPreflightPanel";
import type { PlaylistInspectionCandidate } from "../usePlaylistInspection";
import type { PlaylistPreflightItem } from "@/services/tldw/playlist-ingest";

const virtualizerHarness = vi.hoisted(() => ({
  latestOptions: null as null | {
    count: number;
    overscan?: number;
    getItemKey?: (index: number) => React.Key;
  },
  scrollCalls: [] as number[],
  setNativeRangeStart: null as null | React.Dispatch<
    React.SetStateAction<number>
  >,
}));

vi.mock("@tanstack/react-virtual", async () => {
  const ReactModule = await import("react");
  return {
    useVirtualizer: (options: {
      count: number;
      overscan?: number;
      getItemKey?: (index: number) => React.Key;
    }) => {
      virtualizerHarness.latestOptions = options;
      const [start, setStart] = ReactModule.useState(0);
      virtualizerHarness.setNativeRangeStart = setStart;
      const mountedCount = Math.min(12, options.count);
      const boundedStart = Math.min(
        start,
        Math.max(0, options.count - mountedCount),
      );
      const scrollToIndex = ReactModule.useCallback(
        (index: number) => {
          virtualizerHarness.scrollCalls.push(index);
          setStart(Math.max(0, index - mountedCount + 1));
        },
        [mountedCount],
      );
      return {
        getTotalSize: () => options.count * 76,
        getVirtualItems: () =>
          Array.from({ length: mountedCount }, (_, offset) => {
            const index = boundedStart + offset;
            return {
              index,
              start: index * 76,
              size: 76,
              key: options.getItemKey?.(index) ?? index,
            };
          }),
        measureElement: vi.fn(),
        scrollToIndex,
      };
    },
  };
});

type Task3Candidate = PlaylistInspectionCandidate & {
  selectedOccurrenceIds: ReadonlySet<string>;
  sessionDuplicateOccurrenceIds: ReadonlySet<string>;
  selectionWarning: "changed" | "ambiguous" | null;
};

const item = (
  ordinal: number,
  overrides: Partial<PlaylistPreflightItem> = {},
): PlaylistPreflightItem => ({
  occurrenceId: `occ-${ordinal}`,
  ordinal,
  occurrenceIndexForSource: 1,
  sourceUrl: `https://www.youtube.com/watch?v=video-${ordinal}`,
  normalizedSourceId: `youtube:video:${ordinal}`,
  sourceKind: "youtube_video",
  availability: "available",
  duplicateStatus: "new",
  duplicateOfOccurrenceId: null,
  selectedByDefault: true,
  displayMetadata: {
    title: `Video ${ordinal}`,
    channelOrUploader: "Conference channel",
    durationSeconds: 90 + ordinal,
    playlistId: "PL500",
    playlistTitle: "Conference 500",
  },
  ...overrides,
});

const candidate = (
  items: PlaylistPreflightItem[],
  overrides: Partial<Task3Candidate> = {},
): Task3Candidate => ({
  key: "https://www.youtube.com/playlist?list=PL500",
  url: "https://www.youtube.com/playlist?list=PL500",
  status: "ready",
  preflightId: "preflight-1",
  summary: {
    contractVersion: 2,
    preflightId: "preflight-1",
    status: "ready",
    sourceUrl: "https://www.youtube.com/playlist?list=PL500",
    sourceKind: "youtube_playlist",
    playlistId: "PL500",
    summary: {
      playlistTitle: "Conference 500",
      totalCount: items.length,
      loadedCount: items.length,
      ingestibleCount: items.filter((entry) => entry.sourceUrl).length,
      unavailableCount: items.filter((entry) => !entry.sourceUrl).length,
      duplicateCount: items.filter(
        (entry) =>
          entry.duplicateStatus === "duplicate_existing" ||
          entry.duplicateStatus === "duplicate_in_batch",
      ).length,
      selectedCount: items.filter((entry) => entry.sourceUrl).length,
      warnings: [],
    },
    error: null,
    createdAt: "2026-07-13T00:00:00Z",
    updatedAt: "2026-07-13T00:00:01Z",
    expiresAt: "2026-07-14T00:00:00Z",
  },
  items,
  nextCursor: null,
  error: null,
  selectedOccurrenceIds: new Set(
    items
      .filter((entry) => entry.sourceUrl && entry.selectedByDefault === true)
      .map((entry) => entry.occurrenceId),
  ),
  sessionDuplicateOccurrenceIds: new Set(),
  selectionWarning: null,
  ...overrides,
});

const interpolate = (value: string, options?: Record<string, unknown>) =>
  value.replace(/\{\{(\w+)\}\}/g, (_match, key) =>
    String(options?.[key] ?? ""),
  );

const renderPanel = (
  task3Candidate: Task3Candidate,
  overrides: Partial<React.ComponentProps<typeof PlaylistPreflightPanel>> = {},
) => {
  const qi = vi.fn(
    (key: string, fallback: string, options?: Record<string, unknown>) =>
      interpolate(fallback, options),
  );
  const callbacks = {
    onCancel: vi.fn(),
    onRetry: vi.fn(),
    onRemove: vi.fn(),
    onRefresh: vi.fn(),
    onSelectionChange: vi.fn(),
    onSelectionBatchChange: vi.fn(),
  };
  render(
    <PlaylistPreflightPanel
      candidate={task3Candidate}
      qi={qi}
      {...callbacks}
      {...overrides}
    />,
  );
  return { qi, ...callbacks };
};

describe("PlaylistPreflightPanel v2 virtualization", () => {
  beforeEach(() => {
    virtualizerHarness.latestOptions = null;
    virtualizerHarness.scrollCalls = [];
    virtualizerHarness.setNativeRangeStart = null;
  });

  it("keeps a 500-item list bounded with stable occurrence keys and list position semantics", () => {
    const items = Array.from({ length: 500 }, (_, index) => item(index + 1));

    renderPanel(candidate(items));

    const list = screen.getByRole("list", { name: "Playlist videos" });
    const mountedRows = within(list).getAllByRole("listitem");
    expect(mountedRows.length).toBeGreaterThan(0);
    expect(mountedRows.length).toBeLessThan(30);
    expect(mountedRows[0]).toHaveAttribute("aria-setsize", "500");
    expect(mountedRows[0]).toHaveAttribute("aria-posinset", "1");
    expect(mountedRows[0]).toHaveAttribute("data-occurrence-id", "occ-1");
    expect(virtualizerHarness.latestOptions?.count).toBe(500);
    expect(virtualizerHarness.latestOptions?.overscan).toBeLessThanOrEqual(8);
    expect(virtualizerHarness.latestOptions?.getItemKey?.(0)).toBe("occ-1");
    expect(screen.queryByText("500. Video 500")).not.toBeInTheDocument();
  });

  it("renders title and ordinal first, secondary metadata second, and URLs only in details", () => {
    const first = item(1, { duplicateStatus: "duplicate_existing" });

    renderPanel(candidate([first]));

    expect(screen.getByText("1. Video 1")).toBeInTheDocument();
    expect(
      screen.getByText(
        /Conference 500.*Conference channel.*available.*duplicate/i,
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByText(first.sourceUrl as string).closest("details"),
    ).not.toBeNull();
    expect(document.querySelector("img")).toBeNull();
  });

  it("loads thumbnails only on request without a referrer and treats failures as cosmetic", () => {
    const thumbnailUrl = "https://i.ytimg.com/vi/video-1/hqdefault.jpg";
    renderPanel(
      candidate([
        item(1, {
          displayMetadata: {
            title: "Video 1",
            thumbnailUrl,
          },
        }),
      ]),
    );

    expect(screen.queryByRole("img", { name: "Thumbnail for Video 1" })).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: "Load thumbnail for Video 1" }));

    const thumbnail = screen.getByRole("img", { name: "Thumbnail for Video 1" });
    expect(thumbnail).toHaveAttribute("src", thumbnailUrl);
    expect(thumbnail).toHaveAttribute("loading", "lazy");
    expect(thumbnail).toHaveAttribute("referrerpolicy", "no-referrer");

    fireEvent.error(thumbnail);

    expect(screen.queryByRole("img", { name: "Thumbnail for Video 1" })).toBeNull();
    expect(screen.getByText("Thumbnail unavailable")).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: "Select playlist item 1: Video 1" })).toBeEnabled();
  });

  it("classifies only confirmed statuses as duplicates and only exact new rows in filters", () => {
    const rows = [
      item(1),
      item(2, { duplicateStatus: "duplicate_existing" }),
      item(3, { duplicateStatus: "duplicate_in_batch" }),
      item(4, { duplicateStatus: "unknown", selectedByDefault: false }),
      item(5, { duplicateStatus: null }),
    ];

    const { qi } = renderPanel(candidate(rows));

    expect(screen.getByText("2 duplicates")).toBeInTheDocument();
    const unknownRow = document.querySelector<HTMLElement>(
      '[data-occurrence-id="occ-4"]',
    );
    const nullStatusRow = document.querySelector<HTMLElement>(
      '[data-occurrence-id="occ-5"]',
    );
    expect(unknownRow).not.toBeNull();
    expect(nullStatusRow).not.toBeNull();
    expect(unknownRow).not.toHaveTextContent(/ • duplicate$/i);
    expect(nullStatusRow).not.toHaveTextContent(/ • duplicate$/i);
    expect(
      within(unknownRow as HTMLElement).getByText(/duplicate status unknown/i),
    ).toBeInTheDocument();
    expect(qi).toHaveBeenCalledWith(
      "playlistPreflight.duplicateUnknown",
      "duplicate status unknown",
    );

    const filter = screen.getByRole("combobox", {
      name: "Filter playlist items",
    });
    const list = screen.getByRole("list", { name: "Playlist videos" });
    fireEvent.change(filter, { target: { value: "new" } });
    expect(
      within(list)
        .getAllByRole("listitem")
        .map((row) => row.getAttribute("data-occurrence-id")),
    ).toEqual(["occ-1"]);

    fireEvent.change(filter, { target: { value: "duplicates" } });
    expect(
      within(list)
        .getAllByRole("listitem")
        .map((row) => row.getAttribute("data-occurrence-id")),
    ).toEqual(["occ-2", "occ-3"]);
  });

  it("dispatches one 500-row bulk selection update instead of per-row callbacks", () => {
    const items = Array.from({ length: 500 }, (_, index) => item(index + 1));
    const { onSelectionBatchChange, onSelectionChange } = renderPanel(
      candidate(items),
    );

    fireEvent.click(screen.getByRole("button", { name: "Select none" }));

    expect(onSelectionBatchChange).toHaveBeenCalledTimes(1);
    const updates = onSelectionBatchChange.mock.calls[0]?.[0];
    expect(updates).toHaveLength(500);
    expect(updates?.[0]).toEqual({ occurrenceId: "occ-1", selected: false });
    expect(updates?.[499]).toEqual({
      occurrenceId: "occ-500",
      selected: false,
    });
    expect(onSelectionChange).not.toHaveBeenCalled();
  });

  it("selects only exact new rows when duplicate status is unknown or absent", () => {
    const rows = [
      item(1),
      item(2, { duplicateStatus: "duplicate_existing" }),
      item(3, { duplicateStatus: "duplicate_in_batch" }),
      item(4, { duplicateStatus: "unknown" }),
      item(5, { duplicateStatus: null }),
    ];
    const { onSelectionBatchChange } = renderPanel(candidate(rows));

    fireEvent.click(screen.getByRole("button", { name: "Select new" }));

    expect(onSelectionBatchChange).toHaveBeenCalledWith([
      { occurrenceId: "occ-1", selected: true },
      { occurrenceId: "occ-2", selected: false },
      { occurrenceId: "occ-3", selected: false },
      { occurrenceId: "occ-4", selected: false },
      { occurrenceId: "occ-5", selected: false },
    ]);
  });

  it("treats unknown as duplicate only when independently repeated in session", () => {
    const unknown = item(1, {
      duplicateStatus: "unknown",
      selectedByDefault: false,
    });
    const { onSelectionBatchChange } = renderPanel(
      candidate([unknown], {
        sessionDuplicateOccurrenceIds: new Set([unknown.occurrenceId]),
      }),
    );

    expect(screen.getByText("1 duplicates")).toBeInTheDocument();
    const row = document.querySelector<HTMLElement>(
      '[data-occurrence-id="occ-1"]',
    );
    expect(row).not.toBeNull();
    expect(
      within(row as HTMLElement).getByText(/available • duplicate$/i),
    ).toBeInTheDocument();
    expect(screen.queryByText(/duplicate status unknown/i)).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: "Select new" }));
    expect(onSelectionBatchChange).toHaveBeenCalledWith([
      { occurrenceId: "occ-1", selected: false },
    ]);
  });

  it("scrolls before restoring ArrowDown focus to the next occurrence", async () => {
    const items = Array.from({ length: 500 }, (_, index) => item(index + 1));

    renderPanel(candidate(items));

    const twelfth = document.querySelector<HTMLElement>(
      '[data-occurrence-id="occ-12"]',
    );
    expect(twelfth).not.toBeNull();
    twelfth?.focus();
    fireEvent.keyDown(twelfth as HTMLElement, { key: "ArrowDown" });

    await waitFor(() =>
      expect(document.activeElement).toHaveAttribute(
        "data-occurrence-id",
        "occ-13",
      ),
    );
    expect(virtualizerHarness.scrollCalls).toContain(12);
  });

  it("re-homes list focus when native scrolling unmounts the active row", async () => {
    const items = Array.from({ length: 500 }, (_, index) => item(index + 1));

    renderPanel(candidate(items));

    const sixth = document.querySelector<HTMLElement>(
      '[data-occurrence-id="occ-6"]',
    );
    expect(sixth).not.toBeNull();
    sixth?.focus();

    act(() => virtualizerHarness.setNativeRangeStart?.(100));

    await waitFor(() =>
      expect(document.activeElement).toHaveAttribute(
        "data-occurrence-id",
        "occ-101",
      ),
    );
  });

  it("does not steal focus after the user intentionally leaves the virtual list", async () => {
    const items = Array.from({ length: 500 }, (_, index) => item(index + 1));
    render(
      <button type="button" data-testid="outside-focus-target">
        Outside focus
      </button>,
    );
    renderPanel(candidate(items));

    const sixth = document.querySelector<HTMLElement>(
      '[data-occurrence-id="occ-6"]',
    );
    const outside = screen.getByTestId("outside-focus-target");
    sixth?.focus();
    outside.focus();

    act(() => virtualizerHarness.setNativeRangeStart?.(100));

    await waitFor(() => expect(document.activeElement).toBe(outside));
  });

  it("gives repeated titles unique localized checkbox names by ordinal", () => {
    const rows = [
      item(1, { displayMetadata: { title: "Repeated title" } }),
      item(2, { displayMetadata: { title: "Repeated title" } }),
    ];
    const { qi } = renderPanel(candidate(rows));

    expect(
      screen.getByRole("checkbox", {
        name: "Select playlist item 1: Repeated title",
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("checkbox", {
        name: "Select playlist item 2: Repeated title",
      }),
    ).toBeInTheDocument();
    expect(qi).toHaveBeenCalledWith(
      "playlistPreflight.itemSelectionAria",
      "Select playlist item {{ordinal}}: {{title}}",
      { ordinal: 1, title: "Repeated title" },
    );
  });

  it("localizes known availability values and fails closed for unknown values", () => {
    const rows = [
      item(1, { availability: "needs_auth" }),
      item(2, {
        availability:
          "provider_secret_state" as PlaylistPreflightItem["availability"],
      }),
    ];
    const { qi } = renderPanel(candidate(rows));

    expect(screen.getByText(/Authentication required/)).toBeInTheDocument();
    expect(screen.getByText(/Availability unknown/)).toBeInTheDocument();
    expect(screen.queryByText(/needs_auth/)).not.toBeInTheDocument();
    expect(screen.queryByText(/provider_secret_state/)).not.toBeInTheDocument();
    expect(qi).toHaveBeenCalledWith(
      "playlistPreflight.availabilityNeedsAuth",
      "Authentication required",
    );
    expect(qi).toHaveBeenCalledWith(
      "playlistPreflight.availabilityUnknown",
      "Availability unknown",
    );
  });

  it("keeps unavailable rows visible and limits bulk controls to visible eligible rows", () => {
    const rows = [
      item(1),
      item(2, {
        duplicateStatus: "duplicate_existing",
        duplicateOfOccurrenceId: "library:2",
      }),
      item(3, {
        sourceUrl: null,
        availability: "unavailable",
        selectedByDefault: false,
      }),
      item(4, {
        availability: "needs_auth",
        selectedByDefault: false,
      }),
    ];

    const Controlled = () => {
      const [selected, setSelected] = React.useState<ReadonlySet<string>>(
        new Set(["occ-1", "occ-2"]),
      );
      return (
        <PlaylistPreflightPanel
          candidate={candidate(rows, { selectedOccurrenceIds: selected })}
          qi={(key, fallback, options) => interpolate(fallback, options)}
          onCancel={vi.fn()}
          onRetry={vi.fn()}
          onRemove={vi.fn()}
          onRefresh={vi.fn()}
          onSelectionChange={(occurrenceId, nextSelected) =>
            setSelected((current) => {
              const next = new Set(current);
              if (nextSelected) next.add(occurrenceId);
              else next.delete(occurrenceId);
              return next;
            })
          }
          onSelectionBatchChange={(updates) =>
            setSelected((current) => {
              const next = new Set(current);
              for (const { occurrenceId, selected: nextSelected } of updates) {
                if (nextSelected) next.add(occurrenceId);
                else next.delete(occurrenceId);
              }
              return next;
            })
          }
        />
      );
    };
    render(<Controlled />);

    expect(
      screen.getByRole("checkbox", {
        name: "Select playlist item 3: Video 3",
      }),
    ).toBeDisabled();
    expect(
      screen.getByRole("checkbox", {
        name: "Select playlist item 4: Video 4",
      }),
    ).toBeDisabled();

    fireEvent.change(
      screen.getByRole("combobox", { name: "Filter playlist items" }),
      {
        target: { value: "new" },
      },
    );
    fireEvent.click(screen.getByRole("button", { name: "Select none" }));
    fireEvent.change(
      screen.getByRole("combobox", { name: "Filter playlist items" }),
      {
        target: { value: "all" },
      },
    );

    expect(
      screen.getByRole("checkbox", {
        name: "Select playlist item 1: Video 1",
      }),
    ).not.toBeChecked();
    expect(
      screen.getByRole("checkbox", {
        name: "Select playlist item 2: Video 2",
      }),
    ).toBeChecked();

    fireEvent.click(screen.getByRole("button", { name: "Select new" }));
    expect(
      screen.getByRole("checkbox", {
        name: "Select playlist item 1: Video 1",
      }),
    ).toBeChecked();
    expect(
      screen.getByRole("checkbox", {
        name: "Select playlist item 2: Video 2",
      }),
    ).not.toBeChecked();

    fireEvent.click(screen.getByRole("button", { name: "Select all" }));
    expect(
      screen.getByRole("checkbox", {
        name: "Select playlist item 1: Video 1",
      }),
    ).toBeChecked();
    expect(
      screen.getByRole("checkbox", {
        name: "Select playlist item 2: Video 2",
      }),
    ).toBeChecked();
  });

  it("localizes controls and surfaces refresh ambiguity without enabling Task 4", () => {
    const { qi, onRefresh } = renderPanel(
      candidate([item(1)], { selectionWarning: "ambiguous" }),
    );

    expect(
      screen.getByText(/added, removed, reordered, or could not be matched/i),
    ).toBeInTheDocument();
    fireEvent.click(
      screen.getByRole("button", { name: "Refresh playlist inspection" }),
    );
    expect(onRefresh).toHaveBeenCalledTimes(1);
    expect(screen.getByRole("button", { name: "Add 1 video" })).toBeDisabled();
    expect(qi).toHaveBeenCalledWith(
      "playlistPreflight.refreshAria",
      "Refresh playlist inspection",
    );
    expect(qi).toHaveBeenCalledWith(
      "playlistPreflight.itemSelectionAria",
      "Select playlist item {{ordinal}}: {{title}}",
      { ordinal: 1, title: "Video 1" },
    );
  });

  it("preserves safe Task 2 error and retry behavior", () => {
    const onRetry = vi.fn();
    renderPanel(
      candidate([], {
        status: "failed",
        error: {
          code: "preflight_incomplete",
          message: "Playlist inspection is incomplete. Try again.",
          retryable: true,
        },
      }),
      { onRetry },
    );

    expect(
      screen.getByText("Playlist inspection is incomplete. Try again."),
    ).toBeInTheDocument();
    fireEvent.click(
      screen.getByRole("button", {
        name: `Retry playlist inspection for ${candidate([]).url}`,
      }),
    );
    expect(onRetry).toHaveBeenCalledTimes(1);
  });
});
