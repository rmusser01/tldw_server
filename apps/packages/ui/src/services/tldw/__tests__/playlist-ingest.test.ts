import { describe, expect, it, vi } from "vitest";

import {
  loadCompletePlaylistPreflightItems,
  normalizePlaylistPreflightSummary,
  type ApiPlaylistPreflightSummaryResponse,
  type PlaylistIngestPageParams,
  type PlaylistIngestRequestOptions,
  type PlaylistPreflightItem,
  type PlaylistPreflightItemsPage,
  type PlaylistPreflightSummary,
} from "../playlist-ingest";

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
  },
  ...overrides,
});

const summary = (
  loadedCount: number,
  totalCount: number | null = loadedCount,
): PlaylistPreflightSummary => ({
  contractVersion: 2,
  preflightId: "preflight-1",
  status: "ready",
  sourceUrl: "https://www.youtube.com/playlist?list=PL500",
  sourceKind: "youtube_playlist",
  playlistId: "PL500",
  summary: {
    playlistTitle: "Conference 500",
    totalCount,
    loadedCount,
    ingestibleCount: loadedCount,
    unavailableCount: 0,
    duplicateCount: 0,
    selectedCount: loadedCount,
    warnings: [],
  },
  error: null,
  createdAt: "2026-07-13T00:00:00Z",
  updatedAt: "2026-07-13T00:00:01Z",
  expiresAt: "2026-07-14T00:00:00Z",
});

const page = (
  items: PlaylistPreflightItem[],
  nextCursor: string | null,
  preflightId = "preflight-1",
): PlaylistPreflightItemsPage => ({
  contractVersion: 2,
  preflightId,
  items,
  nextCursor,
});

const apiReadySummary = (
  summaryCounts: Record<string, unknown>,
): ApiPlaylistPreflightSummaryResponse =>
  ({
    contract_version: 2,
    preflight_id: "preflight-1",
    status: "ready",
    source_url: "https://www.youtube.com/playlist?list=PL500",
    source_kind: "youtube_playlist",
    playlist_id: "PL500",
    summary: {
      playlist_title: "Conference 500",
      total_count: 0,
      ingestible_count: 0,
      unavailable_count: 0,
      duplicate_count: 0,
      selected_count: 0,
      warnings: [],
      ...summaryCounts,
    },
    error: null,
    created_at: "2026-07-13T00:00:00Z",
    updated_at: "2026-07-13T00:00:01Z",
    expires_at: "2026-07-14T00:00:00Z",
  }) as unknown as ApiPlaylistPreflightSummaryResponse;

type PageLoader = (
  preflightId: string,
  params: PlaylistIngestPageParams,
  options: PlaylistIngestRequestOptions,
) => Promise<PlaylistPreflightItemsPage>;

describe("loadCompletePlaylistPreflightItems", () => {
  it("loads a 500-item immutable snapshot in exact opaque-cursor order", async () => {
    const items = Array.from({ length: 500 }, (_, index) => item(index + 1));
    const cursors = [
      "opaque:page/+2==",
      "do-not-decode::3?",
      "cursor%2Ffour+raw",
      "final cursor = five",
    ];
    const pages = [
      page(items.slice(0, 100), cursors[0]),
      page(items.slice(100, 200), cursors[1]),
      page(items.slice(200, 300), cursors[2]),
      page(items.slice(300, 400), cursors[3]),
      page(items.slice(400), null),
    ];
    const requestedCursors: Array<string | null> = [];
    const observedSignals: AbortSignal[] = [];
    const loadPage = vi.fn<PageLoader>(async (preflightId, params, options) => {
      expect(preflightId).toBe("preflight-1");
      requestedCursors.push(params.cursor ?? null);
      observedSignals.push(options.signal as AbortSignal);
      const index = requestedCursors.length - 1;
      return pages[index];
    });
    const controller = new AbortController();

    const result = await loadCompletePlaylistPreflightItems({
      preflightId: "preflight-1",
      summary: summary(500),
      signal: controller.signal,
      loadPage,
      pageSize: 100,
    });

    expect(requestedCursors).toEqual([null, ...cursors]);
    expect(new Set(requestedCursors).size).toBe(requestedCursors.length);
    expect(observedSignals).toEqual(Array(5).fill(controller.signal));
    expect(result).toHaveLength(500);
    expect(result.map(({ occurrenceId }) => occurrenceId)).toEqual(
      items.map(({ occurrenceId }) => occurrenceId),
    );
    expect(result.map(({ ordinal }) => ordinal)).toEqual(
      Array.from({ length: 500 }, (_, index) => index + 1),
    );
  });

  it("cancels sequential paging with the shared AbortSignal and returns no partial snapshot", async () => {
    const controller = new AbortController();
    const loadPage = vi
      .fn<PageLoader>()
      .mockResolvedValueOnce(page([item(1)], "next-page"))
      .mockImplementationOnce(
        async (_preflightId, _params, options) =>
          new Promise<PlaylistPreflightItemsPage>((_resolve, reject) => {
            options.signal?.addEventListener(
              "abort",
              () => reject(new DOMException("Aborted", "AbortError")),
              { once: true },
            );
          }),
      );

    const result = loadCompletePlaylistPreflightItems({
      preflightId: "preflight-1",
      summary: summary(2),
      signal: controller.signal,
      loadPage,
      pageSize: 1,
    });
    await vi.waitFor(() => expect(loadPage).toHaveBeenCalledTimes(2));

    controller.abort();

    await expect(result).rejects.toMatchObject({ name: "AbortError" });
    expect(loadPage.mock.calls[0]?.[2].signal).toBe(controller.signal);
    expect(loadPage.mock.calls[1]?.[2].signal).toBe(controller.signal);
  });

  it("rejects a repeated cursor without refetching a completed page", async () => {
    const loadPage = vi
      .fn<PageLoader>()
      .mockResolvedValueOnce(page([item(1)], "repeat-me"))
      .mockResolvedValueOnce(page([item(2)], "repeat-me"));

    const result = loadCompletePlaylistPreflightItems({
      preflightId: "preflight-1",
      summary: summary(2),
      signal: new AbortController().signal,
      loadPage,
      pageSize: 1,
    });

    await expect(result).rejects.toMatchObject({
      code: "preflight_incomplete",
      message: "Playlist inspection is incomplete. Try again.",
      retryable: true,
    });
    expect(loadPage).toHaveBeenCalledTimes(2);
    expect(loadPage.mock.calls.map((call) => call[1].cursor ?? null)).toEqual([
      null,
      "repeat-me",
    ]);
  });

  it.each([
    {
      name: "an empty continuing page",
      readySummary: summary(1),
      firstPage: page([], "must-not-be-followed"),
    },
    {
      name: "a cursor after reaching loaded count",
      readySummary: summary(1),
      firstPage: page([item(1)], "must-not-be-followed"),
    },
    {
      name: "items exceeding loaded count",
      readySummary: summary(1),
      firstPage: page([item(1), item(2)], "must-not-be-followed"),
    },
  ])(
    "rejects $name without following its cursor",
    async ({ readySummary, firstPage }) => {
      const loadPage = vi
        .fn<PageLoader>()
        .mockResolvedValueOnce(firstPage)
        .mockResolvedValueOnce(page([], null));

      const result = loadCompletePlaylistPreflightItems({
        preflightId: "preflight-1",
        summary: readySummary,
        signal: new AbortController().signal,
        loadPage,
        pageSize: 100,
      });

      await expect(result).rejects.toMatchObject({
        code: "preflight_incomplete",
      });
      expect(loadPage).toHaveBeenCalledTimes(1);
    },
  );

  it("rejects a non-v2 ready summary before loading items", async () => {
    const loadPage = vi.fn<PageLoader>().mockResolvedValue(page([], null));
    const invalidSummary = {
      ...summary(0),
      contractVersion: 1,
    } as unknown as PlaylistPreflightSummary;

    await expect(
      loadCompletePlaylistPreflightItems({
        preflightId: "preflight-1",
        summary: invalidSummary,
        signal: new AbortController().signal,
        loadPage,
      }),
    ).rejects.toMatchObject({ code: "preflight_incomplete" });
    expect(loadPage).not.toHaveBeenCalled();
  });

  it("rejects a non-v2 items page without following its cursor", async () => {
    const loadPage = vi.fn<PageLoader>().mockResolvedValue({
      ...page([item(1)], "must-not-be-followed"),
      contractVersion: 1,
    } as unknown as PlaylistPreflightItemsPage);

    await expect(
      loadCompletePlaylistPreflightItems({
        preflightId: "preflight-1",
        summary: summary(1),
        signal: new AbortController().signal,
        loadPage,
      }),
    ).rejects.toMatchObject({ code: "preflight_incomplete" });
    expect(loadPage).toHaveBeenCalledTimes(1);
  });

  it.each([
    ["missing", {}],
    ["invalid", { loaded_count: "0" }],
  ])(
    "rejects a ready summary with %s loaded_count before loading items",
    async (_name, summaryCounts) => {
      const loadPage = vi.fn<PageLoader>().mockResolvedValue(page([], null));
      const normalized = normalizePlaylistPreflightSummary(
        apiReadySummary(summaryCounts),
      );

      await expect(
        loadCompletePlaylistPreflightItems({
          preflightId: "preflight-1",
          summary: normalized,
          signal: new AbortController().signal,
          loadPage,
        }),
      ).rejects.toMatchObject({ code: "preflight_incomplete" });
      expect(loadPage).not.toHaveBeenCalled();
    },
  );

  it.each([
    {
      name: "mismatched preflight ids",
      readySummary: summary(1),
      pages: [page([item(1)], null, "other-preflight")],
    },
    {
      name: "duplicate occurrence ids",
      readySummary: summary(2),
      pages: [
        page([item(1)], "next"),
        page([item(2, { occurrenceId: "occ-1" })], null),
      ],
    },
    {
      name: "loaded-count mismatches",
      readySummary: summary(3, null),
      pages: [page([item(1), item(2)], null)],
    },
    {
      name: "known total-count mismatches",
      readySummary: summary(2, 3),
      pages: [page([item(1), item(2)], null)],
    },
  ])(
    "rejects $name as a safe incomplete snapshot",
    async ({ readySummary, pages }) => {
      const loadPage = vi.fn<PageLoader>(
        async () => pages.shift() as PlaylistPreflightItemsPage,
      );

      const result = loadCompletePlaylistPreflightItems({
        preflightId: "preflight-1",
        summary: readySummary,
        signal: new AbortController().signal,
        loadPage,
        pageSize: 100,
      });

      await expect(result).rejects.toMatchObject({
        code: "preflight_incomplete",
        message: "Playlist inspection is incomplete. Try again.",
        retryable: true,
      });
    },
  );
});
