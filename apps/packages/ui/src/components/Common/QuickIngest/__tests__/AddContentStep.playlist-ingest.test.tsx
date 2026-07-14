// @vitest-environment jsdom
import React from "react";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { IngestWizardState } from "../IngestWizardContext";
import type {
  PlaylistIngestErrorInfo,
  PlaylistMaterialization,
  PlaylistPreflightAccepted,
  PlaylistPreflightItem,
  PlaylistPreflightItemsPage,
  PlaylistPreflightStatus,
  PlaylistPreflightSummary,
} from "@/services/tldw/playlist-ingest";
import { PlaylistIngestPublicError } from "@/services/tldw/playlist-ingest";
import { normalizeUrlForDedupe } from "@/entries/shared/ingest-payloads";

const apiMocks = vi.hoisted(() => ({
  createPlaylistPreflight: vi.fn(),
  getPlaylistPreflight: vi.fn(),
  listPlaylistPreflightItems: vi.fn(),
  cancelPlaylistPreflight: vi.fn(),
  materializePlaylistPreflight: vi.fn(),
}));

const capabilityHarness = vi.hoisted(() => ({
  hasMediaPlaylistIngestV2: true as boolean | null,
  loading: false,
}));

const translationHarness = vi.hoisted(() => ({
  keys: [] as string[],
}));

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, value?: string | Record<string, unknown>) => {
      translationHarness.keys.push(key);
      if (typeof value === "string") return value;
      const message =
        typeof value?.defaultValue === "string" ? value.defaultValue : key;
      return message.replace(/\{\{(\w+)\}\}/g, (_match, token: string) =>
        String(value?.[token] ?? ""),
      );
    },
  }),
}));

vi.mock("antd", () => ({
  Button: ({
    children,
    onClick,
    disabled,
    type: _type,
    danger: _danger,
    size: _size,
    loading: _loading,
    ...props
  }: Omit<React.ButtonHTMLAttributes<HTMLButtonElement>, "type"> & {
    type?: string;
    danger?: boolean;
    size?: string;
    loading?: boolean;
  }) => (
    <button type="button" onClick={onClick} disabled={disabled} {...props}>
      {children}
    </button>
  ),
  Input: {
    TextArea: ({
      autoSize: _autoSize,
      ...props
    }: React.TextareaHTMLAttributes<HTMLTextAreaElement> & {
      autoSize?: unknown;
    }) => <textarea {...props} />,
  },
  Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  Typography: {
    Text: ({ children, ...props }: React.HTMLAttributes<HTMLSpanElement>) => (
      <span {...props}>{children}</span>
    ),
  },
}));

vi.mock("@tanstack/react-virtual", () => ({
  useVirtualizer: ({
    count,
    getItemKey,
  }: {
    count: number;
    getItemKey?: (index: number) => React.Key;
  }) => ({
    getTotalSize: () => count * 76,
    getVirtualItems: () =>
      Array.from({ length: count }, (_, index) => ({
        index,
        start: index * 76,
        size: 76,
        key: getItemKey?.(index) ?? index,
      })),
    measureElement: vi.fn(),
    scrollToIndex: vi.fn(),
  }),
}));

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: apiMocks,
}));

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({
    capabilities:
      capabilityHarness.hasMediaPlaylistIngestV2 === null
        ? null
        : {
            ffmpegAvailable: true,
            hasMediaPlaylistIngestV2:
              capabilityHarness.hasMediaPlaylistIngestV2,
          },
    loading: capabilityHarness.loading,
  }),
}));

vi.mock("../QueueTab/FileDropZone", () => ({
  FileDropZone: () => <div data-testid="file-drop-zone" />,
}));

vi.mock("../BatchMetadataPanel", () => ({
  BatchMetadataPanel: () => <div data-testid="batch-metadata-panel" />,
}));

import { AddContentStep } from "../AddContentStep";
import {
  buildPlaylistIngestRunRequest,
  IngestWizardProvider,
  useIngestWizard,
} from "../IngestWizardContext";

const ORDINARY_URL = "https://example.com/article";
const INVALID_URL = "not a url";
const PLAYLIST_A = "https://www.youtube.com/playlist?list=PL-alpha";
const PLAYLIST_B = "https://youtu.be/video-b?list=PL-beta";
const PLAYLIST_C = "https://www.youtube.com/watch?v=video-c&list=PL-gamma";

const accepted = (preflightId: string): PlaylistPreflightAccepted => ({
  contractVersion: 2,
  preflightId,
  status: "pending",
  statusUrl: `/status/${preflightId}`,
  itemsUrl: `/items/${preflightId}`,
  expiresAt: "2026-07-14T00:00:00Z",
  limits: { maxItems: 500, globalCapacity: 10, ownerCapacity: 4 },
});

const summary = (
  preflightId: string,
  status: PlaylistPreflightStatus = "ready",
  error: PlaylistIngestErrorInfo | null = null,
): PlaylistPreflightSummary => ({
  contractVersion: 2,
  preflightId,
  status,
  sourceUrl: PLAYLIST_A,
  sourceKind: "youtube_playlist",
  playlistId: preflightId,
  summary:
    status === "ready"
      ? {
          playlistTitle: `Playlist ${preflightId}`,
          totalCount: 0,
          loadedCount: 0,
          ingestibleCount: 0,
          unavailableCount: 0,
          duplicateCount: 0,
          selectedCount: 0,
          warnings: [],
        }
      : null,
  error,
  createdAt: "2026-07-13T00:00:00Z",
  updatedAt: "2026-07-13T00:00:01Z",
  expiresAt: "2026-07-14T00:00:00Z",
});

const item = (
  occurrenceId: string,
  sourceUrl: string,
  normalizedSourceId: string,
): PlaylistPreflightItem => ({
  occurrenceId,
  ordinal: 1,
  occurrenceIndexForSource: 1,
  sourceUrl,
  normalizedSourceId,
  sourceKind: "youtube_video",
  availability: "available",
  duplicateStatus: "new",
  duplicateOfOccurrenceId: null,
  selectedByDefault: true,
  displayMetadata: { title: occurrenceId },
});

const itemWith = (
  occurrenceId: string,
  ordinal: number,
  normalizedSourceId: string,
  overrides: Partial<PlaylistPreflightItem> = {},
): PlaylistPreflightItem => ({
  ...item(
    occurrenceId,
    `https://www.youtube.com/watch?v=${occurrenceId}`,
    normalizedSourceId,
  ),
  ordinal,
  occurrenceIndexForSource: 1,
  displayMetadata: { title: occurrenceId },
  ...overrides,
});

const readySummary = (
  preflightId: string,
  loadedCount: number,
  totalCount: number | null = loadedCount,
): PlaylistPreflightSummary => {
  const result = summary(preflightId);
  return {
    ...result,
    summary: {
      ...result.summary!,
      loadedCount,
      totalCount,
      ingestibleCount: loadedCount,
      selectedCount: loadedCount,
    },
  };
};

const page = (
  preflightId: string,
  items: PlaylistPreflightItem[] = [],
  nextCursor: string | null = null,
): PlaylistPreflightItemsPage => ({
  contractVersion: 2,
  preflightId,
  items,
  nextCursor,
});

const materialization = (
  preflightId: string,
  items: PlaylistPreflightItem[],
): PlaylistMaterialization => ({
  contractVersion: 2,
  materializationId: `materialization-${preflightId}`,
  preflightId,
  status: "ready",
  items: items.map((entry) => ({
    occurrenceId: entry.occurrenceId,
    ordinal: entry.ordinal,
    sourceUrl: entry.sourceUrl!,
    normalizedSourceId: entry.normalizedSourceId,
    sourceKind: entry.sourceKind,
    displayMetadata: entry.displayMetadata,
  })),
  expiresAt: "2026-07-20T00:00:00Z",
});

const getIdFromUrl = (url: string): string =>
  new URL(url).searchParams.get("list") || "preflight";

type HarnessProps = {
  initialState?: Partial<IngestWizardState>;
  onQuickProcess?: () => void;
  onStateChange?: (state: IngestWizardState) => void;
};

let currentState: IngestWizardState | null = null;

const StateProbe = () => {
  const { state } = useIngestWizard();
  React.useEffect(() => {
    currentState = state;
  }, [state]);
  return null;
};

const Harness = ({
  initialState,
  onQuickProcess = vi.fn(),
  onStateChange,
}: HarnessProps) => (
  <IngestWizardProvider
    initialState={initialState}
    onStateChange={onStateChange}
  >
    <StateProbe />
    <AddContentStep onQuickProcess={onQuickProcess} />
  </IngestWizardProvider>
);

const addLines = async (lines: string, useEnter = false) => {
  const input = screen.getByRole("textbox", { name: "URL input area" });
  fireEvent.change(input, { target: { value: lines } });
  if (useEnter) {
    fireEvent.keyDown(input, { key: "Enter", code: "Enter" });
  } else {
    await userEvent.click(
      screen.getByRole("button", { name: "Add URLs to queue" }),
    );
  }
};

const expectProceedBlocked = () => {
  expect(
    screen.getByRole("button", { name: /Configure \d+ items/i }),
  ).toBeDisabled();
  expect(
    screen.getByRole("button", { name: "Use defaults & process" }),
  ).toBeDisabled();
};

const deferred = <T,>() => {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
};

describe("AddContentStep playlist inspection controller", () => {
  beforeEach(() => {
    currentState = null;
    capabilityHarness.hasMediaPlaylistIngestV2 = true;
    capabilityHarness.loading = false;
    translationHarness.keys = [];
    apiMocks.createPlaylistPreflight
      .mockReset()
      .mockImplementation(async ({ url }: { url: string }) =>
        accepted(getIdFromUrl(url)),
      );
    apiMocks.getPlaylistPreflight
      .mockReset()
      .mockImplementation(async (preflightId: string) => summary(preflightId));
    apiMocks.listPlaylistPreflightItems
      .mockReset()
      .mockImplementation(async (preflightId: string) => page(preflightId));
    apiMocks.cancelPlaylistPreflight.mockReset().mockResolvedValue(undefined);
    apiMocks.materializePlaylistPreflight.mockReset();
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("stages ordinary valid and invalid lines but never queues a playlist candidate", async () => {
    render(<Harness />);

    await addLines(`${ORDINARY_URL}\n${INVALID_URL}\n${PLAYLIST_A}`);

    await waitFor(() => {
      expect(apiMocks.createPlaylistPreflight).toHaveBeenCalledTimes(1);
    });
    expect(currentState?.queueItems.map((row) => row.url)).toEqual([
      ORDINARY_URL,
      INVALID_URL,
    ]);
    expect(currentState?.queueItems.some((row) => row.url === PLAYLIST_A)).toBe(
      false,
    );
    expect(screen.getByText(PLAYLIST_A)).toBeInTheDocument();
    expectProceedBlocked();

    await waitFor(() => {
      expect(screen.getByText("Inspection ready")).toBeInTheDocument();
    });
    const statusRegion = screen
      .getByText("Inspection ready")
      .closest('[role="status"]');
    expect(statusRegion).toHaveAttribute("aria-live", "polite");
    expect(statusRegion).toHaveAttribute("aria-atomic", "true");
    expect(translationHarness.keys).toEqual(
      expect.arrayContaining([
        "quickIngest.playlistInspection.readyLabel",
        "quickIngest.playlistInspection.readyMessage",
        "quickIngest.playlistInspection.removeAria",
      ]),
    );
    expectProceedBlocked();
  });

  it("materializes the exact selected playlist occurrences before adding authoritative flat rows", async () => {
    const inspectedItems = [
      itemWith("occ-first", 1, "youtube:video:first", {
        displayMetadata: {
          title: "First video",
          playlistId: "PL-alpha",
          playlistTitle: "Alpha playlist",
          channelOrUploader: "Channel A",
          durationSeconds: 125,
        },
      }),
      itemWith("occ-second", 2, "youtube:video:second", {
        displayMetadata: {
          title: "Second video",
          playlistId: "PL-alpha",
          playlistTitle: "Alpha playlist",
        },
      }),
    ];
    apiMocks.getPlaylistPreflight.mockResolvedValue(
      readySummary("PL-alpha", inspectedItems.length),
    );
    apiMocks.listPlaylistPreflightItems.mockResolvedValue(
      page("PL-alpha", inspectedItems),
    );
    apiMocks.materializePlaylistPreflight.mockResolvedValue(
      materialization("PL-alpha", inspectedItems),
    );
    render(<Harness />);

    await addLines(PLAYLIST_A);
    await userEvent.click(
      await screen.findByRole("button", { name: "Add 2 videos" }),
    );

    await waitFor(() => {
      expect(apiMocks.materializePlaylistPreflight).toHaveBeenCalledWith(
        "PL-alpha",
        ["occ-first", "occ-second"],
      );
      expect(currentState?.queueItems).toHaveLength(2);
    });
    expect(currentState?.queueItems[0]).toMatchObject({
      id: "occ-first",
      sourceRef: {
        kind: "materialized_playlist_item",
        materializationId: "materialization-PL-alpha",
        occurrenceId: "occ-first",
      },
      playlist: {
        playlistId: "PL-alpha",
        playlistTitle: "Alpha playlist",
        ordinal: 1,
        title: "First video",
        channelOrUploader: "Channel A",
        durationSeconds: 125,
        materializationExpiresAt: "2026-07-20T00:00:00Z",
      },
    });
    expect(screen.getByText("1. First video")).toBeInTheDocument();
    expect(
      within(
        screen.getByRole("list", { name: "Queued ingest items" }),
      ).getAllByText("Alpha playlist"),
    ).toHaveLength(2);
    expect(screen.queryByText("Inspection ready")).not.toBeInTheDocument();
  });

  it("recomputes in-batch duplicates from only the selected materialized subset", async () => {
    const inspectedItems = [
      itemWith("occ-unselected-first", 1, "youtube:video:shared", {
        duplicateStatus: "new",
      }),
      itemWith("occ-selected-second", 2, "youtube:video:shared", {
        duplicateStatus: "duplicate_in_batch",
        duplicateOfOccurrenceId: "occ-unselected-first",
      }),
    ];
    apiMocks.getPlaylistPreflight.mockResolvedValue(
      readySummary("PL-alpha", inspectedItems.length),
    );
    apiMocks.listPlaylistPreflightItems.mockResolvedValue(
      page("PL-alpha", inspectedItems),
    );
    apiMocks.materializePlaylistPreflight.mockResolvedValue(
      materialization("PL-alpha", [inspectedItems[1]]),
    );
    render(<Harness />);

    await addLines(PLAYLIST_A);
    await userEvent.click(
      await screen.findByRole("checkbox", {
        name: "Select playlist item 1: occ-unselected-first",
      }),
    );
    await userEvent.click(
      screen.getByRole("checkbox", {
        name: "Select playlist item 2: occ-selected-second",
      }),
    );
    expect(screen.getByText("1 duplicates")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Add 1 video" }));

    await waitFor(() => expect(currentState?.queueItems).toHaveLength(1));
    expect(apiMocks.materializePlaylistPreflight).toHaveBeenCalledWith(
      "PL-alpha",
      ["occ-selected-second"],
    );
    expect(currentState?.queueItems[0]).toMatchObject({
      id: "occ-selected-second",
      playlist: { duplicateStatus: "new" },
    });
  });

  it("keeps a selected materialized alias duplicate when it overlaps an ordinary queued URL", async () => {
    const queuedUrl = "https://example.com/video?b=2&a=1#display";
    const queuedAlias = normalizeUrlForDedupe(queuedUrl);
    const inspectedItems = [
      itemWith("occ-direct-alias", 1, queuedAlias, {
        sourceUrl: "https://materialized.example/video/source",
      }),
    ];
    apiMocks.getPlaylistPreflight.mockResolvedValue(readySummary("PL-alpha", 1));
    apiMocks.listPlaylistPreflightItems.mockResolvedValue(
      page("PL-alpha", inspectedItems),
    );
    apiMocks.materializePlaylistPreflight.mockResolvedValue(
      materialization("PL-alpha", inspectedItems),
    );
    render(
      <Harness
        initialState={{
          queueItems: [
            {
              id: "queued-direct",
              sourceRef: {
                kind: "direct_url",
                occurrenceId: "queued-direct",
                url: queuedUrl,
              },
              url: queuedUrl,
              detectedType: "video",
              icon: "Film",
              fileSize: 0,
              validation: { valid: true },
            },
          ],
        }}
      />,
    );

    await addLines(PLAYLIST_A);
    const selected = await screen.findByRole("checkbox", {
      name: "Select playlist item 1: occ-direct-alias",
    });
    expect(selected).not.toBeChecked();
    await userEvent.click(selected);
    await userEvent.click(screen.getByRole("button", { name: "Add 1 video" }));

    await waitFor(() => expect(currentState?.queueItems).toHaveLength(2));
    expect(currentState?.queueItems[1]).toMatchObject({
      id: "occ-direct-alias",
      playlist: { duplicateStatus: "duplicate_in_batch" },
      playlistReview: { selected: true },
    });
    expect(
      buildPlaylistIngestRunRequest(
        [currentState!.queueItems[1]],
        Date.parse("2026-07-13T00:00:00Z"),
      ).block,
    ).toEqual({
      code: "review_required",
      occurrenceIds: ["occ-direct-alias"],
    });
  });

  it("uses materialization order rather than pending-candidate provenance for shared sources", async () => {
    const inspectedByPreflight = {
      "PL-alpha": itemWith("occ-alpha-shared", 1, "youtube:video:shared"),
      "PL-beta": itemWith("occ-beta-shared", 1, "youtube:video:shared"),
    };
    apiMocks.getPlaylistPreflight.mockImplementation(
      async (preflightId: keyof typeof inspectedByPreflight) =>
        readySummary(preflightId, 1),
    );
    apiMocks.listPlaylistPreflightItems.mockImplementation(
      async (preflightId: keyof typeof inspectedByPreflight) =>
        page(preflightId, [inspectedByPreflight[preflightId]]),
    );
    apiMocks.materializePlaylistPreflight.mockImplementation(
      async (preflightId: keyof typeof inspectedByPreflight) =>
        materialization(preflightId, [inspectedByPreflight[preflightId]]),
    );
    render(<Harness />);

    await addLines(`${PLAYLIST_A}\n${PLAYLIST_B}`);
    const laterCandidateSelection = await screen.findByRole("checkbox", {
      name: "Select playlist item 1: occ-beta-shared",
    });
    expect(laterCandidateSelection).not.toBeChecked();
    await userEvent.click(laterCandidateSelection);
    await userEvent.click(
      screen.getAllByRole("button", { name: "Add 1 video" })[1],
    );

    await waitFor(() => expect(currentState?.queueItems).toHaveLength(1));
    expect(currentState?.queueItems[0]).toMatchObject({
      id: "occ-beta-shared",
      playlist: { duplicateStatus: "new" },
      playlistReview: { selected: true },
    });
    expect(currentState?.queueItems[0]?.playlistReview?.duplicatePolicy).toBeUndefined();

    const remainingCandidateSelection = screen.getByRole("checkbox", {
      name: "Select playlist item 1: occ-alpha-shared",
    });
    if (!(remainingCandidateSelection as HTMLInputElement).checked) {
      await userEvent.click(remainingCandidateSelection);
    }
    await userEvent.click(screen.getByRole("button", { name: "Add 1 video" }));

    await waitFor(() => expect(currentState?.queueItems).toHaveLength(2));
    expect(currentState?.queueItems.map((row) => row.playlist?.duplicateStatus)).toEqual([
      "new",
      "duplicate_in_batch",
    ]);
  });

  it("keeps a lone materialized row new when its pending candidate peer is removed", async () => {
    const inspectedByPreflight = {
      "PL-alpha": itemWith("occ-alpha-shared", 1, "youtube:video:shared"),
      "PL-beta": itemWith("occ-beta-shared", 1, "youtube:video:shared"),
    };
    apiMocks.getPlaylistPreflight.mockImplementation(
      async (preflightId: keyof typeof inspectedByPreflight) =>
        readySummary(preflightId, 1),
    );
    apiMocks.listPlaylistPreflightItems.mockImplementation(
      async (preflightId: keyof typeof inspectedByPreflight) =>
        page(preflightId, [inspectedByPreflight[preflightId]]),
    );
    apiMocks.materializePlaylistPreflight.mockImplementation(
      async (preflightId: keyof typeof inspectedByPreflight) =>
        materialization(preflightId, [inspectedByPreflight[preflightId]]),
    );
    render(<Harness />);

    await addLines(`${PLAYLIST_A}\n${PLAYLIST_B}`);
    const laterCandidateSelection = await screen.findByRole("checkbox", {
      name: "Select playlist item 1: occ-beta-shared",
    });
    await userEvent.click(laterCandidateSelection);
    await userEvent.click(
      screen.getAllByRole("button", { name: "Add 1 video" })[1],
    );
    await waitFor(() => expect(currentState?.queueItems).toHaveLength(1));

    await userEvent.click(
      screen.getByRole("button", {
        name: `Remove playlist inspection for ${PLAYLIST_A}`,
      }),
    );
    expect(currentState?.queueItems[0]).toMatchObject({
      id: "occ-beta-shared",
      playlist: { duplicateStatus: "new" },
      playlistReview: { selected: true },
    });
    expect(currentState?.queueItems[0]?.playlistReview?.duplicatePolicy).toBeUndefined();
  });

  it("preserves an inspected unknown duplicate status after materialization", async () => {
    const inspectedItems = [
      itemWith("occ-unknown", 1, "youtube:video:unknown", {
        duplicateStatus: "unknown",
      }),
    ];
    apiMocks.getPlaylistPreflight.mockResolvedValue(readySummary("PL-alpha", 1));
    apiMocks.listPlaylistPreflightItems.mockResolvedValue(
      page("PL-alpha", inspectedItems),
    );
    apiMocks.materializePlaylistPreflight.mockResolvedValue(
      materialization("PL-alpha", inspectedItems),
    );
    render(<Harness />);

    await addLines(PLAYLIST_A);
    await userEvent.click(
      await screen.findByRole("button", { name: "Add 1 video" }),
    );

    await waitFor(() => expect(currentState?.queueItems).toHaveLength(1));
    expect(currentState?.queueItems[0]).toMatchObject({
      id: "occ-unknown",
      playlist: { duplicateStatus: "unknown" },
    });
  });

  it.each([
    [
      "request failure",
      new PlaylistIngestPublicError("server_unreachable"),
      null,
    ],
    ["response mismatch", null, ["occ-first"]],
  ] as const)(
    "keeps the candidate and adds zero rows on materialization %s",
    async (_case, requestError, returnedOccurrenceIds) => {
      const inspectedItems = [
        itemWith("occ-first", 1, "youtube:video:first"),
        itemWith("occ-second", 2, "youtube:video:second"),
      ];
      apiMocks.getPlaylistPreflight.mockResolvedValue(
        readySummary("PL-alpha", inspectedItems.length),
      );
      apiMocks.listPlaylistPreflightItems.mockResolvedValue(
        page("PL-alpha", inspectedItems),
      );
      if (requestError) {
        apiMocks.materializePlaylistPreflight.mockRejectedValue(requestError);
      } else {
        apiMocks.materializePlaylistPreflight.mockResolvedValue(
          materialization(
            "PL-alpha",
            inspectedItems.filter((entry) =>
              returnedOccurrenceIds?.includes(entry.occurrenceId),
            ),
          ),
        );
      }
      render(<Harness />);

      await addLines(PLAYLIST_A);
      await userEvent.click(
        await screen.findByRole("button", { name: "Add 2 videos" }),
      );

      await waitFor(() => {
        expect(
          screen.getByText(
            requestError
              ? "The server could not be reached. Try again."
              : "The selected playlist items are no longer valid.",
          ),
        ).toBeInTheDocument();
      });
      expect(currentState?.queueItems).toEqual([]);
      expect(screen.getByText("Inspection ready")).toBeInTheDocument();
      expect(
        screen.getByRole("button", { name: "Add 2 videos" }),
      ).toBeEnabled();
    },
  );

  it("fails atomically when a returned occurrence id collides with an existing queue row", async () => {
    const inspectedItems = [itemWith("occ-first", 1, "youtube:video:first")];
    apiMocks.getPlaylistPreflight.mockResolvedValue(
      readySummary("PL-alpha", inspectedItems.length),
    );
    apiMocks.listPlaylistPreflightItems.mockResolvedValue(
      page("PL-alpha", inspectedItems),
    );
    apiMocks.materializePlaylistPreflight.mockResolvedValue(
      materialization("PL-alpha", inspectedItems),
    );
    render(
      <Harness
        initialState={{
          queueItems: [
            {
              id: "occ-first",
              sourceRef: {
                kind: "direct_url",
                occurrenceId: "occ-first",
                url: ORDINARY_URL,
              },
              url: ORDINARY_URL,
              detectedType: "web",
              icon: "Globe",
              fileSize: 0,
              validation: { valid: true },
            },
          ],
        }}
      />,
    );

    await addLines(PLAYLIST_A);
    await userEvent.click(
      await screen.findByRole("button", { name: "Add 1 video" }),
    );

    expect(
      await screen.findByText(
        "The selected playlist items are no longer valid.",
      ),
    ).toBeInTheDocument();
    expect(currentState?.queueItems).toHaveLength(1);
    expect(currentState?.queueItems[0]?.url).toBe(ORDINARY_URL);
    expect(screen.getByText("Inspection ready")).toBeInTheDocument();
  });

  it("ignores a second materialization submit while the first request is pending", async () => {
    const inspectedItems = [
      itemWith("occ-pending", 1, "youtube:video:pending"),
    ];
    const pendingMaterialization = deferred<PlaylistMaterialization>();
    apiMocks.getPlaylistPreflight.mockResolvedValue(
      readySummary("PL-alpha", inspectedItems.length),
    );
    apiMocks.listPlaylistPreflightItems.mockResolvedValue(
      page("PL-alpha", inspectedItems),
    );
    apiMocks.materializePlaylistPreflight.mockReturnValue(
      pendingMaterialization.promise,
    );
    render(<Harness />);

    await addLines(PLAYLIST_A);
    const addButton = await screen.findByRole("button", {
      name: "Add 1 video",
    });
    fireEvent.click(addButton);
    fireEvent.click(addButton);

    expect(apiMocks.materializePlaylistPreflight).toHaveBeenCalledTimes(1);
    await waitFor(() => expect(addButton).toBeDisabled());

    pendingMaterialization.resolve(materialization("PL-alpha", inspectedItems));
    await waitFor(() => expect(currentState?.queueItems).toHaveLength(1));
  });

  it("does not lose an ordinary add that commits with materialization", async () => {
    const inspectedItems = [itemWith("occ-race-add", 1, "youtube:video:race-add")];
    const pendingMaterialization = deferred<PlaylistMaterialization>();
    apiMocks.getPlaylistPreflight.mockResolvedValue(readySummary("PL-alpha", 1));
    apiMocks.listPlaylistPreflightItems.mockResolvedValue(page("PL-alpha", inspectedItems));
    apiMocks.materializePlaylistPreflight.mockReturnValue(pendingMaterialization.promise);
    render(<Harness />);

    await addLines(PLAYLIST_A);
    const materializeButton = await screen.findByRole("button", { name: "Add 1 video" });
    fireEvent.change(screen.getByRole("textbox", { name: "URL input area" }), {
      target: { value: ORDINARY_URL },
    });

    await act(async () => {
      materializeButton.click();
      screen.getByRole("button", { name: "Add URLs to queue" }).click();
      pendingMaterialization.resolve(materialization("PL-alpha", inspectedItems));
      await pendingMaterialization.promise;
    });

    await waitFor(() =>
      expect(currentState?.queueItems.map((row) => row.id)).toEqual([
        expect.any(String),
        "occ-race-add",
      ]),
    );
    expect(currentState?.queueItems[0]?.url).toBe(ORDINARY_URL);
  });

  it("keeps the candidate when an in-updater occurrence collision fails closed", async () => {
    const inspectedItems = [
      itemWith("occ-race-collision", 1, "youtube:video:race-collision"),
    ];
    const pendingMaterialization = deferred<PlaylistMaterialization>();
    const uuidSpy = vi
      .spyOn(globalThis.crypto, "randomUUID")
      .mockReturnValue("occ-race-collision");
    apiMocks.getPlaylistPreflight.mockResolvedValue(readySummary("PL-alpha", 1));
    apiMocks.listPlaylistPreflightItems.mockResolvedValue(page("PL-alpha", inspectedItems));
    apiMocks.materializePlaylistPreflight.mockReturnValue(pendingMaterialization.promise);
    render(<Harness />);

    try {
      await addLines(PLAYLIST_A);
      const materializeButton = await screen.findByRole("button", { name: "Add 1 video" });
      fireEvent.change(screen.getByRole("textbox", { name: "URL input area" }), {
        target: { value: ORDINARY_URL },
      });

      await act(async () => {
        materializeButton.click();
        screen.getByRole("button", { name: "Add URLs to queue" }).click();
        pendingMaterialization.resolve(materialization("PL-alpha", inspectedItems));
        await pendingMaterialization.promise;
      });

      expect(currentState?.queueItems).toHaveLength(1);
      expect(currentState?.queueItems[0]).toMatchObject({
        id: "occ-race-collision",
        sourceRef: { kind: "direct_url" },
      });
      expect(await screen.findByText("Inspection ready")).toBeInTheDocument();
      expect(
        screen.getByText("The selected playlist items are no longer valid."),
      ).toBeInTheDocument();
    } finally {
      uuidSpy.mockRestore();
    }
  });

  it.each([
    ["remove", false],
    ["clear", true],
  ])("does not resurrect rows %sd while materialization is pending", async (_action, clearAll) => {
    const inspectedItems = [itemWith("occ-race-delete", 1, "youtube:video:race-delete")];
    const pendingMaterialization = deferred<PlaylistMaterialization>();
    apiMocks.getPlaylistPreflight.mockResolvedValue(readySummary("PL-alpha", 1));
    apiMocks.listPlaylistPreflightItems.mockResolvedValue(page("PL-alpha", inspectedItems));
    apiMocks.materializePlaylistPreflight.mockReturnValue(pendingMaterialization.promise);
    render(
      <Harness
        initialState={{
          queueItems: [
            {
              id: "ordinary-before-race",
              sourceRef: {
                kind: "direct_url",
                occurrenceId: "ordinary-before-race",
                url: ORDINARY_URL,
              },
              url: ORDINARY_URL,
              detectedType: "document",
              icon: "file-text",
              fileSize: 0,
              validation: { valid: true },
            },
          ],
        }}
      />,
    );

    await addLines(PLAYLIST_A);
    const materializeButton = await screen.findByRole("button", { name: "Add 1 video" });
    const deleteButton = clearAll
      ? screen.getByRole("button", { name: "Remove all items from queue" })
      : screen.getByRole("button", { name: "Remove this item from queue" });

    await act(async () => {
      materializeButton.click();
      deleteButton.click();
      pendingMaterialization.resolve(materialization("PL-alpha", inspectedItems));
      await pendingMaterialization.promise;
    });

    await waitFor(() =>
      expect(currentState?.queueItems.map((row) => row.id)).toEqual(["occ-race-delete"]),
    );
  });

  it("locks ready-card refresh, removal, and selection mutations while materializing", async () => {
    const inspectedItems = [
      itemWith("occ-pending-lock", 1, "youtube:video:pending-lock"),
    ];
    const pendingMaterialization = deferred<PlaylistMaterialization>();
    apiMocks.getPlaylistPreflight.mockResolvedValue(readySummary("PL-alpha", 1));
    apiMocks.listPlaylistPreflightItems.mockResolvedValue(
      page("PL-alpha", inspectedItems),
    );
    apiMocks.materializePlaylistPreflight.mockReturnValue(
      pendingMaterialization.promise,
    );
    render(<Harness />);

    await addLines(PLAYLIST_A);
    fireEvent.click(
      await screen.findByRole("button", { name: "Add 1 video" }),
    );

    const refresh = screen.getByRole("button", {
      name: "Refresh playlist inspection",
    });
    const remove = screen.getByRole("button", {
      name: `Remove playlist inspection for ${PLAYLIST_A}`,
    });
    const selectAll = screen.getByRole("button", { name: "Select all" });
    const selectNone = screen.getByRole("button", { name: "Select none" });
    const selectNew = screen.getByRole("button", { name: "Select new" });
    const rowSelection = screen.getByRole("checkbox", {
      name: "Select playlist item 1: occ-pending-lock",
    });
    await waitFor(() => {
      expect(refresh).toBeDisabled();
      expect(remove).toBeDisabled();
      expect(selectAll).toBeDisabled();
      expect(selectNone).toBeDisabled();
      expect(selectNew).toBeDisabled();
      expect(rowSelection).toBeDisabled();
    });

    fireEvent.click(refresh);
    fireEvent.click(remove);
    fireEvent.click(selectNone);
    fireEvent.click(rowSelection);
    expect(apiMocks.createPlaylistPreflight).toHaveBeenCalledTimes(1);
    expect(rowSelection).toBeChecked();
    expect(screen.getByText("Inspection ready")).toBeInTheDocument();

    pendingMaterialization.resolve(materialization("PL-alpha", inspectedItems));
    await waitFor(() => expect(currentState?.queueItems).toHaveLength(1));
  });

  it("keeps the candidate locked until the queued materialization commit is verified", async () => {
    const inspectedItems = [
      itemWith("occ-commit-lock", 1, "youtube:video:commit-lock"),
    ];
    const pendingMaterialization = deferred<PlaylistMaterialization>();
    apiMocks.getPlaylistPreflight.mockResolvedValue(readySummary("PL-alpha", 1));
    apiMocks.listPlaylistPreflightItems.mockResolvedValue(
      page("PL-alpha", inspectedItems),
    );
    apiMocks.materializePlaylistPreflight.mockReturnValue(
      pendingMaterialization.promise,
    );
    render(<Harness />);

    await addLines(PLAYLIST_A);
    const addButton = await screen.findByRole("button", { name: "Add 1 video" });
    fireEvent.click(addButton);
    await waitFor(() => expect(addButton).toBeDisabled());

    const unlockedTransitions: MutationRecord[] = [];
    const observer = new MutationObserver((records) => {
      for (const record of records) {
        if (
          record.attributeName === "disabled" &&
          !(record.target as HTMLButtonElement).disabled
        ) {
          unlockedTransitions.push(record);
        }
      }
    });
    observer.observe(addButton, { attributes: true, attributeFilter: ["disabled"] });

    pendingMaterialization.resolve(materialization("PL-alpha", inspectedItems));
    await waitFor(() => expect(currentState?.queueItems).toHaveLength(1));
    await waitFor(() =>
      expect(screen.queryByRole("button", { name: "Add 1 video" })).not.toBeInTheDocument(),
    );
    observer.disconnect();

    expect(unlockedTransitions).toHaveLength(0);
  });

  it("retries materialization after an error without duplicating queue rows", async () => {
    const inspectedItems = [itemWith("occ-retry", 1, "youtube:video:retry")];
    apiMocks.getPlaylistPreflight.mockResolvedValue(
      readySummary("PL-alpha", inspectedItems.length),
    );
    apiMocks.listPlaylistPreflightItems.mockResolvedValue(
      page("PL-alpha", inspectedItems),
    );
    apiMocks.materializePlaylistPreflight
      .mockRejectedValueOnce(
        new PlaylistIngestPublicError("server_unreachable"),
      )
      .mockResolvedValueOnce(materialization("PL-alpha", inspectedItems));
    render(<Harness />);

    await addLines(PLAYLIST_A);
    await userEvent.click(
      await screen.findByRole("button", { name: "Add 1 video" }),
    );
    expect(
      await screen.findByText("The server could not be reached. Try again."),
    ).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Add 1 video" }));

    await waitFor(() => expect(currentState?.queueItems).toHaveLength(1));
    expect(apiMocks.materializePlaylistPreflight).toHaveBeenCalledTimes(2);
    expect(currentState?.queueItems[0]?.id).toBe("occ-retry");
  });

  it("routes Enter through the same candidate inspection handler", async () => {
    render(<Harness />);

    await addLines(`${ORDINARY_URL}\n${PLAYLIST_A}`, true);

    await waitFor(() => {
      expect(apiMocks.createPlaylistPreflight).toHaveBeenCalledWith(
        { url: PLAYLIST_A },
        expect.objectContaining({ signal: expect.any(AbortSignal) }),
      );
    });
    expect(currentState?.queueItems.map((row) => row.url)).toEqual([
      ORDINARY_URL,
    ]);
    expect(currentState?.queueItems.some((row) => row.url === PLAYLIST_A)).toBe(
      false,
    );
  });

  it("uses the established Quick Ingest cadence between pending polls", async () => {
    vi.useFakeTimers();
    apiMocks.getPlaylistPreflight
      .mockResolvedValueOnce(summary("PL-alpha", "pending"))
      .mockResolvedValueOnce(summary("PL-alpha", "ready"));
    render(<Harness />);

    const input = screen.getByRole("textbox", { name: "URL input area" });
    fireEvent.change(input, { target: { value: PLAYLIST_A } });
    fireEvent.click(screen.getByRole("button", { name: "Add URLs to queue" }));
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(apiMocks.getPlaylistPreflight).toHaveBeenCalledTimes(1);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1_199);
    });
    expect(apiMocks.getPlaylistPreflight).toHaveBeenCalledTimes(1);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1);
    });
    expect(apiMocks.getPlaylistPreflight).toHaveBeenCalledTimes(2);
  });

  it("fails closed when playlist ingest v2 is unavailable", async () => {
    capabilityHarness.hasMediaPlaylistIngestV2 = false;
    render(<Harness />);

    await addLines(`${ORDINARY_URL}\n${PLAYLIST_A}`);

    expect(apiMocks.createPlaylistPreflight).not.toHaveBeenCalled();
    expect(
      screen.getByText(
        "Playlist inspection is unavailable on this server. Update the server or remove this playlist.",
      ),
    ).toBeInTheDocument();
    expect(currentState?.queueItems.map((row) => row.url)).toEqual([
      ORDINARY_URL,
    ]);
    expectProceedBlocked();
  });

  it("waits for capability loading before inspecting a candidate", async () => {
    capabilityHarness.hasMediaPlaylistIngestV2 = null;
    capabilityHarness.loading = true;
    const view = render(<Harness />);

    await addLines(PLAYLIST_A);

    expect(apiMocks.createPlaylistPreflight).not.toHaveBeenCalled();
    expect(currentState?.queueItems).toEqual([]);
    expect(screen.getByText("Inspecting playlist")).toBeInTheDocument();

    capabilityHarness.hasMediaPlaylistIngestV2 = true;
    capabilityHarness.loading = false;
    view.rerender(<Harness />);

    await waitFor(() => {
      expect(apiMocks.createPlaylistPreflight).toHaveBeenCalledTimes(1);
    });
    expect(currentState?.queueItems).toEqual([]);
  });

  it("requeues an unavailable candidate when capability refresh enables inspection", async () => {
    capabilityHarness.hasMediaPlaylistIngestV2 = false;
    const view = render(<Harness />);

    await addLines(PLAYLIST_A);
    expect(apiMocks.createPlaylistPreflight).not.toHaveBeenCalled();

    capabilityHarness.hasMediaPlaylistIngestV2 = true;
    view.rerender(<Harness />);

    await waitFor(() => {
      expect(apiMocks.createPlaylistPreflight).toHaveBeenCalledTimes(1);
    });
    expect(currentState?.queueItems).toEqual([]);
  });

  it("waits for disabled-resource cleanup before a fast capability re-enable", async () => {
    const oldPoll = deferred<PlaylistPreflightSummary>();
    const cleanup = deferred<void>();
    apiMocks.getPlaylistPreflight.mockReturnValueOnce(oldPoll.promise);
    apiMocks.cancelPlaylistPreflight.mockReturnValueOnce(cleanup.promise);
    const view = render(<Harness />);

    await addLines(PLAYLIST_A);
    await waitFor(() => {
      expect(apiMocks.getPlaylistPreflight).toHaveBeenCalledTimes(1);
    });

    capabilityHarness.hasMediaPlaylistIngestV2 = false;
    view.rerender(<Harness />);
    await waitFor(() => {
      expect(apiMocks.cancelPlaylistPreflight).toHaveBeenCalledWith("PL-alpha");
    });

    capabilityHarness.hasMediaPlaylistIngestV2 = true;
    view.rerender(<Harness />);
    oldPoll.resolve(summary("PL-alpha", "running"));
    await act(async () => {
      await oldPoll.promise;
    });
    expect(apiMocks.createPlaylistPreflight).toHaveBeenCalledTimes(1);

    cleanup.resolve();
    await waitFor(() => {
      expect(apiMocks.createPlaylistPreflight).toHaveBeenCalledTimes(2);
    });
  });

  it.each(["create", "poll"] as const)(
    "shows a safe retry state when %s fails",
    async (failurePoint) => {
      if (failurePoint === "create") {
        apiMocks.createPlaylistPreflight.mockRejectedValueOnce(
          new Error("secret extractor details"),
        );
      } else {
        apiMocks.getPlaylistPreflight.mockRejectedValueOnce(
          new Error("secret poll response"),
        );
      }
      render(<Harness />);

      await addLines(PLAYLIST_A);

      await waitFor(() => {
        expect(
          screen.getByText("Playlist ingestion is unavailable. Try again."),
        ).toBeInTheDocument();
      });
      expect(screen.queryByText(/secret/i)).not.toBeInTheDocument();
      expect(
        screen.getByRole("button", {
          name: `Retry playlist inspection for ${PLAYLIST_A}`,
        }),
      ).toBeEnabled();
      expect(currentState?.queueItems).toEqual([]);
      expectProceedBlocked();
    },
  );

  it("shows a safe nonretryable typed error without leaking raw details", async () => {
    const error = Object.assign(
      new PlaylistIngestPublicError("playlist_private_or_auth_required"),
      { cause: new Error("raw provider token secret") },
    );
    apiMocks.createPlaylistPreflight.mockRejectedValueOnce(error);
    render(<Harness />);

    await addLines(PLAYLIST_A);

    await waitFor(() => {
      expect(
        screen.getByText(
          "This playlist is private or requires authentication.",
        ),
      ).toBeInTheDocument();
    });
    expect(
      screen.queryByText(/raw provider token secret/i),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", {
        name: `Retry playlist inspection for ${PLAYLIST_A}`,
      }),
    ).not.toBeInTheDocument();
  });

  it("uses terminal retryable error info for truthful copy and retry", async () => {
    apiMocks.getPlaylistPreflight.mockResolvedValueOnce(
      summary("PL-alpha", "blocked", {
        code: "preflight_busy",
        message: "The server is busy inspecting playlists. Try again shortly.",
        retryable: true,
      }),
    );
    render(<Harness />);

    await addLines(PLAYLIST_A);

    await waitFor(() => {
      expect(
        screen.getByText(
          "The server is busy inspecting playlists. Try again shortly.",
        ),
      ).toBeInTheDocument();
    });
    expect(
      screen.getByRole("button", {
        name: `Retry playlist inspection for ${PLAYLIST_A}`,
      }),
    ).toBeEnabled();
  });

  it("waits for failed-resource cleanup before retrying", async () => {
    const cleanup = deferred<void>();
    apiMocks.getPlaylistPreflight.mockRejectedValueOnce(
      new Error("poll failed"),
    );
    apiMocks.cancelPlaylistPreflight.mockReturnValueOnce(cleanup.promise);
    render(<Harness />);

    await addLines(PLAYLIST_A);
    await waitFor(() => {
      expect(
        screen.getByText("Playlist ingestion is unavailable. Try again."),
      ).toBeInTheDocument();
    });

    await userEvent.click(
      screen.getByRole("button", {
        name: `Retry playlist inspection for ${PLAYLIST_A}`,
      }),
    );

    expect(apiMocks.cancelPlaylistPreflight).toHaveBeenCalledWith("PL-alpha");
    expect(apiMocks.createPlaylistPreflight).toHaveBeenCalledTimes(1);

    cleanup.resolve();
    await waitFor(() => {
      expect(apiMocks.createPlaylistPreflight).toHaveBeenCalledTimes(2);
    });
  });

  it.each([
    ["blocked", "Playlist inspection was blocked. Remove it or try again."],
    ["expired", "Playlist inspection expired. Try again."],
  ] as const)("keeps %s candidates blocked", async (status, message) => {
    apiMocks.getPlaylistPreflight.mockResolvedValueOnce(
      summary("PL-alpha", status),
    );
    render(<Harness />);

    await addLines(PLAYLIST_A);

    await waitFor(() => {
      expect(screen.getByText(message)).toBeInTheDocument();
    });
    expect(apiMocks.listPlaylistPreflightItems).not.toHaveBeenCalled();
    expectProceedBlocked();
  });

  it("inspects multiple candidates with a peak concurrency of two", async () => {
    let active = 0;
    let peak = 0;
    const creates = [
      deferred<PlaylistPreflightAccepted>(),
      deferred<PlaylistPreflightAccepted>(),
      deferred<PlaylistPreflightAccepted>(),
    ];
    apiMocks.createPlaylistPreflight.mockImplementation(() => {
      const next =
        creates[apiMocks.createPlaylistPreflight.mock.calls.length - 1];
      active += 1;
      peak = Math.max(peak, active);
      return next.promise.finally(() => {
        active -= 1;
      });
    });
    render(<Harness />);

    await addLines(`${PLAYLIST_A}\n${PLAYLIST_B}\n${PLAYLIST_C}`);

    await waitFor(() => {
      expect(apiMocks.createPlaylistPreflight).toHaveBeenCalledTimes(2);
    });
    expect(peak).toBe(2);

    await act(async () => {
      creates[0].resolve(accepted("PL-alpha"));
      await creates[0].promise;
    });

    await waitFor(() => {
      expect(apiMocks.createPlaylistPreflight).toHaveBeenCalledTimes(3);
    });
    expect(peak).toBe(2);

    await act(async () => {
      creates[1].resolve(accepted("PL-beta"));
      creates[2].resolve(accepted("PL-gamma"));
      await Promise.all([creates[1].promise, creates[2].promise]);
    });
  });

  it("deduplicates repeated candidate lines into one inspection resource", async () => {
    render(<Harness />);

    await addLines(`${PLAYLIST_A}\n  ${PLAYLIST_A}  \n${PLAYLIST_A}`);

    await waitFor(() => {
      expect(apiMocks.createPlaylistPreflight).toHaveBeenCalledTimes(1);
    });
    expect(screen.getAllByText(PLAYLIST_A)).toHaveLength(1);
  });

  it("does not strand an immediate cancel and retry behind its active generation", async () => {
    const firstCreate = deferred<PlaylistPreflightAccepted>();
    apiMocks.createPlaylistPreflight.mockReturnValueOnce(firstCreate.promise);
    render(<Harness />);

    await addLines(PLAYLIST_A);
    await waitFor(() => {
      expect(apiMocks.createPlaylistPreflight).toHaveBeenCalledTimes(1);
    });
    await userEvent.click(
      screen.getByRole("button", {
        name: `Cancel playlist inspection for ${PLAYLIST_A}`,
      }),
    );
    await userEvent.click(
      screen.getByRole("button", {
        name: `Retry playlist inspection for ${PLAYLIST_A}`,
      }),
    );

    firstCreate.resolve(accepted("PL-old"));
    await waitFor(() => {
      expect(apiMocks.cancelPlaylistPreflight).toHaveBeenCalledWith("PL-old");
      expect(apiMocks.createPlaylistPreflight).toHaveBeenCalledTimes(2);
    });
  });

  it("cancels polling and the server resource without surfacing AbortError", async () => {
    const pollStarted = deferred<void>();
    let pollSignal: AbortSignal | null = null;
    apiMocks.getPlaylistPreflight.mockImplementation(
      (_preflightId: string, options: { signal: AbortSignal }) =>
        new Promise((_resolve, reject) => {
          pollSignal = options.signal;
          pollStarted.resolve();
          options.signal.addEventListener("abort", () => {
            reject(new DOMException("cancelled", "AbortError"));
          });
        }),
    );
    render(<Harness />);

    await addLines(PLAYLIST_A);
    await pollStarted.promise;
    await userEvent.click(
      screen.getByRole("button", {
        name: `Cancel playlist inspection for ${PLAYLIST_A}`,
      }),
    );

    expect(pollSignal?.aborted).toBe(true);
    expect(apiMocks.cancelPlaylistPreflight).toHaveBeenCalledWith("PL-alpha");
    expect(
      screen.getByText("Playlist inspection was cancelled."),
    ).toBeInTheDocument();
    expect(
      screen.queryByText(/AbortError|DOMException/i),
    ).not.toBeInTheDocument();
    expectProceedBlocked();
  });

  it("aborts in-flight inspection silently on unmount", async () => {
    const createStarted = deferred<void>();
    let createSignal: AbortSignal | null = null;
    const consoleError = vi
      .spyOn(console, "error")
      .mockImplementation(() => {});
    apiMocks.createPlaylistPreflight.mockImplementation(
      (_payload: unknown, options: { signal: AbortSignal }) =>
        new Promise((_resolve, reject) => {
          createSignal = options.signal;
          createStarted.resolve();
          options.signal.addEventListener("abort", () => {
            reject(new DOMException("unmounted", "AbortError"));
          });
        }),
    );
    const view = render(<Harness />);

    await addLines(PLAYLIST_A);
    await createStarted.promise;
    view.unmount();

    expect(createSignal?.aborted).toBe(true);
    expect(consoleError).not.toHaveBeenCalled();
  });

  it("cancels a server resource accepted after local cancellation", async () => {
    const create = deferred<PlaylistPreflightAccepted>();
    apiMocks.createPlaylistPreflight.mockReturnValueOnce(create.promise);
    render(<Harness />);

    await addLines(PLAYLIST_A);
    await waitFor(() => {
      expect(apiMocks.createPlaylistPreflight).toHaveBeenCalledTimes(1);
    });
    await userEvent.click(
      screen.getByRole("button", {
        name: `Cancel playlist inspection for ${PLAYLIST_A}`,
      }),
    );
    create.resolve(accepted("PL-alpha"));

    await waitFor(() => {
      expect(apiMocks.cancelPlaylistPreflight).toHaveBeenCalledWith("PL-alpha");
    });
    expect(
      screen.getByText("Playlist inspection was cancelled."),
    ).toBeInTheDocument();
  });

  it("can inspect after React Strict Mode replays effect cleanup", async () => {
    render(
      <React.StrictMode>
        <Harness />
      </React.StrictMode>,
    );

    await addLines(PLAYLIST_A);

    await waitFor(() => {
      expect(apiMocks.createPlaylistPreflight).toHaveBeenCalledTimes(1);
    });
    expect(currentState?.queueItems).toEqual([]);
  });

  it("auto-starts a typed extension seed and clears it exactly once", async () => {
    const onStateChange = vi.fn();
    render(
      <Harness
        initialState={{
          playlistPreflightSeed: {
            source: "extension_active_tab",
            action: "playlist_preflight",
            url: PLAYLIST_A,
            sourceKind: "youtube_playlist",
          },
        }}
        onStateChange={onStateChange}
      />,
    );

    await waitFor(() => {
      expect(apiMocks.createPlaylistPreflight).toHaveBeenCalledTimes(1);
    });
    expect(apiMocks.createPlaylistPreflight).toHaveBeenCalledWith(
      { url: PLAYLIST_A },
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    );
    expect(currentState?.playlistPreflightSeed).toBeNull();
    expect(
      onStateChange.mock.calls.filter(
        ([state]: [IngestWizardState]) => state.playlistPreflightSeed === null,
      ),
    ).toHaveLength(1);
    expect(currentState?.queueItems).toEqual([]);
  });

  it("auto-starts a typed extension seed once after Strict Mode lifecycle replay", async () => {
    const onStateChange = vi.fn();
    render(
      <React.StrictMode>
        <Harness
          initialState={{
            playlistPreflightSeed: {
              source: "extension_active_tab",
              action: "playlist_preflight",
              url: PLAYLIST_A,
              sourceKind: "youtube_playlist",
            },
          }}
          onStateChange={onStateChange}
        />
      </React.StrictMode>,
    );

    await waitFor(() => {
      expect(screen.getByText("Inspection ready")).toBeInTheDocument();
    });
    expect(apiMocks.createPlaylistPreflight).toHaveBeenCalledTimes(1);
    expect(currentState?.playlistPreflightSeed).toBeNull();
    expect(
      onStateChange.mock.calls.filter(
        ([state]: [IngestWizardState]) => state.playlistPreflightSeed === null,
      ),
    ).toHaveLength(1);
    expect(currentState?.queueItems).toEqual([]);
  });

  it("stays inspecting until every opaque-cursor page is loaded atomically", async () => {
    const secondPage = deferred<PlaylistPreflightItemsPage>();
    apiMocks.getPlaylistPreflight.mockResolvedValue(
      readySummary("PL-alpha", 2),
    );
    apiMocks.listPlaylistPreflightItems
      .mockResolvedValueOnce(
        page(
          "PL-alpha",
          [itemWith("occ-first", 1, "youtube:video:first")],
          "opaque:+/next==",
        ),
      )
      .mockReturnValueOnce(secondPage.promise);
    render(<Harness />);

    await addLines(PLAYLIST_A);

    await waitFor(() => {
      expect(apiMocks.listPlaylistPreflightItems).toHaveBeenCalledTimes(2);
    });
    expect(screen.getByText("Inspecting playlist")).toBeInTheDocument();
    expect(screen.queryByText("Inspection ready")).not.toBeInTheDocument();
    expect(
      screen.queryByRole("checkbox", {
        name: "Select playlist item 1: occ-first",
      }),
    ).not.toBeInTheDocument();
    const firstCall = apiMocks.listPlaylistPreflightItems.mock.calls[0];
    const secondCall = apiMocks.listPlaylistPreflightItems.mock.calls[1];
    expect(firstCall?.[1]).toEqual({ limit: 100 });
    expect(secondCall?.[1]).toEqual({ cursor: "opaque:+/next==", limit: 100 });
    expect(secondCall?.[2].signal).toBe(firstCall?.[2].signal);

    secondPage.resolve(
      page("PL-alpha", [itemWith("occ-second", 2, "youtube:video:second")]),
    );

    await waitFor(() => {
      expect(screen.getByText("Inspection ready")).toBeInTheDocument();
    });
    expect(
      screen.getByRole("checkbox", {
        name: "Select playlist item 1: occ-first",
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("checkbox", {
        name: "Select playlist item 2: occ-second",
      }),
    ).toBeInTheDocument();
    expectProceedBlocked();
  });

  it("uses the latest queued URLs when atomic paging publishes defaults", async () => {
    const directUrl = "https://youtu.be/mixed-shared?t=5";
    const secondPage = deferred<PlaylistPreflightItemsPage>();
    apiMocks.getPlaylistPreflight.mockResolvedValue(
      readySummary("PL-alpha", 2),
    );
    apiMocks.listPlaylistPreflightItems
      .mockResolvedValueOnce(
        page(
          "PL-alpha",
          [itemWith("occ-first", 1, "youtube:video:first")],
          "next-page",
        ),
      )
      .mockReturnValueOnce(secondPage.promise);
    render(<Harness />);

    await addLines(PLAYLIST_A);
    await waitFor(() => {
      expect(apiMocks.listPlaylistPreflightItems).toHaveBeenCalledTimes(2);
    });

    await addLines(directUrl);
    expect(currentState?.queueItems.map((row) => row.url)).toContain(directUrl);

    secondPage.resolve(
      page("PL-alpha", [
        itemWith("occ-matching-queue", 2, "youtube:video:mixed-shared", {
          sourceUrl: directUrl,
        }),
      ]),
    );

    const matching = await screen.findByRole("checkbox", {
      name: "Select playlist item 2: occ-matching-queue",
    });
    expect(matching).not.toBeChecked();
  });

  it("recomputes ready defaults from queue changes while explicit selection wins", async () => {
    const directUrl = "https://youtu.be/explicit-shared?t=5";
    apiMocks.getPlaylistPreflight.mockResolvedValue(
      readySummary("PL-alpha", 1),
    );
    apiMocks.listPlaylistPreflightItems.mockResolvedValue(
      page("PL-alpha", [
        itemWith("occ-explicit", 1, "youtube:video:explicit-shared", {
          sourceUrl: directUrl,
          selectedByDefault: false,
        }),
      ]),
    );
    render(<Harness />);

    await addLines(PLAYLIST_A);
    const checkbox = await screen.findByRole("checkbox", {
      name: "Select playlist item 1: occ-explicit",
    });
    expect(checkbox).not.toBeChecked();
    await userEvent.click(checkbox);
    expect(checkbox).toBeChecked();

    await addLines(directUrl);

    await waitFor(() => expect(checkbox).toBeChecked());
    expect(currentState?.queueItems.map((row) => row.url)).toContain(directUrl);
  });

  it("cancels the shared paging signal without exposing a partial snapshot", async () => {
    let pagingSignal: AbortSignal | null = null;
    apiMocks.getPlaylistPreflight.mockResolvedValue(
      readySummary("PL-alpha", 2),
    );
    apiMocks.listPlaylistPreflightItems
      .mockResolvedValueOnce(
        page(
          "PL-alpha",
          [itemWith("occ-first", 1, "youtube:video:first")],
          "next",
        ),
      )
      .mockImplementationOnce(
        async (
          _preflightId: string,
          _params: unknown,
          options: { signal: AbortSignal },
        ) => {
          pagingSignal = options.signal;
          return new Promise<PlaylistPreflightItemsPage>((_resolve, reject) => {
            options.signal.addEventListener(
              "abort",
              () => reject(new DOMException("Aborted", "AbortError")),
              { once: true },
            );
          });
        },
      );
    render(<Harness />);

    await addLines(PLAYLIST_A);
    await waitFor(() => expect(pagingSignal).not.toBeNull());

    await userEvent.click(
      screen.getByRole("button", {
        name: `Cancel playlist inspection for ${PLAYLIST_A}`,
      }),
    );

    expect((pagingSignal as unknown as AbortSignal).aborted).toBe(true);
    expect(
      screen.getByText("Playlist inspection was cancelled."),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("checkbox", {
        name: "Select playlist item 1: occ-first",
      }),
    ).not.toBeInTheDocument();
  });

  it("surfaces incomplete paging safely and never publishes partial rows", async () => {
    apiMocks.getPlaylistPreflight.mockResolvedValue(
      readySummary("PL-alpha", 2),
    );
    apiMocks.listPlaylistPreflightItems
      .mockResolvedValueOnce(
        page(
          "PL-alpha",
          [itemWith("occ-first", 1, "youtube:video:first")],
          "repeat",
        ),
      )
      .mockResolvedValueOnce(
        page(
          "PL-alpha",
          [itemWith("occ-second", 2, "youtube:video:second")],
          "repeat",
        ),
      );
    render(<Harness />);

    await addLines(PLAYLIST_A);

    await waitFor(() => {
      expect(
        screen.getByText("Playlist inspection is incomplete. Try again."),
      ).toBeInTheDocument();
    });
    expect(
      screen.getByRole("button", { name: /Retry playlist inspection/i }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("checkbox", {
        name: "Select playlist item 1: occ-first",
      }),
    ).not.toBeInTheDocument();
    expect(apiMocks.listPlaylistPreflightItems).toHaveBeenCalledTimes(2);
  });

  it("drives row duplicate semantics in queue-first then candidate and ordinal order", async () => {
    apiMocks.getPlaylistPreflight.mockImplementation(
      async (preflightId: string) => readySummary(preflightId, 2),
    );
    apiMocks.listPlaylistPreflightItems.mockImplementation(
      async (preflightId: string) => {
        if (preflightId === "PL-alpha") {
          return page(preflightId, [
            itemWith("alpha-queued-repeat", 1, "youtube:video:queued", {
              sourceUrl: "https://youtu.be/queued-repeat?t=20",
            }),
            itemWith("alpha-session-first", 2, "youtube:video:session-repeat"),
          ]);
        }
        return page(preflightId, [
          itemWith("beta-queued-repeat", 1, "youtube:video:queued", {
            sourceUrl: "https://youtu.be/queued-repeat?t=40",
          }),
          itemWith("beta-session-later", 2, "youtube:video:session-repeat"),
        ]);
      },
    );
    render(
      <Harness
        initialState={{
          queueItems: [
            {
              id: "queued-repeat",
              url: "https://youtu.be/queued-repeat?t=5",
              detectedType: "video",
              icon: "Film",
              fileSize: 0,
              validation: { valid: true },
            },
          ],
        }}
      />,
    );

    await addLines(`${PLAYLIST_A}\n${PLAYLIST_B}`);

    await waitFor(() => {
      expect(
        screen.getByRole("checkbox", {
          name: "Select playlist item 2: beta-session-later",
        }),
      ).toBeInTheDocument();
    });
    expect(
      screen.getByRole("checkbox", {
        name: "Select playlist item 1: alpha-queued-repeat",
      }),
    ).not.toBeChecked();
    expect(
      screen.getByRole("checkbox", {
        name: "Select playlist item 1: beta-queued-repeat",
      }),
    ).not.toBeChecked();
    expect(
      screen.getByRole("checkbox", {
        name: "Select playlist item 2: alpha-session-first",
      }),
    ).toBeChecked();
    expect(
      screen.getByRole("checkbox", {
        name: "Select playlist item 2: beta-session-later",
      }),
    ).not.toBeChecked();

    expect(screen.getByText("1 duplicates")).toBeInTheDocument();
    expect(screen.getByText("2 duplicates")).toBeInTheDocument();

    for (const button of screen.getAllByRole("button", {
      name: "Select new",
    })) {
      await userEvent.click(button);
    }
    expect(
      screen.getByRole("checkbox", {
        name: "Select playlist item 1: alpha-queued-repeat",
      }),
    ).not.toBeChecked();
    expect(
      screen.getByRole("checkbox", {
        name: "Select playlist item 2: alpha-session-first",
      }),
    ).toBeChecked();
    expect(
      screen.getByRole("checkbox", {
        name: "Select playlist item 1: beta-queued-repeat",
      }),
    ).not.toBeChecked();
    expect(
      screen.getByRole("checkbox", {
        name: "Select playlist item 2: beta-session-later",
      }),
    ).not.toBeChecked();

    const filters = screen.getAllByRole("combobox", {
      name: "Filter playlist items",
    });
    const lists = screen.getAllByRole("list", { name: "Playlist videos" });
    fireEvent.change(filters[0], { target: { value: "duplicates" } });
    fireEvent.change(filters[1], { target: { value: "duplicates" } });
    expect(
      within(lists[0])
        .getAllByRole("listitem")
        .map((row) => row.getAttribute("data-occurrence-id")),
    ).toEqual(["alpha-queued-repeat"]);
    expect(
      within(lists[1])
        .getAllByRole("listitem")
        .map((row) => row.getAttribute("data-occurrence-id")),
    ).toEqual(["beta-queued-repeat", "beta-session-later"]);
  });

  it("does not let an unavailable occurrence suppress the first eligible repeat", async () => {
    apiMocks.getPlaylistPreflight.mockResolvedValue(
      readySummary("PL-alpha", 2),
    );
    apiMocks.listPlaylistPreflightItems.mockResolvedValue(
      page("PL-alpha", [
        itemWith("unavailable-repeat", 1, "youtube:video:repeat", {
          availability: "needs_auth",
          selectedByDefault: false,
        }),
        itemWith("eligible-repeat", 2, "youtube:video:repeat"),
      ]),
    );
    render(<Harness />);

    await addLines(PLAYLIST_A);

    expect(
      await screen.findByRole("checkbox", {
        name: "Select playlist item 1: unavailable-repeat",
      }),
    ).toBeDisabled();
    expect(
      screen.getByRole("checkbox", {
        name: "Select playlist item 2: eligible-repeat",
      }),
    ).toBeChecked();
    expect(screen.queryByText("1 duplicates")).not.toBeInTheDocument();

    fireEvent.change(
      screen.getByRole("combobox", { name: "Filter playlist items" }),
      { target: { value: "new" } },
    );
    expect(
      within(screen.getByRole("list", { name: "Playlist videos" }))
        .getAllByRole("listitem")
        .map((row) => row.getAttribute("data-occurrence-id")),
    ).toEqual(["eligible-repeat"]);
  });

  it("promotes a later repeated item when the earlier candidate is removed", async () => {
    apiMocks.getPlaylistPreflight.mockImplementation(
      async (preflightId: string) => readySummary(preflightId, 1),
    );
    apiMocks.listPlaylistPreflightItems.mockImplementation(
      async (preflightId: string) =>
        page(preflightId, [
          itemWith(
            preflightId === "PL-alpha" ? "alpha-shared" : "beta-shared",
            1,
            "youtube:video:shared",
          ),
        ]),
    );
    render(<Harness />);

    await addLines(`${PLAYLIST_A}\n${PLAYLIST_B}`);

    const beta = await screen.findByRole("checkbox", {
      name: "Select playlist item 1: beta-shared",
    });
    expect(beta).not.toBeChecked();
    expect(screen.getByText("1 duplicates")).toBeInTheDocument();

    await userEvent.click(
      screen.getByRole("button", {
        name: `Remove playlist inspection for ${PLAYLIST_A}`,
      }),
    );

    await waitFor(() => expect(beta).toBeChecked());
    expect(screen.queryByText("1 duplicates")).not.toBeInTheDocument();
  });

  it("promotes a later repeated item when refreshing the earlier candidate fails", async () => {
    apiMocks.createPlaylistPreflight
      .mockResolvedValueOnce(accepted("PL-alpha"))
      .mockResolvedValueOnce(accepted("PL-beta"))
      .mockRejectedValueOnce(new Error("refresh failed"));
    apiMocks.getPlaylistPreflight.mockImplementation(
      async (preflightId: string) => readySummary(preflightId, 1),
    );
    apiMocks.listPlaylistPreflightItems.mockImplementation(
      async (preflightId: string) =>
        page(preflightId, [
          itemWith(
            preflightId === "PL-alpha" ? "alpha-shared" : "beta-shared",
            1,
            "youtube:video:shared",
          ),
        ]),
    );
    render(<Harness />);

    await addLines(`${PLAYLIST_A}\n${PLAYLIST_B}`);
    const beta = await screen.findByRole("checkbox", {
      name: "Select playlist item 1: beta-shared",
    });
    expect(beta).not.toBeChecked();

    await userEvent.click(
      screen.getAllByRole("button", {
        name: "Refresh playlist inspection",
      })[0],
    );

    await screen.findByText("Playlist ingestion is unavailable. Try again.");
    await waitFor(() => expect(beta).toBeChecked());
    expect(screen.queryByText("1 duplicates")).not.toBeInTheDocument();
  });

  it("recomputes repeats across cancel and retry while preserving an explicit choice", async () => {
    const refreshAttempt = deferred<PlaylistPreflightAccepted>();
    apiMocks.createPlaylistPreflight
      .mockResolvedValueOnce(accepted("PL-alpha"))
      .mockResolvedValueOnce(accepted("PL-beta"))
      .mockReturnValueOnce(refreshAttempt.promise)
      .mockResolvedValueOnce(accepted("PL-alpha-return"));
    apiMocks.getPlaylistPreflight.mockImplementation(
      async (preflightId: string) => readySummary(preflightId, 1),
    );
    apiMocks.listPlaylistPreflightItems.mockImplementation(
      async (preflightId: string) =>
        page(preflightId, [
          itemWith(
            preflightId === "PL-beta" ? "beta-shared" : "alpha-shared",
            1,
            "youtube:video:shared",
          ),
        ]),
    );
    render(<Harness />);

    await addLines(`${PLAYLIST_A}\n${PLAYLIST_B}`);
    const beta = await screen.findByRole("checkbox", {
      name: "Select playlist item 1: beta-shared",
    });
    expect(beta).not.toBeChecked();
    await userEvent.click(beta);

    await userEvent.click(
      screen.getAllByRole("button", {
        name: "Refresh playlist inspection",
      })[0],
    );
    await waitFor(() => {
      expect(apiMocks.createPlaylistPreflight).toHaveBeenCalledTimes(3);
    });
    expect(beta).toBeChecked();
    expect(screen.queryByText("1 duplicates")).not.toBeInTheDocument();

    await userEvent.click(
      screen.getByRole("button", {
        name: `Cancel playlist inspection for ${PLAYLIST_A}`,
      }),
    );
    refreshAttempt.reject(
      Object.assign(new Error("aborted"), { name: "AbortError" }),
    );
    await screen.findByText("Playlist inspection was cancelled.");

    await userEvent.click(
      screen.getByRole("button", {
        name: `Retry playlist inspection for ${PLAYLIST_A}`,
      }),
    );

    const alpha = await screen.findByRole("checkbox", {
      name: "Select playlist item 1: alpha-shared",
    });
    expect(alpha).toBeChecked();
    expect(beta).toBeChecked();
    expect(screen.getByText("1 duplicates")).toBeInTheDocument();
    const lists = screen.getAllByRole("list", { name: "Playlist videos" });
    const filters = screen.getAllByRole("combobox", {
      name: "Filter playlist items",
    });
    fireEvent.change(filters[1], { target: { value: "duplicates" } });
    expect(
      within(lists[1])
        .getAllByRole("listitem")
        .map((row) => row.getAttribute("data-occurrence-id")),
    ).toEqual(["beta-shared"]);
  });

  it("refreshes through a new preflight and reconciles explicit repeated-source choices", async () => {
    apiMocks.createPlaylistPreflight
      .mockResolvedValueOnce(accepted("preflight-old"))
      .mockResolvedValueOnce(accepted("preflight-new"));
    apiMocks.getPlaylistPreflight.mockImplementation(
      async (preflightId: string) => readySummary(preflightId, 2),
    );
    apiMocks.listPlaylistPreflightItems.mockImplementation(
      async (preflightId: string) =>
        preflightId === "preflight-old"
          ? page(preflightId, [
              itemWith("old-repeat-1", 1, "youtube:video:repeat", {
                occurrenceIndexForSource: null,
              }),
              itemWith("old-repeat-2", 2, "youtube:video:repeat", {
                occurrenceIndexForSource: null,
              }),
            ])
          : page(preflightId, [
              itemWith("new-repeat-1", 1, "youtube:video:repeat", {
                occurrenceIndexForSource: null,
              }),
              itemWith("new-repeat-2", 2, "youtube:video:repeat", {
                occurrenceIndexForSource: null,
              }),
            ]),
    );
    render(<Harness />);

    await addLines(PLAYLIST_A);
    const oldSecond = await screen.findByRole("checkbox", {
      name: "Select playlist item 2: old-repeat-2",
    });
    expect(oldSecond).not.toBeChecked();
    await userEvent.click(oldSecond);

    await userEvent.click(
      screen.getByRole("button", { name: "Refresh playlist inspection" }),
    );

    const newSecond = await screen.findByRole("checkbox", {
      name: "Select playlist item 2: new-repeat-2",
    });
    expect(newSecond).toBeChecked();
    expect(apiMocks.createPlaylistPreflight).toHaveBeenCalledTimes(2);
    expect(apiMocks.cancelPlaylistPreflight).toHaveBeenCalledWith(
      "preflight-old",
    );
    expect(
      screen.queryByText(/added, removed, reordered/i),
    ).not.toBeInTheDocument();
    expect(currentState?.queueItems).toEqual([]);
  });

  it("preserves refresh reconciliation through a transient failure and retry", async () => {
    apiMocks.createPlaylistPreflight
      .mockResolvedValueOnce(accepted("preflight-old"))
      .mockRejectedValueOnce(new Error("transient refresh failure"))
      .mockResolvedValueOnce(accepted("preflight-new"));
    apiMocks.getPlaylistPreflight.mockImplementation(
      async (preflightId: string) => readySummary(preflightId, 2),
    );
    apiMocks.listPlaylistPreflightItems.mockImplementation(
      async (preflightId: string) =>
        preflightId === "preflight-old"
          ? page(preflightId, [
              itemWith("old-repeat-1", 1, "youtube:video:repeat", {
                occurrenceIndexForSource: null,
              }),
              itemWith("old-repeat-2", 2, "youtube:video:repeat", {
                occurrenceIndexForSource: null,
              }),
            ])
          : page(preflightId, [
              itemWith("new-repeat-1", 1, "youtube:video:repeat", {
                occurrenceIndexForSource: null,
              }),
              itemWith("new-repeat-2", 2, "youtube:video:repeat", {
                occurrenceIndexForSource: null,
              }),
            ]),
    );
    render(<Harness />);

    await addLines(PLAYLIST_A);
    const oldSecond = await screen.findByRole("checkbox", {
      name: "Select playlist item 2: old-repeat-2",
    });
    await userEvent.click(oldSecond);

    await userEvent.click(
      screen.getByRole("button", { name: "Refresh playlist inspection" }),
    );
    await screen.findByText("Playlist ingestion is unavailable. Try again.");
    await userEvent.click(
      screen.getByRole("button", {
        name: `Retry playlist inspection for ${PLAYLIST_A}`,
      }),
    );

    const newSecond = await screen.findByRole("checkbox", {
      name: "Select playlist item 2: new-repeat-2",
    });
    expect(newSecond).toBeChecked();
    expect(
      screen.queryByText(/added, removed, reordered/i),
    ).not.toBeInTheDocument();
    expect(apiMocks.createPlaylistPreflight).toHaveBeenCalledTimes(3);
  });

  it("warns after refresh when occurrences are reordered, added, or removed", async () => {
    apiMocks.createPlaylistPreflight
      .mockResolvedValueOnce(accepted("preflight-old"))
      .mockResolvedValueOnce(accepted("preflight-new"));
    apiMocks.getPlaylistPreflight.mockImplementation(
      async (preflightId: string) =>
        readySummary(preflightId, preflightId === "preflight-old" ? 2 : 3),
    );
    apiMocks.listPlaylistPreflightItems.mockImplementation(
      async (preflightId: string) =>
        preflightId === "preflight-old"
          ? page(preflightId, [
              itemWith("old-a", 1, "youtube:video:a"),
              itemWith("old-b", 2, "youtube:video:b"),
            ])
          : page(preflightId, [
              itemWith("new-b", 1, "youtube:video:b"),
              itemWith("new-a", 2, "youtube:video:a"),
              itemWith("new-c", 3, "youtube:video:c"),
            ]),
    );
    render(<Harness />);

    await addLines(PLAYLIST_A);
    await screen.findByRole("checkbox", {
      name: "Select playlist item 2: old-b",
    });
    await userEvent.click(
      screen.getByRole("button", { name: "Refresh playlist inspection" }),
    );

    await waitFor(() => {
      expect(
        screen.getByText(/added, removed, reordered, or could not be matched/i),
      ).toBeInTheDocument();
    });
    expect(
      screen.getByRole("checkbox", { name: "Select playlist item 2: new-a" }),
    ).toBeChecked();
    expect(
      screen.getByRole("checkbox", { name: "Select playlist item 1: new-b" }),
    ).toBeChecked();
    expect(currentState?.queueItems).toEqual([]);
  });

  it("derives session duplicates from queued direct URLs and loaded items across candidates", async () => {
    apiMocks.getPlaylistPreflight.mockImplementation(
      async (preflightId: string) =>
        readySummary(preflightId, preflightId === "PL-alpha" ? 2 : 1),
    );
    apiMocks.listPlaylistPreflightItems.mockImplementation(
      async (preflightId: string, params: { cursor?: string }) => {
        if (preflightId === "PL-alpha") {
          if (params.cursor) {
            return page(preflightId, [
              item(
                "occ-a-only",
                "https://www.youtube.com/watch?v=only-a",
                "youtube:video:only-a",
              ),
            ]);
          }
          return page(
            preflightId,
            [
              item(
                "occ-a-shared",
                "https://www.youtube.com/watch?v=shared",
                "youtube:video:shared",
              ),
            ],
            "more-items",
          );
        }
        return page(preflightId, [
          item(
            "occ-b-shared",
            "https://youtu.be/shared?t=30",
            "youtube:video:shared",
          ),
        ]);
      },
    );
    render(
      <Harness
        initialState={{
          queueItems: [
            {
              id: "queued-shared",
              url: "https://youtu.be/shared?t=5",
              detectedType: "video",
              icon: "Film",
              fileSize: 0,
              validation: { valid: true },
            },
          ],
        }}
      />,
    );

    await addLines(`${PLAYLIST_A}\n${PLAYLIST_B}`);

    await waitFor(() => {
      expect(
        screen.getByText(
          "3 staged or inspected items overlap in this session.",
        ),
      ).toBeInTheDocument();
    });
    expect(
      screen.queryByText("More playlist items are not loaded yet."),
    ).not.toBeInTheDocument();
    expect(currentState?.queueItems).toHaveLength(1);
    expect(
      currentState?.queueItems.some((row) => detectPlaylistUrl(row.url)),
    ).toBe(false);
    expectProceedBlocked();
  });
});

const detectPlaylistUrl = (url?: string): boolean =>
  Boolean(url && new URL(url).searchParams.get("list"));
