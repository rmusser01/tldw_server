// @vitest-environment jsdom
import React from "react";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { IngestWizardState } from "../IngestWizardContext";
import type {
  PlaylistIngestErrorInfo,
  PlaylistPreflightAccepted,
  PlaylistPreflightItem,
  PlaylistPreflightItemsPage,
  PlaylistPreflightStatus,
  PlaylistPreflightSummary,
} from "@/services/tldw/playlist-ingest";
import { PlaylistIngestPublicError } from "@/services/tldw/playlist-ingest";

const apiMocks = vi.hoisted(() => ({
  createPlaylistPreflight: vi.fn(),
  getPlaylistPreflight: vi.fn(),
  listPlaylistPreflightItems: vi.fn(),
  cancelPlaylistPreflight: vi.fn(),
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
import { IngestWizardProvider, useIngestWizard } from "../IngestWizardContext";

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
          totalCount: 3,
          loadedCount: 3,
          ingestibleCount: 3,
          unavailableCount: 0,
          duplicateCount: 0,
          selectedCount: 3,
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

  it("derives session duplicates from queued direct URLs and loaded items across candidates", async () => {
    apiMocks.listPlaylistPreflightItems.mockImplementation(
      async (preflightId: string) => {
        if (preflightId === "PL-alpha") {
          return page(
            preflightId,
            [
              item(
                "occ-a-shared",
                "https://www.youtube.com/watch?v=shared",
                "youtube:video:shared",
              ),
              item(
                "occ-a-only",
                "https://www.youtube.com/watch?v=only-a",
                "youtube:video:only-a",
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
      screen.getByText("More playlist items are not loaded yet."),
    ).toBeInTheDocument();
    expect(currentState?.queueItems).toHaveLength(1);
    expect(
      currentState?.queueItems.some((row) => detectPlaylistUrl(row.url)),
    ).toBe(false);
    expectProceedBlocked();
  });
});

const detectPlaylistUrl = (url?: string): boolean =>
  Boolean(url && new URL(url).searchParams.get("list"));
