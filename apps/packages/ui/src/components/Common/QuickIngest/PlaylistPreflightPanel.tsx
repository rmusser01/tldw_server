import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Button, Typography } from "antd";
import { useVirtualizer } from "@tanstack/react-virtual";
import { ListVideo, RefreshCw } from "lucide-react";

import type {
  PlaylistInspectionCandidate,
  PlaylistSelectionUpdate,
} from "./usePlaylistInspection";
import type { PlaylistPreflightItem } from "@/services/tldw/playlist-ingest";
import { Alert, Badge } from "@/components/ui/primitives";

export type PlaylistPreflightText = (
  key: string,
  fallback: string,
  options?: Record<string, unknown>,
) => string;

type PlaylistPreflightPanelProps = {
  candidate: PlaylistInspectionCandidate;
  qi?: PlaylistPreflightText;
  onCancel: () => void;
  onRetry: () => void;
  onRemove: () => void;
  onRefresh: () => void;
  onAdd?: () => void;
  isAdding?: boolean;
  addError?: string | null;
  onSelectionChange: (occurrenceId: string, selected: boolean) => void;
  onSelectionBatchChange: (updates: readonly PlaylistSelectionUpdate[]) => void;
};

type PlaylistFilter = "all" | "new" | "duplicates" | "unavailable";

const fallbackText: PlaylistPreflightText = (_key, fallback, options) =>
  fallback.replace(/\{\{(\w+)\}\}/g, (_match, token: string) =>
    String(options?.[token] ?? ""),
  );

const isEligible = (item: PlaylistPreflightItem): boolean =>
  Boolean(item.sourceUrl) &&
  (item.availability === null || item.availability === "available");

const isDuplicate = (
  item: PlaylistPreflightItem,
  sessionDuplicates: ReadonlySet<string>,
): boolean =>
  item.duplicateStatus === "duplicate_existing" ||
  item.duplicateStatus === "duplicate_in_batch" ||
  sessionDuplicates.has(item.occurrenceId);

const isNew = (
  item: PlaylistPreflightItem,
  sessionDuplicates: ReadonlySet<string>,
): boolean =>
  item.duplicateStatus === "new" && !sessionDuplicates.has(item.occurrenceId);

const statusBadge = (
  status: PlaylistInspectionCandidate["status"],
): "info" | "success" | "warning" | "danger" => {
  if (status === "ready") return "success";
  if (status === "queued" || status === "inspecting") return "info";
  if (status === "failed" || status === "blocked") return "danger";
  return "warning";
};

const statusCopy = (
  candidate: PlaylistInspectionCandidate,
  qi: PlaylistPreflightText,
): { label: string; message: string } => {
  switch (candidate.status) {
    case "ready":
      return {
        label: qi("playlistInspection.readyLabel", "Inspection ready"),
        message: qi(
          "playlistInspection.readyMessage",
          "{{count}} playlist items loaded for review.",
          { count: candidate.items.length },
        ),
      };
    case "unavailable":
      return {
        label: qi(
          "playlistInspection.unavailableLabel",
          "Inspection unavailable",
        ),
        message: qi(
          "playlistInspection.unavailableMessage",
          "Playlist inspection is unavailable on this server. Update the server or remove this playlist.",
        ),
      };
    case "failed":
      return {
        label: qi("playlistInspection.failedLabel", "Inspection failed"),
        message:
          candidate.error?.message ??
          qi(
            "playlistInspection.failedMessage",
            "Playlist inspection failed. Try again.",
          ),
      };
    case "blocked":
      return {
        label: qi("playlistInspection.blockedLabel", "Inspection blocked"),
        message:
          candidate.error?.message ??
          qi(
            "playlistInspection.blockedMessage",
            "Playlist inspection was blocked. Remove it or try again.",
          ),
      };
    case "expired":
      return {
        label: qi("playlistInspection.expiredLabel", "Inspection expired"),
        message:
          candidate.error?.message ??
          qi(
            "playlistInspection.expiredMessage",
            "Playlist inspection expired. Try again.",
          ),
      };
    case "cancelled":
      return {
        label: qi("playlistInspection.cancelledLabel", "Inspection cancelled"),
        message:
          candidate.error?.message ??
          qi(
            "playlistInspection.cancelledMessage",
            "Playlist inspection was cancelled.",
          ),
      };
    default:
      return {
        label: qi("playlistInspection.inspectingLabel", "Inspecting playlist"),
        message: qi(
          "playlistInspection.inspectingMessage",
          "Loading playlist details from the server.",
        ),
      };
  }
};

const formatDuration = (seconds: number | null | undefined): string | null => {
  if (seconds === null || seconds === undefined || seconds < 0) return null;
  const minutes = Math.floor(seconds / 60);
  return `${minutes}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;
};

const availabilityCopy = (
  availability: PlaylistPreflightItem["availability"],
  qi: PlaylistPreflightText,
): string => {
  switch (availability) {
    case "available":
      return qi("playlistPreflight.availabilityAvailable", "Available");
    case "deleted":
      return qi("playlistPreflight.availabilityDeleted", "Deleted");
    case "needs_auth":
      return qi(
        "playlistPreflight.availabilityNeedsAuth",
        "Authentication required",
      );
    case "premium_only":
      return qi("playlistPreflight.availabilityPremiumOnly", "Premium only");
    case "private":
      return qi("playlistPreflight.availabilityPrivate", "Private");
    case "subscriber_only":
      return qi(
        "playlistPreflight.availabilitySubscriberOnly",
        "Subscribers only",
      );
    case "unavailable":
      return qi("playlistPreflight.availabilityUnavailable", "Unavailable");
    default:
      return qi(
        "playlistPreflight.availabilityUnknown",
        "Availability unknown",
      );
  }
};

export const PlaylistPreflightPanel: React.FC<PlaylistPreflightPanelProps> = ({
  candidate,
  qi = fallbackText,
  onCancel,
  onRetry,
  onRemove,
  onRefresh,
  onAdd,
  isAdding = false,
  addError = null,
  onSelectionChange,
  onSelectionBatchChange,
}) => {
  const [filter, setFilter] = useState<PlaylistFilter>("all");
  const scrollParentRef = useRef<HTMLDivElement | null>(null);
  const rowRefs = useRef(new Map<string, HTMLElement>());
  const listOwnsFocusRef = useRef(false);
  const activeListRowRef = useRef<{
    occurrenceId: string;
    index: number;
  } | null>(null);
  const copy = statusCopy(candidate, qi);
  const isActive =
    candidate.status === "queued" || candidate.status === "inspecting";
  const canRetry =
    ((candidate.status === "failed" || candidate.status === "blocked") &&
      candidate.error?.retryable === true) ||
    candidate.status === "expired" ||
    candidate.status === "cancelled";
  const selectedCount = candidate.selectedOccurrenceIds.size;
  const sessionDuplicates = candidate.sessionDuplicateOccurrenceIds;
  const duplicateCount = candidate.items.filter((item) =>
    isDuplicate(item, sessionDuplicates),
  ).length;
  const filteredItems = useMemo(
    () =>
      candidate.items.filter((item) => {
        if (filter === "new")
          return isNew(item, sessionDuplicates) && isEligible(item);
        if (filter === "duplicates")
          return isDuplicate(item, sessionDuplicates);
        if (filter === "unavailable") return !isEligible(item);
        return true;
      }),
    [candidate.items, filter, sessionDuplicates],
  );
  // TanStack Virtual exposes an imperative object that React Compiler intentionally skips.
  // eslint-disable-next-line react-hooks/incompatible-library
  const virtualizer = useVirtualizer({
    count: filteredItems.length,
    getScrollElement: () => scrollParentRef.current,
    estimateSize: () => 76,
    overscan: 5,
    getItemKey: (index) => filteredItems[index]?.occurrenceId ?? index,
    measureElement: (element) => element?.getBoundingClientRect().height || 76,
  });
  const virtualItems = virtualizer.getVirtualItems();

  const applyVisibleSelection = useCallback(
    (mode: "all" | "none" | "new") => {
      const updates = filteredItems.filter(isEligible).map((item) => ({
        occurrenceId: item.occurrenceId,
        selected:
          mode === "all" || (mode === "new" && isNew(item, sessionDuplicates)),
      }));
      if (updates.length > 0) onSelectionBatchChange(updates);
    },
    [filteredItems, onSelectionBatchChange, sessionDuplicates],
  );

  const restoreRowFocus = useCallback((occurrenceId: string) => {
    const attempt = (remaining: number) => {
      if (!listOwnsFocusRef.current) return;
      const row = rowRefs.current.get(occurrenceId);
      if (row) {
        row.focus();
        return;
      }
      if (remaining > 0) window.setTimeout(() => attempt(remaining - 1), 0);
    };
    window.setTimeout(() => attempt(2), 0);
  }, []);

  useEffect(() => {
    const handleFocusIn = (event: FocusEvent) => {
      const target = event.target;
      if (target instanceof Node && scrollParentRef.current?.contains(target)) {
        return;
      }
      listOwnsFocusRef.current = false;
    };
    document.addEventListener("focusin", handleFocusIn);
    return () => document.removeEventListener("focusin", handleFocusIn);
  }, []);

  useEffect(() => {
    const activeRow = activeListRowRef.current;
    if (
      !listOwnsFocusRef.current ||
      !activeRow ||
      rowRefs.current.has(activeRow.occurrenceId) ||
      virtualItems.length === 0
    ) {
      return;
    }
    const nearestVirtualRow = virtualItems.reduce((nearest, current) =>
      Math.abs(current.index - activeRow.index) <
      Math.abs(nearest.index - activeRow.index)
        ? current
        : nearest,
    );
    const nearestItem = filteredItems[nearestVirtualRow.index];
    if (nearestItem) restoreRowFocus(nearestItem.occurrenceId);
  }, [filteredItems, restoreRowFocus, virtualItems]);

  const handleRowKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLElement>, index: number) => {
      if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return;
      event.preventDefault();
      const targetIndex = Math.max(
        0,
        Math.min(
          filteredItems.length - 1,
          index + (event.key === "ArrowDown" ? 1 : -1),
        ),
      );
      const target = filteredItems[targetIndex];
      if (!target) return;
      virtualizer.scrollToIndex(targetIndex, { align: "auto" });
      restoreRowFocus(target.occurrenceId);
    },
    [filteredItems, restoreRowFocus, virtualizer],
  );

  return (
    <div className="rounded-md border border-border bg-surface px-3 py-2">
      <div className="flex items-start gap-2">
        <ListVideo
          className="mt-0.5 h-4 w-4 flex-shrink-0 text-primary"
          aria-hidden="true"
        />
        <div className="min-w-0 flex-1">
          <Typography.Text className="block text-sm font-medium">
            {candidate.summary?.summary?.playlistTitle ||
              qi("playlistPreflight.detected", "Playlist detected")}
          </Typography.Text>
          <details className="text-[11px] text-text-muted">
            <summary>
              {qi("playlistPreflight.details", "Playlist details")}
            </summary>
            <span>{candidate.url}</span>
          </details>
        </div>
        {candidate.status === "ready" && (
          <Button
            size="small"
            onClick={onRefresh}
            disabled={isAdding}
            aria-label={qi(
              "playlistPreflight.refreshAria",
              "Refresh playlist inspection",
            )}
            icon={<RefreshCw className="h-3.5 w-3.5" aria-hidden="true" />}
          >
            {qi("playlistPreflight.refresh", "Refresh")}
          </Button>
        )}
      </div>

      <div
        className="mt-2 flex flex-wrap items-center justify-between gap-2"
        role="status"
        aria-live="polite"
        aria-atomic="true"
      >
        <div className="flex flex-wrap items-center gap-2">
          <Badge variant={statusBadge(candidate.status)} size="sm">
            {copy.label}
          </Badge>
          <Typography.Text className="text-xs text-text-muted">
            {copy.message}
          </Typography.Text>
        </div>
        <div className="flex items-center gap-1">
          {isActive && (
            <Button
              size="small"
              onClick={onCancel}
              aria-label={qi(
                "playlistInspection.cancelAria",
                "Cancel playlist inspection for {{url}}",
                { url: candidate.url },
              )}
            >
              {qi("playlistInspection.cancel", "Cancel")}
            </Button>
          )}
          {canRetry && (
            <Button
              size="small"
              onClick={onRetry}
              aria-label={qi(
                "playlistInspection.retryAria",
                "Retry playlist inspection for {{url}}",
                { url: candidate.url },
              )}
            >
              {qi("playlistInspection.retry", "Retry")}
            </Button>
          )}
          {!isActive && (
            <Button
              size="small"
              type="text"
              onClick={onRemove}
              disabled={isAdding}
              aria-label={qi(
                "playlistInspection.removeAria",
                "Remove playlist inspection for {{url}}",
                { url: candidate.url },
              )}
            >
              {qi("playlistInspection.remove", "Remove")}
            </Button>
          )}
        </div>
      </div>

      {candidate.status === "ready" && (
        <div className="mt-2 space-y-2">
          <div className="flex flex-wrap items-center gap-1.5">
            <Badge variant="info">
              {qi("playlistPreflight.itemCount", "{{count}} items", {
                count: candidate.items.length,
              })}
            </Badge>
            <Badge variant="success">
              {qi("playlistPreflight.selectedCount", "{{count}} selected", {
                count: selectedCount,
              })}
            </Badge>
            {duplicateCount > 0 && (
              <Badge variant="warning">
                {qi(
                  "playlistPreflight.duplicateCount",
                  "{{count}} duplicates",
                  { count: duplicateCount },
                )}
              </Badge>
            )}
          </div>

          {candidate.selectionWarning && (
            <Alert
              variant="warning"
              title={qi(
                "playlistPreflight.selectionWarning",
                "Playlist items were added, removed, reordered, or could not be matched. Review your selection before continuing.",
              )}
            />
          )}

          {addError && <Alert variant="error" title={addError} />}

          <div className="flex flex-wrap items-center gap-1.5">
            <Button
              size="small"
              onClick={() => applyVisibleSelection("all")}
              disabled={isAdding}
            >
              {qi("playlistPreflight.selectAll", "Select all")}
            </Button>
            <Button
              size="small"
              onClick={() => applyVisibleSelection("none")}
              disabled={isAdding}
            >
              {qi("playlistPreflight.selectNone", "Select none")}
            </Button>
            <Button
              size="small"
              onClick={() => applyVisibleSelection("new")}
              disabled={isAdding}
            >
              {qi("playlistPreflight.selectNew", "Select new")}
            </Button>
            <label className="ml-auto text-xs text-text-muted">
              <span className="sr-only">
                {qi("playlistPreflight.filterAria", "Filter playlist items")}
              </span>
              <select
                value={filter}
                onChange={(event) =>
                  setFilter(event.target.value as PlaylistFilter)
                }
                aria-label={qi(
                  "playlistPreflight.filterAria",
                  "Filter playlist items",
                )}
                className="rounded border border-border bg-surface px-2 py-1 text-xs"
              >
                <option value="all">
                  {qi("playlistPreflight.filterAll", "All items")}
                </option>
                <option value="new">
                  {qi("playlistPreflight.filterNew", "New")}
                </option>
                <option value="duplicates">
                  {qi("playlistPreflight.filterDuplicates", "Duplicates")}
                </option>
                <option value="unavailable">
                  {qi("playlistPreflight.filterUnavailable", "Unavailable")}
                </option>
              </select>
            </label>
          </div>

          <div
            ref={scrollParentRef}
            role="list"
            aria-label={qi("playlistPreflight.listAria", "Playlist videos")}
            className="max-h-80 overflow-y-auto rounded border border-border"
          >
            <div
              className="relative w-full"
              style={{ height: virtualizer.getTotalSize() }}
            >
              {virtualItems.map((virtualRow) => {
                const item = filteredItems[virtualRow.index];
                if (!item) return null;
                const eligible = isEligible(item);
                const title =
                  item.displayMetadata.title ||
                  qi("playlistPreflight.untitled", "Untitled video");
                const duration = formatDuration(
                  item.displayMetadata.durationSeconds,
                );
                const secondary = [
                  item.displayMetadata.playlistTitle,
                  item.displayMetadata.channelOrUploader,
                  duration,
                  availabilityCopy(item.availability, qi),
                  isDuplicate(item, sessionDuplicates)
                    ? qi("playlistPreflight.duplicate", "duplicate")
                    : item.duplicateStatus === "unknown"
                      ? qi(
                          "playlistPreflight.duplicateUnknown",
                          "duplicate status unknown",
                        )
                      : null,
                ].filter(Boolean);
                return (
                  <div
                    key={virtualRow.key}
                    ref={(element) => {
                      if (element) {
                        rowRefs.current.set(item.occurrenceId, element);
                        virtualizer.measureElement(element);
                      } else {
                        rowRefs.current.delete(item.occurrenceId);
                      }
                    }}
                    role="listitem"
                    tabIndex={0}
                    aria-setsize={filteredItems.length}
                    aria-posinset={virtualRow.index + 1}
                    data-occurrence-id={item.occurrenceId}
                    data-index={virtualRow.index}
                    onFocusCapture={() => {
                      listOwnsFocusRef.current = true;
                      activeListRowRef.current = {
                        occurrenceId: item.occurrenceId,
                        index: virtualRow.index,
                      };
                    }}
                    onKeyDown={(event) =>
                      handleRowKeyDown(event, virtualRow.index)
                    }
                    className="absolute left-0 top-0 flex w-full items-start gap-2 border-b border-border px-2 py-2"
                    style={{ transform: `translateY(${virtualRow.start}px)` }}
                  >
                    <input
                      type="checkbox"
                      checked={candidate.selectedOccurrenceIds.has(
                        item.occurrenceId,
                      )}
                      disabled={!eligible || isAdding}
                      aria-label={qi(
                        "playlistPreflight.itemSelectionAria",
                        "Select playlist item {{ordinal}}: {{title}}",
                        { ordinal: item.ordinal, title },
                      )}
                      onChange={(event) =>
                        onSelectionChange(
                          item.occurrenceId,
                          event.target.checked,
                        )
                      }
                    />
                    <div className="min-w-0 flex-1">
                      <Typography.Text className="block truncate text-xs font-medium">
                        {item.ordinal}. {title}
                      </Typography.Text>
                      <Typography.Text className="block truncate text-[11px] text-text-muted">
                        {secondary.join(" • ")}
                      </Typography.Text>
                      {item.sourceUrl && (
                        <details className="text-[11px] text-text-muted">
                          <summary>
                            {qi("playlistPreflight.itemDetails", "Details")}
                          </summary>
                          <span>{item.sourceUrl}</span>
                        </details>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="flex items-center justify-between gap-2">
            <Typography.Text
              className="text-xs text-text-muted"
              role="status"
              aria-live="polite"
            >
              {qi("playlistPreflight.selectionStatus", "{{count}} selected", {
                count: selectedCount,
              })}
            </Typography.Text>
            <Button
              size="small"
              type="primary"
              onClick={onAdd}
              loading={isAdding}
              disabled={!onAdd || selectedCount === 0 || isAdding}
            >
              {qi(
                "playlistPreflight.addVideos",
                selectedCount === 1
                  ? "Add {{count}} video"
                  : "Add {{count}} videos",
                { count: selectedCount },
              )}
            </Button>
          </div>
        </div>
      )}
    </div>
  );
};

export default PlaylistPreflightPanel;
