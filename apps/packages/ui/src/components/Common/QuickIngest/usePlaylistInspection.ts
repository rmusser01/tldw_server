import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { WizardQueueItem } from "./types";
import { normalizeUrlForDedupe } from "@/entries/shared/ingest-payloads";
import { tldwClient } from "@/services/tldw/TldwApiClient";
import {
  loadCompletePlaylistPreflightItems,
  toPlaylistIngestPublicError,
  type PlaylistIngestErrorInfo,
  type PlaylistPreflightItem,
  type PlaylistPreflightSummary,
} from "@/services/tldw/playlist-ingest";
import {
  isQuickIngestPlaylistPreflightDetail,
  type QuickIngestOpenDetail,
} from "@/utils/quick-ingest-open";

const MAX_CONCURRENT_INSPECTIONS = 2;
const FIRST_ITEMS_PAGE_LIMIT = 100;
const POLL_INTERVAL_MS = 1_200;

export type PlaylistInspectionStatus =
  | "queued"
  | "inspecting"
  | "ready"
  | "unavailable"
  | "failed"
  | "blocked"
  | "expired"
  | "cancelled";

export type PlaylistInspectionCandidate = {
  key: string;
  url: string;
  status: PlaylistInspectionStatus;
  preflightId: string | null;
  summary: PlaylistPreflightSummary | null;
  items: PlaylistPreflightItem[];
  nextCursor: string | null;
  error: PlaylistIngestErrorInfo | null;
  selectedOccurrenceIds: ReadonlySet<string>;
  sessionDuplicateOccurrenceIds: ReadonlySet<string>;
  selectionWarning: "changed" | "ambiguous" | null;
};

export type PlaylistSelectionUpdate = {
  occurrenceId: string;
  selected: boolean;
};

export type PlaylistSessionDuplicateReference = {
  id: string;
  source: "queue" | "candidate";
};

export type PlaylistSessionDuplicateIndex = ReadonlyMap<
  string,
  readonly PlaylistSessionDuplicateReference[]
>;

type UsePlaylistInspectionOptions = {
  enabled: boolean | null;
  queueItems: WizardQueueItem[];
  seed: QuickIngestOpenDetail | null;
  clearSeed: () => void;
};

const isAbortError = (error: unknown): boolean =>
  (error as { name?: unknown } | null)?.name === "AbortError";

const candidateAliases = (item: PlaylistPreflightItem): string[] => {
  const aliases: string[] = [];
  if (item.normalizedSourceId) aliases.push(item.normalizedSourceId);
  if (item.sourceUrl) aliases.push(normalizeUrlForDedupe(item.sourceUrl));
  return [...new Set(aliases.filter(Boolean))];
};

const queueAliases = (item: WizardQueueItem): string[] => {
  const aliases: string[] = [];
  if (item.playlist?.normalizedSourceId) {
    aliases.push(item.playlist.normalizedSourceId);
  }
  if (item.url) aliases.push(normalizeUrlForDedupe(item.url));
  return [...new Set(aliases.filter(Boolean))];
};

const isEligiblePlaylistItem = (item: PlaylistPreflightItem): boolean =>
  Boolean(item.sourceUrl) &&
  (item.availability === null || item.availability === "available");

type SelectionOverrides = Map<string, Map<string, boolean>>;

const applyPlaylistSelectionDefaults = (
  queueItems: WizardQueueItem[],
  candidates: Map<string, PlaylistInspectionCandidate>,
  overrides: SelectionOverrides,
): Map<string, PlaylistInspectionCandidate> => {
  const seenAliases = new Set<string>();
  for (const queueItem of queueItems) {
    for (const alias of queueAliases(queueItem)) seenAliases.add(alias);
  }

  const next = new Map(candidates);
  for (const [key, candidate] of next) {
    const selected = new Set<string>();
    const sessionDuplicates = new Set<string>();
    const explicit = overrides.get(key);
    const orderedItems = [...candidate.items].sort(
      (left, right) => left.ordinal - right.ordinal,
    );
    for (const item of orderedItems) {
      const aliases = candidateAliases(item);
      const eligible = isEligiblePlaylistItem(item);
      const isLaterSessionOccurrence = aliases.some((alias) =>
        seenAliases.has(alias),
      );
      if (isLaterSessionOccurrence) {
        sessionDuplicates.add(item.occurrenceId);
      }
      const explicitSelection = explicit?.get(item.occurrenceId);
      const isSelected =
        eligible &&
        (explicitSelection ??
          (item.selectedByDefault === true && !isLaterSessionOccurrence));
      if (isSelected) selected.add(item.occurrenceId);
      if (eligible) {
        for (const alias of aliases) seenAliases.add(alias);
      }
    }
    next.set(key, {
      ...candidate,
      selectedOccurrenceIds: selected,
      sessionDuplicateOccurrenceIds: sessionDuplicates,
    });
  }
  return next;
};

const playlistReconciliationKeys = (
  items: PlaylistPreflightItem[],
): Array<string | null> => {
  const repeatIndexes = new Map<string, number>();
  return items.map((item) => {
    if (!item.normalizedSourceId) return null;
    const fallbackIndex = (repeatIndexes.get(item.normalizedSourceId) ?? 0) + 1;
    repeatIndexes.set(item.normalizedSourceId, fallbackIndex);
    const occurrenceIndex =
      item.occurrenceIndexForSource !== null &&
      Number.isSafeInteger(item.occurrenceIndexForSource) &&
      item.occurrenceIndexForSource > 0
        ? item.occurrenceIndexForSource
        : fallbackIndex;
    return `${item.normalizedSourceId}\u0000${occurrenceIndex}`;
  });
};

const reconcilePlaylistSelection = (
  previousItems: PlaylistPreflightItem[],
  previousSelected: ReadonlySet<string>,
  nextItems: PlaylistPreflightItem[],
): {
  overrides: Map<string, boolean>;
  warning: PlaylistInspectionCandidate["selectionWarning"];
} => {
  const previousKeys = playlistReconciliationKeys(previousItems);
  const nextKeys = playlistReconciliationKeys(nextItems);
  const previousByKey = new Map<string, number[]>();
  const nextByKey = new Map<string, number[]>();
  const add = (target: Map<string, number[]>, key: string, index: number) =>
    target.set(key, [...(target.get(key) ?? []), index]);
  previousKeys.forEach((key, index) => {
    if (key !== null) add(previousByKey, key, index);
  });
  nextKeys.forEach((key, index) => {
    if (key !== null) add(nextByKey, key, index);
  });

  const overrides = new Map<string, boolean>();
  let ambiguous = previousKeys.includes(null) || nextKeys.includes(null);
  for (const [key, previousIndexes] of previousByKey) {
    const nextIndexes = nextByKey.get(key) ?? [];
    if (previousIndexes.length !== 1 || nextIndexes.length !== 1) {
      ambiguous = true;
      continue;
    }
    const previousItem = previousItems[previousIndexes[0]];
    const nextItem = nextItems[nextIndexes[0]];
    overrides.set(
      nextItem.occurrenceId,
      previousSelected.has(previousItem.occurrenceId),
    );
  }

  const changed =
    previousKeys.length !== nextKeys.length ||
    previousKeys.some((key, index) => key !== nextKeys[index]);
  return {
    overrides,
    warning: ambiguous ? "ambiguous" : changed ? "changed" : null,
  };
};

export const buildPlaylistSessionDuplicateIndex = (
  queueItems: WizardQueueItem[],
  candidates: Iterable<PlaylistInspectionCandidate>,
): PlaylistSessionDuplicateIndex => {
  const aliases = new Map<
    string,
    Map<string, PlaylistSessionDuplicateReference>
  >();
  const add = (alias: string, reference: PlaylistSessionDuplicateReference) => {
    const references = aliases.get(alias) ?? new Map();
    references.set(reference.id, reference);
    aliases.set(alias, references);
  };

  for (const queueItem of queueItems) {
    const reference = { id: `queue:${queueItem.id}`, source: "queue" as const };
    for (const alias of queueAliases(queueItem)) add(alias, reference);
  }

  for (const candidate of candidates) {
    for (const item of candidate.items) {
      const reference = {
        id: `candidate:${candidate.key}:${item.occurrenceId}`,
        source: "candidate" as const,
      };
      for (const alias of candidateAliases(item)) add(alias, reference);
    }
  }

  return new Map(
    [...aliases].map(([alias, references]) => [
      alias,
      [...references.values()],
    ]),
  );
};

export const usePlaylistInspection = ({
  enabled,
  queueItems,
  seed,
  clearSeed,
}: UsePlaylistInspectionOptions) => {
  const [candidateMap, setCandidateMap] = useState<
    Map<string, PlaylistInspectionCandidate>
  >(new Map());
  const candidatesRef = useRef(candidateMap);
  const queueItemsRef = useRef(queueItems);
  queueItemsRef.current = queueItems;
  const controllersRef = useRef(new Map<string, AbortController>());
  const cleanupRef = useRef(new Map<string, Promise<void>>());
  const selectionOverridesRef = useRef<SelectionOverrides>(new Map());
  const refreshBaselinesRef = useRef(
    new Map<
      string,
      {
        items: PlaylistPreflightItem[];
        selectedOccurrenceIds: ReadonlySet<string>;
      }
    >(),
  );
  const timersRef = useRef(new Map<string, ReturnType<typeof setTimeout>>());
  const pendingRef = useRef<string[]>([]);
  const activeRef = useRef(new Set<string>());
  const mountedRef = useRef(true);
  const pumpRef = useRef<() => void>(() => {});
  const consumedSeedRef = useRef<QuickIngestOpenDetail | null>(null);

  const publish = useCallback(
    (next: Map<string, PlaylistInspectionCandidate>) => {
      candidatesRef.current = next;
      if (mountedRef.current) setCandidateMap(next);
    },
    [],
  );

  const updateCandidate = useCallback(
    (
      key: string,
      update: (
        current: PlaylistInspectionCandidate,
      ) => PlaylistInspectionCandidate,
    ) => {
      const current = candidatesRef.current.get(key);
      if (!current) return;
      const next = new Map(candidatesRef.current);
      next.set(key, update(current));
      publish(next);
    },
    [publish],
  );

  const publishSelectionDefaults = useCallback(
    (next: Map<string, PlaylistInspectionCandidate>) => {
      publish(
        applyPlaylistSelectionDefaults(
          queueItemsRef.current,
          next,
          selectionOverridesRef.current,
        ),
      );
    },
    [publish],
  );

  const updateCandidateSelectionDefaults = useCallback(
    (
      key: string,
      update: (
        current: PlaylistInspectionCandidate,
      ) => PlaylistInspectionCandidate,
    ) => {
      const current = candidatesRef.current.get(key);
      if (!current) return;
      const next = new Map(candidatesRef.current);
      next.set(key, update(current));
      publishSelectionDefaults(next);
    },
    [publishSelectionDefaults],
  );

  const clearTimer = useCallback((key: string) => {
    const timer = timersRef.current.get(key);
    if (timer !== undefined) clearTimeout(timer);
    timersRef.current.delete(key);
  }, []);

  const waitForNextPoll = useCallback(
    (key: string, signal: AbortSignal) =>
      new Promise<void>((resolve, reject) => {
        if (signal.aborted) {
          reject(new DOMException("Aborted", "AbortError"));
          return;
        }
        const onAbort = () => {
          clearTimer(key);
          reject(new DOMException("Aborted", "AbortError"));
        };
        const timer = setTimeout(() => {
          timersRef.current.delete(key);
          signal.removeEventListener("abort", onAbort);
          resolve();
        }, POLL_INTERVAL_MS);
        timersRef.current.set(key, timer);
        signal.addEventListener("abort", onAbort, { once: true });
      }),
    [clearTimer],
  );

  const scheduleCleanup = useCallback(
    (key: string, preflightId: string): Promise<void> => {
      const previous = cleanupRef.current.get(key) ?? Promise.resolve();
      const cleanup = previous
        .catch(() => {})
        .then(() => tldwClient.cancelPlaylistPreflight(preflightId))
        .catch(() => {});
      cleanupRef.current.set(key, cleanup);
      void cleanup.finally(() => {
        if (cleanupRef.current.get(key) === cleanup) {
          cleanupRef.current.delete(key);
        }
      });
      return cleanup;
    },
    [],
  );

  const runInspection = useCallback(
    async (key: string, controller: AbortController) => {
      const initial = candidatesRef.current.get(key);
      if (!initial) return;
      const canCommit = () =>
        !controller.signal.aborted &&
        mountedRef.current &&
        controllersRef.current.get(key) === controller;

      try {
        await cleanupRef.current.get(key);
        if (!canCommit()) return;
        const accepted = await tldwClient.createPlaylistPreflight(
          { url: initial.url },
          { signal: controller.signal },
        );
        if (!canCommit()) {
          await scheduleCleanup(key, accepted.preflightId);
          return;
        }
        updateCandidate(key, (current) => ({
          ...current,
          preflightId: accepted.preflightId,
          error: null,
        }));

        while (!controller.signal.aborted) {
          const status = await tldwClient.getPlaylistPreflight(
            accepted.preflightId,
            {
              signal: controller.signal,
            },
          );
          if (!canCommit()) return;

          if (status.status === "pending" || status.status === "running") {
            updateCandidate(key, (current) => ({
              ...current,
              status: "inspecting",
              summary: status,
              error: status.error,
            }));
            await waitForNextPoll(key, controller.signal);
            continue;
          }

          if (status.status === "ready") {
            updateCandidate(key, (current) => ({
              ...current,
              status: "inspecting",
              summary: status,
              error: status.error,
            }));
            const items = await loadCompletePlaylistPreflightItems({
              preflightId: accepted.preflightId,
              summary: status,
              signal: controller.signal,
              pageSize: FIRST_ITEMS_PAGE_LIMIT,
              loadPage: (preflightId, params, options) =>
                tldwClient.listPlaylistPreflightItems(
                  preflightId,
                  params,
                  options,
                ),
            });
            if (!canCommit()) return;
            const baseline = refreshBaselinesRef.current.get(key);
            if (baseline) {
              const reconciliation = reconcilePlaylistSelection(
                baseline.items,
                baseline.selectedOccurrenceIds,
                items,
              );
              selectionOverridesRef.current.set(key, reconciliation.overrides);
              refreshBaselinesRef.current.delete(key);
              const next = new Map(candidatesRef.current);
              const current = next.get(key);
              if (!current) return;
              next.set(key, {
                ...current,
                status: "ready",
                summary: status,
                items,
                nextCursor: null,
                error: status.error,
                selectionWarning: reconciliation.warning,
              });
              publishSelectionDefaults(next);
              return;
            }
            const next = new Map(candidatesRef.current);
            const current = next.get(key);
            if (!current) return;
            next.set(key, {
              ...current,
              status: "ready",
              summary: status,
              items,
              nextCursor: null,
              error: status.error,
              selectionWarning: null,
            });
            publishSelectionDefaults(next);
            return;
          }

          updateCandidate(key, (current) => ({
            ...current,
            status:
              status.status === "blocked"
                ? "blocked"
                : status.status === "expired"
                  ? "expired"
                  : "cancelled",
            summary: status,
            error: status.error,
          }));
          return;
        }
      } catch (error) {
        if (controller.signal.aborted || isAbortError(error) || !canCommit()) {
          return;
        }
        const publicError = toPlaylistIngestPublicError(error);
        updateCandidateSelectionDefaults(key, (current) => ({
          ...current,
          status: "failed",
          items: [],
          nextCursor: null,
          selectedOccurrenceIds: new Set(),
          sessionDuplicateOccurrenceIds: new Set(),
          error: {
            code: publicError.code,
            message: publicError.message,
            retryable: publicError.retryable,
          },
        }));
      } finally {
        clearTimer(key);
      }
    },
    [
      clearTimer,
      publishSelectionDefaults,
      scheduleCleanup,
      updateCandidate,
      updateCandidateSelectionDefaults,
      waitForNextPoll,
    ],
  );

  const pump = useCallback(() => {
    if (enabled !== true || !mountedRef.current) return;
    while (
      activeRef.current.size < MAX_CONCURRENT_INSPECTIONS &&
      pendingRef.current.length > 0
    ) {
      const pendingIndex = pendingRef.current.findIndex(
        (key) => !activeRef.current.has(key),
      );
      if (pendingIndex < 0) break;
      const [key] = pendingRef.current.splice(pendingIndex, 1);
      if (!key) continue;
      const candidate = candidatesRef.current.get(key);
      if (!candidate || candidate.status !== "queued") continue;

      const controller = new AbortController();
      controllersRef.current.set(key, controller);
      activeRef.current.add(key);
      updateCandidate(key, (current) => ({
        ...current,
        status: "inspecting",
      }));
      void runInspection(key, controller).finally(() => {
        if (controllersRef.current.get(key) === controller) {
          controllersRef.current.delete(key);
        }
        activeRef.current.delete(key);
        pumpRef.current();
      });
    }
  }, [enabled, runInspection, updateCandidate]);

  useEffect(() => {
    pumpRef.current = pump;
  }, [pump]);

  const addCandidates = useCallback(
    (urls: string[]) => {
      const next = new Map(candidatesRef.current);
      let changed = false;
      for (const rawUrl of urls) {
        const key = rawUrl.trim();
        if (!key || next.has(key)) continue;
        next.set(key, {
          key,
          url: key,
          status: enabled === false ? "unavailable" : "queued",
          preflightId: null,
          summary: null,
          items: [],
          nextCursor: null,
          error: null,
          selectedOccurrenceIds: new Set(),
          sessionDuplicateOccurrenceIds: new Set(),
          selectionWarning: null,
        });
        if (enabled !== false) pendingRef.current.push(key);
        changed = true;
      }
      if (!changed) return;
      publish(next);
      queueMicrotask(() => pumpRef.current());
    },
    [enabled, publish],
  );

  const cancelCandidate = useCallback(
    (key: string) => {
      pendingRef.current = pendingRef.current.filter((value) => value !== key);
      clearTimer(key);
      controllersRef.current.get(key)?.abort();
      const preflightId = candidatesRef.current.get(key)?.preflightId;
      refreshBaselinesRef.current.delete(key);
      selectionOverridesRef.current.delete(key);
      updateCandidateSelectionDefaults(key, (current) => ({
        ...current,
        status: "cancelled",
        preflightId: null,
        items: [],
        nextCursor: null,
        error: null,
        selectedOccurrenceIds: new Set(),
        sessionDuplicateOccurrenceIds: new Set(),
        selectionWarning: null,
      }));
      if (preflightId) {
        void scheduleCleanup(key, preflightId);
      }
    },
    [clearTimer, scheduleCleanup, updateCandidateSelectionDefaults],
  );

  const removeCandidate = useCallback(
    (key: string) => {
      pendingRef.current = pendingRef.current.filter((value) => value !== key);
      clearTimer(key);
      controllersRef.current.get(key)?.abort();
      const preflightId = candidatesRef.current.get(key)?.preflightId;
      refreshBaselinesRef.current.delete(key);
      selectionOverridesRef.current.delete(key);
      const next = new Map(candidatesRef.current);
      next.delete(key);
      publishSelectionDefaults(next);
      if (preflightId) {
        void scheduleCleanup(key, preflightId);
      }
    },
    [clearTimer, publishSelectionDefaults, scheduleCleanup],
  );

  const retryCandidate = useCallback(
    (key: string) => {
      pendingRef.current = pendingRef.current.filter((value) => value !== key);
      clearTimer(key);
      controllersRef.current.get(key)?.abort();
      const preflightId = candidatesRef.current.get(key)?.preflightId;
      if (preflightId) {
        void scheduleCleanup(key, preflightId);
      }
      if (!refreshBaselinesRef.current.has(key)) {
        selectionOverridesRef.current.delete(key);
      }
      updateCandidateSelectionDefaults(key, (current) => ({
        ...current,
        status: enabled === false ? "unavailable" : "queued",
        preflightId: null,
        summary: null,
        items: [],
        nextCursor: null,
        error: null,
        selectedOccurrenceIds: new Set(),
        sessionDuplicateOccurrenceIds: new Set(),
        selectionWarning: null,
      }));
      if (enabled !== false) {
        pendingRef.current.push(key);
        queueMicrotask(() => pumpRef.current());
      }
    },
    [clearTimer, enabled, scheduleCleanup, updateCandidateSelectionDefaults],
  );

  const refreshCandidate = useCallback(
    (key: string) => {
      const candidate = candidatesRef.current.get(key);
      if (!candidate) return;
      pendingRef.current = pendingRef.current.filter((value) => value !== key);
      clearTimer(key);
      controllersRef.current.get(key)?.abort();
      if (candidate.preflightId) {
        void scheduleCleanup(key, candidate.preflightId);
      }
      if (candidate.items.length > 0) {
        refreshBaselinesRef.current.set(key, {
          items: candidate.items,
          selectedOccurrenceIds: candidate.selectedOccurrenceIds,
        });
      } else {
        refreshBaselinesRef.current.delete(key);
      }
      selectionOverridesRef.current.delete(key);
      updateCandidateSelectionDefaults(key, (current) => ({
        ...current,
        status: enabled === false ? "unavailable" : "queued",
        preflightId: null,
        summary: null,
        items: [],
        nextCursor: null,
        error: null,
        selectedOccurrenceIds: new Set(),
        sessionDuplicateOccurrenceIds: new Set(),
        selectionWarning: null,
      }));
      if (enabled !== false) {
        pendingRef.current.push(key);
        queueMicrotask(() => pumpRef.current());
      }
    },
    [clearTimer, enabled, scheduleCleanup, updateCandidateSelectionDefaults],
  );

  const setCandidateSelection = useCallback(
    (key: string, occurrenceId: string, selected: boolean) => {
      const candidate = candidatesRef.current.get(key);
      const item = candidate?.items.find(
        (entry) => entry.occurrenceId === occurrenceId,
      );
      if (!candidate || !item || !isEligiblePlaylistItem(item)) return;
      const explicit = new Map(selectionOverridesRef.current.get(key) ?? []);
      explicit.set(occurrenceId, selected);
      selectionOverridesRef.current.set(key, explicit);
      updateCandidate(key, (current) => {
        const selectedOccurrenceIds = new Set(current.selectedOccurrenceIds);
        if (selected) selectedOccurrenceIds.add(occurrenceId);
        else selectedOccurrenceIds.delete(occurrenceId);
        return { ...current, selectedOccurrenceIds };
      });
    },
    [updateCandidate],
  );

  const setCandidateSelections = useCallback(
    (key: string, updates: readonly PlaylistSelectionUpdate[]) => {
      const candidate = candidatesRef.current.get(key);
      if (!candidate || updates.length === 0) return;
      const requested = new Map(
        updates.map(({ occurrenceId, selected }) => [occurrenceId, selected]),
      );
      const eligibleUpdates = candidate.items.flatMap((item) =>
        requested.has(item.occurrenceId) && isEligiblePlaylistItem(item)
          ? [
              {
                occurrenceId: item.occurrenceId,
                selected: requested.get(item.occurrenceId) as boolean,
              },
            ]
          : [],
      );
      if (eligibleUpdates.length === 0) return;

      const explicit = new Map(selectionOverridesRef.current.get(key) ?? []);
      for (const { occurrenceId, selected } of eligibleUpdates) {
        explicit.set(occurrenceId, selected);
      }
      selectionOverridesRef.current.set(key, explicit);
      updateCandidate(key, (current) => {
        const selectedOccurrenceIds = new Set(current.selectedOccurrenceIds);
        for (const { occurrenceId, selected } of eligibleUpdates) {
          if (selected) selectedOccurrenceIds.add(occurrenceId);
          else selectedOccurrenceIds.delete(occurrenceId);
        }
        return { ...current, selectedOccurrenceIds };
      });
    },
    [updateCandidate],
  );

  useEffect(() => {
    if (candidatesRef.current.size === 0) return;
    publishSelectionDefaults(candidatesRef.current);
  }, [publishSelectionDefaults, queueItems]);

  useEffect(() => {
    if (!isQuickIngestPlaylistPreflightDetail(seed)) {
      if (seed === null) consumedSeedRef.current = null;
      return;
    }
    if (consumedSeedRef.current === seed) return;
    consumedSeedRef.current = seed;
    addCandidates([seed.url]);
    clearSeed();
  }, [addCandidates, clearSeed, seed]);

  useEffect(() => {
    if (enabled === null) return;
    if (enabled) {
      const next = new Map(candidatesRef.current);
      let changed = false;
      for (const [key, candidate] of next) {
        const shouldRequeue = candidate.status === "unavailable";
        if (shouldRequeue) {
          next.set(key, { ...candidate, status: "queued" });
          changed = true;
        }
        if (
          (shouldRequeue || candidate.status === "queued") &&
          !pendingRef.current.includes(key)
        ) {
          pendingRef.current.push(key);
        }
      }
      if (changed) publish(next);
      queueMicrotask(() => pumpRef.current());
      return;
    }
    pendingRef.current = [];
    for (const controller of controllersRef.current.values())
      controller.abort();
    const next = new Map(candidatesRef.current);
    let changed = false;
    for (const [key, candidate] of next) {
      if (candidate.status === "ready") continue;
      if (candidate.preflightId) {
        void scheduleCleanup(key, candidate.preflightId);
      }
      next.set(key, {
        ...candidate,
        status: "unavailable",
        preflightId: null,
        error: null,
      });
      changed = true;
    }
    if (changed) publish(next);
  }, [enabled, publish, scheduleCleanup]);

  useEffect(() => {
    const timers = timersRef.current;
    const controllers = controllersRef.current;
    const candidates = candidatesRef;
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      pendingRef.current = [];
      for (const timer of timers.values()) clearTimeout(timer);
      timers.clear();
      for (const controller of controllers.values()) controller.abort();
      controllers.clear();
      for (const candidate of candidates.current.values()) {
        if (candidate.preflightId) {
          void tldwClient
            .cancelPlaylistPreflight(candidate.preflightId)
            .catch(() => {});
        }
      }
    };
  }, []);

  const candidates = useMemo(() => [...candidateMap.values()], [candidateMap]);
  const sessionDuplicateIndex = useMemo(
    () => buildPlaylistSessionDuplicateIndex(queueItems, candidates),
    [candidates, queueItems],
  );
  const sessionDuplicateCount = useMemo(() => {
    const duplicates = new Set<string>();
    for (const references of sessionDuplicateIndex.values()) {
      if (references.length < 2) continue;
      for (const reference of references) duplicates.add(reference.id);
    }
    return duplicates.size;
  }, [sessionDuplicateIndex]);

  return {
    candidates,
    addCandidates,
    cancelCandidate,
    removeCandidate,
    retryCandidate,
    refreshCandidate,
    setCandidateSelection,
    setCandidateSelections,
    hasUnresolvedCandidates: candidates.length > 0,
    hasTruncatedCandidates: candidates.some(
      (candidate) => candidate.nextCursor !== null,
    ),
    sessionDuplicateIndex,
    sessionDuplicateCount,
  };
};
