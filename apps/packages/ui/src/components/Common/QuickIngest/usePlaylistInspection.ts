import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { WizardQueueItem } from "./types";
import { normalizeUrlForDedupe } from "@/entries/shared/ingest-payloads";
import { tldwClient } from "@/services/tldw/TldwApiClient";
import {
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
  const controllersRef = useRef(new Map<string, AbortController>());
  const cleanupRef = useRef(new Map<string, Promise<void>>());
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
            const firstPage = await tldwClient.listPlaylistPreflightItems(
              accepted.preflightId,
              { limit: FIRST_ITEMS_PAGE_LIMIT },
              { signal: controller.signal },
            );
            if (!canCommit()) return;
            updateCandidate(key, (current) => ({
              ...current,
              status: "ready",
              summary: status,
              items: firstPage.items,
              nextCursor: firstPage.nextCursor,
              error: status.error,
            }));
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
        updateCandidate(key, (current) => ({
          ...current,
          status: "failed",
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
    [clearTimer, scheduleCleanup, updateCandidate, waitForNextPoll],
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
      updateCandidate(key, (current) => ({
        ...current,
        status: "cancelled",
        preflightId: null,
        error: null,
      }));
      if (preflightId) {
        void scheduleCleanup(key, preflightId);
      }
    },
    [clearTimer, scheduleCleanup, updateCandidate],
  );

  const removeCandidate = useCallback(
    (key: string) => {
      pendingRef.current = pendingRef.current.filter((value) => value !== key);
      clearTimer(key);
      controllersRef.current.get(key)?.abort();
      const preflightId = candidatesRef.current.get(key)?.preflightId;
      const next = new Map(candidatesRef.current);
      next.delete(key);
      publish(next);
      if (preflightId) {
        void scheduleCleanup(key, preflightId);
      }
    },
    [clearTimer, publish, scheduleCleanup],
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
      updateCandidate(key, (current) => ({
        ...current,
        status: enabled === false ? "unavailable" : "queued",
        preflightId: null,
        summary: null,
        items: [],
        nextCursor: null,
        error: null,
      }));
      if (enabled !== false) {
        pendingRef.current.push(key);
        queueMicrotask(() => pumpRef.current());
      }
    },
    [clearTimer, enabled, scheduleCleanup, updateCandidate],
  );

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
    hasUnresolvedCandidates: candidates.length > 0,
    hasTruncatedCandidates: candidates.some(
      (candidate) => candidate.nextCursor !== null,
    ),
    sessionDuplicateIndex,
    sessionDuplicateCount,
  };
};
