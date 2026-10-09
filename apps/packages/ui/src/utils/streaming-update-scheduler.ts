/**
 * Trailing-flush scheduler for streaming UI updates.
 *
 * Streaming loops receive high-frequency chunks (often one per token). Writing
 * every chunk straight into a store triggers a full-list setState per token.
 * This scheduler accumulates the latest pending value and flushes it into the
 * store at most once per `intervalMs`, mirroring the cadence already used by
 * `hooks/chat/useChatActions.ts` (STREAMING_UPDATE_INTERVAL_MS = 80).
 *
 * Contract:
 * - `schedule(value)` records the latest value and arms a trailing timer. The
 *   flush happens no sooner than `intervalMs` after the previous flush, so a
 *   slow trickle of chunks still flushes on the trailing edge instead of
 *   stacking timers.
 * - `flushNow()` writes the pending value synchronously and disarms the timer.
 *   Always call it when the stream ends (or before terminal error handling
 *   that needs the streamed content) so no content is lost.
 * - `cancel()` drops the pending value and disarms the timer. Call it when the
 *   turn is discarded/restored so a late trailing flush cannot land on top of
 *   the restored state.
 * - `shouldFlush()` is consulted at flush time; returning false drops the
 *   pending value (used to stop updates after a request-scope invalidation).
 */
export const STREAMING_UPDATE_INTERVAL_MS = 80;

export type StreamingUpdateScheduler<T> = {
  schedule: (value: T) => void;
  flushNow: () => void;
  cancel: () => void;
  hasPending: () => boolean;
};

export type StreamingUpdateSchedulerOptions<T> = {
  apply: (value: T) => void;
  shouldFlush?: () => boolean;
  intervalMs?: number;
};

export const createStreamingUpdateScheduler = <T>(
  options: StreamingUpdateSchedulerOptions<T>,
): StreamingUpdateScheduler<T> => {
  const intervalMs = options.intervalMs ?? STREAMING_UPDATE_INTERVAL_MS;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let pending: { value: T } | null = null;
  let lastFlushAt = 0;

  const flushNow = () => {
    if (timer !== null) {
      clearTimeout(timer);
      timer = null;
    }
    if (pending === null) return;
    const value = pending.value;
    pending = null;
    if (options.shouldFlush && !options.shouldFlush()) return;
    lastFlushAt = Date.now();
    options.apply(value);
  };

  const schedule = (value: T) => {
    pending = { value };
    if (timer !== null) return;
    const elapsed = Date.now() - lastFlushAt;
    const delay = Math.max(0, intervalMs - elapsed);
    timer = setTimeout(() => {
      timer = null;
      flushNow();
    }, delay);
  };

  const cancel = () => {
    if (timer !== null) {
      clearTimeout(timer);
      timer = null;
    }
    pending = null;
  };

  return {
    schedule,
    flushNow,
    cancel,
    hasPending: () => pending !== null,
  };
};
