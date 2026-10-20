/**
 * Shared 80ms coalescing throttle for streamed chat transcript updates
 * (TASK-13521, Batch W1 Stage 1).
 *
 * Streaming loops receive high-frequency chunks (often one per token). Writing
 * every chunk straight into the messages store triggers a full-list setState
 * per token. This module extracts the trailing-flush throttle that
 * `hooks/chat/useChatActions.ts` and `hooks/chat/useCharacterChatMode.ts`
 * each carried as a private inline copy, and that `hooks/useMessage.tsx`
 * (sidepanel) built around `utils/streaming-update-scheduler.ts`, into one
 * helper consumed by all three paths.
 *
 * The throttled update is also narrowed: instead of cloning the whole
 * transcript through `setMessages((prev) => prev.map(...))`, only the
 * streaming message (located by id, searched from the end) is replaced via
 * `Array.prototype.slice`, so non-streaming messages keep their references
 * and per-token array-clone cost is zero.
 *
 * Contract:
 * - `schedule(text, reasoningTime)` records the latest text and arms a
 *   trailing timer; the flush lands no sooner than `intervalMs` after the
 *   previous flush, so a slow trickle of chunks still flushes on the trailing
 *   edge instead of stacking timers.
 * - `flushNow()` writes the pending text synchronously and disarms the timer.
 *   Always call it when the stream ends so the LAST chunk is never dropped.
 * - `cancel()` drops the pending text and disarms the timer. Call it when the
 *   turn is discarded/restored so a late trailing flush cannot land on top of
 *   the restored state.
 * - `disarm()` disarms the timer but KEEPS the pending text. This preserves
 *   the abort semantics of the former inline throttles: an abort listener
 *   disarms, the post-stream `flushNow()` can still land the partial text,
 *   and no trailing flush fires during error handling.
 * - `shouldFlush` is consulted at flush time; returning false drops the
 *   pending value (used to stop updates after a request-scope invalidation).
 */
import type { Message } from "@/store/option";
import { updateActiveVariant } from "@/utils/message-variants";

/** Coalescing window for streamed transcript updates (ms). */
export const STREAMING_UPDATE_INTERVAL_MS = 80;

export type StreamingMessagesSetter = (
  updater: (prev: Message[]) => Message[]
) => void;

/**
 * Replace only the streaming message (located by id, searched from the end)
 * instead of cloning the transcript through `prev.map(...)`. Returns the
 * previous array reference untouched when the id is absent (e.g. the turn was
 * restored), so a stale update cannot notify subscribers for nothing.
 */
export const replaceStreamingMessage = (
  prev: Message[],
  messageId: string,
  update: (message: Message) => Message
): Message[] => {
  for (let index = prev.length - 1; index >= 0; index -= 1) {
    const message = prev[index];
    if (message?.id === messageId) {
      const next = prev.slice();
      next[index] = update(message);
      return next;
    }
  }
  return prev;
};

export type StreamingTextScheduler = {
  /** Record the latest streamed text; a trailing flush lands it in-window. */
  schedule: (text: string, reasoningTime: number) => void;
  /** Write the pending text synchronously and disarm the timer (stream end). */
  flushNow: () => void;
  /** Drop the pending text and disarm the timer (turn restored/errored). */
  cancel: () => void;
  /** Disarm the timer but keep the pending text (abort). */
  disarm: () => void;
  hasPending: () => boolean;
};

export type StreamingTextSchedulerOptions = {
  /** Skip (and drop) a pending flush, e.g. the request scope was invalidated. */
  shouldFlush?: () => boolean;
  intervalMs?: number;
};

export const createStreamingTextScheduler = (
  generateMessageId: string,
  setMessages: StreamingMessagesSetter,
  options: StreamingTextSchedulerOptions = {}
): StreamingTextScheduler => {
  const intervalMs = options.intervalMs ?? STREAMING_UPDATE_INTERVAL_MS;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let pending: { text: string; reasoningTime: number } | null = null;
  let lastFlushAt = 0;

  const apply = (text: string, reasoningTime: number) => {
    if (text.length === 0) return;
    if (options.shouldFlush && !options.shouldFlush()) return;
    setMessages((prev) =>
      replaceStreamingMessage(prev, generateMessageId, (message) =>
        updateActiveVariant(message, {
          message: text,
          reasoning_time_taken: reasoningTime,
        })
      )
    );
  };

  const clearTimer = () => {
    if (timer !== null) {
      clearTimeout(timer);
      timer = null;
    }
  };

  const flushNow = () => {
    clearTimer();
    if (pending === null) return;
    const { text, reasoningTime } = pending;
    pending = null;
    lastFlushAt = Date.now();
    apply(text, reasoningTime);
  };

  const schedule = (text: string, reasoningTime: number) => {
    pending = { text, reasoningTime };
    if (timer !== null) return;
    const elapsed = Date.now() - lastFlushAt;
    const delay = Math.max(0, intervalMs - elapsed);
    timer = setTimeout(() => {
      timer = null;
      flushNow();
    }, delay);
  };

  return {
    schedule,
    flushNow,
    cancel: () => {
      clearTimer();
      pending = null;
    },
    disarm: clearTimer,
    hasPending: () => pending !== null,
  };
};
