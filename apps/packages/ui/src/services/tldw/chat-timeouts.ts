/** First visible output uses the same default as Settings' Balanced preset. */
export const CHAT_STARTUP_TIMEOUT_DEFAULT_MS = 120_000

/** A local Chat watchdog timeout, distinct from Stop or a provider failure. */
export class ChatStreamTimeoutError extends Error {
  constructor(
    readonly phase: "startup" | "idle",
    readonly timeoutMs: number
  ) {
    const seconds = timeoutMs / 1_000
    super(phase === "startup"
      ? `Chat response timed out after ${seconds} seconds before any visible output arrived.`
      : `Chat response stalled for ${seconds} seconds after visible output began.`)
    this.name = "ChatStreamTimeoutError"
  }
}
