import type { Message } from "@/store/option"

/**
 * Whether the reply bubble a request is filling should offer Stop (G19).
 *
 * A request is in flight from the moment it has a controller, before the
 * server accepts it and streaming starts, and the bubble offers Stop until the
 * request is stopped. Only the reply the request is filling counts: the last
 * row once it is a bot row ending in the "▋" placeholder, which every chat mode
 * uses for the reply it fills. While the request is still being prepared, the
 * previous finished reply is last and is never marked.
 */
export const isReplyRequestPending = (
  controller: AbortController | null | undefined,
  messages: readonly Pick<Message, "isBot" | "message">[]
): boolean => {
  const last = messages[messages.length - 1]
  return Boolean(
    controller && !controller.signal.aborted &&
      last?.isBot && String(last.message ?? "").endsWith("▋")
  )
}
