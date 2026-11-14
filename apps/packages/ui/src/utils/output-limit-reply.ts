/**
 * Replies the model cut off at its output limit (UX G06: CM-03).
 *
 * A chat completion stream reports why the model stopped in the final chunk's
 * `finish_reason`. `"length"` means the model ran out of output tokens, so the
 * reply ends mid-thought even though the stream itself finished normally. The
 * chat view marks such a reply as incomplete and offers Continue and Retry.
 */

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value)

/** The `finish_reason` an OpenAI-compatible stream uses for the output limit. */
export const OUTPUT_LIMIT_FINISH_REASON = "length"

/** The `finish_reason` of a stream chunk's first choice, when it carries one. */
export const readChunkFinishReason = (chunk: unknown): string | null => {
  if (!isRecord(chunk) || !Array.isArray(chunk.choices)) return null
  const choice = chunk.choices[0]
  if (!isRecord(choice)) return null
  const raw = choice.finish_reason
  if (typeof raw !== "string") return null
  const reason = raw.trim().toLowerCase()
  return reason.length > 0 ? reason : null
}

/** Whether a `finish_reason` means the model stopped at its output limit. */
export const isOutputLimitFinishReason = (reason: unknown): boolean =>
  typeof reason === "string" &&
  reason.trim().toLowerCase() === OUTPUT_LIMIT_FINISH_REASON

/**
 * The reply's generation info with the output-limit marker added. The marker
 * is the `finish_reason` itself, so it is kept with the reply wherever its
 * generation info is kept.
 */
export const markOutputLimitedReply = (
  generationInfo: unknown
): Record<string, unknown> => ({
  ...(isRecord(generationInfo) ? generationInfo : {}),
  finish_reason: OUTPUT_LIMIT_FINISH_REASON
})

/**
 * Whether a reply's generation info says the model hit its output limit. A
 * reply that was also interrupted shows the interruption instead.
 */
export const isOutputLimitedReply = (generationInfo: unknown): boolean =>
  isRecord(generationInfo) &&
  generationInfo.interrupted !== true &&
  isOutputLimitFinishReason(generationInfo.finish_reason)
