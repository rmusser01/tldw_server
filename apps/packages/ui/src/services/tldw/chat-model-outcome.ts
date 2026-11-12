/**
 * Record how each chat send ended, for the model chip's readiness (CC-01,
 * #3111). This runs at the transport so every send path is covered: plain
 * chat (/chat/completions), character chat (/chats/{id}/complete-v2, streamed
 * or not) and native-history character sends that stream completions
 * directly. The outcome is keyed by the request's own provider and model.
 */
import {
  isChatModelUnreachableError,
  recordChatModelSendOutcome,
  type ChatModelSendOutcome
} from "@/utils/chat-model-readiness"
import { extractTokenFromChunk } from "@/utils/extract-token-from-chunk"

export type ChatModelOutcomeTarget = {
  provider: unknown
  model: string
}

/**
 * The provider and model a chat request body names: `api_provider` for
 * /chat/completions, `provider` for character-chat completions.
 */
export const chatModelOutcomeTarget = (
  body: unknown
): ChatModelOutcomeTarget | null => {
  if (!body || typeof body !== "object" || Array.isArray(body)) return null
  const record = body as Record<string, unknown>
  const model = typeof record.model === "string" ? record.model.trim() : ""
  if (!model) return null
  return { provider: record.api_provider ?? record.provider, model }
}

const recordOutcome = (
  target: ChatModelOutcomeTarget | null,
  outcome: ChatModelSendOutcome
) => {
  if (target) recordChatModelSendOutcome(target.provider, target.model, outcome)
}

/** A request that came back: the model answered. */
export const recordChatModelAnswered = (target: ChatModelOutcomeTarget | null) =>
  recordOutcome(target, "working")

/**
 * A request that failed. Only a 5xx or a provider-unavailable error marks the
 * model unreachable; validation errors, aborts and timeouts are not evidence
 * either way.
 */
export const recordChatModelFailure = (
  target: ChatModelOutcomeTarget | null,
  error: unknown
) => {
  if (isChatModelUnreachableError(error)) recordOutcome(target, "unreachable")
}

const isErrorFrame = (chunk: unknown): boolean =>
  Boolean(
    chunk &&
      typeof chunk === "object" &&
      !Array.isArray(chunk) &&
      (chunk as Record<string, unknown>).error
  )

const hasAnswerText = (chunk: unknown): boolean => {
  if (!chunk || typeof chunk !== "object") return false
  if (extractTokenFromChunk(chunk).length > 0) return true
  const delta = (chunk as { choices?: Array<{ delta?: { reasoning_content?: unknown } }> })
    .choices?.[0]?.delta?.reasoning_content
  return typeof delta === "string" && delta.length > 0
}

/** Stream lines arrive as raw JSON text; parsed chunks pass through as is. */
const toChunkRecord = (chunk: unknown): unknown => {
  if (typeof chunk !== "string") return chunk
  const trimmed = chunk.trim()
  if (!trimmed.startsWith("{")) return null
  try {
    return JSON.parse(trimmed)
  } catch {
    return null
  }
}

/**
 * Pass a chat stream (raw lines or parsed chunks) through unchanged while
 * recording its outcome: answer text marks the model working, a provider error
 * frame or a thrown provider error marks it unreachable. The latest evidence
 * wins. Once the model has answered, only lines that mention an error are
 * parsed, so the token path stays cheap.
 */
export async function* trackChatModelStreamOutcome<T>(
  target: ChatModelOutcomeTarget | null,
  stream: AsyncIterable<T>
): AsyncGenerator<T, void, unknown> {
  let answered = false
  try {
    for await (const chunk of stream) {
      const inspect =
        target !== null &&
        (!answered || typeof chunk !== "string" || chunk.includes('"error"'))
      if (inspect) {
        const frame = toChunkRecord(chunk)
        if (isErrorFrame(frame)) {
          recordChatModelFailure(target, frame)
          answered = false
        } else if (!answered && hasAnswerText(frame)) {
          answered = true
          recordChatModelAnswered(target)
        }
      }
      yield chunk
    }
  } catch (error) {
    recordChatModelFailure(target, error)
    throw error
  }
}

/** Run a non-streaming chat request and record how it ended. */
export const trackChatModelRequestOutcome = async <T>(
  target: ChatModelOutcomeTarget | null,
  run: () => Promise<T>
): Promise<T> => {
  try {
    const result = await run()
    recordChatModelAnswered(target)
    return result
  } catch (error) {
    recordChatModelFailure(target, error)
    throw error
  }
}
