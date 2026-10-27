import { extractTokenFromChunk } from "@/utils/extract-token-from-chunk"

export type StreamingChunk =
  | string
  | {
      content?: string
      error?: unknown
      errors?: unknown
      event?: string
      type?: string
      message?: string
      detail?: string
      choices?: Array<{
        delta?: {
          content?: string
          reasoning_content?: string
        }
      }>
      additional_kwargs?: {
        reasoning_content?: string
      }
    }

const STREAMING_ERROR_FALLBACK_MESSAGE = "Streaming request failed."

type StreamingChunkErrorInfo = {
  message: string
  code?: string
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value)

const readFirstNonEmptyString = (
  value: unknown,
  depth = 0
): string | null => {
  if (depth > 4 || value == null) return null
  if (typeof value === "string") {
    const trimmed = value.trim()
    return trimmed.length > 0 ? trimmed : null
  }
  if (Array.isArray(value)) {
    for (const entry of value) {
      const nested = readFirstNonEmptyString(entry, depth + 1)
      if (nested) return nested
    }
    return null
  }
  if (!isRecord(value)) return null

  const directKeys = [
    "message",
    "detail",
    "description",
    "reason",
    "error_message",
    "error_description"
  ]
  for (const key of directKeys) {
    const text = readFirstNonEmptyString(value[key], depth + 1)
    if (text) return text
  }

  const nestedKeys = ["error", "errors", "details", "cause"]
  for (const key of nestedKeys) {
    const nested = readFirstNonEmptyString(value[key], depth + 1)
    if (nested) return nested
  }

  return null
}

const readFirstErrorCode = (value: unknown, depth = 0): string | undefined => {
  if (depth > 4 || value == null) return undefined
  if (typeof value === "string") {
    const trimmed = value.trim()
    return trimmed.length > 0 ? trimmed : undefined
  }
  if (Array.isArray(value)) {
    for (const entry of value) {
      const nested = readFirstErrorCode(entry, depth + 1)
      if (nested) return nested
    }
    return undefined
  }
  if (!isRecord(value)) return undefined

  const directKeys = ["code", "error_code", "type"]
  for (const key of directKeys) {
    const code = readFirstErrorCode(value[key], depth + 1)
    if (code) return code
  }

  const nestedKeys = ["error", "errors", "details", "cause"]
  for (const key of nestedKeys) {
    const nested = readFirstErrorCode(value[key], depth + 1)
    if (nested) return nested
  }

  return undefined
}

export const extractStreamingChunkError = (
  chunk: unknown
): StreamingChunkErrorInfo | null => {
  if (!isRecord(chunk)) return null

  const eventValue =
    typeof chunk.event === "string" ? chunk.event.toLowerCase() : ""
  const typeValue =
    typeof chunk.type === "string" ? chunk.type.toLowerCase() : ""

  const hasExplicitErrorContainer =
    Object.prototype.hasOwnProperty.call(chunk, "error") ||
    Object.prototype.hasOwnProperty.call(chunk, "errors")

  const hasExplicitErrorEvent =
    eventValue === "error" ||
    typeValue === "error" ||
    eventValue.endsWith("_error")

  if (!hasExplicitErrorContainer && !hasExplicitErrorEvent) {
    return null
  }

  const message =
    readFirstNonEmptyString(
      hasExplicitErrorContainer ? chunk.error ?? chunk.errors : chunk
    ) ??
    readFirstNonEmptyString(chunk) ??
    STREAMING_ERROR_FALLBACK_MESSAGE

  const code = readFirstErrorCode(
    hasExplicitErrorContainer ? chunk.error ?? chunk.errors : chunk
  )

  return { message, code }
}

type StreamingAccumulator = {
  fullText: string
  contentToSave: string
  apiReasoning: boolean
}

type StreamingChunkResult = StreamingAccumulator & {
  token: string
}

export const consumeStreamingChunk = (
  state: StreamingAccumulator,
  chunk: StreamingChunk
): StreamingChunkResult => {
  let { fullText, contentToSave, apiReasoning } = state
  const streamError = extractStreamingChunkError(chunk)
  if (streamError) {
    const error = new Error(streamError.message) as Error & { code?: string }
    if (streamError.code) {
      error.code = streamError.code
    }
    throw error
  }
  const token = extractTokenFromChunk(chunk)
  const reasoningDelta =
    typeof chunk === "string"
      ? undefined
      : chunk?.choices?.[0]?.delta?.reasoning_content ??
        chunk?.additional_kwargs?.reasoning_content

  if (reasoningDelta) {
    const reasoningText =
      typeof reasoningDelta === "string" ? reasoningDelta : ""
    const reasoningContent = `${apiReasoning ? "" : "<think>"}${reasoningText}`
    fullText += reasoningContent
    contentToSave += reasoningContent
    apiReasoning = true
  }
  if (apiReasoning && token) {
    fullText += "</think>"
    contentToSave += "</think>"
    apiReasoning = false
  }

  if (token) {
    fullText += token
    contentToSave += token
  }

  return { fullText, contentToSave, apiReasoning, token }
}
