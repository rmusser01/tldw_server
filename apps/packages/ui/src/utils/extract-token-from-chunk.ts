const extractText = (value: unknown, depth: number = 0): string => {
  if (depth > 8) return ""
  if (typeof value === "string") return value
  if (Array.isArray(value)) {
    return value.map((v) => extractText(v, depth + 1)).filter(Boolean).join("")
  }
  if (value && typeof value === "object") {
    const record = value as Record<string, unknown>
    const text = extractText(record.text, depth + 1)
    if (text) return text
    const content = extractText(record.content, depth + 1)
    if (content) return content
    const parts = extractText(record.parts, depth + 1)
    if (parts) return parts
  }
  return ""
}

/**
 * Detect the synthesized `stream_transport_interrupted` sentinel that the
 * background stream proxy emits when the extension port drops AFTER the first
 * byte. It carries no assistant text (so `extractTokenFromChunk` returns ""),
 * which is why it must be recognized separately and propagated to the chat
 * pipeline so a truncated answer is finalized as interrupted, not complete.
 */
export function extractStreamTransportInterruption(
  chunk: unknown
): { detail: string | null } | null {
  if (!chunk || typeof chunk !== "object" || Array.isArray(chunk)) return null
  const record = chunk as Record<string, unknown>
  const event =
    typeof record.event === "string" ? record.event.toLowerCase() : ""
  if (event !== "stream_transport_interrupted") return null
  const detail =
    typeof record.detail === "string" && record.detail.trim().length > 0
      ? record.detail.trim()
      : null
  return { detail }
}

export function extractTokenFromChunk(chunk: unknown): string {
  if (typeof chunk === "string") return chunk
  if (!chunk || typeof chunk !== "object") return ""

  const record = chunk as Record<string, unknown>
  const choices = Array.isArray(record.choices) ? record.choices : []
  if (choices.length > 0) {
    const choice = choices[0] as Record<string, unknown>
    const deltaText = extractText(choice.delta ?? choice.message)
    if (deltaText) return deltaText
    const choiceText = extractText(choice.text ?? choice.content ?? choice)
    if (choiceText) return choiceText
  }

  const candidates = Array.isArray(record.candidates) ? record.candidates : []
  if (candidates.length > 0) {
    const candidate = candidates[0] as Record<string, unknown>
    const candidateText = extractText(
      candidate.content ?? candidate.message ?? candidate.output ?? candidate.parts ?? candidate.text
    )
    if (candidateText) return candidateText
  }

  const rootText = extractText(
    record.delta ??
      record.content ??
      record.message ??
      record.text ??
      record.completion ??
      record.output_text
  )
  return rootText
}
