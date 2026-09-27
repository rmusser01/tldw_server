import type { ChatMessage } from "@/services/tldw/TldwApiClient"
import type { WeightedImagePromptContextEntry } from "@/utils/image-prompt-strategies"

const DEFAULT_SYSTEM_SEMANTICS =
  "You refine image-generation prompts. Preserve intent while improving clarity, visual specificity, and composition."
const DEFAULT_REWRITE_SEMANTICS =
  "Rewrite the prompt to be concise, concrete, and generation-ready."
const LOCKED_OUTPUT_CONTRACT =
  "Output only the final refined prompt as plain text. Do not include markdown, labels, bullets, or commentary."

const normalizeWhitespace = (value: string): string =>
  value.replace(/\s+/g, " ").trim()

const truncate = (value: string, max = 260): string => {
  if (value.length <= max) return value
  return `${value.slice(0, max - 3).trimEnd()}...`
}

const stripCodeFence = (value: string): string => {
  const trimmed = value.trim()
  const fenced = trimmed.match(/^```[a-z0-9_-]*\s*([\s\S]*?)\s*```$/i)
  if (fenced?.[1]) {
    return fenced[1].trim()
  }
  return trimmed
}

const stripPromptPrefix = (value: string): string => {
  return value.replace(/^prompt\s*:\s*/i, "").trim()
}

const normalizeContextEntries = (
  entries: WeightedImagePromptContextEntry[]
): string => {
  if (!Array.isArray(entries) || entries.length === 0) return ""
  return entries
    .slice(0, 4)
    .map((entry) => {
      const label = normalizeWhitespace(String(entry.label || "Context"))
      const text = normalizeWhitespace(String(entry.text || ""))
      const score =
        typeof entry.score === "number" && Number.isFinite(entry.score)
          ? Math.round(entry.score * 100)
          : null
      if (!text) return ""
      return score == null ? `${label}: ${text}` : `${label} (${score}%): ${text}`
    })
    .filter(Boolean)
    .join("\n")
}

export const buildImagePromptRefineMessages = (args: {
  originalPrompt: string
  strategyLabel?: string | null
  backend?: string | null
  contextEntries?: WeightedImagePromptContextEntry[]
  systemSemantics?: string
  rewriteSemantics?: string
}): ChatMessage[] => {
  const normalizedPrompt = normalizeWhitespace(args.originalPrompt || "")
  const strategyLabel = normalizeWhitespace(args.strategyLabel || "Scene")
  const backend = normalizeWhitespace(args.backend || "default")
  const contextSummary = normalizeContextEntries(args.contextEntries || [])
  const contextBlock = contextSummary
    ? `\n\nContext blend cues:\n${contextSummary}`
    : ""

  return [
    {
      role: "system",
      content: `${args.systemSemantics ?? DEFAULT_SYSTEM_SEMANTICS} ${LOCKED_OUTPUT_CONTRACT}`
    },
    {
      role: "user",
      content: [
        `Prompt mode: ${strategyLabel}`,
        `Backend: ${backend}`,
        `Original prompt:\n${truncate(normalizedPrompt, 1200)}`,
        contextBlock,
        args.rewriteSemantics ?? DEFAULT_REWRITE_SEMANTICS
      ]
        .filter(Boolean)
        .join("\n\n")
    }
  ]
}

export const extractImagePromptRefineCandidate = (
  payload: unknown
): string | null => {
  if (!payload || typeof payload !== "object") return null
  const candidatePayload = payload as Record<string, unknown>

  const resolveArrayContent = (value: unknown): string | null => {
    if (!Array.isArray(value)) return null
    const textChunks = value
      .map((entry) => {
        if (!entry || typeof entry !== "object") return ""
        const text = (entry as Record<string, unknown>).text
        return typeof text === "string" ? text : ""
      })
      .filter(Boolean)
    if (textChunks.length === 0) return null
    return textChunks.join("\n").trim()
  }

  const messageContent =
    (candidatePayload?.choices as Array<Record<string, unknown>> | undefined)?.[0]
      ?.message &&
    typeof (candidatePayload.choices as Array<Record<string, unknown>>)[0]
      ?.message === "object"
      ? ((candidatePayload.choices as Array<Record<string, unknown>>)[0]
          ?.message as Record<string, unknown>).content
      : null

  const candidate =
    (typeof messageContent === "string" ? messageContent : null) ??
    resolveArrayContent(messageContent) ??
    (typeof candidatePayload.output === "string"
      ? candidatePayload.output
      : null) ??
    (typeof candidatePayload.content === "string"
      ? candidatePayload.content
      : null) ??
    (typeof candidatePayload.text === "string" ? candidatePayload.text : null)

  if (!candidate || candidate.trim().length === 0) return null
  const unfenced = stripCodeFence(candidate)
  const withoutPrefix = stripPromptPrefix(unfenced)
  const normalized = normalizeWhitespace(withoutPrefix)
  return normalized || null
}
