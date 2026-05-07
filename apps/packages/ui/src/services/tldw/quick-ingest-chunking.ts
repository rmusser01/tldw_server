import { resolvePerformChunking } from "@/services/tldw/ingest-defaults"

export type QuickIngestChunkingMode = "auto" | "manual"
export type QuickIngestAutoChunkingGoal =
  | "balanced"
  | "qa_search"
  | "navigation_summary"

export type QuickIngestChunkingCommon = {
  perform_chunking?: boolean
  chunking_mode?: QuickIngestChunkingMode | string
  auto_chunking_goal?: QuickIngestAutoChunkingGoal | string
  auto_chunking_use_llm?: boolean
}

export const DEFAULT_QUICK_INGEST_CHUNKING_MODE: QuickIngestChunkingMode =
  "auto"
export const DEFAULT_QUICK_INGEST_AUTO_CHUNKING_GOAL: QuickIngestAutoChunkingGoal =
  "balanced"

const AUTO_CHUNKING_FIELDS = new Set([
  "chunking_mode",
  "auto_chunking_goal",
  "auto_chunking_use_llm",
])

const MANUAL_CHUNKING_FIELDS = new Set([
  "auto_apply_template",
  "chunk_language",
  "chunk_method",
  "chunk_overlap",
  "chunk_size",
  "chunking_template_name",
  "context_strategy",
  "context_token_budget",
  "context_window_size",
  "contextual_llm_model",
  "custom_chapter_pattern",
  "enable_contextual_chunking",
  "hierarchical_chunking",
  "hierarchical_template",
  "use_adaptive_chunking",
  "use_multi_level_chunking",
])

const CONTROLLED_COMMON_FIELDS = new Set([
  "perform_analysis",
  "perform_chunking",
  "overwrite_existing",
])

const fieldNameParts = (fieldName: string): string[] =>
  String(fieldName || "")
    .split(".")
    .map((part) => part.trim())
    .filter(Boolean)

const matchesField = (fieldName: string, names: Set<string>): boolean => {
  const parts = fieldNameParts(fieldName)
  return parts.some((part) => names.has(part))
}

export const resolveQuickIngestChunkingMode = (
  value: QuickIngestChunkingCommon["chunking_mode"],
): QuickIngestChunkingMode =>
  value === "manual" ? "manual" : DEFAULT_QUICK_INGEST_CHUNKING_MODE

export const resolveQuickIngestAutoChunkingGoal = (
  value: QuickIngestChunkingCommon["auto_chunking_goal"],
): QuickIngestAutoChunkingGoal => {
  if (value === "qa_search" || value === "navigation_summary") {
    return value
  }
  return DEFAULT_QUICK_INGEST_AUTO_CHUNKING_GOAL
}

export const shouldSubmitQuickIngestAdvancedField = (
  fieldName: string,
  common?: QuickIngestChunkingCommon,
): boolean => {
  if (matchesField(fieldName, CONTROLLED_COMMON_FIELDS)) return false
  if (matchesField(fieldName, AUTO_CHUNKING_FIELDS)) return false

  const performChunking = resolvePerformChunking(common?.perform_chunking)
  const mode = resolveQuickIngestChunkingMode(common?.chunking_mode)
  if (matchesField(fieldName, MANUAL_CHUNKING_FIELDS)) {
    return performChunking && mode === "manual"
  }

  return true
}

export const applyQuickIngestChunkingFields = (
  target: Record<string, any>,
  options: {
    common?: QuickIngestChunkingCommon
    chunkingTemplateName?: string
    autoApplyTemplate?: boolean
  } = {},
): Record<string, any> => {
  const common = options.common
  const performChunking = resolvePerformChunking(common?.perform_chunking)
  target.perform_chunking = performChunking

  if (!performChunking) {
    return target
  }

  const mode = resolveQuickIngestChunkingMode(common?.chunking_mode)
  target.chunking_mode = mode

  if (mode === "auto") {
    target.auto_chunking_goal = resolveQuickIngestAutoChunkingGoal(
      common?.auto_chunking_goal,
    )
    if (common?.auto_chunking_use_llm === true) {
      target.auto_chunking_use_llm = true
    }
    return target
  }

  if (options.chunkingTemplateName) {
    target.chunking_template_name = options.chunkingTemplateName
  }
  if (options.autoApplyTemplate) {
    target.auto_apply_template = true
  }
  return target
}
