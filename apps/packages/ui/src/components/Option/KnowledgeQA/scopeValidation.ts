import type { RagSource } from "@/services/rag/unified-rag"
import type { RagResult } from "./types"
import {
  getEvidenceOrigin,
  getResultSourceId,
} from "./sourceListUtils"

const RAG_SOURCE_TYPES = new Set<string>([
  "media_db",
  "notes",
  "chats",
  "characters",
  "kanban",
  "prompts",
  "world_books",
  "dictionaries",
])

const MEDIA_CONTENT_SOURCE_TYPES = new Set<string>([
  "media",
  "document",
  "pdf",
  "transcript",
  "video",
  "audio",
  "epub",
  "ebook",
  "html",
  "markdown",
  "xml",
])

export type KnowledgeScopeViolationReason = "excluded_source"

export type KnowledgeScopeViolation = {
  index: number
  sourceId: string
  sourceType: string
  reason: KnowledgeScopeViolationReason
}

export type KnowledgeResultScopeValidationInput = {
  selectedSources: RagSource[]
  selectedMediaIds: number[]
  selectedNoteIds: string[]
  webFallbackEnabled: boolean
  results: RagResult[]
}

export type KnowledgeResultScopeValidation = {
  acceptedResults: RagResult[]
  violations: KnowledgeScopeViolation[]
}

type SourceCategory = RagSource | "web" | null

function normalizeSourceType(result: RagResult): string | null {
  const candidates = [
    result.sourceType,
    result.metadata?.source_type,
  ]
  for (const candidate of candidates) {
    if (typeof candidate === "string" && candidate.trim().length > 0) {
      return candidate.trim().toLowerCase()
    }
  }
  return null
}

function sourceCategoryForType(sourceType: string | null): SourceCategory {
  if (!sourceType) return null
  if (sourceType === "note") return "notes"
  if (sourceType === "web") return "web"
  if (MEDIA_CONTENT_SOURCE_TYPES.has(sourceType)) return "media_db"
  if (RAG_SOURCE_TYPES.has(sourceType)) return sourceType as RagSource
  return null
}

function normalizeMediaId(value: string | null): number | null {
  if (!value || !/^\d+$/.test(value)) return null
  const parsed = Number.parseInt(value, 10)
  return Number.isFinite(parsed) ? parsed : null
}

function hasExplicitScopeBroadening(result: RagResult): boolean {
  const metadata = result.metadata ?? {}
  return (
    metadata.scope_broadened_by_workspace === true ||
    metadata.scope_broadened_reason === "scope_broadened_by_workspace" ||
    metadata.scope_broadening_reason === "scope_broadened_by_workspace" ||
    metadata.scope_reason === "scope_broadened_by_workspace"
  )
}

function isWebFallbackResult(
  result: RagResult,
  sourceCategory: SourceCategory
): boolean {
  return getEvidenceOrigin(result) === "web_fallback" || sourceCategory === "web"
}

function sourceTypeMatchesSelectedCategory(
  sourceCategory: SourceCategory,
  selectedSources: RagSource[]
): boolean {
  if (!sourceCategory || selectedSources.length === 0) return true
  if (sourceCategory === "web") return false
  return selectedSources.includes(sourceCategory)
}

function isExactSourceAllowed(
  result: RagResult,
  sourceCategory: SourceCategory,
  selectedMediaIds: number[],
  selectedNoteIds: string[]
): boolean {
  const sourceId = getResultSourceId(result)
  if (sourceCategory === "media_db" && selectedMediaIds.length > 0) {
    const mediaId = normalizeMediaId(sourceId)
    return mediaId != null && selectedMediaIds.includes(mediaId)
  }

  if (sourceCategory === "notes" && selectedNoteIds.length > 0) {
    return sourceId != null && selectedNoteIds.includes(sourceId)
  }

  return true
}

function withOriginalResultIndex(result: RagResult, index: number): RagResult {
  return {
    ...result,
    metadata: {
      ...(result.metadata ?? {}),
      original_result_index: index,
    },
  }
}

export function validateKnowledgeResultScope({
  selectedSources,
  selectedMediaIds,
  selectedNoteIds,
  webFallbackEnabled,
  results,
}: KnowledgeResultScopeValidationInput): KnowledgeResultScopeValidation {
  const violations: KnowledgeScopeViolation[] = []
  const acceptedResults: RagResult[] = []

  results.forEach((result, index) => {
    const sourceType = normalizeSourceType(result)
    const sourceCategory = sourceCategoryForType(sourceType)
    const sourceId = getResultSourceId(result) ?? "unknown"
    const explicitScopeBroadening = hasExplicitScopeBroadening(result)
    const webFallbackResult = isWebFallbackResult(result, sourceCategory)
    const sourceAllowed =
      explicitScopeBroadening ||
      (webFallbackResult && webFallbackEnabled) ||
      (!webFallbackResult &&
        sourceTypeMatchesSelectedCategory(sourceCategory, selectedSources) &&
        isExactSourceAllowed(
          result,
          sourceCategory,
          selectedMediaIds,
          selectedNoteIds
        ))

    if (sourceAllowed) {
      acceptedResults.push(withOriginalResultIndex(result, index))
      return
    }

    violations.push({
      index,
      sourceId,
      sourceType: sourceType ?? "unknown",
      reason: "excluded_source",
    })
  })

  return {
    acceptedResults:
      violations.length === 0
        ? results
        : acceptedResults,
    violations,
  }
}
