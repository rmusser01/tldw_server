import { normalizeQuickChatRoutePath } from "./workflow-guides"

type QuickChatDocsRagRequestInput = {
  query: string
  currentRoute?: string | null
  scope?: QuickChatDocsScopeConfig
}

export type QuickChatDocsScopeConfig = {
  strictProjectDocsOnly?: boolean
  projectDocsNamespace?: string | null
  projectDocsMediaIds?: number[]
}

export type QuickChatDocsRagRequestProfile = {
  query: string
  options: {
    top_k: number
    search_mode: "hybrid"
    enable_generation: true
    enable_citations: true
    enable_reranking: true
    reranking_strategy: "flashrank"
    sources: string[]
    corpus?: string
    index_namespace?: string
    include_media_ids?: number[]
    min_score: number
    fts_level: "chunk" | "document"
    enable_parent_expansion: true
    parent_context_size: number
    include_sibling_chunks: true
    sibling_window: number
    include_parent_document: boolean
    parent_max_tokens: number
    max_generation_tokens: number
    timeoutMs: number
  }
}

const CURRENT_PAGE_PATTERN =
  /\b(this page|current page|this screen|here|this tab)\b/i

const SYNOPSIS_PATTERN =
  /\b(synopsis|summary|summarize|summarise|overview|abstract|tldr|tl;dr)\b/i

const TROUBLESHOOT_PATTERN =
  /\b(error|broken|issue|not working|fails?|failed|fix|debug|troubleshoot)\b/i

export const QUICK_CHAT_DEFAULT_PROJECT_DOCS_NAMESPACE = "project_docs"

export const normalizeQuickChatDocsMediaIds = (value: unknown): number[] => {
  if (Array.isArray(value)) {
    return value
      .map((item) => Number(item))
      .filter((item) => Number.isFinite(item) && item > 0)
      .map((item) => Math.trunc(item))
  }
  if (typeof value === "string") {
    const trimmed = value.trim()
    if (!trimmed) return []
    try {
      const parsed = JSON.parse(trimmed) as unknown
      return normalizeQuickChatDocsMediaIds(parsed)
    } catch {
      return trimmed
        .split(",")
        .map((item) => Number(item.trim()))
        .filter((item) => Number.isFinite(item) && item > 0)
        .map((item) => Math.trunc(item))
    }
  }
  return []
}

export const toQuickChatDocsMediaIdsInputValue = (value: unknown): string => {
  if (typeof value === "string") return value
  const ids = normalizeQuickChatDocsMediaIds(value)
  return ids.join(", ")
}

const ROUTE_LABELS: Record<string, string> = {
  "/research-workspace": "Research Workspace",
  "/media": "Media",
  "/knowledge": "Knowledge",
  "/characters": "Characters",
  "/world-books": "World Books",
  "/prompts": "Prompts",
  "/evaluations": "Evaluations",
  "/notes": "Notes",
  "/flashcards": "Flashcards",
  "/settings/health": "Health & Diagnostics"
}

const buildContextualizedQuery = (
  query: string,
  normalizedRoute: string | null
): string => {
  if (!normalizedRoute || !CURRENT_PAGE_PATTERN.test(query)) {
    return query
  }
  const routeLabel = ROUTE_LABELS[normalizedRoute] || normalizedRoute
  const hint = `Current page context: ${routeLabel} (${normalizedRoute}).`
  return `${query}\n\n${hint}`
}

export const buildQuickChatDocsRagProfile = ({
  query,
  currentRoute = null,
  scope
}: QuickChatDocsRagRequestInput): QuickChatDocsRagRequestProfile => {
  const normalizedRoute = normalizeQuickChatRoutePath(currentRoute)
  const contextualizedQuery = buildContextualizedQuery(query, normalizedRoute)
  const isSynopsis = SYNOPSIS_PATTERN.test(query)
  const isTroubleshoot = TROUBLESHOOT_PATTERN.test(query)
  const strictProjectDocsOnly = scope?.strictProjectDocsOnly !== false
  const namespace = (scope?.projectDocsNamespace || "").trim()
  const resolvedNamespace =
    namespace.length > 0 ? namespace : QUICK_CHAT_DEFAULT_PROJECT_DOCS_NAMESPACE
  const includeMediaIds = normalizeQuickChatDocsMediaIds(
    scope?.projectDocsMediaIds
  )

  const sources = strictProjectDocsOnly ? ["media_db"] : ["media_db", "notes"]

  return {
    query: contextualizedQuery,
    options: {
      top_k: isSynopsis ? 10 : 7,
      search_mode: "hybrid",
      enable_generation: true,
      enable_citations: true,
      enable_reranking: true,
      reranking_strategy: "flashrank",
      sources,
      ...(strictProjectDocsOnly
        ? {
            corpus: "media_db",
            index_namespace: resolvedNamespace,
            ...(includeMediaIds.length > 0
              ? { include_media_ids: includeMediaIds }
              : {})
          }
        : {}),
      min_score: isTroubleshoot ? 0.1 : 0.18,
      fts_level: isSynopsis ? "document" : "chunk",
      enable_parent_expansion: true,
      parent_context_size: isSynopsis ? 900 : 600,
      include_sibling_chunks: true,
      sibling_window: 1,
      include_parent_document: isSynopsis,
      parent_max_tokens: isSynopsis ? 1600 : 900,
      max_generation_tokens: isSynopsis ? 950 : 700,
      timeoutMs: 45_000
    }
  }
}
