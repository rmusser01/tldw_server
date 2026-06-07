import type { KnowledgeStatus } from "@/types/connection"

export type KnowledgeReadyRecoveryKind =
  | "ready"
  | "backend_unavailable"
  | "no_indexed_sources"
  | "no_indexed_sources_web_only"
  | "no_selected_sources"
  | "web_only"

export type KnowledgeReadyRecoveryState = {
  kind: KnowledgeReadyRecoveryKind
  hasIndexedSources: boolean
  hasSelectedSources: boolean
  webFallbackAvailable: boolean
  webFallbackEnabled: boolean
  canSearchPersonalLibrary: boolean
  canSearchWebOnly: boolean
  searchBlocked: boolean
}

type ClassifyKnowledgeReadyRecoveryInput = {
  knowledgeStatus: KnowledgeStatus
  selectedSourceCount: number
  webFallbackAvailable: boolean
  webFallbackEnabled: boolean
}

export function classifyKnowledgeReadyRecoveryState({
  knowledgeStatus,
  selectedSourceCount,
  webFallbackAvailable,
  webFallbackEnabled,
}: ClassifyKnowledgeReadyRecoveryInput): KnowledgeReadyRecoveryState {
  const hasIndexedSources = knowledgeStatus !== "empty"
  const hasSelectedSources = selectedSourceCount > 0
  const canSearchWebOnly = webFallbackAvailable && webFallbackEnabled
  const canSearchPersonalLibrary = hasIndexedSources && hasSelectedSources

  if (knowledgeStatus === "offline") {
    return {
      kind: "backend_unavailable",
      hasIndexedSources,
      hasSelectedSources,
      webFallbackAvailable,
      webFallbackEnabled,
      canSearchPersonalLibrary: false,
      canSearchWebOnly: false,
      searchBlocked: true,
    }
  }

  if (!hasIndexedSources) {
    return {
      kind: canSearchWebOnly ? "no_indexed_sources_web_only" : "no_indexed_sources",
      hasIndexedSources: false,
      hasSelectedSources,
      webFallbackAvailable,
      webFallbackEnabled,
      canSearchPersonalLibrary: false,
      canSearchWebOnly,
      searchBlocked: !canSearchWebOnly,
    }
  }

  if (!hasSelectedSources) {
    return {
      kind: canSearchWebOnly ? "web_only" : "no_selected_sources",
      hasIndexedSources,
      hasSelectedSources: false,
      webFallbackAvailable,
      webFallbackEnabled,
      canSearchPersonalLibrary: false,
      canSearchWebOnly,
      searchBlocked: !canSearchWebOnly,
    }
  }

  return {
    kind: "ready",
    hasIndexedSources,
    hasSelectedSources,
    webFallbackAvailable,
    webFallbackEnabled,
    canSearchPersonalLibrary,
    canSearchWebOnly: false,
    searchBlocked: false,
  }
}
