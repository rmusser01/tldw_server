import type { WorkspaceSource } from "@/types/workspace"
import type { StagedSourceAvailability, StagedWorkspaceSource } from "./types"

const toAvailability = (source: WorkspaceSource): StagedSourceAvailability => {
  if (source.status === "processing") return "processing"
  if (source.status === "error") return "error"
  if (source.status === "ready" || !source.status) return "ready"
  return "unavailable"
}

export const buildStagedSourceFromWorkspaceSource = (
  source: WorkspaceSource,
  scopeLabel: string
): StagedWorkspaceSource => ({
  sourceId: source.id,
  mediaId: Number.isInteger(source.mediaId) && source.mediaId > 0 ? source.mediaId : null,
  title: source.title,
  type: source.type,
  scopeLabel,
  availability: toAvailability(source),
  statusMessage: source.statusMessage
})

export const stageWorkspaceSources = (
  existing: StagedWorkspaceSource[],
  sources: WorkspaceSource[],
  scopeLabel: string
): StagedWorkspaceSource[] => {
  const byId = new Map(existing.map((item) => [item.sourceId, item]))
  for (const source of sources) {
    byId.set(source.id, buildStagedSourceFromWorkspaceSource(source, scopeLabel))
  }
  return Array.from(byId.values())
}

const formatRowValueForInsert = (
  value: string,
  fallback: string,
  maxLength = 96
): string => {
  const firstLine = value.split(/\r?\n/)[0] ?? ""
  const collapsed = firstLine
    .replace(/[`*_#[\]()>|{}]/g, "")
    .replace(/\s+/g, " ")
    .trim()

  if (collapsed.length <= maxLength) {
    return collapsed || fallback
  }

  return `${collapsed.slice(0, maxLength - 3).trimEnd()}...`
}

export const formatStagedSourceInsertText = (
  sources: StagedWorkspaceSource[]
): string => {
  if (sources.length === 0) return ""
  const lines = sources.map((source, index) => {
    const state = source.availability === "ready" ? "" : ` (${source.availability})`
    const title = formatRowValueForInsert(source.title, "Untitled source")
    const scopeLabel = formatRowValueForInsert(source.scopeLabel, "Unknown scope", 64)
    return `${index + 1}. ${title} [${source.type}, scope: ${scopeLabel}]${state}`
  })
  return `Context sources:\n${lines.join("\n")}\n\n`
}

export const getReadyStagedMediaIds = (
  sources: StagedWorkspaceSource[]
): number[] =>
  Array.from(
    new Set(
      sources
        .filter((source) => source.availability === "ready")
        .map((source) => source.mediaId)
        .filter((mediaId): mediaId is number => Number.isInteger(mediaId) && mediaId > 0)
    )
  )
