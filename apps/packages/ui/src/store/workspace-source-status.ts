import type { WorkspaceSource } from "@/types/workspace"

export const isWorkspaceSourcePartiallyQueryable = (
  source: WorkspaceSource
): boolean => {
  const readiness = source.readiness
  return (
    (source.status || "ready") === "processing" &&
    source.statusDetails?.lifecycleState === "partially_queryable" &&
    Boolean(readiness?.text_extracted) &&
    Boolean(readiness?.fts_ready) &&
    Boolean(readiness?.tool_accessible)
  )
}

export const isWorkspaceSourceSelectable = (
  source: WorkspaceSource
): boolean =>
  (source.status || "ready") === "ready" ||
  isWorkspaceSourcePartiallyQueryable(source)
