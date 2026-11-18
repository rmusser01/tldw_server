import type { WorkspaceSource, WorkspaceSourceStatusDetails } from "@/types/workspace"

/** An unavailable lookup cannot revoke the user's selection intent. */
export const isWorkspaceSourceAvailabilityUnknown = (
  details: WorkspaceSourceStatusDetails | undefined
): boolean => details?.lifecycleState === "unknown" || details?.lifecycleState === "missing_media"

export const isTransientSourceStatusError = (
  error: unknown
): { transient: boolean; message: string } => {
  const status = (error as { status?: number } | null)?.status
  const message =
    error instanceof Error ? error.message : String(error ?? "Unknown error")
  const transient =
    status === 0 ||
    status === 404 ||
    status === 408 ||
    status === 429 ||
    // A failed server lookup cannot establish that the source itself failed.
    (typeof status === "number" && status >= 500 && status < 600) ||
    /network|timeout|abort|failed to fetch|fetch failed|connection|backend unavailable/i.test(message)

  return { transient, message }
}

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

/** A failed optional vector index is a warning, not an operation still running. */
export const isWorkspaceSourceActivelyProcessing = (
  source: WorkspaceSource
): boolean =>
  source.status === "processing" &&
  !(source.statusDetails?.lifecycleState === "partially_queryable" &&
    source.statusDetails.statusReason === "vector_index_failed")
