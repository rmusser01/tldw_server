export const operationId = "13f28c88-0f13-4b19-80e8-87ddc27bf22b"
export const workspaceId = "24f28c88-0f13-4b19-80e8-87ddc27bf22b"
export const clonePayload = (status = "queued", shareId = 42) => ({
  schema_version: 1,
  operation_id: operationId,
  workspace_id: workspaceId,
  command: "shared_workspace_clone",
  share_id: shareId,
  status,
  started_at: "2026-09-13T10:00:00Z",
  updated_at: "2026-09-13T10:00:01Z",
  retryable: status === "failed",
  diagnostics: {},
  poll_href: `/api/v1/sharing/shared-with-me/${shareId}/clone/${operationId}`,
  progress:
    status === "queued" || status === "running"
      ? { phase: "sources", percent: 40, message_code: "copying_sources" }
      : null,
  result:
    status === "succeeded"
      ? {
          schema_version: 1,
          outcome: "complete",
          workspace_id: workspaceId,
          name: "Research (Copy)",
          publication_confirmed: true,
          counts: {
            sources_attempted: 2,
            sources_copied: 2,
            sources_failed: 0,
            notes_attempted: 0,
            notes_copied: 0,
            notes_failed: 0,
            artifacts_attempted: 0,
            artifacts_copied: 0,
            artifacts_failed: 0,
            media_attempted: 2,
            media_copied: 2,
            media_failed: 0,
            operation_owned_media_count: 2
          },
          readiness: {
            text_search: "ready",
            citations: "ready",
            vector_search: "needs_indexing"
          },
          warnings: []
        }
      : null,
  error:
    status === "failed"
      ? {
          code: "clone_failed",
          message_key: "sharing.clone.failed",
          message: "Copy could not be completed.",
          cleanup_state: "complete"
        }
      : null
})
