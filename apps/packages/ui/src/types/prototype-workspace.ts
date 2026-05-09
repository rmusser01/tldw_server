export interface PrototypeWorkspaceCreateInput {
  title: string
  creation_source: string
  description?: string | null
  prompt?: string | null
  preview_policy?: Record<string, unknown>
  share_policy?: Record<string, unknown>
  runtime_policy?: Record<string, unknown>
  designated_promoter_ids?: number[]
}

export interface PrototypeWorkspace {
  id: string
  owner_user_id: number
  title: string
  description?: string | null
  creation_source: string
  canonical_snapshot_id?: string | null
  last_known_good_snapshot_id?: string | null
  canonical_preview_status?: string | null
  publish_validation_status?: string | null
  preview_policy: Record<string, unknown>
  share_policy: Record<string, unknown>
  runtime_policy: Record<string, unknown>
  designated_promoter_ids: number[]
  created_at: string
  updated_at: string
  archived_at?: string | null
  is_archived: boolean
}

export type PrototypeWorkspaceViewerRole = "owner" | "internal_collaborator"

export interface PrototypeWorkspaceSessionSummary {
  id: string
  prototype_workspace_id: string
  base_snapshot_id: string
  actor_user_id?: number | null
  actor_shared_actor_id?: string | null
  actor_type: string
  share_link_id?: number | null
  acp_session_id?: string | null
  sandbox_session_id?: string | null
  sandbox_run_id?: string | null
  runtime_status: string
  preview_handle?: string | null
  preview_status: string
  last_saved_snapshot_id?: string | null
  last_activity_at?: string | null
  expires_at?: string | null
  revoked_at?: string | null
  created_at: string
  updated_at: string
  is_revoked: boolean
}

export interface PrototypeWorkspaceSnapshotSummary {
  snapshot_id: string
  prototype_workspace_id: string
  parent_snapshot_id?: string | null
  created_from_session_id?: string | null
  author_user_id?: number | null
  author_shared_actor_id?: string | null
  storage_ref?: string | null
  diff_summary: Record<string, unknown>
  prompt_summary?: string | null
  preview_health: Record<string, unknown>
  created_at: string
  is_canonical: boolean
  is_last_known_good: boolean
}

export interface PrototypeWorkspaceDetail extends PrototypeWorkspace {
  viewer_role: PrototypeWorkspaceViewerRole
  sessions: PrototypeWorkspaceSessionSummary[]
  snapshots: PrototypeWorkspaceSnapshotSummary[]
}

export interface PrototypeWorkspaceSessionCreateInput {
  request_nonce?: string | null
}

export interface PrototypeCollaboratorSessionCreateInput {
  session_token: string
  request_nonce?: string | null
}

export interface PrototypeSessionJob {
  job_id: string
  job_type: "branch_session_bootstrap"
  status: string
  message: string
  prototype_workspace_id: string
  prototype_session_id: string
  actor_type: string
  shared_actor_id?: string | null
  idempotency_key?: string | null
}

export interface PrototypePromotionCreateInput {
  prototype_workspace_id: string
  prototype_session_id: string
  candidate_snapshot_id: string
  session_token: string
  request_reason?: string | null
}

export interface PrototypePromotionRequest {
  id: string
  prototype_workspace_id: string
  prototype_session_id: string
  candidate_snapshot_id: string
  requested_by_user_id?: number | null
  requested_by_shared_actor_id?: string | null
  status: string
  reviewed_by_user_id?: number | null
  review_notes?: string | null
  created_at: string
  updated_at: string
}
