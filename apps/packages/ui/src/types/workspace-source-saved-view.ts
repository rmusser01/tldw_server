import type {
  WorkspaceSourceLifecycleState,
  WorkspaceSourceReviewState,
  WorkspaceSourceStatus,
  WorkspaceSourceType
} from "./workspace"

export const WORKSPACE_SOURCE_SAVED_VIEW_TYPE_FILTERS = [
  "pdf",
  "video",
  "audio",
  "website",
  "document",
  "text"
] as const satisfies readonly WorkspaceSourceType[]

export type WorkspaceSourceSavedViewTypeFilter =
  (typeof WORKSPACE_SOURCE_SAVED_VIEW_TYPE_FILTERS)[number]

export const WORKSPACE_SOURCE_SAVED_VIEW_STATUS_FILTERS = [
  "processing",
  "ready",
  "error"
] as const satisfies readonly WorkspaceSourceStatus[]

export type WorkspaceSourceSavedViewStatusFilter =
  (typeof WORKSPACE_SOURCE_SAVED_VIEW_STATUS_FILTERS)[number]

export const WORKSPACE_SOURCE_SAVED_VIEW_REVIEW_STATE_FILTERS = [
  "unset",
  "needs_review",
  "reviewed"
] as const satisfies readonly WorkspaceSourceReviewState[]

export type WorkspaceSourceSavedViewReviewStateFilter =
  (typeof WORKSPACE_SOURCE_SAVED_VIEW_REVIEW_STATE_FILTERS)[number]

export const WORKSPACE_SOURCE_SAVED_VIEW_LIFECYCLE_STATE_FILTERS = [
  "queued",
  "ingesting",
  "extracting",
  "chunking",
  "indexing",
  "queryable",
  "partially_queryable",
  "failed",
  "retrying",
  "missing_media",
  "blocked_by_permissions",
  "unknown"
] as const satisfies readonly WorkspaceSourceLifecycleState[]

export type WorkspaceSourceSavedViewLifecycleStateFilter =
  (typeof WORKSPACE_SOURCE_SAVED_VIEW_LIFECYCLE_STATE_FILTERS)[number]

export const WORKSPACE_SOURCE_SAVED_VIEW_DATE_FIELDS = [
  "added_at",
  "source_created_at"
] as const

export type WorkspaceSourceSavedViewDateField =
  (typeof WORKSPACE_SOURCE_SAVED_VIEW_DATE_FIELDS)[number]

export const WORKSPACE_SOURCE_SAVED_VIEW_SORTS = [
  "manual",
  "name_asc",
  "name_desc",
  "added_desc",
  "added_asc",
  "source_created_desc",
  "source_created_asc",
  "file_size_desc",
  "file_size_asc",
  "duration_desc",
  "duration_asc",
  "page_count_desc",
  "page_count_asc"
] as const

export type WorkspaceSourceSavedViewSort =
  (typeof WORKSPACE_SOURCE_SAVED_VIEW_SORTS)[number]

export const WORKSPACE_SOURCE_SAVED_VIEW_INVALID_REASONS = [
  "invalid_json",
  "invalid_state",
  "unsupported_schema_version"
] as const

export type WorkspaceSourceSavedViewInvalidReason =
  (typeof WORKSPACE_SOURCE_SAVED_VIEW_INVALID_REASONS)[number]

export interface WorkspaceSourceSavedViewStateV1 {
  type_filters: WorkspaceSourceSavedViewTypeFilter[]
  status_filters: WorkspaceSourceSavedViewStatusFilter[]
  review_state_filters: WorkspaceSourceSavedViewReviewStateFilter[]
  lifecycle_state_filters: WorkspaceSourceSavedViewLifecycleStateFilter[]
  date_field: WorkspaceSourceSavedViewDateField
  date_from: string | null
  date_to: string | null
  require_url: boolean
  require_file_size: boolean
  require_duration: boolean
  require_page_count: boolean
  file_size_min: number | null
  file_size_max: number | null
  duration_min: number | null
  duration_max: number | null
  page_count_min: number | null
  page_count_max: number | null
  sort: WorkspaceSourceSavedViewSort
}
