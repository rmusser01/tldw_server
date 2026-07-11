import type {
  WorkspaceSourceLifecycleState,
  WorkspaceSourceReviewState,
  WorkspaceSourceStatus,
  WorkspaceSourceType
} from "./workspace"

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
  type_filters: WorkspaceSourceType[]
  status_filters: WorkspaceSourceStatus[]
  review_state_filters: WorkspaceSourceReviewState[]
  lifecycle_state_filters: WorkspaceSourceLifecycleState[]
  date_field: "added_at" | "source_created_at"
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
