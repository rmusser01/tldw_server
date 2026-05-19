export type IngestionSourceType = "local_directory" | "archive_snapshot" | "git_repository"
export type IngestionSinkType = "media" | "notes"
export type IngestionLifecyclePolicy = "canonical" | "import_only"

export interface IngestionSourceSyncSummary {
  changed_count: number
  degraded_count: number
  conflict_count: number
  sink_failure_count: number
  ingestion_failure_count: number
  created_count: number
  updated_count: number
  deleted_count: number
  unchanged_count: number
}

export interface IngestionSourceSummary {
  id: string
  user_id: number
  source_type: IngestionSourceType
  sink_type: IngestionSinkType
  policy: IngestionLifecyclePolicy
  enabled: boolean
  schedule_enabled: boolean
  schedule_config: Record<string, unknown>
  config: Record<string, unknown>
  active_job_id?: string | null
  last_successful_snapshot_id?: string | null
  last_sync_started_at?: string | null
  last_sync_completed_at?: string | null
  last_sync_status?: string | null
  last_error?: string | null
  last_successful_sync_summary?: IngestionSourceSyncSummary | null
  created_at?: string | null
  updated_at?: string | null
}

export interface IngestionSourceListResponse {
  sources: IngestionSourceSummary[]
  total: number
}

export interface IngestionSourceItem {
  id: string
  source_id: string
  normalized_relative_path: string
  content_hash?: string | null
  sync_status: string
  binding: Record<string, unknown>
  present_in_source: boolean
  created_at?: string | null
  updated_at?: string | null
}

export interface IngestionSourceItemsListResponse {
  items: IngestionSourceItem[]
  total: number
}

export interface IngestionSourceDirectoryEntry {
  name: string
  path: string
  is_root: boolean
}

export interface IngestionSourceDirectoryBrowseResponse {
  roots: IngestionSourceDirectoryEntry[]
  current_path?: string | null
  parent_path?: string | null
  entries: IngestionSourceDirectoryEntry[]
  error?: string | null
}

export interface IngestionSourceItemFilters {
  sync_status?: string
  present_in_source?: boolean
}

export interface CreateIngestionSourceRequest {
  source_type: IngestionSourceType
  sink_type: IngestionSinkType
  policy?: IngestionLifecyclePolicy
  enabled?: boolean
  schedule_enabled?: boolean
  schedule?: Record<string, unknown>
  config?: Record<string, unknown>
}

export interface UpdateIngestionSourceRequest {
  source_type?: IngestionSourceType
  sink_type?: IngestionSinkType
  policy?: IngestionLifecyclePolicy
  enabled?: boolean
  schedule_enabled?: boolean
  schedule?: Record<string, unknown>
  config?: Record<string, unknown>
}

export interface IngestionSourceSyncTriggerResponse {
  status: string
  source_id: string
  job_id?: string | null
  snapshot_status?: string | null
}
