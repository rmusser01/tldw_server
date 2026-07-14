/**
 * Skills module types
 * Corresponds to tldw_server2 /api/v1/skills endpoints
 */

export type SkillContext = "inline" | "fork"
export type SkillListSort = "name" | "context" | "created_at" | "last_modified"
export type SkillListOrder = "asc" | "desc"

export interface SkillRuntimeMetadata {
  execution_mode: SkillContext
  test_run_may_call_model: boolean
  declares_tools: boolean
  declared_tool_count: number
  model_override: string | null
  auto_invocation_enabled: boolean
}

export interface SkillsListParams {
  q?: string
  includeHidden?: boolean
  userInvocable?: boolean
  hasTools?: boolean
  context?: SkillContext
  model?: string
  sort?: SkillListSort
  order?: SkillListOrder
  limit?: number
  offset?: number
  abortSignal?: AbortSignal
}

export interface SkillSummary {
  name: string
  description: string | null
  argument_hint: string | null
  user_invocable: boolean
  disable_model_invocation: boolean
  allowed_tools?: string[] | null
  model?: string | null
  context: SkillContext
  runtime?: SkillRuntimeMetadata | null
  version: number
}

export interface SkillResponse {
  id: string
  name: string
  description: string | null
  argument_hint: string | null
  disable_model_invocation: boolean
  user_invocable: boolean
  allowed_tools: string[] | null
  model: string | null
  context: SkillContext
  runtime?: SkillRuntimeMetadata | null
  content: string
  raw_content?: string | null
  supporting_files: Record<string, string> | null
  directory_path: string
  created_at: string
  last_modified: string
  version: number
}

export interface SkillsListResponse {
  skills: SkillSummary[]
  count: number
  total: number
  limit: number
  offset: number
}

export interface SkillsTrashListParams {
  limit?: number
  offset?: number
  abortSignal?: AbortSignal
}

export interface SkillTrashItem extends SkillSummary {
  deleted_at: string
  restorable: boolean
  restore_unavailable_reason: string | null
}

export interface SkillsTrashListResponse {
  skills: SkillTrashItem[]
  count: number
  total: number
  limit: number
  offset: number
}

export interface SkillBulkDeleteItem {
  name: string
  version?: number
}

export interface SkillBulkDeleteResponse {
  deleted: string[]
  count: number
}

export interface SkillCreate {
  name: string
  content: string
  supporting_files?: Record<string, string> | null
}

export interface SkillUpdate {
  content?: string
  supporting_files?: Record<string, string | null> | null
}

export interface SkillExecuteRequest {
  args?: string | null
  dry_run?: boolean
}

export interface SkillExecutionResult {
  skill_name: string
  rendered_prompt: string
  allowed_tools: string[] | null
  model_override: string | null
  execution_mode: SkillContext
  fork_output: string | null
  dry_run: boolean
}

export interface SkillImportRequest {
  name?: string
  content: string
  supporting_files?: Record<string, string> | null
  overwrite?: boolean
  expected_version?: number
}

export interface SkillImportPreviewResponse {
  valid: boolean
  errors: string[]
  name: string | null
  description: string | null
  argument_hint: string | null
  disable_model_invocation: boolean | null
  user_invocable: boolean | null
  allowed_tools: string[] | null
  model: string | null
  context: SkillContext | null
  supporting_file_count: number
  conflict: boolean
  can_overwrite: boolean
  existing_version: number | null
}

export interface SkillContextPayload {
  available_skills: SkillSummary[]
  context_text: string
}
