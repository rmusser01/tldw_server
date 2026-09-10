export type SharedRecoveryAction = "retry" | "refresh" | "reselect_sources"

export interface SharedAllowedAction {
  allowed: boolean
  reason_code: string | null
}

export interface SharedAllowedActions {
  inspect_sources: SharedAllowedAction
  ask_grounded_questions: SharedAllowedAction
  add_sources: SharedAllowedAction
  edit_workspace: SharedAllowedAction
  clone_workspace: SharedAllowedAction
}

export type SharedGenerationDefault =
  | {
      provider: string
      model: string
      ready: true
      reason_code: null
    }
  | {
      provider: null
      model: null
      ready: false
      reason_code: string
    }

export interface SharedPagination {
  offset: number
  limit: number
  total: number
  has_more: boolean
}

export interface SharedPartialError {
  area: string
  code: string
  message: string
  retryable: boolean
}

export interface SharedSource {
  source_id: string
  title: string
  source_type: string
  origin_url: string | null
  origin_host: string | null
  state: string
  reason_code: string | null
  citation_ready: boolean
  retrieval_ready: boolean
  position: number
  added_at: string | null
}

export interface SharedSourceSummary {
  total: number
  queryable: number
  processing: number
  failed: number
}

export interface SharedSourceQuery {
  offset: number
  limit: number
  q?: string
  state?: string
}

export interface SharedSourcePage {
  items: SharedSource[]
  pagination: SharedPagination
  summary: SharedSourceSummary
  partial_errors: SharedPartialError[]
}

export interface SharedPreviewSnippet {
  kind: "content_excerpt" | "chunk"
  text: string
  start_char: number | null
  end_char: number | null
  chunk_index: number | null
}

export interface SharedSourcePreview {
  source_id: string
  title: string
  source_type: string
  origin_url: string | null
  origin_host: string | null
  state: string
  reason_code: string | null
  content_available: boolean
  preview_mode: string
  unavailable_reason: string | null
  text_preview: string | null
  text_total_chars: number | null
  text_truncated: boolean
  snippets: SharedPreviewSnippet[]
  generated_at: string
}

export interface SharedCitationLocator {
  chunk: number | null
  start_char: number | null
  end_char: number | null
}

export interface SharedCitation {
  citation_id: string
  source_id: string
  source_title: string
  locator: SharedCitationLocator
  quote: string
  score: number
}

export interface SharedMessage {
  message_id: string
  role: "user" | "assistant"
  content: string
  created_at: string
  citations: SharedCitation[]
}

export type SharedTurnMessage = Omit<SharedMessage, "citations">

export interface SharedMessagePage {
  conversation_id: string | null
  messages: SharedMessage[]
  next_before: string | null
}

export interface SharedWorkspaceBootstrap {
  schema_version: 1
  generated_at: string
  share: {
    share_id: number
    access_level: string
    allow_clone: boolean
    owner_display_name: string
    shared_at: string | null
  }
  workspace: {
    workspace_id: string
    name: string
    description: string
  }
  allowed_actions: SharedAllowedActions
  generation_default: SharedGenerationDefault
  source_summary: SharedSourceSummary
  sources: {
    items: SharedSource[]
    pagination: SharedPagination
  }
  conversation: SharedMessagePage
  partial_errors: SharedPartialError[]
}

export interface SharedSourceScope {
  mode: "all" | "include"
  source_ids: string[]
}

export interface SharedChatRequest {
  request_id: string
  query: string
  source_scope: SharedSourceScope
  provider?: string | null
  model?: string | null
}

export interface SharedChatResponse {
  schema_version: 1
  request_id: string
  conversation_id: string
  turn: {
    user_message: SharedTurnMessage
    assistant_message: SharedTurnMessage
  }
  citations: SharedCitation[]
  generation: {
    provider: string
    model: string
  }
  source_scope: {
    mode: "all" | "include"
    effective_source_count: number
  }
  replay: {
    replayed: boolean
  }
}
