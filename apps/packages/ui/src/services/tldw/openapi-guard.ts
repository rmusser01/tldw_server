// Type-level guard to keep the extension's API usage aligned with the
// server's OpenAPI spec. This file only exports types and helper
// functions, so it does not increase the runtime bundle size.
//
// NOTE: The openapi.json import was removed to eliminate the 1.4 MB
// JSON from the initial bundle. The ClientPath union below is manually
// maintained. QuickIngestModal dynamically imports the spec when needed.
//
// Maintenance:
// - When you add a new server endpoint that the extension calls (via bgRequest,
//   bgStream, or direct fetch), add its relative path to ClientPath so TS can
//   type-check it against the OpenAPI spec.
// - When you remove or rename an endpoint, update ClientPath (and any entries
//   in API_PATHS) to match the current server API.
// - To verify that ClientPath and MEDIA_ADD_SCHEMA_FALLBACK stay in sync with
//   the current OpenAPI contract, run `bun run verify:openapi` from
//   `apps/packages/ui`.
//   If verification fails, reconcile the differences by updating ClientPath
//   or by refreshing `apps/extension/openapi.json`. If that snapshot is absent,
//   the verifier derives the spec from the checked-out backend using the
//   project Python environment.

// Union of relative API paths that the web UI calls via bgRequest/bgStream
// or direct fetch. If a new endpoint is added in the UI, it should be
// added here so TypeScript can verify it exists in the spec.
export type ClientPath =
  | "/api/v1/health"
  | "/api/v1/llm/models"
  | "/api/v1/llm/models/metadata"
  | "/api/v1/llm/providers"
  | "/api/v1/chat/completions"
  | "/api/v1/chat/commands"
  | "/api/v1/feedback/explicit"
  | "/api/v1/web-clipper/save"
  | "/api/v1/web-clipper/{clip_id}"
  | "/api/v1/web-clipper/{clip_id}/enrichments"
  | "/api/v1/rag/health"
  | "/api/v1/rag/search"
  | "/api/v1/rag/search/stream"
  | "/api/v1/rag/simple"
  | "/api/v1/rag/feedback/implicit"
  | "/api/v1/media"
  | "/api/v1/media/search"
  | "/api/v1/media/metadata-search"
  | "/api/v1/media/add"
  | "/api/v1/media/bulk/keyword-update"
  | "/api/v1/media/statistics"
  | "/api/v1/media/{media_id}/restore"
  | "/api/v1/media/{media_id}/permanent"
  | "/api/v1/media/{media_id}/reprocess"
  | "/api/v1/media/ingest/jobs"
  | "/api/v1/media/ingest/jobs/{job_id}"
  | "/api/v1/media/{media_id}/keywords"
  | "/api/v1/media/process-audios"
  | "/api/v1/media/process-documents"
  | "/api/v1/media/process-ebooks"
  | "/api/v1/media/process-pdfs"
  | "/api/v1/media/process-videos"
  | "/api/v1/media/process-web-scraping"
  | "/api/v1/data-tables"
  | "/api/v1/data-tables/generate"
  | "/api/v1/data-tables/{table_uuid}"
  | "/api/v1/data-tables/{table_uuid}/content"
  | "/api/v1/data-tables/{table_uuid}/regenerate"
  | "/api/v1/data-tables/{table_uuid}/export"
  | "/api/v1/data-tables/jobs/{job_id}"
  | "/api/v1/files/create"
  | "/api/v1/files/{file_id}"
  | "/api/v1/files/{file_id}/export"
  | "/api/v1/notes"
  | "/api/v1/notes/search"
  | "/api/v1/notes/collections"
  | "/api/v1/notes/collections/{collection_id}"
  | "/api/v1/notes/collections/{collection_id}/keywords"
  | "/api/v1/notes/collections/{collection_id}/keywords/{keyword_id}"
  | "/api/v1/notes/collections/keyword-links"
  | "/api/v1/notes/keywords"
  | "/api/v1/notes/keywords/search"
  | "/api/v1/notes/keywords/{keyword_id}"
  | "/api/v1/notes/conversations/{conversation_id}/keywords"
  | "/api/v1/notes/conversations/{conversation_id}/keywords/{keyword_id}"
  | "/api/v1/notes/conversations/keyword-links"
  | "/api/v1/prompts"
  | "/api/v1/prompts/search"
  | "/api/v1/prompts/export"
  | "/api/v1/prompts/collections"
  | "/api/v1/prompts/collections/create"
  | "/api/v1/prompts/collections/{collection_id}"
  | "/api/v1/chat/dictionaries"
  | "/api/v1/chat/dictionaries/import/json"
  | "/api/v1/chat/dictionaries/validate"
  | "/api/v1/chat/dictionaries/process"
  | "/api/v1/chat/grammars"
  | "/api/v1/chat/grammars/{grammar_id}"
  | "/api/v1/chat/knowledge/save"
  | "/api/v1/chat/documents"
  | "/api/v1/chat/documents/generate"
  | "/api/v1/chat/documents/bulk"
  | "/api/v1/chat/documents/{document_id}"
  | "/api/v1/chat/documents/jobs/{job_id}"
  | "/api/v1/chat/documents/prompts"
  | "/api/v1/chat/documents/prompts/{document_type}"
  | "/api/v1/chat/documents/statistics"
  | "/api/v1/chat/queue/status"
  | "/api/v1/chat/queue/activity"
  | "/api/v1/chats"
  | "/api/v1/chats/{chat_id}/completions"
  | "/api/v1/chats/{chat_id}/completions/persist"
  | "/api/v1/chats/{chat_id}/complete-v2"
  | "/api/v1/chats/{chat_id}/settings"
  | "/api/v1/characters"
  | "/api/v1/characters/import"
  | "/api/v1/characters/world-books"
  | "/api/v1/characters/world-books/import"
  | "/api/v1/audio/providers"
  | "/api/v1/audio/speech"
  | "/api/v1/audio/speech/jobs"
  | "/api/v1/audio/speech/jobs/{job_id}/artifacts"
  | "/api/v1/audio/jobs/{job_id}/progress/stream"
  | "/api/v1/audio/transcriptions"
  | "/api/v1/audio/voices"
  | "/api/v1/audio/voices/upload"
  | "/api/v1/audio/voices/encode"
  | "/api/v1/audio/voices/catalog"
  | "/api/v1/audio/voices/{voice_id}"
  | "/api/v1/audio/voices/{voice_id}/preview"
  | "/api/v1/audio/health"
  | "/api/v1/embeddings/models"
  | "/api/v1/embeddings/collections"
  | "/api/v1/embeddings/providers-config"
  | "/api/v1/embeddings/health"
  | "/api/v1/metrics/health"
  | "/api/v1/metrics/chat"
  | "/api/v1/evaluations"
  | "/api/v1/evaluations/datasets"
  | "/api/v1/evaluations/rate-limits"
  | "/api/v1/evaluations/history"
  | "/api/v1/evaluations/webhooks"
  | "/api/v1/mcp/health"
  | "/api/v1/mcp/tools"
  | "/api/v1/mcp/tool_catalogs"
  | "/api/v1/mcp/status"
  | "/api/v1/mcp/tools/execute"
  | "/api/v1/mcp/hub/acp-profiles"
  | "/api/v1/mcp/hub/acp-profiles/{profile_id}"
  | "/api/v1/mcp/hub/external-servers"
  | "/api/v1/mcp/hub/external-servers/{server_id}"
  | "/api/v1/mcp/hub/external-servers/{server_id}/secret"
  | "/api/v1/mcp/hub/tool-registry"
  | "/api/v1/mcp/hub/tool-registry/modules"
  | "/api/v1/mcp/hub/tool-registry/summary"
  | "/api/v1/mcp/hub/permission-profiles"
  | "/api/v1/mcp/hub/permission-profiles/{profile_id}"
  | "/api/v1/mcp/hub/policy-assignments"
  | "/api/v1/mcp/hub/policy-assignments/{assignment_id}"
  | "/api/v1/mcp/hub/policy-assignments/{assignment_id}/override"
  | "/api/v1/mcp/hub/approval-policies"
  | "/api/v1/mcp/hub/approval-policies/{approval_policy_id}"
  | "/api/v1/mcp/hub/approval-decisions"
  | "/api/v1/mcp/hub/effective-policy"
  | "/api/v1/chat-workflows/templates"
  | "/api/v1/chat-workflows/templates/{template_id}"
  | "/api/v1/chat-workflows/generate-draft"
  | "/api/v1/chat-workflows/runs"
  | "/api/v1/chat-workflows/runs/{run_id}"
  | "/api/v1/chat-workflows/runs/{run_id}/transcript"
  | "/api/v1/chat-workflows/runs/{run_id}/answer"
  | "/api/v1/chat-workflows/runs/{run_id}/rounds/{round_index}/respond"
  | "/api/v1/chat-workflows/runs/{run_id}/cancel"
  | "/api/v1/chat-workflows/runs/{run_id}/continue-chat"
  | "/api/v1/workflows/step-types"
  | "/api/v1/items"
  | "/api/v1/items/{item_id}"
  | "/api/v1/items/bulk"
  | "/api/v1/reading/items"
  | "/api/v1/reading/items/{item_id}"
  | "/api/v1/reading/save"
  | "/api/v1/reading/items/{item_id}/summarize"
  | "/api/v1/reading/items/{item_id}/tts"
  | "/api/v1/reading/items/{item_id}/highlight"
  | "/api/v1/reading/items/{item_id}/highlights"
  | "/api/v1/reading/highlights/{highlight_id}"
  | "/api/v1/reading/import"
  | "/api/v1/reading/import/jobs"
  | "/api/v1/reading/import/jobs/{job_id}"
  | "/api/v1/reading/export"
  | "/api/v1/reading/digests/schedules"
  | "/api/v1/reading/digests/schedules/{schedule_id}"
  | "/api/v1/slides/generate/from-media"
  | "/api/v1/slides/presentations"
  | "/api/v1/slides/presentations/{presentation_id}"
  | "/api/v1/slides/presentations/{presentation_id}/export"
  | "/api/v1/slides/presentations/{presentation_id}/render-jobs"
  | "/api/v1/slides/render-jobs/{job_id}"
  | "/api/v1/slides/presentations/{presentation_id}/render-artifacts"
  | "/api/v1/writing/version"
  | "/api/v1/writing/capabilities"
  | "/api/v1/writing/sessions"
  | "/api/v1/writing/sessions/{session_id}"
  | "/api/v1/writing/sessions/{session_id}/clone"
  | "/api/v1/writing/templates"
  | "/api/v1/writing/templates/{name}"
  | "/api/v1/writing/themes"
  | "/api/v1/writing/themes/{name}"
  | "/api/v1/writing/defaults"
  | "/api/v1/writing/snapshot/export"
  | "/api/v1/writing/snapshot/import"
  | "/api/v1/writing/tokenize"
  | "/api/v1/writing/detokenize"
  | "/api/v1/writing/token-count"
  | "/api/v1/writing/wordclouds"
  | "/api/v1/writing/wordclouds/{wordcloud_id}"
  | "/api/v1/outputs/templates"
  | "/api/v1/outputs/templates/{template_id}"
  | "/api/v1/outputs/templates/{template_id}/preview"
  | "/api/v1/outputs"
  | "/api/v1/outputs/{output_id}/download"
  | "/api/v1/quizzes/import/json"
  | "/api/v1/flashcards"
  | "/api/v1/flashcards/decks"
  | "/api/v1/flashcards/templates"
  | "/api/v1/flashcards/templates/{template_id}"
  | "/api/v1/flashcards/generate"
  | "/api/v1/flashcards/review"
  | "/api/v1/flashcards/import"
  | "/api/v1/flashcards/import/json"
  | "/api/v1/flashcards/import/apkg"
  | "/api/v1/flashcards/export"
  | "/api/v1/chatbooks/export"
  | "/api/v1/chatbooks/preview"
  | "/api/v1/chatbooks/import"
  | "/api/v1/chatbooks/export/jobs"
  | "/api/v1/chatbooks/export/jobs/{job_id}"
  | "/api/v1/chatbooks/import/jobs"
  | "/api/v1/chatbooks/import/jobs/{job_id}"
  | "/api/v1/chatbooks/download/{job_id}"
  | "/api/v1/chatbooks/cleanup"
  | "/api/v1/chatbooks/health"
  | "/api/v1/auth/login"
  | "/api/v1/auth/logout"
  | "/api/v1/auth/me"
  | "/api/v1/auth/refresh"
  | "/api/v1/auth/register"
  | "/api/v1/auth/magic-link/request"
  | "/api/v1/auth/magic-link/verify"
  | "/api/v1/persona/profiles"
  | "/api/v1/persona/archetypes"
  | "/api/v1/persona/archetypes/{archetype_key}"
  | "/api/v1/persona/archetypes/{archetype_key}/preview"
  | "/api/v1/mcp/catalog"
  | "/api/v1/mcp/catalog/test-connection"
  | "/api/v1/users/keys"
  | "/api/v1/users/keys/openai/oauth/authorize"
  | "/api/v1/users/keys/openai/oauth/callback"
  | "/api/v1/users/keys/openai/oauth/status"
  | "/api/v1/users/keys/openai/oauth/refresh"
  | "/api/v1/users/keys/openai/oauth"
  | "/api/v1/users/keys/openai/source"
  | "/api/v1/users/storage"
  | "/api/v1/billing/plans"
  | "/api/v1/billing/subscription"
  | "/api/v1/billing/usage"
  | "/api/v1/billing/invoices"
  | "/api/v1/billing/subscription/cancel"
  | "/api/v1/billing/subscription/resume"
  | "/api/v1/billing/checkout"
  | "/api/v1/billing/portal"
  | "/api/v1/orgs"
  | "/api/v1/chunking/chunk_text"
  | "/api/v1/chunking/chunk_file"
  | "/api/v1/chunking/capabilities"
  | "/api/v1/moderation/settings"
  | "/api/v1/moderation/policy/effective"
  | "/api/v1/moderation/reload"
  | "/api/v1/moderation/users"
  | "/api/v1/moderation/users/{user_id}"
  | "/api/v1/moderation/blocklist"
  | "/api/v1/moderation/blocklist/managed"
  | "/api/v1/moderation/blocklist/append"
  | "/api/v1/moderation/blocklist/{item_id}"
  | "/api/v1/moderation/blocklist/lint"
  | "/api/v1/moderation/test"


type ReplacePathParams<Path extends string> =
  Path extends `${infer Head}{${string}}${infer Tail}`
    ? ReplacePathParams<`${Head}${string}${Tail}`>
    : Path

// Runtime path form: replaces OpenAPI-style "{param}" segments with "${string}".
export type ClientPathRuntime = ReplacePathParams<ClientPath>

// OpenAPI paths don't include query strings, but the UI appends them at runtime.
export type ClientPathRuntimeWithQuery = ClientPathRuntime | `${ClientPathRuntime}?${string}`

// Centralized, typed API paths for use across the extension. Values are
// checked against ClientPath so that any drift from the OpenAPI spec is
// caught at compile time.
export const API_PATHS = {
  MEDIA_ADD: "/api/v1/media/add" as const,
  MEDIA_INGEST_JOBS: "/api/v1/media/ingest/jobs" as const
} as const satisfies Record<string, ClientPath>

// Allowed relative API path: anything beginning with a slash. We keep
// this wide to avoid breaking existing call sites, while ClientPath
// provides a manually-maintained list of known endpoints.
export type AllowedPath = `/${string}`

// Absolute URL permitted in a few places
export type AbsoluteURL = `${'http' | 'https'}:${string}`

export type PathOrUrl = AllowedPath | AbsoluteURL

export type ClientPathOrUrl = ClientPathRuntime | AbsoluteURL

export type ClientPathOrUrlWithQuery = ClientPathRuntimeWithQuery | AbsoluteURL

// Common HTTP methods accepted
export type AllowedHttpMethod = 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH' | 'OPTIONS' | 'HEAD'

// Any method is fine for absolute URLs; for paths, use the common set
export type AllowedMethodFor<P extends PathOrUrl> = P extends AbsoluteURL
  ? string
  : AllowedHttpMethod

// Convenience: accept lower/upper/mixed-case method annotations at call sites
export type UpperLower<M extends string> = Uppercase<M> | Lowercase<M> | M

export function normalizeMethod<M extends string>(method: M): Uppercase<M> {
  return String(method).toUpperCase() as Uppercase<M>
}
