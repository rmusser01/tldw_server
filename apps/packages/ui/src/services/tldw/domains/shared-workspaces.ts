import { buildTldwApiError, TldwApiError } from "@/services/tldw/api-error"
import { fetchWithTldwAuth } from "@/services/tldw/auth-fetch"
import { getTldwServerURL } from "@/services/tldw-server"
import type {
  SharedAllowedAction,
  SharedAllowedActions,
  SharedChatRequest,
  SharedChatResponse,
  SharedCitation,
  SharedGenerationDefault,
  SharedMessage,
  SharedMessagePage,
  SharedPagination,
  SharedPartialError,
  SharedPreviewSnippet,
  SharedSource,
  SharedSourcePage,
  SharedSourcePreview,
  SharedSourceQuery,
  SharedSourceSummary,
  SharedTurnMessage,
  SharedWorkspaceBootstrap
} from "@/types/shared-workspace"

const INVALID_RESPONSE_DETAIL = {
  code: "shared_workspace_unavailable",
  message: "Shared workspace returned an invalid response.",
  retryable: true,
  recovery_action: "retry"
} as const

const POST_COMMIT_RESPONSE_DETAIL = {
  code: "shared_chat_response_unconfirmed",
  message: "The answer status is uncertain. Retry to reconcile this question.",
  retryable: true,
  recovery_action: "retry"
} as const

type SharedPostCommitResponseDetail = {
  code: string
  message: string
  retryable: true
  recovery_action: "retry"
}

export class SharedWorkspacePostCommitResponseError extends TldwApiError {
  constructor(
    detail: SharedPostCommitResponseDetail = POST_COMMIT_RESPONSE_DETAIL
  ) {
    super(detail.message, 502, detail)
    this.name = "SharedWorkspacePostCommitResponseError"
  }
}

export const isSharedWorkspacePostCommitResponseError = (
  error: unknown
): error is SharedWorkspacePostCommitResponseError =>
  error instanceof SharedWorkspacePostCommitResponseError

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value)

const record = (value: unknown): Record<string, unknown> => {
  if (!isRecord(value)) throw new Error("Expected object")
  return value
}

const string = (
  value: unknown,
  maxLength: number,
  { allowEmpty = false }: { allowEmpty?: boolean } = {}
): string => {
  if (typeof value !== "string" || value.length > maxLength) {
    throw new Error("Expected bounded string")
  }
  if (!allowEmpty && !value.trim()) throw new Error("Expected non-empty string")
  return value
}

const nullableString = (
  value: unknown,
  maxLength: number,
  options?: { allowEmpty?: boolean }
): string | null => (value === null ? null : string(value, maxLength, options))

const boolean = (value: unknown): boolean => {
  if (typeof value !== "boolean") throw new Error("Expected boolean")
  return value
}

const finiteNumber = (value: unknown): number => {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error("Expected finite number")
  }
  return value
}

const integer = (
  value: unknown,
  { min = 0, max = Number.MAX_SAFE_INTEGER } = {}
): number => {
  const parsed = finiteNumber(value)
  if (!Number.isSafeInteger(parsed) || parsed < min || parsed > max) {
    throw new Error("Expected bounded integer")
  }
  return parsed
}

const nullableInteger = (value: unknown): number | null =>
  value === null ? null : integer(value)

const array = <T>(
  value: unknown,
  maxLength: number,
  parseItem: (item: unknown) => T
): T[] => {
  if (!Array.isArray(value) || value.length > maxLength) {
    throw new Error("Expected bounded array")
  }
  return value.map(parseItem)
}

const parsePagination = (value: unknown): SharedPagination => {
  const item = record(value)
  return {
    offset: integer(item.offset),
    limit: integer(item.limit, { min: 1, max: 200 }),
    total: integer(item.total),
    has_more: boolean(item.has_more)
  }
}

const parsePartialError = (value: unknown): SharedPartialError => {
  const item = record(value)
  return {
    area: string(item.area, 64),
    code: string(item.code, 128),
    message: string(item.message, 320),
    retryable: boolean(item.retryable)
  }
}

const parseSource = (value: unknown): SharedSource => {
  const item = record(value)
  return {
    source_id: string(item.source_id, 512),
    title: string(item.title, 512, { allowEmpty: true }),
    source_type: string(item.source_type, 64),
    origin_url: nullableString(item.origin_url, 2_048),
    origin_host: nullableString(item.origin_host, 255),
    state: string(item.state, 64),
    reason_code: nullableString(item.reason_code, 128),
    citation_ready: boolean(item.citation_ready),
    retrieval_ready: boolean(item.retrieval_ready),
    position: integer(item.position),
    added_at: nullableString(item.added_at, 64)
  }
}

const parseSummary = (value: unknown): SharedSourceSummary => {
  const item = record(value)
  return {
    total: integer(item.total),
    queryable: integer(item.queryable),
    processing: integer(item.processing),
    failed: integer(item.failed)
  }
}

const parseCitation = (value: unknown): SharedCitation => {
  const item = record(value)
  const locator = record(item.locator)
  return {
    citation_id: string(item.citation_id, 512),
    source_id: string(item.source_id, 512),
    source_title: string(item.source_title, 512, { allowEmpty: true }),
    locator: {
      chunk: nullableInteger(locator.chunk),
      start_char: nullableInteger(locator.start_char),
      end_char: nullableInteger(locator.end_char)
    },
    quote: string(item.quote, 1_000),
    score: finiteNumber(item.score)
  }
}

const parseMessage = (value: unknown): SharedMessage => {
  const item = record(value)
  if (item.role !== "user" && item.role !== "assistant") {
    throw new Error("Expected message role")
  }
  return {
    message_id: string(item.message_id, 512),
    role: item.role,
    content: string(item.content, 100_000, { allowEmpty: true }),
    created_at: string(item.created_at, 64),
    citations:
      item.citations === undefined
        ? []
        : array(item.citations, 20, parseCitation)
  }
}

const parseTurnMessage = (value: unknown): SharedTurnMessage => {
  const message = parseMessage(value)
  return {
    message_id: message.message_id,
    role: message.role,
    content: message.content,
    created_at: message.created_at
  }
}

const parseMessagePage = (value: unknown): SharedMessagePage => {
  const item = record(value)
  return {
    conversation_id: nullableString(item.conversation_id, 512),
    messages: array(item.messages, 100, parseMessage),
    next_before: nullableString(item.next_before, 2_048)
  }
}

const deniedAction = (reasonCode: string): SharedAllowedAction => ({
  allowed: false,
  reason_code: reasonCode
})

const parseAction = (value: unknown): SharedAllowedAction => {
  if (!isRecord(value) || typeof value.allowed !== "boolean") {
    return deniedAction("shared_action_unavailable")
  }
  if (value.allowed) {
    return value.reason_code === null
      ? { allowed: true, reason_code: null }
      : deniedAction("shared_action_unavailable")
  }
  try {
    return {
      allowed: false,
      reason_code: string(value.reason_code, 128)
    }
  } catch {
    return deniedAction("shared_action_unavailable")
  }
}

const parseActions = (value: unknown): SharedAllowedActions => {
  const item = isRecord(value) ? value : {}
  return {
    inspect_sources: parseAction(item.inspect_sources),
    ask_grounded_questions: parseAction(item.ask_grounded_questions),
    add_sources: parseAction(item.add_sources),
    edit_workspace: parseAction(item.edit_workspace),
    clone_workspace: parseAction(item.clone_workspace)
  }
}

export const normalizeSharedGenerationDefault = (
  value: unknown
): SharedGenerationDefault => {
  if (isRecord(value)) {
    try {
      if (
        value.ready === true &&
        value.reason_code === null &&
        typeof value.provider === "string" &&
        typeof value.model === "string"
      ) {
        return {
          provider: string(value.provider, 128).trim(),
          model: string(value.model, 512).trim(),
          ready: true,
          reason_code: null
        }
      }
      if (
        value.ready === false &&
        value.provider === null &&
        value.model === null
      ) {
        return {
          provider: null,
          model: null,
          ready: false,
          reason_code: string(value.reason_code, 128).trim()
        }
      }
    } catch {
      // Invalid defaults are normalized below so generation remains denied.
    }
  }
  return {
    provider: null,
    model: null,
    ready: false,
    reason_code: "generation_default_unavailable"
  }
}

const parseBootstrap = (value: unknown): SharedWorkspaceBootstrap => {
  const item = record(value)
  if (item.schema_version !== 1) throw new Error("Unsupported schema")
  const share = record(item.share)
  const workspace = record(item.workspace)
  const sourceEnvelope = record(item.sources)
  const generationDefault = normalizeSharedGenerationDefault(
    item.generation_default
  )
  const allowedActions = parseActions(item.allowed_actions)
  return {
    schema_version: 1,
    generated_at: string(item.generated_at, 64),
    share: {
      share_id: integer(share.share_id, { min: 1 }),
      access_level: string(share.access_level, 64),
      allow_clone: boolean(share.allow_clone),
      owner_display_name: string(share.owner_display_name, 128),
      shared_at: nullableString(share.shared_at, 64)
    },
    workspace: {
      workspace_id: string(workspace.workspace_id, 512),
      name: string(workspace.name, 512, { allowEmpty: true }),
      description: string(workspace.description, 2_000, { allowEmpty: true })
    },
    allowed_actions: allowedActions,
    generation_default: generationDefault,
    source_summary: parseSummary(item.source_summary),
    sources: {
      items: array(sourceEnvelope.items, 50, parseSource),
      pagination: parsePagination(sourceEnvelope.pagination)
    },
    conversation: parseMessagePage(item.conversation),
    partial_errors: array(item.partial_errors, 8, parsePartialError)
  }
}

const parseSourcePage = (value: unknown): SharedSourcePage => {
  const item = record(value)
  return {
    items: array(item.items, 200, parseSource),
    pagination: parsePagination(item.pagination),
    summary: parseSummary(item.summary),
    partial_errors: array(item.partial_errors, 8, parsePartialError)
  }
}

const parseSnippet = (value: unknown): SharedPreviewSnippet => {
  const item = record(value)
  if (item.kind !== "content_excerpt" && item.kind !== "chunk") {
    throw new Error("Expected preview kind")
  }
  return {
    kind: item.kind,
    text: string(item.text, 12_000),
    start_char: nullableInteger(item.start_char),
    end_char: nullableInteger(item.end_char),
    chunk_index: nullableInteger(item.chunk_index)
  }
}

const parsePreview = (value: unknown): SharedSourcePreview => {
  const item = record(value)
  return {
    source_id: string(item.source_id, 512),
    title: string(item.title, 512, { allowEmpty: true }),
    source_type: string(item.source_type, 64),
    origin_url: nullableString(item.origin_url, 2_048),
    origin_host: nullableString(item.origin_host, 255),
    state: string(item.state, 64),
    reason_code: nullableString(item.reason_code, 128),
    content_available: boolean(item.content_available),
    preview_mode: string(item.preview_mode, 64),
    unavailable_reason: nullableString(item.unavailable_reason, 128),
    text_preview: nullableString(item.text_preview, 12_000, { allowEmpty: true }),
    text_total_chars: nullableInteger(item.text_total_chars),
    text_truncated: boolean(item.text_truncated),
    snippets: array(item.snippets, 10, parseSnippet),
    generated_at: string(item.generated_at, 64)
  }
}

const parseChatResponse = (value: unknown): SharedChatResponse => {
  const item = record(value)
  if (item.schema_version !== 1) throw new Error("Unsupported schema")
  const turn = record(item.turn)
  const generation = record(item.generation)
  const scope = record(item.source_scope)
  const replay = record(item.replay)
  if (scope.mode !== "all" && scope.mode !== "include") {
    throw new Error("Expected source scope")
  }
  const citations = array(item.citations, 20, parseCitation)
  if (citations.length === 0) throw new Error("Expected grounded citations")
  const userMessage = parseTurnMessage(turn.user_message)
  const assistantMessage = parseTurnMessage(turn.assistant_message)
  return {
    schema_version: 1,
    request_id: string(item.request_id, 64),
    conversation_id: string(item.conversation_id, 512),
    turn: {
      user_message: userMessage,
      assistant_message: assistantMessage
    },
    citations,
    generation: {
      provider: string(generation.provider, 128),
      model: string(generation.model, 512)
    },
    source_scope: {
      mode: scope.mode,
      effective_source_count: integer(scope.effective_source_count, {
        min: 1,
        max: 500
      })
    },
    replay: { replayed: boolean(replay.replayed) }
  }
}

const apiUrl = async (shareId: number, suffix: string): Promise<string> => {
  const base = (await getTldwServerURL()).replace(/\/+$/, "")
  return `${base}/api/v1/sharing/shared-with-me/${shareId}${suffix}`
}

const requestJson = async <T>(
  url: string,
  init: RequestInit,
  parse: (value: unknown) => T,
  options: { postCommitOnInvalidSuccess?: boolean } = {}
): Promise<T> => {
  const response = await fetchWithTldwAuth(url, init)
  if (!response.ok) throw await buildTldwApiError(response)
  try {
    return parse(await response.json())
  } catch (error) {
    if (error instanceof TldwApiError) throw error
    if (options.postCommitOnInvalidSuccess) {
      throw new SharedWorkspacePostCommitResponseError()
    }
    throw new TldwApiError(
      INVALID_RESPONSE_DETAIL.message,
      502,
      INVALID_RESPONSE_DETAIL
    )
  }
}

export const sharedWorkspacesApi = {
  async bootstrap(
    shareId: number,
    signal?: AbortSignal
  ): Promise<SharedWorkspaceBootstrap> {
    return requestJson(
      await apiUrl(shareId, "/workspace"),
      { signal },
      parseBootstrap
    )
  },

  async listSources(
    shareId: number,
    query: SharedSourceQuery,
    signal?: AbortSignal
  ): Promise<SharedSourcePage> {
    const params = new URLSearchParams({
      offset: String(query.offset),
      limit: String(query.limit)
    })
    if (query.q?.trim()) params.set("q", query.q.trim())
    if (query.state?.trim()) params.set("state", query.state.trim())
    return requestJson(
      await apiUrl(shareId, `/sources?${params.toString()}`),
      { signal },
      parseSourcePage
    )
  },

  async previewSource(
    shareId: number,
    sourceId: string,
    chunkIndex?: number,
    signal?: AbortSignal
  ): Promise<SharedSourcePreview> {
    const params = new URLSearchParams()
    if (chunkIndex !== undefined) {
      params.set("chunk_index", String(chunkIndex))
    }
    const query = params.size ? `?${params.toString()}` : ""
    return requestJson(
      await apiUrl(
        shareId,
        `/sources/${encodeURIComponent(sourceId)}/preview${query}`
      ),
      { signal },
      parsePreview
    )
  },

  async listMessages(
    shareId: number,
    before?: string,
    signal?: AbortSignal
  ): Promise<SharedMessagePage> {
    const params = new URLSearchParams()
    if (before?.trim()) params.set("before", before.trim())
    const query = params.size ? `?${params.toString()}` : ""
    return requestJson(
      await apiUrl(shareId, `/chat/messages${query}`),
      { signal },
      parseMessagePage
    )
  },

  async ask(
    shareId: number,
    request: SharedChatRequest,
    signal?: AbortSignal
  ): Promise<SharedChatResponse> {
    return requestJson(
      await apiUrl(shareId, "/chat"),
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request),
        signal
      },
      parseChatResponse,
      { postCommitOnInvalidSuccess: true }
    )
  }
}

export type SharedWorkspacesApi = typeof sharedWorkspacesApi
