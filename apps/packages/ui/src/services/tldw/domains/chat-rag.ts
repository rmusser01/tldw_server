import { bgRequest, bgStream, bgUpload } from '@/services/background-proxy'
import { buildQuery } from '../client-utils'
import { createJsonResponseLike } from '../json-response-like'
import { appendPathQuery } from '../path-utils'
import { captureChatRequestDebugSnapshot } from '../chat-request-debug'
import type { TldwApiClientCore } from '../TldwApiClient'
import type { ChatScope } from '@/types/chat-scope'
import { toChatScopeParams } from '@/types/chat-scope'
import { normalizeChatRole } from '@/utils/normalize-chat-role'
import type {
  ChatCompletionRequest,
  ServerChatSummary,
  ServerChatMessage,
  ChatSettingsResponse,
  LorebookDiagnosticExportResponse,
  ConversationSharePermission,
  ConversationShareLinkCreateResponse,
  ConversationShareLinksListResponse,
  ConversationShareLinkResolveResponse,
  WorldBookProcessResponse,
} from '../TldwApiClient'

const CHAT_MESSAGES_CACHE_TTL_MS = 60 * 1000

const isConnectionErrorMessage = (message: string): boolean =>
  /network|offline|failed to fetch|connection|unreachable/i.test(message)

const isTimeoutErrorMessage = (message: string): boolean =>
  /timeout|timed out|etimedout/i.test(message)

const isSavedDegradedCharacterPersistError = (error: unknown): boolean => {
  const candidate = error as
    | {
        detail?: unknown
        details?: { detail?: unknown; code?: unknown; saved?: unknown }
      }
    | null
  const detail =
    candidate?.detail &&
    typeof candidate.detail === "object" &&
    !Array.isArray(candidate.detail)
      ? candidate.detail
      : candidate?.details?.detail &&
            typeof candidate.details.detail === "object" &&
            !Array.isArray(candidate.details.detail)
        ? candidate.details.detail
        : candidate?.details &&
              typeof candidate.details === "object" &&
              !Array.isArray(candidate.details)
          ? candidate.details
          : null
  return detail?.code === "persist_validation_degraded" && detail?.saved === true
}

const buildSanitizedRagSearchError = (
  error: unknown
): Error & { status?: number; code?: string } => {
  const status =
    (error as { status?: number; response?: { status?: number }; statusCode?: number } | null)
      ?.status ??
    (error as { response?: { status?: number } } | null)?.response?.status ??
    (error as { statusCode?: number } | null)?.statusCode
  const rawMessage = error instanceof Error ? error.message : String(error ?? "")

  let message = "RAG search failed."
  if (isConnectionErrorMessage(rawMessage)) {
    message = "Cannot reach server. Check your connection and try again."
  } else if (isTimeoutErrorMessage(rawMessage) || status === 408) {
    message = "RAG search timed out. Try again."
  } else if (status === 400 || status === 422) {
    message = "RAG search request is invalid."
  } else if (status === 401) {
    message = "RAG search failed. Authentication is required."
  } else if (status === 403) {
    message = "RAG search failed. Access was denied."
  } else if (status === 404) {
    message = "RAG search endpoint is unavailable."
  } else if (status === 429) {
    message = "RAG search is rate limited. Please wait and try again."
  } else if (typeof status === "number" && status >= 500) {
    message = "RAG search failed due to a server error."
  }

  const sanitizedError = new Error(message) as Error & {
    status?: number
    code?: string
  }
  if (typeof status === "number") {
    sanitizedError.status = status
  }
  return sanitizedError
}

const CHAT_COMPLETION_ERROR_MESSAGE = "Chat completion failed."
const CHAT_COMPLETION_ERRORS_MESSAGE =
  "One or more internal errors were suppressed."

const isSuspiciousChatCompletionString = (value: string): boolean =>
  /traceback|stack(?:\s*trace)?|exception|error|\/Users\/|[A-Za-z]:\\|\.py:\d+/i.test(
    value
  )

const normalizeChatCompletionResponseBody = (
  value: unknown
): Record<string, unknown> | unknown[] => {
  if (typeof value === "string") {
    if (isSuspiciousChatCompletionString(value)) {
      return {
        error: CHAT_COMPLETION_ERROR_MESSAGE,
        errors: [CHAT_COMPLETION_ERRORS_MESSAGE]
      }
    }
    return { content: value }
  }
  const sanitized = sanitizeChatCompletionPayload(value)
  if (Array.isArray(sanitized)) {
    return sanitized
  }
  if (sanitized && typeof sanitized === "object") {
    return sanitized as Record<string, unknown>
  }
  return { content: sanitized ?? "" }
}

const sanitizeChatCompletionPayload = (value: unknown): unknown => {
  if (typeof value === "string") {
    return isSuspiciousChatCompletionString(value)
      ? CHAT_COMPLETION_ERROR_MESSAGE
      : value
  }
  if (Array.isArray(value)) {
    return value.map((item) => sanitizeChatCompletionPayload(item))
  }
  if (value && typeof value === "object") {
    const sanitized: Record<string, unknown> = {}
    for (const [key, item] of Object.entries(value)) {
      if (
        key === "details" ||
        key === "exception" ||
        key === "traceback" ||
        key === "stack" ||
        key === "stack_trace"
      ) {
        continue
      }
      if (key === "error" && item) {
        sanitized[key] = CHAT_COMPLETION_ERROR_MESSAGE
        continue
      }
      if (key === "errors" && item) {
        sanitized[key] = [CHAT_COMPLETION_ERRORS_MESSAGE]
        continue
      }
      sanitized[key] = sanitizeChatCompletionPayload(item)
    }
    return sanitized
  }
  return value
}

export const chatRagMethods = {
  normalizeChatSummary(input: any): ServerChatSummary {
    const created_at = String(input?.created_at || input?.createdAt || "")
    const updated_at =
      input?.updated_at ??
      input?.updatedAt ??
      input?.last_modified ??
      input?.lastModified ??
      null
    const state = input?.state ?? input?.conversation_state ?? null
    const last_active =
      input?.last_active ??
      input?.lastActive ??
      updated_at ??
      created_at ??
      null
    const messageCountRaw = input?.message_count ?? input?.messageCount
    const message_count =
      typeof messageCountRaw === "number"
        ? messageCountRaw
        : typeof messageCountRaw === "string" && messageCountRaw.trim().length > 0
          ? Number.parseFloat(messageCountRaw)
          : null
    const character_id = input?.character_id ?? input?.characterId ?? null
    const assistant_kind =
      input?.assistant_kind ??
      input?.assistantKind ??
      (character_id != null ? "character" : null)
    const assistant_id =
      input?.assistant_id ??
      input?.assistantId ??
      (assistant_kind === "character" && character_id != null
        ? String(character_id)
        : null)
    const scope_type =
      input?.scope_type === "global" || input?.scopeType === "global"
        ? "global"
        : input?.scope_type === "workspace" || input?.scopeType === "workspace"
          ? "workspace"
          : null
    const workspace_id =
      typeof input?.workspace_id === "string" && input.workspace_id.trim().length > 0
        ? input.workspace_id
        : typeof input?.workspaceId === "string" &&
            input.workspaceId.trim().length > 0
          ? input.workspaceId
          : null
    return {
      id: String(input?.id ?? ""),
      title: String(input?.title || ""),
      created_at,
      updated_at: updated_at ? String(updated_at) : null,
      last_active: last_active ? String(last_active) : null,
      message_count: Number.isFinite(message_count as number)
        ? (message_count as number)
        : null,
      source: input?.source ?? null,
      state: state ? String(state) : null,
      topic_label: input?.topic_label ?? input?.topicLabel ?? null,
      cluster_id: input?.cluster_id ?? input?.clusterId ?? null,
      external_ref: input?.external_ref ?? input?.externalRef ?? null,
      bm25_norm:
        typeof input?.bm25_norm === "number"
          ? input?.bm25_norm
          : typeof input?.relevance === "number"
            ? input?.relevance
            : null,
      character_id,
      assistant_kind:
        assistant_kind === "character" || assistant_kind === "persona"
          ? assistant_kind
          : null,
      assistant_id:
        assistant_id == null || assistant_id === ""
          ? null
          : String(assistant_id),
      persona_memory_mode:
        input?.persona_memory_mode === "read_only" ||
        input?.persona_memory_mode === "read_write"
          ? input.persona_memory_mode
          : input?.personaMemoryMode === "read_only" ||
              input?.personaMemoryMode === "read_write"
            ? input.personaMemoryMode
            : null,
      parent_conversation_id:
        input?.parent_conversation_id ?? input?.parentConversationId ?? null,
      root_id: input?.root_id ?? input?.rootId ?? null,
      forked_from_message_id:
        input?.forked_from_message_id ?? input?.forkedFromMessageId ?? null,
      version:
        typeof input?.version === "number"
          ? input.version
          : typeof input?.expected_version === "number"
            ? input.expected_version
            : null,
      scope_type,
      workspace_id
    }
  },

  async createChatCompletion(
    this: TldwApiClientCore,
    request: ChatCompletionRequest,
    options?: { signal?: AbortSignal }
  ): Promise<Response> {
    // Non-stream request via background
    captureChatRequestDebugSnapshot({
      endpoint: "/api/v1/chat/completions",
      method: "POST",
      mode: "non-stream",
      body: request
    })
    const res = await bgRequest<Response>({
      path: '/api/v1/chat/completions',
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: request,
      abortSignal: options?.signal
    })
    // bgRequest returns parsed data; for non-streaming chat we expect a JSON structure or text. To keep existing consumers happy, wrap as Response-like
    // For simplicity, return a minimal object with json() and text()
    const data = res as any
    const safeData = normalizeChatCompletionResponseBody(data)
    return createJsonResponseLike(safeData, { status: 200 })
  },

  async *streamChatCompletion(this: TldwApiClientCore, request: ChatCompletionRequest, options?: { signal?: AbortSignal; streamIdleTimeoutMs?: number }): AsyncGenerator<any, void, unknown> {
    request.stream = true
    captureChatRequestDebugSnapshot({
      endpoint: "/api/v1/chat/completions",
      method: "POST",
      mode: "stream",
      body: request
    })
    for await (const line of bgStream({ path: '/api/v1/chat/completions', method: 'POST', headers: { 'Content-Type': 'application/json' }, body: request, abortSignal: options?.signal, streamIdleTimeoutMs: options?.streamIdleTimeoutMs })) {
      try {
        const parsed = JSON.parse(line)
        yield parsed
      } catch (e) {
        // Ignore empty/whitespace-only lines and SSE comments (": ...")
        const trimmed = line.trim()
        if (trimmed && !trimmed.startsWith(":")) {
          console.warn("[tldw:stream] Unparseable SSE line:", trimmed.slice(0, 200))
        }
      }
    }
  },

  // RAG Methods
  async ragHealth(this: TldwApiClientCore): Promise<any> {
    return await this.request<any>({ path: '/api/v1/rag/health', method: 'GET' })
  },

  async ragSearch(this: TldwApiClientCore, query: string, options?: any): Promise<any> {
    const { timeoutMs, signal, ...rest } = options || {}
    const normalizedQuery = this.normalizeRagQuery(query)
    try {
      return await bgRequest<any>({
        path: '/api/v1/rag/search',
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: { query: normalizedQuery, ...rest },
        timeoutMs,
        abortSignal: signal
      })
    } catch (error) {
      const status = (error as { status?: number } | null)?.status
      const message = error instanceof Error ? error.message : String(error ?? '')
      const aborted =
        (error as { name?: string } | null)?.name === 'AbortError' ||
        /abort|cancel/i.test(message)
      if (aborted) {
        throw error
      }
      const shouldRetryWithoutRerank =
        status === 500 &&
        rest?.enable_reranking !== false &&
        rest?.reranking_strategy !== 'none'

      if (!shouldRetryWithoutRerank) {
        throw buildSanitizedRagSearchError(error)
      }

      // Some local/dev servers fail hard when FlashRank assets are missing.
      // Retry once with reranking disabled so retrieval still works.
      console.warn(
        '[tldw:rag] /api/v1/rag/search failed; retrying once without reranking',
        { status }
      )
      try {
        return await bgRequest<any>({
          path: '/api/v1/rag/search',
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: {
            query: normalizedQuery,
            ...rest,
            enable_reranking: false,
            reranking_strategy: 'none'
          },
          timeoutMs,
          abortSignal: signal
        })
      } catch (retryError) {
        throw buildSanitizedRagSearchError(retryError)
      }
    }
  },

  async *ragSearchStream(
    this: TldwApiClientCore,
    query: string,
    options?: any
  ): AsyncGenerator<any, void, unknown> {
    const { timeoutMs, signal, ...rest } = options || {}
    const normalizedQuery = this.normalizeRagQuery(query)
    for await (const line of bgStream({
      path: '/api/v1/rag/search/stream',
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: { query: normalizedQuery, ...rest },
      abortSignal: signal,
      streamIdleTimeoutMs: timeoutMs
    })) {
      try {
        yield JSON.parse(line)
      } catch {
        // Ignore malformed stream chunks
      }
    }
  },

  async ragSimple(this: TldwApiClientCore, query: string, options?: any): Promise<any> {
    const { timeoutMs, ...rest } = options || {}
    const normalizedQuery = this.normalizeRagQuery(query)
    return await bgRequest<any>({ path: '/api/v1/rag/simple', method: 'POST', headers: { 'Content-Type': 'application/json' }, body: { query: normalizedQuery, ...rest }, timeoutMs })
  },

  // Research / Web search
  async webSearch(this: TldwApiClientCore, options: any): Promise<any> {
    const { timeoutMs, signal, ...rest } = options || {}
    return await bgRequest<any>({
      path: "/api/v1/research/websearch",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: rest,
      timeoutMs,
      abortSignal: signal
    })
  },

  // Chat list / CRUD
  async listChatCommands(this: TldwApiClientCore): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/chat/commands",
      method: "GET"
    })
  },

  async listChats(
    this: TldwApiClientCore,
    params?: Record<string, any>,
    options?: { signal?: AbortSignal; scope?: ChatScope }
  ): Promise<ServerChatSummary[]> {
    const query = buildQuery({ ...toChatScopeParams(options?.scope), ...params })
    const data = await bgRequest<any>({
      path: `/api/v1/chats/${query}`,
      method: "GET",
      abortSignal: options?.signal
    })

    let list: any[] = []

    if (Array.isArray(data)) {
      list = data
    } else if (data && typeof data === "object") {
      const obj: any = data
      if (Array.isArray(obj.chats)) {
        list = obj.chats
      } else if (Array.isArray(obj.items)) {
        list = obj.items
      } else if (Array.isArray(obj.results)) {
        list = obj.results
      } else if (Array.isArray(obj.data)) {
        list = obj.data
      }
    }

    return list.map((c) => this.normalizeChatSummary(c))
  },

  async listChatsWithMeta(
    this: TldwApiClientCore,
    params?: Record<string, any>,
    options?: { signal?: AbortSignal; scope?: ChatScope }
  ): Promise<{ chats: ServerChatSummary[]; total: number }> {
    const query = buildQuery({ ...toChatScopeParams(options?.scope), ...params })
    const data = await bgRequest<any>({
      path: `/api/v1/chats/${query}`,
      method: "GET",
      abortSignal: options?.signal
    })

    let list: any[] = []
    let total: number | null = null

    if (Array.isArray(data)) {
      list = data
    } else if (data && typeof data === "object") {
      const obj: any = data
      if (typeof obj.total === "number") {
        total = obj.total
      } else if (typeof obj.count === "number") {
        total = obj.count
      }
      if (Array.isArray(obj.chats)) {
        list = obj.chats
      } else if (Array.isArray(obj.items)) {
        list = obj.items
      } else if (Array.isArray(obj.results)) {
        list = obj.results
      } else if (Array.isArray(obj.data)) {
        list = obj.data
      }
    }

    const chats = list.map((c) => this.normalizeChatSummary(c))
    return {
      chats,
      total: typeof total === "number" ? total : chats.length
    }
  },

  async searchConversationsWithMeta(
    this: TldwApiClientCore,
    params?: Record<string, any>,
    options?: { signal?: AbortSignal; scope?: ChatScope }
  ): Promise<{ chats: ServerChatSummary[]; total: number }> {
    const query = buildQuery({ ...toChatScopeParams(options?.scope), ...params })
    const data = await bgRequest<any>({
      path: `/api/v1/chats/conversations${query}`,
      method: "GET",
      abortSignal: options?.signal
    })

    let list: any[] = []
    let total: number | null = null

    if (Array.isArray(data)) {
      list = data
    } else if (data && typeof data === "object") {
      const obj: any = data
      if (typeof obj.total === "number") {
        total = obj.total
      } else if (typeof obj.count === "number") {
        total = obj.count
      } else if (obj.pagination && typeof obj.pagination.total === "number") {
        total = obj.pagination.total
      }
      if (Array.isArray(obj.items)) {
        list = obj.items
      } else if (Array.isArray(obj.chats)) {
        list = obj.chats
      } else if (Array.isArray(obj.results)) {
        list = obj.results
      } else if (Array.isArray(obj.data)) {
        list = obj.data
      }
    }

    const chats = list.map((item) => this.normalizeChatSummary(item))
    return {
      chats,
      total: typeof total === "number" ? total : chats.length
    }
  },

  async createChat(this: TldwApiClientCore, payload: Record<string, any>, options?: { scope?: ChatScope }): Promise<ServerChatSummary> {
    const res = await bgRequest<any>({
      path: "/api/v1/chats/",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: { ...toChatScopeParams(options?.scope), ...payload }
    })
    return this.normalizeChatSummary(res)
  },

  async completeCharacterChatTurn(
    this: TldwApiClientCore,
    chat_id: string | number,
    payload: Record<string, any>
  ): Promise<any> {
    const cid = String(chat_id)
    return await bgRequest<any>({
      path: `/api/v1/chats/${cid}/complete-v2`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  },

  async getChat(
    this: TldwApiClientCore,
    chat_id: string | number,
    options?: { scope?: ChatScope }
  ): Promise<ServerChatSummary> {
    const cid = String(chat_id)
    const query = buildQuery(toChatScopeParams(options?.scope))
    const res = await bgRequest<any>({
      path: appendPathQuery(`/api/v1/chats/${cid}`, query),
      method: "GET"
    })
    return this.normalizeChatSummary(res)
  },

  async getChatSettings(
    this: TldwApiClientCore,
    chat_id: string | number,
    options?: { scope?: ChatScope }
  ): Promise<ChatSettingsResponse> {
    const cid = String(chat_id)
    const query = buildQuery(toChatScopeParams(options?.scope))
    return await bgRequest<ChatSettingsResponse>({
      path: appendPathQuery(`/api/v1/chats/${cid}/settings`, query),
      method: "GET"
    })
  },

  async updateChatSettings(
    this: TldwApiClientCore,
    chat_id: string | number,
    settings: Record<string, unknown>,
    options?: { scope?: ChatScope }
  ): Promise<ChatSettingsResponse> {
    const cid = String(chat_id)
    const query = buildQuery(toChatScopeParams(options?.scope))
    return await bgRequest<ChatSettingsResponse>({
      path: appendPathQuery(`/api/v1/chats/${cid}/settings`, query),
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: { settings }
    })
  },

  async getChatLorebookDiagnostics(
    this: TldwApiClientCore,
    chat_id: string | number,
    params?: Record<string, any>,
    options?: { scope?: ChatScope }
  ): Promise<LorebookDiagnosticExportResponse> {
    const cid = String(chat_id)
    const query = buildQuery({
      ...toChatScopeParams(options?.scope),
      ...(params || {})
    })
    return await bgRequest<LorebookDiagnosticExportResponse>({
      path: `/api/v1/chats/${cid}/diagnostics/lorebook${query}`,
      method: "GET"
    })
  },

  async updateChat(
    this: TldwApiClientCore,
    chat_id: string | number,
    payload: Record<string, any>,
    options?: { expectedVersion?: number; scope?: ChatScope }
  ): Promise<ServerChatSummary> {
    const cid = String(chat_id)
    let expectedVersion = options?.expectedVersion
    if (expectedVersion == null) {
      try {
        expectedVersion = await this.getLatestChatVersion(cid, options)
      } catch {
        // ignore and fall back to unversioned update
      }
    }
    const attemptUpdate = async (
      versionToUse: number | undefined,
      hasRetried = false
    ): Promise<ServerChatSummary> => {
      const qp =
        buildQuery({
          ...toChatScopeParams(options?.scope),
          ...(typeof versionToUse === "number"
            ? { expected_version: versionToUse }
            : {})
        })
      try {
        const res = await bgRequest<any>({
          path: appendPathQuery(`/api/v1/chats/${cid}`, qp),
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: payload
        })
        return this.normalizeChatSummary(res)
      } catch (error) {
        if (hasRetried || !this.isVersionConflictError(error)) {
          throw error
        }
        const latestVersion = await this.getLatestChatVersion(cid, options)
        return await attemptUpdate(latestVersion, true)
      }
    }

    return await attemptUpdate(expectedVersion)
  },

  async deleteChat(
    this: TldwApiClientCore,
    chat_id: string | number,
    options?: {
      expectedVersion?: number
      hardDelete?: boolean
      scope?: ChatScope
    }
  ): Promise<void> {
    const cid = String(chat_id)
    const attemptDelete = async (
      versionToUse: number | undefined,
      hasRetried = false
    ): Promise<void> => {
      const query = buildQuery({
        ...toChatScopeParams(options?.scope),
        ...(typeof versionToUse === "number"
          ? { expected_version: versionToUse }
          : {}),
        ...(options?.hardDelete ? { hard_delete: true } : {})
      })
      try {
        await bgRequest<void>({
          path: `/api/v1/chats/${cid}${query}`,
          method: "DELETE"
        })
      } catch (error) {
        if (hasRetried || !this.isVersionConflictError(error)) {
          throw error
        }
        const latestVersion = await this.getLatestChatVersion(cid, options)
        await attemptDelete(latestVersion, true)
      }
    }

    await attemptDelete(options?.expectedVersion)
  },

  async restoreChat(
    this: TldwApiClientCore,
    chat_id: string | number,
    options?: { expectedVersion?: number; scope?: ChatScope }
  ): Promise<ServerChatSummary> {
    const cid = String(chat_id)
    let expectedVersion = options?.expectedVersion
    if (expectedVersion == null) {
      expectedVersion = await this.getLatestChatVersion(cid, options)
    }

    const attemptRestore = async (
      versionToUse: number | undefined,
      hasRetried = false
    ): Promise<ServerChatSummary> => {
      const query = buildQuery(
        {
          ...toChatScopeParams(options?.scope),
          ...(typeof versionToUse === "number"
            ? { expected_version: versionToUse }
            : {})
        }
      )
      try {
        const res = await bgRequest<any>({
          path: `/api/v1/chats/${cid}/restore${query}`,
          method: "POST"
        })
        return this.normalizeChatSummary(res)
      } catch (error) {
        if (hasRetried || !this.isVersionConflictError(error)) {
          throw error
        }
        const latestVersion = await this.getLatestChatVersion(cid, options)
        return await attemptRestore(latestVersion, true)
      }
    }

    return await attemptRestore(expectedVersion)
  },

  async createConversationShareLink(
    this: TldwApiClientCore,
    chat_id: string | number,
    payload?: {
      permission?: ConversationSharePermission
      ttl_seconds?: number
      label?: string
    },
    options?: { scope?: ChatScope }
  ): Promise<ConversationShareLinkCreateResponse> {
    const cid = String(chat_id)
    const query = buildQuery(toChatScopeParams(options?.scope))
    return await bgRequest<ConversationShareLinkCreateResponse>({
      path: appendPathQuery(
        `/api/v1/chat/conversations/${encodeURIComponent(cid)}/share-links`,
        query
      ),
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload || {},
    })
  },

  async listConversationShareLinks(
    this: TldwApiClientCore,
    chat_id: string | number,
    options?: { scope?: ChatScope }
  ): Promise<ConversationShareLinksListResponse> {
    const cid = String(chat_id)
    const query = buildQuery(toChatScopeParams(options?.scope))
    return await bgRequest<ConversationShareLinksListResponse>({
      path: appendPathQuery(
        `/api/v1/chat/conversations/${encodeURIComponent(cid)}/share-links`,
        query
      ),
      method: "GET",
    })
  },

  async revokeConversationShareLink(
    this: TldwApiClientCore,
    chat_id: string | number,
    shareId: string,
    options?: { scope?: ChatScope }
  ): Promise<{ success: boolean; share_id: string }> {
    const cid = encodeURIComponent(String(chat_id))
    const sid = encodeURIComponent(String(shareId))
    const query = buildQuery(toChatScopeParams(options?.scope))
    return await bgRequest<{ success: boolean; share_id: string }>({
      path: appendPathQuery(
        `/api/v1/chat/conversations/${cid}/share-links/${sid}`,
        query
      ),
      method: "DELETE",
    })
  },

  async resolveConversationShareLink(
    this: TldwApiClientCore,
    token: string
  ): Promise<ConversationShareLinkResolveResponse> {
    const encodedToken = encodeURIComponent(token)
    return await bgRequest<ConversationShareLinkResolveResponse>({
      path: `/api/v1/chat/shared/conversations/${encodedToken}`,
      method: "GET",
      noAuth: true,
    })
  },

  async listChatMessages(
    this: TldwApiClientCore,
    chat_id: string | number,
    params?: Record<string, any>,
    options?: { signal?: AbortSignal; scope?: ChatScope }
  ): Promise<ServerChatMessage[]> {
    const cid = String(chat_id)
    const query = buildQuery({
      ...toChatScopeParams(options?.scope),
      ...(params || {})
    })
    const cacheKey = this.getChatMessagesCacheKey(cid, query)
    const cached = this.chatMessagesCache.get(cacheKey)
    if (cached && cached.expiresAt > Date.now()) {
      return cached.value
    }
    if (cached) {
      this.chatMessagesCache.delete(cacheKey)
    }

    const inFlight = this.chatMessagesInFlight.get(cacheKey)
    if (inFlight) {
      return inFlight
    }

    const request = (async () => {
      const data = await bgRequest<any>({
        path: `/api/v1/chats/${cid}/messages${query}`,
        method: "GET",
        abortSignal: options?.signal
      })

      let list: any[] = []

      if (Array.isArray(data)) {
        list = data
      } else if (data && typeof data === "object") {
        const obj: any = data
        if (Array.isArray(obj.messages)) {
          list = obj.messages
        } else if (Array.isArray(obj.items)) {
          list = obj.items
        } else if (Array.isArray(obj.results)) {
          list = obj.results
        } else if (Array.isArray(obj.data)) {
          list = obj.data
        }
      }

      const normalized = list.map((m) => {
        const senderCandidate =
          typeof m.sender === "string"
            ? m.sender
            : typeof m.author === "string"
              ? m.author
              : typeof (m as any)?.message?.sender === "string"
                ? (m as any).message.sender
                : typeof (m as any)?.message?.author === "string"
                  ? (m as any).message.author
                  : undefined
        const roleCandidate =
          typeof m.role === "string"
            ? m.role
            : typeof senderCandidate === "string"
              ? senderCandidate
                : typeof (m as any)?.message?.role === "string"
                  ? (m as any).message.role
                  : undefined
        const senderLower =
          typeof senderCandidate === "string"
            ? senderCandidate.trim().toLowerCase()
            : ""
        const senderLooksLikeUser =
          senderLower === "user" ||
          senderLower === "human" ||
          senderLower.startsWith("user")
        const senderLooksLikeSystem =
          senderLower === "system" || senderLower.startsWith("system")
        const senderLooksLikeTool =
          senderLower === "tool" ||
          senderLower.startsWith("tool") ||
          senderLower === "function"
        const fallbackRole =
          senderLower &&
          !senderLooksLikeUser &&
          !senderLooksLikeSystem &&
          !senderLooksLikeTool
            ? "assistant"
            : "user"
        const role =
          typeof (m as any)?.is_bot === "boolean" ||
          typeof (m as any)?.isBot === "boolean"
            ? (m as any).is_bot || (m as any).isBot
              ? "assistant"
              : "user"
            : normalizeChatRole(roleCandidate, fallbackRole)
        const created_at = String(
          m.created_at || m.createdAt || m.timestamp || ""
        )
        const metadataExtraCandidate =
          (m as any).metadata_extra ?? (m as any).metadataExtra
        const metadataExtra =
          metadataExtraCandidate &&
          typeof metadataExtraCandidate === "object" &&
          !Array.isArray(metadataExtraCandidate)
            ? (metadataExtraCandidate as Record<string, unknown>)
            : undefined
        const rawPinned =
          (metadataExtra?.pinned as unknown) ?? (m as any).pinned
        const pinned =
          typeof rawPinned === "boolean"
            ? rawPinned
            : typeof rawPinned === "string"
              ? ["1", "true", "yes", "on"].includes(rawPinned.trim().toLowerCase())
              : undefined
        return {
          id: String(m.id),
          role,
          sender:
            typeof senderCandidate === "string" && senderCandidate.trim().length > 0
              ? senderCandidate
              : undefined,
          content: String(m.content ?? ""),
          created_at,
          version:
            typeof m.version === "number"
              ? m.version
              : typeof m.expected_version === "number"
                ? m.expected_version
                : undefined,
          metadata_extra: metadataExtra,
          pinned
        } as ServerChatMessage
      })
      this.chatMessagesCache.set(cacheKey, {
        value: normalized,
        expiresAt: Date.now() + CHAT_MESSAGES_CACHE_TTL_MS
      })
      return normalized
    })()

    this.chatMessagesInFlight.set(cacheKey, request)
    try {
      return await request
    } finally {
      this.chatMessagesInFlight.delete(cacheKey)
    }
  },

  async addChatMessage(
    this: TldwApiClientCore,
    chat_id: string | number,
    payload: Record<string, any>,
    options?: { scope?: ChatScope }
  ): Promise<ServerChatMessage> {
    const cid = String(chat_id)
    const query = buildQuery(toChatScopeParams(options?.scope))
    const res = await bgRequest<ServerChatMessage>({
      path: appendPathQuery(`/api/v1/chats/${cid}/messages`, query),
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
    this.invalidateChatMessagesCache(cid)
    return res
  },

  async prepareCharacterCompletion(
    this: TldwApiClientCore,
    chat_id: string | number,
    payload?: Record<string, any>,
    options?: { scope?: ChatScope }
  ): Promise<any> {
    const cid = String(chat_id)
    const body = payload || {}
    const query = buildQuery(toChatScopeParams(options?.scope))
    captureChatRequestDebugSnapshot({
      endpoint: appendPathQuery(`/api/v1/chats/${cid}/completions`, query),
      method: "POST",
      mode: "non-stream",
      body
    })
    return await bgRequest<any>({
      path: appendPathQuery(`/api/v1/chats/${cid}/completions`, query),
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body
    })
  },

  async getCharacterPromptPreview(
    this: TldwApiClientCore,
    chat_id: string | number,
    payload?: Record<string, any>,
    options?: { scope?: ChatScope }
  ): Promise<any> {
    const cid = String(chat_id)
    const query = buildQuery(toChatScopeParams(options?.scope))
    return await bgRequest<any>({
      path: appendPathQuery(`/api/v1/chats/${cid}/prompt-preview`, query),
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload || {}
    })
  },

  async persistCharacterCompletion(
    this: TldwApiClientCore,
    chat_id: string | number,
    payload: Record<string, any>,
    options?: { scope?: ChatScope }
  ): Promise<any> {
    const cid = String(chat_id)
    const query = buildQuery(toChatScopeParams(options?.scope))
    try {
      const res = await bgRequest<any>({
        path: appendPathQuery(`/api/v1/chats/${cid}/completions/persist`, query),
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: payload
      })
      this.invalidateChatMessagesCache(cid)
      return res
    } catch (error) {
      if (isSavedDegradedCharacterPersistError(error)) {
        this.invalidateChatMessagesCache(cid)
      }
      throw error
    }
  },

  async *streamCharacterChatCompletion(
    this: TldwApiClientCore,
    chat_id: string | number,
    payload?: Record<string, any>,
    options?: { signal?: AbortSignal; streamIdleTimeoutMs?: number; scope?: ChatScope }
  ): AsyncGenerator<any> {
    const cid = String(chat_id)
    const body = { ...(payload || {}), stream: true }
    const query = buildQuery(toChatScopeParams(options?.scope))
    captureChatRequestDebugSnapshot({
      endpoint: appendPathQuery(`/api/v1/chats/${cid}/complete-v2`, query),
      method: "POST",
      mode: "stream",
      body
    })
    for await (const line of bgStream({
      path: appendPathQuery(`/api/v1/chats/${cid}/complete-v2`, query),
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
      abortSignal: options?.signal,
      streamIdleTimeoutMs: options?.streamIdleTimeoutMs
    })) {
      if (!line) continue
      try {
        const parsed = JSON.parse(line)
        yield parsed
      } catch {
        yield line
      }
    }
  },

  async searchChatMessages(
    this: TldwApiClientCore,
    chat_id: string | number,
    query: string,
    limit?: number,
    options?: { scope?: ChatScope }
  ): Promise<any> {
    const cid = String(chat_id)
    const qp = buildQuery({
      ...toChatScopeParams(options?.scope),
      query,
      ...(typeof limit === "number" ? { limit } : {})
    })
    return await bgRequest<any>({
      path: `/api/v1/chats/${cid}/messages/search${qp}`,
      method: "GET"
    })
  },

  async completeChat(this: TldwApiClientCore, chat_id: string | number, payload?: Record<string, any>): Promise<any> {
    const cid = String(chat_id)
    const body = payload || {}
    captureChatRequestDebugSnapshot({
      endpoint: `/api/v1/chats/${cid}/complete`,
      method: "POST",
      mode: "non-stream",
      body
    })
    return await bgRequest<any>({ path: `/api/v1/chats/${cid}/complete`, method: 'POST', headers: { 'Content-Type': 'application/json' }, body })
  },

  async *streamCompleteChat(this: TldwApiClientCore, chat_id: string | number, payload?: Record<string, any>): AsyncGenerator<any> {
    const cid = String(chat_id)
    const body = payload || {}
    captureChatRequestDebugSnapshot({
      endpoint: `/api/v1/chats/${cid}/complete`,
      method: "POST",
      mode: "stream",
      body
    })
    for await (const line of bgStream({ path: `/api/v1/chats/${cid}/complete`, method: 'POST', headers: { 'Content-Type': 'application/json' }, body })) {
      try { yield JSON.parse(line) } catch {}
    }
  },

  // Message (single) APIs
  async getMessage(this: TldwApiClientCore, message_id: string | number): Promise<any> {
    const mid = String(message_id)
    return await bgRequest<any>({ path: `/api/v1/messages/${mid}`, method: 'GET' })
  },

  async editMessage(
    this: TldwApiClientCore,
    message_id: string | number,
    content: string,
    expectedVersion: number,
    chatId?: string | number,
    options?: { pinned?: boolean }
  ): Promise<any> {
    const mid = String(message_id)
    const qp = `?expected_version=${encodeURIComponent(String(expectedVersion))}`
    const body: Record<string, unknown> = { content }
    if (typeof options?.pinned === "boolean") {
      body.pinned = options.pinned
    }
    const res = await bgRequest<any>({
      path: `/api/v1/messages/${mid}${qp}`,
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body
    })
    if (chatId != null) {
      this.invalidateChatMessagesCache(chatId)
    }
    return res
  },

  async deleteMessage(
    this: TldwApiClientCore,
    message_id: string | number,
    expectedVersion: number,
    chatId?: string | number
  ): Promise<void> {
    const mid = String(message_id)
    const qp = `?expected_version=${encodeURIComponent(String(expectedVersion))}`
    await bgRequest<void>({
      path: `/api/v1/messages/${mid}${qp}`,
      method: 'DELETE'
    })
    if (chatId != null) {
      this.invalidateChatMessagesCache(chatId)
    }
  },

  async saveChatKnowledge(this: TldwApiClientCore, payload: {
    conversation_id: string | number
    message_id: string | number
    snippet: string
    tags?: string[]
    make_flashcard?: boolean
  }, options?: { scope?: ChatScope }): Promise<any> {
    const body = {
      ...payload,
      ...toChatScopeParams(options?.scope),
      conversation_id: String(payload.conversation_id),
      message_id: String(payload.message_id)
    }
    return await bgRequest<any>({
      path: "/api/v1/chat/knowledge/save",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body
    })
  },

  // World Books
  async listWorldBooks(this: TldwApiClientCore, include_disabled?: boolean): Promise<any> {
    const qp = include_disabled ? `?include_disabled=true` : ''
    return await bgRequest<any>({ path: `/api/v1/characters/world-books${qp}`, method: 'GET' })
  },

  async getWorldBookRuntimeConfig(this: TldwApiClientCore): Promise<{ max_recursive_depth: number }> {
    return await bgRequest<{ max_recursive_depth: number }>({
      path: "/api/v1/characters/world-books/config",
      method: "GET"
    })
  },

  async createWorldBook(this: TldwApiClientCore, payload: Record<string, any>): Promise<any> {
    return await bgRequest<any>({ path: '/api/v1/characters/world-books', method: 'POST', headers: { 'Content-Type': 'application/json' }, body: payload })
  },

  async updateWorldBook(
    this: TldwApiClientCore,
    world_book_id: number | string,
    payload: Record<string, any>,
    options?: { expectedVersion?: number }
  ): Promise<any> {
    const wid = String(world_book_id)
    const query = buildQuery(
      typeof options?.expectedVersion === "number"
        ? { expected_version: options.expectedVersion }
        : {}
    )
    return await bgRequest<any>({
      path: `/api/v1/characters/world-books/${wid}${query}`,
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: payload
    })
  },

  async deleteWorldBook(this: TldwApiClientCore, world_book_id: number | string): Promise<any> {
    const wid = String(world_book_id)
    return await bgRequest<any>({ path: `/api/v1/characters/world-books/${wid}`, method: 'DELETE' })
  },

  async listWorldBookEntries(this: TldwApiClientCore, world_book_id: number | string, enabled_only?: boolean): Promise<any> {
    const wid = String(world_book_id)
    const qp = enabled_only ? `?enabled_only=true` : ''
    return await bgRequest<any>({ path: `/api/v1/characters/world-books/${wid}/entries${qp}`, method: 'GET' })
  },

  async addWorldBookEntry(this: TldwApiClientCore, world_book_id: number | string, payload: Record<string, any>): Promise<any> {
    const wid = String(world_book_id)
    return await bgRequest<any>({ path: `/api/v1/characters/world-books/${wid}/entries`, method: 'POST', headers: { 'Content-Type': 'application/json' }, body: payload })
  },

  async updateWorldBookEntry(this: TldwApiClientCore, entry_id: number | string, payload: Record<string, any>): Promise<any> {
    const eid = String(entry_id)
    return await bgRequest<any>({ path: `/api/v1/characters/world-books/entries/${eid}`, method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: payload })
  },

  async deleteWorldBookEntry(this: TldwApiClientCore, entry_id: number | string): Promise<any> {
    const eid = String(entry_id)
    return await bgRequest<any>({ path: `/api/v1/characters/world-books/entries/${eid}`, method: 'DELETE' })
  },

  async bulkWorldBookEntries(this: TldwApiClientCore, payload: { entry_ids: number[]; operation: string; priority?: number }): Promise<any> {
    return await bgRequest<any>({
      path: '/api/v1/characters/world-books/entries/bulk',
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: payload
    })
  },

  async attachWorldBookToCharacter(
    this: TldwApiClientCore,
    character_id: number | string,
    world_book_id: number | string,
    options?: { enabled?: boolean; priority?: number }
  ): Promise<any> {
    const cid = String(character_id)
    const body: Record<string, any> = { world_book_id: Number(world_book_id) }
    if (typeof options?.enabled === "boolean") {
      body.enabled = options.enabled
    }
    if (typeof options?.priority === "number" && Number.isFinite(options.priority)) {
      body.priority = options.priority
    }
    return await bgRequest<any>({
      path: `/api/v1/characters/${cid}/world-books`,
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body
    })
  },

  async detachWorldBookFromCharacter(this: TldwApiClientCore, character_id: number | string, world_book_id: number | string): Promise<any> {
    const cid = String(character_id)
    const wid = String(world_book_id)
    return await bgRequest<any>({ path: `/api/v1/characters/${cid}/world-books/${wid}`, method: 'DELETE' })
  },

  async listCharacterWorldBooks(this: TldwApiClientCore, character_id: number | string): Promise<any> {
    const cid = String(character_id)
    return await bgRequest<any>({ path: `/api/v1/characters/${cid}/world-books`, method: 'GET' })
  },

  async processWorldBookContext(this: TldwApiClientCore, payload: {
    text: string
    world_book_ids?: number[]
    character_id?: number
    scan_depth?: number
    token_budget?: number
    recursive_scanning?: boolean
  }): Promise<WorldBookProcessResponse> {
    return await bgRequest<WorldBookProcessResponse>({
      path: "/api/v1/characters/world-books/process",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  },

  async exportWorldBook(this: TldwApiClientCore, world_book_id: number | string): Promise<any> {
    const wid = String(world_book_id)
    return await bgRequest<any>({ path: `/api/v1/characters/world-books/${wid}/export`, method: 'GET' })
  },

  async importWorldBook(this: TldwApiClientCore, request: { world_book: Record<string, any>; entries?: any[]; merge_on_conflict?: boolean }): Promise<any> {
    return await bgRequest<any>({ path: '/api/v1/characters/world-books/import', method: 'POST', headers: { 'Content-Type': 'application/json' }, body: request })
  },

  async worldBookStatistics(this: TldwApiClientCore, world_book_id: number | string): Promise<any> {
    const wid = String(world_book_id)
    return await bgRequest<any>({ path: `/api/v1/characters/world-books/${wid}/statistics`, method: 'GET' })
  },

  // Chat Dictionaries
  async createDictionary(this: TldwApiClientCore, payload: Record<string, any>): Promise<any> {
    return await bgRequest<any>({ path: '/api/v1/chat/dictionaries', method: 'POST', headers: { 'Content-Type': 'application/json' }, body: payload })
  },

  async listDictionaries(this: TldwApiClientCore, include_inactive?: boolean, include_usage?: boolean): Promise<any> {
    const params = new URLSearchParams()
    if (include_inactive) params.set('include_inactive', 'true')
    if (include_usage) params.set('include_usage', 'true')
    const qp = params.toString()
    return await bgRequest<any>({ path: `/api/v1/chat/dictionaries${qp ? `?${qp}` : ''}`, method: 'GET' })
  },

  async getDictionary(this: TldwApiClientCore, dictionary_id: number | string): Promise<any> {
    const id = String(dictionary_id)
    return await bgRequest<any>({ path: `/api/v1/chat/dictionaries/${id}`, method: 'GET' })
  },

  async updateDictionary(this: TldwApiClientCore, dictionary_id: number | string, payload: Record<string, any>): Promise<any> {
    const id = String(dictionary_id)
    return await bgRequest<any>({ path: `/api/v1/chat/dictionaries/${id}`, method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: payload })
  },

  async deleteDictionary(this: TldwApiClientCore, dictionary_id: number | string, hard_delete?: boolean): Promise<any> {
    const id = String(dictionary_id)
    const qp = hard_delete ? `?hard_delete=true` : ''
    return await bgRequest<any>({ path: `/api/v1/chat/dictionaries/${id}${qp}`, method: 'DELETE' })
  },

  async listDictionaryEntries(this: TldwApiClientCore, dictionary_id: number | string, group?: string): Promise<any> {
    const id = String(dictionary_id)
    const qp = group ? `?group=${encodeURIComponent(group)}` : ''
    return await bgRequest<any>({ path: `/api/v1/chat/dictionaries/${id}/entries${qp}`, method: 'GET' })
  },

  async addDictionaryEntry(this: TldwApiClientCore, dictionary_id: number | string, payload: Record<string, any>): Promise<any> {
    const id = String(dictionary_id)
    return await bgRequest<any>({ path: `/api/v1/chat/dictionaries/${id}/entries`, method: 'POST', headers: { 'Content-Type': 'application/json' }, body: payload })
  },

  async updateDictionaryEntry(this: TldwApiClientCore, entry_id: number | string, payload: Record<string, any>): Promise<any> {
    const eid = String(entry_id)
    return await bgRequest<any>({ path: `/api/v1/chat/dictionaries/entries/${eid}`, method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: payload })
  },

  async deleteDictionaryEntry(this: TldwApiClientCore, entry_id: number | string): Promise<any> {
    const eid = String(entry_id)
    return await bgRequest<any>({ path: `/api/v1/chat/dictionaries/entries/${eid}`, method: 'DELETE' })
  },

  async bulkDictionaryEntries(this: TldwApiClientCore, payload: {
    entry_ids: number[]
    operation: "delete" | "activate" | "deactivate" | "group"
    group_name?: string
  }): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/chat/dictionaries/entries/bulk",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  },

  async reorderDictionaryEntries(
    this: TldwApiClientCore,
    dictionary_id: number | string,
    payload: {
      entry_ids: number[]
    }
  ): Promise<any> {
    const id = String(dictionary_id)
    return await bgRequest<any>({
      path: `/api/v1/chat/dictionaries/${id}/entries/reorder`,
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  },

  async exportDictionaryMarkdown(this: TldwApiClientCore, dictionary_id: number | string): Promise<any> {
    const id = String(dictionary_id)
    return await bgRequest<any>({ path: `/api/v1/chat/dictionaries/${id}/export`, method: 'GET' })
  },

  async exportDictionaryJSON(this: TldwApiClientCore, dictionary_id: number | string): Promise<any> {
    const id = String(dictionary_id)
    return await bgRequest<any>({ path: `/api/v1/chat/dictionaries/${id}/export/json`, method: 'GET' })
  },

  async importDictionaryJSON(this: TldwApiClientCore, data: any, activate?: boolean): Promise<any> {
    return await bgRequest<any>({ path: '/api/v1/chat/dictionaries/import/json', method: 'POST', headers: { 'Content-Type': 'application/json' }, body: { data, activate: !!activate } })
  },

  async importDictionaryMarkdown(
    this: TldwApiClientCore,
    name: string,
    content: string,
    activate?: boolean
  ): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/chat/dictionaries/import",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: {
        name,
        content,
        activate: !!activate
      }
    })
  },

  async validateDictionary(this: TldwApiClientCore, payload: {
    data: Record<string, any>
    schema_version?: number
    strict?: boolean
  }): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/chat/dictionaries/validate",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  },

  async processDictionary(this: TldwApiClientCore, payload: {
    text: string
    token_budget?: number
    dictionary_id?: number | string
    max_iterations?: number
    chat_id?: string
  }): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/chat/dictionaries/process",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  },

  async dictionaryActivity(
    this: TldwApiClientCore,
    dictionary_id: number | string,
    params?: {
      limit?: number
      offset?: number
    }
  ): Promise<any> {
    const id = String(dictionary_id)
    const query = new URLSearchParams()
    if (typeof params?.limit === "number" && Number.isFinite(params.limit)) {
      query.set("limit", String(Math.max(1, Math.floor(params.limit))))
    }
    if (typeof params?.offset === "number" && Number.isFinite(params.offset)) {
      query.set("offset", String(Math.max(0, Math.floor(params.offset))))
    }
    const qp = query.toString()
    return await bgRequest<any>({
      path: `/api/v1/chat/dictionaries/${id}/activity${qp ? `?${qp}` : ""}`,
      method: "GET"
    })
  },

  async dictionaryStatistics(this: TldwApiClientCore, dictionary_id: number | string): Promise<any> {
    const id = String(dictionary_id)
    return await bgRequest<any>({ path: `/api/v1/chat/dictionaries/${id}/statistics`, method: 'GET' })
  },

  async dictionaryVersions(
    this: TldwApiClientCore,
    dictionary_id: number | string,
    params?: {
      limit?: number
      offset?: number
    }
  ): Promise<any> {
    const id = String(dictionary_id)
    const query = new URLSearchParams()
    if (typeof params?.limit === "number" && Number.isFinite(params.limit)) {
      query.set("limit", String(Math.max(1, Math.floor(params.limit))))
    }
    if (typeof params?.offset === "number" && Number.isFinite(params.offset)) {
      query.set("offset", String(Math.max(0, Math.floor(params.offset))))
    }
    const qp = query.toString()
    return await bgRequest<any>({
      path: `/api/v1/chat/dictionaries/${id}/versions${qp ? `?${qp}` : ""}`,
      method: "GET"
    })
  },

  async dictionaryVersionSnapshot(
    this: TldwApiClientCore,
    dictionary_id: number | string,
    revision: number | string
  ): Promise<any> {
    const id = String(dictionary_id)
    const rev = String(revision)
    return await bgRequest<any>({
      path: `/api/v1/chat/dictionaries/${id}/versions/${rev}`,
      method: "GET"
    })
  },

  async revertDictionaryVersion(
    this: TldwApiClientCore,
    dictionary_id: number | string,
    revision: number | string
  ): Promise<any> {
    const id = String(dictionary_id)
    const rev = String(revision)
    return await bgRequest<any>({
      path: `/api/v1/chat/dictionaries/${id}/versions/${rev}/revert`,
      method: "POST"
    })
  },

  // Chat Documents
  async generateChatDocument(this: TldwApiClientCore, payload: {
    conversation_id: string | number
    document_type: string
    provider: string
    model: string
    specific_message?: string | null
    custom_prompt?: string | null
    stream?: boolean
    async_generation?: boolean
  }): Promise<any> {
    const body = {
      ...payload,
      conversation_id: String(payload.conversation_id)
    }
    return await bgRequest<any>({
      path: "/api/v1/chat/documents/generate",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body
    })
  },

  async listChatDocuments(this: TldwApiClientCore, params?: {
    conversation_id?: string | number
    document_type?: string
    limit?: number
  }): Promise<any> {
    const query = buildQuery(params as Record<string, any>)
    return await bgRequest<any>({
      path: `/api/v1/chat/documents${query}`,
      method: "GET"
    })
  },

  async getChatDocument(this: TldwApiClientCore, document_id: number | string): Promise<any> {
    const id = String(document_id)
    return await bgRequest<any>({
      path: `/api/v1/chat/documents/${id}`,
      method: "GET"
    })
  },

  async deleteChatDocument(this: TldwApiClientCore, document_id: number | string): Promise<any> {
    const id = String(document_id)
    return await bgRequest<any>({
      path: `/api/v1/chat/documents/${id}`,
      method: "DELETE"
    })
  },

  async getChatDocumentJob(this: TldwApiClientCore, job_id: string): Promise<any> {
    const id = String(job_id)
    return await bgRequest<any>({
      path: `/api/v1/chat/documents/jobs/${id}`,
      method: "GET"
    })
  },

  async cancelChatDocumentJob(this: TldwApiClientCore, job_id: string): Promise<any> {
    const id = String(job_id)
    return await bgRequest<any>({
      path: `/api/v1/chat/documents/jobs/${id}`,
      method: "DELETE"
    })
  },

  async saveChatDocumentPrompt(this: TldwApiClientCore, payload: {
    document_type: string
    system_prompt: string
    user_prompt: string
    temperature?: number
    max_tokens?: number
  }): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/chat/documents/prompts",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  },

  async getChatDocumentPrompt(this: TldwApiClientCore, document_type: string): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/chat/documents/prompts/${encodeURIComponent(document_type)}`,
      method: "GET"
    })
  },

  async chatDocumentStatistics(this: TldwApiClientCore): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/chat/documents/statistics",
      method: "GET"
    })
  },

  // Chatbooks
  async exportChatbook(this: TldwApiClientCore, payload: {
    name: string
    description: string
    content_selections: Record<string, string[]>
    author?: string
    include_media?: boolean
    media_quality?: string
    include_embeddings?: boolean
    include_generated_content?: boolean
    tags?: string[]
    categories?: string[]
    async_mode?: boolean
  }): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/chatbooks/export",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  },

  async previewChatbook(this: TldwApiClientCore, file: File): Promise<any> {
    const data = await file.arrayBuffer()
    const name = file.name || "chatbook.zip"
    const type = file.type || "application/zip"
    return await bgUpload<any>({
      path: "/api/v1/chatbooks/preview",
      method: "POST",
      file: { name, type, data }
    })
  },

  async importChatbook(
    this: TldwApiClientCore,
    file: File,
    options?: {
      conflict_resolution?: string
      prefix_imported?: boolean
      import_media?: boolean
      import_embeddings?: boolean
      async_mode?: boolean
      content_selections?: Record<string, string[]>
    }
  ): Promise<any> {
    const data = await file.arrayBuffer()
    const name = file.name || "chatbook.zip"
    const type = file.type || "application/zip"
    const normalized: Record<string, any> = {}
    for (const [k, v] of Object.entries(options || {})) {
      if (typeof v === "undefined" || v === null) continue
      normalized[k] = typeof v === "boolean" ? (v ? "true" : "false") : v
    }
    return await bgUpload<any>({
      path: "/api/v1/chatbooks/import",
      method: "POST",
      fields: normalized,
      file: { name, type, data }
    })
  },

  async listChatbookExportJobs(this: TldwApiClientCore, params?: { limit?: number; offset?: number }): Promise<any> {
    const query = buildQuery(params as Record<string, any>)
    return await bgRequest<any>({
      path: `/api/v1/chatbooks/export/jobs${query}`,
      method: "GET"
    })
  },

  async listChatbookImportJobs(this: TldwApiClientCore, params?: { limit?: number; offset?: number }): Promise<any> {
    const query = buildQuery(params as Record<string, any>)
    return await bgRequest<any>({
      path: `/api/v1/chatbooks/import/jobs${query}`,
      method: "GET"
    })
  },

  async getChatbookExportJob(this: TldwApiClientCore, job_id: string): Promise<any> {
    const id = String(job_id)
    return await bgRequest<any>({
      path: `/api/v1/chatbooks/export/jobs/${id}`,
      method: "GET"
    })
  },

  async getChatbookImportJob(this: TldwApiClientCore, job_id: string): Promise<any> {
    const id = String(job_id)
    return await bgRequest<any>({
      path: `/api/v1/chatbooks/import/jobs/${id}`,
      method: "GET"
    })
  },

  async cancelChatbookExportJob(this: TldwApiClientCore, job_id: string): Promise<any> {
    const id = String(job_id)
    return await bgRequest<any>({
      path: `/api/v1/chatbooks/export/jobs/${id}`,
      method: "DELETE"
    })
  },

  async cancelChatbookImportJob(this: TldwApiClientCore, job_id: string): Promise<any> {
    const id = String(job_id)
    return await bgRequest<any>({
      path: `/api/v1/chatbooks/import/jobs/${id}`,
      method: "DELETE"
    })
  },

  async removeChatbookExportJob(this: TldwApiClientCore, job_id: string): Promise<any> {
    const id = String(job_id)
    return await bgRequest<any>({
      path: `/api/v1/chatbooks/export/jobs/${id}/remove`,
      method: "DELETE"
    })
  },

  async removeChatbookImportJob(this: TldwApiClientCore, job_id: string): Promise<any> {
    const id = String(job_id)
    return await bgRequest<any>({
      path: `/api/v1/chatbooks/import/jobs/${id}/remove`,
      method: "DELETE"
    })
  },

  async cleanupChatbooks(this: TldwApiClientCore): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/chatbooks/cleanup",
      method: "POST"
    })
  },

  async chatbooksHealth(this: TldwApiClientCore): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/chatbooks/health",
      method: "GET"
    })
  },

  async downloadChatbookExport(this: TldwApiClientCore, job_id: string): Promise<{ blob: Blob; filename: string }> {
    await this.ensureConfigForRequest(true)
    const response = await this.request<{
      ok: boolean
      status: number
      data?: ArrayBuffer
      error?: string
      headers?: Record<string, string>
    }>({
      path: `/api/v1/chatbooks/download/${encodeURIComponent(job_id)}`,
      method: "GET",
      headers: { Accept: "application/octet-stream" },
      responseType: "arrayBuffer",
      returnResponse: true
    })
    if (!response) {
      throw new Error("Download failed")
    }
    if (!response.ok) {
      throw new Error(response.error || `Download failed: ${response.status}`)
    }
    const headers = new Headers(response.headers || {})
    const blob = new Blob([response.data ?? new Uint8Array()], {
      type: headers.get("content-type") || "application/octet-stream"
    })
    const disposition = headers.get("content-disposition")
    let filename = `chatbook-${job_id}.zip`
    if (disposition) {
      const utfMatch = disposition.match(/filename\*=UTF-8''([^;]+)/i)
      const plainMatch = disposition.match(/filename="?([^\";]+)"?/i)
      const raw = utfMatch?.[1] || plainMatch?.[1]
      if (raw) {
        try {
          filename = decodeURIComponent(raw)
        } catch {
          filename = raw
        }
      }
    }
    return { blob, filename }
  },

  async chatQueueStatus(this: TldwApiClientCore): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/chat/queue/status",
      method: "GET"
    })
  },

  async chatQueueActivity(this: TldwApiClientCore, limit?: number): Promise<any> {
    const query = buildQuery(
      typeof limit === "number" ? { limit } : undefined
    )
    return await bgRequest<any>({
      path: `/api/v1/chat/queue/activity${query}`,
      method: "GET"
    })
  },
}

export type ChatRagMethods = typeof chatRagMethods
