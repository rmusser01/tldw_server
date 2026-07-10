/**
 * Timeline API Service
 *
 * Handles communication with tldw_server for timeline/branching operations.
 * Leverages existing TldwApiClient and bgRequest infrastructure.
 */

import { bgRequest } from '@/services/background-proxy'
import type { ServerChatSummary, ServerChatMessage } from '@/services/tldw/TldwApiClient'
import { normalizeChatRole } from '@/utils/normalize-chat-role'

// ============================================================================
// Types
// ============================================================================

type UnknownRecord = Record<string, unknown>

function isRecord(value: unknown): value is UnknownRecord {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

function toStringOrNull(value: unknown): string | null {
  if (typeof value === 'string') return value
  if (typeof value === 'number' || typeof value === 'boolean') return String(value)
  return null
}

function toStringOrEmpty(value: unknown): string {
  return toStringOrNull(value) ?? ''
}

function toStringOrNumberOrNull(value: unknown): string | number | null {
  if (typeof value === 'string' || typeof value === 'number') return value
  return null
}

export interface ConversationTreeResponse {
  conversations: ServerChatSummary[]
  root_id: string
}

export interface ForkRequest {
  parent_conversation_id: string
  forked_from_message_id?: string
  title?: string
  character_id?: number
}

export interface ForkResponse {
  id: string
  root_id: string
  parent_conversation_id: string
  forked_from_message_id?: string
  title?: string
  created_at: string
}

export interface MessageWithParent extends ServerChatMessage {
  parent_message_id?: string | null
  conversation_id: string
}

export interface SearchResult {
  message_id: string
  conversation_id: string
  content: string
  role: string
  timestamp: string
  match_score?: number
}

// ============================================================================
// Timeline API Service
// ============================================================================

export class TimelineApiService {
  /**
   * Fetch all conversations that share the same root_id as the given conversation
   * This returns the complete conversation tree for timeline visualization
   */
  async fetchConversationTree(conversationId: string): Promise<ConversationTreeResponse> {
    // First, get the conversation to find its root_id
    const conversation = await this.getConversation(conversationId)

    if (!conversation) {
      throw new Error(`Conversation ${conversationId} not found`)
    }

    const rootId = conversation.root_id || conversationId

    // Fetch all conversations with this root_id
    const conversations = await this.listConversationsByRoot(rootId)

    return {
      conversations,
      root_id: rootId
    }
  }

  /**
   * Get a single conversation by ID
   */
  async getConversation(conversationId: string): Promise<ServerChatSummary | null> {
    try {
      const data = await bgRequest<unknown>({
        path: `/api/v1/chats/${conversationId}`,
        method: 'GET'
      })
      return this.normalizeChatSummary(data)
    } catch (error) {
      console.error("Failed to get conversation", conversationId, error)
      return null
    }
  }

  /**
   * List all conversations with a specific root_id
   */
  async listConversationsByRoot(rootId: string): Promise<ServerChatSummary[]> {
    try {
      // Try with root_id filter parameter
      const data = await bgRequest<unknown>({
        path: `/api/v1/chats/?root_id=${encodeURIComponent(rootId)}`,
        method: 'GET'
      })

      let list: unknown[] = []

      if (Array.isArray(data)) {
        list = data
      } else if (isRecord(data)) {
        if (Array.isArray(data.chats)) list = data.chats
        else if (Array.isArray(data.items)) list = data.items
        else if (Array.isArray(data.results)) list = data.results
        else if (Array.isArray(data.data)) list = data.data
      }

      // Filter to only conversations with matching root_id
      return list
        .map((c) => this.normalizeChatSummary(c))
        .filter((c) => c.root_id === rootId || c.id === rootId)
    } catch (error) {
      console.error("Failed to list conversations by root", rootId, error)
      // Fallback: get single conversation
      const conv = await this.getConversation(rootId)
      return conv ? [conv] : []
    }
  }

  /**
   * Get all messages from a conversation, including parent_message_id
   */
  async getConversationMessages(conversationId: string): Promise<MessageWithParent[]> {
    try {
      const data = await bgRequest<unknown>({
        path: `/api/v1/chats/${conversationId}/messages`,
        method: 'GET'
      })

      let messages: unknown[] = []

      if (Array.isArray(data)) {
        messages = data
      } else if (isRecord(data)) {
        if (Array.isArray(data.messages)) messages = data.messages
        else if (Array.isArray(data.items)) messages = data.items
        else if (Array.isArray(data.data)) messages = data.data
      }

      return messages.map((m) => this.normalizeMessage(m, conversationId))
    } catch (error) {
      console.error("Failed to get conversation messages", conversationId, error)
      return []
    }
  }

  /**
   * Create a fork (new branch) from an existing conversation
   *
   * Note: write operations intentionally propagate errors to the caller (unlike
   * read methods that catch and return safe fallbacks).
   *
   * @throws If the request fails or the server response is invalid.
   */
  async createFork(request: ForkRequest): Promise<ForkResponse> {
    const body: Record<string, unknown> = {
      parent_conversation_id: request.parent_conversation_id
    }

    if (request.forked_from_message_id) {
      body.forked_from_message_id = request.forked_from_message_id
    }

    if (request.title) {
      body.title = request.title
    }

    if (request.character_id !== undefined) {
      body.character_id = request.character_id
    }

    const data = await bgRequest<unknown>({
      path: '/api/v1/chats/',
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body
    })

    if (!isRecord(data) || !toStringOrNull(data.id)) {
      throw new Error('Invalid response from server: missing conversation id')
    }

    return {
      id: toStringOrEmpty(data.id),
      root_id: toStringOrNull(data.root_id ?? data.rootId) || request.parent_conversation_id,
      parent_conversation_id: request.parent_conversation_id,
      forked_from_message_id: request.forked_from_message_id,
      title: toStringOrNull(data.title) ?? request.title,
      created_at: toStringOrEmpty(data.created_at ?? data.createdAt)
    }
  }

  /**
   * Add a message to a conversation with optional parent_message_id
   * This enables creating swipes (alternative responses)
   *
   * Note: write operations intentionally propagate errors to the caller (unlike
   * read methods that catch and return safe fallbacks).
   *
   * @throws If the request fails.
   */
  async addMessage(
    conversationId: string,
    content: string,
    role: 'user' | 'assistant' | 'system',
    parentMessageId?: string
  ): Promise<MessageWithParent> {
    const body: Record<string, unknown> = {
      content,
      role
    }

    if (parentMessageId) {
      body.parent_message_id = parentMessageId
    }

    const data = await bgRequest<unknown>({
      path: `/api/v1/chats/${conversationId}/messages`,
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body
    })

    return this.normalizeMessage(data, conversationId)
  }

  /**
   * Search messages across conversations
   * Uses server-side FTS5 full-text search
   */
  async searchMessages(
    query: string,
    conversationId?: string
  ): Promise<SearchResult[]> {
    try {
      const params = new URLSearchParams({ q: query })
      if (conversationId) {
        params.append('conversation_id', conversationId)
      }

      const data = await bgRequest<unknown>({
        path: `/api/v1/chats/search?${params.toString()}`,
        method: 'GET'
      })

      let results: unknown[] = []

      if (Array.isArray(data)) {
        results = data
      } else if (isRecord(data)) {
        if (Array.isArray(data.results)) results = data.results
        else if (Array.isArray(data.messages)) results = data.messages
        else if (Array.isArray(data.items)) results = data.items
      }

      return results.map((r) => {
        const result = isRecord(r) ? r : {}
        const matchScore = result.score ?? result.bm25_norm ?? result.match_score

        return {
          message_id: toStringOrEmpty(result.id ?? result.message_id),
          conversation_id: toStringOrEmpty(result.conversation_id ?? result.chat_id),
          content: toStringOrEmpty(result.content ?? result.text),
          role: toStringOrEmpty(result.role ?? result.sender) || 'user',
          timestamp: toStringOrEmpty(result.created_at ?? result.timestamp),
          match_score: typeof matchScore === 'number' ? matchScore : undefined
        }
      })
    } catch (error) {
      console.error("Failed to search messages", query, error)
      return []
    }
  }

  // ============================================================================
  // Private Helpers
  // ============================================================================

  private normalizeChatSummary(input: unknown): ServerChatSummary {
    const obj = isRecord(input) ? input : {}

    const created_at = toStringOrEmpty(obj.created_at ?? obj.createdAt)
    const updatedRaw =
      obj.updated_at ??
      obj.updatedAt ??
      obj.last_modified ??
      obj.lastModified

    return {
      id: toStringOrEmpty(obj.id),
      title: toStringOrEmpty(obj.title),
      created_at,
      updated_at: toStringOrNull(updatedRaw),
      source: toStringOrNull(obj.source),
      state: toStringOrNull(obj.state ?? obj.conversation_state),
      topic_label: toStringOrNull(obj.topic_label ?? obj.topicLabel),
      cluster_id: toStringOrNull(obj.cluster_id ?? obj.clusterId),
      external_ref: toStringOrNull(obj.external_ref ?? obj.externalRef),
      bm25_norm: typeof obj.bm25_norm === 'number' ? obj.bm25_norm : null,
      character_id: toStringOrNumberOrNull(obj.character_id ?? obj.characterId),
      parent_conversation_id:
        toStringOrNull(obj.parent_conversation_id ?? obj.parentConversationId),
      root_id: toStringOrNull(obj.root_id ?? obj.rootId)
    }
  }

  private normalizeMessage(input: unknown, conversationId: string): MessageWithParent {
    const obj = isRecord(input) ? input : {}
    const nested = isRecord(obj.message) ? obj.message : null
    const roleCandidate =
      obj.role ??
      obj.sender ??
      obj.author ??
      nested?.role ??
      nested?.sender ??
      nested?.author
    const role = normalizeChatRole(roleCandidate)

    return {
      id: toStringOrEmpty(obj.id),
      role,
      content: toStringOrEmpty(obj.content ?? obj.text),
      created_at: toStringOrEmpty(obj.created_at ?? obj.timestamp ?? obj.createdAt),
      version: typeof obj.version === 'number' ? obj.version : undefined,
      parent_message_id: toStringOrNull(obj.parent_message_id ?? obj.parentMessageId),
      conversation_id: conversationId
    }
  }
}

// ============================================================================
// Singleton Export
// ============================================================================

export const timelineApi = new TimelineApiService()
