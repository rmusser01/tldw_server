import {
  BaseMessage,
  AIMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage,
  FunctionMessage
} from "@/types/messages"
import {
  tldwChat,
  ChatMessage,
  type ChatCompletionContentPart,
  type ChatResearchContext
} from "@/services/tldw"
import type { ToolCall } from "@/types/tool-calls"
import { publishChatLoopEvent } from "@/services/chat-loop/bridge"
import { extractChatLoopEvent } from "@/services/chat-loop/stream"
import { extractStreamTransportInterruption } from "@/utils/extract-token-from-chunk"
import { extractStreamingChunkError } from "@/utils/streaming-chunks"
import type { ChatRequestDebugMetadata } from "@/services/tldw/chat-request-debug"
import { prepareChatCompletionRequest } from "@/services/tldw/TldwChat"
import type { ChatCompletionRequest } from "@/services/tldw/TldwApiClient"
import type { ServicePromptRequestScope } from "@/services/tldw/domains/service-prompts"
import { ImageSupportUnconfirmedError } from "@/utils/chat-error-message"
import type { ChatTurnIdentity } from "@/services/tldw/TldwApiClient"
import type { ChatScope } from "@/types/chat-scope"
import type { HistoryAdmissionV1 } from "@/types/history-selection"
import type { HistoryDurableResultReceiptV1, HistoryDurableSourceV1, HistoryDurableResultV1 } from "@/types/history-durable-turn"
import { canonicalHistoryJson } from "@/db/dexie/history-selection"
import { historyAdmissionReference, prepareHistoryContext } from "@/services/chat-history-selection"
import { historyDurableRequestDigest, validateHistoryDurableAdmission } from "@/services/history-durable-turn"
import { parseHistoryDurableResult, validateHistoryDurableResultReceipt } from "@/utils/history-durable-sources"

export interface ChatTldwOptions {
  model: string
  clientManagedHistory?: boolean
  routing?: {
    strategy?: "llm_router" | "rules_router"
    objective?:
      | "highest_quality"
      | "lowest_cost"
      | "lowest_latency"
      | "balanced"
    mode?: "per_turn" | "sticky_session"
    cross_provider?: boolean
    failure_mode?: "fallback_then_error" | "error"
  }
  temperature?: number
  maxTokens?: number
  topP?: number
  frequencyPenalty?: number
  presencePenalty?: number
  systemPrompt?: string
  streaming?: boolean
  reasoningEffort?: "low" | "medium" | "high"
  toolChoice?: "auto" | "none" | "required"
  tools?: Record<string, unknown>[]
  supportsMultimodal?: boolean
  saveToDb?: boolean
  conversationId?: string
  tldwTurn?: ChatTurnIdentity
  originalUserMessage?: string
  historyMessageLimit?: number
  historyMessageOrder?: string
  slashCommandInjectionMode?: string
  apiProvider?: string
  extraHeaders?: Record<string, unknown>
  extraBody?: Record<string, unknown>
  researchContext?: ChatResearchContext
  chatDebugMetadata?: ChatRequestDebugMetadata
  requestScope?: ServicePromptRequestScope
  scope?: ChatScope
  retryFailedTurn?: boolean
  clientMessageId?: string
  regenerateFromMessageId?: string
}

export class ChatTldw {
  model: string
  clientManagedHistory?: boolean
  routing?: ChatTldwOptions["routing"]
  temperature?: number
  maxTokens?: number
  topP?: number
  frequencyPenalty?: number
  presencePenalty?: number
  systemPrompt?: string
  streaming: boolean
  reasoningEffort?: "low" | "medium" | "high"
  toolChoice?: "auto" | "none" | "required"
  tools?: Record<string, unknown>[]
  supportsMultimodal: boolean
  saveToDb?: boolean
  serverMessagesAlreadyPersisted = false
  serverUserMessageId?: string
  historyAdmission?: HistoryAdmissionV1
  historyResult?: HistoryDurableResultReceiptV1
  conversationId?: string
  serverMessageId?: string
  userServerMessageId?: string
  tldwTurn?: ChatTurnIdentity
  originalUserMessage?: string
  historyMessageLimit?: number
  historyMessageOrder?: string
  slashCommandInjectionMode?: string
  apiProvider?: string
  extraHeaders?: Record<string, unknown>
  extraBody?: Record<string, unknown>
  researchContext?: ChatResearchContext
  chatDebugMetadata?: ChatRequestDebugMetadata
  requestScope?: ServicePromptRequestScope
  scope?: ChatScope
  retryFailedTurn?: boolean
  clientMessageId?: string
  regenerateFromMessageId?: string

  constructor(options: ChatTldwOptions) {
    if (options.clientManagedHistory && options.tldwTurn) {
      throw new Error("unsupported_history_durable_turn")
    }
    // Normalize model id: drop internal prefix like "tldw:" so server receives provider/model
    this.model = String(options.model || "").replace(/^tldw:/, "")
    this.routing = options.routing
    this.temperature = options.temperature ?? 0.7
    this.maxTokens = options.maxTokens
    this.topP = options.topP ?? 1
    this.frequencyPenalty = options.frequencyPenalty ?? 0
    this.presencePenalty = options.presencePenalty ?? 0
    this.systemPrompt = options.systemPrompt
    this.streaming = options.streaming ?? false
    this.reasoningEffort = options.reasoningEffort
    this.toolChoice = options.toolChoice
    this.tools = options.tools
    this.supportsMultimodal = Boolean(options.supportsMultimodal)
    this.clientManagedHistory = options.clientManagedHistory
    this.saveToDb = options.clientManagedHistory ? false : options.saveToDb
    this.conversationId = options.clientManagedHistory
      ? undefined
      : options.conversationId
    this.historyMessageLimit = options.clientManagedHistory
      ? undefined
      : options.historyMessageLimit
    this.historyMessageOrder = options.clientManagedHistory
      ? undefined
      : options.historyMessageOrder
    this.tldwTurn = options.tldwTurn
    this.originalUserMessage = options.originalUserMessage
    this.slashCommandInjectionMode = options.slashCommandInjectionMode
    this.apiProvider = options.apiProvider
    this.extraHeaders = options.extraHeaders
    this.extraBody = options.extraBody
    this.researchContext = options.researchContext
    this.chatDebugMetadata = options.chatDebugMetadata
    this.requestScope = options.requestScope
    this.scope = options.scope
    this.retryFailedTurn = options.retryFailedTurn
    this.clientMessageId = options.clientMessageId
    this.regenerateFromMessageId = options.regenerateFromMessageId
  }

  /**
   * Streaming API used by existing chat modes.
   *
   * This intentionally mirrors the previous `ollama.stream(...)` contract:
   * - yields plain string tokens
   * - optionally calls `callbacks[i].handleLLMEnd(result)` once at the end
   */
  async stream(
    messages: BaseMessage[],
    options?: {
      signal?: AbortSignal
      preparedRequest?: ChatCompletionRequest
      // Matches the shape used in normalChatMode/search/rag, where
      // callbacks: [{ handleLLMEnd(output) { ... } }]
      callbacks?: Array<{ handleLLMEnd?: (output: any) => any }>
    }
  ): Promise<AsyncGenerator<any, void, unknown>> {
    const { signal, callbacks } = options || {}
    this.serverMessageId = undefined
    this.userServerMessageId = undefined
    this.serverMessagesAlreadyPersisted = false
    this.serverUserMessageId = undefined
    this.historyAdmission = undefined
    this.historyResult = undefined
    const selected = options?.preparedRequest?.tldw_turn?.history_v1
    const selectedRequest = selected !== undefined ? options!.preparedRequest! : undefined
    if (selectedRequest) {
      if (!selected) throw new Error("invalid_history_durable_request")
      const digest = historyDurableRequestDigest(selectedRequest)
      const expectedDigest = selected!.kind === "selection"
        ? selected!.selection.request_context_digest : selected!.request_context_digest
      if (selectedRequest.stream !== true || selectedRequest.save_to_db !== true ||
          selectedRequest.model !== this.model || selectedRequest.api_provider !== this.apiProvider ||
          selectedRequest.conversation_id !== this.conversationId || !this.tldwTurn ||
          selectedRequest.tldw_turn!.user_message_id !== this.tldwTurn.user_message_id || digest !== expectedDigest)
        throw new Error("invalid_history_durable_request")
      parseHistoryDurableResult(selectedRequest.tldw_turn!.result_v1,
        selectedRequest.tldw_turn!.result_v1?.sources.length ? "rag" : "plain")
    }
    let observationChunk: Record<string, unknown> | null = null

    const tldwMessages = options?.preparedRequest
      ? []
      : this.prepareRequestMessages(messages)
    const requestedConversationId = this.conversationId
    const toolCalls: ToolCall[] = []
    // Captures the background transport's `stream_transport_interrupted`
    // sentinel. TldwChat surfaces it via onChunk (not as a text token), so we
    // hold onto the raw chunk here and re-emit it after the token loop so the
    // chat pipeline can finalize a truncated answer as interrupted.
    let interruptionChunk: Record<string, unknown> | null = null

    const applyToolCallDelta = (deltas: any[]) => {
      deltas.forEach((delta, fallbackIndex) => {
        const index =
          typeof delta?.index === "number" ? delta.index : fallbackIndex
        if (!toolCalls[index]) {
          toolCalls[index] = {
            id: typeof delta?.id === "string" ? delta.id : `tool-${index}`,
            type: "function",
            function: { name: "", arguments: "" }
          }
        }
        if (typeof delta?.id === "string") {
          toolCalls[index].id = delta.id
        }
        if (typeof delta?.type === "string") {
          toolCalls[index].type = delta.type as ToolCall["type"]
        }
        if (delta?.function) {
          if (typeof delta.function.name === "string") {
            toolCalls[index].function.name += delta.function.name
          }
          if (typeof delta.function.arguments === "string") {
            const prevArgs = toolCalls[index].function.arguments || ""
            toolCalls[index].function.arguments =
              prevArgs + delta.function.arguments
          }
        }
      })
    }

    const handleChunk = (chunk: any) => {
      if (signal?.aborted) return
      if (selectedRequest && selected) {
        const inputId = selectedRequest.tldw_turn!.user_message_id
        const binding = selected.kind === "selection" ? selected.selection : selected.admission
        const hasAdmission = chunk?.tldw_history_admission_v1 !== undefined
        const hasResult = chunk?.tldw_history_result_v1 !== undefined
        if (hasAdmission || hasResult || chunk?.tldw_message_id !== undefined || chunk?.tldw_user_message_id !== undefined) {
          if (chunk?.tldw_conversation_id !== selectedRequest.conversation_id || chunk?.tldw_user_message_id !== inputId)
            throw new Error("invalid_history_durable_receipt")
          const admission = hasAdmission
            ? validateHistoryDurableAdmission(binding, selected, inputId, chunk.tldw_history_admission_v1)
            : this.historyAdmission
          if (!admission || (this.historyAdmission && canonicalHistoryJson(this.historyAdmission) !== canonicalHistoryJson(admission)))
            throw new Error("invalid_history_admission")
          const result = hasResult ? validateHistoryDurableResultReceipt(binding, historyAdmissionReference(admission),
            historyDurableRequestDigest(selectedRequest), selectedRequest.tldw_turn!.result_v1!.sources, chunk.tldw_history_result_v1) : undefined
          if ((chunk?.tldw_message_id !== undefined && (!result || chunk.tldw_message_id !== result.result_message_id)) ||
              (result && chunk?.tldw_message_id !== result.result_message_id) ||
              (this.historyResult && result && canonicalHistoryJson(this.historyResult) !== canonicalHistoryJson(result)))
            throw new Error("invalid_history_durable_receipt")
          const changed = !this.historyAdmission || (result && !this.historyResult)
          this.historyAdmission = admission
          this.serverUserMessageId = inputId
          this.userServerMessageId = inputId
          if (result) {
            this.historyResult = result
            this.serverMessageId = result.result_message_id
            this.serverMessagesAlreadyPersisted = true
          }
          if (changed) observationChunk = {
            tldw_history_admission_v1: admission,
            ...(result ? { tldw_history_result_v1: result } : {})
          }
        }
      } else {
      const streamedConversationId =
        typeof chunk?.tldw_conversation_id === "string" &&
        chunk.tldw_conversation_id.trim().length > 0
          ? chunk.tldw_conversation_id.trim()
          : typeof chunk?.conversation_id === "string" &&
              chunk.conversation_id.trim().length > 0
            ? chunk.conversation_id.trim()
            : null
      const acceptsReceipt = !this.tldwTurn || (
        streamedConversationId === requestedConversationId &&
        chunk?.tldw_user_message_id === this.tldwTurn.user_message_id
      )
      // Nonpersisted completions also carry a request-scoped conversation UUID.
      // It must not turn local history into a link to a nonexistent server chat.
      if (acceptsReceipt && streamedConversationId && this.saveToDb !== false) {
        if (
          streamedConversationId === requestedConversationId &&
          this.tldwTurn &&
          chunk?.tldw_user_message_id === this.tldwTurn.user_message_id
        ) {
          this.serverUserMessageId = chunk.tldw_user_message_id
        }
        this.conversationId = streamedConversationId
        this.saveToDb = true
        // The server attaches the assistant message ID only after its save.
        if (
          typeof chunk?.tldw_message_id === "string" &&
          chunk.tldw_message_id.trim().length > 0
        ) {
          this.serverMessagesAlreadyPersisted = true
        }
      }
      if (acceptsReceipt && this.saveToDb !== false && typeof chunk?.tldw_message_id === "string") {
        const savedMessageId = chunk.tldw_message_id.trim()
        if (savedMessageId) this.serverMessageId = savedMessageId
      }

      if (acceptsReceipt && this.saveToDb !== false && typeof chunk?.tldw_user_message_id === "string") {
        const savedUserMessageId = chunk.tldw_user_message_id.trim()
        if (savedUserMessageId) this.userServerMessageId = savedUserMessageId
      }
      }

      const loopEvent = extractChatLoopEvent(chunk)
      if (loopEvent) {
        publishChatLoopEvent(loopEvent)
      }

      if (extractStreamTransportInterruption(chunk)) {
        interruptionChunk = chunk as Record<string, unknown>
      }

      const deltas =
        chunk?.choices?.[0]?.delta?.tool_calls ??
        chunk?.choices?.[0]?.tool_calls ??
        chunk?.tool_calls
      if (Array.isArray(deltas)) {
        applyToolCallDelta(deltas)
      }
    }

    const stream = tldwChat.streamMessage(
      tldwMessages,
      {
        // Thread the UI AbortSignal so Stop aborts the underlying request in
        // all modes (not just polling `signal.aborted` at the loop top).
        signal,
        preparedRequest: options?.preparedRequest,
        model: this.model,
        temperature: this.temperature,
        maxTokens: this.maxTokens,
        topP: this.topP,
        frequencyPenalty: this.frequencyPenalty,
        presencePenalty: this.presencePenalty,
        systemPrompt: this.systemPrompt,
        stream: true,
        reasoningEffort: this.reasoningEffort,
        routing: this.routing,
        toolChoice: this.toolChoice,
        tools: this.tools,
        saveToDb: this.saveToDb,
        retryFailedTurn: this.retryFailedTurn,
        clientMessageId: this.clientMessageId,
        regenerateFromMessageId: this.regenerateFromMessageId,
        conversationId: this.conversationId,
        tldwTurn: this.tldwTurn,
        historyMessageLimit: this.historyMessageLimit,
        historyMessageOrder: this.historyMessageOrder,
        slashCommandInjectionMode: this.slashCommandInjectionMode,
        apiProvider: this.apiProvider,
        extraHeaders: this.extraHeaders,
        extraBody: this.extraBody,
        researchContext: this.researchContext,
        chatDebugMetadata: this.chatDebugMetadata,
        requestScope: this.requestScope,
        scope: this.scope
      },
      handleChunk
    )

    const getUserMessageId = () => this.serverUserMessageId
    async function* generator() {
      let fullText = ""
      try {
        for await (const token of stream) {
          if (signal?.aborted) {
            break
          }
          if (observationChunk) {
            const observed = observationChunk
            observationChunk = null
            yield observed
          }
          if (selectedRequest && extractStreamingChunkError(token)) {
            yield token
            return
          }
          if (typeof token !== "string") continue
          fullText += token
          // Downstream chat-modes treat chunks as strings or objects with
          // `content` / `choices[0].delta.content`. Yielding the plain
          // string keeps the simple path working (`typeof chunk === 'string'`).
          yield token
        }
        if (observationChunk && !signal?.aborted) {
          yield observationChunk
          observationChunk = null
        }
        // The extension port can drop after the first byte; TldwChat surfaces a
        // synthesized `stream_transport_interrupted` sentinel via onChunk rather
        // than as a text token. Re-emit it (as an object chunk) so downstream
        // chat-modes finalize the partial answer as interrupted, not complete.
        if (interruptionChunk && !signal?.aborted) {
          yield interruptionChunk
        }
      } finally {
        // Synthesize a minimal LangChain-style result for handleLLMEnd
        if (callbacks && callbacks.length > 0) {
          const userMessageId = getUserMessageId()
          const generationInfo = toolCalls.length > 0 || userMessageId
            ? {
                ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
                ...(userMessageId ? { tldw_user_message_id: userMessageId } : {})
              }
            : undefined
          const result = {
            generations: [[{ text: fullText, generationInfo }]]
          }
          for (const cb of callbacks) {
            try {
              await cb?.handleLLMEnd?.(result)
            } catch {
              // Ignore callback errors to avoid breaking chat flow
            }
          }
        }
      }
    }

    return generator()
  }

  /** Bare body only: H1 finalization attaches the excluded history envelope afterwards. */
  prepareSelectedDurableRequest(messages: BaseMessage[], sources: readonly HistoryDurableSourceV1[]):
    ChatCompletionRequest & { tldw_turn: ChatTurnIdentity & { result_v1: HistoryDurableResultV1; history_v1?: import("@/types/history-durable-turn").HistoryDurableEnvelopeV1 } } {
    if (this.clientManagedHistory || !this.tldwTurn || this.saveToDb !== true || !this.conversationId || !this.apiProvider || !this.model ||
        this.tools?.length || this.toolChoice || this.retryFailedTurn || this.regenerateFromMessageId ||
        (this.extraBody && Object.keys(this.extraBody).length) ||
        (this.extraHeaders && Object.entries(this.extraHeaders).some(([key, value]) => key !== "X-TLDW-Loop-Compat" || value !== "1")) ||
        messages.some(message => typeof message.content !== "string"))
      throw new Error("invalid_history_durable_request")
    const result = parseHistoryDurableResult({ version: 1, sources }, sources.length ? "rag" : "plain")
    const request = prepareChatCompletionRequest(this.prepareRequestMessages(messages), {
      model: this.model, apiProvider: this.apiProvider, routing: this.routing, temperature: this.temperature,
      maxTokens: this.maxTokens, topP: this.topP, frequencyPenalty: this.frequencyPenalty,
      presencePenalty: this.presencePenalty, systemPrompt: this.systemPrompt, reasoningEffort: this.reasoningEffort,
      saveToDb: true, conversationId: this.conversationId, extraHeaders: this.extraHeaders,
      researchContext: this.researchContext, slashCommandInjectionMode: this.slashCommandInjectionMode,
      tldwTurn: this.tldwTurn
    }, true)
    // The generic builder trims/wraps user text; this protocol sends the original string exactly.
    const exact = { ...request, messages: request.messages.map(message => message.role === "user"
      ? { role: "user" as const, content: this.originalUserMessage! } : message),
      tldw_turn: { user_message_id: this.tldwTurn.user_message_id, result_v1: result } }
    return prepareHistoryContext(exact, () => true).payload as ReturnType<ChatTldw["prepareSelectedDurableRequest"]>
  }

  /** Freeze after the existing builder resolves tools/settings, before owner admission. */
  prepareClientManagedRequest(messages: BaseMessage[]): ChatCompletionRequest {
    if (!this.clientManagedHistory)
      throw new Error("client_managed_history_required")
    // Custom provider objects can contain credentials. No implicit redaction/projection.
    if (this.extraBody && Object.keys(this.extraBody).length) {
      throw new Error("unsupported_history_custom_provider_body")
    }
    if (
      this.extraHeaders &&
      Object.entries(this.extraHeaders).some(
        ([key, value]) => key !== "X-TLDW-Loop-Compat" || value !== "1"
      )
    ) {
      throw new Error("unsupported_history_custom_provider_headers")
    }
    if (
      !this.supportsMultimodal &&
      messages.some(
        (message) =>
          Array.isArray(message.content) &&
          message.content.some((part) => part.type === "image_url")
      )
    ) {
      throw new Error("unsupported_history_model_images")
    }
    return prepareChatCompletionRequest(this.convertToTldwMessages(messages), {
      model: this.model,
      routing: this.routing,
      temperature: this.temperature,
      maxTokens: this.maxTokens,
      topP: this.topP,
      frequencyPenalty: this.frequencyPenalty,
      presencePenalty: this.presencePenalty,
      systemPrompt: this.systemPrompt,
      reasoningEffort: this.reasoningEffort,
      toolChoice: this.toolChoice,
      tools: this.tools,
      apiProvider: this.apiProvider,
      extraHeaders: this.extraHeaders,
      saveToDb: false,
      researchContext: this.researchContext,
      slashCommandInjectionMode: this.slashCommandInjectionMode
    })
  }

  // Non-streaming helper mirroring the LangChain-style _generate,
  // used only internally if needed.
  async generateOnce(
    messages: BaseMessage[],
    options?: { signal?: AbortSignal }
  ): Promise<{ text: string; message: AIMessage }> {
    const tldwMessages = this.prepareRequestMessages(messages)

    const response = await tldwChat.sendMessage(tldwMessages, {
      model: this.model,
      temperature: this.temperature,
      maxTokens: this.maxTokens,
      topP: this.topP,
      frequencyPenalty: this.frequencyPenalty,
      presencePenalty: this.presencePenalty,
      systemPrompt: this.systemPrompt,
      stream: false,
      reasoningEffort: this.reasoningEffort,
      routing: this.routing,
      toolChoice: this.toolChoice,
      tools: this.tools,
      saveToDb: this.saveToDb,
      retryFailedTurn: this.retryFailedTurn,
      clientMessageId: this.clientMessageId,
      regenerateFromMessageId: this.regenerateFromMessageId,
      conversationId: this.conversationId,
      tldwTurn: this.tldwTurn,
      historyMessageLimit: this.historyMessageLimit,
      historyMessageOrder: this.historyMessageOrder,
      slashCommandInjectionMode: this.slashCommandInjectionMode,
      apiProvider: this.apiProvider,
      extraHeaders: this.extraHeaders,
      extraBody: this.extraBody,
      researchContext: this.researchContext,
      chatDebugMetadata: this.chatDebugMetadata,
      requestScope: this.requestScope,
      scope: this.scope,
      signal: options?.signal
    })

    return {
      text: response,
      message: new AIMessage(response)
    }
  }

  // We don't rely on BaseChatModel's default stream helper in the current
  // chat pipeline; see the custom `stream` implementation above which
  // matches the expected `ollama.stream` contract.

  /**
   * Non-streaming invoke helper to match the simple `.invoke()` shape used
   * by title generation and other one-off calls.
   */
  async invoke(
    messages: BaseMessage[],
    options?: { signal?: AbortSignal }
  ): Promise<{ content: string }> {
    const { text } = await this.generateOnce(messages, options)
    return { content: text }
  }

  private prepareRequestMessages(messages: BaseMessage[]): ChatMessage[] {
    const converted = this.convertToTldwMessages(messages)
    if (!this.tldwTurn) return converted
    if (!this.saveToDb || !this.conversationId?.trim()) {
      throw new Error("A durable user turn requires a persisted conversation.")
    }
    if (!this.originalUserMessage?.trim()) {
      throw new Error("A durable user turn requires its original user message.")
    }

    // The server owns history; only this request's instructions and user are sent.
    const context = converted.filter((entry) => entry.role === "system")
    const currentUser = [...converted].reverse().find((entry) => entry.role === "user")
    const currentContent = currentUser?.content
    const currentText = typeof currentContent === "string"
      ? currentContent
      : (currentContent ?? []).map((part) => {
          if (part.type !== "text") throw new Error("Durable workspace turns require text-only context.")
          return part.text
        }).join("\n")
    if (currentText && currentText !== this.originalUserMessage) {
      context.push({ role: "system", content: currentText })
    }
    return [...context, { role: "user", content: this.originalUserMessage }]
  }

  private normalizeImageUrl(
    value: unknown
  ): { url: string; detail?: "auto" | "low" | "high" | null } | null {
    if (typeof value === "string") {
      return { url: value }
    }
    if (value && typeof value === "object") {
      const candidate = value as { url?: unknown; detail?: unknown }
      if (typeof candidate.url === "string") {
        let detail: "auto" | "low" | "high" | null | undefined
        if (
          candidate.detail === "auto" ||
          candidate.detail === "low" ||
          candidate.detail === "high" ||
          candidate.detail === null
        ) {
          detail = candidate.detail as "auto" | "low" | "high" | null
        }
        return detail === undefined
          ? { url: candidate.url }
          : { url: candidate.url, detail }
      }
    }
    return null
  }

  private normalizeContentPart(
    part: unknown
  ): ChatCompletionContentPart | null {
    if (typeof part === "string") {
      return { type: "text", text: part }
    }
    if (!part || typeof part !== "object") {
      return null
    }
    const candidate = part as {
      type?: unknown
      text?: unknown
      image_url?: unknown
    }
    if (candidate.type === "text" && typeof candidate.text === "string") {
      return { type: "text", text: candidate.text }
    }
    if (candidate.type === "image_url") {
      const imageUrl = this.normalizeImageUrl(candidate.image_url)
      if (!imageUrl) return null
      return { type: "image_url", image_url: imageUrl }
    }
    return null
  }

  private coerceTextContent(content: unknown): string {
    if (typeof content === "string") {
      return content
    }
    if (!Array.isArray(content)) {
      return ""
    }
    return content
      .map((item) => {
        if (typeof item === "string") return item
        if (item && typeof item === "object") {
          const candidate = item as { type?: unknown; text?: unknown }
          if (candidate.type === "text" && typeof candidate.text === "string") {
            return candidate.text
          }
        }
        return ""
      })
      .filter(Boolean)
      .join(" ")
  }

  private normalizeUserContent(
    content: unknown
  ): string | ChatCompletionContentPart[] {
    if (typeof content === "string") {
      return content
    }
    if (!Array.isArray(content)) {
      return ""
    }
    const parts = content
      .map((item) => this.normalizeContentPart(item))
      .filter(Boolean) as ChatCompletionContentPart[]
    if (parts.length === 0) {
      return ""
    }
    const hasImage = parts.some((part) => part.type === "image_url")
    if (!hasImage) {
      return this.coerceTextContent(content)
    }
    return parts
  }

  private convertToTldwMessages(messages: BaseMessage[]): ChatMessage[] {
    return messages.map((msg) => {
      if (
        msg instanceof SystemMessage ||
        (msg as unknown as { role?: string }).role === "system"
      ) {
        return {
          role: "system",
          content: this.coerceTextContent(msg.content)
        }
      }
      if (msg instanceof ToolMessage) {
        return {
          role: "tool",
          content: this.coerceTextContent(msg.content),
          tool_call_id: msg.tool_call_id
        }
      }
      if (
        msg instanceof FunctionMessage ||
        msg.additional_kwargs?.function_call
      ) {
        throw new Error("unsupported_history_function_message")
      }
      if (msg instanceof AIMessage) {
        return {
          role: "assistant",
          content: this.coerceTextContent(msg.content),
          ...(Array.isArray(msg.additional_kwargs?.tool_calls)
            ? { tool_calls: msg.additional_kwargs.tool_calls as ToolCall[] }
            : {})
        }
      }
      if (msg instanceof HumanMessage) {
        if (
          !this.supportsMultimodal &&
          Array.isArray(msg.content) &&
          msg.content.some((part) => part?.type === "image_url")
        ) {
          throw new ImageSupportUnconfirmedError(this.retryFailedTurn === true)
        }
        return {
          role: "user",
          content: this.supportsMultimodal
            ? this.normalizeUserContent(msg.content)
            : this.coerceTextContent(msg.content)
        }
      }

      return {
        role: "user",
        content: this.coerceTextContent(msg.content)
      }
    })
  }

  // Method to check if tldw is available
  static async isAvailable(): Promise<boolean> {
    try {
      return await tldwChat.isReady()
    } catch {
      return false
    }
  }

  // Method to cancel the current stream
  cancelStream(): void {
    tldwChat.cancelStream()
  }
}
