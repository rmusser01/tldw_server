import {
  tldwClient,
  ChatMessage,
  ChatCompletionRequest,
  type ChatCompletionContentPart,
  type ChatResearchContext
} from "./TldwApiClient"
import { extractTokenFromChunk } from "@/utils/extract-token-from-chunk"
import {
  captureChatRequestDebugSnapshot,
  getLastChatCompletionDebugSnapshot,
  type ChatCompletionDebugSnapshot
} from "./chat-request-debug"

type ToolFunctionSchema = Record<string, unknown>
type ToolFunction = {
  name: string
  description?: string
  parameters?: ToolFunctionSchema
}
type OpenAiTool = {
  type: "function"
  function: ToolFunction
}

const TOOL_NAME_PATTERN = /^[a-zA-Z0-9_-]{1,64}$/

const normalizeProvider = (value?: string): string =>
  String(value || "").trim().toLowerCase()

const shouldUseTextParts = (provider?: string, model?: string): boolean => {
  const normalized = normalizeProvider(provider)
  if (normalized) return normalized === "google" || normalized === "gemini"
  const normalizedModel = String(model || "").toLowerCase()
  return normalizedModel.includes("gemini")
}

const normalizeMessagesForProvider = (
  messages: ChatMessage[],
  provider?: string,
  model?: string
): ChatMessage[] => {
  if (!shouldUseTextParts(provider, model)) return messages
  return messages.map((msg) => {
    if (msg.role !== "user") return msg
    if (typeof msg.content === "string") {
      return {
        ...msg,
        content: [{ type: "text", text: msg.content }]
      }
    }
    return msg
  })
}

const normalizeSystemPrompt = (value?: string): string | undefined => {
  if (typeof value !== "string") return undefined
  const trimmed = value.trim()
  return trimmed.length > 0 ? trimmed : undefined
}

const coercePositiveTimeout = (
  value: unknown,
  fallback: number
): number => {
  if (typeof value === "number" && Number.isFinite(value) && value > 0) {
    return Math.trunc(value)
  }
  return fallback
}

const hasReasoningProgress = (chunk: unknown): boolean => {
  if (!chunk || typeof chunk !== "object") return false
  const record = chunk as Record<string, unknown>
  const reasoningDelta =
    (record.choices as Array<Record<string, unknown>> | undefined)?.[0]?.delta &&
    typeof (record.choices as Array<Record<string, unknown>>)[0]?.delta === "object"
      ? (
          (record.choices as Array<Record<string, unknown>>)[0]
            ?.delta as Record<string, unknown>
        )?.reasoning_content
      : undefined
  if (typeof reasoningDelta === "string" && reasoningDelta.length > 0) {
    return true
  }
  const additionalKwargs =
    typeof record.additional_kwargs === "object" &&
    record.additional_kwargs !== null
      ? (record.additional_kwargs as Record<string, unknown>)
      : null
  return Boolean(
    typeof additionalKwargs?.reasoning_content === "string" &&
      additionalKwargs.reasoning_content.length > 0
  )
}

const hasVisibleAssistantProgress = (chunk: unknown): boolean =>
  extractTokenFromChunk(chunk).length > 0 || hasReasoningProgress(chunk)

const readErrorMessage = (error: unknown, fallback = "Aborted"): string => {
  const seen = new Set<unknown>()
  let current: unknown = error
  let firstMessage: string | null = null

  while (current && !seen.has(current)) {
    seen.add(current)
    if (current instanceof Error && current.message) {
      firstMessage ??= current.message
      if (current.name === "AbortError") return current.message
      if (current.message.toLowerCase().includes("abort")) return current.message
    }
    current = (current as { cause?: unknown } | null)?.cause
  }

  return firstMessage ?? fallback
}

const isAbortLikeError = (error: unknown): boolean => {
  const seen = new Set<unknown>()
  let current: unknown = error

  while (current && !seen.has(current)) {
    seen.add(current)
    if (current instanceof Error) {
      if (current.name === "AbortError") return true
      if (current.message.toLowerCase().includes("abort")) return true
    }
    current = (current as { cause?: unknown } | null)?.cause
  }
  return false
}

const createAbortError = (message = "Aborted"): Error => {
  const abortError = new Error(message)
  abortError.name = "AbortError"
  return abortError
}

const resolveStreamTimeoutError = (
  reason: "startup" | "idle" | null,
  sawVisibleProgress: boolean
): Error | null => {
  if (reason === "startup" && !sawVisibleProgress) {
    return new Error("Chat response timed out before any visible output arrived.")
  }
  if (reason === "idle") {
    return new Error("Chat response stalled after visible output began.")
  }
  return null
}

const sanitizeUserContent = (
  content: string | ChatCompletionContentPart[]
): string | ChatCompletionContentPart[] | null => {
  if (typeof content === "string") {
    const trimmed = content.trim()
    return trimmed.length > 0 ? trimmed : null
  }
  if (!Array.isArray(content)) return null

  const cleaned = content
    .map((part) => {
      if (!part || typeof part !== "object") return null
      if (part.type === "text") {
        const text = typeof part.text === "string" ? part.text.trim() : ""
        return text.length > 0 ? { type: "text", text } : null
      }
      if (part.type === "image_url") {
        const candidate = part.image_url
        const url = typeof candidate?.url === "string" ? candidate.url.trim() : ""
        if (!url) return null
        const detail = candidate?.detail
        if (detail === "auto" || detail === "low" || detail === "high" || detail === null) {
          return { type: "image_url", image_url: { url, detail } }
        }
        return { type: "image_url", image_url: { url } }
      }
      return null
    })
    .filter(Boolean) as ChatCompletionContentPart[]

  return cleaned.length > 0 ? cleaned : null
}

const sanitizeMessages = (messages: ChatMessage[]): ChatMessage[] => {
  return messages
    .map((message) => {
      if (message.role === "system") {
        const content = message.content.trim()
        if (!content) return null
        return { ...message, content }
      }

      if (message.role === "user") {
        const content = sanitizeUserContent(message.content)
        if (content == null) return null
        return { ...message, content } as ChatMessage
      }

      if (message.role === "assistant") {
        const content =
          typeof message.content === "string"
            ? message.content.trim()
            : message.content
        const hasToolCalls =
          Array.isArray(message.tool_calls) && message.tool_calls.length > 0
        const hasFunctionCall =
          typeof message.function_call?.name === "string" &&
          message.function_call.name.trim().length > 0
        if (typeof content === "string" && content.length > 0) {
          return { ...message, content }
        }
        if (hasToolCalls || hasFunctionCall) {
          return { ...message, content: null }
        }
        return null
      }

      const rawToolCallId = (message as any).tool_call_id
      const toolCallId =
        typeof rawToolCallId === "string" ? rawToolCallId.trim() : ""
      const rawContent = message.content
      const content =
        typeof rawContent === "string" ? rawContent.trim() : ""
      if (!toolCallId || !content) return null
      return { ...message, tool_call_id: toolCallId, content }
    })
    .filter(Boolean) as ChatMessage[]
}

const buildRequestMessages = (
  messages: ChatMessage[],
  options: TldwChatOptions
): ChatMessage[] => {
  const sanitizedMessages = sanitizeMessages(messages)
  const systemPrompt = normalizeSystemPrompt(options.systemPrompt)
  let withSystemPrompt: ChatMessage[]
  if (systemPrompt) {
    if (sanitizedMessages[0]?.role === "system") {
      // Replace existing system message with the user-configured one
      withSystemPrompt = [
        { role: "system", content: systemPrompt } as ChatMessage,
        ...sanitizedMessages.slice(1)
      ]
    } else {
      withSystemPrompt = [
        { role: "system", content: systemPrompt } as ChatMessage,
        ...sanitizedMessages
      ]
    }
  } else {
    withSystemPrompt = sanitizedMessages
  }

  return normalizeMessagesForProvider(
    withSystemPrompt,
    options.apiProvider,
    options.model
  )
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value)

const sanitizeToolName = (name: string): string | null => {
  const trimmed = name.trim()
  if (!trimmed) return null
  if (TOOL_NAME_PATTERN.test(trimmed)) return trimmed

  let sanitized = trimmed.replace(/[^a-zA-Z0-9_-]+/g, "_")
  sanitized = sanitized.replace(/^_+|_+$/g, "")
  if (!sanitized) return null
  if (sanitized.length > 64) sanitized = sanitized.slice(0, 64)
  return TOOL_NAME_PATTERN.test(sanitized) ? sanitized : null
}

const normalizeChatTools = (
  tools?: Record<string, unknown>[]
): Record<string, unknown>[] | undefined => {
  if (!Array.isArray(tools) || tools.length === 0) return undefined

  const seen = new Set<string>()
  const normalized = tools
    .map((tool) => {
      if (!isRecord(tool)) return null

      const functionRecord = isRecord(tool.function) ? tool.function : undefined
      const rawName =
        (typeof tool.name === "string" && tool.name) ||
        (functionRecord &&
        typeof functionRecord.name === "string" &&
        functionRecord.name
          ? functionRecord.name
          : "")
      const name = rawName ? sanitizeToolName(rawName) : null
      if (!name || seen.has(name)) return null

      if (rawName !== name) {
        console.warn(
          `[tldw] Tool name "${rawName}" normalized to "${name}" to satisfy server schema.`
        )
      }
      seen.add(name)

      const description =
        typeof tool.description === "string"
          ? tool.description
          : functionRecord &&
              typeof functionRecord.description === "string"
            ? functionRecord.description
            : undefined

      const schemaCandidates: Array<unknown> = [
        functionRecord?.parameters,
        tool.parameters,
        tool.input_schema,
        tool.json_schema
      ]
      const parameters =
        (schemaCandidates.find((candidate) =>
          isRecord(candidate)
        ) as ToolFunctionSchema | undefined) || {
          type: "object",
          properties: {}
        }

      return {
        type: "function",
        function: {
          name,
          description,
          parameters
        }
      } as OpenAiTool
    })
    .filter(Boolean) as Record<string, unknown>[]

  return normalized.length > 0 ? normalized : undefined
}

export interface TldwChatOptions {
  model: string
  routing?: ChatCompletionRequest["routing"]
  temperature?: number
  logprobs?: boolean
  topLogprobs?: number
  maxTokens?: number
  topP?: number
  frequencyPenalty?: number
  presencePenalty?: number
  stream?: boolean
  systemPrompt?: string
  reasoningEffort?: "low" | "medium" | "high"
  toolChoice?: "auto" | "none" | "required"
  tools?: Record<string, unknown>[]
  saveToDb?: boolean
  conversationId?: string
  historyMessageLimit?: number
  historyMessageOrder?: string
  slashCommandInjectionMode?: string
  apiProvider?: string
  extraHeaders?: Record<string, unknown>
  extraBody?: Record<string, unknown>
  thinkingBudgetTokens?: number
  grammarMode?: "none" | "library" | "inline"
  grammarId?: string
  grammarInline?: string
  grammarOverride?: string
  jsonMode?: boolean
  researchContext?: ChatResearchContext
}
export { getLastChatCompletionDebugSnapshot }
export type { ChatCompletionDebugSnapshot }

export interface ChatStreamChunk {
  id?: string
  object?: string
  created?: number
  model?: string
  choices?: Array<{
    index: number
    delta: {
      role?: string
      content?: string
      tool_calls?: Array<{
        index?: number
        id?: string
        type?: "function"
        function?: {
          name?: string
          arguments?: string
        }
      }>
    }
    logprobs?: {
      content?: Array<{
        token?: string
        logprob?: number
        top_logprobs?: Array<{
          token?: string
          logprob?: number
        }>
      }>
      tokens?: string[]
      token_logprobs?: number[]
      top_logprobs?: Array<Record<string, number>>
    }
    finish_reason?: string | null
  }>
}

export class TldwChatService {
  private currentController: AbortController | null = null

  /**
   * Send a chat completion request
   */
  async sendMessage(
    messages: ChatMessage[],
    options: TldwChatOptions,
    onResponse?: (response: unknown) => void
  ): Promise<string> {
    try {
      await tldwClient.initialize()
      const normalizedTools = normalizeChatTools(options.tools)
      const toolChoice = normalizedTools ? options.toolChoice : undefined
      const requestMessages = buildRequestMessages(messages, options)
      if (requestMessages.length === 0) {
        throw new Error(
          "Cannot send chat request without any messages. Add a user message or a system prompt."
        )
      }

      const request: ChatCompletionRequest = {
        messages: requestMessages,
        model: options.model,
        routing: options.routing,
        stream: false,
        temperature: options.temperature,
        logprobs: options.logprobs,
        top_logprobs:
          options.logprobs && Number.isFinite(options.topLogprobs)
            ? options.topLogprobs
            : undefined,
        max_tokens: options.maxTokens,
        top_p: options.topP,
        frequency_penalty: options.frequencyPenalty,
        presence_penalty: options.presencePenalty,
        reasoning_effort: options.reasoningEffort,
        tool_choice: toolChoice,
        tools: normalizedTools,
        save_to_db: options.saveToDb,
        conversation_id: options.conversationId,
        history_message_limit: options.historyMessageLimit,
        history_message_order: options.historyMessageOrder,
        slash_command_injection_mode: options.slashCommandInjectionMode,
        api_provider: options.apiProvider,
        extra_headers: options.extraHeaders,
        extra_body: options.extraBody,
        thinking_budget_tokens: options.thinkingBudgetTokens,
        grammar_mode: options.grammarMode,
        grammar_id: options.grammarId,
        grammar_inline: options.grammarInline,
        grammar_override: options.grammarOverride,
        response_format: options.jsonMode ? { type: "json_object" } : undefined,
        research_context: options.researchContext
      }
      captureChatRequestDebugSnapshot({
        endpoint: "/api/v1/chat/completions",
        method: "POST",
        mode: "non-stream",
        body: request
      })

      const response = await tldwClient.createChatCompletion(request)
      const data = await response.json().catch(() => null)
      if (onResponse) {
        onResponse(data)
      }
      const content = data?.choices?.[0]?.message?.content || data?.content || data?.text
      if (typeof content === 'string') {
        return content
      }
      throw new Error('Invalid response format from tldw server')
    } catch (error) {
      console.error('Chat completion failed:', error)
      throw new Error('Chat completion failed', { cause: error })
    }
  }

  /**
   * Send a streaming chat completion request
   */
  async *streamMessage(
    messages: ChatMessage[],
    options: TldwChatOptions,
    onChunk?: (chunk: ChatStreamChunk) => void
  ): AsyncGenerator<string, void, unknown> {
    try {
      await tldwClient.initialize()
      const normalizedTools = normalizeChatTools(options.tools)
      const toolChoice = normalizedTools ? options.toolChoice : undefined
      const requestMessages = buildRequestMessages(messages, options)
      if (requestMessages.length === 0) {
        throw new Error(
          "Cannot send chat request without any messages. Add a user message or a system prompt."
        )
      }

      // Cancel any existing stream
      this.cancelStream()

      // Create new abort controller
      this.currentController = new AbortController()
      const cfg = (await tldwClient.getConfig().catch(() => null)) as
        | {
            chatRequestTimeoutMs?: number
            chatStartupTimeoutMs?: number
            chatStreamIdleTimeoutMs?: number
          }
        | null

      const request: ChatCompletionRequest = {
        messages: requestMessages,
        model: options.model,
        routing: options.routing,
        stream: true,
        temperature: options.temperature,
        logprobs: options.logprobs,
        top_logprobs:
          options.logprobs && Number.isFinite(options.topLogprobs)
            ? options.topLogprobs
            : undefined,
        max_tokens: options.maxTokens,
        top_p: options.topP,
        frequency_penalty: options.frequencyPenalty,
        presence_penalty: options.presencePenalty,
        reasoning_effort: options.reasoningEffort,
        tool_choice: toolChoice,
        tools: normalizedTools,
        save_to_db: options.saveToDb,
        conversation_id: options.conversationId,
        history_message_limit: options.historyMessageLimit,
        history_message_order: options.historyMessageOrder,
        slash_command_injection_mode: options.slashCommandInjectionMode,
        api_provider: options.apiProvider,
        extra_headers: options.extraHeaders,
        extra_body: options.extraBody,
        thinking_budget_tokens: options.thinkingBudgetTokens,
        grammar_mode: options.grammarMode,
        grammar_id: options.grammarId,
        grammar_inline: options.grammarInline,
        grammar_override: options.grammarOverride,
        response_format: options.jsonMode ? { type: "json_object" } : undefined,
        research_context: options.researchContext
      }
      captureChatRequestDebugSnapshot({
        endpoint: "/api/v1/chat/completions",
        method: "POST",
        mode: "stream",
        body: request
      })

      const startupTimeoutMs = coercePositiveTimeout(
        cfg?.chatStartupTimeoutMs,
        10_000
      )
      const streamIdleTimeoutMs = coercePositiveTimeout(
        cfg?.chatStreamIdleTimeoutMs,
        30_000
      )
      const stream = tldwClient.streamChatCompletion(request, {
        signal: this.currentController.signal,
        streamIdleTimeoutMs
      })

      let idleTimer: ReturnType<typeof setTimeout> | null = null
      let startupTimer: ReturnType<typeof setTimeout> | null = null
      const controller = this.currentController
      let sawVisibleProgress = false
      let timeoutReason: "startup" | "idle" | null = null

      const clearIdleTimer = () => {
        if (idleTimer) {
          clearTimeout(idleTimer)
          idleTimer = null
        }
      }

      const clearStartupTimer = () => {
        if (startupTimer) {
          clearTimeout(startupTimer)
          startupTimer = null
        }
      }

      const abortForTimeout = (reason: "startup" | "idle") => {
        timeoutReason = reason
        controller?.abort()
      }

      const armStartupTimer = () => {
        clearStartupTimer()
        startupTimer = setTimeout(() => {
          abortForTimeout("startup")
        }, startupTimeoutMs)
      }

      const resetIdleTimer = () => {
        clearIdleTimer()
        idleTimer = setTimeout(() => {
          abortForTimeout("idle")
        }, streamIdleTimeoutMs)
      }

      try {
        armStartupTimer()
        try {
          for await (const chunk of stream) {
            if (hasVisibleAssistantProgress(chunk)) {
              sawVisibleProgress = true
              clearStartupTimer()
              resetIdleTimer()
            }

            // Check if stream was cancelled
            if (controller?.signal.aborted) {
              break
            }

            // Call the onChunk callback if provided
            if (onChunk) {
              onChunk(chunk)
            }

            const token = extractTokenFromChunk(chunk)
            if (token) {
              yield token
            }
          }
        } catch (error) {
          const timeoutError = resolveStreamTimeoutError(
            timeoutReason,
            sawVisibleProgress
          )
          if (timeoutError) {
            throw timeoutError
          }
          if (controller?.signal.aborted && isAbortLikeError(error)) {
            throw createAbortError(readErrorMessage(error))
          }
          throw error
        }
        const timeoutError = resolveStreamTimeoutError(
          timeoutReason,
          sawVisibleProgress
        )
        if (timeoutError) {
          throw timeoutError
        }
        if (controller?.signal.aborted) {
          throw createAbortError()
        }
      } finally {
        clearIdleTimer()
        clearStartupTimer()
      }
    } catch (error) {
      console.error('Stream completion failed:', error)
      if (isAbortLikeError(error)) {
        throw createAbortError(readErrorMessage(error))
      }
      throw new Error('Stream completion failed', { cause: error })
    } finally {
      this.currentController = null
    }
  }

  /**
   * Cancel the current streaming request
   */
  cancelStream(): void {
    if (this.currentController) {
      this.currentController.abort()
      this.currentController = null
    }
  }

  /**
   * Create a conversation with context
   */
  buildConversation(
    history: ChatMessage[],
    newMessage: string,
    systemPrompt?: string
  ): ChatMessage[] {
    const messages: ChatMessage[] = []

    // Add system prompt if provided
    if (systemPrompt) {
      messages.push({ role: 'system', content: systemPrompt })
    }

    // Add history
    messages.push(...history)

    // Add new user message
    messages.push({ role: 'user', content: newMessage })

    return messages
  }

  /**
   * Format a response for display
   */
  formatResponse(content: string): string {
    // Basic formatting - can be enhanced
    return content.trim()
  }

  /**
   * Check if the service is ready
   */
  async isReady(): Promise<boolean> {
    try {
      await tldwClient.initialize()
      return await tldwClient.healthCheck()
    } catch {
      return false
    }
  }

  /**
   * Get token estimate for messages
   * This is a rough estimate - actual tokens depend on the model
   */
  estimateTokens(messages: ChatMessage[]): number {
    let totalChars = 0
    for (const msg of messages) {
      const content = msg.content
      if (typeof content === "string") {
        totalChars += content.length
      } else if (Array.isArray(content)) {
        let partTokens = 0
        for (const part of content as any[]) {
          if (typeof part === "string") {
            partTokens += part.length
          } else if (part?.type === "text" && typeof part.text === "string") {
            partTokens += part.text.length
          } else if (part?.type === "image_url") {
            // Rough image token estimate: 85 tokens for low detail, 765 for high
            const detail = part.image_url?.detail
            const imageTokens = detail === "low" ? 85 : 765
            partTokens += imageTokens * 4 // convert to char-equivalents
          }
        }
        totalChars += partTokens
      } else if (content != null) {
        try {
          totalChars += JSON.stringify(content).length
        } catch {
          // Fallback if serialization fails
        }
      }
    }
    // Rough estimate: 1 token ≈ 4 characters
    return Math.ceil(totalChars / 4)
  }

  /**
   * Truncate messages to fit within token limit
   */
  truncateMessages(
    messages: ChatMessage[],
    maxTokens: number,
    keepSystemPrompt: boolean = true
  ): ChatMessage[] {
    const result: ChatMessage[] = []
    let currentTokens = 0

    // Keep system prompt if requested
    if (keepSystemPrompt && messages[0]?.role === 'system') {
      result.push(messages[0])
      currentTokens += this.estimateTokens([messages[0]])
    }

    // Add messages from the end (most recent first)
    for (let i = messages.length - 1; i >= 0; i--) {
      const msg = messages[i]

      // Skip system prompt if already added
      if (msg.role === 'system' && result.length > 0 && result[0].role === 'system') {
        continue
      }

      const msgTokens = this.estimateTokens([msg])
      if (currentTokens + msgTokens <= maxTokens) {
        result.splice(keepSystemPrompt ? 1 : 0, 0, msg)
        currentTokens += msgTokens
      } else {
        break
      }
    }

    return result
  }
}

// Singleton instance
export const tldwChat = new TldwChatService()
