import {
  BaseMessage,
  AIMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage
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
import type { ChatRequestDebugMetadata } from "@/services/tldw/chat-request-debug"

export interface ChatTldwOptions {
  model: string
  routing?: {
    strategy?: "llm_router" | "rules_router"
    objective?: "highest_quality" | "lowest_cost" | "lowest_latency" | "balanced"
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
  historyMessageLimit?: number
  historyMessageOrder?: string
  slashCommandInjectionMode?: string
  apiProvider?: string
  extraHeaders?: Record<string, unknown>
  extraBody?: Record<string, unknown>
  researchContext?: ChatResearchContext
  chatDebugMetadata?: ChatRequestDebugMetadata
}

export class ChatTldw {
  model: string
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
  conversationId?: string
  historyMessageLimit?: number
  historyMessageOrder?: string
  slashCommandInjectionMode?: string
  apiProvider?: string
  extraHeaders?: Record<string, unknown>
  extraBody?: Record<string, unknown>
  researchContext?: ChatResearchContext
  chatDebugMetadata?: ChatRequestDebugMetadata

  constructor(options: ChatTldwOptions) {
    // Normalize model id: drop internal prefix like "tldw:" so server receives provider/model
    this.model = String(options.model || '').replace(/^tldw:/, '')
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
    this.saveToDb = options.saveToDb
    this.conversationId = options.conversationId
    this.historyMessageLimit = options.historyMessageLimit
    this.historyMessageOrder = options.historyMessageOrder
    this.slashCommandInjectionMode = options.slashCommandInjectionMode
    this.apiProvider = options.apiProvider
    this.extraHeaders = options.extraHeaders
    this.extraBody = options.extraBody
    this.researchContext = options.researchContext
    this.chatDebugMetadata = options.chatDebugMetadata
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
      // Matches the shape used in normalChatMode/search/rag, where
      // callbacks: [{ handleLLMEnd(output) { ... } }]
      callbacks?: Array<{ handleLLMEnd?: (output: any) => any }>
    }
  ): Promise<AsyncGenerator<any, void, unknown>> {
    const { signal, callbacks } = options || {}

    const tldwMessages = this.convertToTldwMessages(messages)
    const toolCalls: ToolCall[] = []

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
            toolCalls[index].function.arguments = prevArgs + delta.function.arguments
          }
        }
      })
    }

    const handleChunk = (chunk: any) => {
      const loopEvent = extractChatLoopEvent(chunk)
      if (loopEvent) {
        publishChatLoopEvent(loopEvent)
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
        conversationId: this.conversationId,
        historyMessageLimit: this.historyMessageLimit,
        historyMessageOrder: this.historyMessageOrder,
        slashCommandInjectionMode: this.slashCommandInjectionMode,
        apiProvider: this.apiProvider,
        extraHeaders: this.extraHeaders,
        extraBody: this.extraBody,
        researchContext: this.researchContext,
        chatDebugMetadata: this.chatDebugMetadata
      },
      handleChunk
    )

    async function* generator() {
      let fullText = ""
      try {
        for await (const token of stream) {
          if (signal?.aborted) {
            break
          }
          if (typeof token !== "string") continue
          fullText += token
          // Downstream chat-modes treat chunks as strings or objects with
          // `content` / `choices[0].delta.content`. Yielding the plain
          // string keeps the simple path working (`typeof chunk === 'string'`).
          yield token
        }
      } finally {
        // Synthesize a minimal LangChain-style result for handleLLMEnd
        if (callbacks && callbacks.length > 0) {
          const generationInfo =
            toolCalls.length > 0 ? { tool_calls: toolCalls } : undefined
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

  // Non-streaming helper mirroring the LangChain-style _generate,
  // used only internally if needed.
  async generateOnce(
    messages: BaseMessage[]
  ): Promise<{ text: string; message: AIMessage }> {
    const tldwMessages = this.convertToTldwMessages(messages)

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
      conversationId: this.conversationId,
      historyMessageLimit: this.historyMessageLimit,
      historyMessageOrder: this.historyMessageOrder,
      slashCommandInjectionMode: this.slashCommandInjectionMode,
      apiProvider: this.apiProvider,
      extraHeaders: this.extraHeaders,
      extraBody: this.extraBody,
      researchContext: this.researchContext,
      chatDebugMetadata: this.chatDebugMetadata
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
  async invoke(messages: BaseMessage[]): Promise<{ content: string }> {
    const { text } = await this.generateOnce(messages)
    return { content: text }
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
    const candidate = part as { type?: unknown; text?: unknown; image_url?: unknown }
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

  private normalizeUserContent(content: unknown): string | ChatCompletionContentPart[] {
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
      if (msg instanceof SystemMessage) {
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
      if (msg instanceof AIMessage) {
        return {
          role: "assistant",
          content: this.coerceTextContent(msg.content)
        }
      }
      if (msg instanceof HumanMessage) {
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
