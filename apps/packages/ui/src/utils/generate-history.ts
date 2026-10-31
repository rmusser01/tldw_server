import { isCustomModel } from "@/db/dexie/models"
import { decodeChatErrorPayload } from "@/utils/chat-error-message"
import { removeReasoning } from "@/libs/reasoning"
import { isImageGenerationMessageType } from "@/utils/image-generation-chat"
import {
  HumanMessage,
  AIMessage,
  ToolMessage,
  type BaseMessage,
  type MessageContent
} from "@/types/messages"

export const generateHistory = (
  messages: {
    id?: string
    role: string
    content: string
    image?: string
    images?: string[]
    tool_calls?: Record<string, unknown>[] | null
    tool_call_id?: string
    function_call?: Record<string, unknown>
    messageType?: string
  }[],
  model: string,
  options?: { versioned: boolean }
) => {
  const history: BaseMessage[] = []
  const isCustom = isCustomModel(model)
  for (const message of messages) {
    if (isImageGenerationMessageType(message.messageType)) {
      continue
    }
    if (
      options?.versioned &&
      !["system", "user", "assistant", "tool", "function"].includes(
        message.role
      )
    ) {
      throw new Error("unsupported_history_message_role")
    }
    if (
      options?.versioned &&
      message.role !== "user" &&
      message.role !== "system" &&
      (message.images?.some(Boolean) || message.image)
    ) {
      throw new Error("unsupported_history_message_images")
    }
    if (
      options?.versioned &&
      (message.role === "function" || message.function_call)
    ) {
      throw new Error("unsupported_history_function_message")
    }
    if (options?.versioned && message.role === "tool") {
      if (!message.tool_call_id) throw new Error("missing_history_tool_call_id")
      history.push(
        new ToolMessage({
          content: message.content,
          tool_call_id: message.tool_call_id
        })
      )
      continue
    }
    if (message.role === "user") {
      let content: MessageContent = isCustom
        ? message.content
        : [
            {
              type: "text",
              text: message.content
            }
          ]

      const images = message.images?.filter(Boolean).length
        ? message.images.filter(Boolean)
        : message.image
          ? [message.image]
          : []
      if (images.length > 0) {
        content = [
          ...images.map((url) => ({
            type: "image_url" as const,
            image_url: isCustom ? { url } : url
          })),
          {
            type: "text",
            text: message.content
          }
        ]
      }
      history.push(
        new HumanMessage({
          content: content
        })
      )
    } else if (message.role === "assistant") {
      if (decodeChatErrorPayload(message.content)) continue
      history.push(
        new AIMessage({
          ...(options?.versioned && message.tool_calls
            ? { additional_kwargs: { tool_calls: message.tool_calls } }
            : {}),
          content: isCustom
            ? removeReasoning(message.content)
            : [
                {
                  type: "text",
                  text: removeReasoning(message.content)
                }
              ]
        })
      )
    }
  }
  return history
}
