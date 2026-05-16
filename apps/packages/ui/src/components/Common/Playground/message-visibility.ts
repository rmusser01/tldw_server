import { IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE } from "@/utils/image-generation-chat"

export type AssistantResponseVisibilityInput = {
  message?: unknown
  messageType?: unknown
  message_type?: unknown
  images?: unknown
  toolCalls?: unknown
}

export function hasVisibleAssistantResponse(
  input: AssistantResponseVisibilityInput
): boolean {
  const messageText = typeof input.message === "string" ? input.message.trim() : ""
  if (messageText.length > 0) return true

  const messageType =
    typeof input.messageType === "string"
      ? input.messageType
      : typeof input.message_type === "string"
        ? input.message_type
        : undefined
  if (messageType === IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE) return true

  const images = Array.isArray(input.images) ? input.images : []
  if (images.some((image) => typeof image === "string" && image.length > 0)) {
    return true
  }

  const toolCalls = Array.isArray(input.toolCalls) ? input.toolCalls : []
  return toolCalls.length > 0
}
