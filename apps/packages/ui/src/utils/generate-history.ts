import { isCustomModel } from "@/db/dexie/models"
import { decodeChatErrorPayload } from "@/utils/chat-error-message"
import { removeReasoning } from "@/libs/reasoning"
import { isImageGenerationMessageType } from "@/utils/image-generation-chat"
import {
  HumanMessage,
  AIMessage,
  type MessageContent
} from "@/types/messages"

export const generateHistory = (
  messages: {
    role: "user" | "assistant" | "system"
    content: string
    image?: string
    messageType?: string
  }[],
  model: string
) => {
  let history = []
  const isCustom = isCustomModel(model)
  for (const message of messages) {
    if (isImageGenerationMessageType(message.messageType)) {
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

      if (message.image) {
        content = [
          {
            type: "image_url",
            image_url: !isCustom
              ? message.image
              : {
                  url: message.image
                }
          },
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
