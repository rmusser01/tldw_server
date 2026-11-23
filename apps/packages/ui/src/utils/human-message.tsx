import { isCustomModel } from "@/db/dexie/models"
import { processImageForOCR } from "@/utils/ocr"
import {
  HumanMessage,
  type MessageContent,
  type MessageContentPart
} from "@/types/messages"

type HumanMessageType = {
  content: MessageContent
  model: string
  useOCR: boolean
}

const getTextPart = (part?: MessageContentPart): string | null => {
  if (!part || part.type !== "text") return null
  return typeof part.text === "string" ? part.text : null
}

const getImageUrl = (part?: MessageContentPart): string | null => {
  if (!part || part.type !== "image_url") return null
  if (typeof part.image_url === "string") return part.image_url
  return part.image_url?.url ?? null
}

const buildImageContent = (text: string, imageUrl: string): MessageContent => [
  { type: "text", text },
  { type: "image_url", image_url: { url: imageUrl } }
]

export const humanMessageFormatter = async ({
  content,
  model,
  useOCR = false
}: HumanMessageType) => {
  if (Array.isArray(content) && content.length === 1) {
    const part = content[0]
    const text = getTextPart(part)
    if (text !== null && Object.keys(part).every(key => key === "type" || key === "text")) {
      return new HumanMessage({ content: text })
    }
  }
  const isCustom = isCustomModel(model)

  if (isCustom) {
    if (Array.isArray(content)) {
      const text = getTextPart(content[0])
      const imageUrl = getImageUrl(content[1])

      if (useOCR && text && imageUrl) {
        const ocrText = await processImageForOCR(imageUrl)
        const ocrPrompt = `${text}

[IMAGE OCR TEXT]
${ocrText}`
        return new HumanMessage({
          content: ocrPrompt
        })
      }

      if (text && imageUrl) {
        return new HumanMessage({
          content: buildImageContent(text, imageUrl)
        })
      }

      if (text) {
        return new HumanMessage({
          content: text
        })
      }
    }
  }

  if (useOCR) {
    if (Array.isArray(content)) {
      const text = getTextPart(content[0])
      const imageUrl = getImageUrl(content[1])
      if (text && imageUrl) {
        const ocrText = await processImageForOCR(imageUrl)
        const ocrPrompt = `${text}

[IMAGE OCR TEXT]
${ocrText}`
        return new HumanMessage({
          content: ocrPrompt
        })
      }
    }
  }

  return new HumanMessage({
    content
  })
}
