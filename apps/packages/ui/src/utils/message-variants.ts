import type { Message } from "@/store/option"

export type MessageVariant = {
  id?: string
  message: string
  sources?: any[]
  images?: string[]
  generationInfo?: any
  reasoning_time_taken?: number
  createdAt?: number
  serverMessageId?: string
  serverMessageVersion?: number
  metadataExtra?: Message["metadataExtra"]
}

const normalizeId = (value: unknown): string | undefined => {
  if (typeof value === "string" && value.trim().length > 0) return value
  return undefined
}

const isDuplicateVariant = (a: MessageVariant, b: MessageVariant): boolean => {
  const aId = normalizeId(a.id)
  const bId = normalizeId(b.id)
  if (aId && bId && aId === bId) return true
  const aServer = normalizeId(a.serverMessageId)
  const bServer = normalizeId(b.serverMessageId)
  if (aServer && bServer && aServer === bServer) return true
  if (a.message && b.message && a.message === b.message) return true
  return false
}

export const buildMessageVariant = (message: Message): MessageVariant => ({
  id: message.id,
  message: message.message,
  sources: message.sources ?? [],
  images: message.images ?? [],
  generationInfo: message.generationInfo,
  reasoning_time_taken: message.reasoning_time_taken,
  createdAt: message.createdAt,
  serverMessageId: message.serverMessageId,
  serverMessageVersion: message.serverMessageVersion,
  metadataExtra: message.metadataExtra
})

export const mergeMessageVariants = (
  variants: MessageVariant[] | undefined,
  variant: MessageVariant
): MessageVariant[] => {
  const base = Array.isArray(variants) ? [...variants] : []
  if (!base.some((item) => isDuplicateVariant(item, variant))) {
    base.push(variant)
  }
  return base
}

export const normalizeMessageVariants = (message: Message): MessageVariant[] => {
  return mergeMessageVariants(message.variants, buildMessageVariant(message))
}

export const applyVariantToMessage = (
  message: Message,
  variant: MessageVariant,
  activeVariantIndex: number
): Message => {
  return {
    ...message,
    activeVariantIndex,
    message: variant.message,
    sources: variant.sources ?? message.sources,
    images: variant.images ?? message.images,
    generationInfo: variant.generationInfo ?? message.generationInfo,
    reasoning_time_taken:
      variant.reasoning_time_taken ?? message.reasoning_time_taken,
    createdAt: variant.createdAt ?? message.createdAt,
    serverMessageId: variant.serverMessageId ?? message.serverMessageId,
    serverMessageVersion:
      variant.serverMessageVersion ?? message.serverMessageVersion,
    metadataExtra: variant.metadataExtra,
    id: variant.id ?? message.id
  }
}

export const updateActiveVariant = (
  message: Message,
  updates: Partial<Message>
): Message => {
  if (!message.variants || typeof message.activeVariantIndex !== "number") {
    return { ...message, ...updates }
  }

  const idx = Math.max(
    0,
    Math.min(message.activeVariantIndex, message.variants.length - 1)
  )
  const nextVariant: MessageVariant = {
    id: (updates.id as string | undefined) ?? message.id,
    message: (updates.message as string | undefined) ?? message.message,
    sources: (updates.sources as any[] | undefined) ?? message.sources ?? [],
    images: (updates.images as string[] | undefined) ?? message.images ?? [],
    generationInfo: updates.generationInfo ?? message.generationInfo,
    reasoning_time_taken:
      updates.reasoning_time_taken ?? message.reasoning_time_taken,
    createdAt: updates.createdAt ?? message.createdAt,
    serverMessageId: updates.serverMessageId ?? message.serverMessageId,
    serverMessageVersion:
      updates.serverMessageVersion ?? message.serverMessageVersion,
    metadataExtra: Object.prototype.hasOwnProperty.call(
      updates,
      "metadataExtra"
    )
      ? updates.metadataExtra
      : message.metadataExtra
  }
  const variants = [...message.variants]
  variants[idx] = { ...variants[idx], ...nextVariant }

  return {
    ...message,
    ...updates,
    variants
  }
}

export const getLastUserMessageId = (messages: Message[]): string | null => {
  for (let i = messages.length - 1; i >= 0; i--) {
    const candidate = messages[i]
    if (!candidate?.isBot && candidate?.id) {
      return candidate.id
    }
  }
  return null
}

export const getLastUserServerMessageId = (
  messages: Message[]
): string | null => {
  for (let i = messages.length - 1; i >= 0; i--) {
    const candidate = messages[i]
    if (!candidate?.isBot && candidate?.serverMessageId) {
      return candidate.serverMessageId
    }
  }
  return null
}
