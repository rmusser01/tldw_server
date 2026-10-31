import type { Flashcard } from "@/services/flashcards"
import { buildChatThreadPath } from "@/routes/route-paths"

type SourceType = "media" | "message" | "note"

export interface FlashcardSourceMeta {
  type: SourceType
  label: string
  href: string | null
  unavailable: boolean
}

const asCleanString = (value: string | null | undefined): string | null => {
  if (typeof value !== "string") return null
  const trimmed = value.trim()
  return trimmed.length > 0 ? trimmed : null
}

export const getFlashcardSourceMeta = (
  card: Pick<Flashcard, "source_ref_type" | "source_ref_id" | "conversation_id" | "message_id">
): FlashcardSourceMeta | null => {
  const type = card.source_ref_type
  if (!type || type === "manual") return null

  if (type === "media") {
    const sourceId = asCleanString(card.source_ref_id)
    if (!sourceId || !/^[1-9][0-9]*$/.test(sourceId)) {
      return {
        type,
        label: "Media source unavailable",
        href: null,
        unavailable: true
      }
    }
    return {
      type,
      label: `Media #${sourceId}`,
      href: `/media?id=${encodeURIComponent(sourceId)}`,
      unavailable: false
    }
  }

  if (type === "note") {
    const sourceId = asCleanString(card.source_ref_id)
    if (!sourceId) {
      return {
        type,
        label: "Note source unavailable",
        href: null,
        unavailable: true
      }
    }
    return {
      type,
      label: `Note #${sourceId}`,
      href: `/notes?source_ref_id=${encodeURIComponent(sourceId)}`,
      unavailable: false
    }
  }

  const sourceId = asCleanString(card.source_ref_id) ?? asCleanString(card.message_id)
  const conversationId = asCleanString(card.conversation_id)
  if (!sourceId || !conversationId) {
    return {
      type: "message",
      label: "Message source unavailable",
      href: null,
      unavailable: true
    }
  }
  return {
    type: "message",
    label: `Conversation for message #${sourceId}`,
    href: buildChatThreadPath({ serverChatId: conversationId }),
    unavailable: false
  }
}
