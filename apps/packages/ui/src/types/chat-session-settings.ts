import type { ChatResearchContext } from "@/services/tldw/TldwApiClient"

export const CHAT_SETTINGS_SCHEMA_VERSION = 2

export type CharacterMemoryEntry = {
  note: string
  updatedAt?: string
}

export type ChatGenerationOverride = {
  enabled?: boolean
  temperature?: number | null
  top_p?: number | null
  repetition_penalty?: number | null
  stop?: string[]
  updatedAt?: string
}

export type ChatSummary = {
  enabled?: boolean
  content?: string
  sourceRange?: { fromMessageId?: string; toMessageId?: string }
  updatedAt?: string
}

export type ConversationContextSettings = {
  world_book_ids?: number[]
  chat_dictionary_ids?: number[]
}

export type DeepResearchAttachment = ChatResearchContext & {
  attached_at: string
  updatedAt: string
}

export type ChatAssistantOverlay = {
  kind: "character" | "persona"
  id: string
  name: string
  avatar_url?: string | null
  system_prompt_snapshot?: string | null
  updatedAt: string
}

export type AuthorNotePosition =
  | "before_system"
  | `depth:${number}`
  | number
  | {
      mode?: string
      depth?: number
    }

export type ChatSettingsRecord = {
  schemaVersion: number
  updatedAt: string
  autoSummaryEnabled?: boolean
  autoSummaryThresholdMessages?: number | null
  autoSummaryWindowMessages?: number | null
  pinnedMessageIds?: string[]
  greetingSelectionId?: string | null
  greetingsVersion?: number | null
  greetingsChecksum?: string | null
  useCharacterDefault?: boolean
  greetingEnabled?: boolean
  greetingScope?: "chat" | "character"
  presetScope?: "chat" | "character"
  memoryScope?: "shared" | "character" | "both"
  directedCharacterId?: number | null
  chatPresetOverrideId?: string | null
  authorNote?: string
  authorNotePosition?: AuthorNotePosition | null
  characterMemoryById?: Record<string, CharacterMemoryEntry>
  chatGenerationOverride?: ChatGenerationOverride | null
  generationOverrides?: ChatGenerationOverride | null
  turnTakingMode?: "single" | "round_robin"
  participantCharacterIds?: string[]
  conversationContext?: ConversationContextSettings
  chat_dictionary_ids?: number[]
  summary?: ChatSummary | null
  imageEventSyncMode?: "off" | "on"
  assistantOverlay?: ChatAssistantOverlay | null
  deepResearchAttachment?: DeepResearchAttachment | null
  deepResearchPinnedAttachment?: DeepResearchAttachment | null
  deepResearchAttachmentHistory?: DeepResearchAttachment[]
}
