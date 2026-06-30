export type ConversationContextSource =
  | "request"
  | "explicit_chat"
  | "workspace"
  | "character_start"
  | "character_inherited"
  | "global"

export type ConversationContextAssetKind =
  | "character"
  | "worldbook"
  | "dictionary"
  | "workspace"
  | "provider"

export type ConversationContextPieceStatus =
  | "configured"
  | "active"
  | "matched"
  | "skipped"
  | "blocked"
  | "missing"

export interface ConversationContextSelection {
  chatId?: string
  characterId?: number | string | null
  worldBookIds: number[]
  dictionaryIds: number[]
  workspaceId?: string | null
  providerId?: string | null
  modelId?: string | null
}

export interface ConversationContextPiece {
  kind: ConversationContextAssetKind
  id?: number | string | null
  name?: string | null
  source: ConversationContextSource
  status: ConversationContextPieceStatus
  content?: string
  diagnostics?: unknown
  warnings?: string[]
}

export interface ConversationContextPreviewSection {
  name: string
  content: string
  source: ConversationContextSource
}

export interface ConversationContextProviderMessage {
  role: string
  content: string
}

export interface ConversationContextComposition {
  selection: ConversationContextSelection
  inputText: string
  transformedInputText: string
  pieces: ConversationContextPiece[]
  previewSections: ConversationContextPreviewSection[]
  providerMessages: ConversationContextProviderMessage[]
  readiness: "ready" | "partial" | "blocked"
  warnings: string[]
}

export interface ConversationContextSeed {
  chatId?: string
  characterId?: number | string | null
  worldBookIds?: Array<number | string | null | undefined>
  dictionaryIds?: Array<number | string | null | undefined>
  workspaceId?: string | null
  providerId?: string | null
  modelId?: string | null
}

export type ConversationContextSettingsPatch = {
  conversationContext: {
    world_book_ids: number[]
    chat_dictionary_ids: number[]
  }
  chat_dictionary_ids: number[]
}
