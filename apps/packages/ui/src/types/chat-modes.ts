import type { ChatHistory, MessageMetadataExtra } from "~/store/option"
import type { ChatDocuments } from "@/models/ChatTypes"
import type { DynamicUIRequest } from "@/types/dynamic-ui"
import type { ImageGenerationEventSyncPolicy } from "@/utils/image-generation-chat"
import type { ServicePromptRequestScope } from "@/services/tldw/domains/service-prompts"
import type { UploadedFile } from "@/db/dexie/types"

export interface SaveMessageBase {
  historyId: string | null
  setHistoryId: (id: string) => void
  selectedModel: string
  image: string
  userMessageType?: string
  assistantMessageType?: string
  clusterId?: string
  modelId: string
  userModelId?: string
  userMessageId?: string
  assistantMessageId: string
  userParentMessageId?: string | null
  assistantParentMessageId?: string | null
  documents?: ChatDocuments
  saveToDb?: boolean
  conversationId?: string
  imageEventSyncPolicy?: ImageGenerationEventSyncPolicy
  dynamicUIRequest?: DynamicUIRequest
  userMetadataExtra?: MessageMetadataExtra
  assistantMetadataExtra?: MessageMetadataExtra
  scopeSignal?: AbortSignal
  scopeInvalidatedSignal?: AbortSignal
  requestScope?: ServicePromptRequestScope
  deferHistoryMetadata?: boolean
}

export interface SaveMessageData extends SaveMessageBase {
  isRegenerate: boolean
  message: string
  fullText: string
  source: unknown[]
  assistantImages?: string[]
  generationInfo?: Record<string, unknown>
  reasoning_time_taken: number
  prompt_content?: string
  prompt_id?: string
  isContinue?: boolean
  sessionFilesToAdd?: UploadedFile[]
}

export interface SaveMessageErrorData extends SaveMessageBase {
  e: unknown
  botMessage: string
  history: ChatHistory
  setHistory: (history: ChatHistory) => void
  userMessage: string
  isRegenerating: boolean
  prompt_content?: string
  prompt_id?: string
  isContinue?: boolean
  shouldAbortForScopeChange?: () => boolean
}
