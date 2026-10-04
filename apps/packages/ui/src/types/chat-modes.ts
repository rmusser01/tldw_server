import type { HistoryOwnerV1 } from "@/services/chat-history-selection"
import type {
  HistoryAdmissionV1,
  HistorySelectionCaptureV1,
  HistoryViewSelectionV1,
  HistorySelectionV1
} from "@/types/history-selection"
import type {
  HistoryTurnOutcome,
  Message as StoredMessage
} from "@/db/dexie/types"

/** One operation owns its adapter and immutable pending intent across display navigation. */
export interface HistorySendTurn {
  owner: HistoryOwnerV1
  capture: HistorySelectionCaptureV1
  currentView: () => HistoryViewSelectionV1 | null
  validateLease: () => boolean
  canUpdateView: () => boolean
  admission?: HistoryAdmissionV1
  selection?: HistorySelectionV1
  input?: StoredMessage
  resultId?: string
  assistantId?: string
  createdAt?: number
  dispatched?: boolean
  /** The model that writes the reply, recorded with a retained reply. */
  replyModel?: { name: string; id?: string }
  /**
   * How the reply ended when it should stay in the transcript instead of in
   * review (CS-04): see HistoryTurnRecovery.outcome.
   */
  outcome?: HistoryTurnOutcome
  recover: (
    data: {
      content: string
      assistantId: string
      createdAt: number
      outcome?: HistoryTurnOutcome
      interruptionReason?: string
    },
    error: unknown
  ) => Promise<void>
  /** Record the reply so far, throttled, so a reload can keep it. */
  checkpoint?: (content: string) => void
  cancelPreparation?: () => Promise<void>
  beforeDispatch?: () => Promise<void>
  afterAdmission?: () => Promise<void>
  complete?: () => Promise<void>
  /** Resolves false when the view had moved on and could not follow. */
  followResult: (id: string) => Promise<boolean | void>
  /** After a settled reply: clear its record, or keep it until a view follows. */
  finish?: (followed: boolean) => Promise<void>
  /** Put a reply that ended early back into the transcript. */
  keep?: () => Promise<void>
  /** The turn has ended; its record may be kept by any view. */
  release?: () => void
}

import type { ChatHistory, MessageMetadataExtra } from "~/store/option"
import type { ChatDocuments } from "@/models/ChatTypes"
import type { DynamicUIRequest } from "@/types/dynamic-ui"
import type { ImageGenerationEventSyncPolicy } from "@/utils/image-generation-chat"
import type { ServicePromptRequestScope } from "@/services/tldw/domains/service-prompts"
import type { UploadedFile } from "@/db/dexie/types"

export interface SaveMessageBase {
  historyTurn?: HistorySendTurn
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
  userServerMessageId?: string
  retryFailedTurn?: boolean
  assistantServerMessageId?: string
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
