import type { HistoryOwnerV1 } from "@/services/chat-history-selection"
import type {
  HistoryAdmissionReferenceV1,
  HistoryEditTargetV1,
  HistorySelectionCaptureV1,
  HistoryViewSelectionV1,
  HistorySelectionV1
} from "@/types/history-selection"
import type {
  HistoryTurnFailure,
  HistoryTurnOutcome,
  Message as StoredMessage
} from "@/db/dexie/types"

/** Private explicit branch intent; its editor lease never enters a wire payload. */
export type NativeHistoryEditResend = {
  owner_key: string
  conversation_id: string
  message_id: string
  message_revision: string
  isCurrent: () => boolean
  isOriginCurrent: () => boolean
}

/** A private saved-reply action also captures its canonical direct user. */
export type NativeHistoryRegenerate = NativeHistoryEditResend & {
  input_message_id: string
  input_message_revision: string
  /**
   * Native Continue (CM-01) reuses the same saved admission as Regenerate, but
   * the server keeps the reply in context and saves it plus its continuation.
   */
  action?: "continue"
}

/** One operation owns its adapter and immutable pending intent across display navigation. */
export interface HistorySendTurn {
  owner: HistoryOwnerV1
  capture: HistorySelectionCaptureV1
  currentView: () => HistoryViewSelectionV1 | null
  validateLease: () => boolean
  canUpdateView: () => boolean
  /** Original command or its accepted bind, required until user admission. */
  preparationIsCurrent?: () => boolean;
  persistence?: 'server';
  clientManagedContext?: boolean;
  admission?: HistoryAdmissionReferenceV1;
  /**
   * Retry of a reply that ended early (CS-04): the question was admitted when
   * it was first sent, so the turn settles a new reply against this admission
   * instead of admitting the question a second time.
   */
  retryAdmission?: HistoryAdmissionReferenceV1
  historyBranch?: boolean
  historyEdit?: HistoryEditTargetV1
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
      /** The server answered with an error before any reply arrived (CC-02). */
      failure?: HistoryTurnFailure
      /** Why a turn left for review got no reply, when known (CM-06). */
      cause?: HistoryTurnFailure
      /**
       * The user stopped the turn before its owner confirmed the question
       * (CM-02): there is nothing to recover, and the text goes back to the
       * composer.
       */
      cancelled?: boolean
    },
    error: unknown
  ) => Promise<void>
  /** Record the reply so far, throttled, so a reload can keep it. */
  checkpoint?: (content: string) => void
  cancelPreparation?: () => Promise<void>
  beforeDispatch?: () => Promise<void>
  afterAdmission?: () => Promise<void>
  complete?: (content?: string) => Promise<void>
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
