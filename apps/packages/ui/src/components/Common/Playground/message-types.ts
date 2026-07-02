import type { ChatDocuments } from "@/models/ChatTypes"
import type { ToolCall, ToolCallResult } from "@/types/tool-calls"
import type { MessageVariant } from "@/store/option"
import type { Character } from "@/types/character"
import type { MessageSteeringMode } from "@/types/message-steering"
import type { ChatScope } from "@/types/chat-scope"
import type { ImageGenerationRequestSnapshot } from "@/utils/image-generation-chat"

export interface MessageIdentityProps {
  conversationInstanceId: string
  serverChatId?: string | null
  serverMessageId?: string | null
  scope?: ChatScope
  messageId?: string
  historyId?: string
  temporaryChat?: boolean
}

export interface MessageContentProps {
  message: string
  message_type?: string
  images?: string[]
  documents?: ChatDocuments
  toolCalls?: ToolCall[]
  toolResults?: ToolCallResult[]
}

export interface MessageFlowProps {
  currentMessageIndex: number
  totalMessages: number
  isBot: boolean
  name: string
  role?: "user" | "assistant" | "system"
  isProcessing: boolean
  isStreaming: boolean
  isEmbedding?: boolean
  isSearchingInternet?: boolean
  actionInfo?: string | null
}

export interface MessageDisplayProps {
  hideCopy?: boolean
  botAvatar?: JSX.Element
  userAvatar?: JSX.Element
  hideEditAndRegenerate?: boolean
  hideContinue?: boolean
  isTTSEnabled?: boolean
  generationInfo?: any
  reasoningTimeTaken?: number
  openReasoning?: boolean
  modelImage?: string
  modelName?: string
  webSearch?: {}
  createdAt?: number | string
  feedbackQuery?: string | null
  searchQuery?: string
  searchMatch?: "active" | "match" | null
  pinned?: boolean
  suppressDeleteSuccessToast?: boolean
}

export interface MessageCharacterProps {
  characterIdentity?: Character | null
  characterIdentityEnabled?: boolean
  speakerCharacterId?: number | null
  speakerCharacterName?: string
  moodLabel?: string | null
  moodConfidence?: number | null
  moodTopic?: string | null
  visualActorKind?: "character" | "persona" | null
  visualActorId?: number | string | null
  visualPackId?: number | null
  visualPackVersionId?: number | null
  visualExpressionKey?: string | null
  visualAssetId?: number | null
  visualAssetUrl?: string | null
  visualFallbackReason?: string | null
  visualIsAnimated?: boolean | null
  visualPreviewUrl?: string | null
}

export interface MessageCompareProps {
  compareSelectable?: boolean
  compareSelected?: boolean
  onToggleCompareSelect?: () => void
  compareError?: boolean
  compareErrorModelLabel?: string
  onCompareRetry?: () => void
  compareChosen?: boolean
  variants?: MessageVariant[]
  activeVariantIndex?: number
  onSwipePrev?: (messageId: string) => void
  onSwipeNext?: (messageId: string) => void
}

export interface MessageImageActionProps {
  onRegenerateImage?: (payload: {
    messageId?: string
    imageIndex: number
    imageUrl: string
    request: ImageGenerationRequestSnapshot | null
  }) => void | Promise<void>
  onDeleteImage?: (payload: {
    messageId?: string
    imageIndex: number
    imageUrl: string
  }) => void
  onSelectImageVariant?: (payload: {
    messageId?: string
    variantIndex: number
  }) => void
  onKeepImageVariant?: (payload: {
    messageId?: string
    variantIndex: number
  }) => void
  onDeleteImageVariant?: (payload: {
    messageId?: string
    variantIndex: number
  }) => void
  onDeleteAllImageVariants?: (payload: {
    messageId?: string
  }) => void
}

export interface MessageActionCallbacks {
  onRegenerate: () => void
  onEditFormSubmit: (index: number, value: string, isUser: boolean, isSend?: boolean) => void
  onSourceClick?: (source: any) => void
  onContinue?: () => void
  onRunSteeredContinue?: (
    mode: Exclude<MessageSteeringMode, "none">
  ) => void
  onNewBranch?: (index: number) => void
  onStopStreaming?: () => void
  onDeleteMessage?: (index: number) => void
  onSaveToWorkspaceNotes?: (payload: {
    message: string
    isBot: boolean
    name: string
    messageId?: string
    createdAt?: number | string
  }) => void
  onTogglePinned?: (index: number) => void
}

export interface MessageSteeringProps {
  messageSteeringMode?: MessageSteeringMode
  onMessageSteeringModeChange?: (mode: MessageSteeringMode) => void
  messageSteeringForceNarrate?: boolean
  onMessageSteeringForceNarrateChange?: (enabled: boolean) => void
  onClearMessageSteering?: () => void
}

export interface DiscoSkillProps {
  discoSkillComment?: import("@/types/disco-skills").DiscoSkillComment | null
}

/**
 * Full Props type for PlaygroundMessage — intersection of all prop groups.
 */
export type PlaygroundMessageProps = MessageIdentityProps &
  MessageContentProps &
  MessageFlowProps &
  MessageDisplayProps &
  MessageCharacterProps &
  MessageCompareProps &
  MessageImageActionProps &
  MessageActionCallbacks &
  MessageSteeringProps &
  DiscoSkillProps
