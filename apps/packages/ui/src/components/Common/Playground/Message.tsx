import React, { useEffect, useState, useRef } from "react"
import { Tag, Image, Tooltip, Collapse, Avatar, Modal, message } from "antd"
import { LoadingStatus } from "./ActionInfo"
import {
  AlertTriangle,
  CheckCircle2,
  RotateCcw,
  Smile,
  StopCircle as StopCircleIcon,
  Trash2
} from "lucide-react"
import { EditMessageForm } from "./EditMessageForm"
import { useTranslation } from "react-i18next"
import { useTTS, type TtsClipMeta } from "@/hooks/useTTS"
import { useChatMoodBadgePreference } from "@/hooks/useChatMoodBadgePreference"
import { tagColors } from "@/utils/color"
import { removeModelSuffix } from "@/db/dexie/models"
import { parseReasoning } from "@/libs/reasoning"
import {
  decodeChatErrorPayload,
  type ChatErrorPayload
} from "@/utils/chat-error-message"
import { useStorage } from "@plasmohq/storage/hook"
import { PlaygroundUserMessageBubble } from "./PlaygroundUserMessage"
import { copyToClipboard } from "@/utils/clipboard"
import { ChatDocuments } from "@/models/ChatTypes"
import { buildChatTextClass } from "@/utils/chat-style"
import { highlightText } from "@/utils/text-highlight"
import { FeedbackModal } from "@/components/Sidepanel/Chat/FeedbackModal"
import { SourceFeedback } from "@/components/Sidepanel/Chat/SourceFeedback"
import { ToolCallBlock } from "@/components/Sidepanel/Chat/ToolCallBlock"
import type { ToolCall, ToolCallResult } from "@/types/tool-calls"
import { MessageActionsBar } from "./MessageActionsBar"
import { ReasoningBlock } from "./ReasoningBlock"
import { useFeedback } from "@/hooks/useFeedback"
import { useImplicitFeedback } from "@/hooks/useImplicitFeedback"
import { useServerCapabilities } from "@/hooks/useServerCapabilities"
import { useTldwAudioStatus } from "@/hooks/useTldwAudioStatus"
import { getSourceFeedbackKey } from "@/utils/feedback"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import type { ChatScope } from "@/types/chat-scope"
import { useUiModeStore } from "@/store/ui-mode"
import { useStoreMessageOption } from "@/store/option"
import type { MessageVariant } from "@/store/option"
import { useStoreChatModelSettings } from "@/store/model"
import { EDIT_MESSAGE_EVENT } from "@/utils/timeline-actions"
import type { Character } from "@/types/character"
import { useDiscoSkills } from "@/hooks/useDiscoSkills"
import { DiscoSkillAnnotation } from "./DiscoSkillAnnotation"
import type { DiscoSkillComment } from "@/types/disco-skills"
import {
  attemptSkillTrigger,
  buildSkillPrompt,
  createSkillComment
} from "@/utils/disco-skill-check"
import {
  detectCharacterMood,
  normalizeCharacterMoodLabel,
  resolveCharacterBaseAvatarUrl,
  resolveCharacterMoodImageUrl
} from "@/utils/character-mood"
import { useStoreMessage } from "@/store"
import { updateMessageDiscoSkillComment } from "@/db/dexie/helpers"
import type { MessageSteeringMode } from "@/types/message-steering"
import {
  resolveAvatarColumnAlignment,
  resolveMessageRenderSide
} from "./message-layout"
import { formatCost } from "@/utils/model-pricing"
import {
  resolveMessageCostUsd,
  resolveMessageUsage
} from "./message-usage"
import { resolvePlaygroundMessageShortcutAction } from "./playground-message-shortcuts"
import {
  buildQuickMessageActionPrompt,
  type QuickMessageAction
} from "./quick-message-actions"
import { resolveFallbackAudit } from "./routing-fallback-audit"
import {
  IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE,
  resolveImageGenerationMetadata,
  type ImageGenerationRequestSnapshot
} from "@/utils/image-generation-chat"
import { isDeepResearchCompletionMetadata } from "@/components/Option/Playground/research-chat-context"
import {
  DEFAULT_TLDW_TTS_MODEL,
  DEFAULT_TTS_PROVIDER
} from "@/services/tts"

const Markdown = React.lazy(() => import("../../Common/Markdown"))

const ErrorBubble: React.FC<{
  payload: ChatErrorPayload
  toggleLabels: { show: string; hide: string }
  recoveryActions?: Array<{
    id: string
    label: string
    onClick: () => void
  }>
}> = ({ payload, toggleLabels, recoveryActions = [] }) => {
  const [showDetails, setShowDetails] = React.useState(false)

  return (
    <div
      role="alert"
      aria-live="assertive"
      className="rounded-md border border-danger/30 bg-danger/10 p-3 text-sm text-danger">
      <p className="font-semibold">{payload.summary}</p>
      {payload.hint && (
        <p className="mt-1 text-xs text-danger">
          {payload.hint}
        </p>
      )}
      {payload.detail && (
        <button
          type="button"
          onClick={() => setShowDetails((prev) => !prev)}
          title={showDetails ? toggleLabels.hide : toggleLabels.show}
          className="mt-2 text-xs font-medium text-danger underline hover:text-danger">
          {showDetails ? toggleLabels.hide : toggleLabels.show}
        </button>
      )}
      {showDetails && payload.detail && (
        <pre className="mt-2 max-h-40 overflow-auto whitespace-pre-wrap rounded bg-danger/10 p-2 text-xs text-danger">
          {payload.detail}
        </pre>
      )}
      {recoveryActions.length > 0 && (
        <div className="mt-2 flex flex-wrap items-center gap-1.5">
          <span className="sr-only">
            Recommended next actions:{" "}
            {recoveryActions.map((action) => action.label).join(", ")}
          </span>
          {recoveryActions.map((action) => (
            <button
              key={action.id}
              type="button"
              onClick={action.onClick}
              className="rounded border border-danger/40 bg-surface px-2 py-1 text-[11px] font-medium text-danger transition hover:bg-danger/10"
            >
              {action.label}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

const EmptyAssistantResponseNotice: React.FC<{
  summary: string
  detail: string
  recoveryActions: Array<{
    id: string
    label: string
    onClick: () => void
  }>
}> = ({ summary, detail, recoveryActions }) => (
  <div
    role="status"
    aria-live="polite"
    aria-label="Empty assistant response"
    className="rounded-md border border-warn/30 bg-warn/10 p-3 text-sm text-warn">
    <p className="font-semibold">{summary}</p>
    <p className="mt-1 text-xs text-warn">{detail}</p>
    {recoveryActions.length > 0 && (
      <div className="mt-2 flex flex-wrap items-center gap-1.5">
        <span className="sr-only">
          Recommended next actions:{" "}
          {recoveryActions.map((action) => action.label).join(", ")}
        </span>
        {recoveryActions.map((action) => (
          <button
            key={action.id}
            type="button"
            onClick={action.onClick}
            className="rounded border border-warn/40 bg-surface px-2 py-1 text-[11px] font-medium text-warn transition hover:bg-warn/10"
          >
            {action.label}
          </button>
        ))}
      </div>
    )}
  </div>
)

type Props = {
  message: string
  message_type?: string
  hideCopy?: boolean
  botAvatar?: JSX.Element
  userAvatar?: JSX.Element
  isBot: boolean
  name: string
  role?: "user" | "assistant" | "system"
  images?: string[]
  currentMessageIndex: number
  totalMessages: number
  onRegenerate: () => void
  onEditFormSubmit: (value: string, isSend: boolean) => void
  isProcessing: boolean
  webSearch?: {}
  isSearchingInternet?: boolean
  sources?: any[]
  hideEditAndRegenerate?: boolean
  hideContinue?: boolean
  onSourceClick?: (source: any) => void
  isTTSEnabled?: boolean
  generationInfo?: any
  isStreaming: boolean
  reasoningTimeTaken?: number
  openReasoning?: boolean
  modelImage?: string
  modelName?: string
  onContinue?: () => void
  onRunSteeredContinue?: (
    mode: Exclude<MessageSteeringMode, "none">
  ) => void
  documents?: ChatDocuments
  actionInfo?: string | null
  onNewBranch?: () => void
  temporaryChat?: boolean
  onStopStreaming?: () => void
  serverChatId?: string | null
  serverMessageId?: string | null
  messageId?: string
  feedbackQuery?: string | null
  searchQuery?: string
  searchMatch?: "active" | "match" | null
  isEmbedding?: boolean
  createdAt?: number | string
  variants?: MessageVariant[]
  activeVariantIndex?: number
  onSwipePrev?: () => void
  onSwipeNext?: () => void
  // Compare/multi-model metadata (optional)
  compareSelectable?: boolean
  compareSelected?: boolean
  onToggleCompareSelect?: () => void
  compareError?: boolean
  compareErrorModelLabel?: string
  onCompareRetry?: () => void
  compareChosen?: boolean
  // Tool/function calls (optional)
  toolCalls?: ToolCall[]
  toolResults?: ToolCallResult[]
  discoSkillComment?: DiscoSkillComment | null
  historyId?: string
  conversationInstanceId: string
  onDeleteMessage?: () => void
  suppressDeleteSuccessToast?: boolean
  onSaveToWorkspaceNotes?: (payload: {
    message: string
    isBot: boolean
    name: string
    messageId?: string
    createdAt?: number | string
  }) => void
  onTogglePinned?: () => void
  pinned?: boolean
  characterIdentity?: Character | null
  characterIdentityEnabled?: boolean
  speakerCharacterId?: number | null
  speakerCharacterName?: string
  moodLabel?: string | null
  moodConfidence?: number | null
  moodTopic?: string | null
  messageSteeringMode?: MessageSteeringMode
  onMessageSteeringModeChange?: (mode: MessageSteeringMode) => void
  messageSteeringForceNarrate?: boolean
  onMessageSteeringForceNarrateChange?: (enabled: boolean) => void
  onClearMessageSteering?: () => void
  metadataExtra?: Record<string, unknown>
  researchActions?: MessageResearchActions
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
  scope?: ChatScope
}

export type MessageResearchActions = {
  reasonLabel?: string
  primaryLink?: {
    href: string
    label: string
  }
  onUseInChat?: () => void
  onFollowUp?: () => void
}

export const PlaygroundMessage = (props: Props) => {
  const articleRef = useRef<HTMLElement | null>(null)
  const [isBtnPressed, setIsBtnPressed] = React.useState(false)
  const [editMode, setEditMode] = React.useState(false)
  const [checkWideMode] = useStorage("checkWideMode", false)
  const [isUserChatBubble] = useStorage("userChatBubble", true)
  const [autoCopyResponseToClipboard] = useStorage(
    "autoCopyResponseToClipboard",
    false
  )
  const [autoPlayTTS] = useStorage("isTTSAutoPlayEnabled", false)
  const [copyAsFormattedText] = useStorage("copyAsFormattedText", false)
  const [userTextColor] = useStorage("chatUserTextColor", "default")
  const [assistantTextColor] = useStorage("chatAssistantTextColor", "default")
  const [userTextFont] = useStorage("chatUserTextFont", "default")
  const [assistantTextFont] = useStorage("chatAssistantTextFont", "default")
  const [userTextSize] = useStorage("chatUserTextSize", "md")
  const [assistantTextSize] = useStorage("chatAssistantTextSize", "md")
  const [userDisplayName] = useStorage("chatUserDisplayName", "")
  const [showCharacterPortraits] = useStorage("chatShowCharacterPortraits", true)
  const [showMoodBadge] = useChatMoodBadgePreference()
  const moodConfidenceDefault =
    Boolean(props.characterIdentityEnabled) && Boolean(props.characterIdentity?.id)
  const [showMoodConfidence] = useStorage(
    "chatShowMoodConfidence",
    moodConfidenceDefault
  )
  const [userPersonaImage] = useStorage("chatUserPersonaImage", "")
  const [ttsProvider] = useStorage("ttsProvider", DEFAULT_TTS_PROVIDER)
  const [tldwTtsModel] = useStorage("tldwTtsModel", DEFAULT_TLDW_TTS_MODEL)
  const { t } = useTranslation(["common", "playground"])
  const { capabilities } = useServerCapabilities()
  const uiMode = useUiModeStore((state) => state.mode)
  const isProMode = uiMode === "pro"
  const setReplyTarget = useStoreMessageOption((state) => state.setReplyTarget)
  const ragPinnedResults = useStoreMessageOption((state) => state.ragPinnedResults)
  const apiProviderOverride = useStoreChatModelSettings(
    (state) => state.apiProvider
  )
  const updateChatModelSetting = useStoreChatModelSettings(
    (state) => state.updateSetting
  )
  const { cancel, isSpeaking, speak } = useTTS()
  const { healthState: audioHealthState, voicesAvailable } =
    useTldwAudioStatus({
      requireVoices: ttsProvider === "tldw",
      tldwTtsModel
    })
  const [isFeedbackOpen, setIsFeedbackOpen] = React.useState(false)
  const [isAvatarPreviewOpen, setIsAvatarPreviewOpen] = React.useState(false)
  const [compareVariantIndex, setCompareVariantIndex] = React.useState<
    number | null
  >(null)
  const [savingKnowledge, setSavingKnowledge] = React.useState<
    "note" | "flashcard" | null
  >(null)
  const responseDwellSentKeyRef = useRef<string | null>(null)

  // Disco Skills state
  const {
    enabled: discoSkillsEnabled,
    stats: discoSkillsStats,
    triggerProbabilityBase: discoTriggerProbability,
    persistComments: discoPersistComments
  } = useDiscoSkills()
  const setMessages = useStoreMessageOption((state) => state.setMessages)
  const persistedSkillComment = discoPersistComments
    ? props.discoSkillComment ?? null
    : null
  const [skillComment, setSkillComment] = useState<DiscoSkillComment | null>(
    persistedSkillComment
  )
  const [isGeneratingSkillComment, setIsGeneratingSkillComment] = useState(false)
  const skillCommentGeneratedRef = useRef(false)
  const previousMessageRef = useRef<string | null>(null)
  const previousMessageIdRef = useRef<string | null>(null)
  const selectedModel = useStoreMessage((state) => state.selectedModel)

  const isLastMessage: boolean =
    props.currentMessageIndex === props.totalMessages - 1

  // Track streaming completion for aria-live announcement
  const wasStreamingRef = useRef(false)
  const [streamingComplete, setStreamingComplete] = useState(false)
  useEffect(() => {
    if (!props.isBot || !isLastMessage) {
      wasStreamingRef.current = false
      setStreamingComplete(false)
      return
    }

    if (wasStreamingRef.current && !props.isStreaming && !props.isProcessing) {
      setStreamingComplete(true)
      const timer = setTimeout(() => setStreamingComplete(false), 2000)
      return () => clearTimeout(timer)
    }
    wasStreamingRef.current = props.isStreaming || props.isProcessing
  }, [props.isBot, isLastMessage, props.isStreaming, props.isProcessing])

  const errorPayload = decodeChatErrorPayload(props.message)
  const errorFriendlyText = React.useMemo(() => {
    if (!errorPayload) return null
    return [errorPayload.summary, errorPayload.hint, errorPayload.detail]
    .filter(Boolean)
    .join("\n")
  }, [errorPayload])
  const resolvedResponseModel = React.useMemo(() => {
    const info = props.generationInfo as Record<string, unknown> | undefined
    if (!info) return null
    const raw =
      info.model ??
      info.model_name ??
      info.resolved_model ??
      info.resolvedModel ??
      (info.routing as Record<string, unknown> | undefined)?.resolved_model ??
      (info.routing as Record<string, unknown> | undefined)?.model
    if (typeof raw !== "string" || raw.trim().length === 0) return null
    return removeModelSuffix(raw.trim().replace(/accounts\/[^/]+\/models\//g, ""))
  }, [props.generationInfo])
  const messageUsage = React.useMemo(
    () => resolveMessageUsage(props.generationInfo),
    [props.generationInfo]
  )
  const messageCostUsd = React.useMemo(
    () => resolveMessageCostUsd(props.generationInfo),
    [props.generationInfo]
  )
  const showUsageMetadata =
    isProMode && props.isBot && messageUsage.totalTokens > 0
  const interruptedGeneration = Boolean(
    (props.generationInfo as Record<string, unknown> | undefined)?.interrupted
  )
  const interruptionReason = React.useMemo(() => {
    const raw = (props.generationInfo as Record<string, unknown> | undefined)
      ?.interruptionReason
    if (typeof raw !== "string" || raw.trim().length === 0) return null
    return raw.trim()
  }, [props.generationInfo])
  const streamTransportInterrupted = Boolean(
    (props.generationInfo as Record<string, unknown> | undefined)
      ?.streamTransportInterrupted
  )
  const partialResponseSaved = Boolean(
    (props.generationInfo as Record<string, unknown> | undefined)
      ?.partialResponseSaved
  )
  const streamTransportInterruptionReason = React.useMemo(() => {
    const raw = (props.generationInfo as Record<string, unknown> | undefined)
      ?.streamTransportInterruptionReason
    if (typeof raw !== "string" || raw.trim().length === 0) return null
    return raw.trim()
  }, [props.generationInfo])
  const showPartialSaveMarker =
    streamTransportInterrupted &&
    partialResponseSaved &&
    !interruptedGeneration &&
    !errorPayload
  const hasVisibleAssistantText = props.message.trim().length > 0
  const hasVisibleAssistantImages = Boolean(
    props.images?.some((image) => typeof image === "string" && image.length > 0)
  )
  const hasVisibleAssistantToolCalls = Boolean(
    props.toolCalls && props.toolCalls.length > 0
  )
  const messageTimestamp = React.useMemo(() => {
    const info = props.generationInfo as
      | { created_at?: string | number; createdAt?: string | number; timestamp?: string | number }
      | undefined
    const raw =
      props.createdAt ??
      info?.created_at ??
      info?.createdAt ??
      info?.timestamp
    if (!raw) return null
    const date =
      typeof raw === "number"
        ? new Date(raw)
        : new Date(Date.parse(String(raw)))
    if (Number.isNaN(date.getTime())) return null
    return date.toLocaleTimeString([], {
      hour: "numeric",
      minute: "2-digit"
    })
  }, [props.createdAt, props.generationInfo])
  const fallbackAudit = React.useMemo(
    () => resolveFallbackAudit(props.generationInfo),
    [props.generationInfo]
  )
  const fallbackAuditPolicyLabel = React.useMemo(() => {
    if (!fallbackAudit) return null
    if (fallbackAudit.policy === "auto") {
      return t("playground:routing.policyAuto", "Auto fallback")
    }
    if (fallbackAudit.policy === "pinned") {
      return t("playground:routing.policyPinned", "Provider pinned")
    }
    return t("playground:routing.policyGeneric", "Routing")
  }, [fallbackAudit, t])
  const fallbackAuditPathLabel = React.useMemo(() => {
    if (!fallbackAudit) return null
    if (
      fallbackAudit.fallbackApplied &&
      fallbackAudit.requestedTarget &&
      fallbackAudit.resolvedTarget
    ) {
      return `${fallbackAudit.requestedTarget} → ${fallbackAudit.resolvedTarget}`
    }
    return fallbackAudit.resolvedTarget || fallbackAudit.requestedTarget
  }, [fallbackAudit])
  const imageGenerationMetadata = React.useMemo(
    () => resolveImageGenerationMetadata(props.generationInfo),
    [props.generationInfo]
  )
  const canRegenerateImage =
    props.isBot &&
    Boolean(props.onRegenerateImage) &&
    Boolean(imageGenerationMetadata?.request)
  const showInlineImageActions = canRegenerateImage || Boolean(props.onDeleteImage)
  const researchActions = props.researchActions
  const showUseInChat = Boolean(researchActions?.onUseInChat)
  const showFollowUp = Boolean(researchActions?.onFollowUp)
  const isImageGenerationAssistantEvent =
    props.isBot &&
    Boolean(imageGenerationMetadata?.request) &&
    (!props.message_type ||
      props.message_type === IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE)
  const imageGenerationEventSummary = React.useMemo(() => {
    if (!imageGenerationMetadata?.request) return null

    const request = imageGenerationMetadata.request
    const chips: string[] = [
      String(
        t("playground:imageGeneration.eventBackend", "Backend: {{value}}", {
          value: request.backend
        } as any)
      )
    ]

    if (request.model) {
      chips.push(
        String(
          t("playground:imageGeneration.eventModel", "Model: {{value}}", {
            value: request.model
          } as any)
        )
      )
    }
    if (typeof request.width === "number" && typeof request.height === "number") {
      chips.push(
        String(
          t("playground:imageGeneration.eventSize", "Size: {{width}}x{{height}}", {
            width: request.width,
            height: request.height
          } as any)
        )
      )
    }
    if (request.format) {
      chips.push(
        String(
          t("playground:imageGeneration.eventFormat", "Format: {{value}}", {
            value: request.format.toUpperCase()
          } as any)
        )
      )
    }
    if (typeof request.steps === "number") {
      chips.push(
        String(
          t("playground:imageGeneration.eventSteps", "Steps: {{value}}", {
            value: request.steps
          } as any)
        )
      )
    }
    if (typeof request.cfgScale === "number") {
      chips.push(
        String(
          t("playground:imageGeneration.eventCfg", "CFG: {{value}}", {
            value: request.cfgScale
          } as any)
        )
      )
    }
    if (typeof request.seed === "number") {
      chips.push(
        String(
          t("playground:imageGeneration.eventSeed", "Seed: {{value}}", {
            value: request.seed
          } as any)
        )
      )
    }
    if (request.sampler) {
      chips.push(
        String(
          t("playground:imageGeneration.eventSampler", "Sampler: {{value}}", {
            value: request.sampler
          } as any)
        )
      )
    }

    const sourceLabel =
      imageGenerationMetadata.source === "slash-command"
        ? String(
            t("playground:imageGeneration.eventSourceSlash", "Slash command")
          )
        : imageGenerationMetadata.source === "message-regen"
          ? String(
              t("playground:imageGeneration.eventSourceRegen", "Regenerated")
            )
          : imageGenerationMetadata.source === "generate-modal"
            ? String(
                t("playground:imageGeneration.eventSourceModal", "Generate menu")
              )
            : null

    const refineLabel = imageGenerationMetadata.refine
      ? String(
          t(
            "playground:imageGeneration.eventRefined",
            "Refined with {{model}} ({{ms}} ms)",
            {
              model: imageGenerationMetadata.refine.model,
              ms: imageGenerationMetadata.refine.latencyMs
            } as any
          )
        )
      : null
    const syncLabel = imageGenerationMetadata.sync
      ? imageGenerationMetadata.sync.mode === "off"
        ? String(t("playground:imageGeneration.eventSyncOff", "Local only"))
        : imageGenerationMetadata.sync.status === "synced"
          ? String(
              t("playground:imageGeneration.eventSyncOn", "Mirrored to server")
            )
          : imageGenerationMetadata.sync.status === "failed"
            ? String(
                t("playground:imageGeneration.eventSyncFailed", "Mirror failed")
              )
            : String(
                t("playground:imageGeneration.eventSyncPending", "Mirroring...")
              )
      : null
    const syncStatus = imageGenerationMetadata.sync?.status ?? null

    return {
      prompt: request.prompt,
      chips,
      sourceLabel,
      refineLabel,
      syncLabel,
      syncStatus
    }
  }, [imageGenerationMetadata, t])
  const variantCount = props.variants?.length ?? 0
  const resolvedVariantIndex = (() => {
    const fallback =
      typeof props.activeVariantIndex === "number"
        ? props.activeVariantIndex
        : variantCount > 0
          ? variantCount - 1
          : 0
    if (variantCount <= 0) return 0
    return Math.max(0, Math.min(fallback, variantCount - 1))
  })()
  const imageVariantEntries = React.useMemo(() => {
    const variants = Array.isArray(props.variants) ? props.variants : []
    if (variants.length === 0) {
      const baseImages = Array.isArray(props.images)
        ? props.images.filter((image) => typeof image === "string" && image.length > 0)
        : []
      if (baseImages.length === 0) return []
      return [
        {
          index: 0,
          preview: baseImages[0],
          images: baseImages
        }
      ]
    }
    return variants
      .map((variant, index) => {
        const images = Array.isArray(variant?.images)
          ? variant.images.filter((image) => typeof image === "string" && image.length > 0)
          : []
        if (images.length === 0) return null
        return {
          index,
          preview: images[0],
          images
        }
      })
      .filter((entry): entry is { index: number; preview: string; images: string[] } =>
        Boolean(entry)
      )
  }, [props.images, props.variants])
  const activeVariantPreview = React.useMemo(() => {
    if (imageVariantEntries.length === 0) return null
    return (
      imageVariantEntries.find((entry) => entry.index === resolvedVariantIndex) ||
      imageVariantEntries[0]
    )
  }, [imageVariantEntries, resolvedVariantIndex])
  const compareVariantPreview = React.useMemo(() => {
    if (compareVariantIndex == null) return null
    return (
      imageVariantEntries.find((entry) => entry.index === compareVariantIndex) || null
    )
  }, [compareVariantIndex, imageVariantEntries])
  React.useEffect(() => {
    if (compareVariantIndex == null) return
    const compareStillValid = imageVariantEntries.some(
      (entry) => entry.index === compareVariantIndex
    )
    if (!compareStillValid || compareVariantIndex === resolvedVariantIndex) {
      setCompareVariantIndex(null)
    }
  }, [compareVariantIndex, imageVariantEntries, resolvedVariantIndex])
  const showVariantPager = props.isBot && variantCount > 1
  const canSwipePrev =
    showVariantPager && Boolean(props.onSwipePrev) && resolvedVariantIndex > 0
  const canSwipeNext =
    showVariantPager &&
    Boolean(props.onSwipeNext) &&
    resolvedVariantIndex < variantCount - 1
  const hasMessageKeyboardShortcuts =
    (canSwipePrev || canSwipeNext) ||
    Boolean(props.onNewBranch) ||
    Boolean(props.isBot && props.onRegenerate)
  const handleMessageShortcut = React.useCallback(
    (event: React.KeyboardEvent<HTMLElement>) => {
      const action = resolvePlaygroundMessageShortcutAction(event.nativeEvent)
      if (!action) return

      if (action === "variant_prev") {
        if (!canSwipePrev || !props.onSwipePrev) return
        event.preventDefault()
        props.onSwipePrev()
        articleRef.current?.focus()
        return
      }
      if (action === "variant_next") {
        if (!canSwipeNext || !props.onSwipeNext) return
        event.preventDefault()
        props.onSwipeNext()
        articleRef.current?.focus()
        return
      }
      if (action === "new_branch") {
        if (!props.onNewBranch) return
        event.preventDefault()
        props.onNewBranch()
        articleRef.current?.focus()
        return
      }
      if (action === "regenerate") {
        if (!props.isBot || !props.onRegenerate) return
        event.preventDefault()
        props.onRegenerate()
        articleRef.current?.focus()
      }
    },
    [
      canSwipeNext,
      canSwipePrev,
      props.isBot,
      props.onNewBranch,
      props.onRegenerate,
      props.onSwipeNext,
      props.onSwipePrev
    ]
  )
  const resolvedRole = props.role ?? (props.isBot ? "assistant" : "user")
  const isSystemMessage = resolvedRole === "system"
  const speakerMatchesCharacterIdentity =
    props.speakerCharacterId == null ||
    !props.characterIdentity?.id ||
    String(props.speakerCharacterId) === String(props.characterIdentity.id)
  const shouldUseCharacterIdentity =
    props.isBot &&
    Boolean(props.characterIdentityEnabled) &&
    Boolean(props.characterIdentity?.id) &&
    speakerMatchesCharacterIdentity
  const explicitMoodLabel = normalizeCharacterMoodLabel(props.moodLabel)
  const inferredMoodLabel = React.useMemo(() => {
    if (!props.isBot || isSystemMessage) return null
    return detectCharacterMood({ assistantText: props.message }).label
  }, [isSystemMessage, props.isBot, props.message])
  const resolvedMoodLabel = explicitMoodLabel || inferredMoodLabel
  const moodBadgeLabel = React.useMemo(() => {
    if (!resolvedMoodLabel) return null
    const normalizedMood = resolvedMoodLabel.replace(/_/g, " ")
    if (
      showMoodConfidence &&
      typeof props.moodConfidence === "number" &&
      Number.isFinite(props.moodConfidence)
    ) {
      const percent = Math.max(0, Math.min(100, Math.round(props.moodConfidence * 100)))
      return t("playground:message.moodLabelConfidence", "Mood: {{mood}} ({{confidence}}%)", {
        mood: normalizedMood,
        confidence: percent
      })
    }
    return t("playground:message.moodLabel", "Mood: {{mood}}", {
      mood: normalizedMood
    })
  }, [props.moodConfidence, resolvedMoodLabel, showMoodConfidence, t])
  const baseCharacterAvatar = resolveCharacterBaseAvatarUrl(
    props.characterIdentity
  )
  const moodCharacterAvatar = shouldUseCharacterIdentity
    ? resolveCharacterMoodImageUrl(props.characterIdentity, resolvedMoodLabel)
    : ""
  const characterAvatar = moodCharacterAvatar || baseCharacterAvatar
  const resolvedModelImage =
    shouldUseCharacterIdentity && characterAvatar
      ? characterAvatar
      : props.modelImage
  const resolvedModelName =
    shouldUseCharacterIdentity && props.characterIdentity?.name
      ? props.characterIdentity.name
      : props.modelName || props.name
  const resolvedUserPersonaImage = React.useMemo(() => {
    const raw =
      typeof userPersonaImage === "string" ? userPersonaImage.trim() : ""
    if (!raw) return ""
    if (
      raw.startsWith("data:image/") ||
      raw.startsWith("http://") ||
      raw.startsWith("https://")
    ) {
      return raw
    }
    return ""
  }, [userPersonaImage])
  const portraitImage = isSystemMessage
    ? ""
    : props.isBot
      ? resolvedModelImage || ""
      : resolvedUserPersonaImage
  const messageRenderSide = resolveMessageRenderSide({
    isBot: props.isBot,
    isSystemMessage
  })
  const portraitSide: "left" | "right" = messageRenderSide
  const shouldShowPortrait =
    Boolean(showCharacterPortraits) && Boolean(portraitImage)
  const shouldShowAvatarColumn = !shouldShowPortrait
  const avatarColumnAlignmentClass = resolveAvatarColumnAlignment(
    messageRenderSide
  )
  const userAvatarNode = props.userAvatar ? (
    props.userAvatar
  ) : resolvedUserPersonaImage ? (
    <Avatar
      src={resolvedUserPersonaImage}
      alt={userDisplayName.trim() || t("common:you", "You")}
      className="size-8"
    />
  ) : null
  const shouldPreviewAvatar =
    shouldUseCharacterIdentity && Boolean(characterAvatar)
  const ttsClipMeta = React.useMemo<TtsClipMeta>(
    () => ({
      historyId: props.historyId ?? null,
      serverChatId: props.serverChatId ?? null,
      messageId: props.messageId ?? null,
      serverMessageId: props.serverMessageId ?? null,
      role: resolvedRole,
      source: "chat"
    }),
    [
      props.historyId,
      props.serverChatId,
      props.messageId,
      props.serverMessageId,
      resolvedRole
    ]
  )

  const messageKey = React.useMemo(() => {
    if (props.serverMessageId) return `srv:${props.serverMessageId}`
    if (props.messageId) return `local:${props.messageId}`
    // Always include conversation context to prevent key collisions across chats
    const conversationScope =
      props.serverChatId || props.historyId || props.conversationInstanceId
    return `${conversationScope}:${props.currentMessageIndex}`
  }, [
    props.conversationInstanceId,
    props.currentMessageIndex,
    props.historyId,
    props.messageId,
    props.serverChatId,
    props.serverMessageId
  ])

  useEffect(() => {
    if (typeof window === "undefined") return
    if (!props.isBot && isUserChatBubble) return

    const handleEditMessage = (event: Event) => {
      const detail = (event as CustomEvent<{ messageId?: string }>).detail
      if (!detail?.messageId) return
      if (props.isBot) return
      const matches =
        detail.messageId === props.messageId ||
        detail.messageId === props.serverMessageId
      if (matches) {
        setEditMode(true)
      }
    }

    window.addEventListener(EDIT_MESSAGE_EVENT, handleEditMessage)
    return () => {
      window.removeEventListener(EDIT_MESSAGE_EVENT, handleEditMessage)
    }
  }, [isUserChatBubble, props.isBot, props.messageId, props.serverMessageId])

  const {
    thumb,
    detail,
    sourceFeedback,
    canSubmit,
    isSubmitting: isFeedbackSubmitting,
    showThanks,
    submitThumb,
    submitDetail,
    submitSourceThumb
  } = useFeedback({
    messageKey,
    conversationId: props.serverChatId ?? null,
    messageId: props.serverMessageId ?? null,
    query: props.feedbackQuery ?? null
  })

  const feedbackExplicitAvailable = Boolean(capabilities?.hasFeedbackExplicit)
  const feedbackImplicitAvailable = Boolean(capabilities?.hasFeedbackImplicit)
  const canSaveKnowledge =
    Boolean(capabilities?.hasChatKnowledgeSave) &&
    Boolean(capabilities?.hasNotes || capabilities?.hasFlashcards) &&
    Boolean(props.serverChatId) &&
    Boolean(props.serverMessageId) &&
    !props.temporaryChat &&
    !errorPayload
  const canSaveToNotes = canSaveKnowledge && Boolean(capabilities?.hasNotes)
  const canSaveToFlashcards =
    canSaveKnowledge && Boolean(capabilities?.hasFlashcards)
  const canSaveToWorkspaceNotes =
    Boolean(props.onSaveToWorkspaceNotes) &&
    Boolean((errorFriendlyText || props.message || "").trim()) &&
    !errorPayload
  const canGenerateDocument =
    Boolean(capabilities?.hasChatDocuments) &&
    Boolean(props.serverChatId) &&
    props.isBot &&
    !errorPayload
  const replyId = props.messageId ?? props.serverMessageId ?? null
  const canReply =
    isProMode &&
    Boolean(replyId) &&
    !props.compareSelectable &&
    !props.message_type?.startsWith("compare")

  const buildReplyPreview = React.useCallback(
    (value: string) => {
      const collapsed = value.replace(/\s+/g, " ").trim()
      if (!collapsed) {
        return t("common:replyTargetFallback", "Message")
      }
      if (collapsed.length > 140) {
        return `${collapsed.slice(0, 137)}...`
      }
      return collapsed
    },
    [t]
  )

  const {
    trackCopy,
    trackSourcesExpanded,
    trackSourceClick,
    trackCitationUsed,
    trackDwellTime
  } =
    useImplicitFeedback({
      conversationId: props.serverChatId ?? null,
      messageId: props.serverMessageId ?? null,
      query: props.feedbackQuery ?? null,
      sources: props.sources ?? [],
      enabled: feedbackImplicitAvailable
    })

  useEffect(() => {
    const dwellMessageKey = props.serverMessageId ?? null
    if (!dwellMessageKey) return
    if (!props.isBot || !isLastMessage) return
    if (props.isStreaming || props.isProcessing) return
    if (!feedbackImplicitAvailable) return
    if (responseDwellSentKeyRef.current === dwellMessageKey) return

    const timeout = window.setTimeout(() => {
      trackDwellTime(3000)
      responseDwellSentKeyRef.current = dwellMessageKey
    }, 3000)

    return () => {
      window.clearTimeout(timeout)
    }
  }, [
    feedbackImplicitAvailable,
    isLastMessage,
    props.isBot,
    props.isProcessing,
    props.isStreaming,
    props.serverMessageId,
    trackDwellTime
  ])

  const handleReply = React.useCallback(() => {
    if (!replyId) return
    setReplyTarget({
      id: replyId,
      preview: buildReplyPreview(errorFriendlyText || props.message),
      name: props.name,
      isBot: props.isBot
    })
  }, [
    replyId,
    setReplyTarget,
    buildReplyPreview,
    errorFriendlyText,
    props.message,
    props.name,
    props.isBot
  ])

  const handleCopy = React.useCallback(async () => {
    await copyToClipboard({
      text: errorFriendlyText || props.message,
      formatted: copyAsFormattedText
    })
    trackCopy()
    setIsBtnPressed(true)
    setTimeout(() => {
      setIsBtnPressed(false)
    }, 2000)
  }, [copyAsFormattedText, errorFriendlyText, props.message, trackCopy])

  const handleGenerateDocument = React.useCallback(() => {
    if (!props.serverChatId || typeof window === "undefined") return
    window.dispatchEvent(
      new CustomEvent("tldw:open-document-generator", {
        detail: {
          conversationId: props.serverChatId,
          message: errorFriendlyText || props.message,
          messageId: props.serverMessageId
        }
      })
    )
  }, [errorFriendlyText, props.message, props.serverChatId, props.serverMessageId])

  const handleSaveToWorkspaceNotes = React.useCallback(() => {
    const snippet = (errorFriendlyText || props.message || "").trim()
    if (!snippet || !props.onSaveToWorkspaceNotes) return
    props.onSaveToWorkspaceNotes({
      message: snippet,
      isBot: props.isBot,
      name: props.name,
      messageId: props.messageId || props.serverMessageId || undefined,
      createdAt:
        typeof props.createdAt === "number" || typeof props.createdAt === "string"
          ? props.createdAt
          : undefined
    })
  }, [
    errorFriendlyText,
    props.createdAt,
    props.isBot,
    props.message,
    props.messageId,
    props.name,
    props.onSaveToWorkspaceNotes,
    props.serverMessageId
  ])

  const handleSaveKnowledge = async (makeFlashcard: boolean) => {
    if (!props.serverChatId || !props.serverMessageId) return
    const snippet = (errorFriendlyText || props.message || "").trim()
    if (!snippet) {
      message.error(t("saveToNotesEmpty", "Nothing to save yet."))
      return
    }
    setSavingKnowledge(makeFlashcard ? "flashcard" : "note")
    try {
      await tldwClient.initialize().catch(() => null)
      await tldwClient.saveChatKnowledge(
        {
          conversation_id: props.serverChatId,
          message_id: props.serverMessageId,
          snippet,
          make_flashcard: makeFlashcard
        },
        props.scope ? { scope: props.scope } : undefined
      )
      message.success(
        makeFlashcard
          ? t("savedToFlashcards", "Saved to Flashcards")
          : t("savedToNotes", "Saved to Notes")
      )
    } catch (err: unknown) {
      const errorMessage =
        err instanceof Error ? err.message : t("somethingWentWrong")
      message.error(errorMessage)
    } finally {
      setSavingKnowledge(null)
    }
  }

  const autoCopyToClipboard = async () => {
    if (
      autoCopyResponseToClipboard &&
      props.isBot &&
      isLastMessage &&
      !props.isStreaming &&
      !props.isProcessing &&
      props.message.trim().length > 0 &&
      !errorPayload &&
      !ttsActionDisabled
    ) {
      await copyToClipboard({
        text: props.message,
        formatted: copyAsFormattedText
      })
      trackCopy()
      setIsBtnPressed(true)
      setTimeout(() => {
        setIsBtnPressed(false)
      }, 2000)
    }
  }

  useEffect(() => {
    autoCopyToClipboard()
  }, [
    autoCopyResponseToClipboard,
    props.isBot,
    props.currentMessageIndex,
    props.totalMessages,
    props.isStreaming,
    props.isProcessing,
    props.message
  ])

  const userTextClass = React.useMemo(
    () => buildChatTextClass(userTextColor, userTextFont, userTextSize),
    [userTextColor, userTextFont, userTextSize]
  )

  const assistantTextClass = React.useMemo(
    () =>
      buildChatTextClass(
        assistantTextColor,
        assistantTextFont,
        assistantTextSize
      ),
    [assistantTextColor, assistantTextFont, assistantTextSize]
  )

  const chatTextClass = props.isBot ? assistantTextClass : userTextClass
  const renderGreetingMarkdown =
    props.isBot &&
    (props.message_type === "character:greeting" ||
      props.message_type === "greeting")
  const shouldRenderStreamingPlainText =
    props.isBot &&
    isLastMessage &&
    props.isStreaming &&
    !errorPayload &&
    !renderGreetingMarkdown

  const shouldShowLoadingStatus =
    props.isBot &&
    isLastMessage &&
    (props.isProcessing ||
      props.isStreaming ||
      props.isSearchingInternet ||
      props.actionInfo ||
      props.isEmbedding)

  const isActiveResponse =
    props.isBot &&
    isLastMessage &&
    (props.isStreaming || props.isProcessing)
  const feedbackDisabled =
    !canSubmit ||
    Boolean(errorPayload) ||
    !feedbackExplicitAvailable ||
    isActiveResponse
  const showFeedbackControls =
    resolvedRole !== "user" && Boolean(props.serverMessageId)
  const feedbackDisabledReason =
    !canSubmit || Boolean(errorPayload) || !feedbackExplicitAvailable
      ? t(
          "playground:feedback.unavailable",
          "Feedback is unavailable for this message."
        )
      : t(
          "playground:feedback.disabled",
          "Feedback is available after the response finishes."
        )

  const tldwTtsSelected = ttsProvider === "tldw"
  const ttsBlockedByHealth =
    tldwTtsSelected &&
    (audioHealthState === "unhealthy" || audioHealthState === "unavailable")
  const ttsBlockedByVoices = tldwTtsSelected && voicesAvailable === false
  const ttsActionDisabled = ttsBlockedByHealth || ttsBlockedByVoices
  const ttsDisabledReason = ttsBlockedByHealth
    ? audioHealthState === "unavailable"
      ? t(
          "playground:tts.tldwStatusOffline",
          "Audio API not detected; check your tldw server version."
        )
      : t(
          "playground:tts.chatDisabledUnhealthy",
          "Audio service is unhealthy. Check Settings → Health."
        )
    : ttsBlockedByVoices
      ? t(
          "playground:tts.chatDisabledNoVoices",
          "No TTS voices are available on the server."
        )
      : null

  const handleToggleTts = React.useCallback(() => {
    if (ttsActionDisabled) return
    if (isSpeaking) {
      cancel()
      return
    }
    speak({
      utterance: errorFriendlyText || props.message,
      saveClip: props.isBot,
      clipMeta: props.isBot ? ttsClipMeta : undefined
    })
  }, [
    ttsActionDisabled,
    isSpeaking,
    cancel,
    speak,
    errorFriendlyText,
    props.message,
    props.isBot,
    ttsClipMeta
  ])

  const handleDelete = React.useCallback(() => {
    if (!props.onDeleteMessage) return

    Modal.confirm({
      title: t("common:confirmTitle", "Please confirm"),
      content: t("common:deleteMessageConfirm", "Delete this message?"),
      okText: t("common:delete", "Delete"),
      cancelText: t("common:cancel", "Cancel"),
      okButtonProps: { danger: true },
      onOk: async () => {
        try {
          await props.onDeleteMessage?.()
          if (!props.suppressDeleteSuccessToast) {
            message.success(t("common:deleted", "Deleted"))
          }
        } catch (err) {
          console.error("Failed to delete message:", err)
          const fallback = t("common:deleteFailed", "Delete failed")
          const errorMessage = err instanceof Error ? err.message : ""
          message.error(errorMessage || fallback)
        }
      }
    })
  }, [props.onDeleteMessage, props.suppressDeleteSuccessToast, t])
  const handleOpenModelSettings = React.useCallback(() => {
    if (typeof window === "undefined") return
    window.dispatchEvent(new CustomEvent("tldw:open-model-settings"))
  }, [])
  const handleOpenKnowledgePanel = React.useCallback(() => {
    if (typeof window === "undefined") return
    window.dispatchEvent(
      new CustomEvent("tldw:open-knowledge-panel", {
        detail: { tab: "search" }
      })
    )
  }, [])
  const pinnedSourceKeySet = React.useMemo(() => {
    const set = new Set<string>()
    for (const entry of ragPinnedResults || []) {
      const parts = [
        entry?.url,
        entry?.source,
        entry?.title
      ]
        .map((value) =>
          typeof value === "string" ? value.trim().toLowerCase() : ""
        )
        .filter((value) => value.length > 0)
      for (const value of parts) {
        set.add(value)
      }
    }
    return set
  }, [ragPinnedResults])
  const resolveSourcePinnedState = React.useCallback(
    (source: any): "active" | "inactive" | null => {
      if (!pinnedSourceKeySet.size) return null
      const sourceKeys = [
        source?.url,
        source?.name,
        source?.metadata?.source,
        source?.metadata?.title
      ]
        .map((value) =>
          typeof value === "string" ? value.trim().toLowerCase() : ""
        )
        .filter((value) => value.length > 0)
      if (sourceKeys.length === 0) return "inactive"
      const matches = sourceKeys.some((value) => pinnedSourceKeySet.has(value))
      return matches ? "active" : "inactive"
    },
    [pinnedSourceKeySet]
  )
  const buildSourceReference = React.useCallback(
    (source: any, index: number): string => {
      const sourceLabel =
        source?.name ||
        source?.metadata?.title ||
        source?.metadata?.source ||
        source?.url ||
        t("common:sourceLabel", "Source")
      return `[${index + 1}] ${String(sourceLabel).trim()}`
    },
    [t]
  )
  const buildFollowUpPrompt = React.useCallback(
    (sources: any[]) => {
      const references = sources
        .map((source, index) => buildSourceReference(source, index))
        .join("\n")
      return [
        t(
          "playground:sources.askWithTemplateHeader",
          "Use these sources in your next answer:"
        ),
        references,
        "",
        t(
          "playground:sources.askWithTemplateQuestion",
          "Question:"
        )
      ]
        .filter(Boolean)
        .join("\n")
    },
    [buildSourceReference, t]
  )
  const handleAskWithSources = React.useCallback(
    (sources: any[]) => {
      if (typeof window === "undefined" || !Array.isArray(sources) || sources.length === 0) {
        return
      }
      const prompt = buildFollowUpPrompt(sources)
      window.dispatchEvent(
        new CustomEvent("tldw:set-composer-message", {
          detail: { message: prompt }
        })
      )
      window.dispatchEvent(new CustomEvent("tldw:open-knowledge-panel", {
        detail: { tab: "search" }
      }))
      window.dispatchEvent(new CustomEvent("tldw:focus-composer"))
    },
    [buildFollowUpPrompt]
  )
  const handleQuickMessageAction = React.useCallback(
    (action: QuickMessageAction) => {
      if (typeof window === "undefined") return
      const content = (errorFriendlyText || props.message || "").trim()
      if (!content) return

      const lineage =
        props.serverMessageId ||
        props.messageId ||
        `index:${props.currentMessageIndex + 1}`
      const sourceReferences = (props.sources || []).map((source, index) =>
        buildSourceReference(source, index)
      )
      const prompt = buildQuickMessageActionPrompt({
        action,
        message: content,
        lineage,
        sourceReferences
      })

      window.dispatchEvent(
        new CustomEvent("tldw:set-composer-message", {
          detail: { message: prompt }
        })
      )
      window.dispatchEvent(new CustomEvent("tldw:focus-composer"))
    },
    [
      buildSourceReference,
      errorFriendlyText,
      props.currentMessageIndex,
      props.message,
      props.messageId,
      props.serverMessageId,
      props.sources
    ]
  )
  const handleEnableProviderFallback = React.useCallback(() => {
    updateChatModelSetting("apiProvider", undefined)
    message.info(
      apiProviderOverride
        ? t(
            "playground:errorRecovery.fallbackClearedProvider",
            "Provider override cleared. Retrying with fallback routing."
          )
        : t(
            "playground:errorRecovery.fallbackRetrying",
            "Retrying with provider fallback policy."
          )
    )
    props.onRegenerate()
  }, [apiProviderOverride, props.onRegenerate, t, updateChatModelSetting])
  const errorRecoveryActions = React.useMemo(() => {
    if (!errorPayload) return []
    const actions: Array<{ id: string; label: string; onClick: () => void }> = [
      {
        id: "retry",
        label: t(
          "playground:errorRecovery.retrySameModel",
          "Retry same model"
        ),
        onClick: () => props.onRegenerate()
      },
      {
        id: "switch",
        label: t(
          "playground:errorRecovery.switchModel",
          "Switch model"
        ),
        onClick: handleOpenModelSettings
      },
      {
        id: "fallback",
        label: t(
          "playground:errorRecovery.tryProviderFallback",
          "Try provider fallback"
        ),
        onClick: handleEnableProviderFallback
      }
    ]
    if (props.onContinue && !props.hideContinue) {
      actions.push({
        id: "continue",
        label: t(
          "playground:errorRecovery.continueFromPartial",
          "Continue from partial"
        ),
        onClick: props.onContinue
      })
    }
    if (errorPayload.upgradeUrl) {
      actions.unshift({
        id: "upgrade",
        label: t(
          "playground:errorRecovery.upgradePlan",
          "Upgrade plan"
        ),
        onClick: async () => {
          let url = errorPayload.upgradeUrl!
          // Resolve relative upgrade URLs against the configured server origin
          // so they don't resolve against the extension origin in Plasmo.
          if (url.startsWith("/")) {
            try {
              const cfg = await tldwClient.getConfig()
              const base = String(cfg?.serverUrl || "").replace(/\/$/, "")
              if (base) {
                url = `${base}${url}`
              }
            } catch {
              // best-effort; fall through with relative URL
            }
          }
          window.open(url, "_blank", "noopener")
        }
      })
    }
    return actions
  }, [
    errorPayload,
    handleEnableProviderFallback,
    handleOpenModelSettings,
    props.hideContinue,
    props.onContinue,
    props.onRegenerate,
    t
  ])
  const interruptionRecoveryActions = React.useMemo(() => {
    if (!interruptedGeneration || errorPayload) return []
    const actions: Array<{ id: string; label: string; onClick: () => void }> = [
      {
        id: "retry",
        label: t(
          "playground:errorRecovery.retrySameModel",
          "Retry same model"
        ),
        onClick: () => props.onRegenerate()
      },
      {
        id: "switch",
        label: t(
          "playground:errorRecovery.switchModel",
          "Switch model"
        ),
        onClick: handleOpenModelSettings
      },
      {
        id: "fallback",
        label: t(
          "playground:errorRecovery.tryProviderFallback",
          "Try provider fallback"
        ),
        onClick: handleEnableProviderFallback
      }
    ]
    if (props.onContinue && !props.hideContinue) {
      actions.push({
        id: "continue",
        label: t(
          "playground:errorRecovery.continueFromPartial",
          "Continue from partial"
        ),
        onClick: props.onContinue
      })
    }
    return actions
  }, [
    errorPayload,
    handleEnableProviderFallback,
    handleOpenModelSettings,
    interruptedGeneration,
    props.hideContinue,
    props.onContinue,
    props.onRegenerate,
    t
  ])
  const emptyResponseRecoveryActions = React.useMemo(
    () => [
      {
        id: "retry",
        label: t(
          "playground:errorRecovery.retrySameModel",
          "Retry same model"
        ),
        onClick: () => props.onRegenerate()
      },
      {
        id: "switch",
        label: t(
          "playground:errorRecovery.switchModel",
          "Switch model"
        ),
        onClick: handleOpenModelSettings
      },
      {
        id: "fallback",
        label: t(
          "playground:errorRecovery.tryProviderFallback",
          "Try provider fallback"
        ),
        onClick: handleEnableProviderFallback
      }
    ],
    [
      handleEnableProviderFallback,
      handleOpenModelSettings,
      props.onRegenerate,
      t
    ]
  )

  const actionRowVisibility = isProMode
    ? "flex"
    : "hidden group-hover:flex group-focus-within:flex"
  const overflowChipVisibility = isProMode
    ? "hidden"
    : "inline-flex group-hover:hidden"
  const showInlineActions = !props.isProcessing && !editMode
  const showEmptyAssistantResponse =
    props.isBot &&
    !isSystemMessage &&
    !props.isStreaming &&
    !props.isProcessing &&
    !errorPayload &&
    !interruptedGeneration &&
    !showPartialSaveMarker &&
    !isImageGenerationAssistantEvent &&
    !hasVisibleAssistantText &&
    !hasVisibleAssistantImages &&
    !hasVisibleAssistantToolCalls

  const handleThumbUp = React.useCallback(() => {
    void submitThumb("up")
  }, [submitThumb])

  const handleThumbDown = React.useCallback(() => {
    setIsFeedbackOpen(true)
    void submitThumb("down")
  }, [submitThumb])

  const handleOpenDetails = React.useCallback(() => {
    setIsFeedbackOpen(true)
  }, [])

  useEffect(() => {
    if (
      autoPlayTTS &&
      props.isTTSEnabled &&
      props.isBot &&
      isLastMessage &&
      !props.isStreaming &&
      !props.isProcessing &&
      props.message.trim().length > 0 &&
      !errorPayload
    ) {
      let messageToSpeak = props.message

      speak({
        utterance: messageToSpeak,
        saveClip: true,
        clipMeta: ttsClipMeta
      })
    }
  }, [
    autoPlayTTS,
    props.isTTSEnabled,
    props.isBot,
    props.currentMessageIndex,
    props.totalMessages,
    props.isStreaming,
    props.isProcessing,
    props.message,
    errorPayload,
    ttsActionDisabled,
    ttsClipMeta
  ])

  useEffect(() => {
    if (!discoPersistComments) return
    if (persistedSkillComment) {
      setSkillComment(persistedSkillComment)
      skillCommentGeneratedRef.current = true
    }
  }, [discoPersistComments, persistedSkillComment])

  useEffect(() => {
    const currentMessageId = props.messageId ?? null
    const messageIdChanged =
      previousMessageIdRef.current !== null &&
      previousMessageIdRef.current !== currentMessageId
    const messageChanged =
      previousMessageRef.current !== null &&
      previousMessageRef.current !== props.message
    if (messageIdChanged || messageChanged) {
      skillCommentGeneratedRef.current = false
      setSkillComment(null)
      if (discoPersistComments && props.messageId) {
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === props.messageId
              ? { ...msg, discoSkillComment: undefined }
              : msg
          )
        )
        void updateMessageDiscoSkillComment(props.messageId, null).catch(
          () => null
        )
      }
    }
    previousMessageRef.current = props.message
    previousMessageIdRef.current = currentMessageId
  }, [
    props.message,
    props.messageId,
    props.activeVariantIndex,
    discoPersistComments,
    setMessages
  ])

  // Disco Skills comment generation effect
  useEffect(() => {
    // Only trigger for bot messages that are complete
    if (
      !discoSkillsEnabled ||
      !props.isBot ||
      !isLastMessage ||
      props.isStreaming ||
      props.isProcessing ||
      !props.message?.trim() ||
      errorPayload ||
      (discoPersistComments && persistedSkillComment) ||
      skillCommentGeneratedRef.current ||
      isGeneratingSkillComment ||
      skillComment
    ) {
      return
    }

    // Check if a skill should trigger
    const triggerResult = attemptSkillTrigger(
      props.message,
      discoSkillsStats,
      discoTriggerProbability
    )

    if (!triggerResult) {
      skillCommentGeneratedRef.current = true
      return
    }

    const { skill, passed } = triggerResult

    // Prevent re-generation
    skillCommentGeneratedRef.current = true
    setIsGeneratingSkillComment(true)

    // Generate the skill comment via LLM
    const generateComment = async () => {
      try {
        const model = selectedModel || "gpt-4o-mini"
        const prompt = buildSkillPrompt(skill, props.message, passed)

        await tldwClient.initialize().catch(() => null)
        const response = await tldwClient.createChatCompletion({
          model,
          messages: [
            {
              role: "system",
              content:
                "You are a skill voice from Disco Elysium. Stay in character. Be concise (1-3 sentences max). Do not use quotation marks."
            },
            { role: "user", content: prompt }
          ],
          temperature: 0.9,
          max_tokens: 150
        })

        const data = await response.json()
        const commentText =
          data?.choices?.[0]?.message?.content?.trim() ||
          data?.content?.trim() ||
          ""

        if (commentText) {
          const comment = createSkillComment(
            skill,
            commentText,
            passed,
            props.messageId
          )
          setSkillComment(comment)
          if (discoPersistComments && props.messageId) {
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === props.messageId
                  ? { ...msg, discoSkillComment: comment }
                  : msg
              )
            )
            void updateMessageDiscoSkillComment(props.messageId, comment).catch(
              () => null
            )
          }
        }
      } catch (err) {
        console.error("Failed to generate disco skill comment:", err)
      } finally {
        setIsGeneratingSkillComment(false)
      }
    }

    generateComment()
  }, [
    discoSkillsEnabled,
    discoSkillsStats,
    discoTriggerProbability,
    props.isBot,
    isLastMessage,
    props.isStreaming,
    props.isProcessing,
    props.message,
    props.messageId,
    errorPayload,
    selectedModel,
    isGeneratingSkillComment,
    discoPersistComments,
    persistedSkillComment,
    skillComment,
    setMessages
  ])

  const compareLabel = t("playground:composer.compareTag", "Compare")
  const compareSelectedLabel = t(
    "playground:composer.compareSelectedTag",
    "Compared"
  )
  const compareErrorLabel = t("playground:composer.error.label", "Error")
  const systemLabel = t("playground:systemPrompt", "System prompt")
  const messageRole = isSystemMessage
    ? "system"
    : props.isBot
      ? "assistant"
      : "user"
  const messageRoleLabel =
    messageRole === "system"
      ? t("message.role.system", "System")
      : messageRole === "assistant"
        ? t("message.role.assistant", "Assistant")
        : t("message.role.user", "User")
  const messageAriaLabel = t("message.ariaLabel", {
    defaultValue: "{{role}} message {{current}} of {{total}}",
    role: messageRoleLabel,
    current: props.currentMessageIndex + 1,
    total: props.totalMessages
  }) as string

  if (
    isUserChatBubble &&
    !props.isBot &&
    !isSystemMessage &&
    !showCharacterPortraits
  ) {
    return (
      <PlaygroundUserMessageBubble
        {...props}
        role={resolvedRole}
        onDelete={props.onDeleteMessage ? handleDelete : undefined}
      />
    )
  }

  const MARKDOWN_BASE_CLASSES =
    "prose break-words text-message dark:prose-invert prose-p:leading-relaxed prose-pre:p-0 dark:prose-dark max-w-none"
  const hasSources = props.isBot && Boolean(props?.sources?.length)
  const messageSpacing = isProMode
    ? `gap-2 px-4 pt-3 ${hasSources ? "pb-4" : "pb-2.5"}`
    : `gap-1.5 px-3 pt-2 ${hasSources ? "pb-3" : "pb-2"}`
  const messageCardClass = isSystemMessage
    ? `flex max-w-[calc(100%-1.75rem)] flex-col rounded-2xl border border-dashed border-warn/30 bg-warn/10 shadow-sm ${messageSpacing}`
    : props.isBot
      ? `flex max-w-[calc(100%-1.75rem)] flex-col rounded-2xl border border-border/50 bg-surface/60 shadow-sm border-l-2 border-l-primary/20 ${messageSpacing}`
      : `flex max-w-[calc(100%-1.75rem)] flex-col rounded-2xl border border-border/50 bg-surface2/60 shadow-sm ${messageSpacing}`
  const portraitPanel = shouldShowPortrait ? (
    <button
      type="button"
      onClick={() => setIsAvatarPreviewOpen(true)}
      className="relative hidden h-40 w-28 shrink-0 overflow-hidden rounded-2xl border border-border/60 bg-surface/30 shadow-sm transition hover:brightness-110 focus:outline-none focus:ring-2 focus:ring-focus sm:block md:h-52 md:w-36"
      aria-label={t("playground:previewCharacterAvatar", {
        defaultValue: "Preview character avatar"
      }) as string}
    >
      <img
        src={portraitImage}
        alt={
          props.isBot
            ? resolvedModelName || props.name
            : userDisplayName.trim() || t("common:you", "You")
        }
        className="h-full w-full object-cover"
        loading="lazy"
      />
      <span className="pointer-events-none absolute inset-0 bg-gradient-to-t from-black/35 via-transparent to-transparent" />
    </button>
  ) : null
  const avatarColumn = shouldShowAvatarColumn ? (
    <div className={`w-8 flex flex-col relative ${avatarColumnAlignmentClass}`}>
      {props.isBot ? (
        !resolvedModelImage ? (
          <div className="relative h-7 w-7 p-1 rounded-sm text-white flex items-center justify-center  text-opacity-100">
            <div className="absolute h-8 w-8 rounded-full bg-gradient-to-r from-green-300 to-purple-400"></div>
          </div>
        ) : shouldPreviewAvatar ? (
          <button
            type="button"
            onClick={() => setIsAvatarPreviewOpen(true)}
            className="rounded-full focus:outline-none focus:ring-2 focus:ring-focus"
            aria-label={t("playground:previewCharacterAvatar", {
              defaultValue: "Preview character avatar"
            }) as string}
          >
            <Avatar
              src={resolvedModelImage}
              alt={resolvedModelName || props.name}
              className="size-8"
            />
          </button>
        ) : (
          <Avatar
            src={resolvedModelImage}
            alt={resolvedModelName || props.name}
            className="size-8"
          />
        )
      ) : isSystemMessage ? (
        <div className="relative h-7 w-7 p-1 rounded-sm text-warn flex items-center justify-center text-opacity-100">
          <div className="absolute h-8 w-8 rounded-full border border-warn/40 bg-warn/10"></div>
        </div>
      ) : !userAvatarNode ? (
        <div className="relative h-7 w-7 p-1 rounded-sm text-white flex items-center justify-center  text-opacity-100">
          <div className="absolute h-8 w-8 rounded-full from-primary/60 to-primary bg-gradient-to-r"></div>
        </div>
      ) : (
        userAvatarNode
      )}
    </div>
  ) : null
  return (
    <article
      ref={articleRef}
      data-testid="chat-message"
      data-role={messageRole}
      data-message-type={props.message_type}
      data-index={props.currentMessageIndex}
      data-message-id={props.messageId}
      data-server-message-id={props.serverMessageId}
      data-search-match={props.searchMatch || undefined}
      aria-label={messageAriaLabel}
      aria-busy={props.isStreaming && isLastMessage ? true : undefined}
      tabIndex={hasMessageKeyboardShortcuts ? 0 : undefined}
      onKeyDown={hasMessageKeyboardShortcuts ? handleMessageShortcut : undefined}
      className={`group relative flex w-full max-w-5xl flex-col items-end justify-center text-text ${
        isProMode ? "pb-3 md:px-5" : "pb-2 md:px-4"
      } ${checkWideMode ? "max-w-none" : ""} ${
        props.searchMatch === "active"
          ? "rounded-lg ring-2 ring-primary/60 ring-offset-2 ring-offset-bg"
          : props.searchMatch === "match"
            ? "rounded-lg ring-1 ring-primary/35 ring-offset-1 ring-offset-bg"
            : ""
      }`}>
      {/* "Generating..." indicator while streaming on the latest assistant message */}
      {props.isBot && (props.isStreaming || props.isProcessing) && isLastMessage && !props.message && (
        <div className="flex self-start items-center gap-2 px-4 py-2 text-xs text-text-muted">
          <span className="inline-flex gap-0.5">
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-text-muted [animation-delay:0ms]" />
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-text-muted [animation-delay:150ms]" />
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-text-muted [animation-delay:300ms]" />
          </span>
          {t("playground:status.generating", "Generating...")}
        </div>
      )}
      {/* Inline stop button while streaming on the latest assistant message */}
      {props.isBot && (props.isStreaming || props.isProcessing) && isLastMessage && props.onStopStreaming && (
        <div className="absolute right-2 top-0 z-10">
          <Tooltip title={t("playground:tooltip.stopStreaming") as string}>
            <button
              type="button"
              onClick={props.onStopStreaming}
              data-testid="chat-message-stop-streaming"
              title={t("playground:composer.stopStreaming") as string}
              className="rounded-md border border-border bg-surface/70 p-1 text-text backdrop-blur hover:bg-surface">
              <StopCircleIcon className="w-5 h-5" />
              <span className="sr-only">{t("playground:composer.stopStreaming")}</span>
            </button>
          </Tooltip>
        </div>
      )}
      {/* <div className="text-base md:max-w-2xl lg:max-w-xl xl:max-w-3xl flex lg:px-0 m-auto w-full"> */}
      <div
        className={`flex flex-row m-auto w-full ${
          isProMode ? "gap-4 md:gap-6 my-2" : "gap-3 md:gap-4 my-1.5"
        }`}>
        {portraitSide === "left" ? portraitPanel : null}
        {portraitSide === "left" ? avatarColumn : null}
        <div className="flex min-w-0 flex-1 flex-col gap-2">
          <div className={messageCardClass}>
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex flex-wrap items-center gap-2">
                {isSystemMessage ? (
                  <span className="inline-flex items-center rounded-full border border-warn/40 bg-warn/10 px-2 py-0.5 text-xs font-medium text-warn">
                    {systemLabel}
                  </span>
                ) : (
                  <span className="text-caption font-semibold text-text">
                    {props.isBot
                      ? removeModelSuffix(
                          `${resolvedModelName || props?.name}`?.replaceAll(
                            /accounts\/[^\/]+\/models\//g,
                            ""
                          )
                        )
                      : userDisplayName.trim() || t("common:you", "You")}
                  </span>
                )}
                {messageTimestamp && (
                  <span className="text-xs text-text-muted">
                    • {messageTimestamp}
                  </span>
                )}
                {props.isBot && !isSystemMessage && resolvedResponseModel && (
                  <span
                    data-testid="message-model-badge"
                    className="inline-flex items-center rounded-full border border-border bg-surface2 px-1.5 py-0.5 text-[10px] font-medium text-text-muted"
                    title={resolvedResponseModel}
                  >
                    {resolvedResponseModel}
                  </span>
                )}
                {props.isBot && !isSystemMessage && showMoodBadge && moodBadgeLabel && (
                  <span
                    data-testid="message-mood-indicator"
                    className="inline-flex items-center gap-1 rounded-full border border-accent/40 bg-accent/10 px-2 py-0.5 text-xs font-medium text-accent"
                  >
                    <Smile className="h-3 w-3" aria-hidden="true" />
                    <span>{moodBadgeLabel}</span>
                  </span>
                )}
                {props?.message_type && (
                  <Tag
                    className="!m-0"
                    color={tagColors[props?.message_type] || "default"}>
                    {t(`copilot.${props?.message_type}`)}
                  </Tag>
                )}
                {props.isBot && props.message_type === "compare:reply" && (
                  <div className="flex items-center gap-2">
                    {props.compareSelectable && props.onToggleCompareSelect ? (
                      <button
                        type="button"
                        onClick={props.onToggleCompareSelect}
                        aria-label={
                          props.compareSelected
                            ? compareSelectedLabel
                            : compareLabel
                        }
                        aria-pressed={props.compareSelected}
                        title={
                          props.compareSelected
                            ? compareSelectedLabel
                            : compareLabel
                        }
                        className={`inline-flex items-center gap-1 rounded-full px-2 py-1 text-xs font-medium border transition ${
                          props.compareSelected
                            ? "bg-primary text-white border-primary"
                            : "bg-primary/10 text-primary border-primary/30"
                        }`}
                      >
                        {props.compareSelected && (
                          <CheckCircle2 className="h-3 w-3" aria-hidden="true" />
                        )}
                        {props.compareSelected
                          ? compareSelectedLabel
                          : compareLabel}
                      </button>
                    ) : (
                      <span className="inline-flex items-center gap-1 rounded-full bg-primary/10 px-2 py-1 text-xs font-medium text-primary">
                        <CheckCircle2 className="h-3 w-3" aria-hidden="true" />
                        {compareLabel}
                      </span>
                    )}
                    {props.compareError && (
                      <div
                        aria-label={compareErrorLabel}
                        className="flex items-center gap-2 rounded-md border border-danger/30 bg-danger/5 px-3 py-2">
                        <AlertTriangle className="h-4 w-4 flex-shrink-0 text-danger" aria-hidden="true" />
                        <div className="min-w-0 flex-1">
                          <p className="text-xs font-medium text-danger">
                            {t("playground:compareErrorTitle", "Response failed")}
                          </p>
                          {props.compareErrorModelLabel && (
                            <p className="text-[10px] text-text-muted">
                              {props.compareErrorModelLabel}
                            </p>
                          )}
                        </div>
                        {props.onCompareRetry && (
                          <button
                            type="button"
                            onClick={props.onCompareRetry}
                            className="rounded border border-danger/30 px-2 py-0.5 text-[10px] text-danger hover:bg-danger/10"
                          >
                            {t("playground:compareRetry", "Retry")}
                          </button>
                        )}
                      </div>
                    )}
                    {props.compareChosen && (
                      <span className="inline-flex items-center gap-1 rounded-full bg-success/10 px-2 py-1 text-xs font-medium text-success">
                        <CheckCircle2 className="h-3 w-3" aria-hidden="true" />
                        {t(
                          "playground:composer.compareChosenLabel",
                          "Chosen"
                        )}
                      </span>
                    )}
                  </div>
                )}
              </div>
            </div>

          {/* Unified loading status indicator */}
          {shouldShowLoadingStatus && (
            <LoadingStatus
              isProcessing={props.isProcessing}
              isStreaming={props.isStreaming}
              isSearchingInternet={props.isSearchingInternet}
              isEmbedding={props.isEmbedding}
              actionInfo={props.actionInfo}
            />
          )}
          {showUsageMetadata && (
            <div className="text-xs text-text-muted tabular-nums">
              {messageUsage.promptTokens}{" "}
              {t("playground:tokens.prompt", "prompt")} +{" "}
              {messageUsage.completionTokens}{" "}
              {t("playground:tokens.completion", "completion")} ={" "}
              {messageUsage.totalTokens}{" "}
              {t("playground:tokens.total", "tokens")}
              {messageCostUsd != null && (
                <>
                  {" "}
                  • {formatCost(messageCostUsd)}
                </>
              )}
            </div>
          )}
          {props.isBot && !isSystemMessage && fallbackAudit && (
            <Tooltip
              title={
                fallbackAudit.fallbackApplied && fallbackAudit.requestedTarget && fallbackAudit.resolvedTarget
                  ? String(t("playground:routing.fallbackTooltip",
                      "Requested: {{requested}}. Fell back to: {{resolved}}",
                      {
                        requested: fallbackAudit.requestedTarget,
                        resolved: fallbackAudit.resolvedTarget
                      } as any)
                    )
                  : undefined
              }
            >
              <div
                data-testid="message-fallback-audit"
                className="text-[11px] text-text-muted"
              >
                {fallbackAudit.resolvedTarget && (
                  <span>
                    {String(t("playground:routing.using", "Using: {{target}}", {
                      target: fallbackAudit.resolvedTarget
                    } as any))}
                  </span>
                )}
                {!fallbackAudit.resolvedTarget && fallbackAuditPolicyLabel}
                {!fallbackAudit.resolvedTarget && fallbackAuditPathLabel ? ` • ${fallbackAuditPathLabel}` : ""}
                {fallbackAudit.fallbackApplied && fallbackAudit.resolvedTarget && (
                  <span className="ml-1 text-warn">
                    ({t("playground:routing.fellBack", "fallback")})
                  </span>
                )}
                {typeof fallbackAudit.attempts === "number" &&
                fallbackAudit.attempts > 1
                  ? ` • ${t("playground:routing.attempts", "{{count}} attempts", {
                      count: fallbackAudit.attempts
                    } as any)}`
                  : ""}
                {fallbackAudit.reason ? ` • ${fallbackAudit.reason}` : ""}
              </div>
            </Tooltip>
          )}
          {isImageGenerationAssistantEvent && imageGenerationEventSummary && (
            <div
              data-testid="playground-image-event-card"
              className="rounded-md border border-primary/25 bg-primary/5 px-3 py-2 text-xs text-text"
            >
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="font-semibold text-primaryStrong">
                  {t(
                    "playground:imageGeneration.eventTitle",
                    "Image artifact event"
                  )}
                </span>
                <div className="flex flex-wrap items-center gap-1.5">
                  {imageGenerationEventSummary.sourceLabel && (
                    <span className="rounded-full border border-primary/30 bg-surface px-2 py-0.5 text-[11px] text-primaryStrong">
                      {imageGenerationEventSummary.sourceLabel}
                    </span>
                  )}
                  {imageGenerationEventSummary.syncLabel && (
                    <span
                      className={`rounded-full border px-2 py-0.5 text-[11px] ${
                        imageGenerationEventSummary.syncStatus === "failed"
                          ? "border-danger/40 bg-danger/10 text-danger"
                          : imageGenerationEventSummary.syncStatus === "synced"
                            ? "border-success/40 bg-success/10 text-success"
                            : "border-primary/30 bg-surface text-primaryStrong"
                      }`}
                    >
                      {imageGenerationEventSummary.syncLabel}
                    </span>
                  )}
                </div>
              </div>
              <p
                data-testid="playground-image-event-prompt"
                className="mt-2 whitespace-pre-wrap text-sm text-text"
              >
                {imageGenerationEventSummary.prompt}
              </p>
              <div className="mt-2 flex flex-wrap gap-1.5">
                {imageGenerationEventSummary.chips.map((chip, index) => (
                  <span
                    key={`img-event-chip-${index}-${chip}`}
                    className="rounded-full border border-border/70 bg-surface px-2 py-0.5 text-[11px] text-text-muted"
                  >
                    {chip}
                  </span>
                ))}
              </div>
              {imageGenerationEventSummary.refineLabel && (
                <div className="mt-2 text-[11px] text-text-muted">
                  {imageGenerationEventSummary.refineLabel}
                </div>
              )}
              {imageVariantEntries.length > 1 && (
                <div
                  data-testid="playground-image-variant-strip"
                  className="mt-3 rounded-md border border-border/70 bg-surface/50 p-2"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2 text-[11px]">
                    <span className="font-medium text-text-muted">
                      {String(
                        t(
                          "playground:imageGeneration.eventVariants",
                          "Variants {{current}}/{{total}}",
                          {
                            current: resolvedVariantIndex + 1,
                            total: imageVariantEntries.length
                          } as any
                        )
                      )}
                    </span>
                    <div className="flex flex-wrap items-center gap-1.5">
                      {props.onKeepImageVariant && (
                        <button
                          type="button"
                          data-testid="playground-image-variant-keep-active"
                          className="rounded border border-success/35 bg-success/10 px-2 py-0.5 text-[11px] font-medium text-success hover:bg-success/15"
                          onClick={() =>
                            props.onKeepImageVariant?.({
                              messageId: props.messageId,
                              variantIndex: resolvedVariantIndex
                            })
                          }
                        >
                          {t(
                            "playground:imageGeneration.keepActiveVariant",
                            "Keep active"
                          )}
                        </button>
                      )}
                      {props.onDeleteAllImageVariants && (
                        <button
                          type="button"
                          data-testid="playground-image-variant-delete-all"
                          className="rounded border border-danger/35 bg-danger/10 px-2 py-0.5 text-[11px] font-medium text-danger hover:bg-danger/15"
                          onClick={() =>
                            props.onDeleteAllImageVariants?.({
                              messageId: props.messageId
                            })
                          }
                        >
                          {t("playground:imageGeneration.deleteAllVariants", "Delete all")}
                        </button>
                      )}
                    </div>
                  </div>

                  <div className="mt-2 flex flex-wrap items-start gap-2">
                    {imageVariantEntries.map((entry) => {
                      const isActive = entry.index === resolvedVariantIndex
                      const isCompared = compareVariantIndex === entry.index
                      return (
                        <div
                          key={`image-variant-${entry.index}`}
                          className={`rounded border p-1 ${
                            isActive
                              ? "border-primary/60 bg-primary/10"
                              : "border-border/70 bg-surface"
                          }`}
                        >
                          <button
                            type="button"
                            data-testid={`playground-image-variant-select-${entry.index}`}
                            className="flex flex-col items-center gap-1"
                            onClick={() =>
                              props.onSelectImageVariant?.({
                                messageId: props.messageId,
                                variantIndex: entry.index
                              })
                            }
                          >
                            <img
                              src={entry.preview}
                              alt={t(
                                "playground:imageGeneration.variantPreview",
                                "Variant {{index}} preview",
                                { index: entry.index + 1 } as any
                              ) as string}
                              className="h-12 w-12 rounded object-cover"
                              loading="lazy"
                            />
                            <span className="text-[10px] text-text-muted">
                              {String(
                                t("playground:imageGeneration.variantLabel", "V{{index}}", {
                                  index: entry.index + 1
                                } as any)
                              )}
                            </span>
                          </button>
                          <div className="mt-1 flex flex-wrap items-center justify-center gap-1">
                            {!isActive && (
                              <button
                                type="button"
                                data-testid={`playground-image-variant-compare-${entry.index}`}
                                className="rounded border border-border px-1.5 py-0.5 text-[10px] text-text-muted hover:bg-surface2"
                                onClick={() =>
                                  setCompareVariantIndex((prev) =>
                                    prev === entry.index ? null : entry.index
                                  )
                                }
                              >
                                {isCompared
                                  ? t(
                                      "playground:imageGeneration.hideCompareVariant",
                                      "Hide compare"
                                    )
                                  : t(
                                      "playground:imageGeneration.compareVariant",
                                      "Compare"
                                    )}
                              </button>
                            )}
                            {props.onDeleteImageVariant && (
                              <button
                                type="button"
                                data-testid={`playground-image-variant-delete-${entry.index}`}
                                className="rounded border border-danger/35 px-1.5 py-0.5 text-[10px] text-danger hover:bg-danger/10"
                                onClick={() =>
                                  props.onDeleteImageVariant?.({
                                    messageId: props.messageId,
                                    variantIndex: entry.index
                                  })
                                }
                              >
                                {t("common:delete", "Delete")}
                              </button>
                            )}
                          </div>
                        </div>
                      )
                    })}
                  </div>

                  {activeVariantPreview && compareVariantPreview && (
                    <div
                      data-testid="playground-image-variant-compare-preview"
                      className="mt-2 grid gap-2 sm:grid-cols-2"
                    >
                      <div className="rounded border border-border/70 bg-surface p-2">
                        <p className="mb-1 text-[10px] font-medium uppercase tracking-wide text-text-muted">
                          {t("playground:imageGeneration.activeVariant", "Active")}
                        </p>
                        <img
                          src={activeVariantPreview.preview}
                          alt={t(
                            "playground:imageGeneration.activeVariant",
                            "Active"
                          ) as string}
                          className="h-28 w-full rounded object-cover"
                          loading="lazy"
                        />
                      </div>
                      <div className="rounded border border-border/70 bg-surface p-2">
                        <p className="mb-1 text-[10px] font-medium uppercase tracking-wide text-text-muted">
                          {t("playground:imageGeneration.compareVariant", "Compare")}
                        </p>
                        <img
                          src={compareVariantPreview.preview}
                          alt={t(
                            "playground:imageGeneration.compareVariant",
                            "Compare"
                          ) as string}
                          className="h-28 w-full rounded object-cover"
                          loading="lazy"
                        />
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
          {showPartialSaveMarker && (
            <div
              role="status"
              aria-live="polite"
              title={streamTransportInterruptionReason || undefined}
              className="inline-flex items-center rounded-md border border-border/60 bg-surface px-2 py-1 text-[11px] font-medium text-text-muted"
            >
              {t(
                "playground:errorRecovery.partialSavedMarker",
                "Connection dropped. Partial response saved."
              )}
            </div>
          )}
          {interruptedGeneration && !errorPayload && (
            <div
              role="status"
              aria-live="polite"
              className="rounded-md border border-warn/30 bg-warn/10 p-2 text-xs text-warn">
              <p className="font-medium">
                {t(
                  "playground:errorRecovery.interruptedSummary",
                  "Generation was interrupted. You can retry, switch model, or continue from the partial response."
                )}
              </p>
              {interruptionReason && (
                <p className="mt-1 opacity-90">{interruptionReason}</p>
              )}
              {interruptionRecoveryActions.length > 0 && (
                <div className="mt-2 flex flex-wrap items-center gap-1.5">
                  <span className="sr-only">
                    Recommended next actions:{" "}
                    {interruptionRecoveryActions
                      .map((action) => action.label)
                      .join(", ")}
                  </span>
                  {interruptionRecoveryActions.map((action) => (
                    <button
                      key={action.id}
                      type="button"
                      onClick={action.onClick}
                      className="rounded border border-warn/40 bg-surface px-2 py-1 text-[11px] font-medium text-warn transition hover:bg-warn/10"
                    >
                      {action.label}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}
          <div className="flex flex-grow flex-col">
            {!editMode ? (
              props.isBot ? (
                errorPayload ? (
                  <ErrorBubble
                    payload={errorPayload}
                    toggleLabels={{
                      show: t(
                        "error.showDetails",
                        "Show technical details"
                      ) as string,
                      hide: t(
                        "error.hideDetails",
                        "Hide technical details"
                      ) as string
                    }}
                    recoveryActions={errorRecoveryActions}
                  />
                ) : shouldRenderStreamingPlainText ? (
                  <p
                    data-testid="playground-streaming-plain-text"
                    className={`text-body text-text-muted whitespace-pre-wrap ${assistantTextClass}`}
                  >
                    {props.message}
                  </p>
                ) : showEmptyAssistantResponse ? (
                  <EmptyAssistantResponseNotice
                    summary={
                      t(
                        "playground:errorRecovery.emptyResponseSummary",
                        "No response text was returned."
                      ) as string
                    }
                    detail={
                      t(
                        "playground:errorRecovery.emptyResponseDetail",
                        "The request completed, but the assistant did not return visible content."
                      ) as string
                    }
                    recoveryActions={emptyResponseRecoveryActions}
                  />
                ) : renderGreetingMarkdown ? (
                  <React.Suspense
                    fallback={
                      <p
                        className={`text-body text-text-muted ${assistantTextClass}`}>
                        {t("loading.content")}
                      </p>
                    }>
                    <Markdown
                      message={props.message}
                      className={`${MARKDOWN_BASE_CLASSES} ${assistantTextClass}`}
                      searchQuery={props.searchQuery}
                      codeBlockVariant="compact"
                    />
                  </React.Suspense>
                ) : (
                  <>
                    {parseReasoning(props.message).map((e, i) => {
                      if (e.type === "reasoning") {
                        return (
                          <ReasoningBlock
                            key={`reasoning-${i}`}
                            content={e.content}
                            isStreaming={props.isStreaming}
                            reasoningRunning={e.reasoning_running}
                            openReasoning={props.openReasoning}
                            reasoningTimeTaken={props.reasoningTimeTaken}
                            assistantTextClass={assistantTextClass}
                            markdownBaseClasses={MARKDOWN_BASE_CLASSES}
                            searchQuery={props.searchQuery}
                            t={t}
                          />
                        )
                      }

                      return (
                        <React.Suspense
                          key={`message-${i}`}
                          fallback={
                            <p
                              className={`text-body text-text-muted ${assistantTextClass}`}>
                              {t("loading.content")}
                            </p>
                          }>
                          <Markdown
                            message={e.content}
                            className={`${MARKDOWN_BASE_CLASSES} ${assistantTextClass}`}
                            searchQuery={props.searchQuery}
                            codeBlockVariant="github"
                          />
                        </React.Suspense>
                      )
                    })}
                  </>
                )
              ) : (
                <p
                  className={`prose max-w-none dark:prose-invert whitespace-pre-line prose-p:leading-relaxed prose-pre:p-0 dark:prose-dark ${chatTextClass} ${
                    props.message_type &&
                    "italic text-text-muted text-body"
                  }
                  `}>
                  {props.searchQuery
                    ? highlightText(props.message, props.searchQuery)
                    : props.message}
                </p>
              )
            ) : (
              <EditMessageForm
                value={props.message}
                onSumbit={props.onEditFormSubmit}
                onClose={() => setEditMode(false)}
                isBot={props.isBot}
              />
            )}
          </div>
          {/* images if available */}
          {props.images &&
            props.images.filter((img) => img.length > 0).length > 0 && (
              <div className="mt-2 flex flex-wrap gap-3">
                {props.images
                  .filter((image) => image.length > 0)
                  .map((image, index) => (
                    <div key={index} className="group relative">
                      <Image
                        src={image}
                        alt="Uploaded Image"
                        width={180}
                        className="rounded-md relative"
                      />
                      {showInlineImageActions && (
                        <div className="pointer-events-none absolute right-2 top-2 flex items-center gap-1 rounded-full border border-border/70 bg-surface/90 px-1 py-1 opacity-0 shadow-sm transition group-hover:opacity-100 group-focus-within:opacity-100">
                          {canRegenerateImage && (
                            <button
                              type="button"
                              className="pointer-events-auto inline-flex h-7 w-7 items-center justify-center rounded-full text-text-muted transition hover:bg-surface2 hover:text-text"
                              aria-label={t(
                                "playground:imageGeneration.regenerateImage",
                                "Regenerate image"
                              ) as string}
                              title={t(
                                "playground:imageGeneration.regenerateImage",
                                "Regenerate image"
                              ) as string}
                              onClick={() => {
                                void props.onRegenerateImage?.({
                                  messageId: props.messageId,
                                  imageIndex: index,
                                  imageUrl: image,
                                  request: imageGenerationMetadata?.request ?? null
                                })
                              }}
                            >
                              <RotateCcw className="h-3.5 w-3.5" aria-hidden="true" />
                            </button>
                          )}
                          {props.onDeleteImage && (
                            <button
                              type="button"
                              className="pointer-events-auto inline-flex h-7 w-7 items-center justify-center rounded-full text-text-muted transition hover:bg-danger/10 hover:text-danger"
                              aria-label={t(
                                "playground:imageGeneration.deleteImage",
                                "Delete image"
                              ) as string}
                              title={t(
                                "playground:imageGeneration.deleteImage",
                                "Delete image"
                              ) as string}
                              onClick={() => {
                                props.onDeleteImage?.({
                                  messageId: props.messageId,
                                  imageIndex: index,
                                  imageUrl: image
                                })
                              }}
                            >
                              <Trash2 className="h-3.5 w-3.5" aria-hidden="true" />
                            </button>
                          )}
                        </div>
                      )}
                    </div>
                  ))}
              </div>
            )}

          {(showUseInChat || showFollowUp || researchActions?.primaryLink) && (
            <div className="mt-3 flex">
              <div className="flex items-center gap-2">
                {researchActions?.reasonLabel && (
                  <span className="text-sm text-text-subtle">
                    {researchActions.reasonLabel}
                  </span>
                )}
                {researchActions?.primaryLink && (
                  <a
                    href={researchActions.primaryLink.href}
                    className="rounded-full border border-border/70 bg-surface px-3 py-1 text-sm font-medium text-primary transition hover:border-primary/50 hover:text-primary"
                  >
                    {researchActions.primaryLink.label}
                  </a>
                )}
                {showUseInChat && (
                  <button
                    type="button"
                    className="rounded-full border border-border/70 bg-surface px-3 py-1 text-sm font-medium text-text transition hover:border-primary/50 hover:text-primary"
                    onClick={researchActions?.onUseInChat}
                  >
                    Use in Chat
                  </button>
                )}
                {showFollowUp && (
                  <button
                    type="button"
                    className="rounded-full border border-border/70 bg-surface px-3 py-1 text-sm font-medium text-text transition hover:border-primary/50 hover:text-primary"
                    onClick={researchActions?.onFollowUp}
                  >
                    Follow up
                  </button>
                )}
              </div>
            </div>
          )}

          {showInlineActions && (
            <MessageActionsBar
              t={t}
              isProMode={isProMode}
              isBot={props.isBot}
              showVariantPager={showVariantPager}
              resolvedVariantIndex={resolvedVariantIndex}
              variantCount={variantCount}
              canSwipePrev={canSwipePrev}
              canSwipeNext={canSwipeNext}
              onSwipePrev={props.onSwipePrev}
              onSwipeNext={props.onSwipeNext}
              overflowChipVisibility={overflowChipVisibility}
              actionRowVisibility={actionRowVisibility}
              isTtsEnabled={props.isTTSEnabled}
              ttsDisabledReason={ttsDisabledReason}
              ttsActionDisabled={ttsActionDisabled}
              isSpeaking={isSpeaking}
              onToggleTts={handleToggleTts}
              hideCopy={props.hideCopy}
              copyPressed={isBtnPressed}
              onCopy={handleCopy}
              canReply={canReply}
              onReply={handleReply}
              canSaveToWorkspaceNotes={canSaveToWorkspaceNotes}
              onSaveToWorkspaceNotes={handleSaveToWorkspaceNotes}
              canSaveToNotes={canSaveToNotes}
              canSaveToFlashcards={canSaveToFlashcards}
              canGenerateDocument={canGenerateDocument}
              onGenerateDocument={handleGenerateDocument}
              onSaveKnowledge={handleSaveKnowledge}
              savingKnowledge={savingKnowledge}
              generationInfo={props.generationInfo}
              isLastMessage={isLastMessage}
              hideEditAndRegenerate={props.hideEditAndRegenerate}
              onRegenerate={props.onRegenerate}
              onNewBranch={props.onNewBranch}
              temporaryChat={props.temporaryChat}
              hideContinue={props.hideContinue}
              onContinue={props.onContinue}
              onRunSteeredContinue={props.onRunSteeredContinue}
              messageSteeringMode={props.messageSteeringMode}
              onMessageSteeringModeChange={props.onMessageSteeringModeChange}
              messageSteeringForceNarrate={props.messageSteeringForceNarrate}
              onMessageSteeringForceNarrateChange={
                props.onMessageSteeringForceNarrateChange
              }
              onClearMessageSteering={props.onClearMessageSteering}
              onEdit={() => setEditMode(true)}
              editMode={editMode}
              showFeedbackControls={showFeedbackControls}
              feedbackSelected={thumb}
              feedbackDisabled={feedbackDisabled}
              feedbackDisabledReason={feedbackDisabledReason}
              isFeedbackSubmitting={isFeedbackSubmitting}
              showThanks={showThanks}
              onThumbUp={handleThumbUp}
              onThumbDown={handleThumbDown}
              onOpenDetails={handleOpenDetails}
              onDelete={props.onDeleteMessage ? handleDelete : undefined}
              canPin={Boolean(props.serverMessageId)}
              isPinned={Boolean(props.pinned)}
              onTogglePinned={props.onTogglePinned}
              onQuickMessageAction={
                props.isBot ? handleQuickMessageAction : undefined
              }
            />
          )}

          {/* uploaded documents if available */}
          {/* {props.documents && props.documents.length > 0 && (
            <div className="mt-3">
              <div className="flex flex-wrap gap-2">
                {props.documents.map((doc, index) => (
                  <div
                    key={index}
                    className="inline-flex items-center gap-2 px-3 py-2 bg-primary/10 text-primary rounded-lg text-sm border border-primary/30">
                    <FileIcon className="h-4 w-4" />
                    <div className="flex flex-col">
                      <span className="font-medium">{doc.filename || "Unknown file"}</span>
                      {doc.fileSize && (
                        <span className="text-xs opacity-70">
                          {(doc.fileSize / 1024).toFixed(1)} KB
                          {doc.processed !== undefined && (
                            <span className="ml-2">
                              {doc.processed ? "✓ Processed" : "⚠ Processing..."}
                            </span>
                          )}
                        </span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )} */}

          {/* Tool calls (for assistant messages with function calls) */}
          {props.isBot && props.toolCalls && props.toolCalls.length > 0 && (
            <ToolCallBlock
              toolCalls={props.toolCalls}
              results={props.toolResults}
            />
          )}

          {/* Disco Skills annotation */}
          {props.isBot && skillComment && (
            <DiscoSkillAnnotation
              comment={skillComment}
              onDismiss={() => setSkillComment(null)}
            />
          )}

          {props.isBot && props?.sources && props?.sources.length > 0 && (
            <Collapse
              className="mt-6"
              ghost
              onChange={(activeKey) => {
                const opened = Array.isArray(activeKey)
                  ? activeKey.length > 0
                  : Boolean(activeKey)
                if (opened) {
                  trackSourcesExpanded()
                }
              }}
              items={[
                {
                  key: "1",
                  label: (
                    <div className="italic text-text-muted">
                      {t("citations")}
                    </div>
                  ),
                  children: (
                    <div className="mb-3 flex flex-col gap-2">
                      <div className="flex flex-wrap items-center justify-between gap-2 rounded-md border border-border bg-surface2 px-2 py-1 text-[11px] text-text-muted">
                        <span>
                          {t(
                            "playground:sources.citationWorkflowHint",
                            "Inspect source rationale, then seed a follow-up from selected citations."
                          )}
                        </span>
                        <div className="flex items-center gap-2">
                          <button
                            type="button"
                            onClick={() => handleAskWithSources(props.sources || [])}
                            className="rounded border border-border bg-surface px-2 py-0.5 text-[10px] font-medium text-text-subtle hover:bg-surface2 hover:text-text"
                          >
                            {t(
                              "playground:sources.askWithSources",
                              "Ask with these sources"
                            )}
                          </button>
                          <button
                            type="button"
                            onClick={handleOpenKnowledgePanel}
                            className="rounded border border-border bg-surface px-2 py-0.5 text-[10px] font-medium text-text-subtle hover:bg-surface2 hover:text-text"
                          >
                            {t(
                              "playground:sources.openKnowledgePanel",
                              "Open Search & Context"
                            )}
                          </button>
                        </div>
                      </div>
                      {props?.sources?.map((source, index) => {
                        const sourceKey = getSourceFeedbackKey(source, index)
                        const selected =
                          sourceFeedback?.[sourceKey]?.thumb ?? null
                        const pinnedState = resolveSourcePinnedState(source)
                        return (
                          <SourceFeedback
                            key={sourceKey}
                            source={source}
                            sourceKey={sourceKey}
                            sourceIndex={index}
                            pinnedState={pinnedState}
                            selected={selected}
                            disabled={feedbackDisabled || isFeedbackSubmitting}
                            onRate={(key, payload, thumb) =>
                              submitSourceThumb({
                                sourceKey: key,
                                source: payload,
                                thumb
                              })
                            }
                            onAskWithSource={(payload) =>
                              handleAskWithSources([payload])
                            }
                            onOpenKnowledgePanel={handleOpenKnowledgePanel}
                            onSourceClick={props.onSourceClick}
                            onTrackClick={trackSourceClick}
                            onTrackCitation={trackCitationUsed}
                            onTrackDwell={(
                              sourcePayload,
                              dwellMs,
                              sourceIndex
                            ) =>
                              trackDwellTime(dwellMs, sourcePayload, sourceIndex)
                            }
                          />
                        )
                      })}
                    </div>
                  )
                }
              ]}
            />
          )}
          </div>
        </div>
        {portraitSide === "right" ? avatarColumn : null}
        {portraitSide === "right" ? portraitPanel : null}
      </div>
      {isAvatarPreviewOpen && portraitImage && (
        <Modal
          open
          onCancel={() => setIsAvatarPreviewOpen(false)}
          footer={null}
          centered
        >
          <Image
            src={portraitImage}
            alt={
              props.isBot
                ? resolvedModelName || props.name
                : userDisplayName.trim() || t("common:you", "You")
            }
            preview={false}
            className="w-full"
          />
        </Modal>
      )}
      {/* </div> */}
      {showFeedbackControls && (
        <FeedbackModal
          open={isFeedbackOpen}
          onClose={() => setIsFeedbackOpen(false)}
          onSubmit={submitDetail}
          isSubmitting={isFeedbackSubmitting}
          initialRating={detail?.rating ?? null}
          initialIssues={detail?.issues ?? []}
          initialNotes={detail?.notes ?? ""}
        />
      )}
      {streamingComplete && (
        <span aria-live="polite" className="sr-only">
          {t("playground:message.responseComplete", { defaultValue: "Response complete" })}
        </span>
      )}
    </article>
  )
}
