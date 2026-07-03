import React from "react"
import { useTranslation } from "react-i18next"
import { useStorage } from "@plasmohq/storage/hook"
import { useChatMoodBadgePreference } from "@/hooks/useChatMoodBadgePreference"
import { useTTS, type TtsClipMeta } from "@/hooks/useTTS"
import { useTldwAudioStatus } from "@/hooks/useTldwAudioStatus"
import { useFeedback } from "@/hooks/useFeedback"
import { useImplicitFeedback } from "@/hooks/useImplicitFeedback"
import { useServerCapabilities } from "@/hooks/useServerCapabilities"
import { useDiscoSkills } from "@/hooks/useDiscoSkills"
import { useUiModeStore } from "@/store/ui-mode"
import { useStoreMessageOption } from "@/store/option"
import { useStoreChatModelSettings } from "@/store/model"
import { useStoreMessage } from "@/store"
import type { State as MessageOptionState } from "@/store/option/types"
import {
  decodeChatErrorPayload,
  type ChatErrorPayload
} from "@/utils/chat-error-message"
import { buildChatTextClass } from "@/utils/chat-style"
import {
  resolveAvatarColumnAlignment,
  resolveMessageRenderSide
} from "./message-layout"
import { formatCost } from "@/utils/model-pricing"
import {
  resolveMessageCostUsd,
  resolveMessageUsage
} from "./message-usage"
import { resolveFallbackAudit } from "./routing-fallback-audit"
import {
  IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE,
  resolveImageGenerationMetadata
} from "@/utils/image-generation-chat"
import {
  detectCharacterMood,
  normalizeCharacterMoodLabel,
  resolveCharacterBaseAvatarUrl,
  resolveCharacterMoodImageUrl
} from "@/utils/character-mood"
import type { PlaygroundMessageProps } from "./message-types"
import {
  DEFAULT_TLDW_TTS_MODEL,
  DEFAULT_TTS_PROVIDER
} from "@/services/tts"

export type MessageStateProps = PlaygroundMessageProps & { sources?: any[] }

export interface MessageState {
  // Translation
  t: ReturnType<typeof useTranslation>["t"]

  // Storage preferences
  checkWideMode: boolean
  isUserChatBubble: boolean
  autoCopyResponseToClipboard: boolean
  autoPlayTTS: boolean
  copyAsFormattedText: boolean
  userTextColor: string
  assistantTextColor: string
  userTextFont: string
  assistantTextFont: string
  userTextSize: string
  assistantTextSize: string
  userDisplayName: string
  showCharacterPortraits: boolean
  showMoodBadge: boolean
  showMoodConfidence: boolean
  userPersonaImage: string
  ttsProvider: string

  // Server capabilities / UI mode
  capabilities: ReturnType<typeof useServerCapabilities>["capabilities"]
  uiMode: string
  isProMode: boolean

  // TTS
  cancel: ReturnType<typeof useTTS>["cancel"]
  isSpeaking: ReturnType<typeof useTTS>["isSpeaking"]
  speak: ReturnType<typeof useTTS>["speak"]
  ttsActionDisabled: boolean
  ttsDisabledReason: string | null
  audioHealthState: ReturnType<typeof useTldwAudioStatus>["healthState"]
  voicesAvailable: ReturnType<typeof useTldwAudioStatus>["voicesAvailable"]

  // Explicit feedback
  thumb: ReturnType<typeof useFeedback>["thumb"]
  detail: ReturnType<typeof useFeedback>["detail"]
  sourceFeedback: ReturnType<typeof useFeedback>["sourceFeedback"]
  canSubmit: ReturnType<typeof useFeedback>["canSubmit"]
  isFeedbackSubmitting: boolean
  showThanks: ReturnType<typeof useFeedback>["showThanks"]
  submitThumb: ReturnType<typeof useFeedback>["submitThumb"]
  submitDetail: ReturnType<typeof useFeedback>["submitDetail"]
  submitSourceThumb: ReturnType<typeof useFeedback>["submitSourceThumb"]
  feedbackExplicitAvailable: boolean
  feedbackImplicitAvailable: boolean

  // Implicit feedback
  trackCopy: ReturnType<typeof useImplicitFeedback>["trackCopy"]
  trackSourcesExpanded: ReturnType<typeof useImplicitFeedback>["trackSourcesExpanded"]
  trackSourceClick: ReturnType<typeof useImplicitFeedback>["trackSourceClick"]
  trackCitationUsed: ReturnType<typeof useImplicitFeedback>["trackCitationUsed"]
  trackDwellTime: ReturnType<typeof useImplicitFeedback>["trackDwellTime"]

  // Error state
  errorPayload: ChatErrorPayload | null
  errorFriendlyText: string | null

  // Usage / cost
  messageUsage: ReturnType<typeof resolveMessageUsage>
  messageCostUsd: number | null
  showUsageMetadata: boolean

  // Generation / interruption info
  interruptedGeneration: boolean
  interruptionReason: string | null
  streamTransportInterrupted: boolean
  partialResponseSaved: boolean
  streamTransportInterruptionReason: string | null
  showPartialSaveMarker: boolean

  // Timing
  messageTimestamp: string | null

  // Routing / fallback audit
  fallbackAudit: ReturnType<typeof resolveFallbackAudit>
  fallbackAuditPolicyLabel: string | null
  fallbackAuditPathLabel: string | null

  // Image generation
  imageGenerationMetadata: ReturnType<typeof resolveImageGenerationMetadata>
  canRegenerateImage: boolean
  showInlineImageActions: boolean
  isImageGenerationAssistantEvent: boolean
  imageGenerationEventSummary: {
    prompt: string | undefined
    chips: string[]
    sourceLabel: string | null
    refineLabel: string | null
    syncLabel: string | null
    syncStatus: string | null
  } | null

  // Variants
  variantCount: number
  resolvedVariantIndex: number
  imageVariantEntries: Array<{ index: number; preview: string; images: string[] }>
  activeVariantPreview: { index: number; preview: string; images: string[] } | null
  showVariantPager: boolean
  canSwipePrev: boolean
  canSwipeNext: boolean

  // Character / mood
  shouldUseCharacterIdentity: boolean
  resolvedMoodLabel: string | null
  moodBadgeLabel: string | null
  characterAvatar: string
  resolvedModelImage: string | undefined
  resolvedModelName: string | undefined

  // Display / layout
  resolvedUserPersonaImage: string
  portraitImage: string
  messageRenderSide: "left" | "right"
  shouldShowPortrait: boolean
  shouldShowAvatarColumn: boolean
  avatarColumnAlignmentClass: string
  shouldPreviewAvatar: boolean

  // Text classes
  userTextClass: string
  assistantTextClass: string
  chatTextClass: string
  renderGreetingMarkdown: boolean
  shouldRenderStreamingPlainText: boolean

  // Loading state
  shouldShowLoadingStatus: boolean
  isActiveResponse: boolean

  // Feedback UI
  feedbackDisabled: boolean
  showFeedbackControls: boolean
  feedbackDisabledReason: string

  // TTS clip meta
  ttsClipMeta: TtsClipMeta

  // Message identification
  messageKey: string
  isLastMessage: boolean
  resolvedRole: "user" | "assistant" | "system"
  isSystemMessage: boolean

  // Knowledge / notes actions availability
  canSaveKnowledge: boolean
  canSaveToNotes: boolean
  canSaveToFlashcards: boolean
  canSaveToWorkspaceNotes: boolean
  canGenerateDocument: boolean

  // Reply
  replyId: string | null
  canReply: boolean

  // Disco Skills
  discoSkillsEnabled: boolean
  discoSkillsStats: ReturnType<typeof useDiscoSkills>["stats"]
  discoTriggerProbability: ReturnType<typeof useDiscoSkills>["triggerProbabilityBase"]
  discoPersistComments: ReturnType<typeof useDiscoSkills>["persistComments"]
  selectedModel: string | null

  // Store setters / slices
  setReplyTarget: MessageOptionState["setReplyTarget"]
  ragPinnedResults: MessageOptionState["ragPinnedResults"]
  setMessages: MessageOptionState["setMessages"]
  apiProviderOverride: string | undefined
  updateChatModelSetting: ReturnType<typeof useStoreChatModelSettings>
}

/**
 * useMessageState
 *
 * Extracts all hook calls and derived state from the PlaygroundMessage component
 * so the rendering logic can stay clean in Message.tsx.
 *
 * Does NOT include callbacks that own local React state (setIsBtnPressed, etc.),
 * does NOT include JSX, and does NOT include local-only state (editMode,
 * isFeedbackOpen, etc.). Those remain in the component.
 */
export function useMessageState(props: MessageStateProps) {
  // ── Storage ──────────────────────────────────────────────────────────────
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

  // ── Translation ───────────────────────────────────────────────────────────
  const { t } = useTranslation(["common", "playground"])

  // ── Server / UI mode ─────────────────────────────────────────────────────
  const { capabilities } = useServerCapabilities()
  const uiMode = useUiModeStore((state) => state.mode)
  const isProMode = uiMode === "pro"

  // ── Store selectors ───────────────────────────────────────────────────────
  const setReplyTarget = useStoreMessageOption((state) => state.setReplyTarget)
  const ragPinnedResults = useStoreMessageOption((state) => state.ragPinnedResults)
  const setMessages = useStoreMessageOption((state) => state.setMessages)
  const apiProviderOverride = useStoreChatModelSettings(
    (state) => state.apiProvider
  )
  const updateChatModelSetting = useStoreChatModelSettings(
    (state) => state.updateSetting
  )
  const selectedModel = useStoreMessage((state) => state.selectedModel)

  // ── TTS ───────────────────────────────────────────────────────────────────
  const { cancel, isSpeaking, speak } = useTTS()
  const { healthState: audioHealthState, voicesAvailable } = useTldwAudioStatus({
    requireVoices: ttsProvider === "tldw",
    tldwTtsModel
  })

  // ── Disco Skills ──────────────────────────────────────────────────────────
  const {
    enabled: discoSkillsEnabled,
    stats: discoSkillsStats,
    triggerProbabilityBase: discoTriggerProbability,
    persistComments: discoPersistComments
  } = useDiscoSkills()

  // ── Derived: message identification ───────────────────────────────────────
  const isLastMessage: boolean =
    props.currentMessageIndex === props.totalMessages - 1

  const resolvedRole = props.role ?? (props.isBot ? "assistant" : "user")
  const isSystemMessage = resolvedRole === "system"

  const messageKey = React.useMemo(() => {
    if (props.serverMessageId) return `srv:${props.serverMessageId}`
    if (props.messageId) return `local:${props.messageId}`
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

  // ── Derived: error state ──────────────────────────────────────────────────
  const errorPayload = decodeChatErrorPayload(props.message)
  const errorFriendlyText = React.useMemo(() => {
    if (!errorPayload) return null
    return [errorPayload.summary, errorPayload.hint, errorPayload.detail]
      .filter(Boolean)
      .join("\n")
  }, [errorPayload])

  // ── Derived: usage / cost ─────────────────────────────────────────────────
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

  // ── Derived: generation / interruption ────────────────────────────────────
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

  // ── Derived: timestamp ────────────────────────────────────────────────────
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

  // ── Derived: fallback audit ───────────────────────────────────────────────
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

  // ── Derived: image generation ─────────────────────────────────────────────
  const imageGenerationMetadata = React.useMemo(
    () => resolveImageGenerationMetadata(props.generationInfo),
    [props.generationInfo]
  )
  const canRegenerateImage =
    props.isBot &&
    Boolean(props.onRegenerateImage) &&
    Boolean(imageGenerationMetadata?.request)
  const showInlineImageActions = canRegenerateImage || Boolean(props.onDeleteImage)
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
        ? String(t("playground:imageGeneration.eventSourceSlash", "Slash command"))
        : imageGenerationMetadata.source === "message-regen"
          ? String(t("playground:imageGeneration.eventSourceRegen", "Regenerated"))
          : imageGenerationMetadata.source === "generate-modal"
            ? String(t("playground:imageGeneration.eventSourceModal", "Generate menu"))
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
          ? String(t("playground:imageGeneration.eventSyncOn", "Mirrored to server"))
          : imageGenerationMetadata.sync.status === "failed"
            ? String(t("playground:imageGeneration.eventSyncFailed", "Mirror failed"))
            : String(t("playground:imageGeneration.eventSyncPending", "Mirroring..."))
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

  // ── Derived: variants ─────────────────────────────────────────────────────
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
        ? props.images.filter(
            (image) => typeof image === "string" && image.length > 0
          )
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
          ? variant.images.filter(
              (image) => typeof image === "string" && image.length > 0
            )
          : []
        if (images.length === 0) return null
        return {
          index,
          preview: images[0],
          images
        }
      })
      .filter(
        (entry): entry is { index: number; preview: string; images: string[] } =>
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
  const showVariantPager = props.isBot && variantCount > 1
  const canSwipePrev =
    showVariantPager && Boolean(props.onSwipePrev) && resolvedVariantIndex > 0
  const canSwipeNext =
    showVariantPager &&
    Boolean(props.onSwipeNext) &&
    resolvedVariantIndex < variantCount - 1

  // ── Derived: character / mood ─────────────────────────────────────────────
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
      const percent = Math.max(
        0,
        Math.min(100, Math.round(props.moodConfidence * 100))
      )
      return t(
        "playground:message.moodLabelConfidence",
        "Mood: {{mood}} ({{confidence}}%)",
        { mood: normalizedMood, confidence: percent }
      )
    }
    return t("playground:message.moodLabel", "Mood: {{mood}}", {
      mood: normalizedMood
    })
  }, [props.moodConfidence, resolvedMoodLabel, showMoodConfidence, t])
  const baseCharacterAvatar = resolveCharacterBaseAvatarUrl(props.characterIdentity)
  const visualCharacterAvatar =
    shouldUseCharacterIdentity && typeof props.visualAssetUrl === "string"
      ? props.visualAssetUrl.trim()
      : ""
  const moodCharacterAvatar = shouldUseCharacterIdentity
    ? resolveCharacterMoodImageUrl(props.characterIdentity, resolvedMoodLabel)
    : ""
  const characterAvatar =
    visualCharacterAvatar || moodCharacterAvatar || baseCharacterAvatar
  const resolvedModelImage =
    shouldUseCharacterIdentity && characterAvatar
      ? characterAvatar
      : props.modelImage
  const resolvedModelName =
    shouldUseCharacterIdentity && props.characterIdentity?.name
      ? props.characterIdentity.name
      : props.modelName || props.name

  // ── Derived: display / layout ─────────────────────────────────────────────
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
  const shouldShowPortrait =
    Boolean(showCharacterPortraits) && Boolean(portraitImage)
  const shouldShowAvatarColumn = !shouldShowPortrait
  const avatarColumnAlignmentClass = resolveAvatarColumnAlignment(messageRenderSide)
  const shouldPreviewAvatar =
    shouldUseCharacterIdentity && Boolean(characterAvatar)

  // ── Derived: text classes ─────────────────────────────────────────────────
  const userTextClass = React.useMemo(
    () => buildChatTextClass(userTextColor, userTextFont, userTextSize),
    [userTextColor, userTextFont, userTextSize]
  )
  const assistantTextClass = React.useMemo(
    () => buildChatTextClass(assistantTextColor, assistantTextFont, assistantTextSize),
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

  // ── Derived: loading state ────────────────────────────────────────────────
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

  // ── Derived: TTS availability ─────────────────────────────────────────────
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

  // ── TTS clip meta ─────────────────────────────────────────────────────────
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

  // ── Feedback hooks ────────────────────────────────────────────────────────
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

  const {
    trackCopy,
    trackSourcesExpanded,
    trackSourceClick,
    trackCitationUsed,
    trackDwellTime
  } = useImplicitFeedback({
    conversationId: props.serverChatId ?? null,
    messageId: props.serverMessageId ?? null,
    query: props.feedbackQuery ?? null,
    sources: props.sources ?? [],
    enabled: feedbackImplicitAvailable
  })

  // ── Derived: feedback UI ──────────────────────────────────────────────────
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

  // ── Derived: knowledge / notes ────────────────────────────────────────────
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

  // ── Derived: reply ────────────────────────────────────────────────────────
  const replyId = props.messageId ?? props.serverMessageId ?? null
  const canReply =
    isProMode &&
    Boolean(replyId) &&
    !props.compareSelectable &&
    !props.message_type?.startsWith("compare")

  // ─────────────────────────────────────────────────────────────────────────
  return {
    // Translation
    t,

    // Storage preferences
    checkWideMode,
    isUserChatBubble,
    autoCopyResponseToClipboard,
    autoPlayTTS,
    copyAsFormattedText,
    userTextColor,
    assistantTextColor,
    userTextFont,
    assistantTextFont,
    userTextSize,
    assistantTextSize,
    userDisplayName,
    showCharacterPortraits,
    showMoodBadge,
    showMoodConfidence,
    userPersonaImage,
    ttsProvider,

    // Server capabilities / UI mode
    capabilities,
    uiMode,
    isProMode,

    // TTS
    cancel,
    isSpeaking,
    speak,
    ttsActionDisabled,
    ttsDisabledReason,
    audioHealthState,
    voicesAvailable,

    // Explicit feedback
    thumb,
    detail,
    sourceFeedback,
    canSubmit,
    isFeedbackSubmitting,
    showThanks,
    submitThumb,
    submitDetail,
    submitSourceThumb,
    feedbackExplicitAvailable,
    feedbackImplicitAvailable,

    // Implicit feedback
    trackCopy,
    trackSourcesExpanded,
    trackSourceClick,
    trackCitationUsed,
    trackDwellTime,

    // Error state
    errorPayload,
    errorFriendlyText,

    // Usage / cost
    messageUsage,
    messageCostUsd,
    showUsageMetadata,

    // Generation / interruption
    interruptedGeneration,
    interruptionReason,
    streamTransportInterrupted,
    partialResponseSaved,
    streamTransportInterruptionReason,
    showPartialSaveMarker,

    // Timing
    messageTimestamp,

    // Routing / fallback audit
    fallbackAudit,
    fallbackAuditPolicyLabel,
    fallbackAuditPathLabel,

    // Image generation
    imageGenerationMetadata,
    canRegenerateImage,
    showInlineImageActions,
    isImageGenerationAssistantEvent,
    imageGenerationEventSummary,

    // Variants
    variantCount,
    resolvedVariantIndex,
    imageVariantEntries,
    activeVariantPreview,
    showVariantPager,
    canSwipePrev,
    canSwipeNext,

    // Character / mood
    shouldUseCharacterIdentity,
    resolvedMoodLabel,
    moodBadgeLabel,
    characterAvatar,
    resolvedModelImage,
    resolvedModelName,

    // Display / layout
    resolvedUserPersonaImage,
    portraitImage,
    messageRenderSide,
    shouldShowPortrait,
    shouldShowAvatarColumn,
    avatarColumnAlignmentClass,
    shouldPreviewAvatar,

    // Text classes
    userTextClass,
    assistantTextClass,
    chatTextClass,
    renderGreetingMarkdown,
    shouldRenderStreamingPlainText,

    // Loading state
    shouldShowLoadingStatus,
    isActiveResponse,

    // Feedback UI
    feedbackDisabled,
    showFeedbackControls,
    feedbackDisabledReason,

    // TTS clip meta
    ttsClipMeta,

    // Message identification
    messageKey,
    isLastMessage,
    resolvedRole,
    isSystemMessage,

    // Knowledge / notes
    canSaveKnowledge,
    canSaveToNotes,
    canSaveToFlashcards,
    canSaveToWorkspaceNotes,
    canGenerateDocument,

    // Reply
    replyId,
    canReply,

    // Disco Skills
    discoSkillsEnabled,
    discoSkillsStats,
    discoTriggerProbability,
    discoPersistComments,
    selectedModel,

    // Store setters / slices
    setReplyTarget,
    ragPinnedResults,
    setMessages,
    apiProviderOverride,
    updateChatModelSetting,
  } as const
}
