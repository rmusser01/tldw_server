import React from "react"
import { useQuery } from "@tanstack/react-query"
import { Link } from "react-router-dom"
import { useMessageOption } from "@/hooks/useMessageOption"
import { useSelectedCharacter } from "@/hooks/useSelectedCharacter"
import { PlaygroundEmpty } from "./PlaygroundEmpty"
import {
  PlaygroundMessage,
  type MessageResearchActions
} from "@/components/Common/Playground/Message"
import { useStorage } from "@plasmohq/storage/hook"
import { useTranslation } from "react-i18next"
import { generateID, updateMessageMedia } from "@/db/dexie/helpers"
import { fetchChatModels, clearChatModelsCache } from "@/services/tldw-server"
import { useIsConnected } from "@/hooks/useConnectionState"
import { tldwClient, type ChatLinkedResearchRun } from "@/services/tldw/TldwApiClient"
import { NoProviderBanner } from "@/components/Common/NoProviderBanner"
import { applyVariantToMessage } from "@/utils/message-variants"
import {
  buildCharacterChatReadiness,
  CHARACTER_CHAT_MODEL_SETTINGS_PATH,
  getCharacterChatReadinessCopy
} from "@/utils/chat-model-availability"
import {
  getChatLinkedResearchActionPolicy,
  getChatLinkedResearchRefetchInterval,
  getChatReturnedResearchBannerState,
} from "./research-run-status"
import type { Character } from "@/types/character"
import { useAntdNotification } from "@/hooks/useAntdNotification"
import {
  deriveAttachedResearchContext,
  isDeepResearchCompletionMetadata,
  type AttachedResearchContext,
  type ResearchFollowUpTarget
} from "./research-chat-context"
import {
  IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE,
  IMAGE_GENERATION_USER_MESSAGE_TYPE,
  isImageGenerationMessageType,
  normalizeImageGenerationVariantBundle,
  type ImageGenerationRequestSnapshot
} from "@/utils/image-generation-chat"

type TimelineBlock =
  | { kind: "single"; index: number }
  | {
      kind: "compare"
      userIndex: number
      assistantIndices: number[]
      clusterId: string
    }

type TimelineMessageShape = {
  messageType?: string
  message_type?: string
  clusterId?: string
}

const resolveTimelineMessageType = (
  message: TimelineMessageShape
): string | undefined => message.messageType ?? message.message_type

const shouldHideTimelineMessage = (message: TimelineMessageShape): boolean =>
  resolveTimelineMessageType(message) === IMAGE_GENERATION_USER_MESSAGE_TYPE

const LazyPlaygroundCompareCluster = React.lazy(() =>
  import("./PlaygroundCompareCluster").then((module) => ({
    default: module.PlaygroundCompareCluster
  }))
)
const LazyResearchRunStatusStack = React.lazy(() =>
  import("./ResearchRunStatusStack").then((module) => ({
    default: module.ResearchRunStatusStack
  }))
)
const LazyChatGreetingPicker = React.lazy(() =>
  import("@/components/Common/ChatGreetingPicker").then((module) => ({
    default: module.ChatGreetingPicker
  }))
)

type PlaygroundChatProps = {
  showStarterDeck?: boolean
  searchQuery?: string
  matchedMessageIndices?: Set<number>
  activeSearchMessageIndex?: number | null
  onAttachResearchContext?: (context: AttachedResearchContext) => void
  onPrepareResearchFollowUp?: (target: ResearchFollowUpTarget) => void
  returnedResearchRunId?: string | null
  onDismissReturnedResearchRun?: () => void
}

const buildBlocks = (messages: TimelineMessageShape[]): TimelineBlock[] => {
  const blocks: TimelineBlock[] = []
  const used = new Set<number>()

  messages.forEach((msg, idx) => {
    if (used.has(idx)) return
    if (shouldHideTimelineMessage(msg)) {
      used.add(idx)
      return
    }
    const messageType = resolveTimelineMessageType(msg)

    if (messageType === "compare:user" && msg.clusterId) {
      const assistants: number[] = []
      messages.forEach((m, j) => {
        if (j === idx || used.has(j)) return
        if (shouldHideTimelineMessage(m)) {
          used.add(j)
          return
        }
        if (m.clusterId === msg.clusterId) {
          if (resolveTimelineMessageType(m) === "compare:reply") {
            assistants.push(j)
          }
          used.add(j)
        }
      })
      used.add(idx)
      blocks.push({
        kind: "compare",
        userIndex: idx,
        assistantIndices: assistants,
        clusterId: msg.clusterId
      })
    } else {
      blocks.push({ kind: "single", index: idx })
    }
  })

  return blocks
}

export const PlaygroundChat = ({
  showStarterDeck = true,
  searchQuery,
  matchedMessageIndices,
  activeSearchMessageIndex = null,
  onAttachResearchContext,
  onPrepareResearchFollowUp,
  returnedResearchRunId = null,
  onDismissReturnedResearchRun
}: PlaygroundChatProps) => {
  const { t } = useTranslation(["playground", "common"])
  const notification = useAntdNotification()
  const {
    messages,
    setMessages,
    streaming,
    isProcessing,
    regenerateLastMessage,
    isSearchingInternet,
    editMessage,
    deleteMessage,
    toggleMessagePinned,
    ttsEnabled,
    onSubmit,
    actionInfo,
    messageSteeringMode,
    setMessageSteeringMode,
    messageSteeringForceNarrate,
    setMessageSteeringForceNarrate,
    clearMessageSteering,
    createChatBranch,
    createCompareBranch,
    temporaryChat,
    serverChatId,
    serverChatCharacterId,
    serverChatLoadState,
    serverChatLoadError,
    stopStreamingRequest,
    isEmbedding,
    compareMode,
    compareFeatureEnabled,
    compareSelectionByCluster,
    setCompareSelectionForCluster,
    compareActiveModelsByCluster,
    setCompareActiveModelsForCluster,
    setCompareSelectedModels,
    historyId,
    setSelectedModel,
    setCompareMode,
    sendPerModelReply,
    compareCanonicalByCluster,
    setCompareCanonicalForCluster,
    compareContinuationModeByCluster,
    setCompareContinuationModeForCluster,
    setCompareParentForHistory,
    compareSplitChats,
    setCompareSplitChat,
    compareMaxModels
  } = useMessageOption()
  const [openReasoning] = useStorage("openReasoning", false)
  const [selectedCharacter] = useSelectedCharacter<Character | null>(null)
  const isConnected = useIsConnected()
  const { data: chatModels = [], isFetched: chatModelsFetched, refetch: refetchChatModels } = useQuery({
    queryKey: ["playground:chatModels"],
    queryFn: () => fetchChatModels({ returnEmpty: true }),
    enabled: isConnected,
    staleTime: 30_000,
  })
  const {
    data: providersStatus,
    refetch: refetchProvidersStatus,
  } = useQuery({
    queryKey: ["playground:providersStatus"],
    queryFn: async () => {
      await tldwClient.initialize().catch(() => null)
      return await tldwClient.getProvidersStatus()
    },
    enabled: isConnected,
    staleTime: 60_000,
    retry: false,
  })
  const noProvidersConfigured =
    providersStatus != null && providersStatus.any_configured === false
  const selectedCharacterName =
    selectedCharacter?.name ||
    (selectedCharacter as any)?.title ||
    (selectedCharacter as any)?.slug ||
    null
  const characterChatNoModelCopy = React.useMemo(() => {
    if (!selectedCharacter) return null
    const readiness = buildCharacterChatReadiness({
      isServerConnected: isConnected,
      selectedCharacter,
      selectedModel: null,
      availableModels: chatModels as any[]
    })
    return getCharacterChatReadinessCopy(readiness, t, {
      characterName: selectedCharacterName
    })
  }, [chatModels, isConnected, selectedCharacter, selectedCharacterName, t])
  const compareModeActive = compareFeatureEnabled && compareMode
  const stableHistoryId =
    temporaryChat || historyId === "temp" ? null : historyId
  const linkedResearchRunsEnabled = Boolean(serverChatId) && !temporaryChat
  const [conversationInstanceId, setConversationInstanceId] = React.useState(
    () => generateID()
  )
  const [linkedResearchRunErrorCount, setLinkedResearchRunErrorCount] = React.useState(0)
  const previousMessageCount = React.useRef(messages.length)
  const latestLinkedResearchSuccessAt = React.useRef(0)
  const latestLinkedResearchErrorAt = React.useRef(0)

  const linkedResearchRunsQuery = useQuery({
    queryKey: ["playground:chat-linked-research-runs", serverChatId],
    queryFn: async () => {
      if (!serverChatId) {
        return { runs: [] as ChatLinkedResearchRun[] }
      }
      await tldwClient.initialize().catch(() => null)
      return await tldwClient.listChatResearchRuns(serverChatId)
    },
    enabled: linkedResearchRunsEnabled,
    retry: false,
    refetchInterval: (query) => {
      const data = query.state.data as { runs?: ChatLinkedResearchRun[] } | undefined
      const runs = Array.isArray(data?.runs) ? data.runs : []
      return getChatLinkedResearchRefetchInterval(runs, linkedResearchRunErrorCount)
    }
  })

  React.useEffect(() => {
    const hasStableId = Boolean(serverChatId || stableHistoryId)
    if (
      !hasStableId &&
      messages.length === 0 &&
      previousMessageCount.current > 0
    ) {
      setConversationInstanceId(generateID())
    }
    previousMessageCount.current = messages.length
  }, [messages.length, serverChatId, stableHistoryId])

  React.useEffect(() => {
    if (!linkedResearchRunsEnabled) {
      latestLinkedResearchSuccessAt.current = 0
      latestLinkedResearchErrorAt.current = 0
      setLinkedResearchRunErrorCount(0)
    }
  }, [linkedResearchRunsEnabled])

  React.useEffect(() => {
    if (
      linkedResearchRunsQuery.isSuccess &&
      linkedResearchRunsQuery.dataUpdatedAt > 0 &&
      linkedResearchRunsQuery.dataUpdatedAt !== latestLinkedResearchSuccessAt.current
    ) {
      latestLinkedResearchSuccessAt.current = linkedResearchRunsQuery.dataUpdatedAt
      latestLinkedResearchErrorAt.current = 0
      setLinkedResearchRunErrorCount(0)
    }
  }, [linkedResearchRunsQuery.dataUpdatedAt, linkedResearchRunsQuery.isSuccess])

  React.useEffect(() => {
    if (
      linkedResearchRunsQuery.isError &&
      linkedResearchRunsQuery.errorUpdatedAt > 0 &&
      linkedResearchRunsQuery.errorUpdatedAt !== latestLinkedResearchErrorAt.current
    ) {
      latestLinkedResearchErrorAt.current = linkedResearchRunsQuery.errorUpdatedAt
      setLinkedResearchRunErrorCount((current) => current + 1)
    }
  }, [linkedResearchRunsQuery.errorUpdatedAt, linkedResearchRunsQuery.isError])
  const blocks = React.useMemo(() => buildBlocks(messages), [messages])
  const linkedResearchRuns = React.useMemo(() => {
    if (!linkedResearchRunsEnabled || !linkedResearchRunsQuery.isSuccess) {
      return []
    }
    return Array.isArray(linkedResearchRunsQuery.data?.runs)
      ? linkedResearchRunsQuery.data.runs
      : []
  }, [linkedResearchRunsEnabled, linkedResearchRunsQuery.data?.runs, linkedResearchRunsQuery.isSuccess])
  const returnedResearchRun = React.useMemo(
    () =>
      returnedResearchRunId
        ? linkedResearchRuns.find((run) => run.run_id === returnedResearchRunId) ?? null
        : null,
    [linkedResearchRuns, returnedResearchRunId]
  )
  const returnedResearchActionPolicy = React.useMemo(
    () =>
      returnedResearchRun
        ? getChatLinkedResearchActionPolicy(returnedResearchRun)
        : null,
    [returnedResearchRun]
  )
  const returnedResearchBannerState = React.useMemo(
    () =>
      returnedResearchRun
        ? getChatReturnedResearchBannerState({
            run: returnedResearchRun,
            explicitReturn: true
          })
        : null,
    [returnedResearchRun]
  )
  const handleAttachResearchRun = React.useCallback(
    async (runId: string, query: string) => {
      if (!onAttachResearchContext) {
        return
      }
      await tldwClient.initialize().catch(() => null)
      const bundle = await tldwClient.getResearchBundle(runId)
      onAttachResearchContext(
        deriveAttachedResearchContext(bundle, runId, query)
      )
    },
    [onAttachResearchContext]
  )
  const getMessageResearchHandoffState = React.useCallback(
    (metadataExtra?: Record<string, unknown>) => {
      const completion = isDeepResearchCompletionMetadata(
        metadataExtra?.deep_research_completion
      )
        ? metadataExtra.deep_research_completion
        : null
      if (!completion) {
        return null
      }
      const currentRun = linkedResearchRuns.find(
        (run) => run.run_id === completion.run_id
      ) ?? null
      const actionPolicy =
        currentRun !== null
          ? getChatLinkedResearchActionPolicy(currentRun)
          : null
      return {
        completion,
        actionPolicy,
        reviewReason: actionPolicy?.needsReview ? actionPolicy.reasonLabel : null,
        reviewHref: actionPolicy?.needsReview ? actionPolicy.researchHref : null
      }
    },
    [linkedResearchRuns]
  )
  const buildMessageResearchActions = React.useCallback(
    (metadataExtra?: Record<string, unknown>): MessageResearchActions | undefined => {
      const handoff = getMessageResearchHandoffState(metadataExtra)
      if (!handoff) {
        return undefined
      }
      const canUseInChat = handoff.actionPolicy?.canUseInChat !== false
      const canFollowUp =
        handoff.actionPolicy?.canFollowUp !== false &&
        Boolean(onPrepareResearchFollowUp)
      return {
        reasonLabel: handoff.reviewReason ?? undefined,
        primaryLink: handoff.reviewHref
          ? {
              href: handoff.reviewHref,
              label: "Review in Research"
            }
          : undefined,
        onUseInChat: canUseInChat
          ? () => {
              void handleAttachResearchRun(
                handoff.completion.run_id,
                handoff.completion.query
              )
            }
          : undefined,
        onFollowUp: canFollowUp
          ? () => {
              onPrepareResearchFollowUp?.({
                run_id: handoff.completion.run_id,
                query: handoff.completion.query
              })
            }
          : undefined
      }
    },
    [getMessageResearchHandoffState, handleAttachResearchRun, onPrepareResearchFollowUp]
  )
  const showSelectedServerChatLoadFailure =
    messages.length === 0 &&
    Boolean(serverChatId) &&
    serverChatLoadState === "failed"
  const selectedServerChatLoadFailureMessage =
    serverChatLoadError?.trim() ||
    (t(
      "playground:selectedServerChatLoadFailure",
      "Failed to load the selected conversation."
    ) as string)
  const showNoProvidersNotice = isConnected && noProvidersConfigured
  const showNoModelsNotice =
    isConnected &&
    !noProvidersConfigured &&
    chatModelsFetched &&
    chatModels.length === 0
  const showEmptyStarterRegion =
    messages.length === 0 &&
    serverChatLoadState !== "loading" &&
    (showStarterDeck || showNoProvidersNotice || showNoModelsNotice)
  const normalizedSearchQuery =
    typeof searchQuery === "string" ? searchQuery.trim() : ""
  const resolveSearchMatch = React.useCallback(
    (messageIndex: number): "active" | "match" | null => {
      if (!normalizedSearchQuery) return null
      if (!matchedMessageIndices?.has(messageIndex)) return null
      return activeSearchMessageIndex === messageIndex ? "active" : "match"
    },
    [activeSearchMessageIndex, matchedMessageIndices, normalizedSearchQuery]
  )
  const runContinue = React.useCallback(() => {
    void onSubmit({
      image: "",
      message: "",
      isContinue: true
    })
  }, [onSubmit])
  const runSteeredContinue = React.useCallback(
    (mode: "continue_as_user" | "impersonate_user") => {
      void onSubmit({
        image: "",
        message: "",
        isContinue: true,
        messageSteeringOverride: {
          mode,
          forceNarrate: messageSteeringForceNarrate
        },
        continueOutputTarget:
          mode === "impersonate_user" ? "composer_input" : "chat"
      })
    },
    [messageSteeringForceNarrate, onSubmit]
  )
  const handleRegenerateGeneratedImage = React.useCallback(
    async (payload: {
      messageId?: string
      request: ImageGenerationRequestSnapshot | null
    }) => {
      const request = payload.request
      if (!request?.prompt || !request?.backend) {
        notification.warning({
          message: t("warning", { defaultValue: "Warning" }),
          description: t(
            "playground:imageGeneration.regenUnavailable",
            "Original image prompt metadata is unavailable for regeneration."
          )
        })
        return
      }
      const regenerateFromMessage =
        payload.messageId && payload.messageId.length > 0
          ? messages.find((entry) => entry.id === payload.messageId)
          : undefined
      const nextMessages =
        regenerateFromMessage?.id && messages.length > 0
          ? messages.filter((entry) => entry.id !== regenerateFromMessage.id)
          : messages

      if (regenerateFromMessage?.id) {
        setMessages(nextMessages)
      }

      await onSubmit({
        message: request.prompt,
        image: "",
        docs: [],
        isRegenerate: Boolean(regenerateFromMessage),
        regenerateFromMessage,
        messages: nextMessages,
        imageBackendOverride: request.backend,
        userMessageType: IMAGE_GENERATION_USER_MESSAGE_TYPE,
        assistantMessageType: IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE,
        imageGenerationRequest: request,
        imageGenerationSource: "message-regen"
      })
    },
    [messages, notification, onSubmit, setMessages, t]
  )
  const normalizeImageVariantState = React.useCallback(
    (
      entry: any,
      variants: any[],
      activeVariantIndex: number,
      options?: {
        hasVisibleVariant?: boolean
        generationInfo?: unknown
      }
    ) => {
      const normalized = normalizeImageGenerationVariantBundle({
        messageId: entry.id,
        messageGenerationInfo:
          options?.generationInfo !== undefined
            ? options.generationInfo
            : entry.generationInfo,
        variants,
        activeVariantIndex,
        fallbackCreatedAt: Date.now(),
        hasVisibleVariant: options?.hasVisibleVariant ?? true
      })

      if (variants.length === 0) {
        return {
          ...entry,
          variants: [],
          activeVariantIndex: normalized.activeVariantIndex,
          generationInfo: normalized.generationInfo ?? entry.generationInfo
        }
      }

      const activeVariant = normalized.variants[normalized.activeVariantIndex]
      if (!activeVariant) {
        return {
          ...entry,
          variants: normalized.variants,
          activeVariantIndex: normalized.activeVariantIndex,
          generationInfo: normalized.generationInfo ?? entry.generationInfo
        }
      }

      return applyVariantToMessage(
        {
          ...entry,
          variants: normalized.variants,
          activeVariantIndex: normalized.activeVariantIndex,
          generationInfo: normalized.generationInfo ?? entry.generationInfo
        },
        activeVariant,
        normalized.activeVariantIndex
      )
    },
    []
  )
  const handleDeleteGeneratedImage = React.useCallback(
    (payload: { messageId?: string; imageIndex: number }) => {
      if (!payload.messageId) return
      let nextImages: string[] | null = null
      let nextGenerationInfo: unknown = undefined
      setMessages((prev) =>
        prev.map((entry) => {
          if (entry.id !== payload.messageId) return entry
          const variants = Array.isArray(entry.variants) ? entry.variants : []
          if (
            variants.length > 0 &&
            typeof entry.activeVariantIndex === "number"
          ) {
            const activeVariantIndex = Math.max(
              0,
              Math.min(entry.activeVariantIndex, variants.length - 1)
            )
            const currentVariant = variants[activeVariantIndex]
            const currentVariantImages = Array.isArray(currentVariant?.images)
              ? currentVariant.images
              : []
            const remainingVariantImages = currentVariantImages.filter(
              (_, idx) => idx !== payload.imageIndex
            )

            if (remainingVariantImages.length > 0) {
              const nextVariants = [...variants]
              nextVariants[activeVariantIndex] = {
                ...currentVariant,
                images: remainingVariantImages
              }
              const updatedEntry = normalizeImageVariantState(
                {
                  ...entry,
                  variants: nextVariants,
                  activeVariantIndex
                },
                nextVariants,
                activeVariantIndex
              )
              nextImages = Array.isArray(updatedEntry.images)
                ? updatedEntry.images
                : []
              nextGenerationInfo = updatedEntry.generationInfo
              return updatedEntry
            }

            const nextVariants = variants.filter(
              (_, idx) => idx !== activeVariantIndex
            )
            if (nextVariants.length > 0) {
              const nextActiveIndex = Math.max(
                0,
                Math.min(activeVariantIndex, nextVariants.length - 1)
              )
              const updatedEntry = normalizeImageVariantState(
                {
                  ...entry,
                  variants: nextVariants,
                  activeVariantIndex: nextActiveIndex
                },
                nextVariants,
                nextActiveIndex
              )
              nextImages = Array.isArray(updatedEntry.images)
                ? updatedEntry.images
                : []
              nextGenerationInfo = updatedEntry.generationInfo
              return updatedEntry
            }

            nextImages = []
            const updatedEntry = normalizeImageVariantState(
              {
                ...entry,
                images: [],
                variants: [],
                activeVariantIndex: 0
              },
              [],
              0,
              { hasVisibleVariant: false }
            )
            nextGenerationInfo = updatedEntry.generationInfo
            return updatedEntry
          }
          const current = Array.isArray(entry.images) ? entry.images : []
          const remainingImages = current.filter((_, idx) => idx !== payload.imageIndex)
          const updatedEntry = normalizeImageVariantState(
            {
              ...entry,
              images: remainingImages
            },
            [],
            0,
            { hasVisibleVariant: remainingImages.length > 0 }
          )
          nextImages = Array.isArray(updatedEntry.images)
            ? updatedEntry.images
            : remainingImages
          nextGenerationInfo = updatedEntry.generationInfo
          return updatedEntry
        })
      )
      if (nextImages !== null && stableHistoryId) {
        const updates: { images?: string[]; generationInfo?: any } = {
          images: nextImages
        }
        if (nextGenerationInfo !== undefined) {
          updates.generationInfo = nextGenerationInfo
        }
        void updateMessageMedia(payload.messageId, updates).catch(() => null)
      }
    },
    [normalizeImageVariantState, setMessages, stableHistoryId]
  )
  const handleSelectGeneratedImageVariant = React.useCallback(
    (payload: { messageId?: string; variantIndex: number }) => {
      if (!payload.messageId) return
      let nextImages: string[] | null = null
      let nextGenerationInfo: unknown = undefined
      setMessages((prev) =>
        prev.map((entry) => {
          if (entry.id !== payload.messageId) return entry
          const variants = Array.isArray(entry.variants) ? entry.variants : []
          if (variants.length === 0) return entry
          if (
            payload.variantIndex < 0 ||
            payload.variantIndex >= variants.length
          ) {
            return entry
          }
          const updatedEntry = normalizeImageVariantState(
            {
              ...entry,
              variants,
              activeVariantIndex: payload.variantIndex
            },
            variants,
            payload.variantIndex
          )
          nextImages = Array.isArray(updatedEntry.images)
            ? updatedEntry.images
            : []
          nextGenerationInfo = updatedEntry.generationInfo
          return updatedEntry
        })
      )
      if (nextImages !== null && stableHistoryId) {
        const updates: { images?: string[]; generationInfo?: any } = {
          images: nextImages
        }
        if (nextGenerationInfo !== undefined) {
          updates.generationInfo = nextGenerationInfo
        }
        void updateMessageMedia(payload.messageId, updates).catch(() => null)
      }
    },
    [normalizeImageVariantState, setMessages, stableHistoryId]
  )
  const handleKeepGeneratedImageVariant = React.useCallback(
    (payload: { messageId?: string; variantIndex: number }) => {
      if (!payload.messageId) return
      let nextImages: string[] | null = null
      let nextGenerationInfo: unknown = undefined
      setMessages((prev) =>
        prev.map((entry) => {
          if (entry.id !== payload.messageId) return entry
          const variants = Array.isArray(entry.variants) ? entry.variants : []
          if (variants.length === 0) return entry
          const targetIndex = Math.max(
            0,
            Math.min(payload.variantIndex, variants.length - 1)
          )
          const targetVariant = variants[targetIndex]
          const nextVariants = [
            ...variants.filter((_, idx) => idx !== targetIndex),
            targetVariant
          ]
          const nextActiveIndex = nextVariants.length - 1
          const updatedEntry = normalizeImageVariantState(
            {
              ...entry,
              variants: nextVariants,
              activeVariantIndex: nextActiveIndex
            },
            nextVariants,
            nextActiveIndex
          )
          nextImages = Array.isArray(updatedEntry.images)
            ? updatedEntry.images
            : []
          nextGenerationInfo = updatedEntry.generationInfo
          return updatedEntry
        })
      )
      if (nextImages !== null && stableHistoryId) {
        const updates: { images?: string[]; generationInfo?: any } = {
          images: nextImages
        }
        if (nextGenerationInfo !== undefined) {
          updates.generationInfo = nextGenerationInfo
        }
        void updateMessageMedia(payload.messageId, updates).catch(() => null)
      }
    },
    [normalizeImageVariantState, setMessages, stableHistoryId]
  )
  const handleDeleteGeneratedImageVariant = React.useCallback(
    (payload: { messageId?: string; variantIndex: number }) => {
      if (!payload.messageId) return
      let nextImages: string[] | null = null
      let nextGenerationInfo: unknown = undefined
      setMessages((prev) =>
        prev.map((entry) => {
          if (entry.id !== payload.messageId) return entry
          const variants = Array.isArray(entry.variants) ? entry.variants : []
          if (variants.length === 0) return entry
          if (
            payload.variantIndex < 0 ||
            payload.variantIndex >= variants.length
          ) {
            return entry
          }
          const nextVariants = variants.filter(
            (_, idx) => idx !== payload.variantIndex
          )
          if (nextVariants.length === 0) {
            nextImages = []
            const updatedEntry = normalizeImageVariantState(
              {
                ...entry,
                images: [],
                variants: [],
                activeVariantIndex: 0
              },
              [],
              0,
              { hasVisibleVariant: false }
            )
            nextGenerationInfo = updatedEntry.generationInfo
            return updatedEntry
          }
          const nextActiveIndex = Math.max(
            0,
            Math.min(payload.variantIndex, nextVariants.length - 1)
          )
          const updatedEntry = normalizeImageVariantState(
            {
              ...entry,
              variants: nextVariants,
              activeVariantIndex: nextActiveIndex
            },
            nextVariants,
            nextActiveIndex
          )
          nextImages = Array.isArray(updatedEntry.images)
            ? updatedEntry.images
            : []
          nextGenerationInfo = updatedEntry.generationInfo
          return updatedEntry
        })
      )
      if (nextImages !== null && stableHistoryId) {
        const updates: { images?: string[]; generationInfo?: any } = {
          images: nextImages
        }
        if (nextGenerationInfo !== undefined) {
          updates.generationInfo = nextGenerationInfo
        }
        void updateMessageMedia(payload.messageId, updates).catch(() => null)
      }
    },
    [normalizeImageVariantState, setMessages, stableHistoryId]
  )
  const handleDeleteAllGeneratedImageVariants = React.useCallback(
    (payload: { messageId?: string }) => {
      if (!payload.messageId) return
      let nextGenerationInfo: unknown = undefined
      setMessages((prev) =>
        prev.map((entry) =>
          entry.id === payload.messageId
            ? (() => {
                const updatedEntry = normalizeImageVariantState(
                  {
                    ...entry,
                    images: [],
                    variants: [],
                    activeVariantIndex: 0
                  },
                  [],
                  0,
                  { hasVisibleVariant: false }
                )
                nextGenerationInfo = updatedEntry.generationInfo
                return updatedEntry
              })()
            : entry
        )
      )
      if (stableHistoryId) {
        const updates: { images?: string[]; generationInfo?: any } = {
          images: []
        }
        if (nextGenerationInfo !== undefined) {
          updates.generationInfo = nextGenerationInfo
        }
        void updateMessageMedia(payload.messageId, updates).catch(() => null)
      }
    },
    [normalizeImageVariantState, setMessages, stableHistoryId]
  )
  const selectedGreeting = React.useMemo(() => {
    if (!selectedCharacter || typeof selectedCharacter.greeting !== "string") {
      return ""
    }
    return selectedCharacter.greeting.trim()
  }, [selectedCharacter])
  const normalizeGreetingText = React.useCallback(
    (value: string) => value.replace(/\s+/g, " ").trim().toLowerCase(),
    []
  )
  const greetingNeedle = React.useMemo(() => {
    if (!selectedGreeting) return ""
    const normalized = normalizeGreetingText(selectedGreeting)
    if (!normalized) return ""
    return normalized.slice(0, 180)
  }, [normalizeGreetingText, selectedGreeting])
  const firstAssistantIndex = React.useMemo(
    () => messages.findIndex((msg) => msg?.role === "assistant" || msg?.isBot),
    [messages]
  )
  const firstUserIndex = React.useMemo(
    () =>
      messages.findIndex(
        (msg) => msg?.role === "user" || msg?.isBot === false
      ),
    [messages]
  )
  const hasSelectedCharacter = Boolean(selectedCharacter?.id)
  const characterIdentityEnabled = React.useMemo(() => {
    if (!selectedCharacter?.id) return false
    if (compareModeActive) return false
    if (serverChatId) {
      if (serverChatCharacterId == null) return false
      return String(serverChatCharacterId) === String(selectedCharacter.id)
    }
    return true
  }, [
    compareModeActive,
    selectedCharacter?.id,
    serverChatCharacterId,
    serverChatId
  ])
  const resolveMessageType = React.useCallback(
    (message: any, index: number) => {
      const explicit = message?.messageType ?? message?.message_type
      if (explicit) return explicit
      if (!serverChatId && hasSelectedCharacter) {
        const isFirstAssistant = index === firstAssistantIndex
        const hasNoUserBefore = firstUserIndex === -1 || firstUserIndex > index
        if (
          isFirstAssistant &&
          hasNoUserBefore &&
          message?.isBot &&
          typeof message?.message === "string"
        ) {
          const normalizedMessage = normalizeGreetingText(message.message)
          if (greetingNeedle && normalizedMessage.includes(greetingNeedle)) {
            return "character:greeting"
          }
        }
      }
      return undefined
    },
    [
      firstAssistantIndex,
      firstUserIndex,
      hasSelectedCharacter,
      greetingNeedle,
      normalizeGreetingText,
      serverChatId
    ]
  )
  const getPreviousUserMessage = React.useCallback(
    (index: number) => {
      for (let i = index - 1; i >= 0; i--) {
        const candidate = messages[i]
        if (
          !candidate?.isBot &&
          !isImageGenerationMessageType(resolveTimelineMessageType(candidate))
        ) {
          return candidate
        }
      }
      return null
    },
    [messages]
  )
  const modelMetaById = React.useMemo(() => {
    const map = new Map<string, { label: string; provider: string }>()
    const models = (chatModels as any[]) || []
    models.forEach((model) => {
      if (!model?.model) {
        return
      }
      map.set(model.model, {
        label: model.nickname || model.model,
        provider: String(model.provider || "custom").toLowerCase()
      })
    })
    return map
  }, [chatModels])
  const getTokenCount = React.useCallback((generationInfo?: any) => {
    if (!generationInfo || typeof generationInfo !== "object") {
      return null
    }
    const toNumber = (value: unknown) =>
      typeof value === "number" && Number.isFinite(value) ? value : null
    const usage = (generationInfo as any)?.usage
    const prompt =
      toNumber(generationInfo.prompt_eval_count) ??
      toNumber(generationInfo.prompt_tokens) ??
      toNumber(generationInfo.input_tokens) ??
      toNumber(usage?.prompt_tokens) ??
      toNumber(usage?.input_tokens)
    const completion =
      toNumber(generationInfo.eval_count) ??
      toNumber(generationInfo.completion_tokens) ??
      toNumber(generationInfo.output_tokens) ??
      toNumber(usage?.completion_tokens) ??
      toNumber(usage?.output_tokens)
    const total =
      toNumber(generationInfo.total_tokens) ??
      toNumber(generationInfo.total_token_count) ??
      toNumber(usage?.total_tokens)
    const resolvedTotal =
      total ?? (prompt != null && completion != null ? prompt + completion : null)
    if (resolvedTotal == null) {
      return null
    }
    return Math.round(resolvedTotal)
  }, [])

  const handleVariantSwipe = React.useCallback(
    (messageId: string | undefined, direction: "prev" | "next") => {
      if (!messageId) return
      setMessages((prev) =>
        prev.map((msg) => {
          if (msg.id !== messageId) return msg
          const variants = msg.variants ?? []
          if (variants.length < 2) return msg
          const currentIndex =
            typeof msg.activeVariantIndex === "number"
              ? msg.activeVariantIndex
              : variants.length - 1
          const nextIndex =
            direction === "prev" ? currentIndex - 1 : currentIndex + 1
          if (nextIndex < 0 || nextIndex >= variants.length) return msg
          return applyVariantToMessage(msg, variants[nextIndex], nextIndex)
        })
      )
    },
    [setMessages]
  )

  return (
    <>
      <div className="relative flex w-full flex-col items-center pt-8 pb-4">
        {showSelectedServerChatLoadFailure ? (
          <div className="mt-32 w-full px-6">
            <div className="mx-auto max-w-xl rounded-xl border border-destructive/30 bg-destructive/5 px-5 py-4 text-center text-sm text-text">
              {selectedServerChatLoadFailureMessage}
            </div>
          </div>
        ) : showEmptyStarterRegion && (
          <div className="mt-4 w-full">
            {showNoProvidersNotice && (
              <NoProviderBanner
                className="mb-4"
                onRefresh={() => {
                  clearChatModelsCache()
                  void refetchChatModels()
                  void refetchProvidersStatus()
                }}
              />
            )}
            {showNoModelsNotice && (
              <div className="mx-auto mb-4 max-w-xl rounded-xl border border-amber-500/30 bg-amber-500/5 px-5 py-4 text-center text-sm text-text">
                <p className="font-medium">
                  {characterChatNoModelCopy
                    ? characterChatNoModelCopy.title
                    : t("playground:noModelsAvailable", "No AI models available")}
                </p>
                <p className="mt-1 text-xs text-text-muted">
                  {characterChatNoModelCopy
                    ? characterChatNoModelCopy.description
                    : t("playground:addApiKeyInstructions", "Add an LLM provider API key in your server settings and restart, then")}{" "}
                  {characterChatNoModelCopy && (
                    <>
                      <Link
                        to={CHARACTER_CHAT_MODEL_SETTINGS_PATH}
                        className="underline hover:text-text"
                      >
                        {characterChatNoModelCopy.actionLabel}
                      </Link>{" "}
                      {t("common:or", "or")}{" "}
                    </>
                  )}
                  <button
                    type="button"
                    className="underline hover:text-text"
                    onClick={() => { clearChatModelsCache(); void refetchChatModels() }}
                  >
                    {t("playground:refreshModels", "refresh models")}
                  </button>.
                </p>
              </div>
            )}
            {showStarterDeck && <PlaygroundEmpty />}
          </div>
        )}
        <React.Suspense fallback={null}>
          <LazyChatGreetingPicker
            selectedCharacter={selectedCharacter}
            messages={messages}
            historyId={historyId}
            serverChatId={serverChatId}
            className="mb-6 mt-4"
          />
        </React.Suspense>
        {returnedResearchRun &&
        returnedResearchActionPolicy &&
        returnedResearchBannerState ? (
          <section
            className="mb-4 w-full max-w-5xl px-4"
            data-testid="returned-research-banner"
          >
            <div className="rounded-2xl border border-primary/20 bg-primary/5 px-4 py-3 shadow-sm">
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0 flex-1">
                  <div className="text-xs font-semibold uppercase tracking-[0.2em] text-text-subtle">
                    {t("playground:research.returnedBanner.label", "Returned from Research")}
                  </div>
                  <div className="mt-1 truncate text-sm font-medium text-text">
                    {returnedResearchRun.query}
                  </div>
                  {returnedResearchBannerState.supportingText ? (
                    <div className="mt-1 text-xs text-text-subtle">
                      {returnedResearchBannerState.supportingText}
                    </div>
                  ) : null}
                  {!returnedResearchBannerState.supportingText &&
                  returnedResearchActionPolicy.reasonLabel ? (
                    <div className="mt-1 text-xs text-text-subtle">
                      {returnedResearchActionPolicy.reasonLabel}
                    </div>
                  ) : null}
                </div>
                <div className="flex shrink-0 items-center gap-3">
                  {returnedResearchBannerState.mode === "completed" &&
                  returnedResearchActionPolicy.canUseInChat ? (
                    <button
                      type="button"
                      className="text-sm font-medium text-text hover:text-primary"
                      onClick={() => {
                        void handleAttachResearchRun(
                          returnedResearchRun.run_id,
                          returnedResearchRun.query
                        )
                      }}
                    >
                      {t("playground:research.useInChat", "Use in Chat")}
                    </button>
                  ) : null}
                  {returnedResearchBannerState.mode === "review_cleared" ? (
                    <button
                      type="button"
                      className="text-sm font-medium text-text hover:text-primary"
                      onClick={() => {
                        void handleAttachResearchRun(
                          returnedResearchRun.run_id,
                          returnedResearchRun.query
                        )
                      }}
                    >
                      {t(
                        "playground:research.continueReviewed",
                        "Continue with reviewed research"
                      )}
                    </button>
                  ) : null}
                  {returnedResearchBannerState.mode === "completed" &&
                  returnedResearchActionPolicy.canFollowUp &&
                  onPrepareResearchFollowUp ? (
                    <button
                      type="button"
                      className="text-sm font-medium text-text hover:text-primary"
                      onClick={() =>
                        onPrepareResearchFollowUp({
                          run_id: returnedResearchRun.run_id,
                          query: returnedResearchRun.query
                        })
                      }
                    >
                      {t("playground:research.followUp", "Follow up")}
                    </button>
                  ) : null}
                  <a
                    className="text-sm font-medium text-primary hover:underline"
                    href={returnedResearchActionPolicy.researchHref}
                  >
                    {returnedResearchBannerState.mode === "review_cleared"
                      ? t("playground:research.openInResearch", "Open in Research")
                      : returnedResearchActionPolicy.primaryActionLabel}
                  </a>
                  <button
                    type="button"
                    className="text-sm font-medium text-text-subtle hover:text-text"
                    onClick={() => onDismissReturnedResearchRun?.()}
                  >
                    {t("common:dismiss", "Dismiss")}
                  </button>
                </div>
              </div>
            </div>
          </section>
        ) : null}
        <React.Suspense fallback={null}>
          <LazyResearchRunStatusStack
            runs={linkedResearchRuns}
            onUseInChat={(run) => {
              void handleAttachResearchRun(run.run_id, run.query)
            }}
            onFollowUp={onPrepareResearchFollowUp}
          />
        </React.Suspense>
        {blocks.map((block, blockIndex) => {
          if (block.kind === "single") {
            const message = messages[block.index]
            const previousUserMessage = getPreviousUserMessage(block.index)
            const resolvedMessageType = resolveMessageType(message, block.index)
            const isImageGenerationAssistantEvent =
              resolvedMessageType === IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE
            return (
              <PlaygroundMessage
                key={`m-${blockIndex}`}
                isBot={message.isBot}
                message={message.message}
                name={message.name}
                role={message.role}
                images={message.images || []}
                currentMessageIndex={block.index}
                totalMessages={messages.length}
                onRegenerate={regenerateLastMessage}
                onRegenerateImage={(payload) => {
                  void handleRegenerateGeneratedImage(payload)
                }}
                onDeleteImage={handleDeleteGeneratedImage}
                onSelectImageVariant={handleSelectGeneratedImageVariant}
                onKeepImageVariant={handleKeepGeneratedImageVariant}
                onDeleteImageVariant={handleDeleteGeneratedImageVariant}
                onDeleteAllImageVariants={handleDeleteAllGeneratedImageVariants}
                isProcessing={isProcessing}
                isSearchingInternet={isSearchingInternet}
                sources={message.sources}
                onEditFormSubmit={(value, isSend) => {
                  editMessage(block.index, value, !message.isBot, isSend)
                }}
                onDeleteMessage={() => {
                  deleteMessage(block.index)
                }}
                onTogglePinned={() => {
                  void toggleMessagePinned(block.index)
                }}
                onNewBranch={() => {
                  createChatBranch(block.index)
                }}
                isTTSEnabled={ttsEnabled}
                generationInfo={message?.generationInfo}
                toolCalls={message?.toolCalls}
                toolResults={message?.toolResults}
                isStreaming={streaming}
                reasoningTimeTaken={message?.reasoning_time_taken}
                openReasoning={openReasoning}
                modelImage={message?.modelImage}
                modelName={message?.modelName}
                createdAt={message?.createdAt}
                temporaryChat={temporaryChat}
                onStopStreaming={stopStreamingRequest}
                onContinue={runContinue}
                onRunSteeredContinue={runSteeredContinue}
                documents={message?.documents}
                actionInfo={actionInfo}
                serverChatId={serverChatId}
                serverMessageId={message.serverMessageId}
                messageId={message.id}
                pinned={Boolean(message.pinned)}
                metadataExtra={message.metadataExtra}
                researchActions={buildMessageResearchActions(message.metadataExtra)}
                discoSkillComment={message.discoSkillComment}
                historyId={stableHistoryId ?? undefined}
                conversationInstanceId={conversationInstanceId}
                feedbackQuery={previousUserMessage?.message ?? null}
                isEmbedding={isEmbedding}
                characterIdentity={selectedCharacter}
                characterIdentityEnabled={characterIdentityEnabled}
                speakerCharacterId={message.speakerCharacterId ?? null}
                speakerCharacterName={message.speakerCharacterName}
                moodLabel={message.moodLabel ?? null}
                moodConfidence={message.moodConfidence ?? null}
                moodTopic={message.moodTopic ?? null}
                searchQuery={normalizedSearchQuery || undefined}
                searchMatch={resolveSearchMatch(block.index)}
                message_type={resolvedMessageType}
                variants={message.variants}
                activeVariantIndex={message.activeVariantIndex}
                onSwipePrev={() => handleVariantSwipe(message.id, "prev")}
                onSwipeNext={() => handleVariantSwipe(message.id, "next")}
                messageSteeringMode={messageSteeringMode}
                onMessageSteeringModeChange={setMessageSteeringMode}
                messageSteeringForceNarrate={messageSteeringForceNarrate}
                onMessageSteeringForceNarrateChange={setMessageSteeringForceNarrate}
                onClearMessageSteering={clearMessageSteering}
                hideEditAndRegenerate={isImageGenerationAssistantEvent}
                hideContinue={isImageGenerationAssistantEvent}
              />
            )
          }

          return (
            <React.Suspense
              key={`c-${block.clusterId}`}
              fallback={
                <div className="w-full max-w-5xl md:px-4 mb-4">
                  <div className="rounded-md border border-border bg-surface p-3 text-sm text-text-muted shadow-sm">
                    {t(
                      "playground:composer.compareLoading",
                      "Loading comparison…"
                    )}
                  </div>
                </div>
              }>
              <LazyPlaygroundCompareCluster
                block={block}
                blockIndex={blockIndex}
                messages={messages}
                openReasoning={openReasoning}
                isProcessing={isProcessing}
                isSearchingInternet={isSearchingInternet}
                ttsEnabled={ttsEnabled}
                streaming={streaming}
                temporaryChat={temporaryChat}
                serverChatId={serverChatId}
                actionInfo={actionInfo}
                isEmbedding={isEmbedding}
                selectedCharacter={selectedCharacter}
                characterIdentityEnabled={characterIdentityEnabled}
                normalizedSearchQuery={normalizedSearchQuery}
                historyId={historyId}
                stableHistoryId={stableHistoryId}
                conversationInstanceId={conversationInstanceId}
                messageSteeringMode={messageSteeringMode}
                messageSteeringForceNarrate={messageSteeringForceNarrate}
                compareFeatureEnabled={compareFeatureEnabled}
                compareModeActive={compareModeActive}
                compareSelectionByCluster={compareSelectionByCluster}
                compareActiveModelsByCluster={compareActiveModelsByCluster}
                compareCanonicalByCluster={compareCanonicalByCluster}
                compareContinuationModeByCluster={compareContinuationModeByCluster}
                compareSplitChats={compareSplitChats}
                compareMaxModels={compareMaxModels}
                modelMetaById={modelMetaById}
                getTokenCount={getTokenCount}
                getPreviousUserMessage={getPreviousUserMessage}
                resolveSearchMatch={resolveSearchMatch}
                resolveMessageType={resolveMessageType}
                regenerateLastMessage={regenerateLastMessage}
                handleRegenerateGeneratedImage={handleRegenerateGeneratedImage}
                handleDeleteGeneratedImage={handleDeleteGeneratedImage}
                handleSelectGeneratedImageVariant={handleSelectGeneratedImageVariant}
                handleKeepGeneratedImageVariant={handleKeepGeneratedImageVariant}
                handleDeleteGeneratedImageVariant={handleDeleteGeneratedImageVariant}
                handleDeleteAllGeneratedImageVariants={handleDeleteAllGeneratedImageVariants}
                editMessage={editMessage}
                deleteMessage={deleteMessage}
                toggleMessagePinned={toggleMessagePinned}
                createChatBranch={createChatBranch}
                stopStreamingRequest={stopStreamingRequest}
                runContinue={runContinue}
                runSteeredContinue={runSteeredContinue}
                buildMessageResearchActions={buildMessageResearchActions}
                handleVariantSwipe={handleVariantSwipe}
                setMessageSteeringMode={setMessageSteeringMode}
                setMessageSteeringForceNarrate={setMessageSteeringForceNarrate}
                clearMessageSteering={clearMessageSteering}
                setCompareSelectionForCluster={setCompareSelectionForCluster}
                setCompareActiveModelsForCluster={setCompareActiveModelsForCluster}
                setCompareSelectedModels={setCompareSelectedModels}
                setSelectedModel={setSelectedModel}
                setCompareMode={setCompareMode}
                sendPerModelReply={sendPerModelReply}
                setCompareCanonicalForCluster={setCompareCanonicalForCluster}
                setCompareContinuationModeForCluster={
                  setCompareContinuationModeForCluster
                }
                setCompareParentForHistory={setCompareParentForHistory}
                setCompareSplitChat={setCompareSplitChat}
                createCompareBranch={createCompareBranch}
              />
            </React.Suspense>
          )
        })}
      </div>
    </>
  )
}
