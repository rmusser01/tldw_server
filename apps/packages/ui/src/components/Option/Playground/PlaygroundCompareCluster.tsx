import React from "react"
import { useTranslation } from "react-i18next"
import { Clock, DollarSign, Hash } from "lucide-react"
import { PlaygroundMessage } from "@/components/Common/Playground/Message"
import { ProviderIcons } from "@/components/Common/ProviderIcon"
import { useAntdNotification } from "@/hooks/useAntdNotification"
import { decodeChatErrorPayload } from "@/utils/chat-error-message"
import { humanizeMilliseconds } from "@/utils/humanize-milliseconds"
import { resolveMessageCostUsd } from "@/components/Common/Playground/message-usage"
import { formatCost } from "@/utils/model-pricing"
import { tldwModels } from "@/services/tldw"
import { trackCompareMetric } from "@/utils/compare-metrics"
import {
  buildNormalizedPreview,
  computeNormalizedPreviewBudget
} from "./compare-normalized-preview"
import { computeResponseDiffPreview } from "./compare-response-diff"

type CompareBlock = {
  kind: "compare"
  userIndex: number
  assistantIndices: number[]
  clusterId: string
}

type PlaygroundCompareClusterProps = {
  block: CompareBlock
  blockIndex: number
  messages: any[]
  openReasoning: boolean
  isProcessing: boolean
  isSearchingInternet: boolean
  ttsEnabled: boolean
  streaming: boolean
  temporaryChat: boolean
  serverChatId: string | null
  actionInfo: any
  isEmbedding: boolean
  selectedCharacter: any
  characterIdentityEnabled: boolean
  normalizedSearchQuery: string
  historyId: string | null
  stableHistoryId: string | null
  conversationInstanceId: string
  messageSteeringMode: any
  messageSteeringForceNarrate: boolean
  compareFeatureEnabled: boolean
  compareModeActive: boolean
  compareSelectionByCluster: Record<string, string[]>
  compareActiveModelsByCluster: Record<string, string[]>
  compareCanonicalByCluster: Record<string, string | null | undefined>
  compareContinuationModeByCluster: Record<string, string | undefined>
  compareSplitChats: Record<string, Record<string, string>>
  compareMaxModels?: number
  modelMetaById: Map<string, { label: string; provider: string }>
  getTokenCount: (generationInfo?: any) => number | null
  getPreviousUserMessage: (index: number) => any
  resolveSearchMatch: (index: number) => "active" | "match" | null
  resolveMessageType: (message: any, index: number) => string | undefined
  regenerateLastMessage: (...args: any[]) => void
  handleRegenerateGeneratedImage: (payload: any) => Promise<void> | void
  handleDeleteGeneratedImage: (payload: any) => void
  handleSelectGeneratedImageVariant: (payload: any) => void
  handleKeepGeneratedImageVariant: (payload: any) => void
  handleDeleteGeneratedImageVariant: (payload: any) => void
  handleDeleteAllGeneratedImageVariants: (payload: any) => void
  editMessage: (index: number, value: string, isUser: boolean, isSend?: boolean) => void
  deleteMessage: (index: number) => void
  toggleMessagePinned: (index: number) => Promise<void> | void
  createChatBranch: (index: number) => void
  stopStreamingRequest: (...args: any[]) => void
  runContinue: () => void
  runSteeredContinue: (mode: "continue_as_user" | "impersonate_user") => void
  buildMessageResearchActions: (metadataExtra?: Record<string, unknown>) => any
  handleVariantSwipe: (messageId: string | undefined, direction: "prev" | "next") => void
  setMessageSteeringMode: (mode: any) => void
  setMessageSteeringForceNarrate: (next: boolean) => void
  clearMessageSteering: () => void
  setCompareSelectionForCluster: (clusterId: string, next: string[]) => void
  setCompareActiveModelsForCluster: (clusterId: string, next: string[]) => void
  setCompareSelectedModels: (next: string[]) => void
  setSelectedModel: (modelId: string) => void
  setCompareMode: (next: boolean) => void
  sendPerModelReply: (payload: {
    clusterId: string
    modelId: string
    message: string
  }) => Promise<void> | void
  setCompareCanonicalForCluster: (
    clusterId: string,
    messageId: string | null
  ) => void
  setCompareContinuationModeForCluster: (
    clusterId: string,
    mode: string
  ) => void
  setCompareParentForHistory: (
    historyId: string,
    payload: { parentHistoryId: string; clusterId: string }
  ) => void
  setCompareSplitChat: (
    clusterId: string,
    modelId: string,
    historyId: string
  ) => void
  createCompareBranch: (payload: {
    clusterId: string
    modelId: string
    open?: boolean
  }) => Promise<string | null | undefined>
}

const PerModelMiniComposer: React.FC<{
  placeholder: string
  disabled?: boolean
  helperText?: string | null
  onSend: (text: string) => Promise<void> | void
}> = ({ placeholder, disabled = false, helperText, onSend }) => {
  const { t } = useTranslation(["common"])
  const [value, setValue] = React.useState("")

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault()
    const trimmed = value.trim()
    if (!trimmed || disabled) {
      return
    }
    await onSend(trimmed)
    setValue("")
  }

  return (
    <div className="mt-2 space-y-1 text-[11px]">
      <form onSubmit={handleSubmit} className="flex items-center gap-2">
        <input
          className="flex-1 rounded border border-border bg-surface px-2 py-1 text-[11px] text-text placeholder:text-text-muted focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
          type="text"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder={placeholder}
          disabled={disabled}
        />
        <button
          type="submit"
          disabled={disabled || value.trim().length === 0}
          title={t("common:send", "Send") as string}
          className="rounded bg-primary px-2 py-1 text-[11px] font-medium text-surface disabled:cursor-not-allowed disabled:opacity-60 hover:bg-primaryStrong">
          {t("common:send", "Send")}
        </button>
      </form>
      {helperText ? (
        <div className="text-[10px] text-text-subtle">{helperText}</div>
      ) : null}
    </div>
  )
}

export const PlaygroundCompareCluster = ({
  block,
  blockIndex,
  messages,
  openReasoning,
  isProcessing,
  isSearchingInternet,
  ttsEnabled,
  streaming,
  temporaryChat,
  serverChatId,
  actionInfo,
  isEmbedding,
  selectedCharacter,
  characterIdentityEnabled,
  normalizedSearchQuery,
  historyId,
  stableHistoryId,
  conversationInstanceId,
  messageSteeringMode,
  messageSteeringForceNarrate,
  compareFeatureEnabled,
  compareModeActive,
  compareSelectionByCluster,
  compareActiveModelsByCluster,
  compareCanonicalByCluster,
  compareContinuationModeByCluster,
  compareSplitChats,
  compareMaxModels,
  modelMetaById,
  getTokenCount,
  getPreviousUserMessage,
  resolveSearchMatch,
  resolveMessageType,
  regenerateLastMessage,
  handleRegenerateGeneratedImage,
  handleDeleteGeneratedImage,
  handleSelectGeneratedImageVariant,
  handleKeepGeneratedImageVariant,
  handleDeleteGeneratedImageVariant,
  handleDeleteAllGeneratedImageVariants,
  editMessage,
  deleteMessage,
  toggleMessagePinned,
  createChatBranch,
  stopStreamingRequest,
  runContinue,
  runSteeredContinue,
  buildMessageResearchActions,
  handleVariantSwipe,
  setMessageSteeringMode,
  setMessageSteeringForceNarrate,
  clearMessageSteering,
  setCompareSelectionForCluster,
  setCompareActiveModelsForCluster,
  setCompareSelectedModels,
  setSelectedModel,
  setCompareMode,
  sendPerModelReply,
  setCompareCanonicalForCluster,
  setCompareContinuationModeForCluster,
  setCompareParentForHistory,
  setCompareSplitChat,
  createCompareBranch
}: PlaygroundCompareClusterProps) => {
  const { t } = useTranslation(["playground", "common"])
  const tr = React.useCallback(
    (key: string, fallback: string, options?: Record<string, unknown>) =>
      t(key, fallback, options as any) as string,
    [t]
  )
  const notification = useAntdNotification()
  const [collapsed, setCollapsed] = React.useState(true)
  const [hiddenModels, setHiddenModels] = React.useState<string[]>([])
  const [normalizedPreviewEnabled, setNormalizedPreviewEnabled] =
    React.useState(false)
  const [diffPreviewEnabled, setDiffPreviewEnabled] = React.useState(false)

  const userMessage = messages[block.userIndex]
  const previousUserMessage = getPreviousUserMessage(block.userIndex)
  const replyItems = block.assistantIndices.map((i) => {
    const message = messages[i]
    const modelKey = (message as any).modelId || message.modelName || message.name
    return {
      index: i,
      message,
      modelKey
    }
  })
  const clusterSelection = compareSelectionByCluster[block.clusterId] || []
  const clusterActiveModels =
    compareActiveModelsByCluster[block.clusterId] || clusterSelection
  const modelLabels = new Map<string, string>()
  replyItems.forEach(({ message, modelKey }) => {
    if (!modelLabels.has(modelKey)) {
      modelLabels.set(modelKey, message?.modelName || message.name || modelKey)
    }
  })
  const getModelLabel = (modelKey: string) =>
    modelMetaById.get(modelKey)?.label || modelLabels.get(modelKey) || modelKey
  const getModelProvider = (modelKey: string) =>
    modelMetaById.get(modelKey)?.provider || "custom"
  const clusterModelKeys = Array.from(new Set(replyItems.map((item) => item.modelKey)))
  const selectedModelKey = clusterSelection.length === 1 ? clusterSelection[0] : null
  const continuationMode =
    compareContinuationModeByCluster[block.clusterId] ||
    (clusterActiveModels.length > 1 ? "compare" : "winner")
  const isChosenState = !compareModeActive && !!selectedModelKey
  const filteredReplyItems =
    hiddenModels.length > 0
      ? replyItems.filter((item) => !hiddenModels.includes(item.modelKey))
      : replyItems
  const chosenItem = selectedModelKey
    ? replyItems.find((item) => item.modelKey === selectedModelKey) || null
    : null
  let visibleReplyItems =
    collapsed && isChosenState && selectedModelKey
      ? filteredReplyItems.filter((item) => item.modelKey === selectedModelKey)
      : filteredReplyItems
  if (visibleReplyItems.length === 0 && chosenItem) {
    visibleReplyItems = [chosenItem]
  }
  const alternativeCount = selectedModelKey
    ? Math.max(replyItems.length - 1, 0)
    : 0
  const diffBaselineModelKey =
    selectedModelKey ||
    clusterSelection[0] ||
    filteredReplyItems[0]?.modelKey ||
    null
  const diffBaselineItem = diffBaselineModelKey
    ? replyItems.find((item) => item.modelKey === diffBaselineModelKey) || null
    : null
  const diffBaselineText =
    typeof diffBaselineItem?.message?.message === "string"
      ? diffBaselineItem.message.message
      : ""

  const toggleModelFilter = (modelKey: string) => {
    setHiddenModels((current) => {
      const next = new Set(current)
      if (next.has(modelKey)) {
        next.delete(modelKey)
      } else {
        next.add(modelKey)
      }
      return Array.from(next)
    })
  }

  const clearModelFilter = () => {
    setHiddenModels([])
  }

  const handleContinueWithModel = (modelKey: string) => {
    setCompareMode(false)
    setSelectedModel(modelKey)
    setCompareSelectedModels([modelKey])
    setCompareActiveModelsForCluster(block.clusterId, [modelKey])
    setCompareContinuationModeForCluster(block.clusterId, "winner")
    setCollapsed(true)
    notification.success({
      message: tr(
        "playground:composer.compareContinueContract",
        "Next turns continue with {{model}} only. Re-enable Compare to send to multiple models again.",
        { model: getModelLabel(modelKey) }
      )
    })
  }

  const handleCompareAgain = () => {
    if (!compareFeatureEnabled) {
      return
    }
    const maxModels =
      typeof compareMaxModels === "number" && compareMaxModels > 0
        ? compareMaxModels
        : clusterModelKeys.length
    setCompareSelectedModels(clusterModelKeys.slice(0, maxModels))
    setCompareMode(true)
    setCompareContinuationModeForCluster(block.clusterId, "compare")
    setCollapsed(false)
    notification.info({
      message: t(
        "playground:composer.compareResumeContract",
        "Compare mode resumed. Next turn will fan out to selected models."
      )
    })
  }

  const handleBulkSplit = async () => {
    if (!compareFeatureEnabled) {
      return
    }
    const createdIds: string[] = []
    const failedModels: string[] = []
    for (const modelKey of clusterSelection) {
      try {
        const newHistoryId = await createCompareBranch({
          clusterId: block.clusterId,
          modelId: modelKey,
          open: false
        })
        if (newHistoryId && historyId) {
          setCompareParentForHistory(newHistoryId, {
            parentHistoryId: historyId,
            clusterId: block.clusterId
          })
          setCompareSplitChat(block.clusterId, modelKey, newHistoryId)
          createdIds.push(newHistoryId)
        } else {
          failedModels.push(modelKey)
        }
      } catch (error) {
        console.error(`Failed to create branch for ${modelKey}:`, error)
        failedModels.push(modelKey)
      }
    }
    if (createdIds.length > 0) {
      void trackCompareMetric({
        type: "split_bulk",
        count: createdIds.length
      })
      notification.success({
        message: t(
          "playground:composer.compareBulkSplitSuccess",
          "Created {{count}} chats",
          { count: createdIds.length }
        )
      })
    }
    if (failedModels.length > 0) {
      notification.warning({
        message: t(
          "playground:composer.compareBulkSplitPartialFail",
          "Failed to create {{count}} chats",
          { count: failedModels.length }
        )
      })
    }
  }

  return (
    <div className="w-full max-w-5xl md:px-4 mb-4 space-y-2">
      <PlaygroundMessage
        isBot={userMessage.isBot}
        message={userMessage.message}
        name={userMessage.name}
        role={userMessage.role}
        images={userMessage.images || []}
        currentMessageIndex={block.userIndex}
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
        sources={userMessage.sources}
        onEditFormSubmit={(value, isSend) => {
          editMessage(block.userIndex, value, !userMessage.isBot, isSend)
        }}
        onDeleteMessage={() => {
          deleteMessage(block.userIndex)
        }}
        onTogglePinned={() => {
          void toggleMessagePinned(block.userIndex)
        }}
        onNewBranch={() => {
          createChatBranch(block.userIndex)
        }}
        isTTSEnabled={ttsEnabled}
        generationInfo={userMessage?.generationInfo}
        toolCalls={userMessage?.toolCalls}
        toolResults={userMessage?.toolResults}
        isStreaming={streaming}
        reasoningTimeTaken={userMessage?.reasoning_time_taken}
        openReasoning={openReasoning}
        modelImage={userMessage?.modelImage}
        modelName={userMessage?.modelName}
        createdAt={userMessage?.createdAt}
        temporaryChat={temporaryChat}
        onStopStreaming={stopStreamingRequest}
        onContinue={runContinue}
        onRunSteeredContinue={runSteeredContinue}
        documents={userMessage?.documents}
        actionInfo={actionInfo}
        serverChatId={serverChatId}
        serverMessageId={userMessage.serverMessageId}
        messageId={userMessage.id}
        pinned={Boolean(userMessage.pinned)}
        metadataExtra={userMessage.metadataExtra}
        researchActions={buildMessageResearchActions(userMessage.metadataExtra)}
        discoSkillComment={userMessage.discoSkillComment}
        historyId={stableHistoryId ?? undefined}
        conversationInstanceId={conversationInstanceId}
        feedbackQuery={previousUserMessage?.message ?? null}
        isEmbedding={isEmbedding}
        characterIdentity={selectedCharacter}
        characterIdentityEnabled={characterIdentityEnabled}
        speakerCharacterId={userMessage.speakerCharacterId ?? null}
        speakerCharacterName={userMessage.speakerCharacterName}
        moodLabel={userMessage.moodLabel ?? null}
        moodConfidence={userMessage.moodConfidence ?? null}
        moodTopic={userMessage.moodTopic ?? null}
        searchQuery={normalizedSearchQuery || undefined}
        searchMatch={resolveSearchMatch(block.userIndex)}
        message_type={resolveMessageType(userMessage, block.userIndex)}
        variants={userMessage.variants}
        activeVariantIndex={userMessage.activeVariantIndex}
        onSwipePrev={() => handleVariantSwipe(userMessage.id, "prev")}
        onSwipeNext={() => handleVariantSwipe(userMessage.id, "next")}
        messageSteeringMode={messageSteeringMode}
        onMessageSteeringModeChange={setMessageSteeringMode}
        messageSteeringForceNarrate={messageSteeringForceNarrate}
        onMessageSteeringForceNarrateChange={setMessageSteeringForceNarrate}
        onClearMessageSteering={clearMessageSteering}
      />
      <div className="ml-10 space-y-2 border-l border-dashed border-border pl-4">
        <div className="mb-1 flex items-center justify-between text-[11px] text-text-muted">
          <div className="flex items-center gap-2">
            <span className="inline-flex items-center rounded-full bg-surface2 px-2 py-0.5 text-[10px] font-medium text-text">
              {t("playground:composer.compareClusterLabel", "Multi-model answers")}
            </span>
            <span className="text-[10px] text-text-subtle">
              {t("playground:composer.compareClusterCount", "{{count}} models", {
                count: replyItems.length
              })}
            </span>
          </div>
          {isChosenState && alternativeCount > 0 ? (
            <button
              type="button"
              onClick={() => setCollapsed(!collapsed)}
              title={
                collapsed
                  ? (t(
                      "common:timeline.expandAllAlternatives",
                      "Expand all alternatives"
                    ) as string)
                  : (t(
                      "common:timeline.collapseAllAlternatives",
                      "Collapse all alternatives"
                    ) as string)
              }
              className="text-[10px] font-medium text-primary hover:underline">
              {collapsed
                ? t(
                    "common:timeline.expandAllAlternatives",
                    "Expand all alternatives"
                  )
                : t(
                    "common:timeline.collapseAllAlternatives",
                    "Collapse all alternatives"
                  )}{" "}
              ({alternativeCount})
            </button>
          ) : null}
        </div>
        {clusterModelKeys.length > 1 ? (
          <div className="mb-2 flex flex-wrap items-center gap-2 text-[10px] text-text-muted">
            <span className="text-[10px] font-semibold uppercase tracking-wide">
              {t("playground:composer.compareFilterLabel", "Filter models")}
            </span>
            <div className="flex flex-wrap gap-1">
              {clusterModelKeys.map((modelKey) => {
                const isHidden = hiddenModels.includes(modelKey)
                const providerKey = getModelProvider(modelKey)
                return (
                  <button
                    key={`filter-${block.clusterId}-${modelKey}`}
                    type="button"
                    onClick={() => toggleModelFilter(modelKey)}
                    title={getModelLabel(modelKey)}
                    className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-medium transition ${
                      isHidden
                        ? "border-border bg-surface text-text-subtle"
                        : "border-primary bg-surface2 text-primaryStrong"
                    }`}>
                    <ProviderIcons provider={providerKey} className="h-3 w-3" />
                    <span className="max-w-[120px] truncate">
                      {getModelLabel(modelKey)}
                    </span>
                  </button>
                )
              })}
            </div>
            {hiddenModels.length > 0 ? (
              <button
                type="button"
                onClick={clearModelFilter}
                title={
                  t(
                    "playground:composer.compareFilterClear",
                    "Show all"
                  ) as string
                }
                className="text-[10px] font-medium text-primary hover:underline">
                {t("playground:composer.compareFilterClear", "Show all")}
              </button>
            ) : null}
            <span className="text-[10px] text-text-subtle">
              {t("playground:composer.compareFilterCount", "Showing {{visible}} / {{total}}", {
                visible: filteredReplyItems.length,
                total: replyItems.length
              })}
            </span>
            <button
              type="button"
              onClick={() => setNormalizedPreviewEnabled(!normalizedPreviewEnabled)}
              title={
                normalizedPreviewEnabled
                  ? (t(
                      "playground:composer.comparePreviewFullTitle",
                      "Switch to full responses"
                    ) as string)
                  : (t(
                      "playground:composer.comparePreviewNormalizedTitle",
                      "Switch to normalized previews"
                    ) as string)
              }
              className={`rounded-full border px-2 py-0.5 text-[10px] font-medium transition ${
                normalizedPreviewEnabled
                  ? "border-primary bg-primary/10 text-primaryStrong"
                  : "border-border bg-surface text-text-muted hover:bg-surface2 hover:text-text"
              }`}>
              {normalizedPreviewEnabled
                ? t("playground:composer.comparePreviewFull", "Full responses")
                : t(
                    "playground:composer.comparePreviewNormalized",
                    "Normalized previews"
                  )}
            </button>
            <button
              type="button"
              data-testid={`compare-diff-toggle-${block.clusterId}`}
              onClick={() => setDiffPreviewEnabled(!diffPreviewEnabled)}
              title={
                diffPreviewEnabled
                  ? (t(
                      "playground:composer.compareDiffHideTitle",
                      "Hide response differences"
                    ) as string)
                  : (t(
                      "playground:composer.compareDiffShowTitle",
                      "Show response differences"
                    ) as string)
              }
              className={`rounded-full border px-2 py-0.5 text-[10px] font-medium transition ${
                diffPreviewEnabled
                  ? "border-primary bg-primary/10 text-primaryStrong"
                  : "border-border bg-surface text-text-muted hover:bg-surface2 hover:text-text"
              }`}>
              {diffPreviewEnabled
                ? t("playground:composer.compareDiffHide", "Hide differences")
                : t("playground:composer.compareDiffShow", "Diff highlights")}
            </button>
          </div>
        ) : null}
        {compareFeatureEnabled && clusterSelection.length > 1 ? (
          <div className="mb-2 flex items-center justify-between text-[11px] text-text-muted">
            <span>
              {t("playground:composer.compareBulkSplitHint", "Selected models: {{count}}", {
                count: clusterSelection.length
              })}
            </span>
            <button
              type="button"
              onClick={() => {
                void handleBulkSplit()
              }}
              disabled={!compareFeatureEnabled}
              title={
                t(
                  "playground:composer.compareBulkSplit",
                  "Open each selected answer as its own chat"
                ) as string
              }
              className={`rounded border border-border bg-surface px-2 py-0.5 text-[10px] font-medium text-text hover:bg-surface2 ${
                !compareFeatureEnabled ? "cursor-not-allowed opacity-50" : ""
              }`}>
              {t(
                "playground:composer.compareBulkSplit",
                "Open each selected answer as its own chat"
              )}
            </button>
          </div>
        ) : null}
        {visibleReplyItems.map(({ index, message, modelKey }) => {
          const isSelected = clusterSelection.includes(modelKey)
          const errorPayload = decodeChatErrorPayload(message.message)
          const hasError = Boolean(errorPayload)
          const isSelectable = compareFeatureEnabled && !hasError
          const isChosenCard = isChosenState && selectedModelKey === modelKey
          const latencyLabel =
            typeof message?.reasoning_time_taken === "number" &&
            message.reasoning_time_taken > 0
              ? humanizeMilliseconds(message.reasoning_time_taken)
              : null
          const providerKey = getModelProvider(modelKey)
          const providerLabel = tldwModels.getProviderDisplayName(providerKey)
          const tokenCount = getTokenCount(message?.generationInfo)
          const tokenLabel =
            tokenCount !== null
              ? t("playground:composer.compareTokens", "Tokens: {{count}}", {
                  count: tokenCount
                })
              : null
          const costUsd = resolveMessageCostUsd(message?.generationInfo)
          const costLabel = costUsd != null ? formatCost(costUsd) : null
          const splitMap = compareSplitChats[block.clusterId] || {}
          const spawnedHistoryId = splitMap[modelKey]
          const normalizedPreviewBudget = normalizedPreviewEnabled
            ? computeNormalizedPreviewBudget(
                filteredReplyItems.map((item) =>
                  typeof item.message?.message === "string"
                    ? item.message.message
                    : ""
                )
              )
            : 0
          const normalizedPreviewText = normalizedPreviewEnabled
            ? buildNormalizedPreview(
                typeof message?.message === "string" ? message.message : "",
                normalizedPreviewBudget
              )
            : ""
          const isDiffBaselineCard =
            Boolean(diffBaselineModelKey) && modelKey === diffBaselineModelKey
          const diffPreview =
            diffPreviewEnabled && diffBaselineItem
              ? computeResponseDiffPreview({
                  baseline: diffBaselineText,
                  candidate:
                    typeof message?.message === "string" ? message.message : "",
                  maxHighlights: 2
                })
              : null

          const handleToggle = () => {
            if (!isSelectable) {
              return
            }
            const next = isSelected
              ? clusterSelection.filter((id) => id !== modelKey)
              : [...clusterSelection, modelKey]

            if (
              !isSelected &&
              compareMaxModels &&
              clusterSelection.length >= compareMaxModels
            ) {
              notification.warning({
                message: t(
                  "playground:composer.compareMaxModelsTitle",
                  "Compare limit reached"
                ),
                description: t(
                  "playground:composer.compareMaxModels",
                  "You can compare up to {{limit}} models per turn.",
                  { count: compareMaxModels, limit: compareMaxModels }
                )
              })
              return
            }
            setCompareSelectionForCluster(block.clusterId, next)
            setCompareActiveModelsForCluster(block.clusterId, next)
            setCompareSelectedModels(next)
            if (next.length > 1) {
              setCompareContinuationModeForCluster(block.clusterId, "compare")
            } else if (next.length === 1) {
              setCompareContinuationModeForCluster(block.clusterId, "winner")
            }
            void trackCompareMetric({
              type: "selection",
              count: next.length
            })
          }

          const displayName = message?.modelName || message.name
          const clusterMessagesForModel = messages
            .map((m: any, idx) => ({ m, idx }))
            .filter(
              ({ m }) =>
                m.clusterId === block.clusterId &&
                (m.messageType === "compare:user" ||
                  ((m as any).modelId || m.modelName || m.name) === modelKey)
            )
          const threadPreviewItems = clusterMessagesForModel.slice(-4)

          const handleOpenFullChat = async () => {
            if (!compareFeatureEnabled) {
              return
            }
            const newHistoryId = await createCompareBranch({
              clusterId: block.clusterId,
              modelId: modelKey
            })
            if (newHistoryId && historyId) {
              setCompareParentForHistory(newHistoryId, {
                parentHistoryId: historyId,
                clusterId: block.clusterId
              })
              setCompareSplitChat(block.clusterId, modelKey, newHistoryId)
            }
            if (modelKey) {
              setCompareMode(false)
              setSelectedModel(modelKey)
              setCompareSelectedModels([modelKey])
            }
          }

          const placeholder = t(
            "playground:composer.perModelReplyPlaceholder",
            "Reply only to {{model}}",
            { model: displayName }
          )
          const perModelDisabledReason = !compareFeatureEnabled
            ? t(
                "playground:composer.compareDisabled",
                "Compare mode is disabled in settings."
              )
            : null
          const perModelDisabled =
            isProcessing || streaming || Boolean(perModelDisabledReason)

          const handlePerModelSend = async (text: string) => {
            await sendPerModelReply({
              clusterId: block.clusterId,
              modelId: modelKey,
              message: text
            })
          }
          const previousUserMessage = getPreviousUserMessage(index)

          return (
            <div
              key={`c-${blockIndex}-${index}`}
              role="article"
              aria-label={tr(
                "playground:composer.compareCardAria",
                "{{model}} response from {{provider}}",
                {
                  model: getModelLabel(modelKey),
                  provider:
                    providerLabel ||
                    tr(
                      "playground:composer.compareProviderCustom",
                      "Custom provider"
                    )
                }
              )}
              className={`rounded-md border border-border bg-surface p-2 shadow-sm ${
                isChosenCard ? "ring-1 ring-success" : ""
              }`}>
              <div
                data-testid={`compare-model-identity-${block.clusterId}-${modelKey}`}
                className="mb-2 flex items-center justify-between gap-2 rounded-md border border-border bg-surface2 px-2 py-1 text-[11px]">
                <div className="flex min-w-0 items-center gap-1.5">
                  <ProviderIcons
                    provider={providerKey}
                    className="h-3.5 w-3.5 flex-shrink-0 text-text-subtle"
                  />
                  <span className="truncate font-semibold text-text">
                    {getModelLabel(modelKey)}
                  </span>
                  <span className="truncate text-[10px] text-text-subtle">
                    {providerLabel ||
                      tr(
                        "playground:composer.compareProviderCustom",
                        "Custom provider"
                      )}
                  </span>
                </div>
                <span className="rounded-full border border-border bg-surface px-1.5 py-0.5 text-[9px] font-medium text-text-muted">
                  {t("playground:composer.compareModelIdentityTag", "Model")}
                </span>
              </div>
              {normalizedPreviewEnabled ? (
                <div
                  data-testid={`compare-normalized-preview-${block.clusterId}-${modelKey}`}
                  className="mb-2 rounded-md border border-primary/30 bg-primary/5 px-2 py-1.5 text-[11px] text-primaryStrong">
                  <div className="mb-1 flex items-center justify-between gap-2">
                    <span className="text-[10px] font-semibold uppercase tracking-wide">
                      {t(
                        "playground:composer.comparePreviewLabel",
                        "Normalized preview"
                      )}
                    </span>
                    <span className="text-[10px] text-primaryStrong/80">
                      {tr("playground:composer.comparePreviewBudget", "~{{count}} chars", {
                        count: normalizedPreviewBudget
                      })}
                    </span>
                  </div>
                  <p className="whitespace-pre-wrap">
                    {normalizedPreviewText ||
                      tr(
                        "playground:composer.comparePreviewEmpty",
                        "No preview text available."
                      )}
                  </p>
                </div>
              ) : null}
              {diffPreview ? (
                <div
                  data-testid={`compare-diff-preview-${block.clusterId}-${modelKey}`}
                  className="mb-2 rounded-md border border-accent/35 bg-accent/5 px-2 py-1.5 text-[11px] text-text">
                  <div className="mb-1 flex items-center justify-between gap-2">
                    <span className="text-[10px] font-semibold uppercase tracking-wide text-text">
                      {isDiffBaselineCard
                        ? tr(
                            "playground:composer.compareDiffBaselineLabel",
                            "Diff baseline"
                          )
                        : tr(
                            "playground:composer.compareDiffVsLabel",
                            "Diff vs {{model}}",
                            {
                              model: getModelLabel(diffBaselineModelKey || modelKey)
                            }
                          )}
                    </span>
                    <span className="text-[10px] text-text-subtle">
                      {tr("playground:composer.compareDiffOverlap", "{{percent}}% overlap", {
                        percent: Math.round(diffPreview.overlapRatio * 100)
                      })}
                    </span>
                  </div>
                  {isDiffBaselineCard ? (
                    <p className="text-[10px] text-text-subtle">
                      {tr(
                        "playground:composer.compareDiffBaselineHint",
                        "Other responses are compared against this card."
                      )}
                    </p>
                  ) : diffPreview.hasMeaningfulDifference ? (
                    <div className="space-y-1">
                      {diffPreview.addedHighlights.length > 0 ? (
                        <div className="rounded border border-success/30 bg-success/10 px-2 py-1">
                          <div className="mb-0.5 text-[10px] font-medium uppercase tracking-wide text-success">
                            {t("playground:composer.compareDiffAddedLabel", "Added")}
                          </div>
                          {diffPreview.addedHighlights.map((entry) => (
                            <p
                              key={`diff-add-${block.clusterId}-${modelKey}-${entry}`}
                              className="line-clamp-2 text-[11px] text-success">
                              + {entry}
                            </p>
                          ))}
                        </div>
                      ) : null}
                      {diffPreview.removedHighlights.length > 0 ? (
                        <div className="rounded border border-warn/30 bg-warn/10 px-2 py-1">
                          <div className="mb-0.5 text-[10px] font-medium uppercase tracking-wide text-warn">
                            {t(
                              "playground:composer.compareDiffRemovedLabel",
                              "Removed"
                            )}
                          </div>
                          {diffPreview.removedHighlights.map((entry) => (
                            <p
                              key={`diff-remove-${block.clusterId}-${modelKey}-${entry}`}
                              className="line-clamp-2 text-[11px] text-warn">
                              - {entry}
                            </p>
                          ))}
                        </div>
                      ) : null}
                    </div>
                  ) : (
                    <p className="text-[10px] text-text-subtle">
                      {t(
                        "playground:composer.compareDiffNoChanges",
                        "No meaningful differences in previewed segments."
                      )}
                    </p>
                  )}
                </div>
              ) : null}
              <PlaygroundMessage
                isBot={message.isBot}
                message={message.message}
                name={message.name}
                role={message.role}
                images={message.images || []}
                currentMessageIndex={index}
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
                  editMessage(index, value, !message.isBot, isSend)
                }}
                onDeleteMessage={() => {
                  deleteMessage(index)
                }}
                onTogglePinned={() => {
                  void toggleMessagePinned(index)
                }}
                onNewBranch={() => {
                  createChatBranch(index)
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
                searchMatch={resolveSearchMatch(index)}
                message_type={resolveMessageType(message, index)}
                compareSelectable={isSelectable}
                compareSelected={isSelected}
                onToggleCompareSelect={handleToggle}
                compareError={hasError}
                compareErrorModelLabel={hasError ? getModelLabel(modelKey) : undefined}
                compareChosen={isChosenCard}
                variants={message.variants}
                activeVariantIndex={message.activeVariantIndex}
                onSwipePrev={() => handleVariantSwipe(message.id, "prev")}
                onSwipeNext={() => handleVariantSwipe(message.id, "next")}
                messageSteeringMode={messageSteeringMode}
                onMessageSteeringModeChange={setMessageSteeringMode}
                messageSteeringForceNarrate={messageSteeringForceNarrate}
                onMessageSteeringForceNarrateChange={setMessageSteeringForceNarrate}
                onClearMessageSteering={clearMessageSteering}
              />

              {threadPreviewItems.length > 1 ? (
                <div className="mt-2 space-y-1 rounded-md bg-surface2 p-2 text-[11px] text-text">
                  <div className="mb-0.5 text-[11px] font-medium tracking-wide text-text-subtle">
                    {t("playground:composer.compareThreadLabel", "Per-model thread")}
                  </div>
                  {threadPreviewItems.map(({ m, idx: threadIndex }) => (
                    <div
                      key={`thread-${block.clusterId}-${modelKey}-${threadIndex}`}
                      className="flex gap-1">
                      <span className="font-semibold">
                        {m.isBot
                          ? m.modelName || m.name
                          : m.messageType === "compare:user"
                            ? t(
                                "playground:composer.compareThreadShared",
                                "You (shared)"
                              )
                            : t("playground:composer.compareThreadYou", "You")}
                        :
                      </span>
                      <span className="line-clamp-2">{m.message}</span>
                    </div>
                  ))}
                </div>
              ) : null}

              <PerModelMiniComposer
                placeholder={placeholder}
                disabled={perModelDisabled}
                helperText={perModelDisabledReason}
                onSend={handlePerModelSend}
              />

              <div className="mt-1 flex items-center justify-between text-[11px] text-text-muted">
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => {
                      void handleOpenFullChat()
                    }}
                    disabled={!compareFeatureEnabled}
                    title={
                      t(
                        "playground:composer.compareOpenFullChat",
                        "Open as full chat"
                      ) as string
                    }
                    className={`text-primary hover:underline ${
                      !compareFeatureEnabled
                        ? "cursor-not-allowed opacity-50 no-underline"
                        : ""
                    }`}>
                    {t("playground:composer.compareOpenFullChat", "Open as full chat")}
                  </button>
                  {tokenLabel ? (
                    <span
                      className="inline-flex items-center gap-1 text-[10px] text-text-subtle"
                      aria-label={tokenLabel}>
                      <Hash className="h-3 w-3" aria-hidden="true" />
                      {tokenLabel}
                    </span>
                  ) : null}
                  {costLabel ? (
                    <span
                      className="inline-flex items-center gap-1 text-[10px] text-text-subtle"
                      aria-label={tr("playground:composer.compareCost", "Cost: {{cost}}", {
                        cost: costLabel
                      })}>
                      <DollarSign className="h-3 w-3" aria-hidden="true" />
                      {costLabel}
                    </span>
                  ) : null}
                  {latencyLabel ? (
                    <span
                      className="inline-flex items-center gap-1 text-[10px] text-text-subtle"
                      aria-label={t("playground:composer.compareLatency", "Latency")}>
                      <Clock className="h-3 w-3" aria-hidden="true" />
                      {latencyLabel}
                    </span>
                  ) : null}
                </div>
                <div className="flex items-center gap-2">
                  {spawnedHistoryId ? (
                    <button
                      type="button"
                      onClick={() => {
                        window.dispatchEvent(
                          new CustomEvent("tldw:open-history", {
                            detail: { historyId: spawnedHistoryId }
                          })
                        )
                      }}
                      title={
                        t(
                          "playground:composer.compareSpawnedChat",
                          "Open split chat"
                        ) as string
                      }
                      className="text-[10px] text-text-muted hover:text-text underline">
                      {t("playground:composer.compareSpawnedChat", "Open split chat")}
                    </button>
                  ) : null}
                  {message.id ? (
                    <button
                      type="button"
                      onClick={() => {
                        const currentCanonical =
                          compareCanonicalByCluster[block.clusterId] || null
                        const next = currentCanonical === message.id ? null : message.id
                        setCompareCanonicalForCluster(block.clusterId, next)
                      }}
                      title={
                        compareCanonicalByCluster[block.clusterId] === message.id
                          ? (t(
                              "playground:composer.compareCanonicalOn",
                              "Chosen"
                            ) as string)
                          : (t(
                              "playground:composer.compareCanonicalOff",
                              "Choose as answer"
                            ) as string)
                      }
                      className={`rounded px-2 py-0.5 text-[10px] font-medium border transition ${
                        compareCanonicalByCluster[block.clusterId] === message.id
                          ? "border-success bg-success text-white"
                          : "border-success/40 bg-success/10 text-success"
                      }`}>
                      {compareCanonicalByCluster[block.clusterId] === message.id
                        ? t("playground:composer.compareCanonicalOn", "Chosen")
                        : t(
                            "playground:composer.compareCanonicalOff",
                            "Choose as answer"
                          )}
                    </button>
                  ) : null}
                </div>
              </div>
            </div>
          )
        })}

        {clusterSelection.length > 0 ? (
          <div className="mt-2 rounded-md border border-border bg-surface2 px-3 py-2 text-[11px] text-text-muted">
            <div className="flex flex-wrap items-center gap-2">
              <span className="font-medium">
                {t("playground:composer.compareSelectedLabel", "Chosen answer:")}
              </span>
              <div className="flex flex-wrap gap-1">
                {clusterSelection.map((modelKey) => (
                  <span
                    key={`selected-${block.clusterId}-${modelKey}`}
                    className="rounded-full bg-surface px-2 py-0.5 text-[10px] font-medium text-text shadow-sm">
                    {getModelLabel(modelKey)}
                  </span>
                ))}
              </div>
            </div>
            {clusterActiveModels.length === 1 ? (
              <div className="mt-2 flex items-center justify-between gap-2">
                <span className="text-[10px] text-text-muted">
                  {compareModeActive
                    ? t(
                        "playground:composer.compareContinueHint",
                        "Continue this chat with the selected model."
                      )
                    : t(
                        "playground:composer.compareChosenHint",
                        "Continue with the selected answer or keep comparing."
                      )}
                </span>
                {compareModeActive ? (
                  <button
                    type="button"
                    onClick={() => handleContinueWithModel(clusterActiveModels[0])}
                    title={
                      t(
                        "playground:composer.compareContinueWinner",
                        "Continue with winner"
                      ) as string
                    }
                    className="rounded border border-primary bg-primary px-2 py-0.5 text-[10px] font-medium text-white hover:bg-primaryStrong">
                    {t("playground:composer.compareContinueWinner", "Continue with winner")}
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={handleCompareAgain}
                    disabled={!compareFeatureEnabled}
                    title={
                      t(
                        "playground:composer.compareKeepComparing",
                        "Keep comparing"
                      ) as string
                    }
                    className={`rounded border border-primary px-2 py-0.5 text-[10px] font-medium ${
                      compareFeatureEnabled
                        ? "border-primary bg-surface text-primary hover:bg-surface2"
                        : "border-primary/40 bg-surface text-text-subtle cursor-not-allowed opacity-60"
                    }`}>
                    {t("playground:composer.compareKeepComparing", "Keep comparing")}
                  </button>
                )}
              </div>
            ) : (
              <div className="mt-2 text-[10px] text-text-muted">
                {t(
                  "playground:composer.compareActiveModelsHint",
                  "Your next message will be sent to each active model."
                )}
              </div>
            )}
            <div className="mt-2 flex items-center gap-2 text-[10px] text-text-subtle">
              <span className="font-medium">
                {t(
                  "playground:composer.compareContinuationLabel",
                  "Continuation mode:"
                )}
              </span>
              <span className="rounded-full border border-border bg-surface px-2 py-0.5">
                {continuationMode === "compare"
                  ? t(
                      "playground:composer.compareContinuationCompare",
                      "Keep comparing"
                    )
                  : t(
                      "playground:composer.compareContinuationWinner",
                      "Winner only"
                    )}
              </span>
            </div>
          </div>
        ) : null}

        {(() => {
          const canonicalId = compareCanonicalByCluster[block.clusterId] || null
          if (!canonicalId) {
            return null
          }
          const canonical = (messages as any[]).find((m) => m.id && m.id === canonicalId)
          if (!canonical) {
            return null
          }
          return (
            <div className="mt-3 rounded-md border border-success bg-success/10 px-3 py-2 text-[13px] text-success">
              <div className="mb-1 flex items-center gap-2 text-[11px] font-medium">
                <span className="uppercase tracking-wide">
                  {t("playground:composer.compareCanonicalLabel", "Chosen answer")}
                </span>
                <span className="text-success/80">
                  {canonical.modelName || canonical.name}
                </span>
              </div>
              <div className="whitespace-pre-wrap">{canonical.message}</div>
            </div>
          )
        })()}
      </div>
    </div>
  )
}
