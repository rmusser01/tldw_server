import React from "react"
import { useQuery } from "@tanstack/react-query"
import { useTranslation } from "react-i18next"
import { Select, Tooltip, Switch } from "antd"
import {
  Send,
  GitCompare,
  Plus,
  X,
  Trash2,
  StopCircle
} from "lucide-react"
import { useMessageOption } from "@/hooks/useMessageOption"
import { PlaygroundMessage } from "@/components/Common/Playground/Message"
import { PlaygroundEmpty } from "@/components/Option/Playground/PlaygroundEmpty"
import { ProviderIcons } from "@/components/Common/ProviderIcon"
import { fetchChatModels } from "@/services/tldw-server"
import { MAX_COMPARE_MODELS } from "@/hooks/chat/compare-constants"
import { useAntdNotification } from "@/hooks/useAntdNotification"
import useDynamicTextareaSize from "@/hooks/useDynamicTextareaSize"

type TimelineBlock =
  | { kind: "single"; index: number }
  | {
      kind: "compare"
      userIndex: number
      assistantIndices: number[]
      clusterId: string
    }

const buildBlocks = (
  messages: { messageType?: string; clusterId?: string }[]
): TimelineBlock[] => {
  const blocks: TimelineBlock[] = []
  const used = new Set<number>()

  messages.forEach((msg, idx) => {
    if (used.has(idx)) return

    if (msg.messageType === "compare:user" && msg.clusterId) {
      const assistants: number[] = []
      messages.forEach((m, j) => {
        if (j !== idx && m.clusterId === msg.clusterId) {
          if (m.messageType === "compare:reply") {
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

/**
 * ModelPlaygroundChat - Chat interface for Model Playground
 *
 * Always has compare mode available with a simplified, developer-focused UI.
 * Includes the message input directly in this component.
 */
export const ModelPlaygroundChat: React.FC = () => {
  const { t } = useTranslation(["playground", "common"])
  const notification = useAntdNotification()
  const textareaRef = React.useRef<HTMLTextAreaElement>(null)
  const [message, setMessage] = React.useState("")

  const {
    messages,
    streaming,
    isProcessing,
    regenerateLastMessage,
    editMessage,
    deleteMessage,
    onSubmit,
    stopStreamingRequest,
    selectedModel,
    setSelectedModel,
    compareMode,
    setCompareMode,
    compareSelectedModels,
    setCompareSelectedModels,
    compareMaxModels,
    clearChat,
    ttsEnabled,
    historyId
  } = useMessageOption({ forceCompareEnabled: true })

  const { data: chatModels = [] } = useQuery({
    queryKey: ["workspace:chatModels"],
    queryFn: () => fetchChatModels({ returnEmpty: true }),
    enabled: true
  })

  useDynamicTextareaSize(textareaRef, message, 200)

  const blocks = React.useMemo(() => buildBlocks(messages), [messages])

  // Ensure selected model is in compare list when compare mode is on
  React.useEffect(() => {
    if (compareMode && compareSelectedModels.length === 0 && selectedModel) {
      setCompareSelectedModels([selectedModel])
    }
  }, [compareMode, compareSelectedModels.length, selectedModel, setCompareSelectedModels])

  const handleCompareToggle = () => {
    if (!compareMode && selectedModel) {
      setCompareSelectedModels([selectedModel])
    }
    setCompareMode(!compareMode)
  }

  const handleAddCompareModel = (modelId: string) => {
    if (compareSelectedModels.includes(modelId)) return
    const max = compareMaxModels || MAX_COMPARE_MODELS
    if (compareSelectedModels.length >= max) {
      notification.warning({
        message: t("playground:composer.compareMaxModelsTitle", "Compare limit reached"),
        description: t(
          "playground:composer.compareMaxModels",
          "You can compare up to {{limit}} models per turn.",
          { limit: max }
        )
      })
      return
    }
    setCompareSelectedModels([...compareSelectedModels, modelId])
  }

  const handleRemoveCompareModel = (modelId: string) => {
    setCompareSelectedModels(compareSelectedModels.filter((id) => id !== modelId))
  }

  const canSend = React.useMemo(() => {
    if (compareMode) return compareSelectedModels.length > 0
    return Boolean(selectedModel && selectedModel.trim().length > 0)
  }, [compareMode, compareSelectedModels.length, selectedModel])

  const handleSend = async () => {
    if (!message.trim() || isProcessing || streaming) return
    if (!canSend) {
      notification.error({
        message: t("error"),
        description: compareMode
          ? t(
              "playground:composer.validationCompareSelectModels",
              "Select at least one model to use in Compare mode."
            )
          : t("validationSelectModel")
      })
      return
    }

    await onSubmit({
      message: message.trim(),
      image: ""
    })
    setMessage("")
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const modelOptions = React.useMemo(() => {
    return (chatModels as any[]).map((model) => ({
      value: model.model,
      label: (
        <div className="flex items-center gap-2">
          <ProviderIcons
            provider={model.provider}
            className="h-4 w-4 text-text-subtle"
          />
          <span className="truncate">{model.nickname || model.model}</span>
        </div>
      )
    }))
  }, [chatModels])

  const getModelLabel = (modelId: string) => {
    const model = (chatModels as any[]).find((m) => m.model === modelId)
    return model?.nickname || model?.model || modelId
  }

  const getModelProvider = (modelId: string) => {
    const model = (chatModels as any[]).find((m) => m.model === modelId)
    return String(model?.provider || "custom").toLowerCase()
  }

  return (
    <div className="flex flex-col pt-4">
      {/* Empty state */}
      {messages.length === 0 && (
        <div className="mt-16">
          <PlaygroundEmpty />
        </div>
      )}

      {/* Messages */}
      {blocks.map((block, blockIndex) => {
        if (block.kind === "single") {
          const msg = messages[block.index]
          return (
            <PlaygroundMessage
              key={`m-${blockIndex}`}
              isBot={msg.isBot}
              message={msg.message}
              name={msg.name}
              role={msg.role}
              images={msg.images || []}
              currentMessageIndex={block.index}
              totalMessages={messages.length}
              onRegenerate={regenerateLastMessage}
              isProcessing={isProcessing}
              isSearchingInternet={false}
              sources={msg.sources}
              onEditFormSubmit={(value, isSend) => {
                editMessage(block.index, value, !msg.isBot, isSend)
              }}
              onDeleteMessage={() => {
                void deleteMessage(block.index)
              }}
              isTTSEnabled={ttsEnabled}
              generationInfo={msg?.generationInfo}
              toolCalls={msg?.toolCalls}
              toolResults={msg?.toolResults}
              isStreaming={streaming}
              modelImage={msg?.modelImage}
              modelName={msg?.modelName}
              createdAt={msg?.createdAt}
              temporaryChat={false}
              onStopStreaming={stopStreamingRequest}
              onContinue={() => {
                void onSubmit({ image: "", message: "", isContinue: true })
              }}
              documents={msg?.documents}
              messageId={msg.id}
              discoSkillComment={msg.discoSkillComment}
              conversationInstanceId={historyId || "workspace"}
            />
          )
        }

        // Compare block
        const userMessage = messages[block.userIndex]
        const replyItems = block.assistantIndices.map((i) => ({
          index: i,
          message: messages[i],
          modelKey:
            (messages[i] as any).modelId ||
            messages[i].modelName ||
            messages[i].name
        }))

        return (
          <div key={`c-${blockIndex}`} className="mb-4 space-y-2">
            {/* User message */}
            <PlaygroundMessage
              isBot={userMessage.isBot}
              message={userMessage.message}
              name={userMessage.name}
              role={userMessage.role}
              images={userMessage.images || []}
              currentMessageIndex={block.userIndex}
              totalMessages={messages.length}
              onRegenerate={regenerateLastMessage}
              isProcessing={isProcessing}
              isSearchingInternet={false}
              sources={userMessage.sources}
              onEditFormSubmit={(value, isSend) => {
                editMessage(block.userIndex, value, !userMessage.isBot, isSend)
              }}
              onDeleteMessage={() => {
                void deleteMessage(block.userIndex)
              }}
              isTTSEnabled={ttsEnabled}
              generationInfo={userMessage?.generationInfo}
              toolCalls={userMessage?.toolCalls}
              toolResults={userMessage?.toolResults}
              isStreaming={streaming}
              modelImage={userMessage?.modelImage}
              modelName={userMessage?.modelName}
              createdAt={userMessage?.createdAt}
              temporaryChat={false}
              onStopStreaming={stopStreamingRequest}
              documents={userMessage?.documents}
              messageId={userMessage.id}
              discoSkillComment={userMessage.discoSkillComment}
              conversationInstanceId={historyId || "workspace"}
            />

            {/* Compare responses */}
            <div className="ml-10 space-y-2 border-l-2 border-dashed border-primary/30 pl-4">
              <div className="flex items-center gap-2 text-xs text-text-muted">
                <GitCompare className="h-3.5 w-3.5 text-primary" />
                <span className="font-medium">
                  {t("playground:workspace.compareResponses", "Compare Responses")} ({replyItems.length})
                </span>
              </div>
              <div className="grid gap-3 md:grid-cols-2">
                {replyItems.map(({ index, message: replyMsg, modelKey }) => (
                  <div
                    key={`reply-${index}`}
                    className="rounded-lg border border-border bg-surface p-3"
                  >
                    <div className="mb-2 flex items-center gap-2 text-xs">
                      <ProviderIcons
                        provider={getModelProvider(modelKey)}
                        className="h-4 w-4"
                      />
                      <span className="font-medium text-text">
                        {getModelLabel(modelKey)}
                      </span>
                    </div>
                    <PlaygroundMessage
                      isBot={replyMsg.isBot}
                      message={replyMsg.message}
                      name={replyMsg.name}
                      role={replyMsg.role}
                      images={replyMsg.images || []}
                      currentMessageIndex={index}
                      totalMessages={messages.length}
                      onRegenerate={regenerateLastMessage}
                      isProcessing={isProcessing}
                      isSearchingInternet={false}
                      sources={replyMsg.sources}
                      onEditFormSubmit={(value, isSend) => {
                        editMessage(index, value, !replyMsg.isBot, isSend)
                      }}
                      onDeleteMessage={() => {
                        void deleteMessage(index)
                      }}
                      isTTSEnabled={ttsEnabled}
                      generationInfo={replyMsg?.generationInfo}
                      toolCalls={replyMsg?.toolCalls}
                      toolResults={replyMsg?.toolResults}
                      isStreaming={streaming}
                      modelImage={replyMsg?.modelImage}
                      modelName={replyMsg?.modelName}
                      createdAt={replyMsg?.createdAt}
                      temporaryChat={false}
                      onStopStreaming={stopStreamingRequest}
                      documents={replyMsg?.documents}
                      messageId={replyMsg.id}
                      discoSkillComment={replyMsg.discoSkillComment}
                      conversationInstanceId={historyId || "workspace"}
                    />
                  </div>
                ))}
              </div>
            </div>
          </div>
        )
      })}

      {/* Input area */}
      <div className="sticky bottom-0 mt-4 border-t border-border bg-surface/95 px-4 py-4 backdrop-blur">
        {/* Compare mode controls */}
        <div className="mb-3 flex flex-wrap items-center gap-3">
          {/* Compare toggle */}
          <div className="flex items-center gap-2">
            <Switch
              size="small"
              checked={compareMode}
              onChange={handleCompareToggle}
            />
            <span className="text-xs font-medium text-text">
              <GitCompare className="mr-1 inline h-3.5 w-3.5" />
              {t("playground:workspace.compareMode", "Compare Mode")}
            </span>
          </div>

          {/* Model selector (single mode) */}
          {!compareMode && (
            <Select
              value={selectedModel}
              onChange={setSelectedModel}
              options={modelOptions}
              placeholder={t("playground:composer.selectModel", "Select model")}
              className="w-52"
              size="small"
            />
          )}

          {/* Compare model chips */}
          {compareMode && (
            <div className="flex flex-wrap items-center gap-2">
              {compareSelectedModels.map((modelId) => (
                <div
                  key={modelId}
                  className="flex items-center gap-1 rounded-full border border-primary/50 bg-primary/10 px-2 py-0.5 text-xs"
                >
                  <ProviderIcons
                    provider={getModelProvider(modelId)}
                    className="h-3 w-3"
                  />
                  <span className="max-w-24 truncate">{getModelLabel(modelId)}</span>
                  <button
                    type="button"
                    onClick={() => handleRemoveCompareModel(modelId)}
                    className="ml-0.5 rounded p-0.5 hover:bg-primary/20"
                  >
                    <X className="h-3 w-3" />
                  </button>
                </div>
              ))}
              <Select
                value={null}
                onChange={(value) => value && handleAddCompareModel(value)}
                options={modelOptions.filter(
                  (opt) => !compareSelectedModels.includes(opt.value)
                )}
                placeholder={
                  <div className="flex items-center gap-1 text-xs">
                    <Plus className="h-3 w-3" />
                    {t("playground:workspace.addModel", "Add model")}
                  </div>
                }
                className="w-36"
                size="small"
                allowClear={false}
              />
            </div>
          )}

          {/* Actions */}
          <div className="ml-auto flex items-center gap-2">
            <Tooltip title={t("playground:workspace.clearChat", "Clear chat")}>
              <button
                type="button"
                onClick={clearChat}
                disabled={messages.length === 0}
                className="rounded-lg p-2 text-text-muted transition-colors hover:bg-surface2 hover:text-text disabled:opacity-50"
              >
                <Trash2 className="h-4 w-4" />
              </button>
            </Tooltip>
          </div>
        </div>

        {/* Message input */}
        <div className="flex items-end gap-3">
          <textarea
            ref={textareaRef}
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              compareMode && compareSelectedModels.length > 1
                ? t("playground:workspace.messagePlaceholderCompare", "Send to {{count}} models...", {
                    count: compareSelectedModels.length
                  })
                : t("playground:workspace.messagePlaceholder", "Type your message...")
            }
            rows={1}
            className="min-h-[44px] max-h-48 flex-1 resize-none rounded-xl border border-border bg-bg px-4 py-3 text-sm text-text placeholder:text-text-muted focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
          />
          {streaming ? (
            <Tooltip title={t("common:stop", "Stop")}>
              <button
                type="button"
                onClick={stopStreamingRequest}
                className="rounded-xl bg-danger p-3 text-white transition hover:bg-danger/90"
              >
                <StopCircle className="h-5 w-5" />
              </button>
            </Tooltip>
          ) : (
            <Tooltip title={t("common:send", "Send")}>
              <button
                type="button"
                onClick={handleSend}
              disabled={!message.trim() || isProcessing || !canSend}
              className="rounded-xl bg-primary p-3 text-white transition hover:bg-primaryStrong disabled:opacity-50 disabled:cursor-not-allowed"
            >
                <Send className="h-5 w-5" />
              </button>
            </Tooltip>
          )}
        </div>
      </div>
    </div>
  )
}
