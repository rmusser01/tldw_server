import React, { useState, useRef, useCallback, Suspense, useLayoutEffect, useId } from "react"
import { smoothScrollBehavior } from "@/utils/reduced-motion"
import { useTranslation } from "react-i18next"
import { Empty, Spin, Switch, Tooltip } from "antd"
import {
  SendHorizontal,
  Plus,
  AlertCircle,
  BookOpen,
  StopCircle,
  Sparkles
} from "lucide-react"
import { useDocumentWorkspaceStore } from "@/store/document-workspace"
import type { QuickIngestOperation } from "@/services/tldw/quick-ingest-authority"
import { useDocumentChat, type DocumentChatCache } from "@/hooks/document-workspace/useDocumentChat"
import { useDocumentMetadata } from "@/hooks/document-workspace/useDocumentMetadata"
import { useMessageOption } from "@/hooks/useMessageOption"
import { markChatTurnStoppedByUser } from "@/hooks/chat/abort-turn-cleanup"
import { SuggestedQuestions } from "./SuggestedQuestions"

const PlaygroundMessage = React.lazy(() =>
  import("@/components/Common/Playground/Message").then((m) => ({
    default: m.PlaygroundMessage
  }))
)

/**
 * DocumentChat - Chat interface scoped to the current document.
 *
 * This component:
 * - Uses useDocumentChat hook to scope RAG queries to the active document
 * - Displays a simplified chat interface without model selection
 * - Shows suggested questions when chat is empty
 * - Displays message sources for RAG responses
 */
export const DocumentChat: React.FC<{
  operation: QuickIngestOperation
  conversations?: DocumentChatCache
}> = ({ operation, conversations: workspaceConversations }) => {
  const { t } = useTranslation(["option", "common", "playground"])
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const activeTurn = useRef<AbortController | null>(null)
  const activeTabRef = useRef<HTMLButtonElement | null>(null)
  const focusDraftRef = useRef<string | null>(null)
  const panelId = useId()

  // Get active document from workspace store
  const activeDocumentId = useDocumentWorkspaceStore((s) => s.activeDocumentId)
  const pendingSelectionPrompt = useDocumentWorkspaceStore(s => s.pendingSelectionPrompt)
  const activeDocumentType = useDocumentWorkspaceStore(
    (s) => s.activeDocumentType
  )

  // Use the document chat hook to scope RAG
  const {
    messages,
    streaming,
    isProcessing,
    isServerAvailable,
    isChatReady,
    historyDocumentId,
    capturePrivateScene,
    conversations,
    activeConversationKey,
    conversationStatus,
    inputValue,
    setInputValue,
    newConversation,
    selectConversation,
    retryConversation,
    onHistorySaved,
    trackTurn,
    ragEnabled,
    setRagEnabled
  } = useDocumentChat(activeDocumentId, operation, workspaceConversations)

  // Get full message option for onSubmit and other actions
  const {
    onSubmit,
    regenerateLastMessage,
    editMessage,
    deleteMessage,
    ttsEnabled,
    temporaryChat,
    serverChatId,
    serverChatMetaLoaded,
    serverChatLoadState,
    actionInfo,
    selectedModel
  } = useMessageOption({
    hydrateServerChat: isChatReady,
    capturePrivateScene,
    historyDocumentId,
    onDocumentHistorySaved: onHistorySaved
  })
  const waitingForChatDetails = isChatReady && Boolean(serverChatId) && !serverChatMetaLoaded
  const canSend = isChatReady && !waitingForChatDetails

  const { data: documentMetadata } = useDocumentMetadata(activeDocumentId)

  const isResearchPaper = React.useMemo(() => {
    if (activeDocumentType !== "pdf") return false
    if (!documentMetadata) return false
    return Boolean(
      (documentMetadata.authors && documentMetadata.authors.length > 0) ||
        documentMetadata.abstract ||
        (documentMetadata.keywords && documentMetadata.keywords.length > 0)
    )
  }, [activeDocumentType, documentMetadata])

  const questionVariant =
    activeDocumentType === "pdf"
      ? isResearchPaper
        ? "research"
        : "general"
      : undefined

  const hasSelectedModel = Boolean(
    selectedModel && selectedModel.trim().length > 0
  )

  // Track if suggested questions should be shown
  const [suggestionsCollapsed, setSuggestionsCollapsed] = useState(false)

  // Scroll to bottom when new messages arrive
  React.useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: smoothScrollBehavior() })
    }
  }, [messages.length])

  React.useEffect(() => {
    setSuggestionsCollapsed(false)
  }, [activeConversationKey])

  useLayoutEffect(() => {
    activeTabRef.current?.scrollIntoView?.({ block: "nearest", inline: "nearest" })
  }, [activeConversationKey])

  useLayoutEffect(() => {
    if (!activeConversationKey || !pendingSelectionPrompt || !operation.isCurrent() ||
        pendingSelectionPrompt.documentId !== activeDocumentId ||
        useDocumentWorkspaceStore.getState().pendingSelectionPrompt !== pendingSelectionPrompt) return
    useDocumentWorkspaceStore.setState({ pendingSelectionPrompt: null })
    const { prompt } = pendingSelectionPrompt
    setInputValue(draft => draft.trim() ? `${draft}\n\n${prompt}` : prompt)
    focusDraftRef.current = activeConversationKey
  }, [activeConversationKey, activeDocumentId, operation, pendingSelectionPrompt, setInputValue])

  useLayoutEffect(() => {
    const input = textareaRef.current
    if (!activeConversationKey || focusDraftRef.current !== activeConversationKey || !input || input.disabled) return
    focusDraftRef.current = null
    input.focus({ preventScroll: true })
    input.setSelectionRange(input.value.length, input.value.length)
  }, [activeConversationKey, hasSelectedModel, inputValue, isChatReady])

  const stopConversationTurn = useCallback(() => {
    const controller = activeTurn.current
    if (!controller) return
    activeTurn.current = null
    markChatTurnStoppedByUser(controller.signal)
    controller.abort()
  }, [])
  useLayoutEffect(() => stopConversationTurn, [activeDocumentId, operation, stopConversationTurn])

  const runTurn = useCallback(async <T,>(send: (controller: AbortController) => Promise<T>) => {
    if (activeTurn.current || !canSend) return
    const controller = new AbortController()
    activeTurn.current = controller
    try {
      return await trackTurn(() => send(controller))
    } finally {
      if (activeTurn.current === controller) activeTurn.current = null
    }
  }, [canSend, trackTurn])

  const submitQuestion = useCallback((question: string, isContinue = false) =>
    runTurn(controller => onSubmit({ message: question, image: "", controller, isContinue })),
  [onSubmit, runTurn])

  // Handle sending a message
  const handleSend = useCallback(async () => {
    const trimmedMessage = inputValue.trim()
    if (!trimmedMessage || isProcessing || streaming || !hasSelectedModel || !canSend) {
      return
    }

    // Collapse suggestions after first message
    if (!suggestionsCollapsed) {
      setSuggestionsCollapsed(true)
    }

    setInputValue("")

    const result = await submitQuestion(trimmedMessage)
    if (result?.status === "failed" && result.restoreDraft && result.restoreDraftIsCurrent?.() !== false) {
      setInputValue(current => current || inputValue)
    }
  }, [
    canSend,
    hasSelectedModel,
    inputValue,
    isProcessing,
    setInputValue,
    submitQuestion,
    streaming,
    suggestionsCollapsed
  ])

  // Handle suggested question click - auto-send
  const handleQuestionClick = useCallback(
    async (question: string) => {
      if (!hasSelectedModel || !canSend || isProcessing || streaming) return
      setSuggestionsCollapsed(true)
      await submitQuestion(question)
    },
    [canSend, hasSelectedModel, isProcessing, streaming, submitQuestion]
  )

  // Handle keyboard events in textarea
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleTabKeys = (event: React.KeyboardEvent<HTMLDivElement>) => {
    const tabs = Array.from(event.currentTarget.querySelectorAll<HTMLButtonElement>('[role="tab"]'))
    const current = tabs.findIndex(tab => tab === event.target)
    if (current < 0) return
    let next: number
    switch (event.key) {
      case "ArrowRight":
        next = (current + 1) % tabs.length
        break
      case "ArrowLeft":
        next = (current + tabs.length - 1) % tabs.length
        break
      case "Home":
        next = 0
        break
      case "End":
        next = tabs.length - 1
        break
      default:
        return
    }
    event.preventDefault()
    tabs[next]?.focus()
    tabs[next]?.click()
  }

  // No document selected state
  if (activeDocumentId === null) {
    return (
      <div className="flex h-full flex-col items-center justify-center p-4 text-center">
        <BookOpen className="mb-3 h-12 w-12 text-text-muted" />
        <p className="text-sm text-text-muted">
          {t(
            "option:documentWorkspace.noDocumentForChat",
            "Ask questions and get AI-powered answers about your document. Open a document to start."
          )}
        </p>
      </div>
    )
  }

  // Server not available state
  if (!isServerAvailable) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-2 p-4 text-center">
        <AlertCircle className="mb-2 h-10 w-10 text-warning" />
        <p className="text-sm font-medium text-text-muted">
          {t(
            "option:documentWorkspace.serverRequired",
            "Connect to server to use document chat"
          )}
        </p>
        <p className="text-xs text-text-muted max-w-xs">
          {t(
            "option:documentWorkspace.serverRequiredHint",
            "Start the server and configure the connection in Settings. The server provides AI chat, document search, and document analysis."
          )}
        </p>
      </div>
    )
  }

  const hasMessages = isChatReady && messages.length > 0
  const showSuggestions = isChatReady && !suggestionsCollapsed
  const newChatLabel = t("option:documentWorkspace.newConversation", "New chat")
  const useDocumentLabel = t("option:documentWorkspace.enableRag", "Use document content")

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex shrink-0 items-center gap-1 border-b border-border px-2">
        <div
          role="tablist"
          aria-label={t("option:documentWorkspace.conversations", "Conversations")}
          className="flex min-w-0 flex-1 gap-1 overflow-x-auto py-1 scrollbar-thin"
          onKeyDown={handleTabKeys}
        >
          {conversations.map(({ key, title }, index) => {
            const selected = key === activeConversationKey
            const label = title || t("option:documentWorkspace.conversationDraft", "Draft {{number}}", { number: index + 1 })
            return (
              <button
                key={key}
                ref={selected ? activeTabRef : undefined}
                id={`${panelId}-${key}`}
                role="tab"
                type="button"
                title={label}
                aria-selected={selected}
                aria-controls={selected ? panelId : undefined}
                tabIndex={selected ? 0 : -1}
                className="max-w-36 shrink-0 truncate rounded px-2 py-1.5 text-sm text-text-muted hover:bg-hover hover:text-text aria-selected:bg-primary/10 aria-selected:font-medium aria-selected:text-primary focus-visible:outline focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-primary"
                onClick={() => {
                  if (selected) return
                  stopConversationTurn()
                  selectConversation(key)
                }}
              >
                {label}
              </button>
            )
          })}
        </div>
        <button
          type="button"
          disabled={conversations.length === 0}
          className="flex min-h-8 shrink-0 items-center gap-1 rounded px-2 text-sm text-text hover:bg-hover focus-visible:outline focus-visible:outline-2 focus-visible:outline-primary disabled:opacity-50"
          onClick={() => {
            stopConversationTurn()
            newConversation()
          }}
        >
          <Plus className="h-4 w-4" aria-hidden="true" />
          {newChatLabel}
        </button>
      </div>
      <div
        id={panelId}
        role={activeConversationKey ? "tabpanel" : undefined}
        aria-labelledby={activeConversationKey ? `${panelId}-${activeConversationKey}` : undefined}
        className="flex min-h-0 flex-1 flex-col"
      >

        {hasMessages && suggestionsCollapsed && (
          <div className="flex justify-center border-b border-border py-1">
            <button
              type="button"
              onClick={() => setSuggestionsCollapsed(false)}
              className="flex items-center gap-1 rounded px-2 py-1 text-xs text-text-muted transition-colors hover:bg-hover hover:text-text"
            >
              <Sparkles className="h-3 w-3" />
              {t("option:documentWorkspace.showSuggestions", "Show suggestions")}
            </button>
          </div>
        )}

        <div className="flex-1 overflow-y-auto">
          {conversationStatus === "loading" && (
            <div role="status" className="flex items-center gap-2 p-3 text-sm text-text-muted">
              <Spin size="small" />
              {t("option:documentWorkspace.loadingConversation", "Loading conversation…")}
            </div>
          )}
          {conversationStatus === "error" && (
            <div className="flex items-center justify-between gap-2 p-3 text-sm">
              <p role="alert" className="text-error">
                {t("option:documentWorkspace.conversationLoadError", "Could not load this conversation.")}
              </p>
              <button
                type="button"
                onClick={retryConversation}
                className="shrink-0 rounded border border-border px-2 py-1 text-text hover:bg-hover focus-visible:outline focus-visible:outline-2 focus-visible:outline-primary"
              >
                {t("common:retry", "Retry")}
              </button>
            </div>
          )}
          {showSuggestions && (
            <SuggestedQuestions
              documentType={activeDocumentType}
              variant={questionVariant}
              onQuestionClick={handleQuestionClick}
              disabled={!canSend || isProcessing || streaming}
            />
          )}

          {hasMessages && (
            <div className="space-y-4 p-3">
              <Suspense
                fallback={
                  <div className="flex items-center justify-center p-4">
                    <Spin size="small" />
                  </div>
                }
              >
                {messages.map((message, index) => (
                  <div key={message.id || `msg-${index}`}>
                    <PlaygroundMessage
                      isBot={message.isBot}
                      message={message.message}
                      name={message.name}
                      role={message.role}
                      images={message.images || []}
                      currentMessageIndex={index}
                      totalMessages={messages.length}
                      onRegenerate={() => runTurn(controller => regenerateLastMessage({ controller }))}
                      isProcessing={isProcessing}
                      sources={message.sources}
                      onEditFormSubmit={(value, isSend, editIsCurrent, editOriginIsCurrent) => {
                        return trackTurn(() => editMessage(
                          index,
                          value,
                          !message.isBot,
                          isSend,
                          editIsCurrent,
                          editOriginIsCurrent
                        ))
                      }}
                      onDeleteMessage={() => {
                        return deleteMessage(index)
                      }}
                      isTTSEnabled={ttsEnabled}
                      generationInfo={message?.generationInfo}
                      isStreaming={streaming}
                      modelImage={message?.modelImage}
                      modelName={message?.modelName}
                      createdAt={message?.createdAt}
                      temporaryChat={temporaryChat}
                      onStopStreaming={stopConversationTurn}
                      onContinue={() => submitQuestion("", true)}
                      documents={message?.documents}
                      actionInfo={actionInfo}
                      serverChatId={serverChatId}
                      serverMessageId={message.serverMessageId}
                      messageId={message.id}
                      discoSkillComment={message.discoSkillComment}
                      conversationInstanceId={`doc-${activeDocumentId}-${activeConversationKey}`}
                      hideCopy={false}
                      hideEditAndRegenerate={false}
                      toolCalls={message?.toolCalls}
                      toolResults={message?.toolResults}
                    />
                    {message.isBot && message.sources && message.sources.length > 0 && (
                        <div className="ml-10 mt-1 flex items-center gap-1 text-xs text-text-muted">
                          <Tooltip
                            title={t(
                              "option:documentWorkspace.usesDocumentContext",
                              "Uses document context"
                            )}>
                            <span className="flex items-center gap-1">
                              <BookOpen className="h-3 w-3" />
                              <span>
                                {message.sources.length} source
                                {message.sources.length !== 1 ? "s" : ""}
                              </span>
                            </span>
                          </Tooltip>
                        </div>
                      )}
                  </div>
                ))}
              </Suspense>
              <div ref={messagesEndRef} />
            </div>
          )}

          {isChatReady && !hasMessages && suggestionsCollapsed && (
            <div className="flex h-full items-center justify-center p-4">
              <Empty
                description={t(
                  "option:documentWorkspace.startChatting",
                  "Ask a question about this document"
                )}
                image={Empty.PRESENTED_IMAGE_SIMPLE}
              />
            </div>
          )}
        </div>

        <div className="border-t border-border p-3">
          {waitingForChatDetails && (
            serverChatLoadState === "failed" ? (
              <div className="mb-2 flex items-center justify-between gap-2 text-xs">
                <p role="alert" className="text-error">
                  {t("option:documentWorkspace.chatDetailsError", "Could not load chat details.")}
                </p>
                <button
                  type="button"
                  onClick={retryConversation}
                  className="rounded border border-border px-2 py-1 hover:bg-hover focus-visible:outline focus-visible:outline-2 focus-visible:outline-primary"
                >
                  {t("common:retry", "Retry")}
                </button>
              </div>
            ) : (
              <p role="status" className="mb-2 text-xs text-text-muted">
                {t("option:documentWorkspace.loadingChatDetails", "Loading chat details…")}
              </p>
            )
          )}
          <div className="mb-2 flex items-center justify-between text-xs text-text-muted">
            <div className="flex items-center gap-2">
              <Switch
                size="small"
                checked={ragEnabled}
                disabled={!isChatReady}
                aria-label={useDocumentLabel}
                onChange={(checked) => setRagEnabled(checked)}
              />
              <span>{useDocumentLabel}</span>
            </div>
            {!hasSelectedModel && (
              <span className="text-warning">
                {t(
                  "option:documentWorkspace.selectModelHint",
                  "Set up AI chat in Settings"
                )}
              </span>
            )}
          </div>
          <div className="relative">
            <textarea
              ref={textareaRef}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={t(
                "option:documentWorkspace.chatPlaceholder",
                "Ask about this document..."
              )}
              aria-label={t("option:documentWorkspace.chatPlaceholder", "Ask about this document...")}
              disabled={!hasSelectedModel || !isChatReady || (isProcessing && !streaming)}
              rows={1}
              className="
                w-full resize-none rounded-lg border border-border bg-surface
                px-3 py-2 pr-10 text-sm text-text
                placeholder:text-text-muted
                focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary
                disabled:cursor-not-allowed disabled:opacity-50
              "
              style={{ minHeight: "40px", maxHeight: "120px" }}
            />

            {/* Send/Stop button */}
            <div className="absolute bottom-2 right-2">
              {streaming ? (
                <Tooltip title={t("common:stop", "Stop")}>
                  <button
                    type="button"
                    aria-label={t("common:stop", "Stop")}
                    onClick={stopConversationTurn}
                    className="rounded p-1 text-error transition-colors hover:bg-error/10"
                  >
                    <StopCircle className="h-5 w-5" />
                  </button>
                </Tooltip>
              ) : (
                <Tooltip title={t("common:send", "Send")}>
                  <button
                    type="button"
                    onClick={handleSend}
                    aria-label={t("common:send", "Send")}
                    disabled={!hasSelectedModel || !canSend || !inputValue.trim() || isProcessing}
                    className="
                      rounded p-1 text-primary transition-colors
                      hover:bg-primary/10
                      disabled:cursor-not-allowed disabled:opacity-50
                    "
                  >
                    <SendHorizontal className="h-5 w-5" />
                  </button>
                </Tooltip>
              )}
            </div>
          </div>

          {/* Document scope indicator */}
          <div className="mt-2 flex items-center gap-1 text-xs text-text-subtle">
            <BookOpen className="h-3 w-3" />
            <span>
              {ragEnabled
                ? t(
                    "option:documentWorkspace.scopedToDocument",
                    "Answers scoped to this document"
                  )
                : t(
                    "option:documentWorkspace.ragDisabled",
                    "Document content is not being used. Enable to get answers based on this document."
                  )}
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default DocumentChat
