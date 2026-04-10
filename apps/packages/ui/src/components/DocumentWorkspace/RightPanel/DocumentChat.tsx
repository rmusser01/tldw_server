import React, { useState, useRef, useCallback, Suspense, useEffect } from "react"
import { useTranslation } from "react-i18next"
import { Empty, Spin, Switch, Tooltip, Popconfirm } from "antd"
import {
  SendHorizontal,
  Trash2,
  AlertCircle,
  BookOpen,
  StopCircle,
  Sparkles
} from "lucide-react"
import { useDocumentWorkspaceStore } from "@/store/document-workspace"
import { useDocumentChat } from "@/hooks/document-workspace/useDocumentChat"
import { useDocumentMetadata } from "@/hooks/document-workspace/useDocumentMetadata"
import { useMessageOption } from "@/hooks/useMessageOption"
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
export const DocumentChat: React.FC = () => {
  const { t } = useTranslation(["option", "common", "playground"])
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const [inputValue, setInputValue] = useState("")

  // Get active document from workspace store
  const activeDocumentId = useDocumentWorkspaceStore((s) => s.activeDocumentId)
  const activeDocumentType = useDocumentWorkspaceStore(
    (s) => s.activeDocumentType
  )

  // Use the document chat hook to scope RAG
  const {
    messages,
    streaming,
    isProcessing,
    isServerAvailable,
    clearDocumentChat,
    ragEnabled,
    setRagEnabled
  } = useDocumentChat(activeDocumentId)

  // Get full message option for onSubmit and other actions
  const {
    onSubmit,
    stopStreamingRequest,
    regenerateLastMessage,
    editMessage,
    deleteMessage,
    ttsEnabled,
    temporaryChat,
    serverChatId,
    actionInfo,
    selectedModel
  } = useMessageOption()

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
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" })
    }
  }, [messages.length])

  React.useEffect(() => {
    setSuggestionsCollapsed(false)
    setInputValue("")
  }, [activeDocumentId])

  // Listen for "Ask AI" events from the text selection popover
  useEffect(() => {
    const handleAskAI = (event: CustomEvent<{ text: string; prompt: string }>) => {
      const { prompt } = event.detail
      setInputValue(prompt)
      // Focus the input after a short delay to ensure the tab switch has completed
      setTimeout(() => {
        textareaRef.current?.focus()
      }, 100)
    }

    window.addEventListener(
      "document-workspace-ask-ai",
      handleAskAI as EventListener
    )
    return () => {
      window.removeEventListener(
        "document-workspace-ask-ai",
        handleAskAI as EventListener
      )
    }
  }, [])

  // Handle sending a message
  const handleSend = useCallback(async () => {
    const trimmedMessage = inputValue.trim()
    if (!trimmedMessage || isProcessing || streaming || !hasSelectedModel) {
      return
    }

    // Collapse suggestions after first message
    if (!suggestionsCollapsed) {
      setSuggestionsCollapsed(true)
    }

    setInputValue("")

    await onSubmit({
      message: trimmedMessage,
      image: ""
    })
  }, [
    hasSelectedModel,
    inputValue,
    isProcessing,
    onSubmit,
    streaming,
    suggestionsCollapsed
  ])

  // Handle suggested question click - auto-send
  const handleQuestionClick = useCallback(
    async (question: string) => {
      if (!hasSelectedModel || isProcessing || streaming) return
      setSuggestionsCollapsed(true)
      await onSubmit({
        message: question,
        image: ""
      })
    },
    [hasSelectedModel, isProcessing, streaming, onSubmit]
  )

  // Handle keyboard events in textarea
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  // Handle clearing the chat
  const handleClearChat = useCallback(() => {
    clearDocumentChat()
    setSuggestionsCollapsed(false)
    setInputValue("")
  }, [clearDocumentChat])

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

  const hasMessages = messages.length > 0
  const showSuggestions = !suggestionsCollapsed

  return (
    <div className="flex h-full flex-col">
      {/* Chat header with clear button */}
      {hasMessages && (
        <div className="flex items-center justify-between border-b border-border px-3 py-2">
          <span className="text-xs text-text-muted">
            {t("option:documentWorkspace.chatWithDocument", "Chat with document")}
          </span>
          <Popconfirm
            title={t("option:documentWorkspace.clearChatConfirm", "Clear chat history?")}
            onConfirm={handleClearChat}
            okText={t("common:clear", "Clear")}
            cancelText={t("common:cancel", "Cancel")}
            okButtonProps={{ danger: true }}
            placement="bottomRight"
          >
            <Tooltip
              title={t("common:clearChat", "Clear chat")}
              placement="left"
            >
              <button
                type="button"
                disabled={isProcessing || streaming}
                className="rounded p-1 text-text-muted transition-colors hover:bg-hover hover:text-text disabled:opacity-50"
              >
                <Trash2 className="h-4 w-4" />
              </button>
            </Tooltip>
          </Popconfirm>
        </div>
      )}

      {/* Show suggestions button when collapsed */}
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

      {/* Messages area */}
      <div className="flex-1 overflow-y-auto">
        {showSuggestions && (
          <SuggestedQuestions
            documentType={activeDocumentType}
            variant={questionVariant}
            onQuestionClick={handleQuestionClick}
            disabled={isProcessing || streaming}
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
                    onRegenerate={regenerateLastMessage}
                    isProcessing={isProcessing}
                    sources={message.sources}
                    onEditFormSubmit={(idx, value, isUser, isSend) => {
                      editMessage(idx, value, isUser, isSend)
                    }}
                    onDeleteMessage={(idx) => {
                      deleteMessage(idx)
                    }}
                    isTTSEnabled={ttsEnabled}
                    generationInfo={message?.generationInfo}
                    isStreaming={streaming}
                    modelImage={message?.modelImage}
                    modelName={message?.modelName}
                    createdAt={message?.createdAt}
                    temporaryChat={temporaryChat}
                    onStopStreaming={stopStreamingRequest}
                    onContinue={() => {
                      onSubmit({
                        image: "",
                        message: "",
                        isContinue: true
                      })
                    }}
                    documents={message?.documents}
                    actionInfo={actionInfo}
                    serverChatId={serverChatId}
                    serverMessageId={message.serverMessageId}
                    messageId={message.id}
                    discoSkillComment={message.discoSkillComment}
                    conversationInstanceId={`doc-${activeDocumentId}`}
                    hideCopy={false}
                    hideEditAndRegenerate={false}
                    toolCalls={message?.toolCalls}
                    toolResults={message?.toolResults}
                  />
                  {message.isBot && message.sources && message.sources.length > 0 && (
                    <div className="ml-10 mt-1 flex items-center gap-1 text-xs text-text-muted">
                      <Tooltip title={t("option:documentWorkspace.usesDocumentContext", "Uses document context")}>
                        <span className="flex items-center gap-1">
                          <BookOpen className="h-3 w-3" />
                          <span>{message.sources.length} source{message.sources.length !== 1 ? "s" : ""}</span>
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

        {!hasMessages && suggestionsCollapsed && (
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

      {/* Input area */}
      <div className="border-t border-border p-3">
        <div className="mb-2 flex items-center justify-between text-xs text-text-muted">
          <div className="flex items-center gap-2">
            <Switch
              size="small"
              checked={ragEnabled}
              onChange={(checked) => setRagEnabled(checked)}
            />
            <span>
              {t(
                "option:documentWorkspace.enableRag",
                "Use document content"
              )}
            </span>
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
            disabled={!hasSelectedModel || !isServerAvailable || (isProcessing && !streaming)}
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
                  onClick={stopStreamingRequest}
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
                  disabled={!hasSelectedModel || !inputValue.trim() || isProcessing}
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
  )
}

export default DocumentChat
