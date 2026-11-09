import { useQuickChatPopoutRestore } from "@/components/Common/QuickChatHelper/popout-state"
import React, { useEffect, useRef } from "react"
import { Select } from "antd"
import { useQuery } from "@tanstack/react-query"
import { useSearchParams } from "react-router-dom"
import { useTranslation } from "react-i18next"
import { useQuickChatStore } from "@/store/quick-chat"
import { useQuickChat } from "@/hooks/useQuickChat"
import { fetchChatModels } from "@/services/tldw-server"
import { QuickChatMessage as QuickChatMessageView } from "@/components/Common/QuickChatHelper/QuickChatMessage"
import { QuickChatInput } from "@/components/Common/QuickChatHelper/QuickChatInput"
import { AlertCircle } from "lucide-react"
import { useChatModelsSelect } from "@/hooks/useChatModelsSelect"

const QuickChatPopout: React.FC = () => {
  const { t } = useTranslation(["option", "common"])
  const [searchParams] = useSearchParams()
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const {
    messages,
    sendMessage,
    cancelStream,
    isStreaming,
    hasModel,
    activeModel,
    currentModel,
    modelOverride,
    setModelOverride
  } = useQuickChat()
  const {
    data: modelsData,
    isLoading: modelsLoading,
    isError: modelsError
  } = useQuery({
    queryKey: ["quickChatModels"],
    queryFn: () => fetchChatModels({ returnEmpty: false }),
    staleTime: 5 * 60 * 1000,
    gcTime: 5 * 60 * 1000,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
    select: (data) => data.filter((model) => model?.model)
  })
  const models = modelsData ?? []
  const showModelsError = modelsError && models.length === 0
  const modelsErrorHintId = "quick-chat-models-error-hint"

  useQuickChatPopoutRestore(searchParams.get("state"), parsedState => {
    useQuickChatStore.getState().restoreFromState(parsedState)
  })

  // Scroll to bottom when messages change
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" })
    }
  }, [messages])

  // Clear messages and cancel any active stream on window close
  useEffect(() => {
    const handleBeforeUnload = () => {
      cancelStream()
      useQuickChatStore.getState().clearMessages()
    }
    window.addEventListener("beforeunload", handleBeforeUnload)
    return () => {
      window.removeEventListener("beforeunload", handleBeforeUnload)
    }
  }, [cancelStream])

  const title = t("option:quickChatHelper.title", "Quick Chat Helper")
  const emptyState = t(
    "option:quickChatHelper.emptyState",
    "Start a quick side chat to keep your main thread clean."
  )
  let notFoundContent: string | undefined
  if (showModelsError) {
    notFoundContent = t(
      "option:quickChatHelper.modelsLoadError",
      "Unable to load models"
    )
  } else if (models.length === 0 && !modelsLoading) {
    notFoundContent = t(
      "option:quickChatHelper.noModelsAvailable",
      "No models available"
    )
  }
  const { allowClear, modelOptions, modelPlaceholder, handleModelChange } =
    useChatModelsSelect({
      models,
      currentModel,
      modelOverride,
      setModelOverride,
      t
    })

  return (
    <div className="flex flex-col h-screen bg-bg">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border bg-surface">
        <h1 className="text-lg font-semibold text-text">
          {title}
        </h1>
      </div>
      <div className="px-4 py-2 border-b border-border bg-bg">
        <Select
          className="w-full"
          size="small"
          showSearch
          options={modelOptions}
          value={activeModel || undefined}
          placeholder={modelPlaceholder}
          loading={modelsLoading}
          disabled={showModelsError}
          status={showModelsError ? "error" : undefined}
          optionFilterProp="label"
          allowClear={allowClear}
          aria-label={t("option:quickChatHelper.modelLabel", "Model")}
          aria-describedby={showModelsError ? modelsErrorHintId : undefined}
          onChange={handleModelChange}
          notFoundContent={notFoundContent}
        />
        {showModelsError && (
          <div
            className="mt-2 flex items-center gap-2 text-xs text-warn"
            id={modelsErrorHintId}>
            <AlertCircle className="h-4 w-4" />
            <span>
              {t(
                "option:quickChatHelper.modelsLoadErrorHint",
                "Check your server connection, then try again."
              )}
            </span>
          </div>
        )}
      </div>

      {/* Messages area */}
      <div
        className="flex-1 overflow-y-auto px-4 py-3"
        role="log"
        aria-live="polite"
        aria-label={t("common:chatMessages", "Chat messages")}>
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-text-muted text-center px-4">
            <p className="text-sm">{emptyState}</p>
            {!hasModel && (
              <div className="mt-3 flex items-center gap-2 text-warn text-xs">
                <AlertCircle className="h-4 w-4" />
                <span>
                  {t(
                    "option:quickChatHelper.noModelWarning",
                    "Select a model in the main chat or choose one here."
                  )}
                </span>
              </div>
            )}
          </div>
        ) : (
          <>
            {messages.map((message, index) => (
              <QuickChatMessageView
                key={message.id}
                message={message}
                isStreaming={isStreaming}
                isLast={index === messages.length - 1}
              />
            ))}
            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      {/* Input area */}
      <div className="px-4 py-3 border-t border-border bg-surface">
        <QuickChatInput
          onSend={sendMessage}
          onCancel={cancelStream}
          isStreaming={isStreaming}
          disabled={!hasModel}
        />
      </div>
    </div>
  )
}

export default QuickChatPopout
