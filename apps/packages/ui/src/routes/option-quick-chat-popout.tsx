import { useChatModelCatalog } from "@/hooks/useChatModelCatalog"
import React, { useEffect, useRef, useState } from "react"
import { Select, Segmented } from "antd"
import { useNavigate, useSearchParams } from "react-router-dom"
import { useTranslation } from "react-i18next"
import { useQuickChatStore } from "@/store/quick-chat"
import { useQuickChat } from "@/hooks/useQuickChat"
import { QuickChatMessage as QuickChatMessageView } from "@/components/Common/QuickChatHelper/QuickChatMessage"
import { QuickChatInput } from "@/components/Common/QuickChatHelper/QuickChatInput"
import { AlertCircle } from "lucide-react"
import { useChatModelsSelect } from "@/hooks/useChatModelsSelect"
import { QuickChatGuidesPanel } from "@/components/Common/QuickChatHelper/QuickChatGuidesPanel"
import { parseQuickChatPopoutState } from "@/components/Common/QuickChatHelper/popout-state"

const QuickChatPopout: React.FC = () => {
  const { t } = useTranslation(["option", "common"])
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const hasRestoredRef = useRef(false)
  const [sourceRouteContext, setSourceRouteContext] = useState<string | null>(null)
  const { assistantMode, setAssistantMode } = useQuickChatStore()

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
    isError: modelsError,
    refetch: retryModels
  } = useChatModelCatalog()
  const models = modelsData ?? []
  const showModelsError = modelsError && models.length === 0
  const modelsErrorHintId = "quick-chat-models-error-hint"

  // Restore state from sessionStorage on mount
  useEffect(() => {
    if (hasRestoredRef.current) return

    const stateKey = searchParams.get("state")
    if (!stateKey) {
      hasRestoredRef.current = true
      return
    }
    if (!/^[a-zA-Z0-9:_-]{1,128}$/.test(stateKey)) {
      console.warn("Invalid quick chat state key")
      hasRestoredRef.current = true
      return
    }
    try {
      const savedState = sessionStorage.getItem(stateKey)
      if (!savedState) return

      const parsed = JSON.parse(savedState) as unknown
      const parsedState = parseQuickChatPopoutState(parsed)
      if (parsedState) {
        useQuickChatStore.getState().restoreFromState({
          messages: parsedState.messages,
          modelOverride: parsedState.modelOverride,
          assistantMode: parsedState.assistantMode
        })
        setSourceRouteContext(parsedState.sourceRoute)
      } else {
        console.warn("Invalid quick chat state structure in sessionStorage")
      }
    } catch (error) {
      console.error("Failed to restore quick chat state:", error)
    } finally {
      // Clean up sessionStorage regardless of validity or parse errors
      sessionStorage.removeItem(stateKey)
      hasRestoredRef.current = true
    }
  }, [searchParams])

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
  const docsModeActive = assistantMode === "docs_rag"
  const browseModeActive = assistantMode === "browse_guides"
  const currentRouteContext = sourceRouteContext
  const handleModeChange = (value: string | number) => {
    if (value === "chat" || value === "docs_rag" || value === "browse_guides") {
      setAssistantMode(value)
    }
  }
  const handleSendFromInput = (message: string) => {
    void sendMessage(message, {
      mode: docsModeActive ? "docs_rag" : "chat",
      currentRoute: currentRouteContext
    })
  }
  const openWorkflowRoute = (route: string) => {
    const normalized = route.startsWith("/") ? route : `/${route}`
    navigate(normalized)
  }
  const handleAskGuide = (question: string) => {
    setAssistantMode("docs_rag")
    void sendMessage(question, {
      mode: "docs_rag",
      currentRoute: currentRouteContext
    })
  }

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
            id={modelsErrorHintId}
            role="alert">
            <AlertCircle className="h-4 w-4" />
            <span>
              {t(
                "option:quickChatHelper.modelsLoadErrorHint",
                "Check your server connection, then try again."
              )}
            </span>
            <button type="button" className="underline min-h-[44px]" onClick={() => { void retryModels() }}>
              {t("playground:characterChatReadiness.model.retryAction", "Try again")}
            </button>
          </div>
        )}
        <div className="mt-2">
          <Segmented
            block
            size="small"
            value={assistantMode}
            onChange={handleModeChange}
            options={[
              {
                value: "chat",
                label: t("option:quickChatHelper.mode.chat", "Chat")
              },
              {
                value: "docs_rag",
                label: t("option:quickChatHelper.mode.docs", "Docs Q&A")
              },
              {
                value: "browse_guides",
                label: t("option:quickChatHelper.mode.guides", "Browse Guides")
              }
            ]}
          />
        </div>
      </div>

      {browseModeActive ? (
        <div className="flex-1 overflow-hidden px-4 py-3">
          <QuickChatGuidesPanel
            onAskGuide={handleAskGuide}
            onOpenRoute={openWorkflowRoute}
            askDisabled={false}
            currentRoute={currentRouteContext}
          />
        </div>
      ) : (
        <div
          className="flex-1 overflow-y-auto px-4 py-3"
          role="log"
          aria-live="polite"
          aria-label={t("common:chatMessages", "Chat messages")}>
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-text-muted text-center px-4">
              <p className="text-sm">
                {docsModeActive
                  ? t(
                      "option:quickChatHelper.docsEmptyState",
                      "Ask what you want to achieve and I will search indexed docs to suggest tools and steps."
                    )
                  : emptyState}
              </p>
              {!hasModel && !docsModeActive && (
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
      )}

      {/* Input area */}
      {!browseModeActive && (
        <div className="px-4 py-3 border-t border-border bg-surface">
          <QuickChatInput
            onSend={handleSendFromInput}
            onCancel={cancelStream}
            isStreaming={isStreaming}
            disabled={docsModeActive ? false : !hasModel}
            placeholder={
              docsModeActive
                ? t(
                    "option:quickChatHelper.docsInputPlaceholder",
                    "Ask about a workflow or feature and I will search docs..."
                  )
                : undefined
            }
          />
        </div>
      )}
    </div>
  )
}

export default QuickChatPopout
