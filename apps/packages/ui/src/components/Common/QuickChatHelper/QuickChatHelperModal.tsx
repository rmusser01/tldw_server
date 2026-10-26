import { flushSync } from "react-dom"
import { EMPTY_CHAT_MODELS, useChatModelCatalog } from "@/hooks/useChatModelCatalog"
import React, { useRef, useEffect, useCallback, useState } from "react"
import { Modal, Button, Tooltip, Select, Segmented } from "antd"
import { ExternalLink, AlertCircle } from "lucide-react"
import { useTranslation } from "react-i18next"
import { useQuickChat } from "@/hooks/useQuickChat"
import { useQuickChatStore } from "@/store/quick-chat"
import { QuickChatMessage } from "./QuickChatMessage"
import { QuickChatInput } from "./QuickChatInput"
import { browser } from "wxt/browser"
import { useConnectionPhase, useIsConnected } from "@/hooks/useConnectionState"
import { ConnectionPhase } from "@/types/connection"
import { useChatModelsSelect } from "@/hooks/useChatModelsSelect"
import { QuickChatGuidesPanel } from "./QuickChatGuidesPanel"
import { buildQuickChatPopoutState } from "./popout-state"
import { useLocation, useNavigate } from "react-router-dom"
import { useStartTutorial } from "@/hooks/useStartTutorial"
import { smoothScrollBehavior } from "@/utils/reduced-motion"

const EMPTY_POP_OUT_TOOLTIP_STYLES = {
  root: { maxWidth: "200px" }
}

type Props = {
  open: boolean
  onClose: () => void
  getTutorialFocusTarget?: () => HTMLElement | null
}

export const QuickChatHelperModal: React.FC<Props> = ({ open, onClose, getTutorialFocusTarget }) => {
  const { t } = useTranslation(["option", "common"])
  const navigate = useNavigate()
  const location = useLocation()
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const { assistantMode, setAssistantMode } = useQuickChatStore()
  const startTutorial = useStartTutorial()
  const [handedOff, setHandedOff] = useState(false)
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
  const phase = useConnectionPhase()
  const isConnected = useIsConnected()
  const isConnectionReady = isConnected && phase === ConnectionPhase.CONNECTED
  const { data: models = EMPTY_CHAT_MODELS, isLoading: modelsLoading, isError: modelsDiscoveryFailed, refetch: retryModels } = useChatModelCatalog()

  const { allowClear, modelOptions, modelPlaceholder, handleModelChange } =
    useChatModelsSelect({
      models,
      currentModel,
      modelOverride,
      setModelOverride,
      t
    })

  const getModalContainer = useCallback(() => {
    if (typeof document === "undefined") return null
    return document.getElementById("tldw-portal-root") || document.body
  }, [])

  // Scroll to bottom when messages change or streaming completes
  useEffect(() => {
    if (!messagesEndRef.current || messages.length === 0) {
      return
    }
    messagesEndRef.current.scrollIntoView({ behavior: smoothScrollBehavior() })
  }, [messages, isStreaming])

  const handlePopOut = useCallback(() => {
    // Serialize current state to sessionStorage
    const state = useQuickChatStore.getState().getSerializableState()
    const popoutState = buildQuickChatPopoutState(state, location.pathname)
    const stateKey = `quickchat_${Date.now()}`
    sessionStorage.setItem(stateKey, JSON.stringify(popoutState))

    // Open pop-out window
    const popoutUrl = browser.runtime.getURL(
      `/options.html#/quick-chat-popout?state=${stateKey}`
    )
    const popoutWindow = window.open(
      popoutUrl,
      "quickChatHelper",
      "width=480,height=600,menubar=no,toolbar=no,location=no,status=no"
    )

    if (popoutWindow) {
      useQuickChatStore.getState().setPopoutWindow(popoutWindow)
      onClose()
    }
  }, [location.pathname, onClose])

  const title = t("option:quickChatHelper.title", "Quick Chat Helper")
  const emptyState = t(
    "option:quickChatHelper.emptyState",
    "Start a quick side chat to keep your main thread clean."
  )
  const popOutLabel = t("option:quickChatHelper.popOutButton", "Pop out")
  const popOutDisabledTooltip = t(
    "option:quickChatHelper.popOutDisabled",
    "Pop-out is disabled when there are no messages. Start a conversation first."
  )
  const connectionLabel = isConnectionReady
    ? t("common:connected", "Connected")
    : t("common:notConnected", "Not connected")
  const connectionBadgeClass = isConnectionReady
    ? "border-success/30 bg-success/10 text-success"
    : "border-warn/30 bg-warn/10 text-warn"
  const connectionDotClass = isConnectionReady
    ? "bg-success"
    : "bg-warn"
  const descriptionId =
    messages.length === 0 && assistantMode !== "browse_guides"
      ? "quick-chat-description"
      : undefined
  const docsModeActive = assistantMode === "docs_rag"
  const browseModeActive = assistantMode === "browse_guides"

  const handleModeChange = useCallback(
    (value: string | number) => {
      if (value === "chat" || value === "docs_rag" || value === "browse_guides") {
        setAssistantMode(value)
      }
    },
    [setAssistantMode]
  )

  const handleSendFromInput = useCallback(
    (message: string) => {
      void sendMessage(message, {
        mode: docsModeActive ? "docs_rag" : "chat",
        currentRoute: location.pathname
      })
    },
    [docsModeActive, location.pathname, sendMessage]
  )

  const openWorkflowRoute = useCallback(
    (route: string) => {
      const normalized = route.startsWith("/") ? route : `/${route}`
      const isOptionsPage =
        typeof window !== "undefined" && /options\.html$/i.test(window.location.pathname)
      if (isOptionsPage) {
        navigate(normalized)
        return
      }
      const optionsUrl = browser.runtime.getURL(`/options.html#${normalized}`)
      window.open(optionsUrl, "_blank")
    },
    [navigate]
  )

  const handleAskGuide = useCallback(
    (question: string) => {
      setAssistantMode("docs_rag")
      void sendMessage(question, {
        mode: "docs_rag",
        currentRoute: location.pathname
      })
    },
    [location.pathname, sendMessage, setAssistantMode]
  )

  const handleStartTutorial = useCallback(
    (tutorialId: string) => {
      if (!startTutorial(tutorialId, getTutorialFocusTarget?.())) return
      // Commit the handoff before the parent unmounts this modal.
      flushSync(() => setHandedOff(true))
      onClose()
    },
    [onClose, startTutorial, getTutorialFocusTarget]
  )

  return (
    <Modal
      title={
        <div className="flex items-center justify-between pr-8">
          <span id="quick-chat-title">{title}</span>
          <Tooltip
            title={messages.length === 0 ? popOutDisabledTooltip : popOutLabel}
            styles={
              messages.length === 0 ? EMPTY_POP_OUT_TOOLTIP_STYLES : undefined
            }>
            <Button
              type="text"
              size="small"
              icon={<ExternalLink className="h-4 w-4" />}
              onClick={handlePopOut}
              aria-label={popOutLabel}
              disabled={messages.length === 0}
            />
          </Tooltip>
        </div>
      }
      open={open}
      onCancel={onClose}
      focusable={{ focusTriggerAfterClose: !handedOff }}
      footer={null}
      width={480}
      className="quick-chat-helper-modal"
      getContainer={getModalContainer}
      destroyOnHidden={false}
      maskClosable={true}
      keyboard={true}
      aria-labelledby="quick-chat-title"
      aria-describedby={descriptionId}>
      <div className="flex flex-col h-[50vh] max-h-[400px]">
        {modelsDiscoveryFailed && (
          <div role="alert" className="px-1 pb-2 text-xs text-text">
            {t("playground:composer.modelsDiscoveryFailed", "Could not check chat models")}
            <Button size="small" onClick={() => { void retryModels() }}>
              {t("playground:characterChatReadiness.model.retryAction", "Try again")}
            </Button>
          </div>
        )}
        <div className="flex items-center gap-2 px-1 pb-2">
          <Select
            className="flex-1"
            size="small"
            showSearch
            options={modelOptions}
            value={activeModel || undefined}
            placeholder={modelPlaceholder}
            loading={modelsLoading}
            optionFilterProp="label"
            allowClear={allowClear}
            aria-label={t("option:quickChatHelper.modelLabel", "Model")}
            onChange={handleModelChange}
          />
          <span
            role="status"
            aria-live="polite"
            className={`inline-flex items-center gap-1 rounded-full border px-2 py-1 text-[11px] font-medium ${connectionBadgeClass}`}
          >
            <span className={`h-1.5 w-1.5 rounded-full ${connectionDotClass}`} />
            <span>{connectionLabel}</span>
          </span>
        </div>
        <div className="px-1 pb-2">
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
        {/* Messages area */}
        {browseModeActive ? (
          <div className="flex-1 overflow-hidden px-1 py-1">
            <QuickChatGuidesPanel
              onAskGuide={handleAskGuide}
              onOpenRoute={openWorkflowRoute}
              askDisabled={false}
              currentRoute={location.pathname}
              onStartTutorial={handleStartTutorial}
            />
          </div>
        ) : (
          <div
            className="flex-1 overflow-y-auto px-1 py-2"
            role="log"
            aria-live="polite"
            aria-label={t("common:chatMessages", "Chat messages")}>
            {messages.length === 0 ? (
              <div
                id="quick-chat-description"
                className="flex flex-col items-center justify-center h-full text-text-subtle text-center px-4">
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
                  <QuickChatMessage
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
        )}
      </div>
    </Modal>
  )
}

export default QuickChatHelperModal
