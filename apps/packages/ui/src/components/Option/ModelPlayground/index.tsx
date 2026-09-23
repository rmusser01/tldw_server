import React from "react"
import { useTranslation } from "react-i18next"
import { useNavigate } from "react-router-dom"
import { PanelRightClose, PanelRightOpen, Bug, FlaskConical } from "lucide-react"
import { Tooltip } from "antd"
import { useStorage } from "@plasmohq/storage/hook"
import { ModelPlaygroundChat } from "./ModelPlaygroundChat"
import { ParametersSidebar } from "./ParametersSidebar"
import { DebugPanel } from "./DebugPanel"
import { useMessageOption } from "@/hooks/useMessageOption"
import { useSmartScroll } from "@/hooks/useSmartScroll"
import { ChevronDown } from "lucide-react"
import {
  MODEL_PLAYGROUND_DEBUG_KEY,
  MODEL_PLAYGROUND_SIDEBAR_KEY
} from "@/utils/storage-migrations"

/**
 * ModelPlayground - Developer-focused chat interface
 *
 * Features:
 * - Compare mode (multi-model side-by-side) - always available
 * - Full model parameters panel
 * - Full RAG settings (all 30+ options)
 * - Debug panel (traces, tokens, timing)
 * - System prompt editor
 */
export const ModelPlayground: React.FC = () => {
  const { t } = useTranslation(["playground", "option", "common"])
  const navigate = useNavigate()
  const [sidebarOpen, setSidebarOpen] = useStorage(MODEL_PLAYGROUND_SIDEBAR_KEY, true)
  const [debugOpen, setDebugOpen] = useStorage(MODEL_PLAYGROUND_DEBUG_KEY, false)

  const { messages, streaming } = useMessageOption({
    hydrateServerChat: true,
    forceCompareEnabled: true
  })
  const { containerRef, isAutoScrollToBottom, autoScrollToBottom } =
    useSmartScroll(messages, streaming, 120)

  const handleGoToChat = () => {
    navigate("/")
  }

  return (
    <div className="flex h-full flex-col bg-bg text-text">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-border bg-surface px-4 py-3">
        <div className="flex items-center gap-3">
          <FlaskConical className="h-5 w-5 text-primary" />
          <div>
            <h1 className="text-lg font-semibold text-text">
              {t("playground:workspace.title", "Model Playground")}
            </h1>
            <p className="text-xs text-text-muted">
              {t(
                "playground:workspace.subtitle",
                "Developer chat with compare mode and full controls"
              )}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Tooltip title={t("playground:workspace.toggleDebug", "Toggle debug panel")}>
            <button
              type="button"
              onClick={() => setDebugOpen(!debugOpen)}
              className={`rounded-lg p-2 transition-colors ${
                debugOpen
                  ? "bg-primary/10 text-primary"
                  : "text-text-muted hover:bg-surface2 hover:text-text"
              }`}
              aria-pressed={debugOpen}
            >
              <Bug className="h-4 w-4" />
            </button>
          </Tooltip>
          <Tooltip
            title={
              sidebarOpen
                ? t("playground:workspace.hideSidebar", "Hide sidebar")
                : t("playground:workspace.showSidebar", "Show sidebar")
            }
          >
            <button
              type="button"
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="rounded-lg p-2 text-text-muted transition-colors hover:bg-surface2 hover:text-text"
              aria-pressed={sidebarOpen}
            >
              {sidebarOpen ? (
                <PanelRightClose className="h-4 w-4" />
              ) : (
                <PanelRightOpen className="h-4 w-4" />
              )}
            </button>
          </Tooltip>
          <button
            type="button"
            onClick={handleGoToChat}
            className="ml-2 rounded-lg border border-border bg-surface px-3 py-1.5 text-sm font-medium text-text transition hover:bg-surface2"
          >
            {t("playground:workspace.goToSimpleChat", "Simple Chat")}
          </button>
        </div>
      </div>

      {/* Main content */}
      <div className="flex min-h-0 flex-1">
        {/* Chat area */}
        <div className="relative flex min-w-0 flex-1 flex-col">
          <div
            ref={containerRef}
            role="log"
            aria-live="polite"
            aria-relevant="additions"
            aria-label={t("playground:aria.chatTranscript", "Chat messages")}
            className="custom-scrollbar flex-1 min-h-0 w-full overflow-x-hidden overflow-y-auto px-4"
          >
            <div className="mx-auto w-full max-w-4xl pb-6">
              <ModelPlaygroundChat />
            </div>
          </div>
          {/* Scroll to bottom button */}
          {!isAutoScrollToBottom && (
            <div className="pointer-events-none absolute bottom-24 left-0 right-0 flex justify-center">
              <button
                onClick={() => autoScrollToBottom()}
                aria-label={t(
                  "playground:composer.scrollToLatest",
                  "Scroll to latest messages"
                )}
                title={
                  t(
                    "playground:composer.scrollToLatest",
                    "Scroll to latest messages"
                  ) as string
                }
                className="pointer-events-auto rounded-full border border-border bg-surface p-2 text-text-subtle shadow-card transition-colors hover:bg-surface2 focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
              >
                <ChevronDown className="size-4 text-text-subtle" aria-hidden="true" />
              </button>
            </div>
          )}
        </div>

        {/* Right sidebar - Parameters */}
        {sidebarOpen && (
          <div className="hidden w-80 shrink-0 border-l border-border bg-surface lg:block">
            <ParametersSidebar />
          </div>
        )}
      </div>

      {/* Debug panel - bottom drawer */}
      {debugOpen && (
        <div className="h-64 shrink-0 border-t border-border bg-surface">
          <DebugPanel onClose={() => setDebugOpen(false)} />
        </div>
      )}
    </div>
  )
}

export default ModelPlayground
