import React from "react"
import { useTranslation } from "react-i18next"
import { Tooltip } from "antd"
import {
  Bot,
  PanelLeftClose,
  PanelLeftOpen,
  PanelRightClose,
  PanelRightOpen,
} from "lucide-react"
import { useACPSessionsStore } from "@/store/acp-sessions"
import { getSessionMessageCount, getSessionTokenUsage } from "./sessionMetrics"

interface ACPPlaygroundHeaderProps {
  leftPaneOpen: boolean
  rightPaneOpen: boolean
  onToggleLeftPane: () => void
  onToggleRightPane: () => void
  hideToggles?: boolean
  acpHealthy?: boolean
  isHealthLoading?: boolean
}

export const ACPPlaygroundHeader: React.FC<ACPPlaygroundHeaderProps> = ({
  leftPaneOpen,
  rightPaneOpen,
  onToggleLeftPane,
  onToggleRightPane,
  hideToggles = false,
  acpHealthy,
  isHealthLoading = false,
}) => {
  const { t } = useTranslation(["playground", "option", "common"])

  const activeSession = useACPSessionsStore((s) =>
    s.activeSessionId ? s.getSession(s.activeSessionId) : undefined
  )
  const activeSessionMessageCount = React.useMemo(
    () => (activeSession ? getSessionMessageCount(activeSession) : 0),
    [activeSession]
  )
  const activeSessionTokenUsage = React.useMemo(
    () => (activeSession ? getSessionTokenUsage(activeSession) : null),
    [activeSession]
  )

  const getStateColor = (state: string | undefined) => {
    switch (state) {
      case "connected":
        return "bg-success"
      case "running":
        return "bg-primary animate-pulse"
      case "waiting_permission":
        return "bg-warning animate-pulse"
      case "error":
        return "bg-error"
      case "connecting":
        return "bg-info animate-pulse"
      default:
        return "bg-text-muted"
    }
  }

  const getStateLabel = (state: string | undefined) => {
    switch (state) {
      case "connected":
        return t("playground:acp.state.connected", "Connected")
      case "running":
        return t("playground:acp.state.running", "Running")
      case "waiting_permission":
        return t("playground:acp.state.waitingPermission", "Awaiting Permission")
      case "error":
        return t("playground:acp.state.error", "Error")
      case "connecting":
        return t("playground:acp.state.connecting", "Connecting...")
      default:
        return t("playground:acp.state.disconnected", "Disconnected")
    }
  }

  return (
    <header className="flex items-center justify-between border-b border-border bg-surface px-4 py-3">
      <div className="flex items-center gap-3">
        {!hideToggles && (
          <Tooltip
            title={
              leftPaneOpen
                ? t("playground:acp.hideSessions", "Hide sessions")
                : t("playground:acp.showSessions", "Show sessions")
            }
          >
            <button
              type="button"
              onClick={onToggleLeftPane}
              className="hidden rounded-lg p-2 text-text-muted transition-colors hover:bg-surface2 hover:text-text lg:block"
              aria-pressed={leftPaneOpen}
            >
              {leftPaneOpen ? (
                <PanelLeftClose className="h-4 w-4" />
              ) : (
                <PanelLeftOpen className="h-4 w-4" />
              )}
            </button>
          </Tooltip>
        )}

        <div className="flex items-center gap-2">
          <Bot className="h-5 w-5 text-primary" />
          <div>
            <h1 className="flex items-center gap-2 text-lg font-semibold text-text">
              {t("playground:acp.title", "Agent Playground")}
              {!isHealthLoading && acpHealthy !== undefined && (
                <Tooltip
                  title={
                    acpHealthy
                      ? t("playground:acp.health.healthy", "ACP backend is healthy")
                      : t(
                          "playground:acp.health.unhealthy",
                          "ACP backend is not configured or unreachable"
                        )
                  }
                >
                  <span
                    className={`inline-block h-2.5 w-2.5 rounded-full ${
                      acpHealthy ? "bg-success" : "bg-error"
                    }`}
                    data-testid="acp-health-indicator"
                    tabIndex={0}
                    role="status"
                    aria-label={
                      acpHealthy
                        ? t("playground:acp.health.healthy", "ACP backend is healthy")
                        : t(
                            "playground:acp.health.unhealthy",
                            "ACP backend is not configured or unreachable"
                          )
                    }
                  >
                    <span className="sr-only">
                      {acpHealthy
                        ? t("playground:acp.health.healthy", "ACP backend is healthy")
                        : t(
                            "playground:acp.health.unhealthy",
                            "ACP backend is not configured or unreachable"
                          )}
                    </span>
                  </span>
                </Tooltip>
              )}
            </h1>
            <p className="text-xs text-text-muted">
              {t("playground:acp.subtitle", "Interact with AI coding agents via ACP")}
            </p>
          </div>
        </div>
      </div>

      <div className="flex items-center gap-3">
        {/* Session status indicator */}
        {activeSession && (
          <div className="flex flex-wrap items-center gap-1.5 rounded-lg bg-surface2 px-2.5 py-1.5">
            <span className="flex items-center gap-2 rounded bg-bg px-2 py-1">
              <span
                className={`h-2 w-2 rounded-full ${getStateColor(activeSession.state)}`}
              />
              <span className="text-sm text-text-muted">
                {getStateLabel(activeSession.state)}
              </span>
            </span>
            <span className="rounded bg-bg px-1.5 py-0.5 text-xs text-text-muted">
              {t("playground:acp.sessionMeta.messages", `Msgs ${activeSessionMessageCount}`)}
            </span>
            <span className="rounded bg-bg px-1.5 py-0.5 text-xs text-text-muted">
              {t(
                "playground:acp.sessionMeta.tokens",
                `Tokens ${activeSessionTokenUsage !== null ? activeSessionTokenUsage.toLocaleString() : "--"}`
              )}
            </span>
            <span className="rounded bg-bg px-1.5 py-0.5 text-xs text-text-muted">
              {t("playground:acp.sessionMeta.permissions", `Perm ${activeSession.pendingPermissions.length}`)}
            </span>
          </div>
        )}

        {!hideToggles && (
          <Tooltip
            title={
              rightPaneOpen
                ? t("playground:acp.hideTools", "Hide tools")
                : t("playground:acp.showTools", "Show tools")
            }
          >
            <button
              type="button"
              onClick={onToggleRightPane}
              className="hidden rounded-lg p-2 text-text-muted transition-colors hover:bg-surface2 hover:text-text lg:block"
              aria-pressed={rightPaneOpen}
            >
              {rightPaneOpen ? (
                <PanelRightClose className="h-4 w-4" />
              ) : (
                <PanelRightOpen className="h-4 w-4" />
              )}
            </button>
          </Tooltip>
        )}
      </div>
    </header>
  )
}
