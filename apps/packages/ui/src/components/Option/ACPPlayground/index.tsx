import React, { Suspense, useEffect } from "react"
import { useTranslation } from "react-i18next"
import { useQuery } from "@tanstack/react-query"
import { useStorage } from "@plasmohq/storage/hook"
import { Drawer, Tabs } from "antd"
import { Bot, MessageSquare, Wrench, Terminal } from "lucide-react"
import { useCanonicalConnectionConfig } from "@/hooks/useCanonicalConnectionConfig"
import { useMobile } from "@/hooks/useMediaQuery"
import { useACPSession } from "@/hooks/useACPSession"
import { ACPRestClient } from "@/services/acp/client"
import { buildACPClientConfig, buildACPAuthHeaders } from "@/services/acp/connection"
import { useACPSessionsStore } from "@/store/acp-sessions"
import type { ACPPermissionTier } from "@/services/acp/types"
import { ACPPlaygroundHeader } from "./ACPPlaygroundHeader"
import { ACPChatPanel } from "./ACPChatPanel"

const ACPSessionPanel = React.lazy(() =>
  import("./ACPSessionPanel").then((module) => ({ default: module.ACPSessionPanel }))
)
const ACPToolsPanel = React.lazy(() =>
  import("./ACPToolsPanel").then((module) => ({ default: module.ACPToolsPanel }))
)
const ACPWorkspacePanel = React.lazy(() =>
  import("./ACPWorkspacePanel").then((module) => ({ default: module.ACPWorkspacePanel }))
)
const ACPPermissionModal = React.lazy(() =>
  import("./ACPPermissionModal").then((module) => ({ default: module.ACPPermissionModal }))
)

const ACP_LEFT_PANE_KEY = "acp-playground-left-pane"
const ACP_RIGHT_PANE_KEY = "acp-playground-right-pane"
const ACP_SESSION_DETAIL_VIEWS = new Set([
  "session",
  "diagnostics",
  "artifacts",
  "audit",
  "events"
])

const readACPPlaygroundSearch = (): string => {
  if (typeof window === "undefined") return ""
  if (window.location.search) return window.location.search

  const hash = window.location.hash || ""
  const queryIndex = hash.indexOf("?")
  return queryIndex >= 0 ? hash.slice(queryIndex) : ""
}

const normalizeQueryValue = (value: string | null): string | null => {
  const trimmed = value?.trim()
  return trimmed ? trimmed : null
}

/**
 * ACPPlayground - Agent Client Protocol interface
 *
 * Enables interaction with downstream agents like Claude Code through ACP.
 *
 * Layout:
 * - Left pane: Session list and creation
 * - Center: Agent chat/conversation view
 * - Right pane: Tools and capabilities display
 */
export const ACPPlayground: React.FC = () => {
  const { t } = useTranslation(["playground", "option", "common"])
  const isMobile = useMobile()

  // Pane state with persistence
  const [leftPaneOpen, setLeftPaneOpen] = useStorage(ACP_LEFT_PANE_KEY, true)
  const [rightPaneOpen, setRightPaneOpen] = useStorage(ACP_RIGHT_PANE_KEY, true)
  const { config: connectionConfig } = useCanonicalConnectionConfig()

  // Mobile drawer state
  const [leftDrawerOpen, setLeftDrawerOpen] = React.useState(false)
  const [rightDrawerOpen, setRightDrawerOpen] = React.useState(false)

  // Mobile tab state
  const [activeTab, setActiveTab] = React.useState<"sessions" | "chat" | "tools" | "workspace">("chat")
  const [rightTab, setRightTab] = React.useState<"tools" | "workspace">("tools")

  // Store
  const sessionsById = useACPSessionsStore((s) => s.sessions)
  const activeSessionId = useACPSessionsStore((s) => s.activeSessionId)
  const sessions = React.useMemo(
    () =>
      Object.values(sessionsById).sort(
        (a, b) => b.updatedAt.getTime() - a.updatedAt.getTime()
      ),
    [sessionsById]
  )
  const activeSession = React.useMemo(
    () => (activeSessionId ? sessionsById[activeSessionId] : undefined),
    [activeSessionId, sessionsById]
  )
  const updateSessionState = useACPSessionsStore((s) => s.updateSessionState)
  const setSessionCapabilities = useACPSessionsStore((s) => s.setSessionCapabilities)
  const addUpdate = useACPSessionsStore((s) => s.addUpdate)
  const addPendingPermission = useACPSessionsStore((s) => s.addPendingPermission)
  const removePendingPermission = useACPSessionsStore((s) => s.removePendingPermission)
  const clearPendingPermissions = useACPSessionsStore((s) => s.clearPendingPermissions)
  const upsertSessionsFromServerList = useACPSessionsStore((s) => s.upsertSessionsFromServerList)
  const applySessionDetail = useACPSessionsStore((s) => s.applySessionDetail)
  const applySessionUsage = useACPSessionsStore((s) => s.applySessionUsage)
  const setActiveSession = useACPSessionsStore((s) => s.setActiveSession)
  const globalError = useACPSessionsStore((s) => s.globalError)
  const setGlobalError = useACPSessionsStore((s) => s.setGlobalError)
  const cleanupExpiredSessions = useACPSessionsStore((s) => s.cleanupExpiredSessions)
  const workspaceTabLabel = t("playground:acp.workspace.title", "Workspace")
  const [isHydratingSessions, setIsHydratingSessions] = React.useState(false)

  const restClient = React.useMemo(
    () =>
      connectionConfig ? new ACPRestClient(buildACPClientConfig(connectionConfig)) : null,
    [connectionConfig]
  )

  // Health check query to determine ACP backend availability.
  // Include the server URL in the query key so changing the ACP server
  // configuration invalidates the cached health status.
  const acpServerUrl = connectionConfig?.serverUrl ?? ""
  const { data: healthData, isLoading: isHealthLoading } = useQuery({
    queryKey: ["acp", "health", acpServerUrl],
    queryFn: async () => {
      try {
        const resp = await fetch(
          `${connectionConfig!.serverUrl}/api/v1/acp/health`,
          {
            headers: buildACPAuthHeaders(connectionConfig),
          }
        )
        return resp.ok ? await resp.json() : { overall: "unavailable" }
      } catch {
        return { overall: "unavailable" }
      }
    },
    enabled: !!connectionConfig,
    staleTime: 30_000,
    refetchInterval: 60_000,
  })
  // Return undefined while the initial health check is in flight so
  // downstream components can distinguish "loading" from "unhealthy".
  const acpHealthy: boolean | undefined = isHealthLoading
    ? undefined
    : healthData?.overall === "healthy" || healthData?.overall === "degraded"

  const refreshSessionsFromServer = React.useCallback(async () => {
    if (!restClient) {
      return
    }
    setIsHydratingSessions(true)
    try {
      const response = await restClient.listSessions({ limit: 200, offset: 0 })
      upsertSessionsFromServerList(response.sessions ?? [])
    } catch (error) {
      console.warn("Failed to hydrate ACP sessions from backend:", error)
    } finally {
      setIsHydratingSessions(false)
    }
  }, [restClient, upsertSessionsFromServerList])

  // Single ACP connection shared across chat + modal actions.
  const {
    state,
    isConnected,
    error: websocketError,
    reconnectInfo,
    connect,
    sendPrompt,
    cancel,
    approvePermission,
    denyPermission,
  } = useACPSession({
    sessionId: activeSessionId ?? undefined,
    autoConnect: Boolean(activeSessionId && connectionConfig),
    onConnected: (message) => {
      updateSessionState(message.session_id, "connected")
      setGlobalError(null)
      if (message.agent_capabilities) {
        setSessionCapabilities(message.session_id, message.agent_capabilities)
      }
    },
    onUpdate: (message) => {
      addUpdate(message.session_id, {
        type: message.update_type,
        data: message.data,
      })
    },
    onPermissionRequest: (message) => {
      addPendingPermission(message.session_id, {
        request_id: message.request_id,
        tool_name: message.tool_name,
        tool_arguments: message.tool_arguments,
        tier: message.tier,
        approval_requirement: message.approval_requirement,
        governance_reason: message.governance_reason,
        deny_reason: message.deny_reason,
        provenance_summary: message.provenance_summary,
        runtime_narrowing_reason: message.runtime_narrowing_reason,
        policy_snapshot_fingerprint: message.policy_snapshot_fingerprint,
        timeout_seconds: message.timeout_seconds,
        requestedAt: new Date(),
      })
    },
    onPromptComplete: (message) => {
      clearPendingPermissions(message.session_id)
      updateSessionState(message.session_id, "connected")
    },
    onError: (message) => {
      if (message.session_id) {
        updateSessionState(message.session_id, "error")
      }
      setGlobalError(message.message)
    },
  })

  // Cleanup expired sessions on mount
  useEffect(() => {
    cleanupExpiredSessions()
  }, [cleanupExpiredSessions])

  // Honor deep links from Agent Tasks and workspace run history.
  useEffect(() => {
    const params = new URLSearchParams(readACPPlaygroundSearch())
    const requestedSessionId = normalizeQueryValue(params.get("session"))
    const requestedView = normalizeQueryValue(params.get("view"))?.toLowerCase()

    if (requestedSessionId) {
      setActiveSession(requestedSessionId)
    }

    if (!requestedView) {
      return
    }

    if (requestedView === "workspace") {
      setActiveTab("workspace")
      setRightTab("workspace")
      setRightPaneOpen(true)
      return
    }

    if (requestedView === "tools") {
      setActiveTab("tools")
      setRightTab("tools")
      setRightPaneOpen(true)
      return
    }

    if (requestedView === "chat") {
      setActiveTab("chat")
      return
    }

    if (requestedView === "sessions" || ACP_SESSION_DETAIL_VIEWS.has(requestedView)) {
      setActiveTab("sessions")
      setLeftPaneOpen(true)
    }
  }, [setActiveSession, setLeftPaneOpen, setRightPaneOpen])

  // Hydrate persisted ACP sessions from backend when the page loads.
  useEffect(() => {
    if (!restClient) {
      return
    }
    void refreshSessionsFromServer()
  }, [refreshSessionsFromServer, restClient])

  // Keep the active session state in sync with the shared ACP hook.
  useEffect(() => {
    if (!activeSessionId) return
    updateSessionState(activeSessionId, state)
    if (state === "disconnected") {
      clearPendingPermissions(activeSessionId)
    }
  }, [activeSessionId, state, updateSessionState, clearPendingPermissions])

  const previousActiveSessionIdRef = React.useRef<string | null>(null)

  // Mark the previous active session disconnected when switching sessions.
  useEffect(() => {
    const previousSessionId = previousActiveSessionIdRef.current
    if (previousSessionId && previousSessionId !== activeSessionId) {
      updateSessionState(previousSessionId, "disconnected")
      clearPendingPermissions(previousSessionId)
    }
    previousActiveSessionIdRef.current = activeSessionId
  }, [activeSessionId, updateSessionState, clearPendingPermissions])

  useEffect(() => {
    setGlobalError(null)
  }, [activeSessionId, setGlobalError])

  // When switching sessions, refresh detail + usage if the session exists server-side.
  useEffect(() => {
    if (!activeSessionId || !restClient) return

    let cancelled = false
    const loadSessionMetadata = async () => {
      try {
        const [detailResult, usageResult] = await Promise.allSettled([
          restClient.getSessionDetail(activeSessionId),
          restClient.getSessionUsage(activeSessionId),
        ])

        if (cancelled) return

        if (detailResult.status === "fulfilled") {
          applySessionDetail(detailResult.value)
        }
        if (usageResult.status === "fulfilled") {
          applySessionUsage(usageResult.value)
        }
      } catch {
        // Ignore metadata refresh failures for local-only sessions.
      }
    }

    void loadSessionMetadata()

    return () => {
      cancelled = true
    }
  }, [activeSessionId, restClient, applySessionDetail, applySessionUsage])

  const handleToggleLeftPane = () => {
    if (isMobile) {
      setLeftDrawerOpen(!leftDrawerOpen)
    } else {
      setLeftPaneOpen(!leftPaneOpen)
    }
  }

  const handleToggleRightPane = () => {
    if (isMobile) {
      setRightDrawerOpen(!rightDrawerOpen)
    } else {
      setRightPaneOpen(!rightPaneOpen)
    }
  }

  const pendingPermissions = activeSession?.pendingPermissions ?? []
  const hasPendingPermissions = pendingPermissions.length > 0
  const panelFallback = <div className="py-6" data-testid="acp-panel-loading" />

  const handleApprovePermission = React.useCallback((requestId: string, batchApproveTier?: ACPPermissionTier) => {
    try {
      approvePermission(requestId, batchApproveTier)
      if (activeSessionId) {
        removePendingPermission(activeSessionId, requestId)
      }
    } catch (error) {
      setGlobalError(error instanceof Error ? error.message : "Failed to approve permission")
    }
  }, [approvePermission, activeSessionId, removePendingPermission, setGlobalError])

  const handleDenyPermission = React.useCallback((requestId: string) => {
    try {
      denyPermission(requestId)
      if (activeSessionId) {
        removePendingPermission(activeSessionId, requestId)
      }
    } catch (error) {
      setGlobalError(error instanceof Error ? error.message : "Failed to deny permission")
    }
  }, [denyPermission, activeSessionId, removePendingPermission, setGlobalError])

  const renderAcpSessionPanel = React.useCallback(
    (props: React.ComponentProps<typeof ACPSessionPanel>) => (
      <Suspense fallback={panelFallback}>
        <ACPSessionPanel {...props} />
      </Suspense>
    ),
    []
  )

  const renderAcpToolsPanel = React.useCallback(
    (props?: React.ComponentProps<typeof ACPToolsPanel>) => (
      <Suspense fallback={panelFallback}>
        <ACPToolsPanel {...props} />
      </Suspense>
    ),
    []
  )

  const renderAcpWorkspacePanel = React.useCallback(
    () => (
      <Suspense fallback={panelFallback}>
        <ACPWorkspacePanel />
      </Suspense>
    ),
    []
  )

  const renderAcpPermissionModal = React.useCallback(
    () => (
      <Suspense fallback={null}>
        <ACPPermissionModal
          pendingPermissions={pendingPermissions}
          approvePermission={handleApprovePermission}
          denyPermission={handleDenyPermission}
        />
      </Suspense>
    ),
    [handleApprovePermission, handleDenyPermission, pendingPermissions]
  )

  // Mobile tab items
  const mobileTabItems = [
    {
      key: "sessions",
      label: (
        <span className="flex items-center gap-1.5">
          <Bot className="h-4 w-4" />
          <span>{t("playground:acp.sessions", "Sessions")}</span>
          {sessions.length > 0 && (
            <span className="ml-1 rounded-full bg-primary px-1.5 py-0.5 text-xs text-white">
              {sessions.length}
            </span>
          )}
        </span>
      ),
      children: renderAcpSessionPanel({
        onRefreshSessions: refreshSessionsFromServer,
        isRefreshing: isHydratingSessions,
        acpHealthy,
      }),
    },
    {
      key: "chat",
      label: (
        <span className="flex items-center gap-1.5">
          <MessageSquare className="h-4 w-4" />
          <span>{t("playground:acp.chat", "Chat")}</span>
        </span>
      ),
      children: (
        <ACPChatPanel
          state={state}
          isConnected={isConnected}
          updates={activeSession?.updates ?? []}
          connect={connect}
          error={globalError ?? websocketError}
          sendPrompt={sendPrompt}
          cancel={cancel}
          reconnectInfo={reconnectInfo}
        />
      ),
    },
    {
      key: "tools",
      label: (
        <span className="flex items-center gap-1.5">
          <Wrench className="h-4 w-4" />
          <span>{t("playground:acp.tools", "Tools")}</span>
        </span>
      ),
      children: renderAcpToolsPanel(),
    },
    {
      key: "workspace",
      label: (
        <span className="flex items-center gap-1.5">
          <Terminal className="h-4 w-4" />
          <span>{workspaceTabLabel}</span>
        </span>
      ),
      children: renderAcpWorkspacePanel(),
    },
  ]

  // Mobile layout
  if (isMobile) {
    return (
      <div className="relative flex h-full flex-col bg-bg text-text">
        <ACPPlaygroundHeader
          leftPaneOpen={false}
          rightPaneOpen={false}
          onToggleLeftPane={handleToggleLeftPane}
          onToggleRightPane={handleToggleRightPane}
          hideToggles
          acpHealthy={acpHealthy}
          isHealthLoading={isHealthLoading}
        />

        <Tabs
          activeKey={activeTab}
          onChange={(key) => setActiveTab(key as typeof activeTab)}
          items={mobileTabItems}
          centered
          destroyOnHidden
          className="flex-1 [&_.ant-tabs-content]:h-full [&_.ant-tabs-content-holder]:flex-1 [&_.ant-tabs-tabpane]:h-full"
          tabBarStyle={{ marginBottom: 0, borderBottom: "1px solid var(--border)" }}
        />

        {hasPendingPermissions && renderAcpPermissionModal()}
      </div>
    )
  }

  // Desktop layout
  return (
    <div className="relative flex h-full flex-col bg-bg text-text">
      <ACPPlaygroundHeader
        leftPaneOpen={!!leftPaneOpen}
        rightPaneOpen={!!rightPaneOpen}
        onToggleLeftPane={handleToggleLeftPane}
        onToggleRightPane={handleToggleRightPane}
        acpHealthy={acpHealthy}
        isHealthLoading={isHealthLoading}
      />

      <div className="flex min-h-0 flex-1">
        {/* Left pane - Sessions (desktop) */}
        {leftPaneOpen && (
          <aside className="hidden w-72 shrink-0 border-r border-border bg-surface lg:flex lg:flex-col">
            {renderAcpSessionPanel({
              onHide: () => setLeftPaneOpen(false),
              onRefreshSessions: refreshSessionsFromServer,
              isRefreshing: isHydratingSessions,
              acpHealthy,
            })}
          </aside>
        )}

        {/* Left pane - Sessions (tablet drawer) */}
        <Drawer
          title={
            <span className="flex items-center gap-2">
              <Bot className="h-4 w-4" />
              {t("playground:acp.sessions", "Sessions")}
            </span>
          }
          placement="left"
          onClose={() => setLeftDrawerOpen(false)}
          open={leftDrawerOpen}
          size={320}
          className="lg:hidden"
          styles={{ body: { padding: 0 } }}
        >
          {renderAcpSessionPanel({
            onRefreshSessions: refreshSessionsFromServer,
            isRefreshing: isHydratingSessions,
            acpHealthy,
          })}
        </Drawer>

        {/* Center pane - Chat */}
        <main className="flex min-w-0 flex-1 flex-col">
          <ACPChatPanel
            state={state}
            isConnected={isConnected}
            updates={activeSession?.updates ?? []}
            connect={connect}
            error={globalError ?? websocketError}
            sendPrompt={sendPrompt}
            cancel={cancel}
            reconnectInfo={reconnectInfo}
          />
        </main>

        {/* Right pane - Tools (desktop) */}
        {rightPaneOpen && (
          <aside className="hidden w-80 shrink-0 border-l border-border bg-surface lg:flex lg:flex-col">
            <Tabs
              activeKey={rightTab}
              onChange={(key) => setRightTab(key as typeof rightTab)}
              items={[
                {
                  key: "tools",
                  label: t("playground:acp.tools", "Tools"),
                  children: renderAcpToolsPanel({
                    onHide: () => setRightPaneOpen(false),
                  }),
                },
                {
                  key: "workspace",
                  label: workspaceTabLabel,
                  children: renderAcpWorkspacePanel(),
                },
              ]}
              destroyOnHidden
              className="h-full [&_.ant-tabs-content]:h-full [&_.ant-tabs-content-holder]:flex-1 [&_.ant-tabs-tabpane]:h-full"
            />
          </aside>
        )}

        {/* Right pane - Tools (tablet drawer) */}
        <Drawer
          title={
            <span className="flex items-center gap-2">
              <Wrench className="h-4 w-4" />
              {t("playground:acp.tools", "Tools")}
            </span>
          }
          placement="right"
          onClose={() => setRightDrawerOpen(false)}
          open={rightDrawerOpen}
          size={320}
          className="lg:hidden"
          styles={{ body: { padding: 0 } }}
        >
          <Tabs
            activeKey={rightTab}
            onChange={(key) => setRightTab(key as typeof rightTab)}
            items={[
              {
                key: "tools",
                label: t("playground:acp.tools", "Tools"),
                children: renderAcpToolsPanel({
                  onHide: () => {
                    setRightPaneOpen(false)
                    setRightDrawerOpen(false)
                  },
                }),
              },
              {
                key: "workspace",
                label: workspaceTabLabel,
                children: renderAcpWorkspacePanel(),
              },
            ]}
            destroyOnHidden
            className="h-full [&_.ant-tabs-content]:h-full [&_.ant-tabs-content-holder]:flex-1 [&_.ant-tabs-tabpane]:h-full"
          />
        </Drawer>
      </div>

      {/* Permission modal */}
      {hasPendingPermissions && renderAcpPermissionModal()}
    </div>
  )
}

export default ACPPlayground
