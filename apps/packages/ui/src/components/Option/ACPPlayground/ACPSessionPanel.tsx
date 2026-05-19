import React, { useState } from "react"
import { useTranslation } from "react-i18next"
import { Button, Empty, Input, Popconfirm, Tooltip, Tag } from "antd"
import { Plus, Folder, Trash2, X, ChevronRight, Bot, Copy, GitFork, RefreshCw } from "lucide-react"
import { useACPSessionsStore } from "@/store/acp-sessions"
import { useCanonicalConnectionConfig } from "@/hooks/useCanonicalConnectionConfig"
import { ACPRestClient } from "@/services/acp/client"
import { buildACPClientConfig } from "@/services/acp/connection"
import type { ACPSession, ACPAgentType, ACPSessionState } from "@/services/acp/types"
import { AGENT_TYPE_INFO } from "@/services/acp/constants"
import { ACPSessionCreateModal } from "./ACPSessionCreateModal"
import { getSessionMessageCount, getSessionTokenUsage } from "./sessionMetrics"

interface ACPSessionPanelProps {
  onHide?: () => void
  onRefreshSessions?: () => Promise<void> | void
  isRefreshing?: boolean
  acpHealthy?: boolean
}

type SessionSortKey = "recent" | "oldest" | "name_asc" | "name_desc"

export const ACPSessionPanel: React.FC<ACPSessionPanelProps> = ({
  onHide,
  onRefreshSessions,
  isRefreshing = false,
  acpHealthy,
}) => {
  const { t } = useTranslation(["playground", "option", "common"])

  const [showCreateModal, setShowCreateModal] = useState(false)
  const [searchQuery, setSearchQuery] = useState("")
  const [stateFilter, setStateFilter] = useState<ACPSessionState | "all">("all")
  const [sortKey, setSortKey] = useState<SessionSortKey>("recent")
  const { config: connectionConfig } = useCanonicalConnectionConfig()

  // Server config
  const [isRefreshingLocal, setIsRefreshingLocal] = useState(false)

  const restClient = React.useMemo(
    () =>
      connectionConfig ? new ACPRestClient(buildACPClientConfig(connectionConfig)) : null,
    [connectionConfig]
  )

  // Store
  const sessionsById = useACPSessionsStore((s) => s.sessions)
  const sessions = React.useMemo(
    () => Object.values(sessionsById),
    [sessionsById]
  )
  const filteredSessions = React.useMemo(() => {
    const normalizedQuery = searchQuery.trim().toLowerCase()
    let next = sessions

    if (normalizedQuery) {
      next = next.filter((session) => {
        const name = (session.name || "").toLowerCase()
        const cwd = session.cwd.toLowerCase()
        const agentType = (session.agentType || "").toLowerCase()
        const tags = (session.tags || []).join(" ").toLowerCase()
        return (
          name.includes(normalizedQuery)
          || cwd.includes(normalizedQuery)
          || agentType.includes(normalizedQuery)
          || tags.includes(normalizedQuery)
        )
      })
    }

    if (stateFilter !== "all") {
      next = next.filter((session) => session.state === stateFilter)
    }

    return [...next].sort((a, b) => {
      switch (sortKey) {
        case "oldest":
          return a.updatedAt.getTime() - b.updatedAt.getTime()
        case "name_asc":
          return (a.name || a.cwd).localeCompare(b.name || b.cwd, undefined, { sensitivity: "base" })
        case "name_desc":
          return (b.name || b.cwd).localeCompare(a.name || a.cwd, undefined, { sensitivity: "base" })
        case "recent":
        default:
          return b.updatedAt.getTime() - a.updatedAt.getTime()
      }
    })
  }, [sessions, searchQuery, stateFilter, sortKey])
  const activeSessionId = useACPSessionsStore((s) => s.activeSessionId)
  const setActiveSession = useACPSessionsStore((s) => s.setActiveSession)
  const closeSession = useACPSessionsStore((s) => s.closeSession)
  const createSession = useACPSessionsStore((s) => s.createSession)
  const replaceSessionId = useACPSessionsStore((s) => s.replaceSessionId)
  const applySessionDetail = useACPSessionsStore((s) => s.applySessionDetail)
  const applySessionUsage = useACPSessionsStore((s) => s.applySessionUsage)
  const setGlobalError = useACPSessionsStore((s) => s.setGlobalError)
  const hasActiveFilters = searchQuery.trim().length > 0 || stateFilter !== "all" || sortKey !== "recent"
  const refreshLoading = isRefreshing || isRefreshingLocal

  const handleCreateSession = () => {
    setShowCreateModal(true)
  }

  const handleSessionCreated = (sessionId: string) => {
    // The modal handles everything, just close it
    setShowCreateModal(false)
  }

  const handleSelectSession = (sessionId: string) => {
    setActiveSession(sessionId)
  }

  const handleCloseSession = async (sessionId: string) => {
    try {
      await restClient?.closeSession(sessionId).catch(() => {
        // Ignore server close failures - session may already be closed
      })

      // Remove from local store
      closeSession(sessionId)
    } catch (error) {
      console.error("Failed to close session:", error)
    }
  }

  const handleCopySessionId = async (sessionId: string) => {
    try {
      if (typeof navigator !== "undefined" && navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(sessionId)
      }
    } catch (error) {
      console.error("Failed to copy ACP session ID:", error)
    }
  }

  const createLocalFork = (sourceSession: ACPSession): string => {
    const fallbackName = `${sourceSession.name || sourceSession.cwd.split("/").filter(Boolean).pop() || "Session"} (fork)`
    const forkSessionId = createSession({
      cwd: sourceSession.cwd,
      name: fallbackName,
      agentType: sourceSession.agentType,
      tags: sourceSession.tags ? [...sourceSession.tags] : undefined,
      mcpServers: sourceSession.mcpServers?.map((server) => ({
        ...server,
        args: server.args ? [...server.args] : undefined,
        env: server.env ? { ...server.env } : undefined,
      })),
      personaId: sourceSession.personaId ?? undefined,
      workspaceId: sourceSession.workspaceId ?? undefined,
      workspaceGroupId: sourceSession.workspaceGroupId ?? undefined,
      scopeSnapshotId: sourceSession.scopeSnapshotId ?? undefined,
    })

    useACPSessionsStore.setState((state) => {
      const forkSession = state.sessions[forkSessionId]
      if (!forkSession) {
        return state
      }

      return {
        sessions: {
          ...state.sessions,
          [forkSessionId]: {
            ...forkSession,
            capabilities: sourceSession.capabilities ? { ...sourceSession.capabilities } : undefined,
            updates: sourceSession.updates.map((update) => ({
              ...update,
              timestamp: new Date(update.timestamp),
              data: { ...update.data },
            })),
            pendingPermissions: [],
            forkParentSessionId: sourceSession.id,
            state: "disconnected",
            updatedAt: new Date(),
          },
        },
      }
    })

    setActiveSession(forkSessionId)
    return forkSessionId
  }

  const handleForkSession = async (sessionId: string) => {
    const sourceSession = sessionsById[sessionId]
    if (!sourceSession) {
      return
    }
    const isServerBacked = sourceSession.backendStatus !== null
    setGlobalError(null)

    if (!restClient) {
      if (!isServerBacked) {
        createLocalFork(sourceSession)
      } else {
        setGlobalError("Missing ACP connection configuration")
      }
      return
    }

    const fallbackName = `${sourceSession.name || sourceSession.cwd.split("/").filter(Boolean).pop() || "Session"} (fork)`

    try {
      let messageIndex = resolveForkMessageIndex(sourceSession)

      try {
        const detail = await restClient.getSessionDetail(sessionId)
        applySessionDetail(detail)
        if (Array.isArray(detail.messages) && detail.messages.length > 0) {
          messageIndex = detail.messages.length - 1
        }
      } catch {
        // Ignore detail fetch failure and keep local fallback index.
      }

      if (messageIndex < 0) {
        if (!isServerBacked) {
          createLocalFork(sourceSession)
        } else {
          setGlobalError("fork_not_resumable")
        }
        return
      }

      const payload = await restClient.forkSession(sessionId, {
        message_index: messageIndex,
        name: fallbackName,
      })

      const serverSessionId = payload.session_id

      const localForkSessionId = createSession({
        cwd: sourceSession.cwd,
        name: payload.name || fallbackName,
        agentType: sourceSession.agentType,
        tags: sourceSession.tags ? [...sourceSession.tags] : undefined,
        mcpServers: sourceSession.mcpServers?.map((server) => ({
          ...server,
          args: server.args ? [...server.args] : undefined,
          env: server.env ? { ...server.env } : undefined,
        })),
        personaId: sourceSession.personaId ?? undefined,
        workspaceId: sourceSession.workspaceId ?? undefined,
        workspaceGroupId: sourceSession.workspaceGroupId ?? undefined,
        scopeSnapshotId: sourceSession.scopeSnapshotId ?? undefined,
      })

      replaceSessionId(localForkSessionId, serverSessionId, {
        name: payload.name || undefined,
        forkParentSessionId: resolveForkParentSessionId(payload, sourceSession.id),
      })

      void restClient.getSessionUsage(serverSessionId)
        .then((usage) => applySessionUsage(usage))
        .catch(() => undefined)

      setActiveSession(serverSessionId)
    } catch (error) {
      if (!isServerBacked) {
        createLocalFork(sourceSession)
        return
      }
      const message = error instanceof Error && error.message ? error.message : "Failed to fork ACP session"
      setGlobalError(message)
    }
  }

  const getSessionStateColor = (state: string) => {
    switch (state) {
      case "connected":
        return "bg-success"
      case "running":
        return "bg-primary"
      case "waiting_permission":
        return "bg-warning"
      case "error":
        return "bg-error"
      case "connecting":
        return "bg-info"
      default:
        return "bg-text-muted"
    }
  }

  const formatDate = (date: Date) => {
    const now = new Date()
    const diff = now.getTime() - date.getTime()
    const minutes = Math.floor(diff / 60000)
    const hours = Math.floor(minutes / 60)
    const days = Math.floor(hours / 24)

    if (minutes < 1) return t("playground:acp.justNow", "Just now")
    if (minutes < 60) return t("playground:acp.minutesAgo", "{{count}}m ago", { count: minutes })
    if (hours < 24) return t("playground:acp.hoursAgo", "{{count}}h ago", { count: hours })
    return t("playground:acp.daysAgo", "{{count}}d ago", { count: days })
  }

  const clearFilters = () => {
    setSearchQuery("")
    setStateFilter("all")
    setSortKey("recent")
  }

  const handleRefreshSessions = async () => {
    if (!onRefreshSessions) {
      return
    }

    setIsRefreshingLocal(true)
    try {
      await onRefreshSessions()
    } finally {
      setIsRefreshingLocal(false)
    }
  }

  const getStateFilterLabel = (state: ACPSessionState | "all") => {
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
      case "disconnected":
        return t("playground:acp.state.disconnected", "Disconnected")
      case "all":
      default:
        return t("playground:acp.filter.allStates", "All states")
    }
  }

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-border p-3">
        <h2 className="text-sm font-semibold text-text">
          {t("playground:acp.sessions", "Sessions")}
        </h2>
        <div className="flex items-center gap-1">
          {onRefreshSessions && (
            <Tooltip title={t("playground:acp.refreshSessions", "Refresh sessions")}>
              <Button
                type="text"
                size="small"
                icon={<RefreshCw className={`h-4 w-4 ${refreshLoading ? "animate-spin" : ""}`} />}
                onClick={handleRefreshSessions}
                disabled={refreshLoading}
              />
            </Tooltip>
          )}
          <Tooltip title={t("playground:acp.newSession", "New Session")}>
            <Button
              type="text"
              size="small"
              icon={<Plus className="h-4 w-4" />}
              onClick={handleCreateSession}
            />
          </Tooltip>
          {onHide && (
            <Tooltip title={t("common:close", "Close")}>
              <Button
                type="text"
                size="small"
                icon={<X className="h-4 w-4" />}
                onClick={onHide}
              />
            </Tooltip>
          )}
        </div>
      </div>

      {/* Sessions list */}
      <div className="flex-1 overflow-y-auto">
        {sessions.length === 0 ? (
          acpHealthy === false ? (
            <Empty
              image={Empty.PRESENTED_IMAGE_SIMPLE}
              description={
                <div className="space-y-2 text-sm text-text-muted">
                  <p>
                    {t(
                      "playground:acp.notConfigured",
                      "ACP (Agent Client Protocol) is not configured or the backend is unreachable."
                    )}
                  </p>
                  <p>
                    {t(
                      "playground:acp.notConfiguredHelp",
                      "ACP lets you interact with AI coding agents like Claude Code from your browser. Enable the [ACP] section in config.txt and restart the server to get started."
                    )}
                  </p>
                </div>
              }
              className="py-8"
            />
          ) : (
            <Empty
              image={Empty.PRESENTED_IMAGE_SIMPLE}
              description={t("playground:acp.noSessions", "No active sessions")}
              className="py-8"
            >
              <Button
                type="primary"
                icon={<Plus className="h-4 w-4" />}
                onClick={handleCreateSession}
              >
                {t("playground:acp.createFirst", "Create Session")}
              </Button>
            </Empty>
          )
        ) : (
          <>
            <div className="border-b border-border p-2">
              <div className="mb-2 flex items-center justify-between gap-2">
                <Input
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder={t("playground:acp.searchSessions", "Search sessions")}
                  allowClear
                  size="small"
                  aria-label={t("playground:acp.searchSessions", "Search sessions")}
                />
                {hasActiveFilters && (
                  <Button size="small" type="text" onClick={clearFilters}>
                    {t("playground:acp.clearFilters", "Clear")}
                  </Button>
                )}
              </div>

              <div className="flex items-center gap-2">
                <label className="sr-only" htmlFor="acp-session-filter-state">
                  {t("playground:acp.filterByState", "Filter by state")}
                </label>
                <select
                  id="acp-session-filter-state"
                  data-testid="acp-session-filter-state"
                  value={stateFilter}
                  onChange={(e) => setStateFilter(e.target.value as ACPSessionState | "all")}
                  className="h-7 min-w-0 flex-1 rounded border border-border bg-bg px-2 text-xs text-text"
                >
                  {(["all", "connected", "running", "waiting_permission", "error", "connecting", "disconnected"] as const).map((state) => (
                    <option key={state} value={state}>
                      {getStateFilterLabel(state)}
                    </option>
                  ))}
                </select>

                <label className="sr-only" htmlFor="acp-session-sort">
                  {t("playground:acp.sortSessions", "Sort sessions")}
                </label>
                <select
                  id="acp-session-sort"
                  data-testid="acp-session-sort"
                  value={sortKey}
                  onChange={(e) => setSortKey(e.target.value as SessionSortKey)}
                  className="h-7 min-w-0 flex-1 rounded border border-border bg-bg px-2 text-xs text-text"
                >
                  <option value="recent">{t("playground:acp.sort.recent", "Recently updated")}</option>
                  <option value="oldest">{t("playground:acp.sort.oldest", "Oldest updated")}</option>
                  <option value="name_asc">{t("playground:acp.sort.nameAsc", "Name A-Z")}</option>
                  <option value="name_desc">{t("playground:acp.sort.nameDesc", "Name Z-A")}</option>
                </select>
              </div>
            </div>

            {filteredSessions.length === 0 ? (
              <Empty
                image={Empty.PRESENTED_IMAGE_SIMPLE}
                description={t("playground:acp.noSessionsFiltered", "No sessions match the current filters")}
                className="py-8"
              >
                <Button size="small" onClick={clearFilters}>
                  {t("playground:acp.clearFilters", "Clear")}
                </Button>
              </Empty>
            ) : (
              <div className="p-2">
                {filteredSessions.map((session) => (
                  <SessionItem
                    key={session.id}
                    session={session}
                    isActive={session.id === activeSessionId}
                    messageCount={getSessionMessageCount(session)}
                    tokenUsage={getSessionTokenUsage(session)}
                    pendingPermissionCount={session.pendingPermissions.length}
                    onSelect={() => handleSelectSession(session.id)}
                    onCopySessionId={() => handleCopySessionId(session.id)}
                    onFork={() => handleForkSession(session.id)}
                    onClose={() => handleCloseSession(session.id)}
                    getStateColor={getSessionStateColor}
                    formatDate={formatDate}
                  />
                ))}
              </div>
            )}
          </>
        )}
      </div>

      {/* Create Session Modal */}
      <ACPSessionCreateModal
        open={showCreateModal}
        onClose={() => setShowCreateModal(false)}
        onSuccess={handleSessionCreated}
      />
    </div>
  )
}

interface SessionItemProps {
  session: ACPSession
  isActive: boolean
  messageCount: number
  tokenUsage: number | null
  pendingPermissionCount: number
  onSelect: () => void
  onCopySessionId: () => void
  onFork: () => void
  onClose: () => void
  getStateColor: (state: string) => string
  formatDate: (date: Date) => string
}

const getAgentTypeColor = (agentType?: ACPAgentType): string => {
  if (!agentType) return "default"
  const info = AGENT_TYPE_INFO[agentType]
  return info?.color ?? "default"
}

const getAgentTypeName = (agentType?: ACPAgentType): string => {
  if (!agentType) return ""
  const info = AGENT_TYPE_INFO[agentType]
  return info?.name ?? agentType
}

const getPolicySummaryBadges = (session: ACPSession): string[] => {
  const summary = (session.policySummary || null) as Record<string, unknown> | null
  if (!summary) {
    return []
  }

  const badges: string[] = []
  const approvalMode = typeof summary.approval_mode === "string" ? summary.approval_mode : null
  const allowedCount = typeof summary.allowed_tool_count === "number" ? summary.allowed_tool_count : null
  const deniedCount = typeof summary.denied_tool_count === "number" ? summary.denied_tool_count : null

  if (approvalMode) {
    badges.push(`Policy ${approvalMode}`)
  }
  if (allowedCount !== null) {
    badges.push(`Allow ${allowedCount}`)
  }
  if (deniedCount !== null && deniedCount > 0) {
    badges.push(`Deny ${deniedCount}`)
  }

  return badges
}

const SessionItem: React.FC<SessionItemProps> = ({
  session,
  isActive,
  messageCount,
  tokenUsage,
  pendingPermissionCount,
  onSelect,
  onCopySessionId,
  onFork,
  onClose,
  getStateColor,
  formatDate,
}) => {
  const { t } = useTranslation(["playground", "common"])
  const policyBadges = getPolicySummaryBadges(session)
  const hasPolicyError = typeof session.policyRefreshError === "string" && session.policyRefreshError.length > 0

  // Get the last part of the path for display
  const displayPath = session.cwd.split("/").filter(Boolean).slice(-2).join("/") || session.cwd || "/"
  const displayName = session.name || displayPath || session.cwd

  return (
    <div
      data-testid="acp-session-item"
      data-session-id={session.id}
      className={`group mb-1 flex cursor-pointer items-center gap-2 rounded-lg p-2 transition-colors ${
        isActive
          ? "bg-primary/10 text-primary"
          : "hover:bg-surface2"
      }`}
      onClick={onSelect}
    >
      <span className={`h-2 w-2 shrink-0 rounded-full ${getStateColor(session.state)}`} />

      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="truncate text-sm font-medium">
            {displayName}
          </span>
          {session.agentType && (
            <Tag
              className="shrink-0"
              color={getAgentTypeColor(session.agentType)}
              style={{ margin: 0, fontSize: "10px", lineHeight: "16px", padding: "0 4px" }}
            >
              <Bot className="mr-0.5 inline h-2.5 w-2.5" />
              {getAgentTypeName(session.agentType)}
            </Tag>
          )}
        </div>
        <div className="flex items-center gap-2 text-xs text-text-muted">
          <Folder className="h-3 w-3 shrink-0" />
          <span className="truncate">{displayPath}</span>
          <span className="shrink-0">·</span>
          <span className="shrink-0">{formatDate(session.updatedAt)}</span>
        </div>

        <div className="mt-1 flex items-center gap-1.5 text-[11px]">
          <span className="rounded bg-surface2 px-1.5 py-0.5 text-text-muted">
            {t("playground:acp.sessionMeta.messages", `Msgs ${messageCount}`)}
          </span>
          <span className="rounded bg-surface2 px-1.5 py-0.5 text-text-muted">
            {t(
              "playground:acp.sessionMeta.tokens",
              `Tokens ${tokenUsage !== null ? tokenUsage.toLocaleString() : "--"}`
            )}
          </span>
          <span className="rounded bg-surface2 px-1.5 py-0.5 text-text-muted">
            {t("playground:acp.sessionMeta.permissions", `Perm ${pendingPermissionCount}`)}
          </span>
          {session.forkParentSessionId && (
            <Tooltip title={session.forkParentSessionId}>
              <span
                className="rounded bg-surface2 px-1.5 py-0.5 text-text-muted"
                data-testid={`acp-session-fork-parent-${session.id}`}
              >
                {t(
                  "playground:acp.sessionMeta.fork",
                  `Fork ${formatForkParentSessionId(session.forkParentSessionId)}`
                )}
              </span>
            </Tooltip>
          )}
        </div>

        {(policyBadges.length > 0 || hasPolicyError) && (
          <div className="mt-1 flex flex-wrap items-center gap-1.5 text-[11px]">
            {policyBadges.map((badge) => (
              <span
                key={`${session.id}-${badge}`}
                className="rounded bg-primary/10 px-1.5 py-0.5 text-primary"
              >
                {badge}
              </span>
            ))}
            {hasPolicyError && (
              <Tooltip title={session.policyRefreshError}>
                <span className="rounded bg-danger/10 px-1.5 py-0.5 text-danger">
                  {t("playground:acp.sessionMeta.policyError", "Policy refresh error")}
                </span>
              </Tooltip>
            )}
          </div>
        )}
      </div>

      <div className="flex items-center gap-1 opacity-0 transition-opacity group-hover:opacity-100">
        <Tooltip title={t("playground:acp.copySessionId", "Copy session ID")}>
          <Button
            type="text"
            size="small"
            icon={<Copy className="h-3 w-3" />}
            onClick={(e) => {
              e.stopPropagation()
              onCopySessionId()
            }}
            aria-label={t("playground:acp.copySessionId", "Copy session ID")}
            data-testid={`acp-session-copy-${session.id}`}
            className="text-text-muted hover:text-text"
          />
        </Tooltip>
        <Tooltip title={t("playground:acp.forkSession", "Fork session")}>
          <Button
            type="text"
            size="small"
            icon={<GitFork className="h-3 w-3" />}
            onClick={(e) => {
              e.stopPropagation()
              onFork()
            }}
            aria-label={t("playground:acp.forkSession", "Fork session")}
            data-testid={`acp-session-fork-${session.id}`}
            className="text-text-muted hover:text-text"
          />
        </Tooltip>
        <Popconfirm
          title={t("playground:acp.closeSessionConfirm", "Close this session?")}
          onConfirm={(e) => {
            e?.stopPropagation()
            onClose()
          }}
          onCancel={(e) => e?.stopPropagation()}
          okText={t("common:yes", "Yes")}
          cancelText={t("common:no", "No")}
        >
          <Button
            type="text"
            size="small"
            icon={<Trash2 className="h-3 w-3" />}
            onClick={(e) => e.stopPropagation()}
            aria-label={t("playground:acp.closeSession", "Close session")}
            className="text-text-muted hover:text-error"
          />
        </Popconfirm>
      </div>

      {isActive && (
        <ChevronRight className="h-4 w-4 shrink-0 text-primary" />
      )}
    </div>
  )
}

const resolveForkMessageIndex = (sourceSession: ACPSession): number => {
  const inferredMessageCount = getSessionMessageCount(sourceSession)
  return inferredMessageCount > 0 ? inferredMessageCount - 1 : -1
}

const resolveForkParentSessionId = (
  payload: {
    fork_parent_session_id?: unknown
    parent_session_id?: unknown
    forked_from_session_id?: unknown
    forked_from?: unknown
  } | null,
  sourceSessionId: string
): string => {
  const candidate = payload?.fork_parent_session_id
    ?? payload?.parent_session_id
    ?? payload?.forked_from_session_id
    ?? payload?.forked_from
  return typeof candidate === "string" && candidate.length > 0
    ? candidate
    : sourceSessionId
}

const formatForkParentSessionId = (sessionId: string): string => (
  sessionId.length > 8 ? sessionId.slice(0, 8) : sessionId
)
