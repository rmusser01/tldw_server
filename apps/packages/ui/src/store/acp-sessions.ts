/**
 * ACP Sessions Zustand Store
 * Manages state for ACP (Agent Client Protocol) sessions
 */

import { createWithEqualityFn } from "zustand/traditional"
import { createJSONStorage, persist, type StateStorage } from "zustand/middleware"
import type {
  ACPSession,
  ACPSessionDetailResponse,
  ACPSessionListItem,
  ACPSessionState,
  ACPSessionUsageResponse,
  ACPTokenUsage,
  ACPUpdate,
  ACPPendingPermission,
  ACPAgentType,
  ACPBackendSessionStatus,
  ACPMCPServerConfig,
} from "@/services/acp/types"
import { SESSION_CONFIG } from "@/services/acp/constants"

// -----------------------------------------------------------------------------
// Storage Configuration
// -----------------------------------------------------------------------------

const STORAGE_KEY = "tldw-acp-sessions"

const generateId = (): string => {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID()
  }
  return Math.random().toString(36).slice(2)
}

const createMemoryStorage = (): StateStorage => ({
  getItem: () => null,
  setItem: () => {},
  removeItem: () => {},
})

const DATE_FIELDS = new Set(["createdAt", "updatedAt", "requestedAt", "timestamp", "lastActivityAt"])

const MESSAGE_UPDATE_TYPES = new Set(["text", "assistant_text", "user_text"])

const toIsoDate = (value: string | null | undefined): Date | null => {
  if (!value) return null
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? null : date
}

const normalizeBackendSessionStatus = (status: string): ACPBackendSessionStatus => {
  if (status === "closed" || status === "error") {
    return status
  }
  return "active"
}

const resolveStateFromBackendStatus = (
  status: ACPBackendSessionStatus,
  hasWebsocket: boolean
): ACPSessionState => {
  if (status === "error") return "error"
  if (status === "closed") return "disconnected"
  return hasWebsocket ? "connected" : "disconnected"
}

const shouldKeepLiveState = (state: ACPSessionState): boolean => (
  state === "connecting" || state === "connected" || state === "running" || state === "waiting_permission"
)

const mergeUsage = (
  existingUsage: ACPTokenUsage | null | undefined,
  incomingUsage: ACPTokenUsage | null | undefined
): ACPTokenUsage | null => {
  if (!existingUsage && !incomingUsage) {
    return null
  }

  const current: ACPTokenUsage = existingUsage ?? {
    prompt_tokens: 0,
    completion_tokens: 0,
    total_tokens: 0,
  }

  const incoming: ACPTokenUsage = incomingUsage ?? {
    prompt_tokens: 0,
    completion_tokens: 0,
    total_tokens: 0,
  }

  return {
    prompt_tokens: Math.max(current.prompt_tokens, incoming.prompt_tokens),
    completion_tokens: Math.max(current.completion_tokens, incoming.completion_tokens),
    total_tokens: Math.max(current.total_tokens, incoming.total_tokens),
  }
}

const getUsageIncrementFromUpdate = (data: Record<string, unknown>): ACPTokenUsage | null => {
  const usage = (data.usage || null) as Record<string, unknown> | null

  const usagePromptTokens = typeof usage?.prompt_tokens === "number" ? usage.prompt_tokens : 0
  const usageCompletionTokens = typeof usage?.completion_tokens === "number" ? usage.completion_tokens : 0
  const usageTotalTokens = typeof usage?.total_tokens === "number" ? usage.total_tokens : 0
  if (usagePromptTokens > 0 || usageCompletionTokens > 0 || usageTotalTokens > 0) {
    return {
      prompt_tokens: usagePromptTokens,
      completion_tokens: usageCompletionTokens,
      total_tokens: usageTotalTokens || usagePromptTokens + usageCompletionTokens,
    }
  }

  const promptTokens = typeof data.prompt_tokens === "number" ? data.prompt_tokens : 0
  const completionTokens = typeof data.completion_tokens === "number" ? data.completion_tokens : 0
  const totalTokens = typeof data.total_tokens === "number" ? data.total_tokens : 0
  if (promptTokens > 0 || completionTokens > 0 || totalTokens > 0) {
    return {
      prompt_tokens: promptTokens,
      completion_tokens: completionTokens,
      total_tokens: totalTokens || promptTokens + completionTokens,
    }
  }

  return null
}

/**
 * Generate a session name from the working directory.
 * Format: "ProjectName (HH:MM)"
 */
const generateSessionName = (cwd: string): string => {
  // Extract project name from cwd
  const parts = cwd.replace(/\/$/, "").split("/")
  const projectName = parts[parts.length - 1] || "Session"

  // Add time stamp
  const now = new Date()
  const hours = now.getHours().toString().padStart(2, "0")
  const minutes = now.getMinutes().toString().padStart(2, "0")

  return `${projectName} (${hours}:${minutes})`
}

const dateReviver = (key: string, value: unknown): unknown => {
  if (DATE_FIELDS.has(key) && typeof value === "string") {
    const date = new Date(value)
    return isNaN(date.getTime()) ? value : date
  }
  return value
}

const createACPStorage = (): StateStorage => {
  if (typeof window === "undefined") {
    return createMemoryStorage()
  }

  return {
    getItem: (name: string): string | null => {
      const value = localStorage.getItem(name)
      if (!value) return null
      try {
        const parsed = JSON.parse(value, dateReviver)
        return JSON.stringify(parsed)
      } catch {
        return value
      }
    },
    setItem: (name: string, value: string): void => {
      localStorage.setItem(name, value)
    },
    removeItem: (name: string): void => {
      localStorage.removeItem(name)
    },
  }
}

// -----------------------------------------------------------------------------
// State Types
// -----------------------------------------------------------------------------

interface SessionsState {
  /** All known sessions */
  sessions: Record<string, ACPSession>
  /** Currently active session ID */
  activeSessionId: string | null
  /** Session creation in progress */
  isCreating: boolean
  /** Global error message */
  globalError: string | null
}

export interface CreateSessionOptions {
  cwd: string
  name?: string
  agentType?: ACPAgentType
  tags?: string[]
  mcpServers?: ACPMCPServerConfig[]
  personaId?: string | null
  workspaceId?: string | null
  workspaceGroupId?: string | null
  scopeSnapshotId?: string | null
}

interface SessionsActions {
  /** Create a new session entry (before WebSocket connects) */
  createSession: (options: CreateSessionOptions) => string
  /** Update session state */
  updateSessionState: (sessionId: string, state: ACPSessionState) => void
  /** Update session name */
  updateSessionName: (sessionId: string, name: string) => void
  /** Update session tags */
  updateSessionTags: (sessionId: string, tags: string[]) => void
  /** Set session capabilities */
  setSessionCapabilities: (sessionId: string, capabilities: Record<string, unknown>) => void
  /** Update session metadata */
  updateSessionMetadata: (sessionId: string, metadata: Partial<Pick<ACPSession, "sandboxSessionId" | "sandboxRunId" | "sshWsUrl" | "sshUser">>) => void
  /** Replace a local session id with the server session id */
  replaceSessionId: (localSessionId: string, serverSessionId: string, updates?: Partial<ACPSession>) => void
  /** Merge server-listed sessions into the local store */
  upsertSessionsFromServerList: (sessions: ACPSessionListItem[]) => void
  /** Merge server session detail into a local session */
  applySessionDetail: (detail: ACPSessionDetailResponse) => void
  /** Merge server usage totals into a local session */
  applySessionUsage: (usage: ACPSessionUsageResponse) => void
  /** Add an update to a session */
  addUpdate: (sessionId: string, update: Omit<ACPUpdate, "timestamp">) => void
  /** Add a pending permission */
  addPendingPermission: (sessionId: string, permission: ACPPendingPermission) => void
  /** Remove a pending permission */
  removePendingPermission: (sessionId: string, requestId: string) => void
  /** Clear all pending permissions for a session */
  clearPendingPermissions: (sessionId: string) => void
  /** Set the active session */
  setActiveSession: (sessionId: string | null) => void
  /** Close and remove a session */
  closeSession: (sessionId: string) => void
  /** Clear updates for a session */
  clearSessionUpdates: (sessionId: string) => void
  /** Set creating state */
  setIsCreating: (isCreating: boolean) => void
  /** Set global error */
  setGlobalError: (error: string | null) => void
  /** Get a session by ID */
  getSession: (sessionId: string) => ACPSession | undefined
  /** Get all sessions as array */
  getSessions: () => ACPSession[]
  /** Clean up expired sessions */
  cleanupExpiredSessions: () => void
  /** Reset all state */
  reset: () => void
}

export type ACPSessionsStore = SessionsState & SessionsActions

// -----------------------------------------------------------------------------
// Initial State
// -----------------------------------------------------------------------------

const initialState: SessionsState = {
  sessions: {},
  activeSessionId: null,
  isCreating: false,
  globalError: null,
}

// -----------------------------------------------------------------------------
// Store
// -----------------------------------------------------------------------------

interface PersistedState {
  sessions: Record<string, ACPSession>
  activeSessionId: string | null
}

export const useACPSessionsStore = createWithEqualityFn<ACPSessionsStore>()(
  persist<ACPSessionsStore, [], [], PersistedState>(
    (set, get) => ({
      ...initialState,

      upsertSessionsFromServerList: (serverSessions) => {
        set((state) => {
          const mergedSessions = { ...state.sessions }

          for (const serverSession of serverSessions) {
            const serverSessionId = serverSession.session_id
            if (!serverSessionId) {
              continue
            }
            const existing = mergedSessions[serverSessionId]
            const backendStatus = normalizeBackendSessionStatus(serverSession.status)
            const fallbackState = resolveStateFromBackendStatus(
              backendStatus,
              serverSession.has_websocket
            )

            const baseSession: ACPSession = existing ?? {
              id: serverSessionId,
              cwd: "/",
              name: serverSession.name || "Session",
              forkParentSessionId: serverSession.forked_from ?? null,
              agentType: serverSession.agent_type,
              tags: serverSession.tags ?? [],
              mcpServers: undefined,
              personaId: serverSession.persona_id ?? null,
              workspaceId: serverSession.workspace_id ?? null,
              workspaceGroupId: serverSession.workspace_group_id ?? null,
              scopeSnapshotId: serverSession.scope_snapshot_id ?? null,
              policySnapshotVersion: serverSession.policy_snapshot_version ?? null,
              policySnapshotFingerprint: serverSession.policy_snapshot_fingerprint ?? null,
              policySnapshotRefreshedAt: toIsoDate(serverSession.policy_snapshot_refreshed_at),
              policySummary: serverSession.policy_summary ?? null,
              policyProvenanceSummary: serverSession.policy_provenance_summary ?? null,
              policyRefreshError: serverSession.policy_refresh_error ?? null,
              state: fallbackState,
              capabilities: undefined,
              sandboxSessionId: null,
              sandboxRunId: null,
              sshWsUrl: null,
              sshUser: null,
              backendStatus,
              messageCount: serverSession.message_count,
              usage: serverSession.usage ?? null,
              lastActivityAt: toIsoDate(serverSession.last_activity_at),
              updates: [],
              pendingPermissions: [],
              createdAt: toIsoDate(serverSession.created_at) ?? new Date(),
              updatedAt: toIsoDate(serverSession.last_activity_at) ?? new Date(),
            }

            const nextState =
              existing && shouldKeepLiveState(existing.state) && backendStatus === "active"
                ? existing.state
                : fallbackState

            const nextUpdatedAt = toIsoDate(serverSession.last_activity_at) ?? baseSession.updatedAt
            const mergedUsage = mergeUsage(existing?.usage, serverSession.usage)

            mergedSessions[serverSessionId] = {
              ...baseSession,
              id: serverSessionId,
              name: serverSession.name || baseSession.name,
              forkParentSessionId: serverSession.forked_from ?? baseSession.forkParentSessionId ?? null,
              agentType: serverSession.agent_type || baseSession.agentType,
              tags: serverSession.tags ?? baseSession.tags,
              personaId: serverSession.persona_id ?? baseSession.personaId ?? null,
              workspaceId: serverSession.workspace_id ?? baseSession.workspaceId ?? null,
              workspaceGroupId: serverSession.workspace_group_id ?? baseSession.workspaceGroupId ?? null,
              scopeSnapshotId: serverSession.scope_snapshot_id ?? baseSession.scopeSnapshotId ?? null,
              policySnapshotVersion:
                serverSession.policy_snapshot_version ?? baseSession.policySnapshotVersion ?? null,
              policySnapshotFingerprint:
                serverSession.policy_snapshot_fingerprint ?? baseSession.policySnapshotFingerprint ?? null,
              policySnapshotRefreshedAt:
                toIsoDate(serverSession.policy_snapshot_refreshed_at)
                ?? baseSession.policySnapshotRefreshedAt
                ?? null,
              policySummary: serverSession.policy_summary ?? baseSession.policySummary ?? null,
              policyProvenanceSummary:
                serverSession.policy_provenance_summary
                ?? baseSession.policyProvenanceSummary
                ?? null,
              policyRefreshError:
                serverSession.policy_refresh_error ?? baseSession.policyRefreshError ?? null,
              backendStatus,
              messageCount: Math.max(baseSession.messageCount ?? 0, serverSession.message_count),
              usage: mergedUsage,
              lastActivityAt: toIsoDate(serverSession.last_activity_at),
              createdAt: toIsoDate(serverSession.created_at) ?? baseSession.createdAt,
              updatedAt: nextUpdatedAt,
              state: nextState,
            }
          }

          return { sessions: mergedSessions }
        })
      },

      applySessionDetail: (detail) => {
        set((state) => {
          const sessionId = detail.session_id
          if (!sessionId) {
            return state
          }
          const existing = state.sessions[sessionId]
          const backendStatus = normalizeBackendSessionStatus(detail.status)
          const fallbackState = resolveStateFromBackendStatus(
            backendStatus,
            detail.has_websocket
          )
          const nextState =
            existing && shouldKeepLiveState(existing.state) && backendStatus === "active"
              ? existing.state
              : fallbackState

          const baseSession: ACPSession = existing ?? {
            id: sessionId,
            cwd: detail.cwd || "/",
            name: detail.name || "Session",
            forkParentSessionId: detail.forked_from ?? null,
            agentType: detail.agent_type,
            tags: detail.tags ?? [],
            mcpServers: undefined,
            personaId: detail.persona_id ?? null,
            workspaceId: detail.workspace_id ?? null,
            workspaceGroupId: detail.workspace_group_id ?? null,
            scopeSnapshotId: detail.scope_snapshot_id ?? null,
            policySnapshotVersion: detail.policy_snapshot_version ?? null,
            policySnapshotFingerprint: detail.policy_snapshot_fingerprint ?? null,
            policySnapshotRefreshedAt: toIsoDate(detail.policy_snapshot_refreshed_at),
            policySummary: detail.policy_summary ?? null,
            policyProvenanceSummary: detail.policy_provenance_summary ?? null,
            policyRefreshError: detail.policy_refresh_error ?? null,
            state: nextState,
            capabilities: undefined,
            sandboxSessionId: null,
            sandboxRunId: null,
            sshWsUrl: null,
            sshUser: null,
            backendStatus,
            messageCount: detail.message_count,
            usage: detail.usage ?? null,
            lastActivityAt: toIsoDate(detail.last_activity_at),
            updates: [],
            pendingPermissions: [],
            createdAt: toIsoDate(detail.created_at) ?? new Date(),
            updatedAt: toIsoDate(detail.last_activity_at) ?? new Date(),
          }

          return {
            sessions: {
              ...state.sessions,
              [sessionId]: {
                ...baseSession,
                name: detail.name || baseSession.name,
                cwd: detail.cwd || baseSession.cwd,
                forkParentSessionId: detail.forked_from ?? baseSession.forkParentSessionId ?? null,
                agentType: detail.agent_type || baseSession.agentType,
                tags: detail.tags ?? baseSession.tags,
                personaId: detail.persona_id ?? baseSession.personaId ?? null,
                workspaceId: detail.workspace_id ?? baseSession.workspaceId ?? null,
                workspaceGroupId: detail.workspace_group_id ?? baseSession.workspaceGroupId ?? null,
                scopeSnapshotId: detail.scope_snapshot_id ?? baseSession.scopeSnapshotId ?? null,
                policySnapshotVersion:
                  detail.policy_snapshot_version ?? baseSession.policySnapshotVersion ?? null,
                policySnapshotFingerprint:
                  detail.policy_snapshot_fingerprint ?? baseSession.policySnapshotFingerprint ?? null,
                policySnapshotRefreshedAt:
                  toIsoDate(detail.policy_snapshot_refreshed_at)
                  ?? baseSession.policySnapshotRefreshedAt
                  ?? null,
                policySummary: detail.policy_summary ?? baseSession.policySummary ?? null,
                policyProvenanceSummary:
                  detail.policy_provenance_summary
                  ?? baseSession.policyProvenanceSummary
                  ?? null,
                policyRefreshError:
                  detail.policy_refresh_error ?? baseSession.policyRefreshError ?? null,
                backendStatus,
                messageCount: Math.max(baseSession.messageCount ?? 0, detail.message_count),
                usage: mergeUsage(baseSession.usage, detail.usage),
                lastActivityAt: toIsoDate(detail.last_activity_at),
                createdAt: toIsoDate(detail.created_at) ?? baseSession.createdAt,
                updatedAt: toIsoDate(detail.last_activity_at) ?? baseSession.updatedAt,
                state: nextState,
              },
            },
          }
        })
      },

      applySessionUsage: (usagePayload) => {
        set((state) => {
          const sessionId = usagePayload.session_id
          if (!sessionId) {
            return state
          }
          const existing = state.sessions[sessionId]
          if (!existing) {
            return state
          }

          const incomingUsage = usagePayload.usage ?? {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
          }

          return {
            sessions: {
              ...state.sessions,
              [sessionId]: {
                ...existing,
                usage: mergeUsage(existing.usage, incomingUsage),
                messageCount: Math.max(existing.messageCount ?? 0, usagePayload.message_count),
                lastActivityAt: toIsoDate(usagePayload.last_activity_at),
                updatedAt: toIsoDate(usagePayload.last_activity_at) ?? existing.updatedAt,
              },
            },
          }
        })
      },

      createSession: (options: CreateSessionOptions) => {
        const id = generateId()
        const now = new Date()

        // Generate name from cwd if not provided
        const generatedName = options.name || generateSessionName(options.cwd)

        const session: ACPSession = {
          id,
          cwd: options.cwd,
          name: generatedName,
          forkParentSessionId: null,
          agentType: options.agentType,
          tags: options.tags,
          mcpServers: options.mcpServers,
          personaId: options.personaId ?? null,
          workspaceId: options.workspaceId ?? null,
          workspaceGroupId: options.workspaceGroupId ?? null,
          scopeSnapshotId: options.scopeSnapshotId ?? null,
          policySnapshotVersion: null,
          policySnapshotFingerprint: null,
          policySnapshotRefreshedAt: null,
          policySummary: null,
          policyProvenanceSummary: null,
          policyRefreshError: null,
          state: "disconnected",
          capabilities: undefined,
          sandboxSessionId: null,
          sandboxRunId: null,
          sshWsUrl: null,
          sshUser: null,
          backendStatus: null,
          messageCount: 0,
          usage: null,
          lastActivityAt: null,
          updates: [],
          pendingPermissions: [],
          createdAt: now,
          updatedAt: now,
        }

        set((state) => ({
          sessions: { ...state.sessions, [id]: session },
          activeSessionId: id,
        }))

        return id
      },

      updateSessionState: (sessionId: string, sessionState: ACPSessionState) => {
        set((state) => {
          const session = state.sessions[sessionId]
          if (!session) return state

          return {
            sessions: {
              ...state.sessions,
              [sessionId]: {
                ...session,
                state: sessionState,
                updatedAt: new Date(),
              },
            },
          }
        })
      },

      updateSessionName: (sessionId: string, name: string) => {
        set((state) => {
          const session = state.sessions[sessionId]
          if (!session) return state

          return {
            sessions: {
              ...state.sessions,
              [sessionId]: {
                ...session,
                name,
                updatedAt: new Date(),
              },
            },
          }
        })
      },

      updateSessionTags: (sessionId: string, tags: string[]) => {
        set((state) => {
          const session = state.sessions[sessionId]
          if (!session) return state

          return {
            sessions: {
              ...state.sessions,
              [sessionId]: {
                ...session,
                tags,
                updatedAt: new Date(),
              },
            },
          }
        })
      },

      updateSessionMetadata: (sessionId: string, metadata) => {
        set((state) => {
          const session = state.sessions[sessionId]
          if (!session) return state

          return {
            sessions: {
              ...state.sessions,
              [sessionId]: {
                ...session,
                ...metadata,
                updatedAt: new Date(),
              },
            },
          }
        })
      },

      replaceSessionId: (localSessionId: string, serverSessionId: string, updates) => {
        set((state) => {
          const session = state.sessions[localSessionId]
          if (!session) return state

          const updated: ACPSession = {
            ...session,
            ...updates,
            id: serverSessionId,
            backendStatus: updates?.backendStatus ?? session.backendStatus ?? "active",
            updatedAt: new Date(),
          }

          const sessions = { ...state.sessions }
          delete sessions[localSessionId]
          sessions[serverSessionId] = updated

          return {
            sessions,
            activeSessionId: serverSessionId,
          }
        })
      },

      setSessionCapabilities: (sessionId: string, capabilities: Record<string, unknown>) => {
        set((state) => {
          const session = state.sessions[sessionId]
          if (!session) return state

          return {
            sessions: {
              ...state.sessions,
              [sessionId]: {
                ...session,
                capabilities,
                updatedAt: new Date(),
              },
            },
          }
        })
      },

      addUpdate: (sessionId: string, update: Omit<ACPUpdate, "timestamp">) => {
        set((state) => {
          const session = state.sessions[sessionId]
          if (!session) return state

          const newUpdate: ACPUpdate = {
            ...update,
            timestamp: new Date(),
          }

          // Limit updates in memory
          let updates = [...session.updates, newUpdate]
          if (updates.length > SESSION_CONFIG.MAX_UPDATES_PER_SESSION) {
            updates = updates.slice(-SESSION_CONFIG.MAX_UPDATES_PER_SESSION)
          }

          const updateData = newUpdate.data as Record<string, unknown>
          const usageDelta = getUsageIncrementFromUpdate(updateData)
          const currentUsage = session.usage ?? {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
          }
          const nextUsage = usageDelta
            ? {
                prompt_tokens: currentUsage.prompt_tokens + usageDelta.prompt_tokens,
                completion_tokens: currentUsage.completion_tokens + usageDelta.completion_tokens,
                total_tokens: currentUsage.total_tokens + usageDelta.total_tokens,
              }
            : session.usage ?? null

          const messageCountIncrement = MESSAGE_UPDATE_TYPES.has(newUpdate.type) ? 1 : 0
          const nextMessageCount = (session.messageCount ?? 0) + messageCountIncrement

          return {
            sessions: {
              ...state.sessions,
              [sessionId]: {
                ...session,
                updates,
                usage: nextUsage,
                messageCount: nextMessageCount,
                lastActivityAt: newUpdate.timestamp,
                updatedAt: new Date(),
              },
            },
          }
        })
      },

      addPendingPermission: (sessionId: string, permission: ACPPendingPermission) => {
        set((state) => {
          const session = state.sessions[sessionId]
          if (!session) return state

          return {
            sessions: {
              ...state.sessions,
              [sessionId]: {
                ...session,
                pendingPermissions: [...session.pendingPermissions, permission],
                state: "waiting_permission",
                updatedAt: new Date(),
              },
            },
          }
        })
      },

      removePendingPermission: (sessionId: string, requestId: string) => {
        set((state) => {
          const session = state.sessions[sessionId]
          if (!session) return state

          const pendingPermissions = session.pendingPermissions.filter(
            (p) => p.request_id !== requestId
          )

          return {
            sessions: {
              ...state.sessions,
              [sessionId]: {
                ...session,
                pendingPermissions,
                // If no more pending permissions, go back to running
                state: pendingPermissions.length === 0 ? "running" : "waiting_permission",
                updatedAt: new Date(),
              },
            },
          }
        })
      },

      clearPendingPermissions: (sessionId: string) => {
        set((state) => {
          const session = state.sessions[sessionId]
          if (!session) return state

          return {
            sessions: {
              ...state.sessions,
              [sessionId]: {
                ...session,
                pendingPermissions: [],
                updatedAt: new Date(),
              },
            },
          }
        })
      },

      setActiveSession: (sessionId: string | null) => {
        set({ activeSessionId: sessionId })
      },

      closeSession: (sessionId: string) => {
        set((state) => {
          const { [sessionId]: removed, ...rest } = state.sessions
          return {
            sessions: rest,
            activeSessionId:
              state.activeSessionId === sessionId ? null : state.activeSessionId,
          }
        })
      },

      clearSessionUpdates: (sessionId: string) => {
        set((state) => {
          const session = state.sessions[sessionId]
          if (!session) return state

          return {
            sessions: {
              ...state.sessions,
              [sessionId]: {
                ...session,
                updates: [],
                updatedAt: new Date(),
              },
            },
          }
        })
      },

      setIsCreating: (isCreating: boolean) => {
        set({ isCreating })
      },

      setGlobalError: (error: string | null) => {
        set({ globalError: error })
      },

      getSession: (sessionId: string) => {
        return get().sessions[sessionId]
      },

      getSessions: () => {
        const sessions = get().sessions
        return Object.values(sessions).sort(
          (a, b) => b.updatedAt.getTime() - a.updatedAt.getTime()
        )
      },

      cleanupExpiredSessions: () => {
        const now = Date.now()
        set((state) => {
          const sessions: Record<string, ACPSession> = {}
          let activeSessionId = state.activeSessionId

          for (const [id, session] of Object.entries(state.sessions)) {
            const age = now - session.updatedAt.getTime()
            if (age < SESSION_CONFIG.SESSION_EXPIRY_MS) {
              sessions[id] = session
            } else {
              // Session expired
              if (activeSessionId === id) {
                activeSessionId = null
              }
            }
          }

          return { sessions, activeSessionId }
        })
      },

      reset: () => {
        set(initialState)
      },
    }),
    {
      name: STORAGE_KEY,
      storage: createJSONStorage(() => createACPStorage()),
      partialize: (state): PersistedState => ({
        // Only persist session metadata, not transient state like updates
        sessions: Object.fromEntries(
          Object.entries(state.sessions).map(([id, session]) => [
            id,
            {
              ...session,
              // Keep the most recent 50 updates for context after refresh
              updates: session.updates.slice(-50),
              // Don't persist pending permissions (they expire)
              pendingPermissions: [],
              // Reset state to disconnected (connection needs re-establishment)
              state: "disconnected" as const,
              // Don't persist sensitive SSH connection details
              sshWsUrl: null,
              sshUser: null,
              // Don't persist sandbox session IDs (they expire)
              sandboxSessionId: null,
              sandboxRunId: null,
            },
          ])
        ),
        activeSessionId: state.activeSessionId,
      }),
      onRehydrateStorage: () => (state) => {
        if (state) {
          // Convert date strings back to Date objects
          const sessions: Record<string, ACPSession> = {}
          for (const [id, session] of Object.entries(state.sessions)) {
            sessions[id] = {
              ...session,
              createdAt:
                typeof session.createdAt === "string"
                  ? new Date(session.createdAt)
                  : session.createdAt,
              updatedAt:
                typeof session.updatedAt === "string"
                  ? new Date(session.updatedAt)
                  : session.updatedAt,
              lastActivityAt:
                typeof session.lastActivityAt === "string"
                  ? new Date(session.lastActivityAt)
                  : session.lastActivityAt ?? null,
              updates: session.updates.map((u) => ({
                ...u,
                timestamp:
                  typeof u.timestamp === "string" ? new Date(u.timestamp) : u.timestamp,
              })),
              pendingPermissions: session.pendingPermissions.map((p) => ({
                ...p,
                requestedAt:
                  typeof p.requestedAt === "string"
                    ? new Date(p.requestedAt)
                    : p.requestedAt,
              })),
            }
          }
          state.sessions = sessions

          // Cleanup expired sessions on load
          setTimeout(() => {
            useACPSessionsStore.getState().cleanupExpiredSessions()
          }, 0)
        }
      },
    }
  )
)

// Expose for debugging
if (typeof window !== "undefined") {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ;(window as any).__tldw_useACPSessionsStore = useACPSessionsStore
}
