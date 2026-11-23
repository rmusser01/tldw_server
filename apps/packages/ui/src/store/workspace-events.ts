export const WORKSPACE_STORAGE_KEY = "tldw-workspace"
export const WORKSPACE_STORAGE_SPLIT_KEY_PREFIX = `${WORKSPACE_STORAGE_KEY}:workspace:`
const WORKSPACE_FRESH_INITIALIZATION_RUNTIME_MARKER =
  "__tldwResearchWorkspaceFreshInitialization"
export const WORKSPACE_STORAGE_QUOTA_EVENT =
  "tldw:workspace-storage-quota-error"
export const WORKSPACE_STORAGE_RECOVERY_EVENT =
  "tldw:workspace-storage-recovery"
export const WORKSPACE_STORAGE_CHANNEL_NAME = "tldw-workspace-sync"
export const WORKSPACE_BROADCAST_SYNC_FLAG = "tldw:workspace:broadcast-sync"
export const WORKSPACE_CONFLICT_NOTICE_THROTTLE_MS = 8000

type WorkspaceWindow = Window & {
  __TLDW_ENABLE_WORKSPACE_BROADCAST_SYNC__?: boolean
  [WORKSPACE_FRESH_INITIALIZATION_RUNTIME_MARKER]?: Set<string>
}

export const collectResearchWorkspaceLegacyLocalStorageKeys = (): string[] => {
  if (typeof window === "undefined") return []
  const keys: string[] = []
  try {
    for (let index = 0; index < window.localStorage.length; index += 1) {
      const key = window.localStorage.key(index)
      if (!key) continue
      if (
        key === WORKSPACE_STORAGE_KEY ||
        key.startsWith(WORKSPACE_STORAGE_SPLIT_KEY_PREFIX) ||
        key.startsWith("tldw:research-workspace:") ||
        key.startsWith("tldw:workspace:playground:")
      ) {
        keys.push(key)
      }
    }
  } catch {
    return []
  }
  return keys.sort()
}

export const markFreshWorkspaceInitializationForRuntime = (
  workspaceId: string
): void => {
  if (typeof window === "undefined" || !workspaceId) return
  const runtimeWindow = window as WorkspaceWindow
  const initializedWorkspaceIds =
    runtimeWindow[WORKSPACE_FRESH_INITIALIZATION_RUNTIME_MARKER] ??
    new Set<string>()
  initializedWorkspaceIds.add(workspaceId)
  runtimeWindow[WORKSPACE_FRESH_INITIALIZATION_RUNTIME_MARKER] =
    initializedWorkspaceIds
}

export const wasWorkspaceFreshlyInitializedInRuntime = (
  workspaceId: string
): boolean =>
  typeof window !== "undefined" &&
  Boolean(
    (window as WorkspaceWindow)[
      WORKSPACE_FRESH_INITIALIZATION_RUNTIME_MARKER
    ]?.has(workspaceId)
  )

export interface WorkspaceStorageQuotaEventDetail {
  key: string
  reason: string
}

export type WorkspaceStorageRecoveryAction =
  | "archived_workspace_removed"
  | "banner_image_removed"
  | "chat_session_removed"
  | "artifact_removed"
  | "retry_success"
  | "retry_failed"
  | "retry_skipped"

export interface WorkspaceStorageRecoveryEventDetail {
  key: string
  action: WorkspaceStorageRecoveryAction
  beforeBytes: number
  afterBytes: number
  recoveredBytes: number
  workspaceId?: string
  reason?: string
}

export interface WorkspaceBroadcastUpdateMessage {
  type: "workspace-storage-updated"
  key: string
  updatedAt: number
}

export const isWorkspaceBroadcastSyncEnabled = (): boolean => {
  if (typeof window === "undefined") return false

  const typedWindow = window as WorkspaceWindow
  if (
    typeof typedWindow.__TLDW_ENABLE_WORKSPACE_BROADCAST_SYNC__ === "boolean"
  ) {
    return typedWindow.__TLDW_ENABLE_WORKSPACE_BROADCAST_SYNC__
  }

  try {
    return window.localStorage.getItem(WORKSPACE_BROADCAST_SYNC_FLAG) === "1"
  } catch {
    return false
  }
}

export const isWorkspaceBroadcastUpdateMessage = (
  value: unknown
): value is WorkspaceBroadcastUpdateMessage => {
  if (!value || typeof value !== "object") return false
  const candidate = value as Partial<WorkspaceBroadcastUpdateMessage>
  return (
    candidate.type === "workspace-storage-updated" &&
    typeof candidate.key === "string" &&
    typeof candidate.updatedAt === "number"
  )
}

export const shouldSurfaceWorkspaceConflictNotice = (
  lastShownAt: number,
  nextEventAt: number,
  throttleMs: number = WORKSPACE_CONFLICT_NOTICE_THROTTLE_MS
): boolean => {
  if (lastShownAt <= 0) return true
  return nextEventAt - lastShownAt >= throttleMs
}
