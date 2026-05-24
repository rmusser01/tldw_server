import { trackResearchWorkspaceTelemetry } from "@/utils/research-workspace-telemetry"

export const WORKSPACE_UNDO_WINDOW_MS = 10000
const WORKSPACE_UNDO_MAX_STACK_SIZE = 10

export interface WorkspaceUndoActionHandle {
  id: string
  expiresAt: number
}

interface PendingWorkspaceUndoAction {
  id: string
  expiresAt: number
  undo: () => void
  finalize: () => void
  timer: ReturnType<typeof setTimeout>
}

interface ScheduleWorkspaceUndoActionInput {
  apply: () => void
  undo: () => void
  finalize?: () => void
  timeoutMs?: number
}

const pendingWorkspaceUndoActions = new Map<string, PendingWorkspaceUndoAction>()

/**
 * Insertion-order stack of action IDs. Most recent is last.
 * Capped at WORKSPACE_UNDO_MAX_STACK_SIZE entries.
 */
const undoStack: string[] = []

const createWorkspaceUndoActionId = (): string => {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID()
  }
  return `workspace-undo-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
}

const clearPendingWorkspaceUndoAction = (
  id: string
): PendingWorkspaceUndoAction | null => {
  const action = pendingWorkspaceUndoActions.get(id)
  if (!action) return null
  clearTimeout(action.timer)
  pendingWorkspaceUndoActions.delete(id)
  const stackIndex = undoStack.indexOf(id)
  if (stackIndex !== -1) undoStack.splice(stackIndex, 1)
  return action
}

export const scheduleWorkspaceUndoAction = ({
  apply,
  undo,
  finalize,
  timeoutMs = WORKSPACE_UNDO_WINDOW_MS
}: ScheduleWorkspaceUndoActionInput): WorkspaceUndoActionHandle => {
  apply()

  const id = createWorkspaceUndoActionId()
  const expiresAt = Date.now() + timeoutMs

  const timer = setTimeout(() => {
    const action = clearPendingWorkspaceUndoAction(id)
    if (!action) return
    action.finalize()
  }, timeoutMs)

  // Evict oldest entry when at capacity — use shift() directly to avoid
  // the redundant indexOf scan that clearPendingWorkspaceUndoAction performs.
  if (undoStack.length >= WORKSPACE_UNDO_MAX_STACK_SIZE) {
    const oldestId = undoStack.shift()
    if (oldestId) {
      const evicted = pendingWorkspaceUndoActions.get(oldestId)
      if (evicted) {
        clearTimeout(evicted.timer)
        pendingWorkspaceUndoActions.delete(oldestId)
        evicted.finalize()
      }
    }
  }

  pendingWorkspaceUndoActions.set(id, {
    id,
    expiresAt,
    undo,
    finalize: finalize || (() => {}),
    timer
  })
  undoStack.push(id)

  return { id, expiresAt }
}

export const undoWorkspaceAction = (id: string): boolean => {
  const action = clearPendingWorkspaceUndoAction(id)
  if (!action) return false
  action.undo()
  void trackResearchWorkspaceTelemetry({
    type: "undo_triggered"
  })
  return true
}

/**
 * Undo the most recent pending action (Cmd+Z / Ctrl+Z).
 * Returns true if an action was undone.
 */
export const undoLatestWorkspaceAction = (): boolean => {
  if (undoStack.length === 0) return false
  const latestId = undoStack[undoStack.length - 1]
  if (!latestId) return false
  return undoWorkspaceAction(latestId)
}

export const getWorkspaceUndoPendingCount = (): number =>
  pendingWorkspaceUndoActions.size

/**
 * Finalize and discard all pending undo actions.
 * Call on workspace switch to prevent cross-workspace undo.
 */
export const clearAllPendingUndoActions = (): void => {
  for (const action of pendingWorkspaceUndoActions.values()) {
    clearTimeout(action.timer)
    action.finalize()
  }
  pendingWorkspaceUndoActions.clear()
  undoStack.length = 0
}

export const clearWorkspaceUndoActionsForTests = (): void => {
  for (const action of pendingWorkspaceUndoActions.values()) {
    clearTimeout(action.timer)
  }
  pendingWorkspaceUndoActions.clear()
  undoStack.length = 0
}
