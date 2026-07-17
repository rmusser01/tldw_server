import { getDesignSystemState } from "@/design-system"
import type { ChatWorkspaceAssistantSource } from "./types"

export type WorkspaceStatusStripProps = {
  backendAvailable: boolean
  workspaceReady: boolean
  streaming: boolean
  sendError?: string | null
  stagedSourceCount: number
  hasModelSelected: boolean
  selectedPersonaLabel?: string | null
  assistantSource?: ChatWorkspaceAssistantSource
}

const statusPillClass =
  "inline-flex min-h-[24px] items-center rounded-md border border-border bg-surface px-2 text-xs font-medium text-text"

const READY_STATE_LABEL = getDesignSystemState("ready").label

const getRuntimeLabel = ({
  backendAvailable,
  workspaceReady,
  streaming,
  sendError
}: Pick<
  WorkspaceStatusStripProps,
  "backendAvailable" | "workspaceReady" | "streaming" | "sendError"
>) => {
  if (!backendAvailable) return "Server unavailable"
  if (workspaceReady === false) return "Loading workspace context"
  if (sendError) return "Send failed"
  if (streaming) return "Streaming"
  return READY_STATE_LABEL
}

export const WorkspaceStatusStrip = ({
  backendAvailable,
  workspaceReady,
  streaming,
  sendError,
  stagedSourceCount,
  hasModelSelected,
  selectedPersonaLabel,
  assistantSource
}: WorkspaceStatusStripProps) => {
  const runtimeLabel = getRuntimeLabel({
    backendAvailable,
    workspaceReady,
    streaming,
    sendError
  })
  const hasPersona =
    Boolean(selectedPersonaLabel) ||
    assistantSource === "workspace" ||
    assistantSource === "unavailable"

  return (
    <footer
      aria-label="Chat workspace status"
      className="flex min-w-0 flex-wrap items-center justify-between gap-2 border-t border-border bg-surface px-3 py-2 text-xs text-text-muted"
    >
      <div className="flex min-w-0 flex-wrap items-center gap-2">
        <span className={statusPillClass}>{runtimeLabel}</span>
        {stagedSourceCount > 0 ? (
          <span className={statusPillClass}>Context staged</span>
        ) : null}
        {!backendAvailable ? (
          <span className={statusPillClass}>Reconnect server</span>
        ) : null}
        {backendAvailable && workspaceReady === false ? (
          <span className={statusPillClass}>Wait for workspace identity</span>
        ) : null}
        {backendAvailable && workspaceReady && !hasModelSelected ? (
          <span className={statusPillClass}>Select a model</span>
        ) : null}
        {backendAvailable && workspaceReady && !hasPersona ? (
          <span className={statusPillClass}>No persona</span>
        ) : null}
      </div>
      <div className="flex min-w-0 flex-wrap items-center gap-2">
        <span>Ctrl+K command</span>
        <span>Ctrl+Enter send</span>
      </div>
    </footer>
  )
}
