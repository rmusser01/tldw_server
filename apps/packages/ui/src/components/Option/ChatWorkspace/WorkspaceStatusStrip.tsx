import {
  getChatWorkspaceRuntimeLabel,
  type ChatWorkspaceAssistantSource,
  type ChatWorkspaceRuntimeStatus
} from "./types"

export type WorkspaceStatusStripProps = ChatWorkspaceRuntimeStatus & {
  stagedSourceCount: number
  hasModelSelected: boolean
  selectedPersonaLabel?: string | null
  assistantSource?: ChatWorkspaceAssistantSource
}

const statusPillClass =
  "inline-flex min-h-[24px] items-center rounded-md border border-border bg-surface px-2 text-xs font-medium text-text"

export const WorkspaceStatusStrip = ({
  backendAvailable,
  workspaceReady,
  connectionMode,
  streaming,
  sending,
  historyLoading,
  historyLoadError,
  sendError,
  stagedSourceCount,
  hasModelSelected,
  selectedPersonaLabel,
  assistantSource
}: WorkspaceStatusStripProps) => {
  const runtimeLabel = getChatWorkspaceRuntimeLabel({
    backendAvailable,
    workspaceReady,
    connectionMode,
    streaming,
    sending,
    historyLoading,
    historyLoadError,
    sendError,
    hasModelSelected
  })
  const hasPersona =
    Boolean(selectedPersonaLabel) ||
    assistantSource === "workspace" ||
    assistantSource === "unavailable"

  return (
    <footer
      aria-label="Chat workspace status"
      className="flex min-w-0 shrink-0 flex-wrap items-center justify-between gap-2 border-t border-border bg-surface px-3 py-2 text-xs text-text-muted"
    >
      <div
        role="status"
        aria-atomic="true"
        className="flex min-w-0 flex-wrap items-center gap-2"
      >
        <span className="sr-only">
          {stagedSourceCount} source{stagedSourceCount === 1 ? "" : "s"} staged.
        </span>
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
        {backendAvailable && workspaceReady && !hasModelSelected && runtimeLabel !== "Select a model" ? (
          <span className={statusPillClass}>Select a model</span>
        ) : null}
        {backendAvailable && workspaceReady && !hasPersona ? (
          <span className={statusPillClass}>No persona</span>
        ) : null}
      </div>
      <div className="hidden min-w-0 flex-wrap items-center gap-2 xl:flex">
        <span>Ctrl+K command</span>
        <span>Ctrl+Enter send</span>
      </div>
    </footer>
  )
}
