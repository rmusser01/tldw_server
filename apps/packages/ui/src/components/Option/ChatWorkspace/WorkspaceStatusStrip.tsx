import { getDesignSystemState } from "@/design-system"

export type WorkspaceStatusStripProps = {
  backendAvailable: boolean
  streaming: boolean
  stagedSourceCount: number
}

const statusPillClass =
  "inline-flex min-h-[24px] items-center rounded-md border border-border bg-surface px-2 text-xs font-medium text-text"

const READY_STATE_LABEL = getDesignSystemState("ready").label

export const WorkspaceStatusStrip = ({
  backendAvailable,
  streaming,
  stagedSourceCount
}: WorkspaceStatusStripProps) => {
  const runtimeLabel = !backendAvailable
    ? "Server unavailable"
    : streaming
      ? "Streaming"
      : READY_STATE_LABEL

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
      </div>
      <div className="flex min-w-0 flex-wrap items-center gap-2">
        <span>Ctrl+K command</span>
        <span>Ctrl+Enter send</span>
      </div>
    </footer>
  )
}
