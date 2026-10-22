import React from "react"
import { useTranslation } from "react-i18next"
import { Globe, Cpu, Database, Sparkles, Loader2 } from "lucide-react"

type Props = {
  action: string
  /** Shown instead of the action label while the reply has not started. */
  waitingFor?: string | null
  /** Time since the response started, shown after the label. */
  elapsedMs?: number
}

/** Elapsed wait as m:ss, e.g. 0:12 or 2:05 (CM-06). */
export const formatElapsedWait = (elapsedMs: number) => {
  const seconds = Math.max(0, Math.floor(elapsedMs / 1_000))
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`
}

/** When a long wait is announced to screen readers (CM-06). */
export const WAIT_ANNOUNCEMENT_MS = [10_000, 30_000] as const

// Screen reader only announcement component
const SrAnnouncement: React.FC<{ message: string }> = ({ message }) => (
  <div
    role="status"
    aria-live="polite"
    aria-atomic="true"
    className="sr-only"
  >
    {message}
  </div>
)

// Map action types to icons and labels
const actionConfig: Record<string, { icon: React.ElementType; labelKey: string; defaultLabel: string }> = {
  webSearch: { icon: Globe, labelKey: "actionInfo.webSearch", defaultLabel: "Searching the web..." },
  ragSearch: { icon: Database, labelKey: "actionInfo.ragSearch", defaultLabel: "Searching knowledge base..." },
  embedding: { icon: Database, labelKey: "actionInfo.embedding", defaultLabel: "Processing context..." },
  generating: { icon: Sparkles, labelKey: "actionInfo.generating", defaultLabel: "Generating response..." },
  processing: { icon: Cpu, labelKey: "actionInfo.processing", defaultLabel: "Processing..." },
  thinking: { icon: Sparkles, labelKey: "actionInfo.thinking", defaultLabel: "Thinking..." },
}

export const ActionInfo = ({ action, waitingFor, elapsedMs }: Props) => {
  const { t } = useTranslation("common")

  const config = actionConfig[action]
  const IconComponent = config?.icon || Loader2
  const label = waitingFor
    ? t("actionInfo.waitingFor", "Waiting for {{model}}...", { model: waitingFor })
    : config
      ? t(config.labelKey, config.defaultLabel)
      : t(`actionInfo.${action}`, action)

  // Screen readers hear the start and long waits from LoadingStatus; this
  // pill is not a live region, so its ticking timer is never read out.
  return (
    <div
      className="inline-flex items-center gap-2 px-3 py-2 rounded-full bg-primary/10 border border-primary/30"
      data-testid="chat-loading-status"
    >
      <IconComponent className="size-4 text-primary animate-pulse" />
      <span className="text-sm font-medium text-primary shimmer-text">
        {label}
      </span>
      {elapsedMs !== undefined && elapsedMs >= 1_000 && (
        <span
          aria-hidden="true"
          data-testid="chat-loading-elapsed"
          className="text-xs font-medium tabular-nums text-primary/80"
        >
          {formatElapsedWait(elapsedMs)}
        </span>
      )}
    </div>
  )
}

// Unified loading status component for chat messages
type LoadingStatusProps = {
  isProcessing?: boolean
  isStreaming?: boolean
  isSearchingInternet?: boolean
  isEmbedding?: boolean
  actionInfo?: string | null
  /** The model (or character) the reply comes from. */
  modelName?: string | null
  /** No part of the reply has arrived yet. */
  waitingForReply?: boolean
}

export const LoadingStatus = ({
  isProcessing,
  isStreaming,
  isSearchingInternet,
  isEmbedding,
  actionInfo,
  modelName,
  waitingForReply
}: LoadingStatusProps) => {
  const { t } = useTranslation("common")
  const [completionAnnouncement, setCompletionAnnouncement] = React.useState<string | null>(null)
  const [liveAnnouncement, setLiveAnnouncement] = React.useState<string | null>(null)
  const [elapsedMs, setElapsedMs] = React.useState(0)
  const wasActiveRef = React.useRef(false)
  const previousActionRef = React.useRef<string | null>(null)

  // Determine the current action based on state
  let currentAction: string | null = null

  if (isSearchingInternet) {
    currentAction = "webSearch"
  } else if (isEmbedding) {
    currentAction = "embedding"
  } else if (actionInfo) {
    currentAction = actionInfo
  } else if (isStreaming) {
    currentAction = "generating"
  } else if (isProcessing) {
    currentAction = "processing"
  }

  const isActive = currentAction !== null

  React.useEffect(() => {
    if (!currentAction) {
      setLiveAnnouncement(null)
      previousActionRef.current = null
      return
    }
    if (previousActionRef.current === currentAction) {
      return
    }
    previousActionRef.current = currentAction
    const startedLabel = t("actionInfo.started", "{{label}} started", {
      label: t(
        actionConfig[currentAction]?.labelKey || `actionInfo.${currentAction}`,
        actionConfig[currentAction]?.defaultLabel || currentAction
      )
    })
    setLiveAnnouncement(startedLabel)
  }, [currentAction, t])

  // A visible elapsed time, and plain-language announcements at about 10 s
  // and 30 s instead of a numbered checkpoint every 5 s (CM-06).
  const waitingForRef = React.useRef<string | null>(null)
  const model = modelName?.trim() || null
  waitingForRef.current = waitingForReply ? model : null
  const translateRef = React.useRef(t)
  translateRef.current = t
  React.useEffect(() => {
    setElapsedMs(0)
    if (!isStreaming || !isActive) return
    const startedAt = Date.now()
    let announced = 0
    const timer = window.setInterval(() => {
      const elapsed = Date.now() - startedAt
      setElapsedMs(elapsed)
      if (announced >= WAIT_ANNOUNCEMENT_MS.length || elapsed < WAIT_ANNOUNCEMENT_MS[announced])
        return
      while (announced < WAIT_ANNOUNCEMENT_MS.length && elapsed >= WAIT_ANNOUNCEMENT_MS[announced])
        announced += 1
      const waitingFor = waitingForRef.current
      const seconds = Math.round(WAIT_ANNOUNCEMENT_MS[announced - 1] / 1_000)
      const t = translateRef.current
      setLiveAnnouncement(
        waitingFor
          ? t("actionInfo.stillWaitingFor", "Still waiting for {{model}} ({{seconds}} seconds).", {
              model: waitingFor,
              seconds
            })
          : t("actionInfo.stillGenerating", "Still writing the reply ({{seconds}} seconds).", {
              seconds
            })
      )
    }, 1_000)

    return () => {
      window.clearInterval(timer)
    }
  }, [isActive, isStreaming])

  // Announce completion when transitioning from active to inactive
  React.useEffect(() => {
    if (wasActiveRef.current && !isActive) {
      // Was active, now complete - announce to screen readers
      setCompletionAnnouncement(t("actionInfo.complete", "Response complete"))
      // Clear announcement after a short delay
      const timer = setTimeout(() => setCompletionAnnouncement(null), 1000)
      return () => clearTimeout(timer)
    }
    wasActiveRef.current = isActive
  }, [isActive, t])

  return (
    <>
      {/* Live updates for start/progress checkpoints */}
      {liveAnnouncement && (
        <SrAnnouncement message={liveAnnouncement} />
      )}
      {/* Screen reader announcement for completion */}
      {completionAnnouncement && (
        <SrAnnouncement message={completionAnnouncement} />
      )}
      {/* Visual status indicator */}
      {currentAction && (
        <ActionInfo
          action={currentAction}
          waitingFor={currentAction === "generating" && waitingForReply ? model : null}
          elapsedMs={isStreaming ? elapsedMs : undefined}
        />
      )}
    </>
  )
}
