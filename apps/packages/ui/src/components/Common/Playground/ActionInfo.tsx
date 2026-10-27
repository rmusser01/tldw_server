import React from "react"
import { useTranslation } from "react-i18next"
import { Globe, Cpu, Database, Sparkles, Loader2 } from "lucide-react"

type Props = {
  action: string
  /** Seconds since the reply started; shown after a few seconds (CM-06). */
  elapsedSeconds?: number
}

/** Show the elapsed time once a wait is long enough to wonder about. */
const SHOW_ELAPSED_AFTER_SECONDS = 5
/** Spoken updates during a long wait, replacing a 5 s "checkpoint" count (CM-06). */
const STILL_WORKING_AFTER_SECONDS = 10
const TAKING_A_WHILE_AFTER_SECONDS = 30

const formatElapsed = (seconds: number) =>
  `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`

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

export const ActionInfo = ({ action, elapsedSeconds = 0 }: Props) => {
  const { t } = useTranslation("common")

  const config = actionConfig[action]
  const IconComponent = config?.icon || Loader2
  const label = config
    ? t(config.labelKey, config.defaultLabel)
    : t(`actionInfo.${action}`, action)

  return (
    <div
      className="inline-flex items-center gap-2 px-3 py-2 rounded-full bg-primary/10 border border-primary/30"
      role="status"
      aria-live="polite"
      data-testid="chat-loading-status"
    >
      <IconComponent className="size-4 text-primary animate-pulse" />
      <span className="text-sm font-medium text-primary shimmer-text">
        {label}
      </span>
      {elapsedSeconds >= SHOW_ELAPSED_AFTER_SECONDS && (
        // The time is visual only; the spoken updates come from LoadingStatus.
        <span aria-hidden="true" className="text-xs tabular-nums text-text-muted">
          {formatElapsed(elapsedSeconds)}
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
  /** "model (Provider)", named in the long-wait announcements (CM-06). */
  modelLabel?: string
}

export const LoadingStatus = ({
  isProcessing,
  isStreaming,
  isSearchingInternet,
  isEmbedding,
  actionInfo,
  modelLabel
}: LoadingStatusProps) => {
  const { t } = useTranslation("common")
  const [completionAnnouncement, setCompletionAnnouncement] = React.useState<string | null>(null)
  const [liveAnnouncement, setLiveAnnouncement] = React.useState<string | null>(null)
  const wasActiveRef = React.useRef(false)
  const previousActionRef = React.useRef<string | null>(null)
  const [elapsedSeconds, setElapsedSeconds] = React.useState(0)

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

  // Count the seconds of this reply; a new reply starts again from zero.
  React.useEffect(() => {
    setElapsedSeconds(0)
    if (!isActive) return
    const startedAt = Date.now()
    const timer = window.setInterval(() => {
      setElapsedSeconds(Math.floor((Date.now() - startedAt) / 1000))
    }, 1000)
    return () => {
      window.clearInterval(timer)
    }
  }, [isActive])

  const model = modelLabel?.trim()
  React.useEffect(() => {
    if (elapsedSeconds === STILL_WORKING_AFTER_SECONDS) {
      setLiveAnnouncement(
        model
          ? t("actionInfo.stillWorkingModel", "Still working on the reply from {{model}}.", { model })
          : t("actionInfo.stillWorking", "Still working on the reply.")
      )
    } else if (elapsedSeconds === TAKING_A_WHILE_AFTER_SECONDS) {
      setLiveAnnouncement(
        model
          ? t(
              "actionInfo.takingAWhileModel",
              "{{model}} is taking a while. You can stop the reply and try again.",
              { model }
            )
          : t(
              "actionInfo.takingAWhile",
              "This reply is taking a while. You can stop it and try again."
            )
      )
    }
  }, [elapsedSeconds, model, t])

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
        <ActionInfo action={currentAction} elapsedSeconds={elapsedSeconds} />
      )}
    </>
  )
}
