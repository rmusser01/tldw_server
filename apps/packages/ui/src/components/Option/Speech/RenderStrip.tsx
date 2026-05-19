import React from "react"
import { Button, Tag, Tooltip, notification } from "antd"
import { Edit3, Play, RefreshCw, Trash2, X } from "lucide-react"
import { UnifiedAudioPlayer } from "@/components/Common/UnifiedAudioPlayer"
import { TtsJobProgress, type TtsJobProgressStep } from "@/components/Common/TtsJobProgress"

export type RenderStripState = "idle" | "generating" | "ready" | "playing" | "error"

export type RenderStripConfig = {
  provider: string
  voice?: string
  model?: string
  format?: string
  speed?: number
}

export type RenderStripProps = {
  id: string
  state: RenderStripState
  config: RenderStripConfig
  audioUrl?: string
  audioBlob?: Blob
  errorMessage?: string
  errorSettingsHref?: string
  /** Progress 0-100 for long-running generations */
  progress?: number
  /** Whether this strip's audio is currently playing */
  isPlaying?: boolean
  /** Force pause (when another strip starts playing) */
  forcePaused?: boolean
  /** Generation progress steps for long-running jobs */
  jobSteps?: TtsJobProgressStep[]
  jobCurrentStep?: number
  jobMessage?: string
  jobEta?: number
  onGenerate?: (id: string) => void
  onRemove?: (id: string) => void
  onEdit?: (id: string) => void
  onPlay?: (id: string) => void
  onPause?: (id: string) => void
  /** Called when audio playback ends naturally */
  onEnd?: (id: string) => void
  onRetry?: (id: string) => void
  onConfigTagClick?: (id: string, field: string) => void
}

const DEFAULT_JOB_STEPS: TtsJobProgressStep[] = [
  { key: "queued", label: "Queued" },
  { key: "synthesizing", label: "Synthesizing" },
  { key: "complete", label: "Complete" }
]

export const RenderStrip: React.FC<RenderStripProps> = ({
  id,
  state,
  config,
  audioUrl,
  audioBlob,
  errorMessage,
  errorSettingsHref,
  progress,
  isPlaying,
  forcePaused,
  jobSteps,
  jobCurrentStep,
  jobMessage,
  jobEta,
  onGenerate,
  onRemove,
  onEdit,
  onPlay,
  onPause,
  onEnd,
  onRetry,
  onConfigTagClick
}) => {
  const [undoPending, setUndoPending] = React.useState(false)
  const undoTimerRef = React.useRef<ReturnType<typeof setTimeout> | null>(null)

  const handleRemove = () => {
    setUndoPending(true)
    undoTimerRef.current = setTimeout(() => {
      setUndoPending(false)
      onRemove?.(id)
    }, 4000)
    notification.info({
      key: `undo-${id}`,
      message: "Render strip removed",
      description: "Click Undo to restore it.",
      btn: (
        <Button
          size="small"
          onClick={() => {
            if (undoTimerRef.current) {
              clearTimeout(undoTimerRef.current)
              undoTimerRef.current = null
            }
            setUndoPending(false)
            notification.destroy(`undo-${id}`)
          }}
        >
          Undo
        </Button>
      ),
      duration: 4
    })
  }

  React.useEffect(() => {
    return () => {
      if (undoTimerRef.current) clearTimeout(undoTimerRef.current)
    }
  }, [])

  if (undoPending) return null

  const providerLabel =
    config.provider === "browser"
      ? "Browser preview"
      : config.provider === "tldw"
        ? (config.model || "tldw")
        : config.provider
  const ariaConfig = [providerLabel, config.voice].filter(Boolean).join(" ")

  const isGenerating = state === "generating"
  const isReady = state === "ready" || state === "playing"
  const isError = state === "error"
  const isIdle = state === "idle"

  return (
    <div
      role="region"
      aria-label={`Render strip: ${ariaConfig}`}
      className="rounded-lg border border-border bg-card p-3 transition-colors hover:border-border-hover"
      data-strip-id={id}
      data-strip-state={state}
    >
      {/* Config tags row */}
      <div className="mb-2 flex flex-wrap items-center gap-1.5">
        <Tooltip title={`Provider: ${config.provider}`}>
          <Tag
            className="cursor-pointer"
            onClick={() => onConfigTagClick?.(id, "provider")}
          >
            {providerLabel}
          </Tag>
        </Tooltip>

        {config.voice && (
          <Tooltip title={`Voice: ${config.voice}`}>
            <Tag
              className="cursor-pointer"
              onClick={() => onConfigTagClick?.(id, "voice")}
            >
              {config.voice}
            </Tag>
          </Tooltip>
        )}

        {config.format && (
          <Tooltip title={`Format: ${config.format}`}>
            <Tag
              className="cursor-pointer"
              onClick={() => onConfigTagClick?.(id, "format")}
            >
              {config.format.toUpperCase()}
            </Tag>
          </Tooltip>
        )}

        {config.speed && config.speed !== 1 && (
          <Tooltip title={`Speed: ${config.speed}x`}>
            <Tag
              className="cursor-pointer"
              onClick={() => onConfigTagClick?.(id, "speed")}
            >
              {config.speed}x
            </Tag>
          </Tooltip>
        )}

        <div className="flex-1" />

        {/* Action buttons */}
        {isIdle && (
          <Tooltip title="Generate">
            <Button
              type="primary"
              size="small"
              icon={<Play className="h-3.5 w-3.5" />}
              onClick={() => onGenerate?.(id)}
              aria-label="Generate audio"
            >
              Generate
            </Button>
          </Tooltip>
        )}

        <Tooltip title="Edit config">
          <Button
            type="text"
            size="small"
            icon={<Edit3 className="h-3.5 w-3.5" />}
            onClick={() => onEdit?.(id)}
            aria-label="Edit configuration"
          />
        </Tooltip>

        <Tooltip title="Remove">
          <Button
            type="text"
            size="small"
            danger
            icon={<Trash2 className="h-3.5 w-3.5" />}
            onClick={handleRemove}
            aria-label="Remove render strip"
          />
        </Tooltip>
      </div>

      {/* Generating state */}
      {isGenerating && (
        <TtsJobProgress
          title="Generating audio..."
          steps={jobSteps || DEFAULT_JOB_STEPS}
          currentStep={jobCurrentStep ?? 0}
          percent={progress}
          message={jobMessage}
          status="running"
          etaSeconds={jobEta}
        />
      )}

      {/* Ready/Playing state — unified audio player */}
      {isReady && audioUrl && (
        <UnifiedAudioPlayer
          audioUrl={audioUrl}
          audioBlob={audioBlob}
          label={`${providerLabel} ${config.voice} audio`}
          compact
          format={config.format}
          downloadFilename={`tts-${config.provider}-${config.voice}-${Date.now()}`}
          forcePaused={forcePaused}
          onPlay={() => onPlay?.(id)}
          onPause={() => onPause?.(id)}
          onEnd={() => onEnd?.(id)}
        />
      )}

      {/* Error state */}
      {isError && (
        <div className="flex items-center gap-2 rounded border border-red-200 bg-red-50 px-3 py-2 dark:border-red-900 dark:bg-red-950/30">
          <X className="h-4 w-4 shrink-0 text-red-500" />
          <span className="flex-1 text-sm text-red-700 dark:text-red-400">
            {errorMessage || "Generation failed"}
          </span>
          {errorSettingsHref && (
            <Button size="small" type="link" href={errorSettingsHref}>
              Open Settings
            </Button>
          )}
          <Button
            size="small"
            icon={<RefreshCw className="h-3.5 w-3.5" />}
            onClick={() => onRetry?.(id)}
          >
            Retry
          </Button>
        </div>
      )}
    </div>
  )
}

export default RenderStrip
