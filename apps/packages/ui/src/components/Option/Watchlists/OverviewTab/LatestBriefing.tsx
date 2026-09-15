import React, { useEffect, useRef, useState } from "react"
import { Button, Checkbox, Modal, Tag } from "antd"
import { AlertCircle, Download, FileText, Headphones, Play, Pause, RotateCcw, ScrollText } from "lucide-react"
import { useTranslation } from "react-i18next"
import type { TFunction } from "i18next"
import type {
  WatchlistBriefingProjection,
  WatchlistBriefingRetryStage,
  WatchlistRunAudioStatus
} from "@/types/watchlists"
import {
  createWatchlistArtifactObjectUrl,
  fetchWatchlistArtifactBlob,
  fetchWatchlistArtifactText,
  revokeWatchlistArtifactObjectUrl,
  type WatchlistArtifactErrorKind
} from "@/services/watchlists-artifacts"
import { hasCurrentBriefingAudio } from "../shared/watchlists-announcements"

type RetryOptions = { confirm_unknown_delivery_retry?: boolean }
type DeliveryReview = {
  adapter: "email" | "chatbook"
  stageName: WatchlistBriefingRetryStage
  occurrenceId: number
  runId: number
  jobId: number
  outcome: string
  destinationOwner: string
  destinationLabel: string
}

export interface LatestBriefingProps {
  projection: WatchlistBriefingProjection | null
  emptyJobId?: number
  nextRunAt?: string | null
  timezone?: string
  unreadCount?: number
  newCount?: number
  onPlay: (
    audio: WatchlistRunAudioStatus | null,
    briefing: WatchlistBriefingProjection
  ) => void
  onOpenReport: (outputId: number, briefing: WatchlistBriefingProjection) => void
  onInspectRun: (runId: number) => void
  onRetryStage: (
    runId: number,
    stage: WatchlistBriefingRetryStage,
    options?: RetryOptions
  ) => void
  onRegenerate: (runId: number, stage: WatchlistBriefingRetryStage) => void
  onTestNow: (jobId?: number) => void
  onViewReports: () => void
  onReviewDeliverySettings: (jobId: number) => void
}

const ACTION_CLASS = "min-h-11 whitespace-normal text-start"
const AUDIO_STAGES: WatchlistBriefingRetryStage[] = [
  "compose_audio_script",
  "persist_audio_script",
  "generate_audio",
  "persist_audio"
]

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value)

const numberValue = (value: unknown): number | undefined => {
  const number = Number(value)
  return Number.isFinite(number) && number >= 0 ? number : undefined
}

const stringValue = (value: unknown): string | undefined =>
  typeof value === "string" && value.trim() ? value.trim() : undefined

const deliveryDestinationOwner = (
  projection: WatchlistBriefingProjection,
  adapter: "email" | "chatbook"
): string => {
  const destination = projection.delivery?.[adapter]
  return `${destination?.recipient_count ?? ""}:${destination?.masked_label ?? ""}`
}

const formatClock = (seconds: number): string => {
  if (!Number.isFinite(seconds) || seconds < 0) return "0:00"
  const whole = Math.floor(seconds)
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`
}

export const formatWatchlistOccurrenceDate = (
  value: string | null | undefined,
  timezone: string,
  locale: string,
  includeYear = false
): string | null => {
  if (!value) return null
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return null
  try {
    return new Intl.DateTimeFormat(locale, {
      timeZone: timezone,
      weekday: "long",
      month: "long",
      day: "numeric",
      ...(includeYear ? { year: "numeric" as const } : {}),
      hour: "numeric",
      minute: "2-digit",
      timeZoneName: "shortOffset"
    }).format(date)
  } catch {
    return new Intl.DateTimeFormat(locale, {
      timeZone: "UTC",
      weekday: "long",
      month: "long",
      day: "numeric",
      ...(includeYear ? { year: "numeric" as const } : {}),
      hour: "numeric",
      minute: "2-digit",
      timeZoneName: "shortOffset"
    }).format(date)
  }
}

const relativeDate = (value: string, locale: string): string | null => {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return null
  const differenceSeconds = (date.getTime() - Date.now()) / 1000
  const absoluteSeconds = Math.abs(differenceSeconds)
  const [amount, unit] = absoluteSeconds >= 86_400
    ? [differenceSeconds / 86_400, "day" as const]
    : absoluteSeconds >= 3_600
      ? [differenceSeconds / 3_600, "hour" as const]
      : [differenceSeconds / 60, "minute" as const]
  return new Intl.RelativeTimeFormat(locale, { numeric: "auto" }).format(
    Math.round(amount),
    unit
  )
}

const stageLabel = (stage: string, t: TFunction): string => {
  const stages: Record<string, [string, string]> = {
    render_text: ["renderText", "Text report"],
    persist_text: ["persistText", "Save report in Reports"],
    compose_audio_script: ["composeScript", "Audio script"],
    persist_audio_script: ["persistScript", "Save audio script"],
    generate_audio: ["generateAudio", "Audio"],
    persist_audio: ["persistAudio", "Save audio"]
  }
  const entry = stages[stage]
  return entry
    ? t(`watchlists:overview.latest.stages.${entry[0]}`, entry[1])
    : sentenceCase(stage.replaceAll("_", " "))
}

const retryStageLabel = (stage: WatchlistBriefingRetryStage, t: TFunction): string => {
  const stages: Partial<Record<WatchlistBriefingRetryStage, [string, string]>> = {
    render_text: ["renderText", "generating report"],
    persist_text: ["persistText", "saving report"],
    compose_audio_script: ["composeScript", "composing audio script"],
    persist_audio_script: ["persistScript", "saving audio script"],
    generate_audio: ["generateAudio", "generating audio"],
    persist_audio: ["persistAudio", "saving audio"]
  }
  const entry = stages[stage]
  return entry
    ? t(`watchlists:overview.latest.recovery.${entry[0]}`, entry[1])
    : stageLabel(stage, t).toLowerCase()
}

const statusLabel = (status: string, t: TFunction): string => {
  const statuses: Record<string, [string, string]> = {
    not_started: ["notStarted", "Not started"],
    queued: ["queued", "Queued"],
    running: ["running", "Running"],
    ready: ["ready", "Ready"],
    completed: ["ready", "Ready"],
    failed: ["failed", "Failed"],
    skipped: ["notSelected", "Not selected"],
    cancelled: ["cancelled", "Cancelled"],
    successful: ["delivered", "Delivered"],
    delivered: ["delivered", "Delivered"],
    partial: ["partial", "Partial"],
    unknown: ["unknown", "Outcome unknown"],
    sending: ["sending", "Sending"]
  }
  const entry = statuses[status]
  return entry ? t(`watchlists:overview.latest.status.${entry[0]}`, entry[1]) : status
}

const statusColor = (status: string): string => {
  if (["ready", "successful", "delivered"].includes(status)) return "green"
  if (["failed", "cancelled"].includes(status)) return "red"
  if (["queued", "running", "sending", "unknown", "partial"].includes(status)) return "gold"
  return "default"
}

const sentenceCase = (value: string): string =>
  value ? `${value[0].toUpperCase()}${value.slice(1)}` : value

export const LatestBriefing: React.FC<LatestBriefingProps> = ({
  projection,
  emptyJobId,
  nextRunAt,
  timezone,
  unreadCount,
  newCount,
  onPlay,
  onOpenReport,
  onInspectRun,
  onRetryStage,
  onRegenerate,
  onTestNow,
  onViewReports,
  onReviewDeliverySettings
}) => {
  const { t, i18n } = useTranslation(["watchlists", "common"])
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const [playing, setPlaying] = useState(false)
  const [hasPlayed, setHasPlayed] = useState(false)
  const [audioLoading, setAudioLoading] = useState(false)
  const [audioError, setAudioError] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [duration, setDuration] = useState(0)
  const [audioObjectUrl, setAudioObjectUrl] = useState<string | null>(null)
  const audioObjectUrlRef = useRef<string | null>(null)
  const [artifactErrorKind, setArtifactErrorKind] = useState<WatchlistArtifactErrorKind | null>(null)
  const [scriptOpen, setScriptOpen] = useState(false)
  const [scriptLoading, setScriptLoading] = useState(false)
  const [scriptText, setScriptText] = useState("")
  const [scriptErrorKind, setScriptErrorKind] = useState<WatchlistArtifactErrorKind | null>(null)
  const scriptRequestControllerRef = useRef<AbortController | null>(null)
  const scriptRequestGenerationRef = useRef(0)
  const [deliveryReview, setDeliveryReview] = useState<DeliveryReview | null>(null)
  const [deliveryReviewed, setDeliveryReviewed] = useState(false)
  const deliveryTriggerRef = useRef<HTMLElement | null>(null)
  const locale = i18n.resolvedLanguage || i18n.language || "en"
  const effectiveNextRun = projection?.next_run_at ?? nextRunAt
  const effectiveTimezone = projection?.timezone || timezone || "UTC"
  const nextRunLabel = formatWatchlistOccurrenceDate(effectiveNextRun, effectiveTimezone, locale)
  const projectedAudio = projection?.audio
  const scriptArtifact = projectedAudio?.script_artifact
  const scriptArtifactPath = stringValue(scriptArtifact?.download_url)
  const scriptIdentity = projection
    ? `${projection.occurrence_id}:${projection.run_id}:${scriptArtifact?.artifact_id || ""}:${scriptArtifactPath || ""}`
    : ""
  const deliveryReviewStage = deliveryReview && projection?.stages[deliveryReview.stageName]
  const deliveryReviewOutcome = deliveryReviewStage?.outcome || deliveryReviewStage?.status
  const deliveryReviewIsCurrent = Boolean(deliveryReview && projection &&
    projection.occurrence_id === deliveryReview.occurrenceId &&
    projection.run_id === deliveryReview.runId &&
    projection.job_id === deliveryReview.jobId &&
    deliveryReviewOutcome === deliveryReview.outcome &&
    deliveryDestinationOwner(projection, deliveryReview.adapter) === deliveryReview.destinationOwner)
  const hasCurrentAudio = Boolean(projection && hasCurrentBriefingAudio(projection))
  const audioArtifactPath = hasCurrentAudio
    ? stringValue(projectedAudio.download_url)
    : undefined
  const audioIdentity = projection && audioArtifactPath
    ? [
        projection.occurrence_id,
        projectedAudio?.task_id,
        projectedAudio?.audio_request_id,
        projectedAudio?.artifact_id,
        audioArtifactPath
      ].join(":")
    : ""

  useEffect(() => {
    const audio = audioRef.current
    audio?.pause()
    if (audio) {
      audio.removeAttribute("src")
      audio.load()
    }
    if (audioObjectUrlRef.current) {
      revokeWatchlistArtifactObjectUrl(audioObjectUrlRef.current)
      audioObjectUrlRef.current = null
    }
    setAudioObjectUrl(null)
    setPlaying(false)
    setHasPlayed(false)
    setElapsed(0)
    setDuration(0)
    setAudioError(false)
    setArtifactErrorKind(null)
    if (!audioArtifactPath) {
      setAudioLoading(false)
      return
    }

    const controller = new AbortController()
    let createdUrl: string | null = null
    setAudioLoading(true)
    void fetchWatchlistArtifactBlob(audioArtifactPath, {
      signal: controller.signal,
      mimeType: projectedAudio?.mime_type || "audio/mpeg"
    }).then((blob) => {
      if (controller.signal.aborted) return
      createdUrl = createWatchlistArtifactObjectUrl(blob)
      audioObjectUrlRef.current = createdUrl
      setAudioObjectUrl(createdUrl)
      setAudioLoading(false)
    }).catch((error: { name?: string; kind?: WatchlistArtifactErrorKind }) => {
      if (controller.signal.aborted || error?.name === "AbortError") return
      setArtifactErrorKind(error?.kind || "network")
      setAudioLoading(false)
    })

    return () => {
      controller.abort()
      audio?.pause()
      if (createdUrl) revokeWatchlistArtifactObjectUrl(createdUrl)
      if (audioObjectUrlRef.current === createdUrl) audioObjectUrlRef.current = null
    }
  }, [audioArtifactPath, audioIdentity, projectedAudio?.mime_type])

  useEffect(() => {
    scriptRequestGenerationRef.current += 1
    scriptRequestControllerRef.current?.abort()
    scriptRequestControllerRef.current = null
    setScriptOpen(false)
    setScriptLoading(false)
    setScriptText("")
    setScriptErrorKind(null)
    return () => {
      scriptRequestGenerationRef.current += 1
      scriptRequestControllerRef.current?.abort()
      scriptRequestControllerRef.current = null
    }
  }, [scriptIdentity])

  useEffect(() => {
    if (!deliveryReview || deliveryReviewIsCurrent) return
    setDeliveryReview(null)
    setDeliveryReviewed(false)
  }, [deliveryReview, deliveryReviewIsCurrent])

  if (!projection) {
    return (
      <section className="@container rounded-xl border border-border bg-surface p-4" aria-labelledby="latest-briefing-heading">
        <div className="max-w-[72ch] space-y-3">
          <h2 id="latest-briefing-heading" className="text-lg font-semibold text-text">
            {t("watchlists:overview.latest.heading.briefing", "Latest briefing")}
          </h2>
          <p className="text-sm text-text-muted">
            {nextRunLabel
              ? t("watchlists:overview.latest.empty.scheduled", "Your first briefing is scheduled for {{date}}.", { date: nextRunLabel })
              : t("watchlists:overview.latest.empty.manual", "No briefing has been generated yet. Run a safe test when you are ready.")}
          </p>
          <div className="flex flex-col gap-2 @lg:flex-row">
            <Button type="primary" className={ACTION_CLASS} onClick={() => onTestNow(emptyJobId)}>
              {t("watchlists:overview.latest.actions.testNow", "Test now")}
            </Button>
          </div>
        </div>
      </section>
    )
  }

  const output = isRecord(projection.output) ? projection.output : {}
  const metadata = isRecord(output.metadata) ? output.metadata : {}
  const provenance = isRecord(metadata.provenance) ? metadata.provenance : {}
  const provenanceItems = Array.isArray(metadata.provenance) ? metadata.provenance : []
  const showName = stringValue(projection.editorial.show_name)
  const outputTitle = stringValue(output.title)
  const objectName = showName || outputTitle || t(
    "watchlists:overview.latest.untitled",
    "Run {{runId}}",
    { runId: projection.run_id }
  )
  const outcomeNounKey = projection.editorial.outcome_noun === "episode" ? "episode" : "briefing"
  const outcomeNoun = t(
    `watchlists:overview.latest.nouns.${outcomeNounKey}`,
    outcomeNounKey
  )
  const reportNoun = projection.editorial.show_notes
    ? t("watchlists:overview.latest.nouns.showNotes", "show notes")
    : t("watchlists:overview.latest.nouns.report", "report")
  const audioReady = Boolean(audioArtifactPath && audioObjectUrl)
  const audioUnavailable = Boolean(artifactErrorKind)
  const noncurrentAudioStatus = projectedAudio?.status === "completed" && !hasCurrentAudio
    ? projectedAudio.stale
      ? ["stale", "Stale"]
      : projectedAudio.superseded_by
        ? ["superseded", "Superseded"]
        : ["unavailable", "Unavailable"]
    : null
  const textReady = Boolean(projection.output) || projection.stages.persist_text?.status === "ready"
  const overallArtifactStatus = audioUnavailable && textReady
    ? "partial"
    : projection.artifact_status
  const audioFailedStage = AUDIO_STAGES.find((stage) => projection.stages[stage]?.status === "failed")
  const composeScriptStage = projection.stages.compose_audio_script
  const persistScriptStage = projection.stages.persist_audio_script
  const scriptStage = composeScriptStage && !["ready", "skipped"].includes(composeScriptStage.status)
    ? composeScriptStage
    : persistScriptStage || composeScriptStage
  const programFormatKey = stringValue(projection.editorial.program_format)
  const programFormat = programFormatKey
    ? t(
        `watchlists:overview.pipelineSetup.receipt.formats.${programFormatKey}`,
        programFormatKey.replaceAll("_", " ")
      )
    : undefined
  const cast = projection.editorial.cast
  const speakers = Array.isArray(cast?.speakers)
    ? cast.speakers
      .map((speaker) => isRecord(speaker) ? stringValue(speaker.label) : undefined)
      .filter((speaker): speaker is string => Boolean(speaker))
    : []
  const speakerCount = numberValue(cast?.speaker_count) || speakers.length || 1
  const targetMinutes = numberValue(projection.editorial.target_minutes)
  const speakerStyleKey = String(Math.min(4, Math.max(1, speakerCount)))
  const speakerStyle = t(
    `watchlists:overview.pipelineSetup.receipt.speakerStyle.${speakerStyleKey}`,
    speakerCount === 1
      ? "solo"
      : `${speakerCount === 2 ? "two" : speakerCount === 3 ? "three" : "four"}-host`
  )
  const generatedAt = formatWatchlistOccurrenceDate(
    stringValue(output.created_at),
    effectiveTimezone,
    locale,
    true
  )
  const includedCount = numberValue(projection.selection.included_count) || 0
  const provenanceSourceCount = provenanceItems.length
    ? new Set(
        provenanceItems
          .map((item) => (isRecord(item) ? item.source_id : undefined))
          .filter((sourceId) => sourceId !== undefined && sourceId !== null)
          .map(String)
      ).size
    : undefined
  const sourceCount =
    numberValue(provenance.source_count) ?? provenanceSourceCount ?? numberValue(metadata.source_count)
  const noMaterialUpdates = metadata.no_material_updates === true

  const deliveryRows = (["email", "chatbook"] as const).flatMap((adapter) => {
    const stageName = `deliver:${adapter}`
    const stage = projection.stages[stageName]
    return stage ? [{ adapter, stageName: stageName as WatchlistBriefingRetryStage, stage }] : []
  })
  const retryableArtifactStages = Object.entries(projection.stages)
    .filter(([stageName, stage]) =>
      !stageName.startsWith("deliver:") &&
      stage.retryable === true &&
      ["failed", "cancelled"].includes(stage.status)
    )
    .map(([stageName]) => stageName as WatchlistBriefingRetryStage)

  const handlePlayback = async () => {
    const audio = audioRef.current
    if (!audio || !projection.audio) return
    if (playing) {
      audio.pause()
      return
    }
    setAudioError(false)
    setAudioLoading(true)
    onPlay(projection.audio, projection)
    try {
      await audio.play()
    } catch {
      setAudioLoading(false)
      setAudioError(true)
    }
  }

  const retryDelivery = (
    stageName: WatchlistBriefingRetryStage,
    adapter: "email" | "chatbook",
    outcome: string,
    trigger: HTMLElement
  ) => {
    if (outcome !== "unknown") {
      onRetryStage(projection.run_id, stageName)
      return
    }
    const destinationSummary = projection.delivery?.[adapter]
    const destinationLabel = adapter === "email"
      ? t("watchlists:overview.latest.delivery.recipientCount", "{{count}} recipients", {
          count: destinationSummary?.recipient_count || 0
        })
      : t("watchlists:overview.latest.delivery.chatbookDestination", "Chatbook")
    deliveryTriggerRef.current = trigger
    setDeliveryReviewed(false)
    setDeliveryReview({
      adapter,
      stageName,
      occurrenceId: projection.occurrence_id,
      runId: projection.run_id,
      jobId: projection.job_id,
      outcome,
      destinationOwner: deliveryDestinationOwner(projection, adapter),
      destinationLabel
    })
  }

  const handleDownloadAudio = async () => {
    if (!audioArtifactPath) return
    try {
      const blob = await fetchWatchlistArtifactBlob(audioArtifactPath, {
        mimeType: projection.audio?.mime_type || "audio/mpeg"
      })
      const url = createWatchlistArtifactObjectUrl(blob)
      const anchor = document.createElement("a")
      anchor.href = url
      anchor.download = `${objectName.replace(/[^a-z0-9]+/gi, "-").replace(/^-|-$/g, "") || `run-${projection.run_id}`}.mp3`
      document.body.appendChild(anchor)
      anchor.click()
      anchor.remove()
      revokeWatchlistArtifactObjectUrl(url)
    } catch (error) {
      setArtifactErrorKind((error as { kind?: WatchlistArtifactErrorKind } | null)?.kind || "network")
    }
  }

  const handleReviewScript = async () => {
    if (!scriptArtifactPath) return
    scriptRequestControllerRef.current?.abort()
    const controller = new AbortController()
    const generation = ++scriptRequestGenerationRef.current
    scriptRequestControllerRef.current = controller
    setScriptOpen(true)
    setScriptLoading(true)
    setScriptText("")
    setScriptErrorKind(null)
    try {
      const text = await fetchWatchlistArtifactText(scriptArtifactPath, {
        signal: controller.signal
      })
      if (controller.signal.aborted || generation !== scriptRequestGenerationRef.current) return
      setScriptText(text)
    } catch (error) {
      if (
        controller.signal.aborted ||
        generation !== scriptRequestGenerationRef.current ||
        (error as { name?: string } | null)?.name === "AbortError"
      ) return
      setScriptErrorKind((error as { kind?: WatchlistArtifactErrorKind } | null)?.kind || "network")
    } finally {
      if (generation === scriptRequestGenerationRef.current) {
        setScriptLoading(false)
        scriptRequestControllerRef.current = null
      }
    }
  }

  const playbackKey = playing ? "pause" : hasPlayed ? "resume" : "play"
  const playbackLabel = t(
    `watchlists:overview.latest.playback.${playbackKey}`,
    sentenceCase(playbackKey)
  )
  const playbackAria = t(
    `watchlists:overview.latest.playback.${playbackKey}Aria`,
    `${playbackLabel} {{name}}`,
    { name: objectName }
  )

  const reviewedDeliveryLabel = deliveryReview?.destinationLabel || ""
  const closeDeliveryReview = () => {
    setDeliveryReview(null)
    setDeliveryReviewed(false)
  }

  return (
    <>
    <section className="@container rounded-xl border border-border bg-surface p-4" aria-labelledby={`latest-briefing-heading-${projection.occurrence_id}`}>
      <div
        className="grid min-w-0 grid-cols-1 gap-5 @3xl:grid-cols-[minmax(0,1fr)_minmax(13rem,0.35fr)]"
        data-testid="latest-briefing-layout"
      >
        <div className="min-w-0 space-y-5">
          <header className="space-y-1">
            <h2 id={`latest-briefing-heading-${projection.occurrence_id}`} className="text-lg font-semibold text-text">
              {t("watchlists:overview.latest.heading.briefing", "Latest briefing")}
            </h2>
            {showName && <p className="break-words text-base font-semibold text-text">{showName}</p>}
            {outputTitle && outputTitle !== showName && (
              <p className="break-words text-sm text-text">{outputTitle}</p>
            )}
            {generatedAt && (
              <p className="text-xs text-text-muted">
                {t("watchlists:overview.latest.generatedAt", "Generated {{date}}", { date: generatedAt })}
              </p>
            )}
            <div className="flex flex-wrap gap-2 pt-1">
              {programFormat && <Tag>{sentenceCase(programFormat)}</Tag>}
              {speakers.length > 0 && (
                <Tag className="max-w-full whitespace-normal break-words">{speakers.join(", ")}</Tag>
              )}
              {targetMinutes !== undefined && (
                <Tag>
                  {t("watchlists:overview.pipelineSetup.receipt.duration", "targeting {{count}} minutes", {
                    count: targetMinutes
                  })}
                </Tag>
              )}
              <Tag color={statusColor(audioUnavailable && textReady ? "partial" : overallArtifactStatus)}>
                {statusLabel(audioUnavailable && textReady ? "partial" : overallArtifactStatus, t)}
              </Tag>
            </div>
          </header>

          {noMaterialUpdates && (
            <p className="rounded-lg bg-bg p-3 text-sm text-text">
              {t(
                "watchlists:overview.latest.noUpdates",
                "No qualifying updates were found. A status {{noun}} was saved.",
                { noun: outcomeNoun }
              )}
            </p>
          )}

          {audioLoading && audioArtifactPath && !audioObjectUrl && (
            <p className="text-xs text-text-muted">
              {t("watchlists:overview.latest.playback.loading", "Loading audio")}
            </p>
          )}
          {artifactErrorKind && (
            <p className="text-sm text-danger" role="alert">
              {artifactErrorKind === "missing"
                ? t("watchlists:overview.latest.playback.missing", "This audio artifact is no longer available.")
                : t("watchlists:overview.latest.playback.accessError", "Audio access failed. Check your sign-in and server connection.")}
            </p>
          )}
          {audioReady && audioObjectUrl && (
            <div className="space-y-3" aria-label={t("watchlists:overview.latest.playback.region", "Audio playback for {{name}}", { name: objectName })}>
              <audio
                ref={audioRef}
                preload="metadata"
                src={audioObjectUrl}
                onLoadedMetadata={(event) => {
                  setDuration(Number.isFinite(event.currentTarget.duration) ? event.currentTarget.duration : 0)
                  setElapsed(event.currentTarget.currentTime || 0)
                }}
                onTimeUpdate={(event) => setElapsed(event.currentTarget.currentTime || 0)}
                onPlay={() => {
                  setPlaying(true)
                  setHasPlayed(true)
                  setAudioLoading(false)
                }}
                onPause={() => setPlaying(false)}
                onWaiting={() => setAudioLoading(true)}
                onCanPlay={() => setAudioLoading(false)}
                onEnded={() => setPlaying(false)}
                onError={() => {
                  setPlaying(false)
                  setAudioLoading(false)
                  setAudioError(true)
                }}
              />
              <div className="flex flex-col gap-3 @lg:flex-row @lg:items-center">
                <Button
                  type="primary"
                  className={ACTION_CLASS}
                  icon={playing ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                  aria-label={playbackAria}
                  onClick={() => void handlePlayback()}
                >
                  {playbackLabel}
                </Button>
                <label className="min-w-0 flex-1 text-xs text-text-muted">
                  <span className="sr-only">
                    {t("watchlists:overview.latest.playback.seekAria", "Seek {{name}}", { name: objectName })}
                  </span>
                  <input
                    type="range"
                    min={0}
                    max={duration || 0}
                    step={1}
                    value={Math.min(elapsed, duration || 0)}
                    aria-label={t("watchlists:overview.latest.playback.seekAria", "Seek {{name}}", { name: objectName })}
                    onChange={(event) => {
                      const next = Number(event.target.value)
                      if (audioRef.current) audioRef.current.currentTime = next
                      setElapsed(next)
                    }}
                    className="min-h-11 w-full accent-primary"
                  />
                </label>
                <span className="shrink-0 text-xs tabular-nums text-text-muted">
                  {formatClock(elapsed)} / {formatClock(duration)}
                </span>
              </div>
              {audioLoading && <p className="text-xs text-text-muted">{t("watchlists:overview.latest.playback.loading", "Loading audio")}</p>}
              {audioError && (
                <p className="text-sm text-danger" role="alert">
                  {t("watchlists:overview.latest.playback.error", "Audio could not be played. Open the report or regenerate audio.")}
                </p>
              )}
            </div>
          )}

          <div className="grid gap-x-6 gap-y-2 text-sm @lg:grid-cols-2" aria-label={t("watchlists:overview.latest.artifacts", "Artifact status")}>
            <div className="flex items-center justify-between gap-3 border-b border-border py-2">
              <span className="inline-flex min-w-0 items-center gap-2"><FileText className="h-4 w-4 shrink-0" />{sentenceCase(reportNoun)}</span>
              <span className={textReady ? "text-success" : "text-text-muted"}>{textReady ? statusLabel("ready", t) : statusLabel(projection.stages.persist_text?.status || "not_started", t)}</span>
            </div>
            {scriptStage && (
              <div className="flex items-center justify-between gap-3 border-b border-border py-2">
                <span>{t("watchlists:overview.latest.script", "Audio script")}</span>
                <span className={scriptStage.status === "failed" ? "text-danger" : "text-text-muted"}>
                  {t("watchlists:overview.latest.stageWithStatus", "{{stage}} {{status}}", {
                    stage: t("watchlists:overview.latest.script", "Audio script"),
                    status: statusLabel(scriptStage.status, t).toLowerCase()
                  })}
                </span>
              </div>
            )}
            <div className="flex items-center justify-between gap-3 border-b border-border py-2">
              <span className="inline-flex min-w-0 items-center gap-2"><Headphones className="h-4 w-4 shrink-0" />{t("watchlists:overview.latest.audio", "Audio")}</span>
              <span className={audioFailedStage || audioUnavailable ? "text-danger" : audioReady ? "text-success" : "text-text-muted"}>
                {audioFailedStage
                  ? t("watchlists:overview.latest.audioFailed", "Audio failed")
                  : audioUnavailable
                    ? t("watchlists:overview.latest.status.unavailable", "Unavailable")
                  : noncurrentAudioStatus
                    ? t(`watchlists:overview.latest.status.${noncurrentAudioStatus[0]}`, noncurrentAudioStatus[1])
                    : audioReady
                    ? t("watchlists:overview.latest.ready", "Ready")
                    : statusLabel(projection.audio?.status || projection.stages.generate_audio?.status || "not_started", t)}
              </span>
            </div>
            <div className="flex items-center justify-between gap-3 border-b border-border py-2">
              <span>{t("watchlists:overview.latest.delivery.reports", "Reports")}</span>
              <span className={textReady ? "text-success" : "text-text-muted"}>{textReady ? t("watchlists:overview.latest.saved", "Saved") : t("watchlists:overview.latest.waiting", "Waiting")}</span>
            </div>
            {deliveryRows.map(({ adapter, stageName, stage }) => {
              const outcome = stage.outcome || stage.status
              const unknown = outcome === "unknown"
              const canRetry = unknown || (
                stage.retryable === true && ["failed", "cancelled"].includes(stage.status)
              )
              return (
                <div key={stageName} className="flex flex-wrap items-center justify-between gap-2 border-b border-border py-2">
                  <span>{sentenceCase(adapter)}</span>
                  <span className={stage.status === "failed" ? "text-danger" : "text-text-muted"}>
                    {sentenceCase(adapter)} {statusLabel(outcome, t).toLowerCase()}
                  </span>
                  {canRetry && (
                    <Button
                      size="small"
                      className={ACTION_CLASS}
                      aria-label={unknown
                        ? t("watchlists:overview.latest.delivery.retryUnknownAria", "Review and retry {{adapter}} delivery for {{name}}", { adapter, name: objectName })
                        : t("watchlists:overview.latest.delivery.retryAria", "Retry {{adapter}} delivery for {{name}}", { adapter, name: objectName })}
                      onClick={(event) => retryDelivery(stageName, adapter, outcome, event.currentTarget)}
                    >
                      {unknown ? t("watchlists:overview.latest.delivery.reviewRetry", "Review and retry") : t("watchlists:overview.latest.actions.retry", "Retry")}
                    </Button>
                  )}
                </div>
              )
            })}
          </div>

          <div className="space-y-2 text-sm text-text-muted">
            <div className="flex flex-wrap gap-x-4 gap-y-1" aria-label={t("watchlists:overview.latest.counts", "Briefing counts")}>
              <span>{t("watchlists:overview.latest.count.included", "Included {{count}}", { count: includedCount })}</span>
              {unreadCount !== undefined && <span>{t("watchlists:overview.latest.count.unread", "Unread {{count}}", { count: unreadCount })}</span>}
              {newCount !== undefined && <span>{t("watchlists:overview.latest.count.new", "New {{count}}", { count: newCount })}</span>}
            </div>
            {sourceCount !== undefined && (
              <p>{t("watchlists:overview.latest.provenance.sources", "{{count}} tracked sources", { count: sourceCount })}</p>
            )}
            {nextRunLabel && (
              <div>
                <p className="font-medium text-text">
                  {t("watchlists:overview.latest.nextRun", "Next run: {{date}}", { date: nextRunLabel })}
                </p>
                {effectiveNextRun && relativeDate(effectiveNextRun, locale) && (
                  <p className="text-xs">
                    {t("watchlists:overview.latest.nextRunRelative", "Schedule timing: {{relative}}", {
                      relative: relativeDate(effectiveNextRun, locale)
                    })}
                  </p>
                )}
              </div>
            )}
          </div>
        </div>

        <aside
          className="min-w-0 space-y-2 border-t border-border pt-4 @3xl:border-s @3xl:border-t-0 @3xl:ps-4 @3xl:pt-0"
          aria-label={t("watchlists:overview.latest.actions.label", "Latest outcome actions")}
          data-testid="watchlists-repeat-actions"
        >
          {textReady && numberValue(output.id) !== undefined && (
            <Button
              type={audioReady ? "default" : "primary"}
              block
              className={ACTION_CLASS}
              aria-label={t("watchlists:overview.latest.actions.openReportAria", "Open {{report}} for {{name}}", { report: reportNoun, name: objectName })}
              onClick={() => onOpenReport(Number(output.id), projection)}
            >
              {t("watchlists:overview.latest.actions.openReport", "Open {{report}}", { report: reportNoun })}
            </Button>
          )}
          {retryableArtifactStages.map((stageName) => (
            <Button
              key={stageName}
              danger
              block
              className={ACTION_CLASS}
              icon={AUDIO_STAGES.includes(stageName) ? <RotateCcw className="h-4 w-4" /> : undefined}
              aria-label={t(
                "watchlists:overview.latest.actions.retryExactAria",
                "Retry {{stage}} for {{name}}",
                { stage: retryStageLabel(stageName, t), name: objectName }
              )}
              onClick={() => onRetryStage(projection.run_id, stageName)}
            >
              {t("watchlists:overview.latest.actions.retryStage", "Retry {{stage}}", {
                stage: retryStageLabel(stageName, t)
              })}
            </Button>
          ))}
          {retryableArtifactStages.length === 0 && projection.recovery.can_regenerate_audio && (
            <Button
              block
              className={ACTION_CLASS}
              icon={<RotateCcw className="h-4 w-4" />}
              aria-label={t("watchlists:overview.latest.actions.regenerateAudioAria", "Regenerate {{style}} {{noun}} audio for {{name}}", { style: speakerStyle, noun: outcomeNoun, name: objectName })}
              onClick={() => onRegenerate(projection.run_id, "generate_audio")}
            >
              {t("watchlists:overview.latest.actions.regenerateAudio", "Regenerate audio")}
            </Button>
          )}
          {audioArtifactPath && !audioUnavailable && (
            <Button
              block
              className={ACTION_CLASS}
              icon={<Download className="h-4 w-4" />}
              aria-label={t("watchlists:overview.latest.actions.downloadAudioAria", "Download audio for {{name}}", { name: objectName })}
              onClick={() => void handleDownloadAudio()}
            >
              {t("watchlists:overview.latest.actions.downloadAudio", "Download audio")}
            </Button>
          )}
          {scriptArtifactPath && (
            <Button
              block
              className={ACTION_CLASS}
              icon={<ScrollText className="h-4 w-4" />}
              aria-label={t("watchlists:overview.latest.actions.reviewScriptAria", "Review script for {{name}}", { name: objectName })}
              onClick={() => void handleReviewScript()}
            >
              {t("watchlists:overview.latest.actions.reviewScript", "Review script")}
            </Button>
          )}
          <Button
            block
            className={ACTION_CLASS}
            data-testid="watchlists-repeat-open-runs"
            aria-label={t("watchlists:overview.latest.actions.inspectAria", "Inspect run {{runId}} for {{name}}", { runId: projection.run_id, name: objectName })}
            icon={<AlertCircle className="h-4 w-4" />}
            onClick={() => onInspectRun(projection.run_id)}
          >
            {t("watchlists:overview.latest.actions.inspect", "Inspect run")}
          </Button>
          <Button
            block
            className={ACTION_CLASS}
            aria-label={`${t("watchlists:overview.latest.actions.testNow", "Test now")}: ${objectName}`}
            onClick={() => onTestNow(projection.job_id)}
          >
            {t("watchlists:overview.latest.actions.testNow", "Test now")}
          </Button>
          <Button block type="link" className={ACTION_CLASS} onClick={onViewReports}>
            {t("watchlists:overview.latest.actions.viewReports", "View all reports")}
          </Button>
        </aside>
      </div>
    </section>
    <Modal
      open={Boolean(deliveryReview)}
      title={deliveryReview
        ? t(
            "watchlists:overview.latest.delivery.reviewTitle",
            "Review {{adapter}} delivery retry",
            { adapter: deliveryReview.adapter }
          )
        : ""}
      onCancel={closeDeliveryReview}
      afterClose={() => deliveryTriggerRef.current?.focus()}
      footer={deliveryReview ? [
        <Button
          key="settings"
          onClick={() => {
            closeDeliveryReview()
            onReviewDeliverySettings(deliveryReview.jobId)
          }}
        >
          {t("watchlists:overview.latest.delivery.reviewSettings", "Review delivery settings")}
        </Button>,
        <Button key="cancel" onClick={closeDeliveryReview}>
          {t("common:cancel", "Cancel")}
        </Button>,
        <Button
          key="retry"
          type="primary"
          danger
          disabled={!deliveryReviewed}
          aria-label={t(
            "watchlists:overview.latest.delivery.confirmRetryAria",
            "Retry {{adapter}} delivery",
            { adapter: deliveryReview.adapter }
          )}
          onClick={() => {
            if (!deliveryReviewIsCurrent) {
              closeDeliveryReview()
              return
            }
            onRetryStage(deliveryReview.runId, deliveryReview.stageName, {
              confirm_unknown_delivery_retry: true
            })
            closeDeliveryReview()
          }}
        >
          {t("watchlists:overview.latest.delivery.confirmRetry", "Retry delivery")}
        </Button>
      ] : null}
    >
      {deliveryReview && (
        <div className="space-y-4">
          <p>
            {t(
              "watchlists:overview.latest.delivery.unknownConfirmation",
              "The provider did not confirm {{adapter}} delivery to {{destination}}. Retrying may send a duplicate.",
              {
                adapter: deliveryReview.adapter,
                destination: reviewedDeliveryLabel
              }
            )}
          </p>
          <p className="font-medium text-text">{reviewedDeliveryLabel}</p>
          <Checkbox
            checked={deliveryReviewed}
            onChange={(event) => setDeliveryReviewed(event.target.checked)}
          >
            {t(
              "watchlists:overview.latest.delivery.reviewAcknowledgement",
              "I reviewed the destination and accept the duplicate-delivery risk."
            )}
          </Checkbox>
        </div>
      )}
    </Modal>
    <Modal
      open={scriptOpen}
      title={scriptArtifact?.title || t("watchlists:overview.latest.script", "Audio script")}
      onCancel={() => setScriptOpen(false)}
      footer={null}
    >
      {scriptLoading ? (
        <p>{t("watchlists:overview.latest.scriptLoading", "Loading script")}</p>
      ) : scriptErrorKind ? (
        <p className="text-danger" role="alert">
          {scriptErrorKind === "missing"
            ? t("watchlists:overview.latest.scriptMissing", "This script artifact is no longer available.")
            : t("watchlists:overview.latest.scriptAccessError", "Script access failed. Check your sign-in and server connection.")}
        </p>
      ) : (
        <pre className="max-h-[60vh] overflow-auto whitespace-pre-wrap break-words text-sm">{scriptText}</pre>
      )}
    </Modal>
    </>
  )
}
