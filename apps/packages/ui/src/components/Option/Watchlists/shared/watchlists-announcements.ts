import type { TFunction } from "i18next"
import type { WatchlistBriefingProjection } from "@/types/watchlists"

const record = (value: unknown): Record<string, unknown> =>
  value && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {}

const nameOf = (projection: WatchlistBriefingProjection): string => {
  const output = record(projection.output)
  return String(
    projection.editorial.show_name || output.title || `Run ${projection.run_id}`
  )
}

const textReady = (projection: WatchlistBriefingProjection): boolean =>
  Boolean(projection.output) || projection.stages.persist_text?.status === "ready"

const audioReady = (projection: WatchlistBriefingProjection): boolean =>
  projection.audio?.status === "completed" || projection.stages.persist_audio?.status === "ready"

const audioFailed = (projection: WatchlistBriefingProjection): boolean =>
  projection.audio?.status === "failed" || [
    "compose_audio_script",
    "persist_audio_script",
    "generate_audio",
    "persist_audio"
  ].some((stage) => projection.stages[stage]?.status === "failed")

const semanticState = (projection: WatchlistBriefingProjection): string => JSON.stringify({
  artifact: projection.artifact_status,
  delivery: projection.delivery_status,
  textReady: textReady(projection),
  audioReady: audioReady(projection),
  audioFailed: audioFailed(projection),
  nextRun: projection.next_run_at,
  stages: Object.entries(projection.stages)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([name, stage]) => [name, stage.status, stage.outcome])
})

const isBlocking = (projection: WatchlistBriefingProjection): boolean =>
  ["failed", "cancelled"].includes(projection.artifact_status) &&
  !textReady(projection) &&
  !audioReady(projection)

export const blockingFailureAnnouncement = (
  previous: WatchlistBriefingProjection | null | undefined,
  next: WatchlistBriefingProjection,
  t: TFunction
): string | null => {
  if (!previous || !isBlocking(next) || isBlocking(previous)) return null
  return t(
    "watchlists:overview.latest.announcements.blockingFailure",
    "{{name}} failed before an artifact was ready. Inspect run {{runId}} to recover it.",
    { name: nameOf(next), runId: next.run_id }
  )
}

export const transitionAnnouncement = (
  previous: WatchlistBriefingProjection | null | undefined,
  next: WatchlistBriefingProjection,
  t: TFunction
): string | null => {
  if (!previous || semanticState(previous) === semanticState(next) || isBlocking(next)) {
    return null
  }

  const name = nameOf(next)
  if (next.artifact_status === "ready" && previous.artifact_status !== "ready") {
    if (audioReady(next)) {
      return t(
        "watchlists:overview.latest.announcements.readyWithAudio",
        "{{name}} is ready. Audio and show notes are available.",
        { name }
      )
    }
    return t(
      "watchlists:overview.latest.announcements.readyText",
      "{{name}} is ready. The report is available.",
      { name }
    )
  }

  if (textReady(next) && audioFailed(next) && !audioFailed(previous)) {
    return t(
      "watchlists:overview.latest.announcements.partialAudioFailure",
      "{{name}} show notes are ready, but audio failed.",
      { name }
    )
  }

  if (next.delivery_status === "delivered" && previous.delivery_status !== "delivered") {
    return t(
      "watchlists:overview.latest.announcements.delivered",
      "{{name}} delivery completed.",
      { name }
    )
  }

  if (next.next_run_at !== previous.next_run_at) {
    return t(
      "watchlists:overview.latest.announcements.nextRunUpdated",
      "{{name}} next run was updated.",
      { name }
    )
  }

  const activeStage = Object.entries(next.stages).find(([stageName, stage]) => {
    const previousStage = previous.stages[stageName]
    return ["queued", "running"].includes(stage.status) && previousStage?.status !== stage.status
  })
  if (activeStage) {
    return t(
      "watchlists:overview.latest.announcements.stageProgress",
      "{{name}}: {{stage}} is {{status}}.",
      { name, stage: activeStage[0].replaceAll("_", " "), status: activeStage[1].status }
    )
  }

  return null
}
