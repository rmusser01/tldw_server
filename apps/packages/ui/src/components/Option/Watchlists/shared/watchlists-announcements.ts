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

export const hasCurrentBriefingAudio = (projection: WatchlistBriefingProjection): boolean => {
  const audio = projection.audio
  return audio?.status === "completed" &&
    audio.stale !== true &&
    !audio.superseded_by &&
    typeof audio.download_url === "string" &&
    audio.download_url.trim().length > 0
}

const audioReady = hasCurrentBriefingAudio

const audioFailed = (projection: WatchlistBriefingProjection): boolean =>
  projection.audio?.status === "failed" || [
    "compose_audio_script",
    "persist_audio_script",
    "generate_audio",
    "persist_audio"
  ].some((stage) => projection.stages[stage]?.status === "failed")

const sameOccurrence = (
  left: WatchlistBriefingProjection,
  right: WatchlistBriefingProjection
): boolean =>
  left.occurrence_id === right.occurrence_id &&
  left.run_id === right.run_id &&
  left.job_id === right.job_id

const localizedStage = (stage: string, t: TFunction): string => {
  const stages: Record<string, [string, string]> = {
    collect: ["collect", "Collecting updates"],
    select: ["select", "Selecting updates"],
    render_text: ["renderText", "Text report"],
    persist_text: ["persistText", "Saving report"],
    compose_audio_script: ["composeScript", "Audio script"],
    persist_audio_script: ["persistScript", "Saving audio script"],
    generate_audio: ["generateAudio", "Audio"],
    persist_audio: ["persistAudio", "Saving audio"]
  }
  const entry = stages[stage]
  return entry
    ? t(`watchlists:overview.latest.stages.${entry[0]}`, entry[1])
    : stage.replaceAll("_", " ")
}

const localizedStatus = (status: string, t: TFunction): string => {
  const statuses: Record<string, [string, string]> = {
    queued: ["queued", "queued"],
    running: ["running", "running"]
  }
  const entry = statuses[status]
  return entry
    ? t(`watchlists:overview.latest.status.${entry[0]}`, entry[1])
    : status
}

const semanticState = (projection: WatchlistBriefingProjection): string => JSON.stringify({
  occurrence: projection.occurrence_id,
  run: projection.run_id,
  job: projection.job_id,
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
  if (
    !previous ||
    !isBlocking(next) ||
    (sameOccurrence(previous, next) && isBlocking(previous))
  ) return null
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
  const sameIdentity = sameOccurrence(previous, next)
  if (next.artifact_status === "ready" && (!sameIdentity || previous.artifact_status !== "ready")) {
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

  if (textReady(next) && audioFailed(next) && (!sameIdentity || !audioFailed(previous))) {
    return t(
      "watchlists:overview.latest.announcements.partialAudioFailure",
      "{{name}} show notes are ready, but audio failed.",
      { name }
    )
  }

  if (next.delivery_status === "delivered" && (!sameIdentity || previous.delivery_status !== "delivered")) {
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
    const previousStage = sameIdentity ? previous.stages[stageName] : undefined
    return ["queued", "running"].includes(stage.status) && previousStage?.status !== stage.status
  })
  if (activeStage) {
    return t(
      "watchlists:overview.latest.announcements.stageProgress",
      "{{name}}: {{stage}} is {{status}}.",
      {
        name,
        stage: localizedStage(activeStage[0], t),
        status: localizedStatus(activeStage[1].status, t)
      }
    )
  }

  return null
}
