import type {
  JobOutputPrefs,
  SourceType,
  WatchlistAudioCast,
  WatchlistAudioCastSpeaker,
  WatchlistBriefingProjection,
  WatchlistProgramFormat,
  WatchlistSourceCreate
} from "@/types/watchlists"
import {
  buildCronFromPreset,
  formatScheduleTime,
  INTERVAL_HOURS_MAX,
  INTERVAL_HOURS_MIN,
  INTERVAL_MINUTES_MAX,
  INTERVAL_MINUTES_MIN,
  validateCronSchedule,
  type CronScheduleValidationResult,
  type ScheduleIntervalUnit,
  type WeekdayToken
} from "../JobsTab/schedule-utils"
import { MIN_SCHEDULE_INTERVAL_MINUTES } from "../JobsTab/schedule-frequency"
import { getLocalTimezone } from "./quick-setup"
import type { BriefingPipelineDraft } from "./pipeline-contract"

export type PipelineWizardSourceMode = "existing" | "new"
export type PipelineWizardScheduleMode =
  | "manual"
  | "interval"
  | "daily"
  | "weekdays"
  | "weekly"
  | "advanced"

export interface PipelineWizardAudioSpeakerDraft {
  id: string
  label: string
  role?: string
  voice: string
  persona?: string
}

export interface PipelineWizardDraft {
  sourceMode: PipelineWizardSourceMode
  sourceIds: number[]
  sourceName: string
  sourceUrl: string
  sourceType: SourceType
  monitorName: string
  scheduleMode: PipelineWizardScheduleMode
  scheduleIntervalValue: number
  scheduleIntervalUnit: ScheduleIntervalUnit
  scheduleHour: number
  scheduleMinute: number
  scheduleWeekday: WeekdayToken
  scheduleAdvancedCron: string
  timezone: string
  nextRunAt?: string
  followingRunAt?: string
  createScheduledOutput: boolean
  programFormat: WatchlistProgramFormat
  outcomeNoun: "briefing" | "episode"
  showName: string
  premise: string
  audience: string
  tone: string
  episodeTitlePattern: string
  customInstructions: string
  templateName: string
  templateFormat?: "md" | "html"
  showNotes: boolean
  emailDeliveryEnabled: boolean
  emailRecipients: string[]
  chatbookDeliveryEnabled: boolean
  chatbookTitle: string
  audioEnabled: boolean
  audioSpeakers: PipelineWizardAudioSpeakerDraft[]
  targetAudioMinutes: number
  audioProvider: string
  audioModel: string
  runNow: boolean
  preservedOutputPrefs?: JobOutputPrefs | null
}

export interface PipelineWizardValidationResult {
  valid: boolean
  errors: string[]
}

export interface PipelineWizardSourceBinding {
  mode: PipelineWizardSourceMode
  signature: string
  ids: number[]
  createdByWizard: boolean
  sessionKey: string | number
}

export interface PipelineWizardBriefingPollOptions {
  intervalMs?: number
  maxAttempts?: number
  waitForDelivery?: boolean
  signal?: AbortSignal
}

export type PipelineWizardBriefingStatus = "ready" | "running" | "failed" | "cancelled"

export interface PipelineWizardBriefingOutcome {
  status: PipelineWizardBriefingStatus
  message?: string
}

export type PipelineWizardCronValidationError =
  | "required"
  | Exclude<CronScheduleValidationResult, null>
  | null

export interface PipelineWizardReviewSummary {
  sources: string
  cadence: string
  filters: string
  output: string
  delivery: string
  audio: string
}

interface PipelineWizardCadenceCopy {
  manual?: string
  interval?: (value: number, unit: ScheduleIntervalUnit) => string
  daily?: (time: string) => string
  weekly?: (weekday: string, time: string) => string
  weekdays?: (time: string) => string
  advanced?: (cron: string) => string
  weekdayLabels?: Partial<Record<WeekdayToken, string>>
}

export interface PipelineWizardReviewSummaryCopy {
  newFeed?: string
  noFeedsSelected?: string
  feedLabel?: (id: number) => string
  filters?: string
  noTemplate?: string
  outputDigest?: (templateName: string) => string
  email?: string
  chatbook?: string
  inAppReports?: string
  audioBriefing?: (speakerCount: number) => string
  audioDisabled?: string
  cadence?: PipelineWizardCadenceCopy
}

interface SourceLabel {
  id: number
  name?: string | null
}

const EMAIL_PATTERN = /^[^\s@]+@[^\s@]+\.[^\s@]+$/

export const createDefaultPipelineWizardDraft = (): PipelineWizardDraft => ({
  sourceMode: "existing",
  sourceIds: [],
  sourceName: "",
  sourceUrl: "",
  sourceType: "rss",
  monitorName: "",
  scheduleMode: "daily",
  scheduleIntervalValue: 5,
  scheduleIntervalUnit: "hours",
  scheduleHour: 8,
  scheduleMinute: 0,
  scheduleWeekday: "MON",
  scheduleAdvancedCron: "",
  timezone: getLocalTimezone(),
  createScheduledOutput: true,
  programFormat: "concise_briefing",
  outcomeNoun: "briefing",
  showName: "",
  premise: "",
  audience: "",
  tone: "",
  episodeTitlePattern: "",
  customInstructions: "",
  templateName: "briefing_markdown",
  showNotes: false,
  emailDeliveryEnabled: false,
  emailRecipients: [],
  chatbookDeliveryEnabled: false,
  chatbookTitle: "",
  audioEnabled: true,
  audioSpeakers: [
    {
      id: "speaker_1",
      label: "Speaker 1",
      role: "host",
      voice: "alloy"
    }
  ],
  targetAudioMinutes: 8,
  audioProvider: "",
  audioModel: "",
  runNow: true
})

export const mergePipelineWizardDraft = (
  initial?: Partial<PipelineWizardDraft> | null
): PipelineWizardDraft => {
  const defaults = createDefaultPipelineWizardDraft()
  return {
    ...defaults,
    ...(initial || {}),
    sourceIds: Array.isArray(initial?.sourceIds) ? [...initial.sourceIds] : defaults.sourceIds,
    emailRecipients: Array.isArray(initial?.emailRecipients)
      ? [...initial.emailRecipients]
      : defaults.emailRecipients,
    audioSpeakers: Array.isArray(initial?.audioSpeakers)
      ? normalizePipelineWizardSpeakers(initial.audioSpeakers)
      : defaults.audioSpeakers,
    preservedOutputPrefs: initial?.preservedOutputPrefs
      ? structuredClone(initial.preservedOutputPrefs)
      : initial?.preservedOutputPrefs
  }
}

const trim = (value: unknown): string => String(value || "").trim()

export const getPipelineWizardSourceSignature = (
  draft: Pick<
    PipelineWizardDraft,
    "sourceMode" | "sourceIds" | "sourceName" | "sourceUrl" | "sourceType"
  >
): string => draft.sourceMode === "new"
  ? JSON.stringify([
      "new",
      trim(draft.sourceName),
      trim(draft.sourceUrl),
      draft.sourceType || "rss"
    ])
  : JSON.stringify([
      "existing",
      [...new Set(draft.sourceIds.map(Number).filter(Number.isFinite))].sort((a, b) => a - b)
    ])

const createAbortError = (): Error => {
  const error = new Error("The operation was aborted.")
  error.name = "AbortError"
  return error
}

const throwIfAborted = (signal?: AbortSignal): void => {
  if (signal?.aborted) throw createAbortError()
}

const wait = (milliseconds: number, signal?: AbortSignal): Promise<void> => {
  throwIfAborted(signal)
  if (milliseconds <= 0) return Promise.resolve()
  return new Promise((resolve, reject) => {
    const onAbort = () => {
      clearTimeout(timeoutId)
      reject(createAbortError())
    }
    const timeoutId = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort)
      resolve()
    }, milliseconds)
    signal?.addEventListener("abort", onAbort, { once: true })
  })
}

export const getPipelineWizardBriefingStatus = (
  projection: WatchlistBriefingProjection,
  waitForDelivery = false
): PipelineWizardBriefingStatus => {
  if (projection.artifact_status === "failed") return "failed"
  if (projection.artifact_status === "cancelled") return "cancelled"
  const stages = Object.values(projection.stages)
  if (stages.some((stage) => stage.status === "failed")) return "failed"
  if (stages.some((stage) => stage.status === "cancelled")) return "cancelled"
  if (projection.artifact_status !== "ready") return "running"
  if (!waitForDelivery) return "ready"
  if (["waiting_for_artifacts", "delivering"].includes(projection.delivery_status)) {
    return "running"
  }
  if (["failed", "partially_delivered", "unknown"].includes(projection.delivery_status)) {
    return "failed"
  }
  return "ready"
}

const BRIEFING_STAGE_LABELS: Record<string, string> = {
  collect: "Collect sources",
  select: "Select updates",
  render_text: "Create report",
  persist_text: "Save report in Reports",
  compose_audio_script: "Compose audio script",
  persist_audio_script: "Save audio script",
  generate_audio: "Create audio",
  persist_audio: "Save audio in Reports",
  deliver: "Deliver test"
}

export const getPipelineWizardBriefingOutcome = (
  projection: WatchlistBriefingProjection,
  waitForDelivery = false
): PipelineWizardBriefingOutcome => {
  const status = getPipelineWizardBriefingStatus(projection, waitForDelivery)
  if (status === "ready") return { status }
  const stageEntry = Object.entries(projection.stages).find(([, stage]) =>
    stage.status === status || (status === "failed" && stage.status === "failed")
  )
  const stage = stageEntry?.[0] || (waitForDelivery ? "deliver" : "briefing")
  const stageLabel = BRIEFING_STAGE_LABELS[stage] ||
    (stage.startsWith("deliver:") ? `Deliver to ${stage.slice("deliver:".length)}` : stage)
  const code = stageEntry?.[1].code ||
    (stage === "deliver" ? `delivery_${projection.delivery_status}` : undefined)
  if (status === "failed") {
    return {
      status,
      message: `${stageLabel} failed${code ? ` (${code})` : ""}. Open run ${projection.run_id} and retry this stage.`
    }
  }
  if (status === "cancelled") {
    return {
      status,
      message: `${stageLabel} was cancelled${code ? ` (${code})` : ""}. Your draft is saved.`
    }
  }
  const runningStage = Object.entries(projection.stages).find(([, stageState]) =>
    stageState.status === "running" || stageState.status === "queued"
  )?.[0]
  const runningLabel = runningStage
    ? BRIEFING_STAGE_LABELS[runningStage] || runningStage
    : "briefing work"
  return {
    status,
    message: `Run ${projection.run_id} is still running: ${runningLabel}. You can close setup and monitor it from Overview.`
  }
}

const isPipelineWizardBriefingTerminal = (
  projection: WatchlistBriefingProjection,
  waitForDelivery: boolean
): boolean => {
  return getPipelineWizardBriefingStatus(projection, waitForDelivery) !== "running"
}

export const waitForPipelineWizardBriefing = async (
  runId: number,
  getBriefing: (runId: number, signal?: AbortSignal) => Promise<WatchlistBriefingProjection>,
  onProgress: (projection: WatchlistBriefingProjection) => void,
  options: PipelineWizardBriefingPollOptions = {}
): Promise<WatchlistBriefingProjection> => {
  const intervalMs = Math.max(0, options.intervalMs ?? 1_000)
  const maxAttempts = Math.max(1, options.maxAttempts ?? 120)
  let lastProjection: WatchlistBriefingProjection | undefined
  for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
    throwIfAborted(options.signal)
    try {
      const projection = await getBriefing(runId, options.signal)
      lastProjection = projection
      onProgress(projection)
      if (isPipelineWizardBriefingTerminal(projection, Boolean(options.waitForDelivery))) {
        return projection
      }
    } catch (error) {
      const status = (error as { status?: number } | null)?.status
      if (status !== 404 || attempt === maxAttempts - 1) throw error
    }
    if (attempt < maxAttempts - 1) await wait(intervalMs, options.signal)
  }
  if (lastProjection) return lastProjection
  throw new Error("briefing_projection_unavailable")
}

const normalizeEmails = (value: string[] | undefined): string[] =>
  Array.isArray(value)
    ? value.map((entry) => trim(entry).toLowerCase()).filter(Boolean)
    : []

const isHttpUrl = (value: string): boolean => {
  try {
    const parsed = new URL(value)
    return parsed.protocol === "http:" || parsed.protocol === "https:"
  } catch {
    return false
  }
}

export const validatePipelineWizardCron = (
  value: string
): PipelineWizardCronValidationError => {
  const cron = trim(value)
  if (!cron) return "required"
  return validateCronSchedule(cron, MIN_SCHEDULE_INTERVAL_MINUTES)
}

export const normalizePipelineWizardSpeakers = (
  speakers: PipelineWizardAudioSpeakerDraft[] | undefined
): PipelineWizardAudioSpeakerDraft[] => {
  const usedIds = new Set<string>()
  return (Array.isArray(speakers) ? speakers : [])
    .map((speaker, index) => {
      const requestedId = trim(speaker.id)
      let id = requestedId && !usedIds.has(requestedId)
        ? requestedId
        : `speaker_${index + 1}`
      let suffix = 2
      while (usedIds.has(id)) {
        id = `speaker_${index + 1}_${suffix}`
        suffix += 1
      }
      usedIds.add(id)
      return {
      id,
      label: trim(speaker.label) || `Speaker ${index + 1}`,
      role: trim(speaker.role) || undefined,
      voice: trim(speaker.voice),
      persona: trim(speaker.persona) || undefined
      }
    })
    .filter((speaker) => speaker.id.length > 0 || speaker.label.length > 0 || speaker.voice.length > 0)
}

export const validatePipelineWizardDraft = (
  draft: PipelineWizardDraft
): PipelineWizardValidationResult => {
  const errors: string[] = []
  const sourceMode = draft.sourceMode || "existing"
  if (sourceMode === "existing") {
    if (!Array.isArray(draft.sourceIds) || draft.sourceIds.length === 0) {
      errors.push("sourceIds")
    }
  } else {
    if (!trim(draft.sourceName)) errors.push("sourceName")
    const sourceUrl = trim(draft.sourceUrl)
    if (!sourceUrl) {
      errors.push("sourceUrl")
    } else if (!isHttpUrl(sourceUrl)) {
      errors.push("sourceUrl")
    }
  }

  if (!trim(draft.monitorName)) errors.push("monitorName")
  if (!trim(draft.templateName)) errors.push("templateName")
  try {
    new Intl.DateTimeFormat("en", { timeZone: trim(draft.timezone) || "UTC" }).format()
  } catch {
    errors.push("timezone")
  }

  if (draft.scheduleMode === "interval") {
    const value = Number(draft.scheduleIntervalValue)
    const minValue =
      draft.scheduleIntervalUnit === "minutes" ? INTERVAL_MINUTES_MIN : INTERVAL_HOURS_MIN
    const maxValue =
      draft.scheduleIntervalUnit === "minutes" ? INTERVAL_MINUTES_MAX : INTERVAL_HOURS_MAX
    if (!Number.isInteger(value) || value < minValue || value > maxValue) {
      errors.push("scheduleIntervalValue")
    }
    if (draft.scheduleIntervalUnit === "hours") {
      const minute = Number(draft.scheduleMinute)
      if (!Number.isInteger(minute) || minute < 0 || minute > 59) errors.push("scheduleMinute")
    }
  }

  if (
    draft.scheduleMode === "daily" ||
    draft.scheduleMode === "weekdays" ||
    draft.scheduleMode === "weekly"
  ) {
    const hour = Number(draft.scheduleHour)
    const minute = Number(draft.scheduleMinute)
    if (!Number.isInteger(hour) || hour < 0 || hour > 23) errors.push("scheduleHour")
    if (!Number.isInteger(minute) || minute < 0 || minute > 59) errors.push("scheduleMinute")
  }

  if (draft.scheduleMode === "advanced") {
    const cronValidation = validatePipelineWizardCron(draft.scheduleAdvancedCron)
    if (cronValidation === "too_frequent") {
      errors.push("scheduleAdvancedCronTooFrequent")
    } else if (cronValidation) {
      errors.push("scheduleAdvancedCron")
    }
  }

  if (draft.emailDeliveryEnabled) {
    const recipients = normalizeEmails(draft.emailRecipients)
    if (recipients.length === 0 || recipients.some((entry) => !EMAIL_PATTERN.test(entry))) {
      errors.push("emailRecipients")
    }
  }

  if (draft.audioEnabled) {
    const speakers = normalizePipelineWizardSpeakers(draft.audioSpeakers)
    if (speakers.length < 1 || speakers.length > 4) errors.push("audioSpeakers")
    if (speakers.some((speaker) => !trim(speaker.voice))) errors.push("audioSpeakerVoices")
    const minutes = Number(draft.targetAudioMinutes)
    if (!Number.isFinite(minutes) || minutes < 1 || minutes > 60) {
      errors.push("targetAudioMinutes")
    }
  }

  return {
    valid: errors.length === 0,
    errors
  }
}

export const toPipelineWizardSourcePayload = (
  draft: PipelineWizardDraft,
  watchlistId?: number | null
): WatchlistSourceCreate | null => {
  if (draft.sourceMode !== "new") return null
  const payload: WatchlistSourceCreate = {
    name: trim(draft.sourceName),
    url: trim(draft.sourceUrl),
    source_type: draft.sourceType || "rss",
    active: true
  }
  const normalizedWatchlistId = Number(watchlistId)
  if (Number.isFinite(normalizedWatchlistId) && normalizedWatchlistId > 0) {
    payload.watchlist_id = normalizedWatchlistId
  }
  return payload
}

export const buildPipelineWizardSchedule = (
  draft: PipelineWizardDraft
): { schedule_expr?: string; timezone?: string } => {
  if (draft.scheduleMode === "manual") return {}
  if (draft.scheduleMode === "advanced") {
    const cron = trim(draft.scheduleAdvancedCron)
    return cron && !validatePipelineWizardCron(cron)
      ? { schedule_expr: cron, timezone: trim(draft.timezone) || getLocalTimezone() }
      : {}
  }
  const preset =
    draft.scheduleMode === "interval"
      ? "interval"
      : draft.scheduleMode === "weekly"
        ? "weekly"
        : draft.scheduleMode === "weekdays"
          ? "weekdays"
          : "daily"
  return {
    schedule_expr: buildCronFromPreset({
      preset,
      intervalValue: draft.scheduleIntervalValue,
      intervalUnit: draft.scheduleIntervalUnit,
      hour: draft.scheduleHour,
      minute: draft.scheduleMinute,
      weekday: draft.scheduleWeekday
    }),
    timezone: trim(draft.timezone) || getLocalTimezone()
  }
}

const CRON_MONTHS: Record<string, number> = {
  JAN: 1, FEB: 2, MAR: 3, APR: 4, MAY: 5, JUN: 6,
  JUL: 7, AUG: 8, SEP: 9, OCT: 10, NOV: 11, DEC: 12
}
const CRON_WEEKDAYS: Record<string, number> = {
  SUN: 0, MON: 1, TUE: 2, WED: 3, THU: 4, FRI: 5, SAT: 6
}

const cronAtom = (
  value: string,
  names: Record<string, number>
): number => names[value.toUpperCase()] ?? Number(value)

const cronFieldMatches = (
  token: string,
  value: number,
  min: number,
  max: number,
  names: Record<string, number> = {},
  alternateValues: number[] = []
): boolean => token.split(",").some((part) => {
  const [base, rawStep] = part.split("/")
  const step = rawStep ? Number(rawStep) : 1
  if (!Number.isInteger(step) || step <= 0) return false
  let start = min
  let end = max
  if (base !== "*" && base !== "?") {
    const range = base.split("-")
    start = cronAtom(range[0], names)
    end = range[1]
      ? cronAtom(range[1], names)
      : rawStep
        ? max
        : start
  }
  return Number.isFinite(start) && Number.isFinite(end) &&
    [value, ...alternateValues].some((candidate) =>
      candidate >= start && candidate <= end && (candidate - start) % step === 0
    )
})

const cronWildcard = (token: string): boolean => token === "*" || token === "?"

export const projectPipelineWizardOccurrences = (
  draft: PipelineWizardDraft,
  from = new Date()
): { nextRunAt?: string; followingRunAt?: string } => {
  if (draft.nextRunAt) {
    return {
      nextRunAt: draft.nextRunAt,
      ...(draft.followingRunAt ? { followingRunAt: draft.followingRunAt } : {})
    }
  }
  if (draft.scheduleMode === "manual") return {}
  const expression = buildPipelineWizardSchedule(draft).schedule_expr
  if (!expression || validatePipelineWizardCron(expression)) return {}
  const fields = expression.split(/\s+/)
  const timezone = trim(draft.timezone) || "UTC"
  let formatter: Intl.DateTimeFormat
  try {
    formatter = new Intl.DateTimeFormat("en-US", {
      timeZone: timezone,
      year: "numeric",
      month: "numeric",
      day: "numeric",
      weekday: "short",
      hour: "numeric",
      minute: "numeric",
      hourCycle: "h23"
    })
    formatter.format(from)
  } catch {
    return {}
  }
  const candidate = new Date(Math.floor(from.getTime() / 60_000) * 60_000 + 60_000)
  const occurrences: string[] = []
  const maxMinutes = 370 * 24 * 60
  for (let index = 0; index < maxMinutes && occurrences.length < 2; index += 1) {
    const parts = Object.fromEntries(
      formatter.formatToParts(candidate).map((part) => [part.type, part.value])
    )
    const minuteMatch = cronFieldMatches(fields[0], Number(parts.minute), 0, 59)
    const hourMatch = cronFieldMatches(fields[1], Number(parts.hour), 0, 23)
    const monthMatch = cronFieldMatches(fields[3], Number(parts.month), 1, 12, CRON_MONTHS)
    const dayOfMonthMatch = cronFieldMatches(fields[2], Number(parts.day), 1, 31)
    const weekdayValue = CRON_WEEKDAYS[String(parts.weekday || "").toUpperCase()]
    const weekdayMatch = cronFieldMatches(
      fields[4],
      weekdayValue,
      0,
      7,
      CRON_WEEKDAYS,
      weekdayValue === 0 ? [7] : []
    )
    const dayMatch = !cronWildcard(fields[2]) && !cronWildcard(fields[4])
      ? dayOfMonthMatch || weekdayMatch
      : dayOfMonthMatch && weekdayMatch
    if (minuteMatch && hourMatch && monthMatch && dayMatch) {
      occurrences.push(candidate.toISOString())
    }
    candidate.setUTCMinutes(candidate.getUTCMinutes() + 1)
  }
  return {
    ...(occurrences[0] ? { nextRunAt: occurrences[0] } : {}),
    ...(occurrences[1] ? { followingRunAt: occurrences[1] } : {})
  }
}

export const toWatchlistAudioCast = (
  draft: PipelineWizardDraft
): WatchlistAudioCast | undefined => {
  if (!draft.audioEnabled) return undefined
  const speakers = normalizePipelineWizardSpeakers(draft.audioSpeakers)
  if (speakers.length < 1 || speakers.length > 4) return undefined
  return {
    speaker_count: speakers.length as 1 | 2 | 3 | 4,
    speakers: speakers.map((speaker): WatchlistAudioCastSpeaker => ({
      id: speaker.id,
      label: speaker.label,
      role: speaker.role,
      voice: speaker.voice,
      persona: speaker.persona
    }))
  }
}

export const toAudioVoiceMap = (
  draft: PipelineWizardDraft
): Record<string, string> | undefined => {
  if (!draft.audioEnabled) return undefined
  const entries = normalizePipelineWizardSpeakers(draft.audioSpeakers)
    .filter((speaker) => speaker.id && speaker.voice)
    .map((speaker) => [speaker.id, speaker.voice] as const)
  return entries.length > 0 ? Object.fromEntries(entries) : undefined
}

export const toBriefingPipelineDraft = (
  draft: PipelineWizardDraft,
  sourceIdsOverride?: number[]
): BriefingPipelineDraft => {
  const sourceIds =
    sourceIdsOverride && sourceIdsOverride.length > 0
      ? sourceIdsOverride
      : draft.sourceIds
  const schedule = buildPipelineWizardSchedule(draft)
  const audioCast = toWatchlistAudioCast(draft)
  const voiceMap = toAudioVoiceMap(draft)
  const firstSpeakerVoice = audioCast?.speakers[0]?.voice

  return {
    monitorName: trim(draft.monitorName),
    sourceIds,
    active: false,
    schedulePreset: schedule.schedule_expr ? "daily" : "none",
    scheduleExpr: schedule.schedule_expr,
    timezone: schedule.timezone,
    createScheduledOutput: Boolean(schedule.schedule_expr && draft.createScheduledOutput),
    templateName: trim(draft.templateName),
    ...(draft.templateFormat ? { templateFormat: draft.templateFormat } : {}),
    programFormat: draft.programFormat,
    outcomeNoun: draft.outcomeNoun,
    showName: trim(draft.showName) || undefined,
    premise: trim(draft.premise) || undefined,
    audience: trim(draft.audience) || undefined,
    tone: trim(draft.tone) || undefined,
    episodeTitlePattern: trim(draft.episodeTitlePattern) || undefined,
    customInstructions: trim(draft.customInstructions) || undefined,
    showNotes: draft.showNotes,
    includeAudio: Boolean(draft.audioEnabled),
    audioVoice: firstSpeakerVoice,
    audioProvider: trim(draft.audioProvider) || undefined,
    audioModel: trim(draft.audioModel) || undefined,
    audioCast,
    voiceMap,
    targetAudioMinutes: draft.audioEnabled ? Number(draft.targetAudioMinutes) : undefined,
    emailRecipients: draft.emailDeliveryEnabled ? normalizeEmails(draft.emailRecipients) : [],
    createChatbook: Boolean(draft.chatbookDeliveryEnabled),
    chatbookTitle: draft.chatbookDeliveryEnabled ? trim(draft.chatbookTitle) : undefined,
    preservedOutputPrefs: draft.preservedOutputPrefs
  }
}

const WEEKDAY_LABELS: Record<WeekdayToken, string> = {
  MON: "Monday",
  TUE: "Tuesday",
  WED: "Wednesday",
  THU: "Thursday",
  FRI: "Friday",
  SAT: "Saturday",
  SUN: "Sunday"
}

export const formatPipelineWizardCadence = (
  draft: PipelineWizardDraft,
  copy: PipelineWizardCadenceCopy = {}
): string => {
  if (draft.scheduleMode === "manual") return copy.manual || "Manual only"
  if (draft.scheduleMode === "interval") {
    const value = Math.max(1, Math.floor(Number(draft.scheduleIntervalValue) || 1))
    if (copy.interval) return copy.interval(value, draft.scheduleIntervalUnit)
    const unit = draft.scheduleIntervalUnit === "minutes" ? "minute" : "hour"
    return `Every ${value} ${unit}${value === 1 ? "" : "s"}`
  }
  if (draft.scheduleMode === "weekly") {
    const weekday = copy.weekdayLabels?.[draft.scheduleWeekday] || WEEKDAY_LABELS[draft.scheduleWeekday]
    const time = formatScheduleTime(draft.scheduleHour, draft.scheduleMinute)
    if (copy.weekly) return copy.weekly(weekday, time)
    return `Weekly on ${weekday} at ${time}`
  }
  if (draft.scheduleMode === "weekdays") {
    const time = formatScheduleTime(draft.scheduleHour, draft.scheduleMinute)
    if (copy.weekdays) return copy.weekdays(time)
    return `Weekdays at ${time}`
  }
  if (draft.scheduleMode === "advanced") {
    const cron = trim(draft.scheduleAdvancedCron)
    return copy.advanced ? copy.advanced(cron) : `Custom cron: ${cron}`
  }
  const time = formatScheduleTime(draft.scheduleHour, draft.scheduleMinute)
  return copy.daily ? copy.daily(time) : `Daily at ${time}`
}

export const buildPipelineWizardReviewSummary = (
  draft: PipelineWizardDraft,
  existingSources: SourceLabel[] = [],
  copy: PipelineWizardReviewSummaryCopy = {}
): PipelineWizardReviewSummary => {
  const sourceLookup = new Map(
    existingSources.map((source) => [
      source.id,
      trim(source.name) || (copy.feedLabel ? copy.feedLabel(source.id) : `Feed #${source.id}`)
    ])
  )
  const selectedSources = draft.sourceMode === "new"
    ? trim(draft.sourceName) || copy.newFeed || "New feed"
    : (draft.sourceIds || [])
      .map((id) => sourceLookup.get(id) || (copy.feedLabel ? copy.feedLabel(id) : `Feed #${id}`))
      .join(", ") || copy.noFeedsSelected || "No feeds selected"
  const delivery = [
    draft.emailDeliveryEnabled ? copy.email || "Email" : null,
    draft.chatbookDeliveryEnabled ? copy.chatbook || "Chatbook" : null
  ].filter(Boolean).join(", ") || copy.inAppReports || "In-app reports"
  const speakers = draft.audioEnabled ? normalizePipelineWizardSpeakers(draft.audioSpeakers).length : 0
  const templateName = trim(draft.templateName) || copy.noTemplate || "No template"

  return {
    sources: selectedSources,
    cadence: formatPipelineWizardCadence(draft, copy.cadence),
    filters: copy.filters || "Monitor filters can be refined after creation",
    output: copy.outputDigest ? copy.outputDigest(templateName) : `${templateName} digest`,
    delivery,
    audio: speakers > 0
      ? copy.audioBriefing
        ? copy.audioBriefing(speakers)
        : `${speakers} speaker${speakers === 1 ? "" : "s"} audio briefing`
      : copy.audioDisabled || "Audio disabled"
  }
}
