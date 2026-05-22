import type {
  SourceType,
  WatchlistAudioCast,
  WatchlistAudioCastSpeaker,
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
  templateName: string
  templateFormat?: "md" | "html"
  emailDeliveryEnabled: boolean
  emailRecipients: string[]
  chatbookDeliveryEnabled: boolean
  chatbookTitle: string
  audioEnabled: boolean
  audioSpeakers: PipelineWizardAudioSpeakerDraft[]
  targetAudioMinutes: number
  runNow: boolean
}

export interface PipelineWizardValidationResult {
  valid: boolean
  errors: string[]
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
  templateName: "",
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
  runNow: true
})

const trim = (value: unknown): string => String(value || "").trim()

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
): PipelineWizardAudioSpeakerDraft[] =>
  (Array.isArray(speakers) ? speakers : [])
    .map((speaker, index) => ({
      id: trim(speaker.id) || `speaker_${index + 1}`,
      label: trim(speaker.label) || `Speaker ${index + 1}`,
      role: trim(speaker.role) || undefined,
      voice: trim(speaker.voice),
      persona: trim(speaker.persona) || undefined
    }))
    .filter((speaker) => speaker.id.length > 0 || speaker.label.length > 0 || speaker.voice.length > 0)

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
    const ids = speakers.map((speaker) => speaker.id)
    if (new Set(ids).size !== ids.length) errors.push("audioSpeakerIds")
    if (speakers.some((speaker) => !trim(speaker.voice))) errors.push("audioSpeakerVoices")
    const minutes = Number(draft.targetAudioMinutes)
    if (!Number.isFinite(minutes) || minutes <= 0) errors.push("targetAudioMinutes")
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
      ? { schedule_expr: cron, timezone: getLocalTimezone() }
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
    timezone: getLocalTimezone()
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
    schedulePreset: schedule.schedule_expr ? "daily" : "none",
    scheduleExpr: schedule.schedule_expr,
    timezone: schedule.timezone,
    createScheduledOutput: Boolean(schedule.schedule_expr),
    templateName: trim(draft.templateName),
    ...(draft.templateFormat ? { templateFormat: draft.templateFormat } : {}),
    includeAudio: Boolean(draft.audioEnabled),
    audioVoice: firstSpeakerVoice,
    audioCast,
    voiceMap,
    targetAudioMinutes: draft.audioEnabled ? Number(draft.targetAudioMinutes) : undefined,
    emailRecipients: draft.emailDeliveryEnabled ? normalizeEmails(draft.emailRecipients) : [],
    createChatbook: Boolean(draft.chatbookDeliveryEnabled),
    chatbookTitle: draft.chatbookDeliveryEnabled ? trim(draft.chatbookTitle) : undefined
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
