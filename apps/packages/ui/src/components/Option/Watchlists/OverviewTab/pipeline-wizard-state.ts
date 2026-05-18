import type {
  SourceType,
  WatchlistAudioCast,
  WatchlistAudioCastSpeaker,
  WatchlistSourceCreate
} from "@/types/watchlists"
import {
  buildCronFromPreset,
  type ScheduleIntervalUnit,
  type WeekdayToken
} from "../JobsTab/schedule-utils"
import { getLocalTimezone } from "./quick-setup"
import type { BriefingPipelineDraft } from "./pipeline-contract"

export type PipelineWizardSourceMode = "existing" | "new"
export type PipelineWizardScheduleMode = "manual" | "interval" | "daily" | "weekly"

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
  templateName: string
  templateFormat: "md" | "html"
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

export interface PipelineWizardReviewSummary {
  sources: string
  cadence: string
  filters: string
  output: string
  delivery: string
  audio: string
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
  templateName: "",
  templateFormat: "md",
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
    if (!Number.isFinite(value) || value <= 0) errors.push("scheduleIntervalValue")
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
  const preset =
    draft.scheduleMode === "interval"
      ? "interval"
      : draft.scheduleMode === "weekly"
        ? "weekly"
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
    templateName: trim(draft.templateName),
    templateFormat: draft.templateFormat,
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

const formatTime = (hour: number, minute: number): string => {
  const normalizedHour = Math.max(0, Math.min(23, Math.floor(Number(hour) || 0)))
  const normalizedMinute = Math.max(0, Math.min(59, Math.floor(Number(minute) || 0)))
  return `${String(normalizedHour).padStart(2, "0")}:${String(normalizedMinute).padStart(2, "0")}`
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

export const formatPipelineWizardCadence = (draft: PipelineWizardDraft): string => {
  if (draft.scheduleMode === "manual") return "Manual only"
  if (draft.scheduleMode === "interval") {
    const value = Math.max(1, Math.floor(Number(draft.scheduleIntervalValue) || 1))
    const unit = draft.scheduleIntervalUnit === "minutes" ? "minute" : "hour"
    return `Every ${value} ${unit}${value === 1 ? "" : "s"}`
  }
  if (draft.scheduleMode === "weekly") {
    return `Weekly on ${WEEKDAY_LABELS[draft.scheduleWeekday]} at ${formatTime(
      draft.scheduleHour,
      draft.scheduleMinute
    )}`
  }
  return `Daily at ${formatTime(draft.scheduleHour, draft.scheduleMinute)}`
}

export const buildPipelineWizardReviewSummary = (
  draft: PipelineWizardDraft,
  existingSources: SourceLabel[] = []
): PipelineWizardReviewSummary => {
  const sourceLookup = new Map(existingSources.map((source) => [source.id, trim(source.name) || `Feed #${source.id}`]))
  const selectedSources = draft.sourceMode === "new"
    ? trim(draft.sourceName) || "New feed"
    : (draft.sourceIds || [])
      .map((id) => sourceLookup.get(id) || `Feed #${id}`)
      .join(", ") || "No feeds selected"
  const delivery = [
    draft.emailDeliveryEnabled ? "Email" : null,
    draft.chatbookDeliveryEnabled ? "Chatbook" : null
  ].filter(Boolean).join(", ") || "In-app reports"
  const speakers = draft.audioEnabled ? normalizePipelineWizardSpeakers(draft.audioSpeakers).length : 0

  return {
    sources: selectedSources,
    cadence: formatPipelineWizardCadence(draft),
    filters: "Monitor filters can be refined after creation",
    output: `${trim(draft.templateName) || "No template"} digest`,
    delivery,
    audio: speakers > 0
      ? `${speakers} speaker${speakers === 1 ? "" : "s"} audio briefing`
      : "Audio disabled"
  }
}
