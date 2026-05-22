import type {
  WatchlistAudioCast,
  WatchlistJobCreate,
  WatchlistOutputCreate
} from "@/types/watchlists"
import {
  resolveQuickSetupSchedule,
  type WatchlistCadenceDraft,
  type QuickSetupSchedulePreset
} from "./quick-setup"
import {
  formatScheduleTimeValue,
  normalizeWeekdayToken,
  type ScheduleIntervalUnit,
  type WeekdayToken
} from "../JobsTab/schedule-utils"
import { normalizeWatchlistTemplateName } from "../shared/templateNames"

export interface BriefingPipelineDraft {
  monitorName: string
  sourceIds: number[]
  schedulePreset: QuickSetupSchedulePreset
  scheduleCadence?: WatchlistCadenceDraft
  scheduleExpr?: string | null
  timezone?: string
  createScheduledOutput?: boolean
  templateName: string
  templateFormat?: "md" | "html"
  templateVersion?: number
  includeAudio: boolean
  audioVoice?: string
  audioCast?: WatchlistAudioCast
  voiceMap?: Record<string, string>
  targetAudioMinutes?: number
  emailRecipients?: string[]
  createChatbook?: boolean
  chatbookTitle?: string
}

export interface PipelineValidationResult {
  valid: boolean
  errors: string[]
}

export interface PipelineReviewSummary {
  scheduleLabel: string
  artifacts: string[]
  deliveries: string[]
}

export interface PipelineScheduleLabelCopy {
  manual?: string
  hourly?: string
  dailyPreset?: string
  weekdaysPreset?: string
  advanced?: (cron: string) => string
  interval?: (value: number, unit: ScheduleIntervalUnit) => string
  daily?: (time: string) => string
  weekdays?: (time: string) => string
  weekly?: (weekday: string, time: string) => string
  weekdayLabels?: Partial<Record<WeekdayToken, string>>
}

export interface PipelineReviewSummaryCopy {
  schedule?: PipelineScheduleLabelCopy
  textBriefing?: string
  audioBriefing?: string
  email?: string
  chatbook?: string
  inAppReports?: string
}

const SCHEDULE_LABELS: Record<QuickSetupSchedulePreset, string> = {
  none: "Manual only",
  hourly: "Hourly",
  daily: "Daily at 08:00",
  weekdays: "Weekdays at 08:00"
}

const WEEKDAY_LABELS: Record<WeekdayToken, string> = {
  SUN: "Sunday",
  MON: "Monday",
  TUE: "Tuesday",
  WED: "Wednesday",
  THU: "Thursday",
  FRI: "Friday",
  SAT: "Saturday"
}

const normalizeRecipients = (value: string[] | undefined): string[] => {
  if (!Array.isArray(value)) return []
  return value
    .map((entry) => String(entry || "").trim())
    .filter((entry) => entry.length > 0)
}

const formatPresetScheduleLabel = (
  preset: QuickSetupSchedulePreset,
  copy: PipelineScheduleLabelCopy = {}
): string => {
  if (preset === "none") return copy.manual || SCHEDULE_LABELS.none
  if (preset === "hourly") return copy.hourly || SCHEDULE_LABELS.hourly
  if (preset === "weekdays") return copy.weekdaysPreset || SCHEDULE_LABELS.weekdays
  return copy.dailyPreset || SCHEDULE_LABELS.daily
}

const formatCustomCronLabel = (
  cron: string,
  copy: PipelineScheduleLabelCopy = {}
): string => {
  if (copy.advanced) return copy.advanced(cron)
  return cron ? `Custom cron: ${cron}` : "Custom cron"
}

const formatScheduleCadenceLabel = (
  cadence: WatchlistCadenceDraft | undefined,
  fallback: QuickSetupSchedulePreset,
  copy: PipelineScheduleLabelCopy = {}
): string => {
  if (!cadence) return formatPresetScheduleLabel(fallback, copy)
  if (cadence.kind === "manual") return copy.manual || SCHEDULE_LABELS.none
  if (cadence.kind === "advanced") {
    const cron = String(cadence.cron || "").trim()
    return formatCustomCronLabel(cron, copy)
  }
  if (cadence.kind === "interval") {
    const value = Math.max(1, Math.floor(Number(cadence.every) || 1))
    const unit: ScheduleIntervalUnit =
      cadence.unit === "minute" || cadence.unit === "minutes" ? "minutes" : "hours"
    if (copy.interval) return copy.interval(value, unit)
    const unitLabel = unit === "minutes" ? "minute" : "hour"
    return `Every ${value} ${unitLabel}${value === 1 ? "" : "s"}`
  }
  const time = formatScheduleTimeValue("time" in cadence ? cadence.time : undefined)
  if (cadence.kind === "daily") return copy.daily ? copy.daily(time) : `Daily at ${time}`
  if (cadence.kind === "weekdays") {
    return copy.weekdays ? copy.weekdays(time) : `Weekdays at ${time}`
  }
  const weekday =
    copy.weekdayLabels?.[normalizeWeekdayToken(cadence.weekday)] ||
    WEEKDAY_LABELS[normalizeWeekdayToken(cadence.weekday)]
  return copy.weekly ? copy.weekly(weekday, time) : `Weekly on ${weekday} at ${time}`
}

export const validateBriefingPipelineDraft = (
  draft: BriefingPipelineDraft
): PipelineValidationResult => {
  const errors: string[] = []
  if (String(draft.monitorName || "").trim().length === 0) {
    errors.push("monitorName")
  }
  if (!Array.isArray(draft.sourceIds) || draft.sourceIds.length === 0) {
    errors.push("sourceIds")
  }
  if (String(draft.templateName || "").trim().length === 0) {
    errors.push("templateName")
  }
  if (draft.includeAudio && String(draft.audioVoice || "").trim().length === 0) {
    errors.push("audioVoice")
  }
  if (draft.includeAudio) {
    const minutes = Number(draft.targetAudioMinutes)
    if (!Number.isFinite(minutes) || minutes <= 0) {
      errors.push("targetAudioMinutes")
    }
  }
  return {
    valid: errors.length === 0,
    errors
  }
}

export const toPipelineJobCreatePayload = (
  draft: BriefingPipelineDraft
): WatchlistJobCreate => {
  const rawScheduleExpr = String(draft.scheduleExpr ?? "").trim()
  const schedule = draft.scheduleCadence
    ? resolveQuickSetupSchedule(draft.scheduleCadence)
    : rawScheduleExpr
      ? {
          schedule_expr: rawScheduleExpr,
          timezone: draft.timezone
        }
      : resolveQuickSetupSchedule(draft.schedulePreset)
  const recipients = normalizeRecipients(draft.emailRecipients)
  const templateVersionNum = Number(draft.templateVersion)
  const normalizedTemplateVersion =
    Number.isFinite(templateVersionNum) && templateVersionNum > 0
      ? templateVersionNum
      : undefined
  const templateFormat =
    draft.templateFormat === "html" || draft.templateFormat === "md"
      ? draft.templateFormat
      : undefined
  const shouldAutoOutput = Boolean(draft.createScheduledOutput && schedule.schedule_expr)
  const normalizedAudioVoice = String(draft.audioVoice || "").trim() || undefined
  const templateName = normalizeWatchlistTemplateName(draft.templateName)

  return {
    name: String(draft.monitorName || "").trim(),
    scope: { sources: draft.sourceIds },
    active: true,
    ...schedule,
    output_prefs: {
      ...(shouldAutoOutput
        ? {
            auto_output: {
              enabled: true,
              type: "briefing_markdown",
              ...(templateFormat ? { format: templateFormat } : {}),
              ...(templateName ? { template_name: templateName } : {}),
              ...(normalizedTemplateVersion ? { template_version: normalizedTemplateVersion } : {})
            }
          }
        : {}),
      template_name: templateName,
      template: {
        default_name: templateName,
        ...(templateFormat ? { default_format: templateFormat } : {}),
        default_version: normalizedTemplateVersion
      },
      generate_audio: draft.includeAudio,
      audio_voice: draft.includeAudio ? normalizedAudioVoice : undefined,
      audio_cast: draft.includeAudio ? draft.audioCast : undefined,
      voice_map: draft.includeAudio ? draft.voiceMap : undefined,
      target_audio_minutes: draft.includeAudio
        ? Number(draft.targetAudioMinutes)
        : undefined,
      deliveries: {
        email:
          recipients.length > 0
            ? {
                enabled: true,
                recipients
              }
            : undefined,
        chatbook: draft.createChatbook
          ? {
              enabled: true,
              title: String(draft.chatbookTitle || "").trim() || "Watchlists Briefing"
            }
          : undefined
      }
    }
  }
}

export const toPipelineOutputCreatePayload = (
  runId: number,
  draft: BriefingPipelineDraft,
  itemIds?: number[]
): WatchlistOutputCreate => {
  const recipients = normalizeRecipients(draft.emailRecipients)
  const templateFormat =
    draft.templateFormat === "html" || draft.templateFormat === "md"
      ? draft.templateFormat
      : undefined
  const templateName = normalizeWatchlistTemplateName(draft.templateName)
  const payload: WatchlistOutputCreate = {
    run_id: runId,
    item_ids: itemIds,
    type: "briefing_markdown",
    ...(templateFormat ? { format: templateFormat } : {}),
    template_name: templateName
  }

  if (Number.isFinite(Number(draft.templateVersion)) && Number(draft.templateVersion) > 0) {
    payload.template_version = Number(draft.templateVersion)
  }

  if (draft.includeAudio) {
    const normalizedAudioVoice = String(draft.audioVoice || "").trim() || undefined
    payload.generate_audio = true
    if (normalizedAudioVoice) payload.audio_voice = normalizedAudioVoice
    if (draft.audioCast) payload.audio_cast = draft.audioCast
    if (draft.voiceMap) payload.voice_map = draft.voiceMap
    const targetAudioMinutes = Number(draft.targetAudioMinutes)
    if (Number.isFinite(targetAudioMinutes) && targetAudioMinutes > 0) {
      payload.target_audio_minutes = targetAudioMinutes
    }
    payload.metadata = {
      audio: {
        enabled: true,
        voice: normalizedAudioVoice || null,
        ...(draft.audioCast ? { speaker_count: draft.audioCast.speaker_count } : {}),
        target_minutes: targetAudioMinutes
      }
    }
  }

  if (recipients.length > 0 || draft.createChatbook) {
    payload.deliveries = {
      email:
        recipients.length > 0
          ? {
              recipients
            }
          : undefined,
      chatbook: draft.createChatbook
        ? {
            enabled: true,
            title: String(draft.chatbookTitle || "").trim() || "Watchlists Briefing"
          }
        : undefined
    }
  }

  return payload
}

export const buildPipelineReviewSummary = (
  draft: BriefingPipelineDraft,
  copy: PipelineReviewSummaryCopy = {}
): PipelineReviewSummary => {
  const rawScheduleExpr = String(draft.scheduleExpr ?? "").trim()
  const artifacts = [copy.textBriefing || "Text briefing"]
  if (draft.includeAudio) artifacts.push(copy.audioBriefing || "Audio briefing")

  const deliveries: string[] = []
  if (normalizeRecipients(draft.emailRecipients).length > 0) deliveries.push(copy.email || "Email")
  if (draft.createChatbook) deliveries.push(copy.chatbook || "Chatbook")
  if (deliveries.length === 0) deliveries.push(copy.inAppReports || "In-app reports")

  return {
    scheduleLabel: draft.scheduleCadence
      ? formatScheduleCadenceLabel(draft.scheduleCadence, draft.schedulePreset, copy.schedule)
      : rawScheduleExpr
        ? formatCustomCronLabel(rawScheduleExpr, copy.schedule)
        : formatPresetScheduleLabel(draft.schedulePreset, copy.schedule),
    artifacts,
    deliveries
  }
}
