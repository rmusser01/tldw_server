import type {
  WatchlistAudioCast,
  WatchlistJobCreate,
  WatchlistOutputCreate
} from "@/types/watchlists"
import {
  resolveQuickSetupSchedule,
  type QuickSetupSchedulePreset
} from "./quick-setup"
import { normalizeWatchlistTemplateName } from "../shared/templateNames"

export interface BriefingPipelineDraft {
  monitorName: string
  sourceIds: number[]
  schedulePreset: QuickSetupSchedulePreset
  scheduleExpr?: string | null
  timezone?: string
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

const SCHEDULE_LABELS: Record<QuickSetupSchedulePreset, string> = {
  none: "Manual only",
  hourly: "Hourly",
  daily: "Daily at 08:00",
  weekdays: "Weekdays at 08:00"
}

const normalizeRecipients = (value: string[] | undefined): string[] => {
  if (!Array.isArray(value)) return []
  return value
    .map((entry) => String(entry || "").trim())
    .filter((entry) => entry.length > 0)
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
  const schedule = draft.scheduleExpr
    ? {
        schedule_expr: draft.scheduleExpr,
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
  const shouldAutoOutput = Boolean(schedule.schedule_expr)
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
  draft: BriefingPipelineDraft
): PipelineReviewSummary => {
  const artifacts = ["Text briefing"]
  if (draft.includeAudio) artifacts.push("Audio briefing")

  const deliveries: string[] = []
  if (normalizeRecipients(draft.emailRecipients).length > 0) deliveries.push("Email")
  if (draft.createChatbook) deliveries.push("Chatbook")
  if (deliveries.length === 0) deliveries.push("In-app reports")

  return {
    scheduleLabel: SCHEDULE_LABELS[draft.schedulePreset],
    artifacts,
    deliveries
  }
}
