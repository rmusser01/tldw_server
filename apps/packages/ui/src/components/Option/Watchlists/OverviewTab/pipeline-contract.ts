import type { WatchlistJobCreate, WatchlistOutputCreate } from "@/types/watchlists"
import {
  resolveQuickSetupSchedule,
  type QuickSetupSchedulePreset
} from "./quick-setup"

export interface BriefingPipelineDraft {
  monitorName: string
  sourceIds: number[]
  schedulePreset: QuickSetupSchedulePreset
  templateName: string
  templateFormat?: "md" | "html"
  templateVersion?: number
  includeAudio: boolean
  audioVoice?: string
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
  const schedule = resolveQuickSetupSchedule(draft.schedulePreset)
  const recipients = normalizeRecipients(draft.emailRecipients)
  const normalizedTemplateName = String(draft.templateName || "").trim()
  const normalizedTemplateVersion =
    Number.isFinite(Number(draft.templateVersion)) && Number(draft.templateVersion) > 0
      ? Number(draft.templateVersion)
      : undefined
  const templateFormat =
    draft.templateFormat === "html" || draft.templateFormat === "md"
      ? draft.templateFormat
      : undefined
  const shouldAutoOutput = draft.schedulePreset !== "none"

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
              ...(normalizedTemplateName ? { template_name: normalizedTemplateName } : {}),
              ...(normalizedTemplateVersion ? { template_version: normalizedTemplateVersion } : {})
            }
          }
        : {}),
      template_name: normalizedTemplateName,
      template: {
        default_name: normalizedTemplateName,
        ...(templateFormat ? { default_format: templateFormat } : {}),
        default_version: normalizedTemplateVersion
      },
      generate_audio: draft.includeAudio,
      audio_voice: draft.includeAudio
        ? String(draft.audioVoice || "").trim() || undefined
        : undefined,
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
  const payload: WatchlistOutputCreate = {
    run_id: runId,
    item_ids: itemIds,
    type: "briefing_markdown",
    ...(templateFormat ? { format: templateFormat } : {}),
    template_name: String(draft.templateName || "").trim()
  }

  if (Number.isFinite(Number(draft.templateVersion)) && Number(draft.templateVersion) > 0) {
    payload.template_version = Number(draft.templateVersion)
  }

  if (draft.includeAudio) {
    payload.metadata = {
      audio: {
        enabled: true,
        voice: String(draft.audioVoice || "").trim() || null,
        target_minutes: Number(draft.targetAudioMinutes)
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
