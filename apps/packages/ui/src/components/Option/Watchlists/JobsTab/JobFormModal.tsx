import React, { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react"
import { Button, Collapse, Form, Input, InputNumber, Modal, Radio, Select, Switch, Tag, message } from "antd"
import { useTranslation } from "react-i18next"
import {
  createWatchlistJob,
  fetchWatchlistGroups,
  fetchJobOutputTemplates,
  fetchWatchlistSources,
  fetchWatchlistTemplates,
  previewWatchlistJob,
  testWatchlistAudioSettings,
  updateWatchlistJob
} from "@/services/watchlists"
import type {
  JobOutputPrefs,
  JobPreviewResult,
  JobScope,
  PreviewItem,
  WatchlistFilter,
  WatchlistJob,
  WatchlistJobCreate
} from "@/types/watchlists"
import {
  buildWatchlistsModalChrome,
  CronDisplay,
  useWatchlistsViewport,
  WatchlistsHelpTooltip
} from "../shared"
import { mapWatchlistsError } from "../shared/watchlists-error"
import {
  getFocusableActiveElement,
  restoreFocusToElement
} from "../shared/focus-management"
import {
  buildScopeTooltipLines,
  summarizeScopeCounts
} from "./job-summaries"
import { evaluatePreviewItems } from "./filter-preview"
import { ScopeSelector } from "./ScopeSelector"
import { FilterBuilder } from "./FilterBuilder"
import { SchedulePicker } from "./SchedulePicker"
import { findInvalidEmailRecipients } from "./email-utils"
import { analyzeScheduleFrequency, MIN_SCHEDULE_INTERVAL_MINUTES } from "./schedule-frequency"
import {
  durationToSeconds,
  secondsToDurationInput,
  type DurationInputValue,
  type DurationUnit
} from "./duration-utils"
import {
  trackWatchlistsPreventionTelemetry,
  type WatchlistsPreventionRule
} from "@/utils/watchlists-prevention-telemetry"

interface JobFormModalProps {
  open: boolean
  onClose: () => void
  onSuccess: () => void
  initialValues?: WatchlistJob
  watchlistId?: number | null
}

interface FormValues {
  name: string
  description: string
  active: boolean
}

type AuthoringMode = "basic" | "advanced"
type EmailBodyFormat = "auto" | "text" | "html"
type OutputFormat = "md" | "html"
type OutputPresetId = "briefing_md" | "newsletter_html" | "mece_md"
type BasicStepId = "scope" | "schedule" | "output" | "review"
const DEFAULT_AUDIO_VOICE = "alloy"
const DEFAULT_AUDIO_SPEED = 1
const DEFAULT_AUDIO_TARGET_MINUTES = 8
const AUDIO_SPEED_MIN = 0.25
const AUDIO_SPEED_MAX = 4
const AUDIO_TARGET_MINUTES_MIN = 1
const AUDIO_TARGET_MINUTES_MAX = 60
const AUDIO_TEST_SAMPLE_TEXT =
  "Top stories briefing sample. This is a quick audio check before saving your monitor."
const RETENTION_UNITS: DurationUnit[] = ["minutes", "hours", "days", "weeks", "seconds"]
const JOB_PREVIEW_LIMIT = 60
const JOB_PREVIEW_PER_SOURCE = 12
const JOB_SCOPE_CATALOG_PAGE_SIZE = 500
const BASIC_STEPS: BasicStepId[] = ["scope", "schedule", "output", "review"]

type ConfidenceRiskLevel = "blocking" | "warning"

interface ConfidenceRisk {
  id: string
  level: ConfidenceRiskLevel
  message: string
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value)

type WatchlistsValidationErrorDetail = {
  code?: string
  rule?: string
  message_key?: string
  message?: string
  remediation_key?: string
  remediation?: string
  meta?: Record<string, unknown>
}

const KNOWN_PREVENTION_RULES = new Set<WatchlistsPreventionRule>([
  "scope_required",
  "schedule_too_frequent",
  "invalid_email_recipients",
  "group_cycle_parent"
])

const asWatchlistsValidationErrorDetail = (
  candidate: unknown
): WatchlistsValidationErrorDetail | null => {
  if (!isRecord(candidate)) return null
  const detailCandidate = isRecord(candidate.detail) ? candidate.detail : candidate
  if (!isRecord(detailCandidate)) return null
  if (detailCandidate.code !== "watchlists_validation_error") return null
  return detailCandidate as WatchlistsValidationErrorDetail
}

const extractWatchlistsValidationErrorDetail = (
  error: unknown
): WatchlistsValidationErrorDetail | null => {
  const direct = asWatchlistsValidationErrorDetail(error)
  if (direct) return direct
  if (!isRecord(error)) return null
  const fromDetails = asWatchlistsValidationErrorDetail(error.details)
  if (fromDetails) return fromDetails
  const fromCause = asWatchlistsValidationErrorDetail(error.cause)
  if (fromCause) return fromCause
  if (isRecord(error.cause)) {
    return asWatchlistsValidationErrorDetail(error.cause.details)
  }
  return null
}

const toKnownPreventionRule = (
  rule: string | undefined
): WatchlistsPreventionRule | null => {
  if (!rule) return null
  return KNOWN_PREVENTION_RULES.has(rule as WatchlistsPreventionRule)
    ? (rule as WatchlistsPreventionRule)
    : null
}

const toOptionalNumber = (value: unknown): number | null => {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : null
}

const clampNumber = (value: number, min: number, max: number): number =>
  Math.min(max, Math.max(min, value))

const normalizeAudioSpeed = (value: unknown): number => {
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) return DEFAULT_AUDIO_SPEED
  return clampNumber(parsed, AUDIO_SPEED_MIN, AUDIO_SPEED_MAX)
}

const normalizeAudioTargetMinutes = (value: unknown): number => {
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) return DEFAULT_AUDIO_TARGET_MINUTES
  return Math.floor(
    clampNumber(parsed, AUDIO_TARGET_MINUTES_MIN, AUDIO_TARGET_MINUTES_MAX)
  )
}

const isValidBackgroundAudioUri = (value: string): boolean => {
  const trimmedValue = value.trim()
  if (trimmedValue.length === 0) return true
  try {
    const parsed = new URL(trimmedValue)
    return parsed.protocol === "https:" || parsed.protocol === "http:" || parsed.protocol === "file:"
  } catch {
    return false
  }
}

const normalizeAudioVoiceMap = (
  value: unknown
): { value: Record<string, string> | null; valid: boolean } => {
  if (!isRecord(value)) return { value: null, valid: false }
  const entries = Object.entries(value)
    .map(([marker, voice]) => [marker.trim(), String(voice || "").trim()] as const)
    .filter(([marker, voice]) => marker.length > 0 && voice.length > 0)
  if (entries.length === 0) {
    return { value: null, valid: true }
  }
  return {
    value: Object.fromEntries(entries),
    valid: true
  }
}

const cloneRecord = (value: Record<string, unknown>): Record<string, unknown> => {
  try {
    return JSON.parse(JSON.stringify(value)) as Record<string, unknown>
  } catch {
    return { ...value }
  }
}

const isEmailBodyFormat = (value: unknown): value is EmailBodyFormat =>
  value === "auto" || value === "text" || value === "html"

const isOutputFormat = (value: unknown): value is OutputFormat =>
  value === "md" || value === "html"

const OUTPUT_PRESETS: Record<
  OutputPresetId,
  {
    templateName: string
    format: OutputFormat
    emailEnabled: boolean
    emailBodyFormat: EmailBodyFormat
    createScheduledOutput: boolean
  }
> = {
  briefing_md: {
    templateName: "briefing_markdown",
    format: "md",
    emailEnabled: false,
    emailBodyFormat: "auto",
    createScheduledOutput: true,
  },
  newsletter_html: {
    templateName: "newsletter_html",
    format: "html",
    emailEnabled: true,
    emailBodyFormat: "html",
    createScheduledOutput: true,
  },
  mece_md: {
    templateName: "mece_markdown",
    format: "md",
    emailEnabled: false,
    emailBodyFormat: "auto",
    createScheduledOutput: true,
  },
}

export const JobFormModal: React.FC<JobFormModalProps> = ({
  open,
  onClose,
  onSuccess,
  initialValues,
  watchlistId
}) => {
  const { t } = useTranslation(["watchlists", "common"])
  const [form] = Form.useForm<FormValues>()
  const [submitting, setSubmitting] = useState(false)

  const isEditing = !!initialValues
  const restoreFocusTargetRef = useRef<HTMLElement | null>(null)
  const wasOpenRef = useRef(false)
  const { isConstrained } = useWatchlistsViewport()
  const modalChrome = buildWatchlistsModalChrome(isConstrained, 700, {
    maxHeight: "70vh",
    overflowY: "auto"
  })

  useLayoutEffect(() => {
    if (open) {
      if (!wasOpenRef.current) {
        restoreFocusTargetRef.current = getFocusableActiveElement()
      }
      wasOpenRef.current = true
      return
    }

    if (wasOpenRef.current) {
      wasOpenRef.current = false
      restoreFocusToElement(restoreFocusTargetRef.current)
    }
  }, [open])

  // Managed state for complex fields
  const [scope, setScope] = useState<JobScope>({})
  const [authoringMode, setAuthoringMode] = useState<AuthoringMode>("basic")
  const [basicStep, setBasicStep] = useState<BasicStepId>("scope")
  const [filters, setFilters] = useState<WatchlistFilter[]>([])
  const [schedule, setSchedule] = useState<string | null>(null)
  const [timezone, setTimezone] = useState("UTC")
  const [templateOptions, setTemplateOptions] = useState<Array<{ label: string; value: string }>>([])
  const [outputPreset, setOutputPreset] = useState<OutputPresetId | undefined>(undefined)
  const [outputTemplateName, setOutputTemplateName] = useState<string | undefined>(undefined)
  const [outputTemplateVersion, setOutputTemplateVersion] = useState<number | null>(null)
  const [outputTemplateFormat, setOutputTemplateFormat] = useState<OutputFormat | undefined>(undefined)
  const [createScheduledOutput, setCreateScheduledOutput] = useState(false)
  const [retentionDefaultDuration, setRetentionDefaultDuration] = useState<DurationInputValue>({
    value: null,
    unit: "days"
  })
  const [retentionTemporaryDuration, setRetentionTemporaryDuration] = useState<DurationInputValue>({
    value: null,
    unit: "days"
  })
  const [deliveryEmailEnabled, setDeliveryEmailEnabled] = useState(false)
  const [deliveryEmailRecipients, setDeliveryEmailRecipients] = useState<string[]>([])
  const [deliveryEmailSubject, setDeliveryEmailSubject] = useState("")
  const [deliveryEmailAttachFile, setDeliveryEmailAttachFile] = useState(true)
  const [deliveryEmailBodyFormat, setDeliveryEmailBodyFormat] = useState<EmailBodyFormat>("auto")
  const [deliveryChatbookEnabled, setDeliveryChatbookEnabled] = useState(false)
  const [deliveryChatbookTitle, setDeliveryChatbookTitle] = useState("")
  const [deliveryChatbookDescription, setDeliveryChatbookDescription] = useState("")
  const [deliveryChatbookConversationId, setDeliveryChatbookConversationId] = useState<number | null>(null)
  const [audioBriefingEnabled, setAudioBriefingEnabled] = useState(false)
  const [audioVoice, setAudioVoice] = useState(DEFAULT_AUDIO_VOICE)
  const [audioSpeed, setAudioSpeed] = useState<number>(DEFAULT_AUDIO_SPEED)
  const [audioTargetMinutes, setAudioTargetMinutes] = useState<number>(DEFAULT_AUDIO_TARGET_MINUTES)
  const [showAdvancedAudioOptions, setShowAdvancedAudioOptions] = useState(false)
  const [audioBackgroundUri, setAudioBackgroundUri] = useState("")
  const [audioVoiceMapText, setAudioVoiceMapText] = useState("")
  const [scopeSourceNamesById, setScopeSourceNamesById] = useState<Record<number, string>>({})
  const [scopeGroupNamesById, setScopeGroupNamesById] = useState<Record<number, string>>({})
  const [previewCandidates, setPreviewCandidates] = useState<PreviewItem[]>([])
  const [previewLoading, setPreviewLoading] = useState(false)
  const [previewError, setPreviewError] = useState<string | null>(null)
  const [audioTestLoading, setAudioTestLoading] = useState(false)
  const [audioTestError, setAudioTestError] = useState<string | null>(null)
  const [audioTestUrl, setAudioTestUrl] = useState<string | null>(null)
  const authoringContext = isEditing ? "edit" : "create"

  const watchedName = Form.useWatch("name", form)

  const applyOutputPrefsState = (prefs: JobOutputPrefs | null | undefined) => {
    const prefsRecord = isRecord(prefs) ? prefs : {}
    const templateRecord = isRecord(prefsRecord.template) ? prefsRecord.template : {}
    const retentionRecord = isRecord(prefsRecord.retention) ? prefsRecord.retention : {}
    const deliveriesRecord = isRecord(prefsRecord.deliveries) ? prefsRecord.deliveries : {}
    const emailRecord = isRecord(deliveriesRecord.email) ? deliveriesRecord.email : null
    const chatbookRecord = isRecord(deliveriesRecord.chatbook) ? deliveriesRecord.chatbook : null
    const autoOutputRecord = isRecord(prefsRecord.auto_output) ? prefsRecord.auto_output : null

    setOutputTemplateName(
      typeof templateRecord.default_name === "string" && templateRecord.default_name.trim().length > 0
        ? templateRecord.default_name
        : undefined
    )

    const parsedTemplateVersion = Number(templateRecord.default_version)
    setOutputTemplateVersion(
      Number.isFinite(parsedTemplateVersion) && parsedTemplateVersion > 0
        ? Math.floor(parsedTemplateVersion)
        : null
    )
    setOutputTemplateFormat(
      isOutputFormat(templateRecord.default_format) ? templateRecord.default_format : undefined
    )
    setCreateScheduledOutput(Boolean(autoOutputRecord) && autoOutputRecord.enabled === true)
    const parsedDefaultRetention = Number(retentionRecord.default_seconds)
    setRetentionDefaultDuration(
      secondsToDurationInput(
        Number.isFinite(parsedDefaultRetention) && parsedDefaultRetention >= 0
          ? Math.floor(parsedDefaultRetention)
          : null
      )
    )
    const parsedTemporaryRetention = Number(retentionRecord.temporary_seconds)
    setRetentionTemporaryDuration(
      secondsToDurationInput(
        Number.isFinite(parsedTemporaryRetention) && parsedTemporaryRetention >= 0
          ? Math.floor(parsedTemporaryRetention)
          : null
      )
    )

    setDeliveryEmailEnabled(Boolean(emailRecord) && emailRecord.enabled !== false)
    setDeliveryEmailRecipients(
      Array.isArray(emailRecord?.recipients)
        ? emailRecord.recipients
          .filter((entry): entry is string => typeof entry === "string")
          .map((entry) => entry.trim())
          .filter((entry) => entry.length > 0)
        : []
    )
    setDeliveryEmailSubject(
      typeof emailRecord?.subject === "string" ? emailRecord.subject : ""
    )
    setDeliveryEmailAttachFile(
      emailRecord?.attach_file === undefined ? true : Boolean(emailRecord.attach_file)
    )
    setDeliveryEmailBodyFormat(
      isEmailBodyFormat(emailRecord?.body_format) ? emailRecord.body_format : "auto"
    )

    setDeliveryChatbookEnabled(Boolean(chatbookRecord) && chatbookRecord.enabled !== false)
    setDeliveryChatbookTitle(
      typeof chatbookRecord?.title === "string" ? chatbookRecord.title : ""
    )
    setDeliveryChatbookDescription(
      typeof chatbookRecord?.description === "string" ? chatbookRecord.description : ""
    )
    const parsedConversationId = Number(chatbookRecord?.conversation_id)
    setDeliveryChatbookConversationId(
      Number.isFinite(parsedConversationId) && parsedConversationId > 0
        ? Math.floor(parsedConversationId)
        : null
    )

    setAudioBriefingEnabled(Boolean(prefsRecord.generate_audio))
    setAudioVoice(
      typeof prefsRecord.audio_voice === "string" && prefsRecord.audio_voice.trim().length > 0
        ? prefsRecord.audio_voice.trim()
        : DEFAULT_AUDIO_VOICE
    )
    setAudioSpeed(normalizeAudioSpeed(prefsRecord.audio_speed))
    setAudioTargetMinutes(normalizeAudioTargetMinutes(prefsRecord.target_audio_minutes))
    const normalizedBackgroundUri =
      typeof prefsRecord.background_audio_uri === "string"
        ? prefsRecord.background_audio_uri.trim()
        : ""
    setAudioBackgroundUri(normalizedBackgroundUri)
    const voiceMapNormalization = normalizeAudioVoiceMap(prefsRecord.voice_map)
    setAudioVoiceMapText(
      voiceMapNormalization.value
        ? JSON.stringify(voiceMapNormalization.value, null, 2)
        : ""
    )
    setShowAdvancedAudioOptions(
      normalizedBackgroundUri.length > 0 || Boolean(voiceMapNormalization.value)
    )
  }

  const buildOutputPrefs = (
    options?: { audioVoiceMap?: Record<string, string> | null }
  ): JobOutputPrefs | undefined => {
    const basePrefs = (
      isEditing &&
      initialValues?.output_prefs &&
      isRecord(initialValues.output_prefs)
    )
      ? cloneRecord(initialValues.output_prefs)
      : {}

    const templatePrefs = isRecord(basePrefs.template) ? { ...basePrefs.template } : {}
    const normalizedTemplateName = outputTemplateName?.trim() || ""
    if (normalizedTemplateName) {
      templatePrefs.default_name = normalizedTemplateName
    } else {
      delete templatePrefs.default_name
    }
    if (typeof outputTemplateVersion === "number" && outputTemplateVersion > 0) {
      templatePrefs.default_version = Math.floor(outputTemplateVersion)
    } else {
      delete templatePrefs.default_version
    }
    if (isOutputFormat(outputTemplateFormat)) {
      templatePrefs.default_format = outputTemplateFormat
    } else {
      delete templatePrefs.default_format
    }
    if (Object.keys(templatePrefs).length > 0) {
      basePrefs.template = templatePrefs
    } else {
      delete basePrefs.template
    }

    const retentionPrefs = isRecord(basePrefs.retention) ? { ...basePrefs.retention } : {}
    const defaultRetentionSeconds = durationToSeconds(retentionDefaultDuration)
    if (typeof defaultRetentionSeconds === "number" && defaultRetentionSeconds >= 0) {
      retentionPrefs.default_seconds = defaultRetentionSeconds
    } else {
      delete retentionPrefs.default_seconds
    }
    const temporaryRetentionSeconds = durationToSeconds(retentionTemporaryDuration)
    if (typeof temporaryRetentionSeconds === "number" && temporaryRetentionSeconds >= 0) {
      retentionPrefs.temporary_seconds = temporaryRetentionSeconds
    } else {
      delete retentionPrefs.temporary_seconds
    }
    if (Object.keys(retentionPrefs).length > 0) {
      basePrefs.retention = retentionPrefs
    } else {
      delete basePrefs.retention
    }

    const deliveriesPrefs = isRecord(basePrefs.deliveries) ? { ...basePrefs.deliveries } : {}

    const normalizedEmailSubject = deliveryEmailSubject.trim()
    const shouldPersistEmail = (
      deliveryEmailEnabled ||
      deliveryEmailRecipients.length > 0 ||
      normalizedEmailSubject.length > 0
    )
    if (shouldPersistEmail) {
      const emailPrefs = isRecord(deliveriesPrefs.email) ? { ...deliveriesPrefs.email } : {}
      emailPrefs.enabled = deliveryEmailEnabled
      if (deliveryEmailRecipients.length > 0) {
        emailPrefs.recipients = deliveryEmailRecipients
      } else {
        delete emailPrefs.recipients
      }
      emailPrefs.attach_file = deliveryEmailAttachFile
      emailPrefs.body_format = deliveryEmailBodyFormat
      if (normalizedEmailSubject.length > 0) {
        emailPrefs.subject = normalizedEmailSubject
      } else {
        delete emailPrefs.subject
      }
      deliveriesPrefs.email = emailPrefs
    } else {
      delete deliveriesPrefs.email
    }

    const shouldPersistChatbook = (
      deliveryChatbookEnabled ||
      deliveryChatbookTitle.trim().length > 0 ||
      deliveryChatbookDescription.trim().length > 0 ||
      deliveryChatbookConversationId !== null
    )
    if (shouldPersistChatbook) {
      const chatbookPrefs = isRecord(deliveriesPrefs.chatbook) ? { ...deliveriesPrefs.chatbook } : {}
      chatbookPrefs.enabled = deliveryChatbookEnabled
      if (deliveryChatbookTitle.trim().length > 0) {
        chatbookPrefs.title = deliveryChatbookTitle.trim()
      } else {
        delete chatbookPrefs.title
      }
      if (deliveryChatbookDescription.trim().length > 0) {
        chatbookPrefs.description = deliveryChatbookDescription.trim()
      } else {
        delete chatbookPrefs.description
      }
      if (typeof deliveryChatbookConversationId === "number" && deliveryChatbookConversationId > 0) {
        chatbookPrefs.conversation_id = Math.floor(deliveryChatbookConversationId)
      } else {
        delete chatbookPrefs.conversation_id
      }
      deliveriesPrefs.chatbook = chatbookPrefs
    } else {
      delete deliveriesPrefs.chatbook
    }

    if (Object.keys(deliveriesPrefs).length > 0) {
      basePrefs.deliveries = deliveriesPrefs
    } else {
      delete basePrefs.deliveries
    }

    if (audioBriefingEnabled) {
      basePrefs.generate_audio = true
      basePrefs.audio_voice = audioVoice.trim() || DEFAULT_AUDIO_VOICE
      basePrefs.audio_speed = normalizeAudioSpeed(audioSpeed)
      basePrefs.target_audio_minutes = normalizeAudioTargetMinutes(audioTargetMinutes)
      if (audioBackgroundUri.trim().length > 0) {
        basePrefs.background_audio_uri = audioBackgroundUri.trim()
      } else {
        delete basePrefs.background_audio_uri
      }
      if (options?.audioVoiceMap && Object.keys(options.audioVoiceMap).length > 0) {
        basePrefs.voice_map = options.audioVoiceMap
      } else {
        delete basePrefs.voice_map
      }
    } else {
      delete basePrefs.audio_voice
      delete basePrefs.audio_speed
      delete basePrefs.target_audio_minutes
      delete basePrefs.background_audio_uri
      delete basePrefs.voice_map
      if ("generate_audio" in basePrefs) {
        basePrefs.generate_audio = false
      }
    }

    const hasRecurringSchedule = Boolean(schedule && schedule.trim().length > 0)
    const existingAutoOutput = isRecord(basePrefs.auto_output)
      ? { ...basePrefs.auto_output }
      : {}
    const shouldAutoOutput = (
      hasRecurringSchedule &&
      createScheduledOutput
    )
    if (shouldAutoOutput) {
      existingAutoOutput.enabled = true
      existingAutoOutput.type =
        typeof existingAutoOutput.type === "string" && existingAutoOutput.type.trim().length > 0
          ? existingAutoOutput.type.trim()
          : "briefing_markdown"
      if (isOutputFormat(outputTemplateFormat)) {
        existingAutoOutput.format = outputTemplateFormat
      } else {
        delete existingAutoOutput.format
      }
      if (normalizedTemplateName) {
        existingAutoOutput.template_name = normalizedTemplateName
      } else {
        delete existingAutoOutput.template_name
      }
      if (typeof outputTemplateVersion === "number" && outputTemplateVersion > 0) {
        existingAutoOutput.template_version = Math.floor(outputTemplateVersion)
      } else {
        delete existingAutoOutput.template_version
      }
      basePrefs.auto_output = existingAutoOutput
    } else {
      delete basePrefs.auto_output
    }

    if (Object.keys(basePrefs).length > 0) {
      return basePrefs as JobOutputPrefs
    }
    if (isEditing) {
      return {}
    }
    return undefined
  }

  const applyPreset = (presetId: OutputPresetId | undefined) => {
    if (!presetId) return
    const preset = OUTPUT_PRESETS[presetId]
    if (!preset) return
    setOutputTemplateName(preset.templateName)
    setOutputTemplateVersion(null)
    setOutputTemplateFormat(preset.format)
    setDeliveryEmailEnabled(preset.emailEnabled)
    setDeliveryEmailBodyFormat(preset.emailBodyFormat)
    setCreateScheduledOutput(preset.createScheduledOutput)
    setOutputPreset(presetId)
    message.success(t("watchlists:jobs.form.presetApplied", "Preset applied"))
  }

  const clearAudioTestPreview = () => {
    setAudioTestError(null)
    setAudioTestLoading(false)
    setAudioTestUrl((previous) => {
      if (previous && typeof URL !== "undefined" && typeof URL.revokeObjectURL === "function") {
        URL.revokeObjectURL(previous)
      }
      return null
    })
  }

  const parseAudioVoiceMapForValidation = (): Record<string, string> | null | undefined => {
    const trimmedAudioVoiceMap = audioVoiceMapText.trim()
    if (trimmedAudioVoiceMap.length === 0) return null
    try {
      const parsedVoiceMap = JSON.parse(trimmedAudioVoiceMap)
      const normalized = normalizeAudioVoiceMap(parsedVoiceMap)
      if (!normalized.valid) {
        message.error(
          t(
            "watchlists:jobs.form.audioVoiceMapInvalid",
            "Voice map must be valid JSON with marker-to-voice string pairs."
          )
        )
        return undefined
      }
      return normalized.value
    } catch {
      message.error(
        t(
          "watchlists:jobs.form.audioVoiceMapInvalid",
          "Voice map must be valid JSON with marker-to-voice string pairs."
        )
      )
      return undefined
    }
  }

  const handleAudioBriefingToggle = (enabled: boolean) => {
    setAudioBriefingEnabled(enabled)
    if (!enabled) {
      clearAudioTestPreview()
    } else {
      setAudioTestError(null)
    }
  }

  const handleTestAudioSettings = async () => {
    if (!audioBriefingEnabled) return

    const trimmedBackgroundUri = audioBackgroundUri.trim()
    if (!isValidBackgroundAudioUri(trimmedBackgroundUri)) {
      message.error(
        t(
          "watchlists:jobs.form.audioBackgroundTrackInvalid",
          "Background track must start with https://, http://, or file://."
        )
      )
      return
    }

    const parsedVoiceMap = parseAudioVoiceMapForValidation()
    if (parsedVoiceMap === undefined) return

    setAudioTestLoading(true)
    setAudioTestError(null)

    try {
      const audioBuffer = await testWatchlistAudioSettings({
        text: t("watchlists:jobs.form.audioTestSampleText", AUDIO_TEST_SAMPLE_TEXT),
        voice: audioVoice.trim() || DEFAULT_AUDIO_VOICE,
        speed: normalizeAudioSpeed(audioSpeed),
        response_format: "mp3"
      })
      if (!(audioBuffer instanceof ArrayBuffer) || audioBuffer.byteLength === 0) {
        throw new Error("Audio preview did not return playable data.")
      }
      if (typeof URL === "undefined" || typeof URL.createObjectURL !== "function") {
        throw new Error("Audio preview playback is unavailable in this environment.")
      }
      const audioBlob = new Blob([audioBuffer], { type: "audio/mpeg" })
      const nextUrl = URL.createObjectURL(audioBlob)
      setAudioTestUrl((previous) => {
        if (previous && typeof URL.revokeObjectURL === "function") {
          URL.revokeObjectURL(previous)
        }
        return nextUrl
      })
      if (parsedVoiceMap && Object.keys(parsedVoiceMap).length > 0) {
        setShowAdvancedAudioOptions(true)
      }
    } catch (err) {
      const mapped = mapWatchlistsError(err, {
        t,
        context: t("watchlists:jobs.form.audioTestContext", "audio sample"),
        fallbackMessage: t(
          "watchlists:jobs.form.audioTestError",
          "Could not generate audio sample. Check voice/speed settings and try again."
        ),
        operationLabel: "generate"
      })
      setAudioTestError(`${mapped.title} ${mapped.description}`.trim())
      setAudioTestUrl((previous) => {
        if (previous && typeof URL !== "undefined" && typeof URL.revokeObjectURL === "function") {
          URL.revokeObjectURL(previous)
        }
        return null
      })
    } finally {
      setAudioTestLoading(false)
    }
  }

  // Reset form when modal opens/closes or initialValues change
  useEffect(() => {
    if (open) {
      setAudioTestError(null)
      setAudioTestLoading(false)
      setAudioTestUrl((previous) => {
        if (previous && typeof URL !== "undefined" && typeof URL.revokeObjectURL === "function") {
          URL.revokeObjectURL(previous)
        }
        return null
      })
      if (initialValues) {
        form.setFieldsValue({
          name: initialValues.name,
          description: initialValues.description || "",
          active: initialValues.active
        })
        setBasicStep("scope")
        setAuthoringMode("advanced")
        void trackWatchlistsPreventionTelemetry({
          type: "watchlists_authoring_started",
          surface: "job_form",
          mode: "advanced",
          context: "edit"
        })
        setScope(initialValues.scope || {})
        setFilters(initialValues.job_filters?.filters || [])
        setSchedule(initialValues.schedule_expr || null)
        setTimezone(initialValues.timezone || "UTC")
        setOutputPreset(undefined)
        applyOutputPrefsState(initialValues.output_prefs)
      } else {
        form.resetFields()
        form.setFieldsValue({
          name: "",
          description: "",
          active: true
        })
        setBasicStep("scope")
        setAuthoringMode("basic")
        void trackWatchlistsPreventionTelemetry({
          type: "watchlists_authoring_started",
          surface: "job_form",
          mode: "basic",
          context: "create"
        })
        setScope({})
        setFilters([])
        setSchedule(null)
        setTimezone("UTC")
        setOutputPreset(undefined)
        applyOutputPrefsState(null)
      }
    }
  }, [open, initialValues, form])

  useEffect(() => {
    return () => {
      if (audioTestUrl && typeof URL !== "undefined" && typeof URL.revokeObjectURL === "function") {
        URL.revokeObjectURL(audioTestUrl)
      }
    }
  }, [audioTestUrl])

  useEffect(() => {
    if (!open) {
      setScopeSourceNamesById({})
      setScopeGroupNamesById({})
      return
    }
    let cancelled = false

    Promise.all([
      fetchWatchlistSources({
        watchlist_id: watchlistId ?? undefined,
        page: 1,
        size: JOB_SCOPE_CATALOG_PAGE_SIZE
      }),
      fetchWatchlistGroups({ page: 1, size: JOB_SCOPE_CATALOG_PAGE_SIZE })
    ])
      .then(([sourcesResult, groupsResult]) => {
        if (cancelled) return
        const nextSources: Record<number, string> = {}
        const nextGroups: Record<number, string> = {}
        for (const source of sourcesResult.items || []) {
          nextSources[source.id] = source.name
        }
        for (const group of groupsResult.items || []) {
          nextGroups[group.id] = group.name
        }
        setScopeSourceNamesById(nextSources)
        setScopeGroupNamesById(nextGroups)
      })
      .catch((err) => {
        console.warn("Failed to load monitor scope summary catalog:", err)
      })

    return () => {
      cancelled = true
    }
  }, [open, watchlistId])

  useEffect(() => {
    if (!open || !isEditing || !initialValues?.id) {
      setPreviewCandidates([])
      setPreviewError(null)
      setPreviewLoading(false)
      return
    }
    let cancelled = false
    setPreviewLoading(true)
    setPreviewError(null)

    previewWatchlistJob(initialValues.id, {
      limit: JOB_PREVIEW_LIMIT,
      per_source: JOB_PREVIEW_PER_SOURCE
    })
      .then((result: JobPreviewResult) => {
        if (cancelled) return
        setPreviewCandidates(Array.isArray(result.items) ? result.items : [])
      })
      .catch((err) => {
        if (cancelled) return
        console.error("Failed to load monitor preview candidates:", err)
        setPreviewCandidates([])
        const mapped = mapWatchlistsError(err, {
          t,
          context: t("watchlists:jobs.form.previewContext", "monitor preview"),
          fallbackMessage: t(
            "watchlists:jobs.form.previewLoadError",
            "Could not load sample candidates for this monitor."
          ),
          operationLabel: t("watchlists:errors.operation.load", "load")
        })
        setPreviewError(`${mapped.title} ${mapped.description}`.trim())
      })
      .finally(() => {
        if (!cancelled) {
          setPreviewLoading(false)
        }
      })

    return () => {
      cancelled = true
    }
  }, [initialValues?.id, isEditing, open, t])

  useEffect(() => {
    if (!open) return
    let cancelled = false
    const loadTemplates = async () => {
      const [outputsResult, legacyResult] = await Promise.allSettled([
        fetchJobOutputTemplates({ limit: 200, offset: 0 }),
        fetchWatchlistTemplates()
      ])
      if (cancelled) return

      const optionsByName = new Map<string, { label: string; value: string }>()

      if (outputsResult.status === "fulfilled") {
        for (const item of outputsResult.value.items) {
          const normalizedName = item.name.trim()
          if (!normalizedName) continue
          const formatSuffix = item.format ? ` (${String(item.format).toUpperCase()})` : ""
          optionsByName.set(normalizedName, {
            label: `${normalizedName}${formatSuffix} · ${t("watchlists:jobs.form.templateSource.outputs", "Outputs template")}`,
            value: normalizedName
          })
        }
      }

      if (legacyResult.status === "fulfilled") {
        const items = Array.isArray(legacyResult.value.items) ? legacyResult.value.items : []
        for (const item of items) {
          const normalizedName = item.name.trim()
          if (!normalizedName || optionsByName.has(normalizedName)) continue
          optionsByName.set(normalizedName, {
            label: `${normalizedName} · ${t("watchlists:jobs.form.templateSource.legacy", "Legacy watchlists template")}`,
            value: normalizedName
          })
        }
      }

      if (outputsResult.status === "rejected" && legacyResult.status === "rejected") {
        console.error("Failed to load output template options for job form", {
          outputs: outputsResult.reason,
          legacy: legacyResult.reason
        })
      }

      setTemplateOptions(
        Array.from(optionsByName.values()).sort((a, b) => a.label.localeCompare(b.label))
      )
    }

    void loadTemplates().catch((err) => {
      console.error("Failed to load output template options for job form:", err)
      if (!cancelled) {
        setTemplateOptions([])
      }
    })
    return () => {
      cancelled = true
    }
  }, [open, t])

  useEffect(() => {
    const normalizedName = outputTemplateName?.trim()
    if (!normalizedName) return
    setTemplateOptions((previous) => {
      if (previous.some((option) => option.value === normalizedName)) {
        return previous
      }
      return [
        ...previous,
        {
          label: `${normalizedName} · ${t("watchlists:jobs.form.templateSource.current", "Current selection")}`,
          value: normalizedName
        }
      ]
    })
  }, [outputTemplateName, t])

  const handleSubmit = async () => {
    try {
      const values = await form.validateFields()

      // Validate scope
      const hasScope =
        (scope.sources?.length ?? 0) > 0 ||
        (scope.groups?.length ?? 0) > 0 ||
        (scope.tags?.length ?? 0) > 0

      if (!hasScope) {
        void trackWatchlistsPreventionTelemetry({
          type: "watchlists_validation_blocked",
          surface: "job_form",
          rule: "scope_required",
          remediation: "select_scope"
        })
        message.error(t("watchlists:jobs.form.scopeRequired", "Please select at least one feed, group, or tag"))
        return
      }

      const scheduleFrequency = analyzeScheduleFrequency(schedule, MIN_SCHEDULE_INTERVAL_MINUTES)
      if (scheduleFrequency.tooFrequent) {
        void trackWatchlistsPreventionTelemetry({
          type: "watchlists_validation_blocked",
          surface: "job_form",
          rule: "schedule_too_frequent",
          remediation: "increase_interval",
          minutes: MIN_SCHEDULE_INTERVAL_MINUTES
        })
        message.error(
          t(
            "watchlists:jobs.form.scheduleTooFrequent",
            "Schedule is too frequent. Minimum interval is every {{minutes}} minutes.",
            { minutes: MIN_SCHEDULE_INTERVAL_MINUTES }
          )
        )
        return
      }

      if (invalidEmailRecipients.length > 0) {
        void trackWatchlistsPreventionTelemetry({
          type: "watchlists_validation_blocked",
          surface: "job_form",
          rule: "invalid_email_recipients",
          remediation: "fix_recipients",
          count: invalidEmailRecipients.length
        })
        message.error(
          t(
            "watchlists:jobs.form.emailRecipientsInvalidSubmit",
            "Fix invalid email recipients before saving."
          )
        )
        return
      }

      if (audioBriefingEnabled && !isValidBackgroundAudioUri(audioBackgroundUri)) {
        message.error(
          t(
            "watchlists:jobs.form.audioBackgroundTrackInvalid",
            "Background track must start with https://, http://, or file://."
          )
        )
        return
      }

      let parsedAudioVoiceMap: Record<string, string> | null = null
      if (audioBriefingEnabled) {
        const parsedAudioVoiceMapResult = parseAudioVoiceMapForValidation()
        if (parsedAudioVoiceMapResult === undefined) {
          return
        }
        parsedAudioVoiceMap = parsedAudioVoiceMapResult
      }

      const outputPrefs = buildOutputPrefs({ audioVoiceMap: parsedAudioVoiceMap })

      const jobData: WatchlistJobCreate = {
        name: values.name,
        description: values.description || undefined,
        active: values.active,
        scope,
        schedule_expr: schedule || undefined,
        timezone: timezone || undefined,
        output_prefs: isEditing ? (outputPrefs || {}) : outputPrefs,
        job_filters: filters.length > 0 ? { filters } : undefined,
        watchlist_id: watchlistId ?? undefined
      }

      const confirmationItems: string[] = []
      if (createScheduledOutput && deliveryEmailEnabled) {
        confirmationItems.push(
          t(
            "watchlists:jobs.form.confirmationEmail",
            "Email delivery will send each run to {{count}} recipient{{plural}}.",
            {
              count: deliveryEmailRecipients.length,
              plural: deliveryEmailRecipients.length === 1 ? "" : "s"
            }
          )
        )
      }
      if (createScheduledOutput && deliveryChatbookEnabled) {
        confirmationItems.push(
          t(
            "watchlists:jobs.form.confirmationChatbook",
            "Chatbook delivery will publish output artifacts on each run."
          )
        )
      }
      if (audioBriefingEnabled) {
        confirmationItems.push(
          t(
            "watchlists:jobs.form.confirmationAudio",
            "Audio briefing will generate on each run ({{voice}}, {{minutes}} min target).",
            {
              voice: audioVoice,
              minutes: audioTargetMinutes
            }
          )
        )
      }

      if (confirmationItems.length > 0) {
        const confirmed = await new Promise<boolean>((resolve) => {
          Modal.confirm({
            title: t(
              "watchlists:jobs.form.confirmationTitle",
              "Confirm recurring delivery settings"
            ),
            content: (
              <div className="space-y-2">
                <div className="text-sm text-text-muted">
                  {t(
                    "watchlists:jobs.form.confirmationDescription",
                    "Review these recurring actions before saving this monitor."
                  )}
                </div>
                <ul className="list-disc pl-5 space-y-1">
                  {confirmationItems.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              </div>
            ),
            okText: isEditing ? t("common:save", "Save") : t("common:create", "Create"),
            cancelText: t("common:cancel", "Cancel"),
            onOk: () => resolve(true),
            onCancel: () => resolve(false)
          })
        })
        if (!confirmed) return
      }

      setSubmitting(true)

      if (isEditing && initialValues) {
        await updateWatchlistJob(initialValues.id, jobData)
        message.success(t("watchlists:jobs.updated", "Monitor updated"))
      } else {
        await createWatchlistJob(jobData)
        message.success(t("watchlists:jobs.created", "Monitor created"))
      }
      if (authoringMode === "basic") {
        void trackWatchlistsPreventionTelemetry({
          type: "watchlists_basic_step_completed",
          surface: "job_form",
          step: "review"
        })
      }
      void trackWatchlistsPreventionTelemetry({
        type: "watchlists_authoring_saved",
        surface: "job_form",
        mode: authoringMode,
        context: authoringContext
      })

      onSuccess()
    } catch (err) {
      if (err && typeof err === "object" && "errorFields" in err) {
        // Validation error - handled by form
        return
      }
      const validationDetail = extractWatchlistsValidationErrorDetail(err)
      if (validationDetail) {
        const meta = isRecord(validationDetail.meta) ? validationDetail.meta : {}
        const minimumMinutes = toOptionalNumber(meta.minimum_minutes)
        const tValues =
          minimumMinutes != null
            ? { minutes: Math.max(1, Math.round(minimumMinutes)) }
            : undefined
        const localizedMessage = validationDetail.message_key
          ? t(
            validationDetail.message_key,
            validationDetail.message || "Validation failed.",
            tValues
          )
          : validationDetail.message || t("watchlists:jobs.saveError", "Failed to save monitor")
        const localizedRemediation = validationDetail.remediation_key
          ? t(validationDetail.remediation_key, validationDetail.remediation || "", tValues)
          : validationDetail.remediation || ""
        const knownRule = toKnownPreventionRule(validationDetail.rule)
        if (knownRule) {
          void trackWatchlistsPreventionTelemetry({
            type: "watchlists_validation_blocked",
            surface: "job_form",
            rule: knownRule,
            remediation: "server_validation_error"
          })
        }
        const combinedMessage = [localizedMessage, localizedRemediation]
          .map((part) => String(part || "").trim())
          .filter((part) => part.length > 0)
          .join(" ")
        message.error(combinedMessage || t("watchlists:jobs.saveError", "Failed to save monitor"))
        return
      }
      console.error("Form submit error:", err)
      const mapped = mapWatchlistsError(err, {
        t,
        context: t("watchlists:jobs.form.saveContext", "monitor"),
        fallbackMessage: t("watchlists:jobs.saveError", "Failed to save monitor"),
        operationLabel: t("watchlists:errors.operation.save", "save")
      })
      message.error(`${mapped.title} ${mapped.description}`.trim())
    } finally {
      setSubmitting(false)
    }
  }

  const handleCancel = () => {
    form.resetFields()
    onClose()
  }

  const retentionUnitOptions = RETENTION_UNITS.map((unit) => ({
    value: unit,
    label: t(
      `watchlists:jobs.form.retentionUnit.${unit}`,
      unit.charAt(0).toUpperCase() + unit.slice(1)
    )
  }))

  const retentionDefaultSeconds = durationToSeconds(retentionDefaultDuration)
  const retentionTemporarySeconds = durationToSeconds(retentionTemporaryDuration)
  const invalidEmailRecipients = findInvalidEmailRecipients(deliveryEmailRecipients)
  const hasAdvancedConfiguration =
    filters.length > 0 ||
    retentionDefaultDuration.value !== null ||
    retentionTemporaryDuration.value !== null ||
    createScheduledOutput ||
    deliveryEmailEnabled ||
    deliveryEmailRecipients.length > 0 ||
    deliveryEmailSubject.trim().length > 0 ||
    deliveryEmailAttachFile !== true ||
    deliveryEmailBodyFormat !== "auto" ||
    deliveryChatbookEnabled ||
    deliveryChatbookTitle.trim().length > 0 ||
    deliveryChatbookDescription.trim().length > 0 ||
    deliveryChatbookConversationId !== null ||
    audioBackgroundUri.trim().length > 0 ||
    audioVoiceMapText.trim().length > 0
  const hasScopeSelection =
    (scope.sources?.length ?? 0) > 0 ||
    (scope.groups?.length ?? 0) > 0 ||
    (scope.tags?.length ?? 0) > 0
  const hasScheduleSelection = Boolean(schedule && schedule.trim().length > 0)
  const basicStepIndex = Math.max(0, BASIC_STEPS.indexOf(basicStep))
  const watchedNameValue = String(watchedName || "").trim()

  const confidenceChecks = [
    {
      id: "name",
      passed: watchedNameValue.length > 0,
    },
    {
      id: "scope",
      passed: hasScopeSelection,
    },
    {
      id: "schedule",
      passed: hasScheduleSelection,
    },
    {
      id: "delivery",
      passed: invalidEmailRecipients.length === 0,
    },
  ]
  const confidenceCompletedChecks = confidenceChecks.filter((check) => check.passed).length
  const confidenceRisks: ConfidenceRisk[] = []

  if (watchedNameValue.length === 0) {
    confidenceRisks.push({
      id: "name",
      level: "blocking",
      message: t("watchlists:jobs.form.confidenceRiskName", "Add a monitor name.")
    })
  }

  if (!hasScopeSelection) {
    confidenceRisks.push({
      id: "scope",
      level: "blocking",
      message: t(
        "watchlists:jobs.form.confidenceRiskScope",
        "Select at least one feed, group, or tag."
      )
    })
  }

  if (!hasScheduleSelection) {
    confidenceRisks.push({
      id: "schedule",
      level: "warning",
      message: t(
        "watchlists:jobs.form.confidenceRiskSchedule",
        "Schedule is not set; this monitor will only run manually."
      )
    })
  }

  if (invalidEmailRecipients.length > 0) {
    confidenceRisks.push({
      id: "email",
      level: "blocking",
      message: t(
        "watchlists:jobs.form.confidenceRiskEmail",
        "Fix invalid email recipients before saving."
      )
    })
  }

  if (authoringMode === "basic" && hasAdvancedConfiguration) {
    confidenceRisks.push({
      id: "hidden-advanced",
      level: "warning",
      message: t(
        "watchlists:jobs.form.confidenceRiskHiddenAdvanced",
        "Advanced settings are preserved and hidden in Basic mode."
      )
    })
  }

  const confidenceHasBlockingRisk = confidenceRisks.some((risk) => risk.level === "blocking")
  const confidenceStatusLabel = confidenceHasBlockingRisk
    ? t("watchlists:jobs.form.confidenceNeedsAttention", "Needs attention")
    : t("watchlists:jobs.form.confidenceReady", "Ready to save")
  const confidenceStatusColor = confidenceHasBlockingRisk ? "orange" : "green"
  const outputScheduleSummaryText = createScheduledOutput
    ? hasScheduleSelection
      ? t(
        "watchlists:jobs.form.liveSummary.scheduledOutput",
        "Create a report after each scheduled run"
      )
      : t(
        "watchlists:jobs.form.liveSummary.scheduledOutputNeedsSchedule",
        "Add a schedule to create reports automatically"
      )
    : t(
      "watchlists:jobs.form.liveSummary.manualOutput",
      "Manual/test reports only"
    )
  const deliverySummaryParts: string[] = []
  if (deliveryEmailEnabled) {
    deliverySummaryParts.push(
      t(
        "watchlists:jobs.form.liveSummary.deliveryEmail",
        "Deliver by email ({{count}} recipient{{plural}})",
        {
          count: deliveryEmailRecipients.length,
          plural: deliveryEmailRecipients.length === 1 ? "" : "s"
        }
      )
    )
  }
  if (deliveryChatbookEnabled) {
    deliverySummaryParts.push(
      t("watchlists:jobs.form.liveSummary.deliveryChatbook", "Save to Chatbook")
    )
  }
  const deliverySummaryText =
    deliverySummaryParts.length > 0
      ? deliverySummaryParts.join(" + ")
      : t("watchlists:jobs.form.liveSummary.deliveryReportsOnly", "Reports tab only")
  const audioSummaryText = audioBriefingEnabled
    ? t(
      "watchlists:jobs.form.liveSummary.audioEnabled",
      "Audio briefing requested ({{voice}}, {{minutes}} min target)",
      {
        voice: audioVoice,
        minutes: audioTargetMinutes
      }
    )
    : t("watchlists:jobs.form.liveSummary.audioDisabled", "Text report only")
  const hasHiddenAdvancedInBasic = authoringMode === "basic" && hasAdvancedConfiguration

  const handleAuthoringModeChange = (nextMode: AuthoringMode) => {
    if (nextMode === authoringMode) return
    if (nextMode === "basic" && hasAdvancedConfiguration) {
      message.info(
        t(
          "watchlists:jobs.form.modeHiddenSettingsNotice",
          "Advanced settings are preserved and will still apply, but they are hidden in Basic mode."
        )
      )
    }
    if (nextMode === "basic") {
      setBasicStep("scope")
    }
    void trackWatchlistsPreventionTelemetry({
      type: "watchlists_authoring_mode_changed",
      surface: "job_form",
      from_mode: authoringMode,
      to_mode: nextMode,
      context: authoringContext
    })
    setAuthoringMode(nextMode)
  }

  const goToBasicStep = (stepId: BasicStepId) => {
    setBasicStep(stepId)
  }

  const handleBasicNext = () => {
    if (basicStep === "scope" && !hasScopeSelection) {
      message.error(
        t("watchlists:jobs.form.scopeRequired", "Please select at least one feed, group, or tag")
      )
      return
    }
    if (basicStep === "schedule" && !hasScheduleSelection) {
      message.error(
        t(
          "watchlists:jobs.form.scheduleRequiredForBasic",
          "Set a schedule before continuing to review."
        )
      )
      return
    }
    const nextStep = BASIC_STEPS[basicStepIndex + 1]
    if (nextStep) {
      void trackWatchlistsPreventionTelemetry({
        type: "watchlists_basic_step_completed",
        surface: "job_form",
        step: basicStep
      })
      setBasicStep(nextStep)
    }
  }

  const handleBasicBack = () => {
    const prevStep = BASIC_STEPS[basicStepIndex - 1]
    if (prevStep) {
      setBasicStep(prevStep)
    }
  }

  const scopeSummary = summarizeScopeCounts(scope, t)
  const scopeSummaryLines = buildScopeTooltipLines(
    scope,
    {
      sources: scopeSourceNamesById,
      groups: scopeGroupNamesById
    },
    t,
    4
  )

  const scopedPreviewCandidates = useMemo(() => {
    const selectedSourceIds = Array.isArray(scope.sources) ? scope.sources : []
    if (selectedSourceIds.length === 0) return previewCandidates
    const allowedSourceIds = new Set(selectedSourceIds)
    return previewCandidates.filter((item) => allowedSourceIds.has(item.source_id))
  }, [previewCandidates, scope.sources])

  const filterPreviewOutcome = useMemo(
    () => evaluatePreviewItems(scopedPreviewCandidates, filters),
    [filters, scopedPreviewCandidates]
  )

  const filterPreviewUnavailableReason = useMemo(() => {
    if (previewLoading) return null
    if (!isEditing) {
      return t(
        "watchlists:jobs.form.previewCreateHint",
        "Save this monitor once to load sample candidates for live filter preview."
      )
    }
    if (previewError) return previewError
    return null
  }, [isEditing, previewError, previewLoading, t])

  const filterPreviewSummaryText = useMemo(() => {
    if (previewLoading) {
      return t("watchlists:filters.preview.loading", "Loading sample candidates...")
    }
    if (filterPreviewUnavailableReason) {
      return filterPreviewUnavailableReason
    }
    return t(
      "watchlists:jobs.form.previewSummary",
      "{{ingestable}} ingestable, {{filtered}} filtered from {{total}} sample items.",
      {
        ingestable: filterPreviewOutcome.ingestable,
        filtered: filterPreviewOutcome.filtered,
        total: filterPreviewOutcome.total
      }
    )
  }, [
    filterPreviewOutcome.filtered,
    filterPreviewOutcome.ingestable,
    filterPreviewOutcome.total,
    filterPreviewUnavailableReason,
    previewLoading,
    t
  ])

  const collapseItems = [
    {
      key: "scope",
      label: (
        <span className="font-medium">
          {t("watchlists:jobs.form.scope", "Feeds to Include")}
          <span className="text-danger ml-1">*</span>
        </span>
      ),
      children: <ScopeSelector value={scope} onChange={setScope} watchlistId={watchlistId} />,
      forceRender: true
    },
    {
      key: "schedule",
      label: (
        <span className="font-medium">
          {t("watchlists:jobs.form.schedule", "Schedule")}
        </span>
      ),
      children: (
        <SchedulePicker
          value={schedule}
          onChange={setSchedule}
          timezone={timezone}
          onTimezoneChange={setTimezone}
        />
      )
    },
    ...(authoringMode === "advanced"
      ? [{
        key: "filters",
        label: (
          <span className="font-medium">
            {t("watchlists:jobs.form.filters", "Filters")}
            {filters.length > 0 && (
              <span className="ml-2 text-text-muted">({filters.length})</span>
            )}
          </span>
        ),
        children: (
          <FilterBuilder
            value={filters}
            onChange={setFilters}
            preview={{
              loading: previewLoading,
              unavailableReason: filterPreviewUnavailableReason,
              outcome: filterPreviewOutcome
            }}
          />
        )
      }]
      : []),
    {
      key: "output_prefs",
      label: (
        <span className="font-medium">
          {t("watchlists:jobs.form.outputPrefs", "Output & Delivery")}
        </span>
      ),
      children: (
        <div className="space-y-4">
          <div className="rounded-lg border border-border p-3">
            <div className="mb-3 text-sm font-medium">
              {t("watchlists:jobs.form.guidedPresets", "Guided presets")}
            </div>
            <div className="grid grid-cols-1 gap-3 md:grid-cols-[1fr_auto]">
              <Select
                value={outputPreset}
                onChange={(value) => setOutputPreset(value as OutputPresetId)}
                options={[
                  {
                    value: "briefing_md",
                    label: t("watchlists:jobs.form.presetBriefingMd", "Daily briefing (Markdown)"),
                  },
                  {
                    value: "newsletter_html",
                    label: t("watchlists:jobs.form.presetNewsletterHtml", "Newsletter (HTML + email)"),
                  },
                  {
                    value: "mece_md",
                    label: t("watchlists:jobs.form.presetMeceMd", "Structured review (Markdown)"),
                  },
                ]}
                placeholder={t("watchlists:jobs.form.presetPlaceholder", "Choose a preset")}
              />
              <Button
                onClick={() => applyPreset(outputPreset)}
                disabled={!outputPreset}
              >
                {t("watchlists:jobs.form.applyPreset", "Apply preset")}
              </Button>
            </div>
            <div className="mt-2 text-xs text-text-muted">
              {t(
                "watchlists:jobs.form.presetHint",
                "Presets prefill template and delivery defaults. Structured review groups content into non-overlapping sections for easier scanning."
              )}
            </div>
          </div>

          <div className="rounded-lg border border-border p-3">
            <div className="flex items-center justify-between gap-3">
              <div>
                <div className="text-sm font-medium">
                  {t("watchlists:jobs.form.createScheduledOutput", "Scheduled reports")}
                </div>
                <div className="text-xs text-text-muted">
                  {t(
                    "watchlists:jobs.form.createScheduledOutputHint",
                    "Create a report after each scheduled run. Leave off for manual/test reports only."
                  )}
                </div>
              </div>
              <Switch
                checked={createScheduledOutput}
                onChange={setCreateScheduledOutput}
                data-testid="job-form-create-scheduled-output-switch"
              />
            </div>
            {createScheduledOutput && !hasScheduleSelection && (
              <div className="mt-2 text-xs text-text-muted">
                {t(
                  "watchlists:jobs.form.createScheduledOutputNeedsSchedule",
                  "Add a schedule before saving if this monitor should generate recurring reports."
                )}
              </div>
            )}
          </div>

          <div className="rounded-lg border border-border p-3">
            <div className="mb-3 flex items-center gap-1 text-sm font-medium">
              {t("watchlists:jobs.form.defaultTemplate", "Default template")}
              <WatchlistsHelpTooltip topic="jinja2" />
            </div>
            <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
              <div>
                <div className="mb-1 text-xs text-text-muted">
                  {t("watchlists:jobs.form.defaultTemplateName", "Template name")}
                </div>
                <Select
                  allowClear
                  showSearch
                  value={outputTemplateName}
                  options={templateOptions}
                  optionFilterProp="label"
                  placeholder={t("watchlists:jobs.form.defaultTemplatePlaceholder", "Select a template")}
                  onChange={(value) => {
                    setOutputTemplateName(value)
                    if (!value) {
                      setOutputTemplateVersion(null)
                    }
                  }}
                />
              </div>
              <div>
                <div className="mb-1 text-xs text-text-muted">
                  {t("watchlists:jobs.form.defaultTemplateVersion", "Template version")}
                </div>
                <InputNumber
                  min={1}
                  precision={0}
                  value={outputTemplateVersion}
                  disabled={!outputTemplateName}
                  onChange={(value) =>
                    setOutputTemplateVersion(
                      typeof value === "number" && value > 0 ? Math.floor(value) : null
                    )
                  }
                  className="w-full"
                  placeholder={t("watchlists:jobs.form.defaultTemplateVersionAuto", "Latest")}
                />
              </div>
              <div>
                <div className="mb-1 text-xs text-text-muted">
                  {t("watchlists:jobs.form.defaultTemplateFormat", "Default output format")}
                </div>
                <Select
                  allowClear
                  value={outputTemplateFormat}
                  onChange={(value) => setOutputTemplateFormat(value as OutputFormat | undefined)}
                  placeholder={t("watchlists:jobs.form.defaultTemplateFormatAuto", "Template/default")}
                  options={[
                    { value: "md", label: "Markdown (md)" },
                    { value: "html", label: "HTML" },
                  ]}
                />
              </div>
            </div>
          </div>

          {authoringMode === "advanced" && (
            <>
              <div className="rounded-lg border border-border p-3">
            <div className="mb-3 flex items-center gap-1 text-sm font-medium">
              {t("watchlists:jobs.form.retentionDefaults", "Retention defaults")}
              <WatchlistsHelpTooltip topic="ttl" />
            </div>
            <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
              <div>
                <div className="mb-1 text-xs text-text-muted">
                  {t("watchlists:jobs.form.retentionDefault", "Default retention")}
                </div>
                <div className="grid grid-cols-[1fr_140px] gap-2">
                  <InputNumber
                    min={0}
                    precision={0}
                    value={retentionDefaultDuration.value}
                    onChange={(value) =>
                      setRetentionDefaultDuration((prev) => ({
                        ...prev,
                        value: typeof value === "number" && value >= 0 ? Math.floor(value) : null
                      }))
                    }
                    className="w-full"
                    placeholder={t("watchlists:jobs.form.retentionDefaultPlaceholder", "Server default")}
                  />
                  <Select
                    value={retentionDefaultDuration.unit}
                    onChange={(unit: DurationUnit) =>
                      setRetentionDefaultDuration((prev) => ({ ...prev, unit }))
                    }
                    options={retentionUnitOptions}
                  />
                </div>
                {typeof retentionDefaultSeconds === "number" && (
                  <div className="mt-1 text-xs text-text-muted">
                    {t(
                      "watchlists:jobs.form.retentionDefaultPreview",
                      "Saved internally as {{seconds}} seconds",
                      { seconds: retentionDefaultSeconds }
                    )}
                  </div>
                )}
              </div>
              <div>
                <div className="mb-1 text-xs text-text-muted">
                  {t("watchlists:jobs.form.retentionTemporary", "Temporary retention")}
                </div>
                <div className="grid grid-cols-[1fr_140px] gap-2">
                  <InputNumber
                    min={0}
                    precision={0}
                    value={retentionTemporaryDuration.value}
                    onChange={(value) =>
                      setRetentionTemporaryDuration((prev) => ({
                        ...prev,
                        value: typeof value === "number" && value >= 0 ? Math.floor(value) : null
                      }))
                    }
                    className="w-full"
                    placeholder={t(
                      "watchlists:jobs.form.retentionTemporaryPlaceholder",
                      "Server temporary default"
                    )}
                  />
                  <Select
                    value={retentionTemporaryDuration.unit}
                    onChange={(unit: DurationUnit) =>
                      setRetentionTemporaryDuration((prev) => ({ ...prev, unit }))
                    }
                    options={retentionUnitOptions}
                  />
                </div>
                {typeof retentionTemporarySeconds === "number" && (
                  <div className="mt-1 text-xs text-text-muted">
                    {t(
                      "watchlists:jobs.form.retentionTemporaryPreview",
                      "Saved internally as {{seconds}} seconds",
                      { seconds: retentionTemporarySeconds }
                    )}
                  </div>
                )}
              </div>
            </div>
            <div className="mt-2 text-xs text-text-muted">
              {t(
                "watchlists:jobs.form.retentionHint",
                "Set 0 for no expiry. Leave blank to use server defaults. Values are converted safely to seconds when saved."
              )}
            </div>
          </div>

          <div className="rounded-lg border border-border p-3">
            <div className="mb-3 flex items-center justify-between gap-3">
              <div className="text-sm font-medium">
                {t("watchlists:jobs.form.emailDelivery", "Email delivery")}
              </div>
              <Switch checked={deliveryEmailEnabled} onChange={setDeliveryEmailEnabled} />
            </div>
            <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
              <div>
                <div className="mb-1 text-xs text-text-muted">
                  {t("watchlists:jobs.form.emailRecipients", "Recipients")}
                </div>
                <Select
                  mode="tags"
                  value={deliveryEmailRecipients}
                  onChange={setDeliveryEmailRecipients}
                  placeholder={t("watchlists:jobs.form.emailRecipientsPlaceholder", "Enter email addresses")}
                  className="w-full"
                  tokenSeparators={[","]}
                  status={invalidEmailRecipients.length > 0 ? "error" : undefined}
                />
                {invalidEmailRecipients.length > 0 && (
                  <div className="mt-1 text-xs text-danger">
                    {t(
                      "watchlists:jobs.form.emailRecipientsInvalidInline",
                      "Invalid addresses: {{emails}}",
                      { emails: invalidEmailRecipients.join(", ") }
                    )}
                  </div>
                )}
              </div>
              <div>
                <div className="mb-1 text-xs text-text-muted">
                  {t("watchlists:jobs.form.emailSubject", "Default subject")}
                </div>
                <Input
                  value={deliveryEmailSubject}
                  onChange={(event) => setDeliveryEmailSubject(event.target.value)}
                  placeholder={t(
                    "watchlists:jobs.form.emailSubjectPlaceholder",
                    "Defaults to output title"
                  )}
                />
              </div>
              <div>
                <div className="mb-1 text-xs text-text-muted">
                  {t("watchlists:jobs.form.emailBodyFormat", "Body format")}
                </div>
                <Select
                  value={deliveryEmailBodyFormat}
                  onChange={(value: EmailBodyFormat) => setDeliveryEmailBodyFormat(value)}
                  options={[
                    { value: "auto", label: t("watchlists:jobs.form.emailBodyAuto", "Auto") },
                    { value: "text", label: t("watchlists:jobs.form.emailBodyText", "Text") },
                    { value: "html", label: t("watchlists:jobs.form.emailBodyHtml", "HTML") }
                  ]}
                />
              </div>
            </div>
            <div className="mt-3 flex items-center justify-between rounded-md bg-surface px-3 py-2 text-sm">
              <span>{t("watchlists:jobs.form.emailAttachFile", "Attach output file")}</span>
              <Switch
                checked={deliveryEmailAttachFile}
                onChange={setDeliveryEmailAttachFile}
              />
            </div>
              </div>

              <div className="rounded-lg border border-border p-3">
            <div className="mb-3 flex items-center justify-between gap-3">
              <div className="text-sm font-medium">
                {t("watchlists:jobs.form.chatbookDelivery", "Chatbook delivery")}
              </div>
              <Switch checked={deliveryChatbookEnabled} onChange={setDeliveryChatbookEnabled} />
            </div>
            <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
              <Input
                value={deliveryChatbookTitle}
                onChange={(event) => setDeliveryChatbookTitle(event.target.value)}
                placeholder={t("watchlists:jobs.form.chatbookTitle", "Document title (optional)")}
              />
              <InputNumber
                min={1}
                precision={0}
                value={deliveryChatbookConversationId}
                onChange={(value) =>
                  setDeliveryChatbookConversationId(
                    typeof value === "number" && value > 0 ? Math.floor(value) : null
                  )
                }
                className="w-full"
                placeholder={t("watchlists:jobs.form.chatbookConversationId", "Conversation ID (optional)")}
              />
            </div>
            <Input.TextArea
              value={deliveryChatbookDescription}
              onChange={(event) => setDeliveryChatbookDescription(event.target.value)}
              rows={2}
              className="mt-3"
              placeholder={t("watchlists:jobs.form.chatbookDescription", "Description (optional)")}
            />
              </div>
            </>
          )}

          <div className="rounded-lg border border-border p-3">
            <div className="mb-3 flex items-center justify-between gap-3">
              <div className="text-sm font-medium">
                {t("watchlists:jobs.form.audioBriefing", "Audio briefing")}
              </div>
              <Switch
                checked={audioBriefingEnabled}
                onChange={handleAudioBriefingToggle}
                data-testid="job-form-audio-enabled-switch"
              />
            </div>
            <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
              <div>
                <div className="mb-1 text-xs text-text-muted">
                  {t("watchlists:jobs.form.audioVoice", "Voice")}
                </div>
                <Select
                  value={audioVoice}
                  onChange={(value) => setAudioVoice(String(value || DEFAULT_AUDIO_VOICE))}
                  disabled={!audioBriefingEnabled}
                  data-testid="job-form-audio-voice-select"
                  options={[
                    { value: "alloy", label: "Alloy" },
                    { value: "nova", label: "Nova" },
                    { value: "echo", label: "Echo" },
                    { value: "fable", label: "Fable" },
                    { value: "onyx", label: "Onyx" },
                    { value: "shimmer", label: "Shimmer" }
                  ]}
                />
              </div>
              <div>
                <div className="mb-1 text-xs text-text-muted">
                  {t("watchlists:jobs.form.audioSpeed", "Speed")}
                </div>
                <InputNumber
                  min={AUDIO_SPEED_MIN}
                  max={AUDIO_SPEED_MAX}
                  step={0.05}
                  value={audioSpeed}
                  onChange={(value) => setAudioSpeed(normalizeAudioSpeed(value))}
                  disabled={!audioBriefingEnabled}
                  className="w-full"
                  data-testid="job-form-audio-speed-input"
                />
              </div>
              <div>
                <div className="mb-1 text-xs text-text-muted">
                  {t("watchlists:jobs.form.audioTargetMinutes", "Target duration (minutes)")}
                </div>
                <InputNumber
                  min={AUDIO_TARGET_MINUTES_MIN}
                  max={AUDIO_TARGET_MINUTES_MAX}
                  precision={0}
                  value={audioTargetMinutes}
                  onChange={(value) => setAudioTargetMinutes(normalizeAudioTargetMinutes(value))}
                  disabled={!audioBriefingEnabled}
                  className="w-full"
                  data-testid="job-form-audio-target-minutes-input"
                />
              </div>
            </div>
            <div className="mt-2 text-xs text-text-muted">
              {t(
                "watchlists:jobs.form.audioHint",
                "Enable spoken briefings alongside text outputs. Keep defaults for fastest setup."
              )}
            </div>
            <div className="mt-1 text-xs text-text-muted" data-testid="job-form-audio-practical-hint">
              {t(
                "watchlists:jobs.form.audioPracticalHint",
                "Practical default: Alloy voice at 1.0 speed and 8-minute target, then tune after your first run."
              )}
            </div>
            <div className="mt-1 text-xs text-text-muted" data-testid="job-form-audio-item-cap-hint">
              {t(
                "watchlists:jobs.form.audioItemCapHint",
                "Audio briefings currently include up to 100 items per run. Tighten filters for more focused narration."
              )}
            </div>
            <div className="mt-2 space-y-2 rounded-md border border-border bg-surface p-3">
              <Button
                type="default"
                onClick={handleTestAudioSettings}
                disabled={!audioBriefingEnabled || audioTestLoading}
                loading={audioTestLoading}
                data-testid="job-form-audio-test-button"
              >
                {t("watchlists:jobs.form.audioTestButton", "Test audio settings")}
              </Button>
              <div className="text-xs text-text-muted" data-testid="job-form-audio-test-hint">
                {t(
                  "watchlists:jobs.form.audioTestHint",
                  "Generate a short sample so you can confirm voice and speed before saving."
                )}
              </div>
              {audioTestLoading && (
                <div className="text-xs text-text-muted" data-testid="job-form-audio-test-loading">
                  {t("watchlists:jobs.form.audioTestLoading", "Generating sample audio...")}
                </div>
              )}
              {audioTestError && (
                <div className="text-xs text-danger" data-testid="job-form-audio-test-error">
                  {audioTestError}
                </div>
              )}
              {audioTestUrl && !audioTestError && (
                <div className="space-y-2" data-testid="job-form-audio-test-success">
                  <div className="text-xs text-text-muted">
                    {t("watchlists:jobs.form.audioTestReady", "Sample ready. Listen before saving.")}
                  </div>
                  <audio controls preload="none" src={audioTestUrl} data-testid="job-form-audio-test-player">
                    {t(
                      "watchlists:outputs.audioPlayerUnsupported",
                      "Your browser does not support audio playback."
                    )}
                  </audio>
                </div>
              )}
            </div>
            {authoringMode === "advanced" && (
              <>
                <div className="mt-2">
                  <Button
                    type="link"
                    className="px-0"
                    onClick={() => setShowAdvancedAudioOptions((previous) => !previous)}
                  >
                    {showAdvancedAudioOptions
                      ? t("watchlists:jobs.form.hideAudioAdvanced", "Hide advanced audio options")
                      : t("watchlists:jobs.form.showAudioAdvanced", "Show advanced audio options")}
                  </Button>
                </div>
                {showAdvancedAudioOptions && (
                  <div className="mt-2 space-y-3 rounded-md border border-border bg-surface p-3">
                    <div>
                      <div className="mb-1 text-xs text-text-muted">
                        {t("watchlists:jobs.form.audioBackgroundTrack", "Background track URI")}
                      </div>
                      <Input
                        value={audioBackgroundUri}
                        onChange={(event) => setAudioBackgroundUri(event.target.value)}
                        disabled={!audioBriefingEnabled}
                        placeholder={t(
                          "watchlists:jobs.form.audioBackgroundTrackPlaceholder",
                          "file:///path/to/bed.mp3"
                        )}
                      />
                    </div>
                    <div>
                      <div className="mb-1 text-xs text-text-muted">
                        {t("watchlists:jobs.form.audioVoiceMap", "Voice map (JSON)")}
                      </div>
                      <Input.TextArea
                        rows={3}
                        value={audioVoiceMapText}
                        onChange={(event) => setAudioVoiceMapText(event.target.value)}
                        disabled={!audioBriefingEnabled}
                        placeholder={t(
                          "watchlists:jobs.form.audioVoiceMapPlaceholder",
                          "{ \"HOST\": \"af_heart\", \"REPORTER\": \"am_adam\" }"
                        )}
                      />
                    </div>
                    <div className="text-xs text-text-muted">
                      {t(
                        "watchlists:jobs.form.audioAdvancedHint",
                        "Use advanced options only if you need soundtrack mixing or custom speaker-to-voice assignments."
                      )}
                    </div>
                  </div>
                )}
              </>
            )}
            {authoringMode === "basic" &&
              (audioBackgroundUri.trim().length > 0 || audioVoiceMapText.trim().length > 0) && (
                <div className="mt-2 text-xs text-text-muted">
                  {t(
                    "watchlists:jobs.form.modeHiddenSettingsNotice",
                    "Advanced settings are preserved and will still apply, but they are hidden in Basic mode."
                  )}
                </div>
              )}
          </div>
        </div>
      )
    }
  ]

  const basicStepOptions: Array<{ id: BasicStepId; label: string }> = [
    { id: "scope", label: t("watchlists:jobs.form.steps.scope", "Scope") },
    { id: "schedule", label: t("watchlists:jobs.form.steps.schedule", "Schedule") },
    { id: "output", label: t("watchlists:jobs.form.steps.output", "Output") },
    { id: "review", label: t("watchlists:jobs.form.steps.review", "Review") }
  ]

  const basicStepToCollapseKey: Record<Exclude<BasicStepId, "review">, string> = {
    scope: "scope",
    schedule: "schedule",
    output: "output_prefs"
  }
  const activeBasicCollapseItem =
    basicStep === "review"
      ? null
      : collapseItems.find((item) => item.key === basicStepToCollapseKey[basicStep])

  return (
    <Modal
      title={
        isEditing
          ? t("watchlists:jobs.editJob", "Edit Monitor")
          : t("watchlists:jobs.addJob", "Add Monitor")
      }
      open={open}
      onOk={handleSubmit}
      onCancel={handleCancel}
      okText={isEditing ? t("common:save", "Save") : t("common:create", "Create")}
      cancelText={t("common:cancel", "Cancel")}
      confirmLoading={submitting}
      destroyOnHidden
      data-testid="job-form-modal"
      width={modalChrome.width}
      style={modalChrome.style}
      styles={modalChrome.styles}
    >
      <Form form={form} layout="vertical" className="mt-4">
        <div className="mb-4 rounded-lg border border-border bg-surface p-3">
          <div className="mb-2 text-sm font-medium">
            {t("watchlists:jobs.form.modeLabel", "Setup mode")}
          </div>
          <Radio.Group
            value={authoringMode}
            onChange={(event) => handleAuthoringModeChange(event.target.value as AuthoringMode)}
            optionType="button"
            buttonStyle="solid"
            size="small"
          >
            <Radio.Button value="basic" data-testid="job-form-mode-basic">
              {t("watchlists:jobs.form.modeBasic", "Basic")}
            </Radio.Button>
            <Radio.Button value="advanced" data-testid="job-form-mode-advanced">
              {t("watchlists:jobs.form.modeAdvanced", "Advanced")}
            </Radio.Button>
          </Radio.Group>
          <div className="mt-2 text-xs text-text-muted">
            {authoringMode === "basic"
              ? t(
                "watchlists:jobs.form.modeHelpBasic",
                "Basic mode hides optional filters and delivery fine-tuning so you can set up a monitor faster."
              )
              : t(
                "watchlists:jobs.form.modeHelpAdvanced",
                "Advanced mode exposes all scheduling, filtering, retention, and delivery controls."
              )}
          </div>
          {authoringMode === "basic" && hasAdvancedConfiguration && (
            <div className="mt-2 text-xs text-text-muted">
              {t(
                "watchlists:jobs.form.modeHiddenSettingsNotice",
                "Advanced settings are preserved and will still apply, but they are hidden in Basic mode."
              )}
            </div>
          )}
        </div>

        <Form.Item
          name="name"
          label={t("watchlists:jobs.form.name", "Name")}
          rules={[
            {
              required: true,
              message: t("watchlists:jobs.form.nameRequired", "Please enter a name")
            },
            {
              max: 200,
              message: t("watchlists:jobs.form.nameTooLong", "Name must be less than 200 characters")
            }
          ]}
        >
          <Input
            placeholder={t("watchlists:jobs.form.namePlaceholder", "e.g., Daily Tech News")}
          />
        </Form.Item>

        <Form.Item
          name="description"
          label={t("watchlists:jobs.form.description", "Description")}
        >
          <Input.TextArea
            placeholder={t(
              "watchlists:jobs.form.descriptionPlaceholder",
              "Optional description of what this monitor does"
            )}
            rows={2}
          />
        </Form.Item>

        <Form.Item
          name="active"
          label={t("watchlists:jobs.form.active", "Active")}
          valuePropName="checked"
        >
          <Switch />
        </Form.Item>
      </Form>

      <div
        className="mt-4 rounded-lg border border-border bg-surface p-3 space-y-2"
        data-testid="job-form-live-summary"
      >
        <div className="text-sm font-medium">
          {t("watchlists:jobs.form.liveSummaryTitle", "Live setup summary")}
        </div>
        <div className="text-xs text-text-muted" data-testid="job-form-summary-name">
          {t("watchlists:jobs.form.liveSummary.name", "Monitor")}:{" "}
          {String(watchedName || "").trim() || t("watchlists:jobs.form.liveSummary.unnamed", "Untitled monitor")}
        </div>
        <div className="grid grid-cols-1 gap-2 text-xs text-text-muted md:grid-cols-3">
          <div>
            <div className="font-medium text-text">{t("watchlists:jobs.form.liveSummary.scope", "Scope")}</div>
            <div data-testid="job-form-summary-scope">{scopeSummary}</div>
          </div>
          <div>
            <div className="font-medium text-text">{t("watchlists:jobs.form.liveSummary.schedule", "Schedule")}</div>
            <div data-testid="job-form-summary-schedule">
              {schedule ? (
                <CronDisplay expression={schedule} showIcon={false} />
              ) : (
                t("watchlists:jobs.form.liveSummary.notScheduled", "Not scheduled")
              )}
            </div>
          </div>
          <div>
            <div className="font-medium text-text">{t("watchlists:jobs.form.liveSummary.filters", "Filters")}</div>
            <div data-testid="job-form-summary-filters">
              {filters.length > 0
                ? t("watchlists:jobs.form.liveSummary.filtersConfigured", "{{count}} filters configured", {
                  count: filters.length
                })
                : t("watchlists:jobs.form.liveSummary.noFilters", "No filters configured")}
            </div>
          </div>
        </div>
        <div className="text-xs text-text-muted" data-testid="job-form-summary-scope-lines">
          {scopeSummaryLines.join(" · ")}
        </div>
        <div className="text-xs text-text-muted" data-testid="job-form-summary-preview">
          {filterPreviewSummaryText}
        </div>
        <div className="text-xs text-text-muted" data-testid="job-form-summary-output">
          {t("watchlists:jobs.form.liveSummary.output", "Output")}:{" "}
          {outputScheduleSummaryText}
          {" • "}
          {t("watchlists:jobs.form.liveSummary.outputTemplate", "Template")}:{" "}
          {outputTemplateName || t("watchlists:jobs.form.defaultTemplateVersionAuto", "Latest")}
        </div>
        <div className="text-xs text-text-muted" data-testid="job-form-summary-delivery">
          {t("watchlists:jobs.form.liveSummary.delivery", "Delivery")}: {deliverySummaryText}
        </div>
        <div className="text-xs text-text-muted" data-testid="job-form-summary-audio">
          {t("watchlists:jobs.form.liveSummary.audio", "Audio briefing")}: {audioSummaryText}
        </div>
        {hasHiddenAdvancedInBasic && (
          <div className="text-xs text-text-muted" data-testid="job-form-summary-hidden-advanced">
            {t(
              "watchlists:jobs.form.liveSummary.hiddenAdvanced",
              "Advanced settings are active and hidden in Basic mode."
            )}
          </div>
        )}
      </div>

      <div
        className="mt-3 rounded-lg border border-border bg-surface p-3 space-y-2"
        data-testid="job-form-confidence-panel"
      >
        <div className="flex items-center justify-between gap-2">
          <div className="text-sm font-medium">
            {t("watchlists:jobs.form.confidenceTitle", "Configuration confidence")}
          </div>
          <Tag color={confidenceStatusColor} data-testid="job-form-confidence-status">
            {confidenceStatusLabel}
          </Tag>
        </div>
        <div className="text-xs text-text-muted" data-testid="job-form-confidence-checks">
          {t(
            "watchlists:jobs.form.confidenceChecks",
            "{{complete}}/{{total}} checks complete",
            {
              complete: confidenceCompletedChecks,
              total: confidenceChecks.length
            }
          )}
        </div>
        {confidenceRisks.length > 0 ? (
          <div className="space-y-1">
            {confidenceRisks.map((risk) => (
              <div
                key={risk.id}
                className="text-xs text-text-muted"
                data-testid={`job-form-confidence-risk-${risk.id}`}
              >
                {risk.message}
              </div>
            ))}
          </div>
        ) : (
          <div className="text-xs text-text-muted" data-testid="job-form-confidence-no-risks">
            {t("watchlists:jobs.form.confidenceNoRisks", "No unresolved risks.")}
          </div>
        )}
      </div>

      {authoringMode === "advanced" ? (
        <Collapse
          items={collapseItems}
          defaultActiveKey={["scope"]}
          className="mt-4"
          expandIconPlacement="end"
        />
      ) : (
        <div className="mt-4 space-y-3" data-testid="job-form-basic-stepper">
          <div className="flex flex-wrap gap-2">
            {basicStepOptions.map((step, index) => (
              <Button
                key={step.id}
                size="small"
                type={basicStep === step.id ? "primary" : "default"}
                onClick={() => goToBasicStep(step.id)}
                data-testid={`job-form-basic-step-${step.id}`}
              >
                {index + 1}. {step.label}
              </Button>
            ))}
          </div>

          {activeBasicCollapseItem ? (
            <Collapse
              items={[activeBasicCollapseItem]}
              activeKey={[activeBasicCollapseItem.key]}
              className="mt-1"
              expandIconPlacement="end"
            />
          ) : (
            <div
              className="rounded-lg border border-border bg-surface p-3 text-sm text-text-muted"
              data-testid="job-form-basic-review"
            >
              <div className="font-medium text-text mb-2">
                {t("watchlists:jobs.form.steps.reviewTitle", "Review monitor setup")}
              </div>
              <div>
                {t("watchlists:jobs.form.steps.reviewScope", "Scope")}: {scopeSummary}
              </div>
              <div>
                {t("watchlists:jobs.form.steps.reviewSchedule", "Schedule")}:{" "}
                {schedule ? (
                  <CronDisplay expression={schedule} showIcon={false} />
                ) : (
                  t("watchlists:jobs.form.liveSummary.notScheduled", "Not scheduled")
                )}
              </div>
              <div>
                {t("watchlists:jobs.form.steps.reviewOutput", "Output")}:{" "}
                {outputScheduleSummaryText}
                {" • "}
                {outputTemplateName || t("watchlists:jobs.form.defaultTemplatePlaceholder", "Select a template")}
              </div>
              <div>
                {t("watchlists:jobs.form.steps.reviewAudio", "Audio briefing")}:{" "}
                {audioBriefingEnabled
                  ? t("common:enabled", "Enabled")
                  : t("common:disabled", "Disabled")}
              </div>
              <div>
                {t("watchlists:jobs.form.steps.reviewDelivery", "Delivery")}: {deliverySummaryText}
              </div>
              {hasHiddenAdvancedInBasic && (
                <div className="text-xs text-text-muted mt-1">
                  {t(
                    "watchlists:jobs.form.liveSummary.hiddenAdvanced",
                    "Advanced settings are active and hidden in Basic mode."
                  )}
                </div>
              )}
            </div>
          )}

          <div className="flex items-center justify-between gap-2">
            <Button
              onClick={handleBasicBack}
              disabled={basicStepIndex === 0}
              data-testid="job-form-basic-back"
            >
              {t("watchlists:jobs.form.steps.back", "Back")}
            </Button>
            {basicStep !== "review" ? (
              <Button
                type="primary"
                onClick={handleBasicNext}
                data-testid="job-form-basic-next"
              >
                {t("watchlists:jobs.form.steps.next", "Next")}
              </Button>
            ) : (
              <div className="text-xs text-text-muted">
                {t(
                  "watchlists:jobs.form.steps.reviewHint",
                  "Use Create/Save to finalize this monitor."
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </Modal>
  )
}
