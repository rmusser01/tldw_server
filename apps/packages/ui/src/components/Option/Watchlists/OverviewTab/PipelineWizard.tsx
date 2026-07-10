import React, { useEffect, useMemo, useRef, useState } from "react"
import {
  Button,
  Checkbox,
  Form,
  Input,
  InputNumber,
  Modal,
  Radio,
  Select,
  Space,
  Switch
} from "antd"
import { useTranslation } from "react-i18next"
import { Alert as DesignSystemAlert } from "@/components/ui"
import type {
  JobPreviewResult,
  WatchlistBriefingProjection,
  WatchlistProgramFormat,
  WatchlistSource
} from "@/types/watchlists"
import {
  INTERVAL_HOURS_MAX,
  INTERVAL_HOURS_MIN,
  INTERVAL_MINUTES_MAX,
  INTERVAL_MINUTES_MIN,
  type ScheduleIntervalUnit,
  type WeekdayToken
} from "../JobsTab/schedule-utils"
import { buildBriefingReceiptModel } from "../shared/briefing-receipt"
import { toPipelineJobCreatePayload } from "./pipeline-contract"
import {
  mergePipelineWizardDraft,
  getPipelineWizardBriefingOutcome,
  projectPipelineWizardOccurrences,
  toBriefingPipelineDraft,
  type PipelineWizardAudioSpeakerDraft,
  type PipelineWizardDraft,
  type PipelineWizardScheduleMode,
  type PipelineWizardSourceMode,
  validatePipelineWizardDraft
} from "./pipeline-wizard-state"

export type PipelineWizardStep = "sources" | "cadence" | "briefing" | "delivery" | "test"

export interface PipelineWizardTestOptions {
  externalDelivery: boolean
  audioSampleSeconds: 60 | null
  jobId?: number
  requestGeneration: number
  signal: AbortSignal
}

export interface PipelineWizardActionResult {
  jobId?: number
  runId?: number
  status?: "ready" | "running" | "failed" | "active" | "cancelled"
  message?: string
  briefing?: WatchlistBriefingProjection
  sourceTest?: JobPreviewResult
}

interface PipelineWizardProps {
  open: boolean
  sources?: WatchlistSource[]
  sourcesLoading?: boolean
  submitting?: boolean
  submitError?: string | null
  initialStep?: PipelineWizardStep | number
  initialDraft?: Partial<PipelineWizardDraft>
  sessionKey?: string | number
  onCancel: () => void
  onSaveDraft?: (draft: PipelineWizardDraft) => void | Promise<void>
  onTest?: (
    draft: PipelineWizardDraft,
    options: PipelineWizardTestOptions,
    onProgress: (projection: WatchlistBriefingProjection) => void
  ) => PipelineWizardActionResult | void | Promise<PipelineWizardActionResult | void>
  onActivate?: (
    draft: PipelineWizardDraft,
    options: { jobId?: number }
  ) => PipelineWizardActionResult | void | Promise<PipelineWizardActionResult | void>
  onTestSource?: (draft: PipelineWizardDraft) =>
    PipelineWizardActionResult | void | Promise<PipelineWizardActionResult | void>
  previewLoading?: boolean
  previewError?: string | null
  previewRendered?: string | null
  previewRunId?: number | null
  previewWarnings?: string[]
  onPreview?: (draft: PipelineWizardDraft) => void
}

export type { PipelineWizardProps }

const STEPS: PipelineWizardStep[] = ["sources", "cadence", "briefing", "delivery", "test"]
const STEP_FALLBACKS: Record<PipelineWizardStep, string> = {
  sources: "Sources",
  cadence: "Cadence",
  briefing: "Briefing",
  delivery: "Delivery",
  test: "Test"
}
const LAST_STEP = STEPS.length - 1
const DEFAULT_SPEAKER_VOICES = ["alloy", "nova", "echo", "fable"]
const TOUCH_TARGET_CLASS = "[@media(pointer:coarse)]:min-h-11 [@media(pointer:coarse)]:min-w-11"
const FIELD_IDS: Partial<Record<string, string>> = {
  sourceName: "pipeline-source-name",
  sourceUrl: "pipeline-source-url",
  monitorName: "pipeline-monitor-name",
  timezone: "pipeline-timezone",
  scheduleIntervalValue: "pipeline-schedule-interval",
  scheduleHour: "pipeline-schedule-hour",
  scheduleMinute: "pipeline-schedule-minute",
  scheduleAdvancedCron: "pipeline-schedule-cron",
  scheduleAdvancedCronTooFrequent: "pipeline-schedule-cron",
  templateName: "pipeline-template-name",
  targetAudioMinutes: "pipeline-audio-minutes",
  emailRecipients: "pipeline-email-recipients"
}

const PROGRAM_FORMATS: Array<{ value: WatchlistProgramFormat; label: string }> = [
  { value: "concise_briefing", label: "Concise briefing" },
  { value: "solo_update", label: "Solo update" },
  { value: "host_discussion", label: "Host discussion" },
  { value: "sportscast", label: "Sportscast" },
  { value: "culture_roundtable", label: "Culture roundtable" },
  { value: "custom", label: "Custom" }
]

const createSpeakers = (
  count: number,
  existing: PipelineWizardAudioSpeakerDraft[],
  speakerLabel: (index: number) => string,
  speakerRole: (index: number) => string
): PipelineWizardAudioSpeakerDraft[] =>
  Array.from({ length: Math.max(1, Math.min(4, count)) }, (_item, index) => {
    const current = existing[index]
    return {
      id: current?.id || `speaker_${index + 1}`,
      label: current?.label || speakerLabel(index + 1),
      role: current?.role || speakerRole(index + 1),
      voice: current?.voice || DEFAULT_SPEAKER_VOICES[index] || DEFAULT_SPEAKER_VOICES[0],
      persona: current?.persona
    }
  })

const stepIndex = (value: PipelineWizardProps["initialStep"]): number => {
  if (typeof value === "number") return Math.max(0, Math.min(LAST_STEP, value))
  const index = value ? STEPS.indexOf(value) : 0
  return index < 0 ? 0 : index
}

const getErrorMessage = (error: unknown): string =>
  error instanceof Error && error.message.trim() ? error.message.trim() : "Server request failed."

const isAbortError = (error: unknown): boolean =>
  error instanceof Error && error.name === "AbortError"

export const PipelineWizard: React.FC<PipelineWizardProps> = ({
  open,
  sources = [],
  sourcesLoading = false,
  submitting = false,
  submitError = null,
  initialStep,
  initialDraft,
  sessionKey,
  onCancel,
  onSaveDraft,
  onTest,
  onActivate,
  onTestSource,
  previewLoading = false,
  previewError = null,
  previewRendered = null,
  previewRunId = null,
  previewWarnings = [],
  onPreview
}) => {
  const { t, i18n } = useTranslation(["watchlists", "common"])
  const locale = i18n?.resolvedLanguage || i18n?.language || "en-US"
  const initial = useMemo(() => {
    const merged = mergePipelineWizardDraft(initialDraft)
    return {
      ...merged,
      audioSpeakers: merged.audioSpeakers.map((speaker, index) => {
        const isDefaultSpeaker = speaker.id === `speaker_${index + 1}` &&
          (!speaker.label || speaker.label === `Speaker ${index + 1}`)
        return isDefaultSpeaker
          ? {
              ...speaker,
              label: t("watchlists:overview.pipelineSetup.speaker.defaultLabel", "Speaker {{index}}", { index: index + 1 }),
              role: t("watchlists:overview.pipelineSetup.speaker.defaultRole", "Host")
            }
          : speaker
      })
    }
  }, [initialDraft, t])
  const [currentStep, setCurrentStep] = useState(() => stepIndex(initialStep))
  const [draft, setDraft] = useState<PipelineWizardDraft>(initial)
  const draftRef = useRef(draft)
  const wasOpenRef = useRef(false)
  const activeSessionKeyRef = useRef<PipelineWizardProps["sessionKey"]>(sessionKey)
  const requestGenerationRef = useRef(0)
  const saveGenerationRef = useRef(0)
  const actionLockRef = useRef(false)
  const actionControllerRef = useRef<AbortController | null>(null)
  const emailRecipientSearchRef = useRef("")
  const validationRef = useRef<HTMLDivElement | null>(null)
  const advancedBriefingRef = useRef<HTMLDetailsElement | null>(null)
  const [stepErrors, setStepErrors] = useState<string[]>([])
  const [actionBusy, setActionBusy] = useState(false)
  const [actionError, setActionError] = useState<string | null>(null)
  const [actionMessage, setActionMessage] = useState<string | null>(null)
  const [sourceTestMessage, setSourceTestMessage] = useState<string | null>(null)
  const [sourceTestResult, setSourceTestResult] = useState<JobPreviewResult | null>(null)
  const [inactiveJobId, setInactiveJobId] = useState<number | undefined>()
  const [testProjection, setTestProjection] = useState<WatchlistBriefingProjection | null>(null)

  useEffect(() => {
    const sessionChanged = activeSessionKeyRef.current !== sessionKey
    if (open && (!wasOpenRef.current || sessionChanged)) {
      requestGenerationRef.current += 1
      actionControllerRef.current?.abort()
      actionControllerRef.current = null
      saveGenerationRef.current += 1
      actionLockRef.current = false
      const next = initial
      draftRef.current = next
      setDraft(next)
      setCurrentStep(stepIndex(initialStep))
      setStepErrors([])
      setActionError(null)
      setActionMessage(null)
      setSourceTestMessage(null)
      setSourceTestResult(null)
      setInactiveJobId(undefined)
      setTestProjection(null)
      setActionBusy(false)
      emailRecipientSearchRef.current = ""
      activeSessionKeyRef.current = sessionKey
    }
    if (!open && wasOpenRef.current) {
      requestGenerationRef.current += 1
      actionControllerRef.current?.abort()
      actionControllerRef.current = null
    }
    wasOpenRef.current = open
  }, [initial, initialStep, open, sessionKey])

  useEffect(() => () => {
    requestGenerationRef.current += 1
    actionControllerRef.current?.abort()
    actionControllerRef.current = null
  }, [])

  useEffect(() => {
    if (!open || draft.sourceMode !== "existing" || draft.sourceIds.length > 0) return
    if (sources.length !== 1) return
    const next = { ...draftRef.current, sourceIds: [sources[0].id] }
    draftRef.current = next
    setDraft(next)
  }, [draft.sourceIds.length, draft.sourceMode, open, sources])

  useEffect(() => {
    const firstError = stepErrors[0]
    if (!firstError) return
    const focusInvalidControl = () => {
      if (firstError === "audioSpeakerVoices") advancedBriefingRef.current?.setAttribute("open", "")
      const target = firstError === "sourceIds"
        ? document.querySelector<HTMLElement>("[data-pipeline-field='sourceIds'] input")
        : firstError === "audioSpeakerVoices"
          ? document.querySelector<HTMLElement>("[id^='pipeline-speaker-'][id$='-voice'][aria-invalid='true']")
        : document.getElementById(FIELD_IDS[firstError] || "")
      if (target && typeof target.focus === "function") target.focus()
      else validationRef.current?.focus()
    }
    const timeoutId = setTimeout(focusInvalidControl, 0)
    return () => clearTimeout(timeoutId)
  }, [currentStep, stepErrors])

  const updateDraft = (patch: Partial<PipelineWizardDraft>) => {
    if (actionControllerRef.current) {
      requestGenerationRef.current += 1
      actionControllerRef.current.abort()
      actionControllerRef.current = null
      actionLockRef.current = false
      setActionBusy(false)
    }
    const next = { ...draftRef.current, ...patch }
    draftRef.current = next
    setDraft(next)
    setStepErrors([])
    setActionError(null)
    setActionMessage(null)
    setSourceTestResult(null)
    setTestProjection(null)
    const generation = ++saveGenerationRef.current
    void Promise.resolve(onSaveDraft?.(next)).catch((error) => {
      const stale = generation !== saveGenerationRef.current
      if (stale && isAbortError(error)) return
      if (!stale) setActionError(getErrorMessage(error))
    })
  }

  const getSpeakerLabel = (index: number) =>
    t("watchlists:overview.pipelineSetup.speaker.defaultLabel", "Speaker {{index}}", { index })

  const getSpeakerRole = () =>
    t("watchlists:overview.pipelineSetup.speaker.defaultRole", "Host")

  const updateSpeaker = (index: number, patch: Partial<PipelineWizardAudioSpeakerDraft>) => {
    const speakers = createSpeakers(
      draft.audioSpeakers.length || 1,
      draft.audioSpeakers,
      getSpeakerLabel,
      getSpeakerRole
    )
    speakers[index] = { ...speakers[index], ...patch }
    updateDraft({ audioSpeakers: speakers })
  }

  const weekdayOptions = useMemo<Array<{ value: WeekdayToken; label: string }>>(
    () => [
      { value: "MON", label: t("watchlists:overview.pipelineSetup.weekdays.monday", "Monday") },
      { value: "TUE", label: t("watchlists:overview.pipelineSetup.weekdays.tuesday", "Tuesday") },
      { value: "WED", label: t("watchlists:overview.pipelineSetup.weekdays.wednesday", "Wednesday") },
      { value: "THU", label: t("watchlists:overview.pipelineSetup.weekdays.thursday", "Thursday") },
      { value: "FRI", label: t("watchlists:overview.pipelineSetup.weekdays.friday", "Friday") },
      { value: "SAT", label: t("watchlists:overview.pipelineSetup.weekdays.saturday", "Saturday") },
      { value: "SUN", label: t("watchlists:overview.pipelineSetup.weekdays.sunday", "Sunday") }
    ],
    [t]
  )
  const scheduleOptions = useMemo<Array<{ value: PipelineWizardScheduleMode; label: string }>>(
    () => [
      { value: "manual", label: t("watchlists:overview.pipelineSetup.schedule.manual", "Manual only") },
      { value: "interval", label: t("watchlists:overview.pipelineSetup.schedule.interval", "Every N hours or minutes") },
      { value: "daily", label: t("watchlists:overview.pipelineSetup.schedule.daily", "Daily") },
      { value: "weekdays", label: t("watchlists:overview.pipelineSetup.schedule.weekdays", "Weekdays") },
      { value: "weekly", label: t("watchlists:overview.pipelineSetup.schedule.weekly", "Weekly") },
      { value: "advanced", label: t("watchlists:overview.pipelineSetup.schedule.advanced", "Advanced cron") }
    ],
    [t]
  )
  const intervalUnitOptions = useMemo<Array<{ value: ScheduleIntervalUnit; label: string }>>(
    () => [
      { value: "hours", label: t("watchlists:overview.pipelineSetup.intervalUnits.hours", "Hours") },
      { value: "minutes", label: t("watchlists:overview.pipelineSetup.intervalUnits.minutes", "Minutes") }
    ],
    [t]
  )

  const validationMessage = (field: string): string => {
    const messages: Record<string, string> = {
      sourceIds: t("watchlists:overview.pipelineSetup.validation.sourcesRequired", "Choose at least one source before continuing."),
      sourceName: t("watchlists:overview.pipelineSetup.validation.sourceNameRequired", "Enter a source name."),
      sourceUrl: t("watchlists:overview.pipelineSetup.validation.sourceUrlInvalid", "Enter a valid HTTP or HTTPS source URL."),
      monitorName: t("watchlists:overview.pipelineSetup.validation.monitorNameRequired", "Enter a monitor name."),
      timezone: t("watchlists:overview.pipelineSetup.validation.timezoneInvalid", "Enter a valid IANA timezone."),
      scheduleIntervalValue: t("watchlists:overview.pipelineSetup.validation.intervalInvalid", "Enter a supported schedule interval."),
      scheduleHour: t("watchlists:overview.pipelineSetup.validation.hourInvalid", "Enter an hour from 0 through 23."),
      scheduleMinute: t("watchlists:overview.pipelineSetup.validation.minuteInvalid", "Enter a minute from 0 through 59."),
      scheduleAdvancedCron: t("watchlists:overview.pipelineSetup.validation.cronInvalid", "Enter a valid five-field cron expression."),
      scheduleAdvancedCronTooFrequent: t("watchlists:overview.pipelineSetup.validation.cronTooFrequent", "Choose a schedule at least five minutes apart."),
      templateName: t("watchlists:overview.pipelineSetup.validation.templateRequired", "Enter a report template."),
      audioSpeakers: t("watchlists:overview.pipelineSetup.validation.castSizeInvalid", "Choose one to four speakers."),
      audioSpeakerVoices: t("watchlists:overview.pipelineSetup.validation.castVoicesRequired", "Choose a voice for every speaker."),
      targetAudioMinutes: t("watchlists:overview.pipelineSetup.validation.audioMinutesInvalid", "Enter a duration from 1 through 60 minutes."),
      emailRecipients: t("watchlists:overview.pipelineSetup.validation.emailInvalid", "Enter at least one valid email address.")
    }
    return messages[field] || t("watchlists:overview.pipelineSetup.validationError", "Review the highlighted fields before continuing.")
  }

  const fieldError = (field: string): React.ReactNode => stepErrors.includes(field)
    ? <span id={`${FIELD_IDS[field] || `pipeline-${field}`}-error`}>{validationMessage(field)}</span>
    : undefined

  const validateStep = (index = currentStep): boolean => {
    const validation = validatePipelineWizardDraft(draftRef.current)
    const stepFields = index === 0
      ? draft.sourceMode === "new" ? ["sourceName", "sourceUrl"] : ["sourceIds"]
      : index === 1
        ? [
            "monitorName",
            "timezone",
            "scheduleIntervalValue",
            "scheduleHour",
            "scheduleMinute",
            "scheduleAdvancedCron",
            "scheduleAdvancedCronTooFrequent"
          ]
        : index === 2
          ? ["templateName", "audioSpeakers", "audioSpeakerVoices", "targetAudioMinutes"]
          : index === 3
            ? ["emailRecipients"]
            : validation.errors
    const errors = validation.errors.filter((error) => stepFields.includes(error))
    setStepErrors(errors)
    return errors.length === 0
  }

  const commitPendingEmailRecipient = () => {
    const recipient = emailRecipientSearchRef.current.trim().toLowerCase()
    if (!recipient || draftRef.current.emailRecipients.includes(recipient)) return
    emailRecipientSearchRef.current = ""
    updateDraft({
      emailRecipients: [...draftRef.current.emailRecipients, recipient]
    })
  }

  const moveNext = () => {
    if (currentStep === 3 && draftRef.current.emailDeliveryEnabled) {
      commitPendingEmailRecipient()
    }
    if (!validateStep()) return
    setCurrentStep((value) => Math.min(LAST_STEP, value + 1))
  }

  const validateAll = (): boolean => {
    const validation = validatePipelineWizardDraft(draftRef.current)
    if (validation.valid) return true
    const first = validation.errors[0]
    setStepErrors(validation.errors)
    if (["sourceIds", "sourceName", "sourceUrl"].includes(first)) setCurrentStep(0)
    else if (["monitorName", "timezone", "scheduleIntervalValue", "scheduleHour", "scheduleMinute", "scheduleAdvancedCron", "scheduleAdvancedCronTooFrequent"].includes(first)) setCurrentStep(1)
    else if (["templateName", "audioSpeakers", "audioSpeakerVoices", "targetAudioMinutes"].includes(first)) setCurrentStep(2)
    else setCurrentStep(3)
    return false
  }

  const runTest = async (
    options: Omit<PipelineWizardTestOptions, "jobId" | "requestGeneration" | "signal">
  ) => {
    if (!validateAll()) return
    if (actionLockRef.current) return
    actionLockRef.current = true
    const generation = ++requestGenerationRef.current
    const controller = new AbortController()
    actionControllerRef.current = controller
    setActionBusy(true)
    setActionError(null)
    setActionMessage(null)
    setTestProjection(null)
    try {
      const result = await onTest?.(
        draftRef.current,
        { ...options, requestGeneration: generation, signal: controller.signal, ...(inactiveJobId ? { jobId: inactiveJobId } : {}) },
        (projection) => {
          if (generation === requestGenerationRef.current) setTestProjection(projection)
        }
      )
      if (generation !== requestGenerationRef.current) return
      const actionResult: PipelineWizardActionResult | undefined = result || undefined
      if (actionResult?.jobId) setInactiveJobId(actionResult.jobId)
      if (actionResult?.briefing) setTestProjection(actionResult.briefing)
      const observedOutcome = actionResult?.briefing
        ? getPipelineWizardBriefingOutcome(actionResult.briefing, options.externalDelivery)
        : undefined
      const resultStatus = observedOutcome?.status || actionResult?.status
      const resultMessage = actionResult?.message || observedOutcome?.message
      if (resultStatus === "failed") {
        setActionError(resultMessage || t("watchlists:overview.pipelineSetup.test.failed", "Test failed. Your draft is saved."))
      } else if (resultStatus === "cancelled") {
        setActionMessage(resultMessage || t("watchlists:overview.pipelineSetup.test.cancelled", "Test cancelled. Your draft is saved."))
      } else if (resultStatus === "running") {
        setActionMessage(resultMessage || t("watchlists:overview.pipelineSetup.test.running", "Test is still running. Monitor it from Overview."))
      } else {
        setActionMessage(t("watchlists:overview.pipelineSetup.test.ready", "Test started. This draft stays inactive until you activate its schedule."))
      }
    } catch (error) {
      if (generation === requestGenerationRef.current) {
        setActionError(
          t(
            "watchlists:overview.pipelineSetup.test.failed",
            "Test failed. Your draft is saved. {{message}}",
            { message: getErrorMessage(error) }
          )
        )
      }
    } finally {
      const activeController = actionControllerRef.current
      if (!activeController || activeController === controller) {
        actionControllerRef.current = null
        actionLockRef.current = false
        setActionBusy(false)
      }
    }
  }

  const activate = async () => {
    if (!validateAll()) return
    if (actionLockRef.current) return
    actionLockRef.current = true
    const generation = ++requestGenerationRef.current
    setActionBusy(true)
    setActionError(null)
    setActionMessage(null)
    try {
      const result = await onActivate?.(draftRef.current, { jobId: inactiveJobId })
      if (generation !== requestGenerationRef.current) return
      if (result && result.jobId) setInactiveJobId(result.jobId)
      setActionMessage(t("watchlists:overview.pipelineSetup.activate.ready", "Schedule activated."))
    } catch (error) {
      if (generation === requestGenerationRef.current) {
        setActionError(
          t(
            "watchlists:overview.pipelineSetup.activate.failed",
            "Schedule could not be activated. Your draft is saved. {{message}}",
            { message: getErrorMessage(error) }
          )
        )
      }
    } finally {
      if (generation === requestGenerationRef.current) {
        actionLockRef.current = false
        setActionBusy(false)
      }
    }
  }

  const testSource = async () => {
    if (!validateStep(0)) return
    if (actionLockRef.current) return
    actionLockRef.current = true
    const generation = ++requestGenerationRef.current
    setActionBusy(true)
    setActionError(null)
    setSourceTestMessage(null)
    try {
      const result = await onTestSource?.(draftRef.current)
      if (generation !== requestGenerationRef.current) return
      setSourceTestResult(result && result.sourceTest ? result.sourceTest : null)
      setSourceTestMessage(
        result && result.status === "cancelled"
          ? t("watchlists:overview.pipelineSetup.sourceTest.cancelled", "Source test cancelled.")
          : result && result.sourceTest
            ? t(
                "watchlists:overview.pipelineSetup.sourceTest.summary",
                "{{ingestable}} ready, {{filtered}} filtered from {{total}} items.",
                {
                  ingestable: result.sourceTest.ingestable,
                  filtered: result.sourceTest.filtered,
                  total: result.sourceTest.total
                }
              )
          : t("watchlists:overview.pipelineSetup.sourceTest.ready", "Source is ready.")
      )
    } catch (error) {
      if (generation === requestGenerationRef.current) {
        setActionError(
          t("watchlists:overview.pipelineSetup.sourceTest.failed", "Source test failed. {{message}}", {
            message: getErrorMessage(error)
          })
        )
      }
    } finally {
      if (generation === requestGenerationRef.current) {
        actionLockRef.current = false
        setActionBusy(false)
      }
    }
  }

  const occurrences = useMemo(() => projectPipelineWizardOccurrences(draft), [draft])
  const receipt = useMemo(() => {
    const payload = toPipelineJobCreatePayload(toBriefingPipelineDraft(draft))
    const contract = payload.output_prefs?.briefing_pipeline
    if (!contract) return null
    return buildBriefingReceiptModel({
      contract,
      sourceCount: draft.sourceMode === "new" ? 1 : draft.sourceIds.length,
      nextRunAt: occurrences.nextRunAt,
      followingRunAt: occurrences.followingRunAt,
      scheduled: draft.scheduleMode !== "manual",
      timezone: draft.timezone || "UTC",
      locale
    })
  }, [draft, locale, occurrences])

  const receiptArtifacts = receipt?.artifacts.map((artifact) =>
    artifact === "show_notes"
      ? t("watchlists:overview.pipelineSetup.receipt.artifacts.showNotes", "show notes")
      : artifact === "audio"
        ? t("watchlists:overview.pipelineSetup.receipt.artifacts.audio", "audio")
        : t("watchlists:overview.pipelineSetup.receipt.artifacts.textReport", "text report")
  ) || []
  const formatList = (values: string[]): string =>
    new Intl.ListFormat(locale, { style: "long", type: "conjunction" }).format(values)
  const receiptDestinations = receipt
    ? [
        t("watchlists:overview.pipelineSetup.delivery.reports", "Reports"),
        ...(receipt.emailRecipients.length > 0
          ? [t("watchlists:overview.pipelineSetup.receipt.destinations.email", "Email: {{recipients}}", {
              recipients: formatList(receipt.emailRecipients)
            })]
          : []),
        ...(receipt.destinations.includes("chatbook")
          ? [t("watchlists:overview.pipelineSetup.receipt.destinations.chatbook", "Chatbook: {{title}}", {
              title: receipt.chatbookTitle || t("watchlists:overview.pipelineSetup.receipt.untitledChatbook", "Untitled")
            })]
          : [])
      ]
    : []
  const receiptProgramFormat = receipt
    ? t(
        `watchlists:overview.pipelineSetup.receipt.formats.${receipt.programFormat}`,
        receipt.programFormat.replaceAll("_", " ")
      )
    : ""
  const receiptSpeakerStyle = receipt?.speakerCount
    ? t(
        `watchlists:overview.pipelineSetup.receipt.speakerStyle.${receipt.speakerCount}`,
        ["", "solo", "two-host", "three-host", "four-host"][receipt.speakerCount] ||
          "{{count}}-host",
        { count: receipt.speakerCount }
      )
    : ""
  const programFormatLabel = receipt
    ? receipt.speakerCount > 0
      ? t(
          "watchlists:overview.pipelineSetup.receipt.programWithSpeakers",
          "{{speakers}} {{format}}",
          { speakers: receiptSpeakerStyle, format: receiptProgramFormat }
        )
      : receiptProgramFormat
    : ""

  const observedStages = testProjection
    ? Object.entries(testProjection.stages)
    : []
  const stageOrder = [
    "collect",
    "select",
    "render_text",
    "persist_text",
    "compose_audio_script",
    "persist_audio_script",
    "generate_audio",
    "persist_audio",
    "deliver"
  ]
  const orderedStages = [...observedStages].sort(([left], [right]) => {
    const leftIndex = stageOrder.indexOf(left)
    const rightIndex = stageOrder.indexOf(right)
    return (leftIndex < 0 ? stageOrder.length : leftIndex) -
      (rightIndex < 0 ? stageOrder.length : rightIndex)
  })
  const stageLabel = (stage: string): string => {
    const labels: Record<string, string> = {
      collect: t("watchlists:overview.pipelineSetup.test.stages.collection", "Collect sources"),
      select: t("watchlists:overview.pipelineSetup.test.stages.selection", "Select updates"),
      render_text: t("watchlists:overview.pipelineSetup.test.stages.text", "Create report"),
      persist_text: t("watchlists:overview.pipelineSetup.test.stages.persistence", "Save report in Reports"),
      compose_audio_script: t("watchlists:overview.pipelineSetup.test.stages.audioScript", "Compose audio script"),
      persist_audio_script: t("watchlists:overview.pipelineSetup.test.stages.audioScriptPersistence", "Save audio script"),
      generate_audio: t("watchlists:overview.pipelineSetup.test.stages.audio", "Create audio"),
      persist_audio: t("watchlists:overview.pipelineSetup.test.stages.audioPersistence", "Save audio in Reports"),
      deliver: t("watchlists:overview.pipelineSetup.test.stages.delivery", "Check test delivery")
    }
    if (labels[stage]) return labels[stage]
    if (stage.startsWith("deliver:")) {
      return t("watchlists:overview.pipelineSetup.test.stages.deliveryAdapter", "Deliver to {{adapter}}", {
        adapter: stage.slice("deliver:".length)
      })
    }
    return t("watchlists:overview.pipelineSetup.test.stages.unknown", "Stage {{stage}}", { stage })
  }
  const stageStatusLabel = (status: string): string => {
    const fallback = status.replaceAll("_", " ").replace(/^./, (value) => value.toUpperCase())
    return t(`watchlists:overview.pipelineSetup.test.status.${status}`, fallback)
  }

  const currentSpeakerCount = Math.max(1, Math.min(4, draft.audioSpeakers.length || 1))
  const intervalMin = draft.scheduleIntervalUnit === "minutes"
    ? INTERVAL_MINUTES_MIN
    : INTERVAL_HOURS_MIN
  const intervalMax = draft.scheduleIntervalUnit === "minutes"
    ? INTERVAL_MINUTES_MAX
    : INTERVAL_HOURS_MAX
  const hasExternalDelivery = draft.emailDeliveryEnabled || draft.chatbookDeliveryEnabled
  const isBusy = submitting || actionBusy

  const errorMessage = stepErrors.includes("sourceIds")
    ? t("watchlists:overview.pipelineSetup.validation.sourcesRequired", "Choose at least one source before continuing.")
    : t("watchlists:overview.pipelineSetup.validationError", "Review the highlighted fields before continuing.")

  const footerActions = currentStep === LAST_STEP
    ? [
        <Button key="cancel" className="min-h-11 w-full whitespace-normal sm:w-auto" onClick={onCancel} disabled={isBusy}>
          {t("common:cancel", "Cancel")}
        </Button>,
        <Button
          key="back"
          className="min-h-11 w-full whitespace-normal sm:w-auto"
          onClick={() => setCurrentStep((value) => Math.max(0, value - 1))}
          disabled={isBusy}
        >
          {t("common:back", "Back")}
        </Button>
      ]
    : [
        <Button key="cancel" className="min-h-11 w-full whitespace-normal sm:w-auto" onClick={onCancel} disabled={isBusy}>
          {t("common:cancel", "Cancel")}
        </Button>,
        currentStep > 0 ? (
          <Button
            key="back"
            className="min-h-11 w-full whitespace-normal sm:w-auto"
            onClick={() => setCurrentStep((value) => Math.max(0, value - 1))}
            disabled={isBusy}
          >
            {t("common:back", "Back")}
          </Button>
        ) : null,
        <Button key="next" type="primary" className="min-h-11 w-full whitespace-normal sm:w-auto" onClick={moveNext} disabled={isBusy}>
          {t(
            "watchlists:overview.pipelineSetup.actions.next",
            "Next: {{step}}",
            {
              step: t(
                `watchlists:overview.pipelineSetup.steps.${STEPS[currentStep + 1]}`,
                STEP_FALLBACKS[STEPS[currentStep + 1]]
              )
            }
          )}
        </Button>
      ]
  const footer = (
    <div className="flex w-full flex-col-reverse gap-2 sm:flex-row sm:flex-wrap sm:justify-end">
      {footerActions}
    </div>
  )

  return (
    <Modal
      open={open}
      title={t("watchlists:overview.pipelineSetup.title", "Set up briefing")}
      onCancel={isBusy ? undefined : onCancel}
      destroyOnHidden
      maskClosable={!isBusy}
      width="min(760px, calc(100vw - 2rem))"
      className="watchlists-pipeline-wizard w-[min(760px,calc(100vw-2rem))]"
      footer={footer}
    >
      <div
        className="max-h-[calc(100dvh-10rem)] space-y-5 overflow-y-auto pe-1"
        data-testid="watchlists-pipeline-scroll-region"
      >
        <nav aria-label={t("watchlists:overview.pipelineSetup.progress", "Briefing setup steps")}>
          <ol className="flex min-w-0 gap-1 overflow-x-auto" role="list">
            {STEPS.map((step, index) => (
              <li key={step} className="min-w-max flex-1">
                <button
                  type="button"
                  className={`min-h-11 w-full rounded-md px-3 py-2 text-start text-sm focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary ${
                    index === currentStep ? "bg-primary/10 font-semibold text-primary" : "text-text-muted"
                  }`}
                  aria-current={index === currentStep ? "step" : undefined}
                  onClick={() => {
                    if (index <= currentStep) setCurrentStep(index)
                  }}
                >
                  {t(`watchlists:overview.pipelineSetup.steps.${step}`, step[0].toUpperCase() + step.slice(1))}
                </button>
              </li>
            ))}
          </ol>
        </nav>

        {stepErrors.length > 0 && (
          <div ref={validationRef} tabIndex={-1} data-testid="watchlists-pipeline-validation-summary">
            <DesignSystemAlert variant="warning" title={errorMessage} />
          </div>
        )}
        {(submitError || actionError) && (
          <DesignSystemAlert
            variant="error"
            title={submitError || actionError || ""}
            data-testid="watchlists-pipeline-action-error"
          />
        )}
        {actionMessage && <DesignSystemAlert variant="info" title={actionMessage} />}

        {currentStep === 0 && (
          <section aria-labelledby="pipeline-sources-heading" className="space-y-4">
            <div>
              <h2 id="pipeline-sources-heading" className="text-base font-semibold">
                {t("watchlists:overview.pipelineSetup.steps.sources", "Sources")}
              </h2>
              <p className="mt-1 text-sm text-text-muted">
                {t("watchlists:overview.pipelineSetup.sources.help", "Choose existing sources or add one, then test the connection.")}
              </p>
            </div>
            <Radio.Group
              className={TOUCH_TARGET_CLASS}
              value={draft.sourceMode}
              onChange={(event) => updateDraft({ sourceMode: event.target.value as PipelineWizardSourceMode })}
            >
              <Space orientation="vertical">
                <Radio value="existing">{t("watchlists:overview.pipelineSetup.source.existing", "Use existing sources")}</Radio>
                <Radio value="new">{t("watchlists:overview.pipelineSetup.source.new", "Add a new source")}</Radio>
              </Space>
            </Radio.Group>
            {draft.sourceMode === "existing" ? (
              <Form layout="vertical">
                <Form.Item
                  label={t("watchlists:overview.pipelineSetup.fields.sources", "Sources")}
                  validateStatus={stepErrors.includes("sourceIds") ? "error" : undefined}
                  help={fieldError("sourceIds")}
                >
                  <Checkbox.Group
                    className="grid max-h-64 gap-2 overflow-y-auto"
                    data-pipeline-field="sourceIds"
                    aria-invalid={stepErrors.includes("sourceIds")}
                    aria-describedby={stepErrors.includes("sourceIds") ? "pipeline-sourceIds-error" : undefined}
                    value={draft.sourceIds}
                    onChange={(values) => updateDraft({ sourceIds: values.map(Number) })}
                  >
                    {sources.map((source) => (
                      <Checkbox key={source.id} value={source.id} className="min-h-11 py-2">
                        {source.name || t("watchlists:overview.pipelineSetup.source.fallback", "Source {{id}}", { id: source.id })}
                      </Checkbox>
                    ))}
                  </Checkbox.Group>
                </Form.Item>
                {sourcesLoading && <p className="text-sm text-text-muted">{t("watchlists:overview.pipelineSetup.sourcesLoading", "Loading sources...")}</p>}
              </Form>
            ) : (
              <Form layout="vertical">
                <Form.Item label={t("watchlists:overview.pipelineSetup.fields.sourceName", "Source name")} htmlFor="pipeline-source-name" validateStatus={stepErrors.includes("sourceName") ? "error" : undefined} help={fieldError("sourceName")}>
                  <Input id="pipeline-source-name" aria-label={t("watchlists:overview.pipelineSetup.fields.sourceName", "Source name")} aria-invalid={stepErrors.includes("sourceName")} aria-describedby={stepErrors.includes("sourceName") ? "pipeline-source-name-error" : undefined} value={draft.sourceName} onChange={(event) => updateDraft({ sourceName: event.target.value })} />
                </Form.Item>
                <Form.Item label={t("watchlists:overview.pipelineSetup.fields.sourceUrl", "Source URL")} htmlFor="pipeline-source-url" validateStatus={stepErrors.includes("sourceUrl") ? "error" : undefined} help={fieldError("sourceUrl")}>
                  <Input id="pipeline-source-url" aria-label={t("watchlists:overview.pipelineSetup.fields.sourceUrl", "Source URL")} aria-invalid={stepErrors.includes("sourceUrl")} aria-describedby={stepErrors.includes("sourceUrl") ? "pipeline-source-url-error" : undefined} value={draft.sourceUrl} onChange={(event) => updateDraft({ sourceUrl: event.target.value })} />
                </Form.Item>
              </Form>
            )}
            <div className="flex flex-wrap items-center gap-3">
              <Button className="min-h-11 w-full whitespace-normal sm:w-auto" onClick={() => void testSource()} loading={actionBusy}>
                {t("watchlists:overview.pipelineSetup.sourceTest.action", "Test source")}
              </Button>
              {sourceTestMessage && <p className="text-sm text-success">{sourceTestMessage}</p>}
            </div>
            {sourceTestResult && sourceTestResult.items.length > 0 && (
              <ul className="space-y-1 text-sm" aria-label={t("watchlists:overview.pipelineSetup.sourceTest.sample", "Source test sample")}>
                {sourceTestResult.items.slice(0, 6).map((item, index) => (
                  <li key={`${item.source_id}-${item.url || item.title || index}`} className="flex min-h-11 items-center justify-between gap-3 border-b border-border py-2">
                    <span>{item.title || item.url || t("watchlists:overview.pipelineSetup.sourceTest.untitled", "Untitled item")}</span>
                    <span className="text-text-muted">
                      {item.decision === "ingest"
                        ? t("watchlists:overview.pipelineSetup.sourceTest.readyItem", "Ready")
                        : t("watchlists:overview.pipelineSetup.sourceTest.filteredItem", "Filtered")}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </section>
        )}

        {currentStep === 1 && (
          <section aria-labelledby="pipeline-cadence-heading" className="space-y-4">
            <h2 id="pipeline-cadence-heading" className="text-base font-semibold">
              {t("watchlists:overview.pipelineSetup.steps.cadence", "Cadence")}
            </h2>
            <Form layout="vertical">
              <Form.Item label={t("watchlists:overview.pipelineSetup.fields.monitorName", "Monitor name")} htmlFor="pipeline-monitor-name" validateStatus={stepErrors.includes("monitorName") ? "error" : undefined} help={fieldError("monitorName")}>
                <Input id="pipeline-monitor-name" aria-label={t("watchlists:overview.pipelineSetup.fields.monitorName", "Monitor name")} aria-invalid={stepErrors.includes("monitorName")} aria-describedby={stepErrors.includes("monitorName") ? "pipeline-monitor-name-error" : undefined} value={draft.monitorName} onChange={(event) => updateDraft({ monitorName: event.target.value })} />
              </Form.Item>
              <div className="grid gap-3 sm:grid-cols-2">
                <Form.Item label={t("watchlists:overview.pipelineSetup.fields.schedule", "Schedule")}>
                  <Select className={TOUCH_TARGET_CLASS} aria-label={t("watchlists:overview.pipelineSetup.fields.schedule", "Schedule")} value={draft.scheduleMode} options={scheduleOptions} onChange={(value) => updateDraft({ scheduleMode: value, createScheduledOutput: value !== "manual", nextRunAt: undefined, followingRunAt: undefined })} />
                </Form.Item>
                <Form.Item label={t("watchlists:overview.pipelineSetup.fields.timezone", "Timezone")} htmlFor="pipeline-timezone" validateStatus={stepErrors.includes("timezone") ? "error" : undefined} help={fieldError("timezone")}>
                  <Input id="pipeline-timezone" aria-label={t("watchlists:overview.pipelineSetup.fields.timezone", "Timezone")} aria-invalid={stepErrors.includes("timezone")} aria-describedby={stepErrors.includes("timezone") ? "pipeline-timezone-error" : undefined} value={draft.timezone} onChange={(event) => updateDraft({ timezone: event.target.value, nextRunAt: undefined, followingRunAt: undefined })} />
                </Form.Item>
              </div>
              {draft.scheduleMode === "interval" && (
                <div className="grid gap-3 sm:grid-cols-2">
                  <Form.Item label={t("watchlists:overview.pipelineSetup.fields.intervalEvery", "Every")} htmlFor="pipeline-schedule-interval" validateStatus={stepErrors.includes("scheduleIntervalValue") ? "error" : undefined} help={fieldError("scheduleIntervalValue")}>
                    <InputNumber id="pipeline-schedule-interval" aria-label={t("watchlists:overview.pipelineSetup.fields.intervalEvery", "Every")} aria-invalid={stepErrors.includes("scheduleIntervalValue")} aria-describedby={stepErrors.includes("scheduleIntervalValue") ? "pipeline-schedule-interval-error" : undefined} className={`w-full ${TOUCH_TARGET_CLASS}`} min={intervalMin} max={intervalMax} precision={0} value={draft.scheduleIntervalValue} onChange={(value) => updateDraft({ scheduleIntervalValue: Number(value), nextRunAt: undefined, followingRunAt: undefined })} />
                  </Form.Item>
                  <Form.Item label={t("watchlists:overview.pipelineSetup.fields.intervalUnit", "Interval unit")}>
                    <Select className={TOUCH_TARGET_CLASS} aria-label={t("watchlists:overview.pipelineSetup.fields.intervalUnit", "Interval unit")} value={draft.scheduleIntervalUnit} options={intervalUnitOptions} onChange={(value) => updateDraft({ scheduleIntervalUnit: value, nextRunAt: undefined, followingRunAt: undefined })} />
                  </Form.Item>
                </div>
              )}
              {(["daily", "weekdays", "weekly"] as PipelineWizardScheduleMode[]).includes(draft.scheduleMode) && (
                <div className="grid gap-3 sm:grid-cols-3">
                  {draft.scheduleMode === "weekly" && (
                    <Form.Item label={t("watchlists:overview.pipelineSetup.fields.weekday", "Weekday")}>
                      <Select className={TOUCH_TARGET_CLASS} aria-label={t("watchlists:overview.pipelineSetup.fields.weekday", "Weekday")} value={draft.scheduleWeekday} options={weekdayOptions} onChange={(value) => updateDraft({ scheduleWeekday: value, nextRunAt: undefined, followingRunAt: undefined })} />
                    </Form.Item>
                  )}
                  <Form.Item label={t("watchlists:overview.pipelineSetup.fields.hour", "Hour")} htmlFor="pipeline-schedule-hour" validateStatus={stepErrors.includes("scheduleHour") ? "error" : undefined} help={fieldError("scheduleHour")}>
                    <InputNumber id="pipeline-schedule-hour" aria-label={t("watchlists:overview.pipelineSetup.fields.hour", "Hour")} aria-invalid={stepErrors.includes("scheduleHour")} aria-describedby={stepErrors.includes("scheduleHour") ? "pipeline-schedule-hour-error" : undefined} className={`w-full ${TOUCH_TARGET_CLASS}`} min={0} max={23} precision={0} value={draft.scheduleHour} onChange={(value) => updateDraft({ scheduleHour: Number(value), nextRunAt: undefined, followingRunAt: undefined })} />
                  </Form.Item>
                  <Form.Item label={t("watchlists:overview.pipelineSetup.fields.minute", "Minute")} htmlFor="pipeline-schedule-minute" validateStatus={stepErrors.includes("scheduleMinute") ? "error" : undefined} help={fieldError("scheduleMinute")}>
                    <InputNumber id="pipeline-schedule-minute" aria-label={t("watchlists:overview.pipelineSetup.fields.minute", "Minute")} aria-invalid={stepErrors.includes("scheduleMinute")} aria-describedby={stepErrors.includes("scheduleMinute") ? "pipeline-schedule-minute-error" : undefined} className={`w-full ${TOUCH_TARGET_CLASS}`} min={0} max={59} precision={0} value={draft.scheduleMinute} onChange={(value) => updateDraft({ scheduleMinute: Number(value), nextRunAt: undefined, followingRunAt: undefined })} />
                  </Form.Item>
                </div>
              )}
              {draft.scheduleMode === "advanced" && (
                <details className="rounded-md border border-border px-3 py-2">
                  <summary className="min-h-11 cursor-pointer py-3 font-medium">{t("watchlists:overview.pipelineSetup.advancedCron", "Advanced cron")}</summary>
                  <Form.Item label={t("watchlists:overview.pipelineSetup.fields.cronExpression", "Cron expression")} htmlFor="pipeline-schedule-cron" validateStatus={stepErrors.some((error) => error.startsWith("scheduleAdvancedCron")) ? "error" : undefined} help={fieldError(stepErrors.includes("scheduleAdvancedCronTooFrequent") ? "scheduleAdvancedCronTooFrequent" : "scheduleAdvancedCron")}>
                    <Input id="pipeline-schedule-cron" aria-label={t("watchlists:overview.pipelineSetup.fields.cronExpression", "Cron expression")} aria-invalid={stepErrors.some((error) => error.startsWith("scheduleAdvancedCron"))} aria-describedby={stepErrors.some((error) => error.startsWith("scheduleAdvancedCron")) ? "pipeline-schedule-cron-error" : undefined} value={draft.scheduleAdvancedCron} placeholder="0 8 * * MON-FRI" onChange={(event) => updateDraft({ scheduleAdvancedCron: event.target.value, nextRunAt: undefined, followingRunAt: undefined })} />
                  </Form.Item>
                </details>
              )}
            </Form>
            <div className="border-t border-border pt-3 text-sm">
              <span className="font-medium">{t("watchlists:overview.pipelineSetup.nextOccurrence", "Next occurrence")}</span>{" "}
              {receipt?.nextRunLabel
                ? t("watchlists:overview.pipelineSetup.receipt.schedule.scheduled", "{{date}} ({{timezone}})", {
                    date: receipt.nextRunLabel,
                    timezone: receipt.timezone
                  })
                : draft.scheduleMode === "manual"
                  ? t("watchlists:overview.pipelineSetup.manualOccurrence", "Manual only")
                  : t("watchlists:overview.pipelineSetup.nextOccurrencePending", "Save a valid schedule to calculate the exact time.")}
            </div>
          </section>
        )}

        {currentStep === 2 && (
          <section aria-labelledby="pipeline-briefing-heading" className="space-y-5">
            <div>
              <h2 id="pipeline-briefing-heading" className="text-lg font-semibold">
                {t("watchlists:overview.pipelineSetup.briefing.question", "What are you making?")}
              </h2>
              <p className="mt-1 text-sm text-text-muted">{t("watchlists:overview.pipelineSetup.briefing.help", "Choose the editorial shape. Every option stays grounded in your selected sources.")}</p>
            </div>
            <Radio.Group
              className="grid gap-2 sm:grid-cols-2"
              value={draft.programFormat}
              onChange={(event) => {
                const programFormat = event.target.value as WatchlistProgramFormat
                updateDraft({
                  programFormat,
                  outcomeNoun: programFormat === "concise_briefing" ? "briefing" : "episode",
                  showNotes: programFormat !== "concise_briefing"
                })
              }}
            >
              {PROGRAM_FORMATS.map((format) => (
                <Radio key={format.value} value={format.value} className="min-h-11 rounded-md border border-border px-3 py-2">
                  {t(`watchlists:overview.pipelineSetup.formats.${format.value}`, format.label)}
                </Radio>
              ))}
            </Radio.Group>
            {draft.programFormat !== "concise_briefing" && (
              <Form layout="vertical">
                <Form.Item label={t("watchlists:overview.pipelineSetup.fields.showName", "Show name")}>
                  <Input aria-label={t("watchlists:overview.pipelineSetup.fields.showName", "Show name")} value={draft.showName} onChange={(event) => updateDraft({ showName: event.target.value })} />
                </Form.Item>
                <Form.Item label={t("watchlists:overview.pipelineSetup.fields.premise", "Show premise")}>
                  <Input.TextArea aria-label={t("watchlists:overview.pipelineSetup.fields.premise", "Show premise")} value={draft.premise} rows={3} onChange={(event) => updateDraft({ premise: event.target.value })} />
                </Form.Item>
              </Form>
            )}
            <div className="flex min-h-11 items-center justify-between gap-3 border-t border-border pt-4">
              <div>
                <p className="font-medium">{t("watchlists:overview.pipelineSetup.fields.includeAudio", "Audio")}</p>
                <p className="text-sm text-text-muted">{t("watchlists:overview.pipelineSetup.fields.audioHelp", "Add a playable spoken version.")}</p>
              </div>
              <Switch className={TOUCH_TARGET_CLASS} aria-label={t("watchlists:overview.pipelineSetup.fields.includeAudio", "Audio")} checked={draft.audioEnabled} onChange={(checked) => updateDraft({ audioEnabled: checked, audioSpeakers: checked ? createSpeakers(currentSpeakerCount, draft.audioSpeakers, getSpeakerLabel, getSpeakerRole) : [] })} />
            </div>
            {draft.audioEnabled && (
              <div className="space-y-4">
                <div className="grid gap-3 sm:grid-cols-2">
                  <Form.Item label={t("watchlists:overview.pipelineSetup.fields.audioMinutes", "Target duration in minutes")} htmlFor="pipeline-audio-minutes" validateStatus={stepErrors.includes("targetAudioMinutes") ? "error" : undefined} help={fieldError("targetAudioMinutes")}>
                    <Input id="pipeline-audio-minutes" aria-label={t("watchlists:overview.pipelineSetup.fields.audioMinutes", "Target duration in minutes")} aria-invalid={stepErrors.includes("targetAudioMinutes")} aria-describedby={stepErrors.includes("targetAudioMinutes") ? "pipeline-audio-minutes-error" : undefined} type="number" min={1} max={60} value={draft.targetAudioMinutes} onChange={(event) => updateDraft({ targetAudioMinutes: Number(event.target.value) })} />
                  </Form.Item>
                  <Form.Item label={t("watchlists:overview.pipelineSetup.fields.speakerCount", "Cast size")}>
                    <Select className={TOUCH_TARGET_CLASS} aria-label={t("watchlists:overview.pipelineSetup.fields.speakerCount", "Cast size")} value={currentSpeakerCount} options={[1, 2, 3, 4].map((value) => ({ value, label: new Intl.NumberFormat(locale).format(value) }))} onChange={(value) => updateDraft({ audioSpeakers: createSpeakers(value, draft.audioSpeakers, getSpeakerLabel, getSpeakerRole) })} />
                  </Form.Item>
                </div>
                <div className="divide-y divide-border border-y border-border">
                  {draft.audioSpeakers.map((speaker, index) => (
                    <div key={speaker.id || index} className="grid gap-3 py-3 sm:grid-cols-2">
                      <Form.Item label={t("watchlists:overview.pipelineSetup.speaker.labelField", "Speaker {{index}} label", { index: index + 1 })}>
                        <Input aria-label={t("watchlists:overview.pipelineSetup.speaker.labelField", "Speaker {{index}} label", { index: index + 1 })} value={speaker.label} onChange={(event) => updateSpeaker(index, { label: event.target.value })} />
                      </Form.Item>
                      <Form.Item label={t("watchlists:overview.pipelineSetup.speaker.roleField", "Speaker {{index}} role", { index: index + 1 })}>
                        <Input aria-label={t("watchlists:overview.pipelineSetup.speaker.roleField", "Speaker {{index}} role", { index: index + 1 })} value={speaker.role} onChange={(event) => updateSpeaker(index, { role: event.target.value })} />
                      </Form.Item>
                    </div>
                  ))}
                </div>
              </div>
            )}
            <details ref={advancedBriefingRef} className="rounded-md border border-border px-3 py-2">
              <summary className="min-h-11 cursor-pointer py-3 font-medium">{t("watchlists:overview.pipelineSetup.briefing.advanced", "Advanced briefing settings")}</summary>
              <div className="space-y-3 pt-2">
                <Form layout="vertical">
                  <Form.Item label={t("watchlists:overview.pipelineSetup.fields.template", "Report template")} htmlFor="pipeline-template-name" validateStatus={stepErrors.includes("templateName") ? "error" : undefined} help={fieldError("templateName")}>
                    <Input id="pipeline-template-name" aria-label={t("watchlists:overview.pipelineSetup.fields.template", "Report template")} aria-invalid={stepErrors.includes("templateName")} aria-describedby={stepErrors.includes("templateName") ? "pipeline-template-name-error" : undefined} value={draft.templateName} onChange={(event) => updateDraft({ templateName: event.target.value })} />
                  </Form.Item>
                  {draft.audioEnabled && (
                    <div className="grid gap-3 sm:grid-cols-2">
                      <Form.Item label={t("watchlists:overview.pipelineSetup.fields.audioProvider", "Audio provider override")}>
                        <Input aria-label={t("watchlists:overview.pipelineSetup.fields.audioProvider", "Audio provider override")} value={draft.audioProvider} onChange={(event) => updateDraft({ audioProvider: event.target.value })} />
                      </Form.Item>
                      <Form.Item label={t("watchlists:overview.pipelineSetup.fields.audioModel", "Audio model override")}>
                        <Input aria-label={t("watchlists:overview.pipelineSetup.fields.audioModel", "Audio model override")} value={draft.audioModel} onChange={(event) => updateDraft({ audioModel: event.target.value })} />
                      </Form.Item>
                    </div>
                  )}
                  {draft.audioEnabled && draft.audioSpeakers.map((speaker, index) => (
                    <div key={speaker.id || index} className="grid gap-3 sm:grid-cols-2">
                      <Form.Item
                        label={t("watchlists:overview.pipelineSetup.speaker.voiceField", "Speaker {{index}} voice", { index: index + 1 })}
                        validateStatus={stepErrors.includes("audioSpeakerVoices") && !speaker.voice.trim() ? "error" : undefined}
                        help={stepErrors.includes("audioSpeakerVoices") && !speaker.voice.trim()
                          ? <span id={`pipeline-speaker-${index + 1}-voice-error`}>{validationMessage("audioSpeakerVoices")}</span>
                          : undefined}
                      >
                        <Select
                          id={`pipeline-speaker-${index + 1}-voice`}
                          data-pipeline-speaker-voice
                          className={TOUCH_TARGET_CLASS}
                          aria-label={t("watchlists:overview.pipelineSetup.speaker.voiceField", "Speaker {{index}} voice", { index: index + 1 })}
                          aria-invalid={stepErrors.includes("audioSpeakerVoices") && !speaker.voice.trim()}
                          aria-describedby={stepErrors.includes("audioSpeakerVoices") && !speaker.voice.trim() ? `pipeline-speaker-${index + 1}-voice-error` : undefined}
                          value={speaker.voice || undefined}
                          options={DEFAULT_SPEAKER_VOICES.map((voice) => ({ value: voice, label: voice }))}
                          onChange={(value) => updateSpeaker(index, { voice: value })}
                        />
                      </Form.Item>
                      <Form.Item label={t("watchlists:overview.pipelineSetup.speaker.personaField", "Speaker {{index}} persona", { index: index + 1 })}>
                        <Input aria-label={t("watchlists:overview.pipelineSetup.speaker.personaField", "Speaker {{index}} persona", { index: index + 1 })} value={speaker.persona} onChange={(event) => updateSpeaker(index, { persona: event.target.value })} />
                      </Form.Item>
                    </div>
                  ))}
                  {draft.programFormat === "custom" && (
                    <Form.Item label={t("watchlists:overview.pipelineSetup.fields.customInstructions", "Custom editorial instructions")}>
                      <Input.TextArea aria-label={t("watchlists:overview.pipelineSetup.fields.customInstructions", "Custom editorial instructions")} value={draft.customInstructions} rows={4} onChange={(event) => updateDraft({ customInstructions: event.target.value })} />
                    </Form.Item>
                  )}
                </Form>
                {onPreview && (
                  <Button className="min-h-11" onClick={() => onPreview(draft)} loading={previewLoading}>
                    {t("watchlists:overview.pipelineSetup.preview.generate", "Preview report")}
                  </Button>
                )}
                {(previewError || previewRendered || previewRunId != null || previewWarnings.length > 0) && (
                  <details>
                    <summary className="min-h-11 cursor-pointer py-3">{t("watchlists:overview.pipelineSetup.preview.diagnostics", "Preview diagnostics")}</summary>
                    {previewError && <DesignSystemAlert variant="warning" title={previewError} />}
                    {previewRunId != null && <p className="text-sm text-text-muted">{t("watchlists:overview.pipelineSetup.preview.context", "Preview run: {{runId}}", { runId: previewRunId })}</p>}
                    {previewWarnings.map((warning) => <p key={warning} className="text-sm text-text-muted">{warning}</p>)}
                    {previewRendered && <pre className="max-h-48 overflow-auto whitespace-pre-wrap text-xs">{previewRendered}</pre>}
                  </details>
                )}
              </div>
            </details>
          </section>
        )}

        {currentStep === 3 && (
          <section aria-labelledby="pipeline-delivery-heading" className="space-y-5">
            <div>
              <h2 id="pipeline-delivery-heading" className="text-base font-semibold">{t("watchlists:overview.pipelineSetup.steps.delivery", "Delivery")}</h2>
              <p className="mt-1 text-sm text-text-muted">{t("watchlists:overview.pipelineSetup.delivery.help", "Reports is always included. External delivery waits until selected artifacts are ready.")}</p>
            </div>
            <div className="flex min-h-11 items-center justify-between border-y border-border py-3">
              <div><p className="font-medium">{t("watchlists:overview.pipelineSetup.delivery.reports", "Reports")}</p><p className="text-sm text-text-muted">{t("watchlists:overview.pipelineSetup.delivery.required", "Required storage")}</p></div>
              <span className="text-sm font-medium text-success">{t("watchlists:overview.pipelineSetup.delivery.alwaysOn", "Always on")}</span>
            </div>
            <Form layout="vertical">
              <div className="flex min-h-11 items-center justify-between gap-3">
                <span className="font-medium">{t("watchlists:overview.pipelineSetup.fields.emailDelivery", "Email")}</span>
                <Switch className={TOUCH_TARGET_CLASS} aria-label={t("watchlists:overview.pipelineSetup.fields.emailDelivery", "Email")} checked={draft.emailDeliveryEnabled} onChange={(checked) => updateDraft({ emailDeliveryEnabled: checked })} />
              </div>
              {draft.emailDeliveryEnabled && (
                <Form.Item label={t("watchlists:overview.pipelineSetup.fields.emailRecipients", "Email recipients")} htmlFor="pipeline-email-recipients" validateStatus={stepErrors.includes("emailRecipients") ? "error" : undefined} help={fieldError("emailRecipients")}>
                  <Select
                    id="pipeline-email-recipients"
                    className={TOUCH_TARGET_CLASS}
                    mode="tags"
                    tokenSeparators={[","]}
                    aria-label={t("watchlists:overview.pipelineSetup.fields.emailRecipients", "Email recipients")}
                    aria-invalid={stepErrors.includes("emailRecipients")}
                    aria-describedby={stepErrors.includes("emailRecipients") ? "pipeline-email-recipients-error" : undefined}
                    value={draft.emailRecipients}
                    onSearch={(value) => {
                      emailRecipientSearchRef.current = value
                    }}
                    onBlur={commitPendingEmailRecipient}
                    onChange={(value) => {
                      emailRecipientSearchRef.current = ""
                      updateDraft({ emailRecipients: value })
                    }}
                  />
                </Form.Item>
              )}
              <div className="flex min-h-11 items-center justify-between gap-3 border-t border-border pt-3">
                <span className="font-medium">{t("watchlists:overview.pipelineSetup.fields.chatbookDelivery", "Chatbook")}</span>
                <Switch className={TOUCH_TARGET_CLASS} aria-label={t("watchlists:overview.pipelineSetup.fields.chatbookDelivery", "Chatbook")} checked={draft.chatbookDeliveryEnabled} onChange={(checked) => updateDraft({ chatbookDeliveryEnabled: checked })} />
              </div>
              {draft.chatbookDeliveryEnabled && (
                <Form.Item label={t("watchlists:overview.pipelineSetup.fields.chatbookTitle", "Chatbook title")}>
                  <Input aria-label={t("watchlists:overview.pipelineSetup.fields.chatbookTitle", "Chatbook title")} value={draft.chatbookTitle} onChange={(event) => updateDraft({ chatbookTitle: event.target.value })} />
                </Form.Item>
              )}
            </Form>
          </section>
        )}

        {currentStep === 4 && (
          <section aria-labelledby="pipeline-test-heading" className="space-y-5">
            <div>
              <h2 id="pipeline-test-heading" className="text-base font-semibold">{t("watchlists:overview.pipelineSetup.steps.test", "Test")}</h2>
              <p className="mt-1 text-sm text-text-muted">
                {draft.audioEnabled
                  ? t("watchlists:overview.pipelineSetup.test.providerDisclosure", "Tests use your configured LLM and text-to-speech providers. A sample is limited to 60 seconds; a full test uses the target duration.")
                  : t("watchlists:overview.pipelineSetup.test.textProviderDisclosure", "Tests use your configured LLM provider. Text-only tests do not use text-to-speech.")}
              </p>
            </div>
            <div className="border-y border-border py-4" data-testid="watchlists-pipeline-receipt">
              <p className="font-medium">{t("watchlists:overview.pipelineSetup.test.receipt", "Activation receipt")}</p>
              {receipt && (
                <dl className="mt-3 grid gap-x-4 gap-y-2 text-sm sm:grid-cols-[minmax(8rem,auto)_1fr]">
                  <dt className="font-medium">{t("watchlists:overview.pipelineSetup.receipt.labels.schedule", "Schedule")}</dt>
                  <dd className="text-text-muted">
                    {receipt.nextRunLabel
                      ? t("watchlists:overview.pipelineSetup.receipt.schedule.scheduled", "{{date}} ({{timezone}})", { date: receipt.nextRunLabel, timezone: receipt.timezone })
                      : receipt.scheduleMode === "scheduled"
                        ? t("watchlists:overview.pipelineSetup.nextOccurrencePending", "Save a valid schedule to calculate the exact time.")
                        : t("watchlists:overview.pipelineSetup.receipt.schedule.manual", "Manual only")}
                  </dd>
                  <dt className="font-medium">{t("watchlists:overview.pipelineSetup.receipt.labels.sources", "Sources")}</dt>
                  <dd className="text-text-muted">
                    {t("watchlists:overview.pipelineSetup.receipt.sources", "{{count}} sources", {
                      count: receipt.sourceCount
                    })}
                  </dd>
                  <dt className="font-medium">{t("watchlists:overview.pipelineSetup.receipt.labels.artifacts", "Artifacts")}</dt>
                  <dd className="text-text-muted">{formatList(receiptArtifacts)}</dd>
                  <dt className="font-medium">{t("watchlists:overview.pipelineSetup.receipt.labels.format", "Format")}</dt>
                  <dd className="text-text-muted">{programFormatLabel}</dd>
                  {receipt.showName && (
                    <>
                      <dt className="font-medium">{t("watchlists:overview.pipelineSetup.receipt.labels.showName", "Show name")}</dt>
                      <dd className="text-text-muted">{receipt.showName}</dd>
                    </>
                  )}
                  {draft.audioEnabled && (
                    <>
                      <dt className="font-medium">{t("watchlists:overview.pipelineSetup.receipt.labels.cast", "Cast")}</dt>
                      <dd className="text-text-muted">{t("watchlists:overview.pipelineSetup.receipt.cast", "{{count}} speakers", { count: receipt.speakerCount })}</dd>
                      <dt className="font-medium">{t("watchlists:overview.pipelineSetup.receipt.labels.duration", "Target duration")}</dt>
                      <dd className="text-text-muted">{t("watchlists:overview.pipelineSetup.receipt.duration", "targeting {{count}} minutes", { count: receipt.targetMinutes || 0 })}</dd>
                    </>
                  )}
                  <dt className="font-medium">{t("watchlists:overview.pipelineSetup.receipt.labels.destinations", "Destinations")}</dt>
                  <dd className="text-text-muted">{formatList(receiptDestinations)}</dd>
                </dl>
              )}
              {receipt?.hasDstChange && (
                <p className="mt-3 text-sm text-text-muted">
                  {t("watchlists:overview.pipelineSetup.receipt.dstChange", "The following run uses {{timezoneName}} after the daylight-saving offset change: {{date}}.", {
                    timezoneName: receipt.followingTimezoneAbbreviation,
                    date: receipt.followingRunLabel
                  })}
                </p>
              )}
            </div>
            {orderedStages.length > 0 && (
              <ol className="space-y-2 text-sm" aria-label={t("watchlists:overview.pipelineSetup.test.progress", "Test progress")}>
                {orderedStages.map(([stage, state]) => (
                  <li key={stage} className="flex min-h-11 items-center justify-between gap-3 border-b border-border py-2">
                    <span>{stageLabel(stage)}</span>
                    <span className="text-text-muted">{stageStatusLabel(state.status)}</span>
                  </li>
                ))}
              </ol>
            )}
            <div className="grid gap-2 sm:grid-cols-2">
              {draft.audioEnabled ? (
                <>
                  <Button className="min-h-11 w-full whitespace-normal" onClick={() => void runTest({ externalDelivery: false, audioSampleSeconds: 60 })} loading={actionBusy}>
                    {t("watchlists:overview.pipelineSetup.test.sample", "Generate 60-second sample")}
                  </Button>
                  <Button className="min-h-11 w-full whitespace-normal" onClick={() => void runTest({ externalDelivery: false, audioSampleSeconds: null })} disabled={isBusy}>
                    {t("watchlists:overview.pipelineSetup.test.full", "Generate full test episode")}
                  </Button>
                </>
              ) : (
                <Button className="min-h-11 w-full whitespace-normal" onClick={() => void runTest({ externalDelivery: false, audioSampleSeconds: null })} loading={actionBusy}>
                  {t("watchlists:overview.pipelineSetup.test.text", "Generate test report")}
                </Button>
              )}
              <Button className="min-h-11 w-full whitespace-normal" onClick={() => void runTest({ externalDelivery: true, audioSampleSeconds: draft.audioEnabled ? 60 : null })} disabled={!hasExternalDelivery || isBusy}>
                {t("watchlists:overview.pipelineSetup.test.send", "Send test")}
              </Button>
              <Button type="primary" className="min-h-11 w-full whitespace-normal" onClick={() => void activate()} loading={actionBusy}>
                {t("watchlists:overview.pipelineSetup.test.activate", "Activate schedule")}
              </Button>
            </div>
          </section>
        )}
      </div>
    </Modal>
  )
}
