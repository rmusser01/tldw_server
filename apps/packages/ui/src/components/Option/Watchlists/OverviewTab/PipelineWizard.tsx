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
import type { WatchlistProgramFormat, WatchlistSource } from "@/types/watchlists"
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
}

export interface PipelineWizardActionResult {
  jobId?: number
  runId?: number
  status?: "ready" | "active" | "cancelled"
  message?: string
}

interface PipelineWizardProps {
  open: boolean
  sources?: WatchlistSource[]
  sourcesLoading?: boolean
  submitting?: boolean
  submitError?: string | null
  initialStep?: PipelineWizardStep | number
  initialDraft?: Partial<PipelineWizardDraft>
  onCancel: () => void
  onSaveDraft?: (draft: PipelineWizardDraft) => void | Promise<void>
  onTest?: (
    draft: PipelineWizardDraft,
    options: PipelineWizardTestOptions
  ) => PipelineWizardActionResult | void | Promise<PipelineWizardActionResult | void>
  onActivate?: (
    draft: PipelineWizardDraft,
    options: { jobId?: number }
  ) => PipelineWizardActionResult | void | Promise<PipelineWizardActionResult | void>
  onTestSource?: (
    draft: PipelineWizardDraft
  ) => PipelineWizardActionResult | void | Promise<PipelineWizardActionResult | void>
  previewLoading?: boolean
  previewError?: string | null
  previewRendered?: string | null
  previewRunId?: number | null
  previewWarnings?: string[]
  onPreview?: (draft: PipelineWizardDraft) => void
}

export type { PipelineWizardProps }

const STEPS: PipelineWizardStep[] = ["sources", "cadence", "briefing", "delivery", "test"]
const LAST_STEP = STEPS.length - 1
const DEFAULT_SPEAKER_VOICES = ["alloy", "nova", "echo", "fable"]

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
  speakerLabel: (index: number) => string = (index) => `Speaker ${index}`
): PipelineWizardAudioSpeakerDraft[] =>
  Array.from({ length: Math.max(1, Math.min(4, count)) }, (_item, index) => {
    const current = existing[index]
    return {
      id: current?.id || `speaker_${index + 1}`,
      label: current?.label || speakerLabel(index + 1),
      role: current?.role || (index === 0 ? "host" : "speaker"),
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

const zonedParts = (date: Date, formatter: Intl.DateTimeFormat) => {
  const parts = formatter.formatToParts(date)
  const get = (type: Intl.DateTimeFormatPartTypes) =>
    parts.find((part) => part.type === type)?.value || ""
  return {
    weekday: get("weekday").toUpperCase(),
    hour: Number(get("hour")),
    minute: Number(get("minute"))
  }
}

const projectNextRunAt = (draft: PipelineWizardDraft, from = new Date()): string | undefined => {
  if (draft.nextRunAt) return draft.nextRunAt
  if (draft.scheduleMode === "manual" || draft.scheduleMode === "advanced") return undefined
  const timezone = draft.timezone || "UTC"
  let formatter: Intl.DateTimeFormat
  try {
    formatter = new Intl.DateTimeFormat("en-US", {
      timeZone: timezone,
      weekday: "short",
      hour: "numeric",
      minute: "numeric",
      hourCycle: "h23"
    })
    formatter.format(from)
  } catch {
    return undefined
  }
  const candidate = new Date(Math.ceil((from.getTime() + 1) / 60_000) * 60_000)
  const maxMinutes = 8 * 24 * 60
  for (let index = 0; index < maxMinutes; index += 1) {
    const parts = zonedParts(candidate, formatter)
    const matches = draft.scheduleMode === "interval"
      ? draft.scheduleIntervalUnit === "minutes"
        ? parts.minute % Math.max(1, draft.scheduleIntervalValue) === 0
        : parts.hour % Math.max(1, draft.scheduleIntervalValue) === 0 &&
          parts.minute === draft.scheduleMinute
      : parts.hour === draft.scheduleHour &&
        parts.minute === draft.scheduleMinute &&
        (draft.scheduleMode === "daily" ||
          (draft.scheduleMode === "weekdays" && !["SAT", "SUN"].includes(parts.weekday)) ||
          (draft.scheduleMode === "weekly" && parts.weekday === draft.scheduleWeekday))
    if (matches) return candidate.toISOString()
    candidate.setUTCMinutes(candidate.getUTCMinutes() + 1)
  }
  return undefined
}

export const PipelineWizard: React.FC<PipelineWizardProps> = ({
  open,
  sources = [],
  sourcesLoading = false,
  submitting = false,
  submitError = null,
  initialStep,
  initialDraft,
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
  const { t } = useTranslation(["watchlists", "common"])
  const initial = useMemo(
    () => mergePipelineWizardDraft(initialDraft),
    [initialDraft]
  )
  const [currentStep, setCurrentStep] = useState(() => stepIndex(initialStep))
  const [draft, setDraft] = useState<PipelineWizardDraft>(initial)
  const draftRef = useRef(draft)
  const wasOpenRef = useRef(false)
  const validationRef = useRef<HTMLDivElement | null>(null)
  const [stepErrors, setStepErrors] = useState<string[]>([])
  const [actionBusy, setActionBusy] = useState(false)
  const [actionError, setActionError] = useState<string | null>(null)
  const [actionMessage, setActionMessage] = useState<string | null>(null)
  const [sourceTestMessage, setSourceTestMessage] = useState<string | null>(null)
  const [inactiveJobId, setInactiveJobId] = useState<number | undefined>()

  useEffect(() => {
    if (open && !wasOpenRef.current) {
      const next = mergePipelineWizardDraft(initialDraft)
      draftRef.current = next
      setDraft(next)
      setCurrentStep(stepIndex(initialStep))
      setStepErrors([])
      setActionError(null)
      setActionMessage(null)
      setSourceTestMessage(null)
      setInactiveJobId(undefined)
    }
    wasOpenRef.current = open
  }, [initialDraft, initialStep, open])

  useEffect(() => {
    if (!open || draft.sourceMode !== "existing" || draft.sourceIds.length > 0) return
    if (sources.length !== 1) return
    const next = { ...draftRef.current, sourceIds: [sources[0].id] }
    draftRef.current = next
    setDraft(next)
  }, [draft.sourceIds.length, draft.sourceMode, open, sources])

  useEffect(() => {
    if (stepErrors.length === 0) return
    validationRef.current?.focus()
  }, [currentStep, stepErrors])

  const updateDraft = (patch: Partial<PipelineWizardDraft>) => {
    const next = { ...draftRef.current, ...patch }
    draftRef.current = next
    setDraft(next)
    setStepErrors([])
    setActionError(null)
    void Promise.resolve(onSaveDraft?.(next)).catch((error) => {
      if (!isAbortError(error)) setActionError(getErrorMessage(error))
    })
  }

  const getSpeakerLabel = (index: number) =>
    t("watchlists:overview.pipelineSetup.speaker.defaultLabel", "Speaker {{index}}", { index })

  const updateSpeaker = (index: number, patch: Partial<PipelineWizardAudioSpeakerDraft>) => {
    const speakers = createSpeakers(
      draft.audioSpeakers.length || 1,
      draft.audioSpeakers,
      getSpeakerLabel
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
          ? ["templateName", "audioSpeakers", "audioSpeakerIds", "audioSpeakerVoices", "targetAudioMinutes"]
          : index === 3
            ? ["emailRecipients"]
            : validation.errors
    const errors = validation.errors.filter((error) => stepFields.includes(error))
    setStepErrors(errors)
    return errors.length === 0
  }

  const moveNext = () => {
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
    else if (["templateName", "audioSpeakers", "audioSpeakerIds", "audioSpeakerVoices", "targetAudioMinutes"].includes(first)) setCurrentStep(2)
    else setCurrentStep(3)
    return false
  }

  const runTest = async (options: Omit<PipelineWizardTestOptions, "jobId">) => {
    if (!validateAll()) return
    setActionBusy(true)
    setActionError(null)
    setActionMessage(null)
    try {
      const result = await onTest?.(
        draftRef.current,
        { ...options, ...(inactiveJobId ? { jobId: inactiveJobId } : {}) }
      )
      if (result && result.jobId) setInactiveJobId(result.jobId)
      setActionMessage(
        result && result.status === "cancelled"
          ? t("watchlists:overview.pipelineSetup.test.cancelled", "Test cancelled. Your draft is saved.")
          : t("watchlists:overview.pipelineSetup.test.ready", "Test started. This draft stays inactive until you activate its schedule.")
      )
    } catch (error) {
      if (!isAbortError(error)) {
        setActionError(
          t(
            "watchlists:overview.pipelineSetup.test.failed",
            "Test failed. Your draft is saved. {{message}}",
            { message: getErrorMessage(error) }
          )
        )
      }
    } finally {
      setActionBusy(false)
    }
  }

  const activate = async () => {
    if (!validateAll()) return
    setActionBusy(true)
    setActionError(null)
    setActionMessage(null)
    try {
      const result = await onActivate?.(draftRef.current, { jobId: inactiveJobId })
      if (result && result.jobId) setInactiveJobId(result.jobId)
      setActionMessage(t("watchlists:overview.pipelineSetup.activate.ready", "Schedule activated."))
    } catch (error) {
      if (!isAbortError(error)) {
        setActionError(
          t(
            "watchlists:overview.pipelineSetup.activate.failed",
            "Schedule could not be activated. Your draft is saved. {{message}}",
            { message: getErrorMessage(error) }
          )
        )
      }
    } finally {
      setActionBusy(false)
    }
  }

  const testSource = async () => {
    if (!validateStep(0)) return
    setActionBusy(true)
    setActionError(null)
    setSourceTestMessage(null)
    try {
      const result = await onTestSource?.(draftRef.current)
      setSourceTestMessage(
        result && result.status === "cancelled"
          ? t("watchlists:overview.pipelineSetup.sourceTest.cancelled", "Source test cancelled.")
          : t("watchlists:overview.pipelineSetup.sourceTest.ready", "Source is ready.")
      )
    } catch (error) {
      if (!isAbortError(error)) {
        setActionError(
          t("watchlists:overview.pipelineSetup.sourceTest.failed", "Source test failed. {{message}}", {
            message: getErrorMessage(error)
          })
        )
      }
    } finally {
      setActionBusy(false)
    }
  }

  const nextRunAt = useMemo(() => projectNextRunAt(draft), [draft])
  const receipt = useMemo(() => {
    if (!nextRunAt) return null
    const payload = toPipelineJobCreatePayload(toBriefingPipelineDraft(draft))
    const contract = payload.output_prefs?.briefing_pipeline
    if (!contract) return null
    return buildBriefingReceiptModel({
      contract,
      sourceCount: draft.sourceMode === "new" ? 1 : draft.sourceIds.length,
      nextRunAt,
      followingRunAt: draft.followingRunAt,
      timezone: draft.timezone || "UTC"
    })
  }, [draft, nextRunAt])

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

  const footer = currentStep === LAST_STEP
    ? [
        <Button key="cancel" className="min-h-11" onClick={onCancel} disabled={isBusy}>
          {t("common:cancel", "Cancel")}
        </Button>,
        <Button
          key="back"
          className="min-h-11"
          onClick={() => setCurrentStep((value) => Math.max(0, value - 1))}
          disabled={isBusy}
        >
          {t("common:back", "Back")}
        </Button>
      ]
    : [
        <Button key="cancel" className="min-h-11" onClick={onCancel} disabled={isBusy}>
          {t("common:cancel", "Cancel")}
        </Button>,
        currentStep > 0 ? (
          <Button
            key="back"
            className="min-h-11"
            onClick={() => setCurrentStep((value) => Math.max(0, value - 1))}
            disabled={isBusy}
          >
            {t("common:back", "Back")}
          </Button>
        ) : null,
        <Button key="next" type="primary" className="min-h-11" onClick={moveNext} disabled={isBusy}>
          {t(
            "watchlists:overview.pipelineSetup.actions.next",
            "Next: {{step}}",
            { step: ["Cadence", "Briefing", "Delivery", "Test"][currentStep] }
          )}
        </Button>
      ]

  return (
    <Modal
      open={open}
      title={t("watchlists:overview.pipelineSetup.title", "Set up briefing")}
      onCancel={isBusy ? undefined : onCancel}
      destroyOnHidden
      maskClosable={!isBusy}
      width={760}
      footer={footer}
    >
      <div className="space-y-5">
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
                >
                  <Checkbox.Group
                    className="grid max-h-64 gap-2 overflow-y-auto"
                    value={draft.sourceIds}
                    onChange={(values) => updateDraft({ sourceIds: values.map(Number) })}
                  >
                    {sources.map((source) => (
                      <Checkbox key={source.id} value={source.id} className="min-h-11 py-2">
                        {source.name || `Source #${source.id}`}
                      </Checkbox>
                    ))}
                  </Checkbox.Group>
                </Form.Item>
                {sourcesLoading && <p className="text-sm text-text-muted">{t("watchlists:overview.pipelineSetup.sourcesLoading", "Loading sources...")}</p>}
              </Form>
            ) : (
              <Form layout="vertical">
                <Form.Item label={t("watchlists:overview.pipelineSetup.fields.sourceName", "Source name")} validateStatus={stepErrors.includes("sourceName") ? "error" : undefined}>
                  <Input aria-label={t("watchlists:overview.pipelineSetup.fields.sourceName", "Source name")} value={draft.sourceName} onChange={(event) => updateDraft({ sourceName: event.target.value })} />
                </Form.Item>
                <Form.Item label={t("watchlists:overview.pipelineSetup.fields.sourceUrl", "Source URL")} validateStatus={stepErrors.includes("sourceUrl") ? "error" : undefined}>
                  <Input aria-label={t("watchlists:overview.pipelineSetup.fields.sourceUrl", "Source URL")} value={draft.sourceUrl} onChange={(event) => updateDraft({ sourceUrl: event.target.value })} />
                </Form.Item>
              </Form>
            )}
            <div className="flex flex-wrap items-center gap-3">
              <Button className="min-h-11" onClick={() => void testSource()} loading={actionBusy}>
                {t("watchlists:overview.pipelineSetup.sourceTest.action", "Test source")}
              </Button>
              {sourceTestMessage && <p className="text-sm text-success">{sourceTestMessage}</p>}
            </div>
          </section>
        )}

        {currentStep === 1 && (
          <section aria-labelledby="pipeline-cadence-heading" className="space-y-4">
            <h2 id="pipeline-cadence-heading" className="text-base font-semibold">
              {t("watchlists:overview.pipelineSetup.steps.cadence", "Cadence")}
            </h2>
            <Form layout="vertical">
              <Form.Item label={t("watchlists:overview.pipelineSetup.fields.monitorName", "Monitor name")} validateStatus={stepErrors.includes("monitorName") ? "error" : undefined}>
                <Input aria-label={t("watchlists:overview.pipelineSetup.fields.monitorName", "Monitor name")} value={draft.monitorName} onChange={(event) => updateDraft({ monitorName: event.target.value })} />
              </Form.Item>
              <div className="grid gap-3 sm:grid-cols-2">
                <Form.Item label={t("watchlists:overview.pipelineSetup.fields.schedule", "Schedule")}>
                  <Select aria-label={t("watchlists:overview.pipelineSetup.fields.schedule", "Schedule")} value={draft.scheduleMode} options={scheduleOptions} onChange={(value) => updateDraft({ scheduleMode: value, createScheduledOutput: value !== "manual", nextRunAt: undefined, followingRunAt: undefined })} />
                </Form.Item>
                <Form.Item label={t("watchlists:overview.pipelineSetup.fields.timezone", "Timezone")} validateStatus={stepErrors.includes("timezone") ? "error" : undefined}>
                  <Input aria-label={t("watchlists:overview.pipelineSetup.fields.timezone", "Timezone")} value={draft.timezone} onChange={(event) => updateDraft({ timezone: event.target.value, nextRunAt: undefined, followingRunAt: undefined })} />
                </Form.Item>
              </div>
              {draft.scheduleMode === "interval" && (
                <div className="grid gap-3 sm:grid-cols-2">
                  <Form.Item label={t("watchlists:overview.pipelineSetup.fields.intervalEvery", "Every")} validateStatus={stepErrors.includes("scheduleIntervalValue") ? "error" : undefined}>
                    <InputNumber aria-label={t("watchlists:overview.pipelineSetup.fields.intervalEvery", "Every")} className="w-full" min={intervalMin} max={intervalMax} precision={0} value={draft.scheduleIntervalValue} onChange={(value) => updateDraft({ scheduleIntervalValue: Number(value), nextRunAt: undefined, followingRunAt: undefined })} />
                  </Form.Item>
                  <Form.Item label={t("watchlists:overview.pipelineSetup.fields.intervalUnit", "Interval unit")}>
                    <Select aria-label={t("watchlists:overview.pipelineSetup.fields.intervalUnit", "Interval unit")} value={draft.scheduleIntervalUnit} options={intervalUnitOptions} onChange={(value) => updateDraft({ scheduleIntervalUnit: value, nextRunAt: undefined, followingRunAt: undefined })} />
                  </Form.Item>
                </div>
              )}
              {(["daily", "weekdays", "weekly"] as PipelineWizardScheduleMode[]).includes(draft.scheduleMode) && (
                <div className="grid gap-3 sm:grid-cols-3">
                  {draft.scheduleMode === "weekly" && (
                    <Form.Item label={t("watchlists:overview.pipelineSetup.fields.weekday", "Weekday")}>
                      <Select aria-label={t("watchlists:overview.pipelineSetup.fields.weekday", "Weekday")} value={draft.scheduleWeekday} options={weekdayOptions} onChange={(value) => updateDraft({ scheduleWeekday: value, nextRunAt: undefined, followingRunAt: undefined })} />
                    </Form.Item>
                  )}
                  <Form.Item label={t("watchlists:overview.pipelineSetup.fields.hour", "Hour")} validateStatus={stepErrors.includes("scheduleHour") ? "error" : undefined}>
                    <InputNumber aria-label={t("watchlists:overview.pipelineSetup.fields.hour", "Hour")} className="w-full" min={0} max={23} precision={0} value={draft.scheduleHour} onChange={(value) => updateDraft({ scheduleHour: Number(value), nextRunAt: undefined, followingRunAt: undefined })} />
                  </Form.Item>
                  <Form.Item label={t("watchlists:overview.pipelineSetup.fields.minute", "Minute")} validateStatus={stepErrors.includes("scheduleMinute") ? "error" : undefined}>
                    <InputNumber aria-label={t("watchlists:overview.pipelineSetup.fields.minute", "Minute")} className="w-full" min={0} max={59} precision={0} value={draft.scheduleMinute} onChange={(value) => updateDraft({ scheduleMinute: Number(value), nextRunAt: undefined, followingRunAt: undefined })} />
                  </Form.Item>
                </div>
              )}
              {draft.scheduleMode === "advanced" && (
                <details className="rounded-md border border-border px-3 py-2">
                  <summary className="min-h-11 cursor-pointer py-3 font-medium">{t("watchlists:overview.pipelineSetup.advancedCron", "Advanced cron")}</summary>
                  <Form.Item label={t("watchlists:overview.pipelineSetup.fields.cronExpression", "Cron expression")} validateStatus={stepErrors.some((error) => error.startsWith("scheduleAdvancedCron")) ? "error" : undefined}>
                    <Input aria-label={t("watchlists:overview.pipelineSetup.fields.cronExpression", "Cron expression")} value={draft.scheduleAdvancedCron} placeholder="0 8 * * MON-FRI" onChange={(event) => updateDraft({ scheduleAdvancedCron: event.target.value, nextRunAt: undefined, followingRunAt: undefined })} />
                  </Form.Item>
                </details>
              )}
            </Form>
            <div className="border-t border-border pt-3 text-sm">
              <span className="font-medium">{t("watchlists:overview.pipelineSetup.nextOccurrence", "Next occurrence")}:</span>{" "}
              {receipt?.nextRunLabel
                ? `${receipt.nextRunLabel} (${draft.timezone})`
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
              <Switch aria-label={t("watchlists:overview.pipelineSetup.fields.includeAudio", "Audio")} checked={draft.audioEnabled} onChange={(checked) => updateDraft({ audioEnabled: checked, audioSpeakers: checked ? createSpeakers(currentSpeakerCount, draft.audioSpeakers, getSpeakerLabel) : [] })} />
            </div>
            {draft.audioEnabled && (
              <div className="space-y-4">
                <div className="grid gap-3 sm:grid-cols-2">
                  <Form.Item label={t("watchlists:overview.pipelineSetup.fields.audioMinutes", "Target duration in minutes")} validateStatus={stepErrors.includes("targetAudioMinutes") ? "error" : undefined}>
                    <Input aria-label={t("watchlists:overview.pipelineSetup.fields.audioMinutes", "Target duration in minutes")} type="number" min={1} max={60} value={draft.targetAudioMinutes} onChange={(event) => updateDraft({ targetAudioMinutes: Number(event.target.value) })} />
                  </Form.Item>
                  <Form.Item label={t("watchlists:overview.pipelineSetup.fields.speakerCount", "Cast size")}>
                    <Select aria-label={t("watchlists:overview.pipelineSetup.fields.speakerCount", "Cast size")} value={currentSpeakerCount} options={[1, 2, 3, 4].map((value) => ({ value, label: `${value}` }))} onChange={(value) => updateDraft({ audioSpeakers: createSpeakers(value, draft.audioSpeakers, getSpeakerLabel) })} />
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
            <details className="rounded-md border border-border px-3 py-2">
              <summary className="min-h-11 cursor-pointer py-3 font-medium">{t("watchlists:overview.pipelineSetup.briefing.advanced", "Advanced briefing settings")}</summary>
              <div className="space-y-3 pt-2">
                <Form layout="vertical">
                  <Form.Item label={t("watchlists:overview.pipelineSetup.fields.template", "Report template")} validateStatus={stepErrors.includes("templateName") ? "error" : undefined}>
                    <Input aria-label={t("watchlists:overview.pipelineSetup.fields.template", "Report template")} value={draft.templateName} onChange={(event) => updateDraft({ templateName: event.target.value })} />
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
                      <Form.Item label={t("watchlists:overview.pipelineSetup.speaker.voiceField", "Speaker {{index}} voice", { index: index + 1 })}>
                        <Select aria-label={t("watchlists:overview.pipelineSetup.speaker.voiceField", "Speaker {{index}} voice", { index: index + 1 })} value={speaker.voice} options={DEFAULT_SPEAKER_VOICES.map((voice) => ({ value: voice, label: voice }))} onChange={(value) => updateSpeaker(index, { voice: value })} />
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
                <Switch aria-label={t("watchlists:overview.pipelineSetup.fields.emailDelivery", "Email")} checked={draft.emailDeliveryEnabled} onChange={(checked) => updateDraft({ emailDeliveryEnabled: checked })} />
              </div>
              {draft.emailDeliveryEnabled && (
                <Form.Item label={t("watchlists:overview.pipelineSetup.fields.emailRecipients", "Email recipients")} validateStatus={stepErrors.includes("emailRecipients") ? "error" : undefined}>
                  <Select mode="tags" tokenSeparators={[","]} aria-label={t("watchlists:overview.pipelineSetup.fields.emailRecipients", "Email recipients")} value={draft.emailRecipients} onChange={(value) => updateDraft({ emailRecipients: value })} />
                </Form.Item>
              )}
              <div className="flex min-h-11 items-center justify-between gap-3 border-t border-border pt-3">
                <span className="font-medium">{t("watchlists:overview.pipelineSetup.fields.chatbookDelivery", "Chatbook")}</span>
                <Switch aria-label={t("watchlists:overview.pipelineSetup.fields.chatbookDelivery", "Chatbook")} checked={draft.chatbookDeliveryEnabled} onChange={(checked) => updateDraft({ chatbookDeliveryEnabled: checked })} />
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
              <p className="mt-1 text-sm text-text-muted">{t("watchlists:overview.pipelineSetup.test.providerDisclosure", "Tests use your configured LLM and text-to-speech providers. A sample is limited to 60 seconds; a full test uses the target duration.")}</p>
            </div>
            <div className="border-y border-border py-4" data-testid="watchlists-pipeline-receipt">
              <p className="font-medium">{t("watchlists:overview.pipelineSetup.test.receipt", "Activation receipt")}</p>
              <p className="mt-2 max-w-[75ch] text-sm text-text-muted">
                {receipt?.sentence || t("watchlists:overview.pipelineSetup.test.manualReceipt", "This briefing runs manually, saves selected artifacts in Reports, and contacts no external destination unless you choose Send test.")}
              </p>
              {receipt?.dstNote && <p className="mt-2 text-sm text-text-muted">{receipt.dstNote}</p>}
            </div>
            <ol className="space-y-2 text-sm" aria-label={t("watchlists:overview.pipelineSetup.test.progress", "Test progress")}>
              {[
                t("watchlists:overview.pipelineSetup.test.stages.collection", "Collect sources"),
                t("watchlists:overview.pipelineSetup.test.stages.selection", "Select updates"),
                t("watchlists:overview.pipelineSetup.test.stages.text", "Create report"),
                ...(draft.audioEnabled ? [t("watchlists:overview.pipelineSetup.test.stages.audio", "Create audio")] : []),
                t("watchlists:overview.pipelineSetup.test.stages.persistence", "Save in Reports"),
                t("watchlists:overview.pipelineSetup.test.stages.delivery", "Check test delivery")
              ].map((label) => (
                <li key={label} className="flex min-h-11 items-center gap-3 border-b border-border py-2"><span aria-hidden="true">○</span>{label}</li>
              ))}
            </ol>
            <div className="grid gap-2 sm:grid-cols-2">
              <Button className="min-h-11" onClick={() => void runTest({ externalDelivery: false, audioSampleSeconds: 60 })} loading={actionBusy}>
                {t("watchlists:overview.pipelineSetup.test.sample", "Generate 60-second sample")}
              </Button>
              <Button className="min-h-11" onClick={() => void runTest({ externalDelivery: false, audioSampleSeconds: null })} disabled={!draft.audioEnabled || isBusy}>
                {t("watchlists:overview.pipelineSetup.test.full", "Generate full test episode")}
              </Button>
              <Button className="min-h-11" onClick={() => void runTest({ externalDelivery: true, audioSampleSeconds: 60 })} disabled={!hasExternalDelivery || isBusy}>
                {t("watchlists:overview.pipelineSetup.test.send", "Send test")}
              </Button>
              <Button type="primary" className="min-h-11" onClick={() => void activate()} loading={actionBusy}>
                {t("watchlists:overview.pipelineSetup.test.activate", "Activate schedule")}
              </Button>
            </div>
          </section>
        )}
      </div>
    </Modal>
  )
}
