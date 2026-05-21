import React, { useEffect, useMemo, useState } from "react"
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
  Steps,
  Switch
} from "antd"
import { useTranslation } from "react-i18next"
import { Alert as DesignSystemAlert } from "@/components/ui"
import type { WatchlistSource } from "@/types/watchlists"
import {
  INTERVAL_HOURS_MAX,
  INTERVAL_HOURS_MIN,
  INTERVAL_MINUTES_MAX,
  INTERVAL_MINUTES_MIN,
  type ScheduleIntervalUnit,
  type WeekdayToken
} from "../JobsTab/schedule-utils"
import {
  buildPipelineWizardReviewSummary,
  createDefaultPipelineWizardDraft,
  type PipelineWizardAudioSpeakerDraft,
  type PipelineWizardDraft,
  type PipelineWizardScheduleMode,
  type PipelineWizardSourceMode,
  validatePipelineWizardDraft
} from "./pipeline-wizard-state"

interface PipelineWizardProps {
  open: boolean
  sources: WatchlistSource[]
  sourcesLoading: boolean
  submitting: boolean
  previewLoading: boolean
  previewError: string | null
  previewRendered: string | null
  previewRunId: number | null
  previewWarnings: string[]
  submitError: string | null
  onCancel: () => void
  onSubmit: (draft: PipelineWizardDraft, options: { mode: "create" | "test" }) => void
  onPreview: (draft: PipelineWizardDraft) => void
}

export type { PipelineWizardProps }

const STEP_COUNT = 5
const LAST_STEP = STEP_COUNT - 1

const DEFAULT_SPEAKER_VOICES = ["alloy", "nova", "echo", "fable"]

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

export const PipelineWizard: React.FC<PipelineWizardProps> = ({
  open,
  sources,
  sourcesLoading,
  submitting,
  previewLoading,
  previewError,
  previewRendered,
  previewRunId,
  previewWarnings,
  submitError,
  onCancel,
  onSubmit,
  onPreview
}) => {
  const { t } = useTranslation(["watchlists", "common"])
  const [currentStep, setCurrentStep] = useState(0)
  const [draft, setDraft] = useState<PipelineWizardDraft>(() => ({
    ...createDefaultPipelineWizardDraft(),
    templateName: "briefing_md"
  }))
  const [stepErrors, setStepErrors] = useState<string[]>([])

  useEffect(() => {
    if (!open) return
    setCurrentStep(0)
    setStepErrors([])
    setDraft({
      ...createDefaultPipelineWizardDraft(),
      templateName: "briefing_md"
    })
  }, [open])

  useEffect(() => {
    if (!open || draft.sourceIds.length > 0 || draft.sourceMode !== "existing") return
    if (sources.length === 1) {
      setDraft((previous) => ({
        ...previous,
        sourceIds: [sources[0].id]
      }))
    }
  }, [draft.sourceIds.length, draft.sourceMode, open, sources])

  const updateDraft = (patch: Partial<PipelineWizardDraft>) => {
    setDraft((previous) => ({
      ...previous,
      ...patch
    }))
    setStepErrors([])
  }

  const getSpeakerLabel = (index: number) =>
    t("watchlists:overview.pipelineSetup.speaker.defaultLabel", "Speaker {{index}}", { index })

  const updateSpeaker = (
    index: number,
    patch: Partial<PipelineWizardAudioSpeakerDraft>
  ) => {
    setDraft((previous) => {
      const nextSpeakers = createSpeakers(
        previous.audioSpeakers.length || 1,
        previous.audioSpeakers,
        getSpeakerLabel
      )
      nextSpeakers[index] = {
        ...nextSpeakers[index],
        ...patch
      }
      return {
        ...previous,
        audioSpeakers: nextSpeakers
      }
    })
    setStepErrors([])
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
  const weekdayLabelByToken = useMemo(
    () => Object.fromEntries(weekdayOptions.map((option) => [option.value, option.label])) as Record<WeekdayToken, string>,
    [weekdayOptions]
  )
  const scheduleOptions = useMemo<Array<{ value: PipelineWizardScheduleMode; label: string }>>(
    () => [
      { value: "manual", label: t("watchlists:overview.pipelineSetup.schedule.manual", "Manual only") },
      { value: "interval", label: t("watchlists:overview.pipelineSetup.schedule.interval", "Every N hours/minutes") },
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
  const speakerCountOptions = useMemo(
    () => [1, 2, 3, 4].map((count) => ({
      value: count,
      label: t(
        "watchlists:overview.pipelineSetup.speaker.count",
        `${count} speaker${count === 1 ? "" : "s"}`,
        { count }
      )
    })),
    [t]
  )

  const reviewSummaryCopy = useMemo(() => ({
    newFeed: t("watchlists:overview.pipelineSetup.review.newFeed", "New feed"),
    noFeedsSelected: t("watchlists:overview.pipelineSetup.review.noFeedsSelected", "No feeds selected"),
    feedLabel: (id: number) => t("watchlists:overview.pipelineSetup.review.feedLabel", "Feed #{{id}}", { id }),
    filters: t(
      "watchlists:overview.pipelineSetup.review.filtersRefine",
      "Monitor filters can be refined after creation"
    ),
    noTemplate: t("watchlists:overview.pipelineSetup.review.noTemplate", "No template"),
    outputDigest: (templateName: string) =>
      t("watchlists:overview.pipelineSetup.review.outputDigest", "{{templateName}} digest", { templateName }),
    email: t("watchlists:overview.pipelineSetup.review.delivery.email", "Email"),
    chatbook: t("watchlists:overview.pipelineSetup.review.delivery.chatbook", "Chatbook"),
    inAppReports: t("watchlists:overview.pipelineSetup.review.delivery.inAppReports", "In-app reports"),
    audioBriefing: (speakerCount: number) =>
      t(
        "watchlists:overview.pipelineSetup.review.audioBriefing",
        `${speakerCount} speaker${speakerCount === 1 ? "" : "s"} audio briefing`,
        { count: speakerCount }
      ),
    audioDisabled: t("watchlists:overview.pipelineSetup.review.audioDisabled", "Audio disabled"),
    cadence: {
      manual: t("watchlists:overview.pipelineSetup.schedule.manual", "Manual only"),
      interval: (value: number, unit: ScheduleIntervalUnit) =>
        unit === "minutes"
          ? t(
            "watchlists:overview.pipelineSetup.review.cadence.everyMinutes",
            `Every ${value} minute${value === 1 ? "" : "s"}`,
            { count: value }
          )
          : t(
            "watchlists:overview.pipelineSetup.review.cadence.everyHours",
            `Every ${value} hour${value === 1 ? "" : "s"}`,
            { count: value }
          ),
      daily: (time: string) =>
        t("watchlists:overview.pipelineSetup.review.cadence.daily", "Daily at {{time}}", { time }),
      weekly: (weekday: string, time: string) =>
        t(
          "watchlists:overview.pipelineSetup.review.cadence.weekly",
          "Weekly on {{weekday}} at {{time}}",
          { weekday, time }
        ),
      weekdays: (time: string) =>
        t("watchlists:overview.pipelineSetup.review.cadence.weekdays", "Weekdays at {{time}}", { time }),
      advanced: (cron: string) =>
        t("watchlists:overview.pipelineSetup.review.cadence.advanced", "Custom cron: {{cron}}", { cron }),
      weekdayLabels: weekdayLabelByToken
    }
  }), [t, weekdayLabelByToken])

  const summary = useMemo(
    () => buildPipelineWizardReviewSummary(draft, sources, reviewSummaryCopy),
    [draft, reviewSummaryCopy, sources]
  )

  const stepItems = useMemo(
    () => [
      { title: t("watchlists:overview.pipelineSetup.steps.source", "Source") },
      { title: t("watchlists:overview.pipelineSetup.steps.monitor", "Monitor") },
      { title: t("watchlists:overview.pipelineSetup.steps.digest", "Digest") },
      { title: t("watchlists:overview.pipelineSetup.steps.audio", "Audio") },
      { title: t("watchlists:overview.pipelineSetup.steps.review", "Review") }
    ],
    [t]
  )

  const validateCurrentStep = (): boolean => {
    const validation = validatePipelineWizardDraft(draft)
    const currentStepFields: string[] = (() => {
      if (currentStep === 0) {
        return draft.sourceMode === "new" ? ["sourceName", "sourceUrl"] : ["sourceIds"]
      }
      if (currentStep === 1) {
        return [
          "monitorName",
          "scheduleIntervalValue",
          "scheduleHour",
          "scheduleMinute",
          "scheduleAdvancedCron"
        ]
      }
      if (currentStep === 2) return ["templateName", "emailRecipients"]
      if (currentStep === 3) {
        return [
          "audioSpeakers",
          "audioSpeakerIds",
          "audioSpeakerVoices",
          "targetAudioMinutes"
        ]
      }
      return validation.errors
    })()
    const errors = validation.errors.filter((error) => currentStepFields.includes(error))
    setStepErrors(errors)
    return errors.length === 0
  }

  const handleNext = () => {
    if (!validateCurrentStep()) return
    setCurrentStep((previous) => Math.min(LAST_STEP, previous + 1))
  }

  const handleSubmit = (mode: "create" | "test") => {
    const validation = validatePipelineWizardDraft(draft)
    if (!validation.valid) {
      setStepErrors(validation.errors)
      const firstError = validation.errors[0]
      if (["sourceIds", "sourceName", "sourceUrl"].includes(firstError)) setCurrentStep(0)
      else if ([
        "monitorName",
        "scheduleIntervalValue",
        "scheduleHour",
        "scheduleMinute",
        "scheduleAdvancedCron"
      ].includes(firstError)) setCurrentStep(1)
      else if (["templateName", "emailRecipients"].includes(firstError)) setCurrentStep(2)
      else setCurrentStep(3)
      return
    }
    onSubmit(draft, { mode })
  }

  const currentSpeakerCount = Math.max(1, Math.min(4, draft.audioSpeakers.length || 1))
  const intervalMin =
    draft.scheduleIntervalUnit === "minutes" ? INTERVAL_MINUTES_MIN : INTERVAL_HOURS_MIN
  const intervalMax =
    draft.scheduleIntervalUnit === "minutes" ? INTERVAL_MINUTES_MAX : INTERVAL_HOURS_MAX

  return (
    <Modal
      open={open}
      title={t("watchlists:overview.pipelineSetup.title", "Briefing pipeline builder")}
      onCancel={onCancel}
      destroyOnHidden
      maskClosable={!submitting}
      width={760}
      footer={[
        <Button key="cancel" onClick={onCancel} disabled={submitting}>
          {t("common:cancel", "Cancel")}
        </Button>,
        <Button
          key="back"
          onClick={() => setCurrentStep((previous) => Math.max(0, previous - 1))}
          disabled={submitting || currentStep === 0}
        >
          {t("common:back", "Back")}
        </Button>,
        currentStep === LAST_STEP ? (
          <Button
            key="test-generation"
            data-testid="watchlists-pipeline-test-generation"
            onClick={() => handleSubmit("test")}
            loading={submitting}
          >
            {t("watchlists:overview.pipelineSetup.actions.testGeneration", "Run test generation")}
          </Button>
        ) : null,
        <Button
          key="next"
          type="primary"
          loading={submitting}
          onClick={() => {
            if (currentStep === LAST_STEP) {
              handleSubmit("create")
              return
            }
            handleNext()
          }}
        >
          {currentStep === LAST_STEP
            ? t("watchlists:overview.pipelineSetup.actions.finish", "Create pipeline")
            : t("common:next", "Next")}
        </Button>
      ]}
    >
      <div className="space-y-4">
        <Steps size="small" current={currentStep} items={stepItems} />
        {stepErrors.length > 0 && (
          <DesignSystemAlert
            variant="warning"
            title={t(
              "watchlists:overview.pipelineSetup.validationError",
              "Review the highlighted pipeline fields."
            )}
          />
        )}
        {submitError && (
          <DesignSystemAlert
            variant="error"
            title={submitError}
            data-testid="watchlists-pipeline-error"
          />
        )}

        {currentStep === 0 && (
          <div className="space-y-3">
            <Radio.Group
              value={draft.sourceMode}
              onChange={(event) => updateDraft({ sourceMode: event.target.value as PipelineWizardSourceMode })}
            >
              <Space orientation="vertical">
                <Radio value="existing">{t("watchlists:overview.pipelineSetup.source.existing", "Use existing feeds")}</Radio>
                <Radio value="new">{t("watchlists:overview.pipelineSetup.source.new", "Create a new feed")}</Radio>
              </Space>
            </Radio.Group>

            {draft.sourceMode === "existing" ? (
              <Form layout="vertical">
                <Form.Item
                  label={t("watchlists:overview.pipelineSetup.fields.sources", "Feeds")}
                  validateStatus={stepErrors.includes("sourceIds") ? "error" : undefined}
                  help={stepErrors.includes("sourceIds") ? t("watchlists:overview.pipelineSetup.validation.sourcesRequired", "Select at least one feed") : undefined}
                >
                  <Checkbox.Group
                    className="grid gap-2"
                    value={draft.sourceIds}
                    onChange={(values) => updateDraft({ sourceIds: values.map((value) => Number(value)) })}
                  >
                    {sources.map((source) => (
                      <Checkbox key={source.id} value={source.id}>
                        {source.name || `Feed #${source.id}`}
                      </Checkbox>
                    ))}
                  </Checkbox.Group>
                </Form.Item>
                {sourcesLoading && (
                  <p className="text-xs text-text-muted">
                    {t("watchlists:overview.pipelineSetup.sourcesLoading", "Loading feeds...")}
                  </p>
                )}
              </Form>
            ) : (
              <Form layout="vertical">
                <Form.Item
                  label={t("watchlists:overview.pipelineSetup.fields.sourceName", "Feed name")}
                  validateStatus={stepErrors.includes("sourceName") ? "error" : undefined}
                >
                  <Input
                    aria-label={t("watchlists:overview.pipelineSetup.fields.sourceName", "Feed name")}
                    value={draft.sourceName}
                    onChange={(event) => updateDraft({ sourceName: event.target.value })}
                  />
                </Form.Item>
                <Form.Item
                  label={t("watchlists:overview.pipelineSetup.fields.sourceUrl", "Feed URL")}
                  validateStatus={stepErrors.includes("sourceUrl") ? "error" : undefined}
                  help={stepErrors.includes("sourceUrl") ? t("watchlists:overview.pipelineSetup.validation.sourceUrlRequired", "Enter a valid http or https feed URL") : undefined}
                >
                  <Input
                    aria-label={t("watchlists:overview.pipelineSetup.fields.sourceUrl", "Feed URL")}
                    value={draft.sourceUrl}
                    onChange={(event) => updateDraft({ sourceUrl: event.target.value })}
                  />
                </Form.Item>
              </Form>
            )}
          </div>
        )}

        {currentStep === 1 && (
          <Form layout="vertical">
            <Form.Item
              label={t("watchlists:overview.pipelineSetup.fields.monitorName", "Monitor name")}
              validateStatus={stepErrors.includes("monitorName") ? "error" : undefined}
            >
              <Input
                aria-label={t("watchlists:overview.pipelineSetup.fields.monitorName", "Monitor name")}
                value={draft.monitorName}
                onChange={(event) => updateDraft({ monitorName: event.target.value })}
              />
            </Form.Item>
            <Form.Item label={t("watchlists:overview.pipelineSetup.fields.schedule", "Schedule")}>
              <Select
                aria-label={t("watchlists:overview.pipelineSetup.fields.schedule", "Schedule")}
                value={draft.scheduleMode}
                options={scheduleOptions}
                onChange={(value) => updateDraft({ scheduleMode: value as PipelineWizardScheduleMode })}
              />
            </Form.Item>
            {draft.scheduleMode === "interval" && (
              <div className="grid gap-3 sm:grid-cols-2">
                <Form.Item
                  label={t("watchlists:overview.pipelineSetup.fields.intervalEvery", "Every")}
                  validateStatus={stepErrors.includes("scheduleIntervalValue") ? "error" : undefined}
                >
                  <InputNumber
                    aria-label={t("watchlists:overview.pipelineSetup.fields.intervalEvery", "Every")}
                    className="w-full"
                    min={intervalMin}
                    max={intervalMax}
                    precision={0}
                    value={draft.scheduleIntervalValue}
                    onChange={(value) => updateDraft({ scheduleIntervalValue: Number(value) })}
                  />
                </Form.Item>
                <Form.Item label={t("watchlists:overview.pipelineSetup.fields.intervalUnit", "Interval unit")}>
                  <Select
                    aria-label={t("watchlists:overview.pipelineSetup.fields.intervalUnit", "Interval unit")}
                    value={draft.scheduleIntervalUnit}
                    options={intervalUnitOptions}
                    onChange={(value) => updateDraft({ scheduleIntervalUnit: value })}
                  />
                </Form.Item>
              </div>
            )}
            {(
              draft.scheduleMode === "daily" ||
              draft.scheduleMode === "weekdays" ||
              draft.scheduleMode === "weekly"
            ) && (
              <div className="grid gap-3 sm:grid-cols-3">
                {draft.scheduleMode === "weekly" && (
                  <Form.Item label={t("watchlists:overview.pipelineSetup.fields.weekday", "Weekday")}>
                    <Select
                      aria-label={t("watchlists:overview.pipelineSetup.fields.weekday", "Weekday")}
                      value={draft.scheduleWeekday}
                      options={weekdayOptions}
                      onChange={(value) => updateDraft({ scheduleWeekday: value })}
                    />
                  </Form.Item>
                )}
                <Form.Item
                  label={t("watchlists:overview.pipelineSetup.fields.hour", "Hour")}
                  validateStatus={stepErrors.includes("scheduleHour") ? "error" : undefined}
                >
                  <InputNumber
                    aria-label={t("watchlists:overview.pipelineSetup.fields.hour", "Hour")}
                    className="w-full"
                    min={0}
                    max={23}
                    precision={0}
                    value={draft.scheduleHour}
                    onChange={(value) => updateDraft({ scheduleHour: Number(value) })}
                  />
                </Form.Item>
                <Form.Item
                  label={t("watchlists:overview.pipelineSetup.fields.minute", "Minute")}
                  validateStatus={stepErrors.includes("scheduleMinute") ? "error" : undefined}
                >
                  <InputNumber
                    aria-label={t("watchlists:overview.pipelineSetup.fields.minute", "Minute")}
                    className="w-full"
                    min={0}
                    max={59}
                    precision={0}
                    value={draft.scheduleMinute}
                    onChange={(value) => updateDraft({ scheduleMinute: Number(value) })}
                  />
                </Form.Item>
              </div>
            )}
            {draft.scheduleMode === "advanced" && (
              <Form.Item
                label={t("watchlists:overview.pipelineSetup.fields.cronExpression", "Cron expression")}
                validateStatus={stepErrors.includes("scheduleAdvancedCron") ? "error" : undefined}
                help={stepErrors.includes("scheduleAdvancedCron")
                  ? t(
                    "watchlists:overview.pipelineSetup.validation.cronExpression",
                    "Enter a 5-field cron expression."
                  )
                  : undefined}
              >
                <Input
                  aria-label={t("watchlists:overview.pipelineSetup.fields.cronExpression", "Cron expression")}
                  placeholder="0 8 * * MON-FRI"
                  value={draft.scheduleAdvancedCron}
                  onChange={(event) => updateDraft({ scheduleAdvancedCron: event.target.value })}
                />
              </Form.Item>
            )}
            <Form.Item label={t("watchlists:overview.pipelineSetup.fields.runNow", "Run immediately")}>
              <Switch
                aria-label={t("watchlists:overview.pipelineSetup.fields.runNow", "Run immediately")}
                checked={draft.runNow}
                onChange={(checked) => updateDraft({ runNow: checked })}
              />
            </Form.Item>
          </Form>
        )}

        {currentStep === 2 && (
          <Form layout="vertical">
            <Form.Item
              label={t("watchlists:overview.pipelineSetup.fields.template", "Template")}
              validateStatus={stepErrors.includes("templateName") ? "error" : undefined}
            >
              <Input
                aria-label={t("watchlists:overview.pipelineSetup.fields.template", "Template")}
                value={draft.templateName}
                onChange={(event) => updateDraft({ templateName: event.target.value })}
              />
            </Form.Item>
            <Form.Item label={t("watchlists:overview.pipelineSetup.fields.emailDelivery", "Email delivery")}>
              <Switch
                aria-label={t("watchlists:overview.pipelineSetup.fields.emailDelivery", "Email delivery")}
                checked={draft.emailDeliveryEnabled}
                onChange={(checked) => updateDraft({ emailDeliveryEnabled: checked })}
              />
            </Form.Item>
            {draft.emailDeliveryEnabled && (
              <Form.Item
                label={t("watchlists:overview.pipelineSetup.fields.emailRecipients", "Email recipients")}
                validateStatus={stepErrors.includes("emailRecipients") ? "error" : undefined}
              >
                <Select
                  mode="tags"
                  tokenSeparators={[","]}
                  aria-label={t("watchlists:overview.pipelineSetup.fields.emailRecipients", "Email recipients")}
                  value={draft.emailRecipients}
                  onChange={(value) => updateDraft({ emailRecipients: value })}
                />
              </Form.Item>
            )}
            <Form.Item label={t("watchlists:overview.pipelineSetup.fields.chatbookDelivery", "Chatbook delivery")}>
              <Switch
                aria-label={t("watchlists:overview.pipelineSetup.fields.chatbookDelivery", "Chatbook delivery")}
                checked={draft.chatbookDeliveryEnabled}
                onChange={(checked) => updateDraft({ chatbookDeliveryEnabled: checked })}
              />
            </Form.Item>
          </Form>
        )}

        {currentStep === 3 && (
          <Form layout="vertical">
            <Form.Item label={t("watchlists:overview.pipelineSetup.fields.includeAudio", "Audio briefing")}>
              <Switch
                aria-label={t("watchlists:overview.pipelineSetup.fields.includeAudio", "Audio briefing")}
                checked={draft.audioEnabled}
                onChange={(checked) => updateDraft({
                  audioEnabled: checked,
                  audioSpeakers: checked
                    ? createSpeakers(currentSpeakerCount, draft.audioSpeakers, getSpeakerLabel)
                    : []
                })}
              />
            </Form.Item>
            {draft.audioEnabled && (
              <>
                <div className="grid gap-3 sm:grid-cols-2">
                  <Form.Item label={t("watchlists:overview.pipelineSetup.fields.speakerCount", "Speaker count")}>
                    <Select
                      aria-label={t("watchlists:overview.pipelineSetup.fields.speakerCount", "Speaker count")}
                      value={currentSpeakerCount}
                      options={speakerCountOptions}
                      onChange={(value) => updateDraft({
                        audioSpeakers: createSpeakers(value, draft.audioSpeakers, getSpeakerLabel)
                      })}
                    />
                  </Form.Item>
                  <Form.Item
                    label={t("watchlists:overview.pipelineSetup.fields.audioMinutes", "Target audio minutes")}
                    validateStatus={stepErrors.includes("targetAudioMinutes") ? "error" : undefined}
                  >
                    <Input
                      aria-label={t("watchlists:overview.pipelineSetup.fields.audioMinutes", "Target audio minutes")}
                      type="number"
                      min={1}
                      value={draft.targetAudioMinutes}
                      onChange={(event) => updateDraft({ targetAudioMinutes: Number(event.target.value) })}
                    />
                  </Form.Item>
                </div>
                <div className="space-y-3">
                  {draft.audioSpeakers.map((speaker, index) => (
                    <div
                      key={speaker.id || index}
                      className="grid gap-3 rounded-md border border-border bg-surface p-3 sm:grid-cols-2"
                    >
                      <Form.Item
                        label={t(
                          "watchlists:overview.pipelineSetup.speaker.labelField",
                          "Speaker {{index}} label",
                          { index: index + 1 }
                        )}
                      >
                        <Input
                          aria-label={t(
                            "watchlists:overview.pipelineSetup.speaker.labelField",
                            "Speaker {{index}} label",
                            { index: index + 1 }
                          )}
                          value={speaker.label}
                          onChange={(event) => updateSpeaker(index, { label: event.target.value })}
                        />
                      </Form.Item>
                      <Form.Item
                        label={t(
                          "watchlists:overview.pipelineSetup.speaker.voiceField",
                          "Speaker {{index}} voice",
                          { index: index + 1 }
                        )}
                      >
                        <Select
                          aria-label={t(
                            "watchlists:overview.pipelineSetup.speaker.voiceField",
                            "Speaker {{index}} voice",
                            { index: index + 1 }
                          )}
                          value={speaker.voice}
                          options={DEFAULT_SPEAKER_VOICES.map((voice) => ({
                            value: voice,
                            label: voice
                          }))}
                          onChange={(value) => updateSpeaker(index, { voice: value })}
                        />
                      </Form.Item>
                    </div>
                  ))}
                </div>
              </>
            )}
          </Form>
        )}

        {currentStep === 4 && (
          <div className="space-y-3 text-sm">
            <div
              className="rounded-md border border-border bg-surface p-3"
              data-testid="watchlists-pipeline-review-summary"
            >
              <p>
                <span className="font-medium">
                  {t("watchlists:overview.pipelineSetup.review.sources", "Sources")}:
                </span>{" "}
                {summary.sources}
              </p>
              <p>
                <span className="font-medium">
                  {t("watchlists:overview.pipelineSetup.review.cadence", "Cadence")}:
                </span>{" "}
                {summary.cadence}
              </p>
              <p>
                <span className="font-medium">
                  {t("watchlists:overview.pipelineSetup.review.filters", "Filters")}:
                </span>{" "}
                {summary.filters}
              </p>
              <p>
                <span className="font-medium">
                  {t("watchlists:overview.pipelineSetup.review.output", "Output")}:
                </span>{" "}
                {summary.output}
              </p>
              <p>
                <span className="font-medium">
                  {t("watchlists:overview.pipelineSetup.review.delivery", "Delivery")}:
                </span>{" "}
                {summary.delivery}
              </p>
              <p>
                <span className="font-medium">
                  {t("watchlists:overview.pipelineSetup.review.audio", "Audio")}:
                </span>{" "}
                {summary.audio}
              </p>
            </div>
            <div className="rounded-md border border-border bg-surface p-3 space-y-2">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <p className="text-xs text-text-muted">
                  {t(
                    "watchlists:overview.pipelineSetup.preview.description",
                    "Preview template output using the latest completed run context before creating the pipeline."
                  )}
                </p>
                <Button
                  size="small"
                  onClick={() => onPreview(draft)}
                  loading={previewLoading}
                  data-testid="watchlists-pipeline-preview-generate"
                >
                  {t("watchlists:overview.pipelineSetup.preview.generate", "Generate preview")}
                </Button>
              </div>
              {previewError && (
                <DesignSystemAlert
                  variant="warning"
                  data-testid="watchlists-pipeline-preview-error"
                  title={previewError}
                />
              )}
              {previewRunId != null && !previewError && (
                <p className="text-xs text-text-muted">
                  {t(
                    "watchlists:overview.pipelineSetup.preview.context",
                    "Preview context run: #{{runId}}",
                    { runId: previewRunId }
                  )}
                </p>
              )}
              {previewWarnings.length > 0 && (
                <ul className="list-disc pl-5 text-xs text-text-muted">
                  {previewWarnings.map((warning, index) => (
                    <li key={`${warning}-${index}`}>{warning}</li>
                  ))}
                </ul>
              )}
              {previewRendered && (
                <pre
                  className="max-h-48 overflow-auto rounded border border-border bg-background p-2 text-xs"
                  data-testid="watchlists-pipeline-preview-rendered"
                >
                  {previewRendered}
                </pre>
              )}
            </div>
          </div>
        )}
      </div>
    </Modal>
  )
}
