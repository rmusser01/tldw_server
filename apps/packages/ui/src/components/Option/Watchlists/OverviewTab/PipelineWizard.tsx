import React, { useEffect, useMemo, useState } from "react"
import {
  Alert,
  Button,
  Checkbox,
  Form,
  Input,
  Modal,
  Radio,
  Select,
  Space,
  Steps,
  Switch
} from "antd"
import { useTranslation } from "react-i18next"
import type { WatchlistSource } from "@/types/watchlists"
import {
  buildPipelineWizardReviewSummary,
  createDefaultPipelineWizardDraft,
  type PipelineWizardAudioSpeakerDraft,
  type PipelineWizardDraft,
  type PipelineWizardScheduleMode,
  type PipelineWizardSourceMode,
  validatePipelineWizardDraft
} from "./pipeline-wizard-state"
import type { ScheduleIntervalUnit, WeekdayToken } from "../JobsTab/schedule-utils"

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
  onCancel: () => void
  onSubmit: (draft: PipelineWizardDraft, options: { mode: "create" | "test" }) => void
  onPreview: (draft: PipelineWizardDraft) => void
}

export type { PipelineWizardProps }

const STEP_COUNT = 5
const LAST_STEP = STEP_COUNT - 1

const WEEKDAY_OPTIONS: Array<{ value: WeekdayToken; label: string }> = [
  { value: "MON", label: "Monday" },
  { value: "TUE", label: "Tuesday" },
  { value: "WED", label: "Wednesday" },
  { value: "THU", label: "Thursday" },
  { value: "FRI", label: "Friday" },
  { value: "SAT", label: "Saturday" },
  { value: "SUN", label: "Sunday" }
]

const SCHEDULE_OPTIONS: Array<{ value: PipelineWizardScheduleMode; label: string }> = [
  { value: "manual", label: "Manual only" },
  { value: "interval", label: "Every N hours/minutes" },
  { value: "daily", label: "Daily" },
  { value: "weekly", label: "Weekly" }
]

const INTERVAL_UNIT_OPTIONS: Array<{ value: ScheduleIntervalUnit; label: string }> = [
  { value: "hours", label: "Hours" },
  { value: "minutes", label: "Minutes" }
]

const SPEAKER_COUNT_OPTIONS = [
  { value: 1, label: "1 speaker" },
  { value: 2, label: "2 speakers" },
  { value: 3, label: "3 speakers" },
  { value: 4, label: "4 speakers" }
]

const DEFAULT_SPEAKER_VOICES = ["alloy", "nova", "echo", "fable"]

const createSpeakers = (
  count: number,
  existing: PipelineWizardAudioSpeakerDraft[]
): PipelineWizardAudioSpeakerDraft[] =>
  Array.from({ length: Math.max(1, Math.min(4, count)) }, (_item, index) => {
    const current = existing[index]
    return {
      id: current?.id || `speaker_${index + 1}`,
      label: current?.label || `Speaker ${index + 1}`,
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

  const updateSpeaker = (
    index: number,
    patch: Partial<PipelineWizardAudioSpeakerDraft>
  ) => {
    setDraft((previous) => {
      const nextSpeakers = createSpeakers(previous.audioSpeakers.length || 1, previous.audioSpeakers)
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

  const summary = useMemo(
    () => buildPipelineWizardReviewSummary(draft, sources),
    [draft, sources]
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
      if (currentStep === 1) return ["monitorName", "scheduleIntervalValue"]
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
      else if (["monitorName", "scheduleIntervalValue"].includes(firstError)) setCurrentStep(1)
      else if (["templateName", "emailRecipients"].includes(firstError)) setCurrentStep(2)
      else setCurrentStep(3)
      return
    }
    onSubmit(draft, { mode })
  }

  const currentSpeakerCount = Math.max(1, Math.min(4, draft.audioSpeakers.length || 1))

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
          <Alert
            type="warning"
            showIcon
            title={t(
              "watchlists:overview.pipelineSetup.validationError",
              "Review the highlighted pipeline fields."
            )}
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
                options={SCHEDULE_OPTIONS}
                onChange={(value) => updateDraft({ scheduleMode: value })}
              />
            </Form.Item>
            {draft.scheduleMode === "interval" && (
              <div className="grid gap-3 sm:grid-cols-2">
                <Form.Item
                  label={t("watchlists:overview.pipelineSetup.fields.intervalEvery", "Every")}
                  validateStatus={stepErrors.includes("scheduleIntervalValue") ? "error" : undefined}
                >
                  <Input
                    aria-label={t("watchlists:overview.pipelineSetup.fields.intervalEvery", "Every")}
                    type="number"
                    min={1}
                    value={draft.scheduleIntervalValue}
                    onChange={(event) => updateDraft({ scheduleIntervalValue: Number(event.target.value) })}
                  />
                </Form.Item>
                <Form.Item label={t("watchlists:overview.pipelineSetup.fields.intervalUnit", "Interval unit")}>
                  <Select
                    aria-label={t("watchlists:overview.pipelineSetup.fields.intervalUnit", "Interval unit")}
                    value={draft.scheduleIntervalUnit}
                    options={INTERVAL_UNIT_OPTIONS}
                    onChange={(value) => updateDraft({ scheduleIntervalUnit: value })}
                  />
                </Form.Item>
              </div>
            )}
            {(draft.scheduleMode === "daily" || draft.scheduleMode === "weekly") && (
              <div className="grid gap-3 sm:grid-cols-3">
                {draft.scheduleMode === "weekly" && (
                  <Form.Item label={t("watchlists:overview.pipelineSetup.fields.weekday", "Weekday")}>
                    <Select
                      aria-label={t("watchlists:overview.pipelineSetup.fields.weekday", "Weekday")}
                      value={draft.scheduleWeekday}
                      options={WEEKDAY_OPTIONS}
                      onChange={(value) => updateDraft({ scheduleWeekday: value })}
                    />
                  </Form.Item>
                )}
                <Form.Item label={t("watchlists:overview.pipelineSetup.fields.hour", "Hour")}>
                  <Input
                    aria-label={t("watchlists:overview.pipelineSetup.fields.hour", "Hour")}
                    type="number"
                    min={0}
                    max={23}
                    value={draft.scheduleHour}
                    onChange={(event) => updateDraft({ scheduleHour: Number(event.target.value) })}
                  />
                </Form.Item>
                <Form.Item label={t("watchlists:overview.pipelineSetup.fields.minute", "Minute")}>
                  <Input
                    aria-label={t("watchlists:overview.pipelineSetup.fields.minute", "Minute")}
                    type="number"
                    min={0}
                    max={59}
                    value={draft.scheduleMinute}
                    onChange={(event) => updateDraft({ scheduleMinute: Number(event.target.value) })}
                  />
                </Form.Item>
              </div>
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
                  audioSpeakers: checked ? createSpeakers(currentSpeakerCount, draft.audioSpeakers) : []
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
                      options={SPEAKER_COUNT_OPTIONS}
                      onChange={(value) => updateDraft({ audioSpeakers: createSpeakers(value, draft.audioSpeakers) })}
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
                      <Form.Item label={`Speaker ${index + 1} label`}>
                        <Input
                          aria-label={`Speaker ${index + 1} label`}
                          value={speaker.label}
                          onChange={(event) => updateSpeaker(index, { label: event.target.value })}
                        />
                      </Form.Item>
                      <Form.Item label={`Speaker ${index + 1} voice`}>
                        <Select
                          aria-label={`Speaker ${index + 1} voice`}
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
              <p><span className="font-medium">Sources:</span> {summary.sources}</p>
              <p><span className="font-medium">Cadence:</span> {summary.cadence}</p>
              <p><span className="font-medium">Filters:</span> {summary.filters}</p>
              <p><span className="font-medium">Output:</span> {summary.output}</p>
              <p><span className="font-medium">Delivery:</span> {summary.delivery}</p>
              <p><span className="font-medium">Audio:</span> {summary.audio}</p>
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
                <Alert
                  type="warning"
                  showIcon
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
