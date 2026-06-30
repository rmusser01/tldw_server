import React, { useEffect, useMemo, useState } from "react"
import { Button, Input, Modal, Switch, Tag } from "antd"
import { useTranslation } from "react-i18next"
import { Alert } from "@/components/ui/primitives"
import type {
  WatchlistContainer,
  WatchlistCreate,
  WatchlistJob,
  WatchlistJobCreate,
  WatchlistSourceCreate
} from "@/types/watchlists"
import {
  applyWatchlistSetupPreset,
  buildWatchlistSetupJobPayload,
  buildWatchlistSetupPlan,
  type WatchlistSetupDestination,
  type WatchlistSetupPreset,
  type WatchlistSetupSchedulePreset,
  type WatchlistSetupStartMode,
  type WatchlistSetupValues
} from "./watchlist-setup-model"

export interface WatchlistSetupCompleteResult {
  destination: WatchlistSetupDestination
  watchlist: WatchlistContainer
  sourceIds: number[]
  job?: WatchlistJob
}

export interface WatchlistSetupWizardProps {
  open: boolean
  submitting?: boolean
  onCancel: () => void
  onCreateWatchlist: (payload: WatchlistCreate) => Promise<WatchlistContainer>
  onCreateSources: (watchlistId: number, sources: WatchlistSourceCreate[]) => Promise<number[]>
  onCreateJob: (watchlistId: number, job: WatchlistJobCreate) => Promise<WatchlistJob>
  onComplete: (result: WatchlistSetupCompleteResult) => void
}

const createInitialValues = (): WatchlistSetupValues => ({
  preset: "general",
  startMode: "sources",
  name: "",
  objective: "",
  trackedScopeText: "",
  sourceUrlsText: "",
  sourceName: "",
  sourceType: "rss",
  monitorName: "",
  reportGoal: "",
  includeAudioBriefing: false,
  schedulePreset: "daily"
})

const presetOptions: Array<{ key: WatchlistSetupPreset; label: string; description: string }> = [
  {
    key: "cti_osint",
    label: "CTI / OSINT",
    description: "Track vulnerabilities, malware, actors, advisories, and source changes."
  },
  {
    key: "news",
    label: "News",
    description: "Track developing events, people, organizations, and source diversity."
  },
  {
    key: "general",
    label: "General",
    description: "Track a topic and collect updates for review."
  },
  {
    key: "blank",
    label: "Blank",
    description: "Start with an empty Watchlist and configure details yourself."
  }
]

const startModeOptions: Array<{ key: WatchlistSetupStartMode; label: string; description: string }> = [
  {
    key: "sources",
    label: "Start from sources",
    description: "Create a Watchlist with initial feeds and an optional monitor."
  },
  {
    key: "topic",
    label: "Start from topic",
    description: "Create the Watchlist objective first, then add sources before collection starts."
  },
  {
    key: "report_goal",
    label: "Start from report goal",
    description: "Define the briefing goal first, then connect sources and templates."
  }
]

const scheduleOptions: Array<{ key: WatchlistSetupSchedulePreset; label: string }> = [
  { key: "none", label: "Manual" },
  { key: "daily", label: "Daily" },
  { key: "weekdays", label: "Weekdays" },
  { key: "hourly", label: "Hourly" }
]

const stepLabels = ["Start", "Scope", "Collection", "Review"]

const getDestinationLabel = (destination: WatchlistSetupDestination): string => {
  switch (destination) {
    case "jobs":
      return "Monitors"
    case "outputs":
      return "Reports"
    case "sources":
    default:
      return "Feeds"
  }
}

const fieldValue = (event: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>): string =>
  event.target.value

export const WatchlistSetupWizard: React.FC<WatchlistSetupWizardProps> = ({
  open,
  submitting = false,
  onCancel,
  onCreateWatchlist,
  onCreateSources,
  onCreateJob,
  onComplete
}) => {
  const { t } = useTranslation("watchlists")
  const [step, setStep] = useState(0)
  const [values, setValues] = useState<WatchlistSetupValues>(() => createInitialValues())
  const [validationError, setValidationError] = useState<string | null>(null)
  const [internalSubmitting, setInternalSubmitting] = useState(false)

  useEffect(() => {
    if (!open) return
    setStep(0)
    setValues(createInitialValues())
    setValidationError(null)
    setInternalSubmitting(false)
  }, [open])

  const busy = submitting || internalSubmitting
  const presetDefaults = useMemo(() => applyWatchlistSetupPreset(values.preset), [values.preset])
  const setupPlan = useMemo(() => buildWatchlistSetupPlan(values), [values])

  const updateValue = <Key extends keyof WatchlistSetupValues>(
    key: Key,
    value: WatchlistSetupValues[Key]
  ) => {
    setValues((current) => ({ ...current, [key]: value }))
    setValidationError(null)
  }

  const validateName = (): boolean => {
    if (values.name.trim()) return true
    setValidationError(t("watchlists:setupWizard.validation.nameRequired", "Add a Watchlist name."))
    return false
  }

  const handleNext = () => {
    if (step === 1 && !validateName()) return
    setStep((current) => Math.min(current + 1, stepLabels.length - 1))
  }

  const handleBack = () => {
    setValidationError(null)
    setStep((current) => Math.max(current - 1, 0))
  }

  const handleSubmit = async () => {
    if (!validateName()) return
    setInternalSubmitting(true)
    try {
      const latestPlan = buildWatchlistSetupPlan(values)
      const watchlist = await onCreateWatchlist(latestPlan.watchlist)
      let sourceIds: number[] = []
      let job: WatchlistJob | undefined

      if (latestPlan.sources.length > 0) {
        sourceIds = await onCreateSources(watchlist.id, latestPlan.sources)
      }

      if (latestPlan.canCreateMonitor && sourceIds.length > 0) {
        const jobPayload = {
          ...buildWatchlistSetupJobPayload(values, sourceIds),
          watchlist_id: watchlist.id
        }
        job = await onCreateJob(watchlist.id, jobPayload)
      }

      onComplete({
        destination: latestPlan.destination,
        watchlist,
        sourceIds,
        job
      })
    } catch (error) {
      console.error("Failed to create Watchlist setup:", error)
      setValidationError(
        error instanceof Error && error.message
          ? error.message
          : t("watchlists:setupWizard.errors.createFailed", "Failed to create Watchlist setup.")
      )
    } finally {
      setInternalSubmitting(false)
    }
  }

  const renderChoiceButton = <Key extends string>({
    key,
    label,
    description,
    selected,
    onSelect
  }: {
    key: Key
    label: string
    description: string
    selected: boolean
    onSelect: (key: Key) => void
  }) => (
    <button
      key={key}
      type="button"
      aria-label={label}
      className={`rounded-md border px-3 py-2 text-left transition ${
        selected ? "border-primary bg-primary/10 text-text-primary" : "border-border bg-surface text-text-secondary"
      }`}
      aria-pressed={selected}
      onClick={() => onSelect(key)}
    >
      <span className="block text-sm font-semibold">{label}</span>
      <span className="mt-1 block text-xs text-text-muted">{description}</span>
    </button>
  )

  const renderStartStep = () => (
    <div className="space-y-4">
      <section>
        <h3 className="mb-2 text-sm font-semibold">
          {t("watchlists:setupWizard.sections.preset", "Domain preset")}
        </h3>
        <div className="grid gap-2 sm:grid-cols-2">
          {presetOptions.map((option) =>
            renderChoiceButton({
              ...option,
              selected: values.preset === option.key,
              onSelect: (key) => updateValue("preset", key)
            })
          )}
        </div>
      </section>
      <section>
        <h3 className="mb-2 text-sm font-semibold">
          {t("watchlists:setupWizard.sections.startMode", "Starting point")}
        </h3>
        <div className="grid gap-2">
          {startModeOptions.map((option) =>
            renderChoiceButton({
              ...option,
              selected: values.startMode === option.key,
              onSelect: (key) => updateValue("startMode", key)
            })
          )}
        </div>
      </section>
    </div>
  )

  const renderScopeStep = () => (
    <div className="space-y-4">
      <label className="block text-sm font-medium" htmlFor="watchlist-setup-name">
        {t("watchlists:setupWizard.fields.name", "Watchlist name")}
      </label>
      <Input
        id="watchlist-setup-name"
        value={values.name}
        onChange={(event) => updateValue("name", fieldValue(event))}
        placeholder={t("watchlists:setupWizard.fields.namePlaceholder", "e.g., Healthcare ransomware")}
      />

      <label className="block text-sm font-medium" htmlFor="watchlist-setup-objective">
        {t("watchlists:setupWizard.fields.objective", "Objective")}
      </label>
      <Input.TextArea
        id="watchlist-setup-objective"
        value={values.objective}
        onChange={(event) => updateValue("objective", fieldValue(event))}
        placeholder={presetDefaults.objectivePlaceholder}
        rows={3}
      />

      <label className="block text-sm font-medium" htmlFor="watchlist-setup-scope">
        {t("watchlists:setupWizard.fields.trackedScope", "Tracked scope")}
      </label>
      <Input.TextArea
        id="watchlist-setup-scope"
        value={values.trackedScopeText}
        onChange={(event) => updateValue("trackedScopeText", fieldValue(event))}
        placeholder={presetDefaults.trackedScopePlaceholder}
        rows={3}
      />
    </div>
  )

  const renderCollectionStep = () => (
    <div className="space-y-4">
      <Alert
        variant="info"
        title={t("watchlists:setupWizard.boundaries.title", "Collection scope first")}
      >
        {t(
          "watchlists:setupWizard.boundaries.alerts",
          "Content-match alerts come later. This setup defines the Watchlist and its initial collection scope."
        )}
      </Alert>

      <label className="block text-sm font-medium" htmlFor="watchlist-setup-source-name">
        {t("watchlists:setupWizard.fields.sourceName", "Source name")}
      </label>
      <Input
        id="watchlist-setup-source-name"
        value={values.sourceName}
        onChange={(event) => updateValue("sourceName", fieldValue(event))}
        placeholder={t("watchlists:setupWizard.fields.sourceNamePlaceholder", "Optional label for the first feed")}
      />

      <label className="block text-sm font-medium" htmlFor="watchlist-setup-source-urls">
        {t("watchlists:setupWizard.fields.sourceUrls", "Source URLs")}
      </label>
      <Input.TextArea
        id="watchlist-setup-source-urls"
        value={values.sourceUrlsText}
        onChange={(event) => updateValue("sourceUrlsText", fieldValue(event))}
        placeholder={t(
          "watchlists:setupWizard.fields.sourceUrlsPlaceholder",
          "One RSS feed, site, or source URL per line"
        )}
        rows={4}
      />

      <label className="block text-sm font-medium" htmlFor="watchlist-setup-monitor-name">
        {t("watchlists:setupWizard.fields.monitorName", "Monitor name")}
      </label>
      <Input
        id="watchlist-setup-monitor-name"
        value={values.monitorName}
        onChange={(event) => updateValue("monitorName", fieldValue(event))}
        placeholder={t("watchlists:setupWizard.fields.monitorNamePlaceholder", "Defaults to Watchlist monitor")}
      />

      <label className="block text-sm font-medium" htmlFor="watchlist-setup-report-goal">
        {t("watchlists:setupWizard.fields.reportGoal", "Report goal")}
      </label>
      <Input.TextArea
        id="watchlist-setup-report-goal"
        value={values.reportGoal}
        onChange={(event) => updateValue("reportGoal", fieldValue(event))}
        placeholder={presetDefaults.reportGoalPlaceholder}
        rows={3}
      />

      <div className="flex flex-wrap gap-2" aria-label={t("watchlists:setupWizard.fields.schedule", "Schedule")}>
        {scheduleOptions.map((option) => (
          <Button
            key={option.key}
            aria-pressed={values.schedulePreset === option.key}
            onClick={() => updateValue("schedulePreset", option.key)}
          >
            {option.label}
          </Button>
        ))}
      </div>

      <div className="flex items-center gap-3">
        <Switch
          aria-label={t("watchlists:setupWizard.fields.audioBriefing", "Audio briefing")}
          checked={Boolean(values.includeAudioBriefing)}
          onChange={(checked) => updateValue("includeAudioBriefing", checked)}
        />
        <span className="text-sm text-text-secondary">
          {t("watchlists:setupWizard.fields.audioBriefing", "Audio briefing")}
        </span>
      </div>
    </div>
  )

  const renderReviewStep = () => (
    <div className="space-y-4">
      <h3 className="text-base font-semibold">
        {t("watchlists:setupWizard.review.title", "Review Watchlist setup")}
      </h3>
      <dl className="grid gap-3 text-sm sm:grid-cols-2">
        <div>
          <dt className="text-text-muted">{t("watchlists:setupWizard.review.name", "Name")}</dt>
          <dd className="font-medium">{setupPlan.watchlist.name || "Untitled Watchlist"}</dd>
        </div>
        <div>
          <dt className="text-text-muted">{t("watchlists:setupWizard.review.destination", "Next opens")}</dt>
          <dd className="font-medium">{getDestinationLabel(setupPlan.destination)}</dd>
        </div>
        <div>
          <dt className="text-text-muted">{t("watchlists:setupWizard.review.domain", "Domain")}</dt>
          <dd className="font-medium">{setupPlan.watchlist.domain}</dd>
        </div>
        <div>
          <dt className="text-text-muted">{t("watchlists:setupWizard.review.sources", "Sources")}</dt>
          <dd className="font-medium">{setupPlan.sources.length}</dd>
        </div>
      </dl>
      <div className="flex flex-wrap gap-1">
        {(setupPlan.watchlist.tags || []).map((tag) => (
          <Tag key={tag}>{tag}</Tag>
        ))}
      </div>
    </div>
  )

  const renderStep = () => {
    switch (step) {
      case 0:
        return renderStartStep()
      case 1:
        return renderScopeStep()
      case 2:
        return renderCollectionStep()
      case 3:
      default:
        return renderReviewStep()
    }
  }

  const footer = (
    <div className="flex flex-wrap items-center justify-between gap-2">
      <div className="flex flex-wrap gap-1 text-xs text-text-muted" aria-label="Setup steps">
        {stepLabels.map((label, index) => (
          <span key={label} className={index === step ? "font-semibold text-text-primary" : ""}>
            {label}
          </span>
        ))}
      </div>
      <div className="flex gap-2">
        {step > 0 ? (
          <Button onClick={handleBack} disabled={busy}>
            {t("watchlists:setupWizard.actions.back", "Back")}
          </Button>
        ) : null}
        {step < stepLabels.length - 1 ? (
          <Button type="primary" onClick={handleNext} disabled={busy}>
            {t("watchlists:setupWizard.actions.next", "Next")}
          </Button>
        ) : (
          <Button type="primary" onClick={handleSubmit} disabled={busy}>
            {t("watchlists:setupWizard.actions.create", "Create Watchlist")}
          </Button>
        )}
      </div>
    </div>
  )

  return (
    <Modal
      open={open}
      title={t("watchlists:setupWizard.title", "Create Watchlist")}
      onCancel={onCancel}
      footer={footer}
      width={720}
      destroyOnHidden
    >
      <div className="space-y-4">
        {validationError ? (
          <Alert title={validationError} variant="error" />
        ) : null}
        {renderStep()}
      </div>
    </Modal>
  )
}
