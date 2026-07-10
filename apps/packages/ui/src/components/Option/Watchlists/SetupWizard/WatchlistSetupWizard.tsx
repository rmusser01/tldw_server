import React, { useEffect, useRef, useState } from "react"
import { Button, Input, Modal, Select } from "antd"
import { useTranslation } from "react-i18next"
import { Alert } from "@/components/ui/primitives"
import {
  getWatchlistRunBriefing,
  testWatchlistSourceDraft,
  triggerWatchlistRun
} from "@/services/watchlists"
import type {
  WatchlistContainer,
  WatchlistCreate,
  WatchlistJob,
  WatchlistJobCreate,
  WatchlistSource,
  WatchlistSourceCreate
} from "@/types/watchlists"
import {
  PipelineWizard,
  type PipelineWizardTestOptions
} from "../OverviewTab/PipelineWizard"
import {
  toBriefingPipelineDraft,
  getPipelineWizardSourceSignature,
  toPipelineWizardSourcePayload,
  waitForPipelineWizardBriefing,
  type PipelineWizardSourceBinding,
  type PipelineWizardDraft
} from "../OverviewTab/pipeline-wizard-state"
import {
  toPipelineJobCreatePayload,
  toPipelineTestJobCreatePayload
} from "../OverviewTab/pipeline-contract"
import {
  applyWatchlistSetupPreset,
  buildWatchlistSetupPlan,
  type WatchlistSetupDestination,
  type WatchlistSetupPreset,
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
  sources?: WatchlistSource[]
  onCancel: () => void
  onCreateWatchlist: (payload: WatchlistCreate) => Promise<WatchlistContainer>
  onCreateSources: (watchlistId: number, sources: WatchlistSourceCreate[]) => Promise<number[]>
  onCreateJob: (watchlistId: number, job: WatchlistJobCreate) => Promise<WatchlistJob>
  onUpdateJob: (jobId: number, job: WatchlistJobCreate | { active: true }) => Promise<WatchlistJob>
  onComplete: (result: WatchlistSetupCompleteResult) => void
}

const initialValues = (): WatchlistSetupValues => ({
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
  includeAudioBriefing: true,
  schedulePreset: "daily"
})

export const WatchlistSetupWizard: React.FC<WatchlistSetupWizardProps> = ({
  open,
  submitting = false,
  sources = [],
  onCancel,
  onCreateWatchlist,
  onCreateSources,
  onCreateJob,
  onUpdateJob,
  onComplete
}) => {
  const { t } = useTranslation("watchlists")
  const [values, setValues] = useState<WatchlistSetupValues>(() => initialValues())
  const [watchlist, setWatchlist] = useState<WatchlistContainer | null>(null)
  const [pipelineDraft, setPipelineDraft] = useState<Partial<PipelineWizardDraft>>()
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const sourceBindingRef = useRef<PipelineWizardSourceBinding | null>(null)
  const jobRef = useRef<WatchlistJob | null>(null)

  useEffect(() => {
    if (!open) return
    setValues(initialValues())
    setWatchlist(null)
    setPipelineDraft(undefined)
    setBusy(false)
    setError(null)
    sourceBindingRef.current = null
    jobRef.current = null
  }, [open])

  const updateValue = <Key extends keyof WatchlistSetupValues>(
    key: Key,
    value: WatchlistSetupValues[Key]
  ) => {
    setValues((current) => ({ ...current, [key]: value }))
    setError(null)
  }

  const continueToSources = async () => {
    if (!values.name.trim()) {
      setError(t("watchlists:setupWizard.validation.nameRequired", "Add a Watchlist name."))
      return
    }
    setBusy(true)
    setError(null)
    try {
      const created = await onCreateWatchlist(buildWatchlistSetupPlan(values).watchlist)
      setWatchlist(created)
      setPipelineDraft({
        sourceMode: "new",
        monitorName: `${created.name} monitor`,
        templateName: "briefing_markdown",
        audioEnabled: Boolean(values.includeAudioBriefing),
        createScheduledOutput: true
      })
    } catch (cause) {
      setError(
        cause instanceof Error && cause.message
          ? cause.message
          : t("watchlists:setupWizard.errors.createFailed", "Could not create the Watchlist.")
      )
    } finally {
      setBusy(false)
    }
  }

  const persistInactive = async (
    draft: PipelineWizardDraft,
    requestedJobId?: number,
    testOptions?: PipelineWizardTestOptions
  ) => {
    if (!watchlist) throw new Error("Watchlist container is missing.")
    const signature = getPipelineWizardSourceSignature(draft)
    let sourceIds = draft.sourceIds
    if (draft.sourceMode === "new" && sourceBindingRef.current?.signature !== signature) {
      const source = toPipelineWizardSourcePayload(draft, watchlist.id)
      if (!source) throw new Error("Source details are incomplete.")
      sourceIds = await onCreateSources(watchlist.id, [source])
      sourceBindingRef.current = { signature, ids: sourceIds }
    } else if (draft.sourceMode === "new") {
      sourceIds = sourceBindingRef.current?.ids || []
    } else {
      sourceBindingRef.current = { signature, ids: sourceIds }
    }
    const pipeline = toBriefingPipelineDraft(draft, sourceIds)
    const jobPayload = testOptions
      ? toPipelineTestJobCreatePayload(pipeline, testOptions)
      : toPipelineJobCreatePayload({ ...pipeline, active: false })
    const payload: WatchlistJobCreate = {
      ...jobPayload,
      active: false,
      watchlist_id: watchlist.id
    }
    const existingJobId = requestedJobId || jobRef.current?.id
    if (existingJobId) {
      const updated = await onUpdateJob(existingJobId, payload)
      jobRef.current = updated
      return { job: updated, pipeline }
    }
    const created = await onCreateJob(watchlist.id, payload)
    jobRef.current = created
    return { job: created, pipeline }
  }

  const testPipeline = async (
    draft: PipelineWizardDraft,
    options: PipelineWizardTestOptions,
    onProgress: Parameters<typeof waitForPipelineWizardBriefing>[2]
  ) => {
    const persisted = await persistInactive(draft, options.jobId, options)
    const run = await triggerWatchlistRun(persisted.job.id)
    const briefing = await waitForPipelineWizardBriefing(
      run.id,
      getWatchlistRunBriefing,
      onProgress,
      { waitForDelivery: options.externalDelivery }
    )
    return {
      jobId: persisted.job.id,
      runId: run.id,
      status: "ready" as const,
      briefing
    }
  }

  const activatePipeline = async (draft: PipelineWizardDraft, options: { jobId?: number }) => {
    if (!watchlist) throw new Error("Watchlist container is missing.")
    const persisted = await persistInactive(draft, options.jobId)
    const activeJob = await onUpdateJob(persisted.job.id, { active: true })
    jobRef.current = activeJob
    onComplete({
      destination: "jobs",
      watchlist,
      sourceIds: sourceBindingRef.current?.ids || [],
      job: activeJob
    })
    return { jobId: activeJob.id, status: "active" as const }
  }

  const testSource = async (draft: PipelineWizardDraft) => {
    if (draft.sourceMode === "new") {
      const result = await testWatchlistSourceDraft({
        url: draft.sourceUrl,
        source_type: draft.sourceType
      }, { limit: 6 })
      return { status: "ready" as const, sourceTest: result }
    } else {
      const selectedSources = sources.filter((source) => draft.sourceIds.includes(source.id))
      const results = await Promise.all(
        selectedSources.map((source) =>
          testWatchlistSourceDraft(
            { url: source.url, source_type: source.source_type },
            { limit: 6 }
          )
        )
      )
      return {
        status: "ready" as const,
        sourceTest: {
          total: results.reduce((total, result) => total + result.total, 0),
          ingestable: results.reduce((total, result) => total + result.ingestable, 0),
          filtered: results.reduce((total, result) => total + result.filtered, 0),
          items: results.flatMap((result) => result.items).slice(0, 6)
        }
      }
    }
  }

  if (watchlist) {
    return (
      <PipelineWizard
        open={open}
        sessionKey={watchlist.id}
        initialStep="sources"
        initialDraft={pipelineDraft}
        sources={sources}
        sourcesLoading={false}
        submitting={submitting || busy}
        submitError={error}
        onCancel={() => {
          onComplete({ destination: "sources", watchlist, sourceIds: sourceBindingRef.current?.ids || [] })
          onCancel()
        }}
        onSaveDraft={setPipelineDraft}
        onTest={testPipeline}
        onActivate={activatePipeline}
        onTestSource={testSource}
      />
    )
  }

  const preset = applyWatchlistSetupPreset(values.preset)
  return (
    <Modal
      open={open}
      title={t("watchlists:setupWizard.containerTitle", "Create Watchlist")}
      onCancel={busy ? undefined : onCancel}
      destroyOnHidden
      width={640}
      footer={[
        <Button key="cancel" className="min-h-11" onClick={onCancel} disabled={busy}>
          {t("common:cancel", "Cancel")}
        </Button>,
        <Button key="continue" type="primary" className="min-h-11" onClick={() => void continueToSources()} loading={busy}>
          {t("watchlists:setupWizard.actions.continue", "Continue to Sources")}
        </Button>
      ]}
    >
      <div className="space-y-4">
        {error && <Alert title={error} variant="error" />}
        <p className="text-sm text-text-muted">
          {t("watchlists:setupWizard.containerHelp", "Create the project container first. Sources, cadence, briefing, delivery, and Test follow in one setup flow.")}
        </p>
        <label className="block text-sm font-medium" htmlFor="watchlist-setup-name">
          {t("watchlists:setupWizard.fields.name", "Watchlist name")}
        </label>
        <Input
          id="watchlist-setup-name"
          aria-label={t("watchlists:setupWizard.fields.name", "Watchlist name")}
          value={values.name}
          onChange={(event) => updateValue("name", event.target.value)}
        />
        <label className="block text-sm font-medium" htmlFor="watchlist-setup-preset">
          {t("watchlists:setupWizard.sections.preset", "Domain preset")}
        </label>
        <Select
          id="watchlist-setup-preset"
          aria-label={t("watchlists:setupWizard.sections.preset", "Domain preset")}
          className="w-full"
          value={values.preset}
          options={[
            { value: "general", label: t("watchlists:setupWizard.presets.general.label", "General") },
            { value: "news", label: t("watchlists:setupWizard.presets.news.label", "News") },
            { value: "cti_osint", label: t("watchlists:setupWizard.presets.cti_osint.label", "CTI / OSINT") },
            { value: "blank", label: t("watchlists:setupWizard.presets.blank.label", "Blank") }
          ]}
          onChange={(value) => updateValue("preset", value as WatchlistSetupPreset)}
        />
        <label className="block text-sm font-medium" htmlFor="watchlist-setup-objective">
          {t("watchlists:setupWizard.fields.objective", "Objective")}
        </label>
        <Input.TextArea
          id="watchlist-setup-objective"
          aria-label={t("watchlists:setupWizard.fields.objective", "Objective")}
          value={values.objective}
          placeholder={preset.objectivePlaceholder}
          rows={3}
          onChange={(event) => updateValue("objective", event.target.value)}
        />
      </div>
    </Modal>
  )
}
