import React, { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react"
import {
  Button,
  Card,
  Empty,
  List,
  message,
  Space,
  Steps,
  Spin,
  Statistic,
  Tag,
  Tooltip
} from "antd"
import {
  AlertTriangle,
  BellRing,
  CheckCircle2,
  Newspaper,
  RefreshCw,
  Rss,
  Workflow
} from "lucide-react"
import { useTranslation } from "react-i18next"
import { useWatchlistsStore } from "@/store/watchlists"
import {
  createWatchlistJob,
  createWatchlistSource,
  fetchWatchlistRuns,
  fetchWatchlistSources,
  getWatchlistTemplate,
  getWatchlistRunBriefing,
  previewWatchlistSchedule,
  previewWatchlistTemplate,
  retryWatchlistBriefingStage,
  testWatchlistSourceDraft,
  triggerWatchlistRun,
  updateWatchlistJob,
  updateWatchlistSource
} from "@/services/watchlists"
import {
  fetchWatchlistsOverviewData,
  getOverviewTabBadges,
  type WatchlistsOverviewData
} from "@/services/watchlists-overview"
import { formatRelativeTime } from "@/utils/dateFormatters"
import { Alert as DesignSystemAlert } from "@/components/ui"
import {
  toPipelineJobCreatePayload,
  toPipelineTestJobCreatePayload
} from "./pipeline-contract"
import {
  PipelineWizard,
  type PipelineWizardTestOptions
} from "./PipelineWizard"
import {
  buildPipelineWizardSchedule,
  toBriefingPipelineDraft,
  getPipelineWizardBriefingOutcome,
  getPipelineWizardSourceSignature,
  toPipelineWizardSourcePayload,
  type PipelineWizardSourceBinding,
  type PipelineWizardDraft,
  waitForPipelineWizardBriefing
} from "./pipeline-wizard-state"
import {
  type WatchlistsOnboardingPath,
  readWatchlistsOnboardingPath,
  writeWatchlistsOnboardingPath
} from "../shared/onboarding-path"
import {
  getFocusableActiveElement,
  restoreFocusToElement
} from "../shared/focus-management"
import { isWatchlistRunSuccessful } from "../shared/runStatus"
import { normalizeWatchlistTemplateName } from "../shared/templateNames"
import { trackWatchlistsOnboardingTelemetry } from "@/utils/watchlists-onboarding-telemetry"
import type { WatchlistSource } from "@/types/watchlists"
import type {
  WatchlistBriefingProjection,
  WatchlistBriefingRetryStage
} from "@/types/watchlists"
import { LatestBriefing } from "./LatestBriefing"
import {
  blockingFailureAnnouncement,
  transitionAnnouncement
} from "../shared/watchlists-announcements"

const OVERVIEW_REFRESH_INTERVAL_MS = 30_000

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value)

export const extractPipelineErrorMessage = (error: unknown): string => {
  if (typeof error === "string") return error.trim()
  if (!isRecord(error)) {
    return error instanceof Error && error.message.trim().length > 0
      ? error.message.trim()
      : ""
  }

  const response = isRecord(error.response) ? error.response : undefined
  const data = response && isRecord(response.data) ? response.data : undefined
  const detail = data ? data.detail : error.detail

  if (typeof detail === "string") return detail.trim()
  if (isRecord(detail)) {
    const message = detail.message
    if (typeof message === "string" && message.trim().length > 0) {
      return message.trim()
    }
    const code = detail.code
    if (typeof code === "string" && code.trim().length > 0) {
      return code.trim()
    }
  }

  const message = error.message
  if (typeof message === "string" && message.trim().length > 0) {
    return message.trim()
  }

  return error instanceof Error && error.message.trim().length > 0
    ? error.message.trim()
    : ""
}

export const OverviewTab: React.FC = () => {
  const { t } = useTranslation(["watchlists", "common"])
  const [data, setData] = useState<WatchlistsOverviewData | null>(null)
  const [loading, setLoading] = useState(false)
  const [refreshing, setRefreshing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [politeAnnouncement, setPoliteAnnouncement] = useState("")
  const [assertiveAnnouncement, setAssertiveAnnouncement] = useState("")
  const [pipelineSetupOpen, setPipelineSetupOpen] = useState(false)
  const [pipelineSetupSubmitting, setPipelineSetupSubmitting] = useState(false)
  const [pipelineSourcesLoading, setPipelineSourcesLoading] = useState(false)
  const [pipelineSources, setPipelineSources] = useState<WatchlistSource[]>([])
  const [pipelinePreviewLoading, setPipelinePreviewLoading] = useState(false)
  const [pipelinePreviewError, setPipelinePreviewError] = useState<string | null>(null)
  const [pipelinePreviewRendered, setPipelinePreviewRendered] = useState<string | null>(null)
  const [pipelinePreviewRunId, setPipelinePreviewRunId] = useState<number | null>(null)
  const [pipelinePreviewWarnings, setPipelinePreviewWarnings] = useState<string[]>([])
  const [pipelineSetupError, setPipelineSetupError] = useState<string | null>(null)
  const [pipelineInitialDraft, setPipelineInitialDraft] = useState<Partial<PipelineWizardDraft>>()
  const [pipelineSessionKey, setPipelineSessionKey] = useState(0)
  const [onboardingPath, setOnboardingPath] = useState<WatchlistsOnboardingPath>(() =>
    readWatchlistsOnboardingPath()
  )
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const overviewRequestRef = useRef(0)
  const previousBriefingRef = useRef<WatchlistBriefingProjection | null>(null)
  const quickSetupTriggerRef = useRef<HTMLButtonElement | null>(null)
  const pipelineSetupRestoreFocusTargetRef = useRef<HTMLElement | null>(null)
  const pipelineSetupWasOpenRef = useRef(false)
  const pipelinePersistedJobIdRef = useRef<number | null>(null)
  const pipelineSourceBindingRef = useRef<PipelineWizardSourceBinding | null>(null)

  const setActiveTab = useWatchlistsStore((s) => s.setActiveTab)
  const setRunsStatusFilter = useWatchlistsStore((s) => s.setRunsStatusFilter)
  const setOverviewHealth = useWatchlistsStore((s) => s.setOverviewHealth)
  const openRunDetail = useWatchlistsStore((s) => s.openRunDetail)
  const openSourceForm = useWatchlistsStore((s) => s.openSourceForm)
  const openJobForm = useWatchlistsStore((s) => s.openJobForm)
  const openOutputPreview = useWatchlistsStore((s) => s.openOutputPreview)
  const selectedWatchlistId = useWatchlistsStore((s) => s.selectedWatchlistId)
  const hasSelectedWatchlist = selectedWatchlistId != null

  const announceBriefingTransition = useCallback((next: WatchlistBriefingProjection | null) => {
    const previous = previousBriefingRef.current
    if (!next) {
      previousBriefingRef.current = null
      setPoliteAnnouncement("")
      setAssertiveAnnouncement("")
      return
    }
    setPoliteAnnouncement(transitionAnnouncement(previous, next, t) || "")
    setAssertiveAnnouncement(blockingFailureAnnouncement(previous, next, t) || "")
    previousBriefingRef.current = next
  }, [t])

  const applyBriefingProjection = useCallback((next: WatchlistBriefingProjection) => {
    announceBriefingTransition(next)
    setData((current) => current ? {
      ...current,
      fetchedAt: new Date().toISOString(),
      latestBriefing: next
    } : current)
  }, [announceBriefingTransition])

  const loadOverview = useCallback(async (showLoading: boolean) => {
    const requestId = ++overviewRequestRef.current
    if (selectedWatchlistId == null) {
      setData(null)
      setError(null)
      previousBriefingRef.current = null
      setPoliteAnnouncement("")
      setAssertiveAnnouncement("")
      if (typeof setOverviewHealth === "function") {
        setOverviewHealth(null, null)
      }
      setLoading(false)
      setRefreshing(false)
      return
    }

    if (showLoading) {
      setLoading(true)
    } else {
      setRefreshing(true)
    }
    try {
      const result = await fetchWatchlistsOverviewData({
        watchlist_id: selectedWatchlistId ?? undefined
      })
      if (requestId !== overviewRequestRef.current) return
      announceBriefingTransition(result.latestBriefing)
      setData(result)
      if (typeof setOverviewHealth === "function") {
        setOverviewHealth(result.health, result.fetchedAt)
      }
      if (result.outputs.total > 0) {
        void trackWatchlistsOnboardingTelemetry({
          type: "quick_setup_first_run_succeeded",
          source: "overview"
        })
        void trackWatchlistsOnboardingTelemetry({
          type: "quick_setup_first_output_succeeded",
          source: "overview"
        })
      }
      setError(null)
    } catch (err) {
      if ((err as { name?: string } | null)?.name === "AbortError") return
      if (requestId !== overviewRequestRef.current) return
      console.error("Failed to load watchlists overview:", err)
      setError(t("watchlists:overview.fetchError", "Failed to load overview"))
    } finally {
      if (requestId === overviewRequestRef.current) {
        setLoading(false)
        setRefreshing(false)
      }
    }
  }, [announceBriefingTransition, selectedWatchlistId, setOverviewHealth, t])

  useEffect(() => {
    void loadOverview(true)
    intervalRef.current = setInterval(() => {
      void loadOverview(false)
    }, OVERVIEW_REFRESH_INTERVAL_MS)

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
      intervalRef.current = null
    }
  }, [loadOverview])

  const quickSetupAutoShownWatchlistRef = useRef<number | null>(null)

  useLayoutEffect(() => {
    if (pipelineSetupOpen) {
      if (!pipelineSetupWasOpenRef.current) {
        pipelineSetupRestoreFocusTargetRef.current =
          pipelineSetupRestoreFocusTargetRef.current || getFocusableActiveElement()
      }
      pipelineSetupWasOpenRef.current = true
      return
    }

    if (pipelineSetupWasOpenRef.current) {
      pipelineSetupWasOpenRef.current = false
      restoreFocusToElement(pipelineSetupRestoreFocusTargetRef.current)
      pipelineSetupRestoreFocusTargetRef.current = null
    }
  }, [pipelineSetupOpen])

  const handleOpenRun = useCallback((runId: number) => {
    setActiveTab("runs")
    openRunDetail(runId)
  }, [openRunDetail, setActiveTab])

  const handleOpenItems = useCallback(() => {
    setActiveTab("items")
  }, [setActiveTab])

  const handleOpenJobs = useCallback(() => {
    setActiveTab("jobs")
  }, [setActiveTab])

  const handleOpenSources = useCallback(() => {
    setActiveTab("sources")
  }, [setActiveTab])

  const handleStartSourceQuickCreate = useCallback(() => {
    setActiveTab("sources")
    openSourceForm()
  }, [openSourceForm, setActiveTab])

  const handleStartJobQuickCreate = useCallback(() => {
    setActiveTab("jobs")
    openJobForm()
  }, [openJobForm, setActiveTab])

  const handleOpenRuns = useCallback(() => {
    setActiveTab("runs")
  }, [setActiveTab])

  const handleOpenBriefingReport = useCallback((outputId: number) => {
    setActiveTab("outputs")
    openOutputPreview(outputId)
  }, [openOutputPreview, setActiveTab])

  const handleRetryBriefingStage = useCallback(async (
    runId: number,
    stage: WatchlistBriefingRetryStage,
    options?: { confirm_unknown_delivery_retry?: boolean }
  ) => {
    try {
      const next = await retryWatchlistBriefingStage(runId, stage, options)
      applyBriefingProjection(next)
    } catch (err) {
      if ((err as { name?: string } | null)?.name === "AbortError") return
      console.error("Failed to retry watchlists briefing stage:", err)
      message.error(t(
        "watchlists:overview.latest.retryError",
        "Could not retry this briefing stage. Inspect the run for details."
      ))
    }
  }, [applyBriefingProjection, t])

  const handleRegenerateBriefing = useCallback(async (
    runId: number,
    stage: WatchlistBriefingRetryStage
  ) => {
    try {
      const next = await retryWatchlistBriefingStage(runId, stage, { regenerate: true })
      applyBriefingProjection(next)
    } catch (err) {
      if ((err as { name?: string } | null)?.name === "AbortError") return
      console.error("Failed to regenerate watchlists briefing stage:", err)
      message.error(t(
        "watchlists:overview.latest.regenerateError",
        "Could not regenerate audio. Inspect the run for details."
      ))
    }
  }, [applyBriefingProjection, t])

  const handleOpenAlerts = useCallback(() => {
    setActiveTab("alerts")
  }, [setActiveTab])

  const handleOpenFailedRuns = useCallback(() => {
    if (typeof setRunsStatusFilter === "function" && (data?.runs.failed ?? 0) > 0) {
      setRunsStatusFilter("failed")
    }
    setActiveTab("runs")
  }, [data?.runs.failed, setActiveTab, setRunsStatusFilter])

  const handleOpenAttentionOutputs = useCallback(() => {
    setActiveTab("outputs")
  }, [setActiveTab])

  const handleOpenAttentionSources = useCallback(() => {
    setActiveTab("sources")
  }, [setActiveTab])

  const handleOnboardingPathChange = useCallback((path: WatchlistsOnboardingPath) => {
    setOnboardingPath(path)
    writeWatchlistsOnboardingPath(path)
  }, [])

  const markQuickSetupAutoShown = useCallback(() => {
    if (selectedWatchlistId == null) return
    quickSetupAutoShownWatchlistRef.current = selectedWatchlistId
    try {
      localStorage.setItem(`watchlists:quickSetup:autoshown:v1:${selectedWatchlistId}`, "1")
    } catch {
      // Keep setup usable when storage is unavailable.
    }
  }, [selectedWatchlistId])

  const openQuickSetup = useCallback((restoreTarget?: HTMLElement | null) => {
    pipelineSetupRestoreFocusTargetRef.current =
      restoreTarget || quickSetupTriggerRef.current || getFocusableActiveElement()
    markQuickSetupAutoShown()
    setPipelineInitialDraft({
      sourceMode: "new",
      audioEnabled: true,
      createScheduledOutput: true,
      templateName: "briefing_markdown"
    })
    pipelinePersistedJobIdRef.current = null
    pipelineSourceBindingRef.current = null
    setPipelineSessionKey((value) => value + 1)
    setPipelineSetupOpen(true)
    void trackWatchlistsOnboardingTelemetry({ type: "quick_setup_opened" })
  }, [markQuickSetupAutoShown])

  // Auto-open Quick Setup for first-time users with no sources
  useEffect(() => {
    if (selectedWatchlistId == null) return
    const autoShownKey = `watchlists:quickSetup:autoshown:v1:${selectedWatchlistId}`

    if (
      !data ||
      data.sources.total !== 0 ||
      quickSetupAutoShownWatchlistRef.current === selectedWatchlistId ||
      onboardingPath !== "beginner"
    ) {
      return
    }

    try {
      if (localStorage.getItem(autoShownKey)) {
        quickSetupAutoShownWatchlistRef.current = selectedWatchlistId
        return
      }
      localStorage.setItem(autoShownKey, "1")
      quickSetupAutoShownWatchlistRef.current = selectedWatchlistId
      openQuickSetup()
    } catch {
      quickSetupAutoShownWatchlistRef.current = selectedWatchlistId
      openQuickSetup()
    }
  }, [data, onboardingPath, openQuickSetup, selectedWatchlistId])

  const loadPipelineSources = useCallback(async () => {
    setPipelineSourcesLoading(true)
    try {
      const result = await fetchWatchlistSources({
        watchlist_id: selectedWatchlistId ?? undefined,
        page: 1,
        size: 200
      })
      const items = Array.isArray(result.items) ? result.items : []
      setPipelineSources(items)
    } catch (err) {
      console.error("Failed to load watchlist sources for pipeline setup:", err)
      setPipelineSources([])
      message.error(
        t(
          "watchlists:overview.pipelineSetup.sourcesError",
          "Failed to load feeds for the pipeline builder."
        )
      )
    } finally {
      setPipelineSourcesLoading(false)
    }
  }, [selectedWatchlistId, t])

  useEffect(() => {
    if (pipelineSetupOpen && pipelineSources.length === 0) {
      void loadPipelineSources()
    }
  }, [loadPipelineSources, pipelineSetupOpen, pipelineSources.length])

  const openPipelineSetup = useCallback(() => {
    pipelineSetupRestoreFocusTargetRef.current = getFocusableActiveElement()
    setPipelinePreviewError(null)
    setPipelinePreviewRendered(null)
    setPipelinePreviewRunId(null)
    setPipelinePreviewWarnings([])
    setPipelineSetupError(null)
    setPipelineInitialDraft(undefined)
    pipelinePersistedJobIdRef.current = null
    pipelineSourceBindingRef.current = null
    setPipelineSessionKey((value) => value + 1)
    setPipelineSetupOpen(true)
    void trackWatchlistsOnboardingTelemetry({ type: "pipeline_setup_opened" })
    void loadPipelineSources()
  }, [loadPipelineSources])

  const handleTestLatest = useCallback(async (jobId?: number) => {
    if (!jobId) {
      openPipelineSetup()
      return
    }
    try {
      await triggerWatchlistRun(jobId)
      setPoliteAnnouncement(t(
        "watchlists:overview.latest.announcements.testQueued",
        "Test run queued. Progress will appear in the latest briefing."
      ))
      void loadOverview(false)
    } catch (err) {
      if ((err as { name?: string } | null)?.name === "AbortError") return
      console.error("Failed to start watchlists briefing test:", err)
      message.error(t("watchlists:overview.latest.testError", "Could not start the test run."))
    }
  }, [loadOverview, openPipelineSetup, t])

  const closePipelineSetup = useCallback(() => {
    if (pipelineSetupSubmitting) return
    setPipelineSetupOpen(false)
    setPipelinePreviewLoading(false)
    setPipelinePreviewError(null)
    setPipelinePreviewRendered(null)
    setPipelinePreviewRunId(null)
    setPipelinePreviewWarnings([])
    setPipelineSetupError(null)
    setPipelineInitialDraft(undefined)
  }, [pipelineSetupSubmitting])

  const generatePipelineTemplatePreview = useCallback(async (wizardDraft: PipelineWizardDraft) => {
    setPipelinePreviewLoading(true)
    setPipelinePreviewError(null)
    setPipelinePreviewRendered(null)
    setPipelinePreviewRunId(null)
    setPipelinePreviewWarnings([])

    try {
      const draft = toBriefingPipelineDraft(wizardDraft)
      const templateName = normalizeWatchlistTemplateName(draft.templateName)
      if (!templateName) {
        setPipelinePreviewError(
          t(
            "watchlists:overview.pipelineSetup.preview.templateRequired",
            "Select a template before generating preview."
          )
        )
        void trackWatchlistsOnboardingTelemetry({
          type: "pipeline_setup_preview_generated",
          status: "error"
        })
        return
      }

      const runResult = await fetchWatchlistRuns({
        watchlist_id: selectedWatchlistId ?? undefined,
        page: 1,
        size: 50
      })
      const completedRun = (Array.isArray(runResult.items) ? runResult.items : []).find(
        (run) => isWatchlistRunSuccessful(run.status)
      )
      if (!completedRun) {
        setPipelinePreviewError(
          t(
            "watchlists:overview.pipelineSetup.preview.noRunContext",
            "Run any monitor once, then generate template preview."
          )
        )
        void trackWatchlistsOnboardingTelemetry({
          type: "pipeline_setup_preview_generated",
          status: "no_run_context"
        })
        return
      }

      const template = await getWatchlistTemplate(templateName)
      const templateContent = String(template.content || "")
      const templateFormat = template.format === "html" ? "html" : "md"
      if (!templateContent.trim()) {
        setPipelinePreviewError(
          t(
            "watchlists:overview.pipelineSetup.preview.templateEmpty",
            "Template has no content. Save template content before previewing."
          )
        )
        void trackWatchlistsOnboardingTelemetry({
          type: "pipeline_setup_preview_generated",
          status: "template_empty",
          run_id: completedRun.id
        })
        return
      }

      const previewResult = await previewWatchlistTemplate(
        templateContent,
        completedRun.id,
        templateFormat
      )
      const rendered = String(previewResult.rendered || "")
      const warningCount = Array.isArray(previewResult.warnings)
        ? previewResult.warnings.filter(
            (warning) => typeof warning === "string" && warning.trim().length > 0
          ).length
        : 0
      setPipelinePreviewRunId(completedRun.id)
      setPipelinePreviewRendered(rendered)
      setPipelinePreviewWarnings(
        Array.isArray(previewResult.warnings)
          ? previewResult.warnings
            .filter((warning) => typeof warning === "string" && warning.trim().length > 0)
            .map((warning) => warning.trim())
          : []
      )
      if (!rendered.trim()) {
        setPipelinePreviewError(
          t(
            "watchlists:overview.pipelineSetup.preview.emptyResult",
            "Template preview returned no output for this run context."
          )
        )
        void trackWatchlistsOnboardingTelemetry({
          type: "pipeline_setup_preview_generated",
          status: "empty",
          run_id: completedRun.id,
          warning_count: warningCount
        })
      } else {
        void trackWatchlistsOnboardingTelemetry({
          type: "pipeline_setup_preview_generated",
          status: "success",
          run_id: completedRun.id,
          warning_count: warningCount
        })
      }
    } catch (err) {
      console.error("Failed to generate pipeline template preview:", err)
      setPipelinePreviewError(
        t(
          "watchlists:overview.pipelineSetup.preview.error",
          "Template preview failed. Verify template and run context, then retry."
        )
      )
      void trackWatchlistsOnboardingTelemetry({
        type: "pipeline_setup_preview_generated",
        status: "error"
      })
    } finally {
      setPipelinePreviewLoading(false)
    }
  }, [selectedWatchlistId, t])

  const persistInactivePipeline = useCallback(async (
    wizardDraft: PipelineWizardDraft,
    requestedJobId?: number,
    testOptions?: PipelineWizardTestOptions
  ) => {
    const signature = getPipelineWizardSourceSignature(wizardDraft)
    const sourcePayload = toPipelineWizardSourcePayload(
      wizardDraft,
      selectedWatchlistId ?? undefined
    )
    let sourceIds = wizardDraft.sourceIds
    if (wizardDraft.sourceMode === "new" && sourcePayload) {
      const binding = pipelineSourceBindingRef.current
      const ownsBoundSource = binding?.mode === "new" &&
        binding.createdByWizard &&
        binding.sessionKey === pipelineSessionKey &&
        binding.ids.length === 1
      if (ownsBoundSource && binding.signature !== signature) {
        const updatePayload = { ...sourcePayload }
        delete updatePayload.watchlist_id
        await updateWatchlistSource(binding.ids[0], updatePayload)
        sourceIds = binding.ids
        pipelineSourceBindingRef.current = {
          mode: "new",
          signature,
          ids: sourceIds,
          createdByWizard: true,
          sessionKey: pipelineSessionKey
        }
      } else if (ownsBoundSource && binding.signature === signature) {
        sourceIds = binding.ids
      } else {
        const source = await createWatchlistSource(sourcePayload)
        sourceIds = [source.id]
        pipelineSourceBindingRef.current = {
          mode: "new",
          signature,
          ids: sourceIds,
          createdByWizard: true,
          sessionKey: pipelineSessionKey
        }
      }
    } else if (wizardDraft.sourceMode === "new") {
      if (!sourcePayload) throw new Error("Source details are incomplete.")
    } else {
      pipelineSourceBindingRef.current = {
        mode: "existing",
        signature,
        ids: sourceIds,
        createdByWizard: false,
        sessionKey: pipelineSessionKey
      }
    }

    const draft = toBriefingPipelineDraft(wizardDraft, sourceIds)
    const jobPayload = testOptions
      ? toPipelineTestJobCreatePayload(draft, testOptions)
      : toPipelineJobCreatePayload({ ...draft, active: false })
    const payload = {
      ...jobPayload,
      active: false,
      watchlist_id: selectedWatchlistId ?? undefined
    }
    const jobId = requestedJobId || pipelinePersistedJobIdRef.current
    if (jobId) {
      await updateWatchlistJob(jobId, payload)
      pipelinePersistedJobIdRef.current = jobId
      return { jobId, draft }
    }

    const job = await createWatchlistJob(payload)
    pipelinePersistedJobIdRef.current = job.id
    return { jobId: job.id, draft }
  }, [pipelineSessionKey, selectedWatchlistId])

  const testPipeline = useCallback(async (
    wizardDraft: PipelineWizardDraft,
    options: PipelineWizardTestOptions,
    onProgress: Parameters<typeof waitForPipelineWizardBriefing>[2]
  ) => {
    setPipelineSetupSubmitting(true)
    try {
      const persisted = await persistInactivePipeline(wizardDraft, options.jobId, options)
      const run = await triggerWatchlistRun(persisted.jobId)
      const briefing = await waitForPipelineWizardBriefing(
        run.id,
        getWatchlistRunBriefing,
        onProgress,
        { waitForDelivery: options.externalDelivery, signal: options.signal }
      )
      const outcome = getPipelineWizardBriefingOutcome(briefing, options.externalDelivery)
      void loadOverview(false)
      return {
        jobId: persisted.jobId,
        runId: run.id,
        status: outcome.status,
        briefing
      }
    } finally {
      setPipelineSetupSubmitting(false)
    }
  }, [loadOverview, persistInactivePipeline])

  const previewPipelineSchedule = useCallback(async (
    wizardDraft: PipelineWizardDraft,
    options: { signal: AbortSignal }
  ) => {
    const schedule = buildPipelineWizardSchedule(wizardDraft)
    if (!schedule.schedule_expr) return {}
    const preview = await previewWatchlistSchedule({
      schedule_expr: schedule.schedule_expr,
      timezone: schedule.timezone || "UTC"
    }, options.signal)
    return {
      ...(preview.next_run_at ? { nextRunAt: preview.next_run_at } : {}),
      ...(preview.following_run_at ? { followingRunAt: preview.following_run_at } : {})
    }
  }, [])

  const activatePipeline = useCallback(async (
    wizardDraft: PipelineWizardDraft,
    options: { jobId?: number }
  ) => {
    setPipelineSetupSubmitting(true)
    try {
      const persisted = await persistInactivePipeline(wizardDraft, options.jobId)
      await updateWatchlistJob(persisted.jobId, { active: true })
      closePipelineSetup()
      void loadOverview(false)
      setActiveTab("jobs")
      return { jobId: persisted.jobId, status: "active" as const }
    } finally {
      setPipelineSetupSubmitting(false)
    }
  }, [closePipelineSetup, loadOverview, persistInactivePipeline, setActiveTab])

  const testPipelineSource = useCallback(async (wizardDraft: PipelineWizardDraft) => {
    if (wizardDraft.sourceMode === "new") {
      const result = await testWatchlistSourceDraft({
        url: wizardDraft.sourceUrl,
        source_type: wizardDraft.sourceType
      }, { limit: 6 })
      return { status: "ready" as const, sourceTest: result }
    } else {
      const selectedSources = pipelineSources.filter((source) =>
        wizardDraft.sourceIds.includes(source.id)
      )
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
  }, [pipelineSources])

  const overviewBadges = getOverviewTabBadges(data?.health)

  if (!hasSelectedWatchlist) {
    return (
      <div
        className="rounded-md border border-dashed border-border bg-surface p-6"
        data-testid="watchlists-overview-no-watchlist"
      >
        <Empty
          image={Empty.PRESENTED_IMAGE_SIMPLE}
          description={(
            <span>
              {t("watchlists:overview.noWatchlist.title", "Create a Watchlist first")}
            </span>
          )}
        >
          <p className="mx-auto max-w-xl text-sm text-text-muted">
            {t(
              "watchlists:overview.noWatchlist.description",
              "Use the Watchlist setup action above to create the project container before adding sources, monitors, or reports."
            )}
          </p>
        </Empty>
      </div>
    )
  }

  if (loading && !data) {
    return (
      <div className="flex items-center justify-center py-16">
        <Spin size="large" />
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="text-sm text-text-muted">
          {t(
            "watchlists:overview.description",
            "At-a-glance watchlist health, activity, and failure signals."
          )}
        </p>
        <Space size="small">
          {data?.fetchedAt && (
            <Tooltip title={new Date(data.fetchedAt).toLocaleString()}>
              <span className="text-xs text-text-muted">
                {t("watchlists:overview.lastUpdated", "Updated {{time}}", {
                  time: formatRelativeTime(data.fetchedAt, t, { compact: true })
                })}
              </span>
            </Tooltip>
          )}
          <Button
            icon={<RefreshCw className="h-4 w-4" />}
            onClick={() => void loadOverview(false)}
            loading={refreshing}
          >
            {t("common:refresh", "Refresh")}
          </Button>
          <Button
            type="default"
            onClick={openPipelineSetup}
            data-testid="watchlists-overview-cta-pipeline-builder"
            disabled={(data?.sources.total || 0) === 0}
          >
            {t("watchlists:overview.pipelineSetup.open", "Briefing pipeline builder")}
          </Button>
        </Space>
      </div>

      {error && (
        <DesignSystemAlert variant="error" title={error} />
      )}

      <div
        className="sr-only"
        role="status"
        aria-live="polite"
        aria-atomic="true"
        data-testid="watchlists-overview-live-polite"
      >
        {politeAnnouncement}
      </div>
      <div
        className="sr-only"
        role="alert"
        aria-live="assertive"
        aria-atomic="true"
        data-testid="watchlists-overview-live-assertive"
      >
        {assertiveAnnouncement}
      </div>

      {data && (
        <>
          <LatestBriefing
            projection={data.latestBriefing}
            unreadCount={data.items.unread}
            onPlay={() => undefined}
            onOpenReport={handleOpenBriefingReport}
            onInspectRun={handleOpenRun}
            onRetryStage={handleRetryBriefingStage}
            onRegenerate={handleRegenerateBriefing}
            onTestNow={handleTestLatest}
            onViewReports={handleOpenAttentionOutputs}
          />
          {(data.sources.total === 0 || data.jobs.total === 0) && (
            <Card
              size="small"
              title={
                <span className="flex items-center gap-2">
                  {t("watchlists:overview.onboarding.title", "Add initial collection")}
                  <Tag color="green">{t("watchlists:overview.onboarding.recommended", "Recommended")}</Tag>
                </span>
              }
            >
              <p className="mb-3 text-sm text-text-muted">
                {t(
                  "watchlists:overview.onboarding.pipeline",
                  "Add feeds -> Configure monitor -> Check Activity -> Review Updates -> Generate Reports"
                )}
              </p>
              <div className="mb-3 flex flex-wrap items-center gap-2">
                <span className="text-xs font-medium uppercase tracking-wide text-text-muted">
                  {t("watchlists:overview.onboarding.path.label", "Onboarding mode")}
                </span>
                <Button
                  size="small"
                  type={onboardingPath === "beginner" ? "primary" : "default"}
                  onClick={() => handleOnboardingPathChange("beginner")}
                  data-testid="watchlists-overview-onboarding-path-beginner"
                  aria-pressed={onboardingPath === "beginner"}
                >
                  {t("watchlists:overview.onboarding.path.beginner", "Beginner (guided)")}
                </Button>
                <Button
                  size="small"
                  type={onboardingPath === "advanced" ? "primary" : "default"}
                  onClick={() => handleOnboardingPathChange("advanced")}
                  data-testid="watchlists-overview-onboarding-path-advanced"
                  aria-pressed={onboardingPath === "advanced"}
                >
                  {t("watchlists:overview.onboarding.path.advanced", "Advanced (direct forms)")}
                </Button>
              </div>
              <p className="mb-3 text-xs text-text-muted">
                {onboardingPath === "beginner"
                  ? t(
                      "watchlists:overview.onboarding.path.beginnerHint",
                      "Guided collection setup keeps this Watchlist scoped before advanced schedules and templates."
                    )
                  : t(
                      "watchlists:overview.onboarding.path.advancedHint",
                      "Advanced mode opens direct feed and monitor forms for this Watchlist."
                    )}
              </p>
              <Steps
                size="small"
                current={data.sources.total === 0 ? 0 : data.jobs.total === 0 ? 1 : 2}
                items={[
                  {
                    title: t("watchlists:overview.onboarding.steps.addFeed.title", "Add feeds to this Watchlist"),
                    content: t(
                      "watchlists:overview.onboarding.steps.addFeed.description",
                      "Add RSS/site feeds inside the selected Watchlist."
                    )
                  },
                  {
                    title: t("watchlists:overview.onboarding.steps.createMonitor.title", "Create a monitor"),
                    content: t(
                      "watchlists:overview.onboarding.steps.createMonitor.description",
                      "Pick Watchlist feeds, then set a schedule with presets."
                    )
                  },
                  {
                    title: t("watchlists:overview.onboarding.steps.reviewResults.title", "Review results"),
                    content: t(
                      "watchlists:overview.onboarding.steps.reviewResults.description",
                      "Open Updates for content and Activity for run diagnostics."
                    )
                  }
                ]}
              />
              <Space className="mt-4" wrap>
                <Button
                  type={onboardingPath === "beginner" ? "primary" : "default"}
                  ref={quickSetupTriggerRef}
                  onClick={(event) => openQuickSetup(event.currentTarget)}
                  data-testid="watchlists-overview-cta-guided-setup"
                >
                  {t("watchlists:overview.onboarding.cta.guidedSetup", "Add initial collection")}
                </Button>
                {onboardingPath === "advanced" && (
                  <Button
                    type="primary"
                    onClick={
                      data.sources.total === 0 ? handleStartSourceQuickCreate : handleStartJobQuickCreate
                    }
                    data-testid="watchlists-overview-cta-advanced-direct"
                  >
                    {data.sources.total === 0
                      ? t("watchlists:overview.onboarding.cta.addFeed", "Add first feed")
                      : t("watchlists:overview.onboarding.cta.createMonitor", "Create first monitor")}
                  </Button>
                )}
                {data.sources.total === 0 && (
                  <Button
                    type="default"
                    onClick={handleStartSourceQuickCreate}
                    data-testid="watchlists-overview-cta-add-feed"
                  >
                    {t("watchlists:overview.onboarding.cta.addFeed", "Add first feed")}
                  </Button>
                )}
                {data.sources.total > 0 && data.jobs.total === 0 && (
                  <Button
                    type="default"
                    onClick={handleStartJobQuickCreate}
                    data-testid="watchlists-overview-cta-create-monitor"
                  >
                    {t("watchlists:overview.onboarding.cta.createMonitor", "Create first monitor")}
                  </Button>
                )}
                <Button onClick={handleOpenItems}>
                  {t("watchlists:overview.onboarding.cta.reviewArticles", "Open Updates")}
                </Button>
              </Space>
            </Card>
          )}

          <DesignSystemAlert
            variant={data.systemHealth === "degraded" ? "warning" : "success"}
            title={
              data.systemHealth === "degraded"
                ? t("watchlists:overview.health.degradedTitle", "System requires attention")
                : t("watchlists:overview.health.healthyTitle", "System healthy")
            }
          >
            {data.systemHealth === "degraded"
              ? t(
                  "watchlists:overview.health.degradedDescription",
                  "Some sources, recent runs, or reports need review. Open the linked surface to investigate."
                )
              : t(
                  "watchlists:overview.health.healthyDescription",
                  "No recent failed runs and source health is stable."
                )}
          </DesignSystemAlert>

          <Card
            size="small"
            title={t("watchlists:overview.alertHealth.title", "Alerts and health")}
            data-testid="watchlists-overview-alert-health-summary"
          >
            <div className="grid gap-3 md:grid-cols-2">
              <div
                className="rounded-md border border-border bg-background p-3"
                data-testid="watchlists-overview-content-alerts"
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="inline-flex items-center gap-2 font-medium text-text">
                    <BellRing className="h-4 w-4" />
                    {t("watchlists:overview.alertHealth.contentTitle", "Content alerts")}
                  </div>
                  <Tag color={(data.alerts?.unread || 0) > 0 ? "orange" : "default"}>
                    {data.alerts?.unread || 0}
                  </Tag>
                </div>
                <div className="mt-2 text-sm font-medium text-text">
                  {(data.alerts?.unread || 0) > 0
                    ? t("watchlists:overview.alertHealth.unreadAlerts", "Unread content alerts")
                    : t("watchlists:overview.alertHealth.noUnreadAlerts", "No unread content alerts")}
                </div>
                <p className="mb-3 mt-1 text-sm text-text-muted">
                  {t(
                    "watchlists:overview.alertHealth.contentDescription",
                    "New updates matching your Watchlist alert rules."
                  )}
                </p>
                <Button size="small" onClick={handleOpenAlerts}>
                  {(data.alerts?.unread || 0) > 0
                    ? t("watchlists:overview.alertHealth.reviewAlerts", "Review alerts")
                    : t("watchlists:overview.alertHealth.createRule", "Create content alert rule")}
                </Button>
              </div>

              <div
                className="rounded-md border border-border bg-background p-3"
                data-testid="watchlists-overview-health-issues"
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="inline-flex items-center gap-2 font-medium text-text">
                    <AlertTriangle className="h-4 w-4" />
                    {t("watchlists:overview.alertHealth.healthTitle", "Health issues")}
                  </div>
                  <Tag color={data.health.attention.total > 0 ? "red" : "default"}>
                    {data.health.attention.total}
                  </Tag>
                </div>
                <div className="mt-2 text-sm font-medium text-text">
                  {data.health.attention.total > 0
                    ? t("watchlists:overview.alertHealth.healthIssues", "Health issues")
                    : t("watchlists:overview.alertHealth.noHealthIssues", "No health issues")}
                </div>
                <p className="mb-3 mt-1 text-sm text-text-muted">
                  {t(
                    "watchlists:overview.alertHealth.healthDescription",
                    "Run failures and source problems are health issues, not content alerts."
                  )}
                </p>
                <Button size="small" onClick={handleOpenRuns}>
                  {t("watchlists:overview.alertHealth.openActivity", "Open Activity")}
                </Button>
              </div>
            </div>
          </Card>

          {data.health.attention.total > 0 && (
            <Card
              size="small"
              title={t("watchlists:overview.attention.title", "Attention needed")}
            >
              <p className="mb-3 text-sm text-text-muted">
                {t(
                  "watchlists:overview.attention.description",
                  "Open the highest-risk surfaces directly from here."
                )}
              </p>
              <Space wrap>
                {overviewBadges.sources > 0 && (
                  <Button
                    danger
                    onClick={handleOpenAttentionSources}
                    data-testid="watchlists-overview-attention-sources"
                  >
                    {t("watchlists:overview.attention.sources", "Feeds need review ({{count}})", {
                      count: overviewBadges.sources
                    })}
                  </Button>
                )}
                {overviewBadges.runs > 0 && (
                  <Button
                    danger
                    onClick={handleOpenFailedRuns}
                    data-testid="watchlists-overview-attention-runs"
                  >
                    {t("watchlists:overview.attention.runs", "Activity needs review ({{count}})", {
                      count: overviewBadges.runs
                    })}
                  </Button>
                )}
                {overviewBadges.outputs > 0 && (
                  <Button
                    danger
                    onClick={handleOpenAttentionOutputs}
                    data-testid="watchlists-overview-attention-outputs"
                  >
                    {t("watchlists:overview.attention.outputs", "Reports need review ({{count}})", {
                      count: overviewBadges.outputs
                    })}
                  </Button>
                )}
                {data.jobs.attention > 0 && (
                  <Button
                    onClick={handleOpenJobs}
                    data-testid="watchlists-overview-attention-jobs"
                  >
                    {t("watchlists:overview.attention.jobs", "Monitors need schedule fixes ({{count}})", {
                      count: data.jobs.attention
                    })}
                  </Button>
                )}
              </Space>
            </Card>
          )}

          {data.sources.total > 0 && data.jobs.total > 0 && (
            <DesignSystemAlert
              variant="info"
              title={t("watchlists:overview.setupComplete.title", "Setup complete")}
              action={{
                label: t("watchlists:overview.setupComplete.openActivity", "Open Activity"),
                onClick: handleOpenRuns
              }}
            >
              {data.jobs.nextRunAt
                ? t(
                    "watchlists:overview.setupComplete.nextRunDescription",
                    "Your next monitor run is {{time}}. New content will appear in Updates and Activity.",
                    { time: formatRelativeTime(data.jobs.nextRunAt, t) }
                  )
                : t(
                    "watchlists:overview.setupComplete.runNowDescription",
                    "Run a monitor from Monitors to generate your first Updates and Activity entries."
                  )}
            </DesignSystemAlert>
          )}

          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            <Card
              size="small"
              title={(
                <span className="flex items-center gap-2">
                  <Rss className="h-4 w-4" />
                  {t("watchlists:overview.cards.sources.title", "Feeds")}
                </span>
              )}
              extra={(
                <Button type="link" size="small" onClick={handleOpenSources}>
                  {t("watchlists:overview.openSources", "Open")}
                </Button>
              )}
            >
              <Statistic
                value={data.sources.total}
                title={t("watchlists:overview.cards.sources.total", "Total")}
              />
              <div className="mt-3 flex flex-wrap gap-2">
                <Tag color="green">
                  {t("watchlists:overview.cards.sources.healthy", "Healthy {{count}}", {
                    count: data.sources.healthy
                  })}
                </Tag>
                <Tag color={data.sources.degraded > 0 ? "red" : "default"}>
                  {t("watchlists:overview.cards.sources.degraded", "Degraded {{count}}", {
                    count: data.sources.degraded
                  })}
                </Tag>
                <Tag>
                  {t("watchlists:overview.cards.sources.inactive", "Inactive {{count}}", {
                    count: data.sources.inactive
                  })}
                </Tag>
              </div>
            </Card>

            <Card
              size="small"
              title={(
                <span className="flex items-center gap-2">
                  <Workflow className="h-4 w-4" />
                  {t("watchlists:overview.cards.jobs.title", "Monitors")}
                </span>
              )}
              extra={(
                <Button type="link" size="small" onClick={handleOpenJobs}>
                  {t("watchlists:overview.openJobs", "Open")}
                </Button>
              )}
            >
              <Statistic
                value={data.jobs.active}
                title={t("watchlists:overview.cards.jobs.active", "Active")}
                suffix={`/ ${data.jobs.total}`}
              />
              <div className="mt-3 text-xs text-text-muted">
                {data.jobs.nextRunAt
                  ? t("watchlists:overview.cards.jobs.nextRun", "Next run {{time}}", {
                      time: formatRelativeTime(data.jobs.nextRunAt, t, { compact: true })
                    })
                  : t("watchlists:overview.cards.jobs.noNextRun", "No upcoming schedules")}
              </div>
            </Card>

            <Card
              size="small"
              title={(
                <span className="flex items-center gap-2">
                  <Newspaper className="h-4 w-4" />
                  {t("watchlists:overview.cards.items.title", "Updates")}
                </span>
              )}
              extra={(
                <Button type="link" size="small" onClick={handleOpenItems}>
                  {t("watchlists:overview.openItems", "Open")}
                </Button>
              )}
            >
              <Statistic
                value={data.items.unread}
                title={t("watchlists:overview.cards.items.unread", "Unread")}
              />
            </Card>

            <Card
              size="small"
              title={(
                <span className="flex items-center gap-2">
                  {data.runs.running + data.runs.pending > 0 ? (
                    <AlertTriangle className="h-4 w-4 text-warning" />
                  ) : (
                    <CheckCircle2 className="h-4 w-4 text-success" />
                  )}
                  {t("watchlists:overview.cards.runs.title", "Activity")}
                </span>
              )}
              extra={(
                <Button type="link" size="small" onClick={() => setActiveTab("runs")}>
                  {t("watchlists:overview.openRuns", "Open")}
                </Button>
              )}
            >
              <Statistic
                value={data.runs.running}
                title={t("watchlists:overview.cards.runs.running", "Running")}
              />
              <div className="mt-3 text-xs text-text-muted">
                {t("watchlists:overview.cards.runs.pending", "Pending {{count}}", {
                  count: data.runs.pending
                })}
              </div>
            </Card>
          </div>

          <Card
            size="small"
            title={t("watchlists:overview.failedRuns.title", "Recent Failed Runs")}
          >
            {data.runs.recentFailed.length === 0 ? (
              <Empty
                image={Empty.PRESENTED_IMAGE_SIMPLE}
                description={t("watchlists:overview.failedRuns.empty", "No recent failures")}
              />
            ) : (
              <List
                dataSource={data.runs.recentFailed}
                renderItem={(run) => (
                  <List.Item
                    actions={[
                      <Button
                        key={`open-${run.id}`}
                        size="small"
                        type="link"
                        onClick={() => handleOpenRun(run.id)}
                      >
                        {t("watchlists:overview.failedRuns.viewRun", "View run")}
                      </Button>
                    ]}
                  >
                    <List.Item.Meta
                      title={(
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="font-medium">
                            {run.job_name ||
                              t("watchlists:overview.failedRuns.jobFallback", "Monitor #{{id}}", {
                                id: run.job_id
                              })}
                          </span>
                          <Tag color="red">{t("watchlists:overview.failedRuns.failed", "Failed")}</Tag>
                          {run.finished_at && (
                            <span className="text-xs text-text-muted">
                              {formatRelativeTime(run.finished_at, t)}
                            </span>
                          )}
                        </div>
                      )}
                      description={
                        run.error_msg || t("watchlists:overview.failedRuns.noError", "No error details available")
                      }
                    />
                  </List.Item>
                )}
              />
            )}
          </Card>
        </>
      )}

      <PipelineWizard
        open={pipelineSetupOpen}
        sessionKey={pipelineSessionKey}
        initialDraft={pipelineInitialDraft}
        sources={pipelineSources}
        sourcesLoading={pipelineSourcesLoading}
        submitting={pipelineSetupSubmitting}
        previewLoading={pipelinePreviewLoading}
        previewError={pipelinePreviewError}
        previewRendered={pipelinePreviewRendered}
        previewRunId={pipelinePreviewRunId}
        previewWarnings={pipelinePreviewWarnings}
        submitError={pipelineSetupError}
        onCancel={closePipelineSetup}
        onSaveDraft={(draft) => setPipelineInitialDraft(draft)}
        onTest={testPipeline}
        onActivate={activatePipeline}
        onTestSource={testPipelineSource}
        onPreviewSchedule={previewPipelineSchedule}
        onPreview={(draft) => {
          void generatePipelineTemplatePreview(draft)
        }}
      />
    </div>
  )
}
