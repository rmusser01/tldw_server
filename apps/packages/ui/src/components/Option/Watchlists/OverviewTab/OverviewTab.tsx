import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react"
import {
  Alert,
  Button,
  Card,
  Checkbox,
  Empty,
  Form,
  Input,
  List,
  Modal,
  message,
  Select,
  Space,
  Steps,
  Spin,
  Statistic,
  Switch,
  Tag,
  Tooltip
} from "antd"
import type { StepsProps } from "antd"
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
  bulkCreateSources,
  createWatchlistOutput,
  createWatchlistJob,
  createWatchlistSource,
  deleteWatchlistJob,
  fetchWatchlistRuns,
  fetchWatchlistSources,
  getWatchlistTemplate,
  previewWatchlistTemplate,
  testWatchlistSourceDraft,
  triggerWatchlistRun
} from "@/services/watchlists"
import {
  fetchWatchlistsOverviewData,
  getOverviewTabBadges,
  type WatchlistsOverviewData
} from "@/services/watchlists-overview"
import { formatRelativeTime } from "@/utils/dateFormatters"
import {
  parseQuickSetupExtraSourceUrls,
  QUICK_SETUP_DEFAULT_VALUES,
  type QuickSetupValues,
  toQuickSetupJobPayload,
  toQuickSetupSourcePayload
} from "./quick-setup"
import type { JobPreviewResult } from "@/types/watchlists"
import {
  buildPipelineReviewSummary,
  toPipelineJobCreatePayload,
  toPipelineOutputCreatePayload,
  validateBriefingPipelineDraft,
  type BriefingPipelineDraft
} from "./pipeline-contract"
import {
  type WatchlistsOnboardingPath,
  readWatchlistsOnboardingPath,
  writeWatchlistsOnboardingPath
} from "../shared/onboarding-path"
import {
  getFocusableActiveElement,
  restoreFocusToElement
} from "../shared/focus-management"
import {
  trackWatchlistsOnboardingTelemetry,
  type WatchlistsQuickSetupStep
} from "@/utils/watchlists-onboarding-telemetry"
import type { WatchlistSource } from "@/types/watchlists"

const OVERVIEW_REFRESH_INTERVAL_MS = 30_000
const QUICK_SETUP_MAX_STEP = 2
const QUICK_SETUP_STEP_KEYS: WatchlistsQuickSetupStep[] = ["feed", "monitor", "review"]
const QUICK_SETUP_STEP_FIELDS: Array<Array<keyof QuickSetupValues>> = [
  ["sourceName", "sourceUrl", "sourceType", "extraSourceUrls"],
  ["monitorName", "schedulePreset", "setupGoal", "runNow", "includeAudioBriefing"],
  []
]
const PIPELINE_SETUP_MAX_STEP = 2

interface PipelineBuilderValues {
  sourceIds: number[]
  monitorName: string
  schedulePreset: "none" | "hourly" | "daily" | "weekdays"
  templateName: string
  includeAudio: boolean
  audioVoice: string
  targetAudioMinutes: number
  emailDeliveryEnabled: boolean
  emailRecipients: string[]
  chatbookDeliveryEnabled: boolean
  chatbookTitle: string
  runNow: boolean
}

const PIPELINE_DEFAULT_VALUES: PipelineBuilderValues = {
  sourceIds: [],
  monitorName: "",
  schedulePreset: "daily",
  templateName: "briefing_md",
  includeAudio: true,
  audioVoice: "alloy",
  targetAudioMinutes: 8,
  emailDeliveryEnabled: false,
  emailRecipients: [],
  chatbookDeliveryEnabled: false,
  chatbookTitle: "",
  runNow: true
}

const toPipelineDraft = (values: PipelineBuilderValues): BriefingPipelineDraft => ({
  monitorName: values.monitorName,
  sourceIds: values.sourceIds || [],
  schedulePreset: values.schedulePreset,
  templateName: values.templateName,
  includeAudio: values.includeAudio,
  audioVoice: values.includeAudio ? values.audioVoice : undefined,
  targetAudioMinutes: values.includeAudio ? Number(values.targetAudioMinutes) : undefined,
  emailRecipients: values.emailDeliveryEnabled
    ? (values.emailRecipients || []).map((entry) => String(entry || "").trim()).filter((entry) => entry.length > 0)
    : [],
  createChatbook: Boolean(values.chatbookDeliveryEnabled),
  chatbookTitle: values.chatbookDeliveryEnabled ? String(values.chatbookTitle || "").trim() : undefined
})

const PIPELINE_EMAIL_PATTERN = /^[^\s@]+@[^\s@]+\.[^\s@]+$/

const normalizePipelineRecipients = (value: unknown): string[] =>
  Array.isArray(value)
    ? value
      .map((entry) => String(entry || "").trim().toLowerCase())
      .filter((entry) => entry.length > 0)
    : []

export const OverviewTab: React.FC = () => {
  const { t } = useTranslation(["watchlists", "common"])
  const [data, setData] = useState<WatchlistsOverviewData | null>(null)
  const [loading, setLoading] = useState(false)
  const [refreshing, setRefreshing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [quickSetupOpen, setQuickSetupOpen] = useState(false)
  const [quickSetupStep, setQuickSetupStep] = useState(0)
  const [quickSetupSubmitting, setQuickSetupSubmitting] = useState(false)
  const [quickSetupCandidatePreview, setQuickSetupCandidatePreview] = useState<JobPreviewResult | null>(null)
  const [quickSetupCandidatePreviewLoading, setQuickSetupCandidatePreviewLoading] = useState(false)
  const [quickSetupCandidatePreviewError, setQuickSetupCandidatePreviewError] = useState<string | null>(null)
  const [pipelineSetupOpen, setPipelineSetupOpen] = useState(false)
  const [pipelineSetupStep, setPipelineSetupStep] = useState(0)
  const [pipelineSetupSubmitting, setPipelineSetupSubmitting] = useState(false)
  const [pipelineSourcesLoading, setPipelineSourcesLoading] = useState(false)
  const [pipelineSources, setPipelineSources] = useState<WatchlistSource[]>([])
  const [pipelinePreviewLoading, setPipelinePreviewLoading] = useState(false)
  const [pipelinePreviewError, setPipelinePreviewError] = useState<string | null>(null)
  const [pipelinePreviewRendered, setPipelinePreviewRendered] = useState<string | null>(null)
  const [pipelinePreviewRunId, setPipelinePreviewRunId] = useState<number | null>(null)
  const [pipelinePreviewWarnings, setPipelinePreviewWarnings] = useState<string[]>([])
  const [onboardingPath, setOnboardingPath] = useState<WatchlistsOnboardingPath>(() =>
    readWatchlistsOnboardingPath()
  )
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const quickSetupRestoreFocusTargetRef = useRef<HTMLElement | null>(null)
  const pipelineSetupRestoreFocusTargetRef = useRef<HTMLElement | null>(null)
  const quickSetupWasOpenRef = useRef(false)
  const pipelineSetupWasOpenRef = useRef(false)
  const quickSetupPreviewRequestRef = useRef(0)
  const [quickSetupForm] = Form.useForm<QuickSetupValues>()
  const [pipelineSetupForm] = Form.useForm<PipelineBuilderValues>()

  const setActiveTab = useWatchlistsStore((s) => s.setActiveTab)
  const setOutputsRunFilter = useWatchlistsStore((s) => s.setOutputsRunFilter)
  const setRunsStatusFilter = useWatchlistsStore((s) => s.setRunsStatusFilter)
  const setOverviewHealth = useWatchlistsStore((s) => s.setOverviewHealth)
  const openRunDetail = useWatchlistsStore((s) => s.openRunDetail)
  const openOutputPreview = useWatchlistsStore((s) => s.openOutputPreview)
  const openSourceForm = useWatchlistsStore((s) => s.openSourceForm)
  const openJobForm = useWatchlistsStore((s) => s.openJobForm)
  const selectedWatchlistId = useWatchlistsStore((s) => s.selectedWatchlistId)
  const hasSelectedWatchlist = selectedWatchlistId != null

  const loadOverview = useCallback(async (showLoading: boolean) => {
    if (selectedWatchlistId == null) {
      setData(null)
      setError(null)
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
      console.error("Failed to load watchlists overview:", err)
      setError(t("watchlists:overview.fetchError", "Failed to load overview"))
    } finally {
      setLoading(false)
      setRefreshing(false)
    }
  }, [selectedWatchlistId, setOverviewHealth, t])

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

  const quickSetupAutoShownRef = useRef(false)

  useLayoutEffect(() => {
    if (quickSetupOpen) {
      if (!quickSetupWasOpenRef.current) {
        quickSetupRestoreFocusTargetRef.current = getFocusableActiveElement()
      }
      quickSetupWasOpenRef.current = true
      return
    }

    if (quickSetupWasOpenRef.current) {
      quickSetupWasOpenRef.current = false
      restoreFocusToElement(quickSetupRestoreFocusTargetRef.current)
    }
  }, [quickSetupOpen])

  useLayoutEffect(() => {
    if (pipelineSetupOpen) {
      if (!pipelineSetupWasOpenRef.current) {
        pipelineSetupRestoreFocusTargetRef.current = getFocusableActiveElement()
      }
      pipelineSetupWasOpenRef.current = true
      return
    }

    if (pipelineSetupWasOpenRef.current) {
      pipelineSetupWasOpenRef.current = false
      restoreFocusToElement(pipelineSetupRestoreFocusTargetRef.current)
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

  const handleOpenAlerts = useCallback(() => {
    setActiveTab("alerts")
  }, [setActiveTab])

  const handleOpenFailedRuns = useCallback(() => {
    if (typeof setRunsStatusFilter === "function") {
      setRunsStatusFilter("failed")
    }
    setActiveTab("runs")
  }, [setActiveTab, setRunsStatusFilter])

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

  const openQuickSetup = useCallback(() => {
    quickSetupForm.setFieldsValue(QUICK_SETUP_DEFAULT_VALUES)
    setQuickSetupStep(0)
    setQuickSetupOpen(true)
    void trackWatchlistsOnboardingTelemetry({ type: "quick_setup_opened" })
  }, [quickSetupForm])

  // Auto-open Quick Setup for first-time users with no sources
  useEffect(() => {
    const autoShownKey = "watchlists:quickSetup:autoshown:v1"

    // If the wizard is already open, mark as shown to prevent re-opening after close
    if (quickSetupOpen) {
      quickSetupAutoShownRef.current = true
      try {
        localStorage.setItem(autoShownKey, "1")
      } catch {
        // ignore
      }
      return
    }

    if (
      !data ||
      data.sources.total !== 0 ||
      selectedWatchlistId == null ||
      quickSetupAutoShownRef.current ||
      onboardingPath !== "beginner"
    ) {
      return
    }

    try {
      if (localStorage.getItem(autoShownKey)) {
        quickSetupAutoShownRef.current = true
        return
      }
      localStorage.setItem(autoShownKey, "1")
      quickSetupAutoShownRef.current = true
      openQuickSetup()
    } catch {
      quickSetupAutoShownRef.current = true
      openQuickSetup()
    }
  }, [data, quickSetupOpen, onboardingPath, openQuickSetup, selectedWatchlistId])

  const closeQuickSetup = useCallback(() => {
    quickSetupPreviewRequestRef.current += 1
    setQuickSetupOpen(false)
    setQuickSetupStep(0)
    setQuickSetupCandidatePreview(null)
    setQuickSetupCandidatePreviewLoading(false)
    setQuickSetupCandidatePreviewError(null)
    quickSetupForm.resetFields()
  }, [quickSetupForm])

  const loadQuickSetupCandidatePreview = useCallback(async (draftValues?: Partial<QuickSetupValues>) => {
    const mergedValues = {
      ...QUICK_SETUP_DEFAULT_VALUES,
      ...quickSetupForm.getFieldsValue(true),
      ...(draftValues || {})
    } as QuickSetupValues
    const sourceUrl = String(mergedValues.sourceUrl || "").trim()
    if (!sourceUrl) {
      setQuickSetupCandidatePreview(null)
      setQuickSetupCandidatePreviewError(null)
      setQuickSetupCandidatePreviewLoading(false)
      return
    }

    const requestId = quickSetupPreviewRequestRef.current + 1
    quickSetupPreviewRequestRef.current = requestId
    setQuickSetupCandidatePreviewLoading(true)
    setQuickSetupCandidatePreviewError(null)

    try {
      const preview = await testWatchlistSourceDraft(
        {
          url: sourceUrl,
          source_type: mergedValues.sourceType || "rss"
        },
        { limit: 6 }
      )
      if (quickSetupPreviewRequestRef.current !== requestId) return
      setQuickSetupCandidatePreview(preview)
      setQuickSetupCandidatePreviewError(null)
      void trackWatchlistsOnboardingTelemetry({
        type: "quick_setup_preview_loaded",
        preview: "candidate",
        total: preview.total || 0,
        ingestable: preview.ingestable || 0,
        filtered: preview.filtered || 0
      })
    } catch (err) {
      console.error("Failed to load quick setup source preview:", err)
      if (quickSetupPreviewRequestRef.current !== requestId) return
      setQuickSetupCandidatePreview(null)
      setQuickSetupCandidatePreviewError(
        t(
          "watchlists:overview.onboarding.quickSetup.candidatePreview.error",
          "Could not load feed sample preview right now. You can still add the collection and use a test run to verify results."
        )
      )
      void trackWatchlistsOnboardingTelemetry({
        type: "quick_setup_preview_failed",
        preview: "candidate",
        reason: "load_failed"
      })
    } finally {
      if (quickSetupPreviewRequestRef.current === requestId) {
        setQuickSetupCandidatePreviewLoading(false)
      }
    }
  }, [quickSetupForm, t])

  const getQuickSetupStepKey = useCallback(
    (step: number): WatchlistsQuickSetupStep => {
      const clamped = Math.max(0, Math.min(step, QUICK_SETUP_MAX_STEP))
      return QUICK_SETUP_STEP_KEYS[clamped]
    },
    []
  )

  const cancelQuickSetup = useCallback(() => {
    void trackWatchlistsOnboardingTelemetry({
      type: "quick_setup_cancelled",
      step: getQuickSetupStepKey(quickSetupStep)
    })
    closeQuickSetup()
  }, [closeQuickSetup, getQuickSetupStepKey, quickSetupStep])

  const handleQuickSetupNext = useCallback(async () => {
    const fields = QUICK_SETUP_STEP_FIELDS[quickSetupStep] || []
    if (fields.length > 0) {
      await quickSetupForm.validateFields(fields)
    }
    void trackWatchlistsOnboardingTelemetry({
      type: "quick_setup_step_completed",
      step: getQuickSetupStepKey(quickSetupStep)
    })
    if (quickSetupStep === 0) {
      const sourceName = String(quickSetupForm.getFieldValue("sourceName") || "").trim()
      const monitorName = String(quickSetupForm.getFieldValue("monitorName") || "").trim()
      if (sourceName.length > 0 && monitorName.length === 0) {
        quickSetupForm.setFieldsValue({
          monitorName: `${sourceName} Monitor`
        })
      }
    }
    if (quickSetupStep === 1) {
      const values = {
        ...QUICK_SETUP_DEFAULT_VALUES,
        ...quickSetupForm.getFieldsValue(true)
      } as QuickSetupValues
      void trackWatchlistsOnboardingTelemetry({
        type: "quick_setup_preview_loaded",
        preview: "template",
        goal: values.setupGoal,
        audioEnabled: values.includeAudioBriefing
      })
    }
    setQuickSetupStep((prev) => Math.min(prev + 1, QUICK_SETUP_MAX_STEP))
  }, [getQuickSetupStepKey, quickSetupForm, quickSetupStep, t])

  const handleQuickSetupBack = useCallback(() => {
    setQuickSetupStep((prev) => Math.max(prev - 1, 0))
  }, [])

  const completeQuickSetup = useCallback(async () => {
    const currentStep = getQuickSetupStepKey(quickSetupStep)
    void trackWatchlistsOnboardingTelemetry({
      type: "quick_setup_step_completed",
      step: currentStep
    })
    try {
      const values = {
        ...QUICK_SETUP_DEFAULT_VALUES,
        ...quickSetupForm.getFieldsValue(true)
      } as QuickSetupValues
      setQuickSetupSubmitting(true)

      const source = await createWatchlistSource(
        toQuickSetupSourcePayload(values, selectedWatchlistId)
      )
      const sourceIds: number[] = [source.id]
      const rawExtraSourceUrls = String(values.extraSourceUrls || "").trim()
      const extraSourceUrls = parseQuickSetupExtraSourceUrls(rawExtraSourceUrls)

      if (rawExtraSourceUrls.length > 0 && extraSourceUrls.length === 0) {
        throw new Error("quick_setup_invalid_extra_source_urls")
      }

      if (extraSourceUrls.length > 0) {
        const bulkResult = await bulkCreateSources(
          extraSourceUrls.map((url, index) => {
            const host = (() => {
              try {
                return new URL(url).hostname || `Feed ${index + 2}`
              } catch {
                return `Feed ${index + 2}`
              }
            })()
            return {
              name: host,
              url,
              source_type: values.sourceType,
              active: true,
              watchlist_id: selectedWatchlistId ?? undefined
            }
          })
        )

        const createdExtraSourceIds = (bulkResult.items || [])
          .filter((entry) => entry.status === "created" && Number.isFinite(Number(entry.id)))
          .map((entry) => Number(entry.id))

        if (createdExtraSourceIds.length !== extraSourceUrls.length) {
          throw new Error("quick_setup_bulk_source_creation_failed")
        }

        sourceIds.push(...createdExtraSourceIds)
      }

      const job = await createWatchlistJob(
        toQuickSetupJobPayload(values, sourceIds, selectedWatchlistId)
      )

      let runId: number | null = null
      if (values.runNow) {
        try {
          const run = await triggerWatchlistRun(job.id)
          runId = run.id
          void trackWatchlistsOnboardingTelemetry({
            type: "quick_setup_test_run_triggered",
            runId: run.id
          })
        } catch (runError) {
          void trackWatchlistsOnboardingTelemetry({
            type: "quick_setup_test_run_failed"
          })
          throw runError
        }
      }

      message.success(
        values.runNow
          ? t(
              "watchlists:overview.onboarding.quickSetup.testRunPending",
              "Initial collection added. Test run started. Open Activity for progress and Reports for generated briefings."
            )
          : values.setupGoal === "briefing"
            ? t(
                "watchlists:overview.onboarding.quickSetup.createdBriefing",
                "Initial collection added. Feeds and monitor are ready for briefing reports."
              )
          : t(
              "watchlists:overview.onboarding.quickSetup.created",
              "Initial collection added. Feeds and monitor are ready."
            )
      )

      closeQuickSetup()
      void loadOverview(false)

      if (runId != null) {
        setActiveTab("runs")
        openRunDetail(runId)
        void trackWatchlistsOnboardingTelemetry({
          type: "quick_setup_completed",
          goal: values.setupGoal,
          runNow: true,
          destination: "runs"
        })
      } else if (values.setupGoal === "briefing") {
        setActiveTab("outputs")
        void trackWatchlistsOnboardingTelemetry({
          type: "quick_setup_completed",
          goal: values.setupGoal,
          runNow: false,
          destination: "outputs"
        })
      } else {
        setActiveTab("jobs")
        void trackWatchlistsOnboardingTelemetry({
          type: "quick_setup_completed",
          goal: values.setupGoal,
          runNow: false,
          destination: "jobs"
        })
      }
    } catch (err) {
      console.error("Failed to add initial Watchlist collection:", err)
      void trackWatchlistsOnboardingTelemetry({
        type: "quick_setup_failed",
        step: currentStep
      })
      message.error(
        t(
          "watchlists:overview.onboarding.quickSetup.error",
          "Failed to add initial collection."
        )
      )
    } finally {
      setQuickSetupSubmitting(false)
    }
  }, [
    closeQuickSetup,
    getQuickSetupStepKey,
    loadOverview,
    openRunDetail,
    quickSetupForm,
    quickSetupStep,
    selectedWatchlistId,
    setActiveTab,
    t
  ])

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
      const selectedIds = pipelineSetupForm.getFieldValue("sourceIds") as number[] | undefined
      if ((!selectedIds || selectedIds.length === 0) && items.length === 1) {
        pipelineSetupForm.setFieldsValue({ sourceIds: [items[0].id] })
      }
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
  }, [pipelineSetupForm, selectedWatchlistId, t])

  const openPipelineSetup = useCallback(() => {
    pipelineSetupForm.setFieldsValue(PIPELINE_DEFAULT_VALUES)
    setPipelineSetupStep(0)
    setPipelinePreviewError(null)
    setPipelinePreviewRendered(null)
    setPipelinePreviewRunId(null)
    setPipelinePreviewWarnings([])
    setPipelineSetupOpen(true)
    void trackWatchlistsOnboardingTelemetry({ type: "pipeline_setup_opened" })
    void loadPipelineSources()
  }, [loadPipelineSources, pipelineSetupForm])

  const closePipelineSetup = useCallback(() => {
    if (pipelineSetupSubmitting) return
    setPipelineSetupOpen(false)
    setPipelineSetupStep(0)
    setPipelinePreviewLoading(false)
    setPipelinePreviewError(null)
    setPipelinePreviewRendered(null)
    setPipelinePreviewRunId(null)
    setPipelinePreviewWarnings([])
    pipelineSetupForm.resetFields()
  }, [pipelineSetupForm, pipelineSetupSubmitting])

  const handlePipelineSetupBack = useCallback(() => {
    setPipelineSetupStep((prev) => Math.max(0, prev - 1))
  }, [])

  const handlePipelineSetupNext = useCallback(async () => {
    try {
      let completedStep: "scope" | "briefing" | null = null
      if (pipelineSetupStep === 0) {
        await pipelineSetupForm.validateFields(["sourceIds"])
        completedStep = "scope"
      } else if (pipelineSetupStep === 1) {
        const includeAudio = Boolean(pipelineSetupForm.getFieldValue("includeAudio"))
        const emailDeliveryEnabled = Boolean(pipelineSetupForm.getFieldValue("emailDeliveryEnabled"))
        const fields: Array<keyof PipelineBuilderValues> = [
          "monitorName",
          "schedulePreset",
          "templateName"
        ]
        if (includeAudio) {
          fields.push("audioVoice", "targetAudioMinutes")
        }
        if (emailDeliveryEnabled) {
          fields.push("emailRecipients")
        }
        await pipelineSetupForm.validateFields(fields)
        completedStep = "briefing"
      }
      if (completedStep) {
        void trackWatchlistsOnboardingTelemetry({
          type: "pipeline_setup_step_completed",
          step: completedStep
        })
      }
      setPipelineSetupStep((prev) => Math.min(prev + 1, PIPELINE_SETUP_MAX_STEP))
    } catch {
      // Field-level validation state is already surfaced by antd form.
    }
  }, [pipelineSetupForm, pipelineSetupStep])

  const generatePipelineTemplatePreview = useCallback(async () => {
    setPipelinePreviewLoading(true)
    setPipelinePreviewError(null)
    setPipelinePreviewRendered(null)
    setPipelinePreviewRunId(null)
    setPipelinePreviewWarnings([])

    try {
      const values = {
        ...PIPELINE_DEFAULT_VALUES,
        ...pipelineSetupForm.getFieldsValue(true)
      } as PipelineBuilderValues
      const draft = toPipelineDraft(values)
      const templateName = String(draft.templateName || "").trim()
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
        (run) => String(run.status || "").trim().toLowerCase() === "completed"
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
  }, [pipelineSetupForm, selectedWatchlistId, t])

  const completePipelineSetup = useCallback(async (
    options?: { mode?: "create" | "test"; forceRunNow?: boolean }
  ) => {
    let createdJobId: number | null = null
    let createdRunId: number | null = null
    const mode = options?.mode || "create"
    let shouldRunNowForTelemetry = false
    let pipelineFailureStage: "validation" | "job_create" | "run_trigger" | "output_create" | "rollback" =
      "job_create"

    try {
      setPipelineSetupSubmitting(true)
      const values = {
        ...PIPELINE_DEFAULT_VALUES,
        ...pipelineSetupForm.getFieldsValue(true)
      } as PipelineBuilderValues
      const shouldRunNow = Boolean(values.runNow || options?.forceRunNow)
      shouldRunNowForTelemetry = shouldRunNow

      const draft = toPipelineDraft(values)
      const validation = validateBriefingPipelineDraft(draft)
      if (!validation.valid) {
        const errorFields = validation.errors.map((key) => ({
          name: key as keyof PipelineBuilderValues,
          errors: [
            t(
              "watchlists:overview.pipelineSetup.validation.required",
              "Complete this field before continuing."
            )
          ]
        }))
        pipelineSetupForm.setFields(errorFields)
        if (validation.errors.includes("sourceIds")) {
          setPipelineSetupStep(0)
        } else {
          setPipelineSetupStep(1)
        }
        message.error(
          t(
            "watchlists:overview.pipelineSetup.validationError",
            "Review the highlighted pipeline fields."
          )
        )
        void trackWatchlistsOnboardingTelemetry({
          type: "pipeline_setup_failed",
          stage: "validation",
          mode,
          runNow: shouldRunNow
        })
        return
      }

      void trackWatchlistsOnboardingTelemetry({
        type: "pipeline_setup_step_completed",
        step: "review"
      })
      void trackWatchlistsOnboardingTelemetry({
        type: "pipeline_setup_submitted",
        mode,
        runNow: shouldRunNow
      })

      pipelineFailureStage = "job_create"
      const job = await createWatchlistJob({
        ...toPipelineJobCreatePayload(draft),
        watchlist_id: selectedWatchlistId ?? undefined
      })
      createdJobId = job.id

      if (!shouldRunNow) {
        closePipelineSetup()
        void loadOverview(false)
        setActiveTab("jobs")
        void trackWatchlistsOnboardingTelemetry({
          type: "pipeline_setup_completed",
          mode,
          runNow: false,
          destination: "jobs"
        })
        message.success(
          t(
            "watchlists:overview.pipelineSetup.created",
            "Pipeline created. Open Monitors to review schedule and trigger runs."
          )
        )
        return
      }

      pipelineFailureStage = "run_trigger"
      const run = await triggerWatchlistRun(job.id)
      createdRunId = run.id
      pipelineFailureStage = "output_create"
      const output = await createWatchlistOutput(
        toPipelineOutputCreatePayload(run.id, draft)
      )

      closePipelineSetup()
      void loadOverview(false)
      void trackWatchlistsOnboardingTelemetry({
        type: "pipeline_setup_completed",
        mode,
        runNow: true,
        destination: "outputs"
      })
      if (typeof setOutputsRunFilter === "function") {
        setOutputsRunFilter(run.id)
      }
      setActiveTab("outputs")
      openOutputPreview(output.id)
      message.success(
        mode === "test"
          ? t(
              "watchlists:overview.pipelineSetup.testGenerationReady",
              "Test generation complete. Monitor, run, and sample report are ready for review."
            )
          : t(
              "watchlists:overview.pipelineSetup.createdAndRunning",
              "Pipeline created. First run started and report is ready for review."
            )
      )
    } catch (err) {
      console.error("Failed to complete pipeline setup:", err)
      void trackWatchlistsOnboardingTelemetry({
        type: "pipeline_setup_failed",
        stage: pipelineFailureStage,
        mode,
        runNow: shouldRunNowForTelemetry
      })
      if (createdJobId != null && createdRunId == null) {
        try {
          await deleteWatchlistJob(createdJobId)
          message.warning(
            t(
              "watchlists:overview.pipelineSetup.rollbackSuccess",
              "Pipeline setup failed before run start. Monitor creation was rolled back."
            )
          )
        } catch (rollbackError) {
          console.error("Failed to rollback pipeline monitor creation:", rollbackError)
          void trackWatchlistsOnboardingTelemetry({
            type: "pipeline_setup_failed",
            stage: "rollback",
            mode,
            runNow: shouldRunNowForTelemetry
          })
          message.error(
            t(
              "watchlists:overview.pipelineSetup.rollbackFailed",
              "Pipeline setup failed and rollback was incomplete. Review Monitors for cleanup."
            )
          )
        }
      } else {
        if (createdRunId != null) {
          setActiveTab("runs")
          openRunDetail(createdRunId)
        }
        message.error(
          t(
            "watchlists:overview.pipelineSetup.error",
            "Pipeline setup failed. Open Activity or Reports to inspect recovery options."
          )
        )
      }
    } finally {
      setPipelineSetupSubmitting(false)
    }
  }, [
    closePipelineSetup,
    loadOverview,
    openOutputPreview,
    openRunDetail,
    pipelineSetupForm,
    selectedWatchlistId,
    setActiveTab,
    setOutputsRunFilter,
    t
  ])

  const quickSetupValues = Form.useWatch([], quickSetupForm) as Partial<QuickSetupValues> | undefined
  const quickSetupSnapshot = useMemo(
    () =>
      ({
        ...QUICK_SETUP_DEFAULT_VALUES,
        ...quickSetupForm.getFieldsValue(true),
        ...(quickSetupValues || {})
      }) as QuickSetupValues,
    [quickSetupForm, quickSetupValues]
  )

  useEffect(() => {
    if (!quickSetupOpen || quickSetupStep !== QUICK_SETUP_MAX_STEP) return
    void loadQuickSetupCandidatePreview(quickSetupSnapshot)
  }, [
    loadQuickSetupCandidatePreview,
    quickSetupOpen,
    quickSetupStep,
    quickSetupSnapshot.sourceType,
    quickSetupSnapshot.sourceUrl
  ])

  const quickSetupCandidateSummary = useMemo(() => {
    if (quickSetupCandidatePreviewLoading) {
      return t(
        "watchlists:overview.onboarding.quickSetup.candidatePreview.loading",
        "Loading sample candidates..."
      )
    }
    if (quickSetupCandidatePreviewError) return quickSetupCandidatePreviewError
    if (!quickSetupCandidatePreview) {
      return t(
        "watchlists:overview.onboarding.quickSetup.candidatePreview.empty",
        "No sample candidates returned yet. You can still add the collection and validate with a test run."
      )
    }
    return t(
      "watchlists:overview.onboarding.quickSetup.candidatePreview.summary",
      "{{ingestable}} ingestable, {{filtered}} filtered from {{total}} sample items.",
      {
        ingestable: quickSetupCandidatePreview.ingestable,
        filtered: quickSetupCandidatePreview.filtered,
        total: quickSetupCandidatePreview.total
      }
    )
  }, [
    quickSetupCandidatePreview,
    quickSetupCandidatePreviewError,
    quickSetupCandidatePreviewLoading,
    t
  ])

  const quickSetupTemplatePreview = useMemo(() => {
    if (quickSetupSnapshot.setupGoal === "triage") {
      return t(
        "watchlists:overview.onboarding.quickSetup.templatePreview.none",
        "No briefing output will be generated for feed-review-only setup."
      )
    }

    const monitorName = String(quickSetupSnapshot.monitorName || "").trim() || "Briefing"
    const sourceName = String(quickSetupSnapshot.sourceName || "").trim() || "Selected feed"
    const topItems = (quickSetupCandidatePreview?.items || [])
      .slice(0, 3)
      .map((item) => item.title || item.url || `Source #${item.source_id}`)

    return [
      `# ${monitorName}`,
      "",
      `Source: ${sourceName}`,
      "",
      "Top candidate headlines:",
      ...(topItems.length > 0
        ? topItems.map((item, index) => `${index + 1}. ${item}`)
        : ["1. Headlines will appear after the first run completes."]),
      "",
      `Audio briefing: ${quickSetupSnapshot.includeAudioBriefing ? "Enabled" : "Disabled"}`
    ].join("\n")
  }, [
    quickSetupCandidatePreview,
    quickSetupSnapshot.includeAudioBriefing,
    quickSetupSnapshot.monitorName,
    quickSetupSnapshot.setupGoal,
    quickSetupSnapshot.sourceName,
    t
  ])

  const quickSetupIsLastStep = quickSetupStep >= QUICK_SETUP_MAX_STEP
  const quickSetupFinishLabel = quickSetupSnapshot.runNow
    ? t(
        "watchlists:overview.onboarding.quickSetup.actions.finishWithTest",
        "Create collection + run test"
      )
    : t("watchlists:overview.onboarding.quickSetup.actions.finish", "Create collection")
  const quickSetupStepHelp =
    quickSetupStep === 0
      ? t(
          "watchlists:overview.onboarding.quickSetup.help.feed",
          "Add one or more feed URLs to this Watchlist. You can adjust feed settings later."
        )
      : quickSetupStep === 1
        ? t(
            "watchlists:overview.onboarding.quickSetup.help.monitor",
            "Choose how this Watchlist monitor runs; presets avoid cron setup first."
          )
        : t(
            "watchlists:overview.onboarding.quickSetup.help.review",
            "Review the scoped feeds and monitor before adding them to this Watchlist."
          )
  const quickSetupExtraSourceUrls = parseQuickSetupExtraSourceUrls(
    String(quickSetupValues?.extraSourceUrls || "")
  )
  const quickSetupDestinationHint = quickSetupValues?.runNow
    ? t(
        "watchlists:overview.onboarding.quickSetup.destination.runs",
        "After setup, you will land in Activity to monitor the active run."
      )
    : quickSetupValues?.setupGoal === "briefing"
      ? t(
          "watchlists:overview.onboarding.quickSetup.destination.outputs",
          "After setup, you will land in Reports to review generated briefings."
        )
      : t(
          "watchlists:overview.onboarding.quickSetup.destination.jobs",
          "After setup, you will land in Monitors to schedule and tune your workflow."
        )
  const pipelineSetupValues = Form.useWatch([], pipelineSetupForm) as
    | Partial<PipelineBuilderValues>
    | undefined
  const pipelineDraftPreview = toPipelineDraft({
    ...PIPELINE_DEFAULT_VALUES,
    ...(pipelineSetupValues || {})
  } as PipelineBuilderValues)
  const pipelineReviewSummary = buildPipelineReviewSummary(pipelineDraftPreview)
  const pipelineEmailRecipients = normalizePipelineRecipients(pipelineSetupValues?.emailRecipients)
  const pipelineHasValidEmailRecipients = !pipelineSetupValues?.emailDeliveryEnabled || (
    pipelineEmailRecipients.length > 0 &&
    pipelineEmailRecipients.every((entry) => PIPELINE_EMAIL_PATTERN.test(entry))
  )
  const pipelineScopeComplete = Array.isArray(pipelineDraftPreview.sourceIds) && pipelineDraftPreview.sourceIds.length > 0
  const pipelineBriefingComplete = Boolean(
    pipelineDraftPreview.monitorName.trim().length > 0 &&
      pipelineDraftPreview.templateName.trim().length > 0 &&
      (!pipelineDraftPreview.includeAudio ||
        (Boolean(pipelineDraftPreview.audioVoice) &&
          Number(pipelineDraftPreview.targetAudioMinutes || 0) > 0)) &&
      pipelineHasValidEmailRecipients
  )
  const pipelineReviewComplete = pipelineScopeComplete && pipelineBriefingComplete
  const pipelineStepItems = useMemo<NonNullable<StepsProps["items"]>>(() => ([
    {
      title: t("watchlists:overview.pipelineSetup.steps.scope", "Scope"),
      status: pipelineScopeComplete ? "finish" : pipelineSetupStep === 0 ? "process" : "wait"
    },
    {
      title: t("watchlists:overview.pipelineSetup.steps.briefing", "Briefing"),
      status: pipelineBriefingComplete ? "finish" : pipelineSetupStep === 1 ? "process" : "wait"
    },
    {
      title: t("watchlists:overview.pipelineSetup.steps.review", "Review"),
      status: pipelineReviewComplete ? "finish" : pipelineSetupStep === 2 ? "process" : "wait"
    }
  ]), [pipelineBriefingComplete, pipelineReviewComplete, pipelineScopeComplete, pipelineSetupStep, t])
  const pipelineSetupIsLastStep = pipelineSetupStep >= PIPELINE_SETUP_MAX_STEP
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
        <Alert type="error" showIcon title={error} />
      )}

      {data && (
        <>
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
                  onClick={openQuickSetup}
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

          <Alert
            showIcon
            type={data.systemHealth === "degraded" ? "warning" : "success"}
            title={
              data.systemHealth === "degraded"
                ? t("watchlists:overview.health.degradedTitle", "System requires attention")
                : t("watchlists:overview.health.healthyTitle", "System healthy")
            }
            description={
              data.systemHealth === "degraded"
                ? t(
                    "watchlists:overview.health.degradedDescription",
                    "Some sources or recent runs show failures. Open failed runs to investigate."
                  )
                : t(
                    "watchlists:overview.health.healthyDescription",
                    "No recent failed runs and source health is stable."
                  )
            }
          />

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
                    {t("watchlists:overview.attention.runs", "Failed activity runs ({{count}})", {
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
                    {t("watchlists:overview.attention.outputs", "Reports with delivery issues ({{count}})", {
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
            <Alert
              showIcon
              type="info"
              title={t("watchlists:overview.setupComplete.title", "Setup complete")}
              description={
                data.jobs.nextRunAt
                  ? t(
                      "watchlists:overview.setupComplete.nextRunDescription",
                      "Your next monitor run is {{time}}. New content will appear in Updates and Activity.",
                      { time: formatRelativeTime(data.jobs.nextRunAt, t) }
                    )
                  : t(
                      "watchlists:overview.setupComplete.runNowDescription",
                      "Run a monitor from Monitors to generate your first Updates and Activity entries."
                    )
              }
              action={
                <Button size="small" onClick={handleOpenRuns}>
                  {t("watchlists:overview.setupComplete.openActivity", "Open Activity")}
                </Button>
              }
            />
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

      <Modal
        open={quickSetupOpen}
        title={t("watchlists:overview.onboarding.quickSetup.title", "Add initial collection")}
        onCancel={quickSetupSubmitting ? undefined : cancelQuickSetup}
        destroyOnHidden
        maskClosable={!quickSetupSubmitting}
        footer={[
          <Button
            key="cancel"
            onClick={cancelQuickSetup}
            disabled={quickSetupSubmitting}
          >
            {t("common:cancel", "Cancel")}
          </Button>,
          <Button
            key="back"
            onClick={handleQuickSetupBack}
            disabled={quickSetupSubmitting || quickSetupStep === 0}
          >
            {t("common:back", "Back")}
          </Button>,
          <Button
            key="next"
            type="primary"
            loading={quickSetupSubmitting}
            onClick={() => {
              if (quickSetupIsLastStep) {
                void completeQuickSetup()
              } else {
                void handleQuickSetupNext()
              }
            }}
          >
            {quickSetupIsLastStep
              ? quickSetupFinishLabel
              : t("common:next", "Next")}
          </Button>
        ]}
      >
        <div className="space-y-4">
          <Steps
            size="small"
            current={quickSetupStep}
            items={[
              { title: t("watchlists:overview.onboarding.quickSetup.steps.feed", "Feed") },
              { title: t("watchlists:overview.onboarding.quickSetup.steps.monitor", "Monitor") },
              { title: t("watchlists:overview.onboarding.quickSetup.steps.review", "Review") }
            ]}
          />
          <div className="rounded-md border border-border bg-surface px-3 py-2 text-xs text-text-muted">
            {quickSetupStepHelp}
          </div>

          <Form
            form={quickSetupForm}
            layout="vertical"
            initialValues={QUICK_SETUP_DEFAULT_VALUES}
          >
            {quickSetupStep === 0 && (
              <div className="space-y-1">
                <Form.Item
                  label={t("watchlists:overview.onboarding.quickSetup.fields.sourceName", "Feed name")}
                  name="sourceName"
                  rules={[
                    {
                      required: true,
                      message: t(
                        "watchlists:overview.onboarding.quickSetup.validation.sourceNameRequired",
                        "Enter a feed name"
                      )
                    }
                  ]}
                >
                  <Input
                    placeholder={t(
                      "watchlists:overview.onboarding.quickSetup.placeholders.sourceName",
                      "e.g., Daily Tech Feed"
                    )}
                    autoFocus
                  />
                </Form.Item>

                <Form.Item
                  label={t("watchlists:overview.onboarding.quickSetup.fields.sourceUrl", "Feed URL")}
                  name="sourceUrl"
                  rules={[
                    {
                      required: true,
                      message: t(
                        "watchlists:overview.onboarding.quickSetup.validation.sourceUrlRequired",
                        "Enter a feed URL"
                      )
                    },
                    {
                      validator: (_rule, value) => {
                        const raw = String(value || "").trim()
                        if (!raw) return Promise.resolve()
                        try {
                          const parsed = new URL(raw)
                          if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
                            throw new Error("invalid_protocol")
                          }
                          return Promise.resolve()
                        } catch {
                          return Promise.reject(
                            new Error(
                              t(
                                "watchlists:overview.onboarding.quickSetup.validation.sourceUrlInvalid",
                                "Enter a valid http(s) URL"
                              )
                            )
                          )
                        }
                      }
                    }
                  ]}
                >
                  <Input
                    placeholder={t(
                      "watchlists:overview.onboarding.quickSetup.placeholders.sourceUrl",
                      "https://example.com/feed.xml"
                    )}
                  />
                </Form.Item>

                <Form.Item
                  label={t(
                    "watchlists:overview.onboarding.quickSetup.fields.extraSourceUrls",
                    "Additional feed URLs (optional)"
                  )}
                  name="extraSourceUrls"
                  rules={[
                    {
                      validator: (_rule, value) => {
                        const raw = String(value || "").trim()
                        if (!raw) return Promise.resolve()
                        const entries = raw
                          .split(/\r?\n|,/)
                          .map((entry) => entry.trim())
                          .filter((entry) => entry.length > 0)
                        const validEntries = parseQuickSetupExtraSourceUrls(raw)
                        if (entries.length === validEntries.length) {
                          return Promise.resolve()
                        }
                        return Promise.reject(
                          new Error(
                            t(
                              "watchlists:overview.onboarding.quickSetup.validation.extraSourceUrlsInvalid",
                              "Enter valid http(s) URLs separated by commas or new lines"
                            )
                          )
                        )
                      }
                    }
                  ]}
                >
                  <Input.TextArea
                    autoSize={{ minRows: 2, maxRows: 5 }}
                    placeholder={t(
                      "watchlists:overview.onboarding.quickSetup.placeholders.extraSourceUrls",
                      "https://example.com/feed-a.xml\nhttps://example.com/feed-b.xml"
                    )}
                  />
                </Form.Item>

                <Form.Item
                  label={t("watchlists:overview.onboarding.quickSetup.fields.sourceType", "Feed type")}
                  name="sourceType"
                >
                  <Select
                    options={[
                      {
                        value: "rss",
                        label: t("watchlists:sources.types.rss", "RSS Feed")
                      },
                      {
                        value: "site",
                        label: t("watchlists:sources.types.site", "Website")
                      },
                      {
                        value: "forum",
                        label: t("watchlists:sources.types.forumComingSoon", "Forum (coming soon)"),
                        disabled: true
                      }
                    ]}
                  />
                </Form.Item>
              </div>
            )}

            {quickSetupStep === 1 && (
              <div className="space-y-1">
                <Form.Item
                  label={t("watchlists:overview.onboarding.quickSetup.fields.monitorName", "Monitor name")}
                  name="monitorName"
                  rules={[
                    {
                      required: true,
                      message: t(
                        "watchlists:overview.onboarding.quickSetup.validation.monitorNameRequired",
                        "Enter a monitor name"
                      )
                    }
                  ]}
                >
                  <Input
                    placeholder={t(
                      "watchlists:overview.onboarding.quickSetup.placeholders.monitorName",
                      "e.g., Morning Brief"
                    )}
                    autoFocus
                  />
                </Form.Item>

                <Form.Item
                  label={t("watchlists:overview.onboarding.quickSetup.fields.schedule", "Schedule")}
                  name="schedulePreset"
                >
                  <Select
                    options={[
                      {
                        value: "none",
                        label: t("watchlists:overview.onboarding.quickSetup.schedule.none", "Manual only")
                      },
                      {
                        value: "hourly",
                        label: t("watchlists:overview.onboarding.quickSetup.schedule.hourly", "Hourly")
                      },
                      {
                        value: "daily",
                        label: t("watchlists:overview.onboarding.quickSetup.schedule.daily", "Daily at 08:00")
                      },
                      {
                        value: "weekdays",
                        label: t("watchlists:overview.onboarding.quickSetup.schedule.weekdays", "Weekdays at 08:00")
                      }
                    ]}
                  />
                </Form.Item>

                <Form.Item
                  label={t("watchlists:overview.onboarding.quickSetup.fields.setupGoal", "Setup goal")}
                  name="setupGoal"
                >
                  <Select
                    options={[
                      {
                        value: "briefing",
                        label: t(
                          "watchlists:overview.onboarding.quickSetup.goal.briefing",
                          "Generate briefing reports"
                        )
                      },
                      {
                        value: "triage",
                        label: t(
                          "watchlists:overview.onboarding.quickSetup.goal.triage",
                          "Feed review only"
                        )
                      }
                    ]}
                  />
                </Form.Item>

                {(quickSetupValues?.setupGoal || "briefing") === "briefing" && (
                  <Form.Item
                    label={t(
                      "watchlists:overview.onboarding.quickSetup.fields.includeAudioBriefing",
                      "Include audio briefing"
                    )}
                    name="includeAudioBriefing"
                    valuePropName="checked"
                  >
                    <Switch
                      aria-label={t(
                        "watchlists:overview.onboarding.quickSetup.fields.includeAudioBriefing",
                        "Include audio briefing"
                      )}
                    />
                  </Form.Item>
                )}

                <Form.Item
                  label={t(
                    "watchlists:overview.onboarding.quickSetup.fields.runNow",
                    "Run test generation immediately"
                  )}
                  name="runNow"
                  valuePropName="checked"
                >
                  <Switch
                    aria-label={t("watchlists:overview.onboarding.quickSetup.fields.runNow", "Run immediately")}
                  />
                </Form.Item>
                <div className="-mt-3 text-xs text-text-muted">
                  {t(
                    "watchlists:overview.onboarding.quickSetup.hints.runNow",
                    "Runs one test generation after setup so you can verify results before waiting for the next schedule."
                  )}
                </div>
              </div>
            )}

            {quickSetupStep === 2 && (
              <div className="space-y-3 text-sm">
                <p className="text-text-muted">
                  {t(
                    "watchlists:overview.onboarding.quickSetup.reviewDescription",
                    "Preview sample candidates and expected briefing, then add feeds and monitor to this Watchlist."
                  )}
                </p>
                <div className="rounded-md border border-border bg-surface p-3">
                  <p>
                    <span className="font-medium">
                      {t("watchlists:overview.onboarding.quickSetup.review.feed", "Feed")}:
                    </span>{" "}
                    {quickSetupSnapshot.sourceName || "—"}
                  </p>
                  <p className="text-text-muted">
                    {quickSetupSnapshot.sourceUrl || "—"}
                  </p>
                  <p className="mt-1 text-text-muted">
                    {t("watchlists:overview.onboarding.quickSetup.review.feedCount", "Total feeds: {{count}}", {
                      count: (quickSetupValues?.sourceUrl ? 1 : 0) + quickSetupExtraSourceUrls.length
                    })}
                  </p>
                  <p className="mt-2">
                    <span className="font-medium">
                      {t("watchlists:overview.onboarding.quickSetup.review.monitor", "Monitor")}:
                    </span>{" "}
                    {quickSetupSnapshot.monitorName || "—"}
                  </p>
                  <p>
                    <span className="font-medium">
                      {t("watchlists:overview.onboarding.quickSetup.review.schedule", "Schedule")}:
                    </span>{" "}
                    {quickSetupSnapshot.schedulePreset === "hourly"
                      ? t("watchlists:overview.onboarding.quickSetup.schedule.hourly", "Hourly")
                      : quickSetupSnapshot.schedulePreset === "daily"
                        ? t("watchlists:overview.onboarding.quickSetup.schedule.daily", "Daily at 08:00")
                        : quickSetupSnapshot.schedulePreset === "weekdays"
                          ? t("watchlists:overview.onboarding.quickSetup.schedule.weekdays", "Weekdays at 08:00")
                          : t("watchlists:overview.onboarding.quickSetup.schedule.none", "Manual only")}
                  </p>
                  <p>
                    <span className="font-medium">
                      {t("watchlists:overview.onboarding.quickSetup.review.goal", "Goal")}:
                    </span>{" "}
                    {quickSetupSnapshot.setupGoal === "triage"
                      ? t("watchlists:overview.onboarding.quickSetup.goal.triage", "Feed review only")
                      : t(
                          "watchlists:overview.onboarding.quickSetup.goal.briefing",
                          "Generate briefing reports"
                        )}
                  </p>
                  {(quickSetupValues?.setupGoal || "briefing") === "briefing" && (
                    <p>
                      <span className="font-medium">
                        {t("watchlists:overview.onboarding.quickSetup.review.audio", "Audio briefing")}:
                      </span>{" "}
                      {quickSetupValues?.includeAudioBriefing
                        ? t("common:enabled", "Enabled")
                        : t("common:disabled", "Disabled")}
                    </p>
                  )}
                  <p>
                    <span className="font-medium">
                      {t("watchlists:overview.onboarding.quickSetup.review.audio", "Audio")}:
                    </span>{" "}
                    {quickSetupSnapshot.setupGoal === "triage"
                      ? t("watchlists:overview.onboarding.quickSetup.outcome.triage", "Update triage only")
                      : quickSetupSnapshot.includeAudioBriefing
                        ? t("watchlists:overview.onboarding.quickSetup.outcome.textAndAudio", "Text + audio briefing")
                        : t("watchlists:overview.onboarding.quickSetup.outcome.textOnly", "Text briefing")}
                  </p>
                  <p>
                    <span className="font-medium">
                      {t("watchlists:overview.onboarding.quickSetup.review.runNow", "Run test now")}:
                    </span>{" "}
                    {quickSetupSnapshot.runNow
                      ? t("common:yes", "Yes")
                      : t("common:no", "No")}
                  </p>
                  <p>
                    <span className="font-medium">
                      {t("watchlists:overview.onboarding.quickSetup.review.destination", "After create")}:
                    </span>{" "}
                    {quickSetupDestinationHint}
                  </p>
                </div>

                <div className="rounded-md border border-border bg-surface p-3">
                  <div className="mb-1 flex items-center justify-between gap-2">
                    <span className="font-medium">
                      {t("watchlists:overview.onboarding.quickSetup.candidatePreview.title", "Feed sample preview")}
                    </span>
                    <Button
                      type="link"
                      size="small"
                      onClick={() => void loadQuickSetupCandidatePreview(quickSetupSnapshot)}
                    >
                      {t("watchlists:overview.onboarding.quickSetup.actions.retryPreview", "Retry preview")}
                    </Button>
                  </div>
                  <p
                    className={quickSetupCandidatePreviewError ? "text-danger" : "text-text-muted"}
                    data-testid={
                      quickSetupCandidatePreviewError
                        ? "watchlists-overview-quick-setup-candidate-error"
                        : "watchlists-overview-quick-setup-candidate-summary"
                    }
                  >
                    {quickSetupCandidateSummary}
                  </p>
                  {quickSetupCandidatePreviewLoading && (
                    <div className="mt-2">
                      <Spin size="small" />
                    </div>
                  )}
                  {!quickSetupCandidatePreviewLoading &&
                    !quickSetupCandidatePreviewError &&
                    (quickSetupCandidatePreview?.items?.length || 0) > 0 && (
                      <ul className="mt-2 space-y-1">
                        {quickSetupCandidatePreview?.items.slice(0, 4).map((item, index) => (
                          <li key={`${item.url || item.title || "candidate"}-${index}`} className="flex gap-2">
                            <Tag className="m-0" color={item.decision === "ingest" ? "green" : "red"}>
                              {item.decision}
                            </Tag>
                            <span className="text-text-muted">
                              {item.title || item.url || t("watchlists:common.unknown", "Unknown")}
                            </span>
                          </li>
                        ))}
                      </ul>
                    )}
                </div>

                <div className="rounded-md border border-border bg-surface p-3">
                  <p className="mb-2 font-medium">
                    {t(
                      "watchlists:overview.onboarding.quickSetup.templatePreview.title",
                      "Briefing preview (template: {{template}})",
                      { template: "briefing_md" }
                    )}
                  </p>
                  <pre
                    className="max-h-52 overflow-auto whitespace-pre-wrap rounded-md border border-border bg-bg p-3 text-xs leading-5 text-text"
                    data-testid="watchlists-overview-quick-setup-template-preview"
                  >
                    {quickSetupTemplatePreview}
                  </pre>
                </div>
                <p className="text-xs text-text-muted">{quickSetupDestinationHint}</p>
              </div>
            )}
          </Form>
        </div>
      </Modal>

      <Modal
        open={pipelineSetupOpen}
        title={t("watchlists:overview.pipelineSetup.title", "Briefing pipeline builder")}
        onCancel={closePipelineSetup}
        destroyOnHidden
        maskClosable={!pipelineSetupSubmitting}
        footer={[
          <Button
            key="cancel"
            onClick={closePipelineSetup}
            disabled={pipelineSetupSubmitting}
          >
            {t("common:cancel", "Cancel")}
          </Button>,
          <Button
            key="back"
            onClick={handlePipelineSetupBack}
            disabled={pipelineSetupSubmitting || pipelineSetupStep === 0}
          >
            {t("common:back", "Back")}
          </Button>,
          pipelineSetupIsLastStep ? (
            <Button
              key="test-generation"
              data-testid="watchlists-pipeline-test-generation"
              onClick={() => {
                void completePipelineSetup({ mode: "test", forceRunNow: true })
              }}
              loading={pipelineSetupSubmitting}
            >
              {t("watchlists:overview.pipelineSetup.actions.testGeneration", "Run test generation")}
            </Button>
          ) : null,
          <Button
            key="next"
            type="primary"
            loading={pipelineSetupSubmitting}
            onClick={() => {
              if (pipelineSetupIsLastStep) {
                void completePipelineSetup({ mode: "create" })
              } else {
                void handlePipelineSetupNext()
              }
            }}
          >
            {pipelineSetupIsLastStep
              ? t("watchlists:overview.pipelineSetup.actions.finish", "Create pipeline")
              : t("common:next", "Next")}
          </Button>
        ]}
      >
        <div className="space-y-4">
          <Steps
            size="small"
            current={pipelineSetupStep}
            items={pipelineStepItems}
          />

          <Form
            form={pipelineSetupForm}
            layout="vertical"
            initialValues={PIPELINE_DEFAULT_VALUES}
          >
            {pipelineSetupStep === 0 && (
              <div className="space-y-2">
                <Form.Item
                  label={t("watchlists:overview.pipelineSetup.fields.sources", "Feeds")}
                  name="sourceIds"
                  rules={[
                    {
                      validator: (_rule, value) => {
                        if (Array.isArray(value) && value.length > 0) {
                          return Promise.resolve()
                        }
                        return Promise.reject(
                          new Error(
                            t(
                              "watchlists:overview.pipelineSetup.validation.sourcesRequired",
                              "Select at least one feed"
                            )
                          )
                        )
                      }
                    }
                  ]}
                >
                  <Checkbox.Group className="grid gap-2">
                    {pipelineSources.map((source) => (
                      <Checkbox key={source.id} value={source.id}>
                        {source.name || `Feed #${source.id}`}
                      </Checkbox>
                    ))}
                  </Checkbox.Group>
                </Form.Item>
                {pipelineSourcesLoading && (
                  <div className="text-xs text-text-muted">
                    {t("watchlists:overview.pipelineSetup.sourcesLoading", "Loading feeds...")}
                  </div>
                )}
              </div>
            )}

            {pipelineSetupStep === 1 && (
              <div className="space-y-1">
                <Form.Item
                  label={t("watchlists:overview.pipelineSetup.fields.monitorName", "Monitor name")}
                  name="monitorName"
                  rules={[
                    {
                      required: true,
                      message: t(
                        "watchlists:overview.pipelineSetup.validation.monitorNameRequired",
                        "Enter a monitor name"
                      )
                    }
                  ]}
                >
                  <Input autoFocus />
                </Form.Item>

                <Form.Item
                  label={t("watchlists:overview.pipelineSetup.fields.schedule", "Schedule")}
                  name="schedulePreset"
                >
                  <Select
                    options={[
                      { value: "none", label: t("watchlists:overview.onboarding.quickSetup.schedule.none", "Manual only") },
                      { value: "hourly", label: t("watchlists:overview.onboarding.quickSetup.schedule.hourly", "Hourly") },
                      { value: "daily", label: t("watchlists:overview.onboarding.quickSetup.schedule.daily", "Daily at 08:00") },
                      { value: "weekdays", label: t("watchlists:overview.onboarding.quickSetup.schedule.weekdays", "Weekdays at 08:00") }
                    ]}
                  />
                </Form.Item>

                <Form.Item
                  label={t("watchlists:overview.pipelineSetup.fields.template", "Template")}
                  name="templateName"
                  rules={[
                    {
                      required: true,
                      message: t(
                        "watchlists:overview.pipelineSetup.validation.templateRequired",
                        "Enter a template name"
                      )
                    }
                  ]}
                >
                  <Input />
                </Form.Item>

                <Form.Item
                  label={t("watchlists:overview.pipelineSetup.fields.includeAudio", "Include audio briefing")}
                  name="includeAudio"
                  valuePropName="checked"
                >
                  <Switch
                    aria-label={t(
                      "watchlists:overview.pipelineSetup.fields.includeAudio",
                      "Include audio briefing"
                    )}
                  />
                </Form.Item>

                {pipelineSetupValues?.includeAudio && (
                  <>
                    <Form.Item
                      label={t("watchlists:overview.pipelineSetup.fields.audioVoice", "Audio voice")}
                      name="audioVoice"
                      rules={[
                        {
                          required: true,
                          message: t(
                            "watchlists:overview.pipelineSetup.validation.audioVoiceRequired",
                            "Select an audio voice"
                          )
                        }
                      ]}
                    >
                      <Select
                        options={[
                          { value: "alloy", label: "Alloy" },
                          { value: "nova", label: "Nova" },
                          { value: "echo", label: "Echo" }
                        ]}
                      />
                    </Form.Item>
                    <Form.Item
                      label={t("watchlists:overview.pipelineSetup.fields.audioMinutes", "Target audio minutes")}
                      name="targetAudioMinutes"
                      rules={[
                        {
                          required: true,
                          type: "number",
                          min: 1,
                          message: t(
                            "watchlists:overview.pipelineSetup.validation.audioMinutesRequired",
                            "Enter target audio minutes"
                          )
                        }
                      ]}
                    >
                      <Input type="number" min={1} />
                    </Form.Item>
                  </>
                )}

                <Form.Item
                  label={t("watchlists:overview.pipelineSetup.fields.emailDelivery", "Email delivery")}
                  name="emailDeliveryEnabled"
                  valuePropName="checked"
                >
                  <Switch
                    aria-label={t("watchlists:overview.pipelineSetup.fields.emailDelivery", "Email delivery")}
                  />
                </Form.Item>

                {pipelineSetupValues?.emailDeliveryEnabled && (
                  <Form.Item
                    label={t("watchlists:overview.pipelineSetup.fields.emailRecipients", "Email recipients")}
                    name="emailRecipients"
                    rules={[
                      {
                        validator: (_rule, value) => {
                          const recipients = normalizePipelineRecipients(value)
                          if (recipients.length === 0) {
                            return Promise.reject(
                              new Error(
                                t(
                                  "watchlists:overview.pipelineSetup.validation.emailRecipientsRequired",
                                  "Enter at least one recipient email"
                                )
                              )
                            )
                          }
                          const invalidRecipients = recipients.filter(
                            (entry) => !PIPELINE_EMAIL_PATTERN.test(entry)
                          )
                          if (invalidRecipients.length > 0) {
                            return Promise.reject(
                              new Error(
                                t(
                                  "watchlists:overview.pipelineSetup.validation.emailRecipientsInvalid",
                                  "Fix invalid email recipients before continuing."
                                )
                              )
                            )
                          }
                          return Promise.resolve()
                        }
                      }
                    ]}
                  >
                    <Select
                      mode="tags"
                      tokenSeparators={[","]}
                      placeholder={t(
                        "watchlists:overview.pipelineSetup.fields.emailRecipientsPlaceholder",
                        "name@example.com"
                      )}
                    />
                  </Form.Item>
                )}

                <Form.Item
                  label={t("watchlists:overview.pipelineSetup.fields.chatbookDelivery", "Chatbook delivery")}
                  name="chatbookDeliveryEnabled"
                  valuePropName="checked"
                >
                  <Switch
                    aria-label={t(
                      "watchlists:overview.pipelineSetup.fields.chatbookDelivery",
                      "Chatbook delivery"
                    )}
                  />
                </Form.Item>

                {pipelineSetupValues?.chatbookDeliveryEnabled && (
                  <Form.Item
                    label={t("watchlists:overview.pipelineSetup.fields.chatbookTitle", "Chatbook title")}
                    name="chatbookTitle"
                  >
                    <Input
                      placeholder={t(
                        "watchlists:overview.pipelineSetup.fields.chatbookTitlePlaceholder",
                        "Morning Intel"
                      )}
                    />
                  </Form.Item>
                )}

                <Form.Item
                  label={t("watchlists:overview.pipelineSetup.fields.runNow", "Run immediately")}
                  name="runNow"
                  valuePropName="checked"
                >
                  <Switch
                    aria-label={t("watchlists:overview.pipelineSetup.fields.runNow", "Run immediately")}
                  />
                </Form.Item>
              </div>
            )}

            {pipelineSetupStep === 2 && (
              <div className="space-y-3 text-sm">
                <p className="text-text-muted">
                  {t(
                    "watchlists:overview.pipelineSetup.reviewDescription",
                    "Confirm this pipeline before creating monitor, run, and output artifacts."
                  )}
                </p>
                <div className="rounded-md border border-border bg-surface p-3">
                  <p>
                    <span className="font-medium">
                      {t("watchlists:overview.pipelineSetup.review.monitor", "Monitor")}:
                    </span>{" "}
                    {pipelineSetupValues?.monitorName || "—"}
                  </p>
                  <p>
                    <span className="font-medium">
                      {t("watchlists:overview.pipelineSetup.review.feeds", "Feeds")}:
                    </span>{" "}
                    {Array.isArray(pipelineSetupValues?.sourceIds)
                      ? pipelineSetupValues?.sourceIds.length
                      : 0}
                  </p>
                  <p>
                    <span className="font-medium">
                      {t("watchlists:overview.pipelineSetup.review.schedule", "Schedule")}:
                    </span>{" "}
                    {pipelineReviewSummary.scheduleLabel}
                  </p>
                  <p>
                    <span className="font-medium">
                      {t("watchlists:overview.pipelineSetup.review.artifacts", "Artifacts")}:
                    </span>{" "}
                    {pipelineReviewSummary.artifacts.join(", ")}
                  </p>
                  <p>
                    <span className="font-medium">
                      {t("watchlists:overview.pipelineSetup.review.deliveries", "Deliveries")}:
                    </span>{" "}
                    {pipelineReviewSummary.deliveries.join(", ")}
                  </p>
                  {pipelineSetupValues?.emailDeliveryEnabled && (
                    <p>
                      <span className="font-medium">
                        {t("watchlists:overview.pipelineSetup.review.emailRecipients", "Email recipients")}:
                      </span>{" "}
                      {pipelineEmailRecipients.length}
                    </p>
                  )}
                  {pipelineSetupValues?.chatbookDeliveryEnabled && (
                    <p>
                      <span className="font-medium">
                        {t("watchlists:overview.pipelineSetup.review.chatbookTitle", "Chatbook title")}:
                      </span>{" "}
                      {String(pipelineSetupValues?.chatbookTitle || "").trim() || "Watchlists Briefing"}
                    </p>
                  )}
                  <p>
                    <span className="font-medium">
                      {t("watchlists:overview.pipelineSetup.review.runNow", "Run now")}:
                    </span>{" "}
                    {pipelineSetupValues?.runNow ? t("common:yes", "Yes") : t("common:no", "No")}
                  </p>
                </div>
                <div className="rounded-md border border-border bg-surface p-3 text-xs text-text-muted space-y-1">
                  <p data-testid="watchlists-pipeline-review-outcome-text">
                    {t(
                      "watchlists:overview.pipelineSetup.review.textOutcome",
                      "Text outcome: {{template}} template will generate a written report artifact.",
                      {
                        template: String(pipelineSetupValues?.templateName || "briefing_md")
                      }
                    )}
                  </p>
                  <p data-testid="watchlists-pipeline-review-outcome-audio">
                    {pipelineSetupValues?.includeAudio
                      ? t(
                          "watchlists:overview.pipelineSetup.review.audioOutcomeEnabled",
                          "Audio outcome: voice {{voice}} targeting about {{minutes}} minutes.",
                          {
                            voice: String(pipelineSetupValues?.audioVoice || "alloy"),
                            minutes: Number(pipelineSetupValues?.targetAudioMinutes || 8)
                          }
                        )
                      : t(
                          "watchlists:overview.pipelineSetup.review.audioOutcomeDisabled",
                          "Audio outcome: disabled. Reports will be text-only."
                        )}
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
                      onClick={() => {
                        void generatePipelineTemplatePreview()
                      }}
                      loading={pipelinePreviewLoading}
                      data-testid="watchlists-pipeline-preview-generate"
                    >
                      {t("watchlists:overview.pipelineSetup.preview.generate", "Generate preview")}
                    </Button>
                  </div>
                  {pipelinePreviewError && (
                    <Alert
                      type="warning"
                      showIcon
                      data-testid="watchlists-pipeline-preview-error"
                      title={pipelinePreviewError}
                    />
                  )}
                  {pipelinePreviewRunId != null && !pipelinePreviewError && (
                    <p className="text-xs text-text-muted">
                      {t(
                        "watchlists:overview.pipelineSetup.preview.context",
                        "Preview context run: #{{runId}}",
                        { runId: pipelinePreviewRunId }
                      )}
                    </p>
                  )}
                  {pipelinePreviewWarnings.length > 0 && (
                    <ul className="list-disc pl-5 text-xs text-text-muted">
                      {pipelinePreviewWarnings.map((warning, index) => (
                        <li key={`${warning}-${index}`}>{warning}</li>
                      ))}
                    </ul>
                  )}
                  {pipelinePreviewRendered && (
                    <pre
                      className="max-h-48 overflow-auto rounded border border-border bg-background p-2 text-xs"
                      data-testid="watchlists-pipeline-preview-rendered"
                    >
                      {pipelinePreviewRendered}
                    </pre>
                  )}
                </div>
              </div>
            )}
          </Form>
        </div>
      </Modal>
    </div>
  )
}
