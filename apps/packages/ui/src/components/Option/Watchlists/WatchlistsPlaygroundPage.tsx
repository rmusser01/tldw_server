import React, { Suspense, useCallback, useEffect, useMemo, useRef } from "react"
import { Button, Drawer, Input, Modal, Select, Switch, Tabs, Tag, Tooltip } from "antd"
import { DismissibleBetaAlert } from "@/components/Common/DismissibleBetaAlert"
import { Alert as DesignSystemAlert } from "@/components/ui/primitives"
import type { TabsProps } from "antd"
import {
  BellRing,
  CalendarClock,
  ChevronDown,
  ChevronUp,
  Command,
  ExternalLink,
  FileOutput,
  FileText,
  HelpCircle,
  LayoutDashboard,
  Newspaper,
  Pencil,
  Plus,
  Play,
  Rss,
  Settings
} from "lucide-react"
import { useTranslation } from "react-i18next"
import { useAntdNotification } from "@/hooks/useAntdNotification"
import { useServerOnline } from "@/hooks/useServerOnline"
import { PageShell } from "@/components/Common/PageShell"
import WorkspaceConnectionGate from "@/components/Common/WorkspaceConnectionGate"
import { downloadBlob } from "@/utils/download-blob"
import {
  bulkCreateSources,
  createWatchlist,
  createWatchlistJob,
  createWatchlistSource,
  exportOpml,
  exportRunsCsv,
  fetchWatchlistRuns,
  fetchWatchlists,
  triggerWatchlistRun,
  updateWatchlist
} from "@/services/watchlists"
import { useWatchlistsStore } from "@/store/watchlists"
import type {
  WatchlistContainer,
  WatchlistDomain,
  WatchlistJobCreate,
  WatchlistPriority,
  WatchlistRun,
  WatchlistSourceCreate,
  WatchlistStatus
} from "@/types/watchlists"
import type { WatchlistTab } from "@/types/watchlists"
import {
  WatchlistSetupWizard,
  type WatchlistSetupCompleteResult
} from "./SetupWizard"
import {
  WATCHLISTS_ISSUE_REPORT_URL,
  WATCHLISTS_MAIN_DOCS_URL,
  WATCHLISTS_TAB_HELP_DOCS
} from "./shared/help-docs"
import { WatchlistsHealthBar } from "./shared/WatchlistsHealthBar"
import {
  WatchlistsMobileNavigation,
  type WatchlistsMobileNavigationGroup
} from "./shared/WatchlistsMobileNavigation"
import { useWatchlistsViewport } from "./shared/useWatchlistsViewport"
import { WatchlistsCommandPalette, useWatchlistsCommands } from "./shared/WatchlistsCommandPalette"
import { useWatchlistsKeyboardShortcuts } from "./shared/useWatchlistsKeyboardShortcuts"
import {
  buildRunStateNotificationKey,
  dedupeRunNotificationEvents,
  groupRunNotificationEvents,
  getRunFailureHint,
  resolveRunNotificationsPollPlan,
  resolveStalledRunNotification,
  resolveRunTransitionNotification,
  shouldNotifyNewTerminalRun
} from "./RunsTab/run-notifications"
import {
  flushWatchlistsIaExperimentSession,
  trackWatchlistsIaExperimentTransition
} from "@/utils/watchlists-ia-experiment-telemetry"
import { trackWatchlistsOnboardingTelemetry } from "@/utils/watchlists-onboarding-telemetry"
import { resolveWatchlistsIaExperimentRollout } from "@/utils/watchlists-ia-rollout"
import { resolvePreferredWatchlistId } from "./watchlist-selection"

const RUN_NOTIFICATIONS_POLL_MS = 15_000
const RUN_NOTIFICATIONS_PAGE_SIZE = 25
const RUN_NOTIFICATIONS_REDUCED_PAGE_SIZE = 10
const RUN_NOTIFICATIONS_MIN_POLL_MS = 100
const RUN_NOTIFICATIONS_BACKGROUND_POLL_MS = 60_000
const RUN_NOTIFICATIONS_RUNS_TAB_POLL_MS = 30_000
const RUN_STALLED_THRESHOLD_MS = 45 * 60_000
const GUIDED_TOUR_STORAGE_KEY = "watchlists:guided-tour:v1"
const TEACH_POINTS_STORAGE_KEY = "watchlists:teach-points:v1"
const ORIENTATION_DISMISSED_STORAGE_KEY = "watchlists:orientation-dismissed:v1"
const SHOW_ALL_VIEWS_STORAGE_KEY = "watchlists:show-all-views:v1"
const SECONDARY_EXPANDED_STORAGE_KEY = "watchlists:secondary-expanded:v1"
const SUCCESSFUL_RUN_STATUSES = new Set(["completed", "succeeded", "success", "done", "finished"])

const OverviewTab = React.lazy(() =>
  import("./OverviewTab/OverviewTab").then((module) => ({ default: module.OverviewTab }))
)
const SourcesTab = React.lazy(() =>
  import("./SourcesTab/SourcesTab").then((module) => ({ default: module.SourcesTab }))
)
const JobsTab = React.lazy(() =>
  import("./JobsTab/JobsTab").then((module) => ({ default: module.JobsTab }))
)
const RunsTab = React.lazy(() =>
  import("./RunsTab/RunsTab").then((module) => ({ default: module.RunsTab }))
)
const ItemsTab = React.lazy(() =>
  import("./ItemsTab/ItemsTab").then((module) => ({ default: module.ItemsTab }))
)
const OutputsTab = React.lazy(() =>
  import("./OutputsTab/OutputsTab").then((module) => ({ default: module.OutputsTab }))
)
const TemplatesTab = React.lazy(() =>
  import("./TemplatesTab/TemplatesTab").then((module) => ({ default: module.TemplatesTab }))
)
const SettingsTab = React.lazy(() =>
  import("./SettingsTab/SettingsTab").then((module) => ({ default: module.SettingsTab }))
)
const AlertsTab = React.lazy(() =>
  import("./AlertsTab/AlertsTab").then((module) => ({ default: module.AlertsTab }))
)

/** Primary tabs in the progressive disclosure layout */
const PROGRESSIVE_PRIMARY_TABS = ["sources", "alerts", "items", "outputs"] as const

/** Which secondary section lives inside which primary tab */
const SECONDARY_IN_PRIMARY: Record<string, string> = {
  jobs: "sources",    // Monitors section inside Feeds tab
  runs: "items",      // Activity section inside Updates tab
  templates: "outputs" // Templates section inside Reports tab
}

const readShowAllViews = (): boolean => {
  if (typeof window === "undefined") return false
  try {
    return localStorage.getItem(SHOW_ALL_VIEWS_STORAGE_KEY) === "true"
  } catch {
    return false
  }
}

const writeShowAllViews = (value: boolean): void => {
  if (typeof window === "undefined") return
  try {
    localStorage.setItem(SHOW_ALL_VIEWS_STORAGE_KEY, String(value))
  } catch {
    // localStorage may be unavailable
  }
}

type SecondaryExpandedState = Partial<Record<string, boolean>>

const readSecondaryExpanded = (): SecondaryExpandedState => {
  if (typeof window === "undefined") return {}
  try {
    const raw = localStorage.getItem(SECONDARY_EXPANDED_STORAGE_KEY)
    if (!raw) return {}
    const parsed = JSON.parse(raw) as SecondaryExpandedState
    return parsed && typeof parsed === "object" ? parsed : {}
  } catch {
    return {}
  }
}

const writeSecondaryExpanded = (state: SecondaryExpandedState): void => {
  if (typeof window === "undefined") return
  try {
    localStorage.setItem(SECONDARY_EXPANDED_STORAGE_KEY, JSON.stringify(state))
  } catch {
    // localStorage may be unavailable
  }
}

type GuidedTourTab = "sources" | "jobs" | "runs" | "items" | "outputs"
type GuidedTourStatus = "idle" | "in_progress" | "dismissed" | "completed"
type TaskViewKey = "collect" | "review" | "briefings"
type WatchlistsTabKey =
  | "overview"
  | "sources"
  | "jobs"
  | "runs"
  | "items"
  | "alerts"
  | "outputs"
  | "templates"
  | "settings"

interface OrientationAction {
  key: string
  label: string
  target: WatchlistsTabKey
}

interface TabOrientation {
  title: string
  description: string
  actions: OrientationAction[]
}
interface GuidedTourState {
  status: GuidedTourStatus
  step: number
}
type TeachPointKey = "jobsCronFilters" | "templatesAuthoring"
type WatchlistFormMode = "create" | "edit"

interface TeachPointState {
  jobsCronFilters: boolean
  templatesAuthoring: boolean
}
type OrientationDismissState = Partial<Record<WatchlistsTabKey, boolean>>
interface WatchlistFormState {
  name: string
  description: string
  objective: string
  domain: WatchlistDomain
  status: WatchlistStatus
  priority: WatchlistPriority
  tagsText: string
}

const GUIDED_TOUR_TABS: GuidedTourTab[] = ["sources", "jobs", "runs", "items", "outputs"]
const GUIDED_TOUR_LAST_STEP = GUIDED_TOUR_TABS.length - 1
const TASK_VIEW_PRIMARY_TAB: Record<TaskViewKey, "sources" | "items" | "outputs"> = {
  collect: "sources",
  review: "items",
  briefings: "outputs"
}

const WATCHLIST_FORM_DEFAULTS: WatchlistFormState = {
  name: "",
  description: "",
  objective: "",
  domain: "general",
  status: "active",
  priority: "medium",
  tagsText: ""
}

const WATCHLIST_DOMAIN_LABELS: Record<WatchlistDomain, string> = {
  cti_osint: "CTI / OSINT",
  news: "News",
  general: "General"
}

const WATCHLIST_STATUS_LABELS: Record<WatchlistStatus, string> = {
  active: "Active",
  paused: "Paused",
  archived: "Archived"
}

const WATCHLIST_PRIORITY_LABELS: Record<WatchlistPriority, string> = {
  low: "Low",
  medium: "Medium",
  high: "High",
  critical: "Critical"
}

const SETUP_DESTINATION_TAB: Record<WatchlistSetupCompleteResult["destination"], WatchlistsTabKey> = {
  sources: "sources",
  jobs: "jobs",
  outputs: "outputs"
}

const toWatchlistFormState = (watchlist: WatchlistContainer | null): WatchlistFormState => {
  if (!watchlist) return WATCHLIST_FORM_DEFAULTS
  return {
    name: watchlist.name || "",
    description: watchlist.description || "",
    objective: watchlist.objective || "",
    domain: watchlist.domain || "general",
    status: watchlist.status || "active",
    priority: watchlist.priority || "medium",
    tagsText: Array.isArray(watchlist.tags) ? watchlist.tags.join(", ") : ""
  }
}

const toOptionalText = (value: string): string | undefined => {
  const trimmed = value.trim()
  return trimmed.length > 0 ? trimmed : undefined
}

const toTags = (value: string): string[] =>
  value
    .split(",")
    .map((tag) => tag.trim())
    .filter((tag) => tag.length > 0)

const resolveTaskViewForTab = (tab: string): TaskViewKey | null => {
  if (tab === "sources" || tab === "jobs") return "collect"
  if (tab === "runs" || tab === "items" || tab === "alerts") return "review"
  if (tab === "outputs" || tab === "templates") return "briefings"
  return null
}

const clampTourStep = (step: number): number => {
  if (!Number.isFinite(step)) return 0
  return Math.max(0, Math.min(Math.floor(step), GUIDED_TOUR_LAST_STEP))
}

const toGuidedTourStep = (step: number): 1 | 2 | 3 | 4 | 5 =>
  (clampTourStep(step) + 1) as 1 | 2 | 3 | 4 | 5

const readGuidedTourState = (): GuidedTourState => {
  if (typeof window === "undefined") return { status: "idle", step: 0 }
  try {
    const raw = localStorage.getItem(GUIDED_TOUR_STORAGE_KEY)
    if (!raw) return { status: "idle", step: 0 }
    const parsed = JSON.parse(raw) as Partial<GuidedTourState>
    const status =
      parsed.status === "in_progress" ||
      parsed.status === "dismissed" ||
      parsed.status === "completed"
        ? parsed.status
        : "idle"
    return { status, step: clampTourStep(Number(parsed.step || 0)) }
  } catch {
    return { status: "idle", step: 0 }
  }
}

const writeGuidedTourState = (state: GuidedTourState): void => {
  if (typeof window === "undefined") return
  try {
    localStorage.setItem(GUIDED_TOUR_STORAGE_KEY, JSON.stringify(state))
  } catch {
    // localStorage may be unavailable.
  }
}

const readTeachPointState = (): TeachPointState => {
  if (typeof window === "undefined") {
    return {
      jobsCronFilters: false,
      templatesAuthoring: false
    }
  }
  try {
    const raw = localStorage.getItem(TEACH_POINTS_STORAGE_KEY)
    if (!raw) {
      return {
        jobsCronFilters: false,
        templatesAuthoring: false
      }
    }
    const parsed = JSON.parse(raw) as Partial<TeachPointState>
    return {
      jobsCronFilters: Boolean(parsed.jobsCronFilters),
      templatesAuthoring: Boolean(parsed.templatesAuthoring)
    }
  } catch {
    return {
      jobsCronFilters: false,
      templatesAuthoring: false
    }
  }
}

const writeTeachPointState = (state: TeachPointState): void => {
  if (typeof window === "undefined") return
  try {
    localStorage.setItem(TEACH_POINTS_STORAGE_KEY, JSON.stringify(state))
  } catch {
    // localStorage may be unavailable.
  }
}

const readOrientationDismissState = (): OrientationDismissState => {
  if (typeof window === "undefined") return {}
  try {
    const raw = localStorage.getItem(ORIENTATION_DISMISSED_STORAGE_KEY)
    if (!raw) return {}
    const parsed = JSON.parse(raw) as OrientationDismissState
    return parsed && typeof parsed === "object" ? parsed : {}
  } catch {
    return {}
  }
}

const writeOrientationDismissState = (state: OrientationDismissState): void => {
  if (typeof window === "undefined") return
  try {
    localStorage.setItem(ORIENTATION_DISMISSED_STORAGE_KEY, JSON.stringify(state))
  } catch {
    // localStorage may be unavailable.
  }
}

const resolveRunNotificationsPollMs = (): number => {
  if (typeof window === "undefined") return RUN_NOTIFICATIONS_POLL_MS
  const override = Number(
    (window as { __TLDW_WATCHLISTS_RUN_NOTIFICATIONS_POLL_MS?: unknown })
      .__TLDW_WATCHLISTS_RUN_NOTIFICATIONS_POLL_MS
  )
  if (!Number.isFinite(override)) return RUN_NOTIFICATIONS_POLL_MS
  return Math.max(RUN_NOTIFICATIONS_MIN_POLL_MS, Math.floor(override))
}

/**
 * WatchlistsPlaygroundPage
 *
 * Main container for the Watchlists module playground.
 * Provides a tabbed interface for managing sources, jobs, runs, outputs, templates, and settings.
 */
/** Expandable inline secondary section for progressive disclosure layout */
const InlineSecondarySection: React.FC<{
  sectionKey: string
  title: string
  count?: number
  expanded: boolean
  onToggle: (key: string) => void
  children: React.ReactNode
}> = ({ sectionKey, title, count, expanded, onToggle, children }) => (
  <div className="mt-6 border-t border-border pt-4" data-testid={`watchlists-secondary-${sectionKey}`}>
    <div
      className="flex cursor-pointer items-center gap-2 text-sm font-medium text-text-muted hover:text-text"
      onClick={() => onToggle(sectionKey)}
      role="button"
      tabIndex={0}
      aria-expanded={expanded}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault()
          onToggle(sectionKey)
        }
      }}
    >
      {expanded ? (
        <ChevronUp className="h-4 w-4" />
      ) : (
        <ChevronDown className="h-4 w-4" />
      )}
      <span>{title}</span>
      {count !== undefined && count > 0 && (
        <span className="text-xs text-text-muted">({count})</span>
      )}
    </div>
    {expanded && <div className="mt-3">{children}</div>}
  </div>
)

export const WatchlistsPlaygroundPage: React.FC = () => {
  const { t } = useTranslation(["watchlists", "common"])
  const isOnline = useServerOnline()
  const notification = useAntdNotification()

  const activeTab = useWatchlistsStore((s) => s.activeTab)
  const overviewHealth = useWatchlistsStore((s) => s.overviewHealth)
  const runsPollingActive = useWatchlistsStore((s) => s.pollingActive)
  const watchlists = useWatchlistsStore((s) => s.watchlists)
  const watchlistsLoading = useWatchlistsStore((s) => s.watchlistsLoading)
  const watchlistsError = useWatchlistsStore((s) => s.watchlistsError)
  const selectedWatchlistId = useWatchlistsStore((s) => s.selectedWatchlistId)
  const setActiveTab = useWatchlistsStore((s) => s.setActiveTab)
  const openRunDetail = useWatchlistsStore((s) => s.openRunDetail)
  const openSourceForm = useWatchlistsStore((s) => s.openSourceForm)
  const openJobForm = useWatchlistsStore((s) => s.openJobForm)
  const setWatchlists = useWatchlistsStore((s) => s.setWatchlists)
  const setWatchlistsLoading = useWatchlistsStore((s) => s.setWatchlistsLoading)
  const setWatchlistsError = useWatchlistsStore((s) => s.setWatchlistsError)
  const setSelectedWatchlistId = useWatchlistsStore((s) => s.setSelectedWatchlistId)
  const addWatchlist = useWatchlistsStore((s) => s.addWatchlist)
  const updateWatchlistInList = useWatchlistsStore((s) => s.updateWatchlistInList)
  const resetStore = useWatchlistsStore((s) => s.resetStore)
  const runStatusRef = useRef<Map<number, string>>(new Map())
  const notifiedRunStatesRef = useRef<Set<string>>(new Set())
  const initializedRunPollingRef = useRef(false)
  const runNotificationsTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const runNotificationsPollingInFlightRef = useRef(false)
  const sessionStartedAtMsRef = useRef<number>(Date.now())
  const [guidedTourState, setGuidedTourState] = React.useState<GuidedTourState>(() => readGuidedTourState())
  const [guidedTourOpen, setGuidedTourOpen] = React.useState(false)
  const [showGuidedTourCompletion, setShowGuidedTourCompletion] = React.useState(false)
  const [teachPointState, setTeachPointState] = React.useState<TeachPointState>(() => readTeachPointState())
  const [orientationDismissedState, setOrientationDismissedState] = React.useState<OrientationDismissState>(
    () => readOrientationDismissState()
  )
  const [documentVisible, setDocumentVisible] = React.useState<boolean>(() => {
    if (typeof document === "undefined") return true
    return document.visibilityState !== "hidden"
  })
  const iaRollout = React.useMemo(() => resolveWatchlistsIaExperimentRollout(), [])
  const iaExperimentVariant = iaRollout.variant
  const iaExperimentEnabled = iaExperimentVariant === "experimental"
  const previousActiveTabRef = useRef<typeof activeTab | null>(null)

  // Progressive disclosure: 3 primary tabs with inline secondary views
  const [showAllViews, setShowAllViews] = React.useState(readShowAllViews)
  const [secondaryExpanded, setSecondaryExpanded] = React.useState<SecondaryExpandedState>(readSecondaryExpanded)
  const [settingsDrawerOpen, setSettingsDrawerOpen] = React.useState(false)
  const [commandPaletteOpen, setCommandPaletteOpen] = React.useState(false)
  const [shortcutsHelpOpen, setShortcutsHelpOpen] = React.useState(false)
  const [setupWizardOpen, setSetupWizardOpen] = React.useState(false)
  const [watchlistFormOpen, setWatchlistFormOpen] = React.useState(false)
  const [watchlistFormMode, setWatchlistFormMode] = React.useState<WatchlistFormMode>("create")
  const [watchlistFormSaving, setWatchlistFormSaving] = React.useState(false)
  const [watchlistForm, setWatchlistForm] = React.useState<WatchlistFormState>(WATCHLIST_FORM_DEFAULTS)
  const { isConstrained } = useWatchlistsViewport()
  const selectedWatchlistIdRef = useRef<number | null>(selectedWatchlistId)
  const loadWatchlistsRequestRef = useRef(0)
  selectedWatchlistIdRef.current = selectedWatchlistId
  const selectedWatchlist = React.useMemo(
    () =>
      Array.isArray(watchlists)
        ? watchlists.find((watchlist) => watchlist.id === selectedWatchlistId) || null
        : null,
    [selectedWatchlistId, watchlists]
  )

  const loadWatchlists = useCallback(async () => {
    const requestId = loadWatchlistsRequestRef.current + 1
    loadWatchlistsRequestRef.current = requestId
    const isLatestRequest = () => loadWatchlistsRequestRef.current === requestId

    setWatchlistsLoading(true)
    setWatchlistsError(null)
    try {
      const response = await fetchWatchlists({ page: 1, size: 100 })
      if (!isLatestRequest()) return
      const items = Array.isArray(response.items) ? response.items : []
      const nextSelectedWatchlistId = resolvePreferredWatchlistId(items, selectedWatchlistIdRef.current)
      setWatchlists(items, nextSelectedWatchlistId)
    } catch (err) {
      if (!isLatestRequest()) return
      console.error("Failed to load Watchlists:", err)
      setWatchlistsError(t("watchlists:containers.fetchError", "Failed to load Watchlists"))
    } finally {
      if (isLatestRequest()) {
        setWatchlistsLoading(false)
      }
    }
  }, [
    setWatchlists,
    setWatchlistsError,
    setWatchlistsLoading,
    t
  ])

  useEffect(() => {
    void loadWatchlists()
  }, [loadWatchlists])

  const openCreateWatchlistForm = useCallback(() => {
    setSetupWizardOpen(true)
  }, [])

  const openEditWatchlistForm = useCallback(() => {
    setWatchlistFormMode("edit")
    setWatchlistForm(toWatchlistFormState(selectedWatchlist))
    setWatchlistFormOpen(true)
  }, [selectedWatchlist])

  const closeWatchlistForm = useCallback(() => {
    if (watchlistFormSaving) return
    setWatchlistFormOpen(false)
  }, [watchlistFormSaving])

  const saveWatchlistForm = useCallback(async () => {
    const name = watchlistForm.name.trim()
    if (!name) {
      notification.error({
        message: t("watchlists:containers.nameRequired", "Enter a Watchlist name"),
        placement: "bottomRight",
        duration: 5
      })
      return
    }

    const payload = {
      name,
      description: toOptionalText(watchlistForm.description),
      objective: toOptionalText(watchlistForm.objective),
      domain: watchlistForm.domain,
      status: watchlistForm.status,
      priority: watchlistForm.priority,
      tags: toTags(watchlistForm.tagsText)
    }

    setWatchlistFormSaving(true)
    try {
      if (watchlistFormMode === "edit" && selectedWatchlist) {
        const updated = await updateWatchlist(selectedWatchlist.id, payload)
        updateWatchlistInList(updated.id, updated)
        setSelectedWatchlistId(updated.id)
        notification.success({
          message: t("watchlists:containers.updated", "Watchlist updated"),
          placement: "bottomRight",
          duration: 5
        })
      } else {
        const created = await createWatchlist(payload)
        addWatchlist(created)
        setSelectedWatchlistId(created.id)
        notification.success({
          message: t("watchlists:containers.created", "Watchlist created"),
          placement: "bottomRight",
          duration: 5
        })
      }
      setWatchlistFormOpen(false)
    } catch (err) {
      console.error("Failed to save Watchlist:", err)
      notification.error({
        message: t("watchlists:containers.saveError", "Failed to save Watchlist"),
        placement: "bottomRight",
        duration: 5
      })
    } finally {
      setWatchlistFormSaving(false)
    }
  }, [
    addWatchlist,
    notification,
    selectedWatchlist,
    setSelectedWatchlistId,
    t,
    updateWatchlistInList,
    watchlistForm,
    watchlistFormMode
  ])

  const createSetupSources = useCallback(async (
    watchlistId: number,
    sources: WatchlistSourceCreate[]
  ): Promise<number[]> => {
    const scopedSources = sources.map((source) => ({
      ...source,
      watchlist_id: watchlistId
    }))

    if (scopedSources.length === 0) return []
    if (scopedSources.length === 1) {
      const created = await createWatchlistSource(scopedSources[0])
      return [created.id]
    }

    const response = await bulkCreateSources(scopedSources)
    const createdIds = response.items
      .filter((item) => item.status === "created" && typeof item.id === "number")
      .map((item) => item.id as number)

    if (createdIds.length !== scopedSources.length) {
      throw new Error(t("watchlists:setupWizard.errors.sources", "Failed to create all Watchlist sources"))
    }

    return createdIds
  }, [t])

  const createSetupJob = useCallback(async (
    watchlistId: number,
    job: WatchlistJobCreate
  ) => {
    return createWatchlistJob({
      ...job,
      watchlist_id: watchlistId
    })
  }, [])

  const completeSetupWizard = useCallback((result: WatchlistSetupCompleteResult) => {
    addWatchlist(result.watchlist)
    setSelectedWatchlistId(result.watchlist.id)
    setActiveTab(SETUP_DESTINATION_TAB[result.destination])
    setSetupWizardOpen(false)
    notification.success({
      message: t("watchlists:containers.created", "Watchlist created"),
      placement: "bottomRight",
      duration: 5
    })
  }, [addWatchlist, notification, setActiveTab, setSelectedWatchlistId, t])

  const toggleShowAllViews = useCallback(() => {
    setShowAllViews((prev) => {
      const next = !prev
      writeShowAllViews(next)
      return next
    })
  }, [])

  const toggleSecondaryExpanded = useCallback((key: string) => {
    setSecondaryExpanded((prev) => {
      const next = { ...prev, [key]: !prev[key] }
      writeSecondaryExpanded(next)
      return next
    })
  }, [])

  // Navigate to a tab, auto-expanding inline secondary sections in progressive mode
  const navigateToTab = useCallback((key: string) => {
    const isProgressive = !isConstrained && !showAllViews && !iaExperimentEnabled
    if (isProgressive && SECONDARY_IN_PRIMARY[key]) {
      const primaryTab = SECONDARY_IN_PRIMARY[key]
      setActiveTab(primaryTab as typeof activeTab)
      const sectionKey = key === "jobs" ? "monitors" : key === "runs" ? "activity" : "templates"
      setSecondaryExpanded((prev) => {
        const next = { ...prev, [sectionKey]: true }
        writeSecondaryExpanded(next)
        return next
      })
      return
    }
    setActiveTab(key as typeof activeTab)
  }, [isConstrained, setActiveTab, showAllViews, iaExperimentEnabled])

  // Refresh key — incrementing forces tab components to remount and refetch
  const [refreshKey, setRefreshKey] = React.useState(0)
  const triggerRefresh = useCallback(() => setRefreshKey((k) => k + 1), [])

  const exportSourcesFromPalette = useCallback(async () => {
    try {
      const opml = await exportOpml()
      downloadBlob(new Blob([opml], { type: "application/xml" }), `watchlists_sources_${Date.now()}.opml`)
      notification.success({
        message: t("watchlists:sources.exported", "OPML exported"),
        placement: "bottomRight",
        duration: 5
      })
    } catch (err) {
      console.error("Failed to export watchlist feeds from command palette:", err)
      notification.error({
        message: t("watchlists:sources.exportError", "Failed to export OPML"),
        placement: "bottomRight",
        duration: 5
      })
    }
  }, [notification, t])

  const exportRunsFromPalette = useCallback(async () => {
    try {
      const csv = await exportRunsCsv({ scope: "global", include_tallies: true })
      downloadBlob(new Blob([csv], { type: "text/csv;charset=utf-8" }), `watchlists_runs_${Date.now()}.csv`)
      notification.success({
        message: t("watchlists:runs.exported", "Activity CSV exported"),
        placement: "bottomRight",
        duration: 5
      })
    } catch (err) {
      console.error("Failed to export watchlist activity from command palette:", err)
      notification.error({
        message: t("watchlists:runs.exportError", "Failed to export activity CSV"),
        placement: "bottomRight",
        duration: 5
      })
    }
  }, [notification, t])

  const tabHelpLabels = {
    overview: t("watchlists:help.tabs.overview", "Overview guidance"),
    sources: t("watchlists:help.tabs.sources", "Feeds setup"),
    jobs: t("watchlists:help.tabs.jobs", "Monitor scheduling"),
    runs: t("watchlists:help.tabs.runs", "Activity guidance"),
    items: t("watchlists:help.tabs.items", "Updates review"),
    alerts: t("watchlists:help.tabs.alerts", "Alert guidance"),
    outputs: t("watchlists:help.tabs.outputs", "Reports guidance"),
    templates: t("watchlists:help.tabs.templates", "Template authoring"),
    settings: t("watchlists:help.tabs.settings", "Workspace settings")
  } as const

  const taskShortcuts = [
    {
      key: "sources" as const,
      label: t("watchlists:quickActions.sources", "Set up feeds")
    },
    {
      key: "jobs" as const,
      label: t("watchlists:quickActions.jobs", "Configure monitors")
    },
    {
      key: "runs" as const,
      label: t("watchlists:quickActions.runs", "Check activity")
    },
    {
      key: "items" as const,
      label: t("watchlists:quickActions.items", "Review updates")
    },
    {
      key: "alerts" as const,
      label: t("watchlists:quickActions.alerts", "Review alerts")
    },
    {
      key: "outputs" as const,
      label: t("watchlists:quickActions.outputs", "View reports")
    }
  ]
  const repeatUserShortcuts = taskShortcuts.filter((shortcut) =>
    ["runs", "items", "outputs"].includes(shortcut.key)
  )
  const taskViews = [
    {
      key: "collect" as const,
      label: t("watchlists:taskViews.collect", "Collect"),
      hint: t("watchlists:taskViews.collectHint", "Feeds and monitors")
    },
    {
      key: "review" as const,
      label: t("watchlists:taskViews.review", "Review"),
      hint: t("watchlists:taskViews.reviewHint", "Activity and updates")
    },
    {
      key: "briefings" as const,
      label: t("watchlists:taskViews.briefings", "Briefings"),
      hint: t("watchlists:taskViews.briefingsHint", "Reports and templates")
    }
  ]
  const activeTaskView = resolveTaskViewForTab(activeTab)
  const tabOrientation: Record<WatchlistsTabKey, TabOrientation> = {
    overview: {
      title: t("watchlists:orientation.overview.title", "Overview: watchlist health at a glance"),
      description: t(
        "watchlists:orientation.overview.description",
        "Review current health and attention signals, then start by adding or refining feeds."
      ),
      actions: [
        {
          key: "open-feeds",
          label: t("watchlists:orientation.actions.openFeeds", "Open Feeds"),
          target: "sources"
        },
        {
          key: "open-monitors",
          label: t("watchlists:orientation.actions.openMonitors", "Open Monitors"),
          target: "jobs"
        }
      ]
    },
    sources: {
      title: t("watchlists:orientation.sources.title", "Feeds: define what to collect"),
      description: t(
        "watchlists:orientation.sources.description",
        "Add or clean feed inputs here. Next, configure monitors to schedule collection."
      ),
      actions: [
        {
          key: "open-monitors",
          label: t("watchlists:orientation.actions.openMonitors", "Open Monitors"),
          target: "jobs"
        },
        {
          key: "open-activity",
          label: t("watchlists:orientation.actions.openActivity", "Open Activity"),
          target: "runs"
        }
      ]
    },
    jobs: {
      title: t("watchlists:orientation.jobs.title", "Monitors: control schedule and processing"),
      description: t(
        "watchlists:orientation.jobs.description",
        "Monitors run your feed pipeline. Next, check Activity to confirm runs are healthy."
      ),
      actions: [
        {
          key: "open-activity",
          label: t("watchlists:orientation.actions.openActivity", "Open Activity"),
          target: "runs"
        },
        {
          key: "open-articles",
          label: t("watchlists:orientation.actions.openArticles", "Open Updates"),
          target: "items"
        }
      ]
    },
    runs: {
      title: t("watchlists:orientation.runs.title", "Activity: run health and history"),
      description: t(
        "watchlists:orientation.runs.description",
        "Inspect failures and logs here. Next, open Reports to view generated briefings."
      ),
      actions: [
        {
          key: "open-reports",
          label: t("watchlists:orientation.actions.openReports", "Open Reports"),
          target: "outputs"
        },
        {
          key: "open-articles",
          label: t("watchlists:orientation.actions.openArticles", "Open Updates"),
          target: "items"
        }
      ]
    },
    items: {
      title: t("watchlists:orientation.items.title", "Updates: triage captured content"),
      description: t(
        "watchlists:orientation.items.description",
        "Review and prioritize captured updates. Next, tune monitor scope or open reports."
      ),
      actions: [
        {
          key: "open-monitors",
          label: t("watchlists:orientation.actions.openMonitors", "Open Monitors"),
          target: "jobs"
        },
        {
          key: "open-reports",
          label: t("watchlists:orientation.actions.openReports", "Open Reports"),
          target: "outputs"
        }
      ]
    },
    alerts: {
      title: t("watchlists:orientation.alerts.title", "Alerts: review matched content"),
      description: t(
        "watchlists:orientation.alerts.description",
        "Create content alert rules and triage item matches here. Run failures stay in Activity as health issues."
      ),
      actions: [
        {
          key: "open-articles",
          label: t("watchlists:orientation.actions.openArticles", "Open Updates"),
          target: "items"
        },
        {
          key: "open-activity",
          label: t("watchlists:orientation.actions.openActivity", "Open Activity"),
          target: "runs"
        }
      ]
    },
    outputs: {
      title: t("watchlists:orientation.outputs.title", "Reports: generated briefing outputs"),
      description: t(
        "watchlists:orientation.outputs.description",
        "Download, review, or regenerate briefings. Next, edit templates when format needs change."
      ),
      actions: [
        {
          key: "open-templates",
          label: t("watchlists:orientation.actions.openTemplates", "Open Templates"),
          target: "templates"
        },
        {
          key: "open-activity",
          label: t("watchlists:orientation.actions.openActivity", "Open Activity"),
          target: "runs"
        }
      ]
    },
    templates: {
      title: t("watchlists:orientation.templates.title", "Templates: define briefing format"),
      description: t(
        "watchlists:orientation.templates.description",
        "Template changes apply to future output generation. Next, run monitors or open reports."
      ),
      actions: [
        {
          key: "open-monitors",
          label: t("watchlists:orientation.actions.openMonitors", "Open Monitors"),
          target: "jobs"
        },
        {
          key: "open-reports",
          label: t("watchlists:orientation.actions.openReports", "Open Reports"),
          target: "outputs"
        }
      ]
    },
    settings: {
      title: t("watchlists:orientation.settings.title", "Settings: workspace-level watchlists defaults"),
      description: t(
        "watchlists:orientation.settings.description",
        "Adjust watchlists defaults and integration settings. Return to overview to validate health."
      ),
      actions: [
        {
          key: "open-overview",
          label: t("watchlists:orientation.actions.openOverview", "Open Overview"),
          target: "overview"
        },
        {
          key: "open-feeds",
          label: t("watchlists:orientation.actions.openFeeds", "Open Feeds"),
          target: "sources"
        }
      ]
    }
  }
  const activeTabOrientation = tabOrientation[activeTab as WatchlistsTabKey] || tabOrientation.overview
  const activeOrientationKey = (activeTab as WatchlistsTabKey) || "overview"
  const orientationDismissed = Boolean(orientationDismissedState[activeOrientationKey])

  const activeTabHelpHref = WATCHLISTS_TAB_HELP_DOCS[activeTab] || WATCHLISTS_MAIN_DOCS_URL
  const activeTabHelpLabel = tabHelpLabels[activeTab] || t("watchlists:help.docs", "Watchlists docs")

  const guidedTourSteps = [
    {
      tab: "sources" as const,
      title: t("watchlists:guide.steps.sources.title", "1. Add feeds"),
      description: t(
        "watchlists:guide.steps.sources.description",
        "Feeds are inputs for monitors. Add RSS/site feeds before scheduling Activity checks."
      )
    },
    {
      tab: "jobs" as const,
      title: t("watchlists:guide.steps.jobs.title", "2. Create monitors"),
      description: t(
        "watchlists:guide.steps.jobs.description",
        "Monitors define schedule, filters, and template-driven reports, including optional audio."
      )
    },
    {
      tab: "runs" as const,
      title: t("watchlists:guide.steps.runs.title", "3. Check activity"),
      description: t(
        "watchlists:guide.steps.runs.description",
        "Activity shows monitor status, logs, and failures."
      )
    },
    {
      tab: "items" as const,
      title: t("watchlists:guide.steps.items.title", "4. Review updates"),
      description: t(
        "watchlists:guide.steps.items.description",
        "Updates are captured content from successful monitor checks, ready for triage."
      )
    },
    {
      tab: "outputs" as const,
      title: t("watchlists:guide.steps.outputs.title", "5. Deliver reports"),
      description: t(
        "watchlists:guide.steps.outputs.description",
        "Reports contain generated briefings from monitor templates; regenerate with different template or audio settings."
      )
    }
  ]
  const guidedTourStep = guidedTourSteps[clampTourStep(guidedTourState.step)]
  const activeTeachPoint = React.useMemo(() => {
    if (activeTab === "jobs" && !teachPointState.jobsCronFilters) {
      return {
        key: "jobsCronFilters" as TeachPointKey,
        title: t("watchlists:teachPoints.jobs.title", "Monitor setup tip"),
        description: t(
          "watchlists:teachPoints.jobs.description",
          "Start with schedule presets first. Use cron and advanced filters only after your first successful Activity check."
        ),
        actionLabel: t("watchlists:teachPoints.jobs.action", "Open Templates"),
        actionTarget: "templates" as WatchlistsTabKey
      }
    }
    if (activeTab === "templates" && !teachPointState.templatesAuthoring) {
      return {
        key: "templatesAuthoring" as TeachPointKey,
        title: t("watchlists:teachPoints.templates.title", "Template setup tip"),
        description: t(
          "watchlists:teachPoints.templates.description",
          "Start from a preset template, preview changes, then regenerate reports to compare text and audio results."
        ),
        actionLabel: t("watchlists:teachPoints.templates.action", "Open Reports"),
        actionTarget: "outputs" as WatchlistsTabKey
      }
    }
    return null
  }, [activeTab, t, teachPointState.jobsCronFilters, teachPointState.templatesAuthoring])

  const dismissTeachPoint = useCallback((key: TeachPointKey) => {
    setTeachPointState((previous) => ({
      ...previous,
      [key]: true
    }))
  }, [])

  const dismissOrientationForActiveTab = useCallback(() => {
    const orientationKey = (activeTab as WatchlistsTabKey) || "overview"
    setOrientationDismissedState((previous) => ({
      ...previous,
      [orientationKey]: true
    }))
  }, [activeTab])

  const restoreOrientationForActiveTab = useCallback(() => {
    const orientationKey = (activeTab as WatchlistsTabKey) || "overview"
    setOrientationDismissedState((previous) => ({
      ...previous,
      [orientationKey]: false
    }))
  }, [activeTab])

  useEffect(() => {
    writeGuidedTourState(guidedTourState)
  }, [guidedTourState])

  useEffect(() => {
    writeTeachPointState(teachPointState)
  }, [teachPointState])

  useEffect(() => {
    writeOrientationDismissState(orientationDismissedState)
  }, [orientationDismissedState])

  useEffect(() => {
    if (typeof document === "undefined") return
    const handleVisibilityChange = () => {
      setDocumentVisible(document.visibilityState !== "hidden")
    }
    document.addEventListener("visibilitychange", handleVisibilityChange)
    return () => {
      document.removeEventListener("visibilitychange", handleVisibilityChange)
    }
  }, [])

  const startGuidedTour = useCallback(() => {
    const nextState: GuidedTourState = { status: "in_progress", step: 0 }
    setGuidedTourState(nextState)
    setGuidedTourOpen(true)
    setShowGuidedTourCompletion(false)
    setActiveTab(GUIDED_TOUR_TABS[0])
    void trackWatchlistsOnboardingTelemetry({ type: "guided_tour_started" })
    void trackWatchlistsOnboardingTelemetry({ type: "guided_tour_step_viewed", step: 1 })
  }, [setActiveTab])

  const resumeGuidedTour = useCallback(() => {
    const step = clampTourStep(guidedTourState.step)
    setGuidedTourOpen(true)
    setActiveTab(GUIDED_TOUR_TABS[step])
    const stepNumber = toGuidedTourStep(step)
    void trackWatchlistsOnboardingTelemetry({ type: "guided_tour_resumed", step: stepNumber })
    void trackWatchlistsOnboardingTelemetry({ type: "guided_tour_step_viewed", step: stepNumber })
  }, [guidedTourState.step, setActiveTab])

  const handleSkipGuidedTour = useCallback(() => {
    setGuidedTourState((previous) => ({
      ...previous,
      status: "dismissed"
    }))
    setGuidedTourOpen(false)
    void trackWatchlistsOnboardingTelemetry({
      type: "guided_tour_dismissed",
      step: toGuidedTourStep(guidedTourState.step)
    })
  }, [guidedTourState.step])

  const handleGuidedTourBack = useCallback(() => {
    const nextStep = clampTourStep(guidedTourState.step - 1)
    setGuidedTourState({ status: "in_progress", step: nextStep })
    setActiveTab(GUIDED_TOUR_TABS[nextStep])
  }, [guidedTourState.step, setActiveTab])

  const handleGuidedTourNext = useCallback(() => {
    if (guidedTourState.step >= GUIDED_TOUR_LAST_STEP) {
      setGuidedTourState({ status: "completed", step: GUIDED_TOUR_LAST_STEP })
      setGuidedTourOpen(false)
      setShowGuidedTourCompletion(true)
      void trackWatchlistsOnboardingTelemetry({ type: "guided_tour_completed" })
      return
    }
    const nextStep = clampTourStep(guidedTourState.step + 1)
    setGuidedTourState({ status: "in_progress", step: nextStep })
    setActiveTab(GUIDED_TOUR_TABS[nextStep])
    void trackWatchlistsOnboardingTelemetry({
      type: "guided_tour_step_viewed",
      step: toGuidedTourStep(nextStep)
    })
  }, [guidedTourState.step, setActiveTab])

  // Reset store on unmount — use ref to avoid re-firing if selector returns new reference
  const resetStoreRef = useRef(resetStore)
  resetStoreRef.current = resetStore
  useEffect(() => {
    return () => {
      resetStoreRef.current()
    }
  }, [])

  // Handle URL params for progressive disclosure features
  useEffect(() => {
    if (typeof window === "undefined") return
    const params = new URLSearchParams(window.location.search)

    // ?view=all → force show all views
    if (params.get("view") === "all") {
      setShowAllViews(true)
      writeShowAllViews(true)
    }

    // ?settings=open → open settings drawer
    if (params.get("settings") === "open") {
      setSettingsDrawerOpen(true)
    }

    // ?expand=monitors|activity|templates → expand inline secondary section
    const expandParam = params.get("expand")
    if (expandParam) {
      setSecondaryExpanded((prev) => {
        const next = { ...prev, [expandParam]: true }
        writeSecondaryExpanded(next)
        return next
      })
    }
  }, []) // Run once on mount

  // Command palette commands — stabilize actions object to avoid re-creating on every render
  const commandPaletteActions = useMemo(() => ({
    setActiveTab: navigateToTab,
    openSourceForm: () => openSourceForm(),
    openJobForm: () => openJobForm(),
    openSettings: () => setSettingsDrawerOpen(true),
    refreshCurrentView: triggerRefresh,
    startGuidedTour,
    createPipeline: () => setSetupWizardOpen(true),
    exportSources: exportSourcesFromPalette,
    exportRuns: exportRunsFromPalette
  }), [
    navigateToTab,
    openSourceForm,
    openJobForm,
    triggerRefresh,
    startGuidedTour,
    exportSourcesFromPalette,
    exportRunsFromPalette
  ])
  const commandPaletteCommands = useWatchlistsCommands(commandPaletteActions)

  // Keyboard shortcuts — memoize actions to avoid event listener churn
  const keyboardShortcutActions = useMemo(() => ({
    onOpenCommandPalette: () => setCommandPaletteOpen(true),
    onSwitchTab: (index: number) => {
      if (index >= 0 && index < PROGRESSIVE_PRIMARY_TABS.length) {
        navigateToTab(PROGRESSIVE_PRIMARY_TABS[index])
      }
    },
    onNewEntity: () => {
      navigateToTab("sources")
      openSourceForm()
    },
    onRefresh: triggerRefresh,
    onFocusSearch: () => {
      const searchInput = document.querySelector<HTMLInputElement>(
        '[data-testid*="search"] input, .watchlists-tabs input[type="text"]'
      )
      if (searchInput) searchInput.focus()
    },
    onShowHelp: () => setShortcutsHelpOpen(true)
  }), [navigateToTab, openSourceForm, triggerRefresh])
  useWatchlistsKeyboardShortcuts(keyboardShortcutActions, isOnline)

  const openRunFromNotification = useCallback((runId: number, key: string) => {
    notification.destroy(key)
    navigateToTab("runs")
    openRunDetail(runId)
  }, [notification, openRunDetail, navigateToTab])

  const openRunsTabFromNotification = useCallback((key: string) => {
    notification.destroy(key)
    navigateToTab("runs")
  }, [notification, navigateToTab])

  const runNotificationsPollPlan = React.useMemo(() => {
    const basePollMs = resolveRunNotificationsPollMs()
    return resolveRunNotificationsPollPlan({
      isOnline,
      activeTab,
      runsPollingActive,
      documentVisible,
      baseIntervalMs: basePollMs,
      minIntervalMs: RUN_NOTIFICATIONS_MIN_POLL_MS,
      defaultPageSize: RUN_NOTIFICATIONS_PAGE_SIZE,
      reducedPageSize: RUN_NOTIFICATIONS_REDUCED_PAGE_SIZE,
      backgroundIntervalMs: RUN_NOTIFICATIONS_BACKGROUND_POLL_MS,
      runsTabIntervalMs: RUN_NOTIFICATIONS_RUNS_TAB_POLL_MS
    })
  }, [activeTab, documentVisible, isOnline, runsPollingActive])

  const showRunNotification = useCallback((run: WatchlistRun, kind: "completed" | "failed", hint?: string | null) => {
    const key = `watchlists-run-${run.id}-${run.status}`
    const onOpenRun = () => openRunFromNotification(run.id, key)
    const messageText = kind === "failed"
      ? t("watchlists:notifications.runFailedTitle", "Run failed")
      : t("watchlists:notifications.runCompletedTitle", "Run completed")
    const descriptionText = kind === "failed"
      ? t(
          "watchlists:notifications.runFailedDescription",
          "Run #{{id}} failed. {{hint}}",
          {
            id: run.id,
            hint: hint || getRunFailureHint(run.error_msg, t) || ""
          }
        )
      : t("watchlists:notifications.runCompletedDescription", "Run #{{id}} completed successfully.", {
          id: run.id
        })

    const onRetryRun = async () => {
      if (!run.job_id) return
      try {
        notification.destroy(key)
        await triggerWatchlistRun(run.job_id)
        notification.success({
          message: t("watchlists:notifications.retryTriggered", "Retry started"),
          placement: "bottomRight",
          duration: 5
        })
      } catch {
        notification.error({
          message: t("watchlists:notifications.retryFailed", "Failed to retry run"),
          placement: "bottomRight",
          duration: 5
        })
      }
    }

    notification[kind === "failed" ? "error" : "success"]({
      key,
      message: messageText,
      description: descriptionText,
      placement: "bottomRight",
      duration: kind === "failed" ? 0 : 8,
      onClick: onOpenRun,
      btn: (
        <div className="flex items-center gap-2">
          <Button
            size="small"
            type="link"
            onClick={(event) => {
              event.preventDefault()
              event.stopPropagation()
              onOpenRun()
            }}
          >
            {t("watchlists:notifications.viewRun", "View run")}
          </Button>
          {kind === "failed" && run.job_id && (
            <Button
              size="small"
              type="link"
              onClick={(event) => {
                event.preventDefault()
                event.stopPropagation()
                void onRetryRun()
              }}
            >
              {t("watchlists:errors.retry", "Retry")}
            </Button>
          )}
        </div>
      )
    })
  }, [notification, openRunFromNotification, t])

  const showGroupedRunNotification = useCallback(
    (group: { kind: "completed" | "failed" | "stalled"; count: number; deepLinkRunId: number; hint: string | null }) => {
      const key = `watchlists-run-group-${group.kind}-${group.deepLinkRunId}-${group.count}`
      const onOpenPrimaryRun = () => openRunFromNotification(group.deepLinkRunId, key)
      const onOpenActivity = () => openRunsTabFromNotification(key)

      const mode: "success" | "warning" | "error" =
        group.kind === "failed"
          ? "error"
          : group.kind === "stalled"
            ? "warning"
            : "success"
      const messageText =
        group.kind === "failed"
          ? t("watchlists:notifications.runFailedGroupedTitle", "Multiple runs failed")
          : group.kind === "stalled"
            ? t("watchlists:notifications.runStalledTitle", "Run appears stalled")
            : t("watchlists:notifications.runCompletedGroupedTitle", "Runs completed")
      const descriptionText =
        group.kind === "failed"
          ? t("watchlists:notifications.runFailedGroupedDescription", "{{count}} runs failed. {{hint}}", {
              count: group.count,
              hint: group.hint || t(
                "watchlists:notifications.failureHints.unknownEmpty",
                "Open run details to inspect logs and retry."
              )
            })
          : group.kind === "stalled"
            ? t("watchlists:notifications.runStalledDescription", "{{count}} runs appear stalled. {{hint}}", {
                count: group.count,
                hint: group.hint || t(
                  "watchlists:notifications.failureHints.stalled",
                  "Open Activity to inspect logs, then cancel or retry."
                )
              })
            : t("watchlists:notifications.runCompletedGroupedDescription", "{{count}} runs completed successfully.", {
                count: group.count
              })

      notification[mode]({
        key,
        message: messageText,
        description: descriptionText,
        placement: "bottomRight",
        duration: mode === "success" ? 8 : 0,
        onClick: onOpenPrimaryRun,
        btn: (
          <div className="flex items-center gap-2">
            <Button
              size="small"
              type="link"
              onClick={(event) => {
                event.preventDefault()
                event.stopPropagation()
                onOpenPrimaryRun()
              }}
            >
              {t("watchlists:notifications.viewRun", "View run")}
            </Button>
            <Button
              size="small"
              type="link"
              onClick={(event) => {
                event.preventDefault()
                event.stopPropagation()
                onOpenActivity()
              }}
            >
              {t("watchlists:notifications.openActivity", "Open Activity")}
            </Button>
          </div>
        )
      })
    },
    [notification, openRunFromNotification, openRunsTabFromNotification, t]
  )

  const pollRunNotifications = useCallback(async () => {
    if (selectedWatchlistId == null) return
    if (runNotificationsPollingInFlightRef.current) return
    runNotificationsPollingInFlightRef.current = true
    try {
      const response = await fetchWatchlistRuns({
        watchlist_id: selectedWatchlistId ?? undefined,
        page: 1,
        size: runNotificationsPollPlan.pageSize
      })
      const nextRuns = Array.isArray(response.items) ? response.items : []
      const firstSuccessfulRun = nextRuns.find((run) =>
        SUCCESSFUL_RUN_STATUSES.has(String(run.status || "").toLowerCase())
      )
      if (firstSuccessfulRun) {
        void trackWatchlistsOnboardingTelemetry({
          type: "first_run_succeeded",
          runId: firstSuccessfulRun.id
        })
      }
      const previousStatusMap = runStatusRef.current
      const nextStatusMap = new Map<number, string>()
      const initialized = initializedRunPollingRef.current
      const nowMs = Date.now()
      const runById = new Map<number, WatchlistRun>()
      const candidateEvents: Array<{
        eventKey: string
        kind: "completed" | "failed" | "stalled"
        runId: number
        hint?: string | null
      }> = []

      nextRuns.forEach((run) => {
        runById.set(run.id, run)
        nextStatusMap.set(run.id, String(run.status || ""))
        const previousStatus = previousStatusMap.get(run.id)

        const transition = resolveRunTransitionNotification(previousStatus, run, t)
        if (initialized && transition) {
          candidateEvents.push({
            eventKey: buildRunStateNotificationKey(run.id, run.status),
            kind: transition.kind,
            runId: run.id,
            hint: transition.hint || null
          })
        }

        if (
          initialized &&
          !previousStatus &&
          shouldNotifyNewTerminalRun(run, sessionStartedAtMsRef.current)
        ) {
          const status = String(run.status || "").toLowerCase()
          const kind = status === "failed" ? "failed" : "completed"
          const hint = kind === "failed" ? getRunFailureHint(run.error_msg, t) : null
          candidateEvents.push({
            eventKey: buildRunStateNotificationKey(run.id, run.status),
            kind,
            runId: run.id,
            hint
          })
        }

        if (initialized) {
          const stalled = resolveStalledRunNotification(
            run,
            nowMs,
            RUN_STALLED_THRESHOLD_MS,
            t
          )
          if (stalled) {
            candidateEvents.push(stalled)
          }
        }
      })

      const freshEvents = dedupeRunNotificationEvents(
        candidateEvents,
        notifiedRunStatesRef.current
      )
      const groupedEvents = groupRunNotificationEvents(freshEvents).filter((group) =>
        runNotificationsPollPlan.suppressCompleted ? group.kind !== "completed" : true
      )

      groupedEvents.forEach((group) => {
        if (group.count === 1 && group.kind !== "stalled") {
          const run = runById.get(group.deepLinkRunId)
          if (!run) return
          const event = freshEvents.find((entry) => entry.eventKey === group.eventKeys[0])
          showRunNotification(run, group.kind, event?.hint || group.hint)
          return
        }
        showGroupedRunNotification(group)
      })

      runStatusRef.current = nextStatusMap
      initializedRunPollingRef.current = true
    } catch (err) {
      console.debug("Watchlists run notification polling failed:", err)
    } finally {
      runNotificationsPollingInFlightRef.current = false
    }
  }, [
    runNotificationsPollPlan.pageSize,
    runNotificationsPollPlan.suppressCompleted,
    selectedWatchlistId,
    showGroupedRunNotification,
    showRunNotification,
    t
  ])

  useEffect(() => {
    if (!runNotificationsPollPlan.enabled) {
      if (runNotificationsTimerRef.current) {
        clearInterval(runNotificationsTimerRef.current)
        runNotificationsTimerRef.current = null
      }
      return
    }
    const pollIntervalMs = runNotificationsPollPlan.intervalMs
    void pollRunNotifications()
    runNotificationsTimerRef.current = setInterval(() => {
      void pollRunNotifications()
    }, pollIntervalMs)
    return () => {
      if (runNotificationsTimerRef.current) {
        clearInterval(runNotificationsTimerRef.current)
        runNotificationsTimerRef.current = null
      }
      runNotificationsPollingInFlightRef.current = false
    }
  }, [pollRunNotifications, runNotificationsPollPlan.enabled, runNotificationsPollPlan.intervalMs])

  const overviewBadges = overviewHealth?.tabBadges || {
    sources: 0,
    runs: 0,
    outputs: 0
  }
  const tabAttentionBadge = (count: number): React.ReactNode =>
    count > 0 ? (
      <span
        className="inline-flex min-w-5 items-center justify-center rounded-full bg-red-500 px-1.5 text-[10px] font-semibold leading-4 text-white"
        aria-label={t("watchlists:tabs.attentionBadgeAria", "{{count}} attention items", { count })}
      >
        {count > 99 ? "99+" : count}
      </span>
    ) : null

  const tabPanelFallback = (
    <div className="py-6 text-sm text-text-muted" data-testid="watchlists-tab-loading" />
  )

  const renderWatchlistsTab = useCallback((tab: WatchlistsTabKey): React.ReactNode => {
    switch (tab) {
      case "overview":
        return (
          <Suspense fallback={tabPanelFallback}>
            <OverviewTab />
          </Suspense>
        )
      case "sources":
        return (
          <Suspense fallback={tabPanelFallback}>
            <SourcesTab />
          </Suspense>
        )
      case "jobs":
        return (
          <Suspense fallback={tabPanelFallback}>
            <JobsTab />
          </Suspense>
        )
      case "runs":
        return (
          <Suspense fallback={tabPanelFallback}>
            <RunsTab />
          </Suspense>
        )
      case "items":
        return (
          <Suspense fallback={tabPanelFallback}>
            <ItemsTab />
          </Suspense>
        )
      case "alerts":
        return (
          <Suspense fallback={tabPanelFallback}>
            <AlertsTab />
          </Suspense>
        )
      case "outputs":
        return (
          <Suspense fallback={tabPanelFallback}>
            <OutputsTab />
          </Suspense>
        )
      case "templates":
        return (
          <Suspense fallback={tabPanelFallback}>
            <TemplatesTab />
          </Suspense>
        )
      case "settings":
        return (
          <Suspense fallback={tabPanelFallback}>
            <SettingsTab />
          </Suspense>
        )
      default:
        return null
    }
  }, [])

  // Full 8-tab items (used in "show all views" mode)
  const allTabItems: TabsProps["items"] = [
    {
      key: "overview",
      label: (
        <span className="flex items-center gap-2">
          <LayoutDashboard className="h-4 w-4" />
          {t("watchlists:tabs.overview", "Overview")}
        </span>
      ),
      children: renderWatchlistsTab("overview")
    },
    {
      key: "sources",
      label: (
        <Tooltip title={t("watchlists:tabs.sourcesTooltip", "RSS feeds, websites, or forums you want to monitor")}>
          <span className="flex items-center gap-2">
            <Rss className="h-4 w-4" />
            {t("watchlists:tabs.sources", "Feeds")}
            {tabAttentionBadge(overviewBadges.sources)}
          </span>
        </Tooltip>
      ),
      children: renderWatchlistsTab("sources")
    },
    {
      key: "jobs",
      label: (
        <Tooltip title={t("watchlists:tabs.jobsTooltip", "Scheduled tasks that check your sources for new content")}>
          <span className="flex items-center gap-2">
            <CalendarClock className="h-4 w-4" />
            {t("watchlists:tabs.jobs", "Monitors")}
          </span>
        </Tooltip>
      ),
      children: renderWatchlistsTab("jobs")
    },
    {
      key: "runs",
      label: (
        <Tooltip title={t("watchlists:tabs.runsTooltip", "Individual execution records of your monitors")}>
          <span className="flex items-center gap-2">
            <Play className="h-4 w-4" />
            {t("watchlists:tabs.runs", "Activity")}
            {tabAttentionBadge(overviewBadges.runs)}
          </span>
        </Tooltip>
      ),
      children: renderWatchlistsTab("runs")
    },
    {
      key: "items",
      label: (
        <Tooltip title={t("watchlists:tabs.itemsTooltip", "Updates collected from your sources")}>
          <span className="flex items-center gap-2">
            <Newspaper className="h-4 w-4" />
            {t("watchlists:tabs.items", "Updates")}
          </span>
        </Tooltip>
      ),
      children: renderWatchlistsTab("items")
    },
    {
      key: "alerts",
      label: (
        <Tooltip title={t("watchlists:tabs.alertsTooltip", "Content matches from your Watchlist alert rules")}>
          <span className="flex items-center gap-2">
            <BellRing className="h-4 w-4" />
            {t("watchlists:tabs.alerts", "Alerts")}
          </span>
        </Tooltip>
      ),
      children: renderWatchlistsTab("alerts")
    },
    {
      key: "outputs",
      label: (
        <span className="flex items-center gap-2">
          <FileOutput className="h-4 w-4" />
          {t("watchlists:tabs.outputs", "Reports")}
          {tabAttentionBadge(overviewBadges.outputs)}
        </span>
      ),
      children: renderWatchlistsTab("outputs")
    },
    {
      key: "templates",
      label: (
        <span className="flex items-center gap-2">
          <FileText className="h-4 w-4" />
          {t("watchlists:tabs.templates", "Templates")}
        </span>
      ),
      children: renderWatchlistsTab("templates")
    },
    {
      key: "settings",
      label: (
        <span className="flex items-center gap-2">
          <Settings className="h-4 w-4" />
          {t("watchlists:tabs.settings", "Settings")}
        </span>
      ),
      children: renderWatchlistsTab("settings")
    }
  ]

  // Progressive disclosure: 3 primary tabs with inline secondary views
  const progressiveTabItems: TabsProps["items"] = [
    {
      key: "sources",
      label: (
        <Tooltip title={t("watchlists:tabs.sourcesTooltip", "RSS feeds, websites, or forums you want to monitor")}>
          <span className="flex items-center gap-2">
            <Rss className="h-4 w-4" />
            {t("watchlists:tabs.sources", "Feeds")}
            {tabAttentionBadge(overviewBadges.sources)}
          </span>
        </Tooltip>
      ),
      children: (
        <>
          {renderWatchlistsTab("sources")}
          <InlineSecondarySection
            sectionKey="monitors"
            title={t("watchlists:tabs.jobs", "Monitors")}
            expanded={Boolean(secondaryExpanded.monitors)}
            onToggle={toggleSecondaryExpanded}
          >
            {renderWatchlistsTab("jobs")}
          </InlineSecondarySection>
        </>
      )
    },
    {
      key: "alerts",
      label: (
        <Tooltip title={t("watchlists:tabs.alertsTooltip", "Content matches from your Watchlist alert rules")}>
          <span className="flex items-center gap-2">
            <BellRing className="h-4 w-4" />
            {t("watchlists:tabs.alerts", "Alerts")}
          </span>
        </Tooltip>
      ),
      children: renderWatchlistsTab("alerts")
    },
    {
      key: "items",
      label: (
        <Tooltip title={t("watchlists:tabs.itemsTooltip", "Updates collected from your sources")}>
          <span className="flex items-center gap-2">
            <Newspaper className="h-4 w-4" />
            {t("watchlists:tabs.items", "Updates")}
          </span>
        </Tooltip>
      ),
      children: (
        <>
          <InlineSecondarySection
            sectionKey="activity"
            title={t("watchlists:tabs.runs", "Recent Activity")}
            count={overviewBadges.runs}
            expanded={Boolean(secondaryExpanded.activity)}
            onToggle={toggleSecondaryExpanded}
          >
            {renderWatchlistsTab("runs")}
          </InlineSecondarySection>
          {renderWatchlistsTab("items")}
        </>
      )
    },
    {
      key: "outputs",
      label: (
        <Tooltip title={t("watchlists:tabs.outputsTooltip", "Generated reports and summaries from your sources")}>
          <span className="flex items-center gap-2">
            <FileOutput className="h-4 w-4" />
            {t("watchlists:tabs.outputs", "Reports")}
            {tabAttentionBadge(overviewBadges.outputs)}
          </span>
        </Tooltip>
      ),
      children: (
        <>
          {renderWatchlistsTab("outputs")}
          <InlineSecondarySection
            sectionKey="templates"
            title={t("watchlists:tabs.templates", "Templates")}
            expanded={Boolean(secondaryExpanded.templates)}
            onToggle={toggleSecondaryExpanded}
          >
            {renderWatchlistsTab("templates")}
          </InlineSecondarySection>
        </>
      )
    }
  ]

  // Resolve which tab set to render based on mode
  const useProgressiveLayout = !showAllViews && !iaExperimentEnabled
  const renderedTabItems: TabsProps["items"] = useProgressiveLayout
    ? progressiveTabItems
    : iaExperimentEnabled
      ? (() => {
          const reducedIaPrimaryTabKeys = ["overview", "sources", "alerts", "items", "outputs", "settings"] as const
          const primarySet = new Set<string>(reducedIaPrimaryTabKeys)
          const primaryItems = allTabItems.filter((item) => item?.key && primarySet.has(String(item.key)))
          if (primarySet.has(activeTab)) return primaryItems
          const activeSecondaryItem = allTabItems.find((item) => String(item?.key) === activeTab)
          if (!activeSecondaryItem) return primaryItems
          return [...primaryItems, activeSecondaryItem]
        })()
      : allTabItems

  const constrainedActiveTabItem =
    allTabItems.find((item) => String(item?.key) === activeTab) ||
    renderedTabItems?.find((item) => String(item?.key) === activeTab)

  // Resolve active tab for the tab bar (in progressive mode, secondary tabs map to their parent)
  const resolvedActiveTab = useProgressiveLayout && SECONDARY_IN_PRIMARY[activeTab]
    ? SECONDARY_IN_PRIMARY[activeTab]
    : activeTab

  const reducedIaSecondaryTabKeys = ["jobs", "runs", "templates"] as const
  const reducedIaSecondaryButtons = reducedIaSecondaryTabKeys.map((key) => ({
    key,
    label:
      key === "jobs"
        ? t("watchlists:tabs.jobs", "Monitors")
        : key === "runs"
          ? t("watchlists:tabs.runs", "Activity")
          : t("watchlists:tabs.templates", "Templates")
  }))
  const constrainedNavigationGroups = useMemo<WatchlistsMobileNavigationGroup[]>(
    () => [
      {
        key: "overview",
        label: t("watchlists:mobileNav.overviewGroup", "Overview"),
        items: [
          {
            key: "overview",
            label: t("watchlists:tabs.overview", "Overview"),
            description: t("watchlists:mobileNav.overviewDescription", "Intent, health, and next actions")
          }
        ]
      },
      {
        key: "collect",
        label: t("watchlists:mobileNav.collectGroup", "Collect"),
        items: [
          {
            key: "sources",
            label: t("watchlists:tabs.sources", "Feeds"),
            description: t("watchlists:mobileNav.sourcesDescription", "Sources, groups, tags, and imports"),
            count: overviewBadges.sources
          },
          {
            key: "jobs",
            label: t("watchlists:tabs.jobs", "Monitors"),
            description: t("watchlists:mobileNav.jobsDescription", "Schedules, scope, filters, and run now")
          }
        ]
      },
      {
        key: "review",
        label: t("watchlists:mobileNav.reviewGroup", "Review"),
        items: [
          {
            key: "alerts",
            label: t("watchlists:tabs.alerts", "Alerts"),
            description: t("watchlists:mobileNav.alertsDescription", "Content matches and alert rules")
          },
          {
            key: "items",
            label: t("watchlists:tabs.items", "Updates"),
            description: t("watchlists:mobileNav.itemsDescription", "Triage, saved views, and report queue")
          },
          {
            key: "runs",
            label: t("watchlists:tabs.runs", "Activity"),
            description: t("watchlists:mobileNav.runsDescription", "Run history, health, and details"),
            count: overviewBadges.runs
          }
        ]
      },
      {
        key: "reports",
        label: t("watchlists:mobileNav.reportsGroup", "Reports"),
        items: [
          {
            key: "outputs",
            label: t("watchlists:tabs.outputs", "Reports"),
            description: t("watchlists:mobileNav.outputsDescription", "Generated reports, evidence, and downloads"),
            count: overviewBadges.outputs
          },
          {
            key: "templates",
            label: t("watchlists:tabs.templates", "Templates"),
            description: t("watchlists:mobileNav.templatesDescription", "Report template authoring and preview")
          }
        ]
      },
      {
        key: "settings",
        label: t("watchlists:mobileNav.settingsGroup", "Settings"),
        items: [
          {
            key: "settings",
            label: t("watchlists:tabs.settings", "Settings"),
            description: t("watchlists:mobileNav.settingsDescription", "Lifecycle, defaults, and subscriptions")
          }
        ]
      }
    ],
    [overviewBadges.outputs, overviewBadges.runs, overviewBadges.sources, t]
  )
  const watchlistOptions = React.useMemo(
    () =>
      (Array.isArray(watchlists) ? watchlists : []).map((watchlist) => ({
        value: watchlist.id,
        label: watchlist.name
      })),
    [watchlists]
  )
  const watchlistViewsAvailable = Boolean(selectedWatchlist && selectedWatchlistId != null)
  const renderWatchlistContainerShell = (): React.ReactNode => {
    if (watchlistsLoading && (!Array.isArray(watchlists) || watchlists.length === 0)) {
      return (
        <DesignSystemAlert
          variant="info"
          className="mb-4"
          data-testid="watchlists-container-loading"
          title={t("watchlists:containers.loading", "Loading Watchlists")}
        >
          {t(
            "watchlists:containers.loadingDescription",
            "Preparing your monitoring workspaces."
          )}
        </DesignSystemAlert>
      )
    }

    if (watchlistsError && (!Array.isArray(watchlists) || watchlists.length === 0)) {
      return (
        <DesignSystemAlert
          variant="error"
          className="mb-4"
          data-testid="watchlists-container-error"
          title={t("watchlists:containers.errorTitle", "Watchlists unavailable")}
          action={{
            label: t("watchlists:errors.retry", "Retry"),
            onClick: () => void loadWatchlists()
          }}
        >
          {watchlistsError}
        </DesignSystemAlert>
      )
    }

    if (!selectedWatchlist) {
      return (
        <div
          className="mb-4 rounded-lg border border-dashed border-border bg-surface p-5"
          data-testid="watchlists-container-empty"
        >
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <div className="text-base font-semibold text-text">
                {t("watchlists:containers.emptyTitle", "Create a Watchlist")}
              </div>
              <div className="mt-1 text-sm text-text-muted">
                {t(
                  "watchlists:containers.emptyDescription",
                  "Use a Watchlist as the workspace for feeds, monitors, activity, updates, and reports."
                )}
              </div>
            </div>
            <Button
              type="primary"
              icon={<Plus className="h-4 w-4" />}
              data-testid="watchlists-create-container"
              onClick={openCreateWatchlistForm}
            >
              {t("watchlists:containers.create", "Create Watchlist")}
            </Button>
          </div>
        </div>
      )
    }

    return (
      <section
        className="mb-4 rounded-lg border border-border bg-surface p-4"
        data-testid="watchlists-container-shell"
        aria-label={t("watchlists:containers.shellAria", "Selected Watchlist")}
      >
        <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div className="min-w-0 flex-1 space-y-3">
            <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
              <span className="text-xs font-medium uppercase tracking-wide text-text-muted">
                {t("watchlists:containers.selectorLabel", "Watchlist")}
              </span>
              <Select
                aria-label={t("watchlists:containers.selectorLabel", "Watchlist")}
                className="min-w-64 max-w-full sm:w-80"
                value={selectedWatchlistId ?? undefined}
                onChange={(value) => setSelectedWatchlistId(Number(value))}
                options={watchlistOptions}
                data-testid="watchlists-container-selector"
              />
            </div>
            <div>
              <div className="flex flex-wrap items-center gap-2">
                <h2 className="m-0 text-lg font-semibold text-text">
                  {selectedWatchlist.name}
                </h2>
                <Tag>{WATCHLIST_DOMAIN_LABELS[selectedWatchlist.domain] || selectedWatchlist.domain}</Tag>
                <Tag>{WATCHLIST_PRIORITY_LABELS[selectedWatchlist.priority] || selectedWatchlist.priority}</Tag>
                <Tag>{WATCHLIST_STATUS_LABELS[selectedWatchlist.status] || selectedWatchlist.status}</Tag>
              </div>
              {selectedWatchlist.objective && (
                <p className="mt-1 max-w-4xl text-sm text-text-muted">
                  {selectedWatchlist.objective}
                </p>
              )}
              {selectedWatchlist.description && !selectedWatchlist.objective && (
                <p className="mt-1 max-w-4xl text-sm text-text-muted">
                  {selectedWatchlist.description}
                </p>
              )}
              {Array.isArray(selectedWatchlist.tags) && selectedWatchlist.tags.length > 0 && (
                <div className="mt-2 flex flex-wrap gap-1">
                  {selectedWatchlist.tags.slice(0, 6).map((tag) => (
                    <Tag key={tag}>{tag}</Tag>
                  ))}
                </div>
              )}
            </div>
          </div>
          <div className="flex shrink-0 flex-wrap gap-2">
            <Button
              icon={<Plus className="h-4 w-4" />}
              data-testid="watchlists-create-container"
              onClick={openCreateWatchlistForm}
            >
              {t("watchlists:containers.create", "Create Watchlist")}
            </Button>
            <Button
              icon={<Pencil className="h-4 w-4" />}
              data-testid="watchlists-edit-container"
              onClick={openEditWatchlistForm}
            >
              {t("common:edit", "Edit")}
            </Button>
          </div>
        </div>
      </section>
    )
  }

  useEffect(() => {
    trackWatchlistsIaExperimentTransition(
      previousActiveTabRef.current,
      activeTab,
      iaExperimentVariant
    )
    previousActiveTabRef.current = activeTab
  }, [activeTab, iaExperimentVariant])

  useEffect(() => {
    if (typeof window === "undefined") return
    const handlePageHide = () => {
      flushWatchlistsIaExperimentSession(activeTab, iaExperimentVariant)
    }
    window.addEventListener("pagehide", handlePageHide)
    return () => {
      window.removeEventListener("pagehide", handlePageHide)
    }
  }, [activeTab, iaExperimentVariant])

  return (
    <WorkspaceConnectionGate
      featureName={t("watchlists:title", "Watchlists")}
      setupDescription={t(
        "watchlists:setupRequired",
        "Watchlists depends on your connected tldw server to monitor feeds, run scheduled jobs, and save outputs."
      )}
      maxWidthClassName="max-w-[1920px]"
    >
      <PageShell className="min-w-0 w-screen max-w-[100vw] overflow-x-hidden py-6" maxWidthClassName="max-w-[1920px]">
      <div className="mb-6">
        <h1 className="text-2xl font-semibold text-text flex items-center gap-2">
          {t("watchlists:title", "Watchlists")}
          <Tooltip title={t("watchlists:help.docsTooltip", "Open watchlists documentation")}>
            <a
              href={WATCHLISTS_MAIN_DOCS_URL}
              target="_blank"
              rel="noreferrer"
              aria-label={t("watchlists:help.docsTooltip", "Open watchlists documentation")}
              data-testid="watchlists-help-icon"
              className="text-text-muted hover:text-primary"
            >
              <HelpCircle className="h-5 w-5" />
            </a>
          </Tooltip>
        </h1>
        <p className="mt-1 text-sm text-text-muted">
          {t(
            "watchlists:description",
            "Monitor RSS feeds, websites, and forums. Create scheduled monitors to automatically scrape and process content."
          )}
        </p>
        <div className="mt-3 flex flex-wrap items-center gap-3 text-sm">
          <a
            href={WATCHLISTS_MAIN_DOCS_URL}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1 text-primary hover:underline"
            data-testid="watchlists-main-docs-link"
          >
            {t("watchlists:help.docs", "Watchlists docs")}
            <ExternalLink className="h-3.5 w-3.5" />
          </a>
          <a
            href={activeTabHelpHref}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1 text-primary hover:underline"
            data-testid="watchlists-context-docs-link"
          >
            {t("watchlists:help.learnMoreTab", "Learn more: {{tab}}", {
              tab: activeTabHelpLabel
            })}
            <ExternalLink className="h-3.5 w-3.5" />
          </a>
          {guidedTourState.status === "in_progress" ? (
            <Button
              size="small"
              type="default"
              onClick={resumeGuidedTour}
              data-testid="watchlists-resume-guide"
            >
              {t("watchlists:guide.resume", "Resume guided tour")}
            </Button>
          ) : (
            <Button
              size="small"
              type="default"
              onClick={startGuidedTour}
              data-testid="watchlists-start-guide"
            >
              {guidedTourState.status === "completed"
                ? t("watchlists:guide.restart", "Restart guided tour")
                : t("watchlists:guide.start", "Start guided tour")}
            </Button>
          )}
          {/* Show all views toggle */}
          {!iaExperimentEnabled && (
            <Tooltip title={t("watchlists:healthBar.showAllViewsTooltip", "Switch to the full 8-tab layout")}>
              <div className="inline-flex items-center gap-1.5 border-l border-border pl-3 ml-1">
                <Switch
                  size="small"
                  checked={showAllViews}
                  onChange={toggleShowAllViews}
                  data-testid="watchlists-show-all-views-toggle"
                />
                <span className="text-text-muted text-xs">
                  {t("watchlists:healthBar.showAllViews", "Show all views")}
                </span>
              </div>
            </Tooltip>
          )}
          {iaExperimentEnabled && (
            <div className="inline-flex flex-wrap items-center gap-2 border-l border-border pl-3 ml-1">
              <span className="text-text-muted">
                {t("watchlists:tabs.moreViews", "More views")}
              </span>
              {reducedIaSecondaryButtons.map((item) => (
                <Button
                  key={item.key}
                  size="small"
                  type={activeTab === item.key ? "primary" : "default"}
                  data-testid={`watchlists-experimental-tab-${item.key}`}
                  onClick={() => setActiveTab(item.key as typeof activeTab)}
                >
                  {item.label}
                </Button>
              ))}
            </div>
          )}
        </div>
        {iaExperimentEnabled ? (
          <div className="mt-3 flex flex-wrap items-center gap-2 text-sm">
            <span className="text-text-muted">
              {t("watchlists:taskViews.label", "Task views")}
            </span>
            {taskViews.map((taskView) => (
              <Button
                key={taskView.key}
                size="small"
                type={activeTaskView === taskView.key ? "primary" : "default"}
                aria-pressed={activeTaskView === taskView.key}
                onClick={() => setActiveTab(TASK_VIEW_PRIMARY_TAB[taskView.key])}
                data-testid={`watchlists-task-view-${taskView.key}`}
              >
                {taskView.label}
                <span className="ml-1 text-xs text-text-muted">{taskView.hint}</span>
              </Button>
            ))}
          </div>
        ) : showAllViews ? (
          <div className="mt-3 flex flex-wrap items-center gap-2 text-sm">
            <span className="text-text-muted">
              {t("watchlists:quickActions.label", "Jump to")}
            </span>
            {taskShortcuts.map((shortcut) => (
              <Button
                key={shortcut.key}
                size="small"
                type={activeTab === shortcut.key ? "primary" : "default"}
                onClick={() => setActiveTab(shortcut.key)}
                data-testid={`watchlists-task-open-${shortcut.key}`}
              >
                {shortcut.label}
              </Button>
            ))}
          </div>
        ) : null}
      </div>

      {renderWatchlistContainerShell()}

      {watchlistViewsAvailable && (
        <>
          {/* Persistent health bar - replaces Overview tab in progressive layout */}
          <WatchlistsHealthBar onOpenSettings={() => setSettingsDrawerOpen(true)} onNavigate={navigateToTab} />

          <div
            className="mb-4 flex flex-wrap items-center gap-2 text-sm"
            data-testid="watchlists-repeat-actions"
          >
            <span className="text-text-muted">
              {t("watchlists:quickActions.repeatLabel", "Jump to")}
            </span>
            {repeatUserShortcuts.map((shortcut) => (
              <Button
                key={shortcut.key}
                size="small"
                type={activeTab === shortcut.key ? "primary" : "default"}
                onClick={() => navigateToTab(shortcut.key)}
                data-testid={`watchlists-repeat-open-${shortcut.key}`}
              >
                {shortcut.label}
              </Button>
            ))}
            <Button
              size="small"
              type="default"
              icon={<Command className="h-3.5 w-3.5" />}
              onClick={() => setCommandPaletteOpen(true)}
              data-testid="watchlists-open-command-palette"
            >
              {t("watchlists:commandPalette.open", "Command palette")}
            </Button>
          </div>

          {orientationDismissed ? (
            <div className="mb-4">
              <Button
                size="small"
                type="link"
                data-testid="watchlists-orientation-restore"
                onClick={restoreOrientationForActiveTab}
              >
                {t("watchlists:orientation.showTabGuidance", "Show tab guidance")}
              </Button>
            </div>
          ) : (
            <DesignSystemAlert
              variant="info"
              className="mb-4"
              data-testid="watchlists-orientation-alert"
              title={<span data-testid="watchlists-orientation-title">{activeTabOrientation.title}</span>}
              dismissible
              onDismiss={dismissOrientationForActiveTab}
            >
              <div className="space-y-3">
                <span data-testid="watchlists-orientation-description">{activeTabOrientation.description}</span>
                <div className="flex flex-wrap gap-2">
                  {activeTabOrientation.actions.map((action) => (
                    <Button
                      key={action.key}
                      size="small"
                      data-testid={`watchlists-orientation-action-${action.key}`}
                      onClick={() => navigateToTab(action.target)}
                    >
                      {action.label}
                    </Button>
                  ))}
                </div>
              </div>
            </DesignSystemAlert>
          )}

          {activeTeachPoint && (
            <DesignSystemAlert
              variant="info"
              className="mb-4"
              data-testid="watchlists-teach-point-alert"
              title={<span data-testid="watchlists-teach-point-title">{activeTeachPoint.title}</span>}
              dismissible
              onDismiss={() => dismissTeachPoint(activeTeachPoint.key)}
            >
              <div className="space-y-3">
                <span data-testid="watchlists-teach-point-description">{activeTeachPoint.description}</span>
                <Button
                  size="small"
                  data-testid={`watchlists-teach-point-action-${activeTeachPoint.key}`}
                  onClick={() => navigateToTab(activeTeachPoint.actionTarget)}
                >
                  {activeTeachPoint.actionLabel}
                </Button>
              </div>
            </DesignSystemAlert>
          )}
        </>
      )}

      {showGuidedTourCompletion && (
        <DesignSystemAlert
          variant="success"
          className="mb-4"
          title={t("watchlists:guide.completedTitle", "Guided tour complete")}
          dismissible
          onDismiss={() => setShowGuidedTourCompletion(false)}
        >
          <div className="space-y-3">
            <span>
              {t(
                "watchlists:guide.completedDescription",
                "Next: monitor Activity for monitor health, review Updates for captured content, and open Reports for generated briefings."
              )}
            </span>
            <div className="flex flex-wrap gap-2">
              <Button size="small" onClick={() => navigateToTab("runs")}>
                {t("watchlists:guide.openActivity", "Open Activity")}
              </Button>
              <Button size="small" onClick={() => navigateToTab("items")}>
                {t("watchlists:guide.openArticles", "Open Updates")}
              </Button>
            </div>
          </div>
        </DesignSystemAlert>
      )}

      <DismissibleBetaAlert
        storageKey="beta-dismissed:watchlists"
        message={t("watchlists:betaNotice", "Beta Feature")}
        description={(
          <div className="space-y-1">
            <div>
              {t(
                "watchlists:betaDescription",
                "Watchlists is currently in beta. Some features may be incomplete or change."
              )}
            </div>
            <div className="flex flex-wrap items-center gap-3 text-sm">
              <a
                href={WATCHLISTS_MAIN_DOCS_URL}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1 text-primary hover:underline"
                data-testid="watchlists-beta-docs-link"
              >
                {t("watchlists:help.docs", "Watchlists docs")}
                <ExternalLink className="h-3.5 w-3.5" />
              </a>
              <a
                href={WATCHLISTS_ISSUE_REPORT_URL}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1 text-primary hover:underline"
                data-testid="watchlists-beta-report-link"
              >
                {t("watchlists:help.reportIssue", "Report an issue")}
                <ExternalLink className="h-3.5 w-3.5" />
              </a>
            </div>
          </div>
        )}
        className="mb-6"
      />

      {watchlistViewsAvailable && (
        isConstrained ? (
          <>
            <WatchlistsMobileNavigation
              activeKey={activeTab}
              fallbackLabel={t("watchlists:mobileNav.fallbackLabel", "Manage Watchlist")}
              groups={constrainedNavigationGroups}
              navigationLabel={t("watchlists:mobileNav.navigationLabel", "Watchlist management destinations")}
              onNavigate={navigateToTab}
              title={t("watchlists:mobileNav.title", "Manage Watchlist")}
            />
            <div
              key={refreshKey}
              className="min-w-0 max-w-full overflow-x-auto"
              data-testid="watchlists-tab-content-shell"
            >
              {constrainedActiveTabItem?.children}
            </div>
          </>
        ) : (
          <div
            className="min-w-0 max-w-full overflow-x-auto"
            data-testid="watchlists-tab-content-shell"
          >
            <Tabs
              key={refreshKey}
              activeKey={resolvedActiveTab}
              onChange={navigateToTab}
              items={renderedTabItems}
              className="watchlists-tabs"
              destroyOnHidden
            />
          </div>
        )
      )}

      {/* Settings drawer (accessible from health bar gear icon) */}
      <Drawer
        title={t("watchlists:tabs.settings", "Settings")}
        open={settingsDrawerOpen}
        onClose={() => setSettingsDrawerOpen(false)}
        size={isConstrained ? "100%" : 520}
        data-testid="watchlists-settings-drawer"
      >
        {renderWatchlistsTab("settings")}
      </Drawer>

      <Modal
        open={guidedTourOpen}
        onCancel={handleSkipGuidedTour}
        title={t("watchlists:guide.title", "Watchlists guided tour")}
        footer={(
          <div className="flex items-center justify-between gap-2">
            <Button onClick={handleSkipGuidedTour}>
              {t("watchlists:guide.skip", "Skip")}
            </Button>
            <div className="flex items-center gap-2">
              <Button
                onClick={handleGuidedTourBack}
                disabled={guidedTourState.step === 0}
              >
                {t("common:back", "Back")}
              </Button>
              <Button
                type="primary"
                onClick={handleGuidedTourNext}
              >
                {guidedTourState.step >= GUIDED_TOUR_LAST_STEP
                  ? t("watchlists:guide.finish", "Finish")
                  : t("common:next", "Next")}
              </Button>
            </div>
          </div>
        )}
      >
        <div className="space-y-3">
          <div className="text-xs font-medium text-text-muted">
            {t("watchlists:guide.progress", "Step {{current}} of {{total}}", {
              current: clampTourStep(guidedTourState.step) + 1,
              total: guidedTourSteps.length
            })}
          </div>
          <div className="text-base font-semibold">{guidedTourStep.title}</div>
          <div className="text-sm text-text-muted">{guidedTourStep.description}</div>
        </div>
      </Modal>

      <WatchlistSetupWizard
        open={setupWizardOpen}
        onCancel={() => setSetupWizardOpen(false)}
        onCreateWatchlist={createWatchlist}
        onCreateSources={createSetupSources}
        onCreateJob={createSetupJob}
        onComplete={completeSetupWizard}
      />

      <Modal
        open={watchlistFormOpen}
        onCancel={closeWatchlistForm}
        onOk={() => void saveWatchlistForm()}
        title={
          watchlistFormMode === "edit"
            ? t("watchlists:containers.editTitle", "Edit Watchlist")
            : t("watchlists:containers.createTitle", "Create Watchlist")
        }
        okText={
          watchlistFormMode === "edit"
            ? t("common:save", "Save")
            : t("common:create", "Create")
        }
        cancelText={t("common:cancel", "Cancel")}
        confirmLoading={watchlistFormSaving}
        destroyOnHidden
      >
        <div className="space-y-4">
          <div>
            <label htmlFor="watchlist-container-name" className="mb-1 block text-sm font-medium text-text">
              {t("watchlists:containers.nameLabel", "Name")}
            </label>
            <Input
              id="watchlist-container-name"
              aria-label={t("watchlists:containers.nameLabel", "Name")}
              value={watchlistForm.name}
              onChange={(event) => setWatchlistForm((previous) => ({
                ...previous,
                name: event.target.value
              }))}
              placeholder={t("watchlists:containers.namePlaceholder", "Healthcare ransomware")}
            />
          </div>
          <div>
            <label htmlFor="watchlist-container-objective" className="mb-1 block text-sm font-medium text-text">
              {t("watchlists:containers.objectiveLabel", "Objective")}
            </label>
            <Input.TextArea
              id="watchlist-container-objective"
              aria-label={t("watchlists:containers.objectiveLabel", "Objective")}
              value={watchlistForm.objective}
              rows={3}
              onChange={(event) => setWatchlistForm((previous) => ({
                ...previous,
                objective: event.target.value
              }))}
              placeholder={t(
                "watchlists:containers.objectivePlaceholder",
                "Track new updates and alert-worthy changes for this investigation."
              )}
            />
          </div>
          <div>
            <label htmlFor="watchlist-container-description" className="mb-1 block text-sm font-medium text-text">
              {t("watchlists:containers.descriptionLabel", "Description")}
            </label>
            <Input.TextArea
              id="watchlist-container-description"
              aria-label={t("watchlists:containers.descriptionLabel", "Description")}
              value={watchlistForm.description}
              rows={2}
              onChange={(event) => setWatchlistForm((previous) => ({
                ...previous,
                description: event.target.value
              }))}
            />
          </div>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
            <div>
              <label htmlFor="watchlist-container-domain" className="mb-1 block text-sm font-medium text-text">
                {t("watchlists:containers.domainLabel", "Domain")}
              </label>
              <Select
                id="watchlist-container-domain"
                aria-label={t("watchlists:containers.domainLabel", "Domain")}
                className="w-full"
                value={watchlistForm.domain}
                onChange={(domain) => setWatchlistForm((previous) => ({
                  ...previous,
                  domain: domain as WatchlistDomain
                }))}
                options={[
                  { value: "general", label: WATCHLIST_DOMAIN_LABELS.general },
                  { value: "cti_osint", label: WATCHLIST_DOMAIN_LABELS.cti_osint },
                  { value: "news", label: WATCHLIST_DOMAIN_LABELS.news }
                ]}
              />
            </div>
            <div>
              <label htmlFor="watchlist-container-priority" className="mb-1 block text-sm font-medium text-text">
                {t("watchlists:containers.priorityLabel", "Priority")}
              </label>
              <Select
                id="watchlist-container-priority"
                aria-label={t("watchlists:containers.priorityLabel", "Priority")}
                className="w-full"
                value={watchlistForm.priority}
                onChange={(priority) => setWatchlistForm((previous) => ({
                  ...previous,
                  priority: priority as WatchlistPriority
                }))}
                options={[
                  { value: "low", label: WATCHLIST_PRIORITY_LABELS.low },
                  { value: "medium", label: WATCHLIST_PRIORITY_LABELS.medium },
                  { value: "high", label: WATCHLIST_PRIORITY_LABELS.high },
                  { value: "critical", label: WATCHLIST_PRIORITY_LABELS.critical }
                ]}
              />
            </div>
            <div>
              <label htmlFor="watchlist-container-status" className="mb-1 block text-sm font-medium text-text">
                {t("watchlists:containers.statusLabel", "Status")}
              </label>
              <Select
                id="watchlist-container-status"
                aria-label={t("watchlists:containers.statusLabel", "Status")}
                className="w-full"
                value={watchlistForm.status}
                onChange={(status) => setWatchlistForm((previous) => ({
                  ...previous,
                  status: status as WatchlistStatus
                }))}
                options={[
                  { value: "active", label: WATCHLIST_STATUS_LABELS.active },
                  { value: "paused", label: WATCHLIST_STATUS_LABELS.paused },
                  { value: "archived", label: WATCHLIST_STATUS_LABELS.archived }
                ]}
              />
            </div>
          </div>
          <div>
            <label htmlFor="watchlist-container-tags" className="mb-1 block text-sm font-medium text-text">
              {t("watchlists:containers.tagsLabel", "Tags")}
            </label>
            <Input
              id="watchlist-container-tags"
              aria-label={t("watchlists:containers.tagsLabel", "Tags")}
              value={watchlistForm.tagsText}
              onChange={(event) => setWatchlistForm((previous) => ({
                ...previous,
                tagsText: event.target.value
              }))}
              placeholder={t("watchlists:containers.tagsPlaceholder", "ransomware, hospitals")}
            />
          </div>
        </div>
      </Modal>

      {/* Command palette */}
      <WatchlistsCommandPalette
        open={commandPaletteOpen}
        onClose={() => setCommandPaletteOpen(false)}
        commands={commandPaletteCommands}
      />

      {/* Keyboard shortcuts help */}
      <Modal
        open={shortcutsHelpOpen}
        onCancel={() => setShortcutsHelpOpen(false)}
        title={t("watchlists:keyboardShortcuts.title", "Keyboard Shortcuts")}
        footer={null}
        width={400}
        data-testid="watchlists-shortcuts-help"
      >
        <div className="space-y-2 text-sm">
          {[
            { keys: "\u2318/Ctrl + K", label: t("watchlists:keyboardShortcuts.commandPalette", "Command palette") },
            { keys: "1 / 2 / 3", label: t("watchlists:keyboardShortcuts.switchTab", "Switch tab (1-3)") },
            { keys: "N", label: t("watchlists:keyboardShortcuts.newEntity", "New entity") },
            { keys: "R", label: t("watchlists:keyboardShortcuts.refresh", "Refresh") },
            { keys: "/", label: t("watchlists:keyboardShortcuts.focusSearch", "Focus search") },
            { keys: "?", label: t("watchlists:keyboardShortcuts.showHelp", "Show shortcuts") }
          ].map((shortcut) => (
            <div key={shortcut.keys} className="flex items-center justify-between">
              <span>{shortcut.label}</span>
              <kbd className="rounded border border-border bg-surface px-1.5 py-0.5 text-xs font-mono">
                {shortcut.keys}
              </kbd>
            </div>
          ))}
        </div>
      </Modal>
      </PageShell>
    </WorkspaceConnectionGate>
  )
}
