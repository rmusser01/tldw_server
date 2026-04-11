import React, { Suspense, useEffect } from "react"
import { useTranslation } from "react-i18next"
import { Drawer, Tabs, Modal, Input, Empty, Skeleton, Button, message } from "antd"
import type { InputRef } from "antd"
import {
  AlertTriangle,
  FileText,
  Keyboard,
  MessageSquare,
  Sparkles,
  Search,
  Command,
  Loader2
} from "lucide-react"
import { useWorkspaceStore } from "@/store/workspace"
import { useTutorialStore } from "@/store/tutorials"
import {
  WORKSPACE_CONFLICT_NOTICE_THROTTLE_MS,
  WORKSPACE_STORAGE_CHANNEL_NAME,
  WORKSPACE_STORAGE_KEY,
  WORKSPACE_STORAGE_QUOTA_EVENT,
  isWorkspaceBroadcastSyncEnabled,
  isWorkspaceBroadcastUpdateMessage,
  shouldSurfaceWorkspaceConflictNotice,
  type WorkspaceStorageQuotaEventDetail
} from "@/store/workspace-events"
import { useMobile } from "@/hooks/useMediaQuery"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { bgRequest } from "@/services/background-proxy"
import type { AllowedPath } from "@/services/tldw/openapi-guard"
import {
  buildKnowledgeQaSeedNote,
  consumeWorkspacePlaygroundPrefill
} from "@/utils/workspace-playground-prefill"
import { FEATURE_FLAGS, useFeatureFlag } from "@/hooks/useFeatureFlags"
import { trackWorkspacePlaygroundTelemetry } from "@/utils/workspace-playground-telemetry"
import { isMac } from "@/hooks/useKeyboardShortcuts"
import { WorkspaceHeader } from "./WorkspaceHeader"
import { WorkspaceBanner } from "./WorkspaceBanner"
import { SharedWorkspaceBanner } from "./SharedWorkspaceBanner"
import { SharedWorkspaceProvider } from "./SharedWorkspaceContext"
import { WorkspaceStatusBar } from "./WorkspaceStatusBar"
import { ChatPane } from "./ChatPane"
import {
  TransferSourcesModal,
  type TransferSourcesModalLaunchRequest
} from "./TransferSourcesModal"
import { useSourceListViewState } from "./use-source-list-view-state"
import {
  filterSources as applyAdvancedSourceFilters,
  sortSources as applySourceSort
} from "./SourcesPane/source-list-view"
import {
  PaneResizer,
  DEFAULT_LEFT_WIDTH,
  DEFAULT_RIGHT_WIDTH
} from "./PaneResizer"
import {
  WORKSPACE_UNDO_WINDOW_MS,
  scheduleWorkspaceUndoAction,
  undoWorkspaceAction,
  undoLatestWorkspaceAction
} from "./undo-manager"
import {
  buildWorkspaceGlobalSearchResults,
  type WorkspaceGlobalSearchNoteDocument,
  type WorkspaceGlobalSearchResult
} from "./workspace-global-search"
import {
  collectDescendantSourceIds,
  createWorkspaceOrganizationIndex
} from "@/store/workspace-organization"

const SourcesPane = React.lazy(() =>
  import("./SourcesPane").then((module) => ({ default: module.SourcesPane }))
)

const StudioPane = React.lazy(() =>
  import("./StudioPane").then((module) => ({ default: module.StudioPane }))
)

const WORKSPACE_SWITCH_TRANSITION_MS = 420
const WORKSPACE_SOURCE_STATUS_POLL_INTERVAL_MS = 5000
const WORKSPACE_NOTE_SEARCH_LIMIT = 75
const WORKSPACE_STORAGE_PAYLOAD_BUDGET_DEFAULT_MB = 5
const WORKSPACE_STORAGE_PAYLOAD_BUDGET_VITE_ENV =
  "VITE_WORKSPACE_STORAGE_PAYLOAD_BUDGET_MB"
const WORKSPACE_STORAGE_PAYLOAD_BUDGET_NEXT_ENV =
  "NEXT_PUBLIC_WORKSPACE_STORAGE_PAYLOAD_BUDGET_MB"
const WORKSPACE_STORAGE_SPLIT_KEY_PREFIX = `${WORKSPACE_STORAGE_KEY}:workspace:`
const WORKSPACE_STORAGE_USAGE_REFRESH_DELAY_MS = 120
const ACCOUNT_STORAGE_USAGE_REFRESH_DELAY_MS = 1400
const WORKSPACE_ONBOARDING_DISMISSED_STORAGE_KEY =
  "tldw:workspace-playground:onboarding-dismissed:v1"
const WORKSPACE_REFRESH_LOOP_TRACE_SESSION_KEY =
  "tldw:workspace-playground:refresh-loop-trace:v1"
const WORKSPACE_REFRESH_LOOP_PENDING_SIGNAL_SESSION_KEY =
  "tldw:workspace-playground:refresh-loop-pending:v1"
const WORKSPACE_REFRESH_LOOP_WINDOW_MS = 45_000
const WORKSPACE_REFRESH_LOOP_THRESHOLD = 3
const WORKSPACE_CONFLICT_TRACKED_FIELDS = [
  "workspaceName",
  "workspaceBanner",
  "sources",
  "selectedSourceIds",
  "generatedArtifacts",
  "currentNote",
  "workspaceChatSessions",
  "audioSettings"
] as const
const WORKSPACE_CONFLICT_FIELD_LABELS: Record<string, string> = {
  workspaceName: "workspace name",
  workspaceBanner: "workspace banner",
  sources: "sources",
  selectedSourceIds: "source selection",
  generatedArtifacts: "generated outputs",
  currentNote: "quick note",
  workspaceChatSessions: "chat history",
  audioSettings: "audio settings"
}

const normalizeVectorProcessingStatus = (
  value: unknown
): "completed" | "pending" | "failed" | null => {
  if (typeof value === "number" && Number.isFinite(value)) {
    if (value >= 1) return "completed"
    if (value < 0) return "failed"
    return "pending"
  }
  if (typeof value !== "string") {
    return null
  }
  const lowered = value.trim().toLowerCase()
  if (!lowered) return null
  if (["1", "true", "complete", "completed", "done", "success"].includes(lowered)) {
    return "completed"
  }
  if (["-1", "false", "failed", "error"].includes(lowered)) {
    return "failed"
  }
  if (["0", "pending", "queued", "in_progress", "in-progress"].includes(lowered)) {
    return "pending"
  }
  return null
}

type WorkspaceTabKey = "sources" | "chat" | "studio"

type WorkspaceNoteKeywordLike =
  | string
  | {
      keyword?: string
      keyword_text?: string
      text?: string
      name?: string
    }

type WorkspaceNoteSearchItem = {
  id?: number
  title?: string
  content?: string
  version?: number
  keywords?: WorkspaceNoteKeywordLike[]
  metadata?: {
    keywords?: WorkspaceNoteKeywordLike[]
  }
}

type WorkspaceNotesSearchResponse =
  | WorkspaceNoteSearchItem[]
  | {
      notes?: WorkspaceNoteSearchItem[]
      results?: WorkspaceNoteSearchItem[]
      items?: WorkspaceNoteSearchItem[]
    }

type WorkspaceStorageUsageState = {
  usedBytes: number
  quotaBytes: number
  originUsedBytes: number | null
  originQuotaBytes: number | null
  accountUsedBytes: number | null
  accountQuotaBytes: number | null
}

const parseNoteKeyword = (
  keyword: WorkspaceNoteKeywordLike | null | undefined
): string | null => {
  if (!keyword) return null
  if (typeof keyword === "string") {
    const trimmed = keyword.trim()
    return trimmed.length > 0 ? trimmed : null
  }

  const value =
    keyword.keyword ??
    keyword.keyword_text ??
    keyword.text ??
    keyword.name ??
    null

  if (typeof value !== "string") return null
  const trimmed = value.trim()
  return trimmed.length > 0 ? trimmed : null
}

const normalizeNoteKeywords = (keywords: string[]): string[] => {
  const seen = new Set<string>()
  const normalized: string[] = []

  for (const keyword of keywords) {
    const trimmed = keyword.trim()
    if (!trimmed) continue
    const dedupe = trimmed.toLowerCase()
    if (seen.has(dedupe)) continue
    seen.add(dedupe)
    normalized.push(trimmed)
  }

  return normalized
}

const extractNoteKeywords = (
  note: WorkspaceNoteSearchItem | null | undefined
): string[] => {
  if (!note) return []
  const raw = Array.isArray(note.metadata?.keywords)
    ? note.metadata?.keywords
    : Array.isArray(note.keywords)
      ? note.keywords
      : []

  return normalizeNoteKeywords(
    raw
      .map((keyword) => parseNoteKeyword(keyword))
      .filter((keyword): keyword is string => Boolean(keyword))
  )
}

const pickNotesArray = (
  response: WorkspaceNotesSearchResponse
): WorkspaceNoteSearchItem[] => {
  if (Array.isArray(response)) return response
  if (Array.isArray(response.notes)) return response.notes
  if (Array.isArray(response.results)) return response.results
  if (Array.isArray(response.items)) return response.items
  return []
}

const buildWorkspaceNotesSearchPath = (workspaceTag: string): AllowedPath => {
  const params = new URLSearchParams()
  params.append("tokens", workspaceTag)
  params.set("limit", String(WORKSPACE_NOTE_SEARCH_LIMIT))
  params.set("include_keywords", "true")
  return `/api/v1/notes/search/?${params.toString()}` as AllowedPath
}

const buildWorkspaceNotePath = (noteId: number): AllowedPath =>
  `/api/v1/notes/${noteId}` as AllowedPath

const isDesktopLayout = (): boolean => {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
    return true
  }
  return window.matchMedia("(min-width: 1024px)").matches
}

const estimateUtf8ByteLength = (value: string): number => {
  if (typeof TextEncoder !== "undefined") {
    return new TextEncoder().encode(value).length
  }
  return unescape(encodeURIComponent(value)).length
}

const parseStorageBudgetCandidateMb = (
  candidate: unknown
): number | null => {
  if (typeof candidate === "number" && Number.isFinite(candidate) && candidate > 0) {
    return candidate
  }
  if (typeof candidate !== "string") return null
  const parsed = Number(candidate.trim())
  if (!Number.isFinite(parsed) || parsed <= 0) return null
  return parsed
}

const resolveWorkspacePayloadBudgetBytes = (): number => {
  const viteEnv = (import.meta as unknown as { env?: Record<string, unknown> }).env
  const viteBudgetMb = parseStorageBudgetCandidateMb(
    viteEnv?.[WORKSPACE_STORAGE_PAYLOAD_BUDGET_VITE_ENV]
  )
  if (viteBudgetMb != null) {
    return Math.round(viteBudgetMb * 1024 * 1024)
  }

  const nextProcess =
    typeof globalThis !== "undefined"
      ? (globalThis as { process?: { env?: Record<string, string | undefined> } })
          .process
      : undefined
  const nextBudgetMb = parseStorageBudgetCandidateMb(
    nextProcess?.env?.[WORKSPACE_STORAGE_PAYLOAD_BUDGET_NEXT_ENV]
  )
  if (nextBudgetMb != null) {
    return Math.round(nextBudgetMb * 1024 * 1024)
  }

  return WORKSPACE_STORAGE_PAYLOAD_BUDGET_DEFAULT_MB * 1024 * 1024
}

const isWorkspacePersistenceStorageKey = (key: string): boolean =>
  key === WORKSPACE_STORAGE_KEY || key.startsWith(WORKSPACE_STORAGE_SPLIT_KEY_PREFIX)

const estimateWorkspacePersistedPayloadBytes = (
  storage: Storage
): number => {
  let totalBytes = 0
  for (let index = 0; index < storage.length; index += 1) {
    const key = storage.key(index)
    if (!key || !isWorkspacePersistenceStorageKey(key)) continue
    const value = storage.getItem(key)
    if (value == null) continue
    totalBytes += estimateUtf8ByteLength(key)
    totalBytes += estimateUtf8ByteLength(value)
  }
  return totalBytes
}

const WORKSPACE_STORAGE_PAYLOAD_BUDGET_BYTES =
  resolveWorkspacePayloadBudgetBytes()

const sanitizeRefreshLoopTimestamps = (
  candidate: unknown
): number[] => {
  if (!Array.isArray(candidate)) return []
  return candidate
    .filter((value): value is number => typeof value === "number" && Number.isFinite(value))
    .sort((a, b) => a - b)
    .slice(-20)
}

const recordWorkspaceRefreshLoopAttempt = (): {
  count: number
  windowMs: number
} | null => {
  if (typeof window === "undefined") return null
  const now = Date.now()
  try {
    const raw = window.sessionStorage.getItem(WORKSPACE_REFRESH_LOOP_TRACE_SESSION_KEY)
    const parsed = raw ? JSON.parse(raw) : null
    const previousTimestamps = sanitizeRefreshLoopTimestamps(parsed?.timestamps)
    const windowedTimestamps = previousTimestamps.filter(
      (timestamp) => now - timestamp <= WORKSPACE_REFRESH_LOOP_WINDOW_MS
    )
    windowedTimestamps.push(now)
    window.sessionStorage.setItem(
      WORKSPACE_REFRESH_LOOP_TRACE_SESSION_KEY,
      JSON.stringify({ timestamps: windowedTimestamps })
    )
    return {
      count: windowedTimestamps.length,
      windowMs: WORKSPACE_REFRESH_LOOP_WINDOW_MS
    }
  } catch {
    return null
  }
}

const persistWorkspaceRefreshLoopSignal = (signal: {
  count: number
  windowMs: number
}) => {
  if (typeof window === "undefined") return
  try {
    window.sessionStorage.setItem(
      WORKSPACE_REFRESH_LOOP_PENDING_SIGNAL_SESSION_KEY,
      JSON.stringify({
        count: signal.count,
        windowMs: signal.windowMs,
        detectedAt: Date.now()
      })
    )
  } catch {
    // Ignore session storage errors.
  }
}

const consumeWorkspaceRefreshLoopSignal = (): {
  count: number
  windowMs: number
} | null => {
  if (typeof window === "undefined") return null
  try {
    const raw = window.sessionStorage.getItem(
      WORKSPACE_REFRESH_LOOP_PENDING_SIGNAL_SESSION_KEY
    )
    if (!raw) return null
    window.sessionStorage.removeItem(WORKSPACE_REFRESH_LOOP_PENDING_SIGNAL_SESSION_KEY)
    const parsed = JSON.parse(raw) as { count?: unknown; windowMs?: unknown }
    if (
      typeof parsed?.count !== "number" ||
      !Number.isFinite(parsed.count) ||
      parsed.count < 1
    ) {
      return null
    }
    const windowMs =
      typeof parsed.windowMs === "number" && Number.isFinite(parsed.windowMs)
        ? parsed.windowMs
        : WORKSPACE_REFRESH_LOOP_WINDOW_MS
    return {
      count: parsed.count,
      windowMs
    }
  } catch {
    return null
  }
}

const parsePersistedWorkspaceState = (
  serializedValue: string | null | undefined
): Record<string, unknown> | null => {
  if (typeof serializedValue !== "string" || serializedValue.length === 0) {
    return null
  }

  try {
    const parsed = JSON.parse(serializedValue)
    if (!parsed || typeof parsed !== "object") {
      return null
    }

    const candidateState =
      "state" in parsed &&
      parsed.state &&
      typeof parsed.state === "object"
        ? parsed.state
        : parsed

    if (!candidateState || typeof candidateState !== "object") {
      return null
    }

    const baseState = candidateState as Record<string, unknown>
    const workspaceId =
      typeof baseState.workspaceId === "string" ? baseState.workspaceId : null
    const snapshotsCandidate = baseState.workspaceSnapshots
    const snapshots =
      snapshotsCandidate && typeof snapshotsCandidate === "object"
        ? (snapshotsCandidate as Record<string, unknown>)
        : null
    const activeSnapshot =
      workspaceId && snapshots && snapshots[workspaceId] && typeof snapshots[workspaceId] === "object"
        ? (snapshots[workspaceId] as Record<string, unknown>)
        : null

    if (!activeSnapshot) {
      return baseState
    }

    return {
      ...baseState,
      workspaceName:
        baseState.workspaceName ?? activeSnapshot.workspaceName ?? null,
      workspaceBanner:
        baseState.workspaceBanner ?? activeSnapshot.workspaceBanner ?? null,
      sources: baseState.sources ?? activeSnapshot.sources ?? null,
      selectedSourceIds:
        baseState.selectedSourceIds ?? activeSnapshot.selectedSourceIds ?? null,
      generatedArtifacts:
        baseState.generatedArtifacts ?? activeSnapshot.generatedArtifacts ?? null,
      currentNote: baseState.currentNote ?? activeSnapshot.currentNote ?? null,
      audioSettings: baseState.audioSettings ?? activeSnapshot.audioSettings ?? null
    }
  } catch {
    return null
  }
}

const deriveWorkspaceConflictFieldDiff = (
  oldValue: string | null | undefined,
  newValue: string | null | undefined
): string[] => {
  const previousState = parsePersistedWorkspaceState(oldValue)
  const nextState = parsePersistedWorkspaceState(newValue)
  if (!previousState || !nextState) {
    return []
  }

  return WORKSPACE_CONFLICT_TRACKED_FIELDS.filter((field) => {
    try {
      return (
        JSON.stringify(previousState[field]) !== JSON.stringify(nextState[field])
      )
    } catch {
      return previousState[field] !== nextState[field]
    }
  })
    .map((field) => WORKSPACE_CONFLICT_FIELD_LABELS[field] || field)
    .slice(0, 3)
}

const isMediaLikelyReadyForRag = (detail: unknown): boolean => {
  if (!detail || typeof detail !== "object") {
    return false
  }

  const candidate = detail as Record<string, unknown>
  const content = candidate.content as Record<string, unknown> | undefined
  const processing = candidate.processing as Record<string, unknown> | undefined
  const vectorStatusCandidates = [
    candidate.vector_processing,
    candidate.vector_processing_status,
    processing?.vector_processing,
    processing?.vector_processing_status
  ]
  let sawIncompleteVectorStatus = false

  for (const statusCandidate of vectorStatusCandidates) {
    const normalizedStatus = normalizeVectorProcessingStatus(statusCandidate)
    if (normalizedStatus === "completed") {
      return true
    }
    if (normalizedStatus === "pending" || normalizedStatus === "failed") {
      sawIncompleteVectorStatus = true
    }
  }

  if (sawIncompleteVectorStatus) {
    return false
  }

  const contentText =
    typeof content?.text === "string" ? content.text.trim() : ""
  if (contentText.length > 0) {
    return true
  }

  const analysis =
    typeof processing?.analysis === "string" ? processing.analysis.trim() : ""
  if (analysis.length > 0) {
    return true
  }

  const safeMetadata = processing?.safe_metadata
  if (
    safeMetadata &&
    typeof safeMetadata === "object" &&
    Object.keys(safeMetadata as Record<string, unknown>).length > 0
  ) {
    return true
  }

  return false
}

const isTransientSourceStatusError = (
  error: unknown
): { transient: boolean; message: string } => {
  const status = (error as { status?: number } | null)?.status
  const message =
    error instanceof Error ? error.message : String(error ?? "Unknown error")
  const transient =
    status === 0 ||
    status === 404 ||
    status === 408 ||
    status === 429 ||
    status === 502 ||
    status === 503 ||
    status === 504 ||
    /network|timeout|abort/i.test(message)

  return { transient, message }
}

const WorkspacePlaygroundSkeleton: React.FC<{ isMobile: boolean }> = ({
  isMobile
}) => (
  <div
    data-testid="workspace-playground-skeleton"
    className="flex h-full flex-col bg-bg px-3 py-3"
  >
    <div className="border-b border-border pb-3">
      <Skeleton.Input active size="small" className="w-[220px] max-w-full" />
    </div>
    {isMobile ? (
      <div className="flex min-h-0 flex-1 flex-col gap-3 pt-3">
        <div className="grid grid-cols-3 gap-2">
          <Skeleton.Button active size="small" block />
          <Skeleton.Button active size="small" block />
          <Skeleton.Button active size="small" block />
        </div>
        <Skeleton active paragraph={{ rows: 8 }} title={false} />
      </div>
    ) : (
      <div className="grid min-h-0 flex-1 grid-cols-1 gap-3 pt-3 lg:grid-cols-[280px_1fr_320px]">
        <Skeleton active paragraph={{ rows: 9 }} title={false} />
        <Skeleton active paragraph={{ rows: 10 }} title={false} />
        <Skeleton active paragraph={{ rows: 8 }} title={false} />
      </div>
    )}
  </div>
)

const WorkspacePaneFallback: React.FC<{ testId: string }> = ({ testId }) => (
  <div
    data-testid={testId}
    className="flex h-full min-h-0 flex-col gap-3 bg-transparent px-3 py-3"
  >
    <div className="h-8 rounded-md bg-surface2/80" />
    <div className="flex-1 rounded-lg border border-border/70 bg-surface2/60" />
  </div>
)

type WorkspacePlaygroundErrorBoundaryState = {
  hasError: boolean
  errorMessage: string | null
  showDetails: boolean
}

class WorkspacePlaygroundErrorBoundary extends React.Component<
  React.PropsWithChildren,
  WorkspacePlaygroundErrorBoundaryState
> {
  state: WorkspacePlaygroundErrorBoundaryState = {
    hasError: false,
    errorMessage: null,
    showDetails: false
  }

  static getDerivedStateFromError(
    error: unknown
  ): Partial<WorkspacePlaygroundErrorBoundaryState> {
    const message =
      error instanceof Error
        ? error.message
        : typeof error === "string"
          ? error
          : "Unknown error"
    return { hasError: true, errorMessage: message }
  }

  componentDidCatch(error: unknown): void {
    // Surface in console for debugging while showing a recoverable fallback UI.
    console.error("WorkspacePlayground render error", error)
  }

  handleReload = () => {
    if (typeof window !== "undefined") {
      window.location.reload()
    }
  }

  handleExportData = () => {
    try {
      const state = useWorkspaceStore.getState()
      const blob = new Blob([JSON.stringify(state)], { type: "application/json" })
      const url = URL.createObjectURL(blob)
      const anchor = document.createElement("a")
      anchor.href = url
      anchor.download = `workspace-recovery-${Date.now()}.json`
      anchor.click()
      URL.revokeObjectURL(url)
    } catch {
      // Silently fail — user can still reload.
    }
  }

  handleClearCache = () => {
    try {
      useWorkspaceStore.persist.clearStorage()
    } catch {
      // Ignore — reload will still work.
    }
    this.handleReload()
  }

  handleSwitchWorkspace = () => {
    try {
      // Use the Zustand store API directly to clear the workspace ID.
      // This avoids coupling to the internal shape of the persisted state.
      useWorkspaceStore.setState({ workspaceId: null })
    } catch {
      // Ignore — reload will still work.
    }
    this.handleReload()
  }

  render() {
    if (!this.state.hasError) {
      return this.props.children
    }

    const truncatedError = this.state.errorMessage
      ? this.state.errorMessage.length > 200
        ? `${this.state.errorMessage.slice(0, 200)}...`
        : this.state.errorMessage
      : null

    return (
      <div className="flex h-full items-center justify-center p-6">
        <div className="w-full max-w-md rounded-lg border border-border bg-surface p-5 text-center shadow-card">
          <h2 className="text-base font-semibold text-text">
            Something went wrong
          </h2>
          <p className="mt-2 text-sm text-text-muted">
            The workspace hit an unexpected error. You can try the options below
            to recover.
          </p>

          {truncatedError && (
            <div className="mt-3">
              <button
                type="button"
                onClick={() =>
                  this.setState((prev) => ({
                    showDetails: !prev.showDetails
                  }))
                }
                className="text-xs text-text-subtle underline hover:text-text-muted"
                data-testid="workspace-error-details-toggle"
              >
                {this.state.showDetails ? "Hide details" : "Show details"}
              </button>
              {this.state.showDetails && (
                <pre
                  className="mt-1 max-h-24 overflow-auto rounded bg-surface2 p-2 text-left text-[11px] text-text-muted"
                  data-testid="workspace-error-details"
                >
                  {truncatedError}
                </pre>
              )}
            </div>
          )}

          <div className="mt-4 flex flex-col gap-2">
            <button
              type="button"
              onClick={this.handleReload}
              className="w-full rounded bg-primary px-3 py-1.5 text-sm font-medium text-white transition hover:opacity-90"
              data-testid="workspace-reload-button"
            >
              Reload workspace
            </button>
            <button
              type="button"
              onClick={this.handleExportData}
              className="w-full rounded border border-border px-3 py-1.5 text-sm font-medium text-text transition hover:bg-surface2"
              data-testid="workspace-export-recovery-button"
            >
              Export workspace data
            </button>
            <button
              type="button"
              onClick={this.handleClearCache}
              className="w-full rounded border border-border px-3 py-1.5 text-sm font-medium text-text transition hover:bg-surface2"
              data-testid="workspace-clear-cache-button"
            >
              Clear workspace cache
            </button>
            <button
              type="button"
              onClick={this.handleSwitchWorkspace}
              className="w-full rounded border border-border px-3 py-1.5 text-sm font-medium text-text transition hover:bg-surface2"
              data-testid="workspace-switch-from-error-button"
            >
              Switch to different workspace
            </button>
          </div>
        </div>
      </div>
    )
  }
}

/**
 * WorkspacePlayground - NotebookLM-style three-pane research interface
 *
 * Layout at different breakpoints:
 * - lg+ (1024px+): Full three-pane layout
 * - md (768-1023px): Chat main, Sources/Studio as slide-out drawers
 * - sm (<768px): Bottom tab navigation between panes
 *
 * Features:
 * - Sources Pane (left): Add and manage research sources
 * - Chat Pane (middle): RAG-powered conversation with selected sources
 * - Studio Pane (right): Generate outputs (summaries, quizzes, flashcards, etc.)
 */
const WorkspacePlaygroundBody: React.FC = () => {
  const { t } = useTranslation(["playground", "option", "common"])
  const isMobile = useMobile()
  const [messageApi, messageContextHolder] = message.useMessage()
  const [provenanceFlagEnabled] = useFeatureFlag(
    FEATURE_FLAGS.RESEARCH_STUDIO_PROVENANCE_V1
  )
  const [statusGuardrailsFlagEnabled] = useFeatureFlag(
    FEATURE_FLAGS.RESEARCH_STUDIO_STATUS_GUARDRAILS_V1
  )
  const provenanceEnabled = provenanceFlagEnabled !== false
  const statusGuardrailsEnabled = statusGuardrailsFlagEnabled !== false

  // Mobile drawer state
  const [leftDrawerOpen, setLeftDrawerOpen] = React.useState(false)
  const [rightDrawerOpen, setRightDrawerOpen] = React.useState(false)

  // Mobile tab state
  const [activeTab, setActiveTab] = React.useState<WorkspaceTabKey>("chat")

  // Global search state
  const [globalSearchOpen, setGlobalSearchOpen] = React.useState(false)
  const [globalSearchQuery, setGlobalSearchQuery] = React.useState("")
  const [activeSearchResultIndex, setActiveSearchResultIndex] = React.useState(0)
  const [transferSourcesRequest, setTransferSourcesRequest] =
    React.useState<TransferSourcesModalLaunchRequest | null>(null)
  const [workspaceSearchNotes, setWorkspaceSearchNotes] = React.useState<
    WorkspaceGlobalSearchNoteDocument[]
  >([])
  const globalSearchInputRef = React.useRef<InputRef | null>(null)

  // Workspace switch transition cue state
  const [showWorkspaceTransitionCue, setShowWorkspaceTransitionCue] =
    React.useState(false)
  const previousWorkspaceIdRef = React.useRef<string | null>(null)
  const workspaceTransitionTimerRef = React.useRef<number | null>(null)
  const [showStorageQuotaWarning, setShowStorageQuotaWarning] =
    React.useState(false)
  const [storageHighUsageDismissed, setStorageHighUsageDismissed] =
    React.useState(false)
  const [showCrossTabSyncWarning, setShowCrossTabSyncWarning] =
    React.useState(false)
  const [crossTabChangedFields, setCrossTabChangedFields] = React.useState<
    string[]
  >([])
  const [workspaceStorageUsage, setWorkspaceStorageUsage] =
    React.useState<WorkspaceStorageUsageState>({
    usedBytes: 0,
    quotaBytes: WORKSPACE_STORAGE_PAYLOAD_BUDGET_BYTES,
    originUsedBytes: null,
    originQuotaBytes: null,
    accountUsedBytes: null,
    accountQuotaBytes: null
  })
  const storageUsagePercent = React.useMemo(() => {
    if (
      workspaceStorageUsage.quotaBytes <= 0 ||
      !Number.isFinite(workspaceStorageUsage.usedBytes) ||
      !Number.isFinite(workspaceStorageUsage.quotaBytes)
    ) {
      return 0
    }
    return Math.round(
      (workspaceStorageUsage.usedBytes / workspaceStorageUsage.quotaBytes) * 100
    )
  }, [workspaceStorageUsage.usedBytes, workspaceStorageUsage.quotaBytes])

  const showStorageHighUsageWarning =
    statusGuardrailsEnabled &&
    !storageHighUsageDismissed &&
    !showStorageQuotaWarning &&
    storageUsagePercent >= 80

  const lastCrossTabSyncWarningRef = React.useRef(0)
  const onboardingInitializedRef = React.useRef(false)
  const [showTutorialPrompt, setShowTutorialPrompt] = React.useState(false)
  const [showShortcutsModal, setShowShortcutsModal] = React.useState(false)
  const startTutorial = useTutorialStore((s) => s.startTutorial)

  // Shared workspace state (from ?shared= query param)
  const [sharedShareId, setSharedShareId] = React.useState<number | null>(null)
  const [sharedAccessLevel, setSharedAccessLevel] = React.useState<
    "view_chat" | "view_chat_add" | "full_edit" | null
  >(null)
  const [sharedOwnerUserId, setSharedOwnerUserId] = React.useState<number | null>(null)
  const [sharedAllowClone, setSharedAllowClone] = React.useState(false)

  React.useEffect(() => {
    try {
      const params = new URLSearchParams(window.location.search)
      const shareIdStr = params.get("shared")
      if (shareIdStr) {
        const sid = parseInt(shareIdStr, 10)
        if (!isNaN(sid)) {
          setSharedShareId(sid)
          // Fetch share details
          import("@/hooks/useSharing").then(async (mod) => {
            try {
              const { getTldwServerURL } = await import("@/services/tldw-server")
              const { fetchWithTldwAuth } = await import("@/services/tldw/auth-fetch")
              const base = await getTldwServerURL()
              const res = await fetchWithTldwAuth(`${base}/api/v1/sharing/shared-with-me/${sid}/workspace`)
              if (res.ok) {
                const data = await res.json()
                const share = data.share
                if (share) {
                  setSharedAccessLevel(share.access_level)
                  setSharedOwnerUserId(share.owner_user_id)
                  setSharedAllowClone(share.allow_clone)
                }
              }
            } catch { /* ignore fetch failures */ }
          })
        }
      }
    } catch { /* ignore URL parsing failures */ }
  }, [])

  // Workspace store
  const workspaceId = useWorkspaceStore((s) => s.workspaceId)
  const workspaceName = useWorkspaceStore((s) => s.workspaceName) || ""
  const workspaceBanner = useWorkspaceStore((s) => s.workspaceBanner) || {
    title: "",
    subtitle: "",
    image: null
  }
  const initializeWorkspace = useWorkspaceStore((s) => s.initializeWorkspace)
  const createNewWorkspace = useWorkspaceStore((s) => s.createNewWorkspace)
  const addSources = useWorkspaceStore((s) => s.addSources)
  const workspaceTag = useWorkspaceStore((s) => s.workspaceTag)
  const setSelectedSourceIds = useWorkspaceStore((s) => s.setSelectedSourceIds)
  const captureToCurrentNote = useWorkspaceStore((s) => s.captureToCurrentNote)
  const clearCurrentNote = useWorkspaceStore((s) => s.clearCurrentNote)
  const setCurrentNote = useWorkspaceStore((s) => s.setCurrentNote)
  const loadNote = useWorkspaceStore((s) => s.loadNote)
  const duplicateWorkspace = useWorkspaceStore((s) => s.duplicateWorkspace)
  const selectedSourceIds = useWorkspaceStore((s) => s.selectedSourceIds)
  const selectedSourceFolderIds = useWorkspaceStore(
    (s) => s.selectedSourceFolderIds
  )
  const generatedArtifacts = useWorkspaceStore((s) => s.generatedArtifacts)
  const leftPaneCollapsed = useWorkspaceStore((s) => s.leftPaneCollapsed)
  const rightPaneCollapsed = useWorkspaceStore((s) => s.rightPaneCollapsed)
  const setLeftPaneCollapsed = useWorkspaceStore((s) => s.setLeftPaneCollapsed)
  const setRightPaneCollapsed = useWorkspaceStore((s) => s.setRightPaneCollapsed)
  const sources = useWorkspaceStore((s) => s.sources)
  const sourceFolders = useWorkspaceStore((s) => s.sourceFolders)
  const sourceFolderMemberships = useWorkspaceStore(
    (s) => s.sourceFolderMemberships
  )
  const activeFolderId = useWorkspaceStore((s) => s.activeFolderId)
  const sourceSearchQuery = useWorkspaceStore((s) => s.sourceSearchQuery)
  const isGeneratingOutput = useWorkspaceStore((s) => s.isGeneratingOutput)
  const generatingOutputType = useWorkspaceStore((s) => s.generatingOutputType)
  const currentNote = useWorkspaceStore((s) => s.currentNote)
  const workspaceChatSessions = useWorkspaceStore((s) => s.workspaceChatSessions)
  const focusSourceById = useWorkspaceStore((s) => s.focusSourceById)
  const focusChatMessageById = useWorkspaceStore((s) => s.focusChatMessageById)
  const focusWorkspaceNote = useWorkspaceStore((s) => s.focusWorkspaceNote)
  const setSourceStatusByMediaId = useWorkspaceStore(
    (s) => s.setSourceStatusByMediaId
  )
  const storeHydrated = useWorkspaceStore((s) => s.storeHydrated)
  const isStoreHydrated = storeHydrated !== false
  const sourceStatusFailureRef = React.useRef<Record<number, number>>({})
  const lastStatusViewSignatureRef = React.useRef<string | null>(null)
  const {
    sourceListViewState,
    patchSourceListViewState,
    resetAdvancedSourceFilters
  } = useSourceListViewState()
  const organizationIndex = React.useMemo(
    () =>
      createWorkspaceOrganizationIndex({
        sources,
        sourceFolders: sourceFolders || [],
        sourceFolderMemberships: sourceFolderMemberships || []
      }),
    [sourceFolderMemberships, sourceFolders, sources]
  )
  const activeFolderSourceIds = React.useMemo(
    () =>
      activeFolderId
        ? new Set(collectDescendantSourceIds(organizationIndex, activeFolderId))
        : null,
    [activeFolderId, organizationIndex]
  )
  const searchedSources = React.useMemo(() => {
    const scopedSources = activeFolderSourceIds
      ? sources.filter((source) => activeFolderSourceIds.has(source.id))
      : sources
    if (!sourceSearchQuery?.trim()) return scopedSources
    const query = sourceSearchQuery.toLowerCase()
    return scopedSources.filter((source) =>
      source.title.toLowerCase().includes(query)
    )
  }, [activeFolderSourceIds, sourceSearchQuery, sources])
  const filteredSources = React.useMemo(
    () =>
      applySourceSort(
        applyAdvancedSourceFilters(searchedSources, sourceListViewState),
        sourceListViewState.sort
      ),
    [searchedSources, sourceListViewState]
  )
  const effectiveSelectedSourceEntries = React.useMemo(() => {
    const selectedSourceIdSet = new Set(selectedSourceIds || [])
    const selectedFolderIdSet = new Set(selectedSourceFolderIds || [])
    const effectiveSelectedIds = new Set<string>()

    for (const source of sources) {
      if (selectedSourceIdSet.has(source.id)) {
        effectiveSelectedIds.add(source.id)
      }
    }

    for (const folderId of selectedFolderIdSet) {
      for (const sourceId of collectDescendantSourceIds(organizationIndex, folderId)) {
        effectiveSelectedIds.add(sourceId)
      }
    }

    return sources.filter((source) => effectiveSelectedIds.has(source.id))
  }, [
    organizationIndex,
    selectedSourceFolderIds,
    selectedSourceIds,
    sources
  ])
  const readyEffectiveSelectedSourceEntries = React.useMemo(
    () =>
      effectiveSelectedSourceEntries.filter((source) =>
        organizationIndex.readySourceIds.has(source.id)
      ),
    [effectiveSelectedSourceEntries, organizationIndex.readySourceIds]
  )
  const readyEffectiveSelectedSourceIdSet = React.useMemo(
    () => new Set(readyEffectiveSelectedSourceEntries.map((source) => source.id)),
    [readyEffectiveSelectedSourceEntries]
  )
  const visibleReadySelectedCount = React.useMemo(
    () =>
      filteredSources.filter((source) =>
        readyEffectiveSelectedSourceIdSet.has(source.id)
      ).length,
    [filteredSources, readyEffectiveSelectedSourceIdSet]
  )
  const hiddenSelectedCount = Math.max(
    0,
    readyEffectiveSelectedSourceEntries.length - visibleReadySelectedCount
  )
  const ineligibleSelectedCount = Math.max(
    0,
    effectiveSelectedSourceEntries.length -
      readyEffectiveSelectedSourceEntries.length
  )

  const [leftPaneWidth, setLeftPaneWidth] = React.useState(DEFAULT_LEFT_WIDTH)
  const [rightPaneWidth, setRightPaneWidth] = React.useState(DEFAULT_RIGHT_WIDTH)

  const leftPaneOpen = !leftPaneCollapsed
  const rightPaneOpen = !rightPaneCollapsed
  const desktopChatContentWidthMode: "comfortable" | "expanded" | "full" =
    leftPaneOpen && rightPaneOpen
      ? "comfortable"
      : leftPaneOpen || rightPaneOpen
        ? "expanded"
        : "full"

  const workspaceChatMessages = React.useMemo(
    () => (workspaceId ? workspaceChatSessions[workspaceId]?.messages || [] : []),
    [workspaceChatSessions, workspaceId]
  )

  const globalSearchResults = React.useMemo(
    () =>
      buildWorkspaceGlobalSearchResults({
        query: globalSearchQuery,
        sources,
        chatMessages: workspaceChatMessages,
        currentNote,
        workspaceNotes: workspaceSearchNotes
      }),
    [
      currentNote,
      globalSearchQuery,
      sources,
      workspaceChatMessages,
      workspaceSearchNotes
    ]
  )

  const processingMediaIds = React.useMemo(
    () =>
      sources
        .filter((source) => (source.status || "ready") === "processing")
        .map((source) => source.mediaId),
    [sources]
  )
  const activeWorkspaceOperations = React.useMemo(() => {
    const operations: string[] = []

    if (processingMediaIds.length > 0) {
      operations.push(
        `${t("playground:workspace.activityProcessing", "Processing")} ${
          processingMediaIds.length
        } ${t("playground:workspace.activitySource", "source")}${
          processingMediaIds.length === 1 ? "" : "s"
        }`
      )
    }

    if (isGeneratingOutput) {
      const rawType =
        typeof generatingOutputType === "string" ? generatingOutputType.trim() : ""
      const readableType =
        rawType.length > 0
          ? rawType.replace(/_/g, " ")
          : t("playground:workspace.activityOutput", "output")
      operations.push(
        `${t("playground:workspace.activityGenerating", "Generating")} ${readableType}`
      )
    }

    return operations
  }, [generatingOutputType, isGeneratingOutput, processingMediaIds.length, t])

  useEffect(() => {
    if (!statusGuardrailsEnabled) {
      lastStatusViewSignatureRef.current = null
      return
    }
    if (activeWorkspaceOperations.length === 0) {
      lastStatusViewSignatureRef.current = null
      return
    }

    const signature = activeWorkspaceOperations.join("|")
    if (lastStatusViewSignatureRef.current === signature) {
      return
    }
    lastStatusViewSignatureRef.current = signature

    void trackWorkspacePlaygroundTelemetry({
      type: "status_viewed",
      workspace_id: workspaceId || null,
      operations_count: activeWorkspaceOperations.length,
      status: signature
    })
  }, [activeWorkspaceOperations, statusGuardrailsEnabled, workspaceId])

  const refreshWorkspaceStorageUsage = React.useCallback(async () => {
    if (typeof window === "undefined") return
    try {
      const usedBytes = estimateWorkspacePersistedPayloadBytes(window.localStorage)
      let originUsedBytes: number | null = null
      let originQuotaBytes: number | null = null
      if (
        typeof navigator !== "undefined" &&
        navigator.storage &&
        typeof navigator.storage.estimate === "function"
      ) {
        try {
          const estimate = await navigator.storage.estimate()
          const usage = estimate?.usage
          const quota = estimate?.quota
          if (typeof usage === "number" && Number.isFinite(usage) && usage >= 0) {
            originUsedBytes = usage
          }
          if (typeof quota === "number" && Number.isFinite(quota) && quota > 0) {
            originQuotaBytes = quota
          }
        } catch {
          // Ignore storage estimate failures.
        }
      }
      setWorkspaceStorageUsage((previousState) => ({
        ...previousState,
        usedBytes,
        quotaBytes: WORKSPACE_STORAGE_PAYLOAD_BUDGET_BYTES,
        originUsedBytes,
        originQuotaBytes
      }))
    } catch {
      // Ignore storage read errors.
    }
  }, [])

  const refreshAccountStorageUsage = React.useCallback(async () => {
    if (typeof tldwClient.getCurrentUserStorageQuota !== "function") return
    try {
      const response = await tldwClient.getCurrentUserStorageQuota()
      const usedMb = Number(response?.storage_used_mb)
      const quotaMb = Number(response?.storage_quota_mb)
      const accountUsedBytes =
        Number.isFinite(usedMb) && usedMb >= 0 ? usedMb * 1024 * 1024 : null
      const accountQuotaBytes =
        Number.isFinite(quotaMb) && quotaMb > 0 ? quotaMb * 1024 * 1024 : null
      setWorkspaceStorageUsage((previousState) => ({
        ...previousState,
        accountUsedBytes,
        accountQuotaBytes
      }))
    } catch {
      if (typeof tldwClient.getCurrentUserProfile !== "function") return
      try {
        const profile = await tldwClient.getCurrentUserProfile({
          sections: "quotas"
        })
        const quotas = (profile as { quotas?: Record<string, unknown> } | null)?.quotas
        const usedMb = Number(quotas?.storage_used_mb)
        const quotaMb = Number(quotas?.storage_quota_mb)
        const accountUsedBytes =
          Number.isFinite(usedMb) && usedMb >= 0 ? usedMb * 1024 * 1024 : null
        const accountQuotaBytes =
          Number.isFinite(quotaMb) && quotaMb > 0 ? quotaMb * 1024 * 1024 : null
        setWorkspaceStorageUsage((previousState) => ({
          ...previousState,
          accountUsedBytes,
          accountQuotaBytes
        }))
      } catch {
        // Ignore account storage fetch failures and keep existing values.
      }
    }
  }, [])

  const closeGlobalSearch = React.useCallback(() => {
    setGlobalSearchOpen(false)
    setGlobalSearchQuery("")
    setActiveSearchResultIndex(0)
  }, [])

  const openTransferSourcesModal = React.useCallback(
    (request: TransferSourcesModalLaunchRequest) => {
      setTransferSourcesRequest(request)
    },
    []
  )

  const closeTransferSourcesModal = React.useCallback(() => {
    setTransferSourcesRequest(null)
  }, [])

  const dismissOnboardingOverlay = React.useCallback(() => {
    if (typeof window === "undefined") return

    try {
      window.localStorage.setItem(
        WORKSPACE_ONBOARDING_DISMISSED_STORAGE_KEY,
        "1"
      )
    } catch {
      // Ignore storage errors.
    }
  }, [])

  useEffect(() => {
    const normalizedWorkspaceTag =
      typeof workspaceTag === "string" ? workspaceTag.trim() : ""

    if (!normalizedWorkspaceTag) {
      setWorkspaceSearchNotes([])
      return
    }

    let cancelled = false

    const loadWorkspaceSearchNotes = async () => {
      try {
        const response = await bgRequest<WorkspaceNotesSearchResponse>({
          path: buildWorkspaceNotesSearchPath(normalizedWorkspaceTag),
          method: "GET"
        })
        if (cancelled) return

        const noteById = new Map<number, WorkspaceGlobalSearchNoteDocument>()
        for (const note of pickNotesArray(response)) {
          if (typeof note.id !== "number" || !Number.isFinite(note.id)) {
            continue
          }
          noteById.set(note.id, {
            id: note.id,
            title: note.title || "",
            content: note.content || "",
            keywords: extractNoteKeywords(note),
            isDraft: false
          })
        }

        setWorkspaceSearchNotes(Array.from(noteById.values()))
      } catch {
        if (!cancelled) {
          setWorkspaceSearchNotes([])
        }
      }
    }

    void loadWorkspaceSearchNotes()

    return () => {
      cancelled = true
    }
  }, [workspaceId, workspaceTag])

  const hydrateAndFocusNote = React.useCallback(
    async (result: WorkspaceGlobalSearchResult) => {
      if (
        result.noteId != null &&
        Number.isFinite(result.noteId) &&
        currentNote?.id !== result.noteId
      ) {
        try {
          const note = await bgRequest<WorkspaceNoteSearchItem>({
            path: buildWorkspaceNotePath(result.noteId),
            method: "GET"
          })
          loadNote({
            id: result.noteId,
            title: note.title || "",
            content: note.content || "",
            keywords: extractNoteKeywords(note),
            version:
              typeof note.version === "number" && Number.isFinite(note.version)
                ? note.version
                : undefined
          })
        } catch {
          // Keep focus behavior even when loading the target note fails.
        }
      }

      focusWorkspaceNote(result.noteField || "content")
    },
    [currentNote?.id, focusWorkspaceNote, loadNote]
  )

  const focusWorkspacePane = React.useCallback(
    (pane: WorkspaceTabKey) => {
      if (pane === "sources") {
        if (isMobile) {
          setActiveTab("sources")
        } else if (isDesktopLayout()) {
          setLeftPaneCollapsed(false)
        } else {
          setLeftDrawerOpen(true)
        }

        window.setTimeout(() => {
          const panel = document.getElementById("workspace-sources-panel")
          const firstFocusable = panel?.querySelector<HTMLElement>(
            "button, input, textarea, [tabindex]:not([tabindex='-1'])"
          )
          firstFocusable?.focus()
        }, 0)
        return
      }

      if (pane === "studio") {
        if (isMobile) {
          setActiveTab("studio")
        } else if (isDesktopLayout()) {
          setRightPaneCollapsed(false)
        } else {
          setRightDrawerOpen(true)
        }

        window.setTimeout(() => {
          const panel = document.getElementById("workspace-studio-panel")
          const firstFocusable = panel?.querySelector<HTMLElement>(
            "button, input, textarea, [tabindex]:not([tabindex='-1'])"
          )
          firstFocusable?.focus()
        }, 0)
        return
      }

      if (isMobile) {
        setActiveTab("chat")
      }
      window.setTimeout(() => {
        const chatInput = document.querySelector<HTMLElement>(
          "#workspace-main-content textarea"
        )
        if (chatInput) {
          chatInput.focus()
          return
        }
        const main = document.getElementById("workspace-main-content")
        const firstFocusable = main?.querySelector<HTMLElement>(
          "button, input, textarea, [tabindex]:not([tabindex='-1'])"
        )
        firstFocusable?.focus()
      }, 0)
    },
    [isMobile, setLeftPaneCollapsed, setRightPaneCollapsed]
  )

  const handleOpenSplitWorkspace = React.useCallback(() => {
    if (readyEffectiveSelectedSourceEntries.length === 0) {
      focusWorkspacePane("sources")
      messageApi.info(
        t(
          "playground:workspace.splitNoSelection",
          "Select at least one source before splitting this workspace."
        )
      )
      return
    }

    openTransferSourcesModal({
      entryPoint: "header",
      selectedSourceIds: effectiveSelectedSourceEntries.map((source) => source.id),
      eligibleSelectedSourceIds: readyEffectiveSelectedSourceEntries.map(
        (source) => source.id
      ),
      totalSelectedCount: effectiveSelectedSourceEntries.length,
      hiddenSelectedCount,
      ineligibleSelectedCount
    })
  }, [
    effectiveSelectedSourceEntries,
    focusWorkspacePane,
    hiddenSelectedCount,
    ineligibleSelectedCount,
    messageApi,
    openTransferSourcesModal,
    readyEffectiveSelectedSourceEntries,
    t
  ])

  const focusNewNoteTitle = React.useCallback(() => {
    focusWorkspacePane("studio")
    window.setTimeout(() => {
      focusWorkspaceNote("title")
    }, 0)
  }, [focusWorkspaceNote, focusWorkspacePane])

  const startNewNoteWithUndo = React.useCallback(() => {
    const previousNote = {
      ...currentNote,
      keywords: [...currentNote.keywords]
    }
    const undoHandle = scheduleWorkspaceUndoAction({
      apply: () => {
        clearCurrentNote()
        focusNewNoteTitle()
      },
      undo: () => {
        setCurrentNote(previousNote)
        focusNewNoteTitle()
      }
    })

    const undoMessageKey = `workspace-note-shortcut-undo-${undoHandle.id}`
    const maybeOpen = (messageApi as { open?: (config: unknown) => void }).open
    const messageConfig = {
      key: undoMessageKey,
      type: "warning",
      duration: WORKSPACE_UNDO_WINDOW_MS / 1000,
      content: t("playground:studio.noteCleared", "Note cleared."),
      btn: (
        <Button
          size="small"
          type="link"
          onClick={() => {
            if (undoWorkspaceAction(undoHandle.id)) {
              messageApi.success(
                t("playground:studio.noteRestored", "Note restored")
              )
            }
            messageApi.destroy(undoMessageKey)
          }}
        >
          {t("common:undo", "Undo")}
        </Button>
      )
    }

    if (typeof maybeOpen === "function") {
      maybeOpen(messageConfig)
    } else {
      const maybeWarning = (
        messageApi as { warning?: (content: string) => void }
      ).warning
      if (typeof maybeWarning === "function") {
        maybeWarning(t("playground:studio.noteCleared", "Note cleared."))
      }
    }
  }, [clearCurrentNote, currentNote, focusNewNoteTitle, messageApi, setCurrentNote, t])

  const focusSearchResult = React.useCallback(
    (result: WorkspaceGlobalSearchResult) => {
      closeGlobalSearch()

      if (result.domain === "source" && result.sourceId) {
        if (isMobile) {
          setActiveTab("sources")
        } else if (isDesktopLayout()) {
          setLeftPaneCollapsed(false)
        } else {
          setLeftDrawerOpen(true)
        }

        window.setTimeout(() => {
          focusSourceById(result.sourceId!)
        }, 0)
        return
      }

      if (result.domain === "chat" && result.chatMessageId) {
        if (isMobile) {
          setActiveTab("chat")
        }
        window.setTimeout(() => {
          focusChatMessageById(result.chatMessageId!)
        }, 0)
        return
      }

      if (result.domain === "note") {
        if (isMobile) {
          setActiveTab("studio")
        } else if (isDesktopLayout()) {
          setRightPaneCollapsed(false)
        } else {
          setRightDrawerOpen(true)
        }

        window.setTimeout(() => {
          void hydrateAndFocusNote(result)
        }, 0)
      }
    },
    [
      closeGlobalSearch,
      focusChatMessageById,
      focusSourceById,
      hydrateAndFocusNote,
      isMobile,
      setLeftPaneCollapsed,
      setRightPaneCollapsed
    ]
  )

  const handleSearchInputKeyDown = (
    event: React.KeyboardEvent<HTMLInputElement>
  ) => {
    if (event.key === "Escape") {
      event.preventDefault()
      event.stopPropagation()
      closeGlobalSearch()
      return
    }

    if (event.key === "ArrowDown") {
      event.preventDefault()
      if (globalSearchResults.length === 0) return
      setActiveSearchResultIndex((prev) =>
        prev + 1 >= globalSearchResults.length ? 0 : prev + 1
      )
      return
    }

    if (event.key === "ArrowUp") {
      event.preventDefault()
      if (globalSearchResults.length === 0) return
      setActiveSearchResultIndex((prev) =>
        prev - 1 < 0 ? globalSearchResults.length - 1 : prev - 1
      )
      return
    }

    if (event.key === "Enter") {
      event.preventDefault()
      const selectedResult = globalSearchResults[activeSearchResultIndex]
      if (selectedResult) {
        focusSearchResult(selectedResult)
      }
    }
  }

  // Initialize workspace on mount if not already initialized — use ref to keep dep stable
  const initRef = React.useRef(initializeWorkspace)
  initRef.current = initializeWorkspace
  useEffect(() => {
    if (!isStoreHydrated) return
    if (!workspaceId) {
      initRef.current()
    }
  }, [isStoreHydrated, workspaceId])

  useEffect(() => {
    if (!statusGuardrailsEnabled) return
    if (!isStoreHydrated) return
    const pendingRefreshLoop = consumeWorkspaceRefreshLoopSignal()
    if (!pendingRefreshLoop) return
    void trackWorkspacePlaygroundTelemetry({
      type: "confusion_refresh_loop",
      workspace_id: workspaceId || null,
      refresh_count: pendingRefreshLoop.count,
      window_ms: pendingRefreshLoop.windowMs
    })
  }, [isStoreHydrated, statusGuardrailsEnabled, workspaceId])

  useEffect(() => {
    void refreshWorkspaceStorageUsage()
  }, [refreshWorkspaceStorageUsage])

  useEffect(() => {
    void refreshAccountStorageUsage()
  }, [refreshAccountStorageUsage])

  useEffect(() => {
    if (typeof window === "undefined") return
    const timer = window.setTimeout(() => {
      void refreshWorkspaceStorageUsage()
    }, WORKSPACE_STORAGE_USAGE_REFRESH_DELAY_MS)
    return () => {
      window.clearTimeout(timer)
    }
  }, [
    currentNote.content,
    currentNote.keywords.length,
    currentNote.title,
    generatedArtifacts.length,
    refreshWorkspaceStorageUsage,
    selectedSourceIds.length,
    sources.length,
    workspaceChatMessages.length,
    workspaceId
  ])

  useEffect(() => {
    if (typeof window === "undefined") return
    const timer = window.setTimeout(() => {
      void refreshAccountStorageUsage()
    }, ACCOUNT_STORAGE_USAGE_REFRESH_DELAY_MS)
    return () => {
      window.clearTimeout(timer)
    }
  }, [
    generatedArtifacts.length,
    refreshAccountStorageUsage,
    sources.length,
    workspaceId
  ])

  useEffect(() => {
    if (!workspaceId) return

    let isActive = true

    const applyPrefill = async () => {
      const payload = await consumeWorkspacePlaygroundPrefill()
      if (!payload || !isActive) return
      if (payload.kind !== "knowledge_qa_thread") return

      const sourceCandidates = payload.sources
        .filter((source) => typeof source.mediaId === "number")
        .map((source) => ({
          mediaId: source.mediaId as number,
          title: source.title,
          type: source.type
        }))

      if (sourceCandidates.length > 0) {
        addSources(sourceCandidates)

        const stateAfterAdd = useWorkspaceStore.getState()
        const prefillSourceIds = stateAfterAdd.sources
          .filter((source) =>
            sourceCandidates.some((candidate) => candidate.mediaId === source.mediaId)
          )
          .map((source) => source.id)
        const mergedSelectedIds = new Set([
          ...stateAfterAdd.selectedSourceIds,
          ...prefillSourceIds
        ])
        if (mergedSelectedIds.size > 0) {
          setSelectedSourceIds(Array.from(mergedSelectedIds))
        }
      }

      const noteContent = buildKnowledgeQaSeedNote(payload)
      if (noteContent.trim().length > 0) {
        const titleBase =
          payload.query.trim().length > 0 ? payload.query.trim() : "Knowledge QA import"
        captureToCurrentNote({
          title: `Knowledge QA: ${titleBase.slice(0, 80)}`,
          content: noteContent,
          mode: "append"
        })
      }
    }

    void applyPrefill()

    return () => {
      isActive = false
    }
  }, [addSources, captureToCurrentNote, setSelectedSourceIds, workspaceId])

  useEffect(() => {
    if (!isStoreHydrated) return
    if (!workspaceId) return
    if (onboardingInitializedRef.current) return

    onboardingInitializedRef.current = true

    if (typeof window === "undefined") return
    try {
      const dismissed = window.localStorage.getItem(
        WORKSPACE_ONBOARDING_DISMISSED_STORAGE_KEY
      )
      if (dismissed !== "1") {
        setShowTutorialPrompt(true)
      }
    } catch {
      // Ignore storage errors
    }
  }, [isStoreHydrated, workspaceId])

  useEffect(() => {
    if (typeof window === "undefined") return

    const handleKeyboardShortcut = (event: KeyboardEvent) => {
      const key = event.key.toLowerCase()
      const hasModifier = event.metaKey || event.ctrlKey

      if (hasModifier && key === "k") {
        event.preventDefault()
        setGlobalSearchOpen(true)
        return
      }

      if (hasModifier && key === "z" && !event.shiftKey) {
        event.preventDefault()
        undoLatestWorkspaceAction()
        return
      }

      if (hasModifier && !event.shiftKey && key === "1") {
        event.preventDefault()
        focusWorkspacePane("sources")
        return
      }

      if (hasModifier && !event.shiftKey && key === "2") {
        event.preventDefault()
        focusWorkspacePane("chat")
        return
      }

      if (hasModifier && !event.shiftKey && key === "3") {
        event.preventDefault()
        focusWorkspacePane("studio")
        return
      }

      if (hasModifier && event.shiftKey && key === "n") {
        event.preventDefault()
        createNewWorkspace()
        return
      }

      if (hasModifier && !event.shiftKey && key === "n") {
        event.preventDefault()
        const hasNoteContent =
          currentNote.title.trim().length > 0 ||
          currentNote.content.trim().length > 0 ||
          currentNote.keywords.length > 0

        const startNewNote = (withUndo: boolean) => {
          if (withUndo) {
            startNewNoteWithUndo()
            return
          }
          clearCurrentNote()
          focusNewNoteTitle()
        }

        if (currentNote.isDirty || hasNoteContent) {
          Modal.confirm({
            title: t("playground:studio.newNoteTitle", "Start a new note?"),
            content: t(
              "playground:studio.newNoteMessage",
              "This clears your current note draft."
            ),
            okText: t("playground:studio.newNote", "New note"),
            cancelText: t("common:cancel", "Cancel"),
            onOk: () => startNewNote(true)
          })
          return
        }

        startNewNote(false)
        return
      }

      if (event.key === "Escape") {
        event.preventDefault()
        closeGlobalSearch()
        return
      }

      if (
        event.key === "?" &&
        !hasModifier &&
        !event.altKey
      ) {
        const target = event.target as HTMLElement | null
        const tag = target?.tagName?.toLowerCase()
        if (
          tag === "input" ||
          tag === "textarea" ||
          tag === "select" ||
          target?.isContentEditable
        ) {
          return
        }
        event.preventDefault()
        setShowShortcutsModal(true)
      }
    }

    window.addEventListener("keydown", handleKeyboardShortcut)
    return () => {
      window.removeEventListener("keydown", handleKeyboardShortcut)
    }
  }, [
    clearCurrentNote,
    closeGlobalSearch,
    createNewWorkspace,
    currentNote.content,
    currentNote.isDirty,
    currentNote.keywords.length,
    currentNote.title,
    focusNewNoteTitle,
    focusWorkspaceNote,
    focusWorkspacePane,
    startNewNoteWithUndo,
    t
  ])

  useEffect(() => {
    if (!statusGuardrailsEnabled) return
    if (typeof window === "undefined") return

    const handleQuotaExceeded = (event: Event) => {
      const customEvent = event as CustomEvent<WorkspaceStorageQuotaEventDetail>
      if (customEvent.detail?.key !== WORKSPACE_STORAGE_KEY) return
      setShowStorageQuotaWarning(true)
      void refreshWorkspaceStorageUsage()
      void trackWorkspacePlaygroundTelemetry({
        type: "quota_warning_seen",
        workspace_id: workspaceId || null,
        reason:
          typeof customEvent.detail?.reason === "string"
            ? customEvent.detail.reason
            : null
      })
    }

    window.addEventListener(
      WORKSPACE_STORAGE_QUOTA_EVENT,
      handleQuotaExceeded as EventListener
    )
    return () => {
      window.removeEventListener(
        WORKSPACE_STORAGE_QUOTA_EVENT,
        handleQuotaExceeded as EventListener
      )
    }
  }, [refreshWorkspaceStorageUsage, statusGuardrailsEnabled, workspaceId])

  const surfaceCrossTabSyncWarning = React.useCallback(
    (oldValue?: string | null, newValue?: string | null) => {
      if (!statusGuardrailsEnabled) return
      const now = Date.now()
      const shouldShow = shouldSurfaceWorkspaceConflictNotice(
        lastCrossTabSyncWarningRef.current,
        now,
        WORKSPACE_CONFLICT_NOTICE_THROTTLE_MS
      )
      if (!shouldShow) return

      lastCrossTabSyncWarningRef.current = now
      const changedFields = deriveWorkspaceConflictFieldDiff(oldValue, newValue)
      setCrossTabChangedFields(changedFields)
      void trackWorkspacePlaygroundTelemetry({
        type: "conflict_modal_opened",
        workspace_id: workspaceId || null,
        changed_fields_count: changedFields.length
      })
      setShowCrossTabSyncWarning(true)
    },
    [statusGuardrailsEnabled, workspaceId]
  )

  useEffect(() => {
    if (!statusGuardrailsEnabled) return
    if (typeof window === "undefined") return

    const handleStorageEvent = (event: StorageEvent) => {
      if (event.key !== WORKSPACE_STORAGE_KEY) return
      if (event.newValue === event.oldValue) return
      if (event.storageArea && event.storageArea !== window.localStorage) return
      void refreshWorkspaceStorageUsage()
      surfaceCrossTabSyncWarning(event.oldValue, event.newValue)
    }

    window.addEventListener("storage", handleStorageEvent)
    return () => {
      window.removeEventListener("storage", handleStorageEvent)
    }
  }, [
    refreshWorkspaceStorageUsage,
    statusGuardrailsEnabled,
    surfaceCrossTabSyncWarning
  ])

  useEffect(() => {
    if (!statusGuardrailsEnabled) return
    if (typeof window === "undefined") return
    if (!isWorkspaceBroadcastSyncEnabled()) return
    if (typeof BroadcastChannel === "undefined") return

    const channel = new BroadcastChannel(WORKSPACE_STORAGE_CHANNEL_NAME)
    const handleBroadcastUpdate = (event: MessageEvent<unknown>) => {
      if (!isWorkspaceBroadcastUpdateMessage(event.data)) return
      if (event.data.key !== WORKSPACE_STORAGE_KEY) return
      void refreshWorkspaceStorageUsage()
      surfaceCrossTabSyncWarning()
    }

    channel.addEventListener("message", handleBroadcastUpdate)
    return () => {
      channel.removeEventListener("message", handleBroadcastUpdate)
      channel.close()
    }
  }, [
    refreshWorkspaceStorageUsage,
    statusGuardrailsEnabled,
    surfaceCrossTabSyncWarning
  ])

  useEffect(() => {
    setActiveSearchResultIndex(0)
  }, [globalSearchOpen, globalSearchQuery, globalSearchResults.length])

  useEffect(() => {
    if (!workspaceId) return

    const previousWorkspaceId = previousWorkspaceIdRef.current
    previousWorkspaceIdRef.current = workspaceId

    if (!previousWorkspaceId || previousWorkspaceId === workspaceId) {
      return
    }

    setShowWorkspaceTransitionCue(true)
    if (workspaceTransitionTimerRef.current !== null) {
      window.clearTimeout(workspaceTransitionTimerRef.current)
    }
    workspaceTransitionTimerRef.current = window.setTimeout(() => {
      setShowWorkspaceTransitionCue(false)
      workspaceTransitionTimerRef.current = null
    }, WORKSPACE_SWITCH_TRANSITION_MS)
  }, [workspaceId])

  useEffect(() => {
    if (!statusGuardrailsEnabled) return
    if (!isStoreHydrated) return
    if (processingMediaIds.length === 0) return

    let cancelled = false

    const pollStatuses = async () => {
      void trackWorkspacePlaygroundTelemetry({
        type: "source_status_polled",
        workspace_id: workspaceId || null,
        processing_count: processingMediaIds.length
      })
      await Promise.all(
        processingMediaIds.map(async (mediaId) => {
          try {
            const detail = await tldwClient.getMediaDetails(mediaId, {
              include_content: true,
              include_versions: false,
              include_version_content: false
            })
            if (cancelled) return

            if (isMediaLikelyReadyForRag(detail)) {
              setSourceStatusByMediaId(mediaId, "ready")
              delete sourceStatusFailureRef.current[mediaId]
              void trackWorkspacePlaygroundTelemetry({
                type: "source_status_ready",
                workspace_id: workspaceId || null,
                media_id: mediaId
              })
            }
          } catch (error) {
            if (cancelled) return

            const nextFailureCount =
              (sourceStatusFailureRef.current[mediaId] || 0) + 1
            sourceStatusFailureRef.current[mediaId] = nextFailureCount

            const { transient, message } = isTransientSourceStatusError(error)
            if (!transient && nextFailureCount >= 2) {
              setSourceStatusByMediaId(mediaId, "error", message)
              delete sourceStatusFailureRef.current[mediaId]
            }
          }
        })
      )
    }

    void pollStatuses()
    const timer = window.setInterval(
      () => void pollStatuses(),
      WORKSPACE_SOURCE_STATUS_POLL_INTERVAL_MS
    )
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [
    isStoreHydrated,
    processingMediaIds,
    setSourceStatusByMediaId,
    statusGuardrailsEnabled,
    workspaceId
  ])

  useEffect(() => {
    return () => {
      if (workspaceTransitionTimerRef.current !== null) {
        window.clearTimeout(workspaceTransitionTimerRef.current)
      }
    }
  }, [])

  const handleToggleLeftPane = () => {
    if (isMobile) {
      setLeftDrawerOpen(!leftDrawerOpen)
    } else {
      setLeftPaneCollapsed(leftPaneOpen)
    }
  }

  const handleToggleRightPane = () => {
    if (isMobile) {
      setRightDrawerOpen(!rightDrawerOpen)
    } else {
      setRightPaneCollapsed(rightPaneOpen)
    }
  }

  const handleReloadWorkspaceFromSyncWarning = () => {
    if (statusGuardrailsEnabled) {
      const refreshAttempt = recordWorkspaceRefreshLoopAttempt()
      if (
        refreshAttempt &&
        refreshAttempt.count >= WORKSPACE_REFRESH_LOOP_THRESHOLD
      ) {
        persistWorkspaceRefreshLoopSignal(refreshAttempt)
      }
    }
    if (typeof window !== "undefined") {
      try {
        window.location.reload()
      } catch (error) {
        try {
          window.sessionStorage.removeItem(
            WORKSPACE_REFRESH_LOOP_PENDING_SIGNAL_SESSION_KEY
          )
        } catch {
          // Ignore session storage cleanup errors.
        }
        console.warn("Workspace reload unavailable", error)
      }
    }
  }

  const handleDismissCrossTabSyncWarning = () => {
    setShowCrossTabSyncWarning(false)
    setCrossTabChangedFields([])
  }

  const handleForkWorkspaceFromSyncWarning = () => {
    duplicateWorkspace(workspaceId)
    handleDismissCrossTabSyncWarning()
  }

  const getSearchDomainLabel = (domain: WorkspaceGlobalSearchResult["domain"]) => {
    switch (domain) {
      case "source":
        return t("playground:search.sources", "Sources")
      case "chat":
        return t("playground:search.chat", "Chat")
      case "note":
        return t("playground:search.notes", "Notes")
      default:
        return domain
    }
  }

  // Mobile tab items with badges
  const mobileTabItems = [
    {
      key: "sources",
      label: (
        <span className="flex items-center gap-1.5">
          <FileText className="h-4 w-4" />
          <span>{t("playground:sources.title", "Sources")}</span>
          {selectedSourceIds.length > 0 && (
            <span className="ml-1 rounded-full border border-border bg-surface2 px-1.5 py-0.5 text-xs text-text">
              {selectedSourceIds.length}
            </span>
          )}
        </span>
      ),
      children: renderSourcesPane()
    },
    {
      key: "chat",
      label: (
        <span className="flex items-center gap-1.5">
          <MessageSquare className="h-4 w-4" />
          <span>{t("playground:chat.title", "Chat")}</span>
        </span>
      ),
      children: (
        <ChatPane
          provenanceEnabled={provenanceEnabled}
          statusGuardrailsEnabled={statusGuardrailsEnabled}
          contentWidthMode="full"
        />
      )
    },
    {
      key: "studio",
      label: (
        <span className="flex items-center gap-1.5">
          <Sparkles className="h-4 w-4" />
          <span>{t("playground:studio.title", "Studio")}</span>
          {generatedArtifacts.length > 0 && (
            <span className="ml-1 rounded-full border border-border bg-surface2 px-1.5 py-0.5 text-xs text-text">
              {generatedArtifacts.length}
            </span>
          )}
        </span>
      ),
      children: renderStudioPane()
    }
  ]

  function renderSourcesPane(options?: { onHide?: () => void }) {
    return (
      <Suspense
        fallback={<WorkspacePaneFallback testId="workspace-sources-pane" />}
      >
        <SourcesPane
          onHide={options?.onHide}
          onOpenTransferSources={openTransferSourcesModal}
          sourceListViewState={sourceListViewState}
          onPatchSourceListViewState={patchSourceListViewState}
          onResetAdvancedSourceFilters={resetAdvancedSourceFilters}
          statusGuardrailsEnabled={statusGuardrailsEnabled}
        />
      </Suspense>
    )
  }

  function renderStudioPane(options?: { onHide?: () => void }) {
    return (
      <Suspense
        fallback={<WorkspacePaneFallback testId="workspace-studio-pane" />}
      >
        <StudioPane onHide={options?.onHide} />
      </Suspense>
    )
  }

  const sessionSummaryItems = [
    {
      key: "sources",
      label: t("playground:sources.title", "Sources"),
      count: selectedSourceIds.length
    },
    {
      key: "outputs",
      label: t("playground:studio.generatedOutputs", "Generated Outputs"),
      count: generatedArtifacts.length
    }
  ]

  if (!isStoreHydrated) {
    return <WorkspacePlaygroundSkeleton isMobile={isMobile} />
  }

  const tutorialPromptBanner = showTutorialPrompt ? (
    <div className="mx-4 mt-2 flex items-center justify-between rounded-lg border border-primary/20 bg-primary/5 px-4 py-2.5 text-sm">
      <span><strong>New here?</strong> Take a quick tour of the workspace.</span>
      <div className="flex gap-2">
        <button
          type="button"
          onClick={() => {
            startTutorial("workspace-playground-basics")
            dismissOnboardingOverlay()
            setShowTutorialPrompt(false)
          }}
          className="rounded-md bg-primary px-3 py-1 text-xs font-medium text-white hover:bg-primaryStrong transition-colors"
        >
          Start tour
        </button>
        <button
          type="button"
          onClick={() => {
            dismissOnboardingOverlay()
            setShowTutorialPrompt(false)
          }}
          className="rounded-md border border-border px-3 py-1 text-xs text-text-muted hover:bg-surface2 transition-colors"
        >
          Dismiss
        </button>
      </div>
    </div>
  ) : null

  return (
    <SharedWorkspaceProvider
      shareId={sharedShareId}
      ownerUserId={sharedOwnerUserId}
      accessLevel={sharedAccessLevel}
      allowClone={sharedAllowClone}
    >
    <div className="relative flex h-full min-h-0 flex-col overflow-hidden bg-[radial-gradient(circle_at_top_left,var(--surface-2),var(--bg)_45%)] text-text">
      {messageContextHolder}
      <a
        href="#workspace-main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:left-3 focus:top-2 focus:z-[60] focus:rounded focus:bg-surface focus:px-3 focus:py-1.5 focus:text-sm focus:shadow-card"
      >
        {t("playground:workspace.skipToMain", "Skip to chat content")}
      </a>
      <a
        href="#workspace-sources-panel"
        className="sr-only focus:not-sr-only focus:absolute focus:left-3 focus:top-12 focus:z-[60] focus:rounded focus:bg-surface focus:px-3 focus:py-1.5 focus:text-sm focus:shadow-card"
      >
        {t("playground:workspace.skipToSources", "Skip to sources panel")}
      </a>
      <a
        href="#workspace-studio-panel"
        className="sr-only focus:not-sr-only focus:absolute focus:left-3 focus:top-[5.5rem] focus:z-[60] focus:rounded focus:bg-surface focus:px-3 focus:py-1.5 focus:text-sm focus:shadow-card"
      >
        {t("playground:workspace.skipToStudio", "Skip to studio panel")}
      </a>

      {statusGuardrailsEnabled &&
        (showStorageQuotaWarning || showStorageHighUsageWarning || showCrossTabSyncWarning) && (
        <div className="space-y-2 border-b border-border bg-surface px-3 py-2">
          {showStorageQuotaWarning && (
            <div
              data-testid="workspace-storage-quota-banner"
              className="flex flex-wrap items-center justify-between gap-2 rounded border border-warning/40 bg-warning/10 px-3 py-2 text-sm text-text"
              role="status"
              aria-live="polite"
            >
              <span>
                {t(
                  "playground:workspace.storageQuotaExceeded",
                  "Workspace data is too large to save locally. Delete older outputs or sources to reduce size."
                )}
              </span>
              <button
                type="button"
                className="rounded border border-border px-2 py-1 text-xs font-medium hover:bg-surface2"
                onClick={() => setShowStorageQuotaWarning(false)}
              >
                {t("common:dismiss", "Dismiss")}
              </button>
            </div>
          )}

          {showStorageHighUsageWarning && (
            <div
              data-testid="workspace-storage-high-usage-banner"
              className="flex flex-wrap items-center justify-between gap-2 rounded border border-warning/40 bg-warning/10 px-3 py-2.5 text-sm text-text"
              role="status"
              aria-live="polite"
            >
              <span className="flex items-center gap-2">
                <AlertTriangle className="h-4 w-4 shrink-0 text-warning" />
                <span>
                  {t(
                    "playground:workspace.storageHighUsage",
                    "Storage is {{percent}}% full. Consider archiving unused workspaces to free space.",
                    { percent: storageUsagePercent }
                  )}
                </span>
              </span>
              <button
                type="button"
                className="rounded border border-border px-2 py-1 text-xs font-medium hover:bg-surface2"
                onClick={() => setStorageHighUsageDismissed(true)}
              >
                {t("common:dismiss", "Dismiss")}
              </button>
            </div>
          )}

          {showCrossTabSyncWarning && (
            <div
              data-testid="workspace-storage-sync-banner"
              className="flex flex-wrap items-center justify-between gap-2 rounded border border-primary/40 bg-primary/10 px-3 py-2 text-sm text-text"
              role="alert"
              aria-live="assertive"
            >
              <div className="space-y-1">
                <span>
                  {t(
                    "playground:workspace.externalUpdate",
                    "This workspace changed in another tab."
                  )}
                </span>
                {crossTabChangedFields.length > 0 && (
                  <p className="text-xs text-text-muted">
                    {t(
                      "playground:workspace.externalUpdateChangedFields",
                      "Changed fields: {{fields}}",
                      {
                        fields: crossTabChangedFields.join(", ")
                      }
                    )}
                  </p>
                )}
                <p className="text-xs font-medium text-primary">
                  {t(
                    "playground:workspace.externalUpdateRecommendation",
                    "Recommended: Reload to get the latest version."
                  )}
                </p>
                <p className="text-xs text-text-muted">
                  {t(
                    "playground:workspace.externalUpdateActionHint",
                    "Keep this version ignores the update. Save as new workspace copies your current state."
                  )}
                </p>
              </div>
              <div className="flex items-center gap-1.5">
                <button
                  type="button"
                  className="rounded border border-primary/40 bg-primary px-2 py-1 text-xs font-medium text-white hover:opacity-90"
                  onClick={handleReloadWorkspaceFromSyncWarning}
                  title={t("playground:workspace.useLatestTooltip", "Discard your changes in this tab and load the version from the other tab")}
                >
                  {t("playground:workspace.useLatest", "Reload from other tab")}
                </button>
                <button
                  type="button"
                  className="rounded border border-border px-2 py-1 text-xs font-medium hover:bg-surface2"
                  onClick={handleForkWorkspaceFromSyncWarning}
                  title={t("playground:workspace.forkCopyTooltip", "Duplicate your current state into a brand-new workspace")}
                >
                  {t("playground:workspace.forkCopy", "Save as new workspace")}
                </button>
                <button
                  type="button"
                  className="rounded border border-border px-2 py-1 text-xs font-medium hover:bg-surface2"
                  onClick={handleDismissCrossTabSyncWarning}
                  title={t("playground:workspace.keepMineTooltip", "Dismiss this notification and keep working with your current version")}
                >
                  {t("playground:workspace.keepMine", "Keep this version")}
                </button>
              </div>
            </div>
          )}
        </div>
      )}


      {/* Mobile vs desktop layout */}
      {isMobile ? (
        <>
          <WorkspaceHeader
            leftPaneOpen={false}
            rightPaneOpen={false}
            onToggleLeftPane={handleToggleLeftPane}
            onToggleRightPane={handleToggleRightPane}
            onOpenSplitWorkspace={handleOpenSplitWorkspace}
            hideToggles
            storageUsedBytes={workspaceStorageUsage.usedBytes}
            storageQuotaBytes={workspaceStorageUsage.quotaBytes}
            storageOriginUsedBytes={workspaceStorageUsage.originUsedBytes ?? undefined}
            storageOriginQuotaBytes={workspaceStorageUsage.originQuotaBytes ?? undefined}
            storageAccountUsedBytes={workspaceStorageUsage.accountUsedBytes ?? undefined}
            storageAccountQuotaBytes={workspaceStorageUsage.accountQuotaBytes ?? undefined}
            provenanceEnabled={provenanceEnabled}
            statusGuardrailsEnabled={statusGuardrailsEnabled}
          />

          <WorkspaceBanner
            banner={workspaceBanner}
            workspaceName={workspaceName}
            isMobile
          />
          <SharedWorkspaceBanner />

          {tutorialPromptBanner}

          <WorkspaceStatusBar
            storageUsedBytes={workspaceStorageUsage.usedBytes}
            storageQuotaBytes={workspaceStorageUsage.quotaBytes}
            activeOperations={activeWorkspaceOperations}
            statusGuardrailsEnabled={statusGuardrailsEnabled}
          />

          <Tabs
            activeKey={activeTab}
            onChange={(key) => setActiveTab(key as WorkspaceTabKey)}
            items={mobileTabItems}
            centered
            destroyOnHidden
            className="flex-1 [&_.ant-tabs-content]:h-full [&_.ant-tabs-content-holder]:flex-1 [&_.ant-tabs-tabpane]:h-full"
            tabBarStyle={{ marginBottom: 0, borderBottom: "1px solid var(--border)" }}
          />
        </>
      ) : (
        <>
          <WorkspaceHeader
            leftPaneOpen={!!leftPaneOpen}
            rightPaneOpen={!!rightPaneOpen}
            onToggleLeftPane={handleToggleLeftPane}
            onToggleRightPane={handleToggleRightPane}
            onOpenSplitWorkspace={handleOpenSplitWorkspace}
            storageUsedBytes={workspaceStorageUsage.usedBytes}
            storageQuotaBytes={workspaceStorageUsage.quotaBytes}
            storageOriginUsedBytes={workspaceStorageUsage.originUsedBytes ?? undefined}
            storageOriginQuotaBytes={workspaceStorageUsage.originQuotaBytes ?? undefined}
            storageAccountUsedBytes={workspaceStorageUsage.accountUsedBytes ?? undefined}
            storageAccountQuotaBytes={workspaceStorageUsage.accountQuotaBytes ?? undefined}
            provenanceEnabled={provenanceEnabled}
            statusGuardrailsEnabled={statusGuardrailsEnabled}
          />

          <WorkspaceBanner
            banner={workspaceBanner}
            workspaceName={workspaceName}
            isMobile={false}
          />
          <SharedWorkspaceBanner />

          {tutorialPromptBanner}

          <div className="flex min-h-0 flex-1 gap-2 px-2 py-2">
            {leftPaneOpen && (
              <>
                <aside
                  id="workspace-sources-panel"
                  role="complementary"
                  aria-label={t("playground:workspace.sourcesPanel", "Sources panel")}
                  className="hidden shrink-0 overflow-hidden rounded-xl border border-border/80 bg-surface/90 shadow-card lg:flex lg:flex-col"
                  style={{ width: leftPaneWidth }}
                >
                  {renderSourcesPane({
                    onHide: () => setLeftPaneCollapsed(true)
                  })}
                </aside>
                <PaneResizer
                  pane="left"
                  width={leftPaneWidth}
                  onResize={setLeftPaneWidth}
                  onReset={() => setLeftPaneWidth(DEFAULT_LEFT_WIDTH)}
                />
              </>
            )}

            <Drawer
              title={
                <span className="flex items-center gap-2">
                  <FileText className="h-4 w-4" />
                  {t("playground:sources.title", "Sources")}
                </span>
              }
              placement="left"
              onClose={() => setLeftDrawerOpen(false)}
              open={leftDrawerOpen}
              mask={false}
              className="lg:hidden"
              styles={{ wrapper: { width: 320 }, body: { padding: 0 } }}
            >
              {renderSourcesPane()}
            </Drawer>

            <main
              id="workspace-main-content"
              className="flex min-w-0 flex-1 overflow-hidden rounded-xl border border-border/80 bg-surface/90 shadow-card"
            >
              <ChatPane
                provenanceEnabled={provenanceEnabled}
                statusGuardrailsEnabled={statusGuardrailsEnabled}
                contentWidthMode={desktopChatContentWidthMode}
              />
            </main>

            {rightPaneOpen && (
              <>
                <PaneResizer
                  pane="right"
                  width={rightPaneWidth}
                  onResize={setRightPaneWidth}
                  onReset={() => setRightPaneWidth(DEFAULT_RIGHT_WIDTH)}
                />
                <aside
                  id="workspace-studio-panel"
                  role="complementary"
                  aria-label={t("playground:workspace.studioPanel", "Studio panel")}
                  className="hidden min-h-0 shrink-0 overflow-hidden rounded-xl border border-border/80 bg-surface/90 shadow-card lg:flex lg:flex-col"
                  style={{ width: rightPaneWidth }}
                >
                  {renderStudioPane({
                    onHide: () => setRightPaneCollapsed(true)
                  })}
                </aside>
              </>
            )}

            <Drawer
              title={
                <span className="flex items-center gap-2">
                  <Sparkles className="h-4 w-4" />
                  {t("playground:studio.title", "Studio")}
                </span>
              }
              placement="right"
              onClose={() => setRightDrawerOpen(false)}
              open={rightDrawerOpen}
              mask={false}
              className="lg:hidden"
              styles={{ wrapper: { width: 360 }, body: { padding: 0 } }}
            >
              {renderStudioPane()}
            </Drawer>
          </div>

          <WorkspaceStatusBar
            storageUsedBytes={workspaceStorageUsage.usedBytes}
            storageQuotaBytes={workspaceStorageUsage.quotaBytes}
            activeOperations={activeWorkspaceOperations}
            statusGuardrailsEnabled={statusGuardrailsEnabled}
          />
        </>
      )}

      <Modal
        title={
          <span className="flex items-center gap-2 text-base">
            <Search className="h-4 w-4" />
            {t("playground:search.title", "Search workspace")}
          </span>
        }
        open={globalSearchOpen}
        onCancel={closeGlobalSearch}
        footer={null}
        width={680}
        destroyOnHidden
        afterOpenChange={(open) => {
          if (!open) return
          window.setTimeout(() => {
            globalSearchInputRef.current?.focus()
          }, 0)
        }}
      >
        <div className="space-y-3">
          <Input
            ref={globalSearchInputRef}
            value={globalSearchQuery}
            onChange={(event) => setGlobalSearchQuery(event.target.value)}
            onKeyDown={handleSearchInputKeyDown}
            placeholder={t(
              "playground:search.placeholder",
              "Search sources, chat, and notes..."
            )}
            prefix={<Search className="h-4 w-4 text-text-muted" />}
            suffix={
              <span className="hidden items-center gap-0.5 text-xs text-text-muted sm:flex">
                <Command className="h-3 w-3" />K
              </span>
            }
          />
          <p className="text-xs text-text-muted mt-0.5">
            {t(
              "playground:search.prefixHint",
              "Tip: Type source: chat: or note: to filter by category"
            )}
          </p>
          <div className="flex items-center gap-1.5">
            <span className="text-xs text-text-subtle">
              {t("playground:search.filterBy", "Filter:")}
            </span>
            {(["source", "chat", "note"] as const).map((domain) => {
              const isActive = globalSearchQuery.trim().toLowerCase().startsWith(`${domain}:`)
              return (
                <button
                  key={domain}
                  type="button"
                  data-testid={`search-filter-chip-${domain}`}
                  onClick={() => {
                    if (isActive) {
                      // Remove prefix
                      setGlobalSearchQuery(
                        globalSearchQuery.replace(new RegExp(`^${domain}:\\s*`, "i"), "")
                      )
                    } else {
                      // Remove any existing prefix and add new one
                      const stripped = globalSearchQuery.replace(/^\w+:\s*/, "")
                      setGlobalSearchQuery(`${domain}: ${stripped}`)
                    }
                    globalSearchInputRef.current?.focus()
                  }}
                  className={`rounded-full px-2 py-0.5 text-[11px] font-medium transition ${
                    isActive
                      ? "bg-primary/15 text-primary"
                      : "bg-surface2 text-text-muted hover:bg-surface2/80 hover:text-text"
                  }`}
                >
                  {getSearchDomainLabel(domain)}
                </button>
              )
            })}
          </div>

          <div className="custom-scrollbar max-h-[360px] space-y-1 overflow-y-auto rounded-lg border border-border p-1">
            {globalSearchResults.length === 0 ? (
              <Empty
                image={Empty.PRESENTED_IMAGE_SIMPLE}
                description={
                  <span className="text-text-muted">
                    {globalSearchQuery.trim()
                      ? t(
                          "playground:search.noResults",
                          "No matching sources, messages, or notes."
                        )
                      : t(
                          "playground:search.empty",
                          "Start typing to search this workspace."
                        )}
                  </span>
                }
              >
                {!globalSearchQuery.trim() && (
                  <div className="mt-1 space-y-1 text-xs text-text-subtle">
                    <p>{t("playground:search.exampleTitle", "Try searching for:")}</p>
                    {[
                      t("playground:search.example1", "a keyword from your sources"),
                      t("playground:search.example2", "a question you asked in chat"),
                      t("playground:search.example3", "a note title or keyword")
                    ].map((example) => (
                      <button
                        key={example}
                        type="button"
                        className="block w-full rounded px-2 py-0.5 text-left italic hover:bg-surface2"
                        onClick={() => setGlobalSearchQuery(example)}
                      >
                        {example}
                      </button>
                    ))}
                  </div>
                )}
              </Empty>
            ) : (
              globalSearchResults.map((result, index) => {
                const isActive = index === activeSearchResultIndex
                return (
                  <button
                    key={result.id}
                    type="button"
                    onClick={() => focusSearchResult(result)}
                    className={`w-full rounded-md border px-3 py-2 text-left transition ${
                      isActive
                        ? "border-primary/40 bg-primary/10"
                        : "border-border hover:bg-surface2"
                    }`}
                    aria-selected={isActive}
                  >
                    <div className="mb-0.5 flex items-center justify-between gap-2">
                      <span className="truncate text-sm font-medium text-text">
                        {result.title}
                      </span>
                      <span className="shrink-0 text-[11px] font-medium uppercase tracking-wide text-text-muted">
                        {getSearchDomainLabel(result.domain)}
                      </span>
                    </div>
                    <p className="truncate text-xs text-text-muted">{result.subtitle}</p>
                    {result.snippet && (
                      <p className="mt-1 line-clamp-2 text-xs text-text-subtle">
                        {result.snippet}
                      </p>
                    )}
                  </button>
                )
              })
            )}
          </div>
        </div>
      </Modal>

      <TransferSourcesModal
        open={transferSourcesRequest !== null}
        request={transferSourcesRequest}
        onCancel={closeTransferSourcesModal}
      />

      {/* Workspace keyboard shortcuts legend */}
      <Modal
        open={showShortcutsModal}
        onCancel={() => setShowShortcutsModal(false)}
        title={
          <span className="flex items-center gap-2">
            <Keyboard className="h-4 w-4 text-text-muted" />
            {t("playground:workspace.shortcutsTitle", "Keyboard Shortcuts")}
          </span>
        }
        footer={null}
        centered
        destroyOnClose
      >
        <div className="space-y-2 py-2 text-sm">
          {(
            [
              [t("playground:workspace.shortcutFocusSources", "Focus sources pane"), isMac ? "\u2318 1" : "Ctrl + 1"],
              [t("playground:workspace.shortcutFocusChat", "Focus chat pane"), isMac ? "\u2318 2" : "Ctrl + 2"],
              [t("playground:workspace.shortcutFocusStudio", "Focus studio pane"), isMac ? "\u2318 3" : "Ctrl + 3"],
              [t("playground:workspace.shortcutGlobalSearch", "Global search"), isMac ? "\u2318 K" : "Ctrl + K"],
              [t("playground:workspace.shortcutNewNote", "New note"), isMac ? "\u2318 N" : "Ctrl + N"],
              [t("playground:workspace.shortcutNewWorkspace", "New workspace"), isMac ? "\u21E7 \u2318 N" : "Ctrl + Shift + N"],
              [t("playground:workspace.shortcutUndo", "Undo"), isMac ? "\u2318 Z" : "Ctrl + Z"],
              [t("playground:workspace.shortcutShowShortcuts", "Show shortcuts"), "?"]
            ] as const
          ).map(([label, keys]) => (
            <div key={label} className="flex items-center justify-between rounded-lg px-2 py-1.5 hover:bg-surface2">
              <span className="text-text">{label}</span>
              <kbd className="ml-4 rounded border border-border bg-surface2 px-1.5 py-0.5 text-xs font-mono text-text-muted">
                {keys}
              </kbd>
            </div>
          ))}
        </div>
      </Modal>

      {showWorkspaceTransitionCue && (
        <div
          data-testid="workspace-switch-transition"
          className="pointer-events-none absolute inset-0 z-50 flex items-center justify-center bg-bg/60 backdrop-blur-[1px]"
        >
          <div className="rounded-md border border-border bg-surface px-4 py-2 text-sm text-text shadow-card">
            <span className="mr-2 inline-block h-3.5 w-3.5 animate-spin rounded-full border border-primary border-t-transparent align-[-2px]" />
            {t("playground:workspace.switching", "Switching workspace...")}
          </div>
        </div>
      )}
    </div>
    </SharedWorkspaceProvider>
  )
}

export const WorkspacePlayground: React.FC = () => (
  <WorkspacePlaygroundErrorBoundary>
    <WorkspacePlaygroundBody />
  </WorkspacePlaygroundErrorBoundary>
)

export default WorkspacePlayground
