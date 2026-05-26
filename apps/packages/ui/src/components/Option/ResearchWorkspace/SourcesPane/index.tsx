import React from "react"
import { useTranslation } from "react-i18next"
import {
  Plus,
  Search,
  FileText,
  Video,
  Headphones,
  Globe,
  File,
  Type,
  PanelLeftClose,
  Loader2,
  AlertTriangle,
  RefreshCw,
  Info,
  Eye,
  ChevronUp,
  ChevronDown
} from "lucide-react"
import {
  Input,
  Checkbox,
  Empty,
  Button,
  Tooltip,
  message,
  Popconfirm,
  Modal
} from "antd"
import { getDesignSystemState } from "@/design-system"
import { useWorkspaceStore } from "@/store/workspace"
import type {
  WorkspaceSource,
  WorkspaceSourceReadiness,
  WorkspaceSourceStatusDetails,
  WorkspaceSourceType
} from "@/types/workspace"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import type { WorkspaceSourcePreviewResponse } from "@/services/tldw/domains/workspace-api"
import {
  WORKSPACE_SOURCE_DRAG_TYPE,
  serializeWorkspaceSourceDragPayload
} from "../drag-source"
import {
  WORKSPACE_UNDO_WINDOW_MS,
  scheduleWorkspaceUndoAction,
  undoWorkspaceAction
} from "../undo-manager"
import {
  collectDescendantSourceIds,
  createWorkspaceOrganizationIndex,
  deriveEffectiveSelectedSourceIds,
  getFolderSelectionState,
  getSourceSelectionOrigin
} from "@/store/workspace-organization"
import { AddSourceModal } from "./AddSourceModal"
import {
  SourceFolderMembershipMenu,
  type SourceFolderMembershipOption
} from "./SourceFolderMembershipMenu"
import type { TransferSourcesModalLaunchRequest } from "../TransferSourcesModal"
import {
  SourceFolderTree,
  type SourceFolderTreeNode
} from "./SourceFolderTree"
import { SourceAdvancedControls } from "./SourceAdvancedControls"
import {
  buildSourceFilterSummary,
  DEFAULT_SOURCE_LIST_VIEW_STATE,
  filterSources as applyAdvancedSourceFilters,
  hasActiveSourceFilters,
  sortSources as applySourceSort,
  type SourceListViewState
} from "./source-list-view"

// Icon mapping for source types
const SOURCE_TYPE_ICONS: Record<WorkspaceSourceType, React.ElementType> = {
  pdf: FileText,
  video: Video,
  audio: Headphones,
  website: Globe,
  document: File,
  text: Type
}

const SOURCE_VIRTUALIZATION_THRESHOLD = 60
const SOURCE_VIRTUAL_ROW_HEIGHT = 80
const SOURCE_VIRTUAL_OVERSCAN = 5
const SOURCE_PREVIEW_MAX_CHARS = 3000
const SOURCE_PREVIEW_CHUNK_LIMIT = 3
const SOURCE_ANNOTATIONS_STORAGE_KEY =
  "tldw:research-workspace:source-annotations:v1"

type SourcePreviewLoadState = {
  sourceId: string | null
  loading: boolean
  error: string | null
  data: WorkspaceSourcePreviewResponse | null
}

const formatFileSize = (bytes?: number): string | null => {
  if (!Number.isFinite(bytes) || (bytes as number) <= 0) return null
  const value = bytes as number
  if (value >= 1024 * 1024 * 1024) return `${(value / (1024 * 1024 * 1024)).toFixed(1)} GB`
  if (value >= 1024 * 1024) return `${Math.round(value / (1024 * 1024))} MB`
  if (value >= 1024) return `${Math.round(value / 1024)} KB`
  return `${Math.round(value)} B`
}

const formatDuration = (seconds?: number): string | null => {
  if (!Number.isFinite(seconds) || (seconds as number) <= 0) return null
  const totalSeconds = Math.round(seconds as number)
  const hrs = Math.floor(totalSeconds / 3600)
  const mins = Math.floor((totalSeconds % 3600) / 60)
  const secs = totalSeconds % 60
  if (hrs > 0) {
    return `${hrs}h ${mins}m`
  }
  if (mins > 0) {
    return `${mins}m ${secs}s`
  }
  return `${secs}s`
}

const READINESS_LABELS: Array<{
  key: keyof WorkspaceSourceReadiness
  label: string
}> = [
  { key: "metadata_ready", label: "Metadata" },
  { key: "text_extracted", label: "Text" },
  { key: "fts_ready", label: "Search" },
  { key: "vector_ready", label: "Vector" },
  { key: "citation_ready", label: "Citations" },
  { key: "summary_ready", label: "Summary" },
  { key: "tool_accessible", label: "Tools" }
]

const humanizeStatusToken = (value?: string | null): string | null => {
  if (!value) return null
  const normalized = value.replace(/[_-]+/g, " ").trim()
  if (!normalized) return null
  return normalized.replace(/\b\w/g, (character) => character.toUpperCase())
}

const describeSourceOfTruth = (sourceOfTruth?: string): string =>
  sourceOfTruth === "workspace-status-projection"
    ? "Server workspace status projection"
    : sourceOfTruth === "local-cache" || !sourceOfTruth
      ? "Local workspace cache"
      : humanizeStatusToken(sourceOfTruth) || sourceOfTruth

const formatStatusDateTime = (date?: Date): string =>
  date instanceof Date && !Number.isNaN(date.getTime())
    ? date.toLocaleString()
    : "Not reported"

const getProgressPercent = (
  details: WorkspaceSourceStatusDetails | undefined
): number | null => {
  const progressPercent =
    details?.progressPercent ?? details?.job?.progressPercent ?? null
  return typeof progressPercent === "number" && Number.isFinite(progressPercent)
    ? Math.max(0, Math.min(100, Math.round(progressPercent)))
    : null
}

const getProgressMessage = (
  details: WorkspaceSourceStatusDetails | undefined
): string | null =>
  details?.progressMessage?.trim() ||
  details?.job?.progressMessage?.trim() ||
  details?.job?.errorMessage?.trim() ||
  null

const hasIncompleteReadiness = (
  readiness: WorkspaceSourceReadiness | undefined
): boolean =>
  Boolean(readiness && READINESS_LABELS.some(({ key }) => readiness[key] === false))

const hasSourceStatusDrilldown = (
  source: WorkspaceSource,
  sourceStatus: string
): boolean =>
  sourceStatus !== "ready" ||
  Boolean(source.statusMessage?.trim()) ||
  Boolean(source.statusDetails) ||
  hasIncompleteReadiness(source.readiness)

const describeRetryEligibility = (
  sourceStatus: string,
  details: WorkspaceSourceStatusDetails | undefined
): string => {
  if (details?.retryEligible === true) {
    return "Retry eligible after reviewing the failure."
  }
  if (sourceStatus === "processing" || details?.lifecycleState === "retrying") {
    return "Retry not available while processing."
  }
  if (sourceStatus === "error") {
    return "Retry state not reported. Re-add or refresh the source if the error persists."
  }
  return "Retry not needed."
}

const describeStaleState = (
  details: WorkspaceSourceStatusDetails | undefined
): string => {
  if (details?.stale === true) return "Stale status. Refresh workspace status."
  if (details?.stale === false) return "Fresh status"
  return "No stale flag reported"
}

const describeNextStatusAction = (
  sourceStatus: string,
  details: WorkspaceSourceStatusDetails | undefined
): string => {
  if (details?.stale) {
    return "Refresh workspace status before relying on this source."
  }

  switch (details?.lifecycleState) {
    case "queued":
      return "Wait for ingestion to start."
    case "ingesting":
      return "Wait for ingestion to finish."
    case "extracting":
      return "Wait for text extraction to finish."
    case "chunking":
      return "Wait for chunking to finish."
    case "indexing":
      return "Wait for indexing to finish before asking grounded questions."
    case "retrying":
      return "Wait for the retry attempt to finish."
    case "partially_queryable":
      return "You can inspect extracted text, but wait for full indexing before relying on citations."
    case "failed":
      return "Review the failure message, then retry ingestion or re-add the source."
    case "missing_media":
      return "Restore or re-add the missing media item."
    case "blocked_by_permissions":
      return "Check workspace permissions or source access, then refresh status."
    case "queryable":
      return "Source is ready for grounded questions and citations."
    default:
      if (sourceStatus === "processing") {
        return "Wait for processing to finish, then refresh status if it appears stuck."
      }
      if (sourceStatus === "error") {
        return "Review the status message, then re-add or retry the source."
      }
      return "No action needed."
  }
}

const StatusDetailRow: React.FC<{
  label: string
  children: React.ReactNode
}> = ({ label, children }) => (
  <div className="grid gap-1 rounded border border-border bg-surface/60 p-2 sm:grid-cols-[8rem_1fr] sm:items-start">
    <dt className="text-[11px] font-semibold uppercase tracking-[0.04em] text-text-subtle">
      {label}
    </dt>
    <dd className="min-w-0 break-words text-sm text-text">{children}</dd>
  </div>
)

type SourceAnnotation = {
  id: string
  quote: string
  note: string
  createdAt: number
  updatedAt: number
}

const buildSourceAnnotationsStorageKey = (workspaceId: string | null | undefined): string =>
  `${SOURCE_ANNOTATIONS_STORAGE_KEY}:${workspaceId || "local"}`

const isSourceAnnotation = (value: unknown): value is SourceAnnotation => {
  const candidate = value as Partial<SourceAnnotation>
  return (
    typeof candidate?.id === "string" &&
    typeof candidate.quote === "string" &&
    typeof candidate.note === "string" &&
    typeof candidate.createdAt === "number" &&
    typeof candidate.updatedAt === "number"
  )
}

const readPersistedSourceAnnotations = (
  workspaceId: string | null | undefined
): Record<string, SourceAnnotation[]> => {
  if (typeof window === "undefined") return {}
  try {
    const raw = window.localStorage.getItem(
      buildSourceAnnotationsStorageKey(workspaceId)
    )
    if (!raw) return {}
    const parsed = JSON.parse(raw) as Record<string, unknown>
    const next: Record<string, SourceAnnotation[]> = {}
    for (const [sourceId, annotations] of Object.entries(parsed || {})) {
      if (!Array.isArray(annotations)) continue
      next[sourceId] = annotations.filter(isSourceAnnotation)
    }
    return next
  } catch {
    return {}
  }
}

const persistSourceAnnotations = (
  workspaceId: string | null | undefined,
  annotations: Record<string, SourceAnnotation[]>
): void => {
  if (typeof window === "undefined") return
  try {
    window.localStorage.setItem(
      buildSourceAnnotationsStorageKey(workspaceId),
      JSON.stringify(annotations)
    )
  } catch {
    // Annotation persistence is best-effort local UI state.
  }
}

const describePreviewUnavailable = (
  preview: WorkspaceSourcePreviewResponse | null,
  t: (key: string, fallback: string) => string
): string => {
  const reason = preview?.unavailable_reason || preview?.status_reason || ""
  if (reason === "extraction_pending" || preview?.preview_mode === "pending") {
    return t(
      "playground:sources.previewExtractionPending",
      "Text extraction has not completed yet."
    )
  }
  if (reason === "media_not_found" || preview?.preview_mode === "missing_media") {
    return t(
      "playground:sources.previewMediaMissing",
      "Media item is missing or unavailable."
    )
  }
  if (preview?.preview_mode === "failed" || reason.includes("failed")) {
    return t(
      "playground:sources.previewExtractionFailed",
      "Source extraction or indexing failed. Preview content is unavailable."
    )
  }
  return t(
    "playground:sources.previewNoTextAvailable",
    "No captured text is available for this source."
  )
}

interface SourcesPaneProps {
  /** Callback to hide/collapse the pane */
  onHide?: () => void
  /** Open the shared transfer modal for the current effective selection. */
  onOpenTransferSources?: (
    request: TransferSourcesModalLaunchRequest
  ) => void
  /** Rollout gate for status/guardrails source handling. */
  statusGuardrailsEnabled?: boolean
  /** Non-persisted session-local source list view state owned by the page. */
  sourceListViewState?: SourceListViewState
  /** Partial state patcher for source list view state. */
  onPatchSourceListViewState?: (patch: Partial<SourceListViewState>) => void
  /** Reset advanced controls without clearing search/folder state. */
  onResetAdvancedSourceFilters?: () => void
  /** Non-blocking server-side context/status warning for this workspace. */
  statusProjectionError?: string | null
}

/**
 * SourcesPane - Left pane for managing research sources
 */
export const SourcesPane: React.FC<SourcesPaneProps> = ({
  onHide,
  onOpenTransferSources,
  statusGuardrailsEnabled = true,
  sourceListViewState = DEFAULT_SOURCE_LIST_VIEW_STATE,
  onPatchSourceListViewState,
  onResetAdvancedSourceFilters,
  statusProjectionError = null
}) => {
  const { t } = useTranslation(["playground", "common"])
  const readyState = getDesignSystemState("ready")
  const [messageApi, messageContextHolder] = message.useMessage()
  const patchSourceListViewState = React.useCallback(
    (patch: Partial<SourceListViewState>) => {
      onPatchSourceListViewState?.(patch)
    },
    [onPatchSourceListViewState]
  )
  const resetAdvancedSourceFilters = React.useCallback(() => {
    onResetAdvancedSourceFilters?.()
  }, [onResetAdvancedSourceFilters])

  // Store state
  const workspaceId = useWorkspaceStore((s) => s.workspaceId) || "local"
  const sources = useWorkspaceStore((s) => s.sources)
  const selectedSourceIds = useWorkspaceStore((s) => s.selectedSourceIds)
  const sourceFolders = useWorkspaceStore((s) => s.sourceFolders) || []
  const sourceFolderMemberships = useWorkspaceStore(
    (s) => s.sourceFolderMemberships
  ) || []
  const selectedSourceFolderIds = useWorkspaceStore(
    (s) => s.selectedSourceFolderIds
  ) || []
  const activeFolderId = useWorkspaceStore((s) => s.activeFolderId) || null
  const sourceSearchQuery = useWorkspaceStore((s) => s.sourceSearchQuery)
  const sourceFocusTarget = useWorkspaceStore((s) => s.sourceFocusTarget)

  // Store actions
  const toggleSourceSelection = useWorkspaceStore((s) => s.toggleSourceSelection)
  const toggleSourceFolderSelection = useWorkspaceStore(
    (s) => s.toggleSourceFolderSelection
  ) || (() => undefined)
  const selectAllSources = useWorkspaceStore((s) => s.selectAllSources)
  const deselectAllSources = useWorkspaceStore((s) => s.deselectAllSources)
  const setSelectedSourceIds = useWorkspaceStore(
    (s) => s.setSelectedSourceIds
  ) || (() => undefined)
  const setSourceSearchQuery = useWorkspaceStore((s) => s.setSourceSearchQuery)
  const setActiveFolder = useWorkspaceStore((s) => s.setActiveFolder) || (() => undefined)
  const clearSourceFocusTarget = useWorkspaceStore(
    (s) => s.clearSourceFocusTarget
  )
  const openAddSourceModal = useWorkspaceStore((s) => s.openAddSourceModal)
  const addSource = useWorkspaceStore((s) => s.addSource)
  const removeSource = useWorkspaceStore((s) => s.removeSource)
  const removeSources = useWorkspaceStore((s) => s.removeSources)
  const restoreSource = useWorkspaceStore((s) => s.restoreSource)
  const reorderSource = useWorkspaceStore((s) => s.reorderSource)
  const createSourceFolder = useWorkspaceStore((s) => s.createSourceFolder) || null
  const assignSourceToFolders =
    useWorkspaceStore((s) => s.assignSourceToFolders) || (() => undefined)
  const getEffectiveSelectedSources =
    useWorkspaceStore((s) => s.getEffectiveSelectedSources) || null
  const sourceItemRefs = React.useRef<Record<string, HTMLDivElement | null>>({})
  const sourceListContainerRef = React.useRef<HTMLDivElement | null>(null)
  const [highlightedSourceId, setHighlightedSourceId] = React.useState<
    string | null
  >(null)
  const [sourceListScrollTop, setSourceListScrollTop] = React.useState(0)
  const [sourceListViewportHeight, setSourceListViewportHeight] =
    React.useState(420)
  const [confirmingRemovalSourceId, setConfirmingRemovalSourceId] =
    React.useState<string | null>(null)
  const [draggedSourceId, setDraggedSourceId] = React.useState<string | null>(null)
  const [previewSourceId, setPreviewSourceId] = React.useState<string | null>(null)
  const [statusDetailsSourceId, setStatusDetailsSourceId] = React.useState<
    string | null
  >(null)
  const [previewReloadNonce, setPreviewReloadNonce] = React.useState(0)
  const [sourceAnnotations, setSourceAnnotations] = React.useState<
    Record<string, SourceAnnotation[]>
  >(() => readPersistedSourceAnnotations(workspaceId))
  const [sourcePreviewState, setSourcePreviewState] =
    React.useState<SourcePreviewLoadState>({
      sourceId: null,
      loading: false,
      error: null,
      data: null
    })
  const annotationsWorkspaceIdRef = React.useRef(workspaceId)
  const [annotationQuoteDraft, setAnnotationQuoteDraft] = React.useState("")
  const [annotationNoteDraft, setAnnotationNoteDraft] = React.useState("")
  const [editingAnnotationId, setEditingAnnotationId] = React.useState<
    string | null
  >(null)
  const [quickUrlValue, setQuickUrlValue] = React.useState("")
  const [quickUrlLoading, setQuickUrlLoading] = React.useState(false)

  const handleQuickUrlPaste = React.useCallback(
    async (url: string) => {
      const trimmed = url.trim()
      if (!trimmed) return
      try {
        new URL(trimmed)
      } catch {
        return // Not a valid URL — ignore
      }

      setQuickUrlLoading(true)
      try {
        const response = await tldwClient.addMedia(trimmed)
        const root = response as Record<string, unknown>
        const candidates: Array<Record<string, unknown>> = []
        if (Array.isArray(root.results)) {
          for (const item of root.results) {
            if (item && typeof item === "object")
              candidates.push(item as Record<string, unknown>)
          }
        }
        if (root.result && typeof root.result === "object")
          candidates.push(root.result as Record<string, unknown>)
        candidates.push(root)

        for (const candidate of candidates) {
          const mediaId = Number(
            candidate.media_id ?? candidate.db_id ?? candidate.id
          )
          if (!Number.isFinite(mediaId) || mediaId <= 0) continue
          const title =
            typeof candidate.title === "string" && candidate.title.trim()
              ? candidate.title
              : trimmed
          addSource({
            mediaId,
            title,
            type: "website" as WorkspaceSourceType,
            status: "processing",
            url: trimmed
          })
          setQuickUrlValue("")
          messageApi.success(
            t("playground:sources.quickUrlAdded", "Source added from URL")
          )
          return
        }
        messageApi.error(
          t("playground:sources.quickUrlFailed", "Could not add URL")
        )
      } catch {
        messageApi.error(
          t("playground:sources.quickUrlFailed", "Could not add URL")
        )
      } finally {
        setQuickUrlLoading(false)
      }
    },
    [addSource, messageApi, t]
  )

  const organizationIndex = React.useMemo(
    () =>
      createWorkspaceOrganizationIndex({
        sources,
        sourceFolders,
        sourceFolderMemberships
      }),
    [sourceFolderMemberships, sourceFolders, sources]
  )

  const buildFolderTreeNode = React.useCallback(
    (folderId: string): SourceFolderTreeNode | null => {
      const folder = organizationIndex.folderById.get(folderId)
      if (!folder) {
        return null
      }

      return {
        id: folder.id,
        name: folder.name,
        sourceCount: collectDescendantSourceIds(organizationIndex, folder.id).length,
        children: (organizationIndex.childrenByFolderId.get(folder.id) || [])
          .map((childId) => buildFolderTreeNode(childId))
          .filter((node): node is SourceFolderTreeNode => Boolean(node))
      }
    },
    [organizationIndex]
  )

  const folderTreeNodes = React.useMemo(
    () =>
      organizationIndex.rootFolderIds
        .map((folderId) => buildFolderTreeNode(folderId))
        .filter((node): node is SourceFolderTreeNode => Boolean(node)),
    [buildFolderTreeNode, organizationIndex.rootFolderIds]
  )

  const sourceFolderOptions = React.useMemo<SourceFolderMembershipOption[]>(() => {
    const flattened: SourceFolderMembershipOption[] = []

    const walk = (nodes: SourceFolderTreeNode[], depth: number) => {
      for (const node of nodes) {
        flattened.push({
          id: node.id,
          name: node.name,
          depth
        })
        walk(node.children, depth + 1)
      }
    }

    walk(folderTreeNodes, 0)
    return flattened
  }, [folderTreeNodes])

  const selectionStateByFolderId = React.useMemo(
    () =>
      Object.fromEntries(
        sourceFolders.map((folder) => [
          folder.id,
          getFolderSelectionState(
            organizationIndex,
            folder.id,
            selectedSourceIds,
            selectedSourceFolderIds
          )
        ])
      ) as Record<string, "unchecked" | "checked" | "indeterminate">,
    [
      organizationIndex,
      selectedSourceFolderIds,
      selectedSourceIds,
      sourceFolders
    ]
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
    if (!sourceSearchQuery.trim()) return scopedSources
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

  const useVirtualizedSources =
    filteredSources.length > SOURCE_VIRTUALIZATION_THRESHOLD
  const virtualStartIndex = useVirtualizedSources
    ? Math.max(
        0,
        Math.floor(sourceListScrollTop / SOURCE_VIRTUAL_ROW_HEIGHT) -
          SOURCE_VIRTUAL_OVERSCAN
      )
    : 0
  const virtualEndIndex = useVirtualizedSources
    ? Math.min(
        filteredSources.length,
        Math.ceil(
          (sourceListScrollTop + sourceListViewportHeight) /
            SOURCE_VIRTUAL_ROW_HEIGHT
        ) + SOURCE_VIRTUAL_OVERSCAN
      )
    : filteredSources.length
  const visibleSources = useVirtualizedSources
    ? filteredSources.slice(virtualStartIndex, virtualEndIndex)
    : filteredSources
  const sourceFilterSummary = React.useMemo(
    () => buildSourceFilterSummary(sourceListViewState),
    [sourceListViewState]
  )
  const hasFileSizeSources = React.useMemo(
    () => sources.some((source) => Number.isFinite(source.fileSize)),
    [sources]
  )
  const hasDurationSources = React.useMemo(
    () => sources.some((source) => Number.isFinite(source.duration)),
    [sources]
  )
  const hasPageCountSources = React.useMemo(
    () => sources.some((source) => Number.isFinite(source.pageCount)),
    [sources]
  )

  const effectiveSelectedSourceEntries = React.useMemo(() => {
    if (typeof getEffectiveSelectedSources === "function") {
      return getEffectiveSelectedSources()
    }

    const effectiveSelectedIds = new Set(
      deriveEffectiveSelectedSourceIds(
        organizationIndex,
        selectedSourceIds,
        selectedSourceFolderIds
      )
    )
    return sources.filter((source) => effectiveSelectedIds.has(source.id))
  }, [
    getEffectiveSelectedSources,
    organizationIndex,
    selectedSourceFolderIds,
    selectedSourceIds,
    sources
  ])
  const effectiveSelectedCount = effectiveSelectedSourceEntries.length
  const effectiveSelectedSourceIds = React.useMemo(
    () => new Set(effectiveSelectedSourceEntries.map((source) => source.id)),
    [effectiveSelectedSourceEntries]
  )
  const visibleReadySourceIds = React.useMemo(
    () =>
      filteredSources
        .filter((source) => organizationIndex.readySourceIds.has(source.id))
        .map((source) => source.id),
    [filteredSources, organizationIndex.readySourceIds]
  )
  const visibleReadySourceIdSet = React.useMemo(
    () => new Set(visibleReadySourceIds),
    [visibleReadySourceIds]
  )
  const hiddenDirectSelectedSourceIds = React.useMemo(
    () =>
      organizationIndex.sourceIdsInOrder.filter(
        (sourceId) =>
          selectedSourceIds.includes(sourceId) &&
          organizationIndex.readySourceIds.has(sourceId) &&
          !visibleReadySourceIdSet.has(sourceId)
      ),
    [
      organizationIndex.readySourceIds,
      organizationIndex.sourceIdsInOrder,
      selectedSourceIds,
      visibleReadySourceIdSet
    ]
  )
  const visibleEffectiveSelectedCount = React.useMemo(
    () =>
      filteredSources.filter((source) => effectiveSelectedSourceIds.has(source.id))
        .length,
    [effectiveSelectedSourceIds, filteredSources]
  )
  const eligibleSelectedSourceIds = React.useMemo(
    () =>
      effectiveSelectedSourceEntries
        .filter((source) => organizationIndex.readySourceIds.has(source.id))
        .map((source) => source.id),
    [effectiveSelectedSourceEntries, organizationIndex.readySourceIds]
  )
  const hiddenSelectedCount = Math.max(
    0,
    effectiveSelectedCount - visibleEffectiveSelectedCount
  )
  const ineligibleSelectedCount = Math.max(
    0,
    effectiveSelectedCount - eligibleSelectedSourceIds.length
  )
  const isListNarrowed =
    activeFolderId !== null ||
    sourceSearchQuery.trim().length > 0 ||
    hasActiveSourceFilters(sourceListViewState)
  const isTemporarySortActive = sourceListViewState.sort !== "manual"
  const allVisibleSelected =
    visibleReadySourceIds.length > 0 &&
    visibleReadySourceIds.every((sourceId) =>
      effectiveSelectedSourceIds.has(sourceId)
    )
  const someVisibleSelected =
    !allVisibleSelected && visibleEffectiveSelectedCount > 0
  const allSelected =
    organizationIndex.readySourceIds.size > 0 &&
    effectiveSelectedCount === organizationIndex.readySourceIds.size
  const someSelected = effectiveSelectedCount > 0 && !allSelected
  const selectionCheckboxChecked = isListNarrowed ? allVisibleSelected : allSelected
  const selectionCheckboxIndeterminate = isListNarrowed
    ? !allVisibleSelected &&
      (someVisibleSelected || hiddenSelectedCount > 0)
    : someSelected
  const selectionCheckboxLabel = isListNarrowed
    ? t("playground:sources.selectVisible", "Select visible")
    : t("playground:sources.selectAll", "Select all")
  const selectedSourceEntries = effectiveSelectedSourceEntries
  const singleSelectedSource =
    selectedSourceEntries.length === 1 ? selectedSourceEntries[0] : null
  const batchRemoveDescription =
    hiddenSelectedCount > 0
      ? t(
          "playground:sources.batchRemoveHiddenSelection",
          {
            count: hiddenSelectedCount,
            defaultValue:
              "{{count}} selected sources are hidden by current filters and will also be removed."
          }
        )
      : undefined

  const clearEffectiveSelection = React.useCallback(() => {
    deselectAllSources()
    for (const folderId of [...new Set(selectedSourceFolderIds)]) {
      toggleSourceFolderSelection(folderId)
    }
  }, [
    deselectAllSources,
    selectedSourceFolderIds,
    toggleSourceFolderSelection
  ])

  const handleBatchRemoveSelected = React.useCallback(() => {
    if (effectiveSelectedSourceEntries.length === 0) {
      return
    }

    const removedSourceEntries = effectiveSelectedSourceEntries
      .map((source) => ({
        source,
        index: sources.findIndex((entry) => entry.id === source.id),
        wasDirectlySelected: selectedSourceIds.includes(source.id),
        folderIds: [...(organizationIndex.folderIdsBySourceId.get(source.id) || [])]
      }))
      .filter((entry) => entry.index >= 0)
    const selectedIds = removedSourceEntries.map((entry) => entry.source.id)

    const undoHandle = scheduleWorkspaceUndoAction({
      apply: () => {
        removeSources(selectedIds)
      },
      undo: () => {
        for (const entry of [...removedSourceEntries].sort((a, b) => a.index - b.index)) {
          restoreSource(entry.source, {
            index: entry.index,
            select: entry.wasDirectlySelected
          })
          if (entry.folderIds.length > 0) {
            assignSourceToFolders(entry.source.id, entry.folderIds)
          }
        }
      }
    })

    messageApi.open({
      type: "warning",
      duration: WORKSPACE_UNDO_WINDOW_MS / 1000,
      content: (
        <div className="flex items-center gap-2">
          <span>
            {t("playground:sources.batchRemoved", "{{count}} sources removed", {
              count: removedSourceEntries.length
            })}
          </span>
          <Button
            size="small"
            onClick={() => undoWorkspaceAction(undoHandle.id)}
          >
            {t("common:undo", "Undo")}
          </Button>
        </div>
      )
    })
  }, [
    assignSourceToFolders,
    effectiveSelectedSourceEntries,
    messageApi,
    organizationIndex.folderIdsBySourceId,
    removeSources,
    restoreSource,
    selectedSourceIds,
    sources,
    t
  ])

  const handleOpenTransferSources = React.useCallback(() => {
    if (!onOpenTransferSources || effectiveSelectedCount === 0) {
      return
    }

    onOpenTransferSources({
      entryPoint: "sources",
      selectedSourceIds: effectiveSelectedSourceEntries.map((source) => source.id),
      eligibleSelectedSourceIds,
      totalSelectedCount: effectiveSelectedCount,
      hiddenSelectedCount,
      ineligibleSelectedCount
    })
  }, [
    effectiveSelectedCount,
    effectiveSelectedSourceEntries,
    eligibleSelectedSourceIds,
    hiddenSelectedCount,
    ineligibleSelectedCount,
    onOpenTransferSources
  ])
  const previewSource = previewSourceId
    ? sources.find((source) => source.id === previewSourceId) || null
    : null
  const statusDetailsSource = statusDetailsSourceId
    ? sources.find((source) => source.id === statusDetailsSourceId) || null
    : null
  const previewAnnotations = previewSourceId
    ? sourceAnnotations[previewSourceId] || []
    : []

  React.useEffect(() => {
    if (annotationsWorkspaceIdRef.current === workspaceId) return
    annotationsWorkspaceIdRef.current = workspaceId
    setSourceAnnotations(readPersistedSourceAnnotations(workspaceId))
  }, [workspaceId])

  const commitSourceAnnotations = React.useCallback(
    (
      updater: (
        previous: Record<string, SourceAnnotation[]>
      ) => Record<string, SourceAnnotation[]>
    ) => {
      setSourceAnnotations((previous) => {
        const next = updater(previous)
        persistSourceAnnotations(workspaceId, next)
        return next
      })
    },
    [workspaceId]
  )

  React.useEffect(() => {
    if (!previewSourceId) {
      setSourcePreviewState((previous) => {
        if (
          previous.sourceId === null &&
          !previous.loading &&
          previous.error === null &&
          previous.data === null
        ) {
          return previous
        }
        return {
          sourceId: null,
          loading: false,
          error: null,
          data: null
        }
      })
      return
    }

    let cancelled = false
    const activeSourceId = previewSourceId
    setSourcePreviewState({
      sourceId: activeSourceId,
      loading: true,
      error: null,
      data: null
    })

    const loadPreview = async () => {
      if (typeof tldwClient.getWorkspaceSourcePreview !== "function") {
        if (!cancelled) {
          setSourcePreviewState({
            sourceId: activeSourceId,
            loading: false,
            error: "Source preview API is unavailable.",
            data: null
          })
        }
        return
      }

      try {
        const data = await tldwClient.getWorkspaceSourcePreview(
          workspaceId,
          activeSourceId,
          {
            max_chars: SOURCE_PREVIEW_MAX_CHARS,
            chunk_limit: SOURCE_PREVIEW_CHUNK_LIMIT
          }
        )
        if (!cancelled) {
          setSourcePreviewState({
            sourceId: activeSourceId,
            loading: false,
            error: null,
            data
          })
        }
      } catch (error) {
        if (!cancelled) {
          setSourcePreviewState({
            sourceId: activeSourceId,
            loading: false,
            error:
              error instanceof Error
                ? error.message
                : "Source preview could not load.",
            data: null
          })
        }
      }
    }

    void loadPreview()
    return () => {
      cancelled = true
    }
  }, [previewReloadNonce, previewSourceId, workspaceId])

  const handleSelectAllToggle = React.useCallback((event: {
    target: { checked: boolean }
  }) => {
    if (isListNarrowed) {
      const nextSelectedIds = event.target.checked
        ? organizationIndex.sourceIdsInOrder.filter(
            (sourceId) =>
              visibleReadySourceIdSet.has(sourceId) ||
              hiddenDirectSelectedSourceIds.includes(sourceId)
          )
        : hiddenDirectSelectedSourceIds
      setSelectedSourceIds(nextSelectedIds)
      return
    }

    if (allSelected || someSelected) {
      clearEffectiveSelection()
      return
    }

    selectAllSources()
  }, [
    allSelected,
    clearEffectiveSelection,
    hiddenDirectSelectedSourceIds,
    isListNarrowed,
    organizationIndex.sourceIdsInOrder,
    selectAllSources,
    setSelectedSourceIds,
    someSelected,
    visibleReadySourceIdSet
  ])

  const handleCreateSourceFolder = React.useCallback(() => {
    if (typeof createSourceFolder !== "function") {
      return
    }

    createSourceFolder("New folder", activeFolderId)
  }, [activeFolderId, createSourceFolder])

  const resetAnnotationEditor = React.useCallback(() => {
    setAnnotationQuoteDraft("")
    setAnnotationNoteDraft("")
    setEditingAnnotationId(null)
  }, [])

  const handleOpenPreview = React.useCallback(
    (sourceId: string) => {
      setPreviewSourceId(sourceId)
      resetAnnotationEditor()
    },
    [resetAnnotationEditor]
  )

  const handleClosePreview = React.useCallback(() => {
    setPreviewSourceId(null)
    resetAnnotationEditor()
  }, [resetAnnotationEditor])

  const handleOpenStatusDetails = React.useCallback((sourceId: string) => {
    setStatusDetailsSourceId(sourceId)
  }, [])

  const handleCloseStatusDetails = React.useCallback(() => {
    setStatusDetailsSourceId(null)
  }, [])

  const handleSaveAnnotation = React.useCallback(() => {
    if (!previewSourceId) return
    const quote = annotationQuoteDraft.trim()
    const note = annotationNoteDraft.trim()
    if (!quote && !note) {
      messageApi.warning(
        t(
          "playground:sources.annotationEmpty",
          "Add a highlight excerpt or an annotation note."
        )
      )
      return
    }

    commitSourceAnnotations((previous) => {
      const existing = previous[previewSourceId] || []
      const now = Date.now()
      if (editingAnnotationId) {
        return {
          ...previous,
          [previewSourceId]: existing.map((annotation) =>
            annotation.id === editingAnnotationId
              ? {
                  ...annotation,
                  quote,
                  note,
                  updatedAt: now
                }
              : annotation
          )
        }
      }

      const nextAnnotation: SourceAnnotation = {
        id: `${previewSourceId}-${now}-${Math.random().toString(36).slice(2, 7)}`,
        quote,
        note,
        createdAt: now,
        updatedAt: now
      }
      return {
        ...previous,
        [previewSourceId]: [nextAnnotation, ...existing]
      }
    })
    resetAnnotationEditor()
  }, [
    annotationNoteDraft,
    annotationQuoteDraft,
    commitSourceAnnotations,
    editingAnnotationId,
    messageApi,
    previewSourceId,
    resetAnnotationEditor,
    t
  ])

  const handleEditAnnotation = React.useCallback((annotation: SourceAnnotation) => {
    setAnnotationQuoteDraft(annotation.quote)
    setAnnotationNoteDraft(annotation.note)
    setEditingAnnotationId(annotation.id)
  }, [])

  const handleDeleteAnnotation = React.useCallback(
    (annotationId: string) => {
      if (!previewSourceId) return
      const sourceId = previewSourceId
      const existing = sourceAnnotations[sourceId] || []
      const annotationIndex = existing.findIndex(
        (annotation) => annotation.id === annotationId
      )
      if (annotationIndex < 0) return
      const removedAnnotation = existing[annotationIndex]
      const wasEditingRemovedAnnotation = editingAnnotationId === annotationId

      const undoHandle = scheduleWorkspaceUndoAction({
        apply: () => {
          commitSourceAnnotations((previous) => {
            const current = previous[sourceId] || []
            return {
              ...previous,
              [sourceId]: current.filter(
                (annotation) => annotation.id !== annotationId
              )
            }
          })
          if (wasEditingRemovedAnnotation) {
            resetAnnotationEditor()
          }
        },
        undo: () => {
          commitSourceAnnotations((previous) => {
            const current = previous[sourceId] || []
            if (current.some((annotation) => annotation.id === annotationId)) {
              return previous
            }
            const restored = [...current]
            const insertionIndex = Math.max(
              0,
              Math.min(annotationIndex, restored.length)
            )
            restored.splice(insertionIndex, 0, removedAnnotation)
            return {
              ...previous,
              [sourceId]: restored
            }
          })
          if (wasEditingRemovedAnnotation) {
            setAnnotationQuoteDraft(removedAnnotation.quote)
            setAnnotationNoteDraft(removedAnnotation.note)
            setEditingAnnotationId(removedAnnotation.id)
          }
        }
      })

      const undoMessageKey = `workspace-source-annotation-undo-${undoHandle.id}`
      const maybeOpen = (
        messageApi as { open?: (config: unknown) => void }
      ).open
      const messageConfig = {
        key: undoMessageKey,
        type: "warning",
        duration: WORKSPACE_UNDO_WINDOW_MS / 1000,
        content: t(
          "playground:sources.annotationRemoved",
          "Annotation removed."
        ),
        btn: (
          <Button
            size="small"
            type="link"
            onClick={() => {
              if (undoWorkspaceAction(undoHandle.id)) {
                messageApi.success(
                  t(
                    "playground:sources.annotationRestored",
                    "Annotation restored"
                  )
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
          maybeWarning(t("playground:sources.annotationRemoved", "Annotation removed."))
        }
      }
    },
    [
      commitSourceAnnotations,
      editingAnnotationId,
      messageApi,
      previewSourceId,
      resetAnnotationEditor,
      sourceAnnotations,
      t
    ]
  )

  const removeSourceWithUndo = React.useCallback(
    (source: (typeof sources)[number]) => {
      const sourceIndex = sources.findIndex((entry) => entry.id === source.id)
      const wasSelected = selectedSourceIds.includes(source.id)
      const assignedFolderIds = organizationIndex.folderIdsBySourceId.get(source.id) || []
      const undoHandle = scheduleWorkspaceUndoAction({
        apply: () => {
          removeSource(source.id)
        },
        undo: () => {
          restoreSource(source, {
            index: sourceIndex,
            select: wasSelected
          })
          if (assignedFolderIds.length > 0) {
            assignSourceToFolders(source.id, assignedFolderIds)
          }
        }
      })

      const undoMessageKey = `workspace-source-undo-${undoHandle.id}`
      const maybeOpen = (
        messageApi as { open?: (config: unknown) => void }
      ).open
      const messageConfig = {
        key: undoMessageKey,
        type: "warning",
        duration: WORKSPACE_UNDO_WINDOW_MS / 1000,
        content: t(
          "playground:sources.undoRemove",
          "Source removed."
        ),
        btn: (
          <Button
            size="small"
            type="link"
            onClick={() => {
              if (undoWorkspaceAction(undoHandle.id)) {
                messageApi.success(
                  t("playground:sources.restoreSuccess", "Source restored")
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
          maybeWarning(t("playground:sources.undoRemove", "Source removed."))
        }
      }
    },
    [
      assignSourceToFolders,
      messageApi,
      organizationIndex.folderIdsBySourceId,
      removeSource,
      restoreSource,
      selectedSourceIds,
      sources,
      t
    ]
  )

  React.useEffect(() => {
    const container = sourceListContainerRef.current
    if (!container) return

    const syncViewportHeight = () => {
      setSourceListViewportHeight(container.clientHeight || 420)
    }

    syncViewportHeight()

    if (typeof ResizeObserver === "undefined") {
      return
    }

    const observer = new ResizeObserver(() => {
      syncViewportHeight()
    })
    observer.observe(container)
    return () => {
      observer.disconnect()
    }
  }, [filteredSources.length])

  React.useEffect(() => {
    const targetSourceId = sourceFocusTarget?.sourceId
    if (!targetSourceId) return

    const sourceExists = sources.some((source) => source.id === targetSourceId)
    if (!sourceExists) {
      clearSourceFocusTarget()
      return
    }

    const isTargetVisible = filteredSources.some(
      (source) => source.id === targetSourceId
    )
    if (!isTargetVisible && sourceSearchQuery.trim()) {
      setSourceSearchQuery("")
    }

    if (useVirtualizedSources && sourceListContainerRef.current) {
      const targetIndex = filteredSources.findIndex(
        (source) => source.id === targetSourceId
      )
      if (targetIndex >= 0) {
        const targetScrollTop = targetIndex * SOURCE_VIRTUAL_ROW_HEIGHT
        sourceListContainerRef.current.scrollTop = targetScrollTop
        setSourceListScrollTop(targetScrollTop)
      }
    }

    const revealTimer = window.setTimeout(() => {
      const element = sourceItemRefs.current[targetSourceId]
      if (element) {
        element.scrollIntoView({ behavior: "smooth", block: "nearest" })
      }
      setHighlightedSourceId(targetSourceId)
    }, 0)

    const highlightTimer = window.setTimeout(() => {
      setHighlightedSourceId((current) =>
        current === targetSourceId ? null : current
      )
    }, 1800)

    clearSourceFocusTarget()

    return () => {
      window.clearTimeout(revealTimer)
      window.clearTimeout(highlightTimer)
    }
  }, [
    clearSourceFocusTarget,
    filteredSources,
    sourceFocusTarget,
    sourceSearchQuery,
    sources,
    setSourceSearchQuery,
    useVirtualizedSources
  ])

  const renderSourceRow = (source: (typeof filteredSources)[number]) => {
    const Icon = SOURCE_TYPE_ICONS[source.type] || File
    const selectionOrigin = getSourceSelectionOrigin(
      source.id,
      selectedSourceIds,
      selectedSourceFolderIds,
      organizationIndex
    )
    const isSelected = selectionOrigin !== "none"
    const isHighlighted = highlightedSourceId === source.id
    const sourceStatus = statusGuardrailsEnabled
      ? source.status || "ready"
      : "ready"
    const showStatusDrilldown =
      statusGuardrailsEnabled && hasSourceStatusDrilldown(source, sourceStatus)
    const isReady = sourceStatus === "ready"
    const isProcessing = sourceStatus === "processing"
    const isError = sourceStatus === "error"
    const canReorder = !isTemporarySortActive && isReady
    const processingStatusText =
      typeof source.statusMessage === "string" && source.statusMessage.trim().length > 0
        ? source.statusMessage
        : t("playground:sources.statusProcessing", "Processing")
    const metadataParts: string[] = []
    const fileSizeLabel = formatFileSize(source.fileSize)
    const durationLabel = formatDuration(source.duration)
    const pageCountLabel =
      Number.isFinite(source.pageCount) && (source.pageCount as number) > 0
        ? `${source.pageCount} pages`
        : null
    if (fileSizeLabel) metadataParts.push(fileSizeLabel)
    if (durationLabel) metadataParts.push(durationLabel)
    if (pageCountLabel) metadataParts.push(pageCountLabel)
    const sourceDate = source.sourceCreatedAt || source.addedAt
    metadataParts.push(
      source.sourceCreatedAt
        ? t("playground:sources.createdAt", "Created {{date}}", {
            date: sourceDate.toLocaleDateString()
          })
        : t("playground:sources.addedAt", "Added {{date}}", {
            date: sourceDate.toLocaleDateString()
          })
    )
    const metadataPreview = metadataParts.slice(0, 2).join(" • ")
    const metadataTooltip = metadataParts.join(" • ")
    const sourceOrderIndex = sources.findIndex((entry) => entry.id === source.id)
    const canMoveUp = canReorder && sourceOrderIndex > 0
    const canMoveDown =
      canReorder && sourceOrderIndex >= 0 && sourceOrderIndex < sources.length - 1
    const isDropTarget = draggedSourceId != null && draggedSourceId !== source.id
    const assignedFolderIds = organizationIndex.folderIdsBySourceId.get(source.id) || []
    const sourceTypeLabel = t(`playground:sources.type.${source.type}`, source.type)
    const sourceStatusLabel = isProcessing
      ? t("playground:sources.statusProcessing", "Processing")
      : isError
        ? t("playground:sources.statusErrorShort", "Error")
        : readyState.label
    const sourceStatusClass = isProcessing
      ? "border-primary/30 bg-primary/10 text-primary"
      : isError
        ? "border-error/30 bg-error/10 text-error"
        : "border-success/30 bg-success/10 text-success"

    return (
      <div
        key={source.id}
        data-source-id={source.id}
        data-source-draggable="true"
        data-highlighted={isHighlighted ? "true" : "false"}
        ref={(element) => {
          sourceItemRefs.current[source.id] = element
        }}
        draggable={canReorder}
        onDragStart={(event) => {
          if (!canReorder) {
            event.preventDefault()
            return
          }
          setDraggedSourceId(source.id)
          event.dataTransfer.effectAllowed = "copyMove"
          event.dataTransfer.setData(
            WORKSPACE_SOURCE_DRAG_TYPE,
            serializeWorkspaceSourceDragPayload({
              sourceId: source.id,
              mediaId: source.mediaId,
              title: source.title,
              type: source.type
            })
          )
          event.dataTransfer.setData("text/plain", source.title)
        }}
        onDragOver={(event) => {
          if (!canReorder || !draggedSourceId || draggedSourceId === source.id) return
          event.preventDefault()
          event.dataTransfer.dropEffect = "move"
        }}
        onDrop={(event) => {
          if (!canReorder || !draggedSourceId || draggedSourceId === source.id) return
          event.preventDefault()
          const targetIndex = sources.findIndex((entry) => entry.id === source.id)
          if (targetIndex >= 0) {
            reorderSource(draggedSourceId, targetIndex)
          }
          setDraggedSourceId(null)
        }}
        onDragEnd={() => setDraggedSourceId(null)}
        className={`group flex items-start gap-2 rounded-lg p-2 transition-colors ${
          isSelected
            ? "bg-primary/10 border border-primary/30"
            : "hover:bg-surface2 border border-transparent"
        } ${
          isHighlighted
            ? "ring-2 ring-primary/40 border-primary/40 bg-primary/15"
            : ""
        } ${
          isDropTarget ? "border-primary/50 bg-primary/5" : ""
        } ${canReorder ? "cursor-grab active:cursor-grabbing" : "cursor-default"}`}
      >
        <div
          data-testid={`source-checkbox-hitarea-${source.id}`}
          className="mt-0.5 flex items-center justify-center [@media(hover:none)]:min-h-11 [@media(hover:none)]:min-w-11"
        >
          <Checkbox
            checked={isSelected}
            disabled={!isReady}
            onChange={() => {
              if (selectionOrigin === "folder") {
                return
              }
              toggleSourceSelection(source.id)
            }}
          />
        </div>
        <div className="flex min-w-0 flex-1 items-start gap-2">
          <div
            className={`relative flex h-8 w-8 shrink-0 items-center justify-center overflow-hidden rounded ${
              isSelected ? "bg-primary/20 text-primary" : "bg-surface2 text-text-muted"
            }`}
          >
            <Icon className="h-4 w-4" />
            {source.thumbnailUrl && (
              <img
                src={source.thumbnailUrl}
                alt=""
                aria-hidden="true"
                data-testid={`source-thumbnail-${source.id}`}
                className="absolute inset-0 h-full w-full object-cover"
                onError={(event) => {
                  event.currentTarget.remove()
                }}
              />
            )}
          </div>
          <div className="min-w-0 flex-1">
            <p className="truncate text-sm font-medium text-text">
              {source.title}
            </p>
            <div className="mt-0.5 flex flex-wrap items-center gap-1 text-[10px] uppercase tracking-[0.04em]">
              <span className="rounded-full border border-border bg-surface2 px-1.5 py-0.5 font-medium text-text-muted">
                {sourceTypeLabel}
              </span>
              <span
                className={`rounded-full border px-1.5 py-0.5 font-medium ${sourceStatusClass}`}
              >
                {sourceStatusLabel}
              </span>
              {selectionOrigin === "direct" && (
                <span className="rounded-full border border-primary/30 bg-primary/10 px-1.5 py-0.5 font-medium text-primary">
                  {t("playground:sources.selectedDirectBadge", "Direct")}
                </span>
              )}
              {selectionOrigin === "folder" && (
                <span className="rounded-full border border-primary/30 bg-primary/10 px-1.5 py-0.5 font-medium text-primary">
                  {t("playground:sources.selectedFolderBadge", "From folder")}
                </span>
              )}
              {selectionOrigin === "both" && (
                <span className="rounded-full border border-primary/30 bg-primary/10 px-1.5 py-0.5 font-medium text-primary">
                  {t("playground:sources.selectedBothBadge", "Direct + folder")}
                </span>
              )}
            </div>
            <Tooltip title={metadataTooltip}>
              <p className="mt-0.5 inline-flex max-w-full items-center gap-1 truncate text-[11px] text-text-subtle">
                <Info className="h-3 w-3 shrink-0" />
                <span className="truncate">{metadataPreview}</span>
              </p>
            </Tooltip>
            {statusGuardrailsEnabled && isProcessing && (
              <p
                className="mt-0.5 flex items-center gap-1 text-[11px] text-primary"
                title={processingStatusText}
              >
                <Loader2 className="h-3 w-3 animate-spin" />
                {processingStatusText}
              </p>
            )}
            {statusGuardrailsEnabled && isError && (
              <Tooltip
                title={
                  source.statusMessage ? (
                    <span
                      className="cursor-text select-all"
                      onClick={(e) => {
                        e.stopPropagation()
                        if (source.statusMessage) {
                          navigator.clipboard.writeText(source.statusMessage).catch(() => {})
                        }
                      }}
                    >
                      {source.statusMessage}
                      <span className="ml-1 text-[10px] opacity-70">
                        {t("playground:sources.clickToCopy", "(click to copy)")}
                      </span>
                    </span>
                  ) : undefined
                }
                placement="bottom"
              >
                <p className="mt-0.5 flex items-center gap-1 text-[11px] text-error">
                  <AlertTriangle className="h-3 w-3 shrink-0" />
                  <span className="line-clamp-1">
                    {source.statusMessage ||
                      t(
                        "playground:sources.statusError",
                        "Source processing failed"
                      )}
                  </span>
                </p>
              </Tooltip>
            )}
          </div>
        </div>
        <div
          className={`flex shrink-0 items-start gap-1 rounded-md p-0.5 ${
            isSelected ? "border border-primary/20 bg-primary/5" : ""
          }`}
        >
          <SourceFolderMembershipMenu
            sourceTitle={source.title}
            folderOptions={sourceFolderOptions}
            selectedFolderIds={assignedFolderIds}
            onChange={(folderIds) => assignSourceToFolders(source.id, folderIds)}
          />
          <Tooltip title={t("playground:sources.previewAnnotate", "Preview & annotate")}>
            <button
              type="button"
              onClick={() => handleOpenPreview(source.id)}
              data-testid={`preview-source-${source.id}`}
              className="rounded p-1 text-text-muted transition hover:bg-surface hover:text-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus"
              aria-label={t(
                "playground:sources.previewAnnotate",
                "Preview & annotate"
              )}
            >
              <Eye className="h-3.5 w-3.5" />
            </button>
          </Tooltip>
          {showStatusDrilldown && (
            <Tooltip
              title={t(
                "playground:sources.viewStatusDetails",
                "View source status details"
              )}
            >
              <button
                type="button"
                onClick={() => handleOpenStatusDetails(source.id)}
                data-testid={`source-status-details-${source.id}`}
                className="rounded p-1 text-text-muted transition hover:bg-surface hover:text-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus"
                aria-label={t(
                  "playground:sources.viewStatusDetailsForSource",
                  "View source status details for {{title}}",
                  { title: source.title }
                )}
              >
                <Info className="h-3.5 w-3.5" />
              </button>
            </Tooltip>
          )}
          <div className="flex flex-col">
            <button
              type="button"
              className="rounded p-0.5 text-text-muted transition hover:bg-surface focus-visible:bg-surface disabled:cursor-not-allowed disabled:opacity-40"
              aria-label={t("playground:sources.moveUp", "Move source up")}
              data-testid={`move-source-up-${source.id}`}
              disabled={!canMoveUp}
              onClick={() => {
                if (!canMoveUp) return
                reorderSource(source.id, sourceOrderIndex - 1)
              }}
            >
              <ChevronUp className="h-3.5 w-3.5" />
            </button>
            <button
              type="button"
              className="rounded p-0.5 text-text-muted transition hover:bg-surface focus-visible:bg-surface disabled:cursor-not-allowed disabled:opacity-40"
              aria-label={t("playground:sources.moveDown", "Move source down")}
              data-testid={`move-source-down-${source.id}`}
              disabled={!canMoveDown}
              onClick={() => {
                if (!canMoveDown) return
                reorderSource(source.id, sourceOrderIndex + 1)
              }}
            >
              <ChevronDown className="h-3.5 w-3.5" />
            </button>
          </div>
          <Popconfirm
            open={confirmingRemovalSourceId === source.id}
            title={t("playground:sources.confirmRemoveTitle", "Remove source?")}
            description={t(
              "playground:sources.confirmRemoveDescription",
              "Press Remove to confirm. You can still undo for a few seconds."
            )}
            okText={t("common:remove", "Remove")}
            cancelText={t("common:cancel", "Cancel")}
            onConfirm={() => {
              setConfirmingRemovalSourceId(null)
              removeSourceWithUndo(source)
            }}
            onCancel={() => setConfirmingRemovalSourceId(null)}
          >
            <button
              type="button"
              onClick={() => {
                if (confirmingRemovalSourceId === source.id) return
                removeSourceWithUndo(source)
              }}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault()
                  setConfirmingRemovalSourceId(source.id)
                }
              }}
              data-testid={`remove-source-${source.id}`}
              className={`rounded p-1 text-text-muted transition hover:bg-error/10 hover:text-error focus-visible:opacity-100 [@media(hover:none)]:min-h-11 [@media(hover:none)]:min-w-11 [@media(hover:none)]:opacity-100 ${
                isSelected ? "opacity-100" : "opacity-0 group-hover:opacity-100"
              }`}
              aria-label={t("common:remove", "Remove")}
            >
              <svg
                className="h-3.5 w-3.5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
            </button>
          </Popconfirm>
        </div>
      </div>
    )
  }

  return (
    <div
      data-testid="workspace-sources-pane-root"
      className="flex h-full min-h-0 flex-col overflow-hidden"
    >
      {messageContextHolder}
      {/* Header */}
      <div className="flex shrink-0 items-center justify-between border-b border-border px-4 py-3">
        <div className="flex min-w-0 items-center gap-2">
          <h2 className="text-sm font-semibold text-text">
            {t("playground:sources.title", "Sources")}
          </h2>
          {statusProjectionError && (
            <Tooltip
              title={t(
                "playground:sources.statusProjectionWarningTooltip",
                "Some source status details may be incomplete: {{message}}",
                { message: statusProjectionError }
              )}
            >
              <button
                type="button"
                aria-label={t(
                  "playground:sources.statusProjectionWarningLabel",
                  "Source status warning"
                )}
                className="inline-flex h-7 w-7 shrink-0 items-center justify-center rounded text-warning transition hover:bg-warning/10 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-warning"
              >
                <AlertTriangle className="h-3.5 w-3.5" />
              </button>
            </Tooltip>
          )}
        </div>
        <div className="flex items-center gap-2">
          <Button
            type="primary"
            size="small"
            icon={<Plus className="h-3.5 w-3.5" />}
            onClick={() => openAddSourceModal("existing")}
          >
            {t("playground:sources.addSources", "Add Sources")}
          </Button>
          {onHide && (
            <Tooltip title={t("playground:workspace.hideSources", "Hide sources")}>
              <button
                type="button"
                onClick={onHide}
                className="hidden h-9 w-9 items-center justify-center rounded text-text-muted transition hover:bg-surface2 hover:text-text lg:flex"
                aria-label={t("playground:workspace.hideSources", "Hide sources")}
              >
                <PanelLeftClose className="h-4 w-4" />
              </button>
            </Tooltip>
          )}
        </div>
      </div>

      {/* Quick URL paste */}
      <div className="shrink-0 border-b border-border px-4 py-1.5">
        <Input
          data-testid="quick-url-input"
          placeholder={t(
            "playground:sources.quickUrlPlaceholder",
            "Paste a URL to add..."
          )}
          value={quickUrlValue}
          onChange={(e) => {
            setQuickUrlValue(e.target.value)
          }}
          onPaste={(e) => {
            const pasted = e.clipboardData.getData("text/plain").trim()
            try {
              new URL(pasted)
              e.preventDefault()
              setQuickUrlValue(pasted)
              handleQuickUrlPaste(pasted)
            } catch {
              // Not a URL — let normal paste happen
            }
          }}
          onPressEnter={() => handleQuickUrlPaste(quickUrlValue)}
          prefix={<Globe className="h-3.5 w-3.5 text-text-muted" />}
          suffix={
            quickUrlLoading ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin text-primary" />
            ) : null
          }
          size="small"
          allowClear
          disabled={quickUrlLoading}
        />
      </div>

      <div
        data-testid="sources-management-controls"
        className="custom-scrollbar shrink-0 overflow-y-auto border-b border-border"
        style={sources.length > 0 ? { maxHeight: "min(55%, 30rem)" } : undefined}
      >
        {/* Search and select controls */}
        {sources.length > 0 && (
          <div className="px-4 py-2">
            <Input
              prefix={<Search className="h-4 w-4 text-text-muted" />}
              placeholder={t("playground:sources.searchPlaceholder", "Search sources...")}
              value={sourceSearchQuery}
              onChange={(e) => setSourceSearchQuery(e.target.value)}
              size="small"
              allowClear
            />
            <SourceAdvancedControls
              viewState={sourceListViewState}
              summary={sourceFilterSummary}
              hasFileSizeSources={hasFileSizeSources}
              hasDurationSources={hasDurationSources}
              hasPageCountSources={hasPageCountSources}
              onPatchViewState={patchSourceListViewState}
              onResetAdvancedFilters={resetAdvancedSourceFilters}
            />
            {isTemporarySortActive && (
              <p className="mt-2 text-[11px] text-text-subtle">
                {t(
                  "playground:sources.reorderDisabledHint",
                  "Temporary sort is active. Switch back to manual order to reorder sources."
                )}
              </p>
            )}
            <div className="mt-2 flex items-center justify-between text-xs">
              <Checkbox
                aria-label={selectionCheckboxLabel}
                checked={selectionCheckboxChecked}
                indeterminate={selectionCheckboxIndeterminate}
                onChange={handleSelectAllToggle}
                className="[@media(hover:none)]:min-h-11 [@media(hover:none)]:min-w-11"
              >
                <span className="text-text-muted">
                  {effectiveSelectedCount > 0
                    ? t("playground:sources.selectedCount", "{{count}} selected", {
                        count: effectiveSelectedCount
                      })
                    : selectionCheckboxLabel}
                </span>
              </Checkbox>
              {effectiveSelectedCount > 0 && (
                <button
                  type="button"
                  onClick={clearEffectiveSelection}
                  className="text-primary hover:underline"
                >
                  {t("common:clear", "Clear")}
                </button>
              )}
            </div>
            {effectiveSelectedCount > 0 && (
              <div
                data-testid="sources-selected-actions"
                className="mt-2 flex flex-wrap items-center gap-2"
              >
                <span className="rounded-full border border-primary/30 bg-primary/10 px-2 py-0.5 text-[11px] font-medium text-primary">
                  {t(
                    "playground:sources.selectedForChat",
                    "{{count}} selected for grounded chat",
                    { count: effectiveSelectedCount }
                  )}
                </span>
                {eligibleSelectedSourceIds.length > 0 && (
                  <button
                    type="button"
                    onClick={handleOpenTransferSources}
                    className="rounded border border-border bg-surface px-2 py-0.5 text-[11px] text-text-muted transition hover:bg-surface2 hover:text-text"
                  >
                    {t("playground:sources.transferSelected", "Move / Copy")}
                  </button>
                )}
                <button
                  type="button"
                  onClick={() => {
                    if (!singleSelectedSource) return
                    handleOpenPreview(singleSelectedSource.id)
                  }}
                  disabled={!singleSelectedSource}
                  className="rounded border border-border bg-surface px-2 py-0.5 text-[11px] text-text-muted transition hover:bg-surface2 hover:text-text disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {t("playground:sources.previewSelected", "Preview selected")}
                </button>
                <Popconfirm
                  title={t(
                    "playground:sources.batchRemoveConfirm",
                    "Remove {{count}} selected sources?",
                    { count: effectiveSelectedCount }
                  )}
                  description={batchRemoveDescription}
                  onConfirm={handleBatchRemoveSelected}
                  okText={t("common:remove", "Remove")}
                  cancelText={t("common:cancel", "Cancel")}
                  okButtonProps={{ danger: true }}
                >
                  <button
                    type="button"
                    data-testid="batch-remove-sources"
                    className="rounded border border-error/30 bg-error/10 px-2 py-0.5 text-[11px] font-medium text-error transition hover:bg-error/20"
                  >
                    {t("playground:sources.removeCount", "Remove ({{count}})", {
                      count: effectiveSelectedCount
                    })}
                  </button>
                </Popconfirm>
              </div>
            )}
          </div>
        )}

        <div className="px-4 py-2">
        <SourceFolderTree
          nodes={folderTreeNodes}
          activeFolderId={activeFolderId}
          selectionStateByFolderId={selectionStateByFolderId}
          onClearFocus={() => setActiveFolder(null)}
          onCreateFolder={handleCreateSourceFolder}
          onFocusFolder={setActiveFolder}
          onToggleFolderSelection={toggleSourceFolderSelection}
        />
        </div>
      </div>

      {/* Source list */}
      <div
        data-testid="sources-list-region"
        ref={sourceListContainerRef}
        onScroll={(event) =>
          setSourceListScrollTop(event.currentTarget.scrollTop)
        }
        className="custom-scrollbar min-h-0 flex-1 overflow-y-auto"
      >
        {filteredSources.length === 0 ? (
          <div className="flex h-full items-center justify-center p-4">
            <Empty
              image={Empty.PRESENTED_IMAGE_SIMPLE}
              description={
                sources.length === 0 ? (
                  <div className="text-center">
                    <p className="text-text-muted">
                      {t("playground:sources.empty", "No sources yet")}
                    </p>
                    <p className="mt-1 text-xs text-text-subtle">
                      {t(
                        "playground:sources.emptyHint",
                        "Add PDFs, web pages, videos, audio, or notes. tldw stores them in your configured local or self-hosted server and shows processing status here."
                      )}
                    </p>
                  </div>
                ) : (
                  <span className="text-text-muted">
                    {t("playground:sources.noResults", "No matching sources")}
                  </span>
                )
              }
            >
              {sources.length === 0 && (
                <Button
                  type="primary"
                  size="small"
                  icon={<Plus className="h-3.5 w-3.5" />}
                  onClick={() => openAddSourceModal("existing")}
                >
                  {t("playground:sources.addFirst", "Add your first source")}
                </Button>
              )}
            </Empty>
          </div>
        ) : useVirtualizedSources ? (
          <div
            data-testid="sources-virtualized-list"
            style={{
              height: filteredSources.length * SOURCE_VIRTUAL_ROW_HEIGHT,
              position: "relative"
            }}
          >
            <div
              className="space-y-1 p-2"
              style={{
                transform: `translateY(${virtualStartIndex * SOURCE_VIRTUAL_ROW_HEIGHT}px)`
              }}
            >
              {visibleSources.map((source) => renderSourceRow(source))}
            </div>
          </div>
        ) : (
          <div className="space-y-1 p-2">
            {visibleSources.map((source) => renderSourceRow(source))}
          </div>
        )}
      </div>

      {/* Footer with source count */}
      {sources.length > 0 && (
        <div className="shrink-0 border-t border-border px-4 py-2 text-xs text-text-muted">
          {t("playground:sources.totalCount", "{{count}} source(s)", {
            count: sources.length
          })}
        </div>
      )}

      {/* Source status details modal */}
      <Modal
        open={Boolean(statusDetailsSource)}
        title={t(
          "playground:sources.statusDetailsModalTitle",
          "Source status details"
        )}
        onCancel={handleCloseStatusDetails}
        footer={null}
        width={560}
      >
        {statusDetailsSource &&
          (() => {
            const details = statusDetailsSource.statusDetails
            const sourceStatus = statusGuardrailsEnabled
              ? statusDetailsSource.status || "ready"
              : "ready"
            const lifecycleLabel =
              humanizeStatusToken(details?.lifecycleState) ||
              (sourceStatus === "processing"
                ? t("playground:sources.statusProcessing", "Processing")
                : sourceStatus === "error"
                  ? t("playground:sources.statusErrorShort", "Error")
                  : t("playground:sources.statusReady", "Ready"))
            const statusReason =
              details?.statusReason ||
              details?.progressMessage ||
              statusDetailsSource.statusMessage ||
              t("playground:sources.notReported", "Not reported")
            const readiness = statusDetailsSource.readiness
            const readinessReadyCount = readiness
              ? READINESS_LABELS.filter(({ key }) => readiness[key]).length
              : null
            const progressPercent = getProgressPercent(details)
            const progressMessage = getProgressMessage(details)
            const progressSummary =
              progressPercent !== null && progressMessage
                ? `${progressPercent}% - ${progressMessage}`
                : progressPercent !== null
                  ? `${progressPercent}%`
                  : progressMessage ||
                    t("playground:sources.noProgressReported", "No progress reported")
            const jobLabel =
              details?.job?.uuid ||
              (typeof details?.job?.id === "number"
                ? String(details.job.id)
                : null)

            return (
              <div
                data-testid="source-status-details-dialog"
                className="space-y-3"
              >
                <div className="rounded border border-border bg-surface2/50 p-3">
                  <div className="flex flex-wrap items-start justify-between gap-2">
                    <div className="min-w-0">
                      <p className="truncate text-sm font-semibold text-text">
                        {statusDetailsSource.title}
                      </p>
                      <p className="mt-1 text-xs text-text-muted">
                        {t(
                          "playground:sources.statusDetailsSummary",
                          "{{type}} source currently marked {{status}}.",
                          {
                            type: statusDetailsSource.type,
                            status: lifecycleLabel
                          }
                        )}
                      </p>
                    </div>
                    <span
                      className={`rounded-full border px-2 py-0.5 text-[11px] font-semibold ${
                        sourceStatus === "error"
                          ? "border-error/30 bg-error/10 text-error"
                          : sourceStatus === "processing"
                            ? "border-primary/30 bg-primary/10 text-primary"
                            : "border-success/30 bg-success/10 text-success"
                      }`}
                    >
                      {lifecycleLabel}
                    </span>
                  </div>
                </div>

                <dl className="space-y-2">
                  <StatusDetailRow
                    label={t("playground:sources.lifecycleLabel", "Lifecycle")}
                  >
                    {lifecycleLabel}
                  </StatusDetailRow>
                  <StatusDetailRow
                    label={t(
                      "playground:sources.statusReasonLabel",
                      "Status reason"
                    )}
                  >
                    <span className="font-mono text-xs">{statusReason}</span>
                  </StatusDetailRow>
                  <StatusDetailRow
                    label={t(
                      "playground:sources.sourceOfTruthLabel",
                      "Source of truth"
                    )}
                  >
                    {describeSourceOfTruth(details?.sourceOfTruth)}
                  </StatusDetailRow>
                  <StatusDetailRow
                    label={t("playground:sources.lastRefreshLabel", "Last refresh")}
                  >
                    {formatStatusDateTime(details?.updatedAt)}
                  </StatusDetailRow>
                  <StatusDetailRow
                    label={t("playground:sources.progressLabel", "Progress")}
                  >
                    {progressSummary}
                  </StatusDetailRow>
                  <StatusDetailRow
                    label={t(
                      "playground:sources.retryEligibilityLabel",
                      "Retry eligibility"
                    )}
                  >
                    {describeRetryEligibility(sourceStatus, details)}
                  </StatusDetailRow>
                  <StatusDetailRow
                    label={t("playground:sources.staleStateLabel", "Stale state")}
                  >
                    {describeStaleState(details)}
                  </StatusDetailRow>
                  <StatusDetailRow
                    label={t("playground:sources.readinessLabel", "Readiness")}
                  >
                    {readinessReadyCount === null ? (
                      t(
                        "playground:sources.readinessNotReported",
                        "No readiness checklist reported"
                      )
                    ) : (
                      <div className="space-y-2">
                        <p>
                          {t(
                            "playground:sources.readinessSummary",
                            "{{ready}} of {{total}} checks ready",
                            {
                              ready: readinessReadyCount,
                              total: READINESS_LABELS.length
                            }
                          )}
                        </p>
                        <div className="flex flex-wrap gap-1">
                          {READINESS_LABELS.map(({ key, label }) => (
                            <span
                              key={key}
                              className={`rounded-full border px-2 py-0.5 text-[11px] font-medium ${
                                readiness[key]
                                  ? "border-success/30 bg-success/10 text-success"
                                  : "border-warning/30 bg-warning/10 text-warning"
                              }`}
                            >
                              {label}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </StatusDetailRow>
                  <StatusDetailRow
                    label={t("playground:sources.identifiersLabel", "Identifiers")}
                  >
                    <div className="space-y-1 font-mono text-xs">
                      <p>
                        {t("playground:sources.mediaIdLabel", "Media ID")}:{" "}
                        {statusDetailsSource.mediaId}
                      </p>
                      <p>
                        {t("playground:sources.sourceIdLabel", "Source ID")}:{" "}
                        {statusDetailsSource.id}
                      </p>
                      {jobLabel && (
                        <p>
                          {t("playground:sources.jobIdLabel", "Job")}: {jobLabel}
                        </p>
                      )}
                      {details?.job?.jobType && (
                        <p>
                          {t("playground:sources.jobTypeLabel", "Job type")}:{" "}
                          {details.job.jobType}
                        </p>
                      )}
                    </div>
                  </StatusDetailRow>
                  <StatusDetailRow
                    label={t("playground:sources.nextActionLabel", "Next action")}
                  >
                    {describeNextStatusAction(sourceStatus, details)}
                  </StatusDetailRow>
                </dl>
              </div>
            )
          })()}
      </Modal>

      {/* Source preview modal */}
      <Modal
        open={Boolean(previewSource)}
        title={t(
          "playground:sources.previewModalTitle",
          "Source preview and annotations"
        )}
        onCancel={handleClosePreview}
        footer={null}
        width={680}
      >
        {previewSource &&
          (() => {
            const previewStatus = statusGuardrailsEnabled
              ? previewSource.status || "ready"
              : "ready"
            const previewStatusLabel =
              previewStatus === "processing"
                ? t("playground:sources.statusProcessing", "Processing")
                : previewStatus === "error"
                  ? t("playground:sources.statusErrorShort", "Error")
                  : t("playground:sources.statusReady", "Ready")
            const previewData =
              sourcePreviewState.sourceId === previewSource.id
                ? sourcePreviewState.data
                : null
            const previewLoading =
              sourcePreviewState.sourceId === previewSource.id &&
              sourcePreviewState.loading
            const previewError =
              sourcePreviewState.sourceId === previewSource.id
                ? sourcePreviewState.error
                : null
            const sourcePreviewSnippets =
              previewData?.snippets?.filter(
                (snippet) => snippet.kind === "chunk" && snippet.text?.trim()
              ) ||
              []
            const previewTotalChars = previewData?.text_total_chars
            const previewTruncated = Boolean(previewData?.text_truncated)
            const formattedPreviewTotalChars =
              typeof previewTotalChars === "number"
                ? previewTotalChars.toLocaleString()
                : null
            const formattedPreviewShownChars = previewData?.text_preview
              ? previewData.text_preview.length.toLocaleString()
              : null
            const previewCharacterSummary =
              formattedPreviewTotalChars && previewTruncated
                ? t(
                    "playground:sources.previewTruncatedSummary",
                    "Showing first {{shown}} of {{total}} characters.",
                    {
                      shown: formattedPreviewShownChars,
                      total: formattedPreviewTotalChars
                    }
                  )
                : formattedPreviewTotalChars
                  ? t(
                      "playground:sources.previewFullSummary",
                      "Showing {{total}} characters.",
                      {
                        total: formattedPreviewTotalChars
                      }
                    )
                  : null

            return (
          <div className="space-y-4">
            <div className="rounded border border-border bg-surface2/40 p-3">
              <p className="text-sm font-semibold text-text">{previewSource.title}</p>
              <p className="text-xs capitalize text-text-muted">
                {previewSource.type} / {previewStatusLabel}
              </p>
              {previewSource.url && (
                <a
                  href={previewSource.url}
                  target="_blank"
                  rel="noreferrer"
                  className="mt-1 inline-block break-all text-xs text-primary hover:underline"
                >
                  {previewSource.url}
                </a>
              )}
            </div>

            <div className="rounded border border-border bg-surface/50 p-3">
              <div className="mb-2 flex items-center justify-between gap-2">
                <p className="text-xs font-semibold uppercase text-text-muted">
                  {t("playground:sources.capturedContent", "Captured content")}
                </p>
                {previewData?.readiness?.citation_ready && (
                  <span className="rounded bg-success/10 px-2 py-0.5 text-[11px] font-medium text-success">
                    {t("playground:sources.citationReady", "Citation ready")}
                  </span>
                )}
              </div>
              {previewLoading ? (
                <div className="flex items-center gap-2 text-sm text-text-muted">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  {t(
                    "playground:sources.previewLoading",
                    "Loading captured content..."
                  )}
                </div>
              ) : previewError ? (
                <div className="space-y-2">
                  <div className="flex items-start gap-2 text-sm text-warning">
                    <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
                    <span>
                      {t(
                        "playground:sources.previewLoadError",
                        "Source preview could not load."
                      )}
                    </span>
                  </div>
                  <p className="break-words rounded border border-border bg-surface2/40 p-2 font-mono text-xs text-warning">
                    {previewError}
                  </p>
                  <Button
                    size="small"
                    icon={<RefreshCw className="h-3.5 w-3.5" />}
                    onClick={() => setPreviewReloadNonce((value) => value + 1)}
                  >
                    {t("playground:sources.retryPreview", "Retry preview")}
                  </Button>
                </div>
              ) : previewData?.content_available && previewData.text_preview ? (
                <div className="space-y-2">
                  <p className="max-h-52 overflow-y-auto whitespace-pre-wrap rounded border border-border bg-surface2/40 p-2 text-sm leading-6 text-text">
                    {previewData.text_preview}
                  </p>
                  {previewCharacterSummary && (
                    <p className="text-xs text-text-muted">
                      {previewCharacterSummary}
                    </p>
                  )}
                </div>
              ) : (
                <p className="rounded border border-border bg-surface2/40 p-2 text-sm text-text-muted">
                  {describePreviewUnavailable(previewData, t)}
                </p>
              )}
            </div>

            <div className="rounded border border-border bg-surface/50 p-3">
              <p className="mb-2 text-xs font-semibold uppercase text-text-muted">
                {t("playground:sources.evidenceSnippets", "Evidence snippets")}
              </p>
              {sourcePreviewSnippets.length === 0 ? (
                <p className="text-xs text-text-muted">
                  {t(
                    "playground:sources.noEvidenceSnippets",
                    "No chunk evidence is available yet."
                  )}
                </p>
              ) : (
                <div className="max-h-48 space-y-2 overflow-y-auto pr-1">
                  {sourcePreviewSnippets.map((snippet) => (
                    <div
                      key={snippet.id}
                      className="rounded border border-border bg-surface2/40 p-2"
                    >
                      <div className="mb-1 flex flex-wrap items-center gap-2 text-[11px] text-text-muted">
                        <span>
                          {snippet.kind === "chunk"
                            ? t("playground:sources.chunkLabel", "Chunk")
                            : t(
                                "playground:sources.contentExcerptLabel",
                                "Content excerpt"
                              )}
                          {typeof snippet.chunk_index === "number"
                            ? ` ${snippet.chunk_index}`
                            : ""}
                        </span>
                        {typeof snippet.start_char === "number" &&
                          typeof snippet.end_char === "number" && (
                            <span>
                              {snippet.start_char}-{snippet.end_char}
                            </span>
                          )}
                      </div>
                      <p className="whitespace-pre-wrap text-sm text-text">
                        {snippet.text}
                      </p>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="rounded border border-border bg-surface/50 p-3">
              <p className="text-xs font-semibold uppercase text-text-muted">
                {t(
                  "playground:sources.localHighlights",
                  "Local highlights & annotations"
                )}
              </p>
              <p className="mb-2 text-xs text-text-muted">
                {t(
                  "playground:sources.localAnnotationsScope",
                  "Saved in this browser for this workspace."
                )}
              </p>
              <Input
                aria-label={t(
                  "playground:sources.annotationQuoteLabel",
                  "Highlighted excerpt"
                )}
                placeholder={t(
                  "playground:sources.annotationQuotePlaceholder",
                  "Highlighted excerpt (optional)"
                )}
                value={annotationQuoteDraft}
                onChange={(event) => setAnnotationQuoteDraft(event.target.value)}
                className="mb-2"
              />
              <Input.TextArea
                aria-label={t(
                  "playground:sources.annotationNoteLabel",
                  "Annotation note"
                )}
                placeholder={t(
                  "playground:sources.annotationNotePlaceholder",
                  "Annotation note"
                )}
                value={annotationNoteDraft}
                onChange={(event) => setAnnotationNoteDraft(event.target.value)}
                rows={3}
              />
              <div className="mt-2 flex items-center justify-end gap-2">
                {editingAnnotationId && (
                  <Button size="small" onClick={resetAnnotationEditor}>
                    {t("common:cancel", "Cancel")}
                  </Button>
                )}
                <Button
                  type="primary"
                  size="small"
                  onClick={handleSaveAnnotation}
                >
                  {editingAnnotationId
                    ? t("playground:sources.saveAnnotation", "Save annotation")
                    : t("playground:sources.addAnnotation", "Add annotation")}
                </Button>
              </div>
            </div>

            <div className="max-h-64 space-y-2 overflow-y-auto pr-1">
              {previewAnnotations.length === 0 ? (
                <p className="text-xs text-text-muted">
                  {t(
                    "playground:sources.noLocalAnnotations",
                    "No local annotations yet."
                  )}
                </p>
              ) : (
                previewAnnotations.map((annotation) => (
                  <div
                    key={annotation.id}
                    data-testid={`source-annotation-${annotation.id}`}
                    className="rounded border border-border bg-surface2/40 p-2"
                  >
                    {annotation.quote && (
                      <p className="text-xs text-text-muted">
                        "{annotation.quote}"
                      </p>
                    )}
                    {annotation.note && (
                      <p className="mt-1 text-sm text-text">{annotation.note}</p>
                    )}
                    <div className="mt-2 flex items-center justify-between">
                      <span className="text-[11px] text-text-muted">
                        {new Date(annotation.updatedAt).toLocaleString()}
                      </span>
                      <div className="flex items-center gap-1">
                        <Button
                          type="text"
                          size="small"
                          onClick={() => handleEditAnnotation(annotation)}
                        >
                          {t("common:edit", "Edit")}
                        </Button>
                        <Button
                          type="text"
                          danger
                          size="small"
                          onClick={() => handleDeleteAnnotation(annotation.id)}
                        >
                          {t("common:delete", "Delete")}
                        </Button>
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
            )
          })()}
      </Modal>

      <AddSourceModal />
    </div>
  )
}

export default SourcesPane
