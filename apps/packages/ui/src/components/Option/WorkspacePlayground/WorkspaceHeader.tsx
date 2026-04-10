import React from "react"
import { useTranslation } from "react-i18next"
import { useNavigate } from "react-router-dom"
import {
  Tooltip,
  Input,
  Dropdown,
  Modal,
  Button,
  message,
  type MenuProps
} from "antd"
import {
  FlaskConical,
  PanelLeftOpen,
  PanelRightOpen,
  Command,
  Pencil,
  Check,
  X,
  ChevronDown,
  Plus,
  BarChart3,
  Trash2,
  MessageSquare,
  FolderOpen,
  Copy,
  Archive,
  RotateCcw,
  Download,
  Upload,
  MoreHorizontal,
  Settings,
  Star,
  Share2,
  CircleHelp
} from "lucide-react"
import type {
  SavedWorkspace,
  WorkspaceBannerImage,
  WorkspaceCollection
} from "@/types/workspace"
import { useWorkspaceStore } from "@/store/workspace"
import { useConnectionStore } from "@/store/connection"
import { deriveConnectionUxState } from "@/types/connection"
import {
  buildWorkspacePlaygroundConfusionDashboardSnapshot,
  buildWorkspacePlaygroundTelemetryEventsCsv,
  getWorkspacePlaygroundTelemetryState,
  queryWorkspacePlaygroundTelemetryEvents,
  resetWorkspacePlaygroundTelemetryState,
  trackWorkspacePlaygroundTelemetry,
  WORKSPACE_PLAYGROUND_CONFUSION_EVENT_TYPES,
  type WorkspacePlaygroundTelemetryEventType,
  type WorkspacePlaygroundTelemetryState
} from "@/utils/workspace-playground-telemetry"
import {
  createWorkspaceExportFilename,
  createWorkspaceExportZipBlob,
  createWorkspaceExportZipFilename,
  parseWorkspaceImportFile
} from "@/store/workspace-bundle"
import {
  WORKSPACE_TEMPLATE_PRESETS,
  buildWorkspaceBibtex,
  createWorkspaceBibtexFilename,
  filterSavedWorkspaces,
  formatWorkspaceLastAccessed,
  groupWorkspacesByCollection
} from "./workspace-header.utils"
import { ShareDialog } from "./ShareDialog"
import {
  normalizeWorkspaceBannerImage,
  WorkspaceBannerImageNormalizationError
} from "./workspace-banner-image"
import {
  WORKSPACE_UNDO_WINDOW_MS,
  scheduleWorkspaceUndoAction,
  undoWorkspaceAction,
  clearAllPendingUndoActions
} from "./undo-manager"
import { useTutorialStore } from "@/store/tutorials"
import {
  FEATURE_ROLLOUT_PERCENTAGE_STORAGE_KEYS,
  FEATURE_ROLLOUT_SUBJECT_ID_STORAGE_KEY,
  createRolloutSubjectId,
  normalizeRolloutPercentage
} from "@/utils/feature-rollout"

interface WorkspaceHeaderProps {
  leftPaneOpen: boolean
  rightPaneOpen: boolean
  onToggleLeftPane: () => void
  onToggleRightPane: () => void
  onOpenSplitWorkspace?: () => void
  /** Hide pane toggle buttons (for mobile layout) */
  hideToggles?: boolean
  /** Approximate persisted workspace payload bytes in local storage */
  storageUsedBytes?: number
  /** Estimated available local storage budget for workspace payload data */
  storageQuotaBytes?: number
  /** Aggregate storage used by this browser profile/origin (if available). */
  storageOriginUsedBytes?: number
  /** Aggregate storage quota for this browser profile/origin (if available). */
  storageOriginQuotaBytes?: number
  /** Server-side storage used by the current user account (if available). */
  storageAccountUsedBytes?: number
  /** Server-side storage quota for the current user account (if available). */
  storageAccountQuotaBytes?: number
  /** Rollout gate for provenance surfaces (retrieval transparency, telemetry tools). */
  provenanceEnabled?: boolean
  /** Rollout gate for status/guardrails surfaces (connectivity/quota/conflict state). */
  statusGuardrailsEnabled?: boolean
}

const TELEMETRY_EVENT_ORDER: WorkspacePlaygroundTelemetryEventType[] = [
  "status_viewed",
  "citation_provenance_opened",
  "token_cost_rendered",
  "diagnostics_toggled",
  "quota_warning_seen",
  "conflict_modal_opened",
  "undo_triggered",
  "operation_cancelled",
  "artifact_rehydrated_failed",
  "source_status_polled",
  "source_status_ready",
  "connectivity_state_changed",
  "confusion_retry_burst",
  "confusion_refresh_loop",
  "confusion_duplicate_submission"
]

type WorkspaceRolloutControlKey =
  keyof typeof FEATURE_ROLLOUT_PERCENTAGE_STORAGE_KEYS

const WORKSPACE_ROLLOUT_CONTROL_ORDER: WorkspaceRolloutControlKey[] = [
  "research_studio_provenance_v1",
  "research_studio_status_guardrails_v1"
]
const WORKSPACE_ROLLOUT_PRESET_PERCENTAGES = [0, 10, 50, 100] as const

export const WorkspaceHeader: React.FC<WorkspaceHeaderProps> = ({
  leftPaneOpen,
  rightPaneOpen,
  onToggleLeftPane,
  onToggleRightPane,
  onOpenSplitWorkspace,
  hideToggles = false,
  storageUsedBytes,
  storageQuotaBytes,
  storageOriginUsedBytes,
  storageOriginQuotaBytes,
  storageAccountUsedBytes,
  storageAccountQuotaBytes,
  provenanceEnabled = true,
  statusGuardrailsEnabled = true
}) => {
  const { t } = useTranslation(["playground", "option", "common"])
  const navigate = useNavigate()
  const startTutorial = useTutorialStore((s) => s.startTutorial)
  const [messageApi, messageContextHolder] = message.useMessage()
  const [isEditing, setIsEditing] = React.useState(false)
  const [editName, setEditName] = React.useState("")
  const [workspaceBrowserOpen, setWorkspaceBrowserOpen] = React.useState(false)
  const [shortcutsModalOpen, setShortcutsModalOpen] = React.useState(false)
  const [deleteConfirmOpen, setDeleteConfirmOpen] = React.useState(false)
  const [deleteConfirmInput, setDeleteConfirmInput] = React.useState("")
  const [deleteTargetWorkspace, setDeleteTargetWorkspace] = React.useState<{
    id: string
    name: string
    sourceCount: number
  } | null>(null)
  const [telemetrySummaryOpen, setTelemetrySummaryOpen] = React.useState(false)
  const [telemetryLoading, setTelemetryLoading] = React.useState(false)
  const [telemetryResetting, setTelemetryResetting] = React.useState(false)
  const [telemetrySummary, setTelemetrySummary] =
    React.useState<WorkspacePlaygroundTelemetryState | null>(null)
  const [rolloutSubjectId, setRolloutSubjectId] = React.useState("")
  const [rolloutPercentages, setRolloutPercentages] = React.useState<
    Record<WorkspaceRolloutControlKey, number>
  >({
    research_studio_provenance_v1: 100,
    research_studio_status_guardrails_v1: 100,
    watchlists_ia_reduced_nav_v1: 100
  })
  const [workspaceSearchQuery, setWorkspaceSearchQuery] = React.useState("")
  const [workspaceCollectionDraft, setWorkspaceCollectionDraft] = React.useState("")
  const [workspaceCollectionFilter, setWorkspaceCollectionFilter] =
    React.useState("all")
  const PINNED_STORAGE_KEY = "tldw:workspace-playground:pinned-workspaces:v1"
  const [pinnedWorkspaceIds, setPinnedWorkspaceIds] = React.useState<Set<string>>(() => {
    try {
      const raw = window.localStorage.getItem(PINNED_STORAGE_KEY)
      if (raw) return new Set(JSON.parse(raw) as string[])
    } catch { /* ignore */ }
    return new Set()
  })
  const togglePinWorkspace = React.useCallback((id: string) => {
    setPinnedWorkspaceIds((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      try {
        window.localStorage.setItem(PINNED_STORAGE_KEY, JSON.stringify([...next]))
      } catch { /* ignore */ }
      return next
    })
  }, [])
  const [shareDialogOpen, setShareDialogOpen] = React.useState(false)
  const [bannerModalOpen, setBannerModalOpen] = React.useState(false)
  const [bannerTitleDraft, setBannerTitleDraft] = React.useState("")
  const [bannerSubtitleDraft, setBannerSubtitleDraft] = React.useState("")
  const [bannerImageDraft, setBannerImageDraft] =
    React.useState<WorkspaceBannerImage | null>(null)
  const [bannerImageUploading, setBannerImageUploading] = React.useState(false)
  const [bannerModalError, setBannerModalError] = React.useState<string | null>(
    null
  )
  const lastConnectivityStatusRef = React.useRef<string | null>(null)
  const importFileInputRef = React.useRef<HTMLInputElement | null>(null)
  const bannerFileInputRef = React.useRef<HTMLInputElement | null>(null)
  const telemetrySummaryEnabled = provenanceEnabled || statusGuardrailsEnabled
  const shortcutModifierLabel = React.useMemo(() => {
    if (typeof navigator === "undefined") return "Cmd/Ctrl"
    return /mac|iphone|ipad|ipod/i.test(navigator.platform) ? "Cmd" : "Ctrl"
  }, [])
  const formattedStorageUsage = React.useMemo(() => {
    if (
      typeof storageUsedBytes !== "number" ||
      !Number.isFinite(storageUsedBytes) ||
      storageUsedBytes < 0 ||
      typeof storageQuotaBytes !== "number" ||
      !Number.isFinite(storageQuotaBytes) ||
      storageQuotaBytes <= 0
    ) {
      return null
    }

    const usedMb = storageUsedBytes / (1024 * 1024)
    const quotaMb = storageQuotaBytes / (1024 * 1024)
    const roundedUsed = Math.round(usedMb * 10) / 10
    const roundedQuota = Math.round(quotaMb * 10) / 10
    const quotaLabel =
      Number.isInteger(roundedQuota) || Math.abs(roundedQuota - Math.round(roundedQuota)) < 0.05
        ? String(Math.round(roundedQuota))
        : roundedQuota.toFixed(1)
    const workspaceRatio = Math.max(0, Math.min(1, storageUsedBytes / storageQuotaBytes))

    const hasOriginUsage =
      typeof storageOriginUsedBytes === "number" &&
      Number.isFinite(storageOriginUsedBytes) &&
      storageOriginUsedBytes >= 0 &&
      typeof storageOriginQuotaBytes === "number" &&
      Number.isFinite(storageOriginQuotaBytes) &&
      storageOriginQuotaBytes > 0

    const hasAccountUsage =
      typeof storageAccountUsedBytes === "number" &&
      Number.isFinite(storageAccountUsedBytes) &&
      storageAccountUsedBytes >= 0 &&
      typeof storageAccountQuotaBytes === "number" &&
      Number.isFinite(storageAccountQuotaBytes) &&
      storageAccountQuotaBytes > 0

    let accountUsageShortLabel: string | null = null
    let accountRatio: number | null = null
    if (hasAccountUsage) {
      const accountUsedMb = storageAccountUsedBytes / (1024 * 1024)
      const accountQuotaMb = storageAccountQuotaBytes / (1024 * 1024)
      const roundedAccountUsed = Math.round(accountUsedMb * 10) / 10
      const roundedAccountQuota = Math.round(accountQuotaMb * 10) / 10
      const accountQuotaLabel =
        Number.isInteger(roundedAccountQuota) ||
        Math.abs(roundedAccountQuota - Math.round(roundedAccountQuota)) < 0.05
          ? String(Math.round(roundedAccountQuota))
          : roundedAccountQuota.toFixed(1)
      accountUsageShortLabel = `${roundedAccountUsed.toFixed(1)}/${accountQuotaLabel} MB`
      accountRatio = Math.max(
        0,
        Math.min(1, storageAccountUsedBytes / storageAccountQuotaBytes)
      )
    }

    const ratio = accountRatio ?? workspaceRatio
    const toneClass =
      ratio >= 0.95
        ? "border-error/40 bg-error/10 text-error"
        : ratio >= 0.8
          ? "border-warning/40 bg-warning/10 text-warning"
          : "border-border bg-surface2 text-text-muted"

    let profileUsageShortLabel: string | null = null
    if (hasOriginUsage) {
      const originUsedMb = storageOriginUsedBytes / (1024 * 1024)
      const originQuotaMb = storageOriginQuotaBytes / (1024 * 1024)
      const roundedOriginUsed = Math.round(originUsedMb * 10) / 10
      const roundedOriginQuota = Math.round(originQuotaMb * 10) / 10
      const originQuotaLabel =
        Number.isInteger(roundedOriginQuota) ||
        Math.abs(roundedOriginQuota - Math.round(roundedOriginQuota)) < 0.05
          ? String(Math.round(roundedOriginQuota))
          : roundedOriginQuota.toFixed(1)
      profileUsageShortLabel = `${roundedOriginUsed.toFixed(1)}/${originQuotaLabel} MB`
    }

    const workspaceShortLabel = `${roundedUsed.toFixed(1)}/${quotaLabel} MB`
    const workspaceShortSegment = `Payload ${workspaceShortLabel}`
    const shortLabel = accountUsageShortLabel
      ? `${workspaceShortSegment} | Account ${accountUsageShortLabel}`
      : profileUsageShortLabel
        ? `${workspaceShortSegment} | Browser ${profileUsageShortLabel}`
        : workspaceShortSegment

    const longLabel = accountUsageShortLabel
      ? profileUsageShortLabel
        ? t(
            "playground:workspace.storageUsageLabelWithAccountAndProfile",
            "Workspace payload: {{workspace}}. Account storage: {{account}}. Browser profile storage: {{profile}}.",
            {
              workspace: workspaceShortLabel,
              account: accountUsageShortLabel,
              profile: profileUsageShortLabel
            }
          )
        : t(
            "playground:workspace.storageUsageLabelWithAccount",
            "Workspace payload: {{workspace}}. Account storage: {{account}}.",
            {
              workspace: workspaceShortLabel,
              account: accountUsageShortLabel
            }
          )
      : profileUsageShortLabel
        ? t(
            "playground:workspace.storageUsageLabelWithProfile",
            "Workspace payload: {{workspace}}. Browser profile storage: {{profile}}.",
            {
              workspace: workspaceShortLabel,
              profile: profileUsageShortLabel
            }
          )
        : t(
            "playground:workspace.storageUsageLabelWorkspaceOnly",
            "Workspace payload storage: {{workspace}}",
            {
              workspace: workspaceShortLabel
            }
          )

    return {
      ratio,
      toneClass,
      shortLabel,
      longLabel
    }
  }, [
    storageAccountQuotaBytes,
    storageAccountUsedBytes,
    storageOriginQuotaBytes,
    storageOriginUsedBytes,
    storageQuotaBytes,
    storageUsedBytes,
    t
  ])
  const connectionState = useConnectionStore((s) => s.state)
  const connectionIndicator = React.useMemo(() => {
    const uxState = deriveConnectionUxState(connectionState)
    const sharedDescription =
      connectionState.lastError || connectionState.knowledgeError || null

    if (uxState === "connected_ok") {
      return {
        label: t("playground:workspace.connectionConnected", "Connected"),
        detail: t(
          "playground:workspace.connectionConnectedDetail",
          "Connection healthy"
        ),
        toneClass: "border-success/40 bg-success/10 text-success",
        description: sharedDescription
      }
    }

    if (
      uxState === "testing" ||
      uxState === "connected_degraded" ||
      uxState === "demo_mode"
    ) {
      return {
        label: t("playground:workspace.connectionDegraded", "Degraded"),
        detail: t(
          "playground:workspace.connectionDegradedDetail",
          "Connection degraded or still checking"
        ),
        toneClass: "border-warning/40 bg-warning/10 text-warning",
        description: sharedDescription
      }
    }

    return {
      label: t("playground:workspace.connectionDisconnected", "Disconnected"),
      detail: t(
        "playground:workspace.connectionDisconnectedDetail",
        "Cannot reach backend service"
      ),
      toneClass: "border-error/40 bg-error/10 text-error",
      description: sharedDescription
    }
  }, [connectionState, t])

  const workspaceName = useWorkspaceStore((s) => s.workspaceName)
  const workspaceId = useWorkspaceStore((s) => s.workspaceId)
  const workspaceTag = useWorkspaceStore((s) => s.workspaceTag)
  const workspaceBanner = useWorkspaceStore((s) => s.workspaceBanner) || {
    title: "",
    subtitle: "",
    image: null
  }
  const sources = useWorkspaceStore((s) => s.sources)
  const setWorkspaceName = useWorkspaceStore((s) => s.setWorkspaceName)
  const setWorkspaceBanner =
    useWorkspaceStore((s) => s.setWorkspaceBanner) || (() => undefined)
  const clearWorkspaceBannerImage =
    useWorkspaceStore((s) => s.clearWorkspaceBannerImage) || (() => undefined)
  const resetWorkspaceBanner =
    useWorkspaceStore((s) => s.resetWorkspaceBanner) || (() => undefined)
  const currentNote = useWorkspaceStore((s) => s.currentNote)
  const setCurrentNote = useWorkspaceStore((s) => s.setCurrentNote)
  const savedWorkspaces = useWorkspaceStore((s) => s.savedWorkspaces)
  const archivedWorkspaces = useWorkspaceStore((s) => s.archivedWorkspaces)
  const workspaceCollections = useWorkspaceStore((s) => s.workspaceCollections)
  const createNewWorkspace = useWorkspaceStore((s) => s.createNewWorkspace)
  const createWorkspaceCollection = useWorkspaceStore(
    (s) => s.createWorkspaceCollection
  )
  const deleteWorkspaceCollection = useWorkspaceStore(
    (s) => s.deleteWorkspaceCollection
  )
  const assignWorkspaceToCollection = useWorkspaceStore(
    (s) => s.assignWorkspaceToCollection
  )
  const exportWorkspaceBundle = useWorkspaceStore((s) => s.exportWorkspaceBundle)
  const importWorkspaceBundle = useWorkspaceStore((s) => s.importWorkspaceBundle)
  const switchWorkspace = useWorkspaceStore((s) => s.switchWorkspace)
  const duplicateWorkspace = useWorkspaceStore((s) => s.duplicateWorkspace)
  const archiveWorkspace = useWorkspaceStore((s) => s.archiveWorkspace)
  const restoreArchivedWorkspace = useWorkspaceStore(
    (s) => s.restoreArchivedWorkspace
  )
  const deleteWorkspace = useWorkspaceStore((s) => s.deleteWorkspace)
  const captureUndoSnapshot = useWorkspaceStore((s) => s.captureUndoSnapshot)
  const restoreUndoSnapshot = useWorkspaceStore((s) => s.restoreUndoSnapshot)
  const saveCurrentWorkspace = useWorkspaceStore((s) => s.saveCurrentWorkspace)

  React.useEffect(() => {
    if (!statusGuardrailsEnabled) {
      lastConnectivityStatusRef.current = null
      return
    }
    const nextStatus = connectionIndicator.label.toLowerCase()
    const previousStatus = lastConnectivityStatusRef.current
    if (previousStatus === nextStatus) return

    void trackWorkspacePlaygroundTelemetry({
      type: "connectivity_state_changed",
      workspace_id: workspaceId || null,
      from: previousStatus,
      to: nextStatus
    })

    lastConnectivityStatusRef.current = nextStatus
  }, [connectionIndicator.label, statusGuardrailsEnabled, workspaceId])

  const handleStartEdit = () => {
    setEditName(workspaceName || "New Research")
    setIsEditing(true)
  }

  const handleSaveEdit = () => {
    if (editName.trim()) {
      setWorkspaceName(editName.trim())
    }
    setIsEditing(false)
  }

  const handleCancelEdit = () => {
    setIsEditing(false)
    setEditName("")
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleSaveEdit()
    } else if (e.key === "Escape") {
      handleCancelEdit()
    }
  }

  const handleGoToSimpleChat = () => {
    // Save current workspace before navigating
    saveCurrentWorkspace()
    navigate("/")
  }

  const handleCreateNewWorkspace = () => {
    createNewWorkspace()
  }

  const handleCreateWorkspaceFromTemplate = (templateId: string) => {
    const template = WORKSPACE_TEMPLATE_PRESETS.find(
      (candidate) => candidate.id === templateId
    )
    if (!template) return

    createNewWorkspace(template.workspaceName)
    setCurrentNote({
      id: undefined,
      title: template.noteTitle,
      content: template.noteContent,
      keywords: [...template.keywords],
      version: undefined,
      isDirty: true
    })

    messageApi.success(
      t("playground:workspace.templateCreated", {
        defaultValue: "Created workspace from template: {{template}}",
        template: template.label
      })
    )
  }

  const handleSwitchWorkspace = (id: string) => {
    // Finalize and discard pending undo actions to prevent cross-workspace undo
    clearAllPendingUndoActions()
    if (currentNote?.isDirty) {
      Modal.confirm({
        title: t(
          "playground:workspace.unsavedNoteTitle",
          "Unsaved note changes"
        ),
        content: t(
          "playground:workspace.unsavedNoteMessage",
          "You have unsaved changes in your note. What would you like to do?"
        ),
        okText: t("playground:workspace.saveAndSwitch", "Save & Switch"),
        cancelText: t("common:cancel", "Cancel"),
        onOk: () => {
          saveCurrentWorkspace()
          switchWorkspace(id)
        },
        footer: (_, { OkBtn, CancelBtn }) => (
          <>
            <CancelBtn />
            <Button
              onClick={() => {
                Modal.destroyAll()
                switchWorkspace(id)
              }}
            >
              {t("playground:workspace.discardAndSwitch", "Discard & Switch")}
            </Button>
            <OkBtn />
          </>
        )
      })
      return
    }
    switchWorkspace(id)
  }

  const handleOpenWorkspaceBrowser = () => {
    setWorkspaceSearchQuery("")
    setWorkspaceBrowserOpen(true)
  }

  const handleCloseWorkspaceBrowser = () => {
    setWorkspaceBrowserOpen(false)
    setWorkspaceSearchQuery("")
  }

  const handleOpenShortcutsModal = () => {
    setShortcutsModalOpen(true)
  }

  const handleCloseShortcutsModal = () => {
    setShortcutsModalOpen(false)
  }

  const handleOpenCustomizeBannerModal = () => {
    setBannerTitleDraft(workspaceBanner.title || "")
    setBannerSubtitleDraft(workspaceBanner.subtitle || "")
    setBannerImageDraft(workspaceBanner.image || null)
    setBannerModalError(null)
    setBannerModalOpen(true)
  }

  const handleCloseCustomizeBannerModal = () => {
    setBannerModalOpen(false)
    setBannerModalError(null)
    setBannerImageUploading(false)
  }

  const handleSaveCustomizeBanner = () => {
    if (workspaceBanner.image && !bannerImageDraft) {
      clearWorkspaceBannerImage()
    }
    setWorkspaceBanner({
      title: bannerTitleDraft,
      subtitle: bannerSubtitleDraft,
      image: bannerImageDraft
    })
    setBannerModalOpen(false)
    setBannerModalError(null)
    messageApi.success(
      t("playground:workspace.customizeBannerSaved", "Banner updated")
    )
  }

  const handleResetCustomizeBanner = () => {
    Modal.confirm({
      title: t("playground:workspace.customizeBannerResetTitle", "Reset banner?"),
      content: t(
        "playground:workspace.customizeBannerResetMessage",
        "This clears title, subtitle, and image for this workspace banner."
      ),
      okText: t("playground:workspace.customizeBannerReset", "Reset banner"),
      okButtonProps: { danger: true },
      cancelText: t("common:cancel", "Cancel"),
      onOk: () => {
        resetWorkspaceBanner()
        setBannerTitleDraft("")
        setBannerSubtitleDraft("")
        setBannerImageDraft(null)
        setBannerModalOpen(false)
        setBannerModalError(null)
      },
      centered: true,
      maskClosable: false
    })
  }

  const handleBannerImageFileChange = async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = event.target.files?.[0]
    event.target.value = ""
    if (!file) return

    setBannerImageUploading(true)
    setBannerModalError(null)
    try {
      const normalizedImage = await normalizeWorkspaceBannerImage(file)
      setBannerImageDraft(normalizedImage)
    } catch (error) {
      if (error instanceof WorkspaceBannerImageNormalizationError) {
        if (error.code === "unsupported_mime_type") {
          setBannerModalError(
            t(
              "playground:workspace.customizeBannerUnsupportedType",
              "Upload a JPG, PNG, or WebP image."
            )
          )
        } else if (error.code === "image_too_large") {
          setBannerModalError(
            t(
              "playground:workspace.customizeBannerTooLarge",
              "Image is too large after processing. Try a smaller image."
            )
          )
        } else {
          setBannerModalError(
            t(
              "playground:workspace.customizeBannerProcessingError",
              "Could not process that image. Try another file."
            )
          )
        }
      } else {
        setBannerModalError(
          t(
            "playground:workspace.customizeBannerProcessingError",
            "Could not process that image. Try another file."
          )
        )
      }
      return
    } finally {
      setBannerImageUploading(false)
    }
  }

  const handlePromptBannerImageUpload = () => {
    bannerFileInputRef.current?.click()
  }

  const handleRemoveBannerImage = () => {
    setBannerImageDraft(null)
    setBannerModalError(null)
  }

  const loadTelemetrySummary = React.useCallback(async () => {
    setTelemetryLoading(true)
    try {
      const snapshot = await getWorkspacePlaygroundTelemetryState()
      setTelemetrySummary(snapshot)
    } finally {
      setTelemetryLoading(false)
    }
  }, [])

  const loadRolloutControls = React.useCallback(() => {
    if (typeof window === "undefined") return

    try {
      const storedSubjectId = window.localStorage.getItem(
        FEATURE_ROLLOUT_SUBJECT_ID_STORAGE_KEY
      )
      setRolloutSubjectId(storedSubjectId?.trim() || "")
      setRolloutPercentages({
        research_studio_provenance_v1: normalizeRolloutPercentage(
          window.localStorage.getItem(
            FEATURE_ROLLOUT_PERCENTAGE_STORAGE_KEYS.research_studio_provenance_v1
          ),
          100
        ),
        research_studio_status_guardrails_v1: normalizeRolloutPercentage(
          window.localStorage.getItem(
            FEATURE_ROLLOUT_PERCENTAGE_STORAGE_KEYS
              .research_studio_status_guardrails_v1
          ),
          100
        ),
        watchlists_ia_reduced_nav_v1: normalizeRolloutPercentage(
          window.localStorage.getItem(
            FEATURE_ROLLOUT_PERCENTAGE_STORAGE_KEYS.watchlists_ia_reduced_nav_v1
          ),
          100
        )
      })
    } catch {
      setRolloutSubjectId("")
      setRolloutPercentages({
        research_studio_provenance_v1: 100,
        research_studio_status_guardrails_v1: 100,
        watchlists_ia_reduced_nav_v1: 100
      })
    }
  }, [])

  const handleOpenTelemetrySummary = () => {
    setTelemetrySummaryOpen(true)
    loadRolloutControls()
    void loadTelemetrySummary()
  }

  const handleCloseTelemetrySummary = () => {
    setTelemetrySummaryOpen(false)
  }

  const handleResetTelemetrySummary = async () => {
    setTelemetryResetting(true)
    try {
      await resetWorkspacePlaygroundTelemetryState()
      await loadTelemetrySummary()
      messageApi.success(
        t(
          "playground:workspace.telemetrySummaryResetSuccess",
          "Telemetry summary reset."
        )
      )
    } catch {
      messageApi.error(
        t(
          "playground:workspace.telemetrySummaryResetError",
          "Unable to reset telemetry summary."
        )
      )
    } finally {
      setTelemetryResetting(false)
    }
  }

  const telemetryCounters = React.useMemo(
    () =>
      TELEMETRY_EVENT_ORDER.map((eventType) => ({
        eventType,
        count: telemetrySummary?.counters?.[eventType] || 0
      })),
    [telemetrySummary]
  )

  const recentTelemetryEvents = React.useMemo(
    () => (telemetrySummary?.recent_events || []).slice(-12).reverse(),
    [telemetrySummary]
  )
  const confusionDashboard = React.useMemo(
    () => buildWorkspacePlaygroundConfusionDashboardSnapshot(telemetrySummary),
    [telemetrySummary]
  )
  const confusionEventsForExport = React.useMemo(
    () =>
      queryWorkspacePlaygroundTelemetryEvents(telemetrySummary, {
        eventTypes: WORKSPACE_PLAYGROUND_CONFUSION_EVENT_TYPES
      }),
    [telemetrySummary]
  )
  const handleSetRolloutPercentage = React.useCallback(
    (flag: WorkspaceRolloutControlKey, percentage: number) => {
      const normalizedPercentage = normalizeRolloutPercentage(percentage, 100)
      setRolloutPercentages((previousState) => ({
        ...previousState,
        [flag]: normalizedPercentage
      }))
      if (typeof window === "undefined") return

      try {
        window.localStorage.setItem(
          FEATURE_ROLLOUT_PERCENTAGE_STORAGE_KEYS[flag],
          String(normalizedPercentage)
        )
        messageApi.success(
          t(
            "playground:workspace.rolloutControlUpdated",
            "Rollout control updated."
          )
        )
      } catch {
        messageApi.error(
          t(
            "playground:workspace.rolloutControlUpdateError",
            "Unable to update rollout control."
          )
        )
      }
    },
    [messageApi, t]
  )
  const handleRegenerateRolloutSubject = React.useCallback(() => {
    const nextSubjectId = createRolloutSubjectId()
    setRolloutSubjectId(nextSubjectId)
    if (typeof window === "undefined") return

    try {
      window.localStorage.setItem(
        FEATURE_ROLLOUT_SUBJECT_ID_STORAGE_KEY,
        nextSubjectId
      )
      messageApi.success(
        t(
          "playground:workspace.rolloutSubjectRegenerated",
          "Rollout subject regenerated."
        )
      )
    } catch {
      messageApi.error(
        t(
          "playground:workspace.rolloutSubjectRegenerateError",
          "Unable to regenerate rollout subject."
        )
      )
    }
  }, [messageApi, t])
  const handleResetRolloutControls = React.useCallback(() => {
    const resetValue = 100
    setRolloutPercentages({
      research_studio_provenance_v1: resetValue,
      research_studio_status_guardrails_v1: resetValue,
      watchlists_ia_reduced_nav_v1: resetValue
    })
    if (typeof window === "undefined") return

    try {
      for (const flag of WORKSPACE_ROLLOUT_CONTROL_ORDER) {
        window.localStorage.setItem(
          FEATURE_ROLLOUT_PERCENTAGE_STORAGE_KEYS[flag],
          String(resetValue)
        )
      }
      messageApi.success(
        t(
          "playground:workspace.rolloutControlReset",
          "Rollout controls reset to 100%."
        )
      )
    } catch {
      messageApi.error(
        t(
          "playground:workspace.rolloutControlResetError",
          "Unable to reset rollout controls."
        )
      )
    }
  }, [messageApi, t])

  const formatTelemetryEventLabel = React.useCallback(
    (eventType: WorkspacePlaygroundTelemetryEventType) =>
      eventType
        .split("_")
        .map((segment) => segment[0]?.toUpperCase() + segment.slice(1))
        .join(" "),
    []
  )

  const formatTelemetryTimestamp = React.useCallback((value: number) => {
    const date = new Date(value)
    if (Number.isNaN(date.getTime())) return "Unknown"
    return date.toLocaleString()
  }, [])
  const formatTelemetryRate = React.useCallback((value: number) => {
    if (!Number.isFinite(value) || value <= 0) return "0%"
    return `${(value * 100).toFixed(1)}%`
  }, [])
  const createTelemetryExportTimestamp = React.useCallback(() => {
    return new Date().toISOString().replace(/[:.]/g, "-")
  }, [])

  const handleDeleteWorkspace = (id: string, e: React.MouseEvent) => {
    e.stopPropagation()
    const target = savedWorkspaces.find((w) => w.id === id)
    setDeleteTargetWorkspace(
      target
        ? { id: target.id, name: target.name, sourceCount: target.sourceCount }
        : { id, name: workspaceName || "Untitled", sourceCount: sources.length }
    )
    setDeleteConfirmInput("")
    setDeleteConfirmOpen(true)
  }

  const executeDeleteWorkspace = () => {
    if (!deleteTargetWorkspace) return
    const { id } = deleteTargetWorkspace
    setDeleteConfirmOpen(false)
    setDeleteTargetWorkspace(null)
    setDeleteConfirmInput("")

    const undoSnapshot = captureUndoSnapshot()
    const undoHandle = scheduleWorkspaceUndoAction({
      apply: () => {
        deleteWorkspace(id)
      },
      undo: () => {
        restoreUndoSnapshot(undoSnapshot)
      }
    })

    const undoMessageKey = `workspace-delete-undo-${undoHandle.id}`
    const maybeOpen = (messageApi as { open?: (config: unknown) => void })
      .open
    const messageConfig = {
      key: undoMessageKey,
      type: "warning",
      duration: WORKSPACE_UNDO_WINDOW_MS / 1000,
      content: t(
        "playground:workspace.deleted",
        "Workspace deleted."
      ),
      btn: (
        <button
          type="button"
          className="rounded border border-border px-2 py-0.5 text-xs font-medium hover:bg-surface2"
          onClick={() => {
            if (undoWorkspaceAction(undoHandle.id)) {
              messageApi.success(
                t(
                  "playground:workspace.restored",
                  "Workspace restored"
                )
              )
            }
            messageApi.destroy(undoMessageKey)
          }}
        >
          {t("common:undo", "Undo")}
        </button>
      )
    }
    if (typeof maybeOpen === "function") {
      maybeOpen(messageConfig)
    } else {
      const maybeWarning = (
        messageApi as { warning?: (content: string) => void }
      ).warning
      if (typeof maybeWarning === "function") {
        maybeWarning(t("playground:workspace.deleted", "Workspace deleted."))
      }
    }
  }

  const handleDuplicateCurrentWorkspace = () => {
    if (!workspaceId) return
    duplicateWorkspace(workspaceId)
  }

  const handleArchiveCurrentWorkspace = () => {
    if (!workspaceId) return

    Modal.confirm({
      title: t("playground:workspace.archiveTitle", "Archive current workspace?"),
      content: t(
        "playground:workspace.archiveMessage",
        "You can restore archived workspaces later from this menu."
      ),
      okText: t("playground:workspace.archive", "Archive"),
      cancelText: t("common:cancel", "Cancel"),
      onOk: () => {
        const undoSnapshot = captureUndoSnapshot()
        const undoHandle = scheduleWorkspaceUndoAction({
          apply: () => {
            archiveWorkspace(workspaceId)
          },
          undo: () => {
            restoreUndoSnapshot(undoSnapshot)
          }
        })

        const undoMessageKey = `workspace-archive-undo-${undoHandle.id}`
        const maybeOpen = (messageApi as { open?: (config: unknown) => void })
          .open
        const messageConfig = {
          key: undoMessageKey,
          type: "warning",
          duration: WORKSPACE_UNDO_WINDOW_MS / 1000,
          content: t(
            "playground:workspace.archived",
            "Workspace archived."
          ),
          btn: (
            <Button
              size="small"
              type="link"
              onClick={() => {
                if (undoWorkspaceAction(undoHandle.id)) {
                  messageApi.success(
                    t("playground:workspace.archiveRestored", "Workspace restored")
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
            maybeWarning(t("playground:workspace.archived", "Workspace archived."))
          }
        }
      },
      centered: true,
      maskClosable: true
    })
  }

  const handleRestoreWorkspace = (id: string) => {
    restoreArchivedWorkspace(id)
    handleSwitchWorkspace(id)
  }

  const triggerFileDownload = (blob: Blob, filename: string) => {
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement("a")
    anchor.href = url
    anchor.download = filename
    anchor.click()
    URL.revokeObjectURL(url)
  }

  const handleExportTelemetrySummaryJson = () => {
    if (!telemetrySummary) {
      messageApi.info(
        t(
          "playground:workspace.telemetrySummaryNoData",
          "No telemetry data available to export."
        )
      )
      return
    }

    const payload = {
      exported_at: new Date().toISOString(),
      source: "workspace-playground-telemetry",
      telemetry: telemetrySummary
    }
    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: "application/json;charset=utf-8"
    })
    triggerFileDownload(
      blob,
      `workspace-telemetry-summary-${createTelemetryExportTimestamp()}.json`
    )
    messageApi.success(
      t(
        "playground:workspace.telemetrySummaryExported",
        "Telemetry summary exported."
      )
    )
  }

  const handleExportConfusionCsv = () => {
    if (confusionEventsForExport.length === 0) {
      messageApi.info(
        t(
          "playground:workspace.telemetryConfusionNoData",
          "No confusion indicator events available for export."
        )
      )
      return
    }

    const csv = buildWorkspacePlaygroundTelemetryEventsCsv(confusionEventsForExport)
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" })
    triggerFileDownload(
      blob,
      `workspace-telemetry-confusion-${createTelemetryExportTimestamp()}.csv`
    )
    messageApi.success(
      t(
        "playground:workspace.telemetryConfusionExported",
        "Confusion indicators exported (CSV)."
      )
    )
  }

  const handleExportCurrentWorkspace = async () => {
    if (!workspaceId) return
    const bundle = exportWorkspaceBundle(workspaceId)
    if (!bundle) {
      messageApi.error(
        t(
          "playground:workspace.exportFailed",
          "Unable to export this workspace."
        )
      )
      return
    }

    const filename = createWorkspaceExportFilename(
      bundle.workspace.name,
      bundle.exportedAt
    )
    try {
      const zipBlob = await createWorkspaceExportZipBlob(bundle)
      const zipFilename = createWorkspaceExportZipFilename(
        bundle.workspace.name,
        bundle.exportedAt
      )
      triggerFileDownload(zipBlob, zipFilename)
      messageApi.success(
        t("playground:workspace.exportSuccessZip", "Workspace exported (.zip)")
      )
      return
    } catch {
      const jsonBlob = new Blob([JSON.stringify(bundle, null, 2)], {
        type: "application/json;charset=utf-8"
      })
      triggerFileDownload(jsonBlob, filename)
      messageApi.info(
        t(
          "playground:workspace.exportZipFallback",
          "ZIP export unavailable. Downloaded JSON bundle instead."
        )
      )
      messageApi.success(
        t("playground:workspace.exportSuccess", "Workspace exported")
      )
    }
  }

  const handleOpenImportWorkspace = () => {
    importFileInputRef.current?.click()
  }

  const handleExportWorkspaceCitations = () => {
    if (sources.length === 0) {
      messageApi.error(
        t(
          "playground:workspace.exportCitationsEmpty",
          "Add at least one source before exporting citations."
        )
      )
      return
    }

    const bibtex = buildWorkspaceBibtex(sources, { workspaceTag })
    if (!bibtex.trim()) {
      messageApi.error(
        t(
          "playground:workspace.exportCitationsFailed",
          "Unable to build citations for this workspace."
        )
      )
      return
    }

    const filename = createWorkspaceBibtexFilename(
      workspaceName || "workspace"
    )
    const blob = new Blob([bibtex], { type: "text/plain;charset=utf-8" })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement("a")
    anchor.href = url
    anchor.download = filename
    anchor.click()
    URL.revokeObjectURL(url)

    messageApi.success(
      t(
        "playground:workspace.exportCitationsSuccess",
        "Citations exported (BibTeX)"
      )
    )
  }

  const handleImportWorkspaceFile = async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = event.target.files?.[0]
    event.target.value = ""
    if (!file) return

    try {
      const parsed = await parseWorkspaceImportFile(file)
      const importedWorkspaceId = importWorkspaceBundle(parsed)
      if (!importedWorkspaceId) {
        throw new Error("import-failed")
      }

      messageApi.success(
        t("playground:workspace.importSuccess", "Workspace imported")
      )
    } catch {
      messageApi.error(
        t(
          "playground:workspace.importFailed",
          "Unable to import this workspace file."
        )
      )
    }
  }

  const savedCountLabel = (workspace: SavedWorkspace) =>
    `${workspace.sourceCount} ${
      workspace.sourceCount === 1 ? "source" : "sources"
    }`

  const savedRelativeLabel = (workspace: SavedWorkspace) =>
    formatWorkspaceLastAccessed(new Date(workspace.lastAccessedAt))

  const filteredWorkspaceBrowserItems = React.useMemo(() => {
    const searchFiltered = filterSavedWorkspaces(
      savedWorkspaces,
      workspaceSearchQuery
    ).sort((a, b) => {
      const aPinned = pinnedWorkspaceIds.has(a.id) ? 0 : 1
      const bPinned = pinnedWorkspaceIds.has(b.id) ? 0 : 1
      return aPinned - bPinned
    })

    if (workspaceCollectionFilter === "all") {
      return searchFiltered
    }

    if (workspaceCollectionFilter === "unassigned") {
      return searchFiltered.filter((workspace) => workspace.collectionId === null)
    }

    return searchFiltered.filter(
      (workspace) => workspace.collectionId === workspaceCollectionFilter
    )
  }, [
    pinnedWorkspaceIds,
    savedWorkspaces,
    workspaceCollectionFilter,
    workspaceSearchQuery
  ])

  const visibleWorkspaceGroups = React.useMemo(
    () =>
      groupWorkspacesByCollection(
        workspaceCollections,
        filteredWorkspaceBrowserItems
      ).filter(
        (group) =>
          group.workspaces.length > 0 ||
          (workspaceSearchQuery.trim().length === 0 &&
            workspaceCollectionFilter === "all")
      ),
    [
      filteredWorkspaceBrowserItems,
      workspaceCollectionFilter,
      workspaceCollections,
      workspaceSearchQuery
    ]
  )

  const handleCreateWorkspaceCollection = () => {
    const nextName = workspaceCollectionDraft.trim()
    if (!nextName) return
    createWorkspaceCollection(nextName, null)
    setWorkspaceCollectionDraft("")
  }

  const handleDeleteWorkspaceCollection = (collectionId: string) => {
    deleteWorkspaceCollection(collectionId)
    if (workspaceCollectionFilter === collectionId) {
      setWorkspaceCollectionFilter("all")
    }
  }

  const handleWorkspaceCollectionAssignment = (
    workspaceIdToAssign: string,
    nextCollectionId: string
  ) => {
    assignWorkspaceToCollection(
      workspaceIdToAssign,
      nextCollectionId ? nextCollectionId : null
    )
  }

  const workspaceCollectionOptions = React.useMemo<
    Array<Pick<WorkspaceCollection, "id" | "name">>
  >(
    () =>
      workspaceCollections.map((collection) => ({
        id: collection.id,
        name: collection.name
      })),
    [workspaceCollections]
  )

  // ── Workspace Switcher dropdown: recent/pinned workspaces + navigation ──
  const workspaceSwitcherItems: MenuProps["items"] = [
    // Recent workspaces section
    ...(savedWorkspaces.length > 0
      ? [
          {
            key: "recent-header",
            type: "group" as const,
            label: t("playground:workspace.recentWorkspaces", "Recent Workspaces")
          },
          ...savedWorkspaces
            .filter((w) => w.id !== workspaceId) // Don't show current workspace
            .slice(0, 5) // Show max 5 recent
            .map((workspace) => ({
              key: workspace.id,
              label: (
                <div className="flex items-center justify-between gap-2">
                  <div className="flex min-w-0 flex-1 items-center gap-2">
                    <FolderOpen className="h-4 w-4 shrink-0 text-text-muted" />
                    <div className="min-w-0">
                      <div className="truncate">{workspace.name}</div>
                      <div className="text-xs text-text-muted">
                        {savedCountLabel(workspace)} • {savedRelativeLabel(workspace)}
                      </div>
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={(e) => handleDeleteWorkspace(workspace.id, e)}
                    className="shrink-0 rounded p-1 text-text-muted opacity-0 transition hover:bg-error/10 hover:text-error group-hover:opacity-100 [.ant-dropdown-menu-item:hover_&]:opacity-100"
                    title={t("common:delete", "Delete")}
                    aria-label={t("common:delete", "Delete")}
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </div>
              ),
              onClick: () => handleSwitchWorkspace(workspace.id)
            })),
          { type: "divider" as const, key: "divider-1" }
        ]
      : []),
    ...(archivedWorkspaces.length > 0
      ? [
          {
            key: "archived-header",
            type: "group" as const,
            label: t("playground:workspace.archivedWorkspaces", "Archived Workspaces")
          },
          ...archivedWorkspaces.slice(0, 5).map((workspace) => ({
            key: `archived-${workspace.id}`,
            icon: <RotateCcw className="h-4 w-4" />,
            label: (
              <div className="flex min-w-0 items-center gap-2">
                <span className="truncate">{workspace.name}</span>
                <span className="shrink-0 text-xs text-text-muted">
                  ({workspace.sourceCount} {workspace.sourceCount === 1 ? "source" : "sources"})
                </span>
              </div>
            ),
            onClick: () => handleRestoreWorkspace(workspace.id)
          })),
          { type: "divider" as const, key: "divider-archived" }
        ]
      : []),
    ...(savedWorkspaces.length > 0
      ? [
          {
            key: "view-all-workspaces",
            icon: <FolderOpen className="h-4 w-4" />,
            label: t(
              "playground:workspace.viewAll",
              "View all workspaces"
            ),
            onClick: handleOpenWorkspaceBrowser
          },
          { type: "divider" as const, key: "divider-view-all" }
        ]
      : []),
    {
      key: "new",
      icon: <Plus className="h-4 w-4" />,
      label: t("playground:workspace.newWorkspace", "New Workspace"),
      onClick: handleCreateNewWorkspace
    }
  ]

  // ── Settings kebab menu: import/export, templates, tools, navigation ──
  const workspaceSettingsItems: MenuProps["items"] = [
    ...(workspaceId
      ? [
          {
            key: "duplicate-current",
            icon: <Copy className="h-4 w-4" />,
            label: t(
              "playground:workspace.duplicateCurrent",
              "Duplicate Current Workspace"
            ),
            onClick: handleDuplicateCurrentWorkspace
          },
          {
            key: "archive-current",
            icon: <Archive className="h-4 w-4" />,
            label: t(
              "playground:workspace.archiveCurrent",
              "Archive Current Workspace"
            ),
            onClick: handleArchiveCurrentWorkspace
          },
          {
            key: "customize-banner",
            icon: <Pencil className="h-4 w-4" />,
            label: t(
              "playground:workspace.customizeBanner",
              "Customize banner"
            ),
            onClick: handleOpenCustomizeBannerModal
          },
          ...(onOpenSplitWorkspace
            ? [
                {
                  key: "split-workspace",
                  icon: <PanelLeftOpen className="h-4 w-4" />,
                  label: t(
                    "playground:workspace.splitWorkspace",
                    "Split workspace"
                  ),
                  onClick: onOpenSplitWorkspace
                }
              ]
            : []),
          { type: "divider" as const, key: "divider-current-actions" }
        ]
      : []),
    ...(workspaceId
      ? [
          {
            key: "export-workspace",
            icon: <Download className="h-4 w-4" />,
            label: t(
              "playground:workspace.exportWorkspace",
              "Export Workspace"
            ),
            onClick: handleExportCurrentWorkspace
          },
          {
            key: "export-citations-bibtex",
            icon: <Download className="h-4 w-4" />,
            label: t(
              "playground:workspace.exportCitationsBibtex",
              "Export Citations (BibTeX)"
            ),
            onClick: handleExportWorkspaceCitations
          }
        ]
      : []),
    {
      key: "import-workspace",
      icon: <Upload className="h-4 w-4" />,
      label: t("playground:workspace.importWorkspace", "Import Workspace"),
      onClick: handleOpenImportWorkspace
    },
    { type: "divider" as const, key: "divider-import-export" },
    {
      key: "template-header",
      type: "group" as const,
      label: t("playground:workspace.templatesHeader", "Start from Template")
    },
    ...WORKSPACE_TEMPLATE_PRESETS.map((template) => ({
      key: `workspace-template-${template.id}`,
      icon: <Plus className="h-4 w-4" />,
      label: template.label,
      onClick: () => handleCreateWorkspaceFromTemplate(template.id)
    })),
    { type: "divider" as const, key: "divider-templates" },
    {
      key: "replay-tour",
      icon: <FlaskConical className="h-4 w-4" />,
      label: t("playground:workspace.replayTour", "Replay tour"),
      onClick: () => startTutorial("workspace-playground-basics")
    },
    {
      key: "keyboard-shortcuts",
      icon: <Command className="h-4 w-4" />,
      label: t("playground:workspace.keyboardShortcuts", "Keyboard Shortcuts"),
      onClick: handleOpenShortcutsModal
    },
    ...(telemetrySummaryEnabled
      ? [
          {
            key: "telemetry-summary",
            icon: <BarChart3 className="h-4 w-4" />,
            label: t("playground:workspace.telemetrySummary", "Telemetry summary"),
            onClick: handleOpenTelemetrySummary
          }
        ]
      : []),
    { type: "divider" as const, key: "divider-2" },
    {
      key: "simple-chat",
      icon: <MessageSquare className="h-4 w-4" />,
      label: t("playground:workspace.goToSimpleChat", "Simple Chat"),
      onClick: handleGoToSimpleChat
    }
  ]

  return (
    <header
      data-testid="workspace-header"
      className="flex items-center justify-between border-b border-border/70 bg-[linear-gradient(90deg,var(--surface)_0%,var(--surface-2)_100%)] px-4 py-3.5"
    >
      {messageContextHolder}
      <div className="flex items-center gap-3">
        <span className="inline-flex h-8 w-8 items-center justify-center rounded-full border border-primary/30 bg-primary/10">
          <FlaskConical className="h-4 w-4 text-primary" />
        </span>
        <div className="flex items-center gap-2">
          {isEditing ? (
            <div className="flex items-center gap-1">
              <Input
                value={editName}
                onChange={(e) => setEditName(e.target.value)}
                onKeyDown={handleKeyDown}
                autoFocus
                size="small"
                className="w-48"
                placeholder={t(
                  "playground:workspace.namePlaceholder",
                  "Workspace name"
                )}
              />
              <button
                type="button"
                onClick={handleSaveEdit}
                className="rounded p-1 text-primary hover:bg-primary/10"
                aria-label={t("common:save", "Save")}
              >
                <Check className="h-4 w-4" />
              </button>
              <button
                type="button"
                onClick={handleCancelEdit}
                className="rounded p-1 text-text-muted hover:bg-surface2 hover:text-text"
                aria-label={t("common:cancel", "Cancel")}
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          ) : (
            <div className="flex items-center gap-2">
              <h1 className="text-lg font-semibold text-text">
                {workspaceName || t("playground:workspace.title", "Research Workspace")}
              </h1>
              <Tooltip title={t("playground:workspace.rename", "Rename workspace")}>
                <button
                  type="button"
                  onClick={handleStartEdit}
                  className="flex h-9 w-9 items-center justify-center rounded text-text-muted opacity-40 transition hover:bg-surface2 hover:text-text hover:opacity-100"
                  aria-label={t("playground:workspace.rename", "Rename workspace")}
                >
                  <Pencil className="h-3.5 w-3.5" />
                </button>
              </Tooltip>
            </div>
          )}
        </div>
      </div>

      <div
        data-testid="workspace-header-actions"
        className="flex items-center gap-2"
      >
        {/* Left pane expand button (only shown when collapsed) */}
        {!hideToggles && !leftPaneOpen && (
          <Tooltip
            title={t(
              "playground:workspace.showSourcesShortcut",
              `Show sources (${shortcutModifierLabel}+1)`
            )}
          >
            <button
              type="button"
              onClick={onToggleLeftPane}
              className="hidden items-center gap-1.5 rounded-lg bg-primary/10 p-2 text-primary transition-colors hover:bg-primary/20 lg:flex"
              aria-label={t(
                "playground:workspace.showSourcesShortcut",
                `Show sources (${shortcutModifierLabel}+1)`
              )}
            >
              <PanelLeftOpen className="h-4 w-4" />
              <kbd className="rounded bg-primary/15 px-1 py-0.5 text-[10px] font-medium leading-none text-primary/70">{shortcutModifierLabel}+1</kbd>
            </button>
          </Tooltip>
        )}

        {/* Right pane expand button (only shown when collapsed) */}
        {!hideToggles && !rightPaneOpen && (
          <Tooltip
            title={t(
              "playground:workspace.showStudioShortcut",
              `Show studio (${shortcutModifierLabel}+3)`
            )}
          >
            <button
              type="button"
              onClick={onToggleRightPane}
              className="hidden items-center gap-1.5 rounded-lg bg-primary/10 p-2 text-primary transition-colors hover:bg-primary/20 lg:flex"
              aria-label={t(
                "playground:workspace.showStudioShortcut",
                `Show studio (${shortcutModifierLabel}+3)`
              )}
            >
              <PanelRightOpen className="h-4 w-4" />
              <kbd className="rounded bg-primary/15 px-1 py-0.5 text-[10px] font-medium leading-none text-primary/70">{shortcutModifierLabel}+3</kbd>
            </button>
          </Tooltip>
        )}

        {/* Workspace Switcher Dropdown */}
        <Dropdown
          menu={{ items: workspaceSwitcherItems }}
          trigger={["click"]}
          placement="bottomRight"
        >
          <button
            type="button"
            data-testid="workspace-workspaces-button"
            className="flex items-center gap-1.5 rounded-lg border border-border bg-surface px-3 py-1.5 text-sm font-medium text-text transition hover:bg-surface2"
          >
            <span>{t("playground:workspace.workspaces", "Workspaces")}</span>
            <ChevronDown className="h-4 w-4 text-text-muted" />
          </button>
        </Dropdown>

        {/* Share Button */}
        <Tooltip title={t("playground:workspace.share", "Share workspace")}>
          <button
            type="button"
            data-testid="workspace-share-button"
            className="flex items-center justify-center rounded-lg border border-border bg-surface p-1.5 text-text-muted transition hover:bg-surface2 hover:text-text"
            aria-label={t("playground:workspace.share", "Share workspace")}
            onClick={() => setShareDialogOpen(true)}
          >
            <Share2 className="h-4 w-4" />
          </button>
        </Tooltip>

        {/* Help / Tour Button */}
        <Tooltip title={t("playground:workspace.takeTour", "Take a tour")}>
          <button
            type="button"
            data-testid="workspace-help-tour-button"
            onClick={() => startTutorial("workspace-playground-basics")}
            className="inline-flex h-8 w-8 items-center justify-center rounded-md border border-border bg-surface text-text-muted hover:bg-surface2 hover:text-text transition-colors"
            aria-label={t("playground:workspace.takeTour", "Take a tour of the workspace")}
          >
            <CircleHelp className="h-4 w-4" />
          </button>
        </Tooltip>

        {/* Settings Kebab Menu */}
        <Dropdown
          menu={{ items: workspaceSettingsItems }}
          trigger={["click"]}
          placement="bottomRight"
        >
          <Tooltip title={t("playground:workspace.workspaceSettings", "Workspace settings")}>
            <button
              type="button"
              data-testid="workspace-settings-button"
              className="flex items-center justify-center rounded-lg border border-border bg-surface p-1.5 text-text-muted transition hover:bg-surface2 hover:text-text"
              aria-label={t("playground:workspace.workspaceSettings", "Workspace settings")}
            >
              <MoreHorizontal className="h-4 w-4" />
            </button>
          </Tooltip>
        </Dropdown>
      </div>

      <input
        ref={importFileInputRef}
        type="file"
        accept=".json,.workspace.json,.zip,.workspace.zip"
        className="hidden"
        data-testid="workspace-import-input"
        onChange={(event) => {
          void handleImportWorkspaceFile(event)
        }}
      />

      <input
        ref={bannerFileInputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        className="hidden"
        data-testid="workspace-banner-upload-input"
        onChange={(event) => {
          void handleBannerImageFileChange(event)
        }}
      />

      <Modal
        title={t("playground:workspace.customizeBanner", "Customize banner")}
        open={bannerModalOpen}
        onCancel={handleCloseCustomizeBannerModal}
        onOk={handleSaveCustomizeBanner}
        okText={t("common:save", "Save")}
        cancelText={t("common:cancel", "Cancel")}
        width={560}
        destroyOnHidden
        footer={[
          <Button
            key="reset-banner"
            danger
            type="default"
            onClick={handleResetCustomizeBanner}
          >
            {t("playground:workspace.customizeBannerReset", "Reset banner")}
          </Button>,
          <Button key="cancel-banner" onClick={handleCloseCustomizeBannerModal}>
            {t("common:cancel", "Cancel")}
          </Button>,
          <Button key="save-banner" type="primary" onClick={handleSaveCustomizeBanner}>
            {t("common:save", "Save")}
          </Button>
        ]}
      >
        <div className="space-y-3">
          <Input
            value={bannerTitleDraft}
            onChange={(event) => setBannerTitleDraft(event.target.value)}
            placeholder={t(
              "playground:workspace.customizeBannerTitlePlaceholder",
              "Banner title"
            )}
            maxLength={80}
            data-testid="workspace-banner-title-input"
          />
          <Input.TextArea
            value={bannerSubtitleDraft}
            onChange={(event) => setBannerSubtitleDraft(event.target.value)}
            placeholder={t(
              "playground:workspace.customizeBannerSubtitlePlaceholder",
              "Banner subtitle"
            )}
            maxLength={180}
            rows={3}
            data-testid="workspace-banner-subtitle-input"
          />
          <div
            data-testid="workspace-banner-preview"
            className="overflow-hidden rounded-xl border border-border/70"
            style={{
              backgroundImage: bannerImageDraft?.dataUrl
                ? `linear-gradient(125deg, rgba(8, 12, 18, 0.68) 0%, rgba(8, 12, 18, 0.2) 100%), url(${bannerImageDraft.dataUrl})`
                : "linear-gradient(125deg, color-mix(in oklab, var(--primary) 24%, transparent) 0%, color-mix(in oklab, var(--surface-2) 76%, transparent) 100%)",
              backgroundPosition: "center",
              backgroundSize: "cover"
            }}
          >
            <div className="min-h-[132px] space-y-1 px-4 py-3 text-white">
              <p className="line-clamp-2 text-lg font-semibold">
                {bannerTitleDraft.trim() || workspaceName || "Research Workspace"}
              </p>
              {bannerSubtitleDraft.trim().length > 0 && (
                <p className="line-clamp-2 text-sm text-white/90">
                  {bannerSubtitleDraft.trim()}
                </p>
              )}
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Button
              onClick={handlePromptBannerImageUpload}
              loading={bannerImageUploading}
              data-testid="workspace-banner-upload-trigger"
            >
              {t("playground:workspace.customizeBannerUpload", "Upload image")}
            </Button>
            {bannerImageDraft && (
              <Button
                danger
                onClick={handleRemoveBannerImage}
                data-testid="workspace-banner-remove-image"
              >
                {t("playground:workspace.customizeBannerRemoveImage", "Remove image")}
              </Button>
            )}
          </div>
          {bannerModalError && (
            <p
              data-testid="workspace-banner-modal-error"
              className="rounded border border-error/40 bg-error/10 px-3 py-2 text-sm text-error"
            >
              {bannerModalError}
            </p>
          )}
        </div>
      </Modal>

      <Modal
        title={t("playground:workspace.allWorkspaces", "All Workspaces")}
        open={workspaceBrowserOpen}
        onCancel={handleCloseWorkspaceBrowser}
        footer={null}
        width={680}
        destroyOnHidden
      >
        <div className="space-y-3">
          <Input
            value={workspaceSearchQuery}
            onChange={(event) => setWorkspaceSearchQuery(event.target.value)}
            placeholder={t(
              "playground:workspace.searchPlaceholder",
              "Search workspaces by name or tag"
            )}
            allowClear
          />

          <div className="flex flex-wrap items-center gap-2">
            <label className="flex items-center gap-2 text-sm text-text-muted">
              <span>{t("playground:workspace.collectionFilter", "Collection")}</span>
              <select
                aria-label={t(
                  "playground:workspace.collectionFilterLabel",
                  "Filter by collection"
                )}
                value={workspaceCollectionFilter}
                onChange={(event) =>
                  setWorkspaceCollectionFilter(event.target.value)
                }
                className="rounded-md border border-border bg-surface px-2 py-1 text-sm text-text"
              >
                <option value="all">
                  {t("playground:workspace.collectionFilterAll", "All collections")}
                </option>
                <option value="unassigned">
                  {t(
                    "playground:workspace.collectionFilterUnassigned",
                    "Unassigned"
                  )}
                </option>
                {workspaceCollectionOptions.map((collection) => (
                  <option key={collection.id} value={collection.id}>
                    {collection.name}
                  </option>
                ))}
              </select>
            </label>

            <div className="flex min-w-[240px] flex-1 items-center gap-2">
              <Input
                value={workspaceCollectionDraft}
                onChange={(event) =>
                  setWorkspaceCollectionDraft(event.target.value)
                }
                placeholder={t(
                  "playground:workspace.collectionCreatePlaceholder",
                  "New collection name"
                )}
              />
              <Button type="default" onClick={handleCreateWorkspaceCollection}>
                {t("playground:workspace.collectionCreate", "Add collection")}
              </Button>
            </div>
          </div>

          <div className="custom-scrollbar max-h-[360px] space-y-1 overflow-y-auto rounded-lg border border-border p-1">
            {visibleWorkspaceGroups.map((group) => (
              <div key={group.id} className="space-y-1">
                <div className="flex items-center justify-between px-2 pt-2 text-xs font-semibold uppercase tracking-[0.08em] text-text-muted">
                  <span
                    aria-label={t(
                      "playground:workspace.collectionGroupLabel",
                      `Collection group ${group.name}`
                    )}
                  >
                    {group.name}
                  </span>
                  <div className="flex items-center gap-2">
                    <span>{group.workspaces.length}</span>
                    {group.collection && (
                      <button
                        type="button"
                        onClick={() =>
                          handleDeleteWorkspaceCollection(group.collection.id)
                        }
                        className="rounded p-1 text-text-muted transition hover:bg-error/10 hover:text-error"
                        aria-label={t(
                          "playground:workspace.collectionDeleteLabel",
                          `Delete collection ${group.name}`
                        )}
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    )}
                  </div>
                </div>

                {group.workspaces.map((workspace) => {
                  const isCurrent = workspace.id === workspaceId
                  const isPinned = pinnedWorkspaceIds.has(workspace.id)
                  return (
                    <div
                      key={workspace.id}
                      className={`flex w-full items-center gap-2 rounded-md border px-3 py-2 transition ${
                        isCurrent
                          ? "cursor-default border-primary/30 bg-primary/10"
                          : "border-border hover:bg-surface2"
                      }`}
                    >
                      <button
                        type="button"
                        disabled={isCurrent}
                        onClick={() => {
                          handleCloseWorkspaceBrowser()
                          handleSwitchWorkspace(workspace.id)
                        }}
                        className="min-w-0 flex-1 text-left"
                      >
                        <div className="flex items-center justify-between gap-3">
                          <div className="min-w-0">
                            <div className="truncate font-medium text-text">
                              {workspace.name}
                            </div>
                            <div className="truncate text-xs text-text-muted">
                              {workspace.tag}
                            </div>
                          </div>
                          <div className="shrink-0 text-right text-xs text-text-muted">
                            <div>{savedCountLabel(workspace)}</div>
                            <div>
                              {t("playground:workspace.lastAccessed", "Last accessed")}{" "}
                              {savedRelativeLabel(workspace)}
                            </div>
                          </div>
                        </div>
                      </button>

                      <select
                        aria-label={t(
                          "playground:workspace.collectionAssignmentLabel",
                          `Collection for ${workspace.name}`
                        )}
                        value={workspace.collectionId || ""}
                        onChange={(event) =>
                          handleWorkspaceCollectionAssignment(
                            workspace.id,
                            event.target.value
                          )
                        }
                        className="max-w-[150px] rounded-md border border-border bg-surface px-2 py-1 text-xs text-text"
                      >
                        <option value="">
                          {t(
                            "playground:workspace.collectionUnassigned",
                            "Unassigned"
                          )}
                        </option>
                        {workspaceCollectionOptions.map((collection) => (
                          <option key={collection.id} value={collection.id}>
                            {collection.name}
                          </option>
                        ))}
                      </select>

                      <Tooltip
                        title={
                          isPinned
                            ? t("playground:workspace.unpin", "Unpin")
                            : t("playground:workspace.pin", "Pin to top")
                        }
                      >
                        <button
                          type="button"
                          data-testid={`pin-workspace-${workspace.id}`}
                          onClick={(e) => {
                            e.stopPropagation()
                            togglePinWorkspace(workspace.id)
                          }}
                          className={`shrink-0 rounded p-1 transition ${
                            isPinned
                              ? "text-warning"
                              : "text-text-muted opacity-0 group-hover:opacity-100 hover:text-warning"
                          }`}
                          style={isPinned ? undefined : { opacity: 1 }}
                        >
                          <Star
                            className={`h-3.5 w-3.5 ${
                              isPinned ? "fill-current" : ""
                            }`}
                          />
                        </button>
                      </Tooltip>
                    </div>
                  )
                })}
              </div>
            ))}

            {filteredWorkspaceBrowserItems.length === 0 && (
              <div className="px-3 py-6 text-center text-sm text-text-muted">
                {t(
                  "playground:workspace.noMatches",
                  "No workspaces match your search."
                )}
              </div>
            )}
          </div>
        </div>
      </Modal>

      <Modal
        title={t("playground:workspace.keyboardShortcuts", "Keyboard Shortcuts")}
        open={shortcutsModalOpen}
        onCancel={handleCloseShortcutsModal}
        footer={null}
        width={520}
        destroyOnHidden
      >
        <div className="space-y-2">
          {[
            {
              action: t("playground:workspace.shortcutSearch", "Search workspace"),
              combo: `${shortcutModifierLabel}+K`
            },
            {
              action: t("playground:workspace.shortcutFocusSources", "Focus sources"),
              combo: `${shortcutModifierLabel}+1`
            },
            {
              action: t("playground:workspace.shortcutFocusChat", "Focus chat"),
              combo: `${shortcutModifierLabel}+2`
            },
            {
              action: t("playground:workspace.shortcutFocusStudio", "Focus studio"),
              combo: `${shortcutModifierLabel}+3`
            },
            {
              action: t("playground:workspace.shortcutNewNote", "New note"),
              combo: `${shortcutModifierLabel}+N`
            },
            {
              action: t("playground:workspace.shortcutNewWorkspace", "New workspace"),
              combo: `${shortcutModifierLabel}+Shift+N`
            },
            {
              action: t("playground:workspace.shortcutUndo", "Undo last action"),
              combo: `${shortcutModifierLabel}+Z`
            }
          ].map((item) => (
            <div
              key={item.action}
              className="flex items-center justify-between rounded border border-border px-3 py-2"
            >
              <span className="text-sm text-text">{item.action}</span>
              <code className="rounded bg-surface2 px-2 py-0.5 text-xs font-semibold text-text">
                {item.combo}
              </code>
            </div>
          ))}
        </div>
      </Modal>

      <Modal
        title={t("playground:workspace.telemetrySummary", "Telemetry summary")}
        open={telemetrySummaryEnabled && telemetrySummaryOpen}
        onCancel={handleCloseTelemetrySummary}
        width={720}
        destroyOnHidden
        footer={[
          <Button key="export-json" onClick={handleExportTelemetrySummaryJson}>
            {t(
              "playground:workspace.telemetrySummaryExportJson",
              "Export JSON"
            )}
          </Button>,
          <Button key="export-csv" onClick={handleExportConfusionCsv}>
            {t(
              "playground:workspace.telemetrySummaryExportCsv",
              "Export confusion CSV"
            )}
          </Button>,
          <Button key="refresh" onClick={() => void loadTelemetrySummary()}>
            {t("playground:workspace.telemetrySummaryRefresh", "Refresh")}
          </Button>,
          <Button
            key="reset"
            danger
            loading={telemetryResetting}
            onClick={() => void handleResetTelemetrySummary()}
          >
            {t("playground:workspace.telemetrySummaryReset", "Reset")}
          </Button>,
          <Button key="close" type="primary" onClick={handleCloseTelemetrySummary}>
            {t("common:close", "Close")}
          </Button>
        ]}
      >
        <div
          data-testid="workspace-telemetry-summary-modal"
          className="space-y-4"
          aria-live="polite"
        >
          {telemetryLoading ? (
            <div className="rounded border border-border bg-surface2 px-3 py-2 text-sm text-text-muted">
              {t(
                "playground:workspace.telemetrySummaryLoading",
                "Loading telemetry summary..."
              )}
            </div>
          ) : (
            <>
              <div className="grid gap-2 text-xs text-text-muted sm:grid-cols-2">
                <div className="rounded border border-border bg-surface2 px-3 py-2">
                  {t("playground:workspace.telemetrySummaryEvents", "Tracked events")}:{" "}
                  <span className="font-semibold text-text">
                    {telemetryCounters.reduce((total, item) => total + item.count, 0)}
                  </span>
                </div>
                <div className="rounded border border-border bg-surface2 px-3 py-2">
                  {t(
                    "playground:workspace.telemetrySummaryLastEvent",
                    "Last event"
                  )}
                  :{" "}
                  <span className="font-semibold text-text">
                    {telemetrySummary?.last_event_at
                      ? formatTelemetryTimestamp(telemetrySummary.last_event_at)
                      : t(
                          "playground:workspace.telemetrySummaryNoEvents",
                          "No events yet"
                        )}
                  </span>
                </div>
              </div>

              <div>
                <h3 className="mb-2 text-sm font-semibold text-text">
                  {t(
                    "playground:workspace.telemetryConfusionDashboard",
                    "Confusion Indicators Dashboard"
                  )}
                </h3>
                <div className="grid gap-2 text-xs sm:grid-cols-2 lg:grid-cols-3">
                  <div className="rounded border border-border bg-surface2 px-3 py-2">
                    <p className="text-text-muted">
                      {t(
                        "playground:workspace.telemetryConfusionRetryBurst",
                        "Retry bursts"
                      )}
                    </p>
                    <p className="text-sm font-semibold text-text">
                      {confusionDashboard.counters.retryBurst}
                    </p>
                    <p className="text-text-muted">
                      {formatTelemetryRate(confusionDashboard.rates.retryPerStatusView)}{" "}
                      {t(
                        "playground:workspace.telemetryConfusionPerStatusView",
                        "per status view"
                      )}
                    </p>
                  </div>
                  <div className="rounded border border-border bg-surface2 px-3 py-2">
                    <p className="text-text-muted">
                      {t(
                        "playground:workspace.telemetryConfusionRefreshLoop",
                        "Refresh loops"
                      )}
                    </p>
                    <p className="text-sm font-semibold text-text">
                      {confusionDashboard.counters.refreshLoop}
                    </p>
                    <p className="text-text-muted">
                      {formatTelemetryRate(confusionDashboard.rates.refreshPerConflict)}{" "}
                      {t(
                        "playground:workspace.telemetryConfusionPerConflict",
                        "per conflict modal"
                      )}
                    </p>
                  </div>
                  <div className="rounded border border-border bg-surface2 px-3 py-2">
                    <p className="text-text-muted">
                      {t(
                        "playground:workspace.telemetryConfusionDuplicate",
                        "Duplicate submissions"
                      )}
                    </p>
                    <p className="text-sm font-semibold text-text">
                      {confusionDashboard.counters.duplicateSubmission}
                    </p>
                    <p className="text-text-muted">
                      {formatTelemetryRate(
                        confusionDashboard.rates.duplicatePerStatusView
                      )}{" "}
                      {t(
                        "playground:workspace.telemetryConfusionPerStatusView",
                        "per status view"
                      )}
                    </p>
                  </div>
                </div>
                <div className="mt-2 grid gap-2 text-xs text-text-muted sm:grid-cols-2">
                  <div className="rounded border border-border bg-surface2 px-3 py-2">
                    {t("playground:workspace.telemetryConfusion24h", "Confusion events (24h)")}:{" "}
                    <span className="font-semibold text-text">
                      {confusionDashboard.windowedCounts.last24h}
                    </span>
                  </div>
                  <div className="rounded border border-border bg-surface2 px-3 py-2">
                    {t("playground:workspace.telemetryConfusion7d", "Confusion events (7d)")}:{" "}
                    <span className="font-semibold text-text">
                      {confusionDashboard.windowedCounts.last7d}
                    </span>
                  </div>
                </div>
                <div className="mt-2 rounded border border-border bg-surface2 px-3 py-2 text-[11px] text-text-muted">
                  <p className="font-medium text-text-subtle">
                    {t(
                      "playground:workspace.telemetryDashboardQueries",
                      "Dashboard query formulas"
                    )}
                  </p>
                  <p>
                    <code>
                      sessions_with_confusion_retry_burst / active_sessions
                    </code>
                  </p>
                  <p>
                    <code>
                      sessions_with_confusion_refresh_loop / sessions_with_conflicts
                    </code>
                  </p>
                  <p>
                    <code>
                      sessions_with_confusion_duplicate_submission / active_sessions
                    </code>
                  </p>
                </div>
              </div>

              <div>
                <h3 className="mb-2 text-sm font-semibold text-text">
                  {t(
                    "playground:workspace.telemetrySummaryCounters",
                    "Event counters"
                  )}
                </h3>
                <div className="mb-2 rounded border border-border bg-surface2 px-3 py-2 text-[11px] text-text-muted">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <span className="font-medium text-text-subtle">
                      {t(
                        "playground:workspace.rolloutExecutionControls",
                        "Rollout execution controls"
                      )}
                    </span>
                    <div className="flex flex-wrap gap-1.5">
                      <Button
                        size="small"
                        onClick={loadRolloutControls}
                        data-testid="workspace-rollout-refresh"
                      >
                        {t("playground:workspace.rolloutRefresh", "Refresh")}
                      </Button>
                      <Button
                        size="small"
                        onClick={handleRegenerateRolloutSubject}
                        data-testid="workspace-rollout-regenerate-subject"
                      >
                        {t(
                          "playground:workspace.rolloutRegenerateSubject",
                          "Regenerate subject"
                        )}
                      </Button>
                      <Button
                        size="small"
                        onClick={handleResetRolloutControls}
                        data-testid="workspace-rollout-reset"
                      >
                        {t("playground:workspace.rolloutReset", "Reset to 100%")}
                      </Button>
                    </div>
                  </div>
                  <p className="mt-1">
                    {t(
                      "playground:workspace.rolloutSubject",
                      "Subject ID"
                    )}
                    :{" "}
                    <code
                      className="rounded bg-surface px-1 py-0.5 text-text"
                      data-testid="workspace-rollout-subject-id"
                    >
                      {rolloutSubjectId || t("common:unknown", "Unknown")}
                    </code>
                  </p>
                  <div className="mt-2 space-y-2">
                    {WORKSPACE_ROLLOUT_CONTROL_ORDER.map((flag) => {
                      const currentPercentage = rolloutPercentages[flag] ?? 100
                      const isSurfaceEnabled =
                        flag === "research_studio_provenance_v1"
                          ? provenanceEnabled
                          : statusGuardrailsEnabled
                      const label =
                        flag === "research_studio_provenance_v1"
                          ? t(
                              "playground:workspace.rolloutProvenanceLabel",
                              "Provenance trust surfaces"
                            )
                          : t(
                              "playground:workspace.rolloutStatusLabel",
                              "Status guardrail surfaces"
                            )

                      return (
                        <div
                          key={flag}
                          className="rounded border border-border bg-surface px-2 py-1.5"
                          data-testid={`workspace-rollout-control-${flag}`}
                        >
                          <p className="text-xs font-medium text-text">{label}</p>
                          <p
                            className="text-[11px] text-text-muted"
                            data-testid={`workspace-rollout-percentage-${flag}`}
                          >
                            {`${t(
                              "playground:workspace.rolloutCurrentOverride",
                              "Override"
                            )}: ${currentPercentage}%`}{" "}
                            ·{" "}
                            {isSurfaceEnabled
                              ? t(
                                  "playground:workspace.rolloutSurfaceEnabled",
                                  "enabled for this subject"
                                )
                              : t(
                                  "playground:workspace.rolloutSurfaceDisabled",
                                  "disabled for this subject"
                                )}
                          </p>
                          <div className="mt-1 flex flex-wrap gap-1.5">
                            {WORKSPACE_ROLLOUT_PRESET_PERCENTAGES.map((percentage) => (
                              <Button
                                key={`${flag}-${percentage}`}
                                size="small"
                                type={
                                  currentPercentage === percentage
                                    ? "primary"
                                    : "default"
                                }
                                onClick={() =>
                                  handleSetRolloutPercentage(flag, percentage)
                                }
                              >
                                {percentage}%
                              </Button>
                            ))}
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </div>
                <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                  {telemetryCounters.map(({ eventType, count }) => (
                    <div
                      key={eventType}
                      data-testid={`workspace-telemetry-counter-${eventType}`}
                      className="flex items-center justify-between rounded border border-border px-3 py-2 text-xs"
                    >
                      <span className="text-text-muted">
                        {formatTelemetryEventLabel(eventType)}
                      </span>
                      <span className="font-semibold text-text">{count}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <h3 className="mb-2 text-sm font-semibold text-text">
                  {t(
                    "playground:workspace.telemetrySummaryRecent",
                    "Recent events"
                  )}
                </h3>
                <div className="max-h-52 space-y-2 overflow-y-auto rounded border border-border bg-surface2 p-2">
                  {recentTelemetryEvents.length > 0 ? (
                    recentTelemetryEvents.map((event, index) => (
                      <div
                        key={`${event.type}-${event.at}-${index}`}
                        className="rounded border border-border bg-surface px-2 py-1.5"
                      >
                        <div className="flex items-center justify-between gap-3 text-xs">
                          <span className="font-medium text-text">
                            {formatTelemetryEventLabel(event.type)}
                          </span>
                          <span className="text-text-muted">
                            {formatTelemetryTimestamp(event.at)}
                          </span>
                        </div>
                        {Object.keys(event.details).length > 0 && (
                          <pre className="mt-1 overflow-x-auto whitespace-pre-wrap text-[11px] text-text-muted">
                            {JSON.stringify(event.details)}
                          </pre>
                        )}
                      </div>
                    ))
                  ) : (
                    <div className="px-2 py-4 text-center text-xs text-text-muted">
                      {t(
                        "playground:workspace.telemetrySummaryNoEvents",
                        "No events yet"
                      )}
                    </div>
                  )}
                </div>
              </div>
            </>
          )}
        </div>
      </Modal>

      {/* Share Dialog */}
      {workspaceId && (
        <ShareDialog
          workspaceId={workspaceId}
          open={shareDialogOpen}
          onClose={() => setShareDialogOpen(false)}
        />
      )}

      {/* Delete workspace confirmation modal */}
      <Modal
        open={deleteConfirmOpen}
        title={t("playground:workspace.deleteTitle", "Delete workspace?")}
        okText={t("common:delete", "Delete")}
        okButtonProps={{
          danger: true,
          disabled: deleteTargetWorkspace
            ? deleteConfirmInput !== deleteTargetWorkspace.name
            : true
        }}
        cancelText={t("common:cancel", "Cancel")}
        onOk={executeDeleteWorkspace}
        onCancel={() => {
          setDeleteConfirmOpen(false)
          setDeleteTargetWorkspace(null)
          setDeleteConfirmInput("")
        }}
        centered
        maskClosable={false}
        destroyOnClose
      >
        {deleteTargetWorkspace && (
          <div className="space-y-3">
            <p className="text-sm text-text">
              {t(
                "playground:workspace.deleteConfirmMessage",
                "This will permanently remove workspace {{name}} and all its data.",
                { name: deleteTargetWorkspace.name }
              )}
            </p>
            <p className="text-sm text-text-muted">
              {t(
                "playground:workspace.deleteConfirmStats",
                "This workspace has {{sourceCount}} source(s). All associated chat sessions and notes will also be removed.",
                { sourceCount: deleteTargetWorkspace.sourceCount }
              )}
            </p>
            <div>
              <p className="mb-1.5 text-sm text-text-muted">
                {t(
                  "playground:workspace.deleteConfirmTypeName",
                  "Type {{name}} to confirm:",
                  { name: deleteTargetWorkspace.name }
                )}
              </p>
              <Input
                value={deleteConfirmInput}
                onChange={(e) => setDeleteConfirmInput(e.target.value)}
                placeholder={deleteTargetWorkspace.name}
                autoFocus
                onPressEnter={() => {
                  if (deleteConfirmInput === deleteTargetWorkspace.name) {
                    executeDeleteWorkspace()
                  }
                }}
              />
            </div>
          </div>
        )}
      </Modal>
    </header>
  )
}

export default WorkspaceHeader
