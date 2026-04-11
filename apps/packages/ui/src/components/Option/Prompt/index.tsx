import { useQuery, useQueryClient } from "@tanstack/react-query"
import {
  Skeleton,
  Table,
  Tooltip,
  notification,
  Modal,
  Input,
  Form,
  Segmented,
  Tag,
  Select,
  Alert,
  Pagination,
  type InputRef
} from "antd"
import { Computer, Zap, Star, StarOff, UploadCloud, Download, Trash2, Pen, Undo2, AlertTriangle, Layers, Cloud, Clipboard, Copy, Keyboard, FolderPlus, Play, LayoutGrid, List } from "lucide-react"
import { PromptActionsMenu } from "./PromptActionsMenu"
import { SyncStatusBadge } from "./SyncStatusBadge"
import { PromptBulkActionBar } from "./PromptBulkActionBar"
import type { PromptGalleryDensity } from "./PromptGalleryCard"
import {
  PromptListTable,
  type PromptTableDensity
} from "./PromptListTable"
import { PromptListToolbar } from "./PromptListToolbar"
import { PromptSidebar } from "./PromptSidebar"
import { useContextualHints } from "./useContextualHints"
import { useFilterPresets, type FilterPreset } from "./useFilterPresets"
import type { PromptListQueryState, PromptRowVM, PromptSavedView } from "./prompt-workspace-types"
// buildSyncBatchPlan moved to usePromptFilteredData hook
import React, { Suspense, useMemo, useRef, useState, useEffect } from "react"
import { useTranslation } from "react-i18next"
import { useNavigate, useSearchParams } from "react-router-dom"
import {
  getAllPrompts,
  getDeletedPrompts
} from "@/db/dexie/helpers"
// getAllCopilotPrompts, upsertCopilotPrompts moved to usePromptInteractions hook
import { tagColors } from "@/utils/color"
// isFireFoxPrivateMode moved to usePromptUtilities hook
// useConfirmDanger moved to usePromptUtilities hook
import { useServerOnline } from "@/hooks/useServerOnline"
import FeatureEmptyState from "@/components/Common/FeatureEmptyState"
import ConnectFeatureBanner from "@/components/Common/ConnectFeatureBanner"
// useMessageOption moved to usePromptInteractions hook
import { useDebounce } from "@/hooks/useDebounce"
import {
  pullFromStudio
} from "@/services/prompt-sync"
import {
  type PromptCollection
} from "@/services/prompts-api"
// prompt-studio imports moved to usePromptInteractions hook
// execute-playground-provider-utils, tldwClient, usePromptStudioStore moved to usePromptInteractions hook
import {
  type TagMatchMode
} from "./custom-prompts-utils"
// filterCopilotPrompts moved to usePromptInteractions hook
import {
  filterTrashPromptsByName,
  getTrashDaysRemaining,
  getTrashRemainingSeverity
} from "./trash-prompts-utils"
// isPromptInCollection moved to usePromptFilteredData hook
import { usePromptSync } from "./hooks/usePromptSync"
import { usePromptEditor } from "./hooks/usePromptEditor"
import { usePromptBulkActions } from "./hooks/usePromptBulkActions"
import { usePromptImportExport } from "./hooks/usePromptImportExport"
import { usePromptCollections } from "./hooks/usePromptCollections"
import { usePromptUtilities } from "./hooks/usePromptUtilities"
import { usePromptFilteredData } from "./hooks/usePromptFilteredData"
import { usePromptInteractions } from "./hooks/usePromptInteractions"

const PromptDrawer = React.lazy(() =>
  import("./PromptDrawer").then((module) => ({ default: module.PromptDrawer }))
)
const ConflictResolutionModal = React.lazy(() =>
  import("./ConflictResolutionModal").then((module) => ({
    default: module.ConflictResolutionModal
  }))
)
const PromptInspectorPanel = React.lazy(() =>
  import("./PromptInspectorPanel").then((module) => ({
    default: module.PromptInspectorPanel
  }))
)
const PromptFullPageEditor = React.lazy(() =>
  import("./PromptFullPageEditor").then((module) => ({
    default: module.PromptFullPageEditor
  }))
)
const ProjectSelector = React.lazy(() =>
  import("./ProjectSelector").then((module) => ({ default: module.ProjectSelector }))
)
const PromptGalleryCard = React.lazy(() =>
  import("./PromptGalleryCard").then((module) => ({
    default: module.PromptGalleryCard
  }))
)
const PromptStarterCards = React.lazy(() =>
  import("./PromptStarterCards").then((module) => ({
    default: module.PromptStarterCards
  }))
)
const ContextualHint = React.lazy(() =>
  import("./ContextualHint").then((module) => ({ default: module.ContextualHint }))
)
const StudioTabContainer = React.lazy(() =>
  import("./Studio/StudioTabContainer").then((module) => ({
    default: module.StudioTabContainer
  }))
)

type SegmentType = "custom" | "copilot" | "studio" | "trash"

const VALID_SEGMENTS: SegmentType[] = ["custom", "copilot", "studio", "trash"]

const getSegmentFromParam = (param: string | null): SegmentType => {
  if (param && VALID_SEGMENTS.includes(param as SegmentType)) {
    return param as SegmentType
  }
  return "custom"
}

type PromptSortKey = "title" | "modifiedAt" | null
type PromptSortOrder = "ascend" | "descend" | null
type PromptSortState = {
  key: PromptSortKey
  order: PromptSortOrder
}

type LocalQuickTestPrompt = {
  id: string
  name: string
  systemText?: string
  userText?: string
}

const PROMPTS_CUSTOM_SORT_STORAGE_KEY = "tldw-prompts-custom-sort-v1"
const PROMPTS_TABLE_DENSITY_STORAGE_KEY = "tldw-prompts-table-density-v1"
const PROMPTS_VIEW_MODE_STORAGE_KEY = "tldw-prompts-view-mode-v1"
const PROMPTS_GALLERY_DENSITY_STORAGE_KEY = "tldw-prompts-gallery-density-v1"
const PROMPTS_MOBILE_BREAKPOINT_PX = 768

const readPromptSortState = (): PromptSortState => {
  if (typeof window === "undefined") {
    return { key: null, order: null }
  }
  try {
    const raw = window.sessionStorage.getItem(PROMPTS_CUSTOM_SORT_STORAGE_KEY)
    if (!raw) {
      return { key: null, order: null }
    }
    const parsed = JSON.parse(raw) as PromptSortState
    const allowedKeys: PromptSortKey[] = ["title", "modifiedAt", null]
    const allowedOrders: PromptSortOrder[] = ["ascend", "descend", null]
    if (!allowedKeys.includes(parsed?.key) || !allowedOrders.includes(parsed?.order)) {
      return { key: null, order: null }
    }
    return parsed
  } catch {
    return { key: null, order: null }
  }
}

const readPromptTableDensity = (): PromptTableDensity => {
  if (typeof window === "undefined") {
    return "comfortable"
  }
  try {
    const raw = window.localStorage.getItem(PROMPTS_TABLE_DENSITY_STORAGE_KEY)
    if (raw === "compact" || raw === "dense" || raw === "comfortable") {
      return raw
    }
    return "comfortable"
  } catch {
    return "comfortable"
  }
}

type PromptViewMode = "table" | "gallery"

const readPromptViewMode = (): PromptViewMode => {
  if (typeof window === "undefined") return "table"
  try {
    const raw = window.localStorage.getItem(PROMPTS_VIEW_MODE_STORAGE_KEY)
    if (raw === "table" || raw === "gallery") return raw
    return "table"
  } catch {
    return "table"
  }
}

const readPromptGalleryDensity = (): PromptGalleryDensity => {
  if (typeof window === "undefined") return "rich"
  try {
    const raw = window.localStorage.getItem(PROMPTS_GALLERY_DENSITY_STORAGE_KEY)
    if (raw === "rich" || raw === "compact") return raw
    return "rich"
  } catch {
    return "rich"
  }
}

export const PromptBody = () => {
  const queryClient = useQueryClient()
  const [searchParams, setSearchParams] = useSearchParams()
  const { t } = useTranslation(["settings", "common", "option"])
  const navigate = useNavigate()
  const isOnline = useServerOnline()
  // Get initial segment from URL param
  const initialSegment = getSegmentFromParam(searchParams.get("tab"))

  // Track if we've processed the initial prompt deep-link
  const deepLinkProcessedRef = useRef(false)

  // Handle ?project= filter for showing prompts from a specific project
  const projectFilter = searchParams.get("project")

  const [searchText, setSearchText] = useState("")
  const [typeFilter, setTypeFilter] = useState<"all" | "system" | "quick">(
    "all"
  )
  const [usageFilter, setUsageFilter] = useState<"all" | "used" | "unused">(
    "all"
  )
  const [tagFilter, setTagFilter] = useState<string[]>([])
  const [tagMatchMode, setTagMatchMode] = useState<TagMatchMode>("any")
  const [syncFilter, setSyncFilter] = useState<string>("all")
  const [currentPage, setCurrentPage] = useState(1)
  const [resultsPerPage, setResultsPerPage] = useState(20)
  const [tableDensity, setTableDensity] = useState<PromptTableDensity>(() =>
    readPromptTableDensity()
  )
  const [viewMode, setViewMode] = useState<PromptViewMode>(() =>
    readPromptViewMode()
  )
  const [galleryDensity, setGalleryDensity] = useState<PromptGalleryDensity>(() =>
    readPromptGalleryDensity()
  )
  const [promptSort, setPromptSort] = useState<PromptSortState>(() =>
    readPromptSortState()
  )
  const searchInputRef = useRef<InputRef | null>(null)
  const [isCompactViewport, setIsCompactViewport] = useState(() =>
    typeof window !== "undefined"
      ? window.innerWidth < PROMPTS_MOBILE_BREAKPOINT_PX
      : false
  )
  const [trashSearchText, setTrashSearchText] = useState("")
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [savedView, setSavedView] = useState<PromptSavedView>("all")
  const { presets: filterPresets, savePreset: saveFilterPreset, deletePreset: deleteFilterPreset } = useFilterPresets()
  const { shouldShow: shouldShowHint, dismiss: dismissHint, markShown: markHintShown } = useContextualHints()

  const debouncedSearchText = useDebounce(searchText, 300)
  const normalizedSearchText = debouncedSearchText.trim()
  const shouldUseServerSearch = isOnline && normalizedSearchText.length > 0

  const { data, status } = useQuery({
    queryKey: ["fetchAllPrompts"],
    queryFn: getAllPrompts
  })

  const { data: trashData, status: trashStatus } = useQuery({
    queryKey: ["fetchDeletedPrompts"],
    queryFn: getDeletedPrompts
  })

  // --- Utility Hooks ---

  const utils = usePromptUtilities({ t, data })
  const {
    confirmDanger,
    guardPrivateMode,
    getPromptKeywords,
    getPromptTexts,
    getPromptType,
    getPromptModifiedAt,
    getPromptUsageCount,
    getPromptLastUsedAt,
    formatRelativePromptTime,
    getPromptRecordById,
    isFireFoxPrivateMode
  } = utils

  // --- Hooks ---

  const sync = usePromptSync({
    queryClient,
    isOnline,
    t
  })

  const editor = usePromptEditor({
    queryClient,
    isOnline,
    t,
    guardPrivateMode,
    getPromptTexts,
    getPromptKeywords,
    getPromptRecordById,
    confirmDanger,
    syncPromptAfterLocalSave: sync.syncPromptAfterLocalSave,
    onEmptyTrashSuccess: () => {
      bulk.setTrashSelectedRowKeys([])
    }
  })

  const bulk = usePromptBulkActions({
    queryClient,
    data,
    isOnline,
    isFireFoxPrivateMode,
    t,
    guardPrivateMode,
    getPromptKeywords,
    buildPromptUpdatePayload: editor.buildPromptUpdatePayload,
    confirmDanger
  })

  const importExport = usePromptImportExport({
    queryClient,
    data,
    isOnline,
    t,
    guardPrivateMode,
    confirmDanger
  })

  const collections = usePromptCollections({
    queryClient,
    isOnline,
    t,
    setSelectedRowKeys: bulk.setSelectedRowKeys
  })

  const interactions = usePromptInteractions({
    queryClient,
    isOnline,
    initialSegment,
    t,
    getPromptTexts,
    getPromptKeywords,
    getPromptRecordById,
    getPromptModifiedAt,
    getPromptUsageCount,
    getPromptLastUsedAt,
    editorMarkPromptAsUsed: editor.markPromptAsUsed
  })
  const {
    openCopilotEdit, setOpenCopilotEdit,
    editCopilotId, setEditCopilotId,
    editCopilotForm,
    copilotSearchText, setCopilotSearchText,
    copilotKeyFilter, setCopilotKeyFilter,
    copilotData, copilotStatus,
    copilotEditPromptValue, copilotPromptIncludesTextPlaceholder,
    copilotPromptKeyOptions, filteredCopilotData,
    updateCopilotPrompt, isUpdatingCopilotPrompt,
    copyCopilotPromptToClipboard, copyPromptShareLink,
    insertPrompt, setInsertPrompt,
    handleInsertChoice, handleUsePromptInChat,
    localQuickTestPrompt, localQuickTestInput, setLocalQuickTestInput,
    localQuickTestOutput, isRunningLocalQuickTest, localQuickTestRunInfo,
    closeLocalQuickTestModal, handleQuickTest, runLocalQuickTest,
    inspectorOpen, inspectorPrompt,
    closeInspector, openPromptInspector,
    shortcutsHelpOpen, setShortcutsHelpOpen,
    selectedSegment, setSelectedSegment,
    hasStudio
  } = interactions

  // --- Effects ---

  // Sync URL params with selected segment after interactions initialize the tab state.
  useEffect(() => {
    const currentTab = searchParams.get("tab")
    const expectedTab = selectedSegment === "custom" ? null : selectedSegment

    if (currentTab !== expectedTab) {
      if (expectedTab) {
        setSearchParams({ tab: expectedTab }, { replace: true })
      } else {
        const newParams = new URLSearchParams(searchParams)
        newParams.delete("tab")
        setSearchParams(newParams, { replace: true })
      }
    }
  }, [searchParams, selectedSegment, setSearchParams])

  useEffect(() => {
    setCurrentPage(1)
  }, [normalizedSearchText, projectFilter, typeFilter, collections.collectionFilter, tagFilter, tagMatchMode])

  useEffect(() => {
    if (typeof window === "undefined") return
    try {
      window.sessionStorage.setItem(
        PROMPTS_CUSTOM_SORT_STORAGE_KEY,
        JSON.stringify(promptSort)
      )
    } catch {
      // Ignore session storage failures in restricted browser modes.
    }
  }, [promptSort])

  useEffect(() => {
    if (typeof window === "undefined") return
    try {
      window.localStorage.setItem(
        PROMPTS_TABLE_DENSITY_STORAGE_KEY,
        tableDensity
      )
    } catch {
      // Ignore storage failures in restricted browser modes.
    }
  }, [tableDensity])

  useEffect(() => {
    if (typeof window === "undefined") return
    try {
      window.localStorage.setItem(PROMPTS_VIEW_MODE_STORAGE_KEY, viewMode)
    } catch {
      // Ignore storage failures in restricted browser modes.
    }
  }, [viewMode])

  useEffect(() => {
    if (typeof window === "undefined") return
    try {
      window.localStorage.setItem(
        PROMPTS_GALLERY_DENSITY_STORAGE_KEY,
        galleryDensity
      )
    } catch {
      // Ignore storage failures in restricted browser modes.
    }
  }, [galleryDensity])

  // Force table view for trash/copilot/studio segments
  useEffect(() => {
    if (selectedSegment === "trash" || selectedSegment === "copilot" || selectedSegment === "studio") {
      setViewMode("table")
    }
  }, [selectedSegment])

  useEffect(() => {
    if (typeof window === "undefined") return
    const handleResize = () => {
      setIsCompactViewport(window.innerWidth < PROMPTS_MOBILE_BREAKPOINT_PX)
    }
    window.addEventListener("resize", handleResize)
    return () => {
      window.removeEventListener("resize", handleResize)
    }
  }, [])

  // Handle ?prompt= deep-link for opening a specific prompt
  useEffect(() => {
    const promptId = searchParams.get("prompt")
    if (!promptId || deepLinkProcessedRef.current) return
    if (status !== "success" || !Array.isArray(data)) return

    const clearPromptParam = () => {
      const newParams = new URLSearchParams(searchParams)
      newParams.delete("prompt")
      newParams.delete("source")
      setSearchParams(newParams, { replace: true })
    }

    const openPromptDrawer = (promptRecord: any) => {
      clearPromptParam()
      editor.setEditId(promptRecord.id)
      editor.setDrawerOpen(true)
      editor.setDrawerInitialValues({
        id: promptRecord?.id,
        name: promptRecord?.name || promptRecord?.title,
        author: promptRecord?.author,
        details: promptRecord?.details,
        system_prompt:
          promptRecord?.system_prompt ||
          (promptRecord?.is_system ? promptRecord?.content : undefined),
        user_prompt:
          promptRecord?.user_prompt ||
          (!promptRecord?.is_system ? promptRecord?.content : undefined),
        keywords: promptRecord?.keywords ?? promptRecord?.tags ?? [],
        serverId: promptRecord?.serverId,
        syncStatus: promptRecord?.syncStatus,
        sourceSystem: promptRecord?.sourceSystem,
        studioProjectId: promptRecord?.studioProjectId,
        lastSyncedAt: promptRecord?.lastSyncedAt,
        fewShotExamples: promptRecord?.fewShotExamples,
        modulesConfig: promptRecord?.modulesConfig,
        promptFormat: promptRecord?.promptFormat ?? "legacy",
        promptSchemaVersion: promptRecord?.promptSchemaVersion ?? null,
        structuredPromptDefinition:
          promptRecord?.structuredPromptDefinition ?? null,
        changeDescription: promptRecord?.changeDescription,
        versionNumber: promptRecord?.versionNumber
      })
    }

    deepLinkProcessedRef.current = true

    // First try local prompt IDs.
    const localPromptRecord = data.find((p: any) => p.id === promptId)
    if (localPromptRecord) {
      openPromptDrawer(localPromptRecord)
      return
    }

    const source = searchParams.get("source")
    const parsedServerPromptId = Number(promptId)
    const isServerPromptLink =
      Number.isInteger(parsedServerPromptId) &&
      parsedServerPromptId > 0 &&
      (source === "studio" || source === null)

    const getSharedPromptFailureDescription = (errorMessage?: string) => {
      const normalized = String(errorMessage || "").toLowerCase()
      const isAccessDenied =
        normalized.includes("401") ||
        normalized.includes("403") ||
        normalized.includes("forbidden") ||
        normalized.includes("unauthor") ||
        normalized.includes("not authenticated") ||
        normalized.includes("api key")
      if (isAccessDenied) {
        return t("managePrompts.notification.sharedPromptAccessDeniedDesc", {
          defaultValue:
            "You don't have permission to open this shared prompt. Check your server login and project access."
        })
      }
      return t("managePrompts.notification.sharedPromptNotFoundDesc", {
        defaultValue:
          "The shared prompt could not be pulled from the server. It may not exist or you may not have access."
      })
    }

    if (isOnline && isServerPromptLink) {
      clearPromptParam()
      void (async () => {
        const syncResult = await pullFromStudio(parsedServerPromptId)
        if (!syncResult.success) {
          notification.warning({
            message: t("managePrompts.notification.promptNotFound", {
              defaultValue: "Prompt not found"
            }),
            description: getSharedPromptFailureDescription(syncResult.error)
          })
          return
        }
        try {
          const refreshedPrompts = await queryClient.fetchQuery({
            queryKey: ["fetchAllPrompts"],
            queryFn: getAllPrompts
          })
          const importedPrompt = (Array.isArray(refreshedPrompts)
            ? refreshedPrompts
            : []
          ).find(
            (item: any) =>
              item?.id === syncResult.localId ||
              item?.serverId === parsedServerPromptId
          )
          if (!importedPrompt) {
            notification.warning({
              message: t("managePrompts.notification.promptNotFound", {
                defaultValue: "Prompt not found"
              }),
              description: t("managePrompts.notification.sharedPromptImportMissing", {
                defaultValue:
                  "The shared prompt was fetched, but could not be loaded locally."
              })
            })
            return
          }
          notification.success({
            message: t("managePrompts.notification.sharedPromptImported", {
              defaultValue: "Shared prompt imported"
            }),
            description: t("managePrompts.notification.sharedPromptImportedDesc", {
              defaultValue:
                "The prompt was pulled from the server and opened in your workspace."
            })
          })
          openPromptDrawer(importedPrompt)
        } catch {
          notification.warning({
            message: t("managePrompts.notification.promptNotFound", {
              defaultValue: "Prompt not found"
            }),
            description: t("managePrompts.notification.sharedPromptImportMissing", {
              defaultValue:
                "The shared prompt was fetched, but could not be loaded locally."
            })
          })
        }
      })()
      return
    }

    clearPromptParam()
    if (!isOnline && isServerPromptLink) {
      deepLinkProcessedRef.current = true
      notification.warning({
        message: t("managePrompts.notification.promptNotFound", {
          defaultValue: "Prompt not found"
        }),
        description: t("managePrompts.notification.sharedPromptOfflineDesc", {
          defaultValue:
            "This shared prompt link requires an online server connection."
        })
      })
      return
    }

    notification.warning({
      message: t("managePrompts.notification.promptNotFound", {
        defaultValue: "Prompt not found"
      }),
      description: t("managePrompts.notification.promptNotFoundDesc", {
        defaultValue: "The requested prompt could not be found. It may have been deleted."
      })
    })
  }, [searchParams, data, status, setSearchParams, isOnline, queryClient, t, editor])

  const promptLoadFailed = status === "error"
  const copilotLoadFailed = isOnline && copilotStatus === "error"
  const loadErrorDescription = [
    promptLoadFailed
      ? t(
          "managePrompts.loadErrorDetail",
          "Custom prompts couldn't be retrieved from local storage."
        )
      : null,
    copilotLoadFailed
      ? t(
          "managePrompts.copilotLoadErrorDetail",
          "Copilot prompts couldn't be retrieved."
        )
      : null
  ]
    .filter(Boolean)
    .join(" ")

  React.useEffect(() => {
    // Only redirect from copilot/studio tab when offline (trash is local-only so always available)
    if (!isOnline && (selectedSegment === "copilot" || selectedSegment === "studio")) {
      setSelectedSegment("custom")
    }
  }, [isOnline, selectedSegment])

  // Handle ?edit=<id> and ?new=1 URL params for full editor
  useEffect(() => {
    if (status !== "success" || !Array.isArray(data)) return
    const editIdParam = searchParams.get("edit")
    const isNew = searchParams.get("new")
    if (editIdParam && !editor.fullEditorOpen) {
      const prompt = data.find((p: any) => String(p.id) === editIdParam)
      if (prompt) {
        editor.openFullEditor(prompt)
      }
    } else if (isNew === "1" && !editor.fullEditorOpen) {
      editor.openFullEditor()
    }
  }, [status, data, searchParams, editor.fullEditorOpen, editor.openFullEditor])

  // --- Computed ---

  const filteredDataHook = usePromptFilteredData({
    data,
    isOnline,
    normalizedSearchText,
    shouldUseServerSearch,
    projectFilter,
    typeFilter,
    syncFilter,
    usageFilter,
    tagFilter,
    tagMatchMode,
    savedView,
    selectedCollection: collections.selectedCollection,
    currentPage,
    resultsPerPage,
    promptSort,
    getPromptKeywords,
    getPromptTexts,
    getPromptType,
    getPromptModifiedAt,
    getPromptUsageCount,
    getPromptLastUsedAt,
    t
  })
  const {
    serverSearchStatus,
    allTags,
    pendingSyncCount,
    localSyncBatchPlan,
    sidebarCounts,
    sortedFilteredData,
    customPromptRows,
    tableTotal,
    hiddenServerResultsOnPage,
    useServerSearchResults
  } = filteredDataHook

  useEffect(() => {
    if (!shouldUseServerSearch || serverSearchStatus !== "error") return
    notification.warning({
      message: t("managePrompts.searchServerFallback", {
        defaultValue: "Server search unavailable"
      }),
      description: t("managePrompts.searchServerFallbackDesc", {
        defaultValue:
          "Falling back to local search results for this query."
      })
    })
  }, [serverSearchStatus, shouldUseServerSearch, t])

  const filteredTrashData = useMemo(() => {
    if (!Array.isArray(trashData)) return []
    return filterTrashPromptsByName(trashData, trashSearchText)
  }, [trashData, trashSearchText])

  const customPromptTableQuery: PromptListQueryState = {
    searchText,
    typeFilter: typeFilter,
    syncFilter: syncFilter as PromptListQueryState["syncFilter"],
    usageFilter,
    tagFilter,
    tagMatchMode,
    sort: {
      key: promptSort.key,
      order: promptSort.order
    },
    page: currentPage,
    pageSize: resultsPerPage,
    savedView
  }

  const customPromptsLoading =
    status === "pending" ||
    (shouldUseServerSearch && serverSearchStatus === "pending")

  React.useEffect(() => {
    // Only clear selection for items that are no longer visible
    const visibleIds = new Set(sortedFilteredData.map((p: any) => p.id))
    bulk.setSelectedRowKeys((prev) => {
      const stillVisible = prev.filter((key) => visibleIds.has(key as string))
      if (stillVisible.length !== prev.length) {
        if (stillVisible.length < prev.length && prev.length > 0) {
          notification.info({
            message: t("managePrompts.selectionFiltered", {
              defaultValue: "Some selected items were filtered out"
            }),
            duration: 2
          })
        }
        return stillVisible
      }
      return prev
    })
  }, [sortedFilteredData, t, bulk])

  // --- Callbacks ---

  const handleCopyCopilotToCustom = React.useCallback(
    (record: { key?: string; prompt?: string }) => {
      interactions.copyCopilotToCustom(record, editor.openCreateDrawer)
    },
    [interactions.copyCopilotToCustom, editor.openCreateDrawer]
  )

  // Keyboard shortcuts: N = new prompt, / = focus search, Esc = close drawer, ? = open shortcut help
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement
      const isInput = target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.tagName === "SELECT" || target.isContentEditable
      if (e.key === "Escape") {
        if (shortcutsHelpOpen) {
          setShortcutsHelpOpen(false)
          return
        }
        if (editor.drawerOpen) {
          editor.setDrawerOpen(false)
        }
        return
      }
      if (isInput) return
      if (
        (e.key === "?" || (e.key === "/" && e.shiftKey)) &&
        !e.metaKey &&
        !e.ctrlKey &&
        !e.altKey
      ) {
        e.preventDefault()
        setShortcutsHelpOpen(true)
        return
      }
      if (e.key === "n" && !e.metaKey && !e.ctrlKey && !e.altKey) {
        e.preventDefault()
        editor.openFullEditor()
        return
      }
      if (e.key === "/" && !e.metaKey && !e.ctrlKey && !e.altKey) {
        e.preventDefault()
        searchInputRef.current?.focus()
      }
    }
    document.addEventListener("keydown", handler)
    return () => document.removeEventListener("keydown", handler)
  }, [editor.drawerOpen, shortcutsHelpOpen, editor.openFullEditor])

  // Clear project filter
  const clearProjectFilter = () => {
    const newParams = new URLSearchParams(searchParams)
    newParams.delete("project")
    setSearchParams(newParams, { replace: true })
  }

  const handleLoadFilterPreset = React.useCallback((preset: FilterPreset) => {
    setTypeFilter(preset.typeFilter as any)
    setSyncFilter(preset.syncFilter as any)
    setTagFilter(preset.tagFilter)
    setTagMatchMode(preset.tagMatchMode)
    setSavedView(preset.savedView)
  }, [])

  const handleSaveFilterPreset = React.useCallback(
    (name: string) => {
      saveFilterPreset(name, {
        typeFilter,
        syncFilter,
        tagFilter,
        tagMatchMode,
        savedView,
      })
    },
    [typeFilter, syncFilter, tagFilter, tagMatchMode, savedView, saveFilterPreset]
  )

  const handleCustomPromptTableQueryChange = React.useCallback(
    (patch: Partial<PromptListQueryState>) => {
      const nextPage =
        typeof patch.page === "number" ? patch.page : currentPage
      const nextPageSize =
        typeof patch.pageSize === "number" ? patch.pageSize : resultsPerPage

      if (nextPageSize !== resultsPerPage) {
        setResultsPerPage(nextPageSize)
        setCurrentPage(1)
      } else if (nextPage !== currentPage) {
        setCurrentPage(nextPage)
      }

      if (patch.sort) {
        const rawNextKey = patch.sort.key
        const nextKey: PromptSortKey =
          rawNextKey === "title" || rawNextKey === "modifiedAt"
            ? rawNextKey
            : null
        const nextOrder = patch.sort.order || null
        setPromptSort({
          key: nextOrder ? nextKey : null,
          order: nextOrder
        })
      }
    },
    [currentPage, resultsPerPage]
  )

  const renderCustomPromptTitleMeta = React.useCallback(
    (row: PromptRowVM) => {
      if (!isCompactViewport) return null
      return (
        <div className="mt-1 flex flex-wrap items-center gap-2">
          <span className="text-[11px] text-text-muted">
            {formatRelativePromptTime(row.updatedAt)}
          </span>
          <Tooltip
            title={
              !isOnline
                ? t("managePrompts.sync.offlineTooltip", {
                    defaultValue:
                      "Sync unavailable (offline). Showing last known status."
                  })
                : undefined
            }
          >
            <span className={!isOnline ? "opacity-60" : undefined}>
              <SyncStatusBadge
                syncStatus={row.syncStatus}
                sourceSystem={row.sourceSystem}
                serverId={row.serverId}
                compact
                onClick={
                  isOnline && row.syncStatus === "conflict"
                    ? () => sync.openConflictResolution(row.id)
                    : undefined
                }
              />
            </span>
          </Tooltip>
        </div>
      )
    },
    [formatRelativePromptTime, isCompactViewport, isOnline, sync.openConflictResolution, t]
  )

  const renderCustomPromptActions = React.useCallback(
    (row: PromptRowVM) => {
      const promptRecord = getPromptRecordById(row.id)
      const actionDisabled = isFireFoxPrivateMode || !promptRecord
      return (
        <PromptActionsMenu
          promptId={row.id}
          disabled={actionDisabled}
          syncStatus={row.syncStatus}
          serverId={row.serverId}
          inlineUseInChat={false}
          onEdit={() => {
            if (!promptRecord) return
            editor.openFullEditor(promptRecord)
          }}
          onDuplicate={() => {
            if (!promptRecord) return
            editor.handleDuplicatePrompt(promptRecord)
          }}
          onUseInChat={() => {
            if (!promptRecord) return
            void handleUsePromptInChat(promptRecord)
          }}
          onQuickTest={() => {
            if (!promptRecord) return
            void handleQuickTest(promptRecord)
          }}
          onDelete={() => {
            if (!promptRecord) return
            void editor.handleDeletePrompt(promptRecord)
          }}
          onShareLink={
            row.serverId && promptRecord
              ? () => {
                  void copyPromptShareLink(promptRecord)
                }
              : undefined
          }
          onPushToServer={
            isOnline && promptRecord
              ? () => {
                  sync.setPromptToSync(promptRecord.id)
                  sync.setProjectSelectorOpen(true)
                }
              : undefined
          }
          onPullFromServer={
            isOnline && row.serverId && promptRecord
              ? () => {
                  sync.pullFromStudioMutation({
                    serverId: row.serverId as number,
                    localId: promptRecord.id
                  })
                }
              : undefined
          }
          onUnlink={
            isOnline && row.serverId && promptRecord
              ? () => {
                  sync.unlinkPromptMutation(promptRecord.id)
                }
              : undefined
          }
          onResolveConflict={
            isOnline && row.syncStatus === "conflict"
              ? () => {
                  sync.openConflictResolution(row.id)
                }
              : undefined
          }
        />
      )
    },
    [
      copyPromptShareLink,
      getPromptRecordById,
      editor,
      handleQuickTest,
      handleUsePromptInChat,
      isFireFoxPrivateMode,
      isOnline,
      sync
    ]
  )

  function customPrompts() {
    const bulkActionTouchClass = isCompactViewport
      ? "min-h-[44px] px-3 py-2"
      : "px-2 py-1"

    return (
      <div data-testid="prompts-custom">
        {/* Project filter banner - shown when filtering by project */}
        {projectFilter && (
          <Alert
            type="info"
            showIcon
            className="mb-4"
            title={t("managePrompts.projectFilter.active", {
              defaultValue: "Filtering by project"
            })}
            description={t("managePrompts.projectFilter.description", {
              defaultValue: "Showing prompts linked to Project #{{projectId}}. Clear the filter to see all prompts.",
              projectId: projectFilter
            })}
            action={
              <button
                onClick={clearProjectFilter}
                className="text-sm text-primary hover:underline"
                data-testid="prompts-clear-project-filter"
              >
                {t("managePrompts.projectFilter.clear", { defaultValue: "Show all prompts" })}
              </button>
            }
          />
        )}
        <div className="mb-6 space-y-3">
          {/* Bulk action bar - shown when rows are selected (table view only) */}
          {viewMode === "table" && bulk.selectedRowKeys.length > 0 && (
            <PromptBulkActionBar mode="legacy">
              <span className="text-sm text-primary">
                {t("managePrompts.bulk.selected", {
                  defaultValue: "{{count}} selected",
                  count: bulk.selectedRowKeys.length
                })}
              </span>
              <button
                onClick={() => bulk.triggerBulkExport()}
                data-testid="prompts-bulk-export"
                className={`inline-flex items-center gap-1 rounded border border-primary/30 text-sm text-primary hover:bg-primary/10 ${bulkActionTouchClass}`}>
                <Download className="size-3" /> {t("managePrompts.bulk.export", { defaultValue: "Export selected" })}
              </button>
              <button
                onClick={() => bulk.setBulkKeywordModalOpen(true)}
                disabled={bulk.isBulkAddingKeyword}
                data-testid="prompts-bulk-add-keyword"
                className={`inline-flex items-center gap-1 rounded border border-primary/30 text-sm text-primary hover:bg-primary/10 disabled:opacity-50 ${bulkActionTouchClass}`}>
                {t("managePrompts.bulk.addKeyword", { defaultValue: "Add keyword" })}
              </button>
              <button
                onClick={() =>
                  bulk.bulkToggleFavorite({
                    ids: bulk.selectedRowKeys.map((key) => String(key)),
                    favorite: !bulk.allSelectedAreFavorite
                  })
                }
                disabled={bulk.isBulkFavoriting}
                data-testid="prompts-bulk-toggle-favorite"
                className={`inline-flex items-center gap-1 rounded border border-primary/30 text-sm text-primary hover:bg-primary/10 disabled:opacity-50 ${bulkActionTouchClass}`}>
                {bulk.allSelectedAreFavorite ? (
                  <StarOff className="size-3" />
                ) : (
                  <Star className="size-3" />
                )}
                {bulk.allSelectedAreFavorite
                  ? t("managePrompts.bulk.unfavorite", {
                      defaultValue: "Unfavorite selected"
                    })
                  : t("managePrompts.bulk.favorite", {
                      defaultValue: "Favorite selected"
                    })}
              </button>
              {isOnline && (
                <button
                  onClick={() =>
                    bulk.bulkPushToServer(bulk.selectedRowKeys.map((key) => String(key)))
                  }
                  disabled={bulk.isBulkPushing}
                  data-testid="prompts-bulk-push-server"
                  className={`inline-flex items-center gap-1 rounded border border-primary/30 text-sm text-primary hover:bg-primary/10 disabled:opacity-50 ${bulkActionTouchClass}`}>
                  <Cloud className="size-3" />
                  {t("managePrompts.bulk.pushToServer", {
                    defaultValue: "Push to server"
                  })}
                </button>
              )}
              <button
                onClick={async () => {
                  if (guardPrivateMode()) return
                  const ok = await confirmDanger({
                    title: t("common:confirmTitle", { defaultValue: "Please confirm" }),
                    content: t("managePrompts.bulk.deleteConfirm", {
                      defaultValue: "Are you sure you want to delete {{count}} prompts?",
                      count: bulk.selectedRowKeys.length
                    }),
                    okText: t("common:delete", { defaultValue: "Delete" }),
                    cancelText: t("common:cancel", { defaultValue: "Cancel" })
                  })
                  if (!ok) return
                  bulk.bulkDeletePrompts(bulk.selectedRowKeys as string[])
                }}
                disabled={bulk.isBulkDeleting}
                data-testid="prompts-bulk-delete"
                className={`inline-flex items-center gap-1 rounded border border-danger/30 text-sm text-danger hover:bg-danger/10 disabled:opacity-50 ${bulkActionTouchClass}`}>
                <Trash2 className="size-3" /> {t("managePrompts.bulk.delete", { defaultValue: "Delete selected" })}
              </button>
              <button
                onClick={() => bulk.setSelectedRowKeys([])}
                data-testid="prompts-clear-selection"
                className={`ml-auto inline-flex items-center rounded text-sm text-text-muted hover:text-text ${isCompactViewport ? "min-h-[44px] px-2" : ""}`}>
                {t("common:clearSelection", { defaultValue: "Clear selection" })}
              </button>
            </PromptBulkActionBar>
          )}
          {isOnline && (sync.batchSyncState.running || sync.batchSyncState.failed.length > 0) && (
            <div
              data-testid="prompts-batch-sync-status"
              className="flex flex-wrap items-center gap-2 rounded-md border border-border bg-surface2 p-2"
            >
              {sync.batchSyncState.running ? (
                <span className="text-sm text-text-muted">
                  {t("managePrompts.sync.batchProgress", {
                    defaultValue: "Syncing {{completed}} of {{total}} prompts...",
                    completed: sync.batchSyncState.completed,
                    total: sync.batchSyncState.total
                  })}
                </span>
              ) : (
                <span className="text-sm text-warn">
                  {t("managePrompts.sync.batchFailedCount", {
                    defaultValue:
                      "{{count}} prompt(s) failed in the last batch run. Retry to continue.",
                    count: sync.batchSyncState.failed.length
                  })}
                </span>
              )}
            </div>
          )}
          {isOnline && (
            <div
              data-testid="prompts-collections-panel"
              className="flex flex-wrap items-center gap-2 rounded-md border border-border bg-surface2 p-2"
            >
              <Select
                value={collections.collectionFilter}
                onChange={(value) =>
                  collections.setCollectionFilter(
                    value === "all" ? "all" : Number(value)
                  )
                }
                loading={collections.promptCollectionsStatus === "pending"}
                style={{ minWidth: isCompactViewport ? "100%" : 260 }}
                data-testid="prompts-collection-filter"
                options={[
                  {
                    label: t("managePrompts.collections.filterAll", {
                      defaultValue: "All collections"
                    }),
                    value: "all"
                  },
                  ...collections.promptCollections.map((collection) => ({
                    label: `${collection.name} (${collection.prompt_ids?.length || 0})`,
                    value: collection.collection_id
                  }))
                ]}
              />
              <button
                type="button"
                onClick={() => collections.setCreateCollectionModalOpen(true)}
                data-testid="prompts-collection-create"
                className="inline-flex items-center gap-2 rounded-md border border-border px-2 py-2 text-sm font-medium text-text hover:bg-surface2 focus:outline-none focus:ring-2 focus:ring-focus focus:ring-offset-2"
              >
                <FolderPlus className="size-4" />
                {t("managePrompts.collections.create", {
                  defaultValue: "New collection"
                })}
              </button>
              {collections.selectedCollection && bulk.selectedRowKeys.length > 0 && (
                <button
                  type="button"
                  onClick={() =>
                    collections.addPromptsToCollectionMutation({
                      collection: collections.selectedCollection!,
                      prompts: bulk.selectedPromptRows
                    })
                  }
                  disabled={collections.isAssigningPromptCollection}
                  data-testid="prompts-collection-add-selected"
                  className="inline-flex items-center gap-2 rounded-md border border-primary/40 px-2 py-2 text-sm font-medium text-primary hover:bg-primary/10 disabled:opacity-50"
                >
                  {t("managePrompts.collections.addSelected", {
                    defaultValue: "Add selected to collection"
                  })}
                </button>
              )}
            </div>
          )}
          <PromptListToolbar
            mode="legacy"
            className="flex flex-wrap items-start justify-between gap-3 sm:items-center"
          >
            {/* Left: Action buttons */}
            <div className="flex flex-wrap items-center gap-2">
              <Tooltip title={t("managePrompts.newPromptHint", { defaultValue: "New prompt (N)" })}>
              <button
                onClick={() => editor.openFullEditor()}
                data-testid="prompts-add"
                className="inline-flex items-center rounded-md border border-transparent bg-primary px-2 py-2 text-md font-medium leading-4 text-white shadow-sm hover:bg-primaryStrong focus:outline-none focus:ring-2 focus:ring-focus focus:ring-offset-2 disabled:opacity-50">
                {t("managePrompts.newPromptBtn", { defaultValue: "New prompt" })}
              </button>
              </Tooltip>
              <div className="inline-flex items-center rounded-md border border-border overflow-hidden">
                <button
                  onClick={() => importExport.triggerExport()}
                  data-testid="prompts-export"
                  aria-label={t("managePrompts.exportLabel", { defaultValue: "Export prompts" })}
                  className="inline-flex items-center gap-2 px-2 py-2 text-md font-medium leading-4 text-text hover:bg-surface2">
                  <Download className="size-4" /> {t("managePrompts.export", { defaultValue: "Export" })}
                </button>
                <Select
                  value={importExport.exportFormat}
                  onChange={(v) => importExport.setExportFormat(v as "json" | "csv" | "markdown")}
                  data-testid="prompts-export-format"
                  options={[
                    { label: "JSON", value: "json" },
                    { label: "CSV", value: "csv", disabled: !isOnline },
                    { label: "Markdown", value: "markdown", disabled: !isOnline }
                  ]}
                  variant="borderless"
                  style={{ width: 120 }}
                  popupMatchSelectWidth={false}
                />
              </div>
              {isOnline &&
                (localSyncBatchPlan.tasks.length > 0 ||
                  sync.batchSyncState.failed.length > 0 ||
                  sync.batchSyncState.running) && (
                  <Tooltip
                    title={
                      localSyncBatchPlan.skippedConflicts > 0
                        ? t("managePrompts.sync.batchConflictHint", {
                            defaultValue:
                              "{{count}} conflict prompt(s) require manual resolution.",
                            count: localSyncBatchPlan.skippedConflicts
                          })
                        : undefined
                    }
                  >
                    <button
                      type="button"
                      onClick={sync.handleBatchSyncAction}
                      data-testid="prompts-sync-all"
                      className="inline-flex items-center gap-2 rounded-md border border-border px-2 py-2 text-md font-medium leading-4 text-text hover:bg-surface2 disabled:opacity-50"
                    >
                      <Cloud className="size-4" />
                      {sync.batchSyncState.running
                        ? t("managePrompts.sync.batchCancel", {
                            defaultValue: "Cancel sync"
                          })
                        : sync.batchSyncState.failed.length > 0
                          ? t("managePrompts.sync.batchRetryFailed", {
                              defaultValue: "Retry failed ({{count}})",
                              count: sync.batchSyncState.failed.length
                            })
                          : t("managePrompts.sync.batchSyncAll", {
                              defaultValue: "Sync all"
                            })}
                    </button>
                  </Tooltip>
                )}
              {/* Import controls grouped together */}
              <div className="inline-flex items-center rounded-md border border-border overflow-hidden">
                <button
                  onClick={() => {
                    if (guardPrivateMode()) return
                    importExport.fileInputRef.current?.click()
                  }}
                  data-testid="prompts-import"
                  className="inline-flex items-center gap-2 px-2 py-2 text-md font-medium leading-4 text-text hover:bg-surface2">
                  <UploadCloud className="size-4" /> {t("managePrompts.import", { defaultValue: "Import" })}
                </button>
                <Select
                  value={importExport.importMode}
                  onChange={(v) => importExport.setImportMode(v as any)}
                  data-testid="prompts-import-mode"
                  options={[
                    { label: t("managePrompts.importMode.merge", { defaultValue: "Merge" }), value: "merge" },
                    { label: t("managePrompts.importMode.replaceWithBackup", { defaultValue: "Replace (backup)" }), value: "replace" }
                  ]}
                  variant="borderless"
                  style={{ width: 130 }}
                  popupMatchSelectWidth={false}
                />
              </div>
              <Tooltip
                title={t("managePrompts.shortcuts.openHint", {
                  defaultValue: "Keyboard shortcuts (?)"
                })}
              >
                <button
                  type="button"
                  onClick={() => setShortcutsHelpOpen(true)}
                  data-testid="prompts-shortcuts-help-button"
                  aria-label={t("managePrompts.shortcuts.openLabel", {
                    defaultValue: "Open keyboard shortcuts"
                  })}
                  className="inline-flex items-center gap-2 rounded-md border border-border px-2 py-2 text-md font-medium leading-4 text-text hover:bg-surface2 focus:outline-none focus:ring-2 focus:ring-focus focus:ring-offset-2"
                >
                  <Keyboard className="size-4" aria-hidden="true" />
                  {t("managePrompts.shortcuts.openButton", {
                    defaultValue: "Shortcuts"
                  })}
                </button>
              </Tooltip>
              <input
                ref={importExport.fileInputRef}
                type="file"
                accept="application/json"
                className="hidden"
                data-testid="prompts-import-file"
                aria-label={t("managePrompts.importFileLabel", { defaultValue: "Import prompts file" })}
                onChange={(e) => {
                  const file = e.target.files?.[0]
                  if (file) importExport.handleImportFile(file)
                  e.currentTarget.value = ""
                }}
              />
            </div>
            {/* Right: Filters */}
            <div className="flex w-full flex-wrap items-stretch gap-2 sm:w-auto sm:items-center sm:justify-end">
              <div
                data-testid="prompts-search-control"
                className="w-full sm:w-auto"
              >
                <Input
                  ref={searchInputRef}
                  allowClear
                  placeholder={t("managePrompts.searchWithScope", { defaultValue: "Search name, content, keywords..." })}
                  value={searchText}
                  onChange={(e) => setSearchText(e.target.value)}
                  data-testid="prompts-search"
                  aria-label={t("managePrompts.search", { defaultValue: "Search prompts..." })}
                  suffix={<kbd className="rounded border border-border px-1 text-xs text-text-subtle">/</kbd>}
                  style={{ width: isCompactViewport ? "100%" : 260 }}
                />
              </div>
              <div
                data-testid="prompts-type-filter-control"
                className="w-full sm:w-auto"
              >
                <Select
                  value={typeFilter}
                  onChange={(v) => setTypeFilter(v as any)}
                  data-testid="prompts-type-filter"
                  aria-label={t("managePrompts.filter.typeLabel", { defaultValue: "Filter by type" })}
                  style={{ width: isCompactViewport ? "100%" : 130 }}
                  options={[
                    { label: t("managePrompts.filter.all", { defaultValue: "All types" }), value: "all" },
                    { label: t("managePrompts.filter.system", { defaultValue: "System" }), value: "system" },
                    { label: t("managePrompts.filter.quick", { defaultValue: "Quick" }), value: "quick" }
                  ]}
                />
              </div>
              <div
                data-testid="prompts-usage-filter-control"
                className="w-full sm:w-auto"
              >
                <Select
                  value={usageFilter}
                  onChange={(v) => setUsageFilter(v as "all" | "used" | "unused")}
                  data-testid="prompts-usage-filter"
                  aria-label={t("managePrompts.filter.usageLabel", {
                    defaultValue: "Filter by usage"
                  })}
                  style={{ width: isCompactViewport ? "100%" : 150 }}
                  options={[
                    {
                      label: t("managePrompts.filter.usageAll", {
                        defaultValue: "All usage"
                      }),
                      value: "all"
                    },
                    {
                      label: t("managePrompts.filter.usageUsed", {
                        defaultValue: "Used"
                      }),
                      value: "used"
                    },
                    {
                      label: t("managePrompts.filter.usageUnused", {
                        defaultValue: "Unused"
                      }),
                      value: "unused"
                    }
                  ]}
                />
              </div>
              <div
                data-testid="prompts-tag-filter-control"
                className="w-full sm:w-auto"
              >
                <Select
                  mode="multiple"
                  allowClear
                  placeholder={t("managePrompts.tags.placeholder", { defaultValue: "Filter keywords" })}
                  style={{ width: isCompactViewport ? "100%" : 220 }}
                  value={tagFilter}
                  onChange={(v) => setTagFilter(v)}
                  data-testid="prompts-tag-filter"
                  aria-label={t("managePrompts.tags.filterLabel", { defaultValue: "Filter by keywords" })}
                  options={allTags.map((t) => ({ label: t, value: t }))}
                />
              </div>
              <div
                data-testid="prompts-tag-match-mode-control"
                className="w-full sm:w-auto"
              >
                <Segmented
                  value={tagMatchMode}
                  onChange={(value) => setTagMatchMode(value as TagMatchMode)}
                  size="small"
                  data-testid="prompts-tag-match-mode"
                  style={{ width: isCompactViewport ? "100%" : undefined }}
                  options={[
                    {
                      value: "any",
                      label: t("managePrompts.tags.matchAny", {
                        defaultValue: "Match any"
                      })
                    },
                    {
                      value: "all",
                      label: t("managePrompts.tags.matchAll", {
                        defaultValue: "Match all"
                      })
                    }
                  ]}
                />
              </div>
              {selectedSegment === "custom" && (
                <div className="w-full sm:w-auto">
                  <Segmented
                    value={viewMode}
                    onChange={(value) =>
                      setViewMode(value as PromptViewMode)
                    }
                    size="small"
                    data-testid="prompts-view-mode"
                    aria-label="View mode"
                    options={[
                      {
                        value: "table",
                        icon: <List className="h-3.5 w-3.5" />,
                        label: "Table"
                      },
                      {
                        value: "gallery",
                        icon: <LayoutGrid className="h-3.5 w-3.5" />,
                        label: "Gallery"
                      }
                    ]}
                  />
                </div>
              )}
              {viewMode === "gallery" && selectedSegment === "custom" && (
                <div className="w-full sm:w-auto">
                  <Segmented
                    value={galleryDensity}
                    onChange={(value) =>
                      setGalleryDensity(value as PromptGalleryDensity)
                    }
                    size="small"
                    data-testid="prompts-gallery-density"
                    aria-label="Gallery density"
                    options={[
                      { value: "rich", label: "Rich" },
                      { value: "compact", label: "Compact" }
                    ]}
                  />
                </div>
              )}
              {(viewMode === "table" || selectedSegment !== "custom") && (
                <div className="w-full sm:w-auto">
                  <Segmented
                    value={tableDensity}
                    onChange={(value) =>
                      setTableDensity(value as PromptTableDensity)
                    }
                    size="small"
                    data-testid="prompts-table-density"
                    aria-label={t("managePrompts.tableDensity.label", {
                      defaultValue: "Table density"
                    })}
                    options={[
                      {
                        value: "comfortable",
                        label: t("managePrompts.tableDensity.comfortable", {
                          defaultValue: "Comfortable"
                        })
                      },
                      {
                        value: "compact",
                        label: t("managePrompts.tableDensity.compact", {
                          defaultValue: "Compact"
                        })
                      },
                      {
                        value: "dense",
                        label: t("managePrompts.tableDensity.dense", {
                          defaultValue: "Dense"
                        })
                      }
                    ]}
                  />
                </div>
              )}
            </div>
          </PromptListToolbar>
        </div>

        {customPromptsLoading && <Skeleton paragraph={{ rows: 8 }} />}

        {useServerSearchResults && hiddenServerResultsOnPage > 0 && (
          <Alert
            type="info"
            showIcon
            className="mb-3"
            message={t("managePrompts.search.localSubset", {
              defaultValue: "Showing synced local matches only"
            })}
            description={t("managePrompts.search.localSubsetDesc", {
              defaultValue:
                "{{count}} result(s) from this page are not saved locally yet.",
              count: hiddenServerResultsOnPage
            })}
          />
        )}

        {status === "success" && Array.isArray(data) && data.length === 0 && (
          <>
            <FeatureEmptyState
              title={t("settings:managePrompts.emptyTitle", {
                defaultValue: "No custom prompts yet"
              })}
              description={t("settings:managePrompts.emptyDescription", {
                defaultValue:
                  "Create reusable prompts for recurring tasks, workflows, and team conventions."
              })}
              examples={[
                t("settings:managePrompts.emptyExample1", {
                  defaultValue:
                    "Save your favorite system prompt for summaries, explanations, or translations."
                }),
                t("settings:managePrompts.emptyExample2", {
                  defaultValue:
                    "Create quick prompts for common actions like drafting emails or refining notes."
                })
              ]}
              primaryActionLabel={t("settings:managePrompts.emptyPrimaryCta", {
                defaultValue: "Create prompt"
              })}
              onPrimaryAction={() => editor.openFullEditor()}
            />
            <div className="mt-6">
              <h3 className="mb-3 text-sm font-medium text-text-muted">
                Or start with a template
              </h3>
              <Suspense fallback={null}>
                <PromptStarterCards
                  onUse={(starter) => editor.openFullEditor(starter)}
                />
              </Suspense>
            </div>
          </>
        )}

        {status === "success" && Array.isArray(data) && data.length >= 5 && shouldShowHint("keyboard-shortcuts") && (
          <Suspense fallback={null}>
            <ContextualHint
              id="keyboard-shortcuts"
              message="Press Enter to preview, E to edit, or ? for all keyboard shortcuts."
              visible={true}
              onDismiss={dismissHint}
              onShown={markHintShown}
            />
          </Suspense>
        )}

        {status === "success" && Array.isArray(data) && data.length > 0 && viewMode === "table" && (
          <PromptListTable
            rows={customPromptRows}
            total={tableTotal}
            loading={customPromptsLoading}
            isOnline={isOnline}
            isCompactViewport={isCompactViewport}
            query={customPromptTableQuery}
            selectedIds={bulk.selectedRowKeys.map((key) => String(key))}
            onQueryChange={handleCustomPromptTableQueryChange}
            onSelectionChange={(ids) => bulk.setSelectedRowKeys(ids)}
            onRowOpen={openPromptInspector}
            onEdit={editor.handleEditPromptById}
            onToggleFavorite={editor.handleTogglePromptFavorite}
            onOpenConflictResolution={sync.openConflictResolution}
            renderActions={renderCustomPromptActions}
            renderTitleMeta={renderCustomPromptTitleMeta}
            favoriteButtonTestId={(row) => `prompt-favorite-${row.id}`}
            formatRelativeTime={formatRelativePromptTime}
            selectionDisabled={isFireFoxPrivateMode}
            columnLabels={{
              title: t("managePrompts.columns.title"),
              preview: t("managePrompts.columns.prompt"),
              tags: t("managePrompts.tags.label", {
                defaultValue: "Keywords"
              }),
              updated: t("managePrompts.columns.modified", {
                defaultValue: "Updated"
              }),
              lastUsed: t("managePrompts.columns.lastUsed", {
                defaultValue: "Last used"
              }),
              status: t("managePrompts.columns.sync", {
                defaultValue: "Sync"
              }),
              actions: t("managePrompts.columns.actions"),
              author: t("managePrompts.form.author.label", {
                defaultValue: "Author"
              }),
              system: t("managePrompts.form.systemPrompt.shortLabel", {
                defaultValue: "System"
              }),
              user: t("managePrompts.form.userPrompt.shortLabel", {
                defaultValue: "User"
              }),
              unknown: t("common:unknown", {
                defaultValue: "Unknown"
              }),
              offlineStatus: t("managePrompts.sync.offlineTooltip", {
                defaultValue:
                  "Sync unavailable (offline). Showing last known status."
              }),
              edit: t("managePrompts.tooltip.edit")
            }}
            paginationShowTotal={(total, range) =>
              t("managePrompts.pagination.summary", {
                defaultValue: "{{start}}-{{end}} of {{total}} prompts",
                start: range[0],
                end: range[1],
                total
              })
            }
            tableDensity={tableDensity}
          />
        )}

        {status === "success" && Array.isArray(data) && data.length > 0 && viewMode === "gallery" && (
          <div className="space-y-4" data-testid="prompts-gallery-view">
            <div className={`grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 ${
              galleryDensity === "compact" ? "gap-3" : "gap-4"
            }`}>
              <Suspense fallback={null}>
                {customPromptRows.map((prompt) => (
                  <PromptGalleryCard
                    key={prompt.id}
                    prompt={prompt}
                    onClick={() => openPromptInspector(prompt.id)}
                    density={galleryDensity}
                    onToggleFavorite={(next) => editor.handleTogglePromptFavorite(prompt.id, next)}
                  />
                ))}
              </Suspense>
            </div>
            {tableTotal > resultsPerPage && (
              <div className="flex justify-end">
                <Pagination
                  current={currentPage}
                  pageSize={resultsPerPage}
                  total={tableTotal}
                  showSizeChanger
                  pageSizeOptions={["10", "20", "50", "100"]}
                  onChange={(page, pageSize) => {
                    if (pageSize !== resultsPerPage) {
                      setResultsPerPage(pageSize)
                      setCurrentPage(1)
                    } else {
                      setCurrentPage(page)
                    }
                  }}
                  showTotal={(total, range) =>
                    t("managePrompts.pagination.summary", {
                      defaultValue: "{{start}}-{{end}} of {{total}} prompts",
                      start: range[0],
                      end: range[1],
                      total
                    })
                  }
                />
              </div>
            )}
          </div>
        )}
      </div>
    )
  }

  function copilotPrompts() {
    if (!isOnline) {
      return (
        <ConnectFeatureBanner
          title={t("settings:managePrompts.emptyConnectTitle", {
            defaultValue: "Connect to use Prompts"
          })}
          description={t("settings:managePrompts.emptyConnectDescription", {
            defaultValue:
              "To manage reusable prompts, first connect to your tldw server."
          })}
          examples={[
            t("settings:managePrompts.emptyConnectExample1", {
              defaultValue:
                "Open Settings → tldw server to add your server URL."
            }),
            t("settings:managePrompts.emptyConnectExample2", {
              defaultValue:
                "Once connected, create custom prompts you can reuse across chats."
            })
          ]}
        />
      )
    }
    return (
      <div>
        {copilotStatus === "pending" && <Skeleton paragraph={{ rows: 8 }} />}

        {copilotStatus === "success" && Array.isArray(copilotData) && copilotData.length === 0 && (
          <FeatureEmptyState
            title={t("managePrompts.copilotEmptyTitle", {
              defaultValue: "No Copilot prompts available"
            })}
            description={t("managePrompts.copilotEmptyDescription", {
              defaultValue:
                "Copilot prompts are predefined templates provided by your tldw server."
            })}
            examples={[
              t("managePrompts.copilotEmptyExample1", {
                defaultValue:
                  "Check your server version or configuration if you expect Copilot prompts to be available."
              }),
              t("managePrompts.copilotEmptyExample2", {
                defaultValue:
                  "After updating your server, reload the extension and return to this tab."
              })
            ]}
            primaryActionLabel={t("settings:healthSummary.diagnostics", {
              defaultValue: "Open Diagnostics"
            })}
            onPrimaryAction={() => navigate("/settings/health")}
          />
        )}

        {copilotStatus === "success" && Array.isArray(copilotData) && copilotData.length > 0 && (
          <>
            <div className="mb-3 flex flex-wrap items-center gap-2">
              <Input
                value={copilotSearchText}
                onChange={(event) => setCopilotSearchText(event.target.value)}
                allowClear
                placeholder={t("managePrompts.copilot.search.placeholder", {
                  defaultValue: "Search copilot prompts..."
                })}
                style={{ width: 260 }}
                data-testid="copilot-search"
              />
              <Select
                value={copilotKeyFilter}
                onChange={(value) => setCopilotKeyFilter(value)}
                options={[
                  {
                    label: t("managePrompts.copilot.filter.all", {
                      defaultValue: "All prompt types"
                    }),
                    value: "all"
                  },
                  ...copilotPromptKeyOptions
                ]}
                style={{ width: 200 }}
                data-testid="copilot-key-filter"
              />
            </div>

            {filteredCopilotData.length === 0 ? (
              <div className="rounded-md border border-border p-4 text-sm text-text-muted">
                {t("managePrompts.copilot.search.empty", {
                  defaultValue: "No copilot prompts match the current filters."
                })}
              </div>
            ) : (
              <Table
                className={`prompts-table prompts-table-density-${tableDensity}`}
                size={tableDensity === "comfortable" ? "middle" : "small"}
                columns={[
                  {
                    title: t("managePrompts.columns.title"),
                    dataIndex: "key",
                    key: "key",
                    render: (content) => (
                      <span className="line-clamp-1">
                        <Tag color={tagColors[content || "default"]}>
                          {t(`common:copilot.${content}`)}
                        </Tag>
                      </span>
                    )
                  },
                  {
                    title: t("managePrompts.columns.prompt"),
                    dataIndex: "prompt",
                    key: "prompt",
                    render: (content) => <span className="line-clamp-1">{content}</span>
                  },
                  {
                    render: (_, record) => (
                      <div className="flex items-center gap-1">
                        <Tooltip title={t("managePrompts.tooltip.edit")}>
                          <button
                            type="button"
                            aria-label={t("managePrompts.tooltip.edit")}
                            onClick={() => {
                              setEditCopilotId(record.key)
                              editCopilotForm.setFieldsValue(record)
                              setOpenCopilotEdit(true)
                            }}
                            data-testid={`copilot-action-edit-${record.key}`}
                            className="inline-flex min-h-8 min-w-8 items-center justify-center rounded p-2 text-text-muted hover:bg-bg-muted/70 focus:outline-none focus:ring-2 focus:ring-primary"
                          >
                            <Pen className="size-4" />
                          </button>
                        </Tooltip>
                        <Tooltip
                          title={t("managePrompts.copilot.copyToCustom.button", {
                            defaultValue: "Copy to Custom"
                          })}
                        >
                          <button
                            type="button"
                            aria-label={t("managePrompts.copilot.copyToCustom.button", {
                              defaultValue: "Copy to Custom"
                            })}
                            onClick={() => handleCopyCopilotToCustom(record)}
                            data-testid={`copilot-action-copy-custom-${record.key}`}
                            className="inline-flex min-h-8 min-w-8 items-center justify-center rounded p-2 text-text-muted hover:bg-bg-muted/70 focus:outline-none focus:ring-2 focus:ring-primary"
                          >
                            <Copy className="size-4" />
                          </button>
                        </Tooltip>
                        <Tooltip
                          title={t("managePrompts.copilot.copyToClipboard.button", {
                            defaultValue: "Copy to clipboard"
                          })}
                        >
                          <button
                            type="button"
                            aria-label={t("managePrompts.copilot.copyToClipboard.button", {
                              defaultValue: "Copy to clipboard"
                            })}
                            onClick={() => {
                              void copyCopilotPromptToClipboard(record)
                            }}
                            data-testid={`copilot-action-copy-clipboard-${record.key}`}
                            className="inline-flex min-h-8 min-w-8 items-center justify-center rounded p-2 text-text-muted hover:bg-bg-muted/70 focus:outline-none focus:ring-2 focus:ring-primary"
                          >
                            <Clipboard className="size-4" />
                          </button>
                        </Tooltip>
                      </div>
                    )
                  }
                ]}
                dataSource={filteredCopilotData}
                rowKey={(record) => record.key}
              />
            )}
          </>
        )}
      </div>
    )
  }

  function trashPrompts() {
    const trashCount = Array.isArray(trashData) ? trashData.length : 0
    const filteredTrashCount = filteredTrashData.length

    const formatDeletedAt = (timestamp: number | null | undefined) => {
      if (!timestamp) return ""
      const date = new Date(timestamp)
      const now = new Date()
      const diffMs = now.getTime() - date.getTime()
      const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24))
      if (diffDays === 0) return t("managePrompts.trash.today", { defaultValue: "Today" })
      if (diffDays === 1) return t("managePrompts.trash.yesterday", { defaultValue: "Yesterday" })
      if (diffDays < 7) return t("managePrompts.trash.daysAgo", { defaultValue: "{{count}} days ago", count: diffDays })
      return date.toLocaleDateString()
    }

    return (
      <div data-testid="prompts-trash">
        <div className="mb-6">
          {trashCount > 0 && (
            <div className="space-y-3">
              <div className="flex items-center justify-between p-3 bg-warn/10 rounded-md border border-warn/30">
                <div className="flex items-center gap-2">
                  <AlertTriangle className="size-4 text-warn" />
                  <span className="text-sm">
                    {t("managePrompts.trash.autoDeleteWarning", {
                      defaultValue: "Prompts in trash are automatically deleted after 30 days."
                    })}
                  </span>
                </div>
                <button
                  onClick={async () => {
                    const ok = await confirmDanger({
                      title: t("managePrompts.trash.emptyConfirmTitle", { defaultValue: "Empty Trash?" }),
                      content: t("managePrompts.trash.emptyConfirmContent", {
                        defaultValue: "This will permanently delete {{count}} prompts. This action cannot be undone.",
                        count: trashCount
                      }),
                      okText: t("managePrompts.trash.emptyTrash", { defaultValue: "Empty Trash" }),
                      cancelText: t("common:cancel", { defaultValue: "Cancel" })
                    })
                    if (!ok) return
                    editor.emptyTrashMutation()
                  }}
                  disabled={editor.isEmptyingTrash}
                  className="inline-flex items-center gap-1 px-2 py-1 text-sm rounded border border-danger/30 text-danger hover:bg-danger/10 disabled:opacity-50">
                  <Trash2 className="size-3" />
                  {t("managePrompts.trash.emptyTrash", { defaultValue: "Empty Trash" })}
                </button>
              </div>

              <Input
                value={trashSearchText}
                onChange={(event) => setTrashSearchText(event.target.value)}
                allowClear
                placeholder={t("managePrompts.trash.searchPlaceholder", {
                  defaultValue: "Search deleted prompts..."
                })}
                style={{ width: 320, maxWidth: "100%" }}
                data-testid="prompts-trash-search"
              />

              {bulk.trashSelectedRowKeys.length > 0 && (
                <div className="flex flex-wrap items-center justify-between gap-2 p-2 rounded-md border border-primary/20 bg-primary/5">
                  <span className="text-sm text-text-muted">
                    {t("managePrompts.bulk.selectedCount", {
                      defaultValue: "{{count}} selected",
                      count: bulk.trashSelectedRowKeys.length
                    })}
                  </span>
                  <button
                    type="button"
                    data-testid="prompts-trash-bulk-restore"
                    onClick={() =>
                      bulk.bulkRestorePrompts(
                        bulk.trashSelectedRowKeys.map((key) => String(key))
                      )
                    }
                    disabled={bulk.isBulkRestoring}
                    className="inline-flex items-center gap-1 px-3 py-1.5 text-sm rounded border border-primary/30 text-primary hover:bg-primary/10 disabled:opacity-50"
                  >
                    <Undo2 className="size-3" />
                    {t("managePrompts.trash.restoreSelected", {
                      defaultValue: "Restore selected"
                    })}
                  </button>
                </div>
              )}
            </div>
          )}
        </div>

        {trashStatus === "pending" && <Skeleton paragraph={{ rows: 8 }} />}

        {trashStatus === "success" && trashCount === 0 && (
          <FeatureEmptyState
            title={t("managePrompts.trash.emptyTitle", { defaultValue: "Trash is empty" })}
            description={t("managePrompts.trash.emptyDescription", {
              defaultValue: "Deleted prompts will appear here for 30 days before being permanently removed."
            })}
            examples={[
              t("managePrompts.trash.emptyExample1", {
                defaultValue: "You can restore deleted prompts at any time while they're in the trash."
              })
            ]}
          />
        )}

        {trashStatus === "success" && trashCount > 0 && (
          <>
            {filteredTrashCount === 0 ? (
              <div className="rounded-md border border-border p-4 text-sm text-text-muted">
                {t("managePrompts.trash.searchEmpty", {
                  defaultValue: "No deleted prompts match your search."
                })}
              </div>
            ) : (
              <Table
                className={`prompts-table prompts-table-density-${tableDensity}`}
                size={tableDensity === "comfortable" ? "middle" : "small"}
                data-testid="prompts-trash-table"
                columns={[
                  {
                    title: t("managePrompts.columns.title"),
                    dataIndex: "title",
                    key: "title",
                    render: (_: any, record: any) => (
                      <div className="flex max-w-64 flex-col">
                        <span className="line-clamp-1 font-medium text-text-muted">
                          {record?.name || record?.title}
                        </span>
                        {record?.author && (
                          <span className="text-xs text-text-muted opacity-70">
                            {t("managePrompts.form.author.label", { defaultValue: "Author" })}: {record.author}
                          </span>
                        )}
                      </div>
                    )
                  },
                  {
                    title: t("managePrompts.columns.prompt", {
                      defaultValue: "Prompt"
                    }),
                    key: "contentPreview",
                    render: (_: any, record: any) => {
                      const { systemText, userText } = getPromptTexts(record)
                      const preview = (
                        userText ||
                        systemText ||
                        (typeof record?.content === "string" ? record.content : "")
                      ).trim()

                      if (!preview) {
                        return (
                          <span className="text-xs text-text-muted opacity-70">
                            {t("managePrompts.trash.noPreview", {
                              defaultValue: "No content preview"
                            })}
                          </span>
                        )
                      }

                      return (
                        <Tooltip title={preview}>
                          <span className="line-clamp-2 max-w-[26rem] text-sm">
                            {preview}
                          </span>
                        </Tooltip>
                      )
                    }
                  },
                  {
                    title: t("managePrompts.trash.deletedAt", { defaultValue: "Deleted" }),
                    key: "deletedAt",
                    width: 140,
                    render: (_: any, record: any) => (
                      <span className="text-sm text-text-muted">
                        {formatDeletedAt(record.deletedAt)}
                      </span>
                    )
                  },
                  {
                    title: t("managePrompts.trash.remaining", {
                      defaultValue: "Remaining"
                    }),
                    key: "remaining",
                    width: 140,
                    render: (_: any, record: any) => {
                      const remainingDays = getTrashDaysRemaining(record.deletedAt)
                      const severity = getTrashRemainingSeverity(remainingDays)
                      const className =
                        severity === "danger"
                          ? "text-danger"
                          : severity === "warning"
                            ? "text-warn"
                            : "text-text-muted"
                      const label =
                        remainingDays <= 0
                          ? t("managePrompts.trash.remainingExpired", {
                              defaultValue: "Due now"
                            })
                          : `${remainingDays} ${
                              remainingDays === 1
                                ? t("managePrompts.trash.dayLeft", { defaultValue: "day left" })
                                : t("managePrompts.trash.daysLeft", { defaultValue: "days left" })
                            }`

                      return (
                        <span
                          className={`text-sm ${className}`}
                          data-testid={`prompts-trash-remaining-${record.id}`}
                        >
                          {label}
                        </span>
                      )
                    }
                  },
                  {
                    title: t("managePrompts.columns.actions"),
                    width: 160,
                    render: (_: any, record: any) => (
                      <div className="flex items-center gap-2">
                        <Tooltip title={t("managePrompts.trash.restore", { defaultValue: "Restore" })}>
                          <button
                            type="button"
                            data-testid={`prompts-trash-restore-${record.id}`}
                            onClick={() => editor.restorePromptMutation(record.id)}
                            className="inline-flex items-center gap-1 px-2 py-1 text-sm rounded border border-primary/30 text-primary hover:bg-primary/10">
                            <Undo2 className="size-3" />
                            {t("managePrompts.trash.restore", { defaultValue: "Restore" })}
                          </button>
                        </Tooltip>
                        <Tooltip title={t("managePrompts.trash.deletePermanently", { defaultValue: "Delete permanently" })}>
                          <button
                            type="button"
                            onClick={async () => {
                              const ok = await confirmDanger({
                                title: t("managePrompts.trash.permanentDeleteTitle", { defaultValue: "Delete permanently?" }),
                                content: t("managePrompts.trash.permanentDeleteContent", {
                                  defaultValue: "This prompt will be permanently deleted. This action cannot be undone."
                                }),
                                okText: t("common:delete", { defaultValue: "Delete" }),
                                cancelText: t("common:cancel", { defaultValue: "Cancel" })
                              })
                              if (!ok) return
                              editor.permanentDeletePromptMutation(record.id)
                            }}
                            className="text-text-muted hover:text-danger">
                            <Trash2 className="size-4" />
                          </button>
                        </Tooltip>
                      </div>
                    )
                  }
                ]}
                dataSource={filteredTrashData}
                rowKey={(record) => record.id}
                rowSelection={{
                  selectedRowKeys: bulk.trashSelectedRowKeys,
                  onChange: (keys) => bulk.setTrashSelectedRowKeys(keys)
                }}
              />
            )}
          </>
        )}
      </div>
    )
  }

  const deferredPromptSurfaceFallback = null

  const renderStudioTabContainer = () => (
    <Suspense fallback={deferredPromptSurfaceFallback}>
      <StudioTabContainer />
    </Suspense>
  )

  const renderPromptDrawer = () => (
    <Suspense fallback={deferredPromptSurfaceFallback}>
      <PromptDrawer
        open={editor.drawerOpen}
        onClose={() => {
          editor.setDrawerOpen(false)
          editor.setDrawerInitialValues(null)
        }}
        mode={editor.drawerMode}
        initialValues={editor.drawerInitialValues}
        onSubmit={editor.handleDrawerSubmit}
        isLoading={editor.drawerMode === "create" ? editor.savePromptLoading : editor.isUpdatingPrompt}
        allTags={allTags}
      />
    </Suspense>
  )

  const renderPromptFullPageEditor = () => (
    <Suspense fallback={deferredPromptSurfaceFallback}>
      <PromptFullPageEditor
        open={editor.fullEditorOpen}
        onClose={editor.closeFullEditor}
        mode={editor.fullEditorMode}
        initialValues={editor.fullEditorInitialValues}
        onSubmit={editor.handleFullEditorSubmit}
        isLoading={editor.fullEditorMode === "create" ? editor.savePromptLoading : editor.isUpdatingPrompt}
        allTags={allTags}
      />
    </Suspense>
  )

  const renderPromptInspectorPanel = () => (
    <Suspense fallback={deferredPromptSurfaceFallback}>
      <PromptInspectorPanel
        open={inspectorOpen}
        prompt={inspectorPrompt}
        onClose={closeInspector}
        onEdit={(promptId) => {
          const promptRecord = getPromptRecordById(promptId)
          if (!promptRecord) return
          closeInspector()
          editor.openFullEditor(promptRecord)
        }}
        onUseInChat={(promptId) => {
          const promptRecord = getPromptRecordById(promptId)
          if (!promptRecord) return
          closeInspector()
          void handleUsePromptInChat(promptRecord)
        }}
        onDuplicate={(promptId) => {
          const promptRecord = getPromptRecordById(promptId)
          if (!promptRecord) return
          editor.handleDuplicatePrompt(promptRecord)
        }}
        onDelete={(promptId) => {
          const promptRecord = getPromptRecordById(promptId)
          if (!promptRecord) return
          closeInspector()
          void editor.handleDeletePrompt(promptRecord)
        }}
      />
    </Suspense>
  )

  const renderProjectSelector = () => (
    <Suspense fallback={deferredPromptSurfaceFallback}>
      <ProjectSelector
        open={sync.projectSelectorOpen}
        onClose={() => {
          sync.setProjectSelectorOpen(false)
          sync.setPromptToSync(null)
        }}
        onSelect={(projectId) => {
          if (sync.promptToSync) {
            sync.pushToStudioMutation({ localId: sync.promptToSync, projectId })
          }
        }}
        loading={sync.isPushing}
      />
    </Suspense>
  )

  const renderConflictResolutionModal = () => (
    <Suspense fallback={deferredPromptSurfaceFallback}>
      <ConflictResolutionModal
        open={sync.conflictModalOpen}
        loading={sync.isLoadingConflictInfo || sync.isResolvingConflict}
        conflictInfo={sync.conflictInfo}
        onClose={sync.closeConflictResolution}
        onResolve={sync.handleResolveConflict}
      />
    </Suspense>
  )

  return (
    <div>
      {/* Screen reader status announcements */}
      <div
        role="status"
        aria-live="polite"
        aria-atomic="true"
        className="sr-only"
        id="prompts-status-announcer"
      />

      {/* Firefox Private Mode Warning */}
      {isFireFoxPrivateMode && (
        <Alert
          type="warning"
          showIcon
          icon={<AlertTriangle className="size-4" />}
          className="mb-4"
          title={t("managePrompts.privateMode.title", { defaultValue: "Limited functionality in Private Mode" })}
          description={t("managePrompts.privateMode.description", {
            defaultValue: "Firefox Private Mode doesn't support IndexedDB. You can view existing prompts, but creating, editing, or importing prompts is disabled. Use a normal window for full functionality."
          })}
        />
      )}
      {(promptLoadFailed || copilotLoadFailed) && (
        <Alert
          type="error"
          showIcon
          className="mb-4"
          title={t(
            "managePrompts.partialLoad",
            "Some prompt data isn't available"
          )}
          description={
            loadErrorDescription ||
            t(
              "managePrompts.loadErrorHelp",
              "Check your server connection and refresh to try again."
            )
          }
        />
      )}
      <div className="flex gap-0">
        {/* Sidebar - desktop only */}
        {!isCompactViewport && (
          <PromptSidebar
            collapsed={sidebarCollapsed}
            onToggleCollapsed={() => setSidebarCollapsed((p) => !p)}
            selectedSegment={selectedSegment}
            onSegmentChange={(s) => setSelectedSegment(s as SegmentType)}
            trashCount={trashData?.length}
            savedView={savedView}
            onSavedViewChange={setSavedView}
            smartCounts={sidebarCounts.smartCounts}
            typeFilter={typeFilter}
            onTypeFilterChange={(v) => setTypeFilter(v as any)}
            typeCounts={sidebarCounts.typeCounts}
            syncFilter={syncFilter}
            onSyncFilterChange={setSyncFilter}
            syncCounts={sidebarCounts.syncCounts}
            tagFilter={tagFilter}
            onTagFilterChange={setTagFilter}
            tagMatchMode={tagMatchMode}
            onTagMatchModeChange={setTagMatchMode}
            tagCounts={sidebarCounts.tagCounts}
            presets={filterPresets}
            onLoadPreset={handleLoadFilterPreset}
            onSavePreset={handleSaveFilterPreset}
            onDeletePreset={deleteFilterPreset}
          />
        )}

        {/* Main content area */}
        <div className="flex-1 min-w-0">

      {/* Mobile segment tabs */}
      {isCompactViewport && (
      <div className="flex flex-col items-start gap-1 mb-6 px-4">
        <Segmented
          size="large"
          options={[
            {
              label: (
                <span className="flex items-center gap-1">
                  {t("managePrompts.segmented.custom", {
                    defaultValue: "Custom prompts"
                  })}
                  {pendingSyncCount > 0 && (
                    <Tooltip
                      title={t("managePrompts.sync.pendingCountTooltip", {
                        defaultValue:
                          "{{count}} prompt(s) have local changes pending sync.",
                        count: pendingSyncCount
                      })}
                    >
                      <span
                        data-testid="prompts-pending-sync-count"
                        className="text-xs bg-warn/20 text-warn px-1.5 py-0.5 rounded-full"
                      >
                        {pendingSyncCount}
                      </span>
                    </Tooltip>
                  )}
                </span>
              ),
              value: "custom"
            },
            {
              label: (
                <Tooltip title={t("managePrompts.segmented.copilotTooltip", {
                  defaultValue: "Predefined prompts from your tldw server that help with common tasks"
                })}>
                  <span>{t("managePrompts.segmented.copilot", { defaultValue: "Copilot prompts" })}</span>
                </Tooltip>
              ),
              value: "copilot",
              disabled: !isOnline
            },
            {
              label: (
                <Tooltip title={t("managePrompts.segmented.studioTooltip", {
                  defaultValue: "Browse and import prompts from Prompt Studio projects on the server"
                })}>
                  <span className="flex items-center gap-1">
                    <Cloud className="size-3" />
                    {t("managePrompts.segmented.studio", { defaultValue: "Studio" })}
                  </span>
                </Tooltip>
              ),
              value: "studio",
              disabled: !isOnline || hasStudio === false
            },
            {
              label: (
                <span className="flex items-center gap-1">
                  <Trash2 className="size-3" />
                  {t("managePrompts.segmented.trash", { defaultValue: "Trash" })}
                  {(trashData?.length || 0) > 0 && (
                    <span className="text-xs bg-text-muted/20 px-1.5 py-0.5 rounded-full">
                      {trashData?.length}
                    </span>
                  )}
                </span>
              ),
              value: "trash"
            }
          ]}
          data-testid="prompts-segmented"
          value={selectedSegment}
          onChange={(value) => {
            setSelectedSegment(value as SegmentType)
          }}
        />
        <p className="text-xs text-text-muted ">
          {selectedSegment === "custom"
            ? t("managePrompts.segmented.helpCustom", {
                defaultValue:
                  "Create and manage reusable prompts you can insert into chat."
              })
            : selectedSegment === "copilot"
            ? t("managePrompts.segmented.helpCopilot", {
                defaultValue:
                  "View and tweak predefined Copilot prompts provided by your server."
              })
            : selectedSegment === "studio"
            ? t("managePrompts.segmented.helpStudio", {
                defaultValue:
                  "Full Prompt Studio: manage projects, prompts, test cases, evaluations, and optimizations."
              })
            : t("managePrompts.segmented.helpTrash", {
                defaultValue:
                  "Restore or permanently delete prompts. Items auto-delete after 30 days."
              })}
        </p>
      </div>
      )}
      <div className={isCompactViewport ? "px-4" : "p-4"}>
      {selectedSegment === "custom" && customPrompts()}
      {selectedSegment === "copilot" && copilotPrompts()}
      {selectedSegment === "studio" && renderStudioTabContainer()}
      {selectedSegment === "trash" && trashPrompts()}
      </div>

      </div>
      </div>

      {renderPromptDrawer()}

      {renderPromptFullPageEditor()}

      {renderPromptInspectorPanel()}

      <Modal
        title={t("managePrompts.modal.editTitle")}
        open={openCopilotEdit}
        onCancel={() => setOpenCopilotEdit(false)}
        footer={null}>
        <Form
          onFinish={(values) =>
            updateCopilotPrompt({
              key: editCopilotId,
              prompt: values.prompt
            })
          }
          layout="vertical"
          form={editCopilotForm}>
          <Form.Item
            name="prompt"
            label={t("managePrompts.form.prompt.label")}
            extra={
              <div className="flex flex-wrap items-center gap-2 text-xs">
                <span>
                  {t("managePrompts.form.prompt.copilotPlaceholderHint", {
                    defaultValue: "Must include placeholder"
                  })}
                </span>
                <Tag color={copilotPromptIncludesTextPlaceholder ? "green" : "orange"}>
                  {"{text}"}
                </Tag>
                <span
                  data-testid="copilot-text-placeholder-status"
                  className={
                    copilotPromptIncludesTextPlaceholder
                      ? "text-success"
                      : "text-warn"
                  }
                >
                  {copilotPromptIncludesTextPlaceholder
                    ? t("managePrompts.form.prompt.copilotPlaceholderPresent", {
                        defaultValue: "placeholder detected"
                      })
                    : t("managePrompts.form.prompt.copilotPlaceholderMissing", {
                        defaultValue: "missing placeholder"
                      })}
                </span>
              </div>
            }
            rules={[
              {
                required: true,
                message: t("managePrompts.form.prompt.required")
              },
              {
                validator: (_, value) => {
                  if (value && value.includes("{text}")) {
                    return Promise.resolve()
                  }
                  return Promise.reject(
                    new Error(
                      t("managePrompts.form.prompt.missingTextPlaceholder")
                    )
                  )
                }
              }
            ]}>
            <Input.TextArea
              placeholder={t("managePrompts.form.prompt.placeholder")}
              autoSize={{ minRows: 3, maxRows: 10 }}
              data-testid="copilot-edit-prompt-input"
            />
          </Form.Item>

          <Form.Item>
            <button
              data-testid="copilot-edit-save"
              disabled={isUpdatingCopilotPrompt}
              className="inline-flex justify-center w-full text-center mt-4 items-center rounded-md border border-transparent bg-primary px-2 py-2 text-sm font-medium leading-4 text-white shadow-sm hover:bg-primaryStrong focus:outline-none focus:ring-2 focus:ring-focus focus:ring-offset-2 disabled:opacity-50">
              {isUpdatingCopilotPrompt
                ? t("managePrompts.form.btnEdit.saving")
                : t("managePrompts.form.btnEdit.save")}
            </button>
          </Form.Item>
        </Form>
      </Modal>

      <Modal
        title={t("managePrompts.bulk.addKeyword", { defaultValue: "Add keyword" })}
        open={bulk.bulkKeywordModalOpen}
        onCancel={() => {
          bulk.setBulkKeywordModalOpen(false)
          bulk.setBulkKeywordValue("")
        }}
        onOk={() =>
          bulk.bulkAddKeyword({
            ids: bulk.selectedRowKeys.map((key) => String(key)),
            keyword: bulk.bulkKeywordValue
          })
        }
        okText={t("common:add", { defaultValue: "Add" })}
        cancelText={t("common:cancel", { defaultValue: "Cancel" })}
        okButtonProps={{
          disabled:
            bulk.bulkKeywordValue.trim().length === 0 || bulk.isBulkAddingKeyword,
          loading: bulk.isBulkAddingKeyword
        }}
      >
        <Input
          autoFocus
          value={bulk.bulkKeywordValue}
          onChange={(event) => bulk.setBulkKeywordValue(event.target.value)}
          onPressEnter={() => {
            if (
              bulk.bulkKeywordValue.trim().length === 0 ||
              bulk.isBulkAddingKeyword
            ) {
              return
            }
            bulk.bulkAddKeyword({
              ids: bulk.selectedRowKeys.map((key) => String(key)),
              keyword: bulk.bulkKeywordValue
            })
          }}
          placeholder={t("managePrompts.tags.addPlaceholder", {
            defaultValue: "Enter keyword"
          })}
          data-testid="prompts-bulk-keyword-input"
        />
      </Modal>

      <Modal
        title={t("managePrompts.quickTest.modalTitle", {
          defaultValue: "Quick test prompt"
        })}
        open={!!localQuickTestPrompt}
        onCancel={closeLocalQuickTestModal}
        footer={[
          <button
            key="cancel"
            type="button"
            data-testid="prompts-local-quick-test-cancel"
            className="inline-flex items-center justify-center rounded-md border border-border bg-bg px-3 py-2 text-sm text-text hover:bg-surface2"
            onClick={closeLocalQuickTestModal}
            disabled={isRunningLocalQuickTest}
          >
            {t("common:cancel", { defaultValue: "Cancel" })}
          </button>,
          <button
            key="run"
            type="button"
            data-testid="prompts-local-quick-test-run"
            className="inline-flex items-center justify-center rounded-md border border-transparent bg-primary px-3 py-2 text-sm text-white hover:bg-primaryStrong disabled:opacity-50"
            onClick={() => {
              void runLocalQuickTest()
            }}
            disabled={isRunningLocalQuickTest}
          >
            <Play className="mr-1 size-4" />
            {isRunningLocalQuickTest
              ? t("managePrompts.quickTest.running", { defaultValue: "Running..." })
              : t("managePrompts.quickTest.runAction", { defaultValue: "Run test" })}
          </button>
        ]}
        width={720}
      >
        {localQuickTestPrompt && (
          <div className="space-y-3">
            <div className="rounded border border-border bg-surface2 p-3">
              <div className="text-sm font-medium text-text">
                {localQuickTestPrompt.name}
              </div>
              {localQuickTestPrompt.systemText && (
                <div className="mt-2 text-xs text-text-muted">
                  <span className="font-medium">
                    {t("managePrompts.systemPrompt", {
                      defaultValue: "System prompt"
                    })}
                    :
                  </span>{" "}
                  <span className="line-clamp-2">{localQuickTestPrompt.systemText}</span>
                </div>
              )}
              {localQuickTestPrompt.userText && (
                <div className="mt-2 text-xs text-text-muted">
                  <span className="font-medium">
                    {t("managePrompts.quickPrompt", {
                      defaultValue: "Quick prompt"
                    })}
                    :
                  </span>{" "}
                  <span className="line-clamp-3">{localQuickTestPrompt.userText}</span>
                </div>
              )}
            </div>

            <div className="space-y-1">
              <label className="text-sm font-medium text-text">
                {t("managePrompts.quickTest.inputLabel", {
                  defaultValue: "Sample input"
                })}
              </label>
              <Input.TextArea
                value={localQuickTestInput}
                onChange={(event) => setLocalQuickTestInput(event.target.value)}
                placeholder={t("managePrompts.quickTest.inputPlaceholder", {
                  defaultValue:
                    "Optional input text. Used for {{text}} templates or appended to quick prompts."
                })}
                autoSize={{ minRows: 3, maxRows: 8 }}
                data-testid="prompts-local-quick-test-input"
              />
            </div>

            {localQuickTestOutput && (
              <div
                className="rounded border border-border bg-bg p-3"
                data-testid="prompts-local-quick-test-output"
              >
                <div className="mb-1 text-xs text-text-muted">
                  {localQuickTestRunInfo
                    ? t("managePrompts.quickTest.outputMeta", {
                        defaultValue: "Result ({{provider}} / {{model}})",
                        provider:
                          localQuickTestRunInfo.provider ||
                          t("managePrompts.quickTest.defaultProvider", {
                            defaultValue: "default provider"
                          }),
                        model: localQuickTestRunInfo.model
                      })
                    : t("managePrompts.quickTest.outputTitle", {
                        defaultValue: "Result"
                      })}
                </div>
                <pre className="whitespace-pre-wrap break-words text-sm text-text">
                  {localQuickTestOutput}
                </pre>
              </div>
            )}
          </div>
        )}
      </Modal>

      <Modal
        title={t("managePrompts.collections.create", {
          defaultValue: "New collection"
        })}
        open={collections.createCollectionModalOpen}
        onCancel={() => {
          collections.setCreateCollectionModalOpen(false)
          collections.setNewCollectionName("")
          collections.setNewCollectionDescription("")
        }}
        onOk={() =>
          collections.createPromptCollectionMutation({
            name: collections.newCollectionName,
            description: collections.newCollectionDescription
          })
        }
        okText={t("common:create", { defaultValue: "Create" })}
        cancelText={t("common:cancel", { defaultValue: "Cancel" })}
        okButtonProps={{
          loading: collections.isCreatingPromptCollection,
          disabled: collections.newCollectionName.trim().length === 0
        }}
        data-testid="prompts-collection-create-modal"
      >
        <div className="space-y-3">
          <Input
            value={collections.newCollectionName}
            onChange={(event) => collections.setNewCollectionName(event.target.value)}
            placeholder={t("managePrompts.collections.namePlaceholder", {
              defaultValue: "Collection name"
            })}
            data-testid="prompts-collection-name-input"
          />
          <Input.TextArea
            value={collections.newCollectionDescription}
            onChange={(event) => collections.setNewCollectionDescription(event.target.value)}
            placeholder={t("managePrompts.collections.descriptionPlaceholder", {
              defaultValue: "Description (optional)"
            })}
            autoSize={{ minRows: 2, maxRows: 4 }}
            data-testid="prompts-collection-description-input"
          />
        </div>
      </Modal>

      <Modal
        title={t("managePrompts.shortcuts.title", {
          defaultValue: "Keyboard shortcuts"
        })}
        open={shortcutsHelpOpen}
        onCancel={() => setShortcutsHelpOpen(false)}
        footer={null}
        data-testid="prompts-shortcuts-modal"
      >
        <p className="text-sm text-text-muted">
          {t("managePrompts.shortcuts.description", {
            defaultValue:
              "Shortcuts are available when focus is not inside an input field."
          })}
        </p>
        <div className="mt-3 space-y-2 text-sm">
          <div className="flex items-center justify-between rounded border border-border p-2">
            <span>
              {t("managePrompts.shortcuts.newPrompt", {
                defaultValue: "Create new prompt"
              })}
            </span>
            <kbd className="rounded border border-border px-1.5 py-0.5 text-xs">N</kbd>
          </div>
          <div className="flex items-center justify-between rounded border border-border p-2">
            <span>
              {t("managePrompts.shortcuts.focusSearch", {
                defaultValue: "Focus search"
              })}
            </span>
            <kbd className="rounded border border-border px-1.5 py-0.5 text-xs">/</kbd>
          </div>
          <div className="flex items-center justify-between rounded border border-border p-2">
            <span>
              {t("managePrompts.shortcuts.closeDrawer", {
                defaultValue: "Close drawer / modal"
              })}
            </span>
            <kbd className="rounded border border-border px-1.5 py-0.5 text-xs">Esc</kbd>
          </div>
          <div className="flex items-center justify-between rounded border border-border p-2">
            <span>
              {t("managePrompts.shortcuts.openHelp", {
                defaultValue: "Open shortcut help"
              })}
            </span>
            <kbd className="rounded border border-border px-1.5 py-0.5 text-xs">?</kbd>
          </div>
        </div>
      </Modal>

      <Modal
        title={t("option:promptInsert.confirmTitle", {
          defaultValue: "Use prompt in chat?"
        })}
        open={!!insertPrompt}
        onCancel={() => setInsertPrompt(null)}
        footer={null}
        width={520}>
        <div className="space-y-3">
          {/* System option */}
          {insertPrompt?.systemText && (
            <button
              type="button"
              onClick={() => {
                void handleInsertChoice("system")
              }}
              data-testid="prompt-insert-system"
              className="w-full text-left p-4 rounded-lg border border-border hover:border-primary hover:bg-primary/5 transition-colors">
              <div className="flex items-center gap-2 mb-2">
                <Computer className="size-5 text-warn" />
                <span className="font-medium">
                  {t("option:promptInsert.useAsSystem", {
                    defaultValue: "Use as System Instruction"
                  })}
                </span>
              </div>
              <p className="text-xs text-text-muted mb-2">
                {t("option:promptInsert.systemDescription", {
                  defaultValue: "Sets the AI's behavior and persona for the conversation."
                })}
              </p>
              <div className="bg-surface2 rounded p-2 text-xs line-clamp-3 font-mono text-text-muted">
                {insertPrompt.systemText}
              </div>
            </button>
          )}

          {/* Quick/User option */}
          {insertPrompt?.userText && (
            <button
              type="button"
              onClick={() => {
                void handleInsertChoice("quick")
              }}
              data-testid="prompt-insert-quick"
              className="w-full text-left p-4 rounded-lg border border-border hover:border-primary hover:bg-primary/5 transition-colors">
              <div className="flex items-center gap-2 mb-2">
                <Zap className="size-5 text-primary" />
                <span className="font-medium">
                  {t("option:promptInsert.useAsTemplate", {
                    defaultValue: "Insert as Message Template"
                  })}
                </span>
              </div>
              <p className="text-xs text-text-muted mb-2">
                {t("option:promptInsert.templateDescription", {
                  defaultValue: "Adds this text to your message composer."
                })}
              </p>
              <div className="bg-surface2 rounded p-2 text-xs line-clamp-3 font-mono text-text-muted">
                {insertPrompt.userText}
              </div>
            </button>
          )}

          {/* Use Both option - shown when prompt has both system and user */}
          {insertPrompt?.systemText && insertPrompt?.userText && (
            <button
              type="button"
              onClick={() => {
                void handleInsertChoice("both")
              }}
              data-testid="prompt-insert-both"
              className="w-full text-left p-4 rounded-lg border-2 border-primary/50 bg-primary/5 hover:border-primary hover:bg-primary/10 transition-colors">
              <div className="flex items-center gap-2 mb-2">
                <Layers className="size-5 text-primary" />
                <span className="font-medium text-primary">
                  {t("option:promptInsert.useBoth", {
                    defaultValue: "Use Both (Recommended)"
                  })}
                </span>
              </div>
              <p className="text-xs text-text-muted mb-2">
                {t("option:promptInsert.bothDescription", {
                  defaultValue: "Sets the system instruction AND inserts the message template. Best for prompts designed to work together."
                })}
              </p>
            </button>
          )}
        </div>
      </Modal>

      {/* Project Selector for Push to Server */}
      {renderProjectSelector()}

      {renderConflictResolutionModal()}
    </div>
  )
}
