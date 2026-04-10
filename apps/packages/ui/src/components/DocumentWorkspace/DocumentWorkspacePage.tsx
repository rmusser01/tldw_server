import React, { useState, useEffect, useCallback, useRef, Suspense } from "react"
import { useTranslation } from "react-i18next"
import { useSearchParams } from "react-router-dom"
import { useStorage } from "@plasmohq/storage/hook"
import { Alert, Drawer, Dropdown, Modal, notification, Tabs, Tooltip } from "antd"
import {
  FileText,
  MessageSquare,
  PanelLeftClose,
  PanelLeftOpen,
  PanelRightClose,
  PanelRightOpen,
  Lightbulb,
  Image,
  List,
  Info,
  BookOpen,
  Highlighter,
  Quote,
  HelpCircle,
  Keyboard,
  Plus,
  ChevronDown,
  CircleHelp
} from "lucide-react"
import { useDocumentWorkspaceStore } from "@/store/document-workspace"
import { useMobile, useTablet } from "@/hooks/useMediaQuery"
import { useAntdMessage } from "@/hooks/useAntdMessage"
import { bgRequest } from "@/services/background-proxy"
import { tldwClient } from "@/services/tldw"
import type { SidebarTab, RightPanelTab } from "./types"
import { DocumentWorkspaceErrorBoundary } from "./DocumentWorkspaceErrorBoundary"
import { DocumentShortcutsModal } from "./DocumentShortcutsModal"
import { DocumentTabBar } from "./DocumentTabBar"
import { SyncStatusIndicator } from "./SyncStatusIndicator"
import { getDocumentMimeType, inferDocumentTypeFromMedia } from "./document-utils"
import { TabIconLabel } from "./TabIconLabel"
import { WorkspaceTour, HighlightTip, MultiDocTip, resetTour, resetAllTips } from "./WorkspaceTips"
import {
  useAnnotations,
  useAnnotationSync,
  useAnnotationSyncOnClose,
  useReadingProgress,
  useReadingProgressAutoSave,
  useReadingProgressSaveOnClose,
  useResizablePanel,
} from "@/hooks/document-workspace"

const DocumentPickerModal = React.lazy(() => import("./DocumentPickerModal"))
const DocumentViewer = React.lazy(() =>
  import("./DocumentViewer").then((module) => ({
    default: module.DocumentViewer
  }))
)
const QuickInsightsTab = React.lazy(() =>
  import("./LeftSidebar/QuickInsightsTab").then((module) => ({
    default: module.QuickInsightsTab
  }))
)
const PagesTab = React.lazy(() =>
  import("./LeftSidebar/PagesTab").then((module) => ({
    default: module.PagesTab
  }))
)
const TableOfContentsTab = React.lazy(() =>
  import("./LeftSidebar/TableOfContentsTab").then((module) => ({
    default: module.TableOfContentsTab
  }))
)
const DocumentInfoTab = React.lazy(() =>
  import("./LeftSidebar/DocumentInfoTab").then((module) => ({
    default: module.DocumentInfoTab
  }))
)
const ReferencesTab = React.lazy(() =>
  import("./LeftSidebar/ReferencesTab").then((module) => ({
    default: module.ReferencesTab
  }))
)
const DocumentChat = React.lazy(() =>
  import("./RightPanel/DocumentChat").then((module) => ({
    default: module.DocumentChat
  }))
)
const AnnotationsPanel = React.lazy(() =>
  import("./RightPanel/AnnotationsPanel").then((module) => ({
    default: module.AnnotationsPanel
  }))
)
const CitationPanel = React.lazy(() =>
  import("./RightPanel/CitationPanel").then((module) => ({
    default: module.CitationPanel
  }))
)
const QuizPanel = React.lazy(() =>
  import("./RightPanel/QuizPanel").then((module) => ({
    default: module.QuizPanel
  }))
)
const tabPanelFallback = <div className="h-full min-h-0" />
const viewerPanelFallback = (
  <div className="flex h-full items-center justify-center">
    <div className="text-xs text-text-muted">Loading...</div>
  </div>
)

type ErrorWithStatus = {
  status: number
}

const isErrorWithStatus = (err: unknown): err is ErrorWithStatus => {
  if (!err || typeof err !== "object") {
    return false
  }
  return (
    "status" in err && typeof (err as { status?: unknown }).status === "number"
  )
}

/**
 * Left sidebar tab content for the document workspace layout.
 */
const LeftSidebarContent: React.FC = () => {
  const { t } = useTranslation(["option", "common"])
  const activeSidebarTab = useDocumentWorkspaceStore((s) => s.activeSidebarTab)
  const setActiveSidebarTab = useDocumentWorkspaceStore(
    (s) => s.setActiveSidebarTab
  )
  const activeDocumentType = useDocumentWorkspaceStore((s) => s.activeDocumentType)

  // Adaptive: hide Pages tab for EPUB (thumbnails not supported)
  const hiddenTabs: SidebarTab[] = activeDocumentType === "epub" ? ["pages"] : []
  const primaryKeys: SidebarTab[] = ["toc", "insights", "info"]
  const secondaryKeys: SidebarTab[] = (["pages", "references"] as SidebarTab[]).filter(
    (k) => !hiddenTabs.includes(k)
  )

  const renderSidebarTab = (tab: SidebarTab) => {
    switch (tab) {
      case "insights":
        return <QuickInsightsTab />
      case "pages":
        return <PagesTab />
      case "toc":
        return <TableOfContentsTab />
      case "info":
        return <DocumentInfoTab />
      case "references":
        return <ReferencesTab />
      default:
        return null
    }
  }

  const allSidebarTabs: Array<{
    key: SidebarTab
    label: string
    icon: React.ReactNode
  }> = [
    {
      key: "toc",
      label: t("option:documentWorkspace.toc", "Contents"),
      icon: <List className="h-4 w-4" />
    },
    {
      key: "insights",
      label: t("option:documentWorkspace.insights", "Insights"),
      icon: <Lightbulb className="h-4 w-4" />
    },
    {
      key: "info",
      label: t("option:documentWorkspace.info", "Info"),
      icon: <Info className="h-4 w-4" />
    },
    {
      key: "pages",
      label: t("option:documentWorkspace.pages", "Pages"),
      icon: <Image className="h-4 w-4" />
    },
    {
      key: "references",
      label: t("option:documentWorkspace.references", "References"),
      icon: <BookOpen className="h-4 w-4" />
    }
  ]

  // Filter out tabs not applicable to current document type
  const availableTabs = allSidebarTabs.filter((tab) => !hiddenTabs.includes(tab.key))

  // If the active tab is a secondary tab, include it in the visible set
  const visibleTabs = availableTabs.filter(
    (tab) => primaryKeys.includes(tab.key) || tab.key === activeSidebarTab
  )
  const overflowTabs = availableTabs.filter(
    (tab) => secondaryKeys.includes(tab.key) && tab.key !== activeSidebarTab
  )

  const moreDropdown = overflowTabs.length > 0 ? (
    <Dropdown
      menu={{
        items: overflowTabs.map((tab) => ({
          key: tab.key,
          label: (
            <span className="flex items-center gap-2">
              {tab.icon}
              <span className="text-xs">{tab.label}</span>
            </span>
          ),
          onClick: () => setActiveSidebarTab(tab.key)
        }))
      }}
      trigger={["click"]}
    >
      <button className="flex items-center gap-0.5 px-1.5 py-1 text-xs text-text-muted hover:text-text rounded hover:bg-hover">
        {t("common:more", "More")}
        <ChevronDown className="h-3 w-3" />
      </button>
    </Dropdown>
  ) : null

  return (
    <div className="flex h-full min-h-0 flex-col">
      <Tabs
        activeKey={activeSidebarTab}
        onChange={(key) => setActiveSidebarTab(key as SidebarTab)}
        items={visibleTabs.map((tab) => ({
          key: tab.key,
          label: <TabIconLabel label={tab.label} icon={tab.icon} />,
          children: (
            <div className="h-full min-h-0 overflow-auto overscroll-contain">
              {activeSidebarTab === tab.key ? (
                <Suspense fallback={tabPanelFallback}>
                  {renderSidebarTab(tab.key)}
                </Suspense>
              ) : null}
            </div>
          )
        }))}
        size="small"
        tabBarExtraContent={moreDropdown}
        className="flex-1 min-h-0 [&_.ant-tabs-content]:h-full [&_.ant-tabs-content]:min-h-0 [&_.ant-tabs-content-holder]:flex-1 [&_.ant-tabs-content-holder]:min-h-0 [&_.ant-tabs-tabpane]:h-full [&_.ant-tabs-tabpane]:min-h-0"
        tabBarStyle={{ marginBottom: 0, paddingLeft: 8, paddingRight: 8 }}
      />
    </div>
  )
}

/**
 * Right panel tab content for chat, notes, citations, and quizzes.
 */
const RIGHT_PRIMARY_KEYS: RightPanelTab[] = ["chat", "annotations"]
const RIGHT_SECONDARY_KEYS: RightPanelTab[] = ["citations", "quiz"]

const RightPanelContent: React.FC = () => {
  const { t } = useTranslation(["option", "common"])
  const activeRightTab = useDocumentWorkspaceStore((s) => s.activeRightTab)
  const setActiveRightTab = useDocumentWorkspaceStore(
    (s) => s.setActiveRightTab
  )

  const renderRightTab = (tab: RightPanelTab) => {
    switch (tab) {
      case "chat":
        return <DocumentChat />
      case "annotations":
        return <AnnotationsPanel />
      case "citations":
        return <CitationPanel />
      case "quiz":
        return <QuizPanel />
      default:
        return null
    }
  }

  const allRightTabs: Array<{
    key: RightPanelTab
    label: string
    icon: React.ReactNode
  }> = [
    {
      key: "chat",
      label: t("option:documentWorkspace.chat", "Chat"),
      icon: <MessageSquare className="h-4 w-4" />
    },
    {
      key: "annotations",
      label: t("option:documentWorkspace.annotations", "Highlights & Notes"),
      icon: <Highlighter className="h-4 w-4" />
    },
    {
      key: "citations",
      label: t("option:documentWorkspace.citations", "Cite"),
      icon: <Quote className="h-4 w-4" />
    },
    {
      key: "quiz",
      label: t("option:documentWorkspace.quiz", "Quiz"),
      icon: <HelpCircle className="h-4 w-4" />
    }
  ]

  // If the active tab is a secondary tab, include it in the visible set
  const visibleRightTabs = allRightTabs.filter(
    (tab) => RIGHT_PRIMARY_KEYS.includes(tab.key) || tab.key === activeRightTab
  )
  const overflowRightTabs = allRightTabs.filter(
    (tab) => RIGHT_SECONDARY_KEYS.includes(tab.key) && tab.key !== activeRightTab
  )

  const moreDropdown = overflowRightTabs.length > 0 ? (
    <Dropdown
      menu={{
        items: overflowRightTabs.map((tab) => ({
          key: tab.key,
          label: (
            <span className="flex items-center gap-2">
              {tab.icon}
              <span className="text-xs">{tab.label}</span>
            </span>
          ),
          onClick: () => setActiveRightTab(tab.key)
        }))
      }}
      trigger={["click"]}
    >
      <button className="flex items-center gap-0.5 px-1.5 py-1 text-xs text-text-muted hover:text-text rounded hover:bg-hover">
        {t("common:more", "More")}
        <ChevronDown className="h-3 w-3" />
      </button>
    </Dropdown>
  ) : null

  return (
    <div className="flex h-full flex-col">
      <Tabs
        activeKey={activeRightTab}
        onChange={(key) => setActiveRightTab(key as RightPanelTab)}
        items={visibleRightTabs.map((tab) => ({
          key: tab.key,
          label: <TabIconLabel label={tab.label} icon={tab.icon} />,
          children:
            activeRightTab === tab.key ? (
              <Suspense fallback={tabPanelFallback}>
                {renderRightTab(tab.key)}
              </Suspense>
            ) : null
        }))}
        size="small"
        tabBarExtraContent={moreDropdown}
        className="flex-1 [&_.ant-tabs-content]:h-full [&_.ant-tabs-content-holder]:flex-1 [&_.ant-tabs-tabpane]:h-full"
        tabBarStyle={{ marginBottom: 0, paddingLeft: 8, paddingRight: 8 }}
      />
    </div>
  )
}

/**
 * Header bar with pane toggles, status, and quick actions.
 */
const WorkspaceHeader: React.FC<{
  leftPaneOpen: boolean
  rightPaneOpen: boolean
  onToggleLeftPane: () => void
  onToggleRightPane: () => void
  onShowShortcuts: () => void
  onOpenPicker: (tab: "library" | "upload") => void
  onRetrySync?: () => void
  hideToggles?: boolean
}> = ({
  leftPaneOpen,
  rightPaneOpen,
  onToggleLeftPane,
  onToggleRightPane,
  onShowShortcuts,
  onOpenPicker,
  onRetrySync,
  hideToggles
}) => {
  const { t } = useTranslation(["option", "common"])
  const activeDocument = useDocumentWorkspaceStore((s) => {
    const doc = s.openDocuments.find((d) => d.id === s.activeDocumentId)
    return doc
  })
  const leftPaneToggleLabel = leftPaneOpen
    ? t("option:documentWorkspace.collapseSidebar", "Collapse sidebar")
    : t("option:documentWorkspace.expandSidebar", "Expand sidebar")
  const rightPaneToggleLabel = rightPaneOpen
    ? t("option:documentWorkspace.collapseChatPanel", "Collapse chat panel")
    : t("option:documentWorkspace.expandChatPanel", "Expand chat panel")

  return (
    <header className="flex h-12 shrink-0 items-center justify-between border-b border-border bg-surface px-4">
      <div className="flex items-center gap-3">
        {!hideToggles && (
          <Tooltip title={leftPaneToggleLabel}>
            <button
              onClick={onToggleLeftPane}
              className="rounded p-1.5 hover:bg-hover"
              data-testid="document-workspace-toggle-left"
              aria-label={leftPaneToggleLabel}
              title={leftPaneToggleLabel}
            >
              {leftPaneOpen ? (
                <PanelLeftClose className="h-5 w-5" />
              ) : (
                <PanelLeftOpen className="h-5 w-5" />
              )}
            </button>
          </Tooltip>
        )}
        <div className="flex items-center gap-2">
          <FileText className="h-5 w-5 text-primary" />
          <h1 className="text-lg font-medium">
            {activeDocument?.title ||
              t("option:documentWorkspace.title", "Document Workspace")}
          </h1>
        </div>
      </div>

      <div className="flex items-center gap-2">
        {/* Sync status indicator */}
        <SyncStatusIndicator onRetry={onRetrySync} />

        <Tooltip title={t("option:documentWorkspace.openDocument", "Open document")}>
          <button
            onClick={() => onOpenPicker("library")}
            className="rounded p-1.5 hover:bg-hover text-text-subtle hover:text-text"
            aria-label={t("option:documentWorkspace.openDocument", "Open document")}
          >
            <Plus className="h-5 w-5" />
          </button>
        </Tooltip>

        <Dropdown
          menu={{
            items: [
              {
                key: "shortcuts",
                label: (
                  <span className="flex items-center gap-2">
                    <Keyboard className="h-4 w-4" />
                    {t("option:documentWorkspace.keyboardShortcuts", "Keyboard shortcuts")}
                    <span className="ml-auto text-xs text-text-muted">?</span>
                  </span>
                ),
                onClick: onShowShortcuts
              },
              {
                key: "tips",
                label: (
                  <span className="flex items-center gap-2">
                    <Lightbulb className="h-4 w-4" />
                    {t("option:documentWorkspace.showFeatureTips", "Show feature tips")}
                  </span>
                ),
                onClick: () => {
                  resetAllTips()
                  window.location.reload()
                }
              },
              {
                key: "tour",
                label: (
                  <span className="flex items-center gap-2">
                    <CircleHelp className="h-4 w-4" />
                    {t("option:documentWorkspace.takeTour", "Take the tour again")}
                  </span>
                ),
                onClick: () => {
                  resetTour()
                  window.location.reload()
                }
              }
            ]
          }}
          trigger={["click"]}
        >
          <button
            className="flex items-center gap-1 rounded px-1.5 py-1 hover:bg-hover text-text-subtle hover:text-text"
            aria-label={t("option:documentWorkspace.help", "Help")}
          >
            <CircleHelp className="h-4 w-4" />
            <span className="text-xs hidden sm:inline">{t("option:documentWorkspace.help", "Help")}</span>
            <ChevronDown className="h-3 w-3" />
          </button>
        </Dropdown>
        {!hideToggles && (
          <Tooltip title={rightPaneToggleLabel}>
            <button
              onClick={onToggleRightPane}
              className="rounded p-1.5 hover:bg-hover"
              data-testid="document-workspace-toggle-right"
              aria-label={rightPaneToggleLabel}
              title={rightPaneToggleLabel}
            >
              {rightPaneOpen ? (
                <PanelRightClose className="h-5 w-5" />
              ) : (
                <PanelRightOpen className="h-5 w-5" />
              )}
            </button>
          </Tooltip>
        )}
      </div>
    </header>
  )
}

const STORAGE_KEY_LEFT_PANE = "document-workspace-left-pane"
const STORAGE_KEY_RIGHT_PANE = "document-workspace-right-pane"
const DOCUMENT_FILE_TIMEOUT_MS = 30 * 1000

/**
 * DocumentWorkspacePage - Three-panel document reader interface
 *
 * Layout at different breakpoints:
 * - lg+ (1024px+): Full three-pane layout
 * - md (768-1023px): Viewer main, sidebar/chat as slide-out drawers
 * - sm (<768px): Bottom tab navigation between panes
 */
export const DocumentWorkspacePage: React.FC = () => {
  const { t } = useTranslation(["option", "common"])
  const isMobile = useMobile()
  const isTablet = useTablet()
  const message = useAntdMessage()

  // Get active document for hooks
  const activeDocumentId = useDocumentWorkspaceStore((s) => s.activeDocumentId)
  const openDocuments = useDocumentWorkspaceStore((s) => s.openDocuments)
  const openDocument = useDocumentWorkspaceStore((s) => s.openDocument)
  const annotationsHealth = useDocumentWorkspaceStore((s) => s.annotationsHealth)
  const progressHealth = useDocumentWorkspaceStore((s) => s.progressHealth)

  // Initialize annotation fetching and sync
  useAnnotations(activeDocumentId)
  const { retrySync, forceSync } = useAnnotationSync(activeDocumentId)
  useAnnotationSyncOnClose(activeDocumentId, forceSync)

  // Initialize reading progress loading and auto-save
  useReadingProgress(activeDocumentId)
  const { forceSave } = useReadingProgressAutoSave(activeDocumentId, 5000) // Save every 5 seconds
  useReadingProgressSaveOnClose(activeDocumentId, forceSave)

  // Resizable panel widths
  const leftPanel = useResizablePanel({ key: "left", defaultWidth: 288, min: 200, max: 400 })
  const rightPanel = useResizablePanel({ key: "right", defaultWidth: 320, min: 240, max: 480, edge: "left" })

  // Pane state with persistence
  const [leftPaneOpen, setLeftPaneOpen] = useStorage(STORAGE_KEY_LEFT_PANE, true)
  const [rightPaneOpen, setRightPaneOpen] = useStorage(
    STORAGE_KEY_RIGHT_PANE,
    true
  )

  // Mobile drawer state
  const [leftDrawerOpen, setLeftDrawerOpen] = useState(false)
  const [rightDrawerOpen, setRightDrawerOpen] = useState(false)

  // Mobile tab state
  const [activeTab, setActiveTab] = useState<"sidebar" | "viewer" | "chat">(
    "viewer"
  )

  // Keyboard shortcuts modal state
  const [shortcutsModalOpen, setShortcutsModalOpen] = useState(false)
  const [pickerOpen, setPickerOpen] = useState(false)
  const [pickerTab, setPickerTab] = useState<"library" | "upload">("library")
  const [loadingDocumentId, setLoadingDocumentId] = useState<number | null>(null)
  const blobUrlMapRef = useRef<Map<number, string>>(new Map())
  const openDocumentRequestRef = useRef(0)

  const handleShowShortcuts = useCallback(() => {
    setShortcutsModalOpen(true)
  }, [])

  const handleCloseShortcuts = useCallback(() => {
    setShortcutsModalOpen(false)
  }, [])

  const handleOpenPicker = useCallback((tab: "library" | "upload") => {
    setPickerTab(tab)
    setPickerOpen(true)
  }, [])

  const handleClosePicker = useCallback(() => {
    setPickerOpen(false)
  }, [])

  // Close document with confirmation when annotations are pending
  const closeDocument = useDocumentWorkspaceStore((s) => s.closeDocument)
  const undoCloseDocument = useDocumentWorkspaceStore((s) => s.undoCloseDocument)
  const annotationSyncStatus = useDocumentWorkspaceStore((s) => s.annotationSyncStatus)

  const doClose = useCallback((id: number) => {
    const doc = openDocuments.find((d) => d.id === id)
    closeDocument(id)
    if (doc) {
      const key = `undo-close-${id}`
      notification.info({
        key,
        message: t("option:documentWorkspace.documentClosed", "Document closed"),
        description: doc.title,
        btn: (
          <button
            className="text-xs text-primary hover:underline font-medium"
            onClick={() => {
              undoCloseDocument()
              notification.destroy(key)
            }}
          >
            {t("option:documentWorkspace.undo", "Undo")}
          </button>
        ),
        duration: 10,
        placement: "bottomRight"
      })
    }
  }, [closeDocument, undoCloseDocument, openDocuments, t])

  const handleCloseDocument = useCallback((id: number) => {
    if (annotationSyncStatus === "pending") {
      Modal.confirm({
        title: t("option:documentWorkspace.unsavedHighlights", "You have unsaved highlights"),
        content: t(
          "option:documentWorkspace.unsavedHighlightsDesc",
          "Some highlights and notes haven't been saved yet. What would you like to do?"
        ),
        okText: t("option:documentWorkspace.saveAndClose", "Save & close"),
        cancelText: t("option:documentWorkspace.closeWithoutSaving", "Close without saving"),
        closable: false,
        maskClosable: false,
        onOk: async () => {
          await forceSync()
          doClose(id)
        },
        onCancel: () => {
          doClose(id)
        }
      })
    } else {
      doClose(id)
    }
  }, [annotationSyncStatus, doClose, forceSync, t])

  // Workspace-level keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement
      const isInputField =
        target.tagName === "INPUT" ||
        target.tagName === "TEXTAREA" ||
        target.isContentEditable

      const mod = e.metaKey || e.ctrlKey

      // Shortcuts that work even in input fields
      if (mod) {
        // Cmd+[ → toggle left sidebar
        if (e.key === "[") {
          e.preventDefault()
          handleToggleLeftPane()
          return
        }
        // Cmd+] → toggle right panel
        if (e.key === "]") {
          e.preventDefault()
          handleToggleRightPane()
          return
        }
        // Cmd+/ → focus chat input
        if (e.key === "/") {
          e.preventDefault()
          if (isMobile) {
            setActiveTab("chat")
          } else {
            setRightPaneOpen(true)
          }
          // Focus the chat textarea after a tick
          setTimeout(() => {
            const textarea = document.querySelector<HTMLTextAreaElement>(
              '[aria-label*="Ask about this document"]'
            )
            textarea?.focus()
          }, 100)
          return
        }
      }

      if (isInputField) return

      // "?" key (Shift + /)
      if (e.key === "?" || (e.shiftKey && e.key === "/")) {
        e.preventDefault()
        setShortcutsModalOpen(true)
      }
    }

    window.addEventListener("keydown", handleKeyDown)
    return () => window.removeEventListener("keydown", handleKeyDown)
  }, [activeDocumentId, isMobile])

  const handleToggleLeftPane = () => {
    if (isMobile || isTablet) {
      setLeftDrawerOpen(!leftDrawerOpen)
    } else {
      setLeftPaneOpen(!leftPaneOpen)
    }
  }

  const handleToggleRightPane = () => {
    if (isMobile || isTablet) {
      setRightDrawerOpen(!rightDrawerOpen)
    } else {
      setRightPaneOpen(!rightPaneOpen)
    }
  }

  const registerBlobUrl = useCallback((mediaId: number, url: string) => {
    const existing = blobUrlMapRef.current.get(mediaId)
    if (existing && existing !== url && existing.startsWith("blob:")) {
      URL.revokeObjectURL(existing)
    }
    blobUrlMapRef.current.set(mediaId, url)
  }, [])

  const recentlyClosed = useDocumentWorkspaceStore((s) => s.recentlyClosed)

  const cleanupRemovedBlobUrls = useCallback(() => {
    const openIds = new Set(openDocuments.map((doc) => doc.id))
    const closedIds = new Set(recentlyClosed.map((c) => c.doc.id))
    for (const [id, url] of blobUrlMapRef.current.entries()) {
      if (!openIds.has(id) && !closedIds.has(id)) {
        if (url.startsWith("blob:")) {
          URL.revokeObjectURL(url)
        }
        blobUrlMapRef.current.delete(id)
      }
    }
  }, [openDocuments, recentlyClosed])

  useEffect(() => {
    cleanupRemovedBlobUrls()
  }, [cleanupRemovedBlobUrls])

  useEffect(() => {
    return () => {
      for (const url of blobUrlMapRef.current.values()) {
        if (url.startsWith("blob:")) {
          URL.revokeObjectURL(url)
        }
      }
      blobUrlMapRef.current.clear()
    }
  }, [])

  const openDocumentById = useCallback(
    async (mediaId: number, docTypeHint?: "pdf" | "epub" | null) => {
      const requestId = ++openDocumentRequestRef.current
      setLoadingDocumentId(mediaId)
      try {
        const details = await tldwClient.getMediaDetails(mediaId, {
          include_content: false,
          include_versions: false
        })
        if (openDocumentRequestRef.current !== requestId) {
          return
        }

        const metadataSources = [
          details?.processing?.safe_metadata,
          details?.content?.metadata,
          details?.metadata
        ]
          .map((source) =>
            typeof source === "object" && source !== null
              ? (source as Record<string, unknown>)
              : null
          )
          .filter((source): source is Record<string, unknown> => source !== null)

        const getMetadataValue = (keys: string[]): string | undefined => {
          for (const source of metadataSources) {
            for (const key of keys) {
              if (key in source) {
                const value = source[key]
                if (value !== undefined && value !== null && String(value).trim() !== "") {
                  return String(value)
                }
              }
            }
          }
          return undefined
        }

        const filename =
          getMetadataValue([
            "original_filename",
            "file_name",
            "filename",
            "fileName",
            "FileName",
            "File_Name"
          ]) || details?.filename

        const docType = docTypeHint ?? inferDocumentTypeFromMedia(details?.source?.type || details?.type, filename)

        if (!docType) {
          message.error(
            t(
              "option:documentWorkspace.unsupportedType",
              "This media type isn’t supported in the document workspace."
            )
          )
          return
        }

        let data: ArrayBuffer | Blob
        try {
          data = await bgRequest<ArrayBuffer>({
            path: `/api/v1/media/${mediaId}/file`,
            method: "GET",
            responseType: "arrayBuffer",
            timeoutMs: DOCUMENT_FILE_TIMEOUT_MS
          })
        } catch (err: unknown) {
          if (openDocumentRequestRef.current !== requestId) {
            return
          }
          const status = isErrorWithStatus(err) ? err.status : undefined
          if (status === 404) {
            message.error(
              t(
                "option:documentWorkspace.missingFile",
                "This document's original file was not preserved when it was imported. To view it in the workspace, re-upload it using the Upload tab above, or re-import it \u2014 newer imports automatically preserve document files."
              )
            )
            return
          }
          if (status === 0) {
            message.error(
              t(
                "option:documentWorkspace.fileTimeout",
                "Document download timed out. Please try again."
              )
            )
            return
          }
          if (status === 401 || status === 403) {
            message.error(
              t(
                "option:documentWorkspace.fileUnauthorized",
                "You don't have permission to access this document."
              )
            )
            return
          }
          throw err
        }
        if (openDocumentRequestRef.current !== requestId) {
          return
        }

        const blob =
          data instanceof Blob
            ? data
            : new Blob([data], { type: getDocumentMimeType(docType) })
        const url = URL.createObjectURL(blob)
        registerBlobUrl(mediaId, url)

        const titleFromMetadata = getMetadataValue([
          "title",
          "Title",
          "document_title",
          "DocumentTitle",
          "dc:title"
        ])

        openDocument({
          id: mediaId,
          title:
            details?.source?.title ||
            details?.title ||
            titleFromMetadata ||
            `Media ${mediaId}`,
          type: docType,
          url
        })
      } catch (err) {
        if (openDocumentRequestRef.current !== requestId) {
          return
        }
        message.error(
          err instanceof Error
            ? err.message
            : t(
                "option:documentWorkspace.openFailed",
                "Failed to open document"
              )
        )
      } finally {
        if (openDocumentRequestRef.current === requestId) {
          setLoadingDocumentId(null)
        }
      }
    },
    [message, openDocument, registerBlobUrl, setLoadingDocumentId, t]
  )

  // Handle ?open={mediaId} URL parameter for auto-opening documents
  const [searchParams, setSearchParams] = useSearchParams()
  const autoOpenId = searchParams.get("open")
  const autoOpenHandledRef = useRef(false)

  useEffect(() => {
    if (!autoOpenId) {
      autoOpenHandledRef.current = false
      return
    }
    if (autoOpenHandledRef.current) return
    // Wait until we are not already loading another document
    if (loadingDocumentId !== null) return
    const mediaId = Number(autoOpenId)
    if (!Number.isFinite(mediaId) || mediaId <= 0) return

    // Check if this document is already open
    const alreadyOpen = openDocuments.some((d) => d.id === mediaId)
    if (alreadyOpen) {
      // Just clear the param
      autoOpenHandledRef.current = true
      setSearchParams((prev) => {
        prev.delete("open")
        return prev
      }, { replace: true })
      return
    }

    autoOpenHandledRef.current = true
    // Remove the param from the URL first so it doesn't re-trigger
    setSearchParams((prev) => {
      prev.delete("open")
      return prev
    }, { replace: true })
    void openDocumentById(mediaId)
  }, [autoOpenId, loadingDocumentId, openDocuments, openDocumentById, setSearchParams])

  const loadingAlert =
    loadingDocumentId !== null ? (
      <div className="px-4 pt-2">
        <Alert
          type="info"
          showIcon
          title={t(
            "option:documentWorkspace.loadingDocument",
            "Loading document..."
          )}
          description={t(
            "option:documentWorkspace.loadingDocumentHint",
            "Fetching the document file. This can take a moment for large files."
          )}
        />
      </div>
    ) : null

  const healthIssues: string[] = []
  if (annotationsHealth === "error") {
    healthIssues.push(
      t(
        "option:documentWorkspace.annotationsUnavailable",
        "Annotations storage is unavailable on the server."
      )
    )
  }
  if (progressHealth === "error") {
    healthIssues.push(
      t(
        "option:documentWorkspace.progressUnavailable",
        "Reading progress storage is unavailable on the server."
      )
    )
  }

  const healthAlert =
    healthIssues.length > 0 ? (
      <div className="px-4 pt-2">
        <Alert
          type="warning"
          showIcon
          title={t(
            "option:documentWorkspace.healthWarningTitle",
            "Document workspace storage unavailable"
          )}
          description={
            <div className="space-y-1">
              <ul className="list-disc pl-5">
                {healthIssues.map((issue, index) => (
                  <li key={`${index}-${issue}`}>{issue}</li>
                ))}
              </ul>
              <div className="text-xs text-text-muted">
                {t(
                  "option:documentWorkspace.healthWarningHint",
                  "Some workspace features are temporarily unavailable. This usually resolves after restarting the server. If this persists, contact your administrator."
                )}
              </div>
            </div>
          }
        />
      </div>
    ) : null

  // Mobile layout (< 768px): Bottom navigation bar
  if (isMobile) {
    const mobileNavItems = [
      { key: "sidebar" as const, icon: <List className="h-5 w-5" />, label: t("option:documentWorkspace.sidebar", "Sidebar") },
      { key: "viewer" as const, icon: <FileText className="h-5 w-5" />, label: t("option:documentWorkspace.document", "Document") },
      { key: "chat" as const, icon: <MessageSquare className="h-5 w-5" />, label: t("option:documentWorkspace.chat", "Chat") }
    ]

    const mobileContent = {
      sidebar: <LeftSidebarContent />,
       viewer: (
        <Suspense fallback={viewerPanelFallback}>
          <DocumentViewer
            loadingDocumentId={loadingDocumentId}
            onOpenLibrary={() => handleOpenPicker("library")}
            onOpenUpload={() => handleOpenPicker("upload")}
            onReloadDocument={openDocumentById}
          />
        </Suspense>
      ),
      chat: <RightPanelContent />
    }

    return (
      <DocumentWorkspaceErrorBoundary>
        <div className="flex h-full min-h-0 flex-col bg-bg text-text">
          <WorkspaceHeader
            leftPaneOpen={false}
            rightPaneOpen={false}
            onToggleLeftPane={handleToggleLeftPane}
            onToggleRightPane={handleToggleRightPane}
            onShowShortcuts={handleShowShortcuts}
            onOpenPicker={handleOpenPicker}
            onRetrySync={retrySync}
            hideToggles
          />
          <DocumentShortcutsModal
            open={shortcutsModalOpen}
            onClose={handleCloseShortcuts}
          />
          <Suspense fallback={null}>
            {pickerOpen && (
              <DocumentPickerModal
                open={pickerOpen}
                initialTab={pickerTab}
                onClose={handleClosePicker}
                onOpenDocument={openDocumentById}
              />
            )}
          </Suspense>
          {loadingAlert}
          {healthAlert}
          <WorkspaceTour />

          {/* Content area */}
          <div className="relative flex-1 min-h-0 overflow-hidden">
            <HighlightTip />
            {mobileContent[activeTab]}
          </div>

          {/* Fixed bottom navigation bar */}
          <nav className="flex h-12 shrink-0 items-stretch border-t border-border bg-surface" role="tablist">
            {mobileNavItems.map((item) => (
              <button
                key={item.key}
                role="tab"
                aria-selected={activeTab === item.key}
                onClick={() => setActiveTab(item.key)}
                className={`flex flex-1 flex-col items-center justify-center gap-0.5 transition-colors ${
                  activeTab === item.key
                    ? "text-primary"
                    : "text-text-muted hover:text-text"
                }`}
              >
                {item.icon}
                <span className="text-xs font-medium leading-none">{item.label}</span>
              </button>
            ))}
          </nav>
        </div>
      </DocumentWorkspaceErrorBoundary>
    )
  }

  // Tablet/Desktop layout
  return (
    <DocumentWorkspaceErrorBoundary>
      <div className="flex h-full min-h-0 flex-col bg-bg text-text">
        <WorkspaceHeader
          leftPaneOpen={!!leftPaneOpen}
          rightPaneOpen={!!rightPaneOpen}
          onToggleLeftPane={handleToggleLeftPane}
          onToggleRightPane={handleToggleRightPane}
          onShowShortcuts={handleShowShortcuts}
          onOpenPicker={handleOpenPicker}
          onRetrySync={retrySync}
        />
        <DocumentShortcutsModal
          open={shortcutsModalOpen}
          onClose={handleCloseShortcuts}
        />
        <Suspense fallback={null}>
          {pickerOpen && (
            <DocumentPickerModal
              open={pickerOpen}
              initialTab={pickerTab}
              onClose={handleClosePicker}
              onOpenDocument={openDocumentById}
            />
          )}
        </Suspense>
        {loadingAlert}
        {healthAlert}
        <WorkspaceTour />

        {/* Document tabs - shown when multiple documents are open */}
        <div className="relative">
          <DocumentTabBar onOpenPicker={() => handleOpenPicker("library")} onCloseDocument={handleCloseDocument} />
          <MultiDocTip />
        </div>

        <div className="flex min-h-0 flex-1">
          {/* Left pane - Sidebar (desktop) */}
          {leftPaneOpen && (
            <aside className="hidden h-full min-h-0 shrink-0 border-r border-border bg-surface lg:flex lg:flex-col" style={{ width: leftPanel.width }}>
              <LeftSidebarContent />
            </aside>
          )}
          {leftPaneOpen && (
            <div
              className="hidden lg:flex h-full w-1 cursor-col-resize items-center justify-center hover:bg-primary/30 active:bg-primary/50 transition-colors"
              onMouseDown={leftPanel.handleMouseDown}
              role="separator"
              aria-orientation="vertical"
              aria-label="Resize left sidebar"
            />
          )}

          {/* Left pane - Sidebar (tablet drawer) */}
          <Drawer
            title={
              <span className="flex items-center gap-2">
                <List className="h-4 w-4" />
                {t("option:documentWorkspace.sidebar", "Sidebar")}
              </span>
            }
            placement="left"
            onClose={() => setLeftDrawerOpen(false)}
            open={leftDrawerOpen}
            size={Math.min(360, typeof window !== "undefined" ? window.innerWidth * 0.85 : 360)}
            className="lg:hidden"
            styles={{ body: { padding: 0, height: "100%", display: "flex", flexDirection: "column" } }}
          >
            <LeftSidebarContent />
          </Drawer>

          {/* Center pane - Document Viewer */}
          <main className="relative flex min-h-0 min-w-0 flex-1 flex-col bg-bg">
            <HighlightTip />
            <Suspense fallback={viewerPanelFallback}>
              <DocumentViewer
                loadingDocumentId={loadingDocumentId}
                onOpenLibrary={() => handleOpenPicker("library")}
                onOpenUpload={() => handleOpenPicker("upload")}
                onReloadDocument={openDocumentById}
              />
            </Suspense>
          </main>

          {/* Right pane - Chat/Annotations (desktop) */}
          {rightPaneOpen && (
            <div
              className="hidden lg:flex h-full w-1 cursor-col-resize items-center justify-center hover:bg-primary/30 active:bg-primary/50 transition-colors"
              onMouseDown={rightPanel.handleMouseDown}
              role="separator"
              aria-orientation="vertical"
              aria-label="Resize right panel"
            />
          )}
          {rightPaneOpen && (
            <aside className="hidden shrink-0 border-l border-border bg-surface lg:flex lg:flex-col" style={{ width: rightPanel.width }}>
              <RightPanelContent />
            </aside>
          )}

          {/* Right pane - Chat/Annotations (tablet drawer) */}
          <Drawer
            title={
              <span className="flex items-center gap-2">
                <MessageSquare className="h-4 w-4" />
                {t("option:documentWorkspace.chat", "Chat")}
              </span>
            }
            placement="right"
            onClose={() => setRightDrawerOpen(false)}
            open={rightDrawerOpen}
            size={Math.min(360, typeof window !== "undefined" ? window.innerWidth * 0.85 : 360)}
            className="lg:hidden"
            styles={{ body: { padding: 0 } }}
          >
            <RightPanelContent />
          </Drawer>
        </div>
      </div>
    </DocumentWorkspaceErrorBoundary>
  )
}

export default DocumentWorkspacePage
