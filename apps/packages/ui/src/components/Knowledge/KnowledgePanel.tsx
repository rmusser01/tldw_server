import React from "react"
import { Modal } from "antd"
import { X } from "lucide-react"
import { useTranslation } from "react-i18next"
import type { UploadedFile } from "@/db/dexie/types"
import type { TabInfo } from "@/hooks/useTabMentions"
import type { RagPinnedResult } from "@/utils/rag-format"
import { formatRagResult } from "@/utils/rag-format"
import { KnowledgeTabs } from "./KnowledgeTabs"
import { QASearchTab } from "./QASearchTab"
import { FileSearchTab } from "./FileSearchTab"
import { SettingsTab } from "./SettingsTab"
import { ContextTab } from "./ContextTab"
import {
  useKnowledgeSettings,
  useKnowledgeSearch,
  toPinnedResult,
  withFullMediaTextIfAvailable,
  qaDocumentToRagResult,
  type RagResult
} from "./hooks"
import { useFileSearch } from "./hooks/useFileSearch"
import { useQASearch, type QADocument } from "./hooks/useQASearch"

/**
 * Tab identifiers for the 4-tab architecture.
 * "search" is kept as a backward-compat alias that maps to "qa-search".
 */
export type KnowledgeTab = "qa-search" | "file-search" | "settings" | "context" | "search"

/**
 * Props for KnowledgePanel - matches RagSearchBar props for compatibility
 */
export type KnowledgePanelProps = {
  onInsert: (text: string) => void
  onAsk: (text: string, options?: { ignorePinnedResults?: boolean }) => void
  isConnected?: boolean
  open?: boolean
  onOpenChange?: (nextOpen: boolean) => void
  /** Request a specific tab to open (paired with openTabRequestId for repeat requests). */
  openTab?: KnowledgeTab
  /** Increment to force re-applying openTab even if it didn't change. */
  openTabRequestId?: number
  autoFocus?: boolean
  showToggle?: boolean
  variant?: "card" | "embedded"
  currentMessage?: string
  showAttachedContext?: boolean
  attachedImage?: string
  attachedTabs?: TabInfo[]
  availableTabs?: TabInfo[]
  attachedFiles?: UploadedFile[]
  onRemoveImage?: () => void
  onRemoveTab?: (tabId: number) => void
  onAddTab?: (tab: TabInfo) => void
  onClearTabs?: () => void
  onRefreshTabs?: () => void
  onAddFile?: () => void
  onRemoveFile?: (fileId: string) => void
  onClearFiles?: () => void
  /** Whether file retrieval (RAG) is enabled */
  fileRetrievalEnabled?: boolean
  /** Callback to toggle file retrieval */
  onFileRetrievalChange?: (enabled: boolean) => void
}

const normalizeUrl = (value?: string) => value?.trim().toLowerCase()
const noop = () => {}

/**
 * KnowledgePanel - Main container for the 4-tab RAG interface
 *
 * Implements the 4-tab architecture:
 * - QA Search tab: Full RAG pipeline with generated answers and source chunks
 * - File Search tab: Media library search for document discovery and attachment
 * - Settings tab: All RAG settings organized in collapsible sections
 * - Context tab: Manage attached tabs, files, and pinned results
 */
const KnowledgePanelBase: React.FC<KnowledgePanelProps> = ({
  onInsert,
  onAsk,
  isConnected = true,
  open,
  onOpenChange,
  openTab,
  openTabRequestId,
  autoFocus = true,
  showToggle = true,
  variant = "card",
  currentMessage,
  showAttachedContext = false,
  attachedImage,
  attachedTabs = [],
  availableTabs = [],
  attachedFiles = [],
  onRemoveImage,
  onRemoveTab,
  onAddTab,
  onClearTabs,
  onRefreshTabs,
  onAddFile,
  onRemoveFile,
  onClearFiles,
  fileRetrievalEnabled = false,
  onFileRetrievalChange
}) => {
  const { t } = useTranslation(["sidepanel"])

  // Normalize "search" alias to "qa-search" for backward compat
  const normalizeTab = (tab: KnowledgeTab): KnowledgeTab =>
    tab === "search" ? "qa-search" : tab

  // Tab state - default to qa-search tab (or requested tab)
  const [activeTab, setActiveTab] = React.useState<KnowledgeTab>(
    normalizeTab(openTab ?? "qa-search")
  )
  const lastOpenTabRequestRef = React.useRef<number | null>(null)

  const handleTabChange = React.useCallback((tab: KnowledgeTab) => {
    setActiveTab(normalizeTab(tab))
  }, [])

  React.useEffect(() => {
    if (!openTab) return
    if (typeof openTabRequestId === "number") {
      if (openTabRequestId !== lastOpenTabRequestRef.current) {
        setActiveTab(normalizeTab(openTab))
        lastOpenTabRequestRef.current = openTabRequestId
      }
      return
    }
    setActiveTab(normalizeTab(openTab))
  }, [openTab, openTabRequestId])

  // Open/close state (controlled or uncontrolled)
  const [internalOpen, setInternalOpen] = React.useState(false)
  const isControlled = typeof open === "boolean"
  const isOpen = isControlled ? open : internalOpen
  const setOpenState = React.useCallback(
    (next: boolean) => {
      if (isControlled) {
        onOpenChange?.(next)
        return
      }
      setInternalOpen(next)
      onOpenChange?.(next)
    },
    [isControlled, onOpenChange]
  )

  // Preview modal state
  const [previewItem, setPreviewItem] = React.useState<RagPinnedResult | null>(
    null
  )

  // Settings state (from hook)
  const settings = useKnowledgeSettings(currentMessage)

  // Search state (from hook) — used for pinned results and legacy compatibility
  const search = useKnowledgeSearch({
    resolvedQuery: settings.resolvedQuery,
    draftSettings: settings.draftSettings,
    applySettings: settings.applySettings,
    onInsert,
    onAsk
  })

  // File search state (from hook)
  const fileSearch = useFileSearch({
    resolvedQuery: settings.resolvedQuery,
    draftSettings: settings.draftSettings,
    applySettings: settings.applySettings,
    onInsert,
    pinnedResults: search.pinnedResults,
    onPin: search.handlePin
  })

  // QA search state (from hook)
  const qaSearch = useQASearch({
    resolvedQuery: settings.resolvedQuery,
    draftSettings: settings.draftSettings,
    applySettings: settings.applySettings,
    onInsert,
    pinnedResults: search.pinnedResults,
    onPin: search.handlePin
  })

  // Handle preview action (shared across tabs)
  const handlePreview = React.useCallback((result: RagResult) => {
    setPreviewItem(toPinnedResult(result))
  }, [])

  const handlePreviewChunk = React.useCallback(
    (doc: QADocument) => {
      handlePreview(qaDocumentToRagResult(doc))
    },
    [handlePreview]
  )

  // Discard staged settings on close
  const previousOpen = React.useRef(isOpen)
  React.useEffect(() => {
    if (previousOpen.current && !isOpen) {
      settings.discardChanges()
    }
    previousOpen.current = isOpen
  }, [isOpen, settings.discardChanges])

  // Handle Ask with confirmation for pinned results
  const handleAsk = React.useCallback(
    (result: RagResult) => {
      const pinned = toPinnedResult(result)
      if (search.pinnedResults.length > 0) {
        Modal.confirm({
          title: t("sidepanel:rag.askConfirmTitle", "Ask about this item?"),
          content: t(
            "sidepanel:rag.askConfirmContent",
            "Saved results will be ignored for this Ask."
          ),
          okText: t("common:continue", "Continue"),
          cancelText: t("common:cancel", "Cancel"),
          onOk: () =>
            onAsk(formatRagResult(pinned, "markdown"), {
              ignorePinnedResults: true
            })
        })
        return
      }
      onAsk(formatRagResult(pinned, "markdown"), { ignorePinnedResults: true })
    },
    [onAsk, search.pinnedResults.length, t]
  )

  // Context item count for badge (image + tabs + files + pinned results)
  const contextCount = React.useMemo(() => {
    const seen = new Set<string>()

    if (attachedImage) {
      seen.add("image")
    }

    attachedTabs.forEach((tab) => {
      const normalized = normalizeUrl(tab.url)
      if (normalized) {
        seen.add(`url:${normalized}`)
      } else {
        seen.add(`tab:${tab.id}`)
      }
    })

    attachedFiles.forEach((file) => {
      seen.add(`file:${file.id}`)
    })

    search.pinnedResults.forEach((pin) => {
      const normalized = normalizeUrl(pin.url)
      if (normalized) {
        seen.add(`url:${normalized}`)
      } else {
        seen.add(`pin:${pin.id}`)
      }
    })

    return seen.size
  }, [attachedImage, attachedTabs, attachedFiles, search.pinnedResults])

  // Toggle handler for external events
  React.useEffect(() => {
    const handler = () => setOpenState(!isOpen)
    window.addEventListener("tldw:toggle-rag", handler)
    return () => window.removeEventListener("tldw:toggle-rag", handler)
  }, [isOpen, setOpenState])

  const handleApplyAndSearch = React.useCallback(() => {
    if (activeTab === "file-search") {
      void fileSearch.runSearch({ applyFirst: true })
      return
    }

    void qaSearch.runQASearch({ applyFirst: true })
    if (activeTab !== "qa-search") {
      handleTabChange("qa-search")
    }
  }, [activeTab, fileSearch.runSearch, handleTabChange, qaSearch.runQASearch])

  // Don't render if closed
  if (!isOpen) {
    if (showToggle) {
      return (
        <button
          onClick={() => setOpenState(true)}
          className="flex items-center gap-2 px-3 py-1.5 text-sm text-text-muted hover:text-text transition-colors"
        >
          {t("sidepanel:rag.show", "Show Knowledge Search")}
        </button>
      )
    }
    return null
  }

  const wrapperClassName = variant === "embedded" ? "w-full" : "w-full mb-2"
  const panelClassName =
    variant === "embedded"
      ? "panel-elevated relative"
      : "panel-card mb-2 relative"

  const showApplyActions = settings.isDirty

  return (
    <div className={wrapperClassName}>
      <div className={panelClassName}>
        {/* Header */}
        <div className="flex items-center justify-between px-3 py-2 border-b border-border">
          <div className="min-w-0 pr-3">
            <h3 className="text-sm font-semibold text-text">
              {t("sidepanel:knowledge.title", "Knowledge Search")}
            </h3>
            <p className="mt-0.5 text-[11px] leading-4 text-text-muted">
              {t(
                "sidepanel:knowledge.scopeNote",
                "Search and insert context for this chat. Open /knowledge for the full QA workspace."
              )}
            </p>
          </div>
          <button
            onClick={() => setOpenState(false)}
            className="p-1 text-text-muted hover:text-text transition-colors rounded"
            aria-label={t("common:close", "Close")}
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Tabs */}
        <KnowledgeTabs
          activeTab={activeTab}
          onTabChange={handleTabChange}
          contextCount={contextCount}
        />

        {/* Tab content */}
        {activeTab === "qa-search" && (
          <QASearchTab
            query={settings.draftSettings.query}
            onQueryChange={(q) =>
              settings.updateSetting("query", q, { transient: true })
            }
            useCurrentMessage={settings.useCurrentMessage}
            onUseCurrentMessageChange={settings.setUseCurrentMessage}
            preset={settings.preset}
            onPresetChange={settings.applyPreset}
            strategy={settings.draftSettings.strategy ?? "standard"}
            onStrategyChange={(strategy) =>
              settings.updateSetting("strategy", strategy)
            }
            selectedSources={settings.draftSettings.sources}
            onSourcesChange={(sources) =>
              settings.updateSetting("sources", sources)
            }
            generationProvider={settings.draftSettings.generation_provider}
            onGenerationProviderChange={(provider) =>
              settings.updateSetting("generation_provider", provider)
            }
            generationModel={settings.draftSettings.generation_model ?? ""}
            onGenerationModelChange={(model) =>
              settings.updateSetting("generation_model", model || null)
            }
            onSearch={qaSearch.runQASearch}
            loading={qaSearch.loading}
            queryError={qaSearch.queryError}
            response={qaSearch.response}
            hasAttemptedSearch={qaSearch.hasAttemptedSearch}
            timedOut={qaSearch.timedOut}
            pinnedResults={search.pinnedResults}
            onCopyAnswer={qaSearch.copyAnswer}
            onInsertAnswer={qaSearch.insertAnswer}
            onCopyChunk={qaSearch.copyChunk}
            onInsertChunk={qaSearch.insertChunk}
            onPinChunk={qaSearch.pinChunk}
            onPreviewChunk={handlePreviewChunk}
            isConnected={isConnected}
            autoFocus={autoFocus}
            onOpenContext={() => handleTabChange("context")}
          />
        )}

        {activeTab === "file-search" && (
          <FileSearchTab
            query={settings.draftSettings.query}
            onQueryChange={(q) =>
              settings.updateSetting("query", q, { transient: true })
            }
            useCurrentMessage={settings.useCurrentMessage}
            onUseCurrentMessageChange={settings.setUseCurrentMessage}
            onSearch={fileSearch.runSearch}
            loading={fileSearch.loading}
            queryError={fileSearch.queryError}
            results={fileSearch.results}
            sortMode={fileSearch.sortMode}
            onSortModeChange={fileSearch.setSortMode}
            sortResults={fileSearch.sortResults}
            hasAttemptedSearch={fileSearch.hasAttemptedSearch}
            timedOut={fileSearch.timedOut}
            mediaTypes={fileSearch.mediaTypes}
            onMediaTypesChange={fileSearch.setMediaTypes}
            attachedMediaIds={fileSearch.attachedMediaIds}
            pinnedResults={search.pinnedResults}
            onPin={search.handlePin}
            onAttach={fileSearch.handleAttach}
            onPreview={handlePreview}
            onOpen={fileSearch.handleOpen}
            isConnected={isConnected}
            onOpenContext={() => handleTabChange("context")}
          />
        )}

        {activeTab === "settings" && (
          <SettingsTab
            settings={settings.draftSettings}
            preset={settings.preset}
            searchFilter={settings.advancedSearch}
            onSearchFilterChange={settings.setAdvancedSearch}
            onUpdate={settings.updateSetting}
            onPresetChange={settings.applyPreset}
            onResetToBalanced={settings.resetToBalanced}
          />
        )}

        {activeTab === "context" && (
          <ContextTab
            attachedImage={attachedImage}
            onRemoveImage={onRemoveImage}
            attachedTabs={attachedTabs}
            availableTabs={availableTabs}
            onRemoveTab={onRemoveTab || noop}
            onAddTab={onAddTab || noop}
            onClearTabs={onClearTabs || noop}
            onRefreshTabs={onRefreshTabs || noop}
            attachedFiles={attachedFiles}
            onAddFile={onAddFile || noop}
            onRemoveFile={onRemoveFile || noop}
            onClearFiles={onClearFiles || noop}
            pinnedResults={search.pinnedResults}
            onUnpinResult={search.handleUnpin}
            onClearPins={search.handleClearPins}
            fileRetrievalEnabled={fileRetrievalEnabled}
            onFileRetrievalChange={onFileRetrievalChange || noop}
          />
        )}

        {/* Apply actions (shown only while staged settings are dirty) */}
        {showApplyActions && (
          <div className="flex items-center justify-end gap-2 px-3 py-3 border-t border-border bg-surface">
            <button
              onClick={settings.applySettings}
              disabled={activeTab === "file-search"}
              className="px-3 py-1.5 text-sm text-text bg-surface2 rounded hover:bg-surface3 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {t("sidepanel:rag.apply", "Apply")}
            </button>
            <button
              onClick={handleApplyAndSearch}
              className="px-3 py-1.5 text-sm text-white bg-accent rounded hover:bg-accent/90 transition-colors"
            >
              {t("sidepanel:rag.applyAndSearch", "Apply & Search")}
            </button>
          </div>
        )}

        {/* Preview Modal */}
        <Modal
          open={!!previewItem}
          onCancel={() => setPreviewItem(null)}
          footer={null}
          title={previewItem?.title || t("sidepanel:rag.preview", "Preview")}
          width={600}
        >
          {previewItem && (
            <div className="space-y-4">
              {previewItem.source && (
                <p className="text-xs text-text-muted">
                  {t("sidepanel:rag.source", "Source")}: {previewItem.source}
                </p>
              )}
              <p className="text-sm text-text whitespace-pre-wrap">
                {previewItem.snippet}
              </p>
              <div className="flex gap-2 pt-2 border-t border-border">
                <button
                  onClick={() => {
                    void (async () => {
                      const resolved = await withFullMediaTextIfAvailable(
                        previewItem
                      )
                      onInsert(formatRagResult(resolved, "markdown"))
                      setPreviewItem(null)
                    })()
                  }}
                  className="px-3 py-1.5 text-sm bg-accent text-white rounded hover:bg-accent/90"
                >
                  {t("sidepanel:rag.actions.insert", "Insert")}
                </button>
                <button
                  onClick={() => {
                    onAsk(formatRagResult(previewItem, "markdown"), {
                      ignorePinnedResults: true
                    })
                    setPreviewItem(null)
                  }}
                  className="px-3 py-1.5 text-sm bg-surface2 text-text rounded hover:bg-surface3"
                >
                  {t("sidepanel:rag.actions.ask", "Ask")}
                </button>
              </div>
            </div>
          )}
        </Modal>
      </div>
    </div>
  )
}

export const KnowledgePanel = React.memo(KnowledgePanelBase)
KnowledgePanel.displayName = "KnowledgePanel"
