import React from "react"
import { useTranslation } from "react-i18next"
import { Switch, Tooltip } from "antd"
import { X } from "lucide-react"
import type { UploadedFile } from "@/db/dexie/types"
import type { TabInfo } from "@/hooks/useTabMentions"
import type { RagPinnedResult } from "@/utils/rag-format"
import { AttachedTabs } from "./AttachedTabs"
import { AttachedFiles } from "./AttachedFiles"
import { PinnedResults } from "./PinnedResults"

type ContextTabProps = {
  // Attached browser tabs
  attachedTabs: TabInfo[]
  availableTabs: TabInfo[]
  onRemoveTab: (tabId: number) => void
  onAddTab: (tab: TabInfo) => void
  onClearTabs: () => void
  onRefreshTabs: () => void

  // Attached image
  attachedImage?: string
  onRemoveImage?: () => void

  // Attached files
  attachedFiles: UploadedFile[]
  onAddFile: () => void
  onRemoveFile: (fileId: string) => void
  onClearFiles: () => void

  // Pinned results
  pinnedResults: RagPinnedResult[]
  onUnpinResult: (id: string) => void
  onClearPins: () => void

  // File retrieval (RAG) toggle
  fileRetrievalEnabled: boolean
  onFileRetrievalChange: (enabled: boolean) => void
}

/**
 * ContextTab - Unified attachment management
 *
 * Phase 4 implementation: Shows all attached context items
 * (browser tabs, files, and pinned RAG results) in one place.
 */
export const ContextTab: React.FC<ContextTabProps> = ({
  attachedTabs,
  availableTabs,
  onRemoveTab,
  onAddTab,
  onClearTabs,
  onRefreshTabs,
  attachedImage,
  onRemoveImage,
  attachedFiles,
  onAddFile,
  onRemoveFile,
  onClearFiles,
  pinnedResults,
  onUnpinResult,
  onClearPins,
  fileRetrievalEnabled,
  onFileRetrievalChange
}) => {
  const { t } = useTranslation(["sidepanel", "playground", "common"])

  const hasAnyContent =
    Boolean(attachedImage) ||
    attachedTabs.length > 0 ||
    attachedFiles.length > 0 ||
    pinnedResults.length > 0

  return (
    <div
      className="flex flex-col h-full overflow-y-auto"
      role="tabpanel"
      id="knowledge-tabpanel-context"
      aria-labelledby="knowledge-tab-context"
    >
      {/* Header with RAG toggle */}
      <div className="px-3 py-2 border-b border-border bg-surface2/50">
        <div className="flex items-center justify-between gap-2">
          <p className="text-xs text-text-muted flex-1">
            {t(
              "sidepanel:rag.contextExplanation",
              "These items will be included in your next query."
            )}
          </p>
          <Tooltip
            title={t(
              "playground:attachments.enableKnowledgeSearchTooltip",
              "When enabled, attached files and pinned results will be searched for relevant context"
            )}
          >
            <div className="inline-flex items-center gap-1.5 shrink-0">
              <Switch
                size="small"
                checked={fileRetrievalEnabled}
                onChange={onFileRetrievalChange}
                aria-label={
                  t(
                    "playground:attachments.enableKnowledgeSearch",
                    "Include in Search & Context"
                  ) as string
                }
              />
              <span className="text-xs text-text-muted whitespace-nowrap">
                {t(
                  "playground:attachments.enableKnowledgeSearch",
                  "Include in Search & Context"
                )}
              </span>
            </div>
          </Tooltip>
        </div>
      </div>

      {/* Content sections */}
      <div className="flex-1 px-3 py-3 space-y-4">
        {/* Attached Image */}
        {attachedImage && (
          <div className="rounded border border-border bg-surface overflow-hidden">
            <div className="flex items-center justify-between px-3 py-2 bg-surface2/50">
              <span className="text-xs font-semibold text-text">
                {t("playground:attachments.image", "Image")}
              </span>
              {onRemoveImage && (
                <button
                  type="button"
                  onClick={onRemoveImage}
                  className="p-1 text-text-muted hover:text-danger transition-colors rounded hover:bg-surface3"
                  aria-label={t("common:remove", "Remove") as string}
                  title={t("common:remove", "Remove") as string}
                >
                  <X className="h-3.5 w-3.5" />
                </button>
              )}
            </div>
            <div className="px-3 py-2">
              <img
                src={attachedImage}
                alt={t("playground:attachments.imageLabel", "Attached image") as string}
                className="rounded-md max-h-28"
              />
            </div>
          </div>
        )}

        {/* Attached Browser Tabs */}
        <AttachedTabs
          tabs={attachedTabs}
          availableTabs={availableTabs}
          onRemove={onRemoveTab}
          onAdd={onAddTab}
          onClear={onClearTabs}
          onRefresh={onRefreshTabs}
        />

        {/* Attached Files */}
        <AttachedFiles
          files={attachedFiles}
          onAdd={onAddFile}
          onRemove={onRemoveFile}
          onClear={onClearFiles}
        />

        {/* Pinned Results */}
        <PinnedResults
          results={pinnedResults}
          onUnpin={onUnpinResult}
          onClear={onClearPins}
        />

        {/* Empty state */}
        {!hasAnyContent && (
          <div className="text-center py-8 text-text-muted">
            <p className="text-sm">
              {t(
                "sidepanel:rag.noContextItems",
                "No context items attached yet."
              )}
            </p>
            <p className="text-xs mt-2">
              {t(
                "sidepanel:rag.contextHint",
                "Save search results or attach web pages and files to include them in your queries."
              )}
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
