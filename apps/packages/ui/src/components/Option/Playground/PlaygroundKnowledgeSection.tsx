import React from "react"
import { KnowledgePanel, type KnowledgeTab } from "@/components/Knowledge"

export type PlaygroundKnowledgeSectionProps = {
  contextToolsOpen: boolean
  isConnectionReady: boolean
  knowledgePanelTab: KnowledgeTab
  knowledgePanelTabRequestId: number
  deferredComposerInput: string
  attachedImage: string
  attachedTabs: any[]
  availableTabs: any[]
  attachedFiles: any[]
  fileRetrievalEnabled: boolean
  onInsert: (text: string) => void
  onAsk: (text: string, options?: { ignorePinnedResults?: boolean }) => void
  onOpenChange: (open: boolean) => void
  onRemoveImage: () => void
  onRemoveTab: (tab: any) => void
  onAddTab: (tab: any) => void
  onClearTabs: () => void
  onRefreshTabs: () => void
  onAddFile: () => void
  onRemoveFile: (file: any) => void
  onClearFiles: () => void
  onFileRetrievalChange: (enabled: boolean) => void
  wrapComposerProfile: (id: string, element: React.ReactNode) => React.ReactNode
  t: (key: string, defaultValueOrOptions?: any, options?: any) => string
}

export const PlaygroundKnowledgeSection = React.memo(
  function PlaygroundKnowledgeSection(props: PlaygroundKnowledgeSectionProps) {
    const {
      contextToolsOpen,
      isConnectionReady,
      knowledgePanelTab,
      knowledgePanelTabRequestId,
      deferredComposerInput,
      attachedImage,
      attachedTabs,
      availableTabs,
      attachedFiles,
      fileRetrievalEnabled,
      onInsert,
      onAsk,
      onOpenChange,
      onRemoveImage,
      onRemoveTab,
      onAddTab,
      onClearTabs,
      onRefreshTabs,
      onAddFile,
      onRemoveFile,
      onClearFiles,
      onFileRetrievalChange,
      wrapComposerProfile,
      t
    } = props

    return (
      <div
        className={contextToolsOpen ? "mb-2" : "hidden"}
        aria-hidden={!contextToolsOpen}
      >
        <div className="rounded-md bg-surface2/50 p-3">
          <div className="flex flex-col gap-4">
            <div>
              <div className="mb-2 text-xs font-semibold text-text">
                {t(
                  "playground:composer.knowledgeSearch",
                  "Search & Context"
                )}
              </div>
              {wrapComposerProfile(
                "knowledge-panel",
                <KnowledgePanel
                  onInsert={onInsert}
                  onAsk={onAsk}
                  isConnected={isConnectionReady}
                  open={contextToolsOpen}
                  onOpenChange={onOpenChange}
                  openTab={knowledgePanelTab}
                  openTabRequestId={knowledgePanelTabRequestId}
                  autoFocus
                  showToggle={false}
                  variant="embedded"
                  currentMessage={contextToolsOpen ? deferredComposerInput : ""}
                  showAttachedContext
                  attachedImage={attachedImage}
                  attachedTabs={attachedTabs}
                  availableTabs={availableTabs}
                  attachedFiles={attachedFiles}
                  onRemoveImage={onRemoveImage}
                  onRemoveTab={onRemoveTab}
                  onAddTab={onAddTab}
                  onClearTabs={onClearTabs}
                  onRefreshTabs={onRefreshTabs}
                  onAddFile={onAddFile}
                  onRemoveFile={onRemoveFile}
                  onClearFiles={onClearFiles}
                  fileRetrievalEnabled={fileRetrievalEnabled}
                  onFileRetrievalChange={onFileRetrievalChange}
                />
              )}
            </div>
          </div>
        </div>
      </div>
    )
  }
)
