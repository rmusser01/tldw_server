import React, { useRef, useEffect, useCallback } from "react"
import { useTranslation } from "react-i18next"
import { Tooltip } from "antd"
import { X, FileText, BookOpen, Plus } from "lucide-react"
import { useDocumentWorkspaceStore } from "@/store/document-workspace"
import type { OpenDocument } from "./types"

interface DocumentTabProps {
  document: OpenDocument
  isActive: boolean
  onSelect: () => void
  onClose: (e: React.MouseEvent) => void
}

const DocumentTab: React.FC<DocumentTabProps> = ({
  document,
  isActive,
  onSelect,
  onClose
}) => {
  const { t } = useTranslation(["option", "common"])
  const Icon = document.type === "epub" ? BookOpen : FileText

  return (
    <div
      role="tab"
      aria-selected={isActive}
      tabIndex={isActive ? 0 : -1}
      onClick={onSelect}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault()
          onSelect()
        }
      }}
      className={`
        group relative flex items-center gap-2 px-3 py-2 min-w-0 max-w-[200px]
        border-r border-border cursor-pointer select-none
        transition-colors duration-150
        ${isActive
          ? "bg-surface text-text border-b-2 border-b-primary"
          : "bg-bg-subtle text-text-subtle hover:bg-hover hover:text-text"
        }
      `}
    >
      <Icon className="h-4 w-4 shrink-0" />
      <Tooltip title={document.title} mouseEnterDelay={0.5}>
        <span className="truncate text-sm">{document.title}</span>
      </Tooltip>
      <Tooltip title={t("option:documentWorkspace.closeDocument", "Close document")}>
        <button
          onClick={onClose}
          className={`
            ml-1 p-0.5 rounded shrink-0
            opacity-0 group-hover:opacity-100 focus:opacity-100
            hover:bg-hover-strong focus:outline-none focus:ring-1 focus:ring-primary
            transition-opacity duration-150
          `}
          aria-label={t("common:close", "Close")}
        >
          <X className="h-3.5 w-3.5" />
        </button>
      </Tooltip>
      {/* Reading progress bar */}
      {document.viewerState && (
        <div
          className="absolute bottom-0 left-0 h-0.5 bg-primary/40 transition-all duration-300"
          style={{
            width: `${
              document.type === "epub"
                ? document.viewerState.currentPercentage
                : document.viewerState.totalPages > 0
                  ? (document.viewerState.currentPage / document.viewerState.totalPages) * 100
                  : 0
            }%`
          }}
        />
      )}
    </div>
  )
}

/**
 * DocumentTabBar - Displays open documents as clickable tabs
 *
 * Features:
 * - Shows all open documents with icons based on type (PDF/EPUB)
 * - Allows switching between documents by clicking tabs
 * - Close button on each tab (visible on hover)
 * - Active tab is visually highlighted
 * - Horizontal scrolling when many documents are open
 * - Keyboard navigation support
 */
export const DocumentTabBar: React.FC<{
  onOpenPicker?: () => void
  onCloseDocument?: (id: number) => void
}> = ({
  onOpenPicker,
  onCloseDocument
}) => {
  const { t } = useTranslation(["option", "common"])
  const tabsContainerRef = useRef<HTMLDivElement>(null)

  const openDocuments = useDocumentWorkspaceStore((s) => s.openDocuments)
  const activeDocumentId = useDocumentWorkspaceStore((s) => s.activeDocumentId)
  const setActiveDocument = useDocumentWorkspaceStore((s) => s.setActiveDocument)
  const closeDocument = useDocumentWorkspaceStore((s) => s.closeDocument)

  // Scroll active tab into view when it changes
  useEffect(() => {
    if (tabsContainerRef.current && activeDocumentId !== null) {
      const activeTab = tabsContainerRef.current.querySelector(
        '[aria-selected="true"]'
      )
      if (activeTab) {
        activeTab.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "nearest" })
      }
    }
  }, [activeDocumentId])

  const handleSelectDocument = useCallback((id: number) => {
    setActiveDocument(id)
  }, [setActiveDocument])

  const handleCloseDocument = useCallback((e: React.MouseEvent, id: number) => {
    e.stopPropagation()
    if (onCloseDocument) {
      onCloseDocument(id)
    } else {
      closeDocument(id)
    }
  }, [closeDocument, onCloseDocument])

  // Keyboard navigation between tabs
  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (openDocuments.length === 0) return

    const currentIndex = openDocuments.findIndex((d) => d.id === activeDocumentId)
    let newIndex = currentIndex

    if (e.key === "ArrowLeft") {
      e.preventDefault()
      newIndex = currentIndex > 0 ? currentIndex - 1 : openDocuments.length - 1
    } else if (e.key === "ArrowRight") {
      e.preventDefault()
      newIndex = currentIndex < openDocuments.length - 1 ? currentIndex + 1 : 0
    } else if (e.key === "Home") {
      e.preventDefault()
      newIndex = 0
    } else if (e.key === "End") {
      e.preventDefault()
      newIndex = openDocuments.length - 1
    }

    if (newIndex !== currentIndex && openDocuments[newIndex]) {
      setActiveDocument(openDocuments[newIndex].id)
    }
  }, [openDocuments, activeDocumentId, setActiveDocument])

  // Don't render if no documents are open
  if (openDocuments.length === 0) {
    return null
  }

  return (
    <div className="flex h-10 shrink-0 items-center border-b border-border bg-bg-subtle overflow-hidden">
      <div
        ref={tabsContainerRef}
        className="flex flex-1 items-stretch overflow-x-auto scrollbar-thin scrollbar-track-transparent scrollbar-thumb-border"
        role="tablist"
        aria-label={t("option:documentWorkspace.openDocuments", "Open documents")}
        onKeyDown={handleKeyDown}
      >
        {openDocuments.map((doc) => (
          <DocumentTab
            key={doc.id}
            document={doc}
            isActive={doc.id === activeDocumentId}
            onSelect={() => handleSelectDocument(doc.id)}
            onClose={(e) => handleCloseDocument(e, doc.id)}
          />
        ))}
      </div>
      {onOpenPicker && (
        <Tooltip title={t("option:documentWorkspace.openDocument", "Open document")}>
          <button
            type="button"
            onClick={onOpenPicker}
            className="mx-2 flex h-7 w-7 items-center justify-center rounded border border-dashed border-border text-text-subtle hover:bg-hover hover:text-text"
            aria-label={t("option:documentWorkspace.openDocument", "Open document")}
          >
            <Plus className="h-4 w-4" />
          </button>
        </Tooltip>
      )}
    </div>
  )
}

export default DocumentTabBar
