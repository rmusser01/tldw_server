/**
 * HistorySidebar - Past searches sidebar
 */

import React, { useMemo, useState, useEffect, useCallback } from "react"
import {
  History,
  Search,
  FileText,
  Sparkles,
  Trash2,
  Pin,
  ChevronLeft,
  ChevronRight,
  Settings,
  Download,
} from "lucide-react"
import { useKnowledgeQA } from "./KnowledgeQAProvider"
import { cn } from "@/libs/utils"
import type { SearchHistoryItem } from "./types"
import { useMobile } from "@/hooks/useMediaQuery"
import { useAntdMessage } from "@/hooks/useAntdMessage"
import {
  buildGroupedHistorySections,
  buildHistoryExportMarkdown,
  filterHistoryItems,
  isKnowledgeQaHistoryItem,
} from "./historyUtils"

type HistorySidebarProps = {
  className?: string
}

function formatTimestamp(timestamp: string): string {
  const date = new Date(timestamp)
  const now = new Date()
  const diff = now.getTime() - date.getTime()
  const days = Math.floor(diff / (1000 * 60 * 60 * 24))

  if (days === 0) {
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
  } else if (days === 1) {
    return "Yesterday"
  } else if (days < 7) {
    return date.toLocaleDateString([], { weekday: "short" })
  } else {
    return date.toLocaleDateString([], { month: "short", day: "numeric" })
  }
}

function triggerFileDownload(blob: Blob, filename: string) {
  if (typeof window === "undefined") {
    throw new Error("History export is only available in the browser")
  }

  const createObjectUrl = window.URL?.createObjectURL
  const revokeObjectUrl = window.URL?.revokeObjectURL
  if (typeof createObjectUrl !== "function" || typeof revokeObjectUrl !== "function") {
    throw new Error("File download is unavailable in this browser")
  }

  const url = createObjectUrl(blob)
  const link = document.createElement("a")
  link.href = url
  link.download = filename
  link.click()
  revokeObjectUrl(url)
}

// Skeleton loading component for history items
function HistorySkeleton() {
  return (
    <div className="py-2 space-y-3 animate-pulse" aria-label="Loading search history">
      {[1, 2, 3].map((i) => (
        <div key={i} className="px-4">
          <div className="flex items-start gap-2 px-2 py-2">
            <div className="w-4 h-4 bg-bg-subtle rounded flex-shrink-0" />
            <div className="flex-1 space-y-2">
              <div className="h-4 bg-bg-subtle rounded w-3/4" />
              <div className="h-3 bg-bg-subtle rounded w-1/2" />
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}

const EXPAND_HINT_SEEN_KEY = "knowledge_qa_history_expand_hint_seen"

function safePersistExpandHintSeen(): void {
  try {
    localStorage.setItem(EXPAND_HINT_SEEN_KEY, "1")
  } catch {
    // Ignore localStorage failures in private mode.
  }
}

export function HistorySidebar({ className }: HistorySidebarProps) {
  const {
    searchHistory,
    historyHydrated,
    currentThreadId,
    historySidebarOpen,
    setHistorySidebarOpen,
    restoreFromHistory,
    deleteHistoryItem,
    toggleHistoryPin,
    preset,
    setSettingsPanelOpen,
  } = useKnowledgeQA()

  const message = useAntdMessage()

  const [historyFilter, setHistoryFilter] = useState("")
  const [showExpandHint, setShowExpandHint] = useState(false)
  const isMobile = useMobile()
  const isInitialLoad = !historyHydrated && searchHistory.length === 0

  useEffect(() => {
    if (isMobile || historySidebarOpen) {
      setShowExpandHint(false)
      return
    }

    try {
      const seenHint = localStorage.getItem(EXPAND_HINT_SEEN_KEY) === "1"
      if (seenHint) return
      setShowExpandHint(true)
      const timer = window.setTimeout(() => {
        setShowExpandHint(false)
        safePersistExpandHintSeen()
      }, 5000)
      return () => window.clearTimeout(timer)
    } catch {
      return
    }
  }, [historySidebarOpen, isMobile])

  const knowledgeHistory = useMemo(
    () => searchHistory.filter((item) => isKnowledgeQaHistoryItem(item)),
    [searchHistory]
  )

  const visibleHistory = useMemo(
    () => filterHistoryItems(knowledgeHistory, historyFilter),
    [knowledgeHistory, historyFilter]
  )

  const groupedHistory = useMemo(
    () => buildGroupedHistorySections(visibleHistory),
    [visibleHistory]
  )

  const handleExpandSidebar = useCallback(() => {
    setHistorySidebarOpen(true)
    setShowExpandHint(false)
    safePersistExpandHintSeen()
  }, [setHistorySidebarOpen])

  const handleExportAll = useCallback(() => {
    if (knowledgeHistory.length === 0) {
      message.open({
        type: "info",
        content: "No history entries available to export.",
        duration: 3,
      })
      return
    }

    try {
      const markdown = buildHistoryExportMarkdown(knowledgeHistory)
      const blob = new Blob([markdown], { type: "text/markdown;charset=utf-8" })
      const dateSuffix = new Date().toISOString().slice(0, 10)
      triggerFileDownload(blob, `knowledge_qa_history_${dateSuffix}.md`)
      message.open({
        type: "success",
        content: `Exported ${knowledgeHistory.length} history entries.`,
        duration: 3,
      })
    } catch (error) {
      console.error("Failed to export Knowledge QA history:", error)
      message.open({
        type: "error",
        content: "History export failed. Please try again.",
        duration: 4,
      })
    }
  }, [knowledgeHistory, message])

  const renderExpandedContent = () => (
    <>
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border">
        <div className="flex items-center gap-2">
          <History className="w-4 h-4 text-text-muted" />
          <span className="font-medium text-sm">History</span>
        </div>
        <button
          onClick={() => setHistorySidebarOpen(false)}
          className="p-1 rounded hover:bg-hover transition-colors"
          title="Collapse sidebar"
        >
          <ChevronLeft className="w-4 h-4 text-text-muted" />
        </button>
      </div>

      <div className="px-4 py-2 border-b border-border">
        <label htmlFor="knowledge-history-filter" className="sr-only">
          Filter history
        </label>
        <div className="relative">
          <Search className="w-3.5 h-3.5 absolute left-2.5 top-1/2 -translate-y-1/2 text-text-muted" />
          <input
            id="knowledge-history-filter"
            value={historyFilter}
            onChange={(event) => setHistoryFilter(event.target.value)}
            placeholder="Filter history"
            className="w-full rounded-md border border-border bg-surface py-1.5 pl-8 pr-2 text-xs focus:outline-none focus:ring-2 focus:ring-primary"
            aria-label="Filter history"
          />
        </div>
      </div>

      {/* History list */}
      <div className="flex-1 overflow-y-auto">
        {isInitialLoad ? (
          <HistorySkeleton />
        ) : visibleHistory.length === 0 ? (
          <div className="px-4 py-8 text-center text-sm text-text-muted">
            <Search className="w-8 h-8 mx-auto mb-2 opacity-50" />
            <p>{historyFilter.trim().length > 0 ? "No matching history" : "No search history yet"}</p>
            <p className="text-xs mt-1">
              {historyFilter.trim().length > 0
                ? "Try a different keyword"
                : "Your searches will appear here"}
            </p>
          </div>
        ) : (
          <div className="py-2">
            {groupedHistory.pinned.length > 0 && (
              <div className="mb-3">
                <div className="px-4 py-1 text-xs font-medium text-text-muted uppercase tracking-wide">
                  Pinned
                </div>
                <div className="space-y-1">
                  {groupedHistory.pinned.map((item) => (
                    <HistoryItem
                      key={item.id}
                      item={item}
                      isActive={Boolean(
                        currentThreadId &&
                          (item.conversationId === currentThreadId || item.id === currentThreadId)
                      )}
                      alwaysShowActions={isMobile}
                      onSelect={() => restoreFromHistory(item)}
                      onDelete={() => deleteHistoryItem(item.id)}
                      onTogglePin={() => toggleHistoryPin(item.id)}
                    />
                  ))}
                </div>
              </div>
            )}

            {Array.from(groupedHistory.groupedByDate.entries()).map(([group, items]) => (
              <div key={group} className="mb-3">
                <div className="px-4 py-1 text-xs font-medium text-text-muted uppercase tracking-wide">
                  {group}
                </div>
                <div className="space-y-1">
                  {items.map((item) => (
                    <HistoryItem
                      key={item.id}
                      item={item}
                      isActive={Boolean(
                        currentThreadId &&
                          (item.conversationId === currentThreadId || item.id === currentThreadId)
                      )}
                      alwaysShowActions={isMobile}
                      onSelect={() => restoreFromHistory(item)}
                      onDelete={() => deleteHistoryItem(item.id)}
                      onTogglePin={() => toggleHistoryPin(item.id)}
                    />
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Footer actions */}
      <div className="px-4 py-3 border-t border-border">
        <button
          onClick={handleExportAll}
          className="flex items-center gap-2 w-full px-3 py-2 text-sm rounded-md text-text-subtle hover:bg-hover hover:text-text transition-colors"
        >
          <Download className="w-4 h-4 text-text-muted" />
          <span>Export All</span>
        </button>
      </div>
    </>
  )

  if (isMobile) {
    return (
      <>
        {!historySidebarOpen && (
          <button
            type="button"
            onClick={() => setHistorySidebarOpen(true)}
            className="fixed left-3 top-[max(0.75rem,env(safe-area-inset-top))] z-30 rounded-lg border border-border bg-surface p-2 shadow-sm"
            aria-label="Open history panel"
            title="Open history panel"
            data-testid="knowledge-history-mobile-open"
          >
            <History className="w-4 h-4 text-text-muted" />
          </button>
        )}

        {historySidebarOpen && (
          <div
            className="fixed inset-0 z-40 lg:hidden"
            data-testid="knowledge-history-mobile-overlay"
          >
            <button
              type="button"
              className="absolute inset-0 bg-black/45"
              onClick={() => setHistorySidebarOpen(false)}
              aria-label="Close history panel"
            />
            <aside
              className={cn(
                "relative h-full w-[85vw] max-w-sm border-r border-border bg-surface/95 backdrop-blur",
                className
              )}
            >
              <div className="flex h-full flex-col">{renderExpandedContent()}</div>
            </aside>
          </div>
        )}
      </>
    )
  }

  // Collapsed state
  if (!historySidebarOpen) {
    return (
      <div
        className={cn("flex flex-col items-center py-4 px-2 border-r border-border bg-surface", className)}
        data-testid="knowledge-history-desktop-collapsed"
      >
        <button
          onClick={handleExpandSidebar}
          className={cn(
            "p-2 rounded-lg hover:bg-hover transition-colors",
            showExpandHint ? "animate-pulse" : ""
          )}
          title="Expand history sidebar"
          aria-label="Expand history sidebar"
        >
          <ChevronRight className="w-5 h-5 text-text-muted" />
        </button>

        {showExpandHint ? (
          <div className="mt-1 rounded bg-primary/10 px-2 py-1 text-[10px] text-primary" role="status">
            Expand history sidebar
          </div>
        ) : null}

        <div className="mt-4 flex flex-col gap-2">
          <button
            onClick={handleExpandSidebar}
            className="p-2 rounded-lg hover:bg-hover transition-colors"
            title="Search history"
          >
            <History className="w-5 h-5 text-text-muted" />
          </button>

          <button
            onClick={() => setSettingsPanelOpen(true)}
            className="p-2 rounded-lg hover:bg-hover transition-colors"
            title="Settings"
          >
            <Settings className="w-5 h-5 text-text-muted" />
          </button>
        </div>
      </div>
    )
  }

  return (
    <div
      className={cn("flex flex-col border-r border-border bg-surface", className)}
      style={{ width: "var(--sidebar-width)" }}
      data-testid="knowledge-history-desktop-open"
    >
      {renderExpandedContent()}
    </div>
  )
}

// Individual history item
function HistoryItem({
  item,
  isActive,
  alwaysShowActions,
  onSelect,
  onDelete,
  onTogglePin,
}: {
  item: SearchHistoryItem
  isActive: boolean
  alwaysShowActions: boolean
  onSelect: () => void
  onDelete: () => void
  onTogglePin: () => void
}) {
  const [confirmDelete, setConfirmDelete] = React.useState(false)

  // Reset confirm state after 3 seconds
  React.useEffect(() => {
    if (confirmDelete) {
      const timer = setTimeout(() => setConfirmDelete(false), 3000)
      return () => clearTimeout(timer)
    }
  }, [confirmDelete])

  const handleDeleteClick = (e: React.MouseEvent) => {
    e.stopPropagation()
    if (confirmDelete) {
      onDelete()
      setConfirmDelete(false)
    } else {
      setConfirmDelete(true)
    }
  }

  const handlePinClick = (e: React.MouseEvent) => {
    e.stopPropagation()
    onTogglePin()
  }

  const actionVisibilityClass =
    alwaysShowActions || item.pinned || confirmDelete
      ? "opacity-100"
      : "opacity-0 group-hover:opacity-100 group-focus-within:opacity-100"

  return (
    <div className="group relative px-2">
      <button
        onClick={onSelect}
        aria-current={isActive ? "true" : undefined}
        className={cn(
          "flex items-start gap-2 w-full px-2 py-2 text-left rounded-md transition-colors",
          isActive ? "bg-primary/10 border-l-2 border-primary" : "hover:bg-hover"
        )}
      >
        <Search className="w-4 h-4 mt-0.5 text-text-muted flex-shrink-0" />
        <div className="flex-1 min-w-0 pr-12">
          <p className="text-sm truncate">{item.query}</p>
          {item.answerPreview ? (
            <p className="mt-0.5 text-xs text-text-muted line-clamp-1">{item.answerPreview}</p>
          ) : null}
          <div className="flex items-center gap-2 mt-0.5 text-xs text-text-muted">
            <span className="flex items-center gap-1">
              <FileText className="w-3 h-3" />
              {item.sourcesCount}
            </span>
            {item.hasAnswer && (
              <span className="flex items-center gap-1">
                <Sparkles className="w-3 h-3" />
              </span>
            )}
            <span>{formatTimestamp(item.timestamp)}</span>
          </div>
        </div>
      </button>

      <button
        onClick={handlePinClick}
        aria-label={item.pinned ? "Unpin history item" : "Pin history item"}
        className={cn(
          "absolute right-7 top-1/2 -translate-y-1/2 rounded p-1 transition-all hover:bg-hover",
          actionVisibilityClass
        )}
        title={item.pinned ? "Unpin" : "Pin"}
      >
        <Pin className={cn("w-3.5 h-3.5", item.pinned ? "text-primary" : "text-text-muted")} />
      </button>

      {/* Delete button (shown on hover) - requires confirmation */}
      <button
        onClick={handleDeleteClick}
        onBlur={() => setConfirmDelete(false)}
        aria-label={confirmDelete ? "Click again to confirm deletion" : "Delete from history"}
        className={cn(
          "absolute right-2 top-1/2 -translate-y-1/2 rounded transition-all focus-visible:opacity-100",
          confirmDelete
            ? "px-2 py-1 text-xs font-medium bg-danger text-white opacity-100"
            : cn("p-1 hover:bg-danger/10 hover:text-danger", actionVisibilityClass)
        )}
        title={confirmDelete ? "Click to confirm delete" : "Delete from history"}
      >
        {confirmDelete ? "Delete?" : <Trash2 className="w-3.5 h-3.5" />}
      </button>
    </div>
  )
}
