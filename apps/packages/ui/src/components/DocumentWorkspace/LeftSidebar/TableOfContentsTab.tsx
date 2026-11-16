import React from "react"
import { useTranslation } from "react-i18next"
import { Empty, Skeleton } from "antd"
import { ChevronRight, ChevronDown, FileText } from "lucide-react"
import { useDocumentWorkspaceStore } from "@/store/document-workspace"
import { usePdfOutline } from "@/hooks/document-workspace"
import type { TocItem, DocumentType } from "../types"

interface TocEntryProps {
  item: TocItem
  currentPage: number
  documentType: DocumentType | null
  onNavigate: (page: number, href?: string) => void
  itemId: string
  focusedItemId: string | null
  onFocusItem: (id: string) => void
  depth?: number
}

const TocEntry: React.FC<TocEntryProps> = ({
  item,
  currentPage,
  documentType,
  onNavigate,
  itemId,
  focusedItemId,
  onFocusItem,
  depth = 0
}) => {
  const { t } = useTranslation(["option", "common"])
  const [expanded, setExpanded] = React.useState(true)
  const hasChildren = item.children && item.children.length > 0
  const treeItemRef = React.useRef<HTMLDivElement | null>(null)
  const isFocusable = focusedItemId === itemId

  // Determine if this item or any of its children contains the current page
  const containsCurrentPage = React.useMemo(() => {
    const checkContains = (entry: TocItem): boolean => {
      if (entry.page === currentPage) return true
      if (entry.children) {
        return entry.children.some(checkContains)
      }
      return false
    }
    return checkContains(item)
  }, [item, currentPage])

  const isActive = item.page === currentPage

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation()
    onNavigate(item.page, item.href)
    treeItemRef.current?.focus()
  }

  const handleToggle = (e: React.MouseEvent) => {
    e.stopPropagation()
    setExpanded(!expanded)
    treeItemRef.current?.focus()
  }

  // Indent based on depth (max 4 levels of indent)
  const paddingLeft = Math.min(depth, 4) * 12 + 8

  // For EPUB, don't show page numbers (they're meaningless)
  const showPageNumber = documentType === "pdf"

  const focusTreeItem = React.useCallback(
    (target: HTMLElement | null) => {
      if (!target) return
      const targetId = target.dataset.tocId
      if (targetId) {
        onFocusItem(targetId)
      }
      target.focus()
    },
    [onFocusItem]
  )

  const moveFocus = React.useCallback(
    (direction: "next" | "prev" | "first" | "last") => {
      const root = treeItemRef.current?.closest<HTMLElement>('[role="tree"]')
      if (!root || !treeItemRef.current) return
      const items = Array.from(
        root.querySelectorAll<HTMLElement>('[role="treeitem"]')
      )
      if (items.length === 0) return
      const currentIndex = items.indexOf(treeItemRef.current)
      if (currentIndex === -1) return
      let target: HTMLElement | undefined
      if (direction === "next") {
        target = items[currentIndex + 1]
      } else if (direction === "prev") {
        target = items[currentIndex - 1]
      } else if (direction === "first") {
        target = items[0]
      } else {
        target = items[items.length - 1]
      }
      if (target) {
        focusTreeItem(target)
      }
    },
    [focusTreeItem]
  )

  return (
    <div
      ref={treeItemRef}
      role="treeitem"
      aria-expanded={hasChildren ? expanded : undefined}
      aria-current={isActive ? "page" : undefined}
      data-toc-id={itemId}
      tabIndex={isFocusable ? 0 : -1}
      onFocus={() => onFocusItem(itemId)}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault()
          onNavigate(item.page, item.href)
        }
        if (e.key === "ArrowRight" && hasChildren && !expanded) {
          e.preventDefault()
          setExpanded(true)
        }
        if (e.key === "ArrowRight" && hasChildren && expanded) {
          e.preventDefault()
          const firstChild = treeItemRef.current?.querySelector<HTMLElement>(
            '[role="treeitem"]'
          )
          if (firstChild) {
            focusTreeItem(firstChild)
          }
        }
        if (e.key === "ArrowLeft") {
          if (hasChildren && expanded) {
            e.preventDefault()
            setExpanded(false)
            return
          }
          const parentItem =
            treeItemRef.current?.parentElement?.closest<HTMLElement>(
              '[role="treeitem"]'
            )
          if (parentItem) {
            e.preventDefault()
            focusTreeItem(parentItem)
          }
        }
        if (e.key === "ArrowDown") {
          e.preventDefault()
          moveFocus("next")
        }
        if (e.key === "ArrowUp") {
          e.preventDefault()
          moveFocus("prev")
        }
        if (e.key === "Home") {
          e.preventDefault()
          moveFocus("first")
        }
        if (e.key === "End") {
          e.preventDefault()
          moveFocus("last")
        }
      }}
    >
      <div
        className={`
          group flex w-full items-center gap-1 rounded px-2 py-1.5 text-left text-sm
          transition-colors hover:bg-hover
          ${isActive ? "bg-primary/10 text-primary font-medium" : "text-text"}
          ${containsCurrentPage && !isActive ? "text-text-secondary" : ""}
        `}
        style={{ paddingLeft }}
      >
        {hasChildren ? (
          <button
            type="button"
            onMouseDown={(e) => e.preventDefault()}
            onClick={handleToggle}
            tabIndex={-1}
            className="shrink-0 rounded p-0.5 hover:bg-hover-deep"
            aria-label={
              expanded
                ? t("option:documentWorkspace.collapseSection", "Collapse section")
                : t("option:documentWorkspace.expandSection", "Expand section")
            }
            aria-expanded={expanded}
          >
            {expanded ? (
              <ChevronDown className="h-3.5 w-3.5" />
            ) : (
              <ChevronRight className="h-3.5 w-3.5" />
            )}
          </button>
        ) : (
          <span className="w-4" aria-hidden="true" /> // Spacer for alignment
        )}

        <button
          type="button"
          onMouseDown={(e) => e.preventDefault()}
          onClick={handleClick}
          tabIndex={-1}
          className="flex min-w-0 flex-1 items-center gap-1 text-left"
        >
          <span className="flex-1 truncate" title={item.title}>
            {item.title}
          </span>

          {showPageNumber && (
            <span
              className={`
                shrink-0 text-xs tabular-nums
                ${isActive ? "text-primary" : "text-muted"}
                opacity-0 group-hover:opacity-100
                ${isActive ? "opacity-100" : ""}
              `}
              aria-label={t("option:documentWorkspace.pageNumber", "Page {{page}}", { page: item.page })}
            >
              {item.page}
            </span>
          )}
        </button>
      </div>

      {hasChildren && expanded && (
        <div role="group">
          {item.children!.map((child, idx) => (
            <TocEntry
              key={`${child.page}-${idx}`}
              item={child}
              currentPage={currentPage}
              documentType={documentType}
              onNavigate={onNavigate}
              itemId={`${itemId}-${idx}`}
              focusedItemId={focusedItemId}
              onFocusItem={onFocusItem}
              depth={depth + 1}
            />
          ))}
        </div>
      )}
    </div>
  )
}

export const TableOfContentsTab: React.FC<{ onHide?: () => void }> = ({ onHide }) => {
  const { t } = useTranslation(["option", "common"])
  const activeDocumentId = useDocumentWorkspaceStore((s) => s.activeDocumentId)
  const activeDocumentType = useDocumentWorkspaceStore((s) => s.activeDocumentType)
  const currentPage = useDocumentWorkspaceStore((s) => s.currentPage)
  const setCurrentPage = useDocumentWorkspaceStore((s) => s.setCurrentPage)
  const [focusedItemId, setFocusedItemId] = React.useState<string | null>(null)
  const lastDocumentIdRef = React.useRef<number | null>(null)

  // For PDF, fetch outline from server
  // For EPUB, outline comes from the viewer component via custom event
  const {
    data: pdfOutline,
    isLoading: pdfLoading,
    error: pdfError
  } = usePdfOutline(activeDocumentType === "pdf" ? activeDocumentId : null)

  // EPUB outline state - populated by EpubViewer via custom event
  const [epubOutline, setEpubOutline] = React.useState<TocItem[] | null>(null)
  const [epubLoading, setEpubLoading] = React.useState(false)

  // Listen for EPUB outline updates
  React.useEffect(() => {
    const handleEpubOutline = (e: CustomEvent<{ documentId: number; items: TocItem[] }>) => {
      if (e.detail.documentId === activeDocumentId) {
        setEpubOutline(e.detail.items)
        setEpubLoading(false)
      }
    }

    const handleEpubLoading = (e: CustomEvent<{ documentId: number }>) => {
      if (e.detail.documentId === activeDocumentId) {
        setEpubLoading(true)
        setEpubOutline(null)
      }
    }

    window.addEventListener("epub-outline-ready", handleEpubOutline as EventListener)
    window.addEventListener("epub-loading", handleEpubLoading as EventListener)

    return () => {
      window.removeEventListener("epub-outline-ready", handleEpubOutline as EventListener)
      window.removeEventListener("epub-loading", handleEpubLoading as EventListener)
    }
  }, [activeDocumentId])

  // Reset EPUB outline when document changes
  React.useEffect(() => {
    if (activeDocumentType === "epub") {
      setEpubLoading(true)
    } else {
      setEpubOutline(null)
      setEpubLoading(false)
    }
  }, [activeDocumentId, activeDocumentType])

  const handleNavigate = React.useCallback(
    (page: number, href?: string) => {
      if (activeDocumentType === "epub" && href) {
        // For EPUB, dispatch event to navigate to href
        window.dispatchEvent(
          new CustomEvent("epub-navigate", {
            detail: { href, documentId: activeDocumentId }
          })
        )
      } else {
        // For PDF, use page navigation
        setCurrentPage(page)
      }
    },
    [activeDocumentType, activeDocumentId, setCurrentPage]
  )

  // Determine which outline to use
  const outline = activeDocumentType === "epub" ? epubOutline : pdfOutline?.items
  const isLoading = activeDocumentType === "epub" ? epubLoading : pdfLoading
  const error = activeDocumentType === "epub" ? null : pdfError
  const hasOutline = !!outline && outline.length > 0
  const validNodeIds = React.useMemo(() => {
    if (!outline) {
      return new Set<string>()
    }
    const ids = new Set<string>()
    const collectIds = (items: TocItem[], parentId?: string) => {
      items.forEach((item, idx) => {
        const itemId = parentId ? `${parentId}-${idx}` : `${idx}`
        ids.add(itemId)
        if (item.children && item.children.length > 0) {
          collectIds(item.children, itemId)
        }
      })
    }
    collectIds(outline)
    return ids
  }, [outline])

  React.useEffect(() => {
    if (!hasOutline) {
      if (focusedItemId !== null) {
        setFocusedItemId(null)
      }
      lastDocumentIdRef.current = activeDocumentId
      return
    }

    const documentChanged = activeDocumentId !== lastDocumentIdRef.current
    if (focusedItemId !== null && !validNodeIds.has(focusedItemId)) {
      setFocusedItemId("0")
    } else if (documentChanged || focusedItemId === null) {
      setFocusedItemId("0")
    }
    lastDocumentIdRef.current = activeDocumentId
  }, [activeDocumentId, hasOutline, focusedItemId, validNodeIds])

  if (!activeDocumentId) {
    return (
      <div className="flex h-full items-center justify-center p-4">
        <Empty
          image={<FileText className="mx-auto h-12 w-12 text-muted" />}
          description={t(
            "option:documentWorkspace.noDocumentSelected",
            "No document selected"
          )}
        />
      </div>
    )
  }

  if (isLoading) {
    return (
      <div className="space-y-2 p-4">
        <Skeleton active paragraph={{ rows: 1 }} title={false} />
        <Skeleton active paragraph={{ rows: 1 }} title={false} />
        <Skeleton active paragraph={{ rows: 1 }} title={false} />
        <Skeleton active paragraph={{ rows: 1 }} title={false} />
        <Skeleton active paragraph={{ rows: 1 }} title={false} />
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex h-full items-center justify-center p-4">
        <Empty
          description={t(
            "option:documentWorkspace.errorLoadingToc",
            "Failed to load table of contents"
          )}
        />
      </div>
    )
  }

  if (!hasOutline) {
    return (
      <div className="flex h-full items-center justify-center p-4">
        <Empty
          image={<FileText className="mx-auto h-12 w-12 text-muted" />}
          description={
            <div className="text-center">
              <p className="mb-1 text-sm text-text">
                {t(
                  "option:documentWorkspace.noToc",
                  "No table of contents"
                )}
              </p>
              <p className="text-xs text-text-muted">
                {t(
                  "option:documentWorkspace.noTocHint",
                  "This document doesn't have an embedded outline"
                )}
              </p>
              {onHide && (
                <button type="button" onClick={onHide} className="mt-3 min-h-11 rounded border border-border px-3 text-sm text-text hover:bg-surface2 focus-visible:outline focus-visible:outline-2 focus-visible:outline-focus">
                  {t("option:documentWorkspace.hideContents", "Hide contents")}
                </button>
              )}
            </div>
          }
        />
      </div>
    )
  }

  return (
    <nav
      className="h-full overflow-y-auto"
      aria-label={t("option:documentWorkspace.tableOfContents", "Table of contents")}
    >
      <div className="py-2" role="tree">
        {outline.map((item, idx) => (
          <TocEntry
            key={`${item.page}-${idx}`}
            item={item}
            currentPage={currentPage}
            documentType={activeDocumentType}
            onNavigate={handleNavigate}
            itemId={`${idx}`}
            focusedItemId={focusedItemId}
            onFocusItem={setFocusedItemId}
          />
        ))}
      </div>
    </nav>
  )
}

export default TableOfContentsTab
