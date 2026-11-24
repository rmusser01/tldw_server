import React, { useRef, useEffect, useCallback, useMemo } from "react"
import { useTranslation } from "react-i18next"
import { Input, Button, Spin, Tooltip } from "antd"
import { X, ChevronUp, ChevronDown } from "lucide-react"
import { usePdfSearch, type PdfDocumentProxy } from "@/hooks/document-workspace/usePdfSearch"
import { useDocumentWorkspaceStore } from "@/store/document-workspace"

interface PdfSearchProps {
  pdfDocumentRef: React.RefObject<PdfDocumentProxy | null>
}

/**
 * PDF search overlay component.
 * Fixed position in top-right of viewer with search input and navigation.
 */
export const PdfSearch: React.FC<PdfSearchProps> = ({ pdfDocumentRef }) => {
  const { t } = useTranslation(["option", "common"])
  const inputRef = useRef<HTMLInputElement>(null)

  const {
    searchOpen,
    searchQuery,
    searchResults,
    activeSearchIndex,
    isIndexing,
    indexFailed,
    setSearchQuery,
    closeSearch,
    goToNextResult,
    goToPreviousResult
  } = usePdfSearch(pdfDocumentRef)

  const searchMatchCase = useDocumentWorkspaceStore((s) => s.searchMatchCase)
  const setSearchMatchCase = useDocumentWorkspaceStore((s) => s.setSearchMatchCase)
  const searchWordBoundary = useDocumentWorkspaceStore((s) => s.searchWordBoundary)
  const setSearchWordBoundary = useDocumentWorkspaceStore((s) => s.setSearchWordBoundary)
  const annotations = useDocumentWorkspaceStore((s) => s.annotations)
  const setActiveRightTab = useDocumentWorkspaceStore((s) => s.setActiveRightTab)

  const annotationMatches = useMemo(() => {
    const trimmedQuery = searchQuery.trim()
    if (!trimmedQuery) return 0
    const q = searchMatchCase ? trimmedQuery : trimmedQuery.toLowerCase()
    return annotations.reduce((count, ann) => {
      const text = searchMatchCase ? ann.text : ann.text.toLowerCase()
      const note = ann.note ? (searchMatchCase ? ann.note : ann.note.toLowerCase()) : null
      return (text.includes(q) || (note && note.includes(q))) ? count + 1 : count
    }, 0)
  }, [searchQuery, searchMatchCase, annotations])

  // Auto-focus input when search opens
  useEffect(() => {
    if (searchOpen && inputRef.current) {
      inputRef.current.focus()
    }
  }, [searchOpen])

  // Handle keyboard shortcuts
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault()
        closeSearch()
      } else if (e.key === "Enter") {
        e.preventDefault()
        if (e.shiftKey) {
          goToPreviousResult()
        } else {
          goToNextResult()
        }
      }
    },
    [closeSearch, goToNextResult, goToPreviousResult]
  )

  if (!searchOpen) {
    return null
  }

  const matchCount = searchResults.length
  const currentMatch = matchCount > 0 ? activeSearchIndex + 1 : 0

  return (
    <div className="absolute right-4 top-4 z-50 flex items-center gap-2 rounded-lg border border-border bg-surface p-2 shadow-lg">
      <div className="relative">
        <Input
          ref={inputRef as React.RefObject<any>}
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={t("option:documentWorkspace.searchPlaceholder", "Search in document...")}
          size="small"
          className="w-48 pr-16"
          suffix={
            isIndexing ? (
              <Spin size="small" />
            ) : matchCount > 0 ? (
              <span className="text-xs text-muted">
                {currentMatch} / {matchCount}
              </span>
            ) : indexFailed ? (
              <span
                className="text-xs text-danger"
                title={t(
                  "option:documentWorkspace.searchIndexFailedHint",
                  "Could not index this PDF. Close and reopen search to retry."
                )}
              >
                {t("option:documentWorkspace.searchIndexFailed", "Unavailable")}
              </span>
            ) : searchQuery ? (
              <span className="text-xs text-muted">0 / 0</span>
            ) : null
          }
        />
      </div>

      {/* Search options */}
      <div className="flex items-center gap-1">
        <Tooltip title={t("option:documentWorkspace.matchCase", "Match case (Aa)")}>
          <button
            type="button"
            onClick={() => setSearchMatchCase(!searchMatchCase)}
            className={`rounded px-1.5 py-1 text-xs font-medium transition-colors ${
              searchMatchCase
                ? "bg-primary/20 text-primary"
                : "text-text-muted hover:text-text hover:bg-hover"
            }`}
            aria-label={t("option:documentWorkspace.matchCase", "Match case")}
            aria-pressed={searchMatchCase}
          >
            Aa
          </button>
        </Tooltip>
        <Tooltip title={t("option:documentWorkspace.wordBoundary", "Match whole words")}>
          <button
            type="button"
            onClick={() => setSearchWordBoundary(!searchWordBoundary)}
            className={`rounded px-1.5 py-1 text-xs font-bold transition-colors ${
              searchWordBoundary
                ? "bg-primary/20 text-primary"
                : "text-text-muted hover:text-text hover:bg-hover"
            }`}
            aria-label={t("option:documentWorkspace.wordBoundary", "Match whole words")}
            aria-pressed={searchWordBoundary}
          >
            \b
          </button>
        </Tooltip>
      </div>

      <div className="flex items-center gap-1">
        <Button
          type="text"
          size="small"
          icon={<ChevronUp className="h-4 w-4" />}
          onClick={goToPreviousResult}
          disabled={matchCount === 0}
          title={t("option:documentWorkspace.previousMatch", "Previous (Shift+Enter)")}
          aria-label={t("option:documentWorkspace.previousMatch", "Previous (Shift+Enter)")}
        />
        <Button
          type="text"
          size="small"
          icon={<ChevronDown className="h-4 w-4" />}
          onClick={goToNextResult}
          disabled={matchCount === 0}
          title={t("option:documentWorkspace.nextMatch", "Next (Enter)")}
          aria-label={t("option:documentWorkspace.nextMatch", "Next (Enter)")}
        />
        <Button
          type="text"
          size="small"
          icon={<X className="h-4 w-4" />}
          onClick={closeSearch}
          title={t("common:close", "Close (Escape)")}
          aria-label={t("common:close", "Close (Escape)")}
        />
        {annotationMatches > 0 && (
          <button
            type="button"
            onClick={() => setActiveRightTab("annotations")}
            className="rounded-full bg-primary/10 px-2 py-0.5 text-xs text-primary hover:bg-primary/20 transition-colors"
          >
            {annotationMatches} in notes
          </button>
        )}
      </div>
    </div>
  )
}

export default PdfSearch
