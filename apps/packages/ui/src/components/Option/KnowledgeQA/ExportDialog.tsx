/**
 * ExportDialog - Export conversations as markdown/PDF with citations
 */

import React, { useState, useCallback, useEffect, useRef } from "react"
import {
  Download,
  FileText,
  FileDown,
  Book,
  X,
  Loader2,
  Check,
  Copy,
} from "lucide-react"
import { useKnowledgeQA } from "./KnowledgeQAProvider"
import { cn } from "@/libs/utils"
import type {
  CitationRef,
  ExportFormat,
  ExportOptions,
  RagResult,
  SearchRuntimeDetails,
} from "./types"
import type {
  RagCitationStyle,
  RagPresetName,
  RagSettings,
} from "@/services/rag/unified-rag"
import { useAntdMessage } from "@/hooks/useAntdMessage"
import { mapKnowledgeQaExportErrorMessage } from "./errorMessages"
import { tldwClient } from "@/services/tldw/TldwApiClient"

type ExportDialogProps = {
  open: boolean
  onClose: () => void
  className?: string
}

const DEFAULT_OPTIONS: ExportOptions = {
  format: "markdown",
  includeSettingsSnapshot: false,
  includeSourceExcerpts: true,
  citationStyle: "apa",
}

const CITATION_APPROXIMATION_NOTE =
  "Citation formatting is approximate and may omit author, year, or publisher fields when metadata is unavailable."
const SHARE_THREAD_LINKS_ENABLED = true
const SHARE_THREAD_LINKS_HELP_TEXT =
  "Share links use a dedicated token with read-only access and an expiry window, independent from normal thread permissions. Extension shared-link routing is currently TBD."
const SENSITIVE_EXPORT_KEY_PATTERN =
  /(?:api[_-]?key|authorization|bearer|credential|password|secret|token)/i

type ExportContext = {
  citations?: CitationRef[]
  settings?: Partial<RagSettings> | null
  preset?: RagPresetName | string | null
  searchDetails?: Partial<SearchRuntimeDetails> | null
}

function getFocusableElements(container: HTMLElement): HTMLElement[] {
  const focusableSelectors = [
    'button:not([disabled])',
    '[href]',
    'input:not([disabled])',
    'select:not([disabled])',
    'textarea:not([disabled])',
    '[tabindex]:not([tabindex="-1"])',
  ].join(",")

  return Array.from(container.querySelectorAll(focusableSelectors)) as HTMLElement[]
}

export function ExportDialog({ open, onClose, className }: ExportDialogProps) {
  const {
    messages,
    currentThreadId,
    results,
    citations,
    answer,
    query,
    settings,
    preset,
    searchDetails,
  } = useKnowledgeQA()
  const message = useAntdMessage()
  const [options, setOptions] = useState<ExportOptions>(DEFAULT_OPTIONS)
  const [isExporting, setIsExporting] = useState(false)
  const [isSavingNote, setIsSavingNote] = useState(false)
  const [exportedContent, setExportedContent] = useState<string | null>(null)
  const [exportError, setExportError] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)
  const [shareLinkCopied, setShareLinkCopied] = useState(false)
  const [isPreparingShareLink, setIsPreparingShareLink] = useState(false)
  const [isRevokingShareLink, setIsRevokingShareLink] = useState(false)
  const [activeShareLink, setActiveShareLink] = useState<{
    id: string
    sharePath: string
    expiresAt: string
  } | null>(null)
  const panelRef = useRef<HTMLDivElement>(null)
  const previousActiveElement = useRef<HTMLElement | null>(null)
  const pendingPrintTimeoutRef = useRef<number | null>(null)
  const copiedTimeoutRef = useRef<number | null>(null)
  const shareLinkCopiedTimeoutRef = useRef<number | null>(null)
  const activeDialogSessionKeyRef = useRef("")
  const dialogSessionNonceRef = useRef(open ? 1 : 0)
  const wasOpenRef = useRef(open)
  const hasExportableContent =
    query.trim().length > 0 || Boolean(answer) || results.length > 0 || messages.length > 0
  const hasServerThread = Boolean(
    currentThreadId && !currentThreadId.startsWith("local-")
  )
  const canCopyThreadLink = SHARE_THREAD_LINKS_ENABLED && hasServerThread
  if (open && !wasOpenRef.current) {
    dialogSessionNonceRef.current += 1
  }
  wasOpenRef.current = open
  const dialogSessionKey = `${dialogSessionNonceRef.current}::${open ? "open" : "closed"}::${currentThreadId ?? "no-thread"}`
  activeDialogSessionKeyRef.current = dialogSessionKey

  const clearCopiedTimeout = useCallback(() => {
    if (copiedTimeoutRef.current != null) {
      window.clearTimeout(copiedTimeoutRef.current)
      copiedTimeoutRef.current = null
    }
  }, [])

  const clearShareLinkCopiedTimeout = useCallback(() => {
    if (shareLinkCopiedTimeoutRef.current != null) {
      window.clearTimeout(shareLinkCopiedTimeoutRef.current)
      shareLinkCopiedTimeoutRef.current = null
    }
  }, [])

  useEffect(() => {
    clearCopiedTimeout()
    clearShareLinkCopiedTimeout()
    if (pendingPrintTimeoutRef.current != null) {
      window.clearTimeout(pendingPrintTimeoutRef.current)
      pendingPrintTimeoutRef.current = null
    }
    setIsExporting(false)
    setIsSavingNote(false)
    setExportedContent(null)
    setExportError(null)
    setCopied(false)
    setActiveShareLink(null)
    setShareLinkCopied(false)
    setIsPreparingShareLink(false)
    setIsRevokingShareLink(false)
  }, [clearCopiedTimeout, clearShareLinkCopiedTimeout, currentThreadId])

  const handleExport = useCallback(async () => {
    const requestSessionKey = dialogSessionKey
    setIsExporting(true)
    setExportedContent(null)
    setExportError(null)

    try {
      let content = ""

      if (options.format === "markdown") {
        content = generateMarkdown(
          query,
          answer,
          results,
          messages,
          options,
          { citations, settings, preset, searchDetails }
        )
        if (activeDialogSessionKeyRef.current !== requestSessionKey) {
          return
        }
        setExportedContent(content)
      } else if (options.format === "pdf") {
        // For PDF, generate markdown first then trigger print
        content = generateMarkdown(
          query,
          answer,
          results,
          messages,
          options,
          { citations, settings, preset, searchDetails }
        )
        if (activeDialogSessionKeyRef.current !== requestSessionKey) {
          return
        }
        setExportedContent(content)
        // Trigger print after a short delay to allow content to render
        if (pendingPrintTimeoutRef.current != null) {
          window.clearTimeout(pendingPrintTimeoutRef.current)
        }
        pendingPrintTimeoutRef.current = window.setTimeout(() => {
          pendingPrintTimeoutRef.current = null
          window.print()
        }, 500)
      } else if (options.format === "chatbook") {
        if (!currentThreadId || currentThreadId.startsWith("local-")) {
          throw new Error("No server-backed thread selected for export.")
        }

        const trimmedQuery = query.trim()
        const normalizedQuery =
          trimmedQuery.length > 96 ? `${trimmedQuery.slice(0, 93)}...` : trimmedQuery
        const exportResult = await tldwClient.exportChatbook({
          name: normalizedQuery
            ? `Knowledge QA: ${normalizedQuery}`
            : `Knowledge QA Export ${new Date().toLocaleDateString()}`,
          description: normalizedQuery
            ? `Knowledge QA export for query: ${normalizedQuery}`
            : "Knowledge QA chatbook export",
          content_selections: {
            conversation: [currentThreadId],
          },
          include_media: true,
          include_embeddings: false,
          include_generated_content: true,
          async_mode: false,
        })

        if (exportResult?.success === false) {
          throw new Error(String(exportResult?.message || "Export failed"))
        }
        if (activeDialogSessionKeyRef.current !== requestSessionKey) {
          return
        }

        const jobId = resolveChatbookJobId(exportResult)
        if (!jobId) {
          throw new Error("Chatbook export completed but no download job was returned.")
        }

        const { blob, filename } = await tldwClient.downloadChatbookExport(jobId)
        if (activeDialogSessionKeyRef.current !== requestSessionKey) {
          return
        }
        downloadBlob(blob, filename || `knowledge_qa_${Date.now()}.zip`)
        onClose()
        return
      }
    } catch (error) {
      if (activeDialogSessionKeyRef.current !== requestSessionKey) {
        return
      }
      const mappedError =
        options.format === "chatbook"
          ? mapKnowledgeQaExportErrorMessage(error)
          : error instanceof Error
            ? error.message
            : "Export failed"
      setExportError(mappedError)
      message.open({
        type: "error",
        content: mappedError,
        duration: 4,
      })
      console.error("Export failed:", error)
    } finally {
      if (activeDialogSessionKeyRef.current === requestSessionKey) {
        setIsExporting(false)
      }
    }
  }, [
    dialogSessionKey,
    options,
    query,
    answer,
    results,
    messages,
    citations,
    settings,
    preset,
    searchDetails,
    currentThreadId,
    onClose,
    message,
  ])

  const handleDownload = useCallback(() => {
    if (!exportedContent) return

    const blob = new Blob([exportedContent], { type: "text/markdown" })
    downloadBlob(blob, `knowledge_qa_${Date.now()}.md`)
  }, [exportedContent])

  const handleCopy = useCallback(async () => {
    if (!exportedContent) return
    const requestSessionKey = dialogSessionKey

    try {
      await navigator.clipboard.writeText(exportedContent)
      if (activeDialogSessionKeyRef.current !== requestSessionKey) {
        return
      }
      clearCopiedTimeout()
      setCopied(true)
      copiedTimeoutRef.current = window.setTimeout(() => {
        if (activeDialogSessionKeyRef.current !== requestSessionKey) {
          return
        }
        setCopied(false)
        copiedTimeoutRef.current = null
      }, 2000)
    } catch (error) {
      if (activeDialogSessionKeyRef.current !== requestSessionKey) {
        return
      }
      console.error("Copy failed:", error)
    }
  }, [clearCopiedTimeout, dialogSessionKey, exportedContent])

  const handleSaveToNotes = useCallback(async () => {
    if (!hasExportableContent) return
    const requestSessionKey = dialogSessionKey

    setIsSavingNote(true)
    try {
      const noteContent = generateMarkdown(
        query,
        answer,
        results,
        messages,
        { ...options, format: "markdown" },
        { citations, settings, preset, searchDetails }
      )
      const trimmedQuery = query.trim()
      const title =
        trimmedQuery.length > 0
          ? `Knowledge QA: ${
              trimmedQuery.length > 72
                ? `${trimmedQuery.slice(0, 69)}...`
                : trimmedQuery
            }`
          : "Knowledge QA export"
      const metadata: Record<string, unknown> = {
        origin: "knowledge_qa",
        source: "knowledge_export",
        citation_style: options.citationStyle,
        include_source_excerpts: options.includeSourceExcerpts,
        include_settings_snapshot: options.includeSettingsSnapshot,
      }
      if (currentThreadId) {
        metadata.thread_id = currentThreadId
      }

      await tldwClient.createNote(noteContent, {
        title,
        metadata,
      })
      if (activeDialogSessionKeyRef.current !== requestSessionKey) {
        return
      }
      message.open({
        type: "success",
        content: "Saved to Notes.",
        duration: 3,
      })
    } catch (error) {
      if (activeDialogSessionKeyRef.current !== requestSessionKey) {
        return
      }
      const mappedError =
        error instanceof Error && error.message
          ? `Failed to save to Notes. ${error.message}`
          : "Failed to save to Notes."
      message.open({
        type: "error",
        content: mappedError,
        duration: 4,
      })
    } finally {
      if (activeDialogSessionKeyRef.current === requestSessionKey) {
        setIsSavingNote(false)
      }
    }
  }, [
    dialogSessionKey,
    hasExportableContent,
    query,
    answer,
    results,
    messages,
    options,
    citations,
    settings,
    preset,
    searchDetails,
    currentThreadId,
    message,
  ])

  const handleCopyThreadLink = useCallback(async () => {
    if (!canCopyThreadLink || !currentThreadId) return

    const requestSessionKey = dialogSessionKey
    setIsPreparingShareLink(true)
    try {
      const shareLink = await tldwClient.createConversationShareLink(currentThreadId, {
        permission: "view",
      })
      if (activeDialogSessionKeyRef.current !== requestSessionKey) {
        return
      }
      const sharePath =
        typeof shareLink?.share_path === "string" && shareLink.share_path.trim().length > 0
          ? shareLink.share_path.trim()
          : `/knowledge/shared/${encodeURIComponent(String(shareLink?.token || ""))}`
      const shareUrl = `${window.location.origin}${sharePath}`
      const nextShareLink = {
        id: String(shareLink.share_id),
        sharePath,
        expiresAt: String(shareLink.expires_at),
      }
      setActiveShareLink(nextShareLink)
      clearShareLinkCopiedTimeout()
      setShareLinkCopied(false)
      try {
        await navigator.clipboard.writeText(shareUrl)
      } catch (error) {
        if (activeDialogSessionKeyRef.current !== requestSessionKey) {
          return
        }
        message.open({
          type: "error",
          content: "Unable to copy share link, but the link remains active.",
          duration: 4,
        })
        console.error("Share link clipboard copy failed:", error)
        return
      }
      if (activeDialogSessionKeyRef.current !== requestSessionKey) {
        return
      }
      setShareLinkCopied(true)
      message.open({
        type: "success",
        content: "Share link copied.",
        duration: 3,
      })
      shareLinkCopiedTimeoutRef.current = window.setTimeout(() => {
        setShareLinkCopied(false)
        shareLinkCopiedTimeoutRef.current = null
      }, 2000)
    } catch (error) {
      if (activeDialogSessionKeyRef.current !== requestSessionKey) {
        return
      }
      message.open({
        type: "error",
        content: "Unable to generate a share link.",
        duration: 4,
      })
      console.error("Share link copy failed:", error)
    } finally {
      if (activeDialogSessionKeyRef.current === requestSessionKey) {
        setIsPreparingShareLink(false)
      }
    }
  }, [
    canCopyThreadLink,
    clearShareLinkCopiedTimeout,
    currentThreadId,
    dialogSessionKey,
    message,
  ])

  const handleRevokeShareLink = useCallback(async () => {
    if (!canCopyThreadLink || !currentThreadId || !activeShareLink?.id) return
    const requestSessionKey = dialogSessionKey
    setIsRevokingShareLink(true)
    try {
      await tldwClient.revokeConversationShareLink(currentThreadId, activeShareLink.id)
      if (activeDialogSessionKeyRef.current !== requestSessionKey) {
        return
      }
      setActiveShareLink(null)
      message.open({
        type: "success",
        content: "Share link revoked.",
        duration: 3,
      })
    } catch (error) {
      if (activeDialogSessionKeyRef.current !== requestSessionKey) {
        return
      }
      message.open({
        type: "error",
        content: "Unable to revoke share link.",
        duration: 4,
      })
      console.error("Share link revoke failed:", error)
    } finally {
      if (activeDialogSessionKeyRef.current === requestSessionKey) {
        setIsRevokingShareLink(false)
      }
    }
  }, [
    activeShareLink?.id,
    canCopyThreadLink,
    currentThreadId,
    dialogSessionKey,
    message,
  ])

  useEffect(() => {
    if (open) {
      previousActiveElement.current = document.activeElement as HTMLElement
    }
  }, [open])

  useEffect(() => {
    if (open) return
    clearCopiedTimeout()
    clearShareLinkCopiedTimeout()
    if (pendingPrintTimeoutRef.current != null) {
      window.clearTimeout(pendingPrintTimeoutRef.current)
      pendingPrintTimeoutRef.current = null
    }
    setIsExporting(false)
    setIsSavingNote(false)
    setExportedContent(null)
    setExportError(null)
    setCopied(false)
    setShareLinkCopied(false)
    setIsPreparingShareLink(false)
    setIsRevokingShareLink(false)
    setActiveShareLink(null)
  }, [clearCopiedTimeout, clearShareLinkCopiedTimeout, open])

  useEffect(
    () => () => {
      clearCopiedTimeout()
      clearShareLinkCopiedTimeout()
      if (pendingPrintTimeoutRef.current != null) {
        window.clearTimeout(pendingPrintTimeoutRef.current)
        pendingPrintTimeoutRef.current = null
      }
    },
    [clearCopiedTimeout, clearShareLinkCopiedTimeout]
  )

  useEffect(() => {
    if (!open || !panelRef.current) return

    const panel = panelRef.current
    const focusableElements = getFocusableElements(panel)
    if (focusableElements.length > 0) {
      focusableElements[0].focus()
    }

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault()
        onClose()
        return
      }

      if (e.key === "Tab") {
        const focusable = getFocusableElements(panel)
        if (focusable.length === 0) return

        const first = focusable[0]
        const last = focusable[focusable.length - 1]

        if (e.shiftKey) {
          if (document.activeElement === first) {
            e.preventDefault()
            last.focus()
          }
        } else if (document.activeElement === last) {
          e.preventDefault()
          first.focus()
        }
      }
    }

    document.addEventListener("keydown", handleKeyDown)
    return () => document.removeEventListener("keydown", handleKeyDown)
  }, [open, onClose])

  useEffect(() => {
    if (!open && previousActiveElement.current) {
      previousActiveElement.current.focus()
      previousActiveElement.current = null
    }
  }, [open])

  if (!open) return null

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/50 z-50"
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Dialog */}
      <div
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="export-dialog-title"
        className={cn(
          "fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2",
          "w-full max-w-lg max-h-[90vh]",
          "bg-surface rounded-xl shadow-xl border border-border",
          "flex flex-col z-50",
          className
        )}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-border">
          <div className="flex items-center gap-2">
            <Download className="w-5 h-5 text-primary" />
            <h2 id="export-dialog-title" className="font-semibold text-lg">
              Export Conversation
            </h2>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close export dialog"
            className="p-1.5 rounded-lg hover:bg-muted transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {/* Format selection */}
          <div className="space-y-3">
            <label className="text-sm font-medium">Export Format</label>
            <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
              {[
                {
                  value: "markdown",
                  label: "Markdown",
                  icon: FileText,
                  description: "Plain text with formatting, ideal for notes and documentation",
                },
                {
                  value: "pdf",
                  label: "PDF",
                  icon: FileDown,
                  description: "Print-ready document, opens your browser's print dialog",
                },
                {
                  value: "chatbook",
                  label: "Chatbook",
                  icon: Book,
                  description: "Portable format for sharing with the tldw community",
                },
              ].map((fmt) => {
                const Icon = fmt.icon
                const isSelected = options.format === fmt.value
                return (
                  <button
                    key={fmt.value}
                    onClick={() =>
                      setOptions((prev) => ({
                        ...prev,
                        format: fmt.value as ExportFormat,
                      }))
                    }
                    aria-pressed={isSelected}
                    className={cn(
                      "flex flex-col items-center gap-2 p-4 rounded-lg border transition-all text-center",
                      isSelected
                        ? "border-primary bg-primary/5 ring-1 ring-primary"
                        : "border-border hover:border-primary/30"
                    )}
                  >
                    <Icon
                      className={cn(
                        "w-6 h-6",
                        isSelected ? "text-primary" : "text-text-muted"
                      )}
                    />
                    <span className="text-sm font-medium">{fmt.label}</span>
                    <span className="text-[10px] text-text-muted leading-tight">
                      {fmt.description}
                    </span>
                  </button>
                )
              })}
            </div>
          </div>

          {/* Options */}
          {options.format !== "chatbook" && (
            <>
              {/* Citation style */}
              <div className="space-y-2">
                <label className="text-sm font-medium">Citation Style</label>
                <select
                  value={options.citationStyle}
                  onChange={(e) =>
                    setOptions((prev) => ({
                      ...prev,
                      citationStyle: e.target.value as RagCitationStyle,
                    }))
                  }
                  className="w-full px-3 py-2 rounded-md border border-border bg-surface focus:outline-none focus:ring-2 focus:ring-primary"
                >
                  <option value="apa">APA</option>
                  <option value="mla">MLA</option>
                  <option value="chicago">Chicago</option>
                  <option value="harvard">Harvard</option>
                  <option value="ieee">IEEE</option>
                </select>
                <p className="text-xs text-text-muted">
                  {CITATION_APPROXIMATION_NOTE}
                </p>
              </div>

              {/* Include options */}
              <div className="space-y-3">
                <label className="text-sm font-medium">Include</label>
                <div className="space-y-2">
                  <label className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={options.includeSourceExcerpts}
                      onChange={(e) =>
                        setOptions((prev) => ({
                          ...prev,
                          includeSourceExcerpts: e.target.checked,
                        }))
                      }
                      className="rounded"
                    />
                    <span className="text-sm">Source excerpts</span>
                  </label>
                  <label className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={options.includeSettingsSnapshot}
                      onChange={(e) =>
                        setOptions((prev) => ({
                          ...prev,
                          includeSettingsSnapshot: e.target.checked,
                        }))
                      }
                      className="rounded"
                    />
                    <span className="text-sm">Settings snapshot</span>
                  </label>
                </div>
              </div>
            </>
          )}

          <div className="space-y-2 rounded-lg border border-border bg-muted/20 px-3 py-3">
            <p className="text-sm font-medium">Workflow actions</p>
            <div className="flex flex-wrap items-center gap-2">
              <button
                type="button"
                onClick={handleSaveToNotes}
                disabled={!hasExportableContent || isSavingNote}
                className="px-3 py-1.5 text-xs font-medium rounded-md border border-border hover:bg-muted transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isSavingNote ? "Saving..." : "Save to Notes"}
              </button>
              <button
                type="button"
                onClick={handleCopyThreadLink}
                disabled={!canCopyThreadLink || isPreparingShareLink}
                title={SHARE_THREAD_LINKS_HELP_TEXT}
                className="px-3 py-1.5 text-xs font-medium rounded-md border border-border hover:bg-muted transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isPreparingShareLink
                  ? "Generating link..."
                  : shareLinkCopied
                    ? "Link copied"
                    : "Create share link"}
              </button>
              <button
                type="button"
                onClick={handleRevokeShareLink}
                disabled={!activeShareLink || isRevokingShareLink}
                className="px-3 py-1.5 text-xs font-medium rounded-md border border-border hover:bg-muted transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isRevokingShareLink ? "Revoking..." : "Revoke link"}
              </button>
            </div>
            <p className="text-xs text-text-muted">
              {SHARE_THREAD_LINKS_HELP_TEXT}
            </p>
            {activeShareLink ? (
              <p className="text-xs text-text-muted">
                Active link expires{" "}
                {new Date(activeShareLink.expiresAt).toLocaleString([], {
                  month: "short",
                  day: "numeric",
                  hour: "numeric",
                  minute: "2-digit",
                })}
                .
              </p>
            ) : null}
          </div>

          {/* Preview / Result */}
          {exportedContent && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium">Preview</label>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={handleCopy}
                    className="flex items-center gap-1.5 px-2 py-1 text-xs rounded hover:bg-muted transition-colors"
                  >
                    {copied ? (
                      <>
                        <Check className="w-3.5 h-3.5" />
                        Copied
                      </>
                    ) : (
                      <>
                        <Copy className="w-3.5 h-3.5" />
                        Copy
                      </>
                    )}
                  </button>
                  <button
                    type="button"
                    onClick={handleDownload}
                    className="flex items-center gap-1.5 px-2 py-1 text-xs rounded hover:bg-muted transition-colors"
                  >
                    <Download className="w-3.5 h-3.5" />
                    Download
                  </button>
                </div>
              </div>
              <pre className="p-4 bg-muted rounded-lg text-xs overflow-auto max-h-60 font-mono">
                {exportedContent.slice(0, 2000)}
                {exportedContent.length > 2000 && "\n\n... (truncated)"}
              </pre>
            </div>
          )}

          {exportError && (
            <div className="rounded-lg border border-danger/30 bg-danger/10 px-3 py-2">
              <p className="text-sm text-danger">{exportError}</p>
              <button
                type="button"
                onClick={handleExport}
                disabled={isExporting}
                className="mt-2 text-xs font-medium text-danger hover:opacity-80 disabled:opacity-60"
              >
                Retry export
              </button>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 px-6 py-4 border-t border-border">
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 text-sm rounded-md hover:bg-muted transition-colors"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={handleExport}
            disabled={isExporting}
            className="flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-md bg-primary text-white hover:bg-primaryStrong transition-colors disabled:opacity-50"
          >
            {isExporting ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Exporting...
              </>
            ) : (
              <>
                <Download className="w-4 h-4" />
                Export
              </>
            )}
          </button>
        </div>
      </div>
    </>
  )
}

function resolveChatbookJobId(payload: any): string | null {
  const directJobId =
    typeof payload?.job_id === "string" && payload.job_id.trim().length > 0
      ? payload.job_id.trim()
      : null
  if (directJobId) return directJobId

  const downloadUrl =
    typeof payload?.download_url === "string" && payload.download_url.trim().length > 0
      ? payload.download_url.trim()
      : null
  if (!downloadUrl) return null

  const normalizedPath = downloadUrl.split("?")[0].replace(/\/+$/, "")
  const segments = normalizedPath.split("/").filter(Boolean)
  if (segments.length === 0) return null

  const rawJobId = segments[segments.length - 1] || ""
  try {
    return decodeURIComponent(rawJobId)
  } catch {
    return rawJobId
  }
}

// Helper function to download a blob
function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement("a")
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

function getSourceTitle(result: RagResult | undefined, index: number): string {
  return (
    result?.metadata?.title ||
    result?.metadata?.source ||
    `Source ${index + 1}`
  )
}

function collectCitationIndexes(
  citations: CitationRef[] | undefined,
  answer: string | null
): number[] {
  const indexes: number[] = []
  const seen = new Set<number>()
  const pushIndex = (value: unknown) => {
    if (typeof value !== "number" || !Number.isInteger(value) || value <= 0) {
      return
    }
    if (seen.has(value)) return
    seen.add(value)
    indexes.push(value)
  }

  citations?.forEach((citation) => pushIndex(citation.index))

  if (indexes.length === 0 && answer) {
    for (const match of answer.matchAll(/\[(\d+)\]/g)) {
      pushIndex(Number(match[1]))
    }
  }

  return indexes
}

function sanitizeExportValue(value: unknown, depth = 0): unknown {
  if (value == null) return value
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return value
  }
  if (Array.isArray(value)) {
    return value
      .map((item) => sanitizeExportValue(item, depth + 1))
      .filter((item) => item !== undefined)
  }
  if (typeof value === "object" && depth < 5) {
    const output: Record<string, unknown> = {}
    Object.entries(value as Record<string, unknown>).forEach(([key, rawValue]) => {
      if (SENSITIVE_EXPORT_KEY_PATTERN.test(key)) return
      const sanitized = sanitizeExportValue(rawValue, depth + 1)
      if (sanitized !== undefined) {
        output[key] = sanitized
      }
    })
    return output
  }

  return undefined
}

function hasSnapshotContent(value: unknown): value is Record<string, unknown> {
  return (
    value != null &&
    typeof value === "object" &&
    !Array.isArray(value) &&
    Object.keys(value).length > 0
  )
}

function buildExportSettingsSnapshot(
  options: ExportOptions,
  context?: ExportContext
): Record<string, unknown> {
  const snapshot: Record<string, unknown> = {
    format: options.format,
    citationStyle: options.citationStyle,
    includeSourceExcerpts: options.includeSourceExcerpts,
    exportDate: new Date().toISOString(),
  }

  if (context?.preset) {
    snapshot.preset = context.preset
  }

  const settingsSnapshot = sanitizeExportValue(context?.settings)
  if (hasSnapshotContent(settingsSnapshot)) {
    snapshot.settings = settingsSnapshot
  }

  const searchDetailsSnapshot = sanitizeExportValue(context?.searchDetails)
  if (hasSnapshotContent(searchDetailsSnapshot)) {
    snapshot.searchDetails = searchDetailsSnapshot
  }

  return snapshot
}

// Generate markdown content
function generateMarkdown(
  query: string,
  answer: string | null,
  results: RagResult[],
  messages: Array<{
    role: string
    content: string
    timestamp?: string
    ragContext?: {
      retrieved_documents?: Array<{
        title?: string
        excerpt?: string
        url?: string
      }>
    }
  }>,
  options: ExportOptions,
  context?: ExportContext
): string {
  const lines: string[] = []
  const citationIndexes = collectCitationIndexes(context?.citations, answer)

  // Header
  lines.push("# Knowledge QA Export")
  lines.push("")
  lines.push(`Generated: ${new Date().toLocaleString()}`)
  lines.push("")

  // Query
  lines.push("## Query")
  lines.push("")
  lines.push(`> ${query}`)
  lines.push("")

  // Answer
  if (answer) {
    lines.push("## Answer")
    lines.push("")
    lines.push(answer)
    lines.push("")
  }

  // Citation mappings
  if (citationIndexes.length > 0) {
    lines.push("## Citations")
    lines.push("")
    citationIndexes.forEach((citationIndex) => {
      const sourceIndex = citationIndex - 1
      const result = results[sourceIndex]
      const title = getSourceTitle(result, sourceIndex)

      if (!result) {
        lines.push(`- [${citationIndex}] Source unavailable in exported results.`)
        return
      }

      lines.push(`- [${citationIndex}] ${title} maps to Source ${sourceIndex + 1}.`)
      if (result.metadata?.page_number) {
        lines.push(`  Page: ${result.metadata.page_number}`)
      }
      if (result.metadata?.url) {
        lines.push(`  URL: ${result.metadata.url}`)
      }
    })
    lines.push("")
  }

  // Sources
  if (results.length > 0) {
    lines.push("## Sources")
    lines.push("")
    results.forEach((result, index) => {
      const title = getSourceTitle(result, index)
      const url = result.metadata?.url
      const score = result.score
      const content = result.content || result.text || ""

      lines.push(`### [${index + 1}] ${title}`)
      lines.push("")
      if (url) {
        lines.push(`URL: ${url}`)
        lines.push("")
      }
      if (score !== undefined) {
        lines.push(`Relevance: ${Math.round(score * 100)}%`)
        lines.push("")
      }
      if (options.includeSourceExcerpts && content) {
        lines.push("**Excerpt:**")
        lines.push("")
        lines.push(`> ${content.slice(0, 500)}${content.length > 500 ? "..." : ""}`)
        lines.push("")
      }
    })
  }

  // Conversation history
  if (messages.length > 0) {
    lines.push("## Conversation History")
    lines.push("")
    messages.forEach((msg) => {
      const role = msg.role === "user" ? "User" : msg.role === "assistant" ? "Assistant" : msg.role
      lines.push(`**${role}:** ${msg.content}`)
      lines.push("")
    })
  }

  // Bibliography (formatted citations)
  if (results.length > 0) {
    lines.push("## Bibliography")
    lines.push("")
    lines.push(
      "_Citation formatting is approximate and may omit author, year, or publisher details when metadata is unavailable._"
    )
    lines.push("")
    results.forEach((result, index) => {
      const citation = formatCitation(result, index + 1, options.citationStyle)
      lines.push(citation)
      lines.push("")
    })
  }

  // Settings snapshot
  if (options.includeSettingsSnapshot) {
    lines.push("## Settings Used")
    lines.push("")
    lines.push("```json")
    lines.push(
      JSON.stringify(buildExportSettingsSnapshot(options, context), null, 2)
    )
    lines.push("```")
  }

  // Footer
  lines.push("")
  lines.push("---")
  lines.push("*Exported from Knowledge QA*")

  return lines.join("\n")
}

// Format a citation based on style
function formatCitation(
  result: {
    metadata?: {
      title?: string
      source?: string
      url?: string
      page_number?: number
    }
  },
  index: number,
  style: RagCitationStyle
): string {
  const title = result.metadata?.title || result.metadata?.source || "Untitled"
  const url = result.metadata?.url || ""
  const page = result.metadata?.page_number

  switch (style) {
    case "apa":
      return `[${index}] ${title}. ${url ? `Retrieved from ${url}` : ""}${page ? `, p. ${page}` : ""}`
    case "mla":
      return `[${index}] "${title}."${url ? ` Web. <${url}>.` : ""}${page ? ` ${page}.` : ""}`
    case "chicago":
      return `[${index}] ${title}.${url ? ` ${url}.` : ""}${page ? ` ${page}.` : ""}`
    case "harvard":
      return `[${index}] ${title}${url ? ` Available at: ${url}` : ""}${page ? ` (p. ${page})` : ""}`
    case "ieee":
      return `[${index}] ${title}${url ? `, [Online]. Available: ${url}` : ""}${page ? `, p. ${page}` : ""}.`
    default:
      return `[${index}] ${title}${url ? ` - ${url}` : ""}${page ? ` (p. ${page})` : ""}`
  }
}
