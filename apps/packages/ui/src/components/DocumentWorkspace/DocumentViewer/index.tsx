import React, { Suspense, useCallback, useEffect, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { Button, Spin } from "antd"
import { FileText, AlertCircle, Highlighter, MessageSquare, Lightbulb, HelpCircle, X } from "lucide-react"
import type { PdfDocumentProxy } from "@/hooks/document-workspace/usePdfSearch"
import { useDocumentWorkspaceStore } from "@/store/document-workspace"
import { ViewerToolbar } from "./ViewerToolbar"
import type { DocumentType } from "../types"

const PdfDocument = React.lazy(() =>
  import("./PdfViewer/PdfDocument").then((module) => ({
    default: module.PdfDocument
  }))
)

const PdfSearch = React.lazy(() =>
  import("./PdfSearch").then((module) => ({
    default: module.PdfSearch
  }))
)

const EpubViewer = React.lazy(() =>
  import("./EpubViewer").then((module) => ({
    default: module.EpubViewer
  }))
)

const viewerFallback = (
  <div className="flex h-full items-center justify-center">
    <Spin size="large" />
  </div>
)

interface DocumentViewerProps {
  className?: string
  loadingDocumentId?: number | null
  onOpenLibrary?: () => void
  onOpenUpload?: () => void
  onReloadDocument?: (mediaId: number, docTypeHint?: DocumentType | null) => Promise<void> | void
}

const shouldRetryBlobLoad = (error: Error, url?: string): boolean => {
  if (!url || !url.startsWith("blob:")) return false
  const message = String(error?.message || "")
  const status = (error as Error & { status?: number })?.status
  if (status === 0) return true
  if ((error as Error & { name?: string })?.name === "UnexpectedResponseException") return true
  return [
    "Unexpected server response",
    "ERR_FILE_NOT_FOUND",
    "Failed to fetch",
    "NetworkError",
    "Invalid name"
  ].some((needle) => message.includes(needle))
}

export const DocumentViewer: React.FC<DocumentViewerProps> = ({
  className,
  loadingDocumentId,
  onOpenLibrary,
  onOpenUpload,
  onReloadDocument
}) => {
  const { t } = useTranslation(["option", "common"])
  const pdfDocumentRef = useRef<PdfDocumentProxy | null>(null)
  const reloadAttemptsRef = useRef<Map<number, number>>(new Map())
  const reloadInFlightRef = useRef<Set<number>>(new Set())

  const activeDocumentId = useDocumentWorkspaceStore((s) => s.activeDocumentId)
  const activeDocumentType = useDocumentWorkspaceStore(
    (s) => s.activeDocumentType
  )
  const openDocuments = useDocumentWorkspaceStore((s) => s.openDocuments)
  const currentPage = useDocumentWorkspaceStore((s) => s.currentPage)
  const totalPages = useDocumentWorkspaceStore((s) => s.totalPages)
  const zoomLevel = useDocumentWorkspaceStore((s) => s.zoomLevel)
  const viewMode = useDocumentWorkspaceStore((s) => s.viewMode)

  const setCurrentPage = useDocumentWorkspaceStore((s) => s.setCurrentPage)
  const setTotalPages = useDocumentWorkspaceStore((s) => s.setTotalPages)
  const setZoomLevel = useDocumentWorkspaceStore((s) => s.setZoomLevel)
  const setViewMode = useDocumentWorkspaceStore((s) => s.setViewMode)
  const goToNextPage = useDocumentWorkspaceStore((s) => s.goToNextPage)
  const goToPreviousPage = useDocumentWorkspaceStore((s) => s.goToPreviousPage)
  const setSearchOpen = useDocumentWorkspaceStore((s) => s.setSearchOpen)
  const searchOpen = useDocumentWorkspaceStore((s) => s.searchOpen)
  const currentPercentage = useDocumentWorkspaceStore((s) => s.currentPercentage)
  const currentChapterTitle = useDocumentWorkspaceStore((s) => s.currentChapterTitle)

  const activeDocument = openDocuments.find((d) => d.id === activeDocumentId)

  // Keyboard navigation and search shortcut
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!activeDocumentId) return

      const mod = e.metaKey || e.ctrlKey

      // Cmd/Ctrl+F to open search (always handle, even in inputs)
      if (mod && e.key === "f") {
        e.preventDefault()
        setSearchOpen(true)
        return
      }

      // Cmd+G → focus page input for "go to page"
      if (mod && (e.key === "g" || e.key === "G")) {
        e.preventDefault()
        const pageInput = document.querySelector<HTMLInputElement>(
          '[data-testid="document-page-input"] input, [data-testid="document-page-input"]'
        )
        if (pageInput) {
          pageInput.focus()
          pageInput.select()
        }
        return
      }

      // Zoom shortcuts (work even in inputs)
      if (mod && (e.key === "=" || e.key === "+")) {
        e.preventDefault()
        setZoomLevel(Math.min(zoomLevel + 25, 400))
        return
      }
      if (mod && e.key === "-") {
        e.preventDefault()
        setZoomLevel(Math.max(zoomLevel - 25, 25))
        return
      }
      if (mod && e.key === "0") {
        e.preventDefault()
        setZoomLevel(100)
        return
      }

      // Don't handle navigation if focus is in an input
      if (
        document.activeElement?.tagName === "INPUT" ||
        document.activeElement?.tagName === "TEXTAREA"
      ) {
        return
      }

      // F → toggle fullscreen on the viewer
      if (e.key === "f" || e.key === "F") {
        if (!e.shiftKey && !mod) {
          e.preventDefault()
          if (document.fullscreenElement) {
            document.exitFullscreen()
          } else {
            const viewer = document.querySelector('[data-testid="document-viewer"]')
            viewer?.requestFullscreen?.()
          }
          return
        }
      }

      switch (e.key) {
        case "ArrowRight":
        case "PageDown":
          e.preventDefault()
          goToNextPage()
          break
        case "ArrowLeft":
        case "PageUp":
          e.preventDefault()
          goToPreviousPage()
          break
        case "Home":
          e.preventDefault()
          setCurrentPage(1)
          break
        case "End":
          e.preventDefault()
          if (totalPages > 0) {
            setCurrentPage(totalPages)
          }
          break
      }
    }

    window.addEventListener("keydown", handleKeyDown)
    return () => window.removeEventListener("keydown", handleKeyDown)
  }, [
    activeDocumentId,
    totalPages,
    zoomLevel,
    goToNextPage,
    goToPreviousPage,
    setCurrentPage,
    setSearchOpen,
    setZoomLevel
  ])

  const handleLoadSuccess = useCallback(
    (numPages: number) => {
      setTotalPages(numPages)
      if (activeDocumentId !== null) {
        reloadAttemptsRef.current.delete(activeDocumentId)
      }
    },
    [activeDocumentId, setTotalPages]
  )

  const handleLoadError = useCallback(
    (error: Error) => {
      console.error("Failed to load document:", error)
      if (!onReloadDocument) return
      if (activeDocumentType !== "pdf") return
      if (activeDocumentId === null) return
      if (!activeDocument) return
      const { url, type } = activeDocument
      if (!url) return
      if (!shouldRetryBlobLoad(error, url)) return
      const attempts = reloadAttemptsRef.current.get(activeDocumentId) ?? 0
      if (attempts >= 1) return
      if (reloadInFlightRef.current.has(activeDocumentId)) return
      reloadAttemptsRef.current.set(activeDocumentId, attempts + 1)
      reloadInFlightRef.current.add(activeDocumentId)
      Promise.resolve(onReloadDocument(activeDocumentId, type))
        .catch((reloadError) => {
          console.error("Failed to reload document:", reloadError)
          const prevAttempts = reloadAttemptsRef.current.get(activeDocumentId) ?? 0
          if (prevAttempts <= 1) {
            reloadAttemptsRef.current.delete(activeDocumentId)
          } else {
            reloadAttemptsRef.current.set(activeDocumentId, prevAttempts - 1)
          }
        })
        .finally(() => {
          reloadInFlightRef.current.delete(activeDocumentId)
        })
    },
    [activeDocument, activeDocumentId, activeDocumentType, onReloadDocument]
  )

  // Onboarding cards - dismissed via localStorage
  const ONBOARDING_KEY = "document-workspace-onboarding-dismissed"
  const [onboardingDismissed, setOnboardingDismissed] = useState(() => {
    try { return localStorage.getItem(ONBOARDING_KEY) === "true" } catch { return false }
  })

  const dismissOnboarding = useCallback(() => {
    setOnboardingDismissed(true)
    try { localStorage.setItem(ONBOARDING_KEY, "true") } catch { /* noop */ }
  }, [])

  if (!activeDocumentId || !activeDocument) {
    const featureCards = [
      {
        icon: <Highlighter className="h-5 w-5 text-yellow-500" />,
        title: t("option:documentWorkspace.featureHighlight", "Highlight and take notes"),
        description: t("option:documentWorkspace.featureHighlightDesc", "Select text to highlight in multiple colors and add notes.")
      },
      {
        icon: <MessageSquare className="h-5 w-5 text-blue-500" />,
        title: t("option:documentWorkspace.featureChat", "Ask questions about your document"),
        description: t("option:documentWorkspace.featureChatDesc", "Ask AI questions and get answers based on your document's content.")
      },
      {
        icon: <Lightbulb className="h-5 w-5 text-amber-500" />,
        title: t("option:documentWorkspace.featureInsights", "Get AI summaries and key findings"),
        description: t("option:documentWorkspace.featureInsightsDesc", "Get AI-generated summaries, key findings, and analysis.")
      },
      {
        icon: <HelpCircle className="h-5 w-5 text-green-500" />,
        title: t("option:documentWorkspace.featureQuiz", "Test your understanding with quizzes"),
        description: t("option:documentWorkspace.featureQuizDesc", "Generate quizzes from document content to test your understanding.")
      }
    ]

    return (
      <div
        className={`flex h-full flex-col items-center justify-center gap-4 p-8 text-center ${className || ""}`}
      >
        <FileText className="h-16 w-16 text-muted" />
        <div>
          <h3 className="text-lg font-medium">
            {t("option:documentWorkspace.noDocument", "No document selected")}
          </h3>
          <p className="text-sm text-muted">
            {t(
              "option:documentWorkspace.noDocumentHint",
              "Open a document from your media library to start reading"
            )}
          </p>
        </div>
        {(onOpenLibrary || onOpenUpload) && (
          <div className="flex flex-wrap items-center justify-center gap-2">
            {onOpenUpload && (
              <Button type="primary" onClick={onOpenUpload}>
                {t("option:documentWorkspace.upload", "Upload")}
              </Button>
            )}
            {onOpenLibrary && (
              <Button onClick={onOpenLibrary}>
                {t("option:documentWorkspace.openDocument", "Open document")}
              </Button>
            )}
          </div>
        )}

        {/* Show tips link when cards are dismissed */}
        {onboardingDismissed && (
          <button
            onClick={() => {
              try { localStorage.removeItem(ONBOARDING_KEY) } catch { /* noop */ }
              setOnboardingDismissed(false)
            }}
            className="text-xs text-primary hover:underline"
          >
            {t("option:documentWorkspace.showTips", "Show tips")}
          </button>
        )}

        {/* Feature discovery cards */}
        {!onboardingDismissed && (
          <div className="mt-4 w-full max-w-xl">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-medium text-text-muted">
                {t("option:documentWorkspace.whatYouCanDo", "What you can do")}
              </span>
              <button
                onClick={dismissOnboarding}
                className="rounded p-1 text-text-muted hover:text-text hover:bg-hover"
                aria-label={t("common:dismiss", "Dismiss")}
              >
                <X className="h-3.5 w-3.5" />
              </button>
            </div>
            <div className="grid grid-cols-2 gap-3">
              {featureCards.map((card) => (
                <div
                  key={card.title}
                  className="flex items-start gap-3 rounded-lg border border-border bg-surface p-3 text-left"
                >
                  <div className="mt-0.5 shrink-0">{card.icon}</div>
                  <div>
                    <p className="text-sm font-medium">{card.title}</p>
                    <p className="mt-0.5 text-xs text-text-muted">{card.description}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    )
  }

  const renderViewer = () => {
    switch (activeDocumentType) {
      case "pdf":
        return (
          <Suspense fallback={viewerFallback}>
            <PdfDocument
              url={activeDocument.url}
              documentId={activeDocumentId}
              currentPage={currentPage}
              zoomLevel={zoomLevel}
              viewMode={viewMode}
              onLoadSuccess={handleLoadSuccess}
              onLoadError={handleLoadError}
              onPageChange={setCurrentPage}
              pdfDocumentRef={pdfDocumentRef}
            />
          </Suspense>
        )
      case "epub":
        return (
          <Suspense fallback={viewerFallback}>
            <EpubViewer
              url={activeDocument.url!}
              documentId={activeDocumentId}
              onLoadSuccess={({ chapterCount }) => {
                // chapterCount is available if needed
              }}
              onLoadError={handleLoadError}
            />
          </Suspense>
        )
      default:
        return (
          <div className="flex h-full flex-col items-center justify-center gap-4 p-8">
            <AlertCircle className="h-12 w-12 text-warning" />
            <p className="text-muted">Unsupported document type</p>
          </div>
        )
    }
  }

  return (
    <div data-testid="document-viewer" className={`flex h-full min-h-0 flex-col ${className || ""}`}>
      <ViewerToolbar
        currentPage={currentPage}
        totalPages={totalPages}
        zoomLevel={zoomLevel}
        viewMode={viewMode}
        documentType={activeDocumentType}
        percentage={currentPercentage}
        chapterTitle={currentChapterTitle}
        onPageChange={setCurrentPage}
        onZoomChange={setZoomLevel}
        onViewModeChange={setViewMode}
        onPreviousPage={goToPreviousPage}
        onNextPage={goToNextPage}
      />
      <div className="relative min-h-0 flex-1 overflow-hidden bg-surface2">
        {loadingDocumentId !== null && loadingDocumentId !== undefined && (
          <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-3 bg-surface2/80 backdrop-blur-sm">
            <Spin size="large" />
            <p className="text-sm text-text-muted">
              {t("option:documentWorkspace.loadingDocument", "Loading document...")}
            </p>
          </div>
        )}
        {activeDocumentType === "pdf" && (
          <Suspense fallback={null}>
            <PdfSearch pdfDocumentRef={pdfDocumentRef} />
          </Suspense>
        )}
        {renderViewer()}
      </div>
    </div>
  )
}

export default DocumentViewer
