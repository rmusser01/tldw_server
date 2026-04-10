import React from "react"
import { useTranslation } from "react-i18next"
import { Document, Page } from "react-pdf"
import { Empty, Skeleton, Alert } from "antd"
import { Image as ImageIcon } from "lucide-react"
import { useDocumentWorkspaceStore } from "@/store/document-workspace"

const MAX_THUMBNAILS = 24

/**
 * PagesTab - Displays page thumbnails for the current PDF document.
 *
 * Features:
 * - Grid display of page thumbnails
 * - Click to navigate to the source page
 * - Highlights current page
 * - Loading and empty states
 */
export const PagesTab: React.FC = () => {
  const { t } = useTranslation(["option", "common"])
  const activeDocumentId = useDocumentWorkspaceStore((s) => s.activeDocumentId)
  const activeDocumentType = useDocumentWorkspaceStore((s) => s.activeDocumentType)
  const documentUrl = useDocumentWorkspaceStore(
    (s) => s.openDocuments.find((doc) => doc.id === s.activeDocumentId)?.url
  )
  const currentPage = useDocumentWorkspaceStore((s) => s.currentPage)
  const setCurrentPage = useDocumentWorkspaceStore((s) => s.setCurrentPage)
  const totalPages = useDocumentWorkspaceStore((s) => s.totalPages)

  const [numPages, setNumPages] = React.useState<number>(0)
  const [loadError, setLoadError] = React.useState<string | null>(null)

  // Prefer the locally loaded page count from this tab's Document instance.
  const pageCount = numPages > 0 ? numPages : totalPages

  React.useEffect(() => {
    setNumPages(0)
    setLoadError(null)
  }, [documentUrl])

  const safeCurrentPage =
    typeof currentPage === "number" && currentPage > 0 ? currentPage : 1

  const thumbnailRange = React.useMemo(() => {
    if (pageCount <= 0) return null
    if (pageCount <= MAX_THUMBNAILS) {
      return { start: 1, end: pageCount, truncated: false }
    }

    const halfWindow = Math.floor(MAX_THUMBNAILS / 2)
    let start = Math.max(1, safeCurrentPage - halfWindow)
    let end = start + MAX_THUMBNAILS - 1

    if (end > pageCount) {
      end = pageCount
      start = Math.max(1, end - MAX_THUMBNAILS + 1)
    }

    return { start, end, truncated: true }
  }, [pageCount, safeCurrentPage])

  const handleThumbnailClick = (page: number) => {
    setCurrentPage(page)
  }

  // No document selected
  if (!activeDocumentId) {
    return (
      <div className="flex h-full items-center justify-center p-4">
        <Empty
          image={<ImageIcon className="mx-auto h-12 w-12 text-muted" />}
          description={t(
            "option:documentWorkspace.pagesNoDocument",
            "Visual page previews for quick navigation. Open a PDF to browse pages."
          )}
        />
      </div>
    )
  }

  if (activeDocumentType !== "pdf") {
    return (
      <div className="flex h-full items-center justify-center p-4">
        <Empty
          image={<ImageIcon className="mx-auto h-12 w-12 text-muted" />}
          description={t(
            "option:documentWorkspace.pagesPdfOnly",
            "Page previews are available for PDF documents."
          )}
        />
      </div>
    )
  }

  if (!documentUrl) {
    return (
      <div className="flex h-full items-center justify-center p-4">
        <Empty
          image={<ImageIcon className="mx-auto h-12 w-12 text-muted" />}
          description={t(
            "option:documentWorkspace.missingDocumentFile",
            "No document file available"
          )}
        />
      </div>
    )
  }

  return (
    <div className="h-full overflow-auto p-3">
      {loadError ? (
        <Alert
          className="mb-3"
          type="error"
          title={t("option:documentWorkspace.pagesError", "Failed to load document")}
          description={loadError || t("common:unknownError", "An unknown error occurred")}
          showIcon
        />
      ) : null}
      <Document
        file={documentUrl}
        error={null}
        onLoadSuccess={(pdf) => {
          setNumPages(pdf.numPages > 0 ? pdf.numPages : 0)
          setLoadError(null)
        }}
        onLoadError={(error) => {
          setLoadError(error.message || "Failed to load PDF")
        }}
        loading={
          <div className="space-y-3">
            <Skeleton.Image active className="!w-full !h-24" />
            <Skeleton.Image active className="!w-full !h-24" />
            <Skeleton.Image active className="!w-full !h-24" />
          </div>
        }
      >
        {pageCount > 0 ? (
          <>
            <div className="mb-2 text-xs text-muted">
              {t("option:documentWorkspace.pagesCount", "{{count}} pages", {
                count: pageCount,
              })}
              {thumbnailRange?.truncated
                ? t(
                    "option:documentWorkspace.pagesWindow",
                    " · Showing {{start}}–{{end}}",
                    {
                      start: thumbnailRange.start,
                      end: thumbnailRange.end,
                    }
                  )
                : null}
            </div>
            <div className="grid grid-cols-2 gap-3">
              {thumbnailRange
                ? Array.from(
                    { length: thumbnailRange.end - thumbnailRange.start + 1 },
                    (_, index) => {
                      const pageNumber = thumbnailRange.start + index
                      const isActive = pageNumber === currentPage
                      return (
                        <button
                          key={`thumb-${pageNumber}`}
                          onClick={() => handleThumbnailClick(pageNumber)}
                          aria-label={t(
                            "option:documentWorkspace.thumbnailAriaLabel",
                            "Thumbnail, page {{pageNumber}}",
                            { pageNumber }
                          )}
                          aria-current={isActive ? "page" : undefined}
                          className={`group relative overflow-hidden rounded border transition-all focus:outline-none focus:ring-2 focus:ring-primary ${
                            isActive
                              ? "border-primary shadow-md"
                              : "border-border hover:border-primary"
                          }`}
                        >
                          <Page
                            pageNumber={pageNumber}
                            scale={0.25}
                            renderTextLayer={false}
                            renderAnnotationLayer={false}
                            loading=""
                          />
                          <span className="absolute bottom-1 left-1/2 -translate-x-1/2 rounded bg-black/70 px-2 py-0.5 text-xs text-white">
                            {pageNumber}
                          </span>
                        </button>
                      )
                    }
                  )
                : null}
            </div>
          </>
        ) : (
          <div className="flex h-full items-center justify-center p-4">
            <Empty
              image={<ImageIcon className="mx-auto h-12 w-12 text-muted" />}
              description={t(
                "option:documentWorkspace.noPages",
                "No pages found in this document"
              )}
            />
          </div>
        )}
      </Document>
    </div>
  )
}

export default PagesTab
