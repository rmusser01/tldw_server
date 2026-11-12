import { ChevronLeft, ChevronRight } from 'lucide-react'
import { useState, useEffect, useCallback, useMemo } from 'react'
import { Tooltip } from 'antd'
import { useTranslation } from 'react-i18next'

interface PaginationProps {
  currentPage: number
  totalPages: number
  onPageChange: (page: number) => void
  totalItems: number
  itemsPerPage: number
  currentItemsCount: number
  pageSizeOptions?: number[]
  onItemsPerPageChange?: (pageSize: number) => void
}

export function Pagination({
  currentPage,
  totalPages,
  onPageChange,
  totalItems,
  itemsPerPage,
  currentItemsCount,
  pageSizeOptions = [20, 50, 100],
  onItemsPerPageChange
}: PaginationProps) {
  const { t } = useTranslation(['review'])
  const [jumpToPage, setJumpToPage] = useState('')
  const [jumpError, setJumpError] = useState('')

  // Clear error when totalPages changes (pagination state updated)
  // IMPORTANT: All hooks must be called before any early returns
  useEffect(() => {
    setJumpError('')
  }, [totalPages])

  const startItem = (currentPage - 1) * itemsPerPage + 1
  const endItem = (currentPage - 1) * itemsPerPage + currentItemsCount

  const handlePrevious = useCallback(() => {
    if (currentPage > 1) {
      onPageChange(currentPage - 1)
    }
  }, [currentPage, onPageChange])

  const handleNext = useCallback(() => {
    if (currentPage < totalPages) {
      onPageChange(currentPage + 1)
    }
  }, [currentPage, totalPages, onPageChange])

  const handlePageClick = useCallback((page: number) => {
    onPageChange(page)
  }, [onPageChange])

  const handleJumpToPage = useCallback(() => {
    const pageNum = parseInt(jumpToPage, 10)
    if (isNaN(pageNum) || pageNum < 1 || pageNum > totalPages) {
      setJumpError(
        t('mediaPage.pageRangeError', 'Page must be 1-{{max}}', {
          max: totalPages
        })
      )
      return
    }

    setJumpError('')
    setJumpToPage('')
    onPageChange(pageNum)
  }, [jumpToPage, totalPages, onPageChange, t])

  const handleJumpInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setJumpToPage(e.target.value)
    // Clear error when user starts typing again
    setJumpError('')
  }, [])

  const handleJumpKeyPress = useCallback((e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleJumpToPage()
    }
  }, [handleJumpToPage])

  // Generate page numbers to display
  const pageNumbers = useMemo(() => {
    const pages: (number | string)[] = []
    const maxPagesToShow = 5

    if (totalPages <= maxPagesToShow) {
      // Show all pages if total is small
      for (let i = 1; i <= totalPages; i++) {
        pages.push(i)
      }
    } else {
      // Always show first page
      pages.push(1)

      if (currentPage > 3) {
        pages.push('...')
      }

      // Show pages around current page
      const start = Math.max(2, currentPage - 1)
      const end = Math.min(totalPages - 1, currentPage + 1)

      for (let i = start; i <= end; i++) {
        pages.push(i)
      }

      if (currentPage < totalPages - 2) {
        pages.push('...')
      }

      // Always show last page
      if (totalPages > 1) {
        pages.push(totalPages)
      }
    }

    return pages
  }, [totalPages, currentPage])

  // A local filter can hide this page's rows while later pages still have matches.
  if (currentItemsCount === 0 && totalPages <= 1) {
    return (
      <div className="px-4 py-2 border-t border-border bg-surface">
        <div className="text-xs text-text-muted text-center">
          {t(
            'mediaPage.showingZero',
            '0 of {{total}} results',
            { total: totalItems }
          )}
        </div>
      </div>
    )
  }

  if (totalPages <= 1) {
    return (
      <div className="px-4 py-2 border-t border-border bg-surface">
        <div className="text-xs text-text-muted text-center">
          {t(
            'mediaPage.showingCount',
            '{{count}} of {{total}} results',
            { count: currentItemsCount, total: totalItems }
          )}
        </div>
        {onItemsPerPageChange && (
          <div className="mt-2 flex items-center justify-center gap-1.5">
            <label htmlFor="items-per-page-select" className="text-xs text-text-muted">
              {t('mediaPage.itemsPerPage', 'Per page:')}
            </label>
            <select
              id="items-per-page-select"
              value={itemsPerPage}
              onChange={(event) => onItemsPerPageChange(Number(event.target.value))}
              className="rounded border border-border bg-surface2 px-2 py-0.5 text-xs text-text focus:outline-none focus:ring-2 focus:ring-focus"
            >
              {pageSizeOptions.map((size) => (
                <option key={size} value={size}>
                  {size}
                </option>
              ))}
            </select>
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="px-4 py-2 border-t border-border bg-surface">
      {/* Items count */}
      <div className="text-xs text-text-muted mb-2 text-center">
        {currentItemsCount === 0
          ? t('mediaPage.showingZero', '0 of {{total}} results', { total: totalItems })
          : t(
              'mediaPage.showingRange',
              'Showing {{start}}-{{end}} of {{total}}',
              { start: startItem, end: endItem, total: totalItems }
            )}
      </div>

      {/* Pagination controls */}
      <div className="flex items-center justify-center gap-1 mb-2">
        {/* Previous button */}
        <button
          onClick={handlePrevious}
          disabled={currentPage === 1}
          className="px-1.5 py-0.5 rounded hover:bg-surface2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          aria-label={t('mediaPage.previousPage', 'Previous page')}
          title={t('mediaPage.previousPage', 'Previous page')}
        >
          <ChevronLeft className="w-3.5 h-3.5 text-text-muted" />
        </button>

        {/* Page numbers */}
        {pageNumbers.map((page, index) => {
          if (page === '...') {
            return (
              <span
                key={`ellipsis-${index}`}
                className="px-1.5 py-0.5 text-text-subtle text-xs"
                aria-hidden="true"
                role="presentation"
              >
                ...
              </span>
            )
          }

          const pageNum = page as number
          const isActive = pageNum === currentPage

          return (
            <button
              key={pageNum}
              onClick={() => handlePageClick(pageNum)}
              aria-label={t('mediaPage.goToPage', 'Go to page {{num}}', { num: pageNum })}
              aria-current={isActive ? 'page' : undefined}
              title={t('mediaPage.goToPage', 'Go to page {{num}}', { num: pageNum })}
              className={`px-2 py-0.5 rounded text-xs transition-colors focus-visible:ring-2 focus-visible:ring-focus focus-visible:ring-offset-1 ${
                isActive
                  ? 'bg-primary text-white font-medium'
                  : 'text-text-muted hover:bg-surface2'
              }`}
            >
              {pageNum}
            </button>
          )
        })}

        {/* Next button */}
        <button
          onClick={handleNext}
          disabled={currentPage === totalPages}
          className="px-1.5 py-0.5 rounded hover:bg-surface2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          aria-label={t('mediaPage.nextPage', 'Next page')}
          title={t('mediaPage.nextPage', 'Next page')}
        >
          <ChevronRight className="w-3.5 h-3.5 text-text-muted" />
        </button>
      </div>

      {/* Jump to page - only show if there are many pages */}
      {totalPages > 5 && (
        <div className="flex items-center justify-center gap-1.5">
          <label htmlFor="jump-to-page" className="text-xs text-text-muted">
            {t('mediaPage.pageLabel', 'Page:')}
          </label>
          <Tooltip
            title={jumpError}
            open={!!jumpError}
            color="red"
          >
            <input
              id="jump-to-page"
              type="number"
              min="1"
              max={totalPages}
              value={jumpToPage}
              onChange={handleJumpInputChange}
              onKeyDown={handleJumpKeyPress}
              placeholder={`1-${totalPages}`}
              className={`w-16 px-1.5 py-0.5 text-xs border bg-surface text-text rounded focus:outline-none focus:ring-2 focus:border-transparent transition-colors ${
                jumpError
                  ? 'border-danger focus:ring-danger'
                  : 'border-border focus:ring-focus'
              }`}
              aria-invalid={!!jumpError}
              aria-describedby={jumpError ? 'jump-page-error' : undefined}
            />
          </Tooltip>
          {jumpError && (
            <span id="jump-page-error" className="sr-only">
              {jumpError}
            </span>
          )}
          <button
            onClick={handleJumpToPage}
            className="px-2 py-0.5 text-xs bg-surface2 hover:bg-surface text-text rounded transition-colors"
            title={t('mediaPage.go', 'Go')}
          >
            {t('mediaPage.go', 'Go')}
          </button>
        </div>
      )}

      {onItemsPerPageChange && (
        <div className="mt-2 flex items-center justify-center gap-1.5">
          <label htmlFor="items-per-page-select" className="text-xs text-text-muted">
            {t('mediaPage.itemsPerPage', 'Per page:')}
          </label>
          <select
            id="items-per-page-select"
            value={itemsPerPage}
            onChange={(event) => onItemsPerPageChange(Number(event.target.value))}
            className="rounded border border-border bg-surface2 px-2 py-0.5 text-xs text-text focus:outline-none focus:ring-2 focus:ring-focus"
          >
            {pageSizeOptions.map((size) => (
              <option key={size} value={size}>
                {size}
              </option>
            ))}
          </select>
        </div>
      )}
    </div>
  )
}
