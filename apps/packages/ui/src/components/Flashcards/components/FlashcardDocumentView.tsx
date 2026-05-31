import React from "react"
import { Button, Empty, Spin } from "antd"

import { Alert } from "@/components/ui/primitives"
import type { Deck, Flashcard, FlashcardBulkUpdateItem, FlashcardBulkUpdateResponse } from "@/services/flashcards"
import { FlashcardDocumentRow } from "./FlashcardDocumentRow"
import type { DocumentQueryFilterContext } from "../utils/document-cache-policy"

export interface FlashcardDocumentViewProps {
  items: Flashcard[]
  decks: Deck[]
  isLoading: boolean
  isFetchingNextPage: boolean
  hasNextPage: boolean
  isTruncated: boolean
  selectedIds: Set<string>
  selectAllAcross: boolean
  filterContext: DocumentQueryFilterContext
  queryKey: readonly unknown[]
  onToggleSelect: (uuid: string, checked: boolean) => void
  onLoadMore: () => void
  onOpenDrawer?: (card: Flashcard) => void
  bulkUpdate: (items: FlashcardBulkUpdateItem[]) => Promise<FlashcardBulkUpdateResponse>
}

/**
 * Continuous-scroll document surface for deck maintenance with row-local editing.
 */
export const FlashcardDocumentView: React.FC<FlashcardDocumentViewProps> = ({
  items,
  decks,
  isLoading,
  isFetchingNextPage,
  hasNextPage,
  isTruncated,
  selectedIds,
  selectAllAcross,
  filterContext,
  queryKey,
  onToggleSelect,
  onLoadMore,
  onOpenDrawer,
  bulkUpdate
}) => {
  const loadMoreRef = React.useRef<HTMLDivElement | null>(null)

  React.useEffect(() => {
    if (!hasNextPage || isFetchingNextPage) return
    const target = loadMoreRef.current
    if (!target || typeof IntersectionObserver === "undefined") return

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          onLoadMore()
        }
      },
      { rootMargin: "320px 0px" }
    )

    observer.observe(target)
    return () => observer.disconnect()
  }, [hasNextPage, isFetchingNextPage, onLoadMore, items.length])

  if (!isLoading && items.length === 0) {
    return (
      <div
        className="rounded-lg border border-border bg-surface p-6"
        data-testid="flashcards-document-view"
      >
        <Empty
          description="No cards match the current filters."
          image={Empty.PRESENTED_IMAGE_SIMPLE}
        />
      </div>
    )
  }

  return (
    <div
      className="rounded-lg border border-border bg-surface"
      data-testid="flashcards-document-view"
    >
      {isTruncated && (
        <Alert
          variant="warning"
          className="m-3"
          title="Document results are truncated to the current scan limit."
          data-testid="flashcards-document-truncation-banner"
        >
          Refine filters or reduce tags to enable selecting across the full result set.
        </Alert>
      )}

      <div className="divide-y divide-border">
        {items.map((card) => (
          <FlashcardDocumentRow
            key={card.uuid}
            card={card}
            decks={decks}
            selected={selectedIds.has(card.uuid)}
            selectAllAcross={selectAllAcross}
            filterContext={filterContext}
            queryKey={queryKey}
            onToggleSelect={onToggleSelect}
            onOpenDrawer={onOpenDrawer}
            bulkUpdate={bulkUpdate}
          />
        ))}
      </div>

      {(isLoading || isFetchingNextPage || hasNextPage) && (
        <div
          ref={loadMoreRef}
          className="flex items-center justify-center gap-3 border-t border-border px-4 py-4"
        >
          {(isLoading || isFetchingNextPage) && <Spin size="small" />}
          {hasNextPage && typeof IntersectionObserver === "undefined" && (
            <Button onClick={onLoadMore}>Load more</Button>
          )}
        </div>
      )}
    </div>
  )
}

export default FlashcardDocumentView
