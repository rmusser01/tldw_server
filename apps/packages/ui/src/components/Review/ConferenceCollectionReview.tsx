import React from "react"
import { ChevronLeft, ChevronRight, ExternalLink, MessageSquareText } from "lucide-react"
import type {
  MediaCollection,
  MediaCollectionItem,
  MediaCollectionItemStatus,
} from "@/services/tldw/conference-collections"

type ConferenceCollectionReviewProps = {
  collection: MediaCollection
  onAskCollection?: (scope: { collectionId: number; mediaIds: number[] }) => void
  onOpenMedia?: (mediaId: number) => void
  className?: string
}

const READY_STATUSES = new Set<MediaCollectionItemStatus>([
  "completed",
  "skipped_existing",
])

const IN_PROGRESS_STATUSES = new Set<MediaCollectionItemStatus>([
  "planned",
  "processing",
])

const ISSUE_STATUSES = new Set<MediaCollectionItemStatus>([
  "submit_failed",
  "failed",
  "cancelled",
])

const STATUS_LABELS: Record<MediaCollectionItemStatus, string> = {
  planned: "Planned",
  processing: "Processing",
  completed: "Completed",
  skipped_existing: "Skipped existing",
  submit_failed: "Not submitted",
  failed: "Failed",
  cancelled: "Cancelled",
}

const statusTone = (status: MediaCollectionItemStatus): string => {
  if (READY_STATUSES.has(status)) {
    return "border-success/30 bg-success/10 text-success"
  }
  if (status === "processing" || status === "planned") {
    return "border-primary/30 bg-primary/10 text-primary"
  }
  return "border-danger/30 bg-danger/10 text-danger"
}

const isReadyItem = (item: MediaCollectionItem): boolean =>
  READY_STATUSES.has(item.status) &&
  typeof item.mediaId === "number" &&
  Number.isFinite(item.mediaId) &&
  item.mediaId > 0

const getItemTitle = (item: MediaCollectionItem): string =>
  item.title?.trim() || `Talk ${item.ordinal}`

const getMetadataString = (
  item: MediaCollectionItem,
  key: "summary" | "excerpt" | "description"
): string | null => {
  const value = item.metadata?.[key]
  return typeof value === "string" && value.trim().length > 0
    ? value.trim()
    : null
}

const formatTalkMeta = (item: MediaCollectionItem): string => {
  const parts = [item.speaker, item.track, item.publishedAt]
    .filter((value): value is string => typeof value === "string" && value.trim().length > 0)
    .map((value) => value.trim())
  return parts.join(" / ")
}

const sortCollectionItems = (
  items: MediaCollectionItem[]
): MediaCollectionItem[] =>
  [...items].sort((left, right) => {
    if (left.ordinal !== right.ordinal) return left.ordinal - right.ordinal
    return left.id - right.id
  })

export function ConferenceCollectionReview({
  collection,
  onAskCollection,
  onOpenMedia,
  className = "",
}: ConferenceCollectionReviewProps) {
  const orderedItems = React.useMemo(
    () => sortCollectionItems(collection.items),
    [collection.items]
  )
  const readyItems = React.useMemo(
    () => orderedItems.filter(isReadyItem),
    [orderedItems]
  )
  const readyMediaIds = React.useMemo(
    () => Array.from(new Set(readyItems.map((item) => item.mediaId as number))),
    [readyItems]
  )
  const inProgressCount = orderedItems.filter((item) =>
    IN_PROGRESS_STATUSES.has(item.status)
  ).length
  const issueCount = orderedItems.filter((item) =>
    ISSUE_STATUSES.has(item.status)
  ).length
  const [activeItemId, setActiveItemId] = React.useState<number | null>(
    () => orderedItems[0]?.id ?? null
  )
  const [selectedCompareIds, setSelectedCompareIds] = React.useState<number[]>([])

  React.useEffect(() => {
    if (orderedItems.length === 0) {
      setActiveItemId(null)
      return
    }
    setActiveItemId((current) =>
      current != null && orderedItems.some((item) => item.id === current)
        ? current
        : orderedItems[0].id
    )
  }, [orderedItems])

  const activeIndex = Math.max(
    0,
    orderedItems.findIndex((item) => item.id === activeItemId)
  )
  const activeItem = orderedItems[activeIndex] ?? null
  const selectedCompareItems = orderedItems.filter((item) =>
    selectedCompareIds.includes(item.id)
  )
  const qaDisabled = readyMediaIds.length === 0

  const handlePrevious = () => {
    if (activeIndex <= 0) return
    setActiveItemId(orderedItems[activeIndex - 1].id)
  }

  const handleNext = () => {
    if (activeIndex >= orderedItems.length - 1) return
    setActiveItemId(orderedItems[activeIndex + 1].id)
  }

  const toggleCompare = (itemId: number) => {
    setSelectedCompareIds((current) =>
      current.includes(itemId)
        ? current.filter((value) => value !== itemId)
        : [...current, itemId]
    )
  }

  const askCollection = () => {
    if (qaDisabled) return
    onAskCollection?.({
      collectionId: collection.id,
      mediaIds: readyMediaIds,
    })
  }

  return (
    <section className={`space-y-4 text-text ${className}`}>
      <header className="flex flex-col gap-3 border-b border-border pb-3 md:flex-row md:items-start md:justify-between">
        <div className="min-w-0">
          <h2 className="truncate text-lg font-semibold">{collection.name}</h2>
          <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-text-muted">
            <span>{orderedItems.length} talks</span>
            <span>{readyItems.length} ready</span>
            {inProgressCount > 0 && <span>{inProgressCount} in progress</span>}
            {issueCount > 0 && <span>{issueCount} need attention</span>}
          </div>
        </div>
        <div className="flex flex-col items-start gap-1 md:items-end">
          <button
            type="button"
            onClick={askCollection}
            disabled={qaDisabled}
            className="inline-flex h-9 items-center gap-2 rounded-md border border-primary/40 px-3 text-sm font-medium text-primary hover:bg-primary/10 disabled:cursor-not-allowed disabled:border-border disabled:text-text-muted disabled:hover:bg-transparent"
          >
            <MessageSquareText className="h-4 w-4" aria-hidden="true" />
            Ask this collection
          </button>
          <div className="text-xs text-text-muted" aria-live="polite">
            {qaDisabled ? (
              <>
                <div>No ready talks yet</div>
                <div>Waiting for completed or skipped-existing items.</div>
              </>
            ) : (
              <div>{readyItems.length} ready talks will be searched.</div>
            )}
          </div>
        </div>
      </header>

      <div className="grid min-h-[420px] gap-4 lg:grid-cols-[minmax(240px,320px)_1fr]">
        <aside className="min-h-0 rounded-md border border-border bg-surface">
          <div className="border-b border-border px-3 py-2 text-xs font-medium text-text-muted">
            Conference order
          </div>
          <ul
            className="max-h-[520px] divide-y divide-border overflow-auto"
            data-testid="conference-talk-list"
          >
            {orderedItems.map((item) => {
              const title = getItemTitle(item)
              const itemMeta = formatTalkMeta(item)
              const selected = selectedCompareIds.includes(item.id)
              const active = activeItem?.id === item.id
              return (
                <li
                  key={item.id}
                  className={`flex gap-2 px-3 py-2 ${active ? "bg-primary/5" : ""}`}
                >
                  <input
                    type="checkbox"
                    checked={selected}
                    aria-label={`Compare ${title}`}
                    onChange={() => toggleCompare(item.id)}
                    className="mt-1 h-4 w-4 rounded border-border"
                  />
                  <button
                    type="button"
                    onClick={() => setActiveItemId(item.id)}
                    aria-current={active ? "true" : undefined}
                    className="min-w-0 flex-1 text-left"
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="truncate text-sm font-medium">
                        {item.ordinal}. {title}
                      </span>
                      <span
                        className={`shrink-0 rounded-full border px-2 py-0.5 text-[10px] font-medium ${statusTone(item.status)}`}
                      >
                        {STATUS_LABELS[item.status]}
                      </span>
                    </div>
                    {itemMeta && (
                      <div className="mt-1 truncate text-xs text-text-muted">
                        {itemMeta}
                      </div>
                    )}
                    {item.errorSummary && (
                      <div className="mt-1 truncate text-xs text-danger">
                        {item.errorSummary}
                      </div>
                    )}
                  </button>
                </li>
              )
            })}
          </ul>
        </aside>

        <main className="min-w-0 rounded-md border border-border bg-surface p-4">
          {activeItem ? (
            <div className="space-y-4">
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <h3 className="truncate text-lg font-semibold">
                    {getItemTitle(activeItem)}
                  </h3>
                  <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-text-muted">
                    <span
                      className={`rounded-full border px-2 py-0.5 text-[10px] font-medium ${statusTone(activeItem.status)}`}
                    >
                      {STATUS_LABELS[activeItem.status]}
                    </span>
                    {formatTalkMeta(activeItem) && (
                      <span>{formatTalkMeta(activeItem)}</span>
                    )}
                  </div>
                </div>
                <div className="flex shrink-0 items-center gap-2">
                  <button
                    type="button"
                    onClick={handlePrevious}
                    disabled={activeIndex <= 0}
                    className="inline-flex h-8 items-center gap-1 rounded-md border border-border px-2 text-xs hover:bg-surface2 disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    <ChevronLeft className="h-3.5 w-3.5" aria-hidden="true" />
                    Previous talk
                  </button>
                  <button
                    type="button"
                    onClick={handleNext}
                    disabled={activeIndex >= orderedItems.length - 1}
                    className="inline-flex h-8 items-center gap-1 rounded-md border border-border px-2 text-xs hover:bg-surface2 disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    Next talk
                    <ChevronRight className="h-3.5 w-3.5" aria-hidden="true" />
                  </button>
                </div>
              </div>

              <div className="grid gap-3 md:grid-cols-2">
                <div className="rounded-md border border-border bg-surface2 p-3">
                  <div className="text-xs font-medium text-text-muted">Summary</div>
                  <p className="mt-2 text-sm leading-6">
                    {getMetadataString(activeItem, "summary") ||
                      getMetadataString(activeItem, "description") ||
                      "No summary yet."}
                  </p>
                </div>
                <div className="rounded-md border border-border bg-surface2 p-3">
                  <div className="text-xs font-medium text-text-muted">Transcript excerpt</div>
                  <p className="mt-2 text-sm leading-6">
                    {getMetadataString(activeItem, "excerpt") ||
                      "No transcript excerpt yet."}
                  </p>
                </div>
              </div>

              <div className="flex flex-wrap items-center gap-2">
                <button
                  type="button"
                  onClick={() => {
                    if (activeItem.mediaId) onOpenMedia?.(activeItem.mediaId)
                  }}
                  disabled={!activeItem.mediaId}
                  className="inline-flex h-8 items-center gap-1 rounded-md border border-border px-3 text-xs hover:bg-surface2 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  <ExternalLink className="h-3.5 w-3.5" aria-hidden="true" />
                  Open media
                </button>
                {activeItem.sourceUrl && (
                  <a
                    href={activeItem.sourceUrl}
                    target="_blank"
                    rel="noreferrer"
                    className="text-xs text-primary hover:underline"
                  >
                    Source URL
                  </a>
                )}
              </div>
            </div>
          ) : (
            <div className="py-10 text-center text-sm text-text-muted">
              No talks in this collection.
            </div>
          )}
        </main>
      </div>

      {selectedCompareItems.length >= 2 && (
        <section className="rounded-md border border-border bg-surface p-4">
          <h3 className="text-sm font-semibold">
            Comparing {selectedCompareItems.length} talks
          </h3>
          <div className="mt-3 grid gap-3 md:grid-cols-2 xl:grid-cols-3">
            {selectedCompareItems.map((item) => (
              <article
                key={item.id}
                className="rounded-md border border-border bg-surface2 p-3"
              >
                <div className="text-sm font-medium">{getItemTitle(item)}</div>
                {formatTalkMeta(item) && (
                  <div className="mt-1 text-xs text-text-muted">
                    {formatTalkMeta(item)}
                  </div>
                )}
                <p className="mt-2 text-sm leading-6">
                  {getMetadataString(item, "summary") ||
                    getMetadataString(item, "excerpt") ||
                    "No summary or excerpt yet."}
                </p>
              </article>
            ))}
          </div>
        </section>
      )}
    </section>
  )
}

export { isReadyItem as isConferenceCollectionReadyItem }
