import React, { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { Input, Typography } from "antd"
import { useVirtualizer } from "@tanstack/react-virtual"
import type {
  ConferenceItemMetadataOverride,
  PlaylistReviewMetadataField,
  PlaylistReviewState,
  WizardDuplicatePolicy,
  WizardQueueItem,
} from "./types"
import {
  getPlaylistAllowedDuplicatePolicies,
  playlistItemHasValidExplicitMetadataPatch,
  playlistItemIsCurrentDuplicate,
  useIngestWizard,
} from "./IngestWizardContext"

const parseTags = (value: string): string[] =>
  value
    .split(",")
    .map((tag) => tag.trim())
    .filter(Boolean)

const parsePlaylistKeywords = (value: string): string[] => {
  const seen = new Set<string>()
  return parseTags(value).filter((keyword) => {
    const key = keyword.toLocaleLowerCase()
    if (seen.has(key)) return false
    seen.add(key)
    return true
  })
}

const stringifyTags = (tags?: string[]): string => (tags || []).join(", ")

const getItemDisplay = (item: WizardQueueItem, index: number): string =>
  item.playlist?.title ||
  item.conferenceOverride?.title ||
  item.fileName ||
  item.url ||
  `Item ${index + 1}`

const getQueueItemOccurrenceId = (item: WizardQueueItem): string =>
  item.sourceRef?.occurrenceId || item.id

type ItemMetadataTableProps = {
  mode?: "conference" | "playlist"
  visibleItemIds?: ReadonlySet<string>
}

export const ItemMetadataTable: React.FC<ItemMetadataTableProps> = ({
  mode = "conference",
  visibleItemIds,
}) => {
  const { state, updateQueueItems } = useIngestWizard()
  const { queueItems } = state
  const [tagDrafts, setTagDrafts] = useState<Record<string, string>>({})

  const scrollParentRef = useRef<HTMLDivElement | null>(null)
  const rowRefs = useRef(new Map<string, HTMLDivElement>())
  const listOwnsFocusRef = useRef(false)
  const activeRowRef = useRef<{ id: string; index: number } | null>(null)
  const [activeId, setActiveId] = useState<string | null>(null)
  const visibleItems = useMemo(
    () =>
      mode === "playlist"
        ? queueItems.filter(
            (item) =>
              (item.sourceRef?.kind === "materialized_playlist_item" || Boolean(item.playlist)) &&
              (!visibleItemIds || visibleItemIds.has(item.id))
          )
        : queueItems,
    [mode, queueItems, visibleItemIds]
  )
  // TanStack Virtual exposes an imperative object that React Compiler skips.
  // eslint-disable-next-line react-hooks/incompatible-library
  const virtualizer = useVirtualizer({
    count: visibleItems.length,
    getScrollElement: () => scrollParentRef.current,
    estimateSize: () => (mode === "playlist" ? 118 : 56),
    overscan: 6,
    getItemKey: (index) => visibleItems[index]?.id ?? index,
    measureElement: (element) =>
      element?.getBoundingClientRect().height || (mode === "playlist" ? 118 : 56),
  })
  const virtualItems = virtualizer.getVirtualItems()
  const restoreRowFocus = useCallback((id: string) => {
    const attempt = (remaining: number) => {
      if (!listOwnsFocusRef.current) return
      const row = rowRefs.current.get(id)
      if (row) {
        row.focus()
        return
      }
      if (remaining > 0) window.requestAnimationFrame(() => attempt(remaining - 1))
    }
    window.requestAnimationFrame(() => attempt(2))
  }, [])

  useEffect(() => {
    const handleFocusIn = (event: FocusEvent) => {
      const target = event.target
      if (target instanceof Node && scrollParentRef.current?.contains(target)) return
      listOwnsFocusRef.current = false
    }
    document.addEventListener("focusin", handleFocusIn)
    return () => document.removeEventListener("focusin", handleFocusIn)
  }, [])

  useEffect(() => {
    if (visibleItems.length === 0) {
      activeRowRef.current = null
      setActiveId(null)
      return
    }
    const active = activeRowRef.current
    if (!active) {
      const id = getQueueItemOccurrenceId(visibleItems[0])
      activeRowRef.current = { id, index: 0 }
      setActiveId(id)
      return
    }
    const currentIndex = visibleItems.findIndex(
      (item) => getQueueItemOccurrenceId(item) === active.id
    )
    if (currentIndex >= 0) {
      active.index = currentIndex
      if (
        listOwnsFocusRef.current &&
        !rowRefs.current.has(active.id) &&
        virtualItems.length > 0
      ) {
        const nearest = virtualItems.reduce((best, row) =>
          Math.abs(row.index - currentIndex) < Math.abs(best.index - currentIndex) ? row : best
        )
        const nearestItem = visibleItems[nearest.index]
        if (nearestItem) {
          const id = getQueueItemOccurrenceId(nearestItem)
          activeRowRef.current = { id, index: nearest.index }
          setActiveId(id)
          restoreRowFocus(id)
        }
      }
      return
    }
    const index = Math.min(active.index, visibleItems.length - 1)
    const id = getQueueItemOccurrenceId(visibleItems[index])
    activeRowRef.current = { id, index }
    setActiveId(id)
    if (listOwnsFocusRef.current) {
      virtualizer.scrollToIndex(index, { align: "auto" })
      restoreRowFocus(id)
    }
  }, [restoreRowFocus, virtualItems, virtualizer, visibleItems])

  const handleRowKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLDivElement>, index: number) => {
      if (event.target !== event.currentTarget) {
        if (event.key === "Escape") {
          event.preventDefault()
          event.currentTarget.focus()
        }
        return
      }
      if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return
      event.preventDefault()
      const targetIndex = Math.max(
        0,
        Math.min(
          visibleItems.length - 1,
          index + (event.key === "ArrowDown" ? 1 : -1)
        )
      )
      const target = visibleItems[targetIndex]
      if (!target) return
      const id = getQueueItemOccurrenceId(target)
      activeRowRef.current = { id, index: targetIndex }
      setActiveId(id)
      virtualizer.scrollToIndex(targetIndex, { align: "auto" })
      restoreRowFocus(id)
    },
    [restoreRowFocus, virtualizer, visibleItems]
  )

  const updateConferenceOverride = useCallback(
    (itemId: string, patch: Partial<ConferenceItemMetadataOverride>) => {
      updateQueueItems((current) =>
        current.map((item) =>
          item.id === itemId
            ? {
                ...item,
                conferenceOverride: {
                  selected: item.conferenceOverride?.selected ?? true,
                  ...(item.conferenceOverride || {}),
                  ...patch,
                },
              }
            : item
        )
      )
    },
    [updateQueueItems]
  )

  const updatePlaylistReview = useCallback(
    (itemId: string, patch: Partial<PlaylistReviewState>) => {
      updateQueueItems((current) =>
        current.map((item) =>
          item.id === itemId
            ? {
                ...item,
                playlistReview: {
                  selected: item.playlistReview?.selected ?? true,
                  ...(item.playlistReview || {}),
                  ...patch,
                },
              }
            : item
        )
      )
    },
    [updateQueueItems]
  )

  const updatePlaylistMetadata = useCallback(
    (item: WizardQueueItem, field: PlaylistReviewMetadataField, value: string | string[]) => {
      const review = item.playlistReview
      const removeField = typeof value === "string" ? value.trim().length === 0 : value.length === 0
      const metadataPatch = { ...(review?.metadataPatch || {}) }
      const editedFields = [...(review?.editedFields || [])]
      if (removeField) {
        delete metadataPatch[field]
        const fieldIndex = editedFields.indexOf(field)
        if (fieldIndex >= 0) editedFields.splice(fieldIndex, 1)
      } else {
        metadataPatch[field] = value as never
        if (!editedFields.includes(field)) editedFields.push(field)
      }
      const nextReview: PlaylistReviewState = {
        selected: review?.selected ?? true,
        ...(review || {}),
        metadataPatch,
        editedFields,
      }
      if (
        nextReview.duplicatePolicy === "update_metadata_only" &&
        !playlistItemHasValidExplicitMetadataPatch({
          ...item,
          playlistReview: nextReview,
        })
      ) {
        nextReview.duplicatePolicy = undefined
      }
      updatePlaylistReview(item.id, nextReview)
    },
    [updatePlaylistReview]
  )

  const commitConferenceTags = useCallback(
    (itemId: string, rawValue: string) => {
      updateConferenceOverride(itemId, { tags: parseTags(rawValue) })
      setTagDrafts((prev) => {
        if (!(itemId in prev)) return prev
        const next = { ...prev }
        delete next[itemId]
        return next
      })
    },
    [updateConferenceOverride]
  )

  const commitPlaylistKeywords = useCallback(
    (item: WizardQueueItem, rawValue: string) => {
      updatePlaylistMetadata(item, "keywordsAdd", parsePlaylistKeywords(rawValue))
      setTagDrafts((prev) => {
        if (!(item.id in prev)) return prev
        const next = { ...prev }
        delete next[item.id]
        return next
      })
    },
    [updatePlaylistMetadata]
  )

  useEffect(() => {
    setTagDrafts((prev) => {
      const itemIds = new Set(queueItems.map((item) => item.id))
      const next = Object.fromEntries(
        Object.entries(prev).filter(([itemId]) => itemIds.has(itemId))
      )
      return Object.keys(next).length === Object.keys(prev).length ? prev : next
    })
  }, [queueItems])

  if (visibleItems.length === 0) return null

  if (mode === "playlist") {
    return (
      <section
        className="mt-3 rounded-md border border-border"
        aria-label="Playlist review overrides"
      >
        <div className="border-b border-border bg-surface2 px-3 py-2 text-xs font-medium text-text-muted">
          Per-video duplicate action and metadata changes
        </div>
        <div
          ref={scrollParentRef}
          className="max-h-96 overflow-y-auto"
          role="list"
          aria-label="Playlist review override items"
        >
          <div className="relative w-full" style={{ height: virtualizer.getTotalSize() }}>
            {virtualItems.map((virtualRow) => {
              const item = visibleItems[virtualRow.index]
              if (!item) return null
              const review = item.playlistReview
              const occurrenceId = item.sourceRef?.occurrenceId || item.id
              const duplicatePolicy = review?.duplicatePolicy ?? ""
              const isCurrentDuplicate = playlistItemIsCurrentDuplicate(item)
              const hasValidPatch = playlistItemHasValidExplicitMetadataPatch(item)
              return (
                <div
                  key={virtualRow.key}
                  ref={(element) => {
                    if (element) {
                      rowRefs.current.set(occurrenceId, element)
                      virtualizer.measureElement(element)
                    } else {
                      rowRefs.current.delete(occurrenceId)
                    }
                  }}
                  role="listitem"
                  tabIndex={activeId === occurrenceId ? 0 : -1}
                  aria-setsize={visibleItems.length}
                  aria-posinset={virtualRow.index + 1}
                  data-occurrence-id={occurrenceId}
                  data-index={virtualRow.index}
                  onFocusCapture={() => {
                    listOwnsFocusRef.current = true
                    activeRowRef.current = { id: occurrenceId, index: virtualRow.index }
                    setActiveId(occurrenceId)
                  }}
                  onKeyDown={(event) => handleRowKeyDown(event, virtualRow.index)}
                  className="absolute left-0 top-0 grid w-full gap-2 border-b border-border px-3 py-2 sm:grid-cols-[minmax(180px,1.2fr)_minmax(160px,1fr)]"
                  style={{ transform: `translateY(${virtualRow.start}px)` }}
                >
                  <div className="min-w-0">
                    <Typography.Text className="block truncate text-xs font-medium">
                      {item.playlist?.ordinal
                        ? `${item.playlist.ordinal}. ${getItemDisplay(item, virtualRow.index)}`
                        : getItemDisplay(item, virtualRow.index)}
                    </Typography.Text>
                    {isCurrentDuplicate ? (
                      <label className="mt-1 block text-[11px] text-text-muted">
                        Duplicate policy
                        <select
                          className="mt-0.5 block w-full rounded border border-border bg-surface px-2 py-1 text-xs"
                          aria-label={`Duplicate policy for occurrence ${occurrenceId}`}
                          value={duplicatePolicy}
                          onChange={(event) =>
                            updatePlaylistReview(item.id, {
                              duplicatePolicy:
                                (event.target.value as WizardDuplicatePolicy) || undefined,
                            })
                          }
                        >
                          <option value="">Choose action</option>
                          {getPlaylistAllowedDuplicatePolicies(item).map((policy) => (
                            <option
                              key={policy}
                              value={policy}
                              disabled={policy === "update_metadata_only" && !hasValidPatch}
                            >
                              {policy.replaceAll("_", " ")}
                            </option>
                          ))}
                        </select>
                        {duplicatePolicy === "update_metadata_only" && !hasValidPatch && (
                          <span className="mt-1 block text-warn">
                            Add at least one valid metadata change.
                          </span>
                        )}
                      </label>
                    ) : (
                      <Typography.Text className="mt-1 block text-[11px] text-text-muted">
                        No duplicate action needed
                      </Typography.Text>
                    )}
                  </div>
                  {isCurrentDuplicate && (
                    <div className="grid gap-1.5">
                      <Input
                        size="small"
                        maxLength={500}
                        aria-label={`Title override for occurrence ${occurrenceId}`}
                        value={review?.metadataPatch?.title ?? ""}
                        placeholder="Title (optional)"
                        onChange={(event) =>
                          updatePlaylistMetadata(item, "title", event.target.value)
                        }
                      />
                      <Input
                        size="small"
                        maxLength={500}
                        aria-label={`Author override for occurrence ${occurrenceId}`}
                        value={review?.metadataPatch?.author ?? ""}
                        placeholder="Author (optional)"
                        onChange={(event) =>
                          updatePlaylistMetadata(item, "author", event.target.value)
                        }
                      />
                      <Input
                        size="small"
                        aria-label={`Keywords to add for occurrence ${occurrenceId}`}
                        value={
                          tagDrafts[item.id] ?? stringifyTags(review?.metadataPatch?.keywordsAdd)
                        }
                        placeholder="keywords to add"
                        onChange={(event) =>
                          setTagDrafts((prev) => ({
                            ...prev,
                            [item.id]: event.target.value,
                          }))
                        }
                        onBlur={(event) => commitPlaylistKeywords(item, event.target.value)}
                        onKeyDown={(event) => {
                          if (event.key === "Enter") event.currentTarget.blur()
                        }}
                      />
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      </section>
    )
  }

  return (
    <div className="mt-3 rounded-md border border-border">
      <div className="grid grid-cols-[48px_minmax(160px,1.4fr)_minmax(120px,1fr)_minmax(100px,0.8fr)_minmax(120px,1fr)] gap-2 border-b border-border bg-surface2 px-2 py-1.5 text-[11px] font-medium uppercase text-text-muted">
        <span>Use</span>
        <span>Title</span>
        <span>Speaker</span>
        <span>Track</span>
        <span>Tags</span>
      </div>
      <div
        ref={scrollParentRef}
        className="max-h-72 overflow-y-auto"
        role="list"
        aria-label="Conference item metadata"
      >
        <div className="relative w-full" style={{ height: virtualizer.getTotalSize() }}>
          {virtualItems.map((virtualRow) => {
            const item = visibleItems[virtualRow.index]
            if (!item) return null
            const override = item.conferenceOverride
            const selected = override?.selected ?? true
            const itemNumber = virtualRow.index + 1

            return (
              <div
                key={virtualRow.key}
                ref={(element) => {
                  const id = getQueueItemOccurrenceId(item)
                  if (element) {
                    rowRefs.current.set(id, element)
                    virtualizer.measureElement(element)
                  } else {
                    rowRefs.current.delete(id)
                  }
                }}
                role="listitem"
                tabIndex={activeId === getQueueItemOccurrenceId(item) ? 0 : -1}
                aria-setsize={visibleItems.length}
                aria-posinset={virtualRow.index + 1}
                data-occurrence-id={getQueueItemOccurrenceId(item)}
                data-index={virtualRow.index}
                onFocusCapture={() => {
                  const id = getQueueItemOccurrenceId(item)
                  listOwnsFocusRef.current = true
                  activeRowRef.current = { id, index: virtualRow.index }
                  setActiveId(id)
                }}
                onKeyDown={(event) => handleRowKeyDown(event, virtualRow.index)}
                className="absolute left-0 top-0 grid w-full grid-cols-[48px_minmax(160px,1.4fr)_minmax(120px,1fr)_minmax(100px,0.8fr)_minmax(120px,1fr)] gap-2 border-b border-border px-2 py-2"
                style={{ transform: `translateY(${virtualRow.start}px)` }}
              >
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={selected}
                    aria-label={`Include item ${itemNumber}`}
                    onChange={(event) =>
                      updateConferenceOverride(item.id, {
                        selected: event.target.checked,
                      })
                    }
                  />
                </label>
                <div className="min-w-0">
                  <Input
                    size="small"
                    aria-label={`Title override for item ${itemNumber}`}
                    value={override?.title ?? ""}
                    placeholder={getItemDisplay(item, virtualRow.index)}
                    onChange={(event) =>
                      updateConferenceOverride(item.id, {
                        title: event.target.value,
                      })
                    }
                  />
                  <Typography.Text className="mt-0.5 block truncate text-[10px] text-text-muted">
                    {item.url || item.fileName || item.id}
                  </Typography.Text>
                </div>
                <Input
                  size="small"
                  aria-label={`Speaker for item ${itemNumber}`}
                  value={override?.speaker ?? ""}
                  onChange={(event) =>
                    updateConferenceOverride(item.id, {
                      speaker: event.target.value,
                    })
                  }
                />
                <Input
                  size="small"
                  aria-label={`Track for item ${itemNumber}`}
                  value={override?.track ?? ""}
                  onChange={(event) =>
                    updateConferenceOverride(item.id, {
                      track: event.target.value,
                    })
                  }
                />
                <Input
                  size="small"
                  aria-label={`Tags for item ${itemNumber}`}
                  value={tagDrafts[item.id] ?? stringifyTags(override?.tags)}
                  placeholder="tag, tag"
                  onChange={(event) =>
                    setTagDrafts((prev) => ({
                      ...prev,
                      [item.id]: event.target.value,
                    }))
                  }
                  onBlur={(event) => commitConferenceTags(item.id, event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") event.currentTarget.blur()
                  }}
                />
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}

export default ItemMetadataTable
