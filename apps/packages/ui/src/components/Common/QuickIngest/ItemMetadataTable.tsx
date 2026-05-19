import React, { useCallback, useEffect, useState } from "react"
import { Input, Typography } from "antd"
import type {
  ConferenceItemMetadataOverride,
  WizardQueueItem,
} from "./types"
import { useIngestWizard } from "./IngestWizardContext"

const parseTags = (value: string): string[] =>
  value
    .split(",")
    .map((tag) => tag.trim())
    .filter(Boolean)

const stringifyTags = (tags?: string[]): string => (tags || []).join(", ")

const getItemDisplay = (item: WizardQueueItem, index: number): string =>
  item.conferenceOverride?.title ||
  item.fileName ||
  item.url ||
  `Item ${index + 1}`

export const ItemMetadataTable: React.FC = () => {
  const { state, setQueueItems } = useIngestWizard()
  const { queueItems } = state
  const [tagDrafts, setTagDrafts] = useState<Record<string, string>>({})

  const updateOverride = useCallback(
    (
      itemId: string,
      patch: Partial<ConferenceItemMetadataOverride>
    ) => {
      setQueueItems(
        queueItems.map((item) =>
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
    [queueItems, setQueueItems]
  )

  const commitTags = useCallback(
    (itemId: string, rawValue: string) => {
      updateOverride(itemId, { tags: parseTags(rawValue) })
      setTagDrafts((prev) => {
        if (!(itemId in prev)) return prev
        const next = { ...prev }
        delete next[itemId]
        return next
      })
    },
    [updateOverride]
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

  if (queueItems.length === 0) return null

  return (
    <div className="mt-3 rounded-md border border-border">
      <div className="grid grid-cols-[48px_minmax(160px,1.4fr)_minmax(120px,1fr)_minmax(100px,0.8fr)_minmax(120px,1fr)] gap-2 border-b border-border bg-surface2 px-2 py-1.5 text-[11px] font-medium uppercase text-text-muted">
        <span>Use</span>
        <span>Title</span>
        <span>Speaker</span>
        <span>Track</span>
        <span>Tags</span>
      </div>
      <div className="max-h-72 overflow-y-auto">
        {queueItems.map((item, index) => {
          const override = item.conferenceOverride
          const selected = override?.selected ?? true
          const itemNumber = index + 1

          return (
            <div
              key={item.id}
              className="grid grid-cols-[48px_minmax(160px,1.4fr)_minmax(120px,1fr)_minmax(100px,0.8fr)_minmax(120px,1fr)] gap-2 border-b border-border px-2 py-2 last:border-b-0"
            >
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={selected}
                  aria-label={`Include item ${itemNumber}`}
                  onChange={(event) =>
                    updateOverride(item.id, { selected: event.target.checked })
                  }
                />
              </label>
              <div className="min-w-0">
                <Input
                  size="small"
                  aria-label={`Title override for item ${itemNumber}`}
                  value={override?.title ?? ""}
                  placeholder={getItemDisplay(item, index)}
                  onChange={(event) =>
                    updateOverride(item.id, {
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
                  updateOverride(item.id, { speaker: event.target.value })
                }
              />
              <Input
                size="small"
                aria-label={`Track for item ${itemNumber}`}
                value={override?.track ?? ""}
                onChange={(event) =>
                  updateOverride(item.id, { track: event.target.value })
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
                onBlur={(event) => commitTags(item.id, event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") {
                    event.currentTarget.blur()
                  }
                }}
              />
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default ItemMetadataTable
