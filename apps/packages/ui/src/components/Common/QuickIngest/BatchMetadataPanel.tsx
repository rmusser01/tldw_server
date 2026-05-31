import React, { useCallback, useMemo, useState } from "react"
import { Input, Tag, Typography } from "antd"
import type { ConferenceBatchMetadata } from "./types"
import { useIngestWizard } from "./IngestWizardContext"
import { ItemMetadataTable } from "./ItemMetadataTable"

const parseTags = (value: string): string[] =>
  value
    .split(",")
    .map((tag) => tag.trim())
    .filter(Boolean)

const inferCollectionName = (
  current: ConferenceBatchMetadata | null,
  playlistTitle?: string | null
): string => current?.collectionName || playlistTitle || ""

const isLikelyConferenceVideo = (itemUrl?: string): boolean => {
  if (!itemUrl) return false
  try {
    const hostname = new URL(itemUrl).hostname.toLowerCase()
    return hostname === "youtube.com" ||
      hostname.endsWith(".youtube.com") ||
      hostname === "youtu.be"
  } catch {
    return false
  }
}

export const BatchMetadataPanel: React.FC = () => {
  const { state, setConferenceBatchMetadata } = useIngestWizard()
  const { queueItems, conferenceBatchMetadata } = state
  const [sharedTagInput, setSharedTagInput] = useState(
    () => conferenceBatchMetadata?.sharedTags.join(", ") ?? ""
  )

  const firstPlaylist = queueItems.find((item) => item.playlist?.playlistTitle)
    ?.playlist
  const selectedCount = useMemo(
    () =>
      queueItems.filter((item) => item.conferenceOverride?.selected !== false)
        .length,
    [queueItems]
  )
  const hasPlaylistItems = queueItems.some((item) => item.playlist)
  const likelyConferenceVideoBatch =
    queueItems.length > 1 &&
    queueItems.every(
      (item) =>
        item.detectedType === "video" ||
        Boolean(item.playlist) ||
        isLikelyConferenceVideo(item.url)
    )
  const shouldShow = hasPlaylistItems || likelyConferenceVideoBatch

  const metadata: ConferenceBatchMetadata = useMemo(
    () => ({
      collectionName: inferCollectionName(
        conferenceBatchMetadata,
        firstPlaylist?.playlistTitle
      ),
      conferenceName: conferenceBatchMetadata?.conferenceName ?? "",
      eventDate: conferenceBatchMetadata?.eventDate ?? "",
      eventYear: conferenceBatchMetadata?.eventYear ?? "",
      sharedTags: conferenceBatchMetadata?.sharedTags ?? [],
      sourcePlaylistUrl: conferenceBatchMetadata?.sourcePlaylistUrl ?? "",
    }),
    [conferenceBatchMetadata, firstPlaylist?.playlistTitle]
  )

  const updateMetadata = useCallback(
    (patch: Partial<ConferenceBatchMetadata>) => {
      setConferenceBatchMetadata({
        ...metadata,
        ...patch,
        sharedTags: patch.sharedTags ?? metadata.sharedTags,
      })
    },
    [metadata, setConferenceBatchMetadata]
  )

  if (!shouldShow) return null

  return (
    <section
      className="mt-4 rounded-md border border-border bg-surface px-3 py-3"
      aria-label="Conference batch metadata"
    >
      <div className="flex items-center justify-between gap-3">
        <div className="min-w-0">
          <Typography.Text className="block text-sm font-medium text-text">
            Conference batch
          </Typography.Text>
          <Typography.Text className="block text-[11px] text-text-muted">
            {selectedCount} selected
          </Typography.Text>
        </div>
        {metadata.sharedTags.length > 0 && (
          <div className="flex flex-wrap justify-end gap-1">
            {metadata.sharedTags.map((tag) => (
              <Tag key={tag} className="!mr-0">
                {tag}
              </Tag>
            ))}
          </div>
        )}
      </div>

      <div className="mt-3 grid gap-2 sm:grid-cols-2">
        <label className="text-xs text-text-muted">
          Collection name
          <Input
            className="mt-1"
            aria-label="Collection name"
            value={metadata.collectionName}
            onChange={(event) =>
              updateMetadata({ collectionName: event.target.value })
            }
          />
        </label>
        <label className="text-xs text-text-muted">
          Conference name
          <Input
            className="mt-1"
            aria-label="Conference name"
            value={metadata.conferenceName}
            onChange={(event) =>
              updateMetadata({ conferenceName: event.target.value })
            }
          />
        </label>
        <label className="text-xs text-text-muted">
          Event year
          <Input
            className="mt-1"
            aria-label="Event year"
            value={metadata.eventYear}
            onChange={(event) => updateMetadata({ eventYear: event.target.value })}
          />
        </label>
        <label className="text-xs text-text-muted">
          Shared tags
          <Input
            className="mt-1"
            aria-label="Shared tags"
            value={sharedTagInput}
            placeholder="conference, track"
            onChange={(event) => {
              setSharedTagInput(event.target.value)
              updateMetadata({ sharedTags: parseTags(event.target.value) })
            }}
          />
        </label>
      </div>

      <ItemMetadataTable />
    </section>
  )
}

export default BatchMetadataPanel
