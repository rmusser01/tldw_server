import React from "react"
import { fireEvent, render, screen, within } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { ConferenceCollectionReview } from "../ConferenceCollectionReview"
import type {
  MediaCollection,
  MediaCollectionItem,
} from "@/services/tldw/conference-collections"

const makeItem = (
  overrides: Partial<MediaCollectionItem>
): MediaCollectionItem => ({
  id: overrides.id ?? 1,
  collectionId: overrides.collectionId ?? 44,
  ordinal: overrides.ordinal ?? 1,
  sourceUrl: overrides.sourceUrl ?? "https://www.youtube.com/watch?v=talk",
  normalizedSourceId: overrides.normalizedSourceId ?? null,
  sourceKind: overrides.sourceKind ?? "youtube",
  title: overrides.title ?? "Untitled talk",
  speaker: overrides.speaker ?? null,
  publishedAt: overrides.publishedAt ?? null,
  track: overrides.track ?? null,
  duplicateStatus: overrides.duplicateStatus ?? "new",
  status: overrides.status ?? "completed",
  mediaId: overrides.mediaId ?? null,
  contentItemId: overrides.contentItemId ?? null,
  latestJobId: overrides.latestJobId ?? null,
  latestRunId: overrides.latestRunId ?? null,
  idempotencyKey: overrides.idempotencyKey ?? null,
  retryCount: overrides.retryCount ?? 0,
  errorSummary: overrides.errorSummary ?? null,
  warnings: overrides.warnings ?? [],
  metadata: overrides.metadata ?? {},
  tags: overrides.tags ?? [],
  createdAt: overrides.createdAt ?? "2026-05-16T10:00:00.000Z",
  updatedAt: overrides.updatedAt ?? "2026-05-16T10:00:00.000Z",
})

const makeCollection = (
  items: MediaCollectionItem[]
): MediaCollection => ({
  id: 44,
  name: "Strange Loop 2010",
  kind: "conference",
  description: "Conference talks",
  sourceUrl: "https://www.youtube.com/playlist?list=PL0065D9B288E6804B",
  metadata: {
    conferenceName: "Strange Loop",
    year: "2010",
  },
  defaultTags: ["strange-loop", "conference"],
  createdAt: "2026-05-16T10:00:00.000Z",
  updatedAt: "2026-05-16T10:00:00.000Z",
  items,
})

describe("ConferenceCollectionReview", () => {
  it("shows ordered talks, readiness, navigation, comparison, and scoped QA", () => {
    const onAskCollection = vi.fn()
    const onOpenMedia = vi.fn()
    const collection = makeCollection([
      makeItem({
        id: 1,
        ordinal: 2,
        title: "Compiler internals",
        speaker: "Grace",
        track: "Languages",
        status: "skipped_existing",
        mediaId: 102,
        metadata: {
          summary: "Compiler summary",
          excerpt: "Compiler excerpt",
        },
      }),
      makeItem({
        id: 2,
        ordinal: 1,
        title: "Macro keynote",
        speaker: "Ada",
        track: "Keynote",
        status: "completed",
        mediaId: 101,
        metadata: {
          summary: "Macro summary",
          excerpt: "Macro excerpt",
        },
      }),
      makeItem({
        id: 3,
        ordinal: 3,
        title: "Panel discussion",
        speaker: "Alan",
        status: "processing",
        mediaId: 103,
      }),
    ])

    render(
      <ConferenceCollectionReview
        collection={collection}
        onAskCollection={onAskCollection}
        onOpenMedia={onOpenMedia}
      />
    )

    expect(screen.getByRole("heading", { name: "Strange Loop 2010" })).toBeInTheDocument()
    expect(screen.getByText("2 ready")).toBeInTheDocument()
    expect(screen.getByText("1 in progress")).toBeInTheDocument()

    const rows = within(screen.getByTestId("conference-talk-list")).getAllByRole("listitem")
    expect(rows.map((row) => row.textContent)).toEqual([
      expect.stringContaining("Macro keynote"),
      expect.stringContaining("Compiler internals"),
      expect.stringContaining("Panel discussion"),
    ])
    expect(rows[0]).toHaveTextContent("Completed")
    expect(rows[1]).toHaveTextContent("Skipped existing")
    expect(rows[2]).toHaveTextContent("Processing")

    expect(screen.getByRole("heading", { name: "Macro keynote" })).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Next talk" }))
    expect(screen.getByRole("heading", { name: "Compiler internals" })).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Open media" }))
    expect(onOpenMedia).toHaveBeenCalledWith(102)

    fireEvent.click(screen.getByLabelText("Compare Macro keynote"))
    fireEvent.click(screen.getByLabelText("Compare Compiler internals"))
    expect(screen.getByText("Comparing 2 talks")).toBeInTheDocument()
    expect(screen.getByText("Macro summary")).toBeInTheDocument()
    expect(screen.getByText("Compiler excerpt")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Ask this collection" }))
    expect(onAskCollection).toHaveBeenCalledWith({
      collectionId: 44,
      mediaIds: [101, 102],
    })
  })

  it("disables scoped QA with readiness copy when no talks are ready", () => {
    const onAskCollection = vi.fn()
    const collection = makeCollection([
      makeItem({
        id: 1,
        ordinal: 1,
        title: "Queued talk",
        status: "planned",
        mediaId: null,
      }),
      makeItem({
        id: 2,
        ordinal: 2,
        title: "Failed talk",
        status: "failed",
        mediaId: null,
        errorSummary: "Download failed",
      }),
    ])

    render(
      <ConferenceCollectionReview
        collection={collection}
        onAskCollection={onAskCollection}
      />
    )

    const askButton = screen.getByRole("button", { name: "Ask this collection" })
    expect(askButton).toBeDisabled()
    expect(screen.getByText("No ready talks yet")).toBeInTheDocument()
    expect(screen.getByText("Waiting for completed or skipped-existing items.")).toBeInTheDocument()

    fireEvent.click(askButton)
    expect(onAskCollection).not.toHaveBeenCalled()
  })
})
