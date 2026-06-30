import { describe, expect, it } from "vitest"

import {
  isPlaylistPreflightDuplicate,
  normalizePlaylistPreflightResponse
} from "@/services/tldw/playlist-preflight"

describe("playlist preflight normalizers", () => {
  it("normalizes item IDs, selected defaults, and duplicate counts", () => {
    const normalized = normalizePlaylistPreflightResponse({
      source_url: "https://www.youtube.com/playlist?list=PLtest",
      source_kind: "youtube_playlist",
      playlist_id: "PLtest",
      playlist_title: "Conference 2010",
      item_count: 2,
      selected_count: 1,
      duplicate_count: 1,
      warnings: ["One duplicate"],
      items: [
        {
          ordinal: 1,
          source_url: "https://www.youtube.com/watch?v=abc123",
          normalized_source_id: "youtube:video:abc123",
          source_kind: "youtube_video",
          title: "Opening Keynote",
          duplicate_status: "new"
        },
        {
          ordinal: 2,
          source_url: "https://www.youtube.com/watch?v=abc123",
          normalized_source_id: "youtube:video:abc123",
          source_kind: "youtube_video",
          title: "Opening Keynote again",
          duplicate_status: "duplicate_in_batch",
          duplicate_of_ordinal: 1,
          selected: false
        },
        {
          ordinal: 3,
          source_url: "https://www.youtube.com/watch?v=def456",
          normalized_source_id: "youtube:video:def456",
          source_kind: "youtube_video",
          title: "Unknown status talk",
          duplicate_status: "server_future_status"
        }
      ]
    })

    expect(normalized.playlistId).toBe("PLtest")
    expect(normalized.items[0]).toMatchObject({
      id: "youtube:video:abc123",
      ordinal: 1,
      selected: true
    })
    expect(normalized.items[1].selected).toBe(false)
    expect(normalized.items[2]).toMatchObject({
      duplicateStatus: "new",
      selected: true
    })
    expect(normalized.duplicateCount).toBe(1)
    expect(isPlaylistPreflightDuplicate(normalized.items[1])).toBe(true)
  })
})
