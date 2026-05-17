import { describe, expect, it } from "vitest"
import {
  buildQuickIngestOpenDetailFromUrl,
  createQuickIngestSessionSeedFromOpenDetail,
  requestQuickIngestOpen,
} from "../quick-ingest-open"

describe("quick ingest open handoff", () => {
  it("builds a typed extension playlist preflight detail from an active tab URL", () => {
    expect(
      buildQuickIngestOpenDetailFromUrl(
        "https://www.youtube.com/watch?v=PrNmmN6qBiw&list=PL0065D9B288E6804B"
      )
    ).toEqual({
      source: "extension_active_tab",
      url: "https://www.youtube.com/watch?v=PrNmmN6qBiw&list=PL0065D9B288E6804B",
      sourceKind: "youtube_watch_playlist",
      action: "playlist_preflight",
    })

    expect(
      buildQuickIngestOpenDetailFromUrl(
        "https://www.youtube.com/playlist?list=PL0065D9B288E6804B"
      )?.sourceKind
    ).toBe("youtube_playlist")
    expect(
      buildQuickIngestOpenDetailFromUrl("https://www.youtube.com/watch?v=abc123")
    ).toBeNull()
  })

  it("stores typed detail on the pending request and dispatches it in the event", () => {
    const events: unknown[] = []
    const listener = (event: Event) => {
      events.push((event as CustomEvent).detail)
    }
    window.addEventListener("tldw:open-quick-ingest", listener)

    const detail = {
      source: "extension_active_tab" as const,
      url: "https://www.youtube.com/watch?v=a&list=PLx",
      sourceKind: "youtube_watch_playlist" as const,
      action: "playlist_preflight" as const,
    }

    try {
      const request = requestQuickIngestOpen(detail)

      expect(request?.detail).toEqual(detail)
      expect(events).toEqual([detail])
    } finally {
      window.removeEventListener("tldw:open-quick-ingest", listener)
    }
  })

  it("creates a wizard session seed that starts playlist preflight", () => {
    const detail = {
      source: "extension_active_tab" as const,
      url: "https://www.youtube.com/watch?v=a&list=PLx",
      sourceKind: "youtube_watch_playlist" as const,
      action: "playlist_preflight" as const,
    }

    expect(createQuickIngestSessionSeedFromOpenDetail(detail)).toEqual({
      openDetail: detail,
    })
  })
})
