import { describe, expect, it } from "vitest"
import {
  buildQuickIngestOpenDetailFromUrl,
  createQuickIngestSessionSeedFromOpenDetail,
  isQuickIngestPlaylistPreflightDetail,
  requestQuickIngestOpen
} from "../quick-ingest-open"
import type { QuickIngestOpenDetail } from "../quick-ingest-open"

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
      action: "playlist_preflight"
    })

    const playlistDetail = buildQuickIngestOpenDetailFromUrl(
      "https://www.youtube.com/playlist?list=PL0065D9B288E6804B"
    )
    expect(isQuickIngestPlaylistPreflightDetail(playlistDetail)).toBe(true)
    expect(
      isQuickIngestPlaylistPreflightDetail(playlistDetail)
        ? playlistDetail.sourceKind
        : null
    ).toBe("youtube_playlist")
    expect(
      buildQuickIngestOpenDetailFromUrl(
        "https://www.youtube.com/watch?v=abc123"
      )
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
      action: "playlist_preflight" as const
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
      action: "playlist_preflight" as const
    }

    expect(createQuickIngestSessionSeedFromOpenDetail(detail)).toEqual({
      openDetail: detail,
      firstSourceAddMode: null
    })
  })

  it("creates a quick first-source wizard seed from first-source milestone metadata", () => {
    const detail = {
      source: "first_source_milestone" as const,
      preferredPreset: "quick" as const,
      firstSource: true,
      firstSourceKind: "paste_text" as const
    }

    expect(createQuickIngestSessionSeedFromOpenDetail(detail)).toMatchObject({
      openDetail: detail,
      firstSourceAddMode: "paste_text",
      selectedPreset: "quick",
      customBasePreset: "quick",
      presetConfig: {
        storeRemote: true,
        reviewBeforeStorage: false,
        common: {
          perform_analysis: false,
          perform_chunking: true
        },
        typeDefaults: {
          document: { ocr: false }
        }
      }
    })
  })

  it("falls back when first-source metadata names an inherited object key as a preset", () => {
    const detail = {
      source: "first_source_milestone" as const,
      preferredPreset: "toString",
      firstSource: true
    } as unknown as QuickIngestOpenDetail

    expect(createQuickIngestSessionSeedFromOpenDetail(detail)).toMatchObject({
      firstSourceAddMode: "web_url",
      selectedPreset: "quick",
      customBasePreset: "quick",
      presetConfig: {
        storeRemote: true,
        reviewBeforeStorage: false
      }
    })
  })

  it("does not change regular quick-ingest opens without first-source metadata", () => {
    expect(
      createQuickIngestSessionSeedFromOpenDetail({ source: "global" })
    ).toBeNull()
  })
})
