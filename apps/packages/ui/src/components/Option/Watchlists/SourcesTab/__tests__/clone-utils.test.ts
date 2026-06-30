import { describe, expect, it } from "vitest"
import { buildClonedWatchlistSourcePayload } from "../clone-utils"
import type { WatchlistSource } from "@/types/watchlists"

const buildSource = (): WatchlistSource => ({
  id: 9,
  name: "Example News RSS",
  url: "https://example.com/rss.xml",
  source_type: "rss",
  active: true,
  tags: ["news", "daily"],
  group_ids: [3, 4],
  watchlist_ids: [2],
  settings: {
    fetch: { user_agent: "tldw-test" },
    extraction: { title_selector: "h1" },
    dedupe: { identity: ["canonical_url", "title"] }
  },
  last_scraped_at: "2026-05-01T12:00:00Z",
  status: "healthy",
  created_at: "2026-05-01T10:00:00Z",
  updated_at: "2026-05-01T11:00:00Z"
})

describe("buildClonedWatchlistSourcePayload", () => {
  it("preserves source rules and assignments while resetting runtime scrape state", () => {
    const original = buildSource()

    const clone = buildClonedWatchlistSourcePayload(original, 2)

    expect(clone).toEqual({
      name: "Example News RSS copy",
      url: "https://example.com/rss.xml",
      source_type: "rss",
      active: false,
      tags: ["news", "daily"],
      settings: original.settings,
      group_ids: [3, 4],
      watchlist_id: 2
    })
    expect(JSON.stringify(clone)).not.toContain("last_scraped_at")
    expect(JSON.stringify(clone)).not.toContain("status")
    expect(JSON.stringify(clone)).not.toContain("seen")
  })

  it("deep clones nested settings and list fields", () => {
    const original = buildSource()
    const clone = buildClonedWatchlistSourcePayload(original, 2)

    clone.tags!.push("copy")
    ;(clone.settings!.dedupe as Record<string, string[]>).identity.push("published_at")
    clone.group_ids!.push(5)

    expect(original.tags).toEqual(["news", "daily"])
    expect((original.settings!.dedupe as Record<string, string[]>).identity).toEqual([
      "canonical_url",
      "title"
    ])
    expect(original.group_ids).toEqual([3, 4])
  })
})
