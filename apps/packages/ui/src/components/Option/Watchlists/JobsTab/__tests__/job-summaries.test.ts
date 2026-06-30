import { describe, expect, it } from "vitest"
import type { WatchlistFilter } from "@/types/watchlists"
import {
  buildScopeTooltipLines,
  summarizeOutputLinkage,
  summarizeFilters,
  summarizeOverflowList,
  summarizeScopeCounts
} from "../job-summaries"

const t = (_key: string, defaultValue: string): string => defaultValue

describe("job summary helpers", () => {
  it("summarizes scope counts with feed/group/tag totals", () => {
    const summary = summarizeScopeCounts(
      {
        sources: [1, 2],
        groups: [9],
        tags: ["tech", "ai"]
      },
      t
    )
    expect(summary).toBe("2 feeds, 1 group, 2 tags")
  })

  it("truncates long name lists with overflow count", () => {
    const overflow = summarizeOverflowList(["One", "Two", "Three", "Four"], 2)
    expect(overflow.visible).toEqual(["One", "Two"])
    expect(overflow.hiddenCount).toBe(2)
    expect(overflow.text).toBe("One, Two +2")
  })

  it("builds scope tooltip lines with name lookup and fallback IDs", () => {
    const lines = buildScopeTooltipLines(
      {
        sources: [1, 2, 3, 4],
        groups: [10],
        tags: ["tech"]
      },
      {
        sources: {
          1: "TechCrunch",
          2: "Ars Technica"
        },
        groups: {
          10: "Daily News"
        }
      },
      t,
      3
    )

    expect(lines).toEqual([
      "Feeds: TechCrunch, Ars Technica, #3 +1",
      "Groups: Daily News",
      "Tags: tech"
    ])
  })

  it("summarizes filter rules with compact preview and per-filter tooltip text", () => {
    const filters: WatchlistFilter[] = [
      {
        type: "keyword",
        action: "include",
        value: { keywords: ["ai", "safety", "policy"] }
      },
      {
        type: "author",
        action: "exclude",
        value: { authors: ["spam-bot"] }
      }
    ]

    const summary = summarizeFilters(filters, t)
    expect(summary.count).toBe(2)
    expect(summary.preview).toBe("Include keyword: ai, safety +1 (1 more)")
    expect(summary.tooltipLines).toEqual([
      "Include keyword: ai, safety +1",
      "Exclude author: spam-bot"
    ])
  })

  it("only summarizes delivery channels with explicit enabled flags", () => {
    expect(
      summarizeOutputLinkage(
        {
          auto_output: { enabled: true },
          deliveries: {
            email: { recipients: ["digest@example.com"] },
            chatbook: { title: "Daily Digest" }
          }
        },
        t
      )
    ).toBe("Create a report after each scheduled run • Template: default • Reports tab only • Text report only")

    expect(
      summarizeOutputLinkage(
        {
          auto_output: { enabled: true },
          deliveries: {
            email: { enabled: true, recipients: ["digest@example.com"] },
            chatbook: { enabled: true, title: "Daily Digest" }
          }
        },
        t
      )
    ).toBe("Create a report after each scheduled run • Template: default • Deliver by email • Save to Chatbook • Text report only")
  })
})
