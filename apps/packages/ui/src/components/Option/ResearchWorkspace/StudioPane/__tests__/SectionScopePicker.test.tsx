import { describe, expect, it } from "vitest"
import { numberedChapterNodes } from "../SectionScopePicker"
import type { MediaNavigationNode } from "@/hooks/useMediaNavigation"

const node = (id: string, title: string, level = 2): MediaNavigationNode => ({
  id, title, level, parent_id: null, order: 0, path_label: null, target_type: "page",
  target_start: 1, target_end: null, target_href: null, source: "generated_toc", confidence: 0.85
})

describe("reviewed numbered chapter proposals", () => {
  it("selects the consecutive chapter level while retaining contribution children in the outline", () => {
    const nodes = [node("group", "SECTION I", 1), node("first", "1. First chapter"),
      node("for", "Case for retention", 3), node("against", "Case for closure", 3), node("last", "2. Last chapter")]
    expect(numberedChapterNodes(nodes).map(item => item.id)).toEqual(["first", "last"])
    expect(nodes).toHaveLength(5)
  })
  it("does not claim a fragmented or untrusted heading series is a chapter list", () => {
    expect(numberedChapterNodes([node("first", "1. First"), node("third", "3. Third")])).toEqual([])
    expect(numberedChapterNodes([node("first", "1. First"), { ...node("last", "2. Last"), source: "generated" }])).toEqual([])
  })
})
