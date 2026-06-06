import { describe, expect, it } from "vitest"

import {
  parseChecklistItems,
  toggleChecklistItemMarker
} from "@/components/Notes/task-markdown"

describe("task markdown helpers", () => {
  it("parses checklist lines for local dirty toggles", () => {
    const items = parseChecklistItems("- [ ] Draft PRD\n- [x] Review MCP tools")

    expect(items).toEqual([
      expect.objectContaining({
        lineNumber: 1,
        lineIndex: 0,
        checked: false,
        text: "Draft PRD",
        marker: "[ ]",
        hasChildContent: false
      }),
      expect.objectContaining({
        lineNumber: 2,
        lineIndex: 1,
        checked: true,
        text: "Review MCP tools",
        marker: "[x]",
        hasChildContent: false
      })
    ])
  })

  it("toggles a local checkbox marker without removing unknown metadata tokens", () => {
    const markdown = "- [ ] Draft PRD <!-- task:abc --> {#keep-me}\nplain text"

    expect(toggleChecklistItemMarker(markdown, 1, true)).toBe(
      "- [x] Draft PRD <!-- task:abc --> {#keep-me}\nplain text"
    )
    expect(toggleChecklistItemMarker(markdown, 1, false)).toBe(markdown)
  })

  it("detects indented child content below checklist items", () => {
    const markdown = [
      "- [ ] Parent task",
      "  - child bullet",
      "  continued detail",
      "- [ ] Next task"
    ].join("\n")

    const items = parseChecklistItems(markdown)

    expect(items[0]).toEqual(expect.objectContaining({ hasChildContent: true }))
    expect(items[1]).toEqual(expect.objectContaining({ hasChildContent: false }))
  })
})
