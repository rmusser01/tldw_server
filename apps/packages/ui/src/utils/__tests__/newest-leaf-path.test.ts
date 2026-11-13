import { describe, expect, it } from "vitest"
import { newestLeafPath, newestLeafUnder, serverChatActivePath, variantBranchCursor } from "../newest-leaf-path"

type Row = { id: string; parent_id: string | null }
const row = (id: string, parent_id: string | null = null): Row => ({ id, parent_id })
const ids = (rows: readonly Row[]) => rows.map((entry) => entry.id)
const link = (entry: Row) => ({ id: entry.id, parentId: entry.parent_id })

describe("newestLeafPath (CM-N2)", () => {
  it("keeps an unlinked legacy chat linear: a row without a parent continues the row before it", () => {
    const rows = [row("u1"), row("a1"), row("u2"), row("a2")]
    expect(ids(newestLeafPath(rows, link).path)).toEqual(["u1", "a1", "u2", "a2"])
    expect(newestLeafPath(rows, link).hasAlternatives).toBe(false)
  })

  it("leaves out an older alternative reply to the same question", () => {
    const rows = [row("u1"), row("a1", "u1"), row("a1b", "u1"), row("u2"), row("a2", "u2")]
    const { path, hasAlternatives } = newestLeafPath(rows, link)
    expect(ids(path)).toEqual(["u1", "a1b", "u2", "a2"])
    expect(hasAlternatives).toBe(true)
  })

  it("follows the newest branch when a later alternative replaced the rest of the chat", () => {
    // a1 was answered, the chat went on, then a1 was regenerated as a1b.
    const rows = [row("u1"), row("a1", "u1"), row("u2", "a1"), row("a2", "u2"), row("a1b", "u1")]
    expect(ids(newestLeafPath(rows, link).path)).toEqual(["u1", "a1b"])
  })

  it("treats a parent that is missing as unlinked", () => {
    const rows = [row("u1"), row("a1", "deleted"), row("u2"), row("a2", "u2")]
    expect(ids(newestLeafPath(rows, link).path)).toEqual(["u1", "a1", "u2", "a2"])
  })

  it("still ends, keeping every row, when parents form a cycle", () => {
    const rows = [row("u1"), row("a1", "u1"), row("u2", "a9"), row("a9", "u2")]
    expect([...ids(newestLeafPath(rows, link).path)].sort()).toEqual(["a1", "a9", "u1", "u2"])
  })

  it("puts a reply listed before its question back after it (review case)", () => {
    const rows = [row("u1"), row("a1", "u1"), row("a2", "u2"), row("u2", "a1")]
    const { path, hasAlternatives } = newestLeafPath(rows, link)
    expect(ids(path)).toEqual(["u1", "a1", "u2", "a2"])
    expect(hasAlternatives).toBe(false)
  })

  it("returns nothing for an empty chat", () => {
    expect(newestLeafPath([], link)).toEqual({ path: [], hasAlternatives: false })
  })

  it("reads server messages by their parent_message_id", () => {
    const message = (id: string, parent_message_id: string | null) =>
      ({ id, parent_message_id, role: "user" as const, content: id, created_at: "2026-11-13T00:00:00Z" })
    const { messages, hasAlternatives } = serverChatActivePath([
      message("u1", null), message("a1", "u1"), message("a1b", "u1")
    ])
    expect(messages.map((entry) => entry.id)).toEqual(["u1", "a1b"])
    expect(hasAlternatives).toBe(true)
  })
})

describe("newestLeafUnder (CM-N2 variant swipe)", () => {
  const nodes = [row("u1"), row("a1", "u1"), row("u2", "a1"), row("a2", "u2"), row("a1b", "u1"), row("u3", "a1b")]

  it("goes to the newest message on that alternative's branch", () => {
    expect(newestLeafUnder(nodes, "a1")).toBe("a2")
    expect(newestLeafUnder(nodes, "a1b")).toBe("u3")
  })

  it("stays on a reply nothing follows, or one it can't find", () => {
    expect(newestLeafUnder(nodes, "a2")).toBe("a2")
    expect(newestLeafUnder(nodes, "unknown")).toBe("unknown")
  })
})

describe("variantBranchCursor (CM-N2 variant swipe)", () => {
  const nodes = [row("u1"), row("a1", "u1"), row("u2", "a1"), row("a2", "u2"), row("a1b", "u1")]
  const selection = (kind: string) => ({
    view: { interpretation: { kind } },
    capture: { snapshot: { nodes } }
  })

  it("opens the chosen alternative's newest message in a parent-graph chat", () => {
    expect(variantBranchCursor(selection("parent_graph_v1"), "a1")).toEqual({ kind: "after_message", message_id: "a2" })
  })

  it("keeps the alternative itself for a reviewed legacy order or an unknown capture", () => {
    expect(variantBranchCursor(selection("legacy_linear_v1"), "a1")).toEqual({ kind: "after_message", message_id: "a1" })
    expect(variantBranchCursor({ view: null, capture: null }, "a1")).toEqual({ kind: "after_message", message_id: "a1" })
  })
})
