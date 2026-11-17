// @vitest-environment jsdom
/**
 * Characterization tests for data-tables store editing clones.
 *
 * `startEditing` / `discardChanges` deep-clone the table with
 * `structuredClone` (WP-21, previously `JSON.parse(JSON.stringify(...))`).
 * These tests pin the clone's observable shape so the conversion cannot
 * silently drift:
 * - editingTable/originalTable must be deep copies (in-place mutation of the
 *   source table must not leak into the editing session);
 * - for the JSON-origin DataTable shapes this store holds (API responses with
 *   ISO string timestamps, null-able cells, nested objects/arrays), the clone
 *   must be indistinguishable from a JSON round-trip — no consumer relied on
 *   JSON's undefined-dropping or Date-stringification;
 * - discardChanges must restore the editing table from the original snapshot.
 */
import { beforeEach, describe, expect, it } from "vitest"
import { useDataTablesStore } from "../data-tables"
import type { DataTable } from "@/types/data-tables"

const makeTable = (): DataTable => ({
  id: "table-1",
  name: "Test table",
  description: "A table",
  prompt: "extract rows",
  columns: [
    { id: "col-1", name: "name", type: "text" },
    { id: "col-2", name: "age", type: "number" },
    { id: "col-3", name: "meta", type: "text" }
  ],
  rows: [
    { name: "Alice", age: 30, meta: { tags: ["a", "b"] } },
    { name: "Bob", age: null, meta: [] }
  ],
  sources: [
    { type: "chat", id: "chat-1", title: "Chat", snippet: "..." }
  ],
  created_at: "2026-10-01T00:00:00Z", // ISO strings, never Date objects
  updated_at: "2026-10-02T00:00:00Z",
  row_count: 2
})

const resetStore = () =>
  useDataTablesStore.setState({
    editingTable: null,
    originalTable: null,
    editingRows: [],
    editingState: {
      editingCellKey: null,
      isDirty: false,
      pendingChanges: []
    }
  })

describe("data-tables editing clones (structuredClone)", () => {
  beforeEach(() => {
    resetStore()
  })

  it("startEditing deep-clones the table into editingTable and originalTable", () => {
    const table = makeTable()

    useDataTablesStore.getState().startEditing(table)

    // Mutate the source object in place after the clones were taken.
    table.rows[0]!.name = "Mutated"
    table.columns.push({ id: "col-4", name: "extra", type: "text" })
    table.sources[0]!.title = "Mutated source"

    const state = useDataTablesStore.getState()
    expect(state.editingTable!.rows[0]!.name).toBe("Alice")
    expect(state.editingTable!.columns).toHaveLength(3)
    expect(state.editingTable!.sources[0]!.title).toBe("Chat")
    expect(state.originalTable!.rows[0]!.name).toBe("Alice")
  })

  it("editing/original clones are JSON-round-trip identical to the source table", () => {
    const table = makeTable()

    useDataTablesStore.getState().startEditing(table)

    const state = useDataTablesStore.getState()
    // If any cloned value were structuredClone-only (an undefined-valued key,
    // a Date, a Map...), the JSON round-trip would not deep-equal the clone.
    expect(state.editingTable).toEqual(JSON.parse(JSON.stringify(table)))
    expect(state.originalTable).toEqual(JSON.parse(JSON.stringify(table)))
  })

  it("startEditing produces two independent clones and stable row ids", () => {
    const table = makeTable()

    useDataTablesStore.getState().startEditing(table)

    const state = useDataTablesStore.getState()
    expect(state.editingTable).not.toBe(state.originalTable)
    expect(state.editingRows).toHaveLength(2)
    state.editingRows.forEach((row) => {
      expect(typeof row._id).toBe("string")
      expect(row._id.length).toBeGreaterThan(0)
    })
  })

  it("discardChanges restores the editing table from the original snapshot", () => {
    useDataTablesStore.getState().startEditing(makeTable())

    useDataTablesStore.getState().updateCell(0, "name", "Changed")
    expect(useDataTablesStore.getState().editingState.isDirty).toBe(true)

    useDataTablesStore.getState().discardChanges()

    const state = useDataTablesStore.getState()
    expect(state.editingTable).toEqual(state.originalTable)
    expect(state.editingRows[0]!.name).toBe("Alice")
    expect(state.editingState.isDirty).toBe(false)
    expect(state.editingState.pendingChanges).toEqual([])
  })
})
