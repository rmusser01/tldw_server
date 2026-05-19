import fs from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

const readResearchSource = () =>
  fs.readFileSync(path.resolve(__dirname, "..", "ResearchTab.tsx"), "utf8")

const readCharacterWorldSource = () =>
  fs.readFileSync(path.resolve(__dirname, "..", "CharacterWorldTab.tsx"), "utf8")

const readTailwindSource = () =>
  fs.readFileSync(
    path.resolve(__dirname, "..", "..", "..", "..", "assets", "tailwind-shared.css"),
    "utf8"
  )

describe("Writing Phase 2 review fixes", () => {
  it("keeps research results scoped to the submitted scene/query snapshot", () => {
    const source = readResearchSource()

    expect(source).toContain("useRef")
    expect(source).toContain("lastSearchSnapshotRef")
    expect(source).toContain("setResults([])")
    expect(source).toContain("setCitedIds(new Set())")
    expect(source).toContain("query_used: snapshot.query")
  })

  it("guards character, world, and plot creation while offline", () => {
    const source = readCharacterWorldSource()

    expect(source).toContain("disabled={!newCharName.trim() || !isOnline}")
    expect(source).toContain("disabled={!newWorldName.trim() || !isOnline}")
    expect(source).toContain("disabled={!newPlotTitle.trim() || !isOnline}")
    expect(source).toContain("onPressEnter={() => isOnline && newCharName.trim() && !addCharMutation.isPending && addCharMutation.mutate(newCharName.trim())}")
    expect(source).toContain("onPressEnter={() => isOnline && newWorldName.trim() && !addWorldMutation.isPending && addWorldMutation.mutate({ name: newWorldName.trim(), kind: newWorldKind })}")
    expect(source).toContain("onPressEnter={() => isOnline && newPlotTitle.trim() && !addPlotMutation.isPending && addPlotMutation.mutate(newPlotTitle.trim())}")
  })

  it("styles citation marks with theme variables instead of hardcoded blues", () => {
    const source = readTailwindSource()

    expect(source).toContain("background: rgb(var(--color-primary) / 0.1);")
    expect(source).toContain("border-bottom: 1px dashed rgb(var(--color-primary));")
    expect(source).not.toContain("background: rgba(59, 130, 246, 0.1);")
    expect(source).not.toContain("border-bottom: 1px dashed #3b82f6;")
  })
})
