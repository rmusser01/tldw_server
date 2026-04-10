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

    expect(source).toMatch(/disabled=\{!newCharName\.trim\(\)\s*\|\|\s*!isOnline\}/)
    expect(source).toMatch(/disabled=\{!newWorldName\.trim\(\)\s*\|\|\s*!isOnline\}/)
    expect(source).toMatch(/disabled=\{!newPlotTitle\.trim\(\)\s*\|\|\s*!isOnline\}/)
    expect(source).toMatch(/onPressEnter=\{\(\)\s*=>\s*isOnline\s*&&\s*newCharName\.trim\(\)\s*&&\s*!addCharMutation\.isPending\s*&&\s*addCharMutation\.mutate\(newCharName\.trim\(\)\)\}/)
    expect(source).toMatch(/onPressEnter=\{\(\)\s*=>\s*isOnline\s*&&\s*newWorldName\.trim\(\)\s*&&\s*!addWorldMutation\.isPending\s*&&\s*addWorldMutation\.mutate\(\{\s*name:\s*newWorldName\.trim\(\),\s*kind:\s*newWorldKind\s*\}\)\}/)
    expect(source).toMatch(/onPressEnter=\{\(\)\s*=>\s*isOnline\s*&&\s*newPlotTitle\.trim\(\)\s*&&\s*!addPlotMutation\.isPending\s*&&\s*addPlotMutation\.mutate\(newPlotTitle\.trim\(\)\)\}/)
  })

  it("only clears fields when the submitted value still matches the live draft", () => {
    const source = readCharacterWorldSource()

    expect(source).toContain('setNewCharName((current) => current === name ? "" : current)')
    expect(source).toContain('setNewWorldName((current) => current === variables.name ? "" : current)')
    expect(source).toContain('setNewPlotTitle((current) => current === title ? "" : current)')
  })

  it("surfaces create failures with an antd message toast", () => {
    const source = readCharacterWorldSource()

    expect(source).toContain("message.error(err.message || \"Failed to create character\")")
    expect(source).toContain("message.error(err.message || \"Failed to create world info\")")
    expect(source).toContain("message.error(err.message || \"Failed to create plot line\")")
  })

  it("gives icon-only manuscript controls accessible names", () => {
    const source = readCharacterWorldSource()

    expect(source).toContain('aria-label="New character name"')
    expect(source).toContain('aria-label="Add character"')
    expect(source).toContain('aria-label="World info kind"')
    expect(source).toContain('aria-label="New world info entry name"')
    expect(source).toContain('aria-label="Add world info entry"')
    expect(source).toContain('aria-label="New plot line title"')
    expect(source).toContain('aria-label="Add plot line"')
  })

  it("styles citation marks with theme variables instead of hardcoded blues", () => {
    const source = readTailwindSource()

    expect(source).toContain("background: rgb(var(--color-primary) / 0.1);")
    expect(source).toContain("border-bottom: 1px dashed rgb(var(--color-primary));")
    expect(source).not.toContain("background: rgba(59, 130, 246, 0.1);")
    expect(source).not.toContain("border-bottom: 1px dashed #3b82f6;")
  })
})
