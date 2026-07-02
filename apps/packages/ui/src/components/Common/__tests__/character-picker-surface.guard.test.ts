import fs from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

describe("character picker popup surfaces", () => {
  it("keeps shared assistant picker popups on an opaque themed surface", () => {
    const sourcePath = path.resolve(__dirname, "../AssistantSelect.tsx")
    const source = fs.readFileSync(sourcePath, "utf8")

    expect(source).toContain('overlayClassName="assistant-select-dropdown')
    expect(source).toContain('data-testid="assistant-select-panel"')
    expect(source).toContain("bg-surface")
    expect(source).not.toContain("bg-background shadow-lg")
  })

  it("keeps character picker popups on an opaque themed surface", () => {
    const sourcePath = path.resolve(__dirname, "../CharacterSelect.tsx")
    const source = fs.readFileSync(sourcePath, "utf8")

    expect(source).toContain('overlayClassName="character-select-dropdown')
    expect(source).toContain('data-testid="character-select-popup"')
    expect(source).toContain("bg-surface")
  })
})
