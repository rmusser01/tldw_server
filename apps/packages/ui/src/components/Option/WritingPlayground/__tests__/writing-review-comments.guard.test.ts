import fs from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

const readWritingSource = (relativePath: string) =>
  fs.readFileSync(path.resolve(__dirname, "..", relativePath), "utf8")

describe("writing review comment guards", () => {
  it("removes `as any` casts from the new manuscript UI surfaces", () => {
    const files = [
      "AIAgentTab.tsx",
      "CharacterWorldTab.tsx",
      "ResearchTab.tsx",
      path.join("modals", "ConnectionWebModal.tsx"),
    ]

    for (const file of files) {
      expect(readWritingSource(file)).not.toContain("as any")
    }
  })

  it("moves citation mark styling out of the TipTap extension", () => {
    const extensionSource = readWritingSource(path.join("extensions", "CitationExtension.ts"))
    const cssSource = fs.readFileSync(
      path.resolve(__dirname, "../../../../assets/tailwind-shared.css"),
      "utf8",
    )

    expect(extensionSource).toContain('class: "citation-mark"')
    expect(extensionSource).not.toContain("style:")
    expect(cssSource).toContain(".citation-mark")
  })

  it("keeps newer writing drafts intact and surfaces create failures", () => {
    const source = readWritingSource("CharacterWorldTab.tsx")

    expect(source).toContain('setNewCharName((current) => current.trim() === name ? "" : current)')
    expect(source).toContain('setNewWorldName((current) => current.trim() === variables.name ? "" : current)')
    expect(source).toContain('setNewPlotTitle((current) => current.trim() === title ? "" : current)')
    expect(source).toContain('message.error("Failed to create character")')
    expect(source).toContain('message.error("Failed to create world info")')
    expect(source).toContain('message.error("Failed to create plot line")')
  })

  it("keeps Story Pulse loading and analyze state scoped to one chapter at a time", () => {
    const source = readWritingSource(path.join("modals", "StoryPulseModal.tsx"))

    expect(source).toContain("useWritingPlaygroundStore((state) => state.activeProjectId)")
    expect(source).toContain("isStructureLoading || isStructureFetching")
    expect(source).toContain("if (analyzing) return")
    expect(source).toContain("disabled={Boolean(analyzing) && analyzing !== ch.id}")
  })

  it("uses narrow store selectors in the AI agent writing panel", () => {
    const source = readWritingSource("AIAgentTab.tsx")

    expect(source).toContain("useWritingPlaygroundStore((state) => state.activeProjectId)")
    expect(source).toContain("useWritingPlaygroundStore((state) => state.activeNodeId)")
    expect(source).not.toContain("ManuscriptCharacterListResponse")
    expect(source).not.toContain("ManuscriptWorldInfoListResponse")
  })
})
