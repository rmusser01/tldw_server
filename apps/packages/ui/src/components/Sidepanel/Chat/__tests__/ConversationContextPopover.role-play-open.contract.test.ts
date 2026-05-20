import { readFileSync } from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

const root = process.cwd()

const resolveSource = (fileName: string) => {
  const candidates = [
    path.resolve(__dirname, `../${fileName}`),
    path.resolve(root, `src/components/Sidepanel/Chat/${fileName}`),
    path.resolve(root, `../packages/ui/src/components/Sidepanel/Chat/${fileName}`),
    path.resolve(
      root,
      `apps/packages/ui/src/components/Sidepanel/Chat/${fileName}`
    )
  ]

  for (const candidate of candidates) {
    try {
      return readFileSync(candidate, "utf8")
    } catch {
      // try the next known monorepo location
    }
  }

  throw new Error(`Unable to locate ${fileName}`)
}

describe("ConversationContextPopover role-play picker contract", () => {
  const popoverSource = resolveSource("ConversationContextPopover.tsx")
  const characterSelectSource = resolveSource("CharacterSelect.tsx")

  it("opens the mounted context popover before requesting the assistant picker", () => {
    expect(popoverSource).toContain("tldw:open-sidepanel-assistant-select")
    expect(popoverSource).toContain("setOpen(true)")
    expect(popoverSource).toContain("assistantSelectOpenRequest")
    expect(popoverSource).toContain("openRequest={assistantSelectOpenRequest ?? undefined}")
  })

  it("lets CharacterSelect handle explicit open requests after it mounts", () => {
    expect(characterSelectSource).toContain("openRequest")
    expect(characterSelectSource).toContain("openAssistantSelect")
    expect(characterSelectSource).toContain("setDropdownOpen(true)")
  })
})
