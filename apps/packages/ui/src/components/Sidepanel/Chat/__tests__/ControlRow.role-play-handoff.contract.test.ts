import { existsSync, readFileSync } from "node:fs"
import { describe, expect, it } from "vitest"

const controlRowPathCandidates = [
  "src/components/Sidepanel/Chat/ControlRow.tsx",
  "../packages/ui/src/components/Sidepanel/Chat/ControlRow.tsx",
  "apps/packages/ui/src/components/Sidepanel/Chat/ControlRow.tsx"
]

const controlRowPath = controlRowPathCandidates.find((candidate) =>
  existsSync(candidate)
)

if (!controlRowPath) {
  throw new Error("Unable to locate Sidepanel control row source")
}

const controlRowSource = readFileSync(controlRowPath, "utf8")

describe("sidepanel role-play full-app handoff contract", () => {
  it("builds the full app link through the Character Chat handoff helper", () => {
    expect(controlRowSource).toContain("buildSidepanelFullAppChatPath")
    expect(controlRowSource).toContain("selectedAssistant")
    expect(controlRowSource).toContain("selectedCharacterId")
  })

  it("keeps active Character Chat state visible and actionable", () => {
    expect(controlRowSource).toContain('data-testid="sidepanel-character-chat-chip"')
    expect(controlRowSource).toContain('data-testid="sidepanel-character-chat-clear"')
    expect(controlRowSource).toContain("sidepanel:controlRow.characterChatChip")
    expect(controlRowSource).toContain("sidepanel:controlRow.openCharacterChatInFullUI")
    expect(controlRowSource).toContain("SELECTED_ASSISTANT_STORAGE_KEY")
    expect(controlRowSource).toContain("SELECTED_CHARACTER_STORAGE_KEY")
    expect(controlRowSource).toContain("runtime?.id")
    expect(controlRowSource).toContain('window.open(fullAppChatPath')
  })
})
