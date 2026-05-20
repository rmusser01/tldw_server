import { existsSync, readFileSync } from "node:fs"
import { describe, expect, it } from "vitest"

const routePathCandidates = [
  "src/routes/sidepanel-chat.tsx",
  "../packages/ui/src/routes/sidepanel-chat.tsx",
  "apps/packages/ui/src/routes/sidepanel-chat.tsx"
]

const routePath = routePathCandidates.find((candidate) => existsSync(candidate))

if (!routePath) {
  throw new Error("Unable to locate sidepanel chat route")
}

const routeSource = readFileSync(routePath, "utf8")

describe("sidepanel Character Chat command palette handoff", () => {
  it("adds an active role-play command that opens the full app with preserved intent", () => {
    expect(routeSource).toContain("buildSidepanelFullAppChatPath")
    expect(routeSource).toContain("openCharacterChatInFullAppCommand")
    expect(routeSource).toContain("common:commandPalette.openCharacterChatFullApp")
    expect(routeSource).toContain("additionalCommands")
    expect(routeSource).toContain("browser?.runtime?.id")
    expect(routeSource).toContain("window.open(normalizedRoute")
  })
})
