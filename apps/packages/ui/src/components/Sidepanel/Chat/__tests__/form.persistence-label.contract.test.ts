import { readFileSync } from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

// XS-05 (#3104): the side panel "Save chat to history" tooltip claimed
// "Locally + Server" whenever the server was reachable, although plain side
// panel chats are never written to the server. The tooltip must come from the
// shared persistence hook, which derives it from an acknowledged server write
// (see hooks/playground/__tests__/usePersistenceMode.test.tsx).
const formSource = readFileSync(
  path.resolve(__dirname, "../form.tsx"),
  "utf8"
)

describe("sidepanel persistence label contract", () => {
  it("derives the save-to-history tooltip from the shared persistence hook", () => {
    expect(formSource).toMatch(
      /usePersistenceMode\(\{\s*temporaryChat,\s*serverChatId\s*\}\)/
    )
    expect(formSource).toMatch(/<Tooltip title=\{persistenceTooltip\}>/)
  })

  it("does not derive a server persistence claim from connectivity", () => {
    expect(formSource).not.toMatch(/serverChatId\s*\|\|\s*isConnectionReady/)
    expect(formSource).not.toContain(
      '"playground:composer.persistence.serverPill"'
    )
    expect(formSource).not.toContain('"playground:composer.persistence.server"')
  })
})
