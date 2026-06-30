import { readFileSync } from "node:fs"
import { fileURLToPath } from "node:url"
import path from "node:path"
import { describe, expect, it } from "vitest"

const testDir = path.dirname(fileURLToPath(import.meta.url))
const sourcePath = path.resolve(testDir, "../sidepanel-chat.tsx")

describe("sidepanel-chat note capture payload", () => {
  it("does not send ignored arbitrary metadata for captured note provenance", () => {
    const source = readFileSync(sourcePath, "utf8")

    expect(source).not.toContain("source_url: noteSourceUrl")
    expect(source).not.toContain('origin: "context-menu"')
  })
})
