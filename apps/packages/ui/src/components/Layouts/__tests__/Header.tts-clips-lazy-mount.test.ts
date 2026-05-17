import { readFileSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"

const testDir = path.dirname(fileURLToPath(import.meta.url))
const sourcePath = path.resolve(testDir, "../Header.tsx")

describe("Header TTS clips drawer mounting", () => {
  it("only mounts the TtsClipsDrawer when the drawer is open", () => {
    const source = readFileSync(sourcePath, "utf8")

    expect(source).toContain("{ttsClipsOpen ? (")
    expect(source).toContain("<TtsClipsDrawer")
  })
})
