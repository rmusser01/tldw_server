import { readFileSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"

const testDir = path.dirname(fileURLToPath(import.meta.url))
const sourcePath = path.resolve(testDir, "../useMessage.tsx")

describe("useMessage assistant overlay guard", () => {
  it("does not clear server chat state when assistant selection changes", () => {
    const source = readFileSync(sourcePath, "utf8")

    expect(source).not.toContain(
      "// Reset server chat when assistant identity changes"
    )
    expect(source).not.toMatch(
      /React\.useEffect\(\(\) => \{\s*setServerChatId\(null\);\s*\}, \[selectedAssistant\?\.id, selectedAssistant\?\.kind\]\);/m
    )
  })

  it("routes plain draft selections through tracked send-mode fallback", () => {
    const source = readFileSync(sourcePath, "utf8")

    expect(source).toContain("draftAssistantKind: selectedAssistant?.kind ?? null")
  })
})
