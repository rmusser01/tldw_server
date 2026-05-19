import { readFileSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"

const testDir = path.dirname(fileURLToPath(import.meta.url))
const sourcePath = path.resolve(testDir, "../useMessage.tsx")

describe("useMessage preset chat overrides", () => {
  it("threads resolved model and OCR overrides into preset chat mode", () => {
    const source = readFileSync(sourcePath, "utf8")

    expect(source).toContain("const resolvedPresetModel =")
    expect(source).toContain("const resolvedPresetUseOCR =")
    expect(source).toContain("useOCR: resolvedPresetUseOCR")
    expect(source).toContain("selectedModel: resolvedSelectedModel")
    expect(source).toContain("useOCR: resolvedUseOCR")
  })
})
