import { readFileSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"

describe("PipelineWizard coarse-pointer sizing contract", () => {
  it("sizes the actual Ant controls and labels inside the scoped wizard", () => {
    const testDir = path.dirname(fileURLToPath(import.meta.url))
    const css = readFileSync(
      path.resolve(testDir, "../../../../../assets/tailwind-shared.css"),
      "utf8"
    )

    expect(css).toMatch(/@media\s*\(pointer:\s*coarse\)/)
    for (const selector of [
      ".watchlists-pipeline-wizard .ant-input",
      ".watchlists-pipeline-wizard .ant-input-number-input",
      ".watchlists-pipeline-wizard .ant-select-selector",
      ".watchlists-pipeline-wizard .ant-radio-wrapper",
      ".watchlists-pipeline-wizard .ant-checkbox-wrapper"
    ]) {
      expect(css).toContain(selector)
    }
    expect(css).toMatch(/\.watchlists-pipeline-wizard[\s\S]*min-height:\s*44px/)
  })
})
