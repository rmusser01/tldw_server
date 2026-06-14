import { readFileSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"

const testDir = path.dirname(fileURLToPath(import.meta.url))
const frontendRoot = path.resolve(testDir, "..")

function readSource(envName: string, fallbackPath: string) {
  return readFileSync(process.env[envName] || fallbackPath, "utf8")
}

describe("media-multi review contracts", () => {
  it("keeps the UAT driver on explicit auth and current media-multi selectors", () => {
    const source = readSource("MEDIA_MULTI_UAT_DRIVER_SOURCE", path.join(frontendRoot, "scripts/media-multi-uat-driver.mjs"))

    expect(source).toContain("TLDW_API_KEY")
    expect(source).toContain("TLDW_API_KEY is required")
    expect(source).not.toMatch(/TLDW_API_KEY\s*\|\|/)
    expect(source).not.toContain("THIS-IS-A-SECURE-KEY")

    expect(source).toContain('page.getByTestId("media-review-result-row")')
    expect(source).toContain('"media-multi-batch-toolbar"')
    expect(source).toContain('"media-multi-batch-add-tags"')
    expect(source).toContain('"media-multi-batch-export-format"')
    expect(source).toContain('"media-multi-batch-export"')
    expect(source).toContain('"media-multi-batch-reprocess"')
    expect(source).toContain('"media-multi-batch-trash"')
    expect(source).not.toContain("results-select-")
    expect(source).not.toContain("media-bulk-")
  })

  it("builds UAT driver artifact paths through Node path helpers", () => {
    const source = readSource("MEDIA_MULTI_UAT_DRIVER_SOURCE", path.join(frontendRoot, "scripts/media-multi-uat-driver.mjs"))

    expect(source).toContain('import os from "node:os"')
    expect(source).toContain('import path from "node:path"')
    expect(source).toContain("path.join(os.tmpdir()")
    expect(source).toContain("path.join(SHOTS, `${name}.png`)")
    expect(source).toContain('path.join(SHOTS, "observations.json")')
    expect(source).not.toContain('"/tmp/media-multi-uat-shots"')
  })

  it("defers virtualized row measurement outside the React ref commit path", () => {
    const source = readSource("MEDIA_REVIEW_RESULTS_LIST_SOURCE", path.join(frontendRoot, "../packages/ui/src/components/Review/MediaReviewResultsList.tsx"))

    expect(source).toContain("const measure = () =>")
    expect(source).toContain("if (el.isConnected) listVirtualizer.measureElement(el)")
    expect(source).toContain("window.requestAnimationFrame(measure)")
    expect(source).toContain("window.setTimeout(measure, 0)")
    expect(source).not.toContain("if (el) listVirtualizer.measureElement(el)")
  })
})
