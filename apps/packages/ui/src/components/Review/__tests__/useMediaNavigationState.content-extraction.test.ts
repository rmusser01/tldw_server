import { readFileSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"

const testDirectory = path.dirname(fileURLToPath(import.meta.url))
const source = readFileSync(
  path.resolve(testDirectory, "../hooks/useMediaNavigationState.ts"),
  "utf8"
)

describe("media navigation content extraction", () => {
  it("uses the canonical media detail extractor", () => {
    expect(source).toContain(
      "import { extractMediaDetailContent } from '@/utils/media-detail-content'"
    )
    expect(source).toContain("extractMediaDetailContent(detail)")
    expect(source).not.toContain("const contentFromDetail = useCallback")
  })

  it("marks permalink details fetched before publishing the hydrated selection", () => {
    const hydrationStart = source.indexOf("const hydratedSelection: MediaResultItem")
    const hydrationEnd = source.indexOf("} catch (error) {", hydrationStart)
    const hydrationBlock = source.slice(hydrationStart, hydrationEnd)

    expect(hydrationBlock.indexOf("setLastFetchedId(resolvedId)")).toBeGreaterThan(-1)
    expect(hydrationBlock.indexOf("setLastFetchedId(resolvedId)")).toBeLessThan(
      hydrationBlock.indexOf("setSelected(hydratedSelection)")
    )
  })
})
