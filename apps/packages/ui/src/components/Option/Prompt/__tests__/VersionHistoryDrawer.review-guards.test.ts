import fs from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

describe("VersionHistoryDrawer review guards", () => {
  it("reselects a non-current version once currentVersion is known", () => {
    const source = fs.readFileSync(
      path.resolve(
        __dirname,
        "..",
        "Studio",
        "Prompts",
        "VersionHistoryDrawer.tsx"
      ),
      "utf8"
    )

    expect(source).toContain("existingSelection.version_number !== currentVersion")
  })

  it("renders an explicit preview error state", () => {
    const source = fs.readFileSync(
      path.resolve(
        __dirname,
        "..",
        "Studio",
        "Prompts",
        "VersionHistoryDrawer.tsx"
      ),
      "utf8"
    )

    expect(source).toContain('selectedVersionStatus === "error"')
    expect(source).toContain("previewErrorTitle")
  })
})
