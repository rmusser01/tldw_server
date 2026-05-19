import { describe, expect, it } from "vitest"

import { ExportPanel } from "../ImportExport/ExportPanel"
import { GeneratePanel } from "../ImportExport/GeneratePanel"
import { ImportPanel } from "../ImportExport/ImportPanel"
import { StudyPackPanel } from "../ImportExport/StudyPackPanel"

describe("ImportExportTab decomposition", () => {
  it("exposes focused panel modules for study packs, import, export, and generation", () => {
    expect(StudyPackPanel).toBeTypeOf("function")
    expect(ImportPanel).toBeTypeOf("function")
    expect(ExportPanel).toBeTypeOf("function")
    expect(GeneratePanel).toBeTypeOf("function")
  })
})
