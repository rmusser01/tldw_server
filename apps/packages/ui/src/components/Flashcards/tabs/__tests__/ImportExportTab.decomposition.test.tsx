import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { ImportExportTab } from "../ImportExportTab"
import { ExportPanel } from "../ImportExport/ExportPanel"
import { GeneratePanel } from "../ImportExport/GeneratePanel"
import { ImportPanel } from "../ImportExport/ImportPanel"
import { StudyPackPanel } from "../ImportExport/StudyPackPanel"

const mocks = vi.hoisted(() => ({
  useImportLimitsQuery: vi.fn()
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?:
        | string
        | {
            defaultValue?: string
          }
    ) => {
      if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
      if (defaultValueOrOptions?.defaultValue) {
        return defaultValueOrOptions.defaultValue.replace(
          /\{\{(\w+)\}\}/g,
          (_match, token: string) =>
            String((defaultValueOrOptions as Record<string, unknown>)[token] ?? `{{${token}}}`)
        )
      }
      return key
    }
  })
}))

vi.mock("../../hooks", () => ({
  useImportLimitsQuery: (...args: unknown[]) => mocks.useImportLimitsQuery(...args)
}))

vi.mock("../ImportExport/ImportPanel", () => ({
  ImportPanel: () => <div data-testid="mock-import-panel" />
}))

vi.mock("../ImportExport/ExportPanel", () => ({
  ExportPanel: () => <div data-testid="mock-export-panel" />
}))

vi.mock("../ImportExport/GeneratePanel", () => ({
  GeneratePanel: () => <div data-testid="mock-generate-panel" />
}))

vi.mock("../ImportExport/StudyPackPanel", () => ({
  StudyPackPanel: () => <div data-testid="mock-study-pack-panel" />
}))

vi.mock("../ImageOcclusionTransferPanel", () => ({
  ImageOcclusionTransferPanel: () => <div data-testid="mock-image-occlusion-panel" />
}))

vi.mock("../../components/StudyPackCreateDrawer", () => ({
  StudyPackCreateDrawer: () => null
}))

describe("ImportExportTab decomposition", () => {
  it("exposes focused panel modules for study packs, import, export, and generation", () => {
    expect(StudyPackPanel).toBeTypeOf("function")
    expect(ImportPanel).toBeTypeOf("function")
    expect(ExportPanel).toBeTypeOf("function")
    expect(GeneratePanel).toBeTypeOf("function")
  })

  it("renders concrete import limits instead of unresolved interpolation placeholders", () => {
    mocks.useImportLimitsQuery.mockReturnValue({
      data: {
        max_lines: 2500,
        max_line_length: 32768,
        max_field_length: 1048576
      }
    })

    render(<ImportExportTab />)

    const limits = screen.getByTestId("flashcards-transfer-summary-limits")
    expect(limits).not.toHaveTextContent("{{lines}}")
    expect(limits).not.toHaveTextContent("{{lineBytes}}")
    expect(limits).not.toHaveTextContent("{{fieldBytes}}")
    expect(limits).toHaveTextContent(`${(2500).toLocaleString()} lines`)
    expect(limits).toHaveTextContent(`${(32768).toLocaleString()} bytes per line`)
    expect(limits).toHaveTextContent(`${(1048576).toLocaleString()} bytes per field`)
  })

  it("shows the unavailable fallback for malformed import limits", () => {
    mocks.useImportLimitsQuery.mockReturnValue({
      data: {
        max_cards_per_import: 2500,
        max_content_size_bytes: 1048576
      }
    })

    render(<ImportExportTab />)

    const limits = screen.getByTestId("flashcards-transfer-summary-limits")
    expect(limits).toHaveTextContent("Limits unavailable")
  })
})
