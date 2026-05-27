import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
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
  ExportPanel: ({ initialDeckId }: { initialDeckId?: number | null }) => (
    <div data-testid="mock-export-panel" data-initial-deck-id={initialDeckId ?? ""} />
  )
}))

vi.mock("../ImportExport/GeneratePanel", () => ({
  GeneratePanel: ({
    initialIntent
  }: {
    initialIntent?: { text?: string | null } | null
  }) => (
    <div data-testid="mock-generate-panel" data-initial-text={initialIntent?.text ?? ""} />
  )
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

  it("starts with a task-first create workspace instead of exposing every transfer panel", () => {
    mocks.useImportLimitsQuery.mockReturnValue({
      data: null
    })

    render(<ImportExportTab />)

    const taskSwitcher = screen.getByTestId("flashcards-transfer-task-switcher")
    expect(taskSwitcher).toHaveTextContent("Create cards")
    expect(taskSwitcher).toHaveTextContent("Import file")
    expect(taskSwitcher).toHaveTextContent("Export backup")
    expect(screen.getByTestId("flashcards-create-task-panel")).toBeVisible()
    expect(screen.getByTestId("flashcards-create-task-panel")).not.toHaveClass("hidden")
    expect(screen.getByTestId("flashcards-import-task-panel")).toHaveClass("hidden")
    expect(screen.getByTestId("flashcards-export-task-panel")).toHaveClass("hidden")
    expect(screen.getByTestId("mock-study-pack-panel")).toBeVisible()
    expect(screen.getByTestId("mock-generate-panel")).toBeVisible()
    expect(screen.getByTestId("mock-image-occlusion-panel")).toBeVisible()
  })

  it("switches between task-specific create, import, and export workspaces", async () => {
    const user = userEvent.setup()
    mocks.useImportLimitsQuery.mockReturnValue({
      data: null
    })

    render(<ImportExportTab />)

    await user.click(screen.getByText("Import file"))
    expect(screen.getByTestId("flashcards-import-task-panel")).toBeVisible()
    expect(screen.getByTestId("flashcards-import-task-panel")).not.toHaveClass("hidden")
    expect(screen.getByTestId("flashcards-create-task-panel")).toHaveClass("hidden")
    expect(screen.getByTestId("flashcards-export-task-panel")).toHaveClass("hidden")

    await user.click(screen.getByText("Export backup"))
    expect(screen.getByTestId("flashcards-export-task-panel")).toBeVisible()
    expect(screen.getByTestId("flashcards-export-task-panel")).not.toHaveClass("hidden")
    expect(screen.getByTestId("flashcards-create-task-panel")).toHaveClass("hidden")
    expect(screen.getByTestId("flashcards-import-task-panel")).toHaveClass("hidden")
  })

  it("opens the export task when a deck export handoff is present", () => {
    mocks.useImportLimitsQuery.mockReturnValue({
      data: null
    })

    render(<ImportExportTab initialExportDeckId={42} initialExportDeckHandoffKey="deck-42" />)

    expect(screen.getByTestId("flashcards-export-task-panel")).toBeVisible()
    expect(screen.getByTestId("flashcards-export-task-panel")).not.toHaveClass("hidden")
    expect(screen.getByTestId("flashcards-create-task-panel")).toHaveClass("hidden")
    expect(screen.getByTestId("mock-export-panel")).toHaveAttribute(
      "data-initial-deck-id",
      "42"
    )
  })

  it("keeps generated-card handoffs in the create task", () => {
    mocks.useImportLimitsQuery.mockReturnValue({
      data: null
    })

    render(
      <ImportExportTab
        generateIntent={{
          text: "Selected page notes",
          sourceType: "manual",
          sourceTitle: "Captured page"
        }}
      />
    )

    expect(screen.getByTestId("flashcards-create-task-panel")).toBeVisible()
    expect(screen.getByTestId("flashcards-create-task-panel")).not.toHaveClass("hidden")
    expect(screen.getByTestId("flashcards-export-task-panel")).toHaveClass("hidden")
    expect(screen.getByTestId("mock-generate-panel")).toHaveAttribute(
      "data-initial-text",
      "Selected page notes"
    )
  })
})
