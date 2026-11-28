import { beforeEach, describe, expect, it, vi } from "vitest"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { useCollectionsStore } from "@/store/collections"
import type { ReadingImportJobStatus } from "@/types/collections"
import { ImportExportPanel } from "../ImportExportPanel"

const apiMock = vi.hoisted(() => ({
  getReadingList: vi.fn(),
  getReadingItem: vi.fn(),
  getHighlights: vi.fn(),
  importReadingList: vi.fn(),
  getReadingImportJob: vi.fn()
}))

const interpolate = (template: string, values?: Record<string, unknown>) => {
  if (!values) return template
  return template.replace(/\{\{(\w+)\}\}/g, (_match, token) => {
    const value = values[token]
    return value == null ? "" : String(value)
  })
}

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?: string | { defaultValue?: string } | Record<string, unknown>,
      maybeOptions?: Record<string, unknown>
    ) => {
      if (typeof fallbackOrOptions === "string") {
        return interpolate(fallbackOrOptions, maybeOptions)
      }
      if (fallbackOrOptions && typeof fallbackOrOptions === "object") {
        const maybeDefault = (fallbackOrOptions as { defaultValue?: string }).defaultValue
        if (typeof maybeDefault === "string") {
          return interpolate(maybeDefault, maybeOptions)
        }
      }
      return key
    }
  })
}))

vi.mock("@/hooks/useTldwApiClient", () => ({
  useTldwApiClient: () => apiMock
}))

describe("ImportExportPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks()

    if (!window.matchMedia) {
      Object.defineProperty(window, "matchMedia", {
        writable: true,
        value: vi.fn().mockImplementation((query: string) => ({
          matches: false,
          media: query,
          onchange: null,
          addListener: vi.fn(),
          removeListener: vi.fn(),
          addEventListener: vi.fn(),
          removeEventListener: vi.fn(),
          dispatchEvent: vi.fn()
        }))
      })
    }

    useCollectionsStore.getState().resetStore()

    apiMock.getReadingList.mockResolvedValue({
      items: [
        {
          id: "item-1",
          title: "Article A",
          favorite: false,
          tags: []
        }
      ],
      total: 1,
      page: 1,
      size: 200
    })
  })

  it("shows selection filename hint when export items are selected", async () => {
    render(<ImportExportPanel />)

    const option = await screen.findByRole("option", { name: /Article A/i })
    fireEvent.click(option)

    await waitFor(() => {
      const hint = screen.getByTestId("export-filename-hint")
      expect(hint.textContent).toContain("reading_export_selection.jsonl")
      expect(hint.textContent).toContain("Scope: selected items")
    })
  })

  it("shows filtered jsonl filename hint when date filters are active", async () => {
    useCollectionsStore.getState().setFilterDateRange("2026-01-01", "2026-01-31")

    render(<ImportExportPanel />)

    await waitFor(() => {
      const hint = screen.getByTestId("export-filename-hint")
      expect(hint.textContent).toContain("reading_export_filtered.jsonl")
      expect(hint.textContent).toContain("Scope: filtered list")
    })
  })

  it("shows timestamped zip filename hint when zip format is selected", async () => {
    useCollectionsStore.getState().setExportFormat("zip")

    render(<ImportExportPanel />)

    await waitFor(() => {
      const hint = screen.getByTestId("export-filename-hint")
      expect(hint.textContent).toContain("reading_export_<timestamp>.zip")
      expect(hint.textContent).toContain("Scope: filtered list")
    })
  })

  it("announces the ZIP selection warning and keeps selected items when switching to JSONL", async () => {
    useCollectionsStore.getState().setExportFormat("zip")
    render(<ImportExportPanel />)

    const option = await screen.findByRole("option", { name: /Article A/i })
    fireEvent.click(option)

    const warning = screen.getByRole("alert")
    expect(warning.textContent).toBe("ZIP export doesn't support item selection")
    expect(within(warning).queryByRole("button")).toBeNull()

    fireEvent.click(screen.getByRole("button", { name: "Switch to JSONL" }))
    expect(screen.queryByText("ZIP export doesn't support item selection")).toBeNull()
    expect(option.getAttribute("aria-selected")).toBe("true")
    expect(screen.getByTestId("export-filename-hint").textContent).toContain(
      "reading_export_selection.jsonl"
    )
  })

  it("warns about a date-filtered ZIP export with no selected items", async () => {
    const store = useCollectionsStore.getState()
    store.setExportFormat("zip")
    store.setFilterDateRange("2026-01-01", "2026-01-31")
    render(<ImportExportPanel />)

    expect((await screen.findByRole("alert")).textContent).toBe(
      "Date range filter export is available in JSONL mode only."
    )
  })

  it("announces a successful import and can start another import", () => {
    const store = useCollectionsStore.getState()
    store.setImportWizardStep("result")
    store.setImportResult({ imported: 3, updated: 1, skipped: 2, errors: [] })
    render(<ImportExportPanel />)

    expect(screen.getByRole("heading", { name: "Import Complete" })).toBeTruthy()
    expect(screen.getByRole("status", { name: "Import" }).textContent).toContain(
      "Import Complete. Imported: 3, Updated: 1, Skipped: 2"
    )
    expect(screen.queryByText("Top import errors")).toBeNull()

    fireEvent.click(screen.getByRole("button", { name: "Import More" }))
    expect(screen.getByText("Select import source:")).toBeTruthy()
  })

  it.each<{
    outcome: ReadingImportJobStatus
    title: string
    announcement: string
    action: string
  }>([
    {
      outcome: {
        job_id: 42,
        status: "completed",
        result: {
          source: "pocket",
          imported: 3,
          updated: 1,
          skipped: 0,
          errors: ["Could not save a tag"]
        }
      },
      title: "Import Complete",
      announcement: "Import Complete. Imported: 3, Updated: 1, Skipped: 0. Import errors: 1",
      action: "Import More"
    },
    {
      outcome: { job_id: 42, status: "failed", error_message: "The import could not read the file." },
      title: "Import Failed",
      announcement: "Import Failed. The import could not read the file.",
      action: "Try Again"
    }
  ])("announces progress and $title in the same region", async ({ outcome, title, announcement, action }) => {
    const store = useCollectionsStore.getState()
    store.setImportSource("auto")
    store.setImportWizardStep("upload")
    apiMock.importReadingList.mockResolvedValueOnce({ job_id: 42, status: "processing" })
    let finishJob!: (job: ReadingImportJobStatus) => void
    apiMock.getReadingImportJob.mockReturnValueOnce(new Promise<ReadingImportJobStatus>((resolve) => {
      finishJob = resolve
    }))
    const { container } = render(<ImportExportPanel />)
    const status = screen.getByRole("status", { name: "Import" })
    const fileInput = container.querySelector('input[type="file"]')!

    fireEvent.change(fileInput, {
      target: { files: [new File(["{}"], "pocket.json", { type: "application/json" })] }
    })
    await waitFor(() => expect(status.textContent).toContain("Current status: Processing"))

    finishJob(outcome)

    expect(await screen.findByRole("heading", { name: title })).toBeTruthy()
    expect(status.textContent).toBe(announcement)
    expect(within(status).queryByRole("button")).toBeNull()

    fireEvent.click(screen.getByRole("button", { name: action }))
    expect(screen.getByText("Select import source:")).toBeTruthy()
    expect(status.textContent).toBe("")
  })

  it("renders top import errors and truncates overflow with a summary line", async () => {
    useCollectionsStore.getState().setImportWizardStep("result")
    useCollectionsStore.getState().setImportResult({
      imported: 3,
      updated: 1,
      skipped: 2,
      errors: [
        "bad row 1",
        "bad row 2",
        "bad row 3",
        "bad row 4",
        "bad row 5",
        "bad row 6",
        "bad row 7"
      ]
    })

    render(<ImportExportPanel />)

    expect(screen.getByRole("heading", { name: "Import Complete" })).toBeTruthy()
    expect(screen.getByRole("status", { name: "Import" }).textContent).toContain(
      "Imported: 3, Updated: 1, Skipped: 2"
    )
    expect(await screen.findByText("Top import errors")).toBeTruthy()
    expect(screen.getByText("bad row 1")).toBeTruthy()
    expect(screen.getByText("bad row 5")).toBeTruthy()
    expect(screen.queryByText("bad row 6")).toBeNull()
    expect(screen.getByText("+2 more errors")).toBeTruthy()
  })
})
