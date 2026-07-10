// @vitest-environment jsdom

import { beforeEach, describe, expect, it, vi } from "vitest"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { JobFormModal } from "../JobFormModal"

const servicesMock = vi.hoisted(() => ({
  createWatchlistJob: vi.fn(),
  updateWatchlistJob: vi.fn(),
  fetchWatchlistSources: vi.fn(),
  fetchWatchlistGroups: vi.fn(),
  fetchJobOutputTemplates: vi.fn(),
  fetchWatchlistOutputPresets: vi.fn(),
  fetchWatchlistTemplates: vi.fn(),
  previewWatchlistJob: vi.fn()
}))

const translationMock = vi.hoisted(() => ({
  t: (
    key: string,
    fallbackOrOptions?: string | { defaultValue?: string },
    maybeOptions?: Record<string, unknown>
  ) => {
    if (typeof fallbackOrOptions === "string") {
      return interpolate(fallbackOrOptions, maybeOptions)
    }
    if (fallbackOrOptions && typeof fallbackOrOptions === "object") {
      const maybeDefault = fallbackOrOptions.defaultValue
      if (typeof maybeDefault === "string") {
        return interpolate(maybeDefault, maybeOptions)
      }
    }
    return key
  }
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
    t: translationMock.t
  })
}))

vi.mock("@/services/watchlists", () => ({
  createWatchlistJob: (...args: unknown[]) => servicesMock.createWatchlistJob(...args),
  updateWatchlistJob: (...args: unknown[]) => servicesMock.updateWatchlistJob(...args),
  fetchWatchlistSources: (...args: unknown[]) => servicesMock.fetchWatchlistSources(...args),
  fetchWatchlistGroups: (...args: unknown[]) => servicesMock.fetchWatchlistGroups(...args),
  fetchJobOutputTemplates: (...args: unknown[]) => servicesMock.fetchJobOutputTemplates(...args),
  fetchWatchlistOutputPresets: (...args: unknown[]) => servicesMock.fetchWatchlistOutputPresets(...args),
  fetchWatchlistTemplates: (...args: unknown[]) => servicesMock.fetchWatchlistTemplates(...args),
  previewWatchlistJob: (...args: unknown[]) => servicesMock.previewWatchlistJob(...args)
}))

vi.mock("../ScopeSelector", () => ({
  ScopeSelector: () => <div>Scope Selector</div>
}))

vi.mock("../FilterBuilder", () => ({
  FilterBuilder: () => <div>Filter Builder</div>
}))

vi.mock("../SchedulePicker", () => ({
  SchedulePicker: () => <div>Schedule Picker</div>
}))

describe("JobFormModal template source options", () => {
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

    servicesMock.fetchJobOutputTemplates.mockResolvedValue({
      items: [
        {
          id: "11",
          name: "briefing_markdown",
          format: "md",
          updated_at: "2026-01-15T00:00:00Z"
        }
      ],
      total: 1
    })
    servicesMock.fetchWatchlistOutputPresets.mockResolvedValue({
      items: [],
      total: 0,
      page: 1,
      size: 100,
      has_more: false
    })
    servicesMock.fetchWatchlistSources.mockResolvedValue({
      items: [
        {
          id: 1,
          name: "Tech Daily",
          url: "https://example.com/rss.xml",
          source_type: "rss",
          active: true,
          tags: ["tech"],
          created_at: "2026-01-15T00:00:00Z",
          updated_at: "2026-01-15T00:00:00Z",
          last_scraped_at: null,
          status: "healthy"
        }
      ],
      total: 1,
      page: 1,
      size: 500,
      has_more: false
    })
    servicesMock.fetchWatchlistGroups.mockResolvedValue({
      items: [],
      total: 0,
      page: 1,
      size: 500,
      has_more: false
    })
    servicesMock.fetchWatchlistTemplates.mockResolvedValue({
      items: [
        {
          name: "briefing_markdown",
          format: "md",
          content: "legacy duplicate"
        },
        {
          name: "legacy_only",
          format: "md",
          content: "legacy unique"
        }
      ]
    })
    servicesMock.previewWatchlistJob.mockResolvedValue({
      items: [],
      total: 0,
      ingestable: 0,
      filtered: 0
    })
  })

  it("prefers outputs templates over legacy duplicates and labels source correctly", async () => {
    render(
      <JobFormModal
        open
        onClose={vi.fn()}
        onSuccess={vi.fn()}
      />
    )

    await waitFor(() => {
      expect(servicesMock.fetchJobOutputTemplates).toHaveBeenCalled()
      expect(servicesMock.fetchWatchlistTemplates).toHaveBeenCalled()
    })

    fireEvent.click(screen.getByTestId("job-form-mode-advanced"))
    fireEvent.click(screen.getByText("Output & Delivery"))

    await screen.findByText("Template name")
    expect(screen.getByTestId("watchlists-help-jinja2")).toBeInTheDocument()

    const selectorPlaceholder = await screen.findByText("Select a template")
    fireEvent.mouseDown(selectorPlaceholder)

    expect(await screen.findByText("briefing_markdown (MD) · Outputs template")).toBeTruthy()
    expect(await screen.findByText("legacy_only · Legacy watchlists template")).toBeTruthy()
    expect(screen.queryByText("briefing_markdown · Legacy watchlists template")).toBeNull()
  })
})
