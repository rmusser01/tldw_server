// @vitest-environment jsdom

import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { ReportBuilderDrawer } from "../ReportBuilderDrawer"
import type { ScrapedItem, WatchlistContainer, WatchlistRun } from "@/types/watchlists"

const serviceMocks = vi.hoisted(() => ({
  createWatchlistOutput: vi.fn(),
  fetchScrapedItems: vi.fn(),
  fetchWatchlistRuns: vi.fn(),
  fetchWatchlistTemplates: vi.fn()
}))

const uiMocks = vi.hoisted(() => ({
  messageError: vi.fn(),
  messageSuccess: vi.fn()
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, defaultValue?: unknown, options?: Record<string, unknown>) => {
      if (typeof defaultValue === "object" && defaultValue) {
        const optionValue = defaultValue as Record<string, unknown>
        const template = typeof optionValue.defaultValue === "string" ? optionValue.defaultValue : _key
        return template.replace(/\{\{(\w+)\}\}/g, (_, token) => String(optionValue[token] ?? ""))
      }
      if (typeof defaultValue !== "string") return _key
      if (!options) return defaultValue
      return defaultValue.replace(/\{\{(\w+)\}\}/g, (_, token) => String(options[token] ?? ""))
    }
  })
}))

vi.mock("@/design-system/states", () => ({
  READY_STATE_LABEL: "Registry Ready"
}))

vi.mock("antd", () => {
  const Select = ({ value, onChange, options = [], ...rest }: any) => (
    <select
      data-testid={rest["data-testid"] || "antd-select"}
      data-value={value == null ? "" : String(value)}
      value={value == null ? "" : String(value)}
      onChange={(event) => {
        const next = event.currentTarget.value
        const option = options.find((entry: any) => String(entry.value) === next)
        onChange?.(option && typeof option.value === "number" ? Number(next) : next || null)
      }}
    >
      <option value="" />
      {options.map((option: any) => (
        <option key={String(option.value)} value={String(option.value)}>
          {String(option.label)}
        </option>
      ))}
    </select>
  )

  return {
    Alert: ({ title, message, description }: any) => (
      <div role="status">
        <div>{title || message}</div>
        <div>{description}</div>
      </div>
    ),
    Button: ({ children, onClick, loading, disabled, ...rest }: any) => (
      <button
        type="button"
        disabled={Boolean(loading || disabled)}
        onClick={() => onClick?.()}
        {...rest}
      >
        {children}
      </button>
    ),
    Checkbox: ({ children, checked, onChange, ...rest }: any) => (
      <label>
        <input
          type="checkbox"
          checked={Boolean(checked)}
          onChange={(event) => onChange?.(event)}
          {...rest}
        />
        {children}
      </label>
    ),
    Drawer: ({ open, title, children, extra, onClose }: any) =>
      open ? (
        <div data-testid="report-builder-drawer">
          <div>{title}</div>
          {extra}
          <button type="button" onClick={() => onClose?.()}>
            Close
          </button>
          {children}
        </div>
      ) : null,
    Empty: ({ description }: any) => <div>{description}</div>,
    Input: ({ value, onChange, ...rest }: any) => (
      <input value={value || ""} onChange={(event) => onChange?.(event)} {...rest} />
    ),
    Select,
    Spin: () => <div>Loading</div>,
    Tag: ({ children }: any) => <span>{children}</span>,
    message: {
      error: uiMocks.messageError,
      success: uiMocks.messageSuccess
    }
  }
})

vi.mock("@/services/watchlists", () => ({
  createWatchlistOutput: (...args: unknown[]) => serviceMocks.createWatchlistOutput(...args),
  fetchScrapedItems: (...args: unknown[]) => serviceMocks.fetchScrapedItems(...args),
  fetchWatchlistRuns: (...args: unknown[]) => serviceMocks.fetchWatchlistRuns(...args),
  fetchWatchlistTemplates: (...args: unknown[]) => serviceMocks.fetchWatchlistTemplates(...args)
}))

const ctiWatchlist: WatchlistContainer = {
  id: 42,
  name: "Healthcare ransomware",
  description: "Track hospital incidents",
  objective: "Track ransomware impact",
  domain: "cti_osint",
  status: "active",
  priority: "critical",
  tags: ["ransomware"],
  created_at: "2026-05-15T00:00:00Z",
  updated_at: "2026-05-15T00:00:00Z"
}

const newsWatchlist: WatchlistContainer = {
  ...ctiWatchlist,
  id: 43,
  name: "Election developments",
  domain: "news",
  priority: "high"
}

const run: WatchlistRun = {
  id: 81,
  job_id: 9,
  status: "completed",
  started_at: "2026-05-15T10:00:00Z",
  finished_at: "2026-05-15T10:15:00Z",
  stats: { ingested: 2 }
}

const queuedItem: ScrapedItem = {
  id: 101,
  run_id: 81,
  job_id: 9,
  source_id: 11,
  url: "https://vendor.example/cve",
  title: "Vendor advisory",
  summary: "Active exploitation observed.",
  published_at: "2026-05-15T11:00:00Z",
  tags: ["cve"],
  status: "ingested",
  reviewed: false,
  queued_for_briefing: true,
  created_at: "2026-05-15T11:05:00Z",
  alert_summary: null
}

const excludedItem: ScrapedItem = {
  ...queuedItem,
  id: 102,
  title: "Background update",
  queued_for_briefing: false
}

const setupQueueMocks = (queuedItems: ScrapedItem[] = [queuedItem], allItems: ScrapedItem[] = [queuedItem, excludedItem]) => {
  serviceMocks.fetchWatchlistRuns.mockResolvedValue({ items: [run], total: 1, has_more: false })
  serviceMocks.fetchWatchlistTemplates.mockResolvedValue({ items: [], total: 0, has_more: false })
  serviceMocks.fetchScrapedItems.mockImplementation(async (params: any) => {
    if (params?.queued_for_briefing === true) {
      return { items: queuedItems, total: queuedItems.length, has_more: false }
    }
    return { items: allItems, total: allItems.length, has_more: false }
  })
  serviceMocks.createWatchlistOutput.mockResolvedValue({
    id: 9001,
    run_id: 81,
    job_id: 9,
    type: "briefing_markdown",
    format: "md",
    title: "CTI evidence report",
    metadata: {
      report_readiness: {
        state: "warning",
        score: 70,
        warnings: []
      }
    },
    version: 1,
    expired: false,
    created_at: "2026-05-15T12:00:00Z"
  })
}

describe("ReportBuilderDrawer", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    setupQueueMocks()
    Object.defineProperty(window, "innerWidth", {
      configurable: true,
      value: 1024
    })
  })

  it("defaults report preset from selected Watchlist domain", async () => {
    const { rerender } = render(
      <ReportBuilderDrawer
        open
        selectedWatchlist={ctiWatchlist}
        onClose={vi.fn()}
        onCreated={vi.fn()}
      />
    )

    expect(await screen.findByTestId("report-builder-preset")).toHaveAttribute("data-value", "cti_osint")

    rerender(
      <ReportBuilderDrawer
        open
        selectedWatchlist={newsWatchlist}
        onClose={vi.fn()}
        onCreated={vi.fn()}
      />
    )

    expect(await screen.findByTestId("report-builder-preset")).toHaveAttribute("data-value", "news_briefing")
  })

  it("requires a run before generation", async () => {
    serviceMocks.fetchWatchlistRuns.mockResolvedValue({ items: [], total: 0, has_more: false })

    render(
      <ReportBuilderDrawer
        open
        selectedWatchlist={ctiWatchlist}
        onClose={vi.fn()}
        onCreated={vi.fn()}
      />
    )

    await screen.findByText("Select a run to generate a report.")
    expect(screen.getByRole("button", { name: "Generate defensible report" })).toBeDisabled()

    expect(uiMocks.messageError).not.toHaveBeenCalled()
    expect(serviceMocks.createWatchlistOutput).not.toHaveBeenCalled()
  })

  it("shows queued count, empty guidance, and preflight warnings", async () => {
    setupQueueMocks([], [excludedItem])

    render(
      <ReportBuilderDrawer
        open
        selectedWatchlist={ctiWatchlist}
        onClose={vi.fn()}
        onCreated={vi.fn()}
      />
    )

    expect(await screen.findByText("0 queued updates")).toBeInTheDocument()
    expect(screen.getByText("No queued updates found for this run.")).toBeInTheDocument()
    expect(screen.getByText("No included updates")).toBeInTheDocument()
  })

  it("uses the design-system registry label when report readiness is ready", async () => {
    const readyItems: ScrapedItem[] = [
      { ...queuedItem, reviewed: true, source_id: 11 },
      { ...queuedItem, id: 103, reviewed: true, source_id: 12, title: "Second source update" }
    ]
    setupQueueMocks(readyItems, readyItems)

    render(
      <ReportBuilderDrawer
        open
        selectedWatchlist={newsWatchlist}
        onClose={vi.fn()}
        onCreated={vi.fn()}
      />
    )

    expect(await screen.findByText("2 queued updates")).toBeInTheDocument()
    const readyBadge = screen.getByText("Registry Ready")
    expect(readyBadge).toBeInTheDocument()
    expect(readyBadge.closest("[data-ds-component='Badge']")).toHaveAttribute(
      "data-ds-variant",
      "success"
    )
    expect(screen.queryByText("Ready")).not.toBeInTheDocument()
  })

  it("submits Stage 5 report options with queued item ids and warning acknowledgement", async () => {
    const onCreated = vi.fn()

    render(
      <ReportBuilderDrawer
        open
        selectedWatchlist={ctiWatchlist}
        onClose={vi.fn()}
        onCreated={onCreated}
      />
    )

    expect(await screen.findByText("1 queued update")).toBeInTheDocument()
    expect(screen.getByText("1 source")).toBeInTheDocument()
    expect(screen.getByText("1 update not queued")).toBeInTheDocument()
    expect(screen.getByText("No alert evidence")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Proceed with warnings" })).toBeInTheDocument()

    fireEvent.change(screen.getByTestId("report-builder-title"), {
      target: { value: "CTI evidence report" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Proceed with warnings" }))

    await waitFor(() => {
      expect(serviceMocks.createWatchlistOutput).toHaveBeenCalledTimes(1)
    })

    expect(serviceMocks.createWatchlistOutput).toHaveBeenCalledWith({
      run_id: 81,
      item_ids: [101],
      title: "CTI evidence report",
      format: "md",
      report_preset: "cti_osint",
      include_evidence_table: true,
      include_excluded_items: true,
      allow_weak_evidence: true,
      require_reviewed_items: false
    })
    expect(onCreated).toHaveBeenCalled()
  })

  it("stacks management controls at extension-sized widths", async () => {
    Object.defineProperty(window, "innerWidth", {
      configurable: true,
      value: 420
    })

    render(
      <ReportBuilderDrawer
        open
        selectedWatchlist={ctiWatchlist}
        onClose={vi.fn()}
        onCreated={vi.fn()}
      />
    )

    expect(await screen.findByTestId("report-builder-layout")).toHaveAttribute("data-layout", "stacked")
  })
})
