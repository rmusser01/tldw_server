// @vitest-environment jsdom

import React from "react"
import { render, screen, within } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { ReportEvidencePanel } from "../ReportEvidencePanel"
import type { WatchlistOutputEvidenceResponse } from "@/types/watchlists"

const mocks = vi.hoisted(() => ({
  getWatchlistOutputEvidenceMock: vi.fn(),
  tMock: (key: string, defaultValue?: unknown, options?: Record<string, unknown>) => {
    if (typeof defaultValue !== "string") return key
    if (!options) return defaultValue
    return defaultValue.replace(/\{\{(\w+)\}\}/g, (_, token) => String(options[token] ?? ""))
  }
}))

const setViewport = (width: number) => {
  Object.defineProperty(window, "innerWidth", {
    configurable: true,
    value: width
  })
  Object.defineProperty(window, "matchMedia", {
    configurable: true,
    value: vi.fn((query: string) => ({
      matches: query.includes("max-width") ? width <= 767 : width >= 768,
      media: query,
      onchange: null,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      addListener: vi.fn(),
      removeListener: vi.fn(),
      dispatchEvent: vi.fn()
    }))
  })
}

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: mocks.tMock
  })
}))

vi.mock("antd", () => {
  const Table = ({ "aria-label": ariaLabel, dataSource = [], columns = [] }: any) => (
    <table role="table" aria-label={ariaLabel || "Evidence table"}>
      <tbody>
        {dataSource.map((record: any, rowIndex: number) => (
          <tr key={record.id ?? rowIndex}>
            {columns.map((column: any, columnIndex: number) => {
              const value = column.dataIndex ? record[column.dataIndex] : undefined
              const content = column.render ? column.render(value, record, rowIndex) : value
              return <td key={String(column.key ?? column.dataIndex ?? columnIndex)}>{content}</td>
            })}
          </tr>
        ))}
      </tbody>
    </table>
  )
  return {
    Alert: ({ title, message, description, action }: any) => (
      <div>
        <div>{title || message}</div>
        <div>{description}</div>
        {action}
      </div>
    ),
    Button: ({ children, onClick, ...rest }: any) => (
      <button type="button" onClick={() => onClick?.()} {...rest}>
        {children}
      </button>
    ),
    Empty: ({ description }: any) => <div>{description}</div>,
    Spin: () => <div>Loading evidence</div>,
    Table,
    Tag: ({ children }: any) => <span>{children}</span>,
    Tooltip: ({ children }: any) => <>{children}</>
  }
})

vi.mock("@/services/watchlists", () => ({
  getWatchlistOutputEvidence: (...args: unknown[]) =>
    mocks.getWatchlistOutputEvidenceMock(...args)
}))

const evidenceResponse: WatchlistOutputEvidenceResponse = {
  output_id: 42,
  immutable_snapshot: true,
  readiness: {
    state: "warning",
    score: 74,
    warnings: []
  },
  snapshot: {
    schema_version: 1,
    snapshot_id: "snapshot-42",
    generated_at: "2026-05-15T12:00:00Z",
    preset: "cti_osint",
    watchlist_id: 3,
    job_id: 5,
    run_id: 7,
    output_id: 42,
    included_items: [
      {
        id: 101,
        title: "Vendor advisory",
        url: "https://vendor.example/cve",
        source_id: 11,
        source_name: "Vendor feed",
        published_at: "2026-05-15T11:00:00Z",
        summary: "Active exploitation observed.",
        tags: ["cve"],
        reviewed: true,
        queued_for_briefing: true,
        alerts: [
          {
            id: 201,
            rule_id: 301,
            rule_name: "Active exploitation",
            severity: "critical",
            status: "unread",
            title: "Active exploitation observed",
            snippet: "Active exploitation observed.",
            matched_text: "active exploitation",
            evidence: { url: "https://vendor.example/cve" },
            created_at: "2026-05-15T11:05:00Z"
          }
        ]
      }
    ],
    excluded_items: [],
    source_summary: {
      unique_source_count: 2,
      missing_source_count: 0
    },
    included_count: 1,
    excluded_count: 0,
    excluded_total_count: 0,
    excluded_items_truncated: false,
    alert_count: 1,
    critical_alert_count: 1,
    readiness: {
      state: "warning",
      score: 74,
      warnings: []
    }
  }
}

describe("ReportEvidencePanel constrained management", () => {
  it("renders included evidence as cards instead of a table at extension width", async () => {
    setViewport(420)
    render(<ReportEvidencePanel outputId={42} evidenceResponse={evidenceResponse} />)

    expect(await screen.findByTestId("report-evidence-included-constrained-list")).toBeInTheDocument()
    expect(screen.queryByRole("table")).not.toBeInTheDocument()
    const card = screen.getByTestId("report-evidence-included-card-101")
    expect(within(card).getByText("Vendor advisory")).toBeInTheDocument()
    expect(within(card).getByText("Active exploitation observed.")).toBeInTheDocument()
    expect(within(card).getByText("Vendor feed")).toBeInTheDocument()
    expect(within(card).getByText("critical")).toBeInTheDocument()
    expect(within(card).getByText("Reviewed")).toBeInTheDocument()
    expect(within(card).getByText("Queued")).toBeInTheDocument()
    expect(within(card).getByRole("link", { name: "Open source" })).toHaveAttribute(
      "href",
      "https://vendor.example/cve"
    )
  })

  it("preserves the evidence table above the constrained breakpoint", async () => {
    setViewport(1024)
    render(<ReportEvidencePanel outputId={42} evidenceResponse={evidenceResponse} />)

    expect(await screen.findByRole("table", { name: "Evidence table" })).toBeInTheDocument()
    expect(screen.queryByTestId("report-evidence-included-constrained-list")).not.toBeInTheDocument()
  })
})
