// @vitest-environment jsdom

import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { ReportEvidencePanel } from "../ReportEvidencePanel"
import type { WatchlistOutputEvidenceResponse } from "@/types/watchlists"

const serviceMocks = vi.hoisted(() => ({
  getWatchlistOutputEvidence: vi.fn()
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, defaultValue?: unknown, options?: Record<string, unknown>) => {
      if (typeof defaultValue !== "string") return _key
      if (!options) return defaultValue
      return defaultValue.replace(/\{\{(\w+)\}\}/g, (_, token) => String(options[token] ?? ""))
    }
  })
}))

vi.mock("antd", () => {
  const Table = ({ dataSource = [], columns = [] }: any) => (
    <div data-testid="report-evidence-table">
      {dataSource.map((record: any, rowIndex: number) => (
        <div key={record.id ?? rowIndex} data-testid={`report-evidence-row-${record.id}`}>
          {columns.map((column: any, columnIndex: number) => {
            const value = column.dataIndex ? record[column.dataIndex] : undefined
            const content = column.render ? column.render(value, record, rowIndex) : value
            return <div key={String(column.key ?? column.dataIndex ?? columnIndex)}>{content}</div>
          })}
        </div>
      ))}
    </div>
  )

  return {
    Alert: ({ title, message, description, action, type }: any) => (
      <div role={type === "error" ? "alert" : "status"}>
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
    serviceMocks.getWatchlistOutputEvidence(...args)
}))

const evidenceResponse: WatchlistOutputEvidenceResponse = {
  output_id: 42,
  immutable_snapshot: true,
  readiness: {
    state: "warning",
    score: 74,
    warnings: [
      {
        code: "single_source",
        severity: "warning",
        message: "Only one source is represented.",
        affected_item_ids: [101]
      }
    ]
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
    excluded_items: [
      {
        id: 103,
        title: "Background update",
        url: "https://vendor.example/background",
        reason: "not_queued_for_report"
      }
    ],
    source_summary: {
      unique_source_count: 2,
      missing_source_count: 0
    },
    included_count: 1,
    excluded_count: 1,
    excluded_total_count: 1,
    excluded_items_truncated: false,
    alert_count: 1,
    critical_alert_count: 1,
    readiness: {
      state: "warning",
      score: 74,
      warnings: [
        {
          code: "single_source",
          severity: "warning",
          message: "Only one source is represented.",
          affected_item_ids: [101]
        }
      ]
    }
  }
}

describe("ReportEvidencePanel", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("renders immutable evidence rows, source diversity, alerts, and excluded trail", async () => {
    serviceMocks.getWatchlistOutputEvidence.mockResolvedValue(evidenceResponse)

    render(<ReportEvidencePanel outputId={42} />)

    expect(await screen.findByText("Evidence snapshot")).toBeInTheDocument()
    expect(screen.getByText("Immutable snapshot captured at 2026-05-15T12:00:00Z")).toBeInTheDocument()
    expect(screen.getByText("Unique sources: 2")).toBeInTheDocument()
    expect(screen.getByText("Vendor advisory")).toBeInTheDocument()
    expect(screen.getByText("Vendor feed")).toBeInTheDocument()
    expect(screen.getByText("critical")).toBeInTheDocument()
    expect(screen.getByText("Reviewed")).toBeInTheDocument()
    expect(screen.getByText("Queued")).toBeInTheDocument()
    expect(screen.getByText("Background update")).toBeInTheDocument()
    expect(screen.getByText("Not queued for report")).toBeInTheDocument()
  })

  it("shows legacy live-only state without an immutable snapshot", async () => {
    serviceMocks.getWatchlistOutputEvidence.mockResolvedValue({
      output_id: 77,
      immutable_snapshot: false,
      snapshot: null,
      readiness: {
        state: "legacy_live_only",
        score: 0,
        warnings: [
          {
            code: "legacy_live_only",
            severity: "info",
            message: "This older report was created before evidence snapshots were available.",
            affected_item_ids: []
          }
        ]
      }
    })

    render(<ReportEvidencePanel outputId={77} />)

    expect(await screen.findByText("Live provenance only")).toBeInTheDocument()
    expect(screen.getByText("Live provenance only").closest('[data-ds-component="Alert"]')).toBeInTheDocument()
    expect(
      screen.getByText("This older report was created before evidence snapshots were available.")
    ).toBeInTheDocument()
  })

  it("shows a retryable missing snapshot error state", async () => {
    vi.spyOn(console, "error").mockImplementation(() => undefined)
    serviceMocks.getWatchlistOutputEvidence.mockRejectedValueOnce(new Error("report_snapshot_missing"))
    serviceMocks.getWatchlistOutputEvidence.mockResolvedValueOnce(evidenceResponse)

    render(<ReportEvidencePanel outputId={42} />)

    expect(await screen.findByText("Evidence snapshot unavailable")).toBeInTheDocument()
    expect(screen.getByText("Evidence snapshot unavailable").closest('[data-ds-component="Alert"]')).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Retry" }))

    await waitFor(() => {
      expect(serviceMocks.getWatchlistOutputEvidence).toHaveBeenCalledTimes(2)
    })
    expect(await screen.findByText("Vendor advisory")).toBeInTheDocument()
  })
})
