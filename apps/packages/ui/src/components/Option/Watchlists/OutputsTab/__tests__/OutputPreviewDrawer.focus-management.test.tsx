// @vitest-environment jsdom

import React from "react"
import { render, screen, waitFor } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { OutputPreviewDrawer } from "../OutputPreviewDrawer"
import type { WatchlistOutput } from "@/types/watchlists"

const serviceMocks = vi.hoisted(() => ({
  downloadWatchlistOutput: vi.fn(),
  downloadWatchlistOutputBinary: vi.fn(),
  getWatchlistOutputEvidence: vi.fn()
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?: string | { defaultValue?: string }
    ) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      if (
        fallbackOrOptions &&
        typeof fallbackOrOptions === "object" &&
        typeof fallbackOrOptions.defaultValue === "string"
      ) {
        return fallbackOrOptions.defaultValue
      }
      return key
    }
  })
}))

vi.mock("antd", () => {
  const Drawer = ({ open, title, extra, children, onClose, afterOpenChange }: any) => {
    const closeRef = React.useRef<HTMLButtonElement | null>(null)
    React.useEffect(() => {
      afterOpenChange?.(open)
      if (open) {
        closeRef.current?.focus()
      }
    }, [afterOpenChange, open])

    if (!open) return null
    return (
      <div>
        <div>{title}</div>
        {extra}
        <button type="button" ref={closeRef} onClick={() => onClose?.()}>
          Close drawer
        </button>
        {children}
      </div>
    )
  }

  const Button = ({ children, onClick, disabled }: any) => (
    <button type="button" disabled={Boolean(disabled)} onClick={() => onClick?.()}>
      {children}
    </button>
  )

  return {
    Alert: ({ title, message, description }: any) => (
      <div role="status">
        <div>{title || message}</div>
        <div>{description}</div>
      </div>
    ),
    Button,
    Drawer,
    Empty: ({ description }: any) => <div>{description}</div>,
    Segmented: () => null,
    Spin: () => <div>Loading</div>,
    Table: ({ dataSource = [], columns = [] }: any) => (
      <div>
        {dataSource.map((record: any, rowIndex: number) => (
          <div key={record.id ?? rowIndex}>
            {columns.map((column: any, columnIndex: number) => {
              const value = column.dataIndex ? record[column.dataIndex] : undefined
              const content = column.render ? column.render(value, record, rowIndex) : value
              return <div key={String(column.key ?? column.dataIndex ?? columnIndex)}>{content}</div>
            })}
          </div>
        ))}
      </div>
    ),
    Tag: ({ children }: any) => <span>{children}</span>,
    Tooltip: ({ children }: any) => <>{children}</>,
    message: {
      success: vi.fn(),
      error: vi.fn()
    }
  }
})

vi.mock("@/services/watchlists", () => ({
  downloadWatchlistOutput: (...args: unknown[]) =>
    serviceMocks.downloadWatchlistOutput(...args),
  downloadWatchlistOutputBinary: (...args: unknown[]) =>
    serviceMocks.downloadWatchlistOutputBinary(...args),
  getWatchlistOutputEvidence: (...args: unknown[]) =>
    serviceMocks.getWatchlistOutputEvidence(...args)
}))

const buildOutput = (overrides: Partial<WatchlistOutput> = {}): WatchlistOutput => ({
  id: 77,
  run_id: 3,
  job_id: 2,
  type: "briefing",
  format: "md",
  title: "Morning Brief",
  content: null,
  storage_path: "watchlists/morning-brief.md",
  metadata: {},
  media_item_id: null,
  chatbook_path: null,
  version: 1,
  expires_at: null,
  expired: false,
  created_at: "2026-02-23T00:00:00Z",
  ...overrides
})

describe("OutputPreviewDrawer focus management", () => {
  it("restores focus to the launch control after the drawer closes", async () => {
    serviceMocks.downloadWatchlistOutput.mockResolvedValue("# Morning brief")
    serviceMocks.downloadWatchlistOutputBinary.mockResolvedValue(new ArrayBuffer(0))
    serviceMocks.getWatchlistOutputEvidence.mockResolvedValue({
      output_id: 77,
      immutable_snapshot: false,
      snapshot: null,
      readiness: {
        state: "legacy_live_only",
        score: 0,
        warnings: []
      }
    })

    const trigger = document.createElement("button")
    trigger.type = "button"
    trigger.textContent = "Open output preview"
    document.body.appendChild(trigger)
    trigger.focus()

    const { rerender } = render(
      <OutputPreviewDrawer
        open
        output={buildOutput()}
        onClose={vi.fn()}
      />
    )

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Close drawer" })).toHaveFocus()
    })

    rerender(
      <OutputPreviewDrawer
        open={false}
        output={buildOutput()}
        onClose={vi.fn()}
      />
    )

    await waitFor(() => {
      expect(trigger).toHaveFocus()
    })

    trigger.remove()
  })

  it("loads the immutable evidence section from the preview drawer", async () => {
    serviceMocks.downloadWatchlistOutput.mockResolvedValue("# Morning brief")
    serviceMocks.downloadWatchlistOutputBinary.mockResolvedValue(new ArrayBuffer(0))
    serviceMocks.getWatchlistOutputEvidence.mockResolvedValue({
      output_id: 77,
      immutable_snapshot: true,
      readiness: {
        state: "ready",
        score: 98,
        warnings: []
      },
      snapshot: {
        schema_version: 1,
        snapshot_id: "snapshot-77",
        generated_at: "2026-05-15T12:00:00Z",
        preset: "cti_osint",
        watchlist_id: 42,
        job_id: 2,
        run_id: 3,
        output_id: 77,
        included_items: [
          {
            id: 101,
            title: "Vendor advisory",
            url: "https://vendor.example/cve",
            source_id: 11,
            source_name: "Vendor feed",
            published_at: "2026-05-15T11:00:00Z",
            summary: null,
            tags: ["cve"],
            reviewed: true,
            queued_for_briefing: true,
            alerts: []
          }
        ],
        excluded_items: [],
        source_summary: {
          unique_source_count: 1
        },
        included_count: 1,
        excluded_count: 0,
        excluded_items_truncated: false,
        alert_count: 0,
        critical_alert_count: 0,
        readiness: {
          state: "ready",
          score: 98,
          warnings: []
        }
      }
    })

    render(
      <OutputPreviewDrawer
        open
        output={buildOutput({
          metadata: {
            report_snapshot_path: "watchlists/snapshot-77.json"
          }
        })}
        onClose={vi.fn()}
      />
    )

    await screen.findByText("# Morning brief")
    expect(await screen.findByText("Evidence snapshot")).toBeInTheDocument()
    expect(await screen.findByText("Vendor advisory")).toBeInTheDocument()
    expect(serviceMocks.getWatchlistOutputEvidence).toHaveBeenCalledWith(77)
  })
})
