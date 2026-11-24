import React from "react"
import { fireEvent, screen, waitFor } from "@testing-library/react"
import { render } from "../../shared/__tests__/render-with-watchlists-query"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { OutputsTab } from "../OutputsTab"

const mocks = vi.hoisted(() => ({
  createWatchlistOutputMock: vi.fn(),
  fetchWatchlistJobsMock: vi.fn(),
  fetchWatchlistOutputsMock: vi.fn(),
  fetchWatchlistTemplatesMock: vi.fn(),
  downloadWatchlistOutputMock: vi.fn(),
  downloadWatchlistOutputBinaryMock: vi.fn(),
  storeStateRef: { current: {} as Record<string, any> }
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
  const Select = ({ value, onChange, options = [], allowClear, ...rest }: any) => (
    <select
      data-testid={rest["data-testid"] || "antd-select"}
      value={value == null ? "" : String(value)}
      onChange={(event) => {
        const next = event.currentTarget.value
        onChange?.(next === "" ? null : next)
      }}
    >
      {allowClear ? <option value="" /> : null}
      {options.map((option: any) => (
        <option key={String(option.value)} value={String(option.value)} disabled={Boolean(option.disabled)}>
          {String(option.label)}
        </option>
      ))}
    </select>
  )

  const Button = ({ children, onClick, loading, ...rest }: any) => (
    <button type="button" disabled={Boolean(loading)} onClick={() => onClick?.()} {...rest}>
      {children}
    </button>
  )

  const Table = ({ "aria-label": ariaLabel }: any) => (
    <div data-testid="outputs-table" aria-label={ariaLabel} />
  )

  return {
    Button,
    Input: ({ value, onChange, ...rest }: any) => (
      <input
        value={value || ""}
        onChange={(event) => onChange?.(event)}
        {...rest}
      />
    ),
    InputNumber: ({ value, onChange, ...rest }: any) => (
      <input
        type="number"
        value={value ?? ""}
        onChange={(event) => onChange?.(Number(event.currentTarget.value))}
        {...rest}
      />
    ),
    Modal: ({ open, children }: any) => (open ? <div>{children}</div> : null),
    Select,
    Space: ({ children }: any) => <>{children}</>,
    Table,
    Tag: ({ children }: any) => <span>{children}</span>,
    Tooltip: ({ children }: any) => <>{children}</>,
    message: {
      success: vi.fn(),
      error: vi.fn()
    }
  }
})

vi.mock("@/utils/dateFormatters", () => ({
  formatRelativeTime: () => "just now"
}))

vi.mock("@/services/watchlists", () => ({
  recordWatchlistsOnboardingTelemetry: async () => undefined,
  createWatchlistOutput: (...args: any[]) => mocks.createWatchlistOutputMock(...args),
  fetchWatchlistJobs: (...args: any[]) => mocks.fetchWatchlistJobsMock(...args),
  fetchWatchlists: async () => ({ items: [], total: 0 }),
  fetchWatchlistOutputs: (...args: any[]) => mocks.fetchWatchlistOutputsMock(...args),
  fetchWatchlistTemplates: (...args: any[]) => mocks.fetchWatchlistTemplatesMock(...args),
  downloadWatchlistOutput: (...args: any[]) => mocks.downloadWatchlistOutputMock(...args),
  downloadWatchlistOutputBinary: (...args: any[]) => mocks.downloadWatchlistOutputBinaryMock(...args)
}))

vi.mock("@/store/watchlists", () => ({
  useWatchlistsStore: (selector: (state: any) => unknown) => selector(mocks.storeStateRef.current)
}))

vi.mock("../OutputPreviewDrawer", () => ({
  OutputPreviewDrawer: () => null
}))

const buildOutput = (deliveryStatus: string) => ({
  id: 18,
  job_id: 4,
  run_id: 91,
  title: "Daily Brief",
  format: "md",
  type: "briefing",
  content: "hello",
  metadata: {
    deliveries: {
      email: {
        channel: "email",
        status: deliveryStatus
      }
    }
  },
  created_at: "2026-02-23T08:00:00Z",
  expires_at: null,
  expired: false
})

const baseState = (overrides: Record<string, unknown> = {}) => ({
  outputsPage: 1,
  outputsPageSize: 20,
  outputsJobFilter: null,
  outputsRunFilter: null,
  outputPreviewOpen: false,
  selectedOutputId: null,
  setOutputsPage: vi.fn(),
  setOutputsPageSize: vi.fn(),
  setOutputsJobFilter: vi.fn(),
  setOutputsRunFilter: vi.fn(),
  openOutputPreview: vi.fn(),
  closeOutputPreview: vi.fn(),
  ...overrides
})

describe("OutputsTab accessibility live-region behavior", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.storeStateRef.current = baseState()
    mocks.fetchWatchlistOutputsMock.mockResolvedValue({ items: [buildOutput("queued")], total: 1, has_more: false })
    mocks.fetchWatchlistJobsMock.mockResolvedValue({ items: [], total: 0, has_more: false })
    mocks.fetchWatchlistTemplatesMock.mockResolvedValue({ items: [] })
    mocks.createWatchlistOutputMock.mockResolvedValue({})
    mocks.downloadWatchlistOutputMock.mockResolvedValue("")
    mocks.downloadWatchlistOutputBinaryMock.mockResolvedValue(new ArrayBuffer(0))
  })

  it("announces delivery status changes in the SR live region", async () => {
    render(<OutputsTab />)
    await waitFor(() => expect(screen.getByRole("button", { name: "Refresh" })).not.toBeDisabled())
    expect(screen.getByTestId("watchlists-outputs-live-region")).toHaveTextContent("")
    expect(screen.getByTestId("watchlists-outputs-live-region")).toHaveAttribute("role", "status")
    expect(screen.getByTestId("watchlists-outputs-live-region")).toHaveAttribute("aria-live", "polite")
    expect(screen.getByTestId("watchlists-outputs-live-region")).toHaveAttribute("aria-atomic", "true")

    mocks.fetchWatchlistOutputsMock.mockResolvedValueOnce({ items: [buildOutput("sent")], total: 1, has_more: false })
    fireEvent.click(screen.getByRole("button", { name: "Refresh" }))

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-outputs-live-region")).toHaveTextContent(
        "Delivery status updated for Daily Brief."
      )
    })
  })

  it("provides an explicit table label for screen readers", () => {
    render(<OutputsTab />)
    expect(screen.getByTestId("outputs-table")).toHaveAttribute("aria-label", "Reports table")
  })
})
