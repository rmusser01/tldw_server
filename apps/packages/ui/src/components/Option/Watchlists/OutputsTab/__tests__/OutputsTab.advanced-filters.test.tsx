import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { OutputsTab } from "../OutputsTab"

const STORAGE_KEY = "watchlists:outputs:advanced-filters:v1"

const mocks = vi.hoisted(() => ({
  createWatchlistOutputMock: vi.fn(),
  fetchWatchlistJobsMock: vi.fn(),
  fetchWatchlistOutputsMock: vi.fn(),
  fetchWatchlistTemplatesMock: vi.fn(),
  downloadWatchlistOutputMock: vi.fn(),
  trackWatchlistsOnboardingTelemetryMock: vi.fn(),
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

  const Table = ({ dataSource = [] }: any) => (
    <div data-testid="outputs-table-rows">{dataSource.length}</div>
  )
  const Alert = ({ title, message, description, action, ...rest }: any) => (
    <div data-testid={rest["data-testid"] || "outputs-alert"}>
      <div>{title ?? message}</div>
      <div>{description}</div>
      {action}
    </div>
  )
  const Space = ({ children }: any) => <>{children}</>
  const Tag = ({ children }: any) => <span>{children}</span>
  const Tooltip = ({ children }: any) => <>{children}</>
  const Modal = ({ open, children }: any) => (open ? <div>{children}</div> : null)
  const Input = ({ value, onChange, ...rest }: any) => (
    <input
      value={value || ""}
      onChange={(event) => onChange?.(event)}
      {...rest}
    />
  )
  const InputNumber = ({ value, onChange, ...rest }: any) => (
    <input
      type="number"
      value={value ?? ""}
      onChange={(event) => onChange?.(Number(event.currentTarget.value))}
      {...rest}
    />
  )

  return {
    Alert,
    Button,
    Input,
    InputNumber,
    Modal,
    Select,
    Space,
    Table,
    Tag,
    Tooltip,
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
  createWatchlistOutput: (...args: any[]) => mocks.createWatchlistOutputMock(...args),
  fetchWatchlistJobs: (...args: any[]) => mocks.fetchWatchlistJobsMock(...args),
  fetchWatchlistOutputs: (...args: any[]) => mocks.fetchWatchlistOutputsMock(...args),
  fetchWatchlistTemplates: (...args: any[]) => mocks.fetchWatchlistTemplatesMock(...args),
  downloadWatchlistOutput: (...args: any[]) => mocks.downloadWatchlistOutputMock(...args)
}))

vi.mock("@/utils/watchlists-onboarding-telemetry", () => ({
  trackWatchlistsOnboardingTelemetry: (...args: any[]) =>
    mocks.trackWatchlistsOnboardingTelemetryMock(...args)
}))

vi.mock("@/store/watchlists", () => ({
  useWatchlistsStore: (selector: (state: any) => unknown) => selector(mocks.storeStateRef.current)
}))

vi.mock("@/utils/watchlists-onboarding-telemetry", () => ({
  trackWatchlistsOnboardingTelemetry: (...args: any[]) =>
    mocks.trackWatchlistsOnboardingTelemetryMock(...args)
}))

vi.mock("../OutputPreviewDrawer", () => ({
  OutputPreviewDrawer: () => null
}))

const baseState = (overrides: Record<string, unknown> = {}) => ({
  outputs: [],
  outputsLoading: false,
  outputsTotal: 0,
  outputsPage: 1,
  outputsPageSize: 20,
  outputsJobFilter: null,
  outputsRunFilter: null,
  outputPreviewOpen: false,
  selectedOutputId: null,
  setOutputs: vi.fn(),
  setOutputsLoading: vi.fn(),
  setOutputsPage: vi.fn(),
  setOutputsPageSize: vi.fn(),
  setOutputsJobFilter: vi.fn(),
  setOutputsRunFilter: vi.fn(),
  setRunsJobFilter: vi.fn(),
  setRunsStatusFilter: vi.fn(),
  setActiveTab: vi.fn(),
  openOutputPreview: vi.fn(),
  closeOutputPreview: vi.fn(),
  ...overrides
})

const buildOutput = (id: number, deliveryStatus: string) => ({
  id,
  job_id: 8,
  run_id: 40 + id,
  title: `Output ${id}`,
  format: "md",
  content: "body",
  storage_path: null,
  metadata: {
    deliveries: [
      {
        channel: "email",
        status: deliveryStatus
      }
    ]
  },
  expires_at: null,
  expired: false,
  created_at: "2026-02-23T10:00:00Z"
})

describe("OutputsTab advanced filters disclosure", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.removeItem(STORAGE_KEY)
    mocks.storeStateRef.current = baseState()
    mocks.fetchWatchlistOutputsMock.mockResolvedValue({ items: [], total: 0, has_more: false })
    mocks.fetchWatchlistJobsMock.mockResolvedValue({ items: [{ id: 8, name: "Weekly Digest" }], total: 1, has_more: false })
    mocks.fetchWatchlistTemplatesMock.mockResolvedValue({ items: [] })
    mocks.createWatchlistOutputMock.mockResolvedValue({})
    mocks.downloadWatchlistOutputMock.mockResolvedValue("")
  })

  it("starts collapsed, toggles open, and persists disclosure state", async () => {
    render(<OutputsTab />)

    expect(screen.queryByTestId("watchlists-outputs-job-filter")).not.toBeInTheDocument()
    expect(screen.queryByTestId("watchlists-outputs-run-filter")).not.toBeInTheDocument()
    fireEvent.click(screen.getByTestId("watchlists-outputs-advanced-toggle"))
    expect(screen.getByTestId("watchlists-outputs-job-filter")).toBeInTheDocument()
    expect(screen.getByTestId("watchlists-outputs-run-filter")).toBeInTheDocument()

    await waitFor(() => {
      expect(localStorage.getItem(STORAGE_KEY)).toBe("1")
    })
  })

  it("auto-opens when an output filter is active", () => {
    mocks.storeStateRef.current = baseState({ outputsJobFilter: 8 })

    render(<OutputsTab />)

    expect(screen.getByTestId("watchlists-outputs-job-filter")).toBeInTheDocument()
    expect(screen.getByTestId("watchlists-outputs-run-filter")).toBeInTheDocument()
  })

  it("auto-opens when a run filter is active", () => {
    mocks.storeStateRef.current = baseState({ outputsRunFilter: 42 })

    render(<OutputsTab />)

    expect(screen.getByTestId("watchlists-outputs-job-filter")).toBeInTheDocument()
    expect(screen.getByTestId("watchlists-outputs-run-filter")).toBeInTheDocument()
  })

  it("filters visible outputs by delivery status in advanced mode", async () => {
    mocks.storeStateRef.current = baseState({
      outputs: [buildOutput(1, "failed"), buildOutput(2, "sent")],
      outputsTotal: 2
    })

    render(<OutputsTab />)

    fireEvent.click(screen.getByTestId("watchlists-outputs-advanced-toggle"))
    expect(screen.getByTestId("watchlists-outputs-delivery-filter")).toBeInTheDocument()
    expect(screen.getByTestId("outputs-table-rows")).toHaveTextContent("2")

    fireEvent.change(screen.getByTestId("watchlists-outputs-delivery-filter"), {
      target: { value: "failed" }
    })

    await waitFor(() => {
      expect(screen.getByTestId("outputs-table-rows")).toHaveTextContent("1")
    })
  })

  it("highlights delivery issues and links remediation actions", async () => {
    const setRunsStatusFilter = vi.fn()
    const setRunsJobFilter = vi.fn()
    const setActiveTab = vi.fn()

    mocks.storeStateRef.current = baseState({
      outputs: [buildOutput(1, "failed"), buildOutput(2, "sent")],
      outputsTotal: 2,
      setRunsStatusFilter,
      setRunsJobFilter,
      setActiveTab
    })

    render(<OutputsTab />)

    const deliveryIssuesBanner = screen.getByTestId("watchlists-outputs-delivery-issues-banner")
    expect(deliveryIssuesBanner).toHaveAttribute("data-ds-component", "Alert")
    expect(deliveryIssuesBanner).toHaveTextContent(
      "Delivery issues detected in 1 report."
    )
    expect(screen.getByTestId("outputs-table-rows")).toHaveTextContent("2")

    fireEvent.click(screen.getByTestId("watchlists-outputs-banner-show-failed"))

    await waitFor(() => {
      expect(screen.getByTestId("outputs-table-rows")).toHaveTextContent("1")
    })

    fireEvent.click(screen.getByTestId("watchlists-outputs-banner-open-runs"))

    expect(setRunsStatusFilter).toHaveBeenCalledWith("failed")
    expect(setRunsJobFilter).toHaveBeenCalledWith(null)
    expect(setActiveTab).toHaveBeenCalledWith("runs")
  })

  it("tracks first output milestone when outputs load returns items", async () => {
    mocks.fetchWatchlistOutputsMock.mockResolvedValue({
      items: [buildOutput(501, "sent")],
      total: 1,
      has_more: false
    })

    render(<OutputsTab />)

    await waitFor(() => {
      expect(mocks.trackWatchlistsOnboardingTelemetryMock).toHaveBeenCalledWith({
        type: "first_output_succeeded",
        outputId: 501,
        format: "md"
      })
    })
  })
})
