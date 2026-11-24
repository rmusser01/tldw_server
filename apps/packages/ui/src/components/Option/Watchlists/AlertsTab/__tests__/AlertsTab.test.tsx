// @vitest-environment jsdom

import React from "react"
import { act, cleanup, fireEvent, screen, waitFor, within } from "@testing-library/react"
import { render } from "@/components/Option/Watchlists/shared/__tests__/render-with-watchlists-query"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  createRule: vi.fn(),
  deleteRule: vi.fn(),
  fetchAlerts: vi.fn(),
  fetchRules: vi.fn(),
  updateAlert: vi.fn(),
  updateRule: vi.fn(),
  selectedWatchlistId: 42
}))

vi.mock("react-i18next", () => {
  const t = (key: string, defaultValue?: unknown, options?: Record<string, unknown>) => {
    if (typeof defaultValue !== "string") return key
    if (!options) return defaultValue
    return defaultValue.replace(/\{\{(\w+)\}\}/g, (_, token) => String(options[token] ?? ""))
  }
  return { useTranslation: () => ({ t }) }
})

vi.mock("antd", async () => {
  const actual = await vi.importActual<typeof import("antd")>("antd")
  const Alert = ({ title, message, description, action, children }: any) => (
    <div>
      <div>{message || title}</div>
      <div>{description}</div>
      <div>{action}</div>
      {children}
    </div>
  )
  const Button = ({ children, icon, onClick, disabled, type: _type, loading: _loading, danger: _danger, ...rest }: any) => (
    <button type="button" onClick={(event) => onClick?.(event)} disabled={Boolean(disabled)} {...rest}>
      {icon}
      {children}
    </button>
  )
  const Empty = ({ description }: any) => <div>{description}</div>
  const Input = ({ value, onChange, ...rest }: any) => (
    <input value={value ?? ""} onChange={(event) => onChange?.(event)} {...rest} />
  )
  const Select = ({ value, options = [], onChange, "aria-label": ariaLabel, ...rest }: any) => (
    <select
      aria-label={ariaLabel}
      value={value == null ? "" : String(value)}
      onChange={(event) => onChange?.(event.currentTarget.value)}
      {...rest}
    >
      {options.map((option: any) => (
        <option key={String(option.value)} value={String(option.value)}>
          {option.label}
        </option>
      ))}
    </select>
  )
  const Switch = ({ checked, onChange, ...rest }: any) => (
    <button type="button" aria-pressed={Boolean(checked)} onClick={() => onChange?.(!checked)} {...rest} />
  )
  const Tag = ({ children }: any) => <span>{children}</span>
  const Tooltip = ({ children }: any) => <>{children}</>
  return { ...actual, Alert, Button, Empty, Input, Select, Switch, Tag, Tooltip }
})

vi.mock("@/store/watchlists", () => ({
  useWatchlistsStore: (selector: (state: Record<string, unknown>) => unknown) =>
    selector({ selectedWatchlistId: mocks.selectedWatchlistId })
}))

vi.mock("@/services/watchlists", () => ({
  createWatchlistContentAlertRule: (...args: unknown[]) => mocks.createRule(...args),
  deleteWatchlistContentAlertRule: (...args: unknown[]) => mocks.deleteRule(...args),
  fetchWatchlistContentAlertRules: (...args: unknown[]) => mocks.fetchRules(...args),
  fetchWatchlistContentAlerts: (...args: unknown[]) => mocks.fetchAlerts(...args),
  updateWatchlistContentAlert: (...args: unknown[]) => mocks.updateAlert(...args),
  updateWatchlistContentAlertRule: (...args: unknown[]) => mocks.updateRule(...args)
}))

import { AlertsTab } from "../AlertsTab"

const rule = {
  id: 7,
  watchlist_id: 42,
  name: "Active exploitation",
  enabled: true,
  rule_kind: "descriptor",
  match_mode: "contains",
  pattern: "active exploitation",
  severity: "critical",
  source_constraints: { source_tags: ["advisory"] },
  metadata: null,
  created_at: "2026-05-15T00:00:00Z",
  updated_at: "2026-05-15T00:00:00Z"
}

const alert = {
  id: 99,
  watchlist_id: 42,
  rule_id: 7,
  item_id: 501,
  run_id: 33,
  job_id: 22,
  source_id: 11,
  severity: "critical",
  status: "unread",
  title: "CVE-2026-9999 active exploitation observed",
  snippet: "Active exploitation is affecting healthcare providers.",
  matched_text: "Active exploitation",
  evidence: {
    source_name: "Advisory feed",
    source_url: "https://example.com/advisories.xml",
    url: "https://example.com/advisory/cve-2026-9999"
  },
  dedupe_key: "watchlist_content_alert:42:7:501",
  created_at: "2026-05-15T12:00:00Z",
  read_at: null,
  dismissed_at: null
}

describe("AlertsTab", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.selectedWatchlistId = 42
    mocks.fetchRules.mockResolvedValue({ items: [rule], total: 1 })
    mocks.fetchAlerts.mockResolvedValue({ items: [alert], total: 1 })
    mocks.updateAlert.mockResolvedValue({ ...alert, status: "read" })
    mocks.createRule.mockResolvedValue(rule)
  })

  afterEach(() => {
    cleanup()
  })

  it("renders rule management, alert evidence, and review actions for the selected Watchlist", async () => {
    render(<AlertsTab />)

    expect(await screen.findByText("Content alert rules")).toBeInTheDocument()
    expect(mocks.fetchRules).toHaveBeenCalledWith(42, { page: 1, size: 100 }, expect.any(AbortSignal))
    expect(mocks.fetchAlerts).toHaveBeenCalledWith(42, {
      status: "unread",
      severity: undefined,
      rule_id: undefined,
      source_id: undefined,
      page: 1,
      size: 50
    }, expect.any(AbortSignal))
    expect(screen.getByText("Alert inbox")).toBeInTheDocument()
    expect((await screen.findAllByText("Active exploitation")).length).toBeGreaterThan(0)
    expect(screen.getByText("CVE-2026-9999 active exploitation observed")).toBeInTheDocument()
    expect(screen.getByText("Active exploitation is affecting healthcare providers.")).toBeInTheDocument()
    expect(screen.getByText("Advisory feed")).toBeInTheDocument()
    const boundaryTitle = screen.getByText(
      "Run failures and source problems are health issues, not content alerts."
    )
    expect(boundaryTitle.closest('[data-ds-component="Alert"]')).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Mark read" }))

    await waitFor(() =>
      expect(mocks.updateAlert).toHaveBeenCalledWith(42, 99, { status: "read" })
    )
  })

  it("prevents empty content alert rules before save", async () => {
    render(<AlertsTab />)

    fireEvent.click(await screen.findByRole("button", { name: "Create rule" }))
    fireEvent.click(screen.getByRole("button", { name: "Save rule" }))

    expect(screen.getByText("Name and pattern are required")).toBeInTheDocument()
    expect(mocks.createRule).not.toHaveBeenCalled()
  })

  it("keeps the current Watchlist inbox when an earlier Watchlist request resolves last", async () => {
    let resolvePrevious!: (value: { items: Array<typeof alert>; total: number }) => void
    const previousAlerts = new Promise<{ items: Array<typeof alert>; total: number }>((resolve) => {
      resolvePrevious = resolve
    })
    const currentAlert = { ...alert, id: 100, watchlist_id: 43, title: "Current Watchlist alert" }
    mocks.fetchAlerts.mockImplementation((watchlistId) =>
      watchlistId === 42 ? previousAlerts : Promise.resolve({ items: [currentAlert], total: 1 })
    )

    const { rerender } = render(<AlertsTab />)
    await waitFor(() => expect(mocks.fetchAlerts).toHaveBeenCalledWith(42, expect.anything(), expect.any(AbortSignal)))
    mocks.selectedWatchlistId = 43
    rerender(<AlertsTab />)
    await screen.findByText(currentAlert.title)

    await act(async () => resolvePrevious({ items: [alert], total: 1 }))

    expect(screen.getByText(currentAlert.title)).toBeInTheDocument()
    expect(screen.queryByText(alert.title)).toBeNull()
  })

  it("renders load errors with the design-system Alert primitive", async () => {
    mocks.fetchAlerts.mockRejectedValue(new Error("offline"))

    render(<AlertsTab />)

    const errorTitle = await screen.findByText("Failed to load content alerts")
    const errorAlert = errorTitle.closest('[data-ds-component="Alert"]')

    expect(errorAlert).toBeInTheDocument()
    expect(
      within(errorAlert as HTMLElement).getByRole("button", { name: "Refresh" })
    ).toBeInTheDocument()
  })

  it("marks the load-error refresh action busy while retrying", async () => {
    mocks.fetchAlerts.mockRejectedValue(new Error("offline"))

    render(<AlertsTab />)

    const errorTitle = await screen.findByText("Failed to load content alerts")
    const errorAlert = errorTitle.closest('[data-ds-component="Alert"]') as HTMLElement
    const refreshButton = within(errorAlert).getByRole("button", { name: "Refresh" })
    const pendingRules = new Promise(() => {})
    const pendingAlerts = new Promise(() => {})

    mocks.fetchRules.mockImplementation(() => pendingRules)
    mocks.fetchAlerts.mockImplementation(() => pendingAlerts)

    fireEvent.click(refreshButton)

    await waitFor(() => {
      const retryingAlert = screen
        .getByText("Failed to load content alerts")
        .closest('[data-ds-component="Alert"]') as HTMLElement
      const retryingButton = within(retryingAlert).getByRole("button", { name: "Refresh" })

      expect(retryingButton).toHaveAttribute("aria-busy", "true")
      expect(retryingButton).toBeDisabled()
    })
  })
})
