import React from "react"
import { render, screen, waitFor, within } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { SettingsTab } from "../SettingsTab"
import { WATCHLISTS_HELP_DOCS } from "../../shared/help-docs"
import { setViewport } from "../../__tests__/test-utils/viewport"

const ONBOARDING_PATH_STORAGE_KEY = "watchlists:onboarding-path:v1"

const createLocalStorageMock = (): Storage => {
  const values = new Map<string, string>()

  return {
    get length() {
      return values.size
    },
    clear: () => values.clear(),
    getItem: (key: string) => values.get(key) ?? null,
    key: (index: number) => Array.from(values.keys())[index] ?? null,
    removeItem: (key: string) => {
      values.delete(key)
    },
    setItem: (key: string, value: string) => {
      values.set(key, value)
    }
  }
}

let testLocalStorage: Storage | undefined

const ensureLocalStorage = (): void => {
  testLocalStorage ??= createLocalStorageMock()
  Object.defineProperty(globalThis, "localStorage", {
    configurable: true,
    value: testLocalStorage
  })
  Object.defineProperty(window, "localStorage", {
    configurable: true,
    value: testLocalStorage
  })
}

const mocks = vi.hoisted(() => ({
  getWatchlistSettingsMock: vi.fn(),
  fetchWatchlistJobsMock: vi.fn(),
  fetchClaimClustersMock: vi.fn(),
  fetchJobClaimClustersMock: vi.fn(),
  subscribeJobToClusterMock: vi.fn(),
  unsubscribeJobFromClusterMock: vi.fn(),
  messageErrorMock: vi.fn()
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
    t: (_key: string, defaultValue?: unknown, values?: Record<string, unknown>) =>
      typeof defaultValue === "string"
        ? interpolate(defaultValue, values)
        : _key
  })
}))

vi.mock("antd", () => {
  const Button = ({ children, onClick, loading: _loading, ...rest }: any) => (
    <button type="button" {...rest} onClick={() => onClick?.()}>
      {children}
    </button>
  )
  const Card = ({ title, children }: any) => (
    <section>
      <h2>{title}</h2>
      {children}
    </section>
  )
  const DescriptionsComponent = ({ children }: any) => <div>{children}</div>
  ;(DescriptionsComponent as any).Item = ({ label, children }: any) => (
    <div>
      <strong>{label}</strong>
      <span>{children}</span>
    </div>
  )
  const Input = {
    Search: ({ value, onChange, onSearch }: any) => (
      <input
        value={value || ""}
        onChange={(event) => onChange?.(event)}
        onKeyDown={(event) => {
          if (event.key === "Enter") onSearch?.(value)
        }}
      />
    )
  }
  const Select = ({ options = [], value, onChange, ...rest }: any) => (
    <select
      data-testid={rest["data-testid"]}
      value={value ?? ""}
      onChange={(event) => onChange?.(event.currentTarget.value || null)}
    >
      <option value="" />
      {options.map((option: any) => (
        <option key={String(option.value)} value={String(option.value)}>
          {String(option.label)}
        </option>
      ))}
    </select>
  )
  const Skeleton = () => <div>Loading...</div>
  const Switch = ({
    checked,
    onChange,
    disabled,
    loading: _loading,
    checkedChildren,
    unCheckedChildren,
    ...rest
  }: any) => (
    <button
      type="button"
      role="switch"
      aria-checked={checked ? "true" : "false"}
      disabled={Boolean(disabled)}
      onClick={() => onChange?.(!checked)}
      {...rest}
    >
      {checked ? checkedChildren || "On" : unCheckedChildren || "Off"}
    </button>
  )
  const Table = ({ dataSource = [], columns = [], ...rest }: any) => (
    <table data-testid={rest["data-testid"] || "settings-clusters-table"}>
      <tbody>
        {dataSource.map((record: any, rowIndex: number) => (
          <tr key={record.id ?? rowIndex}>
            {columns.map((column: any, columnIndex: number) => {
              const key = String(column.key ?? column.dataIndex ?? columnIndex)
              const value = column.dataIndex ? record[column.dataIndex] : undefined
              const content = column.render
                ? column.render(value, record, rowIndex)
                : value
              return <td key={key}>{content}</td>
            })}
          </tr>
        ))}
      </tbody>
    </table>
  )
  const Tooltip = ({ title, children }: any) => (
    <div>
      {children}
      {title}
    </div>
  )

  return {
    Button,
    Card,
    Descriptions: DescriptionsComponent,
    Empty: ({ description }: any) => <div>{description}</div>,
    Input,
    Select,
    Skeleton,
    Switch,
    Table,
    Tooltip,
    message: {
      error: mocks.messageErrorMock,
      success: vi.fn(),
      warning: vi.fn()
    }
  }
})

vi.mock("@/store/watchlists", () => ({
  useWatchlistsStore: (selector: (state: Record<string, any>) => unknown) =>
    selector({
      settings: {
        default_output_ttl_seconds: 86400,
        temporary_output_ttl_seconds: 3600
      },
      settingsLoading: false,
      setSettings: vi.fn(),
      setSettingsLoading: vi.fn()
    })
}))

vi.mock("@/services/watchlists", () => ({
  fetchClaimClusters: (...args: any[]) => mocks.fetchClaimClustersMock(...args),
  fetchJobClaimClusters: (...args: any[]) => mocks.fetchJobClaimClustersMock(...args),
  fetchWatchlistJobs: (...args: any[]) => mocks.fetchWatchlistJobsMock(...args),
  getWatchlistSettings: (...args: any[]) => mocks.getWatchlistSettingsMock(...args),
  subscribeJobToCluster: (...args: any[]) => mocks.subscribeJobToClusterMock(...args),
  unsubscribeJobFromCluster: (...args: any[]) => mocks.unsubscribeJobFromClusterMock(...args)
}))

vi.mock("@/utils/humanize-milliseconds", () => ({
  humanizeMilliseconds: (value: number) => `${value} ms`
}))

vi.mock("@/utils/dateFormatters", () => ({
  formatRelativeTime: () => "just now"
}))

describe("SettingsTab contextual help", () => {
  const originalDiagnosticsFlag = process.env.NEXT_PUBLIC_WATCHLISTS_SHOW_INTERNAL_DIAGNOSTICS

  beforeEach(() => {
    vi.clearAllMocks()
    ensureLocalStorage()
    setViewport(1024)
    localStorage.removeItem(ONBOARDING_PATH_STORAGE_KEY)
    delete process.env.NEXT_PUBLIC_WATCHLISTS_SHOW_INTERNAL_DIAGNOSTICS
    mocks.getWatchlistSettingsMock.mockResolvedValue({
      default_output_ttl_seconds: 86400,
      temporary_output_ttl_seconds: 3600
    })
    mocks.fetchWatchlistJobsMock.mockResolvedValue({ items: [], total: 0, has_more: false })
    mocks.fetchClaimClustersMock.mockResolvedValue([])
    mocks.fetchJobClaimClustersMock.mockResolvedValue([])
    mocks.subscribeJobToClusterMock.mockResolvedValue(undefined)
    mocks.unsubscribeJobFromClusterMock.mockResolvedValue(undefined)
  })

  afterEach(() => {
    if (originalDiagnosticsFlag == null) {
      delete process.env.NEXT_PUBLIC_WATCHLISTS_SHOW_INTERNAL_DIAGNOSTICS
      return
    }
    process.env.NEXT_PUBLIC_WATCHLISTS_SHOW_INTERNAL_DIAGNOSTICS = originalDiagnosticsFlag
  })

  it("shows claim cluster explanation with docs link and help trigger", async () => {
    render(<SettingsTab />)

    await waitFor(() => {
      expect(
        screen.getByText("Related Topics (Claim Clusters)")
      ).toBeInTheDocument()
    })

    expect(screen.getByTestId("watchlists-help-claimClusters")).toBeInTheDocument()
    const links = screen.getAllByRole("link", { name: "Learn more" })
    expect(
      links.some((link) => link.getAttribute("href") === WATCHLISTS_HELP_DOCS.claimClusters)
    ).toBe(true)
  })

  it("renders settings guidance with design-system Alert primitives", async () => {
    const { container } = render(<SettingsTab />)

    await waitFor(() => {
      expect(
        screen.getByText("TTL values are configured on the server.")
      ).toBeInTheDocument()
    })

    const alerts = Array.from(
      container.querySelectorAll('[data-ds-component="Alert"]')
    )
    expect(alerts).toHaveLength(2)
    expect(
      alerts.some((alert) =>
        within(alert as HTMLElement).queryByText(
          "TTL values are configured on the server."
        )
      )
    ).toBe(true)
    expect(
      alerts.some((alert) =>
        within(alert as HTMLElement).queryByText(
          "Select a monitor to manage cluster subscriptions."
        )
      )
    ).toBe(true)
  })

  it("hides internal diagnostics by default", async () => {
    render(<SettingsTab />)

    await waitFor(() => {
      expect(
        screen.getByText("Related Topics (Claim Clusters)")
      ).toBeInTheDocument()
    })

    expect(screen.queryByText("Internal diagnostics")).not.toBeInTheDocument()
    expect(screen.queryByText("Phase 3 Readiness")).not.toBeInTheDocument()
  })

  it("shows internal diagnostics when explicitly enabled", async () => {
    process.env.NEXT_PUBLIC_WATCHLISTS_SHOW_INTERNAL_DIAGNOSTICS = "true"
    render(<SettingsTab />)

    await waitFor(() => {
      expect(screen.getByText("Internal diagnostics")).toBeInTheDocument()
    })
  })

  it("persists onboarding path selection from settings", async () => {
    render(<SettingsTab />)

    await waitFor(() => {
      expect(screen.getByText("Onboarding")).toBeInTheDocument()
    })

    const select = screen.getByTestId("watchlists-settings-onboarding-path-select")
    expect(select).toBeInTheDocument()

    ;(select as HTMLSelectElement).value = "advanced"
    select.dispatchEvent(new Event("change", { bubbles: true }))

    expect(localStorage.getItem(ONBOARDING_PATH_STORAGE_KEY)).toBe("advanced")
  })

  it("renders related-topic subscriptions as constrained cards instead of a table", async () => {
    setViewport(420)
    mocks.fetchWatchlistJobsMock.mockResolvedValue({
      items: [{ id: 17, name: "CVE monitor" }],
      total: 1,
      has_more: false
    })
    mocks.fetchClaimClustersMock.mockResolvedValue([
      {
        id: 44,
        summary: "OpenSSL vulnerability cluster",
        canonical_claim_text: "OpenSSL advisory",
        member_count: 7,
        updated_at: "2026-02-24T12:00:00Z"
      }
    ])
    mocks.fetchJobClaimClustersMock.mockResolvedValue([{ cluster_id: 44 }])

    render(<SettingsTab />)

    await waitFor(() => {
      expect(screen.getByTestId("settings-clusters-constrained-list")).toBeInTheDocument()
    })

    expect(screen.queryByTestId("settings-clusters-table")).not.toBeInTheDocument()
    expect(screen.getByText("OpenSSL vulnerability cluster")).toBeInTheDocument()
    expect(
      screen.getByRole("switch", { name: /Toggle subscription for OpenSSL vulnerability cluster/i })
    ).toBeInTheDocument()
  })
})
