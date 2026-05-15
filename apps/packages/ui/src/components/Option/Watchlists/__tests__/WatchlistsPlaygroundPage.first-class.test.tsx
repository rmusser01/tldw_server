// @vitest-environment jsdom

import React from "react"
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { WatchlistsPlaygroundPage } from "../WatchlistsPlaygroundPage"

const container = {
  id: 42,
  name: "Healthcare ransomware",
  description: "Track hospital impact",
  objective: "Find new ransomware affecting hospitals",
  domain: "cti_osint",
  status: "active",
  priority: "high",
  tags: ["ransomware", "hospitals"],
  created_at: "2026-05-15T00:00:00Z",
  updated_at: "2026-05-15T00:00:00Z"
}

const mocks = vi.hoisted(() => {
  const state = {
    activeTab: "sources" as
      | "overview"
      | "sources"
      | "jobs"
      | "runs"
      | "items"
      | "outputs"
      | "templates"
      | "settings",
    overviewHealth: null as null | Record<string, unknown>,
    pollingActive: false,
    watchlists: [] as Array<Record<string, unknown>>,
    watchlistsLoading: false,
    watchlistsError: null as string | null,
    selectedWatchlistId: null as number | null,
    setActiveTab: vi.fn((next: string) => {
      state.activeTab = next as typeof state.activeTab
    }),
    setWatchlists: vi.fn((nextWatchlists: Array<Record<string, unknown>>, selectedId?: number | null) => {
      state.watchlists = nextWatchlists
      state.selectedWatchlistId = selectedId !== undefined
        ? selectedId
        : typeof state.selectedWatchlistId === "number" &&
            nextWatchlists.some((watchlist) => watchlist.id === state.selectedWatchlistId)
          ? state.selectedWatchlistId
          : Number(nextWatchlists[0]?.id ?? null) || null
    }),
    setWatchlistsLoading: vi.fn((loading: boolean) => {
      state.watchlistsLoading = loading
    }),
    setWatchlistsError: vi.fn((error: string | null) => {
      state.watchlistsError = error
    }),
    setSelectedWatchlistId: vi.fn((nextId: number | null) => {
      state.selectedWatchlistId = nextId
    }),
    addWatchlist: vi.fn((watchlist: Record<string, unknown>) => {
      state.watchlists = [watchlist, ...state.watchlists]
      state.selectedWatchlistId = Number(watchlist.id)
    }),
    updateWatchlistInList: vi.fn((watchlistId: number, updates: Record<string, unknown>) => {
      state.watchlists = state.watchlists.map((watchlist) =>
        watchlist.id === watchlistId ? { ...watchlist, ...updates } : watchlist
      )
    }),
    resetStore: vi.fn()
  }

  return {
    createWatchlistMock: vi.fn(),
    bulkCreateSourcesMock: vi.fn(),
    createWatchlistJobMock: vi.fn(),
    createWatchlistSourceMock: vi.fn(),
    fetchWatchlistRunsMock: vi.fn(),
    fetchWatchlistsMock: vi.fn(),
    updateWatchlistMock: vi.fn(),
    notificationDestroyMock: vi.fn(),
    openJobFormMock: vi.fn(),
    openRunDetailMock: vi.fn(),
    openSourceFormMock: vi.fn(),
    recordWatchlistsIaExperimentTelemetryMock: vi.fn(),
    trackWatchlistsOnboardingTelemetryMock: vi.fn(),
    state
  }
})

const connectionMocks = vi.hoisted(() => ({
  useConnectionUxState: vi.fn()
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

vi.mock("antd", async () => {
  const actual = await vi.importActual<typeof import("antd")>("antd")
  const Alert = ({ title, description, action, children }: any) => (
    <div>
      <div>{title}</div>
      <div>{description}</div>
      <div>{action}</div>
      {children}
    </div>
  )
  const Button = ({ children, icon, onClick, disabled, ...rest }: any) => (
    <button type="button" onClick={() => onClick?.({ preventDefault: vi.fn(), stopPropagation: vi.fn() })} disabled={Boolean(disabled)} {...rest}>
      {icon}
      {children}
    </button>
  )
  const Drawer = ({ open, title, children }: any) =>
    open ? (
      <div>
        <h3>{title}</h3>
        {children}
      </div>
    ) : null
  const Input = ({ value, onChange, ...rest }: any) => (
    <input value={value ?? ""} onChange={(event) => onChange?.(event)} {...rest} />
  )
  Input.TextArea = ({ value, onChange, ...rest }: any) => (
    <textarea value={value ?? ""} onChange={(event) => onChange?.(event)} {...rest} />
  )
  const Modal = ({ open, title, children, footer, onOk, onCancel, okText, cancelText }: any) =>
    open ? (
      <div role="dialog" aria-label={typeof title === "string" ? title : "dialog"}>
        <h3>{title}</h3>
        {children}
        <div>{footer}</div>
        {onOk ? <button type="button" onClick={() => onOk()}>{okText || "OK"}</button> : null}
        {onCancel ? <button type="button" onClick={() => onCancel()}>{cancelText || "Cancel"}</button> : null}
      </div>
    ) : null
  const Select = ({ value, options = [], onChange, "aria-label": ariaLabel, ...rest }: any) => (
    <select
      aria-label={ariaLabel}
      value={value == null ? "" : String(value)}
      onChange={(event) => {
        const rawValue = event.currentTarget.value
        const numeric = Number(rawValue)
        onChange?.(Number.isFinite(numeric) && rawValue.trim() !== "" ? numeric : rawValue)
      }}
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
    <button
      type="button"
      aria-label={rest["aria-label"] || "Toggle"}
      aria-pressed={Boolean(checked)}
      onClick={() => onChange?.(!checked)}
      {...rest}
    />
  )
  const Tabs = ({ activeKey, items = [] }: any) => {
    const activeItem = items.find((item: any) => item.key === activeKey) || items[0]
    return (
      <div>
        {items.map((item: any) => (
          <button key={item.key} type="button" data-testid={`watchlists-tab-${item.key}`}>
            {item.label}
          </button>
        ))}
        <div>{activeItem?.children}</div>
      </div>
    )
  }
  const Tooltip = ({ children }: any) => <>{children}</>
  return { ...actual, Alert, Button, Drawer, Input, Modal, Select, Switch, Tabs, Tooltip }
})

vi.mock("@/components/Common/WorkspaceConnectionGate", () => ({
  default: ({ children }: { children: React.ReactNode }) => <>{children}</>
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({
    destroy: mocks.notificationDestroyMock,
    success: vi.fn(),
    error: vi.fn(),
    warning: vi.fn()
  })
}))

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => true
}))

vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionUxState: () => connectionMocks.useConnectionUxState()
}))

vi.mock("@/services/watchlists", () => ({
  bulkCreateSources: (...args: unknown[]) => mocks.bulkCreateSourcesMock(...args),
  createWatchlist: (...args: unknown[]) => mocks.createWatchlistMock(...args),
  createWatchlistJob: (...args: unknown[]) => mocks.createWatchlistJobMock(...args),
  createWatchlistSource: (...args: unknown[]) => mocks.createWatchlistSourceMock(...args),
  fetchWatchlistRuns: (...args: unknown[]) => mocks.fetchWatchlistRunsMock(...args),
  fetchWatchlists: (...args: unknown[]) => mocks.fetchWatchlistsMock(...args),
  updateWatchlist: (...args: unknown[]) => mocks.updateWatchlistMock(...args),
  recordWatchlistsIaExperimentTelemetry: (...args: unknown[]) =>
    mocks.recordWatchlistsIaExperimentTelemetryMock(...args)
}))

vi.mock("@/utils/watchlists-onboarding-telemetry", () => ({
  trackWatchlistsOnboardingTelemetry: (...args: unknown[]) =>
    mocks.trackWatchlistsOnboardingTelemetryMock(...args)
}))

vi.mock("@/store/watchlists", () => ({
  useWatchlistsStore: (selector: (state: Record<string, unknown>) => unknown) =>
    selector({
      ...mocks.state,
      openJobForm: mocks.openJobFormMock,
      openRunDetail: mocks.openRunDetailMock,
      openSourceForm: mocks.openSourceFormMock
    })
}))

vi.mock("../OverviewTab/OverviewTab", () => ({
  OverviewTab: () => <div>Overview tab</div>
}))
vi.mock("../SourcesTab/SourcesTab", () => ({
  SourcesTab: () => <div>Sources tab</div>
}))
vi.mock("../JobsTab/JobsTab", () => ({
  JobsTab: () => <div>Jobs tab</div>
}))
vi.mock("../RunsTab/RunsTab", () => ({
  RunsTab: () => <div>Runs tab</div>
}))
vi.mock("../OutputsTab/OutputsTab", () => ({
  OutputsTab: () => <div>Outputs tab</div>
}))
vi.mock("../TemplatesTab/TemplatesTab", () => ({
  TemplatesTab: () => <div>Templates tab</div>
}))
vi.mock("../SettingsTab/SettingsTab", () => ({
  SettingsTab: () => <div>Settings tab</div>
}))
vi.mock("../ItemsTab/ItemsTab", () => ({
  ItemsTab: () => <div>Items tab</div>
}))
vi.mock("../shared/WatchlistsHealthBar", () => ({
  WatchlistsHealthBar: () => <div data-testid="watchlists-health-bar" />
}))

describe("WatchlistsPlaygroundPage first-class Watchlist shell", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    connectionMocks.useConnectionUxState.mockReturnValue({
      uxState: "connected_ok",
      hasCompletedFirstRun: true
    })
    mocks.fetchWatchlistsMock.mockResolvedValue({
      items: [container],
      total: 1,
      has_more: false
    })
    mocks.fetchWatchlistRunsMock.mockResolvedValue({ items: [], total: 0, has_more: false })
    mocks.bulkCreateSourcesMock.mockResolvedValue({ items: [], total: 0, created: 0, errors: 0 })
    mocks.createWatchlistJobMock.mockResolvedValue({
      id: 88,
      name: "Monitor",
      watchlist_id: 77,
      scope: { sources: [] },
      active: true,
      created_at: "2026-05-15T00:00:00Z",
      updated_at: "2026-05-15T00:00:00Z"
    })
    mocks.createWatchlistSourceMock.mockResolvedValue({
      id: 901,
      name: "example.com",
      url: "https://example.com/feed.xml",
      source_type: "rss",
      active: true,
      tags: [],
      created_at: "2026-05-15T00:00:00Z",
      updated_at: "2026-05-15T00:00:00Z"
    })
    mocks.recordWatchlistsIaExperimentTelemetryMock.mockResolvedValue({ accepted: true })
    mocks.trackWatchlistsOnboardingTelemetryMock.mockResolvedValue(undefined)
    mocks.state.activeTab = "sources"
    mocks.state.watchlists = []
    mocks.state.watchlistsLoading = false
    mocks.state.watchlistsError = null
    mocks.state.selectedWatchlistId = null
    delete (window as { __TLDW_WATCHLISTS_IA_EXPERIMENT__?: unknown }).__TLDW_WATCHLISTS_IA_EXPERIMENT__
  })

  afterEach(() => {
    cleanup()
  })

  it("loads Watchlists, selects the first container, and shows project metadata", async () => {
    const view = render(<WatchlistsPlaygroundPage />)

    await waitFor(() => expect(mocks.fetchWatchlistsMock).toHaveBeenCalledWith({ page: 1, size: 100 }))
    await waitFor(() => expect(mocks.state.setWatchlists).toHaveBeenCalledWith([container]))
    view.rerender(<WatchlistsPlaygroundPage />)

    expect(await screen.findByRole("heading", { name: "Healthcare ransomware" })).toBeInTheDocument()
    expect(screen.getByText("Find new ransomware affecting hospitals")).toBeInTheDocument()
    expect(screen.getByText("CTI / OSINT")).toBeInTheDocument()
    expect(screen.getByText("High")).toBeInTheDocument()
  })

  it("creates a topic-only Watchlist from the shell wizard, selects it, and opens Feeds", async () => {
    const created = {
      ...container,
      id: 77,
      name: "Election integrity",
      domain: "news",
      priority: "medium",
      tags: ["elections"]
    }
    mocks.createWatchlistMock.mockResolvedValue(created)
    mocks.state.activeTab = "overview"

    render(<WatchlistsPlaygroundPage />)

    fireEvent.click(await screen.findByTestId("watchlists-create-container"))
    const wizard = within(screen.getByRole("dialog", { name: "Create Watchlist" }))
    fireEvent.click(wizard.getByRole("button", { name: "News" }))
    fireEvent.click(wizard.getByRole("button", { name: "Start from topic" }))
    fireEvent.click(wizard.getByRole("button", { name: "Next" }))
    fireEvent.change(wizard.getByLabelText("Watchlist name"), { target: { value: "Election integrity" } })
    fireEvent.change(wizard.getByLabelText("Objective"), {
      target: { value: "Track source diversity and recency" }
    })
    fireEvent.change(wizard.getByLabelText("Tracked scope"), {
      target: { value: "election officials, state courts" }
    })
    fireEvent.click(wizard.getByRole("button", { name: "Next" }))
    fireEvent.click(wizard.getByRole("button", { name: "Next" }))
    fireEvent.click(wizard.getByRole("button", { name: "Create Watchlist" }))

    await waitFor(() =>
      expect(mocks.createWatchlistMock).toHaveBeenCalledWith(
        expect.objectContaining({
          name: "Election integrity",
          objective: "Track source diversity and recency",
          domain: "news",
          priority: "medium",
          status: "active",
          tags: expect.arrayContaining(["news", "election officials", "state courts"])
        })
      )
    )
    expect(mocks.state.addWatchlist).toHaveBeenCalledWith(created)
    expect(mocks.state.setSelectedWatchlistId).toHaveBeenCalledWith(77)
    expect(mocks.state.setActiveTab).toHaveBeenCalledWith("sources")
  })

  it("creates source-backed setup payloads with watchlist_id and opens Monitors", async () => {
    const created = {
      ...container,
      id: 88,
      name: "Healthcare ransomware"
    }
    mocks.createWatchlistMock.mockResolvedValue(created)
    mocks.bulkCreateSourcesMock.mockResolvedValue({
      items: [
        { url: "https://example.com/feed.xml", id: 901, status: "created" },
        { url: "https://advisories.example.org/rss", id: 902, status: "created" }
      ],
      total: 2,
      created: 2,
      errors: 0
    })
    mocks.createWatchlistJobMock.mockResolvedValue({
      id: 990,
      name: "Healthcare ransomware monitor",
      watchlist_id: 88,
      scope: { sources: [901, 902] },
      active: true,
      created_at: "2026-05-15T00:00:00Z",
      updated_at: "2026-05-15T00:00:00Z"
    })
    mocks.state.activeTab = "overview"

    render(<WatchlistsPlaygroundPage />)

    fireEvent.click(await screen.findByTestId("watchlists-create-container"))
    const wizard = within(screen.getByRole("dialog", { name: "Create Watchlist" }))
    fireEvent.click(wizard.getByRole("button", { name: "CTI / OSINT" }))
    fireEvent.click(wizard.getByRole("button", { name: "Start from sources" }))
    fireEvent.click(wizard.getByRole("button", { name: "Next" }))
    fireEvent.change(wizard.getByLabelText("Watchlist name"), {
      target: { value: "Healthcare ransomware" }
    })
    fireEvent.change(wizard.getByLabelText("Objective"), {
      target: { value: "Find ransomware reports affecting hospitals" }
    })
    fireEvent.change(wizard.getByLabelText("Tracked scope"), {
      target: { value: "hospitals, Germany" }
    })
    fireEvent.click(wizard.getByRole("button", { name: "Next" }))
    fireEvent.change(wizard.getByLabelText("Source URLs"), {
      target: { value: "https://example.com/feed.xml\nhttps://advisories.example.org/rss" }
    })
    fireEvent.change(wizard.getByLabelText("Monitor name"), {
      target: { value: "Healthcare ransomware monitor" }
    })
    fireEvent.click(wizard.getByRole("button", { name: "Next" }))
    fireEvent.click(wizard.getByRole("button", { name: "Create Watchlist" }))

    await waitFor(() =>
      expect(mocks.bulkCreateSourcesMock).toHaveBeenCalledWith(
        expect.arrayContaining([
          expect.objectContaining({
            url: "https://example.com/feed.xml",
            watchlist_id: 88
          }),
          expect.objectContaining({
            url: "https://advisories.example.org/rss",
            watchlist_id: 88
          })
        ])
      )
    )
    expect(mocks.createWatchlistJobMock).toHaveBeenCalledWith(
      expect.objectContaining({
        name: "Healthcare ransomware monitor",
        watchlist_id: 88,
        scope: { sources: [901, 902] }
      })
    )
    expect(mocks.state.setSelectedWatchlistId).toHaveBeenCalledWith(88)
    expect(mocks.state.setActiveTab).toHaveBeenCalledWith("jobs")
  })
})
