// @vitest-environment jsdom

import React from "react"
import axe from "axe-core"
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { WatchlistsPlaygroundPage } from "../WatchlistsPlaygroundPage"
import {
  WATCHLISTS_ISSUE_REPORT_URL,
  WATCHLISTS_MAIN_DOCS_URL,
  WATCHLISTS_TAB_HELP_DOCS
} from "../shared/help-docs"

const mocks = vi.hoisted(() => {
  const state = {
    activeTab: "sources" as
      | "overview"
      | "sources"
      | "jobs"
      | "runs"
      | "items"
      | "alerts"
      | "outputs"
      | "templates"
      | "settings",
    overviewHealth: {
      tabBadges: {
        sources: 0,
        runs: 0,
        outputs: 0
      }
    },
    setActiveTab: vi.fn((next: string) => {
      state.activeTab = next as typeof state.activeTab
    })
  }
  return {
    watchlistContainer: {
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
    },
    createWatchlistMock: vi.fn(),
    fetchWatchlistRunsMock: vi.fn(),
    fetchWatchlistsMock: vi.fn(),
    updateWatchlistMock: vi.fn(),
    recordWatchlistsIaExperimentTelemetryMock: vi.fn(),
    trackWatchlistsOnboardingTelemetryMock: vi.fn(),
    notificationDestroyMock: vi.fn(),
    state
  }
})

const connectionMocks = vi.hoisted(() => ({
  useConnectionUxState: vi.fn()
}))

const navigationMocks = vi.hoisted(() => ({
  navigate: vi.fn()
}))

const layoutMocks = vi.hoisted(() => ({
  isConstrained: false,
  longCopy: false
}))

const runA11yBaselineRules = async (context: Element) =>
  axe.run(context, {
    runOnly: {
      type: "rule",
      values: [
        "button-name",
        "link-name",
        "label",
        "aria-valid-attr",
        "aria-valid-attr-value",
        "aria-required-attr"
      ]
    },
    resultTypes: ["violations"]
  })

const expectNoInvalidAriaViolations = (
  violations: Array<{
    id: string
  }>
) => {
  const disallowedIds = new Set([
    "aria-valid-attr",
    "aria-valid-attr-value",
    "aria-required-attr"
  ])

  const disallowedViolations = violations.filter((violation) =>
    disallowedIds.has(violation.id)
  )

  expect(disallowedViolations).toEqual([])
}

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, defaultValue?: unknown, options?: Record<string, unknown>) => {
      if (typeof defaultValue !== "string") return _key
      if (layoutMocks.longCopy && _key === "watchlists:orientation.sources.description") {
        return `Extremely long localized guidance ${"unbreakablelocalizedcopy".repeat(18)}`
      }
      if (!options) return defaultValue
      return defaultValue.replace(/\{\{(\w+)\}\}/g, (_, token) => String(options[token] ?? ""))
    }
  })
}))

vi.mock("antd", async () => {
  const actual = await vi.importActual<typeof import("antd")>("antd")
  const Alert = ({
    title,
    description,
    action,
    closable,
    onClose,
    className,
    type: _type,
    showIcon: _showIcon,
    ...rest
  }: any) => (
    <div className={className} data-testid={rest["data-testid"]}>
      <div>{title}</div>
      <div>{description}</div>
      <div>{action}</div>
      {closable ? (
        <button type="button" onClick={() => onClose?.()}>
          Dismiss
        </button>
      ) : null}
    </div>
  )

  const Tabs = ({ items = [] }: any) => (
    <div>
      {items.map((item: any) => (
        <button
          key={item.key}
          type="button"
          data-testid={`watchlists-tab-${item.key}`}
        >
          {item.label}
        </button>
      ))}
      <div>{items[0]?.children}</div>
    </div>
  )

  const Modal = ({ open, title, children, footer, onCancel, afterOpenChange, width, className }: any) => {
    React.useEffect(() => {
      afterOpenChange?.(open)
    }, [afterOpenChange, open])

    return open ? (
      <div
        role="dialog"
        aria-label={typeof title === "string" ? title : "dialog"}
        className={className}
        data-modal-width={String(width)}
        tabIndex={-1}
        onKeyDown={(event) => {
          if (event.key === "Escape") onCancel?.()
        }}
      >
        <h3>{title}</h3>
        {children}
        <div>{footer}</div>
      </div>
    ) : null
  }
  const Drawer = ({ open, title, children }: any) =>
    open ? (
      <div>
        <h3>{title}</h3>
        {children}
      </div>
    ) : null

  const Empty = ({ description }: any) => <div>{description}</div>
  const Tooltip = ({ children }: any) => <>{children}</>
  const Button = React.forwardRef<HTMLButtonElement, any>(
    ({ children, onClick, disabled, block: _block, ...rest }, ref) => (
      <button
        ref={ref}
        type="button"
        onClick={() => onClick?.()}
        disabled={Boolean(disabled)}
        {...rest}
      >
        {children}
      </button>
    )
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
  const Tag = ({ children }: any) => <span>{children}</span>
  return { ...actual, Alert, Tabs, Empty, Button, Modal, Drawer, Tooltip, Switch, Select, Tag }
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

vi.mock("../shared/useWatchlistsViewport", () => ({
  useWatchlistsViewport: () => ({ isConstrained: layoutMocks.isConstrained })
}))

vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual<typeof import("react-router-dom")>(
    "react-router-dom"
  )
  return {
    ...actual,
    useNavigate: () => navigationMocks.navigate
  }
})

vi.mock("@/services/watchlists", () => ({
  createWatchlist: (...args: any[]) => mocks.createWatchlistMock(...args),
  fetchWatchlistRuns: (...args: any[]) => mocks.fetchWatchlistRunsMock(...args),
  fetchWatchlists: (...args: any[]) => mocks.fetchWatchlistsMock(...args),
  updateWatchlist: (...args: any[]) => mocks.updateWatchlistMock(...args),
  recordWatchlistsIaExperimentTelemetry: (...args: any[]) =>
    mocks.recordWatchlistsIaExperimentTelemetryMock(...args)
}))

vi.mock("@/utils/watchlists-onboarding-telemetry", () => ({
  trackWatchlistsOnboardingTelemetry: (...args: any[]) =>
    mocks.trackWatchlistsOnboardingTelemetryMock(...args)
}))

vi.mock("@/store/watchlists", () => ({
  useWatchlistsStore: (selector: (state: Record<string, unknown>) => unknown) =>
    selector({
      activeTab: mocks.state.activeTab,
      overviewHealth: mocks.state.overviewHealth,
      watchlists: [mocks.watchlistContainer],
      watchlistsLoading: false,
      watchlistsError: null,
      selectedWatchlistId: 42,
      setActiveTab: mocks.state.setActiveTab,
      setWatchlists: vi.fn(),
      setWatchlistsLoading: vi.fn(),
      setWatchlistsError: vi.fn(),
      setSelectedWatchlistId: vi.fn(),
      addWatchlist: vi.fn(),
      updateWatchlistInList: vi.fn(),
      openRunDetail: vi.fn(),
      resetStore: vi.fn()
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

describe("WatchlistsPlaygroundPage help surfaces", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    connectionMocks.useConnectionUxState.mockReturnValue({
      uxState: "connected_ok",
      hasCompletedFirstRun: true
    })
    mocks.fetchWatchlistsMock.mockResolvedValue({
      items: [mocks.watchlistContainer],
      total: 1,
      has_more: false
    })
    mocks.state.activeTab = "sources"
    layoutMocks.isConstrained = false
    layoutMocks.longCopy = false
    document.documentElement.removeAttribute("dir")
    mocks.state.overviewHealth = {
      tabBadges: {
        sources: 0,
        runs: 0,
        outputs: 0
      }
    }
    mocks.fetchWatchlistRunsMock.mockResolvedValue({ items: [], total: 0, has_more: false })
    mocks.recordWatchlistsIaExperimentTelemetryMock.mockResolvedValue({ accepted: true })
    mocks.trackWatchlistsOnboardingTelemetryMock.mockResolvedValue(undefined)
    ;(window as { __TLDW_WATCHLISTS_IA_EXPERIMENT__?: unknown }).__TLDW_WATCHLISTS_IA_EXPERIMENT__ = false
    localStorage.removeItem("beta-dismissed:watchlists")
    localStorage.removeItem("watchlists:guided-tour:v1")
    localStorage.removeItem("watchlists:teach-points:v1")
    localStorage.removeItem("watchlists:ia-experiment:v1")
    localStorage.removeItem("watchlists:show-all-views:v1")
    document.documentElement.removeAttribute("dir")
  })

  afterEach(() => {
    cleanup()
    delete (window as { __TLDW_WATCHLISTS_IA_EXPERIMENT__?: unknown }).__TLDW_WATCHLISTS_IA_EXPERIMENT__
    localStorage.removeItem("beta-dismissed:watchlists")
    localStorage.removeItem("watchlists:guided-tour:v1")
    localStorage.removeItem("watchlists:teach-points:v1")
    localStorage.removeItem("watchlists:ia-experiment:v1")
    localStorage.removeItem("watchlists:show-all-views:v1")
  })

  it("shows persistent docs links and tab-context help link", () => {
    render(<WatchlistsPlaygroundPage />)
    fireEvent.click(screen.getByTestId("watchlists-help-icon"))

    expect(screen.getByTestId("watchlists-main-docs-link")).toHaveAttribute("href", WATCHLISTS_MAIN_DOCS_URL)
    expect(screen.getByTestId("watchlists-context-docs-link")).toHaveAttribute(
      "href",
      WATCHLISTS_TAB_HELP_DOCS.sources
    )
    expect(screen.getByRole("link", { name: "Report an issue" })).toHaveAttribute(
      "href",
      WATCHLISTS_ISSUE_REPORT_URL
    )
  })

  it("keeps context docs routing aligned with each canonical tab help label", () => {
    localStorage.setItem("watchlists:show-all-views:v1", "true")
    const expectations: Array<{
      tab: typeof mocks.state.activeTab
      href: string
      label: string
    }> = [
      { tab: "overview", href: WATCHLISTS_TAB_HELP_DOCS.overview, label: "Overview guidance" },
      { tab: "sources", href: WATCHLISTS_TAB_HELP_DOCS.sources, label: "Feeds setup" },
      { tab: "jobs", href: WATCHLISTS_TAB_HELP_DOCS.jobs, label: "Monitor scheduling" },
      { tab: "runs", href: WATCHLISTS_TAB_HELP_DOCS.runs, label: "Activity guidance" },
      { tab: "items", href: WATCHLISTS_TAB_HELP_DOCS.items, label: "Updates review" },
      { tab: "outputs", href: WATCHLISTS_TAB_HELP_DOCS.outputs, label: "Reports guidance" },
      { tab: "templates", href: WATCHLISTS_TAB_HELP_DOCS.templates, label: "Template authoring" },
      { tab: "settings", href: WATCHLISTS_TAB_HELP_DOCS.settings, label: "Workspace settings" }
    ]

    const { rerender } = render(<WatchlistsPlaygroundPage />)
    fireEvent.click(screen.getByTestId("watchlists-help-icon"))
    const link = () => screen.getByTestId("watchlists-context-docs-link")

    for (const expectation of expectations) {
      mocks.state.activeTab = expectation.tab
      rerender(<WatchlistsPlaygroundPage />)
      expect(link()).toHaveAttribute("href", expectation.href)
      expect(link()).toHaveTextContent(`Learn more: ${expectation.label}`)
    }
  })

  it("renders the full tab map and quick-action labels when show-all-views is enabled", () => {
    localStorage.setItem("watchlists:show-all-views:v1", "true")
    render(<WatchlistsPlaygroundPage />)

    expect(screen.getByTestId("watchlists-tab-sources")).toHaveTextContent("Feeds")
    expect(screen.getByTestId("watchlists-tab-jobs")).toHaveTextContent("Monitors")
    expect(screen.getByTestId("watchlists-tab-runs")).toHaveTextContent("Activity")
    expect(screen.getByTestId("watchlists-tab-items")).toHaveTextContent("Updates")
    expect(screen.getByTestId("watchlists-tab-outputs")).toHaveTextContent("Reports")

    expect(screen.queryByTestId("watchlists-task-open-sources")).not.toBeInTheDocument()
    expect(screen.queryByTestId("watchlists-repeat-actions")).not.toBeInTheDocument()
  })

  it("keeps beta guidance out of the first viewport and available through Help", () => {
    render(<WatchlistsPlaygroundPage />)

    expect(screen.queryByText("Beta Feature")).not.toBeInTheDocument()
    fireEvent.click(screen.getByTestId("watchlists-help-icon"))
    expect(screen.getByTestId("watchlists-main-docs-link")).toHaveAttribute(
      "href",
      WATCHLISTS_MAIN_DOCS_URL
    )
    expect(screen.getByRole("link", { name: "Report an issue" })).toHaveAttribute(
      "href",
      WATCHLISTS_ISSUE_REPORT_URL
    )
  })

  it("supports guided-tour start and resume with persisted progress", () => {
    const { unmount } = render(<WatchlistsPlaygroundPage />)

    fireEvent.click(screen.getByTestId("watchlists-help-icon"))
    fireEvent.click(screen.getByTestId("watchlists-start-guide"))
    expect(screen.queryByRole("dialog", { name: "Watchlists help" })).not.toBeInTheDocument()
    expect(screen.getAllByRole("dialog")).toHaveLength(1)
    expect(screen.getByRole("dialog", { name: "Watchlists guided tour" })).toBeInTheDocument()
    expect(screen.getByText("Step 1 of 5")).toBeInTheDocument()
    expect(
      screen.getByText("Feeds are inputs for monitors. Add RSS/site feeds before scheduling Activity checks.")
    ).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Next" }))
    expect(screen.getByText("Step 2 of 5")).toBeInTheDocument()
    expect(
      screen.getByText("Monitors define schedule, filters, and template-driven reports, including optional audio.")
    ).toBeInTheDocument()

    const persisted = JSON.parse(localStorage.getItem("watchlists:guided-tour:v1") || "{}")
    expect(persisted.status).toBe("in_progress")
    expect(persisted.step).toBe(1)
    expect(mocks.trackWatchlistsOnboardingTelemetryMock).toHaveBeenCalledWith({
      type: "guided_tour_started"
    })
    expect(mocks.trackWatchlistsOnboardingTelemetryMock).toHaveBeenCalledWith({
      type: "guided_tour_step_viewed",
      step: 1
    })

    unmount()
    render(<WatchlistsPlaygroundPage />)

    fireEvent.click(screen.getByTestId("watchlists-help-icon"))
    expect(screen.getByTestId("watchlists-resume-guide")).toBeInTheDocument()
    fireEvent.click(screen.getByTestId("watchlists-resume-guide"))
    expect(screen.queryByRole("dialog", { name: "Watchlists help" })).not.toBeInTheDocument()
    expect(screen.getAllByRole("dialog")).toHaveLength(1)
    expect(screen.getByRole("dialog", { name: "Watchlists guided tour" })).toBeInTheDocument()
    expect(screen.getByText("Step 2 of 5")).toBeInTheDocument()
    expect(mocks.trackWatchlistsOnboardingTelemetryMock).toHaveBeenCalledWith({
      type: "guided_tour_resumed",
      step: 2
    })
  })

  it("reflows constrained RTL Help with long localized copy and reachable controls", () => {
    layoutMocks.isConstrained = true
    layoutMocks.longCopy = true
    document.documentElement.setAttribute("dir", "rtl")

    render(<WatchlistsPlaygroundPage />)
    fireEvent.click(screen.getByTestId("watchlists-help-icon"))

    const dialog = screen.getByRole("dialog", { name: "Watchlists help" })
    const panel = screen.getByTestId("watchlists-help-panel")
    expect(dialog).toHaveAttribute("data-modal-width", "100%")
    expect(dialog).toHaveClass("max-w-full")
    expect(panel).toHaveClass("min-w-0", "max-w-full", "overflow-x-hidden", "break-words")
    expect(panel).not.toHaveClass("w-[520px]", "min-w-[520px]", "whitespace-nowrap")
    expect(screen.getByTestId("watchlists-orientation-description")).toHaveClass("break-words")

    const mainDocs = screen.getByTestId("watchlists-main-docs-link")
    const contextDocs = screen.getByTestId("watchlists-context-docs-link")
    const reportIssue = screen.getByRole("link", { name: "Report an issue" })
    expect(mainDocs.compareDocumentPosition(contextDocs) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(contextDocs.compareDocumentPosition(reportIssue) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()

    const controls = panel.querySelectorAll<HTMLElement>(
      'a[href], button, [role="switch"]'
    )
    expect(controls.length).toBeGreaterThan(5)
    for (const control of controls) {
      expect(control).toBeVisible()
    }
  })

  it("restores Help trigger focus after escaping the guided tour", async () => {
    render(<WatchlistsPlaygroundPage />)

    const helpTrigger = screen.getByRole("button", { name: "Open Watchlists help" })
    helpTrigger.focus()
    fireEvent.click(helpTrigger)
    fireEvent.click(screen.getByTestId("watchlists-start-guide"))

    const tour = await screen.findByRole("dialog", { name: "Watchlists guided tour" })
    expect(screen.getAllByRole("dialog")).toHaveLength(1)
    fireEvent.keyDown(tour, { key: "Escape" })

    await waitFor(() => {
      expect(screen.queryByRole("dialog", { name: "Watchlists guided tour" })).not.toBeInTheDocument()
      expect(helpTrigger).toHaveFocus()
    })
  })

  it("shows first-time teach points for jobs/templates and persists dismissal", () => {
    localStorage.setItem("watchlists:show-all-views:v1", "true")
    mocks.state.activeTab = "jobs"
    const { rerender } = render(<WatchlistsPlaygroundPage />)
    fireEvent.click(screen.getByTestId("watchlists-help-icon"))

    expect(screen.getByText("Monitor setup tip")).toBeInTheDocument()
    expect(screen.getByText(
      "Start with schedule presets first. Use cron and advanced filters only after your first successful Activity check."
    )).toBeInTheDocument()
    fireEvent.click(
      within(screen.getByTestId("watchlists-help-panel")).getByRole("button", { name: "Dismiss" })
    )

    const persisted = JSON.parse(localStorage.getItem("watchlists:teach-points:v1") || "{}")
    expect(persisted.jobsCronFilters).toBe(true)

    rerender(<WatchlistsPlaygroundPage />)
    expect(screen.queryByText("Monitor setup tip")).not.toBeInTheDocument()

    mocks.state.activeTab = "templates"
    rerender(<WatchlistsPlaygroundPage />)
    expect(screen.getByText("Template setup tip")).toBeInTheDocument()
    fireEvent.click(
      within(screen.getByTestId("watchlists-help-panel")).getByRole("button", { name: "Dismiss" })
    )
    const nextPersisted = JSON.parse(localStorage.getItem("watchlists:teach-points:v1") || "{}")
    expect(nextPersisted.templatesAuthoring).toBe(true)
  })

  it("marks guided tour complete and shows completion notice", () => {
    render(<WatchlistsPlaygroundPage />)

    fireEvent.click(screen.getByTestId("watchlists-help-icon"))
    fireEvent.click(screen.getByTestId("watchlists-start-guide"))
    fireEvent.click(screen.getByRole("button", { name: "Next" }))
    fireEvent.click(screen.getByRole("button", { name: "Next" }))
    fireEvent.click(screen.getByRole("button", { name: "Next" }))
    fireEvent.click(screen.getByRole("button", { name: "Next" }))
    fireEvent.click(screen.getByRole("button", { name: "Finish" }))

    expect(screen.getByText("Guided tour complete")).toBeInTheDocument()
    const persisted = JSON.parse(localStorage.getItem("watchlists:guided-tour:v1") || "{}")
    expect(persisted.status).toBe("completed")
    expect(mocks.trackWatchlistsOnboardingTelemetryMock).toHaveBeenCalledWith({
      type: "guided_tour_completed"
    })
  })

  it("exposes attention badge labels and passes aria-name baseline checks", async () => {
    mocks.state.overviewHealth = {
      tabBadges: {
        sources: 3,
        runs: 2,
        outputs: 1
      }
    }
    localStorage.setItem("watchlists:show-all-views:v1", "true")

    const { container } = render(<WatchlistsPlaygroundPage />)

    expect(screen.getByLabelText("3 attention items")).toBeInTheDocument()
    expect(screen.getByLabelText("2 attention items")).toBeInTheDocument()
    expect(screen.getByLabelText("1 attention items")).toBeInTheDocument()

    const results = await runA11yBaselineRules(container)
    expectNoInvalidAriaViolations(results.violations)
    expect(results.violations.map((violation) => violation.id)).not.toContain("button-name")
    expect(results.violations.map((violation) => violation.id)).not.toContain("link-name")
  }, 15000)
})
