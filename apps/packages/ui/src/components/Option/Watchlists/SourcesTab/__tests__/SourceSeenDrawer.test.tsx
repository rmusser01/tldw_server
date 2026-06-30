import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { SourceSeenDrawer } from "../SourceSeenDrawer"

const mockState = vi.hoisted(() => ({
  getSourceSeenStatsMock: vi.fn(),
  clearSourceSeenMock: vi.fn(),
  messageSuccessMock: vi.fn(),
  messageErrorMock: vi.fn(),
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, defaultValue?: unknown, opts?: unknown) => {
      let text: string | undefined
      let vars: Record<string, unknown> = {}
      if (typeof defaultValue === "string") {
        text = defaultValue
        if (opts && typeof opts === "object") vars = opts as Record<string, unknown>
      } else if (
        defaultValue &&
        typeof defaultValue === "object" &&
        "defaultValue" in (defaultValue as Record<string, unknown>)
      ) {
        const dv = (defaultValue as Record<string, unknown>).defaultValue
        if (typeof dv === "string") text = dv
        vars = defaultValue as Record<string, unknown>
      }
      if (!text) return _key
      return text.replace(/\{\{(\w+)\}\}/g, (_, k) =>
        vars[k] != null ? String(vars[k]) : `{{${k}}}`
      )
    }
  })
}))

vi.mock("antd", () => {
  const Drawer = ({ open, children, title }: any) =>
    open ? (
      <div data-testid="drawer">
        <div data-testid="drawer-title">{title}</div>
        {children}
      </div>
    ) : null

  const Button = ({ children, onClick, loading, danger, ...rest }: any) => (
    <button
      type="button"
      disabled={Boolean(loading)}
      onClick={() => onClick?.()}
      data-testid={rest["data-testid"]}
      data-danger={danger ? "true" : undefined}
    >
      {children}
    </button>
  )

  const Descriptions = ({ children, title }: any) => (
    <div data-testid="descriptions">
      {title && <div data-testid="descriptions-title">{title}</div>}
      {children}
    </div>
  )
  Descriptions.Item = ({ children, label }: any) => (
    <div data-testid={`desc-item-${label}`}>
      <span>{label}</span>
      <span data-testid={`desc-value-${label}`}>{children}</span>
    </div>
  )

  const Spin = () => <div data-testid="loading-spinner" />
  const Tag = ({ children, color }: any) => (
    <span data-testid="tag" data-color={color}>
      {children}
    </span>
  )
  const Alert = ({ message: msg, title }: any) => (
    <div data-testid="alert-error">{title ?? msg}</div>
  )
  const Space = ({ children }: any) => <>{children}</>
  const InputNumber = ({ value, onChange, ...rest }: any) => (
    <input
      type="number"
      data-testid={rest["data-testid"] || "input-number"}
      value={value ?? ""}
      onChange={(e) => onChange?.(e.target.value ? Number(e.target.value) : null)}
    />
  )
  const Popconfirm = ({ children, onConfirm }: any) => {
    // Render children with click triggering onConfirm directly
    return React.cloneElement(React.Children.only(children), {
      onClick: () => onConfirm?.(),
    })
  }

  return {
    Drawer,
    Button,
    Descriptions,
    Spin,
    Tag,
    Alert,
    Space,
    InputNumber,
    Popconfirm,
    message: {
      success: mockState.messageSuccessMock,
      error: mockState.messageErrorMock,
    },
  }
})

vi.mock("@/utils/dateFormatters", () => ({
  formatRelativeTime: (date: string) => `relative(${date})`
}))

vi.mock("@/services/watchlists", () => ({
  getSourceSeenStats: (...args: any[]) => mockState.getSourceSeenStatsMock(...args),
  clearSourceSeen: (...args: any[]) => mockState.clearSourceSeenMock(...args),
}))

const sampleStats = {
  source_id: 42,
  user_id: 1,
  seen_count: 15,
  latest_seen_at: "2026-02-06T12:00:00Z",
  defer_until: "2026-02-07T12:00:00Z",
  consec_not_modified: 3,
  recent_keys: ["key-1", "key-2", "key-3"],
}

describe("SourceSeenDrawer", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockState.getSourceSeenStatsMock.mockResolvedValue(sampleStats)
    mockState.clearSourceSeenMock.mockResolvedValue({
      source_id: 42,
      user_id: 1,
      cleared: 15,
      cleared_backoff: true,
    })
  })

  it("renders nothing when closed", () => {
    render(
      <SourceSeenDrawer open={false} onClose={vi.fn()} sourceId={null} />
    )
    expect(screen.queryByTestId("drawer")).not.toBeInTheDocument()
  })

  it("shows loading spinner initially", () => {
    mockState.getSourceSeenStatsMock.mockReturnValue(new Promise(() => {})) // never resolves
    render(
      <SourceSeenDrawer open={true} onClose={vi.fn()} sourceId={42} />
    )
    expect(screen.getByTestId("loading-spinner")).toBeInTheDocument()
  })

  it("displays stats after loading", async () => {
    render(
      <SourceSeenDrawer open={true} onClose={vi.fn()} sourceId={42} />
    )
    await waitFor(() => {
      expect(screen.getByTestId("desc-value-Seen Count")).toHaveTextContent("15")
    })
    expect(screen.getByTestId("desc-value-Latest Seen")).toHaveTextContent("relative(2026-02-06T12:00:00Z)")
  })

  it("shows backoff badge", async () => {
    render(
      <SourceSeenDrawer open={true} onClose={vi.fn()} sourceId={42} />
    )
    await waitFor(() => {
      const tag = screen.getByTestId("desc-value-Backoff Status").querySelector("[data-testid='tag']")
      expect(tag).toBeInTheDocument()
      expect(tag).toHaveAttribute("data-color", "orange")
    })
  })

  it("shows high backoff for consec >= 5", async () => {
    mockState.getSourceSeenStatsMock.mockResolvedValue({
      ...sampleStats,
      consec_not_modified: 7,
    })
    render(
      <SourceSeenDrawer open={true} onClose={vi.fn()} sourceId={42} />
    )
    await waitFor(() => {
      const tag = screen.getByTestId("desc-value-Backoff Status").querySelector("[data-testid='tag']")
      expect(tag).toHaveAttribute("data-color", "red")
    })
  })

  it("shows green when no backoff", async () => {
    mockState.getSourceSeenStatsMock.mockResolvedValue({
      ...sampleStats,
      defer_until: null,
      consec_not_modified: 0,
    })
    render(
      <SourceSeenDrawer open={true} onClose={vi.fn()} sourceId={42} />
    )
    await waitFor(() => {
      const tag = screen.getByTestId("desc-value-Backoff Status").querySelector("[data-testid='tag']")
      expect(tag).toHaveAttribute("data-color", "green")
    })
  })

  it("shows recent keys list", async () => {
    render(
      <SourceSeenDrawer open={true} onClose={vi.fn()} sourceId={42} />
    )
    await waitFor(() => {
      expect(screen.getByText("key-1")).toBeInTheDocument()
      expect(screen.getByText("key-2")).toBeInTheDocument()
      expect(screen.getByText("key-3")).toBeInTheDocument()
    })
  })

  it("clear seen calls API with clear_backoff false", async () => {
    render(
      <SourceSeenDrawer open={true} onClose={vi.fn()} sourceId={42} />
    )
    await waitFor(() => {
      expect(screen.getByTestId("clear-seen-btn")).toBeInTheDocument()
    })
    fireEvent.click(screen.getByTestId("clear-seen-btn"))
    await waitFor(() => {
      expect(mockState.clearSourceSeenMock).toHaveBeenCalledWith(42, {
        clear_backoff: false,
      })
    })
  })

  it("clear all calls API with clear_backoff true", async () => {
    render(
      <SourceSeenDrawer open={true} onClose={vi.fn()} sourceId={42} />
    )
    await waitFor(() => {
      expect(screen.getByTestId("clear-all-btn")).toBeInTheDocument()
    })
    fireEvent.click(screen.getByTestId("clear-all-btn"))
    await waitFor(() => {
      expect(mockState.clearSourceSeenMock).toHaveBeenCalledWith(42, {
        clear_backoff: true,
      })
    })
  })

  it("refreshes stats after reset", async () => {
    render(
      <SourceSeenDrawer open={true} onClose={vi.fn()} sourceId={42} />
    )
    await waitFor(() => {
      expect(screen.getByTestId("clear-seen-btn")).toBeInTheDocument()
    })
    fireEvent.click(screen.getByTestId("clear-seen-btn"))
    await waitFor(() => {
      // Stats should be fetched twice: initial load + after reset
      expect(mockState.getSourceSeenStatsMock).toHaveBeenCalledTimes(2)
    })
  })

  it("shows error state", async () => {
    const consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {})
    mockState.getSourceSeenStatsMock.mockRejectedValue(new Error("Network error"))
    try {
      render(
        <SourceSeenDrawer open={true} onClose={vi.fn()} sourceId={42} />
      )
      await waitFor(() => {
        const alert = screen.getByTestId("alert-error")
        expect(alert).toBeInTheDocument()
        expect(alert).toHaveTextContent("Network error")
        expect(alert.closest("[data-ds-component='Alert']")).toBeInTheDocument()
      })
    } finally {
      consoleErrorSpy.mockRestore()
    }
  })

  it("does not show admin section for non-admin", async () => {
    render(
      <SourceSeenDrawer open={true} onClose={vi.fn()} sourceId={42} isAdmin={false} />
    )
    await waitFor(() => {
      expect(screen.getByTestId("desc-value-Seen Count")).toBeInTheDocument()
    })
    expect(screen.queryByTestId("target-user-input")).not.toBeInTheDocument()
  })

  it("shows admin section when isAdmin is true", async () => {
    render(
      <SourceSeenDrawer open={true} onClose={vi.fn()} sourceId={42} isAdmin={true} />
    )
    await waitFor(() => {
      expect(screen.getByTestId("target-user-input")).toBeInTheDocument()
    })
    expect(screen.getByTestId("load-target-btn")).toBeInTheDocument()
  })

  it("shows source name in title when provided", async () => {
    render(
      <SourceSeenDrawer
        open={true}
        onClose={vi.fn()}
        sourceId={42}
        sourceName="My Feed"
      />
    )
    await waitFor(() => {
      expect(screen.getByTestId("drawer-title")).toHaveTextContent("My Feed")
    })
  })
})
