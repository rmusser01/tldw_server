// @vitest-environment jsdom

import React from "react"
import { render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { JobPreviewModal } from "../JobPreviewModal"

const mocks = vi.hoisted(() => ({
  previewWatchlistJobMock: vi.fn()
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

vi.mock("@/services/watchlists", () => ({
  previewWatchlistJob: (...args: unknown[]) => mocks.previewWatchlistJobMock(...args)
}))

vi.mock("antd", () => {
  const Modal = ({ open, title, children, onCancel, width, styles, ...rest }: any) => {
    const closeRef = React.useRef<HTMLButtonElement | null>(null)

    React.useEffect(() => {
      if (open) {
        closeRef.current?.focus()
      }
    }, [open])

    if (!open) return null

    return (
      <div
        data-testid={rest["data-testid"]}
        data-width={String(width ?? "")}
        data-body-max-height={String(styles?.body?.maxHeight ?? "")}
      >
        <h2>{title}</h2>
        <button type="button" ref={closeRef} onClick={() => onCancel?.()}>
          Close
        </button>
        {children}
      </div>
    )
  }

  const Spin = () => <div>Loading...</div>
  const Tag = ({ children }: any) => <span>{children}</span>
  const Table = ({ dataSource = [] }: any) => (
    <div data-testid="job-preview-table">
      {dataSource.map((item: any, index: number) => (
        <div key={`${item.url || "item"}-${index}`}>{item.title || item.url || "-"}</div>
      ))}
    </div>
  )

  return {
    Modal,
    Spin,
    Tag,
    Table
  }
})

const buildJob = () => ({
  id: 11,
  name: "Morning monitor",
  description: null,
  scope: { sources: [1] },
  schedule_expr: "0 9 * * *",
  timezone: "UTC",
  active: true,
  output_prefs: {},
  created_at: "2026-02-18T00:00:00Z"
})

const matchesViewportQuery = (query: string, width: number): boolean => {
  const maxWidth = query.match(/max-width:\s*(\d+)px/)
  const minWidth = query.match(/min-width:\s*(\d+)px/)
  if (maxWidth && width > Number(maxWidth[1])) return false
  if (minWidth && width < Number(minWidth[1])) return false
  return Boolean(maxWidth || minWidth)
}

const setViewport = (width: number) => {
  Object.defineProperty(window, "innerWidth", {
    configurable: true,
    value: width
  })
  Object.defineProperty(window, "matchMedia", {
    configurable: true,
    writable: true,
    value: vi.fn().mockImplementation((query: string) => ({
      matches: matchesViewportQuery(query, width),
      media: query,
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn()
    }))
  })
}

describe("JobPreviewModal focus restoration", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    setViewport(1024)
    mocks.previewWatchlistJobMock.mockResolvedValue({
      items: [],
      total: 0,
      ingestable: 0,
      filtered: 0
    })
  })

  it("restores focus to the launch control when preview closes", async () => {
    const trigger = document.createElement("button")
    trigger.type = "button"
    trigger.textContent = "Open preview"
    document.body.appendChild(trigger)
    trigger.focus()

    const { rerender } = render(
      <JobPreviewModal
        job={buildJob() as any}
        open
        onClose={vi.fn()}
      />
    )

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Close" })).toHaveFocus()
    })

    rerender(
      <JobPreviewModal
        job={buildJob() as any}
        open={false}
        onClose={vi.fn()}
      />
    )

    await waitFor(() => {
      expect(trigger).toHaveFocus()
    })

    trigger.remove()
  })

  it("renders preview candidates as constrained cards in a full-width dialog", async () => {
    setViewport(420)
    mocks.previewWatchlistJobMock.mockResolvedValue({
      items: [
        {
          title: "CVE briefing",
          url: "https://example.com/cve",
          source_id: 5,
          decision: "ingest",
          matched_action: "alert",
          matched_filter_key: "cve",
          matched_filter_type: "keyword",
          matched_filter_id: 9
        }
      ],
      total: 1,
      ingestable: 1,
      filtered: 0
    })

    render(
      <JobPreviewModal
        job={buildJob() as any}
        open
        onClose={vi.fn()}
      />
    )

    await waitFor(() => {
      const modal = screen.getByTestId("job-preview-modal")
      expect(modal).toHaveAttribute("data-width", "100vw")
      expect(modal.getAttribute("data-body-max-height")).toContain("calc(100vh")
      expect(screen.getByTestId("job-preview-constrained-list")).toBeInTheDocument()
      expect(screen.queryByTestId("job-preview-table")).not.toBeInTheDocument()
      expect(screen.getByText("CVE briefing")).toBeInTheDocument()
    })
  })
})
