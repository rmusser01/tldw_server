import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { SourcesBulkImport } from "../SourcesBulkImport"

const mocks = vi.hoisted(() => ({
  fetchWatchlistSourcesMock: vi.fn(),
  importOpmlMock: vi.fn(),
  messageSuccessMock: vi.fn(),
  messageErrorMock: vi.fn(),
  messageWarningMock: vi.fn(),
  tMock: vi.fn()
}))

const interpolate = (template: string, vars: Record<string, unknown>) =>
  template.replace(/\{\{\s*([^}]+)\s*\}\}/g, (_match, key) => {
    const value = vars[key.trim()]
    return value == null ? "" : String(value)
  })

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: mocks.tMock
  })
}))

vi.mock("antd", () => {
  const Button = ({ children, onClick, disabled, loading, ...rest }: any) => (
    <button
      type="button"
      {...rest}
      disabled={Boolean(disabled) || Boolean(loading)}
      onClick={() => onClick?.()}
    >
      {children}
    </button>
  )

  const Modal = ({ open, title, footer, children, width, styles, ...rest }: any) =>
    open ? (
      <div
        data-testid={rest["data-testid"]}
        data-width={String(width ?? "")}
        data-body-max-height={String(styles?.body?.maxHeight ?? "")}
      >
        <h2>{title}</h2>
        {children}
        <div>{footer}</div>
      </div>
    ) : null

  const Select = ({ value, onChange, options = [], allowClear, mode }: any) => (
    <select
      multiple={mode === "multiple"}
      value={value ?? (mode === "multiple" ? [] : "")}
      onChange={(event) => {
        if (mode === "multiple") {
          const selected = Array.from(event.currentTarget.selectedOptions).map((option) => option.value)
          onChange?.(selected)
          return
        }
        const next = event.currentTarget.value
        onChange?.(next === "" && allowClear ? undefined : next)
      }}
    >
      {allowClear ? <option value="" /> : null}
      {options.map((option: any) => (
        <option key={String(option.value)} value={String(option.value)}>
          {String(option.label)}
        </option>
      ))}
    </select>
  )

  const Switch = ({ checked, onChange }: any) => (
    <input
      type="checkbox"
      checked={Boolean(checked)}
      onChange={(event) => onChange?.(event.currentTarget.checked)}
    />
  )

  const Upload = {
    Dragger: ({ beforeUpload, disabled, children }: any) => (
      <div>
        <input
          data-testid="opml-upload-input"
          type="file"
          disabled={Boolean(disabled)}
          onChange={async (event) => {
            const file = event.currentTarget.files?.[0]
            if (!file) return
            await beforeUpload?.({
              originFileObj: file,
              name: file.name
            })
          }}
        />
        {children}
      </div>
    )
  }

  const Table = ({ dataSource = [], ...rest }: any) => (
    <div data-testid={rest["data-testid"] || "preflight-table-rows"}>{dataSource.length}</div>
  )

  const Tag = ({ children }: any) => <span>{children}</span>
  const Space = ({ children }: any) => <div>{children}</div>
  const Tooltip = ({ children, title }: any) => (
    <div>
      {children}
      {title}
    </div>
  )
  const Alert = ({ title, message, description }: any) => (
    <div>
      <div>{title ?? message}</div>
      <div>{description}</div>
    </div>
  )

  return {
    Alert,
    Button,
    Modal,
    Select,
    Space,
    Switch,
    Table,
    Tag,
    Tooltip,
    Upload,
    message: {
      success: mocks.messageSuccessMock,
      error: mocks.messageErrorMock,
      warning: mocks.messageWarningMock
    }
  }
})

vi.mock("@/services/watchlists", () => ({
  fetchWatchlistSources: (...args: any[]) => mocks.fetchWatchlistSourcesMock(...args),
  importOpml: (...args: any[]) => mocks.importOpmlMock(...args)
}))

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

describe("SourcesBulkImport preflight and commit", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    setViewport(1024)
    mocks.tMock.mockImplementation(
      (_key: string, defaultValue?: unknown, options?: Record<string, unknown>) => {
        if (typeof defaultValue === "string") {
          return options ? interpolate(defaultValue, options) : defaultValue
        }
        if (defaultValue && typeof defaultValue === "object") {
          const maybeDefault = (defaultValue as { defaultValue?: unknown }).defaultValue
          if (typeof maybeDefault === "string") {
            return interpolate(maybeDefault, (defaultValue as Record<string, unknown>) || {})
          }
        }
        return _key
      }
    )
    mocks.fetchWatchlistSourcesMock.mockResolvedValue({
      items: [
        {
          id: 11,
          name: "Existing Feed",
          url: "https://existing.example.com/rss.xml",
          source_type: "rss",
          active: true,
          tags: []
        }
      ],
      total: 1,
      page: 1,
      size: 200,
      has_more: false
    })
    mocks.importOpmlMock.mockResolvedValue({
      items: [{ status: "created", url: "https://new.example.com/rss.xml" }]
    })
  })

  it("shows preflight summary and commits import only for ready entries", async () => {
    const onImported = vi.fn()
    render(
      <SourcesBulkImport
        open
        onClose={vi.fn()}
        groups={[]}
        tags={[]}
        defaultGroupId={null}
        onImported={onImported}
      />
    )

    expect(screen.getByTestId("watchlists-help-opml")).toBeInTheDocument()

    const opml = `<?xml version="1.0"?>
<opml version="2.0">
  <body>
    <outline text="Existing" xmlUrl="https://existing.example.com/rss.xml"/>
    <outline text="Fresh Feed" xmlUrl="https://new.example.com/rss.xml"/>
  </body>
</opml>`
    const file = new File([opml], "feeds.opml", { type: "text/xml" })

    const uploadInput = screen.getByTestId("opml-upload-input")
    fireEvent.change(uploadInput, { target: { files: [file] } })

    await waitFor(() => {
      const preflightTitle = screen.getByText("Preflight Summary")
      expect(preflightTitle).toBeInTheDocument()
      expect(preflightTitle.closest('[data-ds-component="Alert"]')).toBeInTheDocument()
      expect(screen.getByText(/1 ready, 1 existing duplicates/i)).toBeInTheDocument()
      expect(screen.getByTestId("preflight-table-rows")).toHaveTextContent("2")
    })

    const importButton = screen.getByRole("button", { name: /Import 1 feed/i })
    expect(importButton).toBeEnabled()
    fireEvent.click(importButton)

    await waitFor(() => {
      expect(mocks.importOpmlMock).toHaveBeenCalledWith(
        file,
        expect.objectContaining({
          active: true,
          tags: [],
          group_id: undefined
        })
      )
      expect(onImported).toHaveBeenCalledTimes(1)
      expect(mocks.messageSuccessMock).toHaveBeenCalledWith("OPML imported")
      expect(screen.getByText("Import Summary").closest('[data-ds-component="Alert"]')).toBeInTheDocument()
    })
  })

  it("keeps commit disabled when OPML parse fails", async () => {
    render(
      <SourcesBulkImport
        open
        onClose={vi.fn()}
        groups={[]}
        tags={[]}
        defaultGroupId={null}
        onImported={vi.fn()}
      />
    )

    const invalidOpml = "<not-opml />"
    const file = new File([invalidOpml], "invalid.opml", { type: "text/xml" })

    fireEvent.change(screen.getByTestId("opml-upload-input"), { target: { files: [file] } })

    const importButton = await screen.findByRole("button", { name: /Import 0 feeds/i })
    expect(importButton).toBeDisabled()
    fireEvent.click(importButton)

    await waitFor(() => {
      expect(mocks.importOpmlMock).not.toHaveBeenCalled()
    })
  })

  it("retries only failed import entries and preserves recovery scope", async () => {
    mocks.importOpmlMock
      .mockResolvedValueOnce({
        items: [
          { status: "created", url: "https://good.example.com/rss.xml" },
          {
            status: "error",
            url: "https://failed.example.com/rss.xml",
            error: "timeout while fetching"
          }
        ]
      })
      .mockResolvedValueOnce({
        items: [{ status: "created", url: "https://failed.example.com/rss.xml" }]
      })

    render(
      <SourcesBulkImport
        open
        onClose={vi.fn()}
        groups={[]}
        tags={[]}
        defaultGroupId={null}
        onImported={vi.fn()}
      />
    )

    const opml = `<?xml version="1.0"?>
<opml version="2.0">
  <body>
    <outline text="Good Feed" xmlUrl="https://good.example.com/rss.xml"/>
    <outline text="Failed Feed" xmlUrl="https://failed.example.com/rss.xml"/>
  </body>
</opml>`
    const file = new File([opml], "retry-feeds.opml", { type: "text/xml" })
    fireEvent.change(screen.getByTestId("opml-upload-input"), { target: { files: [file] } })

    const importButton = await screen.findByRole("button", { name: /Import 2 feeds/i })
    fireEvent.click(importButton)

    await waitFor(() => {
      expect(mocks.importOpmlMock).toHaveBeenCalledTimes(1)
      expect(screen.getByRole("button", { name: "Retry failed only" })).toBeEnabled()
    })

    fireEvent.click(screen.getByRole("button", { name: "Retry failed only" }))

    await waitFor(() => {
      expect(mocks.importOpmlMock).toHaveBeenCalledTimes(2)
      expect(mocks.messageSuccessMock).toHaveBeenCalledWith("Retried 1 failed feed.")
    })

    const retryFile = mocks.importOpmlMock.mock.calls[1]?.[0] as File
    const retryOpmlText = await retryFile.text()
    expect(retryOpmlText).toContain("https://failed.example.com/rss.xml")
    expect(retryOpmlText).not.toContain("https://good.example.com/rss.xml")
  })

  it("disables retry-failed when failures are non-retryable duplicates", async () => {
    mocks.importOpmlMock.mockResolvedValueOnce({
      items: [
        {
          status: "error",
          url: "https://dupe.example.com/rss.xml",
          error: "duplicate source already exists"
        }
      ]
    })

    render(
      <SourcesBulkImport
        open
        onClose={vi.fn()}
        groups={[]}
        tags={[]}
        defaultGroupId={null}
        onImported={vi.fn()}
      />
    )

    const opml = `<?xml version="1.0"?>
<opml version="2.0">
  <body>
    <outline text="Dup Feed" xmlUrl="https://dupe.example.com/rss.xml"/>
  </body>
</opml>`
    const file = new File([opml], "duplicate.opml", { type: "text/xml" })
    fireEvent.change(screen.getByTestId("opml-upload-input"), { target: { files: [file] } })

    fireEvent.click(await screen.findByRole("button", { name: /Import 1 feed/i }))

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Retry failed only" })).toBeDisabled()
    })
  })

  it("uses constrained cards for OPML preflight and keeps footer actions reachable", async () => {
    setViewport(420)

    render(
      <SourcesBulkImport
        open
        onClose={vi.fn()}
        groups={[]}
        tags={[]}
        defaultGroupId={null}
        onImported={vi.fn()}
      />
    )

    const modal = screen.getByTestId("sources-bulk-import-modal")
    expect(modal).toHaveAttribute("data-width", "100vw")
    expect(modal.getAttribute("data-body-max-height")).toContain("calc(100vh")

    const opml = `<?xml version="1.0"?>
<opml version="2.0">
  <body>
    <outline text="Fresh Feed" xmlUrl="https://new.example.com/rss.xml"/>
  </body>
</opml>`
    const file = new File([opml], "feeds.opml", { type: "text/xml" })

    fireEvent.change(screen.getByTestId("opml-upload-input"), { target: { files: [file] } })

    await waitFor(() => {
      expect(screen.getByTestId("opml-preflight-constrained-list")).toBeInTheDocument()
      expect(screen.queryByTestId("preflight-table-rows")).not.toBeInTheDocument()
      expect(screen.getByRole("button", { name: /Import 1 feed/i })).toBeEnabled()
      expect(screen.getByRole("button", { name: "Close" })).toBeInTheDocument()
    })
  })
})
