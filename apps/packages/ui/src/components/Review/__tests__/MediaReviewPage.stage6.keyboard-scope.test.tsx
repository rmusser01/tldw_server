import React from "react"
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import MediaReviewPage from "../MediaReviewPage"

const mediaItems = Array.from({ length: 8 }).map((_, idx) => ({
  id: idx + 1,
  title: `Item ${idx + 1}`,
  snippet: `Snippet ${idx + 1}`,
  type: "pdf",
  created_at: "2026-02-20T00:00:00.000Z"
}))

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn(),
  messageInfo: vi.fn(),
  messageWarning: vi.fn(),
  messageSuccess: vi.fn(),
  messageError: vi.fn(),
  getSetting: vi.fn(),
  setSetting: vi.fn(),
  clearSetting: vi.fn(),
  setHelpDismissed: vi.fn(),
  setChatMode: vi.fn(),
  setSelectedKnowledge: vi.fn(),
  setRagMediaIds: vi.fn(),
  navigate: vi.fn()
}))

vi.mock('@/services/tldw/quick-ingest-authority', () => ({
  useQuickIngestAuthority: () => 'verified-alice',
  quickIngestAuthority: { capture: () => ({ isCurrent: () => true, signal: new AbortController().signal }) }
}))

const interpolate = (template: string, values?: Record<string, unknown>) =>
  template.replace(/\{\{(\w+)\}\}/g, (_, key) => String(values?.[key] ?? ""))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?: string | { defaultValue?: string; [key: string]: unknown },
      maybeOptions?: Record<string, unknown>
    ) => {
      if (typeof fallbackOrOptions === "string") {
        return interpolate(fallbackOrOptions, maybeOptions)
      }
      const template = fallbackOrOptions?.defaultValue || key
      return interpolate(template, fallbackOrOptions as Record<string, unknown> | undefined)
    }
  })
}))

vi.mock("react-router-dom", () => ({
  useNavigate: () => mocks.navigate,
  useLocation: () => ({ key: 'initial' })
}))

vi.mock("@/hooks/useMessageOption", () => ({
  useMessageOption: () => ({
    setChatMode: mocks.setChatMode,
    setSelectedKnowledge: mocks.setSelectedKnowledge,
    setRagMediaIds: mocks.setRagMediaIds
  })
}))

vi.mock("@/hooks/useAntdMessage", () => ({
  useAntdMessage: () => ({
    info: mocks.messageInfo,
    warning: mocks.messageWarning,
    success: mocks.messageSuccess,
    error: mocks.messageError
  })
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: mocks.bgRequest
}))

vi.mock("@tanstack/react-query", () => ({
  keepPreviousData: {},
  useQuery: () => ({
    data: mediaItems,
    isFetching: false,
    refetch: vi.fn()
  })
}))

vi.mock("@tanstack/react-virtual", () => ({
  useVirtualizer: ({ count }: { count: number }) => ({
    getTotalSize: () => count * 90,
    getVirtualItems: () =>
      Array.from({ length: count }).map((_, index) => ({
        index,
        start: index * 90,
        size: 90,
        key: index
      })),
    scrollToIndex: vi.fn(),
    measureElement: vi.fn()
  })
}))

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => true
}))

vi.mock("@/services/note-keywords", () => ({
  getNoteKeywords: vi.fn().mockResolvedValue([]),
  searchNoteKeywords: vi.fn().mockResolvedValue([])
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: () => [true, mocks.setHelpDismissed, { isLoading: false }]
}))

vi.mock("@/hooks/useSetting", () => {
  const React = require("react") as typeof import("react")
  return {
    useSetting: (setting: { defaultValue: unknown }) => {
      const [value, setValue] = React.useState(setting.defaultValue)
      const setter = async (next: unknown | ((prev: unknown) => unknown)) => {
        setValue((prev: unknown) =>
          typeof next === "function" ? (next as (prev: unknown) => unknown)(prev) : next
        )
      }
      return [value, setter, { isLoading: false }] as const
    }
  }
})

vi.mock("@/services/settings/registry", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/services/settings/registry")>()
  return {
    ...actual,
    getSetting: mocks.getSetting,
    setSetting: mocks.setSetting,
    clearSetting: mocks.clearSetting
  }
})

vi.mock("@/services/settings/ui-settings", () => ({
  DISCUSS_MEDIA_PROMPT_SETTING: { key: "discussMediaPrompt", defaultValue: null },
  LAST_MEDIA_ID_SETTING: { key: "lastMediaId", defaultValue: null },
  MEDIA_HIDE_TRANSCRIPT_TIMINGS_SETTING: { key: "mediaHideTranscriptTimings", defaultValue: true },
  MEDIA_REVIEW_AUTO_VIEW_MODE_SETTING: { key: "mediaReviewAutoViewMode", defaultValue: true },
  MEDIA_REVIEW_FILTERS_COLLAPSED_SETTING: { key: "mediaReviewFiltersCollapsed", defaultValue: false },
  MEDIA_REVIEW_FOCUSED_ID_SETTING: { key: "mediaReviewFocusedId", defaultValue: null },
  MEDIA_REVIEW_ORIENTATION_SETTING: { key: "mediaReviewOrientation", defaultValue: "vertical" },
  MEDIA_REVIEW_SELECTION_SNAPSHOT_SETTING: { key: 'media-review-selection-snapshot', defaultValue: null },
  MEDIA_REVIEW_SELECTION_SETTING: { key: "mediaReviewSelection", defaultValue: [] },
  MEDIA_REVIEW_VIEW_MODE_SETTING: { key: "mediaReviewViewMode", defaultValue: "spread" }
}))

vi.mock("@/utils/media-detail-content", async (importOriginal) => ({
  // Keep the module's real exports so a new helper (e.g. extractMediaDetailAnalysis)
  // does not break this test; override only the content extraction it controls.
  ...(await importOriginal<typeof import("@/utils/media-detail-content")>()),
  extractMediaDetailContent: (detail: any) => detail?.content || detail?.text || ""
}))

vi.mock("@/components/Media/DiffViewModal", () => ({
  DiffViewModal: ({ open }: { open: boolean }) =>
    open ? <div role="dialog" aria-label="Diff View">Diff</div> : null
}))

vi.mock("antd", async (importOriginal) => {
  const React = await import("react")
  const actual = await importOriginal<typeof import("antd")>()

  const Input = React.forwardRef<HTMLInputElement, any>(
    ({ value, onChange, onPressEnter, placeholder, ...rest }, ref) => (
      <input
        ref={ref}
        value={value || ""}
        onChange={onChange}
        placeholder={placeholder}
        onKeyDown={(event) => {
          if (event.key === "Enter") onPressEnter?.(event)
        }}
        {...rest}
      />
    )
  )

  const Button = ({ children, onClick, disabled, icon, danger: _danger, iconPlacement: _iconPlacement, ...rest }: any) => (
    <button type="button" onClick={onClick} disabled={disabled} {...rest}>
      {icon}
      {children}
    </button>
  )

  const Tag = ({ children }: any) => <span>{children}</span>
  const Tooltip = ({ children }: any) => <>{children}</>
  const Spin = () => <span>loading</span>
  const Pagination = () => <div>pagination</div>
  const Empty = ({ description }: any) => <div>{description}</div>
  ;(Empty as any).PRESENTED_IMAGE_SIMPLE = null

  const Checkbox = ({ checked, onChange, children, ...rest }: any) => (
    <label>
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange?.({ target: { checked: event.target.checked } })}
        {...rest}
      />
      {children}
    </label>
  )

  const SelectComponent = ({ value, onChange, options = [], children, mode, ...rest }: any) => {
    const resolvedOptions =
      options.length > 0
        ? options
        : React.Children.toArray(children).map((child: any) => ({
            value: child?.props?.value,
            label: child?.props?.children
          }))
    const isMultiple = mode === "multiple" || mode === "tags"
    const selected = isMultiple ? (Array.isArray(value) ? value : []) : (value ?? "")
    return (
      <select
        multiple={isMultiple}
        value={selected as any}
        onChange={(event) => {
          if (isMultiple) {
            const values = Array.from(event.currentTarget.selectedOptions).map((opt) => opt.value)
            onChange?.(values)
          } else {
            onChange?.(event.currentTarget.value)
          }
        }}
        aria-label={rest["aria-label"]}
      >
        {resolvedOptions.map((opt: any) => (
          <option key={String(opt.value)} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
    )
  }
  ;(SelectComponent as any).Option = ({ value, children }: any) => <option value={value}>{children}</option>

  const RadioButton = ({ value, children, __groupValue, __groupOnChange }: any) => (
    <button type="button" aria-pressed={__groupValue === value} onClick={() => __groupOnChange?.({ target: { value } })}>
      {children}
    </button>
  )

  const RadioGroup = ({ value, onChange, children }: any) => (
    <div>
      {React.Children.map(children, (child: any) =>
        React.cloneElement(child, {
          __groupValue: value,
          __groupOnChange: onChange
        })
      )}
    </div>
  )

  const Switch = ({ checked, onChange }: any) => (
    <input type="checkbox" checked={checked} onChange={(e) => onChange?.(e.target.checked)} />
  )

  const Typography = {
    Text: ({ children }: any) => <span>{children}</span>
  }

  const Skeleton = () => <div>loading-skeleton</div>
  const Alert = ({ title, action }: any) => (
    <div>
      {title}
      {action}
    </div>
  )

  const Dropdown = ({ menu, children }: any) => (
    <div data-testid="mock-dropdown-root">
      {children}
      <div>
        {Array.isArray(menu?.items)
          ? menu.items
              .flatMap((item: any) => item?.children ?? [item])
              .filter((item: any) => item && item.type !== "divider")
              .map((item: any, idx: number) => (
                <button
                  key={`${item.key}-${idx}`}
                  type="button"
                  disabled={item.disabled}
                  onClick={() => item.onClick?.()}
                >
                  {typeof item.label === "string" ? item.label : String(item.key)}
                </button>
              ))
          : null}
      </div>
    </div>
  )

  const Modal = ({ open, title, onCancel, footer, children }: any) => {
    if (!open) return null
    return (
      <div role="dialog" aria-label={typeof title === "string" ? title : "Modal"}>
        <button type="button" onClick={onCancel}>Close</button>
        {children}
        {footer}
      </div>
    )
  }

  const Drawer = ({ open, title, onClose, children }: any) => {
    if (!open) return null
    return (
      <div role="dialog" aria-label={typeof title === "string" ? title : "Drawer"}>
        <button type="button" onClick={onClose}>Close drawer</button>
        {children}
      </div>
    )
  }

  return {
    ...actual,
    Input,
    Button,
    Spin,
    Tag,
    Tooltip,
    Radio: { Group: RadioGroup, Button: RadioButton },
    Pagination,
    Empty,
    Select: SelectComponent,
    Checkbox,
    Typography,
    Skeleton,
    Switch,
    Alert,
    Collapse: ({ children }: any) => <div>{children}</div>,
    Dropdown,
    Modal,
    Drawer
  }
})

vi.mock("@/components/Common/Markdown", () => ({
  Markdown: ({ message }: { message: string }) => <div data-testid="mock-markdown">{message}</div>
}))

vi.mock("@/components/Media/diff-worker-client", () => ({
  computeDiffSync: () => [],
  shouldUseWorkerDiff: () => false,
  shouldRequireSampling: () => false,
  sampleTextForDiff: (t: string) => t,
  computeDiffWithWorker: async () => [],
  createDiffWorker: () => null,
  DIFF_SYNC_LINE_THRESHOLD: 4000,
  DIFF_HARD_CHAR_THRESHOLD: 300_000,
  DIFF_SAMPLED_CHAR_BUDGET: 120_000
}))

describe("MediaReviewPage stage6 keyboard shortcut scope", () => {
  beforeEach(() => {
    mocks.bgRequest.mockReset()
    mocks.messageInfo.mockReset()
    mocks.messageWarning.mockReset()
    mocks.messageSuccess.mockReset()
    mocks.messageError.mockReset()
    mocks.getSetting.mockReset()
    mocks.setSetting.mockReset()
    mocks.clearSetting.mockReset()

    mocks.getSetting.mockResolvedValue(null)
    mocks.setSetting.mockResolvedValue(undefined)
    mocks.clearSetting.mockResolvedValue(undefined)

    mocks.bgRequest.mockImplementation(async (request: { path?: string }) => {
      const path = String(request?.path || "")
      const idMatch = path.match(/\/api\/v1\/media\/([^?]+)/)
      const id = idMatch ? Number(idMatch[1]) : null
      return {
        id: id ?? 1,
        title: `Item ${id ?? 1}`,
        type: "pdf",
        content: `Content ${id ?? 1}`
      }
    })

    Object.defineProperty(window, "matchMedia", {
      writable: true,
      configurable: true,
      value: vi.fn().mockImplementation(() => ({
        matches: false,
        media: "",
        onchange: null,
        addListener: vi.fn(),
        removeListener: vi.fn(),
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        dispatchEvent: vi.fn()
      }))
    })
  })

  const getResultRowByTitle = (title: string): HTMLElement => {
    const list = screen.getByTestId("media-review-results-list")
    const text = within(list).getByText(title)
    const row = text.closest('[role="button"]')
    if (!row) throw new Error(`Missing row for title: ${title}`)
    return row as HTMLElement
  }

  const selectItemByCheckbox = (title: string, options?: Record<string, unknown>): void => {
    const row = getResultRowByTitle(title)
    const checkbox = within(row).getByRole('checkbox')
    fireEvent.click(checkbox, options)
  }

  it("does not trigger global j/k navigation when options menu is open", async () => {
    render(<MediaReviewPage />)

    await waitFor(() => {
      expect(screen.getByTestId('media-review-selection-count')).toHaveTextContent('0 selected')
    })

    mocks.bgRequest.mockClear()
    const optionsButton = screen.getByRole("button", { name: /options/i })
    fireEvent.click(optionsButton)

    fireEvent.keyDown(optionsButton, { key: "j" })

    await waitFor(() => {
      expect(mocks.bgRequest).not.toHaveBeenCalled()
    })
  })

  it("does not clear selection when inline comparison is active", async () => {
    render(<MediaReviewPage />)

    await waitFor(() => {
      expect(screen.getByTestId('media-review-selection-count')).toHaveTextContent('0 selected')
    })

    selectItemByCheckbox("Item 1")
    selectItemByCheckbox("Item 2")

    await waitFor(() => {
      expect(screen.getByTestId('media-review-selection-count')).toHaveTextContent('2 selected')
      expect(screen.getByRole("button", { name: /compare content/i })).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole("button", { name: /compare content/i }))

    await waitFor(() => {
      expect(screen.getByTestId("comparison-split")).toBeInTheDocument()
    })

    // Selection should be preserved while comparing
    expect(screen.getByTestId('media-review-selection-count')).toHaveTextContent('2 selected')
  })

  it("does not hijack Space on focused action buttons", async () => {
    render(<MediaReviewPage />)

    await waitFor(() => {
      expect(screen.getByTestId('media-review-selection-count')).toHaveTextContent('0 selected')
    })

    selectItemByCheckbox("Item 1")

    await waitFor(() => {
      expect(screen.getByTestId('media-review-selection-count')).toHaveTextContent('1 selected')
    })

    const clearButton = within(screen.getByTestId("panel-center")).getByRole("button", {
      name: /^clear$/i
    })
    clearButton.focus()

    fireEvent.keyDown(clearButton, { key: " " })

    await waitFor(() => {
      expect(screen.getByTestId('media-review-selection-count')).toHaveTextContent('1 selected')
    })
  })

  it("keeps j/k navigation active after previewing a result row", async () => {
    render(<MediaReviewPage />)

    await waitFor(() => {
      expect(screen.getByTestId('media-review-selection-count')).toHaveTextContent('0 selected')
    })

    const row = getResultRowByTitle("Item 1")
    row.focus()
    expect(row).toHaveFocus()

    fireEvent.click(row)

    await waitFor(() => {
      expect(screen.getByText("Item 1 of 8")).toBeInTheDocument()
    })

    fireEvent.keyDown(document.activeElement ?? row, { key: "j" })

    await waitFor(() => {
      expect(screen.getByText("Item 2 of 8")).toBeInTheDocument()
    })
  })

  it("restores viewer focus after the browser re-focuses the clicked result row", async () => {
    render(<MediaReviewPage />)

    await waitFor(() => {
      expect(screen.getByTestId('media-review-selection-count')).toHaveTextContent('0 selected')
    })

    const row = getResultRowByTitle("Item 1")
    row.focus()
    expect(row).toHaveFocus()

    vi.useFakeTimers()
    try {
      fireEvent.click(row)
      expect(screen.getByText("Item 1 of 8")).toBeInTheDocument()

      // Real browsers can leave focus on the clicked row after the click handler runs.
      row.focus()
      expect(row).toHaveFocus()

      act(() => {
        vi.runOnlyPendingTimers()
      })

      expect(document.activeElement).not.toBe(row)

      fireEvent.keyDown(document.activeElement ?? document.body, { key: "j" })
      expect(screen.getByText("Item 2 of 8")).toBeInTheDocument()
    } finally {
      vi.useRealTimers()
    }
  })

  it("does not let late focused-id restore overwrite keyboard navigation", async () => {
    let resolveFocusedRestore: ((value: unknown) => void) | null = null
    mocks.getSetting.mockImplementation((setting: { key?: string }) => {
      switch (setting?.key) {
        case "lastMediaId":
          return Promise.resolve(null)
        case "mediaReviewSelection":
          return Promise.resolve([])
        case "mediaReviewFocusedId":
          return new Promise((resolve) => {
            resolveFocusedRestore = resolve
          })
        default:
          return Promise.resolve(null)
      }
    })

    render(<MediaReviewPage />)

    await waitFor(() => {
      expect(screen.getByTestId('media-review-selection-count')).toHaveTextContent('0 selected')
    })

    const row = getResultRowByTitle("Item 1")
    fireEvent.click(row)

    await waitFor(() => {
      expect(screen.getByText("Item 1 of 8")).toBeInTheDocument()
    })

    fireEvent.keyDown(document.activeElement ?? document.body, { key: "j" })

    await waitFor(() => {
      expect(screen.getByText("Item 2 of 8")).toBeInTheDocument()
      expect(screen.getByText("Previewing 2 of 8")).toBeInTheDocument()
    })

    await act(async () => {
      resolveFocusedRestore?.(mediaItems[0].id)
      await Promise.resolve()
    })

    await waitFor(() => {
      expect(screen.getByText("Item 2 of 8")).toBeInTheDocument()
      expect(screen.getByText("Previewing 2 of 8")).toBeInTheDocument()
    })

    fireEvent.keyDown(document.activeElement ?? document.body, { key: "j" })

    await waitFor(() => {
      expect(screen.getByText("Item 3 of 8")).toBeInTheDocument()
      expect(screen.getByText("Previewing 3 of 8")).toBeInTheDocument()
    })
  })
})
