import React from "react"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import MediaReviewPage from "../MediaReviewPage"
import { getContentLayout } from "../card-content-density"

const sourceItems = [
  {
    id: 1,
    title: "Short note",
    snippet: "short snippet",
    type: "pdf",
    created_at: "2026-02-01T00:00:00.000Z"
  },
  {
    id: 2,
    title: "Long transcript",
    snippet: "long snippet",
    type: "audio",
    created_at: "2026-02-02T00:00:00.000Z"
  }
]

const shortContent = "brief content"
const longContent = "Long document line.\n".repeat(700)

const detailById: Record<number, { content: string; summary?: string }> = {
  1: {
    content: shortContent,
    summary: "Short analysis is available."
  },
  2: {
    content: longContent,
    summary: ""
  }
}

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

const interpolate = (template: string, values?: Record<string, unknown>) =>
  template.replace(/\{\{(\w+)\}\}/g, (_, key) => String(values?.[key] ?? ""))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?:
        | string
        | {
            defaultValue?: string
            [key: string]: unknown
          },
      maybeOptions?: Record<string, unknown>
    ) => {
      if (typeof fallbackOrOptions === "string") {
        return interpolate(fallbackOrOptions, maybeOptions)
      }
      return interpolate(
        fallbackOrOptions?.defaultValue || key,
        fallbackOrOptions as Record<string, unknown> | undefined
      )
    }
  })
}))

vi.mock("react-router-dom", () => ({
  useNavigate: () => mocks.navigate
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

vi.mock("@tanstack/react-query", () => {
  const React = require("react") as typeof import("react")
  return {
    keepPreviousData: {},
    useQuery: ({ queryFn, queryKey }: { queryFn: () => Promise<unknown>; queryKey: unknown[] }) => {
      const [data, setData] = React.useState<unknown>([])
      const [isFetching, setIsFetching] = React.useState(false)
      const queryFnRef = React.useRef(queryFn)
      queryFnRef.current = queryFn
      const queryHash = JSON.stringify(queryKey)

      const run = React.useCallback(async () => {
        setIsFetching(true)
        const result = await queryFnRef.current()
        setData(result)
        setIsFetching(false)
        return { data: result }
      }, [])

      React.useEffect(() => {
        void run()
      }, [run, queryHash])

      return { data, isFetching, refetch: run }
    }
  }
})

vi.mock("@tanstack/react-virtual", () => ({
  useVirtualizer: ({ count }: { count: number }) => ({
    getTotalSize: () => count * 120,
    getVirtualItems: () =>
      Array.from({ length: count }).map((_, index) => ({
        index,
        start: index * 120,
        size: 120,
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
        setValue((prev: unknown) => (typeof next === "function" ? (next as (prev: unknown) => unknown)(prev) : next))
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
  MEDIA_REVIEW_SELECTION_SETTING: { key: "mediaReviewSelection", defaultValue: [] },
  MEDIA_REVIEW_VIEW_MODE_SETTING: { key: "mediaReviewViewMode", defaultValue: "spread" }
}))

vi.mock("@/utils/media-detail-content", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/utils/media-detail-content")>()),
  extractMediaDetailContent: (detail: any) => detail?.content || detail?.text || ""
}))

vi.mock("@/components/Media/DiffViewModal", () => ({
  DiffViewModal: () => null
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

  const Button = ({ children, onClick, disabled, icon, iconPlacement: _iconPlacement, danger: _danger, ...rest }: any) => (
    <button type="button" onClick={onClick} disabled={disabled} {...rest}>
      {icon}
      {children}
    </button>
  )

  const Tag = ({ children }: any) => <span>{children}</span>
  const Tooltip = ({ children }: any) => <>{children}</>
  const Spin = () => <span>loading</span>
  const Pagination = () => <div className="ant-pagination">pagination</div>
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
    const resolvedOptions = options.length > 0
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
    <button
      type="button"
      aria-pressed={__groupValue === value}
      onClick={() => __groupOnChange?.({ target: { value } })}
    >
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
    <input type="checkbox" checked={checked} onChange={(event) => onChange?.(event.target.checked)} />
  )

  const Typography = {
    Text: ({ children }: any) => <span>{children}</span>
  }

  const Skeleton = () => <div>loading-skeleton</div>
  const Alert = ({ title, action }: any) => <div>{title}{action}</div>
  const Dropdown = ({ menu, children }: any) => {
    const [open, setOpen] = React.useState(false)
    const visibleItems = (menu?.items || []).filter((item: any) => item && item.type !== "divider")
    return (
      <div>
        <div
          role="button"
          tabIndex={0}
          onClick={() => setOpen((prev) => !prev)}
          onKeyDown={(event) => {
            if (event.key === "Enter" || event.key === " ") {
              event.preventDefault()
              setOpen((prev) => !prev)
            }
          }}
        >
          {children}
        </div>
        {open && (
          <div>
            {visibleItems.map((item: any, idx: number) => (
              <button
                key={`${item.key}-${idx}`}
                type="button"
                disabled={item.disabled}
                onClick={() => {
                  if (typeof menu?.onClick === "function") {
                    menu.onClick({ key: item.key })
                  } else {
                    item.onClick?.()
                  }
                }}
              >
                {typeof item.label === "string" ? item.label : String(item.key)}
              </button>
            ))}
          </div>
        )}
      </div>
    )
  }
  const Modal = () => null
  const Drawer = ({ open, title, children }: any) => (
    open ? (
      <div role="dialog" aria-label={typeof title === "string" ? title : "drawer"}>{children}</div>
    ) : null
  )

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

describe("MediaReviewPage stage4 card density improvements", () => {
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

    mocks.bgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      const path = String(request?.path || "")
      const method = String(request?.method || "GET").toUpperCase()

      const detailMatch = path.match(/\/api\/v1\/media\/(\d+)/)
      if (detailMatch) {
        const id = Number(detailMatch[1])
        return {
          id,
          title: sourceItems.find((item) => item.id === id)?.title || `Item ${id}`,
          type: sourceItems.find((item) => item.id === id)?.type || "pdf",
          content: detailById[id]?.content || "",
          summary: detailById[id]?.summary || ""
        }
      }

      if (path.startsWith("/api/v1/media/search") && method === "POST") {
        return {
          items: sourceItems,
          pagination: {
            total_items: sourceItems.length
          }
        }
      }

      if (path.startsWith("/api/v1/media/")) {
        return {
          items: sourceItems,
          pagination: {
            total_items: sourceItems.length
          }
        }
      }

      return {}
    })

    Object.defineProperty(window, "matchMedia", {
      writable: true,
      configurable: true,
      value: vi.fn().mockImplementation((query: string) => ({
        matches: false,
        media: query,
        onchange: null,
        addListener: vi.fn(),
        removeListener: vi.fn(),
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        dispatchEvent: vi.fn()
      }))
    })

    Object.defineProperty(global.navigator, "clipboard", {
      configurable: true,
      value: {
        writeText: vi.fn().mockResolvedValue(undefined)
      }
    })
  })

  const selectItems = async (...indices: number[]) => {
    const resultsList = await screen.findByTestId("media-review-results-list")
    const rows = within(resultsList).getAllByRole("button")
    indices.forEach((index) => {
      const checkbox = within(rows[index]).getByRole('checkbox')
      fireEvent.click(checkbox)
    })
  }

  it("uses adaptive content-height behavior for short vs long content", async () => {
    render(<MediaReviewPage />)
    await selectItems(0, 1)

    await waitFor(() => {
      expect(screen.getByTestId("media-review-content-body-1")).toBeInTheDocument()
      expect(screen.getByTestId("media-review-content-body-2")).toBeInTheDocument()
    })

    const shortCardContent = screen.getByTestId("media-review-content-body-1") as HTMLDivElement
    const longCardContent = screen.getByTestId("media-review-content-body-2") as HTMLDivElement

    expect(shortCardContent.style.minHeight).toBe("6em")
    expect(shortCardContent.style.maxHeight).toBe("")
    expect(longCardContent.style.minHeight).toBe("14em")
    expect(longCardContent.style.maxHeight).toBe("14em")
  })

  it("uses a unified copy menu and suppresses empty analysis by default", async () => {
    render(<MediaReviewPage />)
    await selectItems(1)

    await waitFor(() => {
      expect(screen.getByTestId("media-review-copy-menu-2")).toBeInTheDocument()
    })

    expect(screen.getByTestId("media-review-analysis-empty-2")).toBeInTheDocument()
    expect(screen.queryByText(/No analysis available/i)).not.toBeInTheDocument()

    fireEvent.click(screen.getByTestId("media-review-copy-menu-2"))
    fireEvent.click(screen.getByRole("button", { name: /copy content/i }))

    await waitFor(() => {
      expect(navigator.clipboard.writeText).toHaveBeenCalledWith(longContent)
      expect(mocks.messageSuccess).toHaveBeenCalled()
    })

    fireEvent.click(screen.getByRole("button", { name: /show panel/i }))
    expect(screen.getByTestId("media-review-analysis-panel-2")).toBeInTheDocument()
    expect(screen.getByText(/No analysis available/i)).toBeInTheDocument()
  })

  it("keeps helper thresholds aligned with card behavior", () => {
    expect(getContentLayout(100)).toEqual({ minHeightEm: 6, capped: false })
    expect(getContentLayout(700)).toEqual({ minHeightEm: 10, capped: true })
    expect(getContentLayout(8_000)).toEqual({ minHeightEm: 14, capped: true })
  })
})
