import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import MediaReviewPage from "../MediaReviewPage"

import { isServicePromptRequestPath } from "@/services/tldw/service-prompt-scope-error"

const sourceItems = [
  {
    id: 1,
    title: "Alpha paper",
    snippet: "alpha summary",
    type: "pdf",
    created_at: "2026-01-10T00:00:00.000Z"
  },
  {
    id: 2,
    title: "Beta notes",
    snippet: "beta summary",
    type: "video",
    created_at: "2026-01-20T00:00:00.000Z"
  },
  {
    id: 3,
    title: "Gamma report",
    snippet: "gamma summary",
    type: "pdf",
    created_at: "2026-01-30T00:00:00.000Z"
  }
]

const detailById: Record<number, { content: string }> = {
  1: { content: "alpha full body" },
  2: { content: "beta full body" },
  3: { content: "gamma full body" }
}

const mocks = vi.hoisted(() => ({
  realQuery: false,
  authorityKey: "verified-alice" as string | null,
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
  navigate: vi.fn(),
  searchBodies: [] as Array<Record<string, unknown>>
}))

vi.mock('@/services/tldw/quick-ingest-authority', () => ({
  useQuickIngestAuthority: () => mocks.authorityKey,
  quickIngestAuthority: { capture: () => ({authorityKey: mocks.authorityKey, isCurrent: () => true, signal: new AbortController().signal}) }
}))

const applySearchPayload = (items: typeof sourceItems, body: Record<string, unknown>) => {
  let next = [...items]
  const mediaTypes = Array.isArray(body.media_types) ? body.media_types.map(String) : []
  if (mediaTypes.length > 0) {
    next = next.filter((item) => mediaTypes.includes(item.type))
  }

  const query = typeof body.query === "string" ? body.query.toLowerCase() : ""
  if (query) {
    next = next.filter((item) => `${item.title} ${item.snippet}`.toLowerCase().includes(query))
  }

  const sortBy = String(body.sort_by || "relevance")
  if (sortBy === "date_desc") {
    next.sort((left, right) => right.created_at.localeCompare(left.created_at))
  } else if (sortBy === "date_asc") {
    next.sort((left, right) => left.created_at.localeCompare(right.created_at))
  } else if (sortBy === "title_asc") {
    next.sort((left, right) => left.title.localeCompare(right.title))
  } else if (sortBy === "title_desc") {
    next.sort((left, right) => right.title.localeCompare(left.title))
  }

  return next
}

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?:
        | string
        | { defaultValue?: string; [key: string]: unknown },
      maybeOptions?: Record<string, unknown>
    ) => {
      if (typeof fallbackOrOptions === "string") {
        return fallbackOrOptions.replace(/\{\{(\w+)\}\}/g, (_, token) => String(maybeOptions?.[token] ?? ""))
      }
      const template = fallbackOrOptions?.defaultValue || key
      return template.replace(/\{\{(\w+)\}\}/g, (_, token) => String((fallbackOrOptions as Record<string, unknown>)?.[token] ?? ""))
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

vi.mock("@tanstack/react-query", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@tanstack/react-query")>()
  const React = require("react") as typeof import("react")
  return {
    ...actual,
    keepPreviousData: {},
    useQuery: (options: {
      queryFn: () => Promise<unknown>
      queryKey: unknown[]
      enabled?: boolean
    }) => {
      const { queryFn, queryKey } = options
      if (mocks.realQuery) return actual.useQuery(options)
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
  getNoteKeywords: vi.fn().mockResolvedValue(["alpha", "beta", "gamma"]),
  searchNoteKeywords: vi.fn().mockResolvedValue(["alpha", "alphabet", "beta alpha"])
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
  // Keep real exports (e.g. extractMediaDetailAnalysis); override only what this test controls.
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

  const Button = ({ children, onClick, disabled, icon, iconPlacement: _iconPlacement, ...rest }: any) => (
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
    const selected = isMultiple
      ? Array.isArray(value)
        ? value
        : []
      : (value ?? "")
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
  ;(SelectComponent as any).Option = ({ value, children }: any) => (
    <option value={value}>{children}</option>
  )

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
  const Alert = ({ title, action }: any) => (
    <div>
      {title}
      {action}
    </div>
  )
  const Dropdown = ({ menu, children }: any) => (
    <div>
      {children}
      <div>
        {(menu?.items || [])
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
          ))}
      </div>
    </div>
  )
  const Modal = () => null
  const Drawer = ({ open, title, children }: any) =>
    open ? (
      <div role="dialog" aria-label={typeof title === "string" ? title : "drawer"}>
        {children}
      </div>
    ) : null

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
  Markdown: ({ message }: { message: string }) => (
    <div data-testid="mock-markdown">{message}</div>
  )
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

describe("MediaReviewPage stage3 search/filter/sort", () => {
  beforeEach(() => {
    mocks.realQuery = false
    mocks.authorityKey = "verified-alice"
    mocks.bgRequest.mockReset()
    mocks.messageInfo.mockReset()
    mocks.messageWarning.mockReset()
    mocks.messageSuccess.mockReset()
    mocks.messageError.mockReset()
    mocks.getSetting.mockReset()
    mocks.setSetting.mockReset()
    mocks.clearSetting.mockReset()
    mocks.searchBodies.length = 0

    mocks.getSetting.mockResolvedValue(null)
    mocks.setSetting.mockResolvedValue(undefined)
    mocks.clearSetting.mockResolvedValue(undefined)

    mocks.bgRequest.mockImplementation(async (request: { path?: string; method?: string; body?: any }) => {
      const path = String(request?.path || "")
      const method = String(request?.method || "GET").toUpperCase()
      if (path.startsWith("/api/v1/media/search") && method === "POST") {
        const body = (request?.body || {}) as Record<string, unknown>
        mocks.searchBodies.push(body)
        const next = applySearchPayload(sourceItems, body)
        return {
          items: next,
          pagination: {
            total_items: next.length
          }
        }
      }

      const detailMatch = path.match(/\/api\/v1\/media\/(\d+)/)
      if (detailMatch) {
        const id = Number(detailMatch[1])
        await new Promise((resolve) => setTimeout(resolve, 120))
        return {
          id,
          title: sourceItems.find((item) => item.id === id)?.title || `Item ${id}`,
          type: sourceItems.find((item) => item.id === id)?.type || "pdf",
          content: detailById[id]?.content || ""
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
        matches: query === "(prefers-reduced-motion: reduce)" ? false : false,
        media: query,
        onchange: null,
        addListener: vi.fn(),
        removeListener: vi.fn(),
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        dispatchEvent: vi.fn()
      }))
    })
  })

  it.each([null, "verified-alice"])(
    "lists media from initial authority %s through the real query lifecycle without another Search",
    async (initialAuthority) => {
      mocks.realQuery = true
      mocks.authorityKey = initialAuthority
      const { QueryClient, QueryClientProvider } = await vi.importActual<
        typeof import("@tanstack/react-query")
      >("@tanstack/react-query")
      const client = new QueryClient({
        defaultOptions: { queries: { retry: false } }
      })
      const view = render(
        <React.StrictMode>
          <QueryClientProvider client={client}>
            <MediaReviewPage />
          </QueryClientProvider>
        </React.StrictMode>
      )
      mocks.authorityKey = "verified-alice"
      view.rerender(
        <React.StrictMode>
          <QueryClientProvider client={client}>
            <MediaReviewPage />
          </QueryClientProvider>
        </React.StrictMode>
      )
      await waitFor(() =>
        expect(screen.getByText("Alpha paper")).toBeInTheDocument()
      )
    }
  )

  it("loads the initial list through the existing exact scoped request policy", async () => {
    mocks.bgRequest.mockImplementation(
      async (request: { path: string; method: string }) => {
        if (!isServicePromptRequestPath(request.path, request.method))
          throw new Error("Scoped request rejected")
        return {
          items: sourceItems,
          pagination: {
            total_items: sourceItems.length
          }
        }
      }
    )
    render(<MediaReviewPage />)
    expect(await screen.findByText("Alpha paper")).toBeInTheDocument()
  })

  it("renders sort control and date range group in filter controls", async () => {
    render(<MediaReviewPage />)

    await waitFor(() => {
      expect(screen.getByRole("combobox", { name: /sort/i })).toBeInTheDocument()
      expect(screen.getByRole("group", { name: /date range/i })).toBeInTheDocument()
    })
  })

  it("sends selected sort/date filters in search payload", async () => {
    render(<MediaReviewPage />)

    await waitFor(() => {
      expect(screen.getByRole("combobox", { name: /sort/i })).toBeInTheDocument()
    })

    fireEvent.change(screen.getByRole("combobox", { name: /sort/i }), {
      target: { value: "date_desc" }
    })
    fireEvent.change(screen.getByLabelText(/start date/i), {
      target: { value: "2026-01-01" }
    })
    fireEvent.change(screen.getByLabelText(/end date/i), {
      target: { value: "2026-01-31" }
    })
    fireEvent.click(screen.getByRole("button", { name: /search/i }))

    await waitFor(() => {
      const lastPayload = mocks.searchBodies.at(-1)
      expect(lastPayload).toBeTruthy()
      expect(lastPayload).toMatchObject({
        sort_by: "date_desc",
        date_range: {
          start: "2026-01-01",
          end: "2026-01-31"
        }
      })
    })
  })

  it("shows full-content search scope copy and progress feedback", async () => {
    render(<MediaReviewPage />)

    await waitFor(() => {
      expect(screen.getByRole("checkbox", { name: /search full content/i })).toBeInTheDocument()
    })

    fireEvent.change(screen.getByLabelText(/search media/i), {
      target: { value: "alpha" }
    })
    fireEvent.click(screen.getByRole("checkbox", { name: /search full content/i }))
    fireEvent.click(screen.getAllByRole("button", { name: /search/i })[0])

    expect(screen.getByText(/scans current page results/i)).toBeInTheDocument()
    await waitFor(() => {
      expect(
        screen.getByRole("status", { name: /content filtering progress/i })
      ).toBeInTheDocument()
    })
  })

  it("shows removable filter chips and removes chip actions", async () => {
    render(<MediaReviewPage />)

    await waitFor(() => {
      expect(screen.getByRole("combobox", { name: /sort/i })).toBeInTheDocument()
    })

    fireEvent.change(screen.getByRole("combobox", { name: /sort/i }), {
      target: { value: "date_desc" }
    })
    fireEvent.change(screen.getByLabelText(/start date/i), {
      target: { value: "2026-01-01" }
    })

    // In three-panel layout, filter chips are always visible in the left panel
    const sortChipRemove = await screen.findByRole("button", {
      name: /remove filter sort:/i
    })
    fireEvent.click(sortChipRemove)

    await waitFor(() => {
      expect(
        screen.queryByRole("button", { name: /remove filter sort:/i })
      ).not.toBeInTheDocument()
    })
  })
})
