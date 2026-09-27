import React from "react"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import MediaReviewPage from "../MediaReviewPage"

vi.mock("@/hooks/useMediaCapabilities", () => ({
  useMediaCapabilities: () => ({ canDelete: true, loading: false })
}))

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

const detailById: Record<number, { content: string; summary?: string }> = {
  1: { content: "alpha full body", summary: "alpha analysis" },
  2: { content: "beta full body", summary: "beta analysis" },
  3: { content: "gamma full body", summary: "gamma analysis" }
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
  navigate: vi.fn(),
  downloadBlob: vi.fn()
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
      const template = fallbackOrOptions?.defaultValue || key
      return interpolate(
        template,
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

vi.mock("@/utils/download-blob", () => ({
  downloadBlob: (...args: unknown[]) => mocks.downloadBlob(...args)
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
  searchNoteKeywords: vi.fn().mockResolvedValue(["alpha", "beta"])
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
  MEDIA_REVIEW_SELECTION_SETTING: { key: "mediaReviewSelection", defaultValue: [] },
  MEDIA_REVIEW_VIEW_MODE_SETTING: { key: "mediaReviewViewMode", defaultValue: "spread" }
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

  const Button = ({ children, onClick, disabled, icon, ...rest }: any) => (
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
        data-testid={rest["data-testid"]}
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
    </div>
  )
  const Modal = () => null
  ;(Modal as any).confirm = ({ onOk }: { onOk?: () => void }) => {
    onOk?.()
  }
  const Drawer = ({ open, title, children }: any) =>
    open ? <div role="dialog" aria-label={typeof title === "string" ? title : "drawer"}>{children}</div> : null

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

describe("MediaReviewPage stage5 export and trash handoff", () => {
  beforeEach(() => {
    mocks.bgRequest.mockReset()
    mocks.messageInfo.mockReset()
    mocks.messageWarning.mockReset()
    mocks.messageSuccess.mockReset()
    mocks.messageError.mockReset()
    mocks.getSetting.mockReset()
    mocks.setSetting.mockReset()
    mocks.clearSetting.mockReset()
    mocks.downloadBlob.mockReset()
    mocks.navigate.mockReset()

    mocks.getSetting.mockResolvedValue(null)
    mocks.setSetting.mockResolvedValue(undefined)
    mocks.clearSetting.mockResolvedValue(undefined)

    mocks.bgRequest.mockImplementation(async (request: { path?: string; method?: string; body?: any }) => {
      const path = String(request?.path || "")
      if (request?.method === "POST" && path.startsWith("/api/v1/media/search")) {
        return {
          items: sourceItems,
          pagination: { total_items: sourceItems.length }
        }
      }
      if (request?.method === "GET" && path.startsWith("/api/v1/media/?")) {
        return {
          items: sourceItems,
          pagination: { total_items: sourceItems.length }
        }
      }
      const detailMatch = path.match(/\/api\/v1\/media\/(\d+)\?include_content=true/)
      if (request?.method === "GET" && detailMatch) {
        const id = Number(detailMatch[1])
        return {
          id,
          title: sourceItems.find((item) => item.id === id)?.title || `Item ${id}`,
          type: sourceItems.find((item) => item.id === id)?.type || "pdf",
          ...(detailById[id] || { content: `content-${id}` })
        }
      }
      if (request?.method === "DELETE" && path.startsWith("/api/v1/media/")) {
        return undefined
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
  })

  const getResultRowByTitle = (title: string): HTMLElement => {
    const list = screen.getByTestId("media-review-results-list")
    const label = within(list).getByText(title)
    const row = label.closest('[role="button"]')
    if (!row) throw new Error(`Missing row for ${title}`)
    return row as HTMLElement
  }

  const selectItemByCheckbox = (title: string, options?: Record<string, unknown>): void => {
    const row = getResultRowByTitle(title)
    const checkbox = within(row).getByRole('checkbox')
    fireEvent.click(checkbox, options)
  }

  it("exports selected items in json/markdown/text formats", async () => {
    render(<MediaReviewPage />)

    await waitFor(() => {
      expect(screen.getByText("0 / 30 selected")).toBeInTheDocument()
    })

    selectItemByCheckbox("Alpha paper")
    selectItemByCheckbox("Beta notes")

    const formatSelect = screen.getByRole("combobox", { name: /export format/i })
    const exportButton = screen.getByRole("button", { name: /export selected/i })

    fireEvent.change(formatSelect, { target: { value: "json" } })
    fireEvent.click(exportButton)
    fireEvent.change(formatSelect, { target: { value: "markdown" } })
    fireEvent.click(exportButton)
    fireEvent.change(formatSelect, { target: { value: "text" } })
    fireEvent.click(exportButton)

    await waitFor(() => {
      expect(mocks.downloadBlob).toHaveBeenCalledTimes(3)
    })

    const [, jsonName] = mocks.downloadBlob.mock.calls[0] as [Blob, string]
    const [, markdownName] = mocks.downloadBlob.mock.calls[1] as [Blob, string]
    const [textBlob, textName] = mocks.downloadBlob.mock.calls[2] as [Blob, string]

    expect(jsonName.endsWith(".json")).toBe(true)
    expect(markdownName.endsWith(".md")).toBe(true)
    expect(textName.endsWith(".txt")).toBe(true)
    await expect(textBlob.text()).resolves.toContain("Alpha paper")
  })

  it("navigates to trash view after successful batch delete when user chooses review trash", async () => {
    render(<MediaReviewPage />)

    await waitFor(() => {
      expect(screen.getByText("0 / 30 selected")).toBeInTheDocument()
    })

    selectItemByCheckbox("Alpha paper")
    selectItemByCheckbox("Beta notes")

    fireEvent.click(screen.getByRole("button", { name: /move to trash/i }))

    await waitFor(() => {
      expect(mocks.bgRequest).toHaveBeenCalledWith(
        expect.objectContaining({
          path: "/api/v1/media/1",
          method: "DELETE"
        })
      )
      expect(mocks.bgRequest).toHaveBeenCalledWith(
        expect.objectContaining({
          path: "/api/v1/media/2",
          method: "DELETE"
        })
      )
    })

    const handoff = await screen.findByTestId("media-multi-trash-handoff")
    fireEvent.click(
      within(handoff).getByRole("button", { name: /open trash/i })
    )

    expect(mocks.navigate).toHaveBeenCalledWith(
      expect.stringContaining("/media-trash?")
    )
    expect(mocks.navigate).toHaveBeenCalledWith(
      expect.stringContaining("from=media-multi")
    )
  })
})
