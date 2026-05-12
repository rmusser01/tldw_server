import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { MemoryRouter } from "react-router-dom"

import { ReviewPage } from "../ReviewPage"

const mocks = vi.hoisted(() => ({
  isOnline: true,
  demoEnabled: false,
  uxState: "connected_ok" as
    | "connected_ok"
    | "testing"
    | "configuring_url"
    | "configuring_auth"
    | "error_auth"
    | "error_unreachable"
    | "unconfigured",
  hasCompletedFirstRun: true,
  navigate: vi.fn(),
  checkOnce: vi.fn(),
  queryData: [] as Array<Record<string, unknown>>,
  promptSearch: {
    query: "",
    setQuery: vi.fn(),
    results: [],
    loading: false,
    includeLocal: true,
    setIncludeLocal: vi.fn(),
    includeServer: true,
    setIncludeServer: vi.fn(),
    search: vi.fn(),
    clearResults: vi.fn()
  }
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallbackOrOptions?: string | { defaultValue?: string }) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      return fallbackOrOptions?.defaultValue ?? _key
    }
  })
}))

vi.mock("@/design-system", () => ({
  getDesignSystemState: (key: string) => ({
    key,
    label: key === "ready" ? "Registry Ready" : key,
    severity: "neutral"
  })
}))

vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual<typeof import("react-router-dom")>(
    "react-router-dom"
  )
  return {
    ...actual,
    useNavigate: () => mocks.navigate
  }
})

vi.mock("@tanstack/react-query", () => ({
  useQuery: () => ({
    data: mocks.queryData,
    isFetching: false,
    refetch: vi.fn()
  })
}))

vi.mock("antd", () => {
  const Button = ({
    children,
    onClick
  }: {
    children?: React.ReactNode
    onClick?: () => void
  }) => (
    <button type="button" onClick={onClick}>
      {children}
    </button>
  )
  const InputBase = ({
    allowClear: _allowClear,
    onPressEnter: _onPressEnter,
    ...props
  }: React.InputHTMLAttributes<HTMLInputElement> & {
    allowClear?: boolean
    onPressEnter?: () => void
  }) => <input {...props} />

  return {
    Button,
    Checkbox: () => <input type="checkbox" />,
    Divider: () => <hr />,
    Empty: ({ description }: { description?: React.ReactNode }) => <div>{description}</div>,
    Input: Object.assign(InputBase, {
      TextArea: (
        props: React.TextareaHTMLAttributes<HTMLTextAreaElement>
      ) => <textarea {...props} />
    }),
    List: Object.assign(
      ({
        children,
        dataSource = [],
        renderItem
      }: {
        children?: React.ReactNode
        dataSource?: unknown[]
        renderItem?: (item: unknown, index: number) => React.ReactNode
      }) => (
        <div>
          {children}
          {dataSource.map((item, index) => (
            <React.Fragment key={index}>
              {renderItem?.(item, index)}
            </React.Fragment>
          ))}
        </div>
      ),
      {
        Item: ({ children }: { children?: React.ReactNode }) => (
          <div>{children}</div>
        )
      }
    ),
    Space: ({ children }: { children?: React.ReactNode }) => <div>{children}</div>,
    Spin: () => <div>loading</div>,
    Tag: ({ children }: { children?: React.ReactNode }) => <span>{children}</span>,
    Tooltip: ({ children }: { children?: React.ReactNode }) => <div>{children}</div>,
    Typography: {
      Text: ({ children }: { children?: React.ReactNode }) => <span>{children}</span>,
      Title: ({ children }: { children?: React.ReactNode }) => <div>{children}</div>,
      Paragraph: ({ children }: { children?: React.ReactNode }) => <div>{children}</div>
    },
    Select: ({ children }: { children?: React.ReactNode }) => <select>{children}</select>,
    Pagination: () => <div />,
    Radio: {
      Button: ({ children }: { children?: React.ReactNode }) => (
        <button type="button">{children}</button>
      ),
      Group: ({ children }: { children?: React.ReactNode }) => <div>{children}</div>
    },
    Modal: ({ children }: { children?: React.ReactNode }) => <div>{children}</div>,
    Dropdown: ({ children }: { children?: React.ReactNode }) => <div>{children}</div>
  }
})

vi.mock("@/services/background-proxy", () => ({
  bgRequest: vi.fn().mockResolvedValue({ items: [], pagination: { total_pages: 1 } })
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getConfig: vi.fn().mockResolvedValue({})
  }
}))

vi.mock("@/services/note-keywords", () => ({
  getNoteKeywords: vi.fn().mockResolvedValue([]),
  searchNoteKeywords: vi.fn().mockResolvedValue([])
}))

vi.mock("@/hooks/useMessageOption", () => ({
  useMessageOption: () => ({
    selectedModel: "test-model",
    messages: [],
    setMessages: vi.fn(),
    setChatMode: vi.fn(),
    setSelectedKnowledge: vi.fn(),
    setRagMediaIds: vi.fn()
  })
}))

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => mocks.isOnline
}))

vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionUxState: () => ({
    uxState: mocks.uxState,
    hasCompletedFirstRun: mocks.hasCompletedFirstRun
  }),
  useConnectionActions: () => ({
    checkOnce: mocks.checkOnce
  })
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({
    success: vi.fn(),
    error: vi.fn(),
    warning: vi.fn()
  })
}))

vi.mock("@/hooks/useAntdMessage", () => ({
  useAntdMessage: () => ({
    success: vi.fn(),
    error: vi.fn(),
    warning: vi.fn()
  })
}))

vi.mock("@/context/demo-mode", () => ({
  useDemoMode: () => ({ demoEnabled: mocks.demoEnabled })
}))

vi.mock("@/hooks/useScrollToServerCard", () => ({
  useScrollToServerCard: () => vi.fn()
}))

vi.mock("@/components/Common/confirm-danger", () => ({
  useConfirmDanger: () => vi.fn().mockResolvedValue(true)
}))

vi.mock("@/components/Common/MarkdownErrorBoundary", () => ({
  MarkdownErrorBoundary: ({ children }: { children?: React.ReactNode }) => (
    <div>{children}</div>
  )
}))

vi.mock("@/components/Review/PromptDropdown", () => ({
  PromptDropdown: () => <div />
}))

vi.mock("@/components/Review/usePromptSearch", () => ({
  usePromptSearch: () => mocks.promptSearch
}))

vi.mock("@/utils/demo-content", () => ({
  getDemoMediaItems: () => [
    {
      title: "Demo review item",
      meta: "Demo metadata",
      statusKey: "ready",
      statusLabel: "Ready"
    }
  ]
}))

vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({
    get: vi.fn().mockResolvedValue(null),
    set: vi.fn().mockResolvedValue(undefined),
    watch: vi.fn(),
    unwatch: vi.fn()
  })
}))

vi.mock("@/components/Common/FeatureEmptyState", () => ({
  default: ({
    title,
    description,
    primaryActionLabel,
    onPrimaryAction,
    secondaryActionLabel,
    onSecondaryAction
  }: {
    title: React.ReactNode
    description?: React.ReactNode
    primaryActionLabel?: React.ReactNode
    onPrimaryAction?: () => void
    secondaryActionLabel?: React.ReactNode
    onSecondaryAction?: () => void
  }) => (
    <div data-testid="feature-empty-state">
      <div>{title}</div>
      {description ? <div>{description}</div> : null}
      {primaryActionLabel ? (
        <button type="button" onClick={onPrimaryAction}>
          {primaryActionLabel}
        </button>
      ) : null}
      {secondaryActionLabel ? (
        <button type="button" onClick={onSecondaryAction}>
          {secondaryActionLabel}
        </button>
      ) : null}
    </div>
  )
}))

vi.mock("@/components/Common/ConnectionProblemBanner", () => ({
  default: ({
    title,
    description,
    primaryActionLabel,
    onPrimaryAction,
    retryActionLabel,
    onRetry,
    secondaryActionLabel,
    onSecondaryAction
  }: {
    title: React.ReactNode
    description?: React.ReactNode
    primaryActionLabel?: React.ReactNode
    onPrimaryAction?: () => void
    retryActionLabel?: React.ReactNode
    onRetry?: () => void
    secondaryActionLabel?: React.ReactNode
    onSecondaryAction?: () => void
  }) => (
    <div data-testid="connection-problem-banner">
      <div>{title}</div>
      {description ? <div>{description}</div> : null}
      {primaryActionLabel ? (
        <button type="button" onClick={onPrimaryAction}>
          {primaryActionLabel}
        </button>
      ) : null}
      {retryActionLabel ? (
        <button type="button" onClick={onRetry}>
          {retryActionLabel}
        </button>
      ) : null}
      {secondaryActionLabel ? (
        <button type="button" onClick={onSecondaryAction}>
          {secondaryActionLabel}
        </button>
      ) : null}
    </div>
  )
}))

const renderPage = (props?: React.ComponentProps<typeof ReviewPage>) =>
  render(
    <MemoryRouter>
      <ReviewPage {...props} />
    </MemoryRouter>
  )

describe("ReviewPage connection states", () => {
  beforeEach(() => {
    mocks.isOnline = true
    mocks.demoEnabled = false
    mocks.uxState = "connected_ok"
    mocks.hasCompletedFirstRun = true
    mocks.navigate.mockReset()
    mocks.checkOnce.mockReset()
    mocks.queryData = []
    mocks.promptSearch.setQuery.mockReset()
    mocks.promptSearch.setIncludeLocal.mockReset()
    mocks.promptSearch.setIncludeServer.mockReset()
    mocks.promptSearch.search.mockReset()
    mocks.promptSearch.clearResults.mockReset()
  })

  it("keeps the demo preview visible while showing auth recovery guidance", () => {
    mocks.isOnline = false
    mocks.demoEnabled = true
    mocks.uxState = "error_auth"

    renderPage()

    expect(screen.getByText("Add your credentials to use Review")).toBeInTheDocument()
    expect(screen.getByText("Demo review item")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Open Settings" }))
    expect(mocks.navigate).toHaveBeenCalledWith("/settings/tldw")
  })

  it("shows setup guidance when demo mode is disabled", () => {
    mocks.isOnline = false
    mocks.demoEnabled = false
    mocks.uxState = "unconfigured"
    mocks.hasCompletedFirstRun = false

    renderPage()

    expect(screen.getByText("Finish setup to use Review")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Finish Setup" }))
    expect(mocks.navigate).toHaveBeenCalledWith("/")
  })

  it("preserves the generic offline path when forceOffline is set", () => {
    mocks.isOnline = true
    mocks.demoEnabled = false
    mocks.uxState = "error_auth"

    renderPage({ forceOffline: true })

    expect(screen.getByText("Connect to use Review")).toBeInTheDocument()
    expect(
      screen.queryByText("Add your credentials to use Review")
    ).not.toBeInTheDocument()
  })

  it("uses the design-system registry fallback for ready status labels", () => {
    mocks.queryData = [
      {
        kind: "media",
        id: "ready-media",
        title: "Ready media",
        meta: { status: "ready" },
        raw: {}
      }
    ]

    renderPage()

    expect(screen.getByText("Registry Ready")).toBeInTheDocument()
  })
})
