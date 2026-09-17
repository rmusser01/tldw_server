import React from "react"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { MemoryRouter } from "react-router-dom"

import ViewMediaPage from "../ViewMediaPage"
import { useConnectionStore } from "@/store/connection"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { bgRequest } from "@/services/background-proxy"

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
  capsLoading: false,
  capabilities: { hasMedia: true },
  navigate: vi.fn(),
  refetch: vi.fn(),
  checkOnce: vi.fn()
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
            [k: string]: unknown
          }
    ) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      const template = fallbackOrOptions?.defaultValue || key
      return interpolate(template, fallbackOrOptions as Record<string, unknown> | undefined)
    }
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

vi.mock("@plasmohq/storage", () => ({
  Storage: class {
    async get() {
      return null
    }
    async set() {
      return undefined
    }
    async remove() {
      return undefined
    }
  }
}))

vi.mock("@plasmohq/storage/hook", async () => {
  const React = await import("react")
  return {
    useStorage: (_key: string, initialValue: unknown) => {
      const [value, setValue] = React.useState(initialValue)
      return [value, setValue] as const
    }
  }
})

vi.mock("@/services/background-proxy", () => ({
  bgRequest: vi.fn().mockResolvedValue({})
}))

vi.mock("@/services/settings/registry", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/services/settings/registry")>()
  return {
    ...actual,
    getSetting: vi.fn(),
    setSetting: vi.fn(),
    clearSetting: vi.fn()
  }
})

vi.mock("@/hooks/useDebounce", () => ({
  useDebounce: (value: string) => value
}))

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => mocks.isOnline
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({
    capabilities: mocks.capabilities,
    loading: mocks.capsLoading
  })
}))

vi.mock("@/context/demo-mode", () => ({
  useDemoMode: () => ({ demoEnabled: mocks.demoEnabled })
}))

vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionState: () => ({ serverUrl: "http://localhost:8000" }),
  useConnectionUxState: () => ({
    uxState: mocks.uxState,
    hasCompletedFirstRun: mocks.hasCompletedFirstRun
  }),
  useConnectionActions: () => ({
    checkOnce: mocks.checkOnce
  })
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

vi.mock("@/hooks/useAntdMessage", () => ({
  useAntdMessage: () => ({
    success: vi.fn(),
    error: vi.fn(),
    warning: vi.fn()
  })
}))

vi.mock("@/hooks/useUndoNotification", () => ({
  useUndoNotification: () => ({
    showUndoNotification: vi.fn()
  })
}))

vi.mock("@/hooks/useFeatureFlags", () => ({
  useMediaNavigationPanel: () => [false],
  useMediaNavigationGeneratedFallbackDefault: () => [false],
  useMediaRichRendering: () => [false],
  useMediaAnalysisDisplayModeSelector: () => [false]
}))

vi.mock("@/hooks/useMediaNavigation", () => ({
  useMediaNavigation: () => ({
    data: { nodes: [] },
    isLoading: false,
    error: null,
    refetch: vi.fn()
  })
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getConfig: vi.fn().mockResolvedValue({})
  }
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

vi.mock("@/components/Media/SearchBar", () => ({
  SearchBar: () => <div />
}))

vi.mock("@/components/Media/FilterPanel", () => ({
  FilterPanel: () => <div />
}))

vi.mock("@/components/Media/JumpToNavigator", () => ({
  JumpToNavigator: () => <div />
}))

vi.mock("@/components/Media/KeyboardShortcutsOverlay", () => ({
  KeyboardShortcutsOverlay: () => null
}))

vi.mock("@/components/Media/FilterChips", () => ({
  FilterChips: () => <div />
}))

vi.mock("@/components/Media/Pagination", () => ({
  Pagination: () => <div />
}))

vi.mock("@/components/Media/MediaSectionNavigator", () => ({
  MediaSectionNavigator: () => <div />
}))

vi.mock("@/components/Media/ResultsList", () => ({
  ResultsList: ({ results }: { results: Array<{ title: string }> }) => <div>{results.map(item => item.title).join(",")}</div>
}))

vi.mock("@/components/Media/ContentViewer", () => ({
  ContentViewer: () => <div />
}))

vi.mock("@/components/Media/MediaIngestJobsPanel", () => ({
  MediaIngestJobsPanel: () => <div />
}))

vi.mock("@/components/Media/MediaLibraryStatsPanel", () => ({
  MediaLibraryStatsPanel: () => <div />
}))

const renderPage = () =>
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <MemoryRouter><ViewMediaPage /></MemoryRouter>
    </QueryClientProvider>
  )

describe("ViewMediaPage connection states", () => {
  beforeEach(() => {
    mocks.isOnline = true
    mocks.demoEnabled = false
    mocks.uxState = "connected_ok"
    mocks.hasCompletedFirstRun = true
    mocks.capsLoading = false
    mocks.capabilities = { hasMedia: true }
    mocks.navigate.mockReset()
    mocks.refetch.mockReset()
    mocks.checkOnce.mockReset()
  })

  it("does not start protected Media reads during initial testing before the credential gate", async () => {
    mocks.isOnline = false
    mocks.uxState = "testing"
    vi.mocked(bgRequest).mockClear()
    const view = renderPage()
    await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent("Checking connection"))
    expect(bgRequest).not.toHaveBeenCalled()
    mocks.uxState = "configuring_auth"
    view.rerender(<QueryClientProvider client={new QueryClient()}><MemoryRouter><ViewMediaPage /></MemoryRouter></QueryClientProvider>)
    expect(screen.getByText("Add your credentials to use Media")).toBeInTheDocument()
    expect(bgRequest).not.toHaveBeenCalled()
  })

  it("remounts private catalogue state after an observed batched authority replacement", async () => {
    useConnectionStore.setState(({ state }) => ({ state: { ...state, isConnected: true } }))
    vi.mocked(bgRequest).mockResolvedValue({ items: [{ id: 1, title: 'Alice source' }], pagination: { total_items: 1 } })
    renderPage()
    await screen.findByText('Alice source')
    vi.mocked(bgRequest).mockResolvedValue({ items: [{ id: 2, title: 'Bob source' }], pagination: { total_items: 1 } })
    act(() => {
      useConnectionStore.setState(({ state }) => ({ state: { ...state, isConnected: false } }))
      useConnectionStore.setState(({ state }) => ({ state: { ...state, isConnected: true } }))
    })
    expect(await screen.findByText('Bob source')).toBeInTheDocument()
    expect(screen.queryByText('Alice source')).not.toBeInTheDocument()
  })

  it("shows credential guidance and opens settings when auth is missing", () => {
    mocks.isOnline = false
    mocks.uxState = "error_auth"

    renderPage()

    expect(screen.getByText("Add your credentials to use Media")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Open Settings" }))
    expect(mocks.navigate).toHaveBeenCalledWith("/settings/tldw")
  })

  it("shows setup guidance and routes first-run users to setup", () => {
    mocks.isOnline = false
    mocks.uxState = "unconfigured"
    mocks.hasCompletedFirstRun = false

    renderPage()

    expect(screen.getByText("Finish setup to use Media")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Finish Setup" }))
    expect(mocks.navigate).toHaveBeenCalledWith("/")
  })

  it("shows unreachable guidance with retry and diagnostics actions", () => {
    mocks.isOnline = false
    mocks.uxState = "error_unreachable"

    renderPage()

    expect(
      screen.getByText("Can't reach your tldw server right now")
    ).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Retry connection" }))
    expect(mocks.checkOnce).toHaveBeenCalled()

    fireEvent.click(screen.getByRole("button", { name: "Health & diagnostics" }))
    expect(mocks.navigate).toHaveBeenCalledWith("/settings/health")
  })
})
