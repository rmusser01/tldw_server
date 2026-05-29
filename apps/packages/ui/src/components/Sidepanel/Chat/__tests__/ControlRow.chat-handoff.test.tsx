import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { ControlRow } from "../ControlRow"

type MockSelectProps = {
  children?: React.ReactNode
  value?: string | string[] | null
  onChange?: (value: string) => void
  "aria-label"?: string
}

type MockChildrenProps = {
  children?: React.ReactNode
}

type MockPopoverProps = MockChildrenProps & {
  content?: React.ReactNode
}

type MockSwitchProps = {
  checked?: boolean
  disabled?: boolean
  onChange?: (checked: boolean) => void
  "aria-label"?: string
}

type MockInputProps = React.InputHTMLAttributes<HTMLInputElement> & {
  onPressEnter?: () => void
}

const createDeferred = <T,>() => {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((promiseResolve, promiseReject) => {
    resolve = promiseResolve
    reject = promiseReject
  })
  return { promise, resolve, reject }
}

const mocks = vi.hoisted(() => ({
  selectedAssistant: null as null | {
    kind: "character"
    id: string
    name: string
  },
  setSelectedAssistant: vi.fn(),
  setSelectedCharacterId: vi.fn(),
  setMoodBadge: vi.fn(),
  setStorage: vi.fn(),
  fetchChatModels: vi.fn(async () => []),
  runtimeGetURL: vi.fn((path: string) => `chrome-extension://handoff${path}`),
  tabsCreate: vi.fn(),
  createSidepanelChatHandoff: vi.fn(),
  buildSidepanelChatHandoffRoute: vi.fn(
    (basePath: string, handoffId: string) => {
      const separator = basePath.includes("?") ? "&" : "?"
      return `${basePath}${separator}handoff=${handoffId}`
    }
  ),
  setToolCatalog: vi.fn(),
  setToolCatalogId: vi.fn(),
  setToolModules: vi.fn(),
  setToolCatalogStrict: vi.fn(),
  setMcpToolEnabled: vi.fn(),
  resetMcpToolFilter: vi.fn()
}))

vi.mock("wxt/browser", () => ({
  browser: {
    runtime: {
      id: "handoff-extension",
      getURL: mocks.runtimeGetURL
    },
    tabs: {
      create: mocks.tabsCreate
    }
  }
}))

vi.mock("@/services/sidepanel-chat-handoff", () => ({
  createSidepanelChatHandoff: mocks.createSidepanelChatHandoff,
  buildSidepanelChatHandoffRoute: mocks.buildSidepanelChatHandoffRoute
}))

vi.mock("antd", () => {
  const Select = ({ children, ...props }: MockSelectProps) => (
    <select
      aria-label={props["aria-label"]}
      value={Array.isArray(props.value) ? props.value[0] ?? "" : props.value ?? ""}
      onChange={(event) => props.onChange?.(event.target.value)}
    >
      {children}
    </select>
  )
  Select.Option = ({ children, value }: MockChildrenProps & { value?: string }) => (
    <option value={value}>{children}</option>
  )
  Select.OptGroup = ({
    children,
    label
  }: MockChildrenProps & { label?: string }) => (
    <optgroup label={label}>{children}</optgroup>
  )

  const Radio = {
    Group: ({ children }: MockChildrenProps) => <div>{children}</div>,
    Button: ({ children, value }: MockChildrenProps & { value?: string }) => (
      <button type="button" data-value={value}>
        {children}
      </button>
    )
  }

  return {
    Input: ({ onPressEnter: _onPressEnter, ...props }: MockInputProps) => (
      <input {...props} />
    ),
    InputNumber: (props: React.InputHTMLAttributes<HTMLInputElement>) => (
      <input type="number" {...props} />
    ),
    Popover: ({ children, content }: MockPopoverProps) => (
      <>
        {children}
        <div data-testid="mock-popover-content">{content}</div>
      </>
    ),
    Radio,
    Select,
    Switch: ({
      checked,
      disabled,
      onChange,
      ...props
    }: MockSwitchProps) => (
      <button
        type="button"
        role="switch"
        aria-checked={Boolean(checked)}
        aria-label={props["aria-label"]}
        disabled={disabled}
        onClick={() => onChange?.(!checked)}
      />
    ),
    Tooltip: ({ children }: MockChildrenProps) => <>{children}</>,
    Upload: ({ children }: MockChildrenProps) => <>{children}</>
  }
})

vi.mock("@tanstack/react-query", () => ({
  useQuery: () => ({ data: [] })
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string, options?: Record<string, unknown>) =>
      String(fallback ?? _key).replace(/\{\{(\w+)\}\}/g, (_, key) =>
        String(options?.[key] ?? "")
      )
  })
}))

vi.mock("@/components/Common/ModelSelect", () => ({
  ModelSelect: () => <button type="button">Model</button>
}))

vi.mock("@/components/Common/PromptSelect", () => ({
  PromptSelect: () => <button type="button">Prompt</button>
}))

vi.mock("@/components/Common/FeatureHint", () => ({
  FeatureHint: ({ children }: { children?: React.ReactNode }) => <>{children}</>,
  useFeatureHintSeen: () => false
}))

vi.mock("@/components/Common/McpToolSelector", () => ({
  McpToolSelector: () => <div data-testid="mcp-tool-selector" />
}))

vi.mock("../ConversationContextPopover", () => ({
  ConversationContextPopover: () => <button type="button">Context</button>
}))

vi.mock("@/hooks/useChatMoodBadgePreference", () => ({
  useChatMoodBadgePreference: () => [false, mocks.setMoodBadge] as const
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({
    capabilities: {
      hasMcp: false,
      hasMediaPlaylistPreflight: false,
      hasWebSearch: true
    }
  })
}))

vi.mock("@/hooks/useMcpTools", () => ({
  useMcpTools: () => ({
    hasMcp: false,
    healthState: "unavailable",
    discoveredTools: [],
    chatTools: [],
    toolCounts: { enabled: 0, total: 0 },
    toolsLoading: false,
    catalogs: [],
    catalogsLoading: false,
    toolCatalog: "",
    toolCatalogId: null,
    toolModules: [],
    moduleOptions: [],
    moduleOptionsLoading: false,
    toolCatalogStrict: false,
    setToolCatalog: mocks.setToolCatalog,
    setToolCatalogId: mocks.setToolCatalogId,
    setToolModules: mocks.setToolModules,
    setToolCatalogStrict: mocks.setToolCatalogStrict,
    setToolEnabled: mocks.setMcpToolEnabled,
    resetToolFilter: mocks.resetMcpToolFilter
  })
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: unknown, initialValue: unknown) =>
    [initialValue ?? null, mocks.setStorage] as const
}))

vi.mock("@/services/tldw-server", () => ({
  fetchChatModels: mocks.fetchChatModels
}))

vi.mock("@/utils/quick-ingest-open", () => ({
  buildQuickIngestOpenDetailFromUrl: vi.fn(),
  requestQuickIngestOpen: vi.fn()
}))

vi.mock("@/hooks/useSelectedAssistant", () => ({
  useSelectedAssistant: () =>
    [mocks.selectedAssistant, mocks.setSelectedAssistant] as const
}))

const defaultProps = () => ({
  selectedSystemPrompt: undefined,
  setSelectedSystemPrompt: vi.fn(),
  setSelectedQuickPrompt: vi.fn(),
  selectedCharacterId: null,
  setSelectedCharacterId: mocks.setSelectedCharacterId,
  webSearch: false,
  setWebSearch: vi.fn(),
  chatMode: "normal" as const,
  setChatMode: vi.fn(),
  toolChoice: "auto" as const,
  setToolChoice: vi.fn(),
  onImageUpload: vi.fn(),
  onToggleRag: vi.fn(),
  isConnected: true
})

describe("ControlRow sidepanel chat handoff", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.selectedAssistant = null
    mocks.createSidepanelChatHandoff.mockResolvedValue({
      id: "handoff-123",
      source: "sidepanel-chat",
      createdAt: "2026-05-29T00:00:00.000Z",
      expiresAt: "2026-05-29T00:10:00.000Z",
      draft: { text: "Summarize this" }
    })
    vi.spyOn(window, "open").mockImplementation(() => null)
  })

  it("keeps Open full app route-only with no handoff parameter", () => {
    render(<ControlRow {...defaultProps()} draftMessage="draft text" />)

    fireEvent.click(screen.getByTestId("chat-open-full-app"))

    expect(mocks.runtimeGetURL).toHaveBeenCalledWith("/options.html#/chat")
    expect(mocks.tabsCreate).toHaveBeenCalledWith({
      url: "chrome-extension://handoff/options.html#/chat"
    })
    expect(mocks.createSidepanelChatHandoff).not.toHaveBeenCalled()
    expect(mocks.buildSidepanelChatHandoffRoute).not.toHaveBeenCalled()
  })

  it("creates a handoff package and opens /chat?handoff=<id>", async () => {
    render(<ControlRow {...defaultProps()} draftMessage="Summarize this" />)

    fireEvent.click(screen.getByTestId("chat-continue-in-webui"))

    await waitFor(() => {
      expect(mocks.createSidepanelChatHandoff).toHaveBeenCalledWith({
        draftText: "Summarize this",
        pageContext: undefined,
        routeIntent: {
          path: "/chat"
        }
      })
    })
    expect(mocks.buildSidepanelChatHandoffRoute).toHaveBeenCalledWith(
      "/chat",
      "handoff-123"
    )
    expect(mocks.runtimeGetURL).toHaveBeenCalledWith(
      "/options.html#/chat?handoff=handoff-123"
    )
    expect(mocks.tabsCreate).toHaveBeenCalledWith({
      url: "chrome-extension://handoff/options.html#/chat?handoff=handoff-123"
    })
  })

  it("ignores a second click while handoff creation is still pending", async () => {
    const deferred = createDeferred<{
      id: string
      source: "sidepanel-chat"
      createdAt: string
      expiresAt: string
      draft: { text: string }
    }>()
    mocks.createSidepanelChatHandoff.mockReturnValueOnce(deferred.promise)
    render(<ControlRow {...defaultProps()} draftMessage="Summarize this" />)

    const continueButton = screen.getByTestId("chat-continue-in-webui")
    fireEvent.click(continueButton)
    fireEvent.click(continueButton)

    await waitFor(() => {
      expect(mocks.createSidepanelChatHandoff).toHaveBeenCalledTimes(1)
    })
    expect(continueButton).toBeDisabled()
    expect(mocks.tabsCreate).not.toHaveBeenCalled()

    deferred.resolve({
      id: "handoff-pending",
      source: "sidepanel-chat",
      createdAt: "2026-05-29T00:00:00.000Z",
      expiresAt: "2026-05-29T00:10:00.000Z",
      draft: { text: "Summarize this" }
    })

    await waitFor(() => {
      expect(mocks.tabsCreate).toHaveBeenCalledTimes(1)
    })
    expect(mocks.tabsCreate).toHaveBeenCalledWith({
      url: "chrome-extension://handoff/options.html#/chat?handoff=handoff-pending"
    })
  })

  it("merges handoff into active character route params", async () => {
    mocks.selectedAssistant = {
      kind: "character",
      id: "char-review",
      name: "Review Guide"
    }

    render(
      <ControlRow
        {...defaultProps()}
        selectedCharacterId="char-review"
        draftMessage="Continue the scene"
      />
    )

    fireEvent.click(screen.getByTestId("chat-continue-in-webui"))

    await waitFor(() => {
      expect(mocks.createSidepanelChatHandoff).toHaveBeenCalledWith({
        draftText: "Continue the scene",
        pageContext: undefined,
        routeIntent: {
          path: "/chat?mode=character&characterId=char-review",
          mode: "character",
          characterId: "char-review"
        }
      })
    })
    expect(mocks.buildSidepanelChatHandoffRoute).toHaveBeenCalledWith(
      "/chat?mode=character&characterId=char-review",
      "handoff-123"
    )
    expect(mocks.runtimeGetURL).toHaveBeenCalledWith(
      "/options.html#/chat?mode=character&characterId=char-review&handoff=handoff-123"
    )
  })

  it("does not serialize draft, snippets, title, or URL content into the URL", async () => {
    const pageContext = {
      title: "Sensitive Article",
      url: "https://example.test/private",
      snippets: [
        {
          kind: "visible-context" as const,
          label: "Selected tab",
          text: "private selected snippet"
        }
      ]
    }
    render(
      <ControlRow
        {...defaultProps()}
        draftMessage="private draft text"
        hasVisiblePageContextForHandoff
        getVisiblePageContextForHandoff={() => pageContext}
      />
    )

    fireEvent.click(screen.getByTestId("chat-continue-in-webui"))

    await waitFor(() => {
      expect(mocks.tabsCreate).toHaveBeenCalled()
    })
    const openedUrl = String(mocks.tabsCreate.mock.calls.at(-1)?.[0]?.url)
    expect(openedUrl).toContain("handoff=handoff-123")
    expect(openedUrl).not.toContain("private%20draft")
    expect(openedUrl).not.toContain("private selected snippet")
    expect(openedUrl).not.toContain("Sensitive")
    expect(openedUrl).not.toContain("example.test")
    expect(mocks.createSidepanelChatHandoff).toHaveBeenCalledWith({
      draftText: "private draft text",
      pageContext,
      routeIntent: {
        path: "/chat"
      }
    })
  })

  it("shows a disabled Continue in WebUI action when no draft or context exists", () => {
    render(<ControlRow {...defaultProps()} draftMessage="   " />)

    const continueButton = screen.getByTestId("chat-continue-in-webui")

    expect(continueButton).toBeDisabled()
  })

  it("shows a warning and does not open when visible context is gone at click time", async () => {
    render(
      <ControlRow
        {...defaultProps()}
        draftMessage="   "
        hasVisiblePageContextForHandoff
        getVisiblePageContextForHandoff={() => undefined}
      />
    )

    fireEvent.click(screen.getByTestId("chat-continue-in-webui"))

    expect(await screen.findByRole("status")).toHaveTextContent(
      "Nothing to continue in WebUI"
    )
    expect(mocks.createSidepanelChatHandoff).not.toHaveBeenCalled()
    expect(mocks.tabsCreate).not.toHaveBeenCalled()
  })

  it("shows an error and does not open a tab when handoff creation fails", async () => {
    mocks.createSidepanelChatHandoff.mockRejectedValueOnce(new Error("storage failed"))
    render(<ControlRow {...defaultProps()} draftMessage="Summarize this" />)

    fireEvent.click(screen.getByTestId("chat-continue-in-webui"))

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Could not continue in WebUI"
    )
    expect(mocks.tabsCreate).not.toHaveBeenCalled()
    expect(window.open).not.toHaveBeenCalled()
  })
})
