import React from "react"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { ControlRow } from "../ControlRow"
import { browser } from "wxt/browser"

type MutableBrowser = {
  tabs: {
    create?: (input: { url: string }) => unknown
  }
}

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
  open?: boolean
  onOpenChange?: (open: boolean) => void
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

const mocks = vi.hoisted(() => ({
  runtimeId: "review-extension" as string | undefined,
  selectedAssistant: {
    kind: "character",
    id: "char-review",
    name: "Review Guide"
  },
  setSelectedAssistant: vi.fn(),
  setSelectedCharacterId: vi.fn(),
  setMoreMenuFocus: vi.fn(),
  setToolCatalog: vi.fn(),
  setToolCatalogId: vi.fn(),
  setToolModules: vi.fn(),
  setToolCatalogStrict: vi.fn(),
  setMcpToolEnabled: vi.fn(),
  resetMcpToolFilter: vi.fn(),
  setMoodBadge: vi.fn(),
  setStorage: vi.fn(),
  fetchChatModels: vi.fn(async () => []),
  runtimeGetURL: vi.fn((path: string) => `chrome-extension://review${path}`),
  tabsCreate: vi.fn()
}))

vi.mock("wxt/browser", () => ({
  browser: {
    runtime: {
      get id() { return mocks.runtimeId },
      getURL: mocks.runtimeGetURL
    },
    tabs: {}
  }
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
    Popover: ({ children, content, open, onOpenChange }: MockPopoverProps) => (
      <>
        {React.isValidElement(children)
          ? React.cloneElement(
              children as React.ReactElement<{ onClick?: () => void }>,
              { onClick: () => onOpenChange?.(!open) }
            )
          : children}
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

vi.mock("@tanstack/react-query", async (original) => {
  const actual = await original<typeof import("@tanstack/react-query")>()
  const queryClient = new actual.QueryClient()
  return {
    ...actual,
    useQuery: () => ({ data: [] }),
    useQueryClient: () => queryClient
  }
})

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string, options?: Record<string, unknown>) =>
      String(fallback ?? _key).replace(/\{\{(\w+)\}\}/g, (_, key) =>
        String(options?.[key] ?? "")
      )
  })
}))

vi.mock("@/components/Common/ModelSelect", () => ({
  ModelSelect: React.forwardRef<{ openAndFocus: () => void }>(
    function MockModelSelect(_props, ref) {
      const trigger = React.useRef<HTMLButtonElement>(null)
      React.useImperativeHandle(
        ref,
        () => ({ openAndFocus: () => trigger.current?.focus() }),
        []
      )
      return (
        <button ref={trigger} type="button">
          Model
        </button>
      )
    }
  )
}))

vi.mock("@/components/Common/PromptSelect", () => ({
  PromptSelect: () => <button type="button">Prompt</button>
}))

// Keep the real FeatureHint; all instances observe one external setting fixture.
const hintSettings = vi.hoisted(() => ({
  seen: {} as Record<string, boolean>,
  listeners: new Set<() => void>()
}))
vi.mock("@/hooks/useSetting", () => ({
  useSetting: () => {
    const seen = React.useSyncExternalStore(
      (listener) => {
        hintSettings.listeners.add(listener)
        return () => hintSettings.listeners.delete(listener)
      },
      () => hintSettings.seen
    )
    return [
      seen,
      async (
        next: Record<string, boolean> |
          ((previous: Record<string, boolean>) => Record<string, boolean>)
      ) => {
        hintSettings.seen =
          typeof next === "function" ? next(hintSettings.seen) : next
        hintSettings.listeners.forEach((listener) => listener())
      }
    ] as const
  }
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
  selectedCharacterId: "char-review",
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

describe("ControlRow role-play handoff behavior", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    hintSettings.seen = {}
    vi.spyOn(window, "open").mockImplementation(() => null)
    delete (browser as unknown as MutableBrowser).tabs.create
  })

  it("clears role-play selection through the selected-assistant hook", () => {
    render(<ControlRow {...defaultProps()} />)

    fireEvent.click(screen.getByTestId("sidepanel-character-chat-clear"))

    expect(mocks.setSelectedCharacterId).toHaveBeenCalledWith(null)
    expect(mocks.setSelectedAssistant).toHaveBeenCalledWith(null)
  })

  it("offers the one Open in full page action for a character chat too", () => {
    const open = vi.fn(async () => true)
    render(<ControlRow {...defaultProps()} fullPage={{ label: "Open in full page", description: "Opens this chat in a new tab.", pending: false, disabled: false, disabledReason: null, webUi: null, target: "extension", open }} />)
    fireEvent.click(screen.getByRole("button", { name: "Open in full page" }))
    expect(open).toHaveBeenCalledOnce()
    expect(screen.queryByTestId("chat-continue-in-webui")).toBeNull()
  })
})

// Native UAT372 recorded floating hint paragraphs intercepting the model picker.
// Native narrow/wide hit testing is still required for the final packaged layout.
describe("ControlRow first-use guidance", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    hintSettings.seen = {}
  })

  it("shows one eligible non-tooltip hint before the controls", () => {
    render(<ControlRow {...defaultProps()} />)
    expect(
      screen.getAllByRole("button", { name: "Dismiss" })
    ).toHaveLength(1)
    expect(screen.getByRole("status")).toHaveTextContent("Open Search & Context to find snippets")
    expect(screen.queryByText(/are under Tools\./)).not.toBeInTheDocument()
    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument()
  })

  it("dismisses by keyboard without sending the form and returns focus to the feature", async () => {
    const user = userEvent.setup()
    const submit = vi.fn((event: React.FormEvent) => event.preventDefault())
    render(
      <form onSubmit={submit}><ControlRow {...defaultProps()} /></form>
    )
    screen.getAllByRole("button", { name: "Dismiss" })[0].focus()
    await user.keyboard("{Enter}")
    expect(submit).not.toHaveBeenCalled()
    await waitFor(() => expect(
      screen.getByRole("button", { name: "Search & Context" })
    ).toHaveFocus())
    expect(screen.queryByText(/Open Search & Context to find snippets/)).not.toBeInTheDocument()
    expect(screen.getByText(/are under Tools\./)).toBeVisible()
    screen.getByRole("button", { name: "Dismiss" }).focus()
    await user.keyboard(" ")
    await waitFor(() => expect(
      screen.getByRole("button", { name: "Tools" })
    ).toHaveFocus())
    expect(screen.queryByRole("button", { name: "Dismiss" })).not.toBeInTheDocument()
    expect(submit).not.toHaveBeenCalled()
  })

  it("keeps both dismissals after remount while preserving other seen hints", async () => {
    const user = userEvent.setup()
    hintSettings.seen = { "another-feature": true }
    const first = render(<ControlRow {...defaultProps()} />)
    await user.click(screen.getAllByRole("button", { name: "Dismiss" })[0])
    first.unmount()
    const second = render(<ControlRow {...defaultProps()} />)
    expect(screen.queryByText(/Open Search & Context to find snippets/)).not.toBeInTheDocument()
    expect(screen.getByText(/are under Tools\./)).toBeVisible()
    await user.click(screen.getByRole("button", { name: "Dismiss" }))
    second.unmount()
    render(<ControlRow {...defaultProps()} />)
    expect(screen.queryByRole("button", { name: "Dismiss" })).not.toBeInTheDocument()
    expect(hintSettings.seen["another-feature"]).toBe(true)
  })

  it("defers knowledge guidance until connected without marking it seen", async () => {
    const user = userEvent.setup()
    const props = defaultProps()
    const view = render(<ControlRow {...props} isConnected={false} />)
    expect(screen.queryByText(/Open Search & Context to find snippets/)).not.toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Dismiss" }))
    expect(screen.queryByRole("button", { name: "Dismiss" })).not.toBeInTheDocument()
    view.rerender(<ControlRow {...props} isConnected />)
    expect(screen.getByText(/Open Search & Context to find snippets/)).toBeVisible()
    expect(screen.queryByText(/are under Tools\./)).not.toBeInTheDocument()
  })

  it("preserves More tools Escape and trigger focus while a hint is visible", async () => {
    render(<ControlRow {...defaultProps()} />)
    const more = screen.getByRole("button", { name: "Tools" })
    fireEvent.click(more)
    expect(more).toHaveAttribute("aria-expanded", "true")
    fireEvent.keyDown(screen.getByTestId("chat-quick-ingest"), { key: "Escape" })
    expect(more).toHaveAttribute("aria-expanded", "false")
    await waitFor(() => expect(more).toHaveFocus())
    expect(screen.getByText(/Open Search & Context to find snippets/)).toBeVisible()
  })
})

it("closes More and returns focus to it when Open in full page is chosen", async () => {
  const open = vi.fn(async () => true)
  render(<ControlRow {...defaultProps()} fullPage={{ label: "Open in full page", description: "Opens this chat in a new tab.", pending: false, disabled: false, disabledReason: null, webUi: null, target: "extension", open }} />)
  const more = screen.getByRole("button", { name: "Tools" })
  fireEvent.click(more)
  fireEvent.click(screen.getByTestId("chat-open-full-app"))
  expect(open).toHaveBeenCalledOnce()
  expect(more).toHaveAttribute("aria-expanded", "false")
  await waitFor(() => expect(more).toHaveFocus())
})
