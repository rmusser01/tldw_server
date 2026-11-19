import React from "react"
import { type PopoverProps } from "antd"
import { fireEvent, render, screen } from "@testing-library/react"
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

type MockPopoverProps = PopoverProps

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
  useNativePopover: false,
  runtimeId: "handoff-extension" as string | undefined,
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
  consumeSidepanelChatHandoff: vi.fn(),
  buildSidepanelChatHandoffRoute:
    vi.fn<(basePath: string, handoffId: string) => string>(),
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
      get id() { return mocks.runtimeId },
      getURL: mocks.runtimeGetURL
    },
    tabs: {
      create: mocks.tabsCreate
    }
  }
}))

vi.mock("@/services/sidepanel-chat-handoff", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/services/sidepanel-chat-handoff")>();
  mocks.buildSidepanelChatHandoffRoute.mockImplementation(
    actual.buildSidepanelChatHandoffRoute,
  );
  return {
    createSidepanelChatHandoff: mocks.createSidepanelChatHandoff,
    consumeSidepanelChatHandoff: mocks.consumeSidepanelChatHandoff,
    buildSidepanelChatHandoffRoute: mocks.buildSidepanelChatHandoffRoute,
  };
});

vi.mock("antd", async (importOriginal) => {
  const actual = await importOriginal<typeof import("antd")>()
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
    ConfigProvider: actual.ConfigProvider,
    Input: ({ onPressEnter: _onPressEnter, ...props }: MockInputProps) => (
      <input {...props} />
    ),
    InputNumber: (props: React.InputHTMLAttributes<HTMLInputElement>) => (
      <input type="number" {...props} />
    ),
    Popover: (props: MockPopoverProps) => mocks.useNativePopover
      ? <actual.Popover {...props} />
      : <>{props.children}<div data-testid="mock-popover-content">{props.content as React.ReactNode}</div></>,
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

const fullPage = (overrides: Record<string, unknown> = {}) => ({
  label: "Open in full page",
  description: "Opens this chat in a new tab, with your draft and page references. Your draft also stays here.",
  pending: false,
  disabled: false,
  target: "extension" as const,
  open: vi.fn(async () => true),
  ...overrides
})

/**
 * XP-09 (#3101): More tools has one "Open in full page" action. It replaces
 * "Expand in full page" and "Copy draft & page references"; the composer's
 * useSidepanelOpenFullPage decides what it carries (see its own tests).
 */
describe("ControlRow Open in full page", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.useNativePopover = false
    mocks.runtimeId = "handoff-extension"
    mocks.selectedAssistant = null
  })

  it("offers one Open in full page action and no separate draft copy", () => {
    render(<ControlRow {...defaultProps()} fullPage={fullPage()} />)
    expect(screen.getAllByRole("button", { name: /full page|full chat|full app|full extension/i })).toHaveLength(1)
    expect(screen.getByRole("button", { name: "Open in full page" })).toHaveAccessibleDescription(
      "Opens this chat in a new tab, with your draft and page references. Your draft also stays here."
    )
    expect(screen.queryByRole("button", { name: "Copy draft & page references" })).toBeNull()
    expect(screen.queryByTestId("chat-continue-in-webui")).toBeNull()
  })

  it("runs the composer's action", () => {
    const action = fullPage()
    render(<ControlRow {...defaultProps()} fullPage={action} />)
    fireEvent.click(screen.getByRole("button", { name: "Open in full page" }))
    expect(action.open).toHaveBeenCalledOnce()
    expect(mocks.createSidepanelChatHandoff).not.toHaveBeenCalled()
    expect(mocks.tabsCreate).not.toHaveBeenCalled()
  })

  it.each([
    ["not ready", { disabled: true }],
    ["already opening", { pending: true }]
  ])("is disabled while %s", (_state, overrides) => {
    const action = fullPage(overrides)
    render(<ControlRow {...defaultProps()} fullPage={action} />)
    const open = screen.getByRole("button", { name: "Open in full page" })
    expect(open).toBeDisabled()
    fireEvent.click(open)
    expect(action.open).not.toHaveBeenCalled()
  })

  it("leaves the action out when the composer offers none", () => {
    render(<ControlRow {...defaultProps()} />)
    expect(screen.queryByRole("button", { name: "Open in full page" })).toBeNull()
  })
})
