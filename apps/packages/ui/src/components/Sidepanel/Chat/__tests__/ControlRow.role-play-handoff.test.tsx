import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
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
      id: "review-extension",
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
    vi.spyOn(window, "open").mockImplementation(() => null)
    delete (browser as unknown as MutableBrowser).tabs.create
  })

  it("clears role-play selection through the selected-assistant hook", () => {
    render(<ControlRow {...defaultProps()} />)

    fireEvent.click(screen.getByTestId("sidepanel-character-chat-clear"))

    expect(mocks.setSelectedCharacterId).toHaveBeenCalledWith(null)
    expect(mocks.setSelectedAssistant).toHaveBeenCalledWith(null)
  })

  it("uses the resolved extension URL when tabs.create is unavailable", () => {
    render(<ControlRow {...defaultProps()} />)

    const openFullApp = screen.getByTestId("chat-open-full-app")

    expect(openFullApp).toHaveAccessibleDescription(
      "Opens /chat in a new tab with the active role-play route. Sidepanel draft, current page context, and unsaved chat state stay in the sidepanel."
    )

    fireEvent.click(openFullApp)

    expect(mocks.runtimeGetURL).toHaveBeenCalledWith(
      "/options.html#/chat?mode=character&characterId=char-review"
    )
    expect(window.open).toHaveBeenCalledWith(
      "chrome-extension://review/options.html#/chat?mode=character&characterId=char-review",
      "_blank"
    )
    expect(mocks.tabsCreate).not.toHaveBeenCalled()
  })

  it("uses tabs.create when the extension tab API is available", () => {
    ;(browser as unknown as MutableBrowser).tabs.create = mocks.tabsCreate
    render(<ControlRow {...defaultProps()} />)

    fireEvent.click(screen.getByTestId("chat-open-full-app"))

    expect(mocks.tabsCreate).toHaveBeenCalledWith({
      url: "chrome-extension://review/options.html#/chat?mode=character&characterId=char-review"
    })
    expect(window.open).not.toHaveBeenCalled()
  })
})
