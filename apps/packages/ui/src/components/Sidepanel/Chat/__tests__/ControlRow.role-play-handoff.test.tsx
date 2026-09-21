import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
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
  ModelSelect: React.forwardRef((_props, ref) => {
    const button = React.useRef<HTMLButtonElement>(null)
    React.useImperativeHandle(ref, () => ({
      openAndFocus: () => button.current?.focus()
    }))
    return (
      <button ref={button} type="button">Model</button>
    )
  })
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

  it("uses the resolved extension URL when tabs.create is unavailable", () => {
    render(<ControlRow {...defaultProps()} />)

    const openFullApp = screen.getByTestId("chat-open-full-app")

    expect(openFullApp).toHaveAccessibleDescription(
      "Opens /chat in a new tab with the active role-play route. Use Continue in WebUI to carry a draft or page context."
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
    expect(screen.getByRole("status")).toHaveTextContent("Search your knowledge")
    expect(screen.queryByText("More tools available")).not.toBeInTheDocument()
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
      screen.getByRole("button", { name: "Open knowledge search" })
    ).toHaveFocus())
    expect(screen.queryByText("Search your knowledge")).not.toBeInTheDocument()
    expect(screen.getByText("More tools available")).toBeVisible()
    screen.getByRole("button", { name: "Dismiss" }).focus()
    await user.keyboard(" ")
    await waitFor(() => expect(
      screen.getByRole("button", { name: "More tools" })
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
    expect(screen.queryByText("Search your knowledge")).not.toBeInTheDocument()
    expect(screen.getByText("More tools available")).toBeVisible()
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
    expect(screen.queryByText("Search your knowledge")).not.toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Dismiss" }))
    expect(screen.queryByRole("button", { name: "Dismiss" })).not.toBeInTheDocument()
    view.rerender(<ControlRow {...props} isConnected />)
    expect(screen.getByText("Search your knowledge")).toBeVisible()
    expect(screen.queryByText("More tools available")).not.toBeInTheDocument()
  })

  it("preserves More tools Escape and trigger focus while a hint is visible", async () => {
    render(<ControlRow {...defaultProps()} />)
    const more = screen.getByRole("button", { name: "More tools" })
    fireEvent.click(more)
    expect(more).toHaveAttribute("aria-expanded", "true")
    fireEvent.keyDown(screen.getByTestId("chat-open-full-app"), { key: "Escape" })
    expect(more).toHaveAttribute("aria-expanded", "false")
    await waitFor(() => expect(more).toHaveFocus())
    expect(screen.getByText("Search your knowledge")).toBeVisible()
  })
})
