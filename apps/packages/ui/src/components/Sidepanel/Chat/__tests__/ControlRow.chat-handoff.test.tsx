import React from "react"
import { ConfigProvider, type PopoverProps } from "antd"
import { readVisiblePageContextForHandoff } from "../sidepanel-chat-handoff-context"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
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
      get id() { return mocks.runtimeId },
      getURL: mocks.runtimeGetURL
    },
    tabs: {
      create: mocks.tabsCreate
    }
  }
}))

vi.mock("@/services/sidepanel-chat-handoff", () => ({
  createSidepanelChatHandoff: mocks.createSidepanelChatHandoff,
  consumeSidepanelChatHandoff: mocks.consumeSidepanelChatHandoff,
  buildSidepanelChatHandoffRoute: mocks.buildSidepanelChatHandoffRoute
}))

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
  draftReady: true,
  captureHandoffGuard: () => () => true,
  handoffSourceKey: "owned-source-chat",
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
    mocks.useNativePopover = false
    mocks.runtimeId = "handoff-extension"
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

  it("does not offer copying before the owned source draft is restored", () => {
    render(<ControlRow {...defaultProps()} draftReady={false} draftMessage="RESTORING" />)
    expect(screen.getByRole("button", { name: "Copy draft & page references" })).toBeDisabled()
    fireEvent.click(screen.getByRole("button", { name: "Copy draft & page references" }))
    expect(mocks.createSidepanelChatHandoff).not.toHaveBeenCalled()
  })

  it.each(["page read", "package write"])("does not open or announce a held %s after source ownership retires", async boundary => {
    const held = createDeferred<unknown>()
    let current = true
    const props = { ...defaultProps(), draftMessage: "SOURCE DRAFT  ", captureHandoffGuard: () => () => current }
    if (boundary === "package write") mocks.createSidepanelChatHandoff.mockReturnValueOnce(held.promise)
    render(<ControlRow {...props} getVisiblePageContextForHandoff={boundary === "page read" ? () => held.promise : undefined} />)
    fireEvent.click(screen.getByRole("button", { name: "Copy draft & page references" }))
    current = false
    await act(async () => held.resolve(boundary === "page read" ? { title: "Old private page", snippets: [] } : { id: "old-package", draft: { text: "SOURCE DRAFT  " } }))
    expect(mocks.tabsCreate).not.toHaveBeenCalled()
    expect(window.open).not.toHaveBeenCalled()
    expect(screen.queryByTestId("chat-handoff-feedback")).toBeNull()
    if (boundary === "page read") expect(mocks.createSidepanelChatHandoff).not.toHaveBeenCalled()
  })

  it("allows a new source copy while the old page read is held and ignores the old result", async () => {
    const held = createDeferred<unknown>()
    let oldCurrent = true
    const oldProps = { ...defaultProps(), draftMessage: "OLD TEXT", captureHandoffGuard: () => () => oldCurrent, getVisiblePageContextForHandoff: () => held.promise }
    const view = render(<ControlRow {...oldProps} />)
    fireEvent.click(screen.getByRole("button", { name: "Copy draft & page references" }))
    oldCurrent = false
    view.rerender(<ControlRow {...defaultProps()} handoffSourceKey="new-owner-and-chat" draftMessage="NEW TEXT" />)
    expect(screen.getByRole("button", { name: "Copy draft & page references" })).toBeEnabled()
    fireEvent.click(screen.getByRole("button", { name: "Copy draft & page references" }))
    await waitFor(() => expect(mocks.tabsCreate).toHaveBeenCalledOnce())
    expect(mocks.createSidepanelChatHandoff).toHaveBeenCalledWith(expect.objectContaining({ draftText: "NEW TEXT" }), expect.any(Function))
    await act(async () => held.resolve({ title: "OLD PAGE", snippets: [] }))
    expect(mocks.tabsCreate).toHaveBeenCalledOnce()
    expect(mocks.createSidepanelChatHandoff).toHaveBeenCalledOnce()
    expect(screen.getByTestId("chat-handoff-feedback")).toHaveTextContent("Full chat page opened")
  })

  it("does not publish a late open failure after the source has changed", async () => {
    const opened = createDeferred<unknown>()
    mocks.tabsCreate.mockReturnValueOnce(opened.promise)
    let current = true
    render(<ControlRow {...defaultProps()} draftMessage="SOURCE" captureHandoffGuard={() => () => current} />)
    fireEvent.click(screen.getByRole("button", { name: "Copy draft & page references" }))
    await waitFor(() => expect(mocks.tabsCreate).toHaveBeenCalledOnce())
    current = false
    await act(async () => opened.reject(new Error("late failure")))
    expect(window.open).not.toHaveBeenCalled()
    expect(screen.queryByTestId("chat-handoff-feedback")).toBeNull()
  })

  it("retires a held copy when source text changes without changing its chat", async () => {
    const held = createDeferred<undefined>()
    const props = { ...defaultProps(), getVisiblePageContextForHandoff: () => held.promise }
    const view = render(<ControlRow {...props} draftMessage="BEFORE" />)
    fireEvent.click(screen.getByRole("button", { name: "Copy draft & page references" }))
    view.rerender(<ControlRow {...props} draftMessage="AFTER  " />)
    expect(screen.getByRole("button", { name: "Copy draft & page references" })).toBeEnabled()
    await act(async () => held.resolve(undefined))
    expect(mocks.createSidepanelChatHandoff).not.toHaveBeenCalled()
    expect(mocks.tabsCreate).not.toHaveBeenCalled()
  })

  it("does not open a held package after the source control unmounts", async () => {
    const held = createDeferred<unknown>()
    mocks.createSidepanelChatHandoff.mockReturnValueOnce(held.promise)
    const view = render(<ControlRow {...defaultProps()} draftMessage="KEEP SOURCE" />)
    fireEvent.click(screen.getByRole("button", { name: "Copy draft & page references" }))
    await waitFor(() => expect(mocks.createSidepanelChatHandoff).toHaveBeenCalledOnce())
    view.unmount()
    await act(async () => held.resolve({ id: "unpublished-copy", draft: { text: "KEEP SOURCE" } }))
    expect(mocks.tabsCreate).not.toHaveBeenCalled()
    expect(window.open).not.toHaveBeenCalled()
    expect(mocks.consumeSidepanelChatHandoff).toHaveBeenCalledWith("unpublished-copy")
  })

  it("does not let an old completion unlock a new pending copy", async () => {
    const oldRead = createDeferred<undefined>()
    const newRead = createDeferred<undefined>()
    let oldCurrent = true
    const view = render(<ControlRow {...defaultProps()} draftMessage="OLD" captureHandoffGuard={() => () => oldCurrent} getVisiblePageContextForHandoff={() => oldRead.promise} />)
    fireEvent.click(screen.getByRole("button", { name: "Copy draft & page references" }))
    oldCurrent = false
    view.rerender(<ControlRow {...defaultProps()} draftMessage="NEW" handoffSourceKey="new-owner" getVisiblePageContextForHandoff={() => newRead.promise} />)
    fireEvent.click(screen.getByRole("button", { name: "Copy draft & page references" }))
    await act(async () => oldRead.resolve(undefined))
    expect(screen.getByRole("button", { name: "Copy draft & page references" })).toBeDisabled()
    await act(async () => newRead.resolve(undefined))
    await waitFor(() => expect(mocks.tabsCreate).toHaveBeenCalledOnce())
    expect(mocks.createSidepanelChatHandoff).toHaveBeenCalledWith(expect.objectContaining({ draftText: "NEW" }), expect.any(Function))
  })

  it("does not announce or refocus after an already issued open resolves for a retired source", async () => {
    const opened = createDeferred<undefined>()
    mocks.tabsCreate.mockReturnValueOnce(opened.promise)
    let current = true
    render(<ControlRow {...defaultProps()} draftMessage="SOURCE" captureHandoffGuard={() => () => current} />)
    fireEvent.click(screen.getByRole("button", { name: "Copy draft & page references" }))
    await waitFor(() => expect(mocks.tabsCreate).toHaveBeenCalledOnce())
    const focus = vi.spyOn(HTMLElement.prototype, "focus")
    current = false
    await act(async () => opened.resolve(undefined))
    expect(focus).not.toHaveBeenCalled()
    expect(window.open).not.toHaveBeenCalled()
    expect(screen.queryByTestId("chat-handoff-feedback")).toBeNull()
    focus.mockRestore()
  })

  it("refuses to start a copy from a retired rendered source", () => {
    render(<ControlRow {...defaultProps()} draftMessage="OLD VISIBLE TEXT" captureHandoffGuard={() => () => false} />)
    fireEvent.click(screen.getByRole("button", { name: "Copy draft & page references" }))
    expect(mocks.createSidepanelChatHandoff).not.toHaveBeenCalled()
    expect(mocks.tabsCreate).not.toHaveBeenCalled()
    expect(screen.queryByTestId("chat-handoff-feedback")).toBeNull()
  })

  it("allows a fresh copy after restoration readiness returns and ignores the held earlier read", async () => {
    const oldRead = createDeferred<undefined>()
    const getContext = vi.fn().mockReturnValueOnce(oldRead.promise).mockResolvedValue(undefined)
    const props = { ...defaultProps(), draftMessage: "RETAINED SOURCE  ", getVisiblePageContextForHandoff: getContext }
    const view = render(<ControlRow {...props} />)
    fireEvent.click(screen.getByRole("button", { name: "Copy draft & page references" }))
    view.rerender(<ControlRow {...props} draftReady={false} />)
    expect(screen.getByRole("button", { name: "Copy draft & page references" })).toBeDisabled()
    view.rerender(<ControlRow {...props} />)
    expect(screen.getByRole("button", { name: "Copy draft & page references" })).toBeEnabled()
    fireEvent.click(screen.getByRole("button", { name: "Copy draft & page references" }))
    await waitFor(() => expect(mocks.tabsCreate).toHaveBeenCalledOnce())
    await act(async () => oldRead.resolve(undefined))
    expect(mocks.createSidepanelChatHandoff).toHaveBeenCalledOnce()
    expect(mocks.createSidepanelChatHandoff).toHaveBeenCalledWith(expect.objectContaining({ draftText: "RETAINED SOURCE  " }), expect.any(Function))
  })

  it("keeps one source copy usable through StrictMode effect replay", async () => {
    render(<React.StrictMode><ControlRow {...defaultProps()} draftMessage="STRICT SOURCE" /></React.StrictMode>)
    fireEvent.click(screen.getByRole("button", { name: "Copy draft & page references" }))
    await waitFor(() => expect(mocks.tabsCreate).toHaveBeenCalledOnce())
    expect(mocks.createSidepanelChatHandoff).toHaveBeenCalledOnce()
    expect(screen.getByTestId("chat-handoff-feedback")).toHaveTextContent("Full chat page opened")
  })

  it("keeps a current copy receipt after its capture callback rerenders", async () => {
    const props = { ...defaultProps(), draftMessage: "SOURCE" }
    const view = render(<ControlRow {...props} />)
    fireEvent.click(screen.getByRole("button", { name: "Copy draft & page references" }))
    await waitFor(() => expect(screen.getByTestId("chat-handoff-feedback")).toHaveTextContent("Full chat page opened"))
    view.rerender(<ControlRow {...props} captureHandoffGuard={() => () => true} />)
    expect(screen.getByTestId("chat-handoff-feedback")).toHaveTextContent("Full chat page opened")
    fireEvent.click(screen.getByRole("button", { name: "Copy draft & page references" }))
    await waitFor(() => expect(mocks.tabsCreate).toHaveBeenCalledTimes(2))
  })

  it("names the bounded copy destination and its carried-state limits", () => {
    render(<ControlRow {...defaultProps()} draftMessage="draft text" />)
    const copy = screen.getByRole("button", { name: "Copy draft & page references" })
    expect(copy).toHaveAccessibleDescription("Opens the extension full page with a short-lived copy of your draft and page titles and URLs. Page content, history, and chat settings are not included. Review the copy in the destination.")
  })

  it.each([false, true])("keeps an opened copy outcome outside More, shortened=%s", async shortened => {
    if (shortened) mocks.createSidepanelChatHandoff.mockResolvedValueOnce({ id: "handoff-123", draft: { text: "Short copy", truncated: true } })
    render(<ControlRow {...defaultProps()} draftMessage="Original source draft  " />)
    fireEvent.click(screen.getByTestId("chat-continue-in-webui"))
    const feedback = await screen.findByTestId("chat-handoff-feedback")
    expect(feedback).toHaveTextContent(shortened ? "partial copy" : "Full chat page opened")
    expect(feedback).toHaveTextContent("Your source draft remains here")
    expect(feedback).not.toHaveTextContent(/Imported/i)
    expect(feedback.closest('[data-testid="mock-popover-content"]')).toBeNull()
    expect(mocks.createSidepanelChatHandoff).toHaveBeenCalledWith(expect.objectContaining({ draftText: "Original source draft  " }), expect.any(Function))
  })

  it("copies within the current WebUI origin independently of its configured callback", async () => {
    mocks.runtimeId = undefined
    const callback = vi.fn()
    vi.mocked(window.open).mockImplementationOnce(() => ({ closed: false }) as Window)
    render(<ControlRow {...defaultProps()} draftMessage="Source draft" onOpenChatInWebUi={callback} />)
    const copy = screen.getByRole("button", { name: "Copy draft & page references" })
    expect(copy).toHaveAccessibleDescription("Opens /chat in this WebUI with a short-lived copy of your draft and page titles and URLs. Page content, history, and chat settings are not included. Review the copy in the destination.")
    fireEvent.click(copy)
    await waitFor(() => expect(window.open).toHaveBeenCalledWith("/chat?handoff=handoff-123", "_blank"))
    expect(callback).not.toHaveBeenCalled()
    expect(mocks.tabsCreate).not.toHaveBeenCalled()
  })

  it("reports a partial copy when requested page context cannot be read", async () => {
    render(<ControlRow {...defaultProps()} draftMessage="Source text" hasVisiblePageContextForHandoff getVisiblePageContextForHandoff={async () => { throw new Error("page unavailable") }} />)
    fireEvent.click(screen.getByTestId("chat-continue-in-webui"))
    const feedback = await screen.findByTestId("chat-handoff-feedback")
    expect(feedback).toHaveTextContent("partial copy")
    expect(feedback).toHaveTextContent("unavailable")
    expect(mocks.tabsCreate).toHaveBeenCalledTimes(1)
  })

  it("reports the actual producer's failed active-page read even when document references survive", async () => {
    const readTabs = vi.fn().mockRejectedValue(new Error("tab unavailable"))
    mocks.createSidepanelChatHandoff.mockImplementationOnce(async input => ({ id: "handoff-123", draft: { text: input.draftText }, pageContext: input.pageContext }))
    render(<ControlRow {...defaultProps()} draftMessage="Source draft" hasVisiblePageContextForHandoff
      getVisiblePageContextForHandoff={() => readVisiblePageContextForHandoff([{ title: "Research note" }], true, readTabs)} />)
    fireEvent.click(screen.getByTestId("chat-continue-in-webui"))
    expect(await screen.findByTestId("chat-handoff-feedback")).toHaveTextContent("partial copy")
    expect(mocks.createSidepanelChatHandoff).toHaveBeenCalledWith(expect.objectContaining({ pageContext: expect.objectContaining({ truncated: true }) }), expect.any(Function))
    expect(readTabs).toHaveBeenCalledTimes(1)
  })

  it("offers retry for the actual producer's failed context-only read without opening an empty destination", async () => {
    render(<ControlRow {...defaultProps()} hasVisiblePageContextForHandoff
      getVisiblePageContextForHandoff={() => readVisiblePageContextForHandoff([], true, async () => { throw new Error("tab unavailable") })} />)
    fireEvent.click(screen.getByTestId("chat-continue-in-webui"))
    expect(await screen.findByTestId("chat-handoff-feedback")).toHaveTextContent("Page references could not be read. Try again.")
    expect(mocks.createSidepanelChatHandoff).not.toHaveBeenCalled()
    expect(mocks.tabsCreate).not.toHaveBeenCalled()
  })

  it("keeps the copy outcome visible after the installed Popover closes", async () => {
    mocks.useNativePopover = true
    render(<ConfigProvider theme={{ token: { motion: false } }}><ControlRow {...defaultProps()} draftMessage="Source draft" /></ConfigProvider>)
    const more = screen.getByRole("button", { name: "More tools" })
    fireEvent.click(more)
    expect(more).toHaveAttribute("aria-expanded", "true")
    fireEvent.click(await screen.findByRole("button", { name: "Copy draft & page references" }))
    const feedback = await screen.findByTestId("chat-handoff-feedback")
    expect(feedback).toBeVisible()
    expect(feedback).toHaveTextContent("Full chat page opened")
    await waitFor(() => expect(more).toHaveAttribute("aria-expanded", "false"))
    expect(feedback.closest(".ant-popover")).toBeNull()
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
      }, expect.any(Function))
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
      }, expect.any(Function))
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
    }, expect.any(Function))
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
      "Page references could not be read. Try again. Your source draft remains here."
    )
    expect(mocks.createSidepanelChatHandoff).not.toHaveBeenCalled()
    expect(mocks.tabsCreate).not.toHaveBeenCalled()
  })

  it("shows an error and does not open a tab when handoff creation fails", async () => {
    mocks.createSidepanelChatHandoff.mockRejectedValueOnce(new Error("storage failed"))
    render(<ControlRow {...defaultProps()} draftMessage="Summarize this" />)

    fireEvent.click(screen.getByTestId("chat-continue-in-webui"))

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Could not open the full chat page. Your source draft remains here."
    )
    expect(mocks.tabsCreate).not.toHaveBeenCalled()
    expect(window.open).not.toHaveBeenCalled()
  })

  it("falls back to window.open when Continue in WebUI tab creation rejects", async () => {
    mocks.tabsCreate.mockRejectedValueOnce(new Error("tab create blocked"))
    vi.mocked(window.open).mockImplementationOnce(() => ({ closed: false }) as Window)
    render(<ControlRow {...defaultProps()} draftMessage="Summarize this" />)

    fireEvent.click(screen.getByTestId("chat-continue-in-webui"))

    await waitFor(() => {
      expect(window.open).toHaveBeenCalledWith(
        "chrome-extension://handoff/options.html#/chat?handoff=handoff-123",
        "_blank"
      )
    })
    expect(mocks.consumeSidepanelChatHandoff).not.toHaveBeenCalled()
    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
  })

  it("consumes the handoff and shows an error when Continue in WebUI cannot open", async () => {
    mocks.tabsCreate.mockRejectedValueOnce(new Error("tab create blocked"))
    render(<ControlRow {...defaultProps()} draftMessage="Summarize this" />)

    fireEvent.click(screen.getByTestId("chat-continue-in-webui"))

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Could not open the full chat page. Your source draft remains here."
    )
    expect(mocks.consumeSidepanelChatHandoff).toHaveBeenCalledWith("handoff-123", expect.any(Function))
  })
})
