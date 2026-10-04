import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest"
import type { useMessageOption as useMessageOptionHook } from "@/hooks/useMessageOption"
import type { StagedWorkspaceSource } from "../types"
import type { EffectiveWorkspaceAssistantDefault } from "@/types/workspace"
import { contrastRatio } from "@/themes/contrast"
import { getBuiltinPresets } from "@/themes/presets"

type UseMessageOptionHook = typeof useMessageOptionHook
type UseMessageOptionState = ReturnType<UseMessageOptionHook>
type SubmitPayload = Parameters<UseMessageOptionState["onSubmit"]>[0]

const chatHookState = vi.hoisted(() => {
  const onSubmit = vi.fn<UseMessageOptionState["onSubmit"]>(
    async (): Promise<Awaited<ReturnType<UseMessageOptionState["onSubmit"]>>> => ({
      status: "submitted"
    })
  )
  const stopStreamingRequest = vi.fn()
  const value = {
    messages: [],
    onSubmit,
    streaming: false,
    isLoading: false,
    isProcessing: false,
    stopStreamingRequest,
    selectedModel: "gpt-test",
    selectedAssistant: { kind: "persona", id: "p1", name: "Analyst" },
    selectedAssistantSource: "explicit",
    serverChatId: null,
    serverChatAssistantKind: null,
    serverChatAssistantId: null,
    serverChatMetaLoaded: false
  } as unknown as UseMessageOptionState
  const useMessageOption = vi.fn<UseMessageOptionHook>(() => value)

  return { onSubmit, stopStreamingRequest, useMessageOption, value }
})

const macroServiceMocks = vi.hoisted(() => ({
  cancelChatMacroRun: vi.fn(),
  getChatMacroRun: vi.fn()
}))

const fetchChatModels = vi.hoisted(() => vi.fn())
vi.mock("@/services/tldw-server", () => ({ fetchChatModels }))
const resolveServicePromptScope = vi.hoisted(() => vi.fn())
vi.mock("@/services/service-prompts", () => ({ resolveServicePromptScope }))
const originalRequestScope = {
  config: { serverUrl: "http://127.0.0.1:8000", authMode: "multi-user" as const, authSource: "manual" as const },
  userId: 42
}
const modelSettings = vi.hoisted(() => ({ apiProvider: "openai", temperature: 0.42, topP: 0.8 }))
const themeSettings = vi.hoisted(() => ({ themeId: "default", mode: "light" }))
vi.mock("@/hooks/useSetting", () => ({
  useSetting: (setting: { key: string }) => [
    setting.key === "tldw:themePreset" ? themeSettings.themeId : []
  ]
}))
vi.mock("@/hooks/useDarkmode", () => ({
  useDarkModeStore: (selector: (state: { mode: string }) => unknown) => selector({ mode: themeSettings.mode })
}))
vi.mock("@/store/model", () => ({
  useStoreChatModelSettings: (selector?: (state: typeof modelSettings) => unknown) =>
    selector ? selector(modelSettings) : modelSettings
}))

vi.mock("@/hooks/useMessageOption", () => ({
  useMessageOption: (...args: Parameters<UseMessageOptionHook>) =>
    chatHookState.useMessageOption(...args)
}))

vi.mock("@/hooks/chat/chat-action-utils", () => ({
  isChatSubmitSuccess: (result: { status: string }) => result.status === "submitted",
  normalizeChatSubmitResult: (result?: { status: string }) =>
    result ?? { status: "submitted" }
}))

vi.mock("@/services/chat-macros", () => ({
  cancelChatMacroRun: (...args: unknown[]) =>
    macroServiceMocks.cancelChatMacroRun(...args),
  getChatMacroRun: (...args: unknown[]) => macroServiceMocks.getChatMacroRun(...args)
}))

vi.mock("../MacroRunDetailDrawer", () => ({
  MacroRunDetailDrawer: () => null
}))

vi.mock("@/components/Common/Playground/Message", () => ({
  PlaygroundMessage: (props: {
    conversationInstanceId: string
    message: string
    headingOffset?: number
    isBot?: boolean
    role?: "user" | "assistant" | "system"
    sources?: unknown[]
    hideSourceActions?: boolean
    generationInfo?: { streamTransportInterrupted?: boolean }
    recoveryActions?: Array<{
      id: string
      label: string
      onClick: () => void
      disabled?: boolean
      disabledReason?: string
    }>
  }) => (
    <article
      data-testid="workspace-panel-message"
      data-conversation-instance-id={props.conversationInstanceId}
      data-heading-offset={props.headingOffset}
      data-message-role={props.role}
      data-citation-sources={JSON.stringify(props.sources)}
      data-hide-source-actions={props.hideSourceActions}
    >
      {props.message}
      {props.isBot && (props.message === "Failed workspace answer" || props.generationInfo?.streamTransportInterrupted) &&
        props.recoveryActions?.map((action) => (
          <button key={action.id} onClick={action.onClick} disabled={action.disabled}>
            {action.label}
          </button>
        ))}
    </article>
  )
}))

let WorkspaceChatPanel: typeof import("../WorkspaceChatPanel").WorkspaceChatPanel

beforeAll(async () => {
  WorkspaceChatPanel = (await import("../WorkspaceChatPanel")).WorkspaceChatPanel
})

const staged: StagedWorkspaceSource[] = [
  {
    sourceId: "source-1",
    mediaId: 101,
    title: "Operator Notes",
    type: "document",
    scopeLabel: "Default workspace",
    availability: "ready"
  }
]

const stagedWithoutReadyMedia: StagedWorkspaceSource[] = [
  {
    sourceId: "source-processing",
    mediaId: 202,
    title: "Indexing Notes",
    type: "document",
    scopeLabel: "Default workspace",
    availability: "processing"
  }
]

const stagedWithMixedAvailability: StagedWorkspaceSource[] = [
  staged[0],
  {
    sourceId: "source-processing",
    mediaId: 202,
    title: "Indexing Notes",
    type: "document",
    scopeLabel: "Default workspace",
    availability: "processing"
  }
]

const getSubmitPayload = (): SubmitPayload => {
  const payload = chatHookState.onSubmit.mock.calls[0]?.[0]
  if (!payload) {
    throw new Error("Expected workspace chat submit payload")
  }
  return payload
}

const availableWorkspaceDefault: EffectiveWorkspaceAssistantDefault = {
  status: "available",
  source: "workspace",
  assistantKind: "persona",
  assistantId: "workspace-persona",
  label: "Workspace Analyst",
  personaMemoryMode: "read_write",
  degradedReason: null
}

const unavailableWorkspaceDefault: EffectiveWorkspaceAssistantDefault = {
  status: "unavailable",
  source: "workspace",
  assistantKind: "persona",
  assistantId: "workspace-persona",
  label: "Workspace Analyst",
  personaMemoryMode: "read_write",
  degradedReason: "persona_deleted"
}

describe("WorkspaceChatPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    modelSettings.apiProvider = "openai"
    modelSettings.temperature = 0.42
    modelSettings.topP = 0.8
    chatHookState.value.messages = []
    chatHookState.value.history = []
    chatHookState.value.setMessages = vi.fn((messages) => {
      chatHookState.value.messages = typeof messages === "function"
        ? messages(chatHookState.value.messages)
        : messages
    })
    chatHookState.value.streaming = false
    chatHookState.value.isLoading = false
    chatHookState.value.isProcessing = false
    chatHookState.value.selectedModel = "gpt-test"
    chatHookState.value.selectedAssistant = {
      kind: "persona",
      id: "p1",
      name: "Analyst"
    }
    chatHookState.value.selectedAssistantSource = "explicit"
    chatHookState.value.serverChatId = null
    chatHookState.value.serverChatAssistantKind = null
    chatHookState.value.serverChatAssistantId = null
    chatHookState.value.serverChatMetaLoaded = false
    chatHookState.value.serverChatLoadState = "idle"
    chatHookState.value.serverChatLoadError = null
    themeSettings.themeId = "default"
    themeSettings.mode = "light"
    chatHookState.value.temporaryChat = false
    chatHookState.onSubmit.mockReset().mockResolvedValue({ status: "submitted" })
    resolveServicePromptScope.mockReset().mockResolvedValue({
      ...originalRequestScope, scopeKey: "original-scope", clientPrincipalVerified: true
    })
    fetchChatModels.mockResolvedValue([
      { model: "ollama/gemma", nickname: "Gemma", provider: "ollama" },
      { model: "ollama/other", nickname: "Other local model", provider: "ollama" }
    ])
    macroServiceMocks.cancelChatMacroRun.mockResolvedValue({
      ok: true,
      status: 200,
      data: { run_id: "run-1", status: "cancel_requested" }
    })
  })

  afterEach(() => vi.unstubAllGlobals())

  const failWorkspaceTurn = () => {
    chatHookState.onSubmit.mockImplementationOnce(async (request) => {
      chatHookState.value.messages = [
        ...chatHookState.value.messages,
        { id: "user-failed", isBot: false, role: "user", message: request.message },
        { id: "assistant-failed", isBot: true, role: "assistant", message: "Failed workspace answer", parentMessageId: "user-failed" }
      ] as UseMessageOptionState["messages"]
      chatHookState.value.serverChatId = "workspace-conversation"
      return { status: "failed", errorMessage: "Provider unavailable" }
    })
  }

  it("forwards a one-level heading offset to workspace transcript messages", () => {
    chatHookState.value.messages = [{
      id: "assistant-heading", isBot: true, role: "assistant", message: "# Model response"
    }] as UseMessageOptionState["messages"]
    render(<WorkspaceChatPanel workspaceId="workspace-1" stagedSources={[]}
      backendAvailable onClearStagedSources={vi.fn()} />)
    expect(screen.getByTestId("workspace-panel-message")).toHaveAttribute("data-heading-offset", "1")
  })

  it("forwards existing RAG citation sources without projecting away metadata", () => {
    const sources = [{ title: "Operator Notes", media_id: 101, chunk_id: "chunk-1", citation_number: 1 }]
    chatHookState.value.messages = [{
      id: "grounded-answer", isBot: true, role: "assistant", message: "Answer [1]", sources
    }] as UseMessageOptionState["messages"]
    render(<WorkspaceChatPanel workspaceId="workspace-1" stagedSources={[]}
      backendAvailable onClearStagedSources={vi.fn()} />)
    expect(screen.getByTestId("workspace-panel-message")).toHaveAttribute("data-citation-sources", JSON.stringify(sources))
  })

  it("hides citation workflow commands unsupported by the standalone workspace", () => {
    chatHookState.value.messages = [{
      id: "grounded-answer", isBot: true, role: "assistant", message: "Answer [1]",
      sources: [{ name: "Operator Notes", content: "Verified evidence" }]
    }] as UseMessageOptionState["messages"]
    render(<WorkspaceChatPanel workspaceId="workspace-1" stagedSources={[]}
      backendAvailable onClearStagedSources={vi.fn()} />)
    expect(screen.getByTestId("workspace-panel-message")).toHaveAttribute("data-hide-source-actions", "true")
  })

  it("keeps an explicit system role when a restored row is not a bot", () => {
    chatHookState.value.messages = [{
      id: "system-row", isBot: false, role: "system", name: "System", message: "System instructions", sources: []
    }] as UseMessageOptionState["messages"]
    render(<WorkspaceChatPanel workspaceId="workspace-1" stagedSources={[]}
      backendAvailable onClearStagedSources={vi.fn()} />)
    expect(screen.getByTestId("workspace-panel-message")).toHaveAttribute("data-message-role", "system")
  })

  it.each(["isLoading", "isProcessing"] as const)("publishes %s as sending", (busyField) => {
    const onRuntimeStateChange = vi.fn()
    const props = { workspaceId: "workspace-1", stagedSources: [], backendAvailable: true,
      onClearStagedSources: vi.fn(), onRuntimeStateChange }
    const { rerender } = render(<WorkspaceChatPanel {...props} />)
    chatHookState.value[busyField] = true
    rerender(<WorkspaceChatPanel {...props} />)
    expect(onRuntimeStateChange).toHaveBeenLastCalledWith(expect.objectContaining({ sending: true }))
    chatHookState.value[busyField] = false
    rerender(<WorkspaceChatPanel {...props} />)
    expect(onRuntimeStateChange).toHaveBeenLastCalledWith(expect.objectContaining({ sending: false }))
  })

  it.each([false, true])("blocks submissions and publishes pending request preparation (staged: %s)", async (withStagedContext) => {
    let resolveScope!: (value: typeof originalRequestScope) => void
    resolveServicePromptScope.mockImplementationOnce(() => new Promise((resolve) => { resolveScope = resolve }))
    const onRuntimeStateChange = vi.fn()
    render(<WorkspaceChatPanel workspaceId="workspace-1" stagedSources={withStagedContext ? staged : []}
      backendAvailable onClearStagedSources={vi.fn()} onRuntimeStateChange={onRuntimeStateChange} />)
    const composer = screen.getByRole("textbox", { name: "Chat workspace message" })
    fireEvent.change(composer, { target: { value: "Prepare this request" } })
    fireEvent.click(screen.getByRole("button", { name: "Send message" }))

    expect(onRuntimeStateChange).toHaveBeenLastCalledWith(expect.objectContaining({ sending: true, streaming: false }))
    expect(screen.getByText("Sending")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Send message" })).toBeDisabled()
    if (withStagedContext) {
      expect(screen.getByRole("button", { name: "Send with staged context" })).toBeDisabled()
    }
    fireEvent.keyDown(composer, { key: "Enter", ctrlKey: true })
    fireEvent.submit(composer.closest("form")!)
    expect(resolveServicePromptScope).toHaveBeenCalledTimes(1)
    expect(chatHookState.onSubmit).not.toHaveBeenCalled()

    await act(async () => resolveScope(originalRequestScope))

    expect(chatHookState.onSubmit).toHaveBeenCalledTimes(1)
    expect(onRuntimeStateChange).toHaveBeenLastCalledWith(expect.objectContaining({ sending: false }))
    expect(screen.queryByText("Sending")).not.toBeInTheDocument()
    fireEvent.change(composer, { target: { value: "Next request" } })
    expect(screen.getByRole("button", { name: "Send message" })).toBeEnabled()
  })

  it("clears preparation after a failed scope lookup while preserving the draft", async () => {
    let rejectScope!: (error: Error) => void
    resolveServicePromptScope.mockImplementationOnce(() => new Promise((_resolve, reject) => { rejectScope = reject }))
    const onRuntimeStateChange = vi.fn()
    render(<WorkspaceChatPanel workspaceId="workspace-1" stagedSources={[]}
      backendAvailable onClearStagedSources={vi.fn()} onRuntimeStateChange={onRuntimeStateChange} />)
    const composer = screen.getByRole("textbox", { name: "Chat workspace message" })
    fireEvent.change(composer, { target: { value: "Preserved request" } })
    fireEvent.click(screen.getByRole("button", { name: "Send message" }))
    expect(onRuntimeStateChange).toHaveBeenLastCalledWith(expect.objectContaining({ sending: true }))

    await act(async () => rejectScope(new Error("Scope lookup unavailable")))

    expect(onRuntimeStateChange).toHaveBeenLastCalledWith(expect.objectContaining({ sending: false }))
    expect(screen.queryByText("Sending")).not.toBeInTheDocument()
    expect(composer).toHaveValue("Preserved request")
    expect(screen.getByRole("button", { name: "Send message" })).toBeEnabled()
    expect(screen.getByRole("alert")).toHaveTextContent("Scope lookup unavailable")
    expect(chatHookState.onSubmit).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole("button", { name: "Send message" }))
    await waitFor(() => expect(chatHookState.onSubmit).toHaveBeenCalledTimes(1))
  })

  it.each(["resolved", "rejected"] as const)("keeps newer preparation owned after an ABA workspace switch and stale lookup %s", async (outcome) => {
    let resolveOldScope!: (value: typeof originalRequestScope) => void
    let rejectOldScope!: (error: Error) => void
    let resolveNewScope!: (value: typeof originalRequestScope) => void
    resolveServicePromptScope
      .mockImplementationOnce(() => new Promise((resolve, reject) => { resolveOldScope = resolve; rejectOldScope = reject }))
      .mockImplementationOnce(() => new Promise((resolve) => { resolveNewScope = resolve }))
    const onRuntimeStateChange = vi.fn()
    const props = { stagedSources: [], backendAvailable: true, onClearStagedSources: vi.fn(), onRuntimeStateChange }
    const { rerender } = render(<WorkspaceChatPanel {...props} workspaceId="workspace-1" />)
    const composer = screen.getByRole("textbox", { name: "Chat workspace message" })
    fireEvent.change(composer, { target: { value: "Old scoped request" } })
    fireEvent.click(screen.getByRole("button", { name: "Send message" }))
    expect(onRuntimeStateChange).toHaveBeenLastCalledWith(expect.objectContaining({ sending: true }))

    rerender(<WorkspaceChatPanel {...props} workspaceId="workspace-2" />)
    expect(onRuntimeStateChange).toHaveBeenLastCalledWith(expect.objectContaining({ sending: false }))
    expect(screen.queryByText("Sending")).not.toBeInTheDocument()
    rerender(<WorkspaceChatPanel {...props} workspaceId="workspace-1" />)
    fireEvent.change(composer, { target: { value: "New scoped request" } })
    fireEvent.click(screen.getByRole("button", { name: "Send message" }))

    await act(async () => {
      if (outcome === "resolved") resolveOldScope(originalRequestScope)
      else rejectOldScope(new Error("Stale scope lookup unavailable"))
    })

    expect(onRuntimeStateChange).toHaveBeenLastCalledWith(expect.objectContaining({ sending: true }))
    expect(screen.getByRole("button", { name: "Send message" })).toBeDisabled()
    expect(screen.getByText("Sending")).toBeInTheDocument()
    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
    expect(composer).toHaveValue("New scoped request")
    fireEvent.keyDown(composer, { key: "Enter", ctrlKey: true })
    expect(resolveServicePromptScope).toHaveBeenCalledTimes(2)
    expect(chatHookState.onSubmit).not.toHaveBeenCalled()

    await act(async () => resolveNewScope(originalRequestScope))

    expect(chatHookState.onSubmit).toHaveBeenCalledTimes(1)
    expect(getSubmitPayload().message).toBe("New scoped request")
    expect(onRuntimeStateChange).toHaveBeenLastCalledWith(expect.objectContaining({ sending: false }))
    expect(screen.queryByText("Sending")).not.toBeInTheDocument()
  })

  it.each([
    { state: "loading" as const, error: null, loading: true, expectedError: null },
    { state: "failed" as const, error: "History unavailable", loading: false, expectedError: "History unavailable" },
    { state: "failed" as const, error: null, loading: false, expectedError: "Chat history unavailable" }
  ])("publishes and blocks incomplete history (%j)", ({ state, error, loading, expectedError }) => {
    chatHookState.value.serverChatLoadState = state
    chatHookState.value.serverChatLoadError = error
    const onRuntimeStateChange = vi.fn()
    render(<WorkspaceChatPanel workspaceId="workspace-1" stagedSources={staged}
      backendAvailable onClearStagedSources={vi.fn()} onRuntimeStateChange={onRuntimeStateChange} />)
    expect(onRuntimeStateChange).toHaveBeenLastCalledWith(expect.objectContaining({
      historyLoading: loading, historyLoadError: expectedError
    }))
    expect(screen.getByRole("button", { name: "Send message" })).toBeDisabled()
    expect(screen.getByRole("button", { name: "Send with staged context" })).toBeDisabled()
  })

  it("does not hydrate or send for an unhydrated workspace with an identity", () => {
    render(<WorkspaceChatPanel workspaceId="workspace-1" workspaceReady={false}
      stagedSources={staged} backendAvailable onClearStagedSources={vi.fn()} />)
    expect(chatHookState.useMessageOption).toHaveBeenCalledWith(expect.objectContaining({ hydrateServerChat: false }))
    expect(screen.getByRole("button", { name: "Send message" })).toBeDisabled()
  })

  it("publishes pending recovery without changing the global model label", async () => {
    failWorkspaceTurn()
    const onRuntimeStateChange = vi.fn()
    render(<WorkspaceChatPanel workspaceId="workspace-1" stagedSources={[]}
      backendAvailable onClearStagedSources={vi.fn()} onRuntimeStateChange={onRuntimeStateChange} />)
    fireEvent.change(screen.getByRole("textbox", { name: "Chat workspace message" }), { target: { value: "Retry this" } })
    fireEvent.click(screen.getByRole("button", { name: "Send message" }))
    await screen.findByText("Provider unavailable")
    fireEvent.click(screen.getByRole("button", { name: "Switch model" }))
    await screen.findByRole("option", { name: "Gemma (ollama)" })
    fireEvent.change(screen.getByRole("combobox", { name: "Retry model" }), { target: { value: JSON.stringify(["ollama", "ollama/gemma"]) } })
    let resolveRetry!: (value: { status: "submitted" }) => void
    chatHookState.onSubmit.mockImplementationOnce(() => new Promise((resolve) => { resolveRetry = resolve }))
    fireEvent.click(screen.getByRole("button", { name: "Retry with selected model" }))
    expect(onRuntimeStateChange).toHaveBeenLastCalledWith(expect.objectContaining({ sending: true, selectedModelLabel: "gpt-test" }))
    await act(async () => resolveRetry({ status: "submitted" }))
    expect(onRuntimeStateChange).toHaveBeenLastCalledWith(expect.objectContaining({ sending: false, selectedModelLabel: "gpt-test" }))
  })

  it.each(getBuiltinPresets().flatMap((preset) => ["light", "dark"].map((mode) => ({ preset, mode }))))(
    "keeps enabled Send text AA for $preset.id/$mode", ({ preset, mode }) => {
      themeSettings.themeId = preset.id
      themeSettings.mode = mode
      const { container } = render(<WorkspaceChatPanel workspaceId="workspace-1" stagedSources={staged}
        backendAvailable onClearStagedSources={vi.fn()} />)
      const utilities = document.createElement("style")
      utilities.textContent = ".text-white { color: rgb(255, 255, 255); }"
      container.appendChild(utilities)
      const button = screen.getByRole("button", { name: "Send message" })
      expect(button).toBeEnabled()
      const foreground = getComputedStyle(button).color.match(/\d+/g)?.slice(0, 3).join(" ") || "255 255 255"
      const background = preset.palette[mode as "light" | "dark"].primaryStrong
      expect(contrastRatio(foreground, background)).toBeGreaterThanOrEqual(4.5)
    }
  )

  it.each([
    { switchModel: false, bound: false },
    { switchModel: true, bound: false },
    { switchModel: false, bound: true },
    { switchModel: true, bound: true }
  ])("retains the original request scope for recovery (%j)", async ({ switchModel, bound }) => {
    if (bound) failWorkspaceTurn()
    else chatHookState.onSubmit.mockResolvedValueOnce({ status: "failed", errorMessage: "Provider unavailable" })
    render(<WorkspaceChatPanel workspaceId="workspace-1" stagedSources={[]}
      onClearStagedSources={vi.fn()} backendAvailable />)
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "Original question" } })
    fireEvent.click(screen.getByRole("button", { name: "Send message" }))
    await screen.findByText("Provider unavailable")

    resolveServicePromptScope.mockResolvedValue({
      config: { serverUrl: "https://other.example.test", authMode: "multi-user" },
      userId: 84, scopeKey: "other-scope", clientPrincipalVerified: true
    })
    if (switchModel) {
      fireEvent.click(screen.getByRole("button", { name: "Switch model" }))
      const selector = await screen.findByRole("combobox", { name: "Retry model" })
      const option = await screen.findByRole<HTMLOptionElement>("option", { name: /Other local model/ })
      fireEvent.change(selector, { target: { value: option.value } })
      fireEvent.click(screen.getByRole("button", { name: "Retry with selected model" }))
    } else fireEvent.click(screen.getByRole("button", { name: "Retry same model" }))

    await waitFor(() => expect(chatHookState.onSubmit).toHaveBeenCalledTimes(2))
    expect(getSubmitPayload().requestOverrides?.requestScope).toEqual(originalRequestScope)
    expect(chatHookState.onSubmit.mock.calls[1][0].requestOverrides?.requestScope).toEqual(originalRequestScope)
    expect(resolveServicePromptScope).toHaveBeenCalledTimes(1)
  })

  it.each(["unavailable", "throws"])("keeps Send usable after UUID allocation %s", async (failure) => {
    const cryptoApi = globalThis.crypto
    const clearStaged = vi.fn()
    vi.stubGlobal("crypto", {
      randomUUID: failure === "throws" ? () => { throw new Error("Random source failed") } : undefined
    })
    render(<WorkspaceChatPanel workspaceId="workspace-1" stagedSources={staged}
      onClearStagedSources={clearStaged} backendAvailable />)
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "Keep this question" } })
    fireEvent.click(screen.getByRole("button", { name: "Send message" }))

    expect(await screen.findByRole("alert")).toHaveTextContent(/secure|HTTPS|localhost/i)
    expect(screen.getByRole("textbox")).toHaveValue("Keep this question")
    expect(chatHookState.onSubmit).not.toHaveBeenCalled()
    expect(clearStaged).not.toHaveBeenCalled()

    vi.stubGlobal("crypto", cryptoApi)
    fireEvent.click(screen.getByRole("button", { name: "Send message" }))
    await waitFor(() => expect(chatHookState.onSubmit).toHaveBeenCalledTimes(1))
    expect(getSubmitPayload().requestOverrides?.tldwTurn).toEqual({
      user_message_id: expect.stringMatching(/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/)
    })
  })

  it.each([{ sources: [] }, { sources: staged }])("retries the captured failed turn without duplicating its user or consuming current input (%j)", async ({ sources }) => {
    const clearStaged = vi.fn()
    chatHookState.value.selectedAssistantSource = "workspace"
    chatHookState.value.selectedAssistant = null
    failWorkspaceTurn()
    const { rerender } = render(
      <WorkspaceChatPanel workspaceId="workspace-1" stagedSources={sources}
        effectiveAssistantDefault={availableWorkspaceDefault}
        onClearStagedSources={clearStaged} backendAvailable />
    )
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "Original question" } })
    fireEvent.click(screen.getByRole("button", { name: "Send message" }))
    await screen.findByText("Provider unavailable")
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "New unsent draft" } })
    chatHookState.value.selectedModel = "global-model-changed"
    rerender(
      <WorkspaceChatPanel workspaceId="workspace-1" stagedSources={stagedWithMixedAvailability}
        effectiveAssistantDefault={availableWorkspaceDefault}
        onClearStagedSources={clearStaged} backendAvailable />
    )
    let finishRetry!: (result: { status: "submitted" }) => void
    chatHookState.onSubmit.mockImplementationOnce(() => new Promise((resolve) => { finishRetry = resolve }))
    const retry = screen.getByRole("button", { name: "Retry same model" })
    fireEvent.click(retry)
    fireEvent.click(retry)
    expect(chatHookState.onSubmit).toHaveBeenCalledTimes(2)
    const firstTurn = chatHookState.onSubmit.mock.calls[0][0].requestOverrides?.tldwTurn
    expect(firstTurn).toEqual({ user_message_id: expect.stringMatching(/^[0-9a-f-]{36}$/) })
    expect(chatHookState.onSubmit.mock.calls[1][0].requestOverrides?.tldwTurn).toEqual(firstTurn)
    expect(chatHookState.onSubmit.mock.calls[1][0]).toMatchObject({
      message: "Original question",
      isRegenerate: true,
      memory: [],
      messages: [{ id: "user-failed", message: "Original question" }],
      serverChatIdOverride: "workspace-conversation",
      regenerateFromMessage: { id: "assistant-failed", parentMessageId: "user-failed" },
      requestOverrides: {
        selectedModel: "gpt-test",
        ragMediaIds: sources.length ? [101] : [],
        fileRetrievalEnabled: sources.length > 0,
        chatMode: sources.length ? "rag" : "normal",
        assistant_kind: "persona",
        assistant_id: "workspace-persona",
        persona_memory_mode: "read_write"
      }
    })
    expect(screen.getByRole("textbox")).toHaveValue("New unsent draft")
    expect(clearStaged).not.toHaveBeenCalled()
    finishRetry({ status: "submitted" })
    await waitFor(() => expect(screen.queryByText("Provider unavailable")).not.toBeInTheDocument())
    expect(screen.getByRole("textbox")).toHaveValue("New unsent draft")
    expect(clearStaged).not.toHaveBeenCalled()
  })

  it("opens a local model selector and retries the failed request with its choice", async () => {
    failWorkspaceTurn()
    render(<WorkspaceChatPanel workspaceId="workspace-1" stagedSources={staged}
      onClearStagedSources={vi.fn()} backendAvailable />)
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "Original question" } })
    fireEvent.click(screen.getByRole("button", { name: "Send message" }))
    await screen.findByText("Provider unavailable")
    modelSettings.temperature = 0.9
    fireEvent.click(screen.getByRole("button", { name: "Switch model" }))
    const selector = await screen.findByRole("combobox", { name: "Retry model" })
    const modelOption = await screen.findByRole<HTMLOptionElement>("option", { name: /Other local model/ })
    expect(selector).toHaveFocus()
    fireEvent.change(selector, { target: { value: modelOption.value } })
    fireEvent.click(screen.getByRole("button", { name: "Retry with selected model" }))
    await waitFor(() => expect(chatHookState.onSubmit).toHaveBeenCalledTimes(2))
    expect(chatHookState.onSubmit.mock.calls[0][0].requestOverrides?.tldwTurn).toBeDefined()
    expect(chatHookState.onSubmit.mock.calls[1][0].requestOverrides?.tldwTurn)
      .toEqual(chatHookState.onSubmit.mock.calls[0][0].requestOverrides?.tldwTurn)
    expect(chatHookState.onSubmit.mock.calls[1][0]).toMatchObject({
      message: "Original question", isRegenerate: true,
      serverChatIdOverride: "workspace-conversation",
      requestOverrides: { selectedModel: "ollama/other", ragMediaIds: [101] }
    })
    expect(chatHookState.value.selectedModel).toBe("gpt-test")
    expect(screen.getByRole("textbox")).toHaveValue("Original question")
  })

  it.each(["failed", "throws", "switch", "cancelled"])("restores the original partial and recovery request after an early retry %s", async (failure) => {
    chatHookState.onSubmit.mockImplementationOnce(async (request) => {
      chatHookState.value.messages = [
        { id: "user-partial", isBot: false, message: request.message },
        { id: "assistant-partial", isBot: true, message: "Partial workspace answer", parentMessageId: "user-partial",
          generationInfo: { interrupted: true, streamTransportInterrupted: true } }
      ] as UseMessageOptionState["messages"]
      chatHookState.value.serverChatId = "workspace-conversation"
      return { status: "skipped", reason: "Connection dropped" }
    })
    render(<WorkspaceChatPanel workspaceId="workspace-1" stagedSources={[]}
      onClearStagedSources={vi.fn()} backendAvailable />)
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "Original question" } })
    fireEvent.click(screen.getByRole("button", { name: "Send message" }))
    await screen.findByText("Connection dropped")
    const originalMessages = [...chatHookState.value.messages]
    const originalOverrides = getSubmitPayload().requestOverrides
    chatHookState.onSubmit.mockImplementationOnce(async () => {
      if (failure === "throws") throw new Error("Conversation lookup failed (503)")
      if (failure === "cancelled") return { status: "skipped", reason: "Request cancelled" }
      return { status: "failed", errorMessage: "Conversation lookup failed (503)" }
    })
    if (failure === "switch") {
      fireEvent.click(screen.getByRole("button", { name: "Switch model" }))
      const option = await screen.findByRole<HTMLOptionElement>("option", { name: /Other local model/ })
      fireEvent.change(screen.getByRole("combobox"), { target: { value: option.value } })
      fireEvent.click(screen.getByRole("button", { name: "Retry with selected model" }))
    } else fireEvent.click(screen.getByRole("button", { name: "Retry same model" }))

    if (failure === "cancelled") await waitFor(() => expect(screen.queryByText("Connection dropped")).not.toBeInTheDocument())
    else await screen.findByText(failure === "throws" ? "Send failed" : "Conversation lookup failed (503)")
    expect(chatHookState.value.messages).toEqual(originalMessages)
    expect(screen.getAllByText("Partial workspace answer")).toHaveLength(1)
    fireEvent.click(screen.getByRole("button", { name: "Retry same model" }))
    await waitFor(() => expect(chatHookState.onSubmit).toHaveBeenCalledTimes(3))
    expect(chatHookState.onSubmit.mock.calls[2][0]).toMatchObject({
      requestOverrides: originalOverrides,
      regenerateFromMessage: originalMessages[1],
      serverChatIdOverride: "workspace-conversation"
    })
  })

  it.each(["failed", "throws", "submitted", "interrupted"])("keeps a replacement assistant without restoring a duplicate after retry %s", async (status) => {
    failWorkspaceTurn()
    render(<WorkspaceChatPanel workspaceId="workspace-1" stagedSources={[]}
      onClearStagedSources={vi.fn()} backendAvailable />)
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "Original question" } })
    fireEvent.click(screen.getByRole("button", { name: "Send message" }))
    await screen.findByText("Provider unavailable")
    chatHookState.onSubmit.mockImplementationOnce(async (request) => {
      chatHookState.value.messages = [...request.messages!, {
        id: "replacement-assistant", isBot: true, message: "New partial answer", parentMessageId: "user-failed",
        serverMessageId: "persisted-replacement", generationInfo: { interrupted: true, streamTransportInterrupted: true }
      }] as UseMessageOptionState["messages"]
      if (status === "throws") throw new Error("Connection dropped")
      if (status === "interrupted") return { status: "skipped", reason: "Connection dropped" }
      return status === "submitted" ? { status: "submitted" } : { status: "failed", errorMessage: "Connection dropped" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Retry same model" }))
    await waitFor(() => expect(screen.queryByText("Provider unavailable")).not.toBeInTheDocument())
    expect(chatHookState.value.messages.map((message) => message.id)).toEqual(["user-failed", "replacement-assistant"])
    expect(screen.getAllByText("New partial answer")).toHaveLength(1)
  })

  it("gives the model selector a full mobile row before its recovery buttons", async () => {
    failWorkspaceTurn()
    render(<WorkspaceChatPanel workspaceId="workspace-1" stagedSources={staged}
      onClearStagedSources={vi.fn()} backendAvailable />)
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "Original question" } })
    fireEvent.click(screen.getByRole("button", { name: "Send message" }))
    await screen.findByText("Provider unavailable")
    fireEvent.click(screen.getByRole("button", { name: "Switch model" }))
    const selector = await screen.findByRole("combobox", { name: "Retry model" })
    expect(screen.getByRole("group", { name: "Failed request model" })).toHaveClass("flex-wrap")
    expect(selector.closest("label")).toHaveClass("w-full", "flex-none", "sm:w-auto", "sm:flex-1")
  })

  it("assigns distinct durable identities to new submissions with identical text", async () => {
    render(<WorkspaceChatPanel workspaceId="workspace-1" stagedSources={[]}
      onClearStagedSources={vi.fn()} backendAvailable />)
    for (let index = 0; index < 2; index += 1) {
      fireEvent.change(screen.getByRole("textbox"), { target: { value: "Same question" } })
      fireEvent.click(screen.getByRole("button", { name: "Send message" }))
      await waitFor(() => expect(screen.getByRole("textbox")).toHaveValue(""))
    }
    const [first, second] = chatHookState.onSubmit.mock.calls.map(([request]) => request.requestOverrides?.tldwTurn)
    expect(first).toBeDefined()
    expect(second).toBeDefined()
    expect(first).not.toEqual(second)
  })

  it("does not force durable persistence for temporary chats", async () => {
    chatHookState.value.temporaryChat = true
    render(<WorkspaceChatPanel workspaceId="workspace-1" stagedSources={[]}
      onClearStagedSources={vi.fn()} backendAvailable />)
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "Temporary question" } })
    fireEvent.click(screen.getByRole("button", { name: "Send message" }))
    await waitFor(() => expect(chatHookState.onSubmit).toHaveBeenCalledTimes(1))
    expect(getSubmitPayload().requestOverrides?.tldwTurn).toBeUndefined()
  })

  it("does not recover a failed turn into a different conversation", async () => {
    failWorkspaceTurn()
    const props = { workspaceId: "workspace-1", stagedSources: staged, onClearStagedSources: vi.fn(), backendAvailable: true }
    const { rerender } = render(<WorkspaceChatPanel {...props} />)
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "Original question" } })
    fireEvent.click(screen.getByRole("button", { name: "Send message" }))
    await screen.findByText("Provider unavailable")
    chatHookState.value.serverChatId = "different-conversation"
    rerender(<WorkspaceChatPanel {...props} />)
    const retry = screen.queryByRole("button", { name: "Retry same model" })
    if (retry) {
      expect(retry).toBeDisabled()
      fireEvent.click(retry)
    }
    expect(chatHookState.onSubmit).toHaveBeenCalledTimes(1)
  })

  it("keeps duplicate model IDs distinct and overrides the failed request provider locally", async () => {
    fetchChatModels.mockResolvedValue([
      { model: "shared-model", provider: "openai" },
      { model: "shared-model", provider: "ollama" }
    ])
    failWorkspaceTurn()
    render(<WorkspaceChatPanel workspaceId="workspace-1" stagedSources={staged}
      onClearStagedSources={vi.fn()} backendAvailable />)
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "Original question" } })
    fireEvent.click(screen.getByRole("button", { name: "Send message" }))
    await screen.findByText("Provider unavailable")
    fireEvent.click(screen.getByRole("button", { name: "Switch model" }))
    const openai = await screen.findByRole<HTMLOptionElement>("option", { name: "shared-model (openai)" })
    const ollama = screen.getByRole<HTMLOptionElement>("option", { name: "shared-model (ollama)" })
    expect(openai.value).not.toBe(ollama.value)
    fireEvent.change(screen.getByRole("combobox", { name: "Retry model" }), { target: { value: ollama.value } })
    fireEvent.click(screen.getByRole("button", { name: "Retry with selected model" }))
    await waitFor(() => expect(chatHookState.onSubmit).toHaveBeenCalledTimes(2))
    expect(chatHookState.onSubmit.mock.calls[1][0].requestOverrides).toMatchObject({
      selectedModel: "shared-model", ragMediaIds: [101],
      currentChatModelSettings: { apiProvider: "ollama", temperature: 0.42, topP: 0.8 }
    })
    expect(modelSettings.apiProvider).toBe("openai")
  })

  it("retries the original provider even after the global provider changes", async () => {
    failWorkspaceTurn()
    const props = { workspaceId: "workspace-1", stagedSources: staged, onClearStagedSources: vi.fn(), backendAvailable: true }
    const { rerender } = render(<WorkspaceChatPanel {...props} />)
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "Original question" } })
    fireEvent.click(screen.getByRole("button", { name: "Send message" }))
    await screen.findByText("Provider unavailable")
    modelSettings.apiProvider = "anthropic"
    rerender(<WorkspaceChatPanel {...props} />)
    fireEvent.click(screen.getByRole("button", { name: "Retry same model" }))
    await waitFor(() => expect(chatHookState.onSubmit).toHaveBeenCalledTimes(2))
    expect(chatHookState.onSubmit.mock.calls[1][0].requestOverrides).toMatchObject({
      selectedModel: "gpt-test",
      currentChatModelSettings: { apiProvider: "openai", temperature: 0.42, topP: 0.8 }
    })
    expect(modelSettings.apiProvider).toBe("anthropic")
  })

  it.each([
    { reason: "Stream transport interrupted; partial response saved.", recoverable: true },
    { reason: "Request cancelled", recoverable: false },
    { reason: "Request scope changed", recoverable: false }
  ])("only offers recovery for an unintended partial interruption: $reason", async ({ reason, recoverable }) => {
    const clearStaged = vi.fn()
    chatHookState.onSubmit.mockImplementationOnce(async (request) => {
      chatHookState.value.messages = [
        { id: "user-partial", isBot: false, role: "user", message: request.message },
        {
          id: "assistant-partial", isBot: true, role: "assistant", message: "Partial workspace answer",
          parentMessageId: "user-partial",
          generationInfo: { interrupted: true, streamTransportInterrupted: true, partialResponseSaved: true }
        }
      ] as UseMessageOptionState["messages"]
      return { status: "skipped", reason }
    })
    const props = { workspaceId: "workspace-1", stagedSources: staged, onClearStagedSources: clearStaged, backendAvailable: true }
    const { rerender } = render(<WorkspaceChatPanel {...props} />)
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "Original question" } })
    await act(async () => { fireEvent.click(screen.getByRole("button", { name: "Send message" })) })
    rerender(<WorkspaceChatPanel {...props} />)
    if (recoverable) {
      fireEvent.click(screen.getByRole("button", { name: "Retry same model" }))
      await waitFor(() => expect(chatHookState.onSubmit).toHaveBeenCalledTimes(2))
      expect(chatHookState.onSubmit.mock.calls[1][0]).toMatchObject({
        isRegenerate: true, message: "Original question",
        regenerateFromMessage: { id: "assistant-partial" }
      })
    } else {
      expect(screen.queryByRole("button", { name: "Retry same model" })).not.toBeInTheDocument()
    }
    expect(screen.getByRole("textbox")).toHaveValue("Original question")
    expect(clearStaged).not.toHaveBeenCalled()
  })

  it("never adopts an unrelated conversation created after a preflight failure", async () => {
    chatHookState.onSubmit.mockResolvedValueOnce({ status: "failed", errorMessage: "Preflight failed" })
    const props = { workspaceId: "workspace-1", stagedSources: staged, onClearStagedSources: vi.fn(), backendAvailable: true }
    const { rerender } = render(<WorkspaceChatPanel {...props} />)
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "Original question" } })
    fireEvent.click(screen.getByRole("button", { name: "Send message" }))
    await screen.findByText("Preflight failed")
    expect(screen.getByRole("button", { name: "Retry same model" })).toBeEnabled()
    chatHookState.value.serverChatId = "unrelated-later-conversation"
    rerender(<WorkspaceChatPanel {...props} />)
    const retry = screen.queryByRole("button", { name: "Retry same model" })
    if (retry) {
      expect(retry).toBeDisabled()
      fireEvent.click(retry)
    }
    expect(chatHookState.onSubmit).toHaveBeenCalledTimes(1)
  })

  it.each([false, true])("waits for delayed interruption metadata without following a changed conversation (%s)", async (changeConversation) => {
    chatHookState.value.serverChatId = "original-conversation"
    chatHookState.onSubmit.mockImplementationOnce(async (request) => {
      chatHookState.value.messages = [
        { id: "user-delayed", isBot: false, role: "user", message: request.message },
        { id: "assistant-delayed", isBot: true, role: "assistant", message: "Partial workspace answer", parentMessageId: "user-delayed" }
      ] as UseMessageOptionState["messages"]
      return { status: "skipped", reason: "Connection dropped" }
    })
    const props = { workspaceId: "workspace-1", stagedSources: staged, onClearStagedSources: vi.fn(), backendAvailable: true }
    const { rerender } = render(<WorkspaceChatPanel {...props} />)
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "Original question" } })
    await act(async () => { fireEvent.click(screen.getByRole("button", { name: "Send message" })) })
    expect(screen.queryByRole("button", { name: "Retry same model" })).not.toBeInTheDocument()
    if (changeConversation) chatHookState.value.serverChatId = "unrelated-conversation"
    chatHookState.value.messages = chatHookState.value.messages.map((message) => message.isBot ? {
      ...message,
      generationInfo: { interrupted: true, streamTransportInterrupted: true, partialResponseSaved: true }
    } : message)
    rerender(<WorkspaceChatPanel {...props} />)
    const retry = screen.queryByRole("button", { name: "Retry same model" })
    if (changeConversation) {
      if (retry) expect(retry).toBeDisabled()
      expect(chatHookState.onSubmit).toHaveBeenCalledTimes(1)
    } else {
      expect(retry).toBeEnabled()
      fireEvent.click(retry!)
      await waitFor(() => expect(chatHookState.onSubmit).toHaveBeenCalledTimes(2))
      expect(chatHookState.onSubmit.mock.calls[1][0]).toMatchObject({
        isRegenerate: true, serverChatIdOverride: "original-conversation",
        regenerateFromMessage: { id: "assistant-delayed" }
      })
    }
  })

  it("recovers a character turn after its seeded greeting and preserves both greeting and user IDs", async () => {
    chatHookState.value.selectedAssistant = { kind: "character", id: "character-1", name: "Character" }
    chatHookState.onSubmit.mockImplementationOnce(async (request) => {
      chatHookState.value.messages = [
        { id: "greeting-1", isBot: true, role: "assistant", message: "Hello", messageType: "character:greeting" },
        { id: "user-character", isBot: false, role: "user", message: request.message },
        { id: "error-character", isBot: true, role: "assistant", message: "Failed workspace answer", parentMessageId: "user-character" }
      ] as UseMessageOptionState["messages"]
      chatHookState.value.serverChatId = "character-conversation"
      return { status: "failed", errorMessage: "Provider unavailable" }
    })
    render(<WorkspaceChatPanel workspaceId="workspace-1" stagedSources={[]}
      onClearStagedSources={vi.fn()} backendAvailable />)
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "Original question" } })
    fireEvent.click(screen.getByRole("button", { name: "Send message" }))
    await screen.findByText("Provider unavailable")
    fireEvent.click(screen.getByRole("button", { name: "Retry same model" }))
    await waitFor(() => expect(chatHookState.onSubmit).toHaveBeenCalledTimes(2))
    expect(chatHookState.onSubmit.mock.calls[1][0]).toMatchObject({
      isRegenerate: true,
      serverChatIdOverride: "character-conversation",
      messages: [{ id: "greeting-1" }, { id: "user-character" }],
      regenerateFromMessage: { id: "error-character", parentMessageId: "user-character" }
    })
  })

  it("does not recover a different user turn with identical text in the same conversation", async () => {
    failWorkspaceTurn()
    const props = { workspaceId: "workspace-1", stagedSources: staged, onClearStagedSources: vi.fn(), backendAvailable: true }
    const { rerender } = render(<WorkspaceChatPanel {...props} />)
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "Original question" } })
    fireEvent.click(screen.getByRole("button", { name: "Send message" }))
    await screen.findByText("Provider unavailable")
    chatHookState.value.messages = chatHookState.value.messages.map((message) => ({
      ...message, id: `other-${message.id}`, parentMessageId: message.isBot ? "other-user-failed" : null
    }))
    rerender(<WorkspaceChatPanel {...props} />)
    const retry = screen.queryByRole("button", { name: "Retry same model" })
    if (retry) {
      expect(retry).toBeDisabled()
      fireEvent.click(retry)
    }
    expect(chatHookState.onSubmit).toHaveBeenCalledTimes(1)
  })

  it("renders an active macro response as a status card", () => {
    chatHookState.value.messages = [
      {
        id: "macro-status-1",
        role: "assistant",
        message: "Started /wrapup.",
        metadataExtra: {
          chat_macro: {
            run_id: "run-1",
            command: "wrapup",
            status: "pending",
            detail_url: "/api/v1/chat/macros/runs/run-1",
            output_profile: "default"
          }
        }
      }
    ] as unknown as UseMessageOptionState["messages"]

    render(
      <WorkspaceChatPanel
        stagedSources={[]}
        onClearStagedSources={vi.fn()}
        backendAvailable
        workspaceId="workspace-1"
      />
    )

    expect(screen.getByRole("article", { name: "/wrapup macro run pending" })).toBeVisible()
    expect(screen.queryByTestId("workspace-panel-message")).not.toBeInTheDocument()
  })

  it("applies a successful macro cancellation to the status card", async () => {
    chatHookState.value.messages = [
      {
        id: "macro-status-1",
        role: "assistant",
        message: "Started /wrapup.",
        metadataExtra: {
          chat_macro: { run_id: "run-1", command: "wrapup", status: "running" }
        }
      }
    ] as unknown as UseMessageOptionState["messages"]
    render(
      <WorkspaceChatPanel
        stagedSources={[]}
        onClearStagedSources={vi.fn()}
        backendAvailable
        workspaceId="workspace-1"
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Cancel macro run" }))

    await waitFor(() =>
      expect(macroServiceMocks.cancelChatMacroRun).toHaveBeenCalledWith("run-1")
    )
    expect(
      screen.getByRole("article", { name: "/wrapup macro run cancel_requested" })
    ).toBeVisible()
    expect(screen.queryByRole("button", { name: "Cancel macro run" }))
      .not.toBeInTheDocument()
  })

  it("shows an error when macro cancellation fails", async () => {
    macroServiceMocks.cancelChatMacroRun.mockResolvedValueOnce({
      ok: false,
      status: 503,
      error: "Jobs manager unavailable"
    })
    chatHookState.value.messages = [
      {
        id: "macro-status-1",
        role: "assistant",
        message: "Started /wrapup.",
        metadataExtra: {
          chat_macro: { run_id: "run-1", command: "wrapup", status: "running" }
        }
      }
    ] as unknown as UseMessageOptionState["messages"]
    render(
      <WorkspaceChatPanel
        stagedSources={[]}
        onClearStagedSources={vi.fn()}
        backendAvailable
        workspaceId="workspace-1"
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Cancel macro run" }))

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Jobs manager unavailable"
    )
    expect(screen.getByRole("button", { name: "Cancel macro run" })).toBeVisible()
  })

  it("renders completed macro output as a normal assistant message", () => {
    chatHookState.value.messages = [
      {
        id: "macro-result-1",
        role: "assistant",
        message: "## Summary\nFinal wrapup.",
        metadataExtra: {
          chat_macro: {
            run_id: "run-1",
            command: "wrapup",
            status: "completed"
          }
        }
      }
    ] as unknown as UseMessageOptionState["messages"]

    render(
      <WorkspaceChatPanel
        stagedSources={[]}
        onClearStagedSources={vi.fn()}
        backendAvailable
        workspaceId="workspace-1"
      />
    )

    expect(screen.getByTestId("workspace-panel-message")).toHaveTextContent("Final wrapup.")
    expect(screen.queryByRole("article", { name: /macro run/ })).not.toBeInTheDocument()
  })

  it("inserts staged source summary into the composer without sending and clears structured staging", () => {
    const onClearStagedSources = vi.fn()

    render(
      <WorkspaceChatPanel
        stagedSources={staged}
        onClearStagedSources={onClearStagedSources}
        backendAvailable
        workspaceId="workspace-1"
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Insert context summary" }))

    expect(screen.getByRole("textbox", { name: "Chat workspace message" })).toHaveValue(
      "Context sources:\n1. Operator Notes [document, scope: Default workspace]\n\n"
    )
    expect(chatHookState.onSubmit).not.toHaveBeenCalled()
    expect(onClearStagedSources).toHaveBeenCalledTimes(1)
  })

  it("clears inserted draft context when the workspace changes", () => {
    const onClearStagedSources = vi.fn()

    const { rerender } = render(
      <WorkspaceChatPanel
        stagedSources={staged}
        onClearStagedSources={onClearStagedSources}
        backendAvailable
        workspaceId="workspace-1"
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Insert context summary" }))
    expect(
      (screen.getByRole("textbox", {
        name: "Chat workspace message"
      }) as HTMLTextAreaElement).value
    ).toContain("Operator Notes")

    rerender(
      <WorkspaceChatPanel
        stagedSources={[]}
        onClearStagedSources={onClearStagedSources}
        backendAvailable
        workspaceId="workspace-2"
      />
    )

    expect(screen.getByRole("textbox", { name: "Chat workspace message" }))
      .toHaveValue("")
  })

  it("sends with staged context through the shared chat path", async () => {
    const onClearStagedSources = vi.fn()

    render(
      <WorkspaceChatPanel
        stagedSources={staged}
        onClearStagedSources={onClearStagedSources}
        backendAvailable
        workspaceId="workspace-1"
      />
    )

    fireEvent.change(screen.getByRole("textbox", { name: "Chat workspace message" }), {
      target: { value: "Summarize this" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Send with staged context" }))

    await waitFor(() => expect(chatHookState.onSubmit).toHaveBeenCalledTimes(1))
    expect(getSubmitPayload()).toMatchObject({
      message: expect.stringContaining("Summarize this"),
      image: "",
      requestOverrides: expect.objectContaining({
        ragMediaIds: [101],
        fileRetrievalEnabled: true
      })
    })
    expect(onClearStagedSources).toHaveBeenCalledTimes(1)
  })

  it("sends the composer draft with Ctrl+Enter", async () => {
    const onClearStagedSources = vi.fn()

    render(
      <WorkspaceChatPanel
        stagedSources={[]}
        onClearStagedSources={onClearStagedSources}
        backendAvailable
        workspaceId="workspace-1"
      />
    )

    const composer = screen.getByRole("textbox", { name: "Chat workspace message" })
    fireEvent.change(composer, { target: { value: "Keyboard send" } })
    fireEvent.keyDown(composer, { key: "Enter", ctrlKey: true })

    await waitFor(() => expect(chatHookState.onSubmit).toHaveBeenCalledTimes(1))
    expect(getSubmitPayload()).toMatchObject({
      message: "Keyboard send",
      requestOverrides: expect.objectContaining({
        ragMediaIds: [],
        fileRetrievalEnabled: false,
        chatMode: "normal"
      })
    })
  })

  it("includes staged source summary in the submitted message when no ready media ids can carry it", async () => {
    const onClearStagedSources = vi.fn()

    render(
      <WorkspaceChatPanel
        stagedSources={stagedWithoutReadyMedia}
        onClearStagedSources={onClearStagedSources}
        backendAvailable
        workspaceId="workspace-1"
      />
    )

    fireEvent.change(screen.getByRole("textbox", { name: "Chat workspace message" }), {
      target: { value: "Draft instruction" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Send with staged context" }))

    await waitFor(() => expect(chatHookState.onSubmit).toHaveBeenCalledTimes(1))
    expect(getSubmitPayload()).toMatchObject({
      message: expect.stringContaining("Draft instruction"),
      requestOverrides: expect.objectContaining({
        ragMediaIds: [],
        fileRetrievalEnabled: false,
        chatMode: "normal"
      })
    })
    expect(getSubmitPayload().message).toEqual(
      expect.stringContaining("Indexing Notes")
    )
    expect(onClearStagedSources).toHaveBeenCalledTimes(1)
  })

  it("includes staged source summary when mixed staged sources cannot all be sent as ready media ids", async () => {
    const onClearStagedSources = vi.fn()

    render(
      <WorkspaceChatPanel
        stagedSources={stagedWithMixedAvailability}
        onClearStagedSources={onClearStagedSources}
        backendAvailable
        workspaceId="workspace-1"
      />
    )

    fireEvent.change(screen.getByRole("textbox", { name: "Chat workspace message" }), {
      target: { value: "Draft instruction" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Send with staged context" }))

    await waitFor(() => expect(chatHookState.onSubmit).toHaveBeenCalledTimes(1))
    expect(getSubmitPayload()).toMatchObject({
      requestOverrides: expect.objectContaining({
        ragMediaIds: [101],
        fileRetrievalEnabled: true,
        chatMode: "rag"
      })
    })
    expect(getSubmitPayload().message).toEqual(
      expect.stringContaining("Draft instruction")
    )
    expect(getSubmitPayload().message).toEqual(
      expect.stringContaining("Indexing Notes")
    )
    expect(onClearStagedSources).toHaveBeenCalledTimes(1)
  })

  it("sends staged-only context when the composer is empty", async () => {
    const onClearStagedSources = vi.fn()

    render(
      <WorkspaceChatPanel
        stagedSources={staged}
        onClearStagedSources={onClearStagedSources}
        backendAvailable
        workspaceId="workspace-1"
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Send with staged context" }))

    await waitFor(() => expect(chatHookState.onSubmit).toHaveBeenCalledTimes(1))
    expect(getSubmitPayload()).toMatchObject({
      message: expect.stringContaining("Operator Notes"),
      requestOverrides: expect.objectContaining({
        ragMediaIds: [101],
        fileRetrievalEnabled: true,
        chatMode: "rag"
      })
    })
    expect(onClearStagedSources).toHaveBeenCalledTimes(1)
  })

  it("does not submit while a stream is active", () => {
    chatHookState.value.streaming = true
    const onClearStagedSources = vi.fn()

    render(
      <WorkspaceChatPanel
        stagedSources={[]}
        onClearStagedSources={onClearStagedSources}
        backendAvailable
        workspaceId="workspace-1"
      />
    )

    fireEvent.change(screen.getByRole("textbox", { name: "Chat workspace message" }), {
      target: { value: "Do not send yet" }
    })

    const sendButton = screen.getByRole("button", { name: "Send message" })
    expect(sendButton).toBeDisabled()
    fireEvent.click(sendButton)
    expect(chatHookState.onSubmit).not.toHaveBeenCalled()
  })

  it("does not submit while the backend is unavailable", () => {
    const onClearStagedSources = vi.fn()

    render(
      <WorkspaceChatPanel
        stagedSources={[]}
        onClearStagedSources={onClearStagedSources}
        backendAvailable={false}
        workspaceId="workspace-1"
      />
    )

    fireEvent.change(screen.getByRole("textbox", { name: "Chat workspace message" }), {
      target: { value: "Do not send offline" }
    })

    const sendButton = screen.getByRole("button", { name: "Send message" })
    expect(sendButton).toBeDisabled()
    fireEvent.click(sendButton)

    expect(chatHookState.onSubmit).not.toHaveBeenCalled()
    expect(onClearStagedSources).not.toHaveBeenCalled()
  })

  it("treats an empty workspace id as loading and does not submit under a global scope", () => {
    chatHookState.value.messages = [
      {
        id: "message-1",
        isBot: true,
        name: "Analyst",
        role: "assistant",
        message: "Hydrating",
        sources: []
      }
    ]
    const onClearStagedSources = vi.fn()
    const onRuntimeStateChange = vi.fn()

    render(
      <WorkspaceChatPanel
        stagedSources={staged}
        onClearStagedSources={onClearStagedSources}
        backendAvailable
        workspaceId=""
        onRuntimeStateChange={onRuntimeStateChange}
      />
    )

    expect(chatHookState.useMessageOption).toHaveBeenCalledWith({
      hydrateServerChat: false,
      scope: { type: "global" }
    })
    expect(screen.getByTestId("workspace-panel-message")).toHaveAttribute(
      "data-conversation-instance-id",
      "workspace-chat"
    )

    const composer = screen.getByRole("textbox", { name: "Chat workspace message" })
    fireEvent.change(composer, { target: { value: "Do not send during hydration" } })

    const sendButton = screen.getByRole("button", { name: "Send message" })
    expect(sendButton).toBeDisabled()
    fireEvent.click(sendButton)
    expect(screen.getByRole("button", { name: "Send with staged context" }))
      .toBeDisabled()
    fireEvent.click(screen.getByRole("button", { name: "Send with staged context" }))
    expect(chatHookState.onSubmit).not.toHaveBeenCalled()
    expect(onClearStagedSources).not.toHaveBeenCalled()
    expect(onRuntimeStateChange).toHaveBeenCalledWith(
      expect.objectContaining({ backendAvailable: false })
    )
    expect(screen.getByText("Loading workspace context")).toBeInTheDocument()
  })

  it("preserves draft and staged context when submit returns a failed result", async () => {
    chatHookState.onSubmit.mockResolvedValueOnce({
      status: "failed",
      errorMessage: "network"
    })
    const onClearStagedSources = vi.fn()

    render(
      <WorkspaceChatPanel
        stagedSources={staged}
        onClearStagedSources={onClearStagedSources}
        backendAvailable
        workspaceId="workspace-1"
      />
    )

    fireEvent.change(screen.getByRole("textbox", { name: "Chat workspace message" }), {
      target: { value: "Keep this draft" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Send with staged context" }))

    await screen.findByText("network")
    expect(screen.getByRole("textbox", { name: "Chat workspace message" })).toHaveValue("Keep this draft")
    expect(onClearStagedSources).not.toHaveBeenCalled()
  })

  it("reports failed sends through runtime state", async () => {
    chatHookState.onSubmit.mockResolvedValueOnce({
      status: "failed",
      errorMessage: "network"
    })
    const onRuntimeStateChange = vi.fn()

    render(
      <WorkspaceChatPanel
        stagedSources={staged}
        onClearStagedSources={vi.fn()}
        backendAvailable
        workspaceId="workspace-1"
        onRuntimeStateChange={onRuntimeStateChange}
      />
    )

    fireEvent.change(screen.getByRole("textbox", { name: "Chat workspace message" }), {
      target: { value: "Keep this draft" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Send with staged context" }))

    await screen.findByText("network")
    expect(onRuntimeStateChange).toHaveBeenCalledWith(
      expect.objectContaining({ sendError: "network" })
    )
  })

  it("preserves draft and staged context without an error when submit is skipped", async () => {
    chatHookState.onSubmit.mockResolvedValueOnce({
      status: "skipped",
      reason: "empty"
    })
    const onClearStagedSources = vi.fn()

    render(
      <WorkspaceChatPanel
        stagedSources={staged}
        onClearStagedSources={onClearStagedSources}
        backendAvailable
        workspaceId="workspace-1"
      />
    )

    fireEvent.change(screen.getByRole("textbox", { name: "Chat workspace message" }), {
      target: { value: "Keep this draft" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Send with staged context" }))

    await waitFor(() => expect(chatHookState.onSubmit).toHaveBeenCalledTimes(1))
    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
    expect(screen.getByRole("textbox", { name: "Chat workspace message" })).toHaveValue("Keep this draft")
    expect(onClearStagedSources).not.toHaveBeenCalled()
  })

  it("also preserves draft and staged context when submit rejects unexpectedly", async () => {
    chatHookState.onSubmit.mockRejectedValueOnce(new Error("network"))
    const onClearStagedSources = vi.fn()

    render(
      <WorkspaceChatPanel
        stagedSources={staged}
        onClearStagedSources={onClearStagedSources}
        backendAvailable
        workspaceId="workspace-1"
      />
    )

    fireEvent.change(screen.getByRole("textbox", { name: "Chat workspace message" }), {
      target: { value: "Keep this draft" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Send with staged context" }))

    await screen.findByText("Send failed")
    expect(screen.getByRole("textbox", { name: "Chat workspace message" })).toHaveValue("Keep this draft")
    expect(onClearStagedSources).not.toHaveBeenCalled()
  })

  it("shows loading state and wires stop streaming to the shared abort handler", () => {
    chatHookState.value.streaming = true
    chatHookState.value.isProcessing = true

    render(
      <WorkspaceChatPanel
        stagedSources={staged}
        onClearStagedSources={vi.fn()}
        backendAvailable
        workspaceId="workspace-1"
      />
    )

    expect(screen.getByText("Streaming")).toBeInTheDocument()
    const stop = screen.getByRole("button", { name: "Stop generating" })
    stop.focus()
    fireEvent.click(stop)
    expect(chatHookState.stopStreamingRequest).toHaveBeenCalledTimes(1)
    expect(chatHookState.stopStreamingRequest).toHaveBeenCalledWith()
    expect(screen.getByRole("textbox", { name: "Chat workspace message" })).toHaveFocus()
  })

  it("uses workspace chat scope and reports runtime state", () => {
    const onRuntimeStateChange = vi.fn()

    render(
      <WorkspaceChatPanel
        stagedSources={staged}
        onClearStagedSources={vi.fn()}
        backendAvailable
        workspaceId="workspace-1"
        onRuntimeStateChange={onRuntimeStateChange}
      />
    )

    expect(chatHookState.useMessageOption).toHaveBeenCalledWith(
      expect.objectContaining({
        scope: { type: "workspace", workspaceId: "workspace-1" }
      })
    )
    expect(onRuntimeStateChange).toHaveBeenCalledWith(
      expect.objectContaining({
        streaming: false,
        selectedModelLabel: "gpt-test",
        hasModelSelected: true,
        selectedPersonaLabel: "Analyst",
        assistantSource: "explicit"
      })
    )
  })

  it("reports missing model selection without relying on display text", () => {
    chatHookState.value.selectedModel = ""
    const onRuntimeStateChange = vi.fn()

    render(
      <WorkspaceChatPanel
        stagedSources={staged}
        onClearStagedSources={vi.fn()}
        backendAvailable
        workspaceId="workspace-1"
        onRuntimeStateChange={onRuntimeStateChange}
      />
    )

    expect(onRuntimeStateChange).toHaveBeenCalledWith(
      expect.objectContaining({
        selectedModelLabel: "No model selected",
        hasModelSelected: false
      })
    )
  })

  it.each(["typed", "staged", "typed-and-staged"] as const)(
    "blocks %s sends without a model and restores readiness without auto-submit",
    (kind) => {
      chatHookState.value.selectedModel = null
      chatHookState.value.selectedAssistant = null
      const onClearStagedSources = vi.fn()
      const onRuntimeStateChange = vi.fn()
      const props = {
        workspaceId: "workspace-1",
        stagedSources: kind === "typed" ? [] : staged,
        backendAvailable: true,
        onClearStagedSources,
        onRuntimeStateChange
      }
      const { rerender } = render(<WorkspaceChatPanel {...props} />)
      const composer = screen.getByRole("textbox", { name: "Chat workspace message" })
      const draft = kind === "staged" ? "" : "Preserved model-less draft"
      if (draft) fireEvent.change(composer, { target: { value: draft } })

      expect(composer).toBeEnabled()
      expect(screen.getByRole("button", { name: "Send message" })).toBeDisabled()
      if (kind !== "typed") {
        expect(screen.getByRole("button", { name: "Send with staged context" })).toBeDisabled()
        expect(screen.getByText("Operator Notes")).toBeInTheDocument()
      }
      fireEvent.keyDown(composer, { key: "Enter", ctrlKey: true })
      fireEvent.submit(composer.closest("form")!)
      expect(resolveServicePromptScope).not.toHaveBeenCalled()
      expect(chatHookState.onSubmit).not.toHaveBeenCalled()
      expect(composer).toHaveValue(draft)
      expect(onClearStagedSources).not.toHaveBeenCalled()

      chatHookState.value.selectedModel = "tldw:gemma"
      rerender(<WorkspaceChatPanel {...props} />)
      expect(screen.getByRole("button", { name: "Send message" })).toBeEnabled()
      if (kind !== "typed") {
        expect(screen.getByRole("button", { name: "Send with staged context" })).toBeEnabled()
        expect(screen.getByText("Operator Notes")).toBeInTheDocument()
      }
      expect(composer).toHaveValue(draft)
      expect(onClearStagedSources).not.toHaveBeenCalled()
      expect(resolveServicePromptScope).not.toHaveBeenCalled()
      expect(chatHookState.onSubmit).not.toHaveBeenCalled()
      expect(onRuntimeStateChange).toHaveBeenLastCalledWith(expect.objectContaining({
        hasModelSelected: true,
        selectedPersonaLabel: null
      }))
    }
  )

  it("keeps explicit Auto server routing selectable without a persona", () => {
    chatHookState.value.selectedModel = "auto"
    chatHookState.value.selectedAssistant = null
    render(<WorkspaceChatPanel workspaceId="workspace-1" stagedSources={staged}
      backendAvailable onClearStagedSources={vi.fn()} />)
    expect(screen.getByRole("button", { name: "Send message" })).toBeEnabled()
    expect(screen.getByRole("button", { name: "Send with staged context" })).toBeEnabled()
    expect(chatHookState.onSubmit).not.toHaveBeenCalled()
  })

  it("inherits the available workspace persona default on first submit", async () => {
    chatHookState.value.selectedAssistant = {
      kind: "persona",
      id: "workspace-persona",
      name: "Workspace Analyst"
    }
    chatHookState.value.selectedAssistantSource = "workspace"
    const onRuntimeStateChange = vi.fn()

    render(
      <WorkspaceChatPanel
        stagedSources={[]}
        onClearStagedSources={vi.fn()}
        backendAvailable
        workspaceId="workspace-1"
        effectiveAssistantDefault={availableWorkspaceDefault}
        onRuntimeStateChange={onRuntimeStateChange}
      />
    )

    expect(chatHookState.useMessageOption).toHaveBeenCalledWith(
      expect.objectContaining({
        scope: { type: "workspace", workspaceId: "workspace-1" },
        inheritedAssistant: expect.objectContaining({
          kind: "persona",
          id: "workspace-persona",
          name: "Workspace Analyst",
          metadata: expect.objectContaining({
            selectionMode: "tracked",
            source: "workspace",
            personaMemoryMode: "read_write"
          })
        })
      })
    )
    expect(onRuntimeStateChange).toHaveBeenCalledWith(
      expect.objectContaining({
        selectedPersonaLabel: "Workspace Analyst",
        assistantSource: "workspace"
      })
    )

    fireEvent.change(screen.getByRole("textbox", { name: "Chat workspace message" }), {
      target: { value: "Use the default persona" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Send message" }))

    await waitFor(() => expect(chatHookState.onSubmit).toHaveBeenCalledTimes(1))
    expect(getSubmitPayload()).toMatchObject({
      requestOverrides: expect.objectContaining({
        assistant_kind: "persona",
        assistant_id: "workspace-persona",
        persona_memory_mode: "read_write"
      })
    })
  })

  it("keeps workspace provenance after the inherited persona chat is created", () => {
    chatHookState.value.selectedAssistant = {
      kind: "persona",
      id: "workspace-persona",
      name: "Workspace Analyst"
    }
    chatHookState.value.selectedAssistantSource = "workspace"
    chatHookState.value.serverChatId = "workspace-persona-chat"
    chatHookState.value.serverChatAssistantKind = "persona"
    chatHookState.value.serverChatAssistantId = "workspace-persona"
    const onRuntimeStateChange = vi.fn()

    render(
      <WorkspaceChatPanel
        stagedSources={[]}
        onClearStagedSources={vi.fn()}
        backendAvailable
        workspaceId="workspace-1"
        effectiveAssistantDefault={availableWorkspaceDefault}
        onRuntimeStateChange={onRuntimeStateChange}
      />
    )

    expect(onRuntimeStateChange).toHaveBeenCalledWith(
      expect.objectContaining({
        selectedPersonaLabel: "Workspace Analyst",
        assistantSource: "workspace"
      })
    )
  })

  it("keeps workspace provenance when reopening an inherited persona chat", () => {
    chatHookState.value.selectedAssistant = null
    chatHookState.value.selectedAssistantSource = "none"
    chatHookState.value.serverChatId = "workspace-persona-chat"
    chatHookState.value.serverChatAssistantKind = "persona"
    chatHookState.value.serverChatAssistantId = "workspace-persona"
    const onRuntimeStateChange = vi.fn()

    render(
      <WorkspaceChatPanel
        stagedSources={[]}
        onClearStagedSources={vi.fn()}
        backendAvailable
        workspaceId="workspace-1"
        effectiveAssistantDefault={availableWorkspaceDefault}
        onRuntimeStateChange={onRuntimeStateChange}
      />
    )

    expect(onRuntimeStateChange).toHaveBeenCalledWith(
      expect.objectContaining({
        selectedPersonaLabel: "Workspace Analyst",
        assistantSource: "workspace"
      })
    )
  })

  it("keeps an explicit selected persona ahead of the workspace default", async () => {
    chatHookState.value.selectedAssistant = {
      kind: "persona",
      id: "explicit-persona",
      name: "Explicit Analyst",
      metadata: {
        selectionMode: "tracked"
      }
    }

    render(
      <WorkspaceChatPanel
        stagedSources={[]}
        onClearStagedSources={vi.fn()}
        backendAvailable
        workspaceId="workspace-1"
        effectiveAssistantDefault={availableWorkspaceDefault}
      />
    )

    fireEvent.change(screen.getByRole("textbox", { name: "Chat workspace message" }), {
      target: { value: "Use explicit persona" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Send message" }))

    await waitFor(() => expect(chatHookState.onSubmit).toHaveBeenCalledTimes(1))
    expect(getSubmitPayload().requestOverrides).not.toMatchObject({
      assistant_id: "workspace-persona"
    })
  })

  it("does not inherit an unavailable workspace default", async () => {
    chatHookState.value.selectedAssistant = null
    chatHookState.value.selectedAssistantSource = "none"
    const onRuntimeStateChange = vi.fn()

    render(
      <WorkspaceChatPanel
        stagedSources={[]}
        onClearStagedSources={vi.fn()}
        backendAvailable
        workspaceId="workspace-1"
        effectiveAssistantDefault={unavailableWorkspaceDefault}
        onRuntimeStateChange={onRuntimeStateChange}
      />
    )

    expect(onRuntimeStateChange).toHaveBeenCalledWith(
      expect.objectContaining({
        selectedPersonaLabel: null,
        assistantSource: "unavailable",
        workspaceAssistantDegradedReason: "persona_deleted"
      })
    )

    fireEvent.change(screen.getByRole("textbox", { name: "Chat workspace message" }), {
      target: { value: "No default persona" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Send message" }))

    await waitFor(() => expect(chatHookState.onSubmit).toHaveBeenCalledTimes(1))
    expect(getSubmitPayload().requestOverrides).not.toMatchObject({
      assistant_kind: "persona",
      assistant_id: "workspace-persona"
    })
  })

  it("does not mutate existing chat assistant metadata when the workspace default changes", async () => {
    chatHookState.value.selectedAssistant = null
    chatHookState.value.selectedAssistantSource = "none"
    chatHookState.value.serverChatAssistantKind = "persona"
    chatHookState.value.serverChatAssistantId = "session-persona"
    chatHookState.value.serverChatMetaLoaded = true

    render(
      <WorkspaceChatPanel
        stagedSources={[]}
        onClearStagedSources={vi.fn()}
        backendAvailable
        workspaceId="workspace-1"
        effectiveAssistantDefault={availableWorkspaceDefault}
      />
    )

    fireEvent.change(screen.getByRole("textbox", { name: "Chat workspace message" }), {
      target: { value: "Continue existing chat" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Send message" }))

    await waitFor(() => expect(chatHookState.onSubmit).toHaveBeenCalledTimes(1))
    expect(getSubmitPayload().requestOverrides).not.toMatchObject({
      assistant_id: "workspace-persona"
    })
  })
})
