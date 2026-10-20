import "@/hooks/chat/__tests__/native-plasmo-storage-fixture"
import React from "react"
import { useModelSettingsOverlayState } from "@/hooks/useModelSettingsOverlayState"
import { ConfigProvider, Modal } from "antd"
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { notifyManager, QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { normalizeSelectedModel } from "@/hooks/chat/useSelectedModel"
import { normalChatMode } from "@/hooks/chat-modes/normalChatMode"
import type { BaseMessage } from "@/types/messages"
import { readFileSync } from "node:fs"
import path from "node:path"
import ts from "typescript"
import { useStoreMessageOption } from "@/store/option"
import { validateNativeCharacterConfiguration } from "@/hooks/chat/native-character-preflight"
import { CurrentChatModelSettings } from "../CurrentChatModelSettings"
import { getAllModelSettings } from "@/services/model-settings"
import { saveActorSettingsForChat } from "@/services/actor-settings"
import { resolveSelectedSystemPromptContent } from "@/components/Common/system-prompt-utils"
import { useStoreChatModelSettings } from "@/store/model"
import { useActorStore } from "@/store/actor"
import { settingsTestOwner } from "@/services/__tests__/chat-settings-fixture"
import { buildCockpitProviderRouteSummary } from "@/components/Option/Playground/playground-cockpit-summaries"

const inputs = vi.hoisted(() => ({
  selectedSystemPrompt: null as string | null,
  character: null as { id: string; name: string } | null,
  provider: "llama.cpp",
  selectedModel: "tldw:/models/gemma.gguf",
  otherProviderFirst: false,
  realModelSelection: false,
  modelWrite: null as null | (() => Promise<void>),
  accountCurrent: vi.fn(() => true),
  conversationWrite: vi.fn(async (_patch: unknown) => null),
  templateText: "Selected template instructions",
  templateRead: null as null | (() => Promise<string>),
  settingsRead: null as null | (() => Promise<{ temperature: number }>),
}))
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string | { defaultValue?: string }) =>
      typeof fallback === "string" ? fallback : (fallback?.defaultValue ?? key),
  }),
}))
vi.mock("@/hooks/useMessageOption", () => ({
  useMessageOption: () => {
    const core = useStoreMessageOption()
    return ({
    historyId: null,
    serverChatId: null,
    selectedSystemPrompt: inputs.selectedSystemPrompt,
    selectedModel: inputs.realModelSelection ? core.selectedModel : inputs.selectedModel,
    uploadedFiles: [],
    removeUploadedFile: vi.fn(),
    setSelectedModel: inputs.realModelSelection ? (value: string | null) => {
      core.setSelectedModel(normalizeSelectedModel(value))
      return inputs.modelWrite?.()
    } : vi.fn(),
    retrySelectedModelSave: () => inputs.modelWrite?.() ?? Promise.resolve(),
    setServerChatState: vi.fn(),
    setServerChatTopic: vi.fn(),
    setServerChatVersion: vi.fn(),
    setServerChatPersonaMemoryMode: vi.fn(),
  })},
}))
const settingsController = {
  getCurrent: () => ({ status: "idle", owner: null, view: null }),
  getSignal: () => undefined,
  fence: () => inputs.accountCurrent,
  draftAccount: { requestScope: settingsTestOwner.requestScope, isCurrent: inputs.accountCurrent }
};
vi.mock("@/hooks/chat/useHistorySelection", async original => ({
  ...await original<typeof import("@/hooks/chat/useHistorySelection")>(),
  useHistorySelectionContext: () => settingsController
}));
vi.mock("@/hooks/useSelectedCharacter", () => ({
  useSelectedCharacter: () => [inputs.character, vi.fn(), { isLoading: false }],
}))
vi.mock("@/hooks/useSelectedAssistant", () => ({
  useSelectedAssistant: () => [null, vi.fn(), { isCurrent: inputs.accountCurrent }],
}))
vi.mock("@/hooks/chat/useChatSettingsRecord", () => ({
  useChatSettingsRecord: () => ({
    settings: null,
    updateSettings: inputs.conversationWrite,
    chatKey: "local:test",
    isReady: true, mode: "ordinary", isCurrent: () => inputs.accountCurrent(),
    captureTarget: () => ({ historyId: "test", serverChatId: null, validate_lease: () => inputs.accountCurrent() }),
  }),
}))
vi.mock("@/services/tldw-server", async original => ({
  ...await original<typeof import("@/services/tldw-server")>(),
  fetchChatModels: async () => [
    ...(inputs.otherProviderFirst
      ? [{ model: "tldw:/models/gemma.gguf", provider: "openai" }]
      : []),
    { model: "tldw:/models/gemma.gguf", provider: inputs.provider },
  ],
  systemPromptForNonRagOption: async () => "",
}))
vi.mock("@/services/model-settings", () => ({
  getAllModelSettings: async () => inputs.settingsRead?.() ?? ({ temperature: 0.7 }),
}))
vi.mock("@/services/ocr", () => ({ getOCRLanguage: async () => null }))
vi.mock("@/services/actor-settings", async original => ({
  ...await original<typeof import("@/services/actor-settings")>(),
  getActorSettingsForChatWithCharacterFallback: async () => null,
  saveActorSettingsForChat: vi.fn(async () => true),
}))
vi.mock("@/components/Common/system-prompt-utils", async original => {
  const actual = await original<typeof import("@/components/Common/system-prompt-utils")>()
  const read = vi.fn(async (id?: string | null) => id ? await (inputs.templateRead?.() ?? inputs.templateText) : "")
  return { ...actual, resolveSelectedSystemPromptContent: read,
    resolveEffectiveSystemPromptState: (params: Parameters<typeof actual.resolveEffectiveSystemPromptState>[0]) =>
      actual.resolveEffectiveSystemPromptState({ ...params, getPromptByIdFn: async id => ({ id, content: await read(id) }) as any }) }
})
vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: async () => undefined,
    listCharacters: async () => [],
    listChatMessages: async () => [],
  },
}))
vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({ error: vi.fn(), success: vi.fn() }),
}))
vi.mock("@/components/Common/Settings/PromptAssemblyPreview", () => ({
  PromptAssemblyPreview: () => null,
}))
vi.mock("@/components/Common/Settings/LorebookDebugPanel", () => ({
  LorebookDebugPanel: () => null,
}))
vi.mock("../tabs", async () => ({
  ConversationTab: (await import("../tabs/ConversationTab")).ConversationTab,
  ModelBasicsTab: (await import("../tabs/ModelBasicsTab")).ModelBasicsTab,
  ActorTab: () => null,
  AdvancedParamsTab: () => null,
}))
vi.mock("../LlamaCppAdvancedControls", () => ({
  LlamaCppAdvancedControls: () => null,
}))
vi.mock("@/components/Common/ProviderIcon", () => ({
  ProviderIcons: () => null,
}))

// Keep ordinary prompt assembly and text/image formatters real; stop only at
// the network/persistence pipeline so these editor tests never contact an API.
vi.mock("@/db/dexie/helpers", () => ({
  getPromptById: async (id: string) => {
    if (!id) return undefined
    const content = await (inputs.templateRead?.() ?? inputs.templateText)
    return { id, title: "Selected template", content, system_prompt: content, is_system: true, createdAt: 0 }
  },
}))
vi.mock("@/utils/model", () => ({ getSelectedModelName: async () => "gemma" }))
vi.mock("@/hooks/chat-modes/chatModePipeline", () => ({
  runChatPipeline: async (...args: unknown[]) => {
    const definition = args[0] as {
      preparePrompt: (context: Record<string, unknown>) => Promise<unknown>
    }
    const [, message, image, isRegenerate, messages, history, signal, params] =
      args
    return definition.preparePrompt({
      ...(params as object),
      message,
      image,
      isRegenerate,
      messages,
      history,
      signal,
    })
  },
}))

// Scope role queries to actual form actions and fields; inactive AntD panels dominated the deadline.
const getMainSaveButton = () => {
  const submits = document.querySelectorAll('form button[type="submit"]')
  if (submits.length !== 1 || !submits[0].parentElement) throw new Error("expected one model settings submit action")
  return within(submits[0].parentElement).getByRole("button", { name: /^save$/i })
}

const getPromptResetButton = (name: string) => {
  const field = screen.getByPlaceholderText("Enter System Prompt").closest(".ant-form-item")
  if (!(field instanceof HTMLElement)) throw new Error("expected model default prompt field")
  return within(field).getByRole("button", { name })
}

const getSettingsButton = (name: string) => {
  const candidates = Array.from(document.querySelectorAll("button")).filter(button =>
    button.getAttribute("aria-label") === name || button.textContent?.trim() === name)
  if (candidates.length !== 1 || !candidates[0].parentElement) throw new Error(`expected one settings action: ${name}`)
  return within(candidates[0].parentElement).getByRole("button", { name, exact: true })
}

const getSettingsTab = (name: string) => {
  const tabs = Array.from(document.querySelectorAll('form [role="tab"]'))
    .filter(tab => tab.textContent?.trim() === name)
  if (tabs.length !== 1 || !tabs[0].parentElement) throw new Error(`expected one settings tab: ${name}`)
  return within(tabs[0].parentElement).getByRole("tab", { name, exact: true })
}

const getModelCombobox = () => {
  const activeTabs = document.querySelectorAll('form [role="tab"][aria-selected="true"]')
  if (activeTabs.length !== 1 || activeTabs[0].textContent?.trim() !== "Model") throw new Error("expected the active Model tab")
  const panel = document.getElementById(activeTabs[0].getAttribute("aria-controls") ?? "")
  if (!(panel instanceof HTMLElement) || panel.getAttribute("role") !== "tabpanel") throw new Error("expected the Model tab panel")
  const label = within(panel).getByText("API / model", { selector: "label", exact: true })
  const field = label.closest(".ant-form-item")
  if (!(field instanceof HTMLElement)) throw new Error("expected the API / model field")
  return within(field).getByRole("combobox")
}

const settingsClients = new Set<QueryClient>()

const instruction =
  "UAT363 RETRY ORBIT742. Describe visible image contents accurately in one short sentence."
const renderSettings = (
  settingsScope?: string,
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } }),
) => {
  settingsClients.add(client)
  const close = vi.fn()
  const view = render(
    <QueryClientProvider client={client}>
      {/* jsdom does not complete AntD's CSS appear animation. */}
      <ConfigProvider theme={{ token: { motion: false } }}>
      <CurrentChatModelSettings
        open
        setOpen={close}
        settingsScope={settingsScope}
      />
      </ConfigProvider>
    </QueryClientProvider>,
  )
  return { ...view, close }
}

const mountSettings = async (settingsScope?: string, client?: QueryClient) => {
  const view = renderSettings(settingsScope, client)
  fireEvent.click(await waitFor(() => getSettingsTab("Conversation")))
  const editor = await screen.findByPlaceholderText("Enter System Prompt")
  return { ...view, editor }
}

describe("Conversation prompt editor through real model settings Save", () => {
  afterEach(() => {
    cleanup()
    Modal.destroyAll()
    for (const client of settingsClients) client.clear()
    settingsClients.clear()
  })
  beforeEach(() => {
    inputs.selectedSystemPrompt = null
    inputs.character = null
    inputs.provider = "llama.cpp"
    inputs.selectedModel = "tldw:/models/gemma.gguf"
    inputs.otherProviderFirst = false
    inputs.realModelSelection = false
    inputs.modelWrite = null
    inputs.conversationWrite.mockReset().mockResolvedValue(null)
    inputs.accountCurrent.mockReset().mockReturnValue(true)
    inputs.templateText = "Selected template instructions"
    inputs.templateRead = null
    inputs.settingsRead = null
    useStoreChatModelSettings.getState().reset()
    useStoreChatModelSettings
      .getState()
      .updateSetting("apiProvider", "llama.cpp")
    const route = buildCockpitProviderRouteSummary({
      selectedProvider: "llama.cpp",
      selectedModel: "tldw:/models/gemma.gguf",
    })
    useStoreChatModelSettings
      .getState()
      .setActiveSettingsScope(route.providerRouteLabel)
    useActorStore.getState().reset()
    useStoreMessageOption.getState().startNewChatDraft(false)
    vi.mocked(saveActorSettingsForChat).mockReset().mockResolvedValue(true)
    vi.mocked(resolveSelectedSystemPromptContent).mockReset().mockImplementation(async id => id ? "Selected template instructions" : "")
  })

  it.each([false, true])("forwards only native close completion and unmounts the editor (drawer=%s)", async useDrawer => {
    const completed = vi.fn()
    const Owner = () => {
      const dialog = useModelSettingsOverlayState()
      return <>
        <button onClick={() => dialog.setOpen(true)}>Open model editor</button>
        {dialog.present && <section data-testid="native-model-editor">
          <CurrentChatModelSettings open={dialog.open} setOpen={dialog.setOpen} useDrawer={useDrawer}
            afterClose={() => { if (dialog.afterClose()) completed() }} />
        </section>}
      </>
    }
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const mounted = render(<QueryClientProvider client={client}><ConfigProvider theme={{ token: { motion: false } }}><Owner /></ConfigProvider></QueryClientProvider>)
    try {
      fireEvent.click(getSettingsButton("Open model editor"))
      fireEvent.click(await waitFor(() => getSettingsTab("Conversation")))
      await screen.findByPlaceholderText("Enter System Prompt")
      expect(completed).not.toHaveBeenCalled()
      fireEvent.click(getSettingsButton("Cancel"))
      await waitFor(() => expect(completed).toHaveBeenCalledOnce())
      expect(screen.queryByTestId("native-model-editor")).toBeNull()
    } finally { mounted.unmount(); client.clear() }
  })

  it.each(["inferred", "explicit"])(
    "keeps the entered prompt immediately after Save with %s model scope",
    async (mode) => {
      const scope =
        mode === "explicit"
          ? useStoreChatModelSettings.getState().activeSettingsScope
          : undefined
      const { editor, close } = await mountSettings(scope)
      fireEvent.change(editor, { target: { value: instruction } })
      expect(useStoreChatModelSettings.getState().systemPrompt).toBeUndefined()
      fireEvent.click(getMainSaveButton())
      await waitFor(() => expect(close).toHaveBeenCalledWith(false))
      expect(useStoreChatModelSettings.getState().systemPrompt).toBe(
        instruction,
      )
    },
  )

  it("stages prompt edits and cancels without changing live model defaults", async () => {
    const { editor, close } = await mountSettings()
    const before = useStoreChatModelSettings.getState().systemPrompt
    fireEvent.change(editor, { target: { value: instruction } })
    expect(useStoreChatModelSettings.getState().systemPrompt).toBe(before)
    fireEvent.click(getSettingsButton("Cancel"))
    expect(close).toHaveBeenCalledWith(false)
    expect(useStoreChatModelSettings.getState().systemPrompt).toBe(before)
    expect(saveActorSettingsForChat).not.toHaveBeenCalled()
  })

  it("keeps model defaults unchanged when scene Save fails and retries the staged proposal", async () => {
    vi.mocked(saveActorSettingsForChat).mockResolvedValueOnce(false)
    const { editor, close } = await mountSettings()
    const before = useStoreChatModelSettings.getState().systemPrompt
    fireEvent.change(editor, { target: { value: instruction } })
    fireEvent.click(getMainSaveButton())
    expect(await screen.findByRole("alert")).toHaveTextContent(/could not be saved/i)
    expect(useStoreChatModelSettings.getState().systemPrompt).toBe(before)
    expect(close).not.toHaveBeenCalled()
    expect(editor).toHaveValue(instruction)
    fireEvent.click(getMainSaveButton())
    await waitFor(() => expect(close).toHaveBeenCalledWith(false))
    expect(useStoreChatModelSettings.getState().systemPrompt).toBe(instruction)
    expect(saveActorSettingsForChat).toHaveBeenCalledTimes(2)
  })

  it.each(["edit", "model"])("does not publish a held prompt Reset after a newer %s", async change => {
    inputs.selectedSystemPrompt = "template-1"
    inputs.realModelSelection = true
    useStoreMessageOption.getState().setSelectedModel(inputs.selectedModel)
    const { editor } = await mountSettings()
    let release!: (value: string) => void
    const pending = new Promise<string>(resolve => { release = resolve })
    vi.mocked(resolveSelectedSystemPromptContent).mockReturnValueOnce(pending)
    fireEvent.click(getPromptResetButton("Reset to template"))
    if (change === "edit") fireEvent.change(editor, { target: { value: "Newer draft" } })
    else act(() => useStoreMessageOption.getState().setSelectedModel("openai:replacement"))
    await act(async () => { release("Old reset result"); await pending })
    expect(editor).not.toHaveValue("Old reset result")
    expect(useStoreChatModelSettings.getState().systemPrompt).not.toBe("Old reset result")
  })

  it("protects a typed conversation note from main Save and dirty close before blur", async () => {
    const { close } = await mountSettings()
    const note = screen.getByPlaceholderText("E.g., Keep responses grounded, avoid repetition, and progress the scene.")
    fireEvent.change(note, { target: { value: "Staged conversation only" } })
    expect(getMainSaveButton()).toBeDisabled()
    fireEvent.click(getMainSaveButton())
    expect(close).not.toHaveBeenCalled()
    expect(saveActorSettingsForChat).not.toHaveBeenCalled()
    fireEvent.click(getSettingsButton("Close"))
    await screen.findAllByText("Discard unsaved settings?")
    fireEvent.click(getSettingsButton("Keep editing"))
    expect(note).toHaveValue("Staged conversation only")
    expect(inputs.conversationWrite).not.toHaveBeenCalled()
    expect(close).not.toHaveBeenCalled()
  })

  it("cannot use an older discard admission to discard a newer conversation edit", async () => {
    const { close } = await mountSettings()
    const note = screen.getByPlaceholderText("E.g., Keep responses grounded, avoid repetition, and progress the scene.")
    fireEvent.change(note, { target: { value: "First draft" } })
    fireEvent.click(getSettingsButton("Close"))
    await screen.findAllByText("Discard unsaved settings?")
    fireEvent.change(note, { target: { value: "Newer draft" } })
    fireEvent.click(getSettingsButton("Discard"))
    await waitFor(() => expect(screen.queryAllByText("Discard unsaved settings?")).toHaveLength(0))
    expect(close).not.toHaveBeenCalled()
    expect(note).toHaveValue("Newer draft")
  })

  it("blocks panel dismissal while conversation Apply is held", async () => {
    let release!: () => void
    inputs.conversationWrite.mockImplementation(() => new Promise<null>(resolve => { release = () => resolve(null) }))
    const { close } = await mountSettings()
    fireEvent.change(screen.getByPlaceholderText("E.g., Keep responses grounded, avoid repetition, and progress the scene."), { target: { value: "Captured note" } })
    fireEvent.click(getSettingsButton("Apply conversation changes"))
    await waitFor(() => expect(inputs.conversationWrite).toHaveBeenCalledTimes(1))
    expect(getSettingsButton("Cancel")).toBeDisabled()
    fireEvent.click(getSettingsButton("Close"))
    expect(close).not.toHaveBeenCalled()
    expect(screen.queryByText("Discard unsaved settings?")).not.toBeInTheDocument()
    await act(async () => release())
    await waitFor(() => expect(getMainSaveButton()).toBeEnabled())
  })

  it("keeps edits when dirty-close admission is cancelled", async () => {
    const { editor, close } = await mountSettings()
    fireEvent.change(editor, { target: { value: instruction } })
    fireEvent.click(getSettingsButton("Close"))
    expect((await screen.findAllByText("Discard unsaved settings?")).length).toBeGreaterThan(0)
    expect(close).not.toHaveBeenCalled()
    fireEvent.click(getSettingsButton("Keep editing"))
    await waitFor(() => expect(screen.queryAllByText("Discard unsaved settings?")).toHaveLength(0))
    expect(editor).toHaveValue(instruction)
    expect(close).not.toHaveBeenCalled()
  })

  it("does not discard a newer edit when an older dirty-close question is accepted", async () => {
    const { editor, close } = await mountSettings()
    fireEvent.change(editor, { target: { value: instruction } })
    fireEvent.click(getSettingsButton("Close"))
    await screen.findAllByText("Discard unsaved settings?")
    fireEvent.change(editor, { target: { value: "Newer draft" } })
    fireEvent.click(getSettingsButton("Discard"))
    await waitFor(() => expect(screen.queryAllByText("Discard unsaved settings?")).toHaveLength(0))
    expect(editor).toHaveValue("Newer draft")
    expect(close).not.toHaveBeenCalled()
  })

  it("keeps a raw model prompt through catalog arrival, reordering and real Save", async () => {
    useStoreChatModelSettings.getState().reset()
    useStoreChatModelSettings.getState().setActiveSettingsScope(inputs.selectedModel)
    useStoreChatModelSettings.getState().setSystemPrompt(instruction)
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const { editor, close } = await mountSettings(undefined, client)
    for (const models of [[], [{ model: inputs.selectedModel, provider: "openai" }, { model: inputs.selectedModel, provider: "llama.cpp" }], [{ model: inputs.selectedModel, provider: "llama.cpp" }]]) {
      await act(async () => {
        client.setQueriesData({ queryKey: ["playground:chatModels"], type: "active" }, models)
        await new Promise<void>(resolve => notifyManager.schedule(resolve))
        expect(client.getQueryCache().findAll({ queryKey: ["playground:chatModels"], type: "active" })
          .map(query => query.state.data)).toEqual([models])
      })
      expect(editor).toHaveValue(instruction)
      expect(useStoreChatModelSettings.getState().activeSettingsScope).toBe(inputs.selectedModel)
      expect(useStoreChatModelSettings.getState().systemPrompt).toBe(instruction)
    }
    fireEvent.click(getMainSaveButton())
    await waitFor(() => expect(close).toHaveBeenCalledWith(false))
    expect(useStoreChatModelSettings.getState().activeSettingsScope).toBe(inputs.selectedModel)
    expect(useStoreChatModelSettings.getState().systemPrompt).toBe(instruction)
  })

  it("does not label a providerless current model as Custom when the catalog is empty", async () => {
    inputs.realModelSelection = true
    useStoreMessageOption.getState().setSelectedModel(inputs.selectedModel)
    useStoreChatModelSettings.getState().reset()
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    await mountSettings(undefined, client)
    await act(async () => {
      client.setQueriesData({ queryKey: ["playground:chatModels"], type: "active" }, [])
      await new Promise<void>(resolve => notifyManager.schedule(resolve))
      expect(client.getQueryCache().findAll({ queryKey: ["playground:chatModels"], type: "active" })
        .map(query => query.state.data)).toEqual([[]])
    })
    fireEvent.click(getSettingsTab("Model"))
    await waitFor(() => expect(screen.getByText("Choose provider - /models/gemma.gguf")).toBeVisible())
    expect(screen.queryByText("Custom - tldw:/models/gemma.gguf")).toBeNull()
    fireEvent.mouseDown(getModelCombobox())
    const labels = await screen.findAllByText("Choose provider - /models/gemma.gguf")
    fireEvent.click(labels.at(-1)!)
    expect(useStoreMessageOption.getState().selectedModel).toBe(inputs.selectedModel)
  }, 10000)

  it.each([undefined, "llamacpp:/models/gemma.gguf"])("commits the provider chosen in the Model tab and retires the previous dialog scope (%s)", async explicitScope => {
    inputs.realModelSelection = true
    inputs.otherProviderFirst = true
    useStoreMessageOption.getState().setSelectedModel("llamacpp:/models/gemma.gguf")
    useStoreChatModelSettings.getState().reset()
    const state = useStoreChatModelSettings.getState()
    state.setActiveSettingsScope("llamacpp:/models/gemma.gguf")
    state.updateSettings({ systemPrompt: "KEEP A", temperature: 0.2 })
    state.updateScopedSetting("openai:/models/gemma.gguf", "temperature", 0.6)
    const { close } = renderSettings(explicitScope)
    expect(await waitFor(() => getSettingsTab("Model"))).toHaveAttribute("aria-selected", "true")
    await waitFor(() => expect(screen.getByLabelText("modelSettings.form.temperature.label")).toHaveValue("0.2"))
    fireEvent.mouseDown(getModelCombobox())
    fireEvent.click(await screen.findByText("OpenAI - tldw:/models/gemma.gguf"))
    expect(useStoreMessageOption.getState().selectedModel).toBe("openai:/models/gemma.gguf")
    expect(useStoreChatModelSettings.getState().activeSettingsScope).toBe("openai:/models/gemma.gguf")
    expect(useStoreChatModelSettings.getState().getEffectiveSettings("llamacpp:/models/gemma.gguf")).toMatchObject({ systemPrompt: "KEEP A", temperature: 0.2 })
    expect(await screen.findByLabelText("modelSettings.form.temperature.label")).toHaveValue("0.6")
    expect(validateNativeCharacterConfiguration({ temporary: false, clientToolCount: 0, settings: {}, model: useStoreMessageOption.getState().selectedModel! })).toMatchObject({ provider: "openai", model: "/models/gemma.gguf" })
    fireEvent.click(getSettingsTab("Conversation"))
    fireEvent.change(await screen.findByPlaceholderText("Enter System Prompt"), { target: { value: instruction } })
    fireEvent.click(getMainSaveButton())
    await waitFor(() => expect(close).toHaveBeenCalledWith(false))
    expect(useStoreChatModelSettings.getState().activeSettingsScope).toBe("openai:/models/gemma.gguf")
    expect(useStoreChatModelSettings.getState().systemPrompt).toBe(instruction)
    expect(useStoreChatModelSettings.getState().getEffectiveSettings("llamacpp:/models/gemma.gguf")).toMatchObject({ systemPrompt: "KEEP A", temperature: 0.2 })
    expect(useStoreChatModelSettings.getState().temperature).toBe(0.6)
  }, 10000)

  it("does not change model if its scope publication retires the initiating account", async () => {
    inputs.realModelSelection = true
    inputs.otherProviderFirst = true
    useStoreMessageOption.getState().setSelectedModel("llamacpp:/models/gemma.gguf")
    useStoreChatModelSettings.getState().setActiveSettingsScope("llamacpp:/models/gemma.gguf")
    await mountSettings()
    const unsubscribe = useStoreChatModelSettings.subscribe(state => {
      if (state.activeSettingsScope === "openai:/models/gemma.gguf") inputs.accountCurrent.mockReturnValue(false)
    })
    try {
      fireEvent.click(getSettingsTab("Model"))
      fireEvent.mouseDown(getModelCombobox())
      fireEvent.click(await screen.findByText("OpenAI - tldw:/models/gemma.gguf"))
      expect(useStoreMessageOption.getState().selectedModel).toBe("llamacpp:/models/gemma.gguf")
    } finally { unsubscribe() }
  }, 10_000)

  it("reports an active model choice whose device save fails", async () => {
    inputs.realModelSelection = true
    inputs.otherProviderFirst = true
    const write = vi.fn().mockRejectedValueOnce(new Error("device unavailable")).mockResolvedValue(undefined)
    inputs.modelWrite = write
    useStoreMessageOption.getState().setSelectedModel("llamacpp:/models/gemma.gguf")
    useStoreChatModelSettings.getState().setActiveSettingsScope("llamacpp:/models/gemma.gguf")
    await mountSettings()
    fireEvent.click(getSettingsTab("Model"))
    fireEvent.mouseDown(getModelCombobox())
    fireEvent.click(await screen.findByText("OpenAI - tldw:/models/gemma.gguf"))
    expect(await screen.findByRole("alert")).toHaveTextContent(/model choice is active.*could not be saved on this device/i)
    expect(useStoreMessageOption.getState().selectedModel).toBe("openai:/models/gemma.gguf")
    await waitFor(() => expect(getSettingsButton("Retry model save")).toBeEnabled())
    fireEvent.click(getSettingsButton("Retry model save"))
    await waitFor(() => expect(screen.queryByRole("alert")).not.toBeInTheDocument())
    expect(write).toHaveBeenCalledTimes(2)
  }, 10_000)

  it("retries only the failed model device write and retains newer staged edits", async () => {
    inputs.realModelSelection = true
    inputs.otherProviderFirst = true
    const write = vi.fn().mockRejectedValueOnce(new Error("device unavailable")).mockResolvedValue(undefined)
    inputs.modelWrite = write
    useStoreMessageOption.getState().setSelectedModel("llamacpp:/models/gemma.gguf")
    useStoreChatModelSettings.getState().setActiveSettingsScope("llamacpp:/models/gemma.gguf")
    await mountSettings()
    fireEvent.click(getSettingsTab("Model"))
    fireEvent.mouseDown(getModelCombobox())
    fireEvent.click(await screen.findByText("OpenAI - tldw:/models/gemma.gguf"))
    await screen.findByRole("alert")
    const temperature = await screen.findByLabelText("modelSettings.form.temperature.label")
    fireEvent.change(temperature, { target: { value: "0.31" } })
    fireEvent.click(getSettingsTab("Conversation"))
    const prompt = await screen.findByPlaceholderText("Enter System Prompt")
    fireEvent.change(prompt, { target: { value: "Newer staged instructions" } })
    const revision = useStoreMessageOption.getState().selectedModelRevision
    fireEvent.click(getSettingsButton("Retry model save"))
    await waitFor(() => expect(write).toHaveBeenCalledTimes(2))
    expect({ revision: useStoreMessageOption.getState().selectedModelRevision, prompt: (prompt as HTMLInputElement).value })
      .toEqual({ revision, prompt: "Newer staged instructions" })
    expect(screen.queryByRole("button", { name: "Discard and switch" })).not.toBeInTheDocument()
    fireEvent.click(getSettingsTab("Model"))
    expect(screen.getByLabelText("modelSettings.form.temperature.label")).toHaveValue("0.31")
  }, 15_000)

  it("reports and retries a failed model Clear without reviving the previous choice", async () => {
    inputs.realModelSelection = true
    const write = vi.fn().mockRejectedValueOnce(new Error("device unavailable")).mockResolvedValue(undefined)
    inputs.modelWrite = write
    useStoreMessageOption.getState().setSelectedModel("llamacpp:/models/gemma.gguf")
    useStoreChatModelSettings.getState().setActiveSettingsScope("llamacpp:/models/gemma.gguf")
    await mountSettings()
    fireEvent.click(getSettingsTab("Model"))
    const select = getModelCombobox().closest(".ant-select")!
    fireEvent.mouseDown(select.querySelector(".ant-select-clear")!)
    expect(await screen.findByRole("alert")).toHaveTextContent(/could not be saved on this device/i)
    expect(useStoreMessageOption.getState().selectedModel).toBeNull()
    await waitFor(() => expect(getSettingsButton("Retry model save")).toBeEnabled())
    const revision = useStoreMessageOption.getState().selectedModelRevision
    fireEvent.click(getSettingsButton("Retry model save"))
    await waitFor(() => expect(write).toHaveBeenCalledTimes(2))
    expect([useStoreMessageOption.getState().selectedModel, useStoreMessageOption.getState().selectedModelRevision]).toEqual([null, revision])
  }, 15_000)

  it("does not publish an old model device error after model selection ABA", async () => {
    inputs.realModelSelection = true
    inputs.otherProviderFirst = true
    let reject!: (error: Error) => void
    const pending = new Promise<void>((_, rejectPromise) => { reject = rejectPromise })
    inputs.modelWrite = () => pending
    useStoreMessageOption.getState().setSelectedModel("llamacpp:/models/gemma.gguf")
    useStoreChatModelSettings.getState().setActiveSettingsScope("llamacpp:/models/gemma.gguf")
    await mountSettings()
    fireEvent.click(getSettingsTab("Model"))
    fireEvent.mouseDown(getModelCombobox())
    fireEvent.click(await screen.findByText("OpenAI - tldw:/models/gemma.gguf"))
    await screen.findByLabelText("modelSettings.form.temperature.label")
    await act(async () => {
      const state = useStoreMessageOption.getState()
      state.setSelectedModel("llamacpp:/models/gemma.gguf")
      state.setSelectedModel("openai:/models/gemma.gguf")
    })
    await screen.findByLabelText("modelSettings.form.temperature.label")
    await act(async () => { reject(new Error("old device failure")); await pending.catch(() => undefined) })
    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
  }, 10_000)

  it("opens the cockpit's owned alias scope and keeps its prompt through real Save", async () => {
    inputs.selectedModel = "llama_cpp:/models/gemma.gguf"
    inputs.provider = "llama"
    useStoreChatModelSettings.getState().reset()
    const state = useStoreChatModelSettings.getState()
    const ownedScope = "llamacpp:/models/gemma.gguf"
    state.setActiveSettingsScope(ownedScope)
    state.setSystemPrompt(instruction)
    const file = path.resolve(__dirname, "../../../Option/Playground/Playground.tsx")
    const ast = ts.createSourceFile(file, readFileSync(file, "utf8"), ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX)
    let callback: ts.Expression | undefined
    const visit = (node: ts.Node) => { if (ts.isVariableDeclaration(node) && node.name.getText(ast) === "openModelSettingsFromCockpit" && ts.isCallExpression(node.initializer!)) callback = node.initializer.arguments[0]; ts.forEachChild(node, visit) }
    visit(ast)
    if (!callback) throw new Error("Missing production cockpit settings opener")
    const open = vi.fn()
    const invoke = new Function("setActiveSettingsScope", "selectedModelSettingsScope", "providerRouteSummary", "openModelSettings", "COCKPIT_MODEL_SETTINGS_TRIGGER_SELECTOR", `return (${callback.getText(ast)})();`)
    act(() => invoke(state.setActiveSettingsScope, ownedScope, buildCockpitProviderRouteSummary({ selectedModel: inputs.selectedModel, selectedProvider: null }), open, "model-settings-trigger"))
    expect(open).toHaveBeenCalledWith({ returnFocusSelector: "model-settings-trigger", settingsScope: ownedScope })
    const { editor, close } = await mountSettings(open.mock.lastCall![0].settingsScope)
    expect(editor).toHaveValue(instruction)
    fireEvent.click(getMainSaveButton())
    await waitFor(() => expect(close).toHaveBeenCalledWith(false))
    expect(useStoreChatModelSettings.getState().activeSettingsScope).toBe(ownedScope)
    expect(useStoreChatModelSettings.getState().systemPrompt).toBe(instruction)
  })

  it("shows an existing conversation override when reopening the editor", async () => {
    useStoreChatModelSettings.getState().setSystemPrompt(instruction)
    const { editor } = await mountSettings()
    expect(editor).toHaveValue(instruction)
  })

  it("C01 settings metadata loading preserves the selected Conversation tab", async () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const view = await mountSettings(undefined, client)
    expect(screen.getByRole("tab", { name: "Conversation" })).toHaveAttribute("aria-selected", "true")
    let finish!: (value: { temperature: number }) => void
    inputs.settingsRead = () => new Promise(resolve => { finish = resolve })
    inputs.character = { id: "42", name: "Late loaded character" }
    view.rerender(
      <ConfigProvider theme={{ token: { motion: false } }}>
        <QueryClientProvider client={client}>
          <CurrentChatModelSettings open setOpen={view.close} />
        </QueryClientProvider>
      </ConfigProvider>
    )
    await waitFor(() => expect(finish).toBeTypeOf("function"))
    await act(async () => { finish({ temperature: 0.7 }) })
    const conversation = await screen.findByRole("tab", { name: "Conversation" })
    expect(conversation).toHaveAttribute("aria-selected", "true")
    expect(screen.getByPlaceholderText("Enter System Prompt")).toBeVisible()
    expect(view.close).not.toHaveBeenCalled()
  })

  it("C01 settings metadata tab state resets for a new dialog opening", async () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const close = vi.fn()
    const body = (open: boolean) => (
      <ConfigProvider theme={{ token: { motion: false } }}>
        <QueryClientProvider client={client}>
          <CurrentChatModelSettings open={open} setOpen={close} />
        </QueryClientProvider>
      </ConfigProvider>
    )
    const view = render(body(true))
    fireEvent.click(await screen.findByRole("tab", { name: "Conversation" }))
    await screen.findByPlaceholderText("Enter System Prompt")
    view.rerender(body(false))
    view.rerender(body(true))
    expect(await screen.findByRole("tab", { name: "Model" })).toHaveAttribute("aria-selected", "true")
    expect(close).not.toHaveBeenCalled()
  })

  it("keeps generation settings through a cached dialog remount and Save", async () => {
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    })
    const first = await mountSettings(undefined, client)
    fireEvent.click(getSettingsTab("Model"))
    fireEvent.change(screen.getByLabelText("modelSettings.form.numPredict.label"), {
      target: { value: "48" },
    })
    fireEvent.change(screen.getByLabelText("modelSettings.form.temperature.label"), {
      target: { value: "0.25" },
    })
    fireEvent.click(getMainSaveButton())
    await waitFor(() => expect(first.close).toHaveBeenCalledWith(false))
    expect(useStoreChatModelSettings.getState().numPredict).toBe(48)
    first.unmount()

    const reopened = await mountSettings(undefined, client)
    fireEvent.click(getSettingsTab("Model"))
    expect(screen.getByLabelText("modelSettings.form.numPredict.label")).toHaveValue("48")
    expect(screen.getByLabelText("modelSettings.form.temperature.label")).toHaveValue("0.25")
    fireEvent.click(getMainSaveButton())
    await waitFor(() => expect(reopened.close).toHaveBeenCalledWith(false))
    expect(useStoreChatModelSettings.getState()).toMatchObject({
      numPredict: 48,
      temperature: 0.25,
      apiProvider: "llama.cpp",
    })
  }, 10000)

  it("does not revive a cached default after explicitly clearing a numeric setting", async () => {
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    })
    await client.prefetchQuery({
      queryKey: ["fetchModelConfig2", true, null],
      queryFn: getAllModelSettings,
    })
    useStoreChatModelSettings.getState().setTemperature(0.25)
    const clearing = await mountSettings(undefined, client)
    fireEvent.click(getSettingsTab("Model"))
    expect(screen.getByLabelText("modelSettings.form.temperature.label")).toHaveValue("0.25")
    fireEvent.change(screen.getByLabelText("modelSettings.form.temperature.label"), {
      target: { value: "" },
    })
    fireEvent.click(getMainSaveButton())
    await waitFor(() =>
      expect(useStoreChatModelSettings.getState().temperature).toBeUndefined(),
    )
    clearing.unmount()
    const reopened = await mountSettings(undefined, client)
    fireEvent.click(getSettingsTab("Model"))
    expect(screen.getByLabelText("modelSettings.form.temperature.label")).toHaveValue("")
    fireEvent.click(getMainSaveButton())
    await waitFor(() => expect(reopened.close).toHaveBeenCalledWith(false))
    expect(useStoreChatModelSettings.getState().temperature).toBeUndefined()
  }, 10000)

  it.each(["saved", "reset-default", "reset-template", "owner-changed"])(
    "shows the current prompt when the dialog remounts with the existing query cache (%s)",
    async (state) => {
      if (state === "reset-template") inputs.selectedSystemPrompt = "template-1"
      const client = new QueryClient({
        defaultOptions: { queries: { retry: false } },
      })
      const first = await mountSettings(undefined, client)
      fireEvent.change(first.editor, { target: { value: instruction } })
      fireEvent.click(getMainSaveButton())
      await waitFor(() => expect(first.close).toHaveBeenCalledWith(false))
      let expected = instruction
      if (state.startsWith("reset")) {
        const template = state === "reset-template"
        fireEvent.click(
          getPromptResetButton(template ? "Reset to template" : "Reset to default"),
        )
        expected = template ? "Selected template instructions" : ""
        await waitFor(() => expect(first.editor).toHaveValue(expected))
        fireEvent.click(getMainSaveButton())
        await waitFor(() => expect(useStoreChatModelSettings.getState().systemPrompt).toBe(expected))
      } else if (state === "owner-changed") {
        act(() =>
          window.dispatchEvent(new Event("tldw:auth-principal-changed")),
        )
        expected = ""
        useStoreMessageOption.getState().startNewChatDraft(false)
      }
      first.unmount()
      const reopened = await mountSettings(undefined, client)
      expect(reopened.editor).toHaveValue(expected)
      expect(useStoreChatModelSettings.getState().systemPrompt).toBe(
        state === "owner-changed" ? undefined : expected,
      )
    },
    10_000,
  )

  it.each([
    "llama-cpp:tldw:/models/gemma.gguf",
    "tldw:llama-cpp:/models/gemma.gguf",
  ])(
    "keeps a recognized legacy provider alias for the same model (%s)",
    async (legacyScope) => {
      useStoreChatModelSettings.getState().setActiveSettingsScope(legacyScope)
      const { editor, close } = await mountSettings()
      fireEvent.change(editor, { target: { value: instruction } })
      fireEvent.click(getMainSaveButton())
      await waitFor(() => expect(close).toHaveBeenCalledWith(false))
      expect(useStoreChatModelSettings.getState().systemPrompt).toBe(
        instruction,
      )
      expect(useStoreChatModelSettings.getState().activeSettingsScope).toBe(
        legacyScope,
      )
    },
  )

  it.each(["lmstudio", "llamafile"])(
    "keeps a same-model cockpit scope for catalog provider %s",
    async (provider) => {
      inputs.provider = provider
      const settings = useStoreChatModelSettings.getState()
      settings.setActiveSettingsScope(undefined)
      settings.updateSetting("apiProvider", provider)
      const route = buildCockpitProviderRouteSummary({
        selectedProvider: provider,
        selectedModel: "tldw:/models/gemma.gguf",
      })
      settings.setActiveSettingsScope(route.providerRouteLabel)
      const { editor, close } = await mountSettings()
      fireEvent.change(editor, { target: { value: instruction } })
      fireEvent.click(getMainSaveButton())
      await waitFor(() => expect(close).toHaveBeenCalledWith(false))
      expect(useStoreChatModelSettings.getState().systemPrompt).toBe(
        instruction,
      )
      expect(useStoreChatModelSettings.getState().activeSettingsScope).toBe(
        route.providerRouteLabel,
      )
    },
  )

  it.each(["tldw:/models/gemma.gguf", "llamacpp:/models/gemma.gguf"])(
    "keeps the selected provider when another catalog provider has the same model ID (%s)",
    async (selectedModel) => {
      inputs.selectedModel = selectedModel
      inputs.otherProviderFirst = true
      const cockpitScope =
        useStoreChatModelSettings.getState().activeSettingsScope
      const { editor, close } = await mountSettings()
      fireEvent.change(editor, { target: { value: instruction } })
      fireEvent.click(getMainSaveButton())
      await waitFor(() => expect(close).toHaveBeenCalledWith(false))
      expect(useStoreChatModelSettings.getState().systemPrompt).toBe(
        instruction,
      )
      expect(useStoreChatModelSettings.getState().activeSettingsScope).toBe(
        cockpitScope,
      )
    },
  )

  it.each(["llama.cpp:/models/other.gguf", "openai:/models/gemma.gguf"])(
    "still saves to the selected model when the active scope belongs to another model/provider (%s)",
    async (previousScope) => {
      const settings = useStoreChatModelSettings.getState()
      settings.setActiveSettingsScope(previousScope)
      settings.setSystemPrompt("Previous model instructions")
      settings.updateScopedSetting(
        "llamacpp:/models/gemma.gguf",
        "systemPrompt",
        "Selected model instructions",
      )
      const { close } = await mountSettings()
      fireEvent.click(getMainSaveButton())
      await waitFor(() => expect(close).toHaveBeenCalledWith(false))
      expect(useStoreChatModelSettings.getState().activeSettingsScope).toBe(
        "llamacpp:/models/gemma.gguf",
      )
      expect(useStoreChatModelSettings.getState().systemPrompt).toBe(
        "Selected model instructions",
      )
      expect(
        useStoreChatModelSettings.getState().scopedSettingsByModelKey[
          previousScope
        ].systemPrompt,
      ).toBe("Previous model instructions")
    },
  )

  it("retains an explicitly requested scope ahead of the active model alias", async () => {
    const explicitScope = "llamacpp:/models/gemma.gguf"
    useStoreChatModelSettings
      .getState()
      .updateScopedSetting(
        explicitScope,
        "systemPrompt",
        "Explicit scope instructions",
      )
    const { close } = await mountSettings(explicitScope)
    fireEvent.click(getMainSaveButton())
    await waitFor(() => expect(close).toHaveBeenCalledWith(false))
    expect(useStoreChatModelSettings.getState().activeSettingsScope).toBe(
      explicitScope,
    )
    expect(useStoreChatModelSettings.getState().systemPrompt).toBe(
      "Explicit scope instructions",
    )
  })

  it("does not mistake a foreign provider scope for the raw selected model when provider preference is absent", async () => {
    useStoreChatModelSettings.getState().reset()
    const foreignScope = "openai:/models/gemma.gguf"
    useStoreChatModelSettings.getState().setActiveSettingsScope(foreignScope)
    useStoreChatModelSettings
      .getState()
      .setSystemPrompt("Foreign model instructions")
    useStoreChatModelSettings
      .getState()
      .updateScopedSetting(
        inputs.selectedModel,
        "systemPrompt",
        "Selected model instructions",
      )
    const { close } = await mountSettings()
    fireEvent.click(getMainSaveButton())
    await waitFor(() => expect(close).toHaveBeenCalledWith(false))
    expect(useStoreChatModelSettings.getState().activeSettingsScope).toBe(
      inputs.selectedModel,
    )
    expect(useStoreChatModelSettings.getState().systemPrompt).toBe(
      "Selected model instructions",
    )
  })

  it.each([
    { raw: "", cached: false },
    { raw: "", cached: true },
    { raw: " \n ", cached: false },
    { raw: " \n ", cached: true },
  ])("C01 a cleared override displays the selected instruction (raw=$raw cached=$cached)", async ({ raw, cached }) => {
    inputs.selectedSystemPrompt = "pirate-template"
    useStoreChatModelSettings.getState().setSystemPrompt(raw)
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    if (cached) await client.prefetchQuery({ queryKey: ["fetchModelConfig2", true, null], queryFn: getAllModelSettings })
    const view = await mountSettings(undefined, client)
    await waitFor(() => expect(view.editor).toHaveValue("Selected template instructions"))
    expect(useStoreChatModelSettings.getState().systemPrompt).toBe(raw)
    view.unmount()
    client.clear()
  })

  it("C01 reopening the same selected template retains its instruction with cached settings", async () => {
    inputs.selectedSystemPrompt = "pirate-template"
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const first = await mountSettings(undefined, client)
    await waitFor(() => expect(first.editor).toHaveValue("Selected template instructions"))
    first.unmount()
    const reopened = await mountSettings(undefined, client)
    expect(reopened.editor).toHaveValue("Selected template instructions")
    reopened.unmount()
    client.clear()
  })

  it("C01 a same-ID template edit is resolved again on dialog reopening", async () => {
    inputs.selectedSystemPrompt = "pirate-template"
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const first = await mountSettings(undefined, client)
    await waitFor(() => expect(first.editor).toHaveValue("Selected template instructions"))
    first.unmount()
    inputs.templateText = "Edited pirate instructions: ARRR"
    const reopened = await mountSettings(undefined, client)
    await waitFor(() => expect(reopened.editor).toHaveValue("Edited pirate instructions: ARRR"))
    reopened.unmount()
    client.clear()
  })

  it.each(["typed edit", "account change"])("C01 a delayed instruction cannot overwrite %s", async boundary => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    await client.prefetchQuery({ queryKey: ["fetchModelConfig2", true, null], queryFn: getAllModelSettings })
    let finish!: (value: string) => void
    inputs.selectedSystemPrompt = "pirate-template"
    inputs.templateRead = () => new Promise(resolve => { finish = resolve })
    const view = await mountSettings(undefined, client)
    if (boundary === "typed edit") fireEvent.change(view.editor, { target: { value: "Newer user instruction" } })
    else act(() => window.dispatchEvent(new Event("tldw:auth-principal-changed")))
    await act(async () => { finish("Retired private instruction"); await Promise.resolve() })
    expect(view.editor).toHaveValue(boundary === "typed edit" ? "Newer user instruction" : "")
    expect(useStoreChatModelSettings.getState().systemPrompt).toBe(boundary === "typed edit" ? "Newer user instruction" : undefined)
    view.unmount()
    client.clear()
  })

  it("C01 Reset finishing after dialog unmount cannot restore a retired instruction", async () => {
    inputs.selectedSystemPrompt = "pirate-template"
    const view = await mountSettings()
    await waitFor(() => expect(view.editor).toHaveValue("Selected template instructions"))
    let finish!: (value: string) => void
    inputs.templateRead = () => new Promise(resolve => { finish = resolve })
    fireEvent.click(screen.getByRole("button", { name: "Reset to template" }))
    expect(finish).toBeTypeOf("function")
    view.unmount()
    act(() => window.dispatchEvent(new Event("tldw:auth-principal-changed")))
    await act(async () => { finish("Retired private instruction"); await Promise.resolve() })
    expect(useStoreChatModelSettings.getState().systemPrompt).toBeUndefined()
  })

  it("C01 cached defaults cannot hide the newly selected instruction", async () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    await client.prefetchQuery({
      queryKey: ["fetchModelConfig2", true, null],
      queryFn: getAllModelSettings,
    })
    inputs.selectedSystemPrompt = "pirate-template"
    const reopened = await mountSettings(undefined, client)
    await waitFor(() => expect(reopened.editor).toHaveValue("Selected template instructions"))
    reopened.unmount()
    client.clear()
  })

  it.each([null, "template-1"])(
    "stages the reset value until Save (template=%s)",
    async (template) => {
      inputs.selectedSystemPrompt = template
      const { editor, close } = await mountSettings()
      fireEvent.change(editor, { target: { value: instruction } })
      fireEvent.click(
        getPromptResetButton(template ? "Reset to template" : "Reset to default"),
      )
      await waitFor(() =>
        expect(editor).toHaveValue(
          template ? "Selected template instructions" : "",
        ),
      )
      expect(useStoreChatModelSettings.getState().systemPrompt).toBeUndefined()
      fireEvent.click(getMainSaveButton())
      await waitFor(() => expect(close).toHaveBeenCalledWith(false))
      expect(useStoreChatModelSettings.getState().systemPrompt).toBe(template ? "Selected template instructions" : "")
    },
  )

  it.each([null, "template-1"])(
    "does not revive a saved override when the cockpit reactivates its model scope after Reset (template=%s)",
    async (template) => {
      inputs.selectedSystemPrompt = template
      const cockpitScope =
        useStoreChatModelSettings.getState().activeSettingsScope
      const { editor, close } = await mountSettings()
      fireEvent.change(editor, { target: { value: instruction } })
      fireEvent.click(getMainSaveButton())
      await waitFor(() => expect(close).toHaveBeenCalledWith(false))
      fireEvent.click(
        getPromptResetButton(template ? "Reset to template" : "Reset to default"),
      )
      const resetValue = template ? "Selected template instructions" : ""
      await waitFor(() => expect(editor).toHaveValue(resetValue))
      fireEvent.click(getMainSaveButton())
      await waitFor(() => expect(useStoreChatModelSettings.getState().systemPrompt).toBe(resetValue))
      act(() =>
        useStoreChatModelSettings
          .getState()
          .setActiveSettingsScope(cockpitScope),
      )
      expect(useStoreChatModelSettings.getState().systemPrompt).toBe(resetValue)
    },
  )

  it("preserves same-owner readiness notifications but clears instructions at an account boundary", async () => {
    const { editor } = await mountSettings(
      useStoreChatModelSettings.getState().activeSettingsScope,
    )
    fireEvent.change(editor, { target: { value: instruction } })
    fireEvent.click(getMainSaveButton())
    await act(async () => {
      window.dispatchEvent(
        new CustomEvent("tldw:config-updated", {
          detail: { authorityChanged: false },
        }),
      )
    })
    expect(useStoreChatModelSettings.getState().systemPrompt).toBe(instruction)
    await act(async () => {
      window.dispatchEvent(new Event("tldw:auth-principal-changed"))
    })
    expect(useStoreChatModelSettings.getState().systemPrompt).toBeUndefined()
  })
  it.each(
    [false, true].flatMap((readiness) =>
      [false, true].map((explicitProvider) => ({
        readiness,
        explicitProvider,
      })),
    ),
  )(
    "uses the saved instruction in the next ordinary image prompt (same-owner readiness=$readiness, explicit provider=$explicitProvider)",
    async ({ readiness, explicitProvider }) => {
      if (!explicitProvider) {
        useStoreChatModelSettings.getState().reset()
        useStoreChatModelSettings
          .getState()
          .setActiveSettingsScope(
            buildCockpitProviderRouteSummary({
              selectedProvider: undefined,
              selectedModel: inputs.selectedModel,
            }).providerRouteLabel,
          )
      }
      const { editor, close } = await mountSettings()
      fireEvent.change(editor, { target: { value: instruction } })
      fireEvent.click(getMainSaveButton())
      await waitFor(() => expect(close).toHaveBeenCalledWith(false))
      if (readiness)
        await act(async () => {
          window.dispatchEvent(
            new CustomEvent("tldw:config-updated", {
              detail: { authorityChanged: false },
            }),
          )
        })
      const image =
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAFklEQVR4nGP8z8DAwMDAxMDAwMDAAAANHQEDasKb6QAAAABJRU5ErkJggg=="
      const prepared = (await normalChatMode(
        "Describe this image",
        image,
        false,
        [],
        [],
        new AbortController().signal,
        {
          selectedModel: "tldw:/models/gemma.gguf",
          useOCR: false,
          selectedSystemPrompt: "",
          currentChatModelSettings: useStoreChatModelSettings.getState(),
          setMessages: vi.fn(),
          setHistory: vi.fn(),
          setIsProcessing: vi.fn(),
          setStreaming: vi.fn(),
          setAbortController: vi.fn(),
          historyId: null,
          setHistoryId: vi.fn(),
          saveMessageOnSuccess: vi.fn(),
          saveMessageOnError: vi.fn(),
        },
      )) as unknown as { chatHistory: BaseMessage[]; humanMessage: BaseMessage }
      expect(
        prepared.chatHistory.map((message) => ({
          role: message._getType(),
          content: message.content,
        })),
      ).toEqual([{ role: "system", content: instruction }])
      expect(prepared.humanMessage.content).toEqual([
        { type: "text", text: "Describe this image" },
        { type: "image_url", image_url: image },
      ])
    },
  )

  it("cannot restore the previous owner's cached form prompt by clicking Save after invalidation", async () => {
    const { editor, close } = await mountSettings()
    fireEvent.change(editor, { target: { value: instruction } })
    await act(async () => {
      window.dispatchEvent(new Event("tldw:auth-principal-changed"))
    })
    expect(screen.queryByRole("button", { name: /^save$/i })).not.toBeInTheDocument()
    expect(saveActorSettingsForChat).not.toHaveBeenCalled()
    expect(close).not.toHaveBeenCalled()
    expect(useStoreChatModelSettings.getState().systemPrompt).toBeUndefined()
    act(() => useStoreMessageOption.getState().startNewChatDraft(false))
    fireEvent.click(await screen.findByRole("button", { name: /^save$/i }))
    await waitFor(() => expect(close).toHaveBeenCalledWith(false))
    expect(useStoreChatModelSettings.getState().systemPrompt).toBe("")
    expect(
      Object.values(
        useStoreChatModelSettings.getState().scopedSettingsByModelKey,
      ).some((settings) => settings.systemPrompt === instruction),
    ).toBe(false)
  })

  it("keeps Character selection and template identity through Save and Reset", async () => {
    inputs.character = { id: "testbot", name: "TestBot" }
    inputs.selectedSystemPrompt = "template-1"
    useStoreChatModelSettings
      .getState()
      .updateSetting("systemPromptTemplateId", "template-1")
    const { editor, close } = await mountSettings()
    fireEvent.change(editor, { target: { value: instruction } })
    fireEvent.click(getMainSaveButton())
    await waitFor(() => expect(close).toHaveBeenCalledWith(false))
    expect(useStoreChatModelSettings.getState().systemPromptTemplateId).toBe(
      "template-1",
    )
    fireEvent.click(getPromptResetButton("Reset to template"))
    await waitFor(() =>
      expect(editor).toHaveValue("Selected template instructions"),
    )
    expect(useStoreChatModelSettings.getState().systemPrompt).toBe(instruction)
    fireEvent.click(getMainSaveButton())
    await waitFor(() => expect(useStoreChatModelSettings.getState().systemPrompt).toBe("Selected template instructions"))
    expect(inputs.character?.id).toBe("testbot")
  })
})
