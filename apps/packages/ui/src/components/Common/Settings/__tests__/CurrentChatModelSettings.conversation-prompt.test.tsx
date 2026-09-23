import React from "react"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { normalChatMode } from "@/hooks/chat-modes/normalChatMode"
import type { BaseMessage } from "@/types/messages"
import { CurrentChatModelSettings } from "../CurrentChatModelSettings"
import { useStoreChatModelSettings } from "@/store/model"
import { useActorStore } from "@/store/actor"
import { buildCockpitProviderRouteSummary } from "@/components/Option/Playground/playground-cockpit-summaries"

const inputs = vi.hoisted(() => ({
  selectedSystemPrompt: null as string | null,
  character: null as { id: string; name: string } | null,
  provider: "llama.cpp",
  selectedModel: "tldw:/models/gemma.gguf",
  otherProviderFirst: false,
}))
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string | { defaultValue?: string }) =>
      typeof fallback === "string" ? fallback : (fallback?.defaultValue ?? key),
  }),
}))
vi.mock("@/hooks/useMessageOption", () => ({
  useMessageOption: () => ({
    historyId: null,
    serverChatId: null,
    selectedSystemPrompt: inputs.selectedSystemPrompt,
    selectedModel: inputs.selectedModel,
    uploadedFiles: [],
    removeUploadedFile: vi.fn(),
    setSelectedModel: vi.fn(),
    setServerChatState: vi.fn(),
    setServerChatTopic: vi.fn(),
    setServerChatVersion: vi.fn(),
    setServerChatPersonaMemoryMode: vi.fn(),
  }),
}))
vi.mock("@/hooks/useSelectedCharacter", () => ({
  useSelectedCharacter: () => [inputs.character, vi.fn(), { isLoading: false }],
}))
vi.mock("@/hooks/useSelectedAssistant", () => ({
  useSelectedAssistant: () => [null],
}))
vi.mock("@/hooks/chat/useChatSettingsRecord", () => ({
  useChatSettingsRecord: () => ({
    settings: null,
    updateSettings: vi.fn(),
    chatKey: null,
  }),
}))
vi.mock("@/services/tldw-server", () => ({
  fetchChatModels: async () => [
    ...(inputs.otherProviderFirst
      ? [{ model: "tldw:/models/gemma.gguf", provider: "openai" }]
      : []),
    { model: "tldw:/models/gemma.gguf", provider: inputs.provider },
  ],
  systemPromptForNonRagOption: async () => "",
}))
vi.mock("@/services/model-settings", () => ({
  getAllModelSettings: async () => ({ temperature: 0.7 }),
}))
vi.mock("@/services/ocr", () => ({ getOCRLanguage: async () => null }))
vi.mock("@/services/actor-settings", () => ({
  getActorSettingsForChatWithCharacterFallback: async () => null,
  saveActorSettingsForChat: vi.fn(),
}))
vi.mock("@/components/Common/system-prompt-utils", () => ({
  resolveSelectedSystemPromptContent: async (id?: string | null) =>
    id ? "Selected template instructions" : "",
}))
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
  ModelBasicsTab: () => null,
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
vi.mock("@/db/dexie/helpers", () => ({ getPromptById: async () => null }))
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

const instruction =
  "UAT363 RETRY ORBIT742. Describe visible image contents accurately in one short sentence."
const mountSettings = async (
  settingsScope?: string,
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } }),
) => {
  const close = vi.fn()
  const view = render(
    <QueryClientProvider client={client}>
      <CurrentChatModelSettings
        open
        setOpen={close}
        settingsScope={settingsScope}
      />
    </QueryClientProvider>,
  )
  fireEvent.click(await screen.findByRole("tab", { name: "Conversation" }))
  const editor = await screen.findByPlaceholderText("Enter System Prompt")
  return { ...view, close, editor }
}

describe("Conversation prompt editor through real model settings Save", () => {
  beforeEach(() => {
    inputs.selectedSystemPrompt = null
    inputs.character = null
    inputs.provider = "llama.cpp"
    inputs.selectedModel = "tldw:/models/gemma.gguf"
    inputs.otherProviderFirst = false
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
    useActorStore.setState({ settings: null })
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
      expect(useStoreChatModelSettings.getState().systemPrompt).toBe(
        instruction,
      )
      fireEvent.click(screen.getByRole("button", { name: /^save$/i }))
      await waitFor(() => expect(close).toHaveBeenCalledWith(false))
      expect(useStoreChatModelSettings.getState().systemPrompt).toBe(
        instruction,
      )
    },
  )

  it("shows an existing conversation override when reopening the editor", async () => {
    useStoreChatModelSettings.getState().setSystemPrompt(instruction)
    const { editor } = await mountSettings()
    expect(editor).toHaveValue(instruction)
  })

  it.each(["saved", "reset-default", "reset-template", "owner-changed"])(
    "shows the current prompt when the dialog remounts with the existing query cache (%s)",
    async (state) => {
      if (state === "reset-template") inputs.selectedSystemPrompt = "template-1"
      const client = new QueryClient({
        defaultOptions: { queries: { retry: false } },
      })
      const first = await mountSettings(undefined, client)
      fireEvent.change(first.editor, { target: { value: instruction } })
      fireEvent.click(screen.getByRole("button", { name: /^save$/i }))
      await waitFor(() => expect(first.close).toHaveBeenCalledWith(false))
      let expected = instruction
      if (state.startsWith("reset")) {
        const template = state === "reset-template"
        fireEvent.click(
          screen.getByRole("button", {
            name: template ? "Reset to template" : "Reset to default",
          }),
        )
        expected = template ? "Selected template instructions" : ""
        await waitFor(() => expect(first.editor).toHaveValue(expected))
      } else if (state === "owner-changed") {
        act(() =>
          window.dispatchEvent(new Event("tldw:auth-principal-changed")),
        )
        expected = ""
      }
      first.unmount()
      const reopened = await mountSettings(undefined, client)
      expect(reopened.editor).toHaveValue(expected)
      expect(useStoreChatModelSettings.getState().systemPrompt).toBe(
        state === "owner-changed" ? undefined : expected,
      )
    },
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
      fireEvent.click(screen.getByRole("button", { name: /^save$/i }))
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
      fireEvent.click(screen.getByRole("button", { name: /^save$/i }))
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
      fireEvent.click(screen.getByRole("button", { name: /^save$/i }))
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
      fireEvent.click(screen.getByRole("button", { name: /^save$/i }))
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
    fireEvent.click(screen.getByRole("button", { name: /^save$/i }))
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
        "llamacpp:/models/gemma.gguf",
        "systemPrompt",
        "Selected model instructions",
      )
    const { close } = await mountSettings()
    fireEvent.click(screen.getByRole("button", { name: /^save$/i }))
    await waitFor(() => expect(close).toHaveBeenCalledWith(false))
    expect(useStoreChatModelSettings.getState().activeSettingsScope).toBe(
      "llamacpp:/models/gemma.gguf",
    )
    expect(useStoreChatModelSettings.getState().systemPrompt).toBe(
      "Selected model instructions",
    )
  })

  it.each([null, "template-1"])(
    "shows the reset value in the editor and active settings (template=%s)",
    async (template) => {
      inputs.selectedSystemPrompt = template
      const { editor } = await mountSettings()
      fireEvent.change(editor, { target: { value: instruction } })
      fireEvent.click(
        screen.getByRole("button", {
          name: template ? "Reset to template" : "Reset to default",
        }),
      )
      await waitFor(() =>
        expect(editor).toHaveValue(
          template ? "Selected template instructions" : "",
        ),
      )
      expect(useStoreChatModelSettings.getState().systemPrompt).toBe(
        template ? "Selected template instructions" : "",
      )
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
      fireEvent.click(screen.getByRole("button", { name: /^save$/i }))
      await waitFor(() => expect(close).toHaveBeenCalledWith(false))
      fireEvent.click(
        screen.getByRole("button", {
          name: template ? "Reset to template" : "Reset to default",
        }),
      )
      const resetValue = template ? "Selected template instructions" : ""
      await waitFor(() =>
        expect(useStoreChatModelSettings.getState().systemPrompt).toBe(
          resetValue,
        ),
      )
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
    fireEvent.click(screen.getByRole("button", { name: /^save$/i }))
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
      fireEvent.click(screen.getByRole("button", { name: /^save$/i }))
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
    fireEvent.click(screen.getByRole("button", { name: /^save$/i }))
    await waitFor(() => expect(close).toHaveBeenCalledWith(false))
    expect(useStoreChatModelSettings.getState().systemPrompt).toBeUndefined()
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
    fireEvent.click(screen.getByRole("button", { name: /^save$/i }))
    await waitFor(() => expect(close).toHaveBeenCalledWith(false))
    expect(useStoreChatModelSettings.getState().systemPromptTemplateId).toBe(
      "template-1",
    )
    fireEvent.click(screen.getByRole("button", { name: "Reset to template" }))
    await waitFor(() =>
      expect(editor).toHaveValue("Selected template instructions"),
    )
    expect(useStoreChatModelSettings.getState().systemPrompt).toBe(
      "Selected template instructions",
    )
    expect(inputs.character?.id).toBe("testbot")
  })
})
