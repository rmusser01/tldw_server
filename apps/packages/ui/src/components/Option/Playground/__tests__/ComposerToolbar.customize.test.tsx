import React from "react"
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within
} from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, afterEach, describe, expect, it, vi } from "vitest"
import { Storage } from "@plasmohq/storage"
import i18next from "i18next"
import { I18nextProvider } from "react-i18next"
import playground from "@/assets/locale/en/playground.json"
import common from "@/assets/locale/en/common.json"
import { ComposerToolbar } from "../ComposerToolbar"
import { PlaygroundMcpControl } from "../PlaygroundMcpControl"
import { PlaygroundModeLauncher } from "../PlaygroundModeLauncher"
import { PlaygroundToolsPopover } from "../PlaygroundToolsPopover"
import { MemoryRouter } from "react-router-dom"
import { ConfigProvider } from "antd"
import { PLAYGROUND_APPEND_FORMATTING_GUIDE_PROMPT_STORAGE_KEY } from "@/utils/output-formatting-guide"

const io = vi.hoisted(() => ({
  assistant: vi.fn(),
  prompt: vi.fn(),
  realAssistant: false
}))
// Actual toolbar/AntD/English/WebUI storage shim; only existing domain editors are leaf doubles.
vi.mock("@/components/Common/AssistantSelect", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/components/Common/AssistantSelect")>()
  return {
    AssistantSelect: (
      props: React.ComponentProps<typeof actual.AssistantSelect>
    ) => {
      io.assistant(props)
      return io.realAssistant ? (
        <actual.AssistantSelect {...props} />
      ) : (
        <button type="button">Choose character or persona</button>
      )
    }
  }
})
vi.mock("@/hooks/useSelectedAssistant", () => ({
  useSelectedAssistant: () => [null, vi.fn(), { isLoading: false }]
}))
vi.mock("@/hooks/chat/useChatSettingsRecord", () => ({
  useChatSettingsRecord: () => ({ settings: null, updateSettings: vi.fn() })
}))
vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: async () => undefined,
    listAllCharacters: async () => [{ id: 7, name: "Mira" }],
    listPersonaProfiles: async () => []
  }
}))

vi.mock("@/components/Common/PromptSelect", () => ({
  PromptSelect: (props: unknown) => {
    io.prompt(props)
    return <button type="button">Choose system prompt</button>
  }
}))
vi.mock("@/components/Common/PersonaBuddy/BuddyManagementButton", () => ({
  BuddyManagementButton: () => <button type="button">Buddy & Persona</button>
}))
vi.mock("@/components/Layouts/ConnectionStatus", () => ({
  ConnectionStatus: () => <span>Connection status</span>
}))
vi.mock("../playground-features", () => ({
  ParameterPresets: () => <button type="button">Parameter presets</button>,
  ParameterPresetsDropdown: ({
    onChange
  }: {
    onChange?: (id: string) => void
  }) => (
    <button onClick={() => onChange?.("balanced")}>Balanced generation</button>
  ),
  SystemPromptTemplatesButton: () => (
    <button type="button">System prompt templates</button>
  ),
  SystemPromptTemplatesModal: () => null,
  SessionCostEstimation: () => <button type="button">Session cost</button>
}))

const translations = i18next.createInstance()
await translations.init({
  lng: "en",
  fallbackLng: "en",
  resources: { en: { playground, common } },
  interpolation: { escapeValue: false }
})
type Props = React.ComponentProps<typeof ComposerToolbar>
const props = (overrides: Partial<Props> = {}): Props => ({
  isProMode: false,
  isMobile: false,
  isConnectionReady: true,
  isSending: false,
  modelSelectButton: <button type="button">Choose model</button>,
  mcpControl: <button type="button">MCP tools</button>,
  sendControl: <button type="button">Send</button>,
  attachmentButton: <button type="button">Attach</button>,
  toolsButton: <button type="button">Tools</button>,
  voiceChatButton: <button type="button">Voice chat</button>,
  modelUsageBadge: null,
  selectedSystemPrompt: undefined,
  systemPrompt: "",
  setSystemPrompt: vi.fn(),
  setSelectedSystemPrompt: vi.fn(),
  setSelectedQuickPrompt: vi.fn(),
  temporaryChat: false,
  onToggleTemporaryChat: vi.fn(),
  privateChatLocked: false,
  isFireFoxPrivateMode: false,
  persistenceTooltip: "History on",
  contextToolsOpen: false,
  onToggleKnowledgePanel: vi.fn(),
  webSearch: false,
  onToggleWebSearch: vi.fn(),
  hasWebSearch: true,
  onOpenModelSettings: vi.fn(),
  modelSummaryLabel: "Mira model",
  promptSummaryLabel: "No prompt",
  hasDictation: true,
  speechAvailable: true,
  speechUsesServer: false,
  isListening: false,
  isServerDictating: false,
  voiceChatEnabled: false,
  speechTooltip: "Dictation",
  onDictationToggle: vi.fn(),
  onTemplateSelect: vi.fn(),
  selectedModel: "mira-model",
  currentProvider: "openai",
  resolvedProviderKey: "openai",
  messages: [],
  selectedDocumentsCount: 0,
  uploadedFilesCount: 0,
  serverChatId: null,
  showServerPersistenceHint: false,
  onDismissServerPersistenceHint: vi.fn(),
  onFocusConnectionCard: vi.fn(),
  modeLauncherButton: <button type="button">Chat modes</button>,
  compareControl: <button type="button">Compare models</button>,
  openUIRequestButton: <button type="button">Interactive response</button>,
  researchLaunchButton: <button type="button">Research</button>,
  rolePlayActions: { onOpenRolePlaySetup: vi.fn() },
  contextItems: [],
  ...overrides
})
const mount = (value = props()) => {
  const node = (next: Props) => (
    <ConfigProvider theme={{ token: { motion: false } }}>
      <I18nextProvider i18n={translations}>
        <ComposerToolbar {...next} />
      </I18nextProvider>
    </ConfigProvider>
  )
  const rendered = render(node(value))
  return {
    ...rendered,
    rerenderToolbar: (next: Props) => rendered.rerender(node(next))
  }
}
const customize = () => screen.getByRole("button", { name: "Customize chat" })
const openCustomize = async () => {
  await waitFor(() => expect(customize()).toBeEnabled())
  fireEvent.click(customize())
  return screen.findByRole("region", { name: "Customize chat" })
}
beforeEach(() => {
  io.realAssistant = false
  localStorage.clear()
  io.assistant.mockClear()
  io.prompt.mockClear()
})
afterEach(() => vi.restoreAllMocks())

describe("Casual task-based customization", () => {
  it("uses the visible message fallback when mode replacement has no visible toolbar control", async () => {
    const Harness = ({ pro }: { pro: boolean }) => {
      const fallback = React.useRef<HTMLInputElement>(null)
      return (
        <I18nextProvider i18n={translations}>
          <input aria-label="Message" ref={fallback} />
          <ComposerToolbar
            {...props({
              isProMode: pro,
              optionsExpanded: false,
              focusFallbackRef: fallback
            })}
          />
        </I18nextProvider>
      )
    }
    const view = render(<Harness pro={false} />)
    const region = await openCustomize()
    within(region).getByRole("button", { name: "MCP tools" }).focus()
    view.rerender(<Harness pro />)
    await waitFor(() =>
      expect(screen.getByRole("textbox", { name: "Message" })).toHaveFocus()
    )
  })

  it("preserves focus on another task when the actual picker closes after a watched close", async () => {
    io.realAssistant = true
    const user = userEvent.setup()
    render(
      <ConfigProvider theme={{ token: { motion: false } }}>
        <I18nextProvider i18n={translations}>
          <button>Other task</button>
          <ComposerToolbar {...props()} />
        </I18nextProvider>
      </ConfigProvider>
    )
    await openCustomize()
    await user.click(screen.getByTestId("character-select"))
    const tab = await screen.findByRole("tab", { name: "Characters" })
    tab.focus()
    await act(async () => {
      await new Storage().set(
        "playgroundComposerCasualAdvancedControlsOpen",
        false
      )
    })
    await user.click(screen.getByRole("button", { name: "Other task" }))
    await act(async () => {
      await new Promise<void>((resolve) =>
        window.requestAnimationFrame(() =>
          window.requestAnimationFrame(() => resolve())
        )
      )
    })
    expect(screen.getByRole("button", { name: "Other task" })).toHaveFocus()
  })

  it("returns to visible Customize after a watched close and the actual portalled picker closes later", async () => {
    io.realAssistant = true
    const user = userEvent.setup()
    mount()
    await openCustomize()
    await user.click(screen.getByTestId("character-select"))
    const tab = await screen.findByRole("tab", { name: "Characters" })
    tab.focus()
    await act(async () => {
      await new Storage().set(
        "playgroundComposerCasualAdvancedControlsOpen",
        false
      )
    })
    expect(customize()).toHaveAttribute("aria-expanded", "false")
    expect(tab).toBeVisible()
    await user.keyboard("{Escape}")
    await waitFor(() => expect(customize()).toHaveFocus())
    expect(customize()).toBeVisible()
  })

  it("returns focus to visible Customize when another window hides its currently focused controls", async () => {
    mount()
    const region = await openCustomize()
    within(region).getByRole("button", { name: "MCP tools" }).focus()
    await act(async () => {
      await new Storage().set(
        "playgroundComposerCasualAdvancedControlsOpen",
        false
      )
    })
    expect(customize()).toHaveAttribute("aria-expanded", "false")
    expect(customize()).toHaveFocus()
  })

  it("leaves another task focused when a watched preference closes Customize", async () => {
    render(
      <I18nextProvider i18n={translations}>
        <button>Other task</button>
        <ComposerToolbar {...props()} />
      </I18nextProvider>
    )
    const region = await openCustomize()
    within(region).getByRole("button", { name: "MCP tools" }).focus()
    screen.getByRole("button", { name: "Other task" }).focus()
    await act(async () => {
      await new Storage().set(
        "playgroundComposerCasualAdvancedControlsOpen",
        false
      )
    })
    expect(customize()).toHaveAttribute("aria-expanded", "false")
    expect(screen.getByRole("button", { name: "Other task" })).toHaveFocus()
  })

  it("keeps customization usable for this visit after a failed preference read, without overwriting the unread preference", async () => {
    const original = Storage.prototype.get
    vi.spyOn(Storage.prototype, "get").mockImplementation(function (
      key: string
    ) {
      return key === "playgroundComposerCasualAdvancedControlsOpen"
        ? Promise.reject(new Error("read unavailable"))
        : original.call(this, key)
    })
    const writes = vi.spyOn(Storage.prototype, "set")
    mount()
    const region = await openCustomize()
    expect(region).toBeVisible()
    expect(
      screen.getByText(
        "Display preference unavailable. Customize chat still works for this visit."
      )
    ).toBeVisible()
    fireEvent.click(customize())
    expect(customize()).toHaveAttribute("aria-expanded", "false")
    expect(writes).not.toHaveBeenCalled()
  })

  it("preserves explicit session customization if saving the display preference fails", async () => {
    mount()
    await waitFor(() => expect(customize()).toBeEnabled())
    vi.spyOn(Storage.prototype, "set").mockRejectedValueOnce(
      new Error("write unavailable")
    )
    fireEvent.click(customize())
    expect(
      await screen.findByRole("region", { name: "Customize chat" })
    ).toBeVisible()
    expect(
      await screen.findByText(
        "Display preference unavailable. Customize chat still works for this visit."
      )
    ).toBeVisible()
    fireEvent.click(customize())
    expect(customize()).toHaveAttribute("aria-expanded", "false")
  })

  it("keeps a newer watched preference when the initial read resolves late", async () => {
    const original = Storage.prototype.get
    let finish!: (value: boolean) => void
    let held = false
    vi.spyOn(Storage.prototype, "get").mockImplementation(function (
      key: string
    ) {
      if (key === "playgroundComposerCasualAdvancedControlsOpen" && !held) {
        held = true
        return new Promise((resolve) => {
          finish = resolve
        })
      }
      return original.call(this, key)
    })
    mount()
    await act(async () => {
      await new Storage().set(
        "playgroundComposerCasualAdvancedControlsOpen",
        false
      )
    })
    await waitFor(() => expect(customize()).toBeEnabled())
    await act(async () => {
      finish(true)
    })
    expect(customize()).toHaveAttribute("aria-expanded", "false")
  })

  it.each([
    { kind: "MCP", focus: "content" },
    { kind: "Modes", focus: "content" },
    { kind: "Tools", focus: "content" },
    { kind: "MCP", focus: "trigger" },
    { kind: "Modes", focus: "trigger" },
    { kind: "Tools", focus: "trigger" }
  ])(
    "closes the actual $kind popup from $focus before closing Customize",
    async ({ kind, focus }) => {
      const noop = () => undefined
      const t = (key: string, fallback?: string, options?: unknown) =>
        translations.t(key, { defaultValue: fallback, ...(options as object) })
      const Harness = () => {
        const [open, setOpen] = React.useState(false)
        const control =
          kind === "MCP" ? (
            <PlaygroundMcpControl
              hasMcp
              mcpHealthState="healthy"
              mcpToolsLoading={false}
              mcpToolsCount={2}
              toolChoice="auto"
              onToolChoiceChange={noop}
              toolRunStatusLabel="Ready"
              mcpAriaLabel="MCP tools"
              mcpSummaryLabel="2 tools"
              mcpChoiceLabel="Auto"
              mcpDisabledReason=""
              mcpPopoverOpen={open}
              onMcpPopoverChange={setOpen}
              onOpenMcpSettings={noop}
              t={t}
            />
          ) : kind === "Modes" ? (
            <PlaygroundModeLauncher
              open={open}
              onOpenChange={setOpen}
              compareModeActive={false}
              compareFeatureEnabled
              onToggleCompare={noop}
              selectedCharacterName={null}
              onOpenActorSettings={noop}
              contextToolsOpen={false}
              onToggleKnowledgePanel={noop}
              voiceChatEnabled={false}
              voiceChatAvailable={false}
              isSending={false}
              onVoiceChatToggle={noop}
              webSearch={false}
              hasWebSearch
              onToggleWebSearch={noop}
              onModeAnnouncement={noop}
              t={t}
            />
          ) : (
            <PlaygroundToolsPopover
              toolsPopoverOpen={open}
              onToolsPopoverChange={setOpen}
              isProMode={false}
              onOpenImageGenerate={noop}
              onOpenKnowledgePanel={noop}
              useOCR={false}
              onUseOCRChange={noop}
              hasWebSearch
              webSearch={false}
              onWebSearchChange={noop}
              simpleInternetSearch={false}
              onSimpleInternetSearchChange={noop}
              defaultInternetSearchOn={false}
              onDefaultInternetSearchOnChange={noop}
              onNavigateWebSearchSettings={noop}
              advancedToolsExpanded={false}
              onAdvancedToolsExpandedChange={noop}
              allowExternalImages={false}
              onAllowExternalImagesChange={noop}
              showMoodBadge={false}
              onShowMoodBadgeChange={noop}
              showMoodConfidence={false}
              onShowMoodConfidenceChange={noop}
              onOpenRawRequest={noop}
              voiceChatAvailable={false}
              voiceChatEnabled={false}
              voiceChatState="idle"
              voiceChatStatusLabel="Idle"
              onVoiceChatToggle={noop}
              isSending={false}
              voiceChatSettingsFields={null}
              imageProviderControl={null}
              historyLength={0}
              onClearContext={noop}
              t={t}
            />
          )
        return (
          <MemoryRouter>
            <ConfigProvider theme={{ token: { motion: false } }}>
              <I18nextProvider i18n={translations}>
                <ComposerToolbar
                  {...props({
                    nestedDisclosureOpen: open,
                    ...(kind === "MCP"
                      ? { mcpControl: control }
                      : kind === "Modes"
                        ? { modeLauncherButton: control }
                        : { toolsButton: control })
                  })}
                />
              </I18nextProvider>
            </ConfigProvider>
          </MemoryRouter>
        )
      }
      const user = userEvent.setup()
      render(<Harness />)
      await openCustomize()
      const trigger = screen.getByRole("button", {
        name:
          kind === "MCP"
            ? "MCP tools"
            : kind === "Modes"
              ? "Modes"
              : "More tools"
      })
      trigger.focus()
      await user.keyboard("{Enter}")
      const child =
        kind === "MCP"
          ? await screen.findByRole("radio", { name: "Required" })
          : await screen.findByRole("button", {
              name: kind === "Modes" ? /Compare responses/ : "Generate image"
            })
      if (focus === "content") child.focus()
      else expect(trigger).toHaveFocus()
      await user.keyboard("{Escape}")
      await waitFor(() =>
        expect(child.closest(".ant-popover")).toHaveClass("ant-popover-hidden")
      )
      expect(customize()).toHaveAttribute("aria-expanded", "true")
      expect(trigger).toHaveFocus()
      await user.keyboard("{Escape}")
      expect(customize()).toHaveAttribute("aria-expanded", "false")
      expect(customize()).toHaveFocus()
    }
  )

  it.each([
    { label: "desktop", isMobile: false, optionsExpanded: true },
    { label: "collapsed desktop", isMobile: false, optionsExpanded: false },
    { label: "mobile", isMobile: true, optionsExpanded: true },
    { label: "collapsed mobile", isMobile: true, optionsExpanded: false }
  ])(
    "keeps primary controls and discloses secondary actions on $label",
    async (layout) => {
      mount(props(layout))
      await waitFor(() => expect(customize()).toBeEnabled())
      for (const name of ["Choose model", "Attach", "Send", "Customize chat"])
        expect(screen.getByRole("button", { name })).toBeVisible()
      for (const name of [
        "MCP tools",
        "Choose character or persona",
        "Choose system prompt",
        "Tools",
        "Research",
        "Chat modes",
        "Compare models",
        "Interactive response",
        "Voice chat",
        "Start dictation"
      ])
        expect(screen.queryByRole("button", { name })).not.toBeInTheDocument()
      const region = await openCustomize()
      const research = within(region).getByRole("group", {
        name: "Research & context"
      })
      const character = within(region).getByRole("group", {
        name: "Character & your role"
      })
      const response = within(region).getByRole("group", {
        name: "Response settings"
      })
      expect(
        within(research).getByRole("button", { name: "MCP tools" })
      ).toBeVisible()
      expect(
        within(research).getByRole("button", { name: "Tools" })
      ).toBeVisible()
      expect(
        within(character).getByRole("button", {
          name: "Choose character or persona"
        })
      ).toBeVisible()
      expect(
        within(character).getByRole("button", { name: "Role-play setup" })
      ).toBeVisible()
      expect(
        within(response).getByRole("button", { name: "Compare models" })
      ).toBeVisible()
      expect(
        within(response).getByRole("button", { name: "Choose system prompt" })
      ).toBeVisible()
      expect(customize()).toHaveAttribute("aria-expanded", "true")
    }
  )

  it("uses the established explicit disclosure preference without mount or mode writes", async () => {
    const store = new Storage()
    await store.set("playgroundComposerCasualAdvancedControlsOpen", true)
    const writes = vi.spyOn(Storage.prototype, "set")
    const value = props()
    const rendered = mount(value)
    await screen.findByRole("region", { name: "Customize chat" })
    expect(writes).not.toHaveBeenCalled()
    rendered.rerenderToolbar({ ...value, isProMode: true })
    rendered.rerenderToolbar(value)
    expect(
      await screen.findByRole("region", { name: "Customize chat" })
    ).toBeVisible()
    expect(writes).not.toHaveBeenCalled()
    fireEvent.click(customize())
    await waitFor(() =>
      expect(customize()).toHaveAttribute("aria-expanded", "false")
    )
    rendered.unmount()
    mount(value)
    await waitFor(() => expect(customize()).toBeEnabled())
    expect(customize()).toHaveAttribute("aria-expanded", "false")
    expect(writes).toHaveBeenCalledExactlyOnceWith(
      "playgroundComposerCasualAdvancedControlsOpen",
      false
    )
  })

  it("waits for the saved disclosure read before accepting a conflicting toggle", async () => {
    const original = Storage.prototype.get
    let finish!: (value: boolean) => void
    vi.spyOn(Storage.prototype, "get").mockImplementation(function (
      key: string
    ) {
      if (key === "playgroundComposerCasualAdvancedControlsOpen")
        return new Promise((resolve) => {
          finish = resolve
        })
      return original.call(this, key)
    })
    const writes = vi.spyOn(Storage.prototype, "set")
    mount()
    expect(customize()).toBeDisabled()
    expect(customize()).toHaveAttribute("aria-busy", "true")
    fireEvent.click(customize())
    expect(writes).not.toHaveBeenCalled()
    await act(async () => {
      finish(true)
    })
    expect(
      await screen.findByRole("region", { name: "Customize chat" })
    ).toBeVisible()
    expect(customize()).toBeEnabled()
    expect(writes).not.toHaveBeenCalled()
  })

  it("active identity opens setup without clearing it while the customization controls stay closed", async () => {
    const clear = vi.fn(),
      setup = vi.fn()
    mount(
      props({
        rolePlayActions: { onOpenRolePlaySetup: setup },
        contextItems: [
          {
            id: "rolePlayIdentity",
            label: "Character",
            value: "Mira",
            tone: "active",
            onClick: clear
          },
          {
            id: "rolePlayCompatibility",
            label: "Character setup",
            value: "Excluded in this mode",
            tone: "warning"
          },
          {
            id: "rolePlayContext",
            label: "Context",
            value: "2 pinned",
            tone: "active"
          }
        ]
      })
    )
    const chip = screen.getByRole("button", { name: /Character: Mira/ })
    fireEvent.click(chip)
    expect(setup).toHaveBeenCalledTimes(1)
    expect(clear).not.toHaveBeenCalled()
    expect(screen.getByText("Excluded in this mode")).toBeVisible()
    expect(screen.getByText("2 pinned")).toBeVisible()
    expect(
      screen.queryByRole("region", { name: "Customize chat" })
    ).not.toBeInTheDocument()
  })

  it("keeps selected tool/formatting/interactive response effects visible without turning them off", async () => {
    await new Storage().set(
      PLAYGROUND_APPEND_FORMATTING_GUIDE_PROMPT_STORAGE_KEY,
      true
    )
    const value = props({
      toolChoice: "required",
      openUIRequestMode: true,
      webSearch: true,
      contextItems: [
        { id: "webSearch", label: "Web search", value: "On", tone: "active" },
        {
          id: "prompt",
          label: "Prompt",
          value: "Story instructions",
          tone: "active"
        },
        {
          id: "json",
          label: "JSON mode",
          value: "Object responses",
          tone: "active"
        }
      ]
    })
    mount(value)
    expect(
      await screen.findByText("MCP tools selected: Required")
    ).toBeVisible()
    expect(screen.getByText("Formatting guide on")).toBeVisible()
    expect(screen.getByText("Interactive response on")).toBeVisible()
    expect(screen.getByText("Story instructions")).toBeVisible()
    expect(screen.getByText("Object responses")).toBeVisible()
    expect(value.onToggleWebSearch).not.toHaveBeenCalled()
    expect(value.setSystemPrompt).not.toHaveBeenCalled()
  })

  it.each([
    { isListening: true },
    { speechUsesServer: true, isServerDictating: true }
  ])("keeps active dictation stop outside customization: %j", async (state) => {
    const stop = vi.fn()
    mount(props({ ...state, onDictationToggle: stop }))
    const button = screen.getByRole("button", { name: "Stop dictation" })
    expect(button).toBeVisible()
    fireEvent.click(button)
    expect(stop).toHaveBeenCalledOnce()
    expect(
      screen.queryByRole("region", { name: "Customize chat" })
    ).not.toBeInTheDocument()
  })

  it("keeps active voice stop and send/queue controls outside customization", async () => {
    mount(
      props({
        voiceChatEnabled: true,
        isSending: true,
        voiceChatButton: <button>Stop voice chat</button>,
        sendControl: <button>Stop generation</button>
      })
    )
    expect(
      screen.getByRole("button", { name: "Stop voice chat" })
    ).toBeVisible()
    expect(
      screen.getByRole("button", { name: "Stop generation" })
    ).toBeVisible()
  })

  it("external send placement does not duplicate the send/stop action", async () => {
    mount(
      props({
        sendControlPlacement: "external",
        optionsExpanded: false,
        isMobile: true
      })
    )
    expect(screen.getByRole("button", { name: "Choose model" })).toBeVisible()
    expect(screen.getByRole("button", { name: "Attach" })).toBeVisible()
    expect(
      screen.queryByRole("button", { name: "Send" })
    ).not.toBeInTheDocument()
    expect(customize()).toBeVisible()
  })

  it("retains Pro direct selectors and expert controls instead of a compulsory disclosure", async () => {
    mount(
      props({ isProMode: true, messages: [{ role: "user", content: "Hello" }] })
    )
    for (const name of [
      "Choose model",
      "Attach",
      "Send",
      "MCP tools",
      "Choose character or persona",
      "Choose system prompt",
      "Tools",
      "Compare models",
      "Interactive response",
      "Voice chat",
      "Start dictation"
    ])
      expect(screen.getByRole("button", { name })).toBeVisible()
    expect(
      screen.queryByRole("button", { name: "Customize chat" })
    ).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Advanced controls" }))
    expect(
      await screen.findByRole("button", { name: "Parameter presets" })
    ).toBeVisible()
    expect(screen.getByRole("button", { name: "Session cost" })).toBeVisible()
  })

  it("keyboard opens customization and Escape returns focus to its persistent visible trigger", async () => {
    const user = userEvent.setup()
    mount()
    await waitFor(() => expect(customize()).toBeEnabled())
    customize().focus()
    await user.keyboard("{Enter}")
    const region = await screen.findByRole("region", { name: "Customize chat" })
    within(region).getByRole("button", { name: "MCP tools" }).focus()
    await user.keyboard("{Escape}")
    await waitFor(() =>
      expect(customize()).toHaveAttribute("aria-expanded", "false")
    )
    expect(customize()).toHaveFocus()
  })

  it("does not dismiss the parent customization layer while an existing nested editor owns Escape", async () => {
    const value = props({ nestedDisclosureOpen: true })
    const rendered = mount(value)
    const region = await openCustomize()
    const input = within(region).getByRole("button", { name: "MCP tools" })
    input.focus()
    fireEvent.keyDown(input, { key: "Escape" })
    expect(customize()).toHaveAttribute("aria-expanded", "true")
    rendered.rerenderToolbar({ ...value, nestedDisclosureOpen: false })
    fireEvent.keyDown(input, { key: "Escape" })
    await waitFor(() =>
      expect(customize()).toHaveAttribute("aria-expanded", "false")
    )
    expect(customize()).toHaveFocus()
  })

  it("closing a disclosure restores focus before hiding the focused child", async () => {
    mount()
    const region = await openCustomize()
    within(region).getByRole("button", { name: "MCP tools" }).focus()
    fireEvent.click(customize())
    await waitFor(() =>
      expect(customize()).toHaveAttribute("aria-expanded", "false")
    )
    expect(customize()).toHaveFocus()
  })

  it("mode replacement cannot strand focus when the old focused control unmounts", async () => {
    const value = props({ isProMode: true })
    const rendered = mount(value)
    screen.getByRole("button", { name: "MCP tools" }).focus()
    rendered.rerenderToolbar({ ...value, isProMode: false })
    await waitFor(() => expect(customize()).toBeEnabled())
    expect(customize()).toHaveFocus()
  })

  it("mode replacement leaves an unrelated focused control alone", async () => {
    const value = props({ isProMode: true })
    const node = (next: Props) => (
      <I18nextProvider i18n={translations}>
        <button>Other task</button>
        <ComposerToolbar {...next} />
      </I18nextProvider>
    )
    const rendered = render(node(value))
    screen.getByRole("button", { name: "Other task" }).focus()
    rendered.rerender(node({ ...value, isProMode: false }))
    await waitFor(() => expect(customize()).toBeEnabled())
    expect(screen.getByRole("button", { name: "Other task" })).toHaveFocus()
  })

  it("presentation changes preserve the entered draft and selected settings", async () => {
    const value = props()
    const Harness = () => {
      const [message, setMessage] = React.useState("unsent character opening"),
        [pro, setPro] = React.useState(false)
      return (
        <I18nextProvider i18n={translations}>
          <input
            aria-label="Message"
            value={message}
            onChange={(event) => setMessage(event.target.value)}
          />
          <button onClick={() => setPro(!pro)}>Change UI mode</button>
          <ComposerToolbar {...value} isProMode={pro} />
        </I18nextProvider>
      )
    }
    render(<Harness />)
    await openCustomize()
    fireEvent.click(customize())
    fireEvent.click(screen.getByRole("button", { name: "Change UI mode" }))
    fireEvent.click(screen.getByRole("button", { name: "Change UI mode" }))
    expect(screen.getByRole("textbox", { name: "Message" })).toHaveValue(
      "unsent character opening"
    )
    expect(value.setSystemPrompt).not.toHaveBeenCalled()
    expect(value.onToggleTemporaryChat).not.toHaveBeenCalled()
  })
})
