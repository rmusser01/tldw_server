import { fireEvent, render, screen, within } from "@testing-library/react"
import React from "react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"

import { ComposerToolbar } from "../ComposerToolbar"

const assistantSelectMock = vi.hoisted(() => vi.fn())
const promptSelectMock = vi.hoisted(() => vi.fn())

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string | { defaultValue?: string }) =>
      (typeof fallback === "string" ? fallback : fallback?.defaultValue) || key
  })
}))

vi.mock("antd", () => ({
  Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  Modal: ({ open, children }: { open?: boolean; children: React.ReactNode }) =>
    open ? <div data-testid="toolbar-modal">{children}</div> : null
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: string, defaultValue: unknown) =>
    React.useState(defaultValue)
}))

vi.mock("@/components/Common/PromptSelect", () => ({
  PromptSelect: (props: unknown) => {
    promptSelectMock(props)
    return <div data-testid="prompt-select" />
  }
}))

vi.mock("@/components/Common/AssistantSelect", () => ({
  AssistantSelect: (props: { variant?: string }) => {
    assistantSelectMock(props)
    return (
      <div data-testid="character-select" data-variant={props.variant ?? ""} />
    )
  }
}))

vi.mock("@/components/Layouts/ConnectionStatus", () => ({
  ConnectionStatus: ({ showLabel = true }: { showLabel?: boolean }) => (
    <div data-testid="connection-status" data-show-label={String(showLabel)} />
  )
}))

vi.mock("@/components/Common/Button", () => ({
  Button: ({
    children,
    onClick
  }: {
    children: React.ReactNode
    onClick?: () => void
  }) => (
    <button type="button" onClick={onClick}>
      {children}
    </button>
  )
}))

vi.mock("../playground-features", () => ({
  ParameterPresets: ({
    compact,
    showLabel
  }: {
    compact?: boolean
    showLabel?: boolean
  }) => (
    <div
      data-testid="parameter-presets"
      data-compact={String(Boolean(compact))}
      data-show-label={String(Boolean(showLabel))}
    />
  ),
  ParameterPresetsDropdown: () => (
    <div data-testid="parameter-presets-dropdown" />
  ),
  SystemPromptTemplatesButton: ({ size }: { size?: string }) => (
    <button type="button" data-size={size ?? "default"}>
      Templates
    </button>
  ),
  SystemPromptTemplatesModal: () => null,
  SessionCostEstimation: () => <div data-testid="session-cost" />
}))

vi.mock("../ComposerToolbarOverflow", () => ({
  ComposerToolbarOverflow: () => <div data-testid="toolbar-overflow" />
}))

vi.mock("../hooks/useCasualComposerDisclosure", () => ({
  useCasualComposerDisclosure: () => {
    const [open, setExplicitOpen] = React.useState(false)
    return { open, setExplicitOpen, loading: false, unavailable: false }
  }
}))

const createProps = (
  overrides: Partial<React.ComponentProps<typeof ComposerToolbar>> = {}
): React.ComponentProps<typeof ComposerToolbar> => ({
  isProMode: false,
  isMobile: false,
  isConnectionReady: true,
  isSending: false,
  modelSelectButton: <button type="button">Model selector</button>,
  mcpControl: <button type="button">MCP</button>,
  sendControl: <button type="button">Send</button>,
  attachmentButton: <button type="button">Attach</button>,
  toolsButton: <button type="button">Tools</button>,
  voiceChatButton: null,
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
  persistenceTooltip: "Persist",
  contextToolsOpen: false,
  onToggleKnowledgePanel: vi.fn(),
  webSearch: false,
  onToggleWebSearch: vi.fn(),
  hasWebSearch: true,
  onOpenModelSettings: vi.fn(),
  modelSummaryLabel: "Model",
  promptSummaryLabel: "Prompt",
  researchLaunchButton: null,
  hasDictation: false,
  speechAvailable: false,
  speechUsesServer: false,
  isListening: false,
  isServerDictating: false,
  voiceChatEnabled: false,
  speechTooltip: "Dictation unavailable",
  onDictationToggle: vi.fn(),
  onTemplateSelect: vi.fn(),
  selectedModel: null,
  currentProvider: "openai",
  resolvedProviderKey: "openai",
  messages: [],
  selectedDocumentsCount: 0,
  uploadedFilesCount: 0,
  serverChatId: null,
  showServerPersistenceHint: false,
  onDismissServerPersistenceHint: vi.fn(),
  onFocusConnectionCard: vi.fn(),
  contextItems: [
    {
      id: "model",
      label: "Model",
      value: "deepseek-chat",
      tone: "active",
      onClick: vi.fn()
    }
  ],
  ...overrides
})

const openCustomization = () => {
  fireEvent.click(screen.getByRole("button", { name: "Customize chat" }));
  const region = screen.queryByRole("region", { name: "Customize chat" });
  const disclosure = region && within(region).getByText("Advanced customization");
  if (disclosure && !disclosure.closest("details")?.open) fireEvent.click(disclosure);
};

describe("ComposerToolbar web search", () => {
  it.each([
    {
      label: "legacy casual",
      isProMode: false,
      isMobile: false,
      optionsExpanded: true
    },
    {
      label: "desktop pro",
      isProMode: true,
      isMobile: false,
      optionsExpanded: true
    },
    {
      label: "mobile",
      isProMode: false,
      isMobile: true,
      optionsExpanded: true
    },
    {
      label: "collapsed mobile",
      isProMode: false,
      isMobile: true,
      optionsExpanded: false
    }
  ])("does not render the external composer action for $label", (layout) => {
    render(
      <ComposerToolbar
        {...createProps({
          isProMode: layout.isProMode,
          isMobile: layout.isMobile,
          optionsExpanded: layout.optionsExpanded
        })}
      />
    )

    expect(
      screen.queryByRole("button", { name: "Improve prompt" })
    ).not.toBeInTheDocument()
  })

  it("forwards settled tracked model discovery to the existing assistant control", () => {
    const tracked = vi.fn()
    const props = createProps({ isProMode: true, onTrackedAssistantSelection: tracked, trackedAssistantSelectionLoading: true })
    const f = render(<ComposerToolbar {...props} />)
    expect(assistantSelectMock.mock.lastCall?.[0]).toMatchObject({ onTrackedSelection: tracked, trackedSelectionLoading: true })
    f.rerender(<ComposerToolbar {...props} trackedAssistantSelectionLoading={false} />)
    expect(assistantSelectMock.mock.lastCall?.[0]).toMatchObject({ onTrackedSelection: tracked, trackedSelectionLoading: false })
  })

  it("passes the active chat route to system prompt assist", () => {
    render(
      <ComposerToolbar
        {...createProps({
          selectedModel: "gpt-5-mini",
          currentProvider: "custom-openai",
          serverChatId: null,
          promptAssistContextKey: "local:history-42"
        })}
      />
    )
    openCustomization()

    expect(promptSelectMock).toHaveBeenCalledWith(
      expect.objectContaining({
        selectedModel: "gpt-5-mini",
        currentProvider: "custom-openai",
        promptAssistContextKey: "local:history-42"
      })
    )
  })

  it("hands prompt model recovery to the existing Playground selector", () => {
    const openListener = vi.fn()
    window.addEventListener("tldw:open-model-selector", openListener)
    render(<ComposerToolbar {...createProps()} />)
    openCustomization()

    const promptProps = promptSelectMock.mock.calls.at(-1)?.[0] as {
      onSelectModel?: () => void
    }
    promptProps.onSelectModel?.()

    expect(openListener).toHaveBeenCalledTimes(1)
    window.removeEventListener("tldw:open-model-selector", openListener)
  })

  it("owns the dropdown assistant selector used by chat starter events", () => {
    render(<ComposerToolbar {...createProps()} />)
    openCustomization()

    expect(screen.getByTestId("character-select")).toHaveAttribute(
      "data-variant",
      "dropdown"
    )
    expect(assistantSelectMock).toHaveBeenCalledWith(
      expect.objectContaining({ variant: "dropdown" })
    )
  })

  it("keeps the desktop status chip row when controls are collapsed for external send placement", () => {
    render(
      <ComposerToolbar
        {...({
          ...createProps(),
          optionsExpanded: false,
          sendControlPlacement: "external"
        } as any)}
      />
    )

    const panel = screen.getByTestId("composer-options-panel")
    expect(panel).toHaveClass("min-w-0", "flex-col")
    expect(screen.getByText("Model selector")).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Send" })).toBeNull()
  })

  it("opens casual advanced controls from the collapsed desktop context strip", () => {
    render(
      <ComposerToolbar
        {...createProps({
          optionsExpanded: false,
          sendControlPlacement: "external"
        })}
      />
    )

    fireEvent.click(screen.getByTestId("composer-casual-advanced-chip"))

    expect(
      screen.getByTestId("composer-casual-advanced-controls-row")
    ).toBeInTheDocument()
  })

  it("keeps mobile image attachment discoverable when options are collapsed", () => {
    render(
      <ComposerToolbar
        {...createProps({
          isMobile: true,
          optionsExpanded: false,
          sendControlPlacement: "external",
          attachmentButton: <button type="button">Attach image</button>
        })}
      />
    )

    expect(screen.getByRole("button", { name: "Attach image" })).toBeVisible()
    expect(screen.getByText("Model selector")).toBeVisible()
    expect(screen.queryByRole("button", { name: "Send" })).toBeNull()
  })

  it("uses a concise Casual primary row with secondary controls disclosed", () => {
    render(<ComposerToolbar {...createProps()} />)
    expect(screen.getByRole("button", { name: "Customize chat" })).toBeVisible()
    expect(screen.getByRole("button", { name: "Send" })).toBeVisible()
    expect(screen.getByText("Model selector")).toBeVisible()
    expect(screen.getByTestId("composer-casual-persistence-chip")).toHaveTextContent("History on")
    expect(screen.queryByTestId("composer-session-status-chip")).toBeNull()
    expect(screen.queryByTestId("composer-casual-runtime-context-chip")).toBeNull()
    expect(screen.queryByRole("button", { name: "MCP" })).toBeNull()
    expect(screen.getByTestId("prompt-select")).not.toBeVisible()
    expect(screen.getByTestId("character-select")).not.toBeVisible()
    expect(screen.queryByText("Provider")).toBeNull()
    expect(screen.queryByText("Routing")).toBeNull()
  })

  it("labels customization by user tasks and keeps Send outside it", () => {
    render(<ComposerToolbar {...createProps({ modeLauncherButton: <button>Modes</button>, voiceChatButton: <button>Start voice chat</button> })} />)
    openCustomization()
    const research = screen.getByRole("group", { name: "Research & context" })
    const response = screen.getByRole("group", { name: "Response settings" })
    expect(screen.getByRole("group", { name: "Advanced composer controls" })).toContainElement(screen.getByRole("button", { name: "Modes" }))
    expect(research).toContainElement(screen.getByRole("button", { name: "MCP" }))
    expect(screen.getByRole("group", { name: "Advanced composer controls" })).toContainElement(screen.getByRole("button", { name: "Start voice chat" }))
    expect(response).toContainElement(screen.getByRole("button", { name: "Chat Settings" }))
    expect(response).not.toContainElement(screen.getByRole("button", { name: "Send" }))
  })

  it("only exposes casual advanced aria-controls when the controlled group is mounted", () => {
    render(<ComposerToolbar {...createProps()} />)

    const toggle = screen.getByTestId("composer-casual-advanced-chip")
    expect(toggle).not.toHaveAttribute("aria-controls")

    fireEvent.click(toggle)

    expect(toggle).toHaveAttribute(
      "aria-controls",
      "composer-casual-advanced-controls-row"
    )
    expect(
      screen.getByRole("region", { name: "Customize chat" })
    ).toHaveAttribute("id", "composer-casual-advanced-controls-row")
  })

  it("reuses role-play setup inside the Casual character group", () => {
    const setup = vi.fn()
    render(<ComposerToolbar {...createProps({ rolePlayActions: { onOpenRolePlaySetup: setup } })} />)
    openCustomization()
    fireEvent.click(within(screen.getByRole("group", { name: "Character & your role" })).getByRole("button", { name: "Role-play setup" }))
    expect(setup).toHaveBeenCalledOnce()
  })

  it("places token usage in the casual bottom context chip row", () => {
    render(
      <ComposerToolbar
        {...createProps({
          modelUsageBadge: <span data-testid="model-usage">~0 tokens</span>
        })}
      />
    )

    const usageBadge = screen.getByTestId("model-usage")
    expect(
      usageBadge.closest('[data-testid="composer-casual-token-chip"]')
    ).not.toBeNull()
    expect(
      usageBadge.closest('[data-playground-toolbar-row="actions"]')
    ).toBeNull()
  })

  it("keeps Modes and MCP in the research customization group", () => {
    render(<ComposerToolbar {...createProps({ modeLauncherButton: <button>Modes</button> })} />)
    openCustomization()
    const group = screen.getByRole("group", { name: "Research & context" })
    expect(screen.getByRole("group", { name: "Advanced composer controls" })).toContainElement(screen.getByRole("button", { name: "Modes" }))
    expect(group).toContainElement(screen.getByRole("button", { name: "MCP" }))
  })

  it("keeps idle voice inside Customize and Attach and Send in the primary row", () => {
    render(<ComposerToolbar {...createProps({ voiceChatButton: <button>Start voice chat</button> })} />)
    openCustomization()
    expect(screen.getByRole("group", { name: "Advanced composer controls" })).toContainElement(screen.getByRole("button", { name: "Start voice chat" }))
    const primary = screen.getByRole("group", { name: "Composer quick actions" })
    expect(primary).toContainElement(screen.getByRole("button", { name: "Attach" }))
    expect(primary).toContainElement(screen.getByRole("button", { name: "Send" }))
  })

  it("keeps Customize after primary controls and active context", () => {
    render(<ComposerToolbar {...createProps()} />)
    openCustomization()
    const primary = screen.getByRole("group", { name: "Composer quick actions" })
    const context = screen.getByTestId("composer-context-strip")
    const customize = screen.getByRole("region", { name: "Customize chat" })
    expect(context.compareDocumentPosition(primary)).toBe(Node.DOCUMENT_POSITION_PRECEDING)
    expect(customize.compareDocumentPosition(context)).toBe(Node.DOCUMENT_POSITION_PRECEDING)
  })

  it("keeps history status outside Customize and customization beside the model", () => {
    render(<ComposerToolbar {...createProps()} />)
    const primary = screen.getByRole("group", { name: "Composer quick actions" })
    expect(primary).toContainElement(screen.getByRole("button", { name: "Customize chat" }))
    expect(primary).toContainElement(screen.getByText("Model selector"))
    expect(screen.getByTestId("composer-context-strip")).toContainElement(screen.getByTestId("composer-casual-persistence-chip"))
  })

  it("keeps deep research reachable in the research group", () => {
    render(<ComposerToolbar {...createProps({ researchLaunchButton: <button>Deep Research</button> })} />)
    openCustomization()
    expect(screen.getByRole("group", { name: "Advanced composer controls" })).toContainElement(screen.getByRole("button", { name: "Deep Research" }))
  })

  it("does not render a standalone Generate control in the toolbar row", () => {
    render(<ComposerToolbar {...createProps()} />)

    expect(screen.queryByRole("button", { name: "Generate" })).toBeNull()
  })

  it("keeps current model settings reachable in Response settings", () => {
    const settings = vi.fn()
    render(<ComposerToolbar {...createProps({ onOpenModelSettings: settings })} />)
    openCustomization()
    fireEvent.click(within(screen.getByRole("group", { name: "Response settings" })).getByRole("button", { name: "Chat Settings" }))
    expect(settings).toHaveBeenCalledOnce()
  })

  it("wraps customization control groups without a horizontal scrolling row", () => {
    render(<ComposerToolbar {...createProps()} />)
    openCustomization()
    for (const name of ["Research & context", "Character & your role", "Response settings"]) {
      const group = screen.getByRole("group", { name })
      expect(group.querySelector("div")).toHaveClass("flex-wrap", "min-w-0")
      expect(group.className).not.toContain("overflow-x-auto")
    }
  })

  it("exposes every existing secondary editor once in its customization group", () => {
    render(<ComposerToolbar {...createProps()} />)
    openCustomization()
    expect(screen.getByRole("group", { name: "Research & context" })).toContainElement(screen.getByRole("button", { name: "MCP" }))
    expect(screen.getByRole("group", { name: "Response settings" })).toContainElement(screen.getByTestId("prompt-select"))
    expect(screen.getByRole("group", { name: "Character & your role" })).toContainElement(screen.getByTestId("character-select"))
    expect(screen.getAllByRole("button", { name: "Chat Settings" })).toHaveLength(1)
    expect(screen.getByTestId("composer-formatting-guide-toggle")).toBeVisible()
    expect(screen.getByRole("button", { name: "Generation style" })).toBeVisible()
    expect(screen.getByRole("button", { name: "System prompts" })).toBeVisible()
  })

  it("toggles output formatting guide prompt appending from advanced controls", () => {
    render(<ComposerToolbar {...createProps()} />)

    fireEvent.click(screen.getByTestId("composer-casual-advanced-chip"))
    fireEvent.click(screen.getByText("Advanced customization"))
    const toggle = screen.getByTestId("composer-formatting-guide-toggle")
    expect(toggle).toHaveAttribute("aria-pressed", "false")

    fireEvent.click(toggle)

    expect(toggle).toHaveAttribute("aria-pressed", "true")
  })

  it("uses split context/generation panels in pro mode", () => {
    render(<ComposerToolbar {...createProps({ isProMode: true })} />)

    expect(screen.getByTestId("composer-pro-context-panel")).toBeInTheDocument()
    expect(
      screen.getByTestId("composer-pro-generation-panel")
    ).toBeInTheDocument()
    expect(screen.getByText("MCP")).toBeInTheDocument()
    expect(screen.getByTestId("prompt-select")).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Advanced controls" })
    ).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Advanced controls" }))
    expect(
      screen.getByTestId("composer-formatting-guide-toggle")
    ).toBeInTheDocument()
  })

  // CC-09 (#3101): Advanced controls showed four bare preset icons and a
  // connection dot that read as an on/off switch.
  it("labels the Pro generation-style presets instead of showing bare icons", () => {
    render(<ComposerToolbar {...createProps({ isProMode: true })} />)
    fireEvent.click(screen.getByRole("button", { name: "Advanced controls" }))

    const presets = screen.getByTestId("parameter-presets")
    expect(presets).toHaveAttribute("data-compact", "false")
    expect(presets).toHaveAttribute("data-show-label", "true")
  })

  it("sizes System prompts like the other Pro advanced controls", () => {
    render(<ComposerToolbar {...createProps({ isProMode: true })} />)
    fireEvent.click(screen.getByRole("button", { name: "Advanced controls" }))

    expect(screen.getByRole("button", { name: "Templates" })).toHaveAttribute(
      "data-size",
      "compact"
    )
  })

  it("never shows the server status as an unlabeled dot in the Pro or mobile toolbar", () => {
    const { unmount } = render(
      <ComposerToolbar {...createProps({ isProMode: true })} />
    )
    fireEvent.click(screen.getByRole("button", { name: "Advanced controls" }))
    for (const status of screen.queryAllByTestId("connection-status")) {
      expect(status).toHaveAttribute("data-show-label", "true")
    }
    unmount()

    render(
      <ComposerToolbar {...createProps({ isMobile: true, isProMode: true })} />
    )
    for (const status of screen.queryAllByTestId("connection-status")) {
      expect(status).toHaveAttribute("data-show-label", "true")
    }
  })

  it("only exposes pro advanced aria-controls when the controlled group is mounted", () => {
    render(<ComposerToolbar {...createProps({ isProMode: true })} />)

    const toggle = screen.getByTestId("composer-advanced-toggle")
    expect(toggle).not.toHaveAttribute("aria-controls")

    fireEvent.click(toggle)

    expect(toggle).toHaveAttribute(
      "aria-controls",
      "composer-pro-advanced-controls-row"
    )
    expect(
      screen.getByRole("group", { name: "Advanced composer controls" })
    ).toHaveAttribute("id", "composer-pro-advanced-controls-row")
  })

  it("labels pro cockpit panels by task area without hiding existing controls", () => {
    render(
      <ComposerToolbar
        {...createProps({
          isProMode: true,
          modeLauncherButton: <button type="button">Modes</button>,
          compareControl: <button type="button">Compare</button>,
          researchLaunchButton: <button type="button">Deep Research</button>
        })}
      />
    )

    const contextPanel = screen.getByRole("group", {
      name: "Context setup"
    })
    const generationPanel = screen.getByRole("group", {
      name: "Model, tools, and run"
    })

    expect(
      screen.getByRole("heading", { name: "Context setup" })
    ).toBeInTheDocument()
    expect(
      screen.getByRole("heading", { name: "Model, tools, and run" })
    ).toBeInTheDocument()
    expect(contextPanel).toContainElement(
      screen.getByRole("button", { name: "Modes" })
    )
    expect(contextPanel).toContainElement(
      screen.getByRole("button", { name: "Compare" })
    )
    expect(generationPanel).toContainElement(
      screen.getByRole("button", { name: "MCP" })
    )
    expect(generationPanel).toContainElement(
      screen.getByRole("button", { name: "Deep Research" })
    )
    expect(generationPanel).toContainElement(
      screen.getByRole("button", { name: "Send" })
    )
  })

  it("labels mobile composer groups without moving controls into a bottom bar", () => {
    render(
      <ComposerToolbar
        {...createProps({
          isMobile: true,
          isProMode: true,
          modeLauncherButton: <button type="button">Modes</button>
        })}
      />
    )

    const primaryGroup = screen.getByRole("group", {
      name: "Mobile mode and model controls"
    })
    const contextGroup = screen.getByRole("group", {
      name: "Mobile context controls"
    })
    const runGroup = screen.getByRole("group", {
      name: "Mobile run input controls"
    })

    expect(primaryGroup).toContainElement(
      screen.getByRole("button", { name: "Modes" })
    )
    expect(primaryGroup).toContainElement(screen.getByText("Model selector"))
    expect(contextGroup).toContainElement(
      screen.getByRole("button", { name: "Chat history: History on" })
    )
    expect(contextGroup).toContainElement(
      screen.getByRole("button", { name: "Search & Context" })
    )
    expect(runGroup).toContainElement(screen.getByTestId("toolbar-overflow"))
    expect(runGroup).toContainElement(
      screen.getByRole("button", { name: "Attach" })
    )
    expect(screen.queryByTestId("composer-bottom-bar")).toBeNull()
  })

  it("invokes toggle callback when web search button is clicked", () => {
    const onToggleWebSearch = vi.fn()
    render(
      <ComposerToolbar
        {...createProps({
          isProMode: true,
          onToggleWebSearch,
          hasWebSearch: true,
          webSearch: false
        })}
      />
    )

    fireEvent.click(screen.getByTestId("web-search-toggle"))

    expect(onToggleWebSearch).toHaveBeenCalledTimes(1)
  })

  it("does not render web search button when capability is unavailable", () => {
    render(
      <ComposerToolbar
        {...createProps({ isProMode: true, hasWebSearch: false })}
      />
    )

    expect(screen.queryByTestId("web-search-toggle")).toBeNull()
  })

  it("renders mode launcher, compare control, and context strip when provided", () => {
    const onClick = vi.fn()
    render(
      <ComposerToolbar
        {...createProps({
          modelSelectButton: null,
          modeLauncherButton: <button type="button">Modes</button>,
          compareControl: <button type="button">Compare</button>,
          contextItems: [
            {
              id: "model",
              label: "Model",
              value: "gpt-4.1",
              tone: "active",
              onClick
            }
          ]
        })}
      />
    )

    openCustomization()
    expect(screen.getByText("Modes")).toBeInTheDocument()
    expect(screen.getByText("Compare")).toBeInTheDocument()
    const contextStrip = screen.getByTestId("composer-context-strip")
    const modelChipButton = screen.getByTitle("Model: gpt-4.1")
    fireEvent.click(modelChipButton)
    expect(onClick).toHaveBeenCalledTimes(1)
    expect(contextStrip).toBeInTheDocument()
  })

  it("renders session status chip when provided and keeps other context actions", () => {
    const onSessionStatusClick = vi.fn()
    const onRiskClick = vi.fn()

    render(
      <ComposerToolbar
        {...createProps({
          contextItems: [
            {
              id: "sessionStatus",
              label: "Session status",
              value: "Degraded",
              tone: "warning",
              onClick: onSessionStatusClick
            },
            {
              id: "truncationRisk",
              label: "Truncation",
              value: "Medium risk",
              tone: "warning",
              onClick: onRiskClick
            }
          ]
        })}
      />
    )

    fireEvent.click(screen.getByTestId("composer-session-status-chip"))
    fireEvent.click(screen.getByRole("button", { name: /Truncation/i }))

    expect(onSessionStatusClick).toHaveBeenCalledTimes(1)
    expect(onRiskClick).toHaveBeenCalledTimes(1)
    expect(
      screen.getByTestId("composer-session-status-chip")
    ).toHaveTextContent("Session status")
    expect(
      screen.getByTestId("composer-session-status-chip")
    ).toHaveTextContent("Degraded")
    expect(screen.getByText("Medium risk")).toBeInTheDocument()
  })

  it("applies warning styling for degraded session status", () => {
    render(
      <ComposerToolbar
        {...createProps({
          contextItems: [
            {
              id: "sessionStatus",
              label: "Session status",
              value: "Degraded",
              tone: "warning"
            }
          ]
        })}
      />
    )

    const sessionChip = screen.getByTestId("composer-session-status-chip")
    expect(sessionChip).toHaveTextContent("Session status")
    expect(sessionChip).toHaveTextContent("Degraded")
    expect(sessionChip.className).toContain("border-warn/40")
  })

  it("renders session status chip in pro and mobile context strips", () => {
    const degradedItem = [
      {
        id: "sessionStatus",
        label: "Session status",
        value: "Offline",
        tone: "warning"
      } as const
    ]

    const { rerender } = render(
      <ComposerToolbar
        {...createProps({
          isProMode: true,
          contextItems: degradedItem
        })}
      />
    )

    expect(
      screen.getByTestId("composer-session-status-chip")
    ).toHaveTextContent("Offline")

    rerender(
      <ComposerToolbar
        {...createProps({
          isMobile: true,
          contextItems: degradedItem
        })}
      />
    )

    expect(
      screen.getByTestId("composer-session-status-chip")
    ).toHaveTextContent("Offline")
  })

  it("uses a bottom persistence chip to toggle history mode vs temporary", () => {
    const onToggleTemporaryChat = vi.fn()
    render(
      <ComposerToolbar
        {...createProps({
          onToggleTemporaryChat
        })}
      />
    )

    fireEvent.click(screen.getByTestId("composer-casual-persistence-chip"))

    expect(onToggleTemporaryChat).toHaveBeenCalledWith(true)
  })
})


describe("history mode labels", () => {
  it.each(
    [
      { name: "Casual desktop", isMobile: false, isProMode: false },
      { name: "Pro desktop", isMobile: false, isProMode: true },
      { name: "mobile", isMobile: true, isProMode: false }
    ].flatMap((layout) =>
      [false, true].map((temporaryChat) => ({ ...layout, temporaryChat }))
    )
  )(
    "$name history toggle agrees with Temporary=$temporaryChat and responds to keyboard",
    async (layout) => {
      const onToggleTemporaryChat = vi.fn()
      render(
        <ComposerToolbar
          {...createProps({ ...layout, onToggleTemporaryChat })}
        />
      )
      const stateLabel = layout.temporaryChat ? "Temporary" : "History on"
      const toggle = screen.getByRole("button", {
        name: `Chat history: ${stateLabel}`
      })
      expect(toggle).toHaveTextContent(stateLabel)
      expect(toggle).toHaveAttribute(
        "aria-pressed",
        String(!layout.temporaryChat)
      )
      toggle.focus()
      await userEvent.setup().keyboard("{Enter}")
      expect(onToggleTemporaryChat).toHaveBeenCalledWith(!layout.temporaryChat)
    }
  )
})
