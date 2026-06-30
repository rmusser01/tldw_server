import { fireEvent, render, screen } from "@testing-library/react"
import React from "react"
import { describe, expect, it, vi } from "vitest"

import { ComposerToolbar } from "../ComposerToolbar"

const assistantSelectMock = vi.hoisted(() => vi.fn())

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string) => fallback || key
  })
}))

vi.mock("antd", () => ({
  Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  Modal: ({
    open,
    children
  }: {
    open?: boolean
    children: React.ReactNode
  }) => (open ? <div data-testid="toolbar-modal">{children}</div> : null)
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: string, defaultValue: unknown) =>
    React.useState(defaultValue)
}))

vi.mock("@/components/Common/PromptSelect", () => ({
  PromptSelect: () => <div data-testid="prompt-select" />
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
  ConnectionStatus: () => <div data-testid="connection-status" />
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
  ParameterPresets: () => <div data-testid="parameter-presets" />,
  ParameterPresetsDropdown: () => <div data-testid="parameter-presets-dropdown" />,
  SystemPromptTemplatesButton: () => <button type="button">Templates</button>,
  SystemPromptTemplatesModal: () => null,
  SessionCostEstimation: () => <div data-testid="session-cost" />
}))

vi.mock("../ComposerToolbarOverflow", () => ({
  ComposerToolbarOverflow: () => <div data-testid="toolbar-overflow" />
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

describe("ComposerToolbar web search", () => {
  it("owns the dropdown assistant selector used by chat starter events", () => {
    render(<ComposerToolbar {...createProps()} />)

    expect(screen.getByTestId("character-select")).toHaveAttribute(
      "data-variant",
      "dropdown"
    )
    expect(assistantSelectMock).toHaveBeenCalledWith(
      expect.objectContaining({ variant: "dropdown" })
    )
  })

  it("hides the options panel when rendered collapsed for external send placement", () => {
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
    expect(panel.className).toBe("mt-2 flex flex-col gap-1")
    expect(screen.queryByText("Model selector")).toBeNull()
    expect(screen.queryByRole("button", { name: "Send" })).toBeNull()
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

    expect(
      screen.getByRole("button", { name: "Attach image" })
    ).toBeVisible()
    expect(screen.queryByText("Model selector")).toBeNull()
    expect(screen.queryByRole("button", { name: "Send" })).toBeNull()
  })

  it("uses casual focus-first layout by default", () => {
    render(<ComposerToolbar {...createProps()} />)

    expect(
      screen.getByRole("button", { name: "Advanced controls" })
    ).toBeInTheDocument()
    expect(screen.getByText("Send")).toBeInTheDocument()
    expect(screen.getByText("Model selector")).toBeInTheDocument()
    expect(screen.queryByText("Provider")).toBeNull()
    expect(screen.queryByText("Routing")).toBeNull()
    expect(
      screen.getByTestId("composer-casual-model-selector-chip")
    ).toBeInTheDocument()
    expect(screen.queryByTestId("composer-session-status-chip")).toBeNull()
    expect(
      screen.getByTestId("composer-casual-persistence-chip")
    ).toHaveTextContent("Saved")
    expect(
      screen
        .getByTestId("composer-casual-advanced-chip")
        .closest('[data-testid="composer-context-strip"]')
    ).not.toBeNull()
    expect(
      screen.queryByTestId("composer-casual-runtime-context-chip")
    ).toBeNull()
    expect(screen.queryByTestId("web-search-toggle")).toBeNull()
    expect(screen.getByText("MCP")).toBeInTheDocument()
    expect(screen.getByTestId("prompt-select")).toBeInTheDocument()
    expect(screen.getByTestId("character-select")).toBeInTheDocument()
  })

  it("labels casual composer control groups for scanning and keyboard focus", () => {
    render(
      <ComposerToolbar
        {...createProps({
          modeLauncherButton: <button type="button">Modes</button>,
          voiceChatButton: <button type="button">Start voice chat</button>
        })}
      />
    )

    const contextGroup = screen.getByRole("group", {
      name: "Mode and context controls"
    })
    const runGroup = screen.getByRole("group", {
      name: "Run input controls"
    })

    expect(contextGroup).toContainElement(
      screen.getByRole("button", { name: "Modes" })
    )
    expect(contextGroup).toContainElement(
      screen.getByRole("button", { name: "MCP" })
    )
    expect(runGroup).toContainElement(
      screen.getByRole("button", { name: "Start voice chat" })
    )
    expect(runGroup).toContainElement(
      screen.getByRole("button", { name: "Chat Settings" })
    )
    expect(runGroup).toContainElement(
      screen.getByRole("button", { name: "Send" })
    )
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
      screen.getByRole("group", { name: "Advanced composer controls" })
    ).toHaveAttribute("id", "composer-casual-advanced-controls-row")
  })

  it("exposes role-play setup directly in the desktop casual toolbar", () => {
    const onOpenRolePlaySetup = vi.fn()
    render(
      <ComposerToolbar
        {...createProps({
          rolePlayActions: {
            onOpenRolePlaySetup
          }
        })}
      />
    )

    const setupButton = screen.getByRole("button", {
      name: "Role-play setup"
    })
    fireEvent.click(setupButton)

    expect(onOpenRolePlaySetup).toHaveBeenCalledTimes(1)
    expect(
      setupButton.closest('[data-playground-toolbar-row="actions"]')
    ).not.toBeNull()
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

  it("keeps Modes first and MCP immediately to its right in the casual controls row", () => {
    render(
      <ComposerToolbar
        {...createProps({
          modeLauncherButton: <button type="button">Modes</button>
        })}
      />
    )

    const actionsRow = document.querySelector<HTMLElement>(
      '[data-playground-toolbar-row="actions"]'
    )
    expect(actionsRow).not.toBeNull()
    const buttons = actionsRow?.querySelectorAll("button")
    expect(buttons?.[0]).not.toBeNull()
    expect(buttons?.[0]).toHaveTextContent("Modes")
    expect(buttons?.[1]).not.toBeNull()
    expect(buttons?.[1]).toHaveTextContent("MCP")
  })

  it("places voice chat, attachment, and send controls in the casual middle actions row", () => {
    render(
      <ComposerToolbar
        {...createProps({
          voiceChatButton: <button type="button">Start voice chat</button>
        })}
      />
    )

    const voiceButton = screen.getByRole("button", {
      name: "Start voice chat"
    })
    const attachmentButton = screen.getByRole("button", { name: "Attach" })
    const sendButton = screen.getByRole("button", { name: "Send" })
    expect(
      voiceButton.closest('[data-playground-toolbar-row="actions"]')
    ).not.toBeNull()
    expect(
      attachmentButton.closest('[data-playground-toolbar-row="actions"]')
    ).not.toBeNull()
    expect(
      sendButton.closest('[data-playground-toolbar-row="actions"]')
    ).not.toBeNull()
    expect(
      voiceButton.closest('[data-playground-toolbar-row="primary"]')
    ).toBeNull()
    expect(
      attachmentButton.closest('[data-playground-toolbar-row="primary"]')
    ).toBeNull()
    expect(
      sendButton.closest('[data-playground-toolbar-row="primary"]')
    ).toBeNull()
  })

  it("keeps advanced controls below the casual actions row", () => {
    render(<ComposerToolbar {...createProps()} />)

    const actionsRow = document.querySelector<HTMLElement>(
      '[data-playground-toolbar-row="actions"]'
    )
    const contextStrip = screen.getByTestId("composer-context-strip")
    fireEvent.click(screen.getByTestId("composer-casual-advanced-chip"))
    const advancedRow = screen.getByTestId(
      "composer-casual-advanced-controls-row"
    )

    expect(actionsRow).not.toBeNull()
    expect(contextStrip.compareDocumentPosition(actionsRow as Node)).toBe(
      Node.DOCUMENT_POSITION_PRECEDING
    )
    expect(advancedRow.compareDocumentPosition(actionsRow as Node)).toBe(
      Node.DOCUMENT_POSITION_PRECEDING
    )
    expect(advancedRow.compareDocumentPosition(contextStrip)).toBe(
      Node.DOCUMENT_POSITION_PRECEDING
    )
  })

  it("places Advanced controls immediately to the right of Saved in the casual context strip", () => {
    render(<ComposerToolbar {...createProps()} />)

    const contextStrip = screen.getByTestId("composer-context-strip")
    const contextButtons = contextStrip.querySelectorAll("button")
    const savedButton = screen.getByTestId(
      "composer-casual-persistence-chip"
    ) as HTMLButtonElement
    const advancedButton = screen.getByTestId(
      "composer-casual-advanced-chip"
    ) as HTMLButtonElement

    const savedIndex = Array.from(contextButtons).indexOf(savedButton)
    const advancedIndex = Array.from(contextButtons).indexOf(advancedButton)

    expect(savedIndex).toBeGreaterThanOrEqual(0)
    expect(advancedIndex).toBe(savedIndex + 1)
  })

  it("renders a deep research launch control in the casual actions row when provided", () => {
    render(
      <ComposerToolbar
        {...createProps({
          researchLaunchButton: <button type="button">Deep Research</button>
        })}
      />
    )

    const researchButton = screen.getByRole("button", {
      name: "Deep Research"
    })
    expect(
      researchButton.closest('[data-playground-toolbar-row="actions"]')
    ).not.toBeNull()
  })

  it("does not render a standalone Generate control in the toolbar row", () => {
    render(<ComposerToolbar {...createProps()} />)

    expect(screen.queryByRole("button", { name: "Generate" })).toBeNull()
  })

  it("places current chat model settings control in the casual middle actions row", () => {
    render(<ComposerToolbar {...createProps()} />)

    const chatSettingsButton = screen.getByRole("button", {
      name: "Chat Settings"
    })
    expect(
      chatSettingsButton.closest('[data-playground-toolbar-row="actions"]')
    ).not.toBeNull()
    expect(
      chatSettingsButton.closest('[data-playground-toolbar-row="advanced"]')
    ).toBeNull()
  })

  it("wraps casual controls below desktop while keeping the dense desktop row", () => {
    render(<ComposerToolbar {...createProps()} />)

    const actionsRow = document.querySelector<HTMLElement>(
      '[data-playground-toolbar-row="actions"]'
    )
    expect(actionsRow).not.toBeNull()
    expect(actionsRow?.className).toContain("flex-wrap")
    expect(actionsRow?.className).toContain("lg:flex-nowrap")
    expect(actionsRow?.className).toContain("lg:overflow-x-auto")
  })

  it("keeps MCP in the casual actions row when advanced controls are expanded", () => {
    render(<ComposerToolbar {...createProps()} />)

    fireEvent.click(screen.getByTestId("composer-casual-advanced-chip"))

    const mcpButton = screen.getByRole("button", { name: "MCP" })
    expect(screen.getByTestId("prompt-select")).toBeInTheDocument()
    expect(screen.getByTestId("character-select")).toBeInTheDocument()
    expect(
      mcpButton.closest('[data-playground-toolbar-row="actions"]')
    ).not.toBeNull()
    expect(
      mcpButton.closest('[data-playground-toolbar-row="advanced"]')
    ).toBeNull()
    expect(
      screen
        .getByTestId("prompt-select")
        .closest('[data-playground-toolbar-row="actions"]')
    ).not.toBeNull()
    expect(
      screen
        .getByTestId("character-select")
        .closest('[data-playground-toolbar-row="actions"]')
    ).not.toBeNull()
    expect(
      screen.getAllByRole("button", { name: "Chat Settings" })
    ).toHaveLength(1)
    expect(
      screen.getByTestId("composer-formatting-guide-toggle")
    ).toBeInTheDocument()
    const advancedRow = screen.getByTestId(
      "composer-casual-advanced-controls-row"
    )
    expect(advancedRow.className).toContain("flex-nowrap")
    expect(advancedRow.className).toContain("overflow-x-auto")
    expect(advancedRow.className).not.toContain("flex-wrap")
  })

  it("toggles output formatting guide prompt appending from advanced controls", () => {
    render(<ComposerToolbar {...createProps()} />)

    fireEvent.click(screen.getByTestId("composer-casual-advanced-chip"))
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
      screen.getByRole("button", { name: "Saved" })
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

  it("uses a bottom persistence chip to toggle saved vs temporary", () => {
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
