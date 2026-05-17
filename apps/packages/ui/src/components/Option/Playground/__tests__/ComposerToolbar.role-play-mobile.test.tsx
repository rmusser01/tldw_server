import { fireEvent, render, screen } from "@testing-library/react"
import React from "react"
import { describe, expect, it, vi } from "vitest"

import { ComposerToolbar } from "../ComposerToolbar"
import { ComposerToolbarOverflow } from "../ComposerToolbarOverflow"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string) => fallback || key
  })
}))

vi.mock("antd", () => ({
  Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  Modal: ({
    open,
    title,
    children
  }: {
    open?: boolean
    title?: React.ReactNode
    children: React.ReactNode
  }) =>
    open ? (
      <section role="dialog" aria-label={String(title || "Dialog")}>
        {children}
      </section>
    ) : null,
  Popover: ({
    children,
    content,
    open,
    onOpenChange
  }: {
    children: React.ReactElement
    content: React.ReactNode
    open?: boolean
    onOpenChange?: (open: boolean) => void
  }) => (
    <div>
      {React.cloneElement(children, {
        onClick: () => onOpenChange?.(!open)
      } as any)}
      {open ? <div data-testid="toolbar-overflow-content">{content}</div> : null}
    </div>
  )
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: string, defaultValue: unknown) =>
    React.useState(defaultValue)
}))

vi.mock("@/components/Common/PromptSelect", () => ({
  PromptSelect: () => <div data-testid="prompt-select" />
}))

vi.mock("@/components/Common/AssistantSelect", () => ({
  AssistantSelect: () => <div data-testid="character-select" />
}))

vi.mock("@/components/Layouts/ConnectionStatus", () => ({
  ConnectionStatus: () => <div data-testid="connection-status" />
}))

vi.mock("@/components/Common/Button", () => ({
  Button: ({
    children,
    onClick,
    ariaLabel
  }: {
    children: React.ReactNode
    onClick?: () => void
    ariaLabel?: string
  }) => (
    <button type="button" onClick={onClick} aria-label={ariaLabel}>
      {children}
    </button>
  )
}))

vi.mock("../playground-features", () => ({
  ParameterPresets: () => <div data-testid="parameter-presets" />,
  ParameterPresetsDropdown: ({ onChange }: { onChange?: (key: string) => void }) => (
    <div data-testid="generation-style-panel">
      <button type="button" onClick={() => onChange?.("balanced")}>
        Balanced
      </button>
    </div>
  ),
  SystemPromptTemplatesButton: () => <button type="button">System prompts</button>,
  SystemPromptTemplatesModal: ({
    open,
    onSelect
  }: {
    open?: boolean
    onSelect: (template: { id: string; content: string }) => void
  }) =>
    open ? (
      <section role="dialog" aria-label="System prompts">
        <button
          type="button"
          onClick={() =>
            onSelect({
              id: "character-actor",
              content: "Stay in character."
            })
          }>
          Character Actor
        </button>
      </section>
    ) : null,
  SessionCostEstimation: () => <div data-testid="session-cost" />
}))

const createToolbarProps = (
  overrides: Partial<React.ComponentProps<typeof ComposerToolbar>> = {}
): React.ComponentProps<typeof ComposerToolbar> => ({
  isProMode: false,
  isMobile: true,
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
  contextItems: [],
  ...overrides
})

const createOverflowProps = (
  overrides: Partial<React.ComponentProps<typeof ComposerToolbarOverflow>> = {}
): React.ComponentProps<typeof ComposerToolbarOverflow> => ({
  isProMode: false,
  isConnectionReady: true,
  contextToolsOpen: false,
  onToggleKnowledgePanel: vi.fn(),
  webSearch: false,
  onToggleWebSearch: vi.fn(),
  hasWebSearch: true,
  onOpenModelSettings: vi.fn(),
  hasDictation: false,
  speechAvailable: false,
  speechUsesServer: false,
  isListening: false,
  isServerDictating: false,
  voiceChatEnabled: false,
  onDictationToggle: vi.fn(),
  temporaryChat: false,
  onFocusConnectionCard: vi.fn(),
  ...overrides
})

describe("ComposerToolbar mobile role-play parity", () => {
  it("exposes system prompts and generation style from the mobile overflow", () => {
    const onTemplateSelect = vi.fn()
    render(
      <ComposerToolbar
        {...createToolbarProps({
          onTemplateSelect
        })}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "More options" }))

    expect(screen.getByRole("button", { name: "System prompts" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Generation style" })).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "System prompts" }))
    fireEvent.click(screen.getByRole("button", { name: "Character Actor" }))

    expect(onTemplateSelect).toHaveBeenCalledWith(
      expect.objectContaining({
        id: "character-actor",
        content: "Stay in character."
      })
    )

    fireEvent.click(screen.getByRole("button", { name: "More options" }))
    fireEvent.click(screen.getByRole("button", { name: "Generation style" }))

    expect(screen.getByTestId("generation-style-panel")).toBeInTheDocument()
  })

  it("keeps active role-play chips reachable and actionable on mobile", () => {
    const onClearIdentity = vi.fn()
    const onClearBehavior = vi.fn()
    const onResetGeneration = vi.fn()
    render(
      <ComposerToolbar
        {...createToolbarProps({
          contextItems: [
            {
              id: "rolePlayIdentity",
              label: "Character",
              value: "Mira",
              tone: "active",
              onClick: onClearIdentity
            },
            {
              id: "rolePlayBehavior",
              label: "Behavior",
              value: "Character Actor",
              tone: "active",
              onClick: onClearBehavior
            },
            {
              id: "rolePlayGenerationStyle",
              label: "Generation style",
              value: "Creative",
              tone: "active",
              onClick: onResetGeneration
            },
            {
              id: "rolePlayContext",
              label: "Context",
              value: "2 pinned",
              tone: "active"
            }
          ]
        })}
      />
    )

    fireEvent.click(screen.getByTitle("Character: Mira"))
    fireEvent.click(screen.getByTitle("Behavior: Character Actor"))
    fireEvent.click(screen.getByTitle("Generation style: Creative"))

    expect(onClearIdentity).toHaveBeenCalledTimes(1)
    expect(onClearBehavior).toHaveBeenCalledTimes(1)
    expect(onResetGeneration).toHaveBeenCalledTimes(1)
    expect(screen.getByTitle("Context: 2 pinned").tagName).toBe("SPAN")
  })

  it("routes mobile role-play setup through a reusable overflow callback", () => {
    const onOpenRolePlaySetup = vi.fn()
    render(
      <ComposerToolbarOverflow
        {...createOverflowProps({
          rolePlayActions: {
            onOpenSystemPrompts: vi.fn(),
            onOpenGenerationStyle: vi.fn(),
            onOpenRolePlaySetup
          }
        } as any)}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "More options" }))
    fireEvent.click(screen.getByRole("button", { name: "Role-play setup" }))

    expect(onOpenRolePlaySetup).toHaveBeenCalledTimes(1)
  })
})
