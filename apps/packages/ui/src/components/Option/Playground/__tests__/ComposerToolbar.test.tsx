import { fireEvent, render, screen } from "@testing-library/react";
import React from "react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { ComposerToolbar } from "../ComposerToolbar";

const assistantSelectMock = vi.hoisted(() => vi.fn());
const promptSelectMock = vi.hoisted(() => vi.fn());

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string | { defaultValue?: string }) =>
      (typeof fallback === "string" ? fallback : fallback?.defaultValue) || key,
  }),
}));

vi.mock("antd", () => ({
  Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  Modal: ({ open, children }: { open?: boolean; children: React.ReactNode }) =>
    open ? <div data-testid="toolbar-modal">{children}</div> : null,
}));

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: string, defaultValue: unknown) =>
    React.useState(defaultValue),
}));

vi.mock("@/components/Common/PromptSelect", () => ({
  PromptSelect: (props: unknown) => {
    promptSelectMock(props);
    return <div data-testid="prompt-select" />;
  },
}));

vi.mock("@/components/Common/AssistantSelect", () => ({
  AssistantSelect: (props: { variant?: string }) => {
    assistantSelectMock(props);
    return (
      <div data-testid="character-select" data-variant={props.variant ?? ""} />
    );
  },
}));

vi.mock("@/components/Layouts/ConnectionStatus", () => ({
  ConnectionStatus: () => <div data-testid="connection-status" />,
}));

vi.mock("@/components/Common/Button", () => ({
  Button: ({
    children,
    onClick,
  }: {
    children: React.ReactNode;
    onClick?: () => void;
  }) => (
    <button type="button" onClick={onClick}>
      {children}
    </button>
  ),
}));

vi.mock("../playground-features", () => ({
  ParameterPresets: ({ compact }: { compact?: boolean }) => (
    <div data-testid="parameter-presets" data-compact={String(Boolean(compact))} />
  ),
  ParameterPresetsDropdown: () => (
    <div data-testid="parameter-presets-dropdown" />
  ),
  SystemPromptTemplatesButton: () => <button type="button">Templates</button>,
  SystemPromptTemplatesModal: () => null,
  SessionCostEstimation: () => <div data-testid="session-cost" />,
}));

vi.mock("../ComposerToolbarOverflow", () => ({
  ComposerToolbarOverflow: () => <div data-testid="toolbar-overflow" />,
}));

const createProps = (
  overrides: Partial<React.ComponentProps<typeof ComposerToolbar>> = {},
): React.ComponentProps<typeof ComposerToolbar> => ({
  isProMode: false,
  onToggleOptions: vi.fn(),
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
  onOpenKnowledgePanel: vi.fn(),
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
      onClick: vi.fn(),
    },
  ],
  ...overrides,
});

const ToolbarHarness = (
  props: Partial<React.ComponentProps<typeof ComposerToolbar>>,
) => {
  const [expanded, setExpanded] = React.useState(
    props.optionsExpanded ?? false,
  );
  return (
    <ComposerToolbar
      {...createProps(props)}
      optionsExpanded={expanded}
      onToggleOptions={() => setExpanded(!expanded)}
    />
  );
};

describe("calm composer", () => {
  it("forwards settled tracked model discovery to the existing assistant control", () => {
    const tracked = vi.fn();
    const props = createProps({
      isProMode: true,
      onTrackedAssistantSelection: tracked,
      trackedAssistantSelectionLoading: true,
    });
    const f = render(<ComposerToolbar {...props} />);
    expect(assistantSelectMock.mock.lastCall?.[0]).toMatchObject({
      onTrackedSelection: tracked,
      trackedSelectionLoading: true,
    });
    f.rerender(
      <ComposerToolbar {...props} trackedAssistantSelectionLoading={false} />,
    );
    expect(assistantSelectMock.mock.lastCall?.[0]).toMatchObject({
      onTrackedSelection: tracked,
      trackedSelectionLoading: false,
    });
  });

  it.each([
    { label: "desktop", isMobile: false, isProMode: false },
    { label: "pro", isMobile: false, isProMode: true },
    { label: "mobile", isMobile: true, isProMode: false },
  ])(
    "keeps essential actions available with tools closed on $label",
    (layout) => {
      render(<ToolbarHarness {...layout} />);
      for (const name of ["Model selector", "Attach", "Tools", "Send"]) {
        expect(screen.getByRole("button", { name, exact: true })).toBeVisible();
      }
      expect(
        screen.queryByRole("region", { name: "Composer tools" }),
      ).toBeNull();
      expect(
        screen.queryByRole("button", { name: "Search & Context" }),
      ).toBeNull();
      expect(
        screen.queryByRole("button", { name: "Advanced controls" }),
      ).toBeNull();
    },
  );

  it("reveals the tool groups with one labelled disclosure", () => {
    render(
      <ToolbarHarness
        modeLauncherButton={<button type="button">Modes</button>}
      />,
    );
    const toggle = screen.getByRole("button", { name: "Tools", exact: true });
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    fireEvent.click(toggle);
    expect(toggle).toHaveAttribute("aria-expanded", "true");
    expect(
      screen.getByRole("region", { name: "Composer tools" }),
    ).toHaveAttribute("id", toggle.getAttribute("aria-controls"));
    expect(screen.getByRole("button", { name: "Modes" })).toBeVisible();
    expect(
      screen.getByRole("button", { name: "Search & Context" }),
    ).toBeVisible();
    expect(
      screen.getByRole("button", { name: "Formatting guide" }),
    ).toBeVisible();
    expect(
      screen.queryByRole("button", { name: "Advanced controls" }),
    ).toBeNull();
    fireEvent.click(toggle);
    expect(screen.queryByRole("region", { name: "Composer tools" })).toBeNull();
    expect(
      screen.getByRole("button", { name: "Model selector" }),
    ).toBeVisible();
  });

  it("honors an already expanded preference", () => {
    render(<ToolbarHarness optionsExpanded />);
    expect(
      screen.getByRole("region", { name: "Composer tools" }),
    ).toBeVisible();
  });

  it("closes Tools with Escape and restores focus to its trigger", () => {
    render(<ToolbarHarness optionsExpanded />);
    fireEvent.keyDown(screen.getByRole("region", { name: "Composer tools" }), {
      key: "Escape",
    });
    expect(screen.queryByRole("region", { name: "Composer tools" })).toBeNull();
    expect(
      screen.getByRole("button", { name: "Tools", exact: true }),
    ).toHaveFocus();
  });

  it.each([false, true])(
    "keeps active context and warnings actionable while tools are closed (mobile %s)",
    (isMobile) => {
      const onWeb = vi.fn();
      const onWarning = vi.fn();
      render(
        <ToolbarHarness
          isMobile={isMobile}
          contextItems={[
            {
              id: "web",
              label: "Web",
              value: "On",
              tone: "active",
              onClick: onWeb,
            },
            {
              id: "truncationRisk",
              label: "Context",
              value: "Near limit",
              tone: "warning",
              onClick: onWarning,
            },
            {
              id: "messageCount",
              label: "Messages",
              value: "0",
              tone: "neutral",
            },
          ]}
        />,
      );
      fireEvent.click(screen.getByTitle("Web: On"));
      fireEvent.click(screen.getByTitle("Context: Near limit"));
      expect(onWeb).toHaveBeenCalledOnce();
      expect(onWarning).toHaveBeenCalledOnce();
      expect(screen.queryByTitle("Messages: 0")).toBeNull();
    },
  );

  it("leaves a warning visible even when its ordinary status is hidden", () => {
    render(
      <ToolbarHarness
        contextItems={[
          {
            id: "providerStatus",
            label: "Provider",
            value: "Unavailable",
            tone: "warning",
          },
        ]}
      />,
    );
    expect(screen.getByTitle("Provider: Unavailable")).toBeVisible();
  });

  it("keeps temporary-chat state visible and respects its privacy lock", () => {
    const onToggleTemporaryChat = vi.fn();
    render(
      <ToolbarHarness
        temporaryChat
        privateChatLocked
        onToggleTemporaryChat={onToggleTemporaryChat}
      />,
    );
    const temporary = screen.getByRole("button", {
      name: "Chat history: Temporary",
    });
    expect(temporary).toBeVisible();
    expect(temporary).toBeDisabled();
    fireEvent.click(temporary);
    expect(onToggleTemporaryChat).not.toHaveBeenCalled();
  });

  it("opens prompt assistance only from Tools", () => {
    render(
      <ToolbarHarness
        promptAssistAction={<button type="button">Improve prompt</button>}
      />,
    );
    expect(screen.queryByRole("button", { name: "Improve prompt" })).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Tools", exact: true }));
    expect(
      screen.getByRole("button", { name: "Improve prompt" }),
    ).toBeVisible();
  });

  it("keeps active dictation visible so it can be stopped with Tools closed", () => {
    const onDictationToggle = vi.fn();
    render(
      <ToolbarHarness
        hasDictation
        speechAvailable
        isListening
        onDictationToggle={onDictationToggle}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Stop dictation" }));
    expect(onDictationToggle).toHaveBeenCalledOnce();
  });

  it("toggles the formatting guide inside Tools", () => {
    render(<ToolbarHarness optionsExpanded />);
    const toggle = screen.getByRole("button", { name: "Formatting guide" });
    expect(toggle).toHaveAttribute("aria-pressed", "false");
    fireEvent.click(toggle);
    expect(toggle).toHaveAttribute("aria-pressed", "true");
  });

  it("uses the active conversation route for prompt assistance", () => {
    render(
      <ToolbarHarness
        optionsExpanded
        selectedModel="gpt-5-mini"
        currentProvider="custom-openai"
        promptAssistContextKey="local:history-42"
      />,
    );
    expect(promptSelectMock).toHaveBeenCalledWith(
      expect.objectContaining({
        selectedModel: "gpt-5-mini",
        currentProvider: "custom-openai",
        promptAssistContextKey: "local:history-42",
      }),
    );
  });

  it("reuses the model recovery event from prompt selection", () => {
    const listener = vi.fn();
    window.addEventListener("tldw:open-model-selector", listener);
    render(<ToolbarHarness optionsExpanded />);
    const props = promptSelectMock.mock.calls.at(-1)?.[0] as {
      onSelectModel?: () => void;
    };
    props.onSelectModel?.();
    expect(listener).toHaveBeenCalledOnce();
    window.removeEventListener("tldw:open-model-selector", listener);
  });

  it("keeps the existing web-search handler behind Tools", () => {
    const onToggleWebSearch = vi.fn();
    render(
      <ToolbarHarness optionsExpanded onToggleWebSearch={onToggleWebSearch} />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Web search" }));
    expect(onToggleWebSearch).toHaveBeenCalledOnce();
  });

  it("does not offer web search without the server capability", () => {
    render(<ToolbarHarness optionsExpanded hasWebSearch={false} />);
    expect(screen.queryByRole("button", { name: "Web search" })).toBeNull();
  });

  // CC-09: the Pro presets read as named styles, not four bare icons.
  it("names the Pro generation-style presets inside Tools", () => {
    render(<ToolbarHarness isProMode optionsExpanded />);
    expect(screen.getByTestId("parameter-presets")).toHaveAttribute(
      "data-compact",
      "false",
    );
  });

  // CC-11: '@' mentions work only in the extension with tab mentions on.
  it("names '@' mentions in the shortcut hint only where they work", () => {
    const { unmount } = render(<ToolbarHarness />);
    expect(screen.getByText("/ commands")).toBeVisible();
    expect(screen.queryByText(/@ mentions/)).toBeNull();
    unmount();

    render(<ToolbarHarness mentionsAvailable />);
    expect(screen.getByText("/ commands · @ mentions")).toBeVisible();
  });
});

describe("composer starter events", () => {
  it.each(["tldw:open-assistant-select", "tldw:open-prompt-select"])(
    "reveals a hidden selector for %s",
    (name) => {
      render(<ToolbarHarness />);
      fireEvent(window, new CustomEvent(name));
      expect(
        screen.getByRole("region", { name: "Composer tools" }),
      ).toBeVisible();
    },
  );
});

describe("composer keyboard disclosure", () => {
  it("closes Tools when Escape is pressed on its opening trigger", () => {
    render(<ToolbarHarness />);
    const toggle = screen.getByRole("button", { name: "Tools", exact: true });
    fireEvent.click(toggle);
    toggle.focus();
    fireEvent.keyDown(toggle, { key: "Escape" });
    expect(
      screen.getByRole("button", { name: "Tools", exact: true }),
    ).toHaveAttribute("aria-expanded", "false");
  });
});

describe("composer context review actions", () => {
  it.each([
    {
      id: "rolePlayIdentity",
      label: "Character",
      value: "Mira",
      opensSetup: true,
    },
    { id: "character", label: "Character", value: "Mira", opensSetup: true },
    {
      id: "rolePlayBehavior",
      label: "System prompt",
      value: "Story instructions",
      opensSetup: false,
    },
    {
      id: "rolePlayScene",
      label: "Scene draft",
      value: "A quiet library",
      opensSetup: false,
    },
  ])(
    "keeps neutral selected $id visible and actionable with Tools closed",
    (selection) => {
      const action = vi.fn();
      const setup = vi.fn();
      render(
        <ToolbarHarness
          rolePlayActions={{ onOpenRolePlaySetup: setup }}
          contextItems={[
            {
              id: selection.id,
              label: selection.label,
              value: selection.value,
              tone: "neutral",
              onClick: action,
            },
          ]}
        />,
      );
      const chip = screen.getByTitle(`${selection.label}: ${selection.value}`);
      expect(chip).toBeVisible();
      fireEvent.click(chip);
      expect(selection.opensSetup ? setup : action).toHaveBeenCalledOnce();
      fireEvent.click(screen.getByTestId("composer-options-toggle"));
      expect(screen.getAllByText(selection.value)).toHaveLength(1);
      if (selection.opensSetup) expect(action).not.toHaveBeenCalled();
    },
  );

  it("opens attached context instead of closing an already open search panel", () => {
    const onOpenKnowledgePanel = vi.fn();
    const onToggleKnowledgePanel = vi.fn();
    render(
      <ToolbarHarness
        contextToolsOpen
        uploadedFilesCount={2}
        onOpenKnowledgePanel={onOpenKnowledgePanel}
        onToggleKnowledgePanel={onToggleKnowledgePanel}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /files/ }));
    expect(onOpenKnowledgePanel).toHaveBeenCalledExactlyOnceWith("context");
    expect(onToggleKnowledgePanel).not.toHaveBeenCalled();
  });

  it("keeps session insights actionable inside Tools", () => {
    const onClick = vi.fn();
    render(
      <ToolbarHarness
        optionsExpanded
        contextItems={[
          {
            id: "sessionUsage",
            label: "Session",
            value: "120 tokens",
            tone: "neutral",
            onClick,
          },
        ]}
      />,
    );
    fireEvent.click(
      screen.getByRole("button", { name: "Session: 120 tokens" }),
    );
    expect(onClick).toHaveBeenCalledOnce();
  });
});

const AudioHarness = ({ mode }: { mode: "dictation" | "voice" }) => {
  const [active, setActive] = React.useState(false);
  return (
    <ToolbarHarness
      optionsExpanded
      {...(mode === "dictation"
        ? {
            hasDictation: true,
            speechAvailable: true,
            isListening: active,
            onDictationToggle: () => setActive(!active),
          }
        : {
            voiceChatEnabled: active,
            voiceChatButton: (
              <button type="button" onClick={() => setActive(!active)}>
                {active ? "Stop voice chat" : "Start voice chat"}
              </button>
            ),
          })}
    />
  );
};

describe("composer audio focus", () => {
  it.each(["dictation", "voice"] as const)(
    "keeps focus through %s start and stop",
    (mode) => {
      render(<AudioHarness mode={mode} />);
      const startName =
        mode === "dictation" ? "Start dictation" : "Start voice chat";
      const stopName =
        mode === "dictation" ? "Stop dictation" : "Stop voice chat";
      const start = screen.getByRole("button", { name: startName });
      start.focus();
      fireEvent.click(start);
      const stop = screen.getByRole("button", { name: stopName });
      expect(stop).toHaveFocus();
      fireEvent.click(stop);
      expect(screen.getByRole("button", { name: startName })).toHaveFocus();
    },
  );

  it.each(["dictation", "voice"] as const)(
    "returns focus to Tools when stopping %s with Tools closed",
    (mode) => {
      render(<AudioHarness mode={mode} />);
      fireEvent.click(
        screen.getByRole("button", {
          name: mode === "dictation" ? "Start dictation" : "Start voice chat",
        }),
      );
      fireEvent.click(screen.getByTestId("composer-options-toggle"));
      const stop = screen.getByRole("button", {
        name: mode === "dictation" ? "Stop dictation" : "Stop voice chat",
      });
      stop.focus();
      fireEvent.click(stop);
      expect(screen.getByTestId("composer-options-toggle")).toHaveFocus();
    },
  );
});

describe("history mode labels", () => {
  it.each(
    [
      { name: "Casual desktop", isMobile: false, isProMode: false },
      { name: "Pro desktop", isMobile: false, isProMode: true },
      { name: "mobile", isMobile: true, isProMode: false },
    ].flatMap((layout) =>
      [false, true].map((temporaryChat) => ({ ...layout, temporaryChat })),
    ),
  )(
    "$name history toggle agrees with Temporary=$temporaryChat and responds to keyboard",
    async (layout) => {
      const onToggleTemporaryChat = vi.fn();
      render(
        <ComposerToolbar
          {...createProps({ ...layout, onToggleTemporaryChat })}
        />,
      );
      const stateLabel = layout.temporaryChat ? "Temporary" : "History on";
      const toggle = screen.getByRole("button", {
        name: `Chat history: ${stateLabel}`,
      });
      expect(toggle).toHaveTextContent(stateLabel);
      expect(toggle).toHaveAttribute(
        "aria-pressed",
        String(!layout.temporaryChat),
      );
      toggle.focus();
      await userEvent.setup().keyboard("{Enter}");
      expect(onToggleTemporaryChat).toHaveBeenCalledWith(!layout.temporaryChat);
    },
  );
});
