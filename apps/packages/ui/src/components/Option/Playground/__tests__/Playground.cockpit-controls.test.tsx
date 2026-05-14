// @vitest-environment jsdom
import React from "react";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { Playground } from "../Playground";
import {
  OPEN_ASSISTANT_SELECT_EVENT,
  OPEN_MCP_SETTINGS_EVENT,
  SET_TEMPORARY_CHAT_EVENT,
  TOGGLE_WEB_SEARCH_EVENT,
} from "../playground-cockpit-actions";

const messageOptionState = vi.hoisted(() => ({
  value: {
    messages: [
      { id: "message-1", role: "user", content: "Hello" },
      { id: "message-2", role: "assistant", content: "Hi" },
    ],
    history: [],
    historyId: "history-1" as string | null,
    serverChatId: "chat-1" as string | null,
    isLoading: false,
    setHistoryId: vi.fn(),
    setHistory: vi.fn(),
    setMessages: vi.fn(),
    selectedSystemPrompt: "" as string | null,
    setSelectedSystemPrompt: vi.fn(),
    selectedQuickPrompt: null as string | null,
    setSelectedQuickPrompt: vi.fn(),
    selectedModel: "openai:gpt-4.1-mini" as string | null,
    setSelectedModel: vi.fn(),
    setServerChatId: vi.fn(),
    contextFiles: [{ id: "file-1", name: "brief.pdf" }],
    setContextFiles: vi.fn(),
    createChatBranch: vi.fn(),
    streaming: true,
    selectedCharacter: { id: "character-1", name: "Mira Vale" },
    setSelectedCharacter: vi.fn(),
    selectedAssistant: {
      kind: "character",
      id: "character-1",
      name: "Mira Vale",
    } as { kind: "character" | "persona"; id: string; name: string } | null,
    serverChatPersonaMemoryMode: null as "read_only" | "read_write" | null,
    setSelectedAssistant: vi.fn(),
    compareMode: false,
    compareFeatureEnabled: false,
    temporaryChat: true,
    webSearch: true,
    toolChoice: "auto" as const,
    setToolChoice: vi.fn(),
    selectedKnowledge: [{ id: "knowledge-1", title: "Research notes" }],
    ragMediaIds: [101, 202],
    setSelectedKnowledge: vi.fn(),
    setRagMediaIds: vi.fn(),
    stopStreamingRequest: vi.fn(),
    regenerateLastMessage: vi.fn(),
  },
}));

const sessionPersistenceState = vi.hoisted(() => ({
  value: {
    restoreSession: vi.fn(async () => false),
    sessionScopeReady: true,
    hasPersistedSession: false,
    persistedHistoryId: null as string | null,
    persistedServerChatId: null as string | null,
  },
}));

const storageState = vi.hoisted(() => ({
  values: new Map<string, unknown>(),
}));

const modelSettingsState = vi.hoisted(() => ({
  value: {
    systemPrompt: "",
    setSystemPrompt: vi.fn(),
    temperature: 0.7 as number | undefined,
    topP: 0.9 as number | undefined,
    topK: undefined as number | undefined,
    numCtx: 8192 as number | undefined,
    numPredict: undefined as number | undefined,
    reasoningEffort: undefined as string | undefined,
    apiProvider: "openai" as string | undefined,
    activeSettingsScope: "openai:gpt-4.1-mini" as string | undefined,
    scopedSettingsByModelKey: {
      "openai:gpt-4.1-mini": { numCtx: 8192 },
    } as Record<string, Record<string, unknown>>,
  },
}));

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, defaultValue?: string) => defaultValue || key,
  }),
}));

vi.mock("@/components/Option/Playground/PlaygroundForm", () => ({
  PlaygroundForm: () => <div data-testid="playground-form" />,
}));

vi.mock("@/components/Option/Playground/PlaygroundChat", () => ({
  PlaygroundChat: () => <div data-testid="playground-chat" />,
}));

vi.mock("@/components/Sidepanel/Chat/ArtifactsPanel", () => ({
  ArtifactsPanel: () => null,
}));

vi.mock("@/hooks/useMessageOption", () => ({
  useMessageOption: () => messageOptionState.value,
}));

vi.mock("@/hooks/usePlaygroundSessionPersistence", () => ({
  usePlaygroundSessionPersistence: () => sessionPersistenceState.value,
}));

vi.mock("@/hooks/playground-session-restore", () => ({
  shouldRestorePersistedPlaygroundSession: () => false,
}));

vi.mock("@/services/app", () => ({
  webUIResumeLastChat: vi.fn(async () => false),
}));

vi.mock("@/db/dexie/helpers", () => ({
  formatToChatHistory: vi.fn(),
  formatToMessage: vi.fn(),
  getHistoryByServerChatId: vi.fn(async () => null),
  getPromptById: vi.fn(async () => null),
  getRecentChatFromWebUI: vi.fn(async () => null),
}));

vi.mock("@/store/model", () => ({
  useStoreChatModelSettings: () => modelSettingsState.value,
}));

vi.mock("@/hooks/useSmartScroll", () => ({
  useSmartScroll: () => ({
    containerRef: { current: null },
    isAutoScrollToBottom: true,
    autoScrollToBottom: vi.fn(),
  }),
}));

vi.mock("@/services/settings/ui-settings", () => ({
  CHAT_BACKGROUND_IMAGE_SETTING: "chatBackgroundImage",
}));

vi.mock("../Knowledge/utils/unsupported-types", () => ({
  otherUnsupportedTypes: [],
}));

vi.mock("@/store/option", () => ({
  useStoreMessageOption: (
    selector?: (state: {
      compareParentByHistory: Record<string, never>;
    }) => unknown,
  ) =>
    typeof selector === "function"
      ? selector({ compareParentByHistory: {} })
      : { compareParentByHistory: {} },
}));

vi.mock("@/store/artifacts", () => ({
  useArtifactsStore: (
    selector: (state: {
      isOpen: boolean;
      active: null;
      isPinned: boolean;
      history: never[];
      unreadCount: number;
      setOpen: ReturnType<typeof vi.fn>;
      closeArtifact: ReturnType<typeof vi.fn>;
      markRead: ReturnType<typeof vi.fn>;
    }) => unknown,
  ) =>
    selector({
      isOpen: false,
      active: null,
      isPinned: false,
      history: [],
      unreadCount: 0,
      setOpen: vi.fn(),
      closeArtifact: vi.fn(),
      markRead: vi.fn(),
    }),
}));

vi.mock("@/hooks/useSetting", () => ({
  useSetting: () => [""],
}));

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (key: string, defaultValue: unknown) => {
    const [value, setValue] = React.useState(
      storageState.values.has(key)
        ? storageState.values.get(key)
        : defaultValue,
    );
    const setStoredValue = (nextValue: unknown) => {
      const resolvedValue =
        typeof nextValue === "function"
          ? (nextValue as (previous: unknown) => unknown)(value)
          : nextValue;
      storageState.values.set(key, resolvedValue);
      setValue(resolvedValue);
    };
    return [value, setStoredValue];
  },
}));

vi.mock("@/hooks/useMediaQuery", () => ({
  useMobile: () => false,
}));

vi.mock("@/hooks/useLoadLocalConversation", () => ({
  useLoadLocalConversation: () => vi.fn(async () => {}),
}));

vi.mock("../playground-shortcuts", () => ({
  resolvePlaygroundShortcutAction: () => null,
}));

vi.mock("@/hooks/useCharacterGreeting", () => ({
  useCharacterGreeting: () => undefined,
}));

vi.mock("react-router-dom", async () => {
  const actual =
    await vi.importActual<typeof import("react-router-dom")>(
      "react-router-dom",
    );
  return {
    ...actual,
    useNavigate: () => vi.fn(),
  };
});

describe("Playground cockpit controls", () => {
  beforeEach(() => {
    storageState.values.clear();
    modelSettingsState.value.systemPrompt = "";
    modelSettingsState.value.setSystemPrompt = vi.fn();
    modelSettingsState.value.temperature = 0.7;
    modelSettingsState.value.topP = 0.9;
    modelSettingsState.value.topK = undefined;
    modelSettingsState.value.numCtx = 8192;
    modelSettingsState.value.numPredict = undefined;
    modelSettingsState.value.reasoningEffort = undefined;
    modelSettingsState.value.apiProvider = "openai";
    modelSettingsState.value.activeSettingsScope = "openai:gpt-4.1-mini";
    modelSettingsState.value.scopedSettingsByModelKey = {
      "openai:gpt-4.1-mini": { numCtx: 8192 },
    };
    messageOptionState.value.messages = [
      { id: "message-1", role: "user", content: "Hello" },
      { id: "message-2", role: "assistant", content: "Hi" },
    ];
    messageOptionState.value.historyId = "history-1";
    messageOptionState.value.serverChatId = "chat-1";
    messageOptionState.value.streaming = true;
    messageOptionState.value.selectedSystemPrompt = "";
    messageOptionState.value.setSelectedSystemPrompt = vi.fn();
    messageOptionState.value.selectedQuickPrompt = null;
    messageOptionState.value.setSelectedQuickPrompt = vi.fn();
    messageOptionState.value.selectedModel = "openai:gpt-4.1-mini";
    messageOptionState.value.selectedCharacter = {
      id: "character-1",
      name: "Mira Vale",
    };
    messageOptionState.value.selectedAssistant = {
      kind: "character",
      id: "character-1",
      name: "Mira Vale",
    };
    messageOptionState.value.serverChatPersonaMemoryMode = null;
    messageOptionState.value.temporaryChat = true;
    messageOptionState.value.webSearch = true;
    messageOptionState.value.toolChoice = "auto";
    messageOptionState.value.setToolChoice = vi.fn();
    messageOptionState.value.contextFiles = [{ id: "file-1", name: "brief.pdf" }];
    messageOptionState.value.setContextFiles = vi.fn();
    messageOptionState.value.selectedKnowledge = [
      { id: "knowledge-1", title: "Research notes" },
    ];
    messageOptionState.value.setSelectedKnowledge = vi.fn();
    messageOptionState.value.ragMediaIds = [101, 202];
    messageOptionState.value.setRagMediaIds = vi.fn();
    messageOptionState.value.stopStreamingRequest = vi.fn();
    messageOptionState.value.regenerateLastMessage = vi.fn();
  });

  it("surfaces existing context and runtime state in cockpit rails", async () => {
    messageOptionState.value.selectedQuickPrompt = "Draft a concise summary";
    const openKnowledgePanel = vi.fn();
    const openModelSettings = vi.fn();
    const openActorSettings = vi.fn();
    const openAssistantSelect = vi.fn();
    const openMcpSettings = vi.fn();
    const toggleWebSearch = vi.fn();
    const setTemporaryChat = vi.fn();
    window.addEventListener("tldw:open-knowledge-panel", openKnowledgePanel);
    window.addEventListener("tldw:open-model-settings", openModelSettings);
    window.addEventListener("tldw:open-actor-settings", openActorSettings);
    window.addEventListener(OPEN_ASSISTANT_SELECT_EVENT, openAssistantSelect);
    window.addEventListener(OPEN_MCP_SETTINGS_EVENT, openMcpSettings);
    window.addEventListener(TOGGLE_WEB_SEARCH_EVENT, toggleWebSearch);
    window.addEventListener(SET_TEMPORARY_CHAT_EVENT, setTemporaryChat);

    try {
      render(<Playground />);

      const leftRail = await screen.findByTestId("playground-cockpit-left-rail");
      const contextRail = within(leftRail).getByTestId("playground-context-rail");
      expect(within(contextRail).getByText("Context active")).toBeInTheDocument();
      expect(within(contextRail).getByText("1 file")).toBeInTheDocument();
      expect(
        within(contextRail).getByText("1 knowledge item"),
      ).toBeInTheDocument();
      expect(within(contextRail).getByText("2 media scopes")).toBeInTheDocument();
      expect(within(contextRail).getByText("Temporary chat")).toBeInTheDocument();
      expect(within(contextRail).getByText("History linked")).toBeInTheDocument();
      expect(within(contextRail).getByText("Prompts")).toBeInTheDocument();
      expect(within(contextRail).getByText("Quick prompt")).toBeInTheDocument();
      expect(
        within(contextRail).getByText("Draft a concise summary"),
      ).toBeInTheDocument();
      fireEvent.click(
        within(contextRail).getByRole("button", { name: "Clear prompt" }),
      );
      expect(messageOptionState.value.setSelectedQuickPrompt).toHaveBeenCalledWith(
        null,
      );
      expect(messageOptionState.value.setSelectedSystemPrompt).toHaveBeenCalledWith(
        "",
      );
      expect(
        within(contextRail).getByRole("button", { name: "Select a prompt" }),
      ).toHaveFocus();
      expect(messageOptionState.value.setContextFiles).not.toHaveBeenCalled();
      expect(within(contextRail).queryByText("1 file(s)")).toBeNull();
      fireEvent.click(
        within(contextRail).getByRole("button", { name: "Clear files" }),
      );
      fireEvent.click(
        within(contextRail).getByRole("button", { name: "Clear knowledge" }),
      );
      fireEvent.click(
        within(contextRail).getByRole("button", {
          name: "Clear media scopes",
        }),
      );
      expect(messageOptionState.value.setContextFiles).toHaveBeenCalledWith([]);
      expect(messageOptionState.value.setSelectedKnowledge).toHaveBeenCalledWith(
        null,
      );
      expect(messageOptionState.value.setRagMediaIds).toHaveBeenCalledWith(null);
      const webSearchControl = within(contextRail).getByRole("button", {
        name: "Web search",
      });
      expect(webSearchControl).toHaveAttribute("aria-pressed", "true");
      fireEvent.click(webSearchControl);
      expect(toggleWebSearch).toHaveBeenCalledTimes(1);

      fireEvent.click(
        within(contextRail).getByRole("button", {
          name: /open search & context/i,
        }),
      );
      expect(openKnowledgePanel).toHaveBeenCalledTimes(1);
      fireEvent.click(
        within(contextRail).getByRole("button", {
          name: /save conversation/i,
        }),
      );
      expect(setTemporaryChat).toHaveBeenCalledTimes(1);
      expect(
        (setTemporaryChat.mock.calls[0]?.[0] as CustomEvent<{ next: boolean }>)
          .detail,
      ).toEqual({ next: false });

      const rightRail = screen.getByTestId("playground-cockpit-right-rail");
      const runtimeInspector = within(rightRail).getByTestId(
        "playground-runtime-inspector",
      );
      expect(within(runtimeInspector).getByText("Streaming")).toBeInTheDocument();
      expect(within(runtimeInspector).getByText("Provider")).toBeInTheDocument();
      expect(within(runtimeInspector).getByText("openai")).toBeInTheDocument();
      expect(within(runtimeInspector).getByText("Model")).toBeInTheDocument();
      expect(within(runtimeInspector).getByText("gpt-4.1-mini")).toBeInTheDocument();
      expect(
        within(runtimeInspector).getByText("Route openai:gpt-4.1-mini"),
      ).toBeInTheDocument();
      expect(within(runtimeInspector).getByText("Temperature")).toBeInTheDocument();
      expect(within(runtimeInspector).getByText("0.7")).toBeInTheDocument();
      expect(
        within(runtimeInspector).getAllByText("Inherited").length,
      ).toBeGreaterThan(0);
      expect(within(runtimeInspector).getByText("Context")).toBeInTheDocument();
      expect(within(runtimeInspector).getByText("8192")).toBeInTheDocument();
      expect(within(runtimeInspector).getByText("Override")).toBeInTheDocument();
      fireEvent.click(
        within(runtimeInspector).getByRole("button", {
          name: "Stop generation",
        }),
      );
      expect(messageOptionState.value.stopStreamingRequest).toHaveBeenCalledTimes(
        1,
      );
      expect(
        within(runtimeInspector).queryByRole("button", {
          name: "Regenerate last response",
        }),
      ).toBeNull();
      expect(within(runtimeInspector).getByText("2 messages")).toBeInTheDocument();
      expect(within(runtimeInspector).getByText("Mira Vale")).toBeInTheDocument();

      fireEvent.click(
        within(runtimeInspector).getByRole("button", {
          name: /open model & chat settings/i,
        }),
      );
      fireEvent.click(
        within(runtimeInspector).getByRole("button", {
          name: /select character or persona/i,
        }),
      );
      fireEvent.click(
        within(runtimeInspector).getByRole("button", {
          name: "MCP tool choice Required",
        }),
      );
      fireEvent.click(
        within(runtimeInspector).getByRole("button", {
          name: "Configure MCP tools",
        }),
      );

      expect(openModelSettings).toHaveBeenCalledTimes(1);
      expect(
        (openModelSettings.mock.calls[0]?.[0] as CustomEvent<{
          returnFocusSelector: string;
        }>).detail,
      ).toEqual({
        returnFocusSelector: "[data-cockpit-model-settings-trigger]",
      });
      expect(openAssistantSelect).toHaveBeenCalledTimes(1);
      expect(openActorSettings).not.toHaveBeenCalled();
      expect(messageOptionState.value.setToolChoice).toHaveBeenCalledWith(
        "required",
      );
      expect(openMcpSettings).toHaveBeenCalledTimes(1);
    } finally {
      window.removeEventListener("tldw:open-knowledge-panel", openKnowledgePanel);
      window.removeEventListener("tldw:open-model-settings", openModelSettings);
      window.removeEventListener("tldw:open-actor-settings", openActorSettings);
      window.removeEventListener(OPEN_ASSISTANT_SELECT_EVENT, openAssistantSelect);
      window.removeEventListener(OPEN_MCP_SETTINGS_EVENT, openMcpSettings);
      window.removeEventListener(TOGGLE_WEB_SEARCH_EVENT, toggleWebSearch);
      window.removeEventListener(SET_TEMPORARY_CHAT_EVENT, setTemporaryChat);
    }
  });

  it("passes persona memory mode through to the cockpit assistant summary", async () => {
    messageOptionState.value.streaming = false;
    messageOptionState.value.selectedAssistant = {
      kind: "persona",
      id: "persona-1",
      name: "Research Persona",
    };
    messageOptionState.value.selectedCharacter = {
      id: "legacy-character",
      name: "Legacy Character",
    };
    messageOptionState.value.serverChatPersonaMemoryMode = "read_write";

    render(<Playground />);

    const runtimeInspector = within(
      await screen.findByTestId("playground-cockpit-right-rail"),
    ).getByTestId("playground-runtime-inspector");
    expect(
      within(runtimeInspector).getByText("Research Persona"),
    ).toBeInTheDocument();
    expect(
      within(runtimeInspector).getByText("Persona selected - memory read/write"),
    ).toBeInTheDocument();
    expect(
      within(runtimeInspector).queryByRole("button", {
        name: "Open Scene Director",
      }),
    ).toBeNull();
    expect(
      within(runtimeInspector).getByText(
        "Scene Director is available for character-backed chats.",
      ),
    ).toBeInTheDocument();
  });

  it("opens the assistant selector on the character tab for character mode", async () => {
    const openAssistantSelect = vi.fn();
    window.addEventListener(OPEN_ASSISTANT_SELECT_EVENT, openAssistantSelect);

    try {
      render(<Playground />);

      const runtimeInspector = within(
        await screen.findByTestId("playground-cockpit-right-rail"),
      ).getByTestId("playground-runtime-inspector");

      fireEvent.click(
        within(runtimeInspector).getByRole("button", {
          name: /select character or persona/i,
        }),
      );

      expect(openAssistantSelect).toHaveBeenCalledTimes(1);
      expect(
        (openAssistantSelect.mock.calls[0]?.[0] as CustomEvent).detail,
      ).toEqual(
        expect.objectContaining({
          tab: "character",
          source: "playground-cockpit",
          returnFocusSelector: "[data-cockpit-assistant-select-trigger]",
        }),
      );
    } finally {
      window.removeEventListener(OPEN_ASSISTANT_SELECT_EVENT, openAssistantSelect);
    }
  });

  it("opens the assistant selector on the persona tab for persona mode", async () => {
    messageOptionState.value.streaming = false;
    messageOptionState.value.selectedAssistant = {
      kind: "persona",
      id: "persona-1",
      name: "Research Persona",
    };
    messageOptionState.value.selectedCharacter = {
      id: "legacy-character",
      name: "Legacy Character",
    };
    const openAssistantSelect = vi.fn();
    window.addEventListener(OPEN_ASSISTANT_SELECT_EVENT, openAssistantSelect);

    try {
      render(<Playground />);

      const runtimeInspector = within(
        await screen.findByTestId("playground-cockpit-right-rail"),
      ).getByTestId("playground-runtime-inspector");

      fireEvent.click(
        within(runtimeInspector).getByRole("button", {
          name: /select character or persona/i,
        }),
      );

      expect(openAssistantSelect).toHaveBeenCalledTimes(1);
      expect(
        (openAssistantSelect.mock.calls[0]?.[0] as CustomEvent).detail,
      ).toEqual(
        expect.objectContaining({
          tab: "persona",
          source: "playground-cockpit",
          returnFocusSelector: "[data-cockpit-assistant-select-trigger]",
        }),
      );
    } finally {
      window.removeEventListener(OPEN_ASSISTANT_SELECT_EVENT, openAssistantSelect);
    }
  });

  it("opens the assistant selector on characters when no assistant is selected", async () => {
    messageOptionState.value.streaming = false;
    messageOptionState.value.selectedAssistant = null;
    messageOptionState.value.selectedCharacter = null;
    const openAssistantSelect = vi.fn();
    window.addEventListener(OPEN_ASSISTANT_SELECT_EVENT, openAssistantSelect);

    try {
      render(<Playground />);

      const runtimeInspector = within(
        await screen.findByTestId("playground-cockpit-right-rail"),
      ).getByTestId("playground-runtime-inspector");

      fireEvent.click(
        within(runtimeInspector).getByRole("button", {
          name: /select character or persona/i,
        }),
      );

      expect(
        (openAssistantSelect.mock.calls[0]?.[0] as CustomEvent).detail,
      ).toEqual(
        expect.objectContaining({
          tab: "character",
          returnFocusSelector: "[data-cockpit-assistant-select-trigger]",
        }),
      );
    } finally {
      window.removeEventListener(OPEN_ASSISTANT_SELECT_EVENT, openAssistantSelect);
    }
  });

  it("clears canonical assistant state and the legacy character mirror from the runtime rail", async () => {
    messageOptionState.value.streaming = false;
    messageOptionState.value.selectedAssistant = {
      kind: "persona",
      id: "persona-1",
      name: "Research Persona",
    };
    messageOptionState.value.selectedCharacter = {
      id: "legacy-character",
      name: "Legacy Character",
    };

    const { rerender } = render(<Playground />);

    const runtimeInspector = within(
      await screen.findByTestId("playground-cockpit-right-rail"),
    ).getByTestId("playground-runtime-inspector");
    expect(within(runtimeInspector).getByText("Research Persona")).toBeInTheDocument();
    expect(within(runtimeInspector).queryByText("Legacy Character")).toBeNull();

    const clearButton = within(runtimeInspector).getByRole("button", {
      name: "Clear assistant",
    });
    clearButton.focus();
    fireEvent.click(clearButton);

    expect(messageOptionState.value.setSelectedAssistant).toHaveBeenCalledWith(null);
    expect(messageOptionState.value.setSelectedCharacter).toHaveBeenCalledWith(null);
    expect(document.activeElement).toBe(clearButton);

    messageOptionState.value.selectedAssistant = null;
    messageOptionState.value.selectedCharacter = null;
    rerender(<Playground />);
    const updatedRuntimeInspector = within(
      screen.getByTestId("playground-cockpit-right-rail"),
    ).getByTestId("playground-runtime-inspector");
    expect(
      within(updatedRuntimeInspector).getAllByText("No assistant selected")
        .length,
    ).toBeGreaterThan(0);
    expect(within(updatedRuntimeInspector).queryByText("Legacy Character")).toBeNull();
    await waitFor(() => {
      expect(document.activeElement).toBe(
        within(updatedRuntimeInspector).getByRole("button", {
          name: /select character or persona/i,
        }),
      );
    });
  });

  it("does not render cockpit control rails in focus mode", async () => {
    storageState.values.set("playgroundChatLayoutMode", "focus");

    render(<Playground />);

    expect(
      await screen.findByTestId("playground-cockpit-shell"),
    ).toHaveAttribute("data-mode", "focus");
    expect(screen.queryByTestId("playground-context-rail")).toBeNull();
    expect(screen.queryByTestId("playground-runtime-inspector")).toBeNull();
    expect(screen.getByTestId("playground-chat")).toBeInTheDocument();
    expect(screen.getByTestId("playground-form")).toBeInTheDocument();
  });

  it("wires ready-state runtime regenerate to the shared chat handler", async () => {
    messageOptionState.value.streaming = false;

    render(<Playground />);

    const runtimeInspector = within(
      await screen.findByTestId("playground-cockpit-right-rail"),
    ).getByTestId("playground-runtime-inspector");
    expect(
      within(runtimeInspector).queryByRole("button", {
        name: "Stop generation",
      }),
    ).toBeNull();

    fireEvent.click(
      within(runtimeInspector).getByRole("button", {
        name: "Regenerate last response",
      }),
    );

    expect(messageOptionState.value.regenerateLastMessage).toHaveBeenCalledTimes(
      1,
    );
  });

  it("reflects degraded server readiness in the cockpit runtime and status strip", async () => {
    messageOptionState.value.streaming = false;

    render(<Playground />);

    await screen.findByTestId("playground-cockpit-shell");
    act(() => {
      window.dispatchEvent(
        new CustomEvent("tldw:server-readiness-state", {
          detail: { state: "degraded", degradedChecks: ["chacha_notes"] },
        }),
      );
    });

    const runtimeInspector = within(
      screen.getByTestId("playground-cockpit-right-rail"),
    ).getByTestId("playground-runtime-inspector");
    expect(within(runtimeInspector).getByText("Degraded")).toBeInTheDocument();
    expect(
      within(runtimeInspector).getByText("Degraded: chacha_notes"),
    ).toBeInTheDocument();
    expect(screen.getByRole("status", { name: "Chat status" })).toHaveTextContent(
      "chacha_notes",
    );
  });

  it("keeps the cockpit visibly degraded when readiness details are empty", async () => {
    messageOptionState.value.streaming = false;

    render(<Playground />);

    await screen.findByTestId("playground-cockpit-shell");
    act(() => {
      window.dispatchEvent(
        new CustomEvent("tldw:server-readiness-state", {
          detail: { state: "degraded", degradedChecks: [] },
        }),
      );
    });

    const runtimeInspector = within(
      screen.getByTestId("playground-cockpit-right-rail"),
    ).getByTestId("playground-runtime-inspector");
    expect(within(runtimeInspector).getByText("Degraded")).toBeInTheDocument();
    expect(screen.getByRole("status", { name: "Chat status" })).toHaveTextContent(
      "Degraded",
    );
  });
});
