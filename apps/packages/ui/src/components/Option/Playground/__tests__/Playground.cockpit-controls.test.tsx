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
import { getPromptById } from "@/db/dexie/helpers";
import { useMcpToolsStore } from "@/store/mcp-tools";
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
    serverChatTitle: "Research session" as string | null,
    serverChatLoadState: "loaded" as "idle" | "loading" | "loaded" | "failed",
    serverChatLoadError: null as string | null,
    serverChatState: "active" as string | null,
    serverChatTopic: "Research" as string | null,
    serverChatSource: "webui" as string | null,
    serverChatCharacterId: null as string | number | null,
    setServerChatCharacterId: vi.fn(),
    serverChatAssistantKind: null as "character" | "persona" | null,
    setServerChatAssistantKind: vi.fn(),
    serverChatAssistantId: null as string | number | null,
    setServerChatAssistantId: vi.fn(),
    serverChatMetaLoaded: true,
    setServerChatMetaLoaded: vi.fn(),
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
    isSearchingInternet: false,
    selectedCharacter: { id: "character-1", name: "Mira Vale" },
    setSelectedCharacter: vi.fn(),
    selectedAssistant: {
      kind: "character",
      id: "character-1",
      name: "Mira Vale",
    } as { kind: "character" | "persona"; id: string; name: string } | null,
    serverChatPersonaMemoryMode: null as "read_only" | "read_write" | null,
    setServerChatPersonaMemoryMode: vi.fn(),
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
    clearPersistedSession: vi.fn(),
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

const chatSettingsState = vi.hoisted(() => ({
  syncChatSettingsForServerChat: vi.fn(async (_params: unknown) => null),
  applyChatSettingsPatch: vi.fn(async (_params: unknown) => null),
}));

const serverChatHistoryState = vi.hoisted(() => ({
  value: [
    {
      id: "tracked-character-chat",
      title: "Mira field notes",
      assistant_kind: "character",
      character_id: "character-1",
      created_at: "2026-05-22T12:00:00.000Z",
      updated_at: "2026-05-22T12:15:00.000Z",
    },
    {
      id: "plain-chat",
      title: "Plain chat",
      assistant_kind: null,
      character_id: null,
      created_at: "2026-05-21T12:00:00.000Z",
      updated_at: "2026-05-21T12:15:00.000Z",
    },
  ],
}));

const tldwServerState = vi.hoisted(() => ({
  fetchChatModels: vi.fn(async (): Promise<any[]> => [
    {
      model: "openai:gpt-4.1-mini",
      provider: "openai",
      is_configured: true,
      provider_is_configured: true,
    },
  ]),
}));

const tldwClientState = vi.hoisted(() => ({
  initialize: vi.fn(async () => null),
  getProvidersStatus: vi.fn(async () => ({
    providers: [
      {
        name: "openai",
        configured: true,
        requires_api_key: true,
      },
    ],
    any_configured: true,
  })),
  getResearchBundle: vi.fn(async () => ({})),
}));

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?: string | { defaultValue?: string },
      interpolationOptions?: Record<string, unknown>,
    ) => {
      const defaultValue =
        typeof defaultValueOrOptions === "string"
          ? defaultValueOrOptions
          : defaultValueOrOptions?.defaultValue || key;
      const values =
        typeof defaultValueOrOptions === "object" && defaultValueOrOptions
          ? defaultValueOrOptions
          : interpolationOptions;

      return defaultValue.replace(/\{\{(\w+)\}\}/g, (_match, token) =>
        values?.[token] == null ? `{{${token}}}` : String(values[token]),
      );
    },
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

vi.mock("@/services/chat-settings", () => ({
  applyChatSettingsPatch: (params: unknown) =>
    chatSettingsState.applyChatSettingsPatch(params),
  syncChatSettingsForServerChat: (params: unknown) =>
    chatSettingsState.syncChatSettingsForServerChat(params),
}));

vi.mock("@/services/tldw-server", () => ({
  fetchChatModels: tldwServerState.fetchChatModels,
}));

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: tldwClientState,
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

vi.mock("@/hooks/useServerChatHistory", () => ({
  useServerChatHistory: () => ({
    data: serverChatHistoryState.value,
    total: serverChatHistoryState.value.length,
    isLoading: false,
    sidebarRefreshState: "ready",
    hasUsableData: true,
    isShowingStaleData: false,
  }),
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
    useLocation: () => ({
      pathname: window.location.pathname || "/chat",
      search: window.location.search || "",
      hash: window.location.hash || "",
      state: null,
      key: "test-location",
    }),
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
    messageOptionState.value.serverChatTitle = "Research session";
    messageOptionState.value.serverChatLoadState = "loaded";
    messageOptionState.value.serverChatLoadError = null;
    messageOptionState.value.serverChatState = "active";
    messageOptionState.value.serverChatTopic = "Research";
    messageOptionState.value.serverChatSource = "webui";
    messageOptionState.value.serverChatCharacterId = null;
    messageOptionState.value.setServerChatCharacterId = vi.fn();
    messageOptionState.value.serverChatAssistantKind = null;
    messageOptionState.value.setServerChatAssistantKind = vi.fn();
    messageOptionState.value.serverChatAssistantId = null;
    messageOptionState.value.setServerChatAssistantId = vi.fn();
    messageOptionState.value.serverChatMetaLoaded = true;
    messageOptionState.value.setServerChatMetaLoaded = vi.fn();
    messageOptionState.value.setServerChatId = vi.fn();
    messageOptionState.value.setHistoryId = vi.fn();
    messageOptionState.value.setHistory = vi.fn();
    messageOptionState.value.setMessages = vi.fn();
    messageOptionState.value.streaming = true;
    messageOptionState.value.isSearchingInternet = false;
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
    messageOptionState.value.setServerChatPersonaMemoryMode = vi.fn();
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
    messageOptionState.value.setSelectedAssistant = vi.fn();
    messageOptionState.value.setSelectedCharacter = vi.fn();
    sessionPersistenceState.value.clearPersistedSession = vi.fn();
    chatSettingsState.syncChatSettingsForServerChat.mockClear();
    chatSettingsState.applyChatSettingsPatch.mockClear();
    tldwServerState.fetchChatModels.mockClear();
    tldwClientState.initialize.mockClear();
    tldwClientState.getProvidersStatus.mockClear();
    tldwClientState.getProvidersStatus.mockResolvedValue({
      providers: [
        {
          name: "openai",
          configured: true,
          requires_api_key: true,
        },
      ],
      any_configured: true,
    });
    tldwClientState.getResearchBundle.mockClear();
    useMcpToolsStore.setState({
      healthState: "healthy",
      toolsLoading: false,
      discoveredTools: [{ name: "search" } as never],
      chatTools: [{ name: "search" } as never],
      toolCounts: {
        discovered: 4,
        executable: 3,
        disabled: 1,
        colliding: 0,
        chatEnabled: 2,
      },
    });
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
      const compositionPreview = within(contextRail).getByRole("region", {
        name: "Next message composition",
      });
      expect(within(compositionPreview).getByText("Draft a concise summary")).toBeInTheDocument();
      expect(within(compositionPreview).getByText("Mira Vale")).toBeInTheDocument();
      expect(
        within(compositionPreview).getAllByText("openai:gpt-4.1-mini").length,
      ).toBeGreaterThan(0);
      expect(within(compositionPreview).getAllByText("MCP tools").length).toBeGreaterThan(0);
      expect(
        within(compositionPreview).getByText("Scope: openai:gpt-4.1-mini"),
      ).toBeInTheDocument();
      expect(within(contextRail).getByText("Context active")).toBeInTheDocument();
      expect(within(contextRail).getByText("1 file")).toBeInTheDocument();
      expect(
        within(contextRail).getByText("1 knowledge item"),
      ).toBeInTheDocument();
      expect(within(contextRail).getByText("2 media scopes")).toBeInTheDocument();
      const statusStrip = screen.getByRole("status", { name: "Chat status" });
      expect(statusStrip).toHaveTextContent("Web search on");
      expect(statusStrip).toHaveTextContent("1 file");
      expect(statusStrip).toHaveTextContent("1 knowledge item");
      expect(statusStrip).toHaveTextContent("+1 more");
      expect(
        within(statusStrip).getByRole("button", {
          name: /open search & context/i,
        }),
      ).toBeInTheDocument();
      expect(within(contextRail).getByText("Temporary chat")).toBeInTheDocument();
      expect(within(contextRail).getByText("History linked")).toBeInTheDocument();
      expect(
        within(contextRail).getByRole("heading", { name: "Prompt" }),
      ).toBeInTheDocument();
      expect(
        within(contextRail).getAllByText("Quick prompt").length,
      ).toBeGreaterThan(0);
      expect(
        within(contextRail).getAllByText("Draft a concise summary").length,
      ).toBeGreaterThan(0);
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
        contextRail.querySelector("[data-cockpit-prompt-select-trigger]"),
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
        within(runtimeInspector).getByRole("button", {
          name: "Regenerate last response",
        }),
      ).toBeDisabled();
      expect(
        within(runtimeInspector).getByText(
          "Wait for the current turn to finish before regenerating.",
        ),
      ).toBeInTheDocument();
      expect(within(runtimeInspector).getByText("2 messages")).toBeInTheDocument();
      expect(within(runtimeInspector).getByText("Mira Vale")).toBeInTheDocument();
      const mcpCounts = within(runtimeInspector).getByLabelText(
        "MCP tool state counts",
      );
      expect(within(mcpCounts).getByText("Discovered")).toBeInTheDocument();
      expect(
        within(mcpCounts).getByText("Discovered").nextElementSibling,
      ).toHaveTextContent("4");
      expect(within(mcpCounts).getByText("Executable")).toBeInTheDocument();
      expect(
        within(mcpCounts).getByText("Executable").nextElementSibling,
      ).toHaveTextContent("3");
      expect(within(mcpCounts).getByText("Chat-enabled")).toBeInTheDocument();
      expect(
        within(mcpCounts).getByText("Chat-enabled").nextElementSibling,
      ).toHaveTextContent("2");
      expect(within(mcpCounts).getByText("User-disabled")).toBeInTheDocument();
      expect(
        within(mcpCounts).getByText("User-disabled").nextElementSibling,
      ).toHaveTextContent("1");
      expect(within(mcpCounts).getByText("Unavailable")).toBeInTheDocument();
      expect(
        within(mcpCounts).getByText("Unavailable").nextElementSibling,
      ).toHaveTextContent("1");

      fireEvent.click(
        within(runtimeInspector).getByRole("button", {
          name: /open model settings/i,
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
          settingsScope: string;
        }>).detail,
      ).toEqual({
        returnFocusSelector: "[data-cockpit-model-settings-trigger]",
        settingsScope: "openai:gpt-4.1-mini",
      });
      expect(openAssistantSelect).toHaveBeenCalledTimes(1);
      expect(openActorSettings).not.toHaveBeenCalled();
      expect(messageOptionState.value.setToolChoice).toHaveBeenCalledWith(
        "required",
      );
      expect(openMcpSettings).toHaveBeenCalledTimes(1);
      expect(
        (openMcpSettings.mock.calls[0]?.[0] as CustomEvent<{
          returnFocusSelector: string;
        }>).detail,
      ).toEqual({
        returnFocusSelector: "[data-cockpit-mcp-settings-trigger]",
      });
    } finally {
      window.removeEventListener("tldw:open-knowledge-panel", openKnowledgePanel);
      window.removeEventListener("tldw:open-model-settings", openModelSettings);
      window.removeEventListener("tldw:open-actor-settings", openActorSettings);
      window.removeEventListener(OPEN_ASSISTANT_SELECT_EVENT, openAssistantSelect);
      window.removeEventListener(OPEN_MCP_SETTINGS_EVENT, openMcpSettings);
      window.removeEventListener(TOGGLE_WEB_SEARCH_EVENT, toggleWebSearch);
      window.removeEventListener(SET_TEMPORARY_CHAT_EVENT, setTemporaryChat);
    }
  }, 15000);

  it("surfaces active web search progress in the cockpit status strip", async () => {
    messageOptionState.value.streaming = false;
    messageOptionState.value.isSearchingInternet = true;
    messageOptionState.value.selectedAssistant = null;
    messageOptionState.value.selectedCharacter = null;

    render(<Playground />);

    const statusStrip = await screen.findByRole("status", {
      name: "Chat status",
    });
    expect(statusStrip).toHaveTextContent("Searching web");
    expect(statusStrip).toHaveTextContent("Web search on");
    expect(statusStrip).not.toHaveTextContent("Ready");
  });

  it("logs selected prompt lookup failures while keeping the prompt unavailable state visible", async () => {
    const promptLookupWarning = vi
      .spyOn(console, "warn")
      .mockImplementation(() => undefined);
    vi.mocked(getPromptById).mockRejectedValueOnce(
      new Error("prompt cache unavailable"),
    );
    messageOptionState.value.selectedSystemPrompt = "prompt-missing";

    try {
      render(<Playground />);

      const contextRail = within(
        await screen.findByTestId("playground-cockpit-left-rail"),
      ).getByTestId("playground-context-rail");

      await waitFor(() => {
        expect(promptLookupWarning).toHaveBeenCalledWith(
          "[Playground] Failed to resolve selected system prompt",
          expect.objectContaining({
            promptId: "prompt-missing",
            error: expect.any(Error),
          }),
        );
      });
      expect(
        within(contextRail).getAllByText("Prompt details unavailable").length,
      ).toBeGreaterThan(0);
    } finally {
      promptLookupWarning.mockRestore();
    }
  });

  it("lists prompt and assistant as first-class context sources and removes one knowledge item at a time", async () => {
    messageOptionState.value.streaming = false;
    messageOptionState.value.selectedQuickPrompt = "Draft a concise summary";
    messageOptionState.value.selectedKnowledge = [
      { id: "knowledge-1", title: "Research notes" },
      { id: "knowledge-2", title: "Protocol notes" },
    ];

    render(<Playground />);

    const contextRail = within(
      await screen.findByTestId("playground-cockpit-left-rail"),
    ).getByTestId("playground-context-rail");
    const sourceList = within(contextRail).getByRole("list", {
      name: "Context sources",
    });

    const promptSource = within(sourceList).getByText("Quick prompt").closest("li");
    expect(promptSource).not.toBeNull();
    expect(within(promptSource as HTMLElement).getByText("Prompt")).toBeInTheDocument();
    expect(
      within(promptSource as HTMLElement).getByText("Draft a concise summary"),
    ).toBeInTheDocument();

    const assistantSource = within(sourceList).getByText("Mira Vale").closest("li");
    expect(assistantSource).not.toBeNull();
    expect(
      within(assistantSource as HTMLElement).getByText("Character"),
    ).toBeInTheDocument();

    const protocolSource = within(sourceList).getByText("Protocol notes").closest("li");
    expect(protocolSource).not.toBeNull();
    fireEvent.click(
      within(protocolSource as HTMLElement).getByRole("button", {
        name: "Remove Protocol notes",
      }),
    );
    expect(messageOptionState.value.setSelectedKnowledge).toHaveBeenCalledWith([
      { id: "knowledge-1", title: "Research notes" },
    ]);

    fireEvent.click(
      within(promptSource as HTMLElement).getByRole("button", {
        name: "Clear prompt context",
      }),
    );
    expect(messageOptionState.value.setSelectedQuickPrompt).toHaveBeenCalledWith(
      null,
    );
    expect(messageOptionState.value.setSelectedSystemPrompt).toHaveBeenCalledWith(
      "",
    );
    expect(messageOptionState.value.setContextFiles).not.toHaveBeenCalled();

    fireEvent.click(
      within(assistantSource as HTMLElement).getByRole("button", {
        name: "Clear assistant",
      }),
    );
    await waitFor(() => {
      expect(messageOptionState.value.setSelectedAssistant).toHaveBeenCalledWith(
        null,
      );
      expect(messageOptionState.value.setSelectedCharacter).toHaveBeenCalledWith(
        null,
      );
    });
  });

  it("shows server session title and recoverable load errors in the context rail", async () => {
    messageOptionState.value.streaming = false;
    messageOptionState.value.temporaryChat = false;
    messageOptionState.value.historyId = null;
    messageOptionState.value.serverChatId = "chat-2";
    messageOptionState.value.serverChatTitle = "Archived investigation";
    messageOptionState.value.serverChatLoadState = "failed";
    messageOptionState.value.serverChatLoadError = "Conversation no longer exists";

    render(<Playground />);

    const contextRail = within(
      await screen.findByTestId("playground-cockpit-left-rail"),
    ).getByTestId("playground-context-rail");

    expect(within(contextRail).getByText("Server chat")).toBeInTheDocument();
    expect(
      within(contextRail).getByText("Archived investigation"),
    ).toBeInTheDocument();
    expect(within(contextRail).getByText("Load failed")).toBeInTheDocument();
    expect(
      within(contextRail).getByText("Conversation no longer exists"),
    ).toBeInTheDocument();
    expect(within(contextRail).getByText("No saved history yet")).toBeInTheDocument();
  });

  it("surfaces unavailable MCP without enabling cockpit-only tool choice", async () => {
    useMcpToolsStore.setState({
      healthState: "unavailable",
      toolsLoading: false,
      discoveredTools: [],
      chatTools: [],
    });
    const openMcpSettings = vi.fn();
    window.addEventListener(OPEN_MCP_SETTINGS_EVENT, openMcpSettings);

    try {
      render(<Playground />);

      const runtimeInspector = within(
        await screen.findByTestId("playground-cockpit-right-rail"),
      ).getByTestId("playground-runtime-inspector");

      expect(
        within(runtimeInspector).getByText("MCP unavailable"),
      ).toBeInTheDocument();
      expect(
        within(runtimeInspector).getByText("MCP tools unavailable"),
      ).toBeInTheDocument();
      expect(
        within(runtimeInspector).queryByRole("button", {
          name: "MCP tool choice Auto",
        }),
      ).toBeNull();

      fireEvent.click(
        within(runtimeInspector).getByRole("button", {
          name: "Configure MCP tools",
        }),
      );

      expect(messageOptionState.value.setToolChoice).not.toHaveBeenCalled();
      expect(openMcpSettings).toHaveBeenCalledTimes(1);
    } finally {
      window.removeEventListener(OPEN_MCP_SETTINGS_EVENT, openMcpSettings);
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

    expect(document.activeElement).toBe(clearButton);
    await waitFor(() => {
      expect(messageOptionState.value.setSelectedAssistant).toHaveBeenCalledWith(
        null,
      );
      expect(messageOptionState.value.setSelectedCharacter).toHaveBeenCalledWith(
        null,
      );
    });

    messageOptionState.value.selectedAssistant = null;
    messageOptionState.value.selectedCharacter = null;
    rerender(<Playground />);
    const updatedRuntimeInspector = within(
      screen.getByTestId("playground-cockpit-right-rail"),
    ).getByTestId("playground-runtime-inspector");
    expect(
      within(updatedRuntimeInspector).getAllByText("No runtime assistant selected")
        .length,
    ).toBeGreaterThan(0);
    expect(
      within(updatedRuntimeInspector).queryByText("No assistant selected"),
    ).toBeNull();
    expect(within(updatedRuntimeInspector).queryByText("Legacy Character")).toBeNull();
    await waitFor(() => {
      expect(document.activeElement).toBe(
        within(updatedRuntimeInspector).getByRole("button", {
          name: /select character or persona/i,
        }),
      );
    });
  });

  it("clears tracked server-chat assistant metadata from the runtime rail", async () => {
    messageOptionState.value.streaming = false;
    messageOptionState.value.selectedAssistant = {
      kind: "character",
      id: "character-1",
      name: "Mira Vale",
    };
    messageOptionState.value.selectedCharacter = {
      id: "character-1",
      name: "Mira Vale",
    };
    messageOptionState.value.serverChatId = "tracked-character-chat";
    messageOptionState.value.serverChatCharacterId = "character-1";
    messageOptionState.value.serverChatAssistantKind = "character";
    messageOptionState.value.serverChatAssistantId = "character-1";
    messageOptionState.value.serverChatPersonaMemoryMode = null;
    messageOptionState.value.serverChatMetaLoaded = true;

    render(<Playground />);

    const runtimeInspector = within(
      await screen.findByTestId("playground-cockpit-right-rail"),
    ).getByTestId("playground-runtime-inspector");
    fireEvent.click(
      within(runtimeInspector).getByRole("button", {
        name: "Clear assistant",
      }),
    );

    await waitFor(() => {
      expect(messageOptionState.value.setSelectedAssistant).toHaveBeenCalledWith(
        null,
      );
      expect(messageOptionState.value.setSelectedCharacter).toHaveBeenCalledWith(
        null,
      );
      expect(messageOptionState.value.setServerChatCharacterId).toHaveBeenCalledWith(
        null,
      );
      expect(
        messageOptionState.value.setServerChatAssistantKind,
      ).toHaveBeenCalledWith(null);
      expect(messageOptionState.value.setServerChatAssistantId).toHaveBeenCalledWith(
        null,
      );
      expect(
        messageOptionState.value.setServerChatPersonaMemoryMode,
      ).toHaveBeenCalledWith(null);
      expect(messageOptionState.value.setServerChatMetaLoaded).toHaveBeenCalledWith(
        false,
      );
      expect(messageOptionState.value.setServerChatId).toHaveBeenCalledWith(null);
      expect(sessionPersistenceState.value.clearPersistedSession).toHaveBeenCalled();
    });
  });

  it("clears persisted assistant overlay settings when returning to plain chat", async () => {
    messageOptionState.value.streaming = false;
    messageOptionState.value.historyId = "history-overlay";
    messageOptionState.value.serverChatId = "overlay-chat";
    messageOptionState.value.selectedAssistant = {
      kind: "persona",
      id: "persona-overlay",
      name: "Overlay Persona",
      metadata: {
        selectionMode: "overlay",
      },
    } as any;
    messageOptionState.value.selectedCharacter = null;

    render(<Playground />);

    const runtimeInspector = within(
      await screen.findByTestId("playground-cockpit-right-rail"),
    ).getByTestId("playground-runtime-inspector");
    fireEvent.click(
      within(runtimeInspector).getByRole("button", {
        name: "Clear assistant",
      }),
    );

    await waitFor(() => {
      expect(chatSettingsState.applyChatSettingsPatch).toHaveBeenCalledWith({
        historyId: "history-overlay",
        serverChatId: "overlay-chat",
        patch: {
          assistantOverlay: null,
        },
      });
    });
    expect(messageOptionState.value.setSelectedAssistant).toHaveBeenCalledWith(null);
    expect(messageOptionState.value.setSelectedCharacter).toHaveBeenCalledWith(null);
  });

  it("does not render cockpit control rails in focus mode", async () => {
    storageState.values.set("playgroundChatLayoutMode", "focus");

    render(<Playground />);

    expect(
      await screen.findByTestId("playground-cockpit-shell"),
    ).toHaveAttribute("data-mode", "focus");
    expect(screen.queryByTestId("playground-context-rail")).toBeNull();
    expect(screen.queryByTestId("playground-runtime-inspector")).toBeNull();
    expect(
      screen.queryByRole("region", { name: "Next message composition" }),
    ).toBeNull();
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
      within(runtimeInspector).getByRole("button", {
        name: "Stop generation",
      }),
    ).toBeDisabled();
    expect(within(runtimeInspector).getByText("No turn is running.")).toBeInTheDocument();

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
    messageOptionState.value.selectedAssistant = null;
    messageOptionState.value.selectedCharacter = null;

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

  it("keeps streaming primary when degraded readiness is warning-only", async () => {
    messageOptionState.value.streaming = true;
    messageOptionState.value.selectedAssistant = null;
    messageOptionState.value.selectedCharacter = null;

    render(<Playground />);

    await screen.findByTestId("playground-cockpit-shell");
    act(() => {
      window.dispatchEvent(
        new CustomEvent("tldw:server-readiness-state", {
          detail: { state: "degraded", degradedChecks: ["embeddings"] },
        }),
      );
    });

    const runtimeInspector = within(
      screen.getByTestId("playground-cockpit-right-rail"),
    ).getByTestId("playground-runtime-inspector");
    expect(within(runtimeInspector).getByText("Streaming")).toBeInTheDocument();

    const status = screen.getByRole("status", { name: "Chat status" });
    expect(status).toHaveTextContent("Streaming");
    expect(status).toHaveTextContent("embeddings");
    expect(status).toHaveTextContent("Chat remains available.");
  });

  it("surfaces blocked server readiness as chat-critical unavailable state", async () => {
    messageOptionState.value.streaming = false;
    messageOptionState.value.selectedAssistant = null;
    messageOptionState.value.selectedCharacter = null;

    render(<Playground />);

    await screen.findByTestId("playground-cockpit-shell");
    act(() => {
      window.dispatchEvent(
        new CustomEvent("tldw:server-readiness-state", {
          detail: { state: "blocked", degradedChecks: ["chat"] },
        }),
      );
    });

    const runtimeInspector = within(
      screen.getByTestId("playground-cockpit-right-rail"),
    ).getByTestId("playground-runtime-inspector");
    expect(within(runtimeInspector).getByText("Error")).toBeInTheDocument();
    expect(
      within(runtimeInspector).getByText(
        "Server is unavailable. Check the server connection before sending.",
      ),
    ).toBeInTheDocument();

    const status = screen.getByRole("status", { name: "Chat status" });
    expect(status).toHaveTextContent("Server unavailable");
    expect(status).toHaveTextContent(
      "Reconnect to the server or review server settings before sending.",
    );
    expect(status).not.toHaveTextContent("Chat remains available.");
  });

  it("keeps provider setup blocking consistent across empty-state and cockpit rails", async () => {
    messageOptionState.value.messages = [];
    messageOptionState.value.historyId = null;
    messageOptionState.value.serverChatId = null;
    messageOptionState.value.streaming = false;
    messageOptionState.value.selectedAssistant = null;
    messageOptionState.value.selectedCharacter = null;
    messageOptionState.value.selectedModel = "tldw:gpt-4o";
    modelSettingsState.value.apiProvider = undefined;
    modelSettingsState.value.activeSettingsScope = "tldw:gpt-4o";
    tldwServerState.fetchChatModels.mockResolvedValueOnce([
      {
        id: "gpt-4o",
        model: "tldw:gpt-4o",
        name: "GPT-4o",
        provider: "openai",
        type: "chat",
      },
    ]);
    tldwClientState.getProvidersStatus.mockResolvedValueOnce({
      providers: [
        {
          name: "openai",
          configured: false,
          requires_api_key: true,
        },
      ],
      any_configured: false,
    });

    render(<Playground />);

    const leftRail = await screen.findByTestId("playground-cockpit-left-rail");
    expect(leftRail).toBeInTheDocument();
    expect(screen.getByTestId("playground-chat")).toBeInTheDocument();
    expect(
      within(leftRail).getByRole("button", { name: "Expand Prompt" }),
    ).toHaveAttribute("aria-expanded", "false");
    const rightRail = await screen.findByTestId("playground-cockpit-right-rail");
    expect(
      within(rightRail).getByRole("button", { name: "Expand Assistant" }),
    ).toHaveAttribute("aria-expanded", "false");

    const runtimeInspector = within(
      rightRail,
    ).getByTestId("playground-runtime-inspector");

    await waitFor(() => {
      expect(within(runtimeInspector).getByText("Error")).toBeInTheDocument();
    });
    expect(runtimeInspector).toHaveTextContent("Provider setup needed");

    const status = screen.getByRole("status", { name: "Chat status" });
    expect(status).toHaveTextContent("Model setup needed");
    expect(status).toHaveTextContent("Provider setup needed");
    expect(status).not.toHaveTextContent("Ready");
  });

  it("keeps the cockpit visibly degraded when readiness details are empty", async () => {
    messageOptionState.value.streaming = false;
    messageOptionState.value.selectedAssistant = null;
    messageOptionState.value.selectedCharacter = null;

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
