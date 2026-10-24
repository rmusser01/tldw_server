// @vitest-environment jsdom
import React from "react";
import {
  act,
  cleanup,
  fireEvent,
  render,
  renderHook,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { CurrentChatModelSettings } from "@/components/Common/Settings/CurrentChatModelSettings";
import { useStoreChatModelSettings } from "@/store/model";
import { useActorStore } from "@/store/actor";
import { usePlaygroundSessionStore } from "@/store/playground-session";
import { useStoreMessageOption } from "@/store/option";
import { captureChatDraftSource } from "@/hooks/chat/chat-draft-transition";
import { useChatSettingsRecord } from "@/hooks/chat/useChatSettingsRecord";

import { Playground } from "../Playground";
import { resolveEffectiveAssistantState } from "@/hooks/chat/effective-assistant-state";
import type { AssistantSelection } from "@/types/assistant-selection";
import { PlaygroundCockpitShell } from "../PlaygroundCockpitShell";
import { getPromptById } from "@/db/dexie/helpers";
import { useMcpToolsStore } from "@/store/mcp-tools";
import {
  OPEN_ASSISTANT_SELECT_EVENT,
  OPEN_MCP_SETTINGS_EVENT,
  SET_TEMPORARY_CHAT_EVENT,
  TOGGLE_WEB_SEARCH_EVENT,
} from "../playground-cockpit-actions";

const forkSettings = vi.hoisted(() => ({
  controller: null as any,
  mode: "ordinary",
  current: true,
  fence: () => () => forkSettings.current,
  update: vi.fn<(...args: any[]) => Promise<any>>(async () => ({}))
}))
vi.mock("@/hooks/chat/useHistorySelection", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/hooks/chat/useHistorySelection")>()
  return {
    ...actual,
    useHistorySelectionContext: () => {
      const controller = actual.useHistorySelectionContext()
      const override = forkSettings.controller;
      const mode = forkSettings.mode;
      return React.useMemo(() => override ? { ...controller, ...override } : {
        ...controller,
        settingsMode: () => forkSettings.mode,
        updateForkSettings: forkSettings.update,
        fence: forkSettings.fence
      }, [controller, override, mode]);
    }
  }
})

vi.mock("@/db/dexie/history-selection", () => ({ensureLocalProfileId: async () => "profile", loadHistoryBookmark: async () => null, saveHistoryBookmark: async () => {}, loadHistoryTurnRecoveries: async () => []}))
vi.mock("@/db/dexie/fork-operations", () => ({findForkCandidate: async () => null, loadForkOperations: async () => []}))
vi.mock("@/services/chat-history-selection", () => ({captureHistorySnapshot: async (owner: any, view: any) => ({status: "legacy_review_required", code: "legacy_review_required", view, snapshot: {version: 1, owner_key: owner.owner_key, conversation_id: owner.conversation_id, nodes: [], source_digest: "source", storage_context_digest: "storage", fences: {}, interpretation_status: {kind: "legacy_review_required"}}})}))
import {useHistorySelection, useHistorySelectionContext} from "@/hooks/chat/useHistorySelection"

const messageOptionState = vi.hoisted(() => ({
  value: {
    messages: [
      { id: "message-1", role: "user", content: "Hello" },
      { id: "message-2", role: "assistant", content: "Hi" },
    ],
    history: [],
    uploadedFiles: [],
    removeUploadedFile: vi.fn(),
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
    selectedModelRevision: 0,
    chatDraftAddress: null as import("@/utils/chat-draft-address").ChatDraftAddress | null,
    chatDraftRevision: 0,
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
    } as AssistantSelection | null,
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

const layoutShellOverrideState = vi.hoisted(() => ({
  overrides: [] as unknown[],
}));

const modelSettingsState = vi.hoisted(() => ({
  useRealStore: false,
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

const connectionState = vi.hoisted(() => ({ isConnected: true, phase: "connected" }));

vi.mock("@/hooks/useConnectionState", async (original) => {
  const actual = await original<typeof import("@/hooks/useConnectionState")>()
  return {
    ...actual,
    useConnectionState: () => connectionState
  }
})

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

// Keep Playground's actual scope effect and the actual settings dialog. The
// composer surface is reduced to its open/close lifecycle for this integration.
const SettingsDialogCaller = () => {
  const [open, setOpen] = React.useState(false);
  return (
    <>
      <button onClick={() => setOpen(true)}>Current Chat Model Settings</button>
      {open && <CurrentChatModelSettings open={open} setOpen={setOpen} />}
    </>
  );
};
vi.mock("@/components/Option/Playground/PlaygroundForm", () => ({
  PlaygroundForm: () =>
    modelSettingsState.useRealStore ? (
      <SettingsDialogCaller />
    ) : (
      <div data-testid="playground-form" />
    ),
}));

vi.mock("@/services/model-settings", () => ({
  getAllModelSettings: async () => ({ temperature: 0.7 }),
}));
vi.mock("@/services/ocr", () => ({ getOCRLanguage: async () => null }));
vi.mock("@/services/actor-settings", async importOriginal => ({
  ...await importOriginal<typeof import("@/services/actor-settings")>(),
  getActorSceneForChat: async ({ target }) => ({
    target, originalRecord: null, settings: (await import("@/types/actor")).createDefaultActorSettings(), origin: "default", profileUnavailable: false
  }),
  saveActorSettingsForChat: vi.fn(async ({ settings }) => settings),
}));
vi.mock("@/hooks/useSelectedCharacter", () => ({
  useSelectedCharacter: () => [null, vi.fn(), { isLoading: false }],
}));
vi.mock("@/hooks/useSelectedAssistant", () => ({
  useSelectedAssistant: () => [null],
}));
vi.mock("@/hooks/chat/useChatSettingsRecord", () => ({
  useChatSettingsRecord: ({ historyId, serverChatId }: { historyId: string | null; serverChatId: string | null }) => {
    const controller = useHistorySelectionContext();
    const mode = controller!.settingsMode(serverChatId);
    const target = React.useMemo(() => ({ historyId, serverChatId }), [historyId, serverChatId]);
    const rendered = useStoreMessageOption.getState();
    const session = usePlaygroundSessionStore.getState();
    const source = React.useMemo(() => {
      const fence = controller!.fence();
      return captureChatDraftSource(() => fence() && controller!.settingsMode(serverChatId) === mode);
    }, [controller, mode, serverChatId, rendered.historyId, rendered.serverChatId, rendered.chatDraftAddress, rendered.chatDraftRevision,
      session.restoreRevision, session.sourceSelectionRevision]);
    const updateSettings = React.useCallback(async (patch: unknown) => {
      if (!source.isCurrent() || mode === "pending") throw new Error("chat_settings_owner_unavailable");
      const result = mode === "fork"
        ? await controller!.updateForkSettings(patch as any)
        : await chatSettingsState.applyChatSettingsPatch({ ...target, patch });
      if (!source.isCurrent()) throw new Error("chat_settings_owner_unavailable");
      return result;
    }, [controller, mode, target, source]);
    return { settings: null, updateSettings, target, chatKey: null };
  },
}));
vi.mock("@/components/Common/Settings/tabs", async () => ({
  ConversationTab: (
    await import("@/components/Common/Settings/tabs/ConversationTab")
  ).ConversationTab,
  ModelBasicsTab: () => null,
  ActorTab: () => null,
  AdvancedParamsTab: () => null,
}));
vi.mock("@/components/Common/Settings/PromptAssemblyPreview", () => ({
  PromptAssemblyPreview: () => null,
}));
vi.mock("@/components/Common/Settings/LorebookDebugPanel", () => ({
  LorebookDebugPanel: () => null,
}));
vi.mock("@/components/Common/Settings/LlamaCppAdvancedControls", () => ({
  LlamaCppAdvancedControls: () => null,
}));

vi.mock("@/components/Option/Playground/PlaygroundChat", () => ({
  PlaygroundChat: () => <div data-testid="playground-chat" />,
}));

vi.mock("@/components/Layouts/Layout", () => ({
  default: ({ children }: { children: React.ReactNode }) => children,
  useOptionLayoutShellOverrides: (overrides: unknown) => {
    layoutShellOverrideState.overrides.push(overrides);
  },
}));

vi.mock("@/components/Sidepanel/Chat/ArtifactsPanel", () => ({
  ArtifactsPanel: () => null,
}));

vi.mock("@/hooks/useMessageOption", () => ({
  useMessageOption: () => ({
    ...messageOptionState.value,
    effectiveAssistantState: resolveEffectiveAssistantState({
      tracked: {
        assistantKind: messageOptionState.value.serverChatAssistantKind,
        assistantId: messageOptionState.value.serverChatAssistantId,
        characterId: messageOptionState.value.serverChatCharacterId,
      },
      draftSelection: messageOptionState.value.selectedAssistant,
    }),
  }),
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

vi.mock("@/services/chat-settings", async importOriginal => ({
  ...await importOriginal<typeof import("@/services/chat-settings")>(),
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

vi.mock("@/store/model", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/store/model")>();
  const useSettings = Object.assign(
    (...args: Parameters<typeof actual.useStoreChatModelSettings>) =>
      modelSettingsState.useRealStore
        ? actual.useStoreChatModelSettings(...args)
        : modelSettingsState.value,
    actual.useStoreChatModelSettings,
    { getState: () => modelSettingsState.useRealStore ? actual.useStoreChatModelSettings.getState() : modelSettingsState.value },
  );
  return { ...actual, useStoreChatModelSettings: useSettings };
});

vi.mock("@/hooks/useSmartScroll", () => ({
  useSmartScroll: () => ({
    containerRef: { current: null },
    isAutoScrollToBottom: true,
    autoScrollToBottom: vi.fn(),
  }),
}));

vi.mock("@/services/settings/ui-settings", () => ({
  CHAT_BACKGROUND_IMAGE_SETTING: "chatBackgroundImage",
  CHAT_WINDOW_OPACITY_SETTING: "chatWindowOpacity",
  CHAT_MESSAGE_OPACITY_SETTING: "chatMessageOpacity",
  CHAT_CHARACTER_IMAGE_OPACITY_SETTING: "chatCharacterImageOpacity",
  resolveOpacityAlpha: (value: unknown, fallback = 35) =>
    typeof value === "number" && Number.isFinite(value)
      ? value / 100
      : fallback / 100,
  THEME_SETTING: {
    key: "theme",
    defaultValue: "dark"
  },
  HEADER_SHORTCUT_IDS: [],
  SIDEBAR_SHORTCUT_IDS: [],
}));

vi.mock("../Knowledge/utils/unsupported-types", () => ({
  otherUnsupportedTypes: [],
}));

// The selected option/action fixture remains an I/O boundary; synchronous
// route guards and render selectors must read the same current selection.
vi.mock("@/store/option", () => {
  const getState = () => ({
    ...messageOptionState.value,
    compareParentByHistory: {} as Record<string, never>,
  });
  const useOptions = (
    selector?: (state: ReturnType<typeof getState>) => unknown,
  ) => (typeof selector === "function" ? selector(getState()) : getState());
  return { useStoreMessageOption: Object.assign(useOptions, { getState }) };
});

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
  useSetting: (setting: string) => {
    if (setting === "chatWindowOpacity") return [35];
    if (setting === "chatMessageOpacity") return [60];
    if (setting === "chatCharacterImageOpacity") return [100];
    return [""];
  },
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
  useDesktop: () => true,
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

// Mount the deployed shared catalog hook against a real query client. The
// global no-provider fallback represents unknown discovery and never runs IO.
let queryClient: QueryClient;
const renderPlayground = () => render(<Playground />, {
  wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>,
});
afterEach(() => { cleanup(); queryClient.clear(); });

describe("Playground cockpit controls", () => {
  beforeEach(() => {
    Object.assign(connectionState, { isConnected: true, phase: "connected" });
    queryClient = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
    useStoreChatModelSettings.setState(useStoreChatModelSettings.getInitialState(), true);
    useActorStore.getState().reset();
    usePlaygroundSessionStore.getState().clearSession();
    forkSettings.controller = null; forkSettings.mode = "ordinary"; forkSettings.current = true; forkSettings.update.mockReset(); forkSettings.update.mockResolvedValue({});
    modelSettingsState.useRealStore = false;
    messageOptionState.value.chatDraftAddress = null;
    messageOptionState.value.chatDraftRevision = 0;
    storageState.values.clear();
    storageState.values.set("playgroundChatContextRailVisible", true);
    storageState.values.set("playgroundChatRuntimeRailVisible", true);
    layoutShellOverrideState.overrides = [];
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
      metadata: { selectionMode: "tracked" },
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
    messageOptionState.value.setSelectedAssistant = vi.fn(async (value, options) => {
      if (options?.isCurrent?.() === false) return false;
      messageOptionState.value.setSelectedCharacter(value);
      return true;
    });
    messageOptionState.value.setSelectedCharacter = vi.fn();
    sessionPersistenceState.value.clearPersistedSession = vi.fn();
    chatSettingsState.syncChatSettingsForServerChat.mockClear();
    chatSettingsState.applyChatSettingsPatch.mockClear();
    tldwServerState.fetchChatModels.mockReset();
    tldwServerState.fetchChatModels.mockResolvedValue([
      {
        model: "openai:gpt-4.1-mini",
        provider: "openai",
        is_configured: true,
        provider_is_configured: true,
      },
    ]);
    tldwClientState.initialize.mockClear();
    tldwClientState.getProvidersStatus.mockReset();
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

  it.each(["different-chat", "same-id-new-draft"])("refuses a retained settings callback after %s retirement", async retirement => {
    messageOptionState.value.historyId = "original-local";
    messageOptionState.value.serverChatId = "original-chat";
    forkSettings.controller = { settingsMode: () => "ordinary", fence: forkSettings.fence };
    const { result } = renderHook(() => useChatSettingsRecord({ historyId: "original-local", serverChatId: "original-chat" }));
    const retained = result.current.updateSettings;
    if (retirement === "different-chat") {
      messageOptionState.value.historyId = "new-local";
      messageOptionState.value.serverChatId = "new-chat";
    } else messageOptionState.value.chatDraftRevision++;
    await expect(retained({ assistantOverlay: null })).rejects.toThrow("chat_settings_owner_unavailable");
    expect(chatSettingsState.applyChatSettingsPatch).not.toHaveBeenCalled();
    expect(forkSettings.update).not.toHaveBeenCalled();
    expect(messageOptionState.value.setSelectedAssistant).not.toHaveBeenCalled();
  });

  it("keeps a freshly entered prompt through the real Playground scope effect and dialog remount without an explicit provider", async () => {
    modelSettingsState.useRealStore = true;
    useStoreChatModelSettings.getState().reset();
    const selectedModel =
      "tldw:../../../Working/Language_Models/gemma-4-26B-A4B/gemma-4-26B-A4B-it-ultra-uncensored-heretic-Q4_K_M.gguf";
    messageOptionState.value.selectedModel = selectedModel;
    messageOptionState.value.selectedCharacter = null;
    messageOptionState.value.selectedAssistant = null;
    messageOptionState.value.messages = [];
    messageOptionState.value.historyId = null;
    messageOptionState.value.serverChatId = null;
    messageOptionState.value.chatDraftAddress = { kind: "fresh", id: "settings-draft", scope: { type: "global" }, temporary: false };
    forkSettings.controller = {
      settingsMode: () => "ordinary", fence: forkSettings.fence,
      getCurrent: () => ({ status: "idle", owner: null, view: null }),
      getSignal: () => undefined,
      draftAccount: {
        requestScope: { config: { serverUrl: "https://settings.test", authMode: "single-user" }, userId: 1 },
        isCurrent: () => forkSettings.current
      }
    };
    messageOptionState.value.streaming = false;
    tldwServerState.fetchChatModels.mockResolvedValue([
      { model: selectedModel, provider: "llama.cpp" },
    ]);
    render(
      <QueryClientProvider client={queryClient}>
        <Playground />
      </QueryClientProvider>,
    );
    await waitFor(() =>
      expect(useStoreChatModelSettings.getState().activeSettingsScope).toBe(
        selectedModel,
      ),
    );
    expect(useStoreChatModelSettings.getState().apiProvider).toBeUndefined();
    const settingsButton = screen.getByRole("button", {
      name: "Current Chat Model Settings",
    });
    fireEvent.click(settingsButton);
    fireEvent.click(
      await screen.findByRole("tab", { name: "Conversation" }),
    );
    const instruction =
      "UAT367 NATIVE ORBIT742. Describe visible image contents accurately in one short sentence.";
    fireEvent.change(
      await screen.findByPlaceholderText("Enter System Prompt"),
      { target: { value: instruction } },
    );
    fireEvent.click(
      within(screen.getByRole("dialog")).getByRole("button", { name: /^save$/i }),
    );
    await waitFor(() =>
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument(),
    );
    expect(useStoreChatModelSettings.getState().systemPrompt).toBe(instruction);
    fireEvent.click(settingsButton);
    fireEvent.click(
      await screen.findByRole("tab", { name: "Conversation" }),
    );
    expect(
      await screen.findByPlaceholderText("Enter System Prompt"),
    ).toHaveValue(instruction);
  }, 15_000);

  it("surfaces existing context and runtime state in cockpit rails", async () => {
    messageOptionState.value.selectedQuickPrompt = "Draft a concise summary";
    modelSettingsState.value.systemPrompt = "Be curious.";
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
      renderPlayground();

      const leftRail = await screen.findByTestId("playground-cockpit-left-rail");
      const contextRail = within(leftRail).getByTestId("playground-context-rail");
      const compositionPreview = within(contextRail).getByRole("region", {
        name: "Next message composition",
      });
      expect(within(compositionPreview).getByText("Inline system prompt selected")).toBeInTheDocument();
      expect(within(compositionPreview).getByText("Mira Vale")).toBeInTheDocument();
      // The model reads "model (Provider)", not the raw route (CC-06).
      expect(
        within(compositionPreview).getAllByText("gpt-4.1-mini (OpenAI)").length,
      ).toBeGreaterThan(0);
      expect(within(compositionPreview).getAllByText("MCP tools").length).toBeGreaterThan(0);
      expect(
        within(compositionPreview).getByText("Scope: gpt-4.1-mini (OpenAI)"),
      ).toBeInTheDocument();
      expect(within(contextRail).getByText("Context selected")).toBeInTheDocument();
      expect(within(contextRail).getByText("1 file")).toBeInTheDocument();
      expect(
        within(contextRail).getByText("1 knowledge item"),
      ).toBeInTheDocument();
      expect(within(contextRail).getByText("2 media scopes")).toBeInTheDocument();
      expect(
        screen.queryByRole("status", { name: "Chat status" }),
      ).not.toBeInTheDocument();
      expect(within(contextRail).getByText("Temporary chat")).toBeInTheDocument();
      expect(within(contextRail).getByText("History linked")).toBeInTheDocument();
      expect(
        within(contextRail).getByRole("heading", { name: "Prompt" }),
      ).toBeInTheDocument();
      expect(
        within(contextRail).queryByText("Quick prompt"),
      ).not.toBeInTheDocument();
      expect(
        within(contextRail).queryByText("Draft a concise summary"),
      ).not.toBeInTheDocument();
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
      expect(within(runtimeInspector).getByText("OpenAI")).toBeInTheDocument();
      expect(within(runtimeInspector).getByText("Model")).toBeInTheDocument();
      expect(within(runtimeInspector).getByText("gpt-4.1-mini")).toBeInTheDocument();
      // The raw route stays available under Details (CC-06).
      expect(
        within(runtimeInspector).getByText("openai:gpt-4.1-mini").closest("details"),
      ).not.toBeNull();
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

  it("surfaces active web search state in the cockpit context rail", async () => {
    messageOptionState.value.streaming = false;
    messageOptionState.value.isSearchingInternet = true;
    messageOptionState.value.selectedAssistant = null;
    messageOptionState.value.selectedCharacter = null;

    renderPlayground();

    const contextRail = within(
      await screen.findByTestId("playground-cockpit-left-rail"),
    ).getByTestId("playground-context-rail");
    expect(within(contextRail).getByText("Web search on")).toBeInTheDocument();
    expect(
      screen.queryByRole("status", { name: "Chat status" }),
    ).not.toBeInTheDocument();
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
      renderPlayground();

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
    modelSettingsState.value.systemPrompt = "Be curious.";
    messageOptionState.value.selectedKnowledge = [
      { id: "knowledge-1", title: "Research notes" },
      { id: "knowledge-2", title: "Protocol notes" },
    ];

    renderPlayground();

    const contextRail = within(
      await screen.findByTestId("playground-cockpit-left-rail"),
    ).getByTestId("playground-context-rail");
    const sourceList = within(contextRail).getByRole("list", {
      name: "Context sources",
    });

    const promptSource = within(sourceList).getByText("Custom prompt").closest("li");
    expect(promptSource).not.toBeNull();
    expect(within(promptSource as HTMLElement).getByText("Prompt")).toBeInTheDocument();
    expect(
      within(promptSource as HTMLElement).getByText("Inline system prompt selected"),
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
      expect(messageOptionState.value.setSelectedAssistant).toHaveBeenCalledWith(null, expect.objectContaining({ isCurrent: expect.any(Function) }));
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

    renderPlayground();

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
      renderPlayground();

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
      metadata: { selectionMode: "tracked" },
    };
    messageOptionState.value.selectedCharacter = {
      id: "legacy-character",
      name: "Legacy Character",
    };
    messageOptionState.value.serverChatPersonaMemoryMode = "read_write";

    renderPlayground();

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
      renderPlayground();

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
          applyAs: "tracked",
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
      metadata: { selectionMode: "tracked" },
    };
    messageOptionState.value.selectedCharacter = {
      id: "legacy-character",
      name: "Legacy Character",
    };
    const openAssistantSelect = vi.fn();
    window.addEventListener(OPEN_ASSISTANT_SELECT_EVENT, openAssistantSelect);

    try {
      renderPlayground();

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
          applyAs: "tracked",
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
      renderPlayground();

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
          applyAs: "tracked",
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
      metadata: { selectionMode: "tracked" },
    };
    messageOptionState.value.selectedCharacter = {
      id: "legacy-character",
      name: "Legacy Character",
    };

    const { rerender } = renderPlayground();

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
      expect(messageOptionState.value.setSelectedAssistant).toHaveBeenCalledWith(null, expect.objectContaining({ isCurrent: expect.any(Function) }));
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

    renderPlayground();

    const runtimeInspector = within(
      await screen.findByTestId("playground-cockpit-right-rail"),
    ).getByTestId("playground-runtime-inspector");
    fireEvent.click(
      within(runtimeInspector).getByRole("button", {
        name: "Clear assistant",
      }),
    );

    await waitFor(() => {
      expect(messageOptionState.value.setSelectedAssistant).toHaveBeenCalledWith(null, expect.objectContaining({ isCurrent: expect.any(Function) }));
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

    renderPlayground();

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
    expect(messageOptionState.value.setSelectedAssistant).toHaveBeenCalledWith(null, expect.objectContaining({ isCurrent: expect.any(Function) }));
    expect(messageOptionState.value.setSelectedCharacter).toHaveBeenCalledWith(null);
  });

  it.each(["success", "scope-change", "failed", "pending"])("clears a copied child's overlay only through its current scoped settings action: %s", async outcome => {
    forkSettings.mode = outcome === "pending" ? "pending" : "fork";
    messageOptionState.value.streaming = false;
    messageOptionState.value.historyId = null;
    messageOptionState.value.serverChatId = "fork-child";
    messageOptionState.value.selectedAssistant = {kind: "persona", id: "overlay", name: "Overlay", metadata: {selectionMode: "overlay"}} as any;
    messageOptionState.value.selectedCharacter = null;
    let finish!: () => void;
    forkSettings.update.mockImplementation(() => new Promise((resolve, reject) => { finish = () => outcome === "failed" ? reject(new Error("lost response")) : resolve({}); }));
    renderPlayground();
    const inspector = within(await screen.findByTestId("playground-cockpit-right-rail")).getByTestId("playground-runtime-inspector");
    fireEvent.click(within(inspector).getByRole("button", {name: "Clear assistant"}));
    if (outcome !== "pending") {
      await waitFor(() => expect(forkSettings.update).toHaveBeenCalledWith({assistantOverlay: null}));
      expect(messageOptionState.value.setServerChatId).not.toHaveBeenCalled();
      if (outcome === "scope-change") forkSettings.current = false;
      await act(async () => finish());
    } else await act(async () => {});
    expect(chatSettingsState.applyChatSettingsPatch).not.toHaveBeenCalled();
    if (outcome === "success") expect(messageOptionState.value.setServerChatId).toHaveBeenCalledWith(null);
    else {
      expect(messageOptionState.value.setServerChatId).not.toHaveBeenCalled();
      expect(messageOptionState.value.setSelectedAssistant).not.toHaveBeenCalled();
      expect(sessionPersistenceState.value.clearPersistedSession).not.toHaveBeenCalled();
    }
  });

  it("clears an ordinary legacy chat assistant through the real qualified controller before ancestry review", async () => {
    messageOptionState.value.streaming = false;
    messageOptionState.value.historyId = null;
    messageOptionState.value.serverChatId = "legacy";
    messageOptionState.value.selectedAssistant = {kind: "persona", id: "overlay", name: "Overlay", metadata: {selectionMode: "overlay"}} as any;
    messageOptionState.value.selectedCharacter = null;
    const actual = renderHook(() => useHistorySelection());
    await act(async () => {await actual.result.current.open({kind: "native", owner_key: "owner", conversation_id: "legacy", validate_lease: () => true} as any)});
    forkSettings.controller = actual.result.current;
    renderPlayground();
    const inspector = within(await screen.findByTestId("playground-cockpit-right-rail")).getByTestId("playground-runtime-inspector");
    fireEvent.click(within(inspector).getByRole("button", {name: "Clear assistant"}));
    await waitFor(() => expect(messageOptionState.value.setServerChatId).toHaveBeenCalledWith(null));
    expect(chatSettingsState.applyChatSettingsPatch).toHaveBeenCalledWith({historyId: null, serverChatId: "legacy", patch: {assistantOverlay: null}});
    expect(actual.result.current.status).toBe("legacy_review_required");
  });

  it("reveals the existing conversation shell for an explicit Find event in Focus", async () => {
    storageState.values.set("playgroundChatLayoutMode", "focus")
    const originalMessages = structuredClone(messageOptionState.value.messages)
    renderPlayground()
    expect(await screen.findByTestId("playground-cockpit-shell")).toHaveAttribute(
      "data-mode",
      "focus"
    )
    act(() => window.dispatchEvent(new Event("tldw:open-chat-sidebar")))
    await waitFor(() =>
      expect(screen.getByTestId("playground-cockpit-shell")).toHaveAttribute(
        "data-mode",
        "cockpit"
      )
    )
    expect(layoutShellOverrideState.overrides.at(-1)).toBeNull()
    expect(messageOptionState.value.messages).toEqual(originalMessages)
    expect(messageOptionState.value.setServerChatId).not.toHaveBeenCalled()
    expect(messageOptionState.value.setHistoryId).not.toHaveBeenCalled()
  })

  it("does not render cockpit control rails in focus mode", async () => {
    storageState.values.set("playgroundChatLayoutMode", "focus");

    renderPlayground();

    expect(
      await screen.findByTestId("playground-cockpit-shell"),
    ).toHaveAttribute("data-mode", "focus");
    expect(screen.queryByTestId("playground-context-rail")).toBeNull();
    expect(screen.queryByTestId("playground-runtime-inspector")).toBeNull();
    expect(
      screen.queryByRole("region", { name: "Next message composition" }),
    ).toBeNull();
    await waitFor(() => {
      expect(layoutShellOverrideState.overrides).toContainEqual({
        hideHeader: true,
        hideSidebar: true,
      });
    });
    expect(
      screen.getByTestId("playground-chat-layout-mode-trigger"),
    ).toHaveAccessibleName("Exit focus");
    expect(
      screen.queryByTestId("playground-shortcuts-help-trigger"),
    ).toBeNull();
    const exitFocus = screen.getByTestId("playground-chat-layout-mode-trigger");
    expect(exitFocus).toHaveTextContent("Exit focus");
    expect(screen.getByTestId("playground-chat")).toBeInTheDocument();
    expect(screen.getByTestId("playground-form")).toBeInTheDocument();

    fireEvent.click(exitFocus);

    await waitFor(() => {
      expect(storageState.values.get("playgroundChatLayoutMode")).toBe(
        "cockpit",
      );
      expect(
        layoutShellOverrideState.overrides[
          layoutShellOverrideState.overrides.length - 1
        ],
      ).toBeNull();
    });
  });

  it("wires ready-state runtime regenerate to the shared chat handler", async () => {
    messageOptionState.value.streaming = false;

    renderPlayground();

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

  it("disables cockpit regenerate when a saved conversation's reply cannot be regenerated", async () => {
    messageOptionState.value.streaming = false;
    messageOptionState.value.temporaryChat = false;
    const selection = renderHook(() => useHistorySelection());
    await act(async () => {
      await selection.result.current.open({
        kind: "native",
        owner_key: "owner",
        conversation_id: "chat-1",
        validate_lease: () => true,
      } as any);
    });
    forkSettings.controller = selection.result.current;

    renderPlayground();

    const inspector = within(
      await screen.findByTestId("playground-cockpit-right-rail"),
    ).getByTestId("playground-runtime-inspector");
    expect(
      within(inspector).getByRole("button", { name: "Regenerate last response" }),
    ).toBeDisabled();
    // CM-01: only a saved reply the server cannot regenerate disables it, with a plain reason.
    expect(within(inspector).getByText(/can't be regenerated here.*New Branch/i)).toBeInTheDocument();
    expect(messageOptionState.value.regenerateLastMessage).not.toHaveBeenCalled();
  });

  it("requires connection before model setup even before a health event arrives", async () => {
    Object.assign(connectionState, { isConnected: false, phase: "unconfigured" });
    messageOptionState.value.selectedAssistant = { kind: "character", id: "character-1", name: "Mira Vale", metadata: { selectionMode: "tracked" } };
    messageOptionState.value.selectedModel = null;
    messageOptionState.value.serverChatId = null;
    messageOptionState.value.historyId = null;
    messageOptionState.value.streaming = false;
    const setup = vi.fn((event: Event) => event.preventDefault());
    const modelSettings = vi.fn();
    window.addEventListener("tldw:settings-navigation-request", setup);
    window.addEventListener("tldw:open-model-settings", modelSettings);
    try {
      renderPlayground();
      const panel = await screen.findByTestId("character-chat-readiness-panel");
      fireEvent.click(within(panel).getByRole("button", { name: "Open server settings" }));
      expect(setup).toHaveBeenCalledExactlyOnceWith(expect.objectContaining({ detail: { destination: "/settings/tldw" } }));
      expect(modelSettings).not.toHaveBeenCalled();
    } finally {
      window.removeEventListener("tldw:settings-navigation-request", setup);
      window.removeEventListener("tldw:open-model-settings", modelSettings);
    }
  });

  it("reflects degraded server readiness in the cockpit runtime rail", async () => {
    messageOptionState.value.streaming = false;
    messageOptionState.value.selectedAssistant = null;
    messageOptionState.value.selectedCharacter = null;

    renderPlayground();

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
    expect(
      await within(runtimeInspector).findByText("Degraded"),
    ).toBeInTheDocument();
    expect(
      within(runtimeInspector).getByText("Degraded: chacha_notes"),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("status", { name: "Chat status" }),
    ).not.toBeInTheDocument();
  });

  it("keeps streaming primary when degraded readiness is warning-only", async () => {
    messageOptionState.value.streaming = true;
    messageOptionState.value.selectedAssistant = null;
    messageOptionState.value.selectedCharacter = null;

    renderPlayground();

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

    expect(
      screen.queryByRole("status", { name: "Chat status" }),
    ).not.toBeInTheDocument();
  });

  it("surfaces blocked server readiness as chat-critical unavailable state", async () => {
    messageOptionState.value.streaming = false;
    messageOptionState.value.selectedAssistant = null;
    messageOptionState.value.selectedCharacter = null;

    renderPlayground();

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

    expect(
      screen.queryByRole("status", { name: "Chat status" }),
    ).not.toBeInTheDocument();
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
    tldwServerState.fetchChatModels.mockResolvedValue([
      {
        id: "gpt-4o",
        model: "tldw:gpt-4o",
        name: "GPT-4o",
        provider: "openai",
        type: "chat",
        is_configured: false,
        provider_is_configured: false,
      },
    ]);
    tldwClientState.getProvidersStatus.mockResolvedValue({
      providers: [
        {
          name: "openai",
          configured: false,
          requires_api_key: true,
        },
      ],
      any_configured: false,
    });

    renderPlayground();

    const leftRail = await screen.findByTestId("playground-cockpit-left-rail");
    expect(leftRail).toBeInTheDocument();
    expect(screen.getByTestId("playground-chat")).toBeInTheDocument();
    const rightRail = await screen.findByTestId(
      "playground-cockpit-right-rail",
    );

    const runtimeInspector = within(rightRail).getByTestId(
      "playground-runtime-inspector",
    );

    await waitFor(() => {
      expect(within(runtimeInspector).getByText("Error")).toBeInTheDocument();
    });
    expect(runtimeInspector).toHaveTextContent("Provider setup needed");
    expect(tldwServerState.fetchChatModels).toHaveBeenCalledWith({
      requireFresh: true,
    });
    expect(
      within(leftRail).getByRole("button", { name: "Expand Prompt" }),
    ).toHaveAttribute("aria-expanded", "false");
    expect(
      within(rightRail).getByRole("button", { name: "Expand Assistant" }),
    ).toHaveAttribute("aria-expanded", "false");

    expect(
      screen.queryByRole("status", { name: "Chat status" }),
    ).not.toBeInTheDocument();
  });

  it("keeps the cockpit visibly degraded when readiness details are empty", async () => {
    messageOptionState.value.streaming = false;
    messageOptionState.value.selectedAssistant = null;
    messageOptionState.value.selectedCharacter = null;

    renderPlayground();

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
    expect(
      await within(runtimeInspector).findByText("Degraded"),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("status", { name: "Chat status" }),
    ).not.toBeInTheDocument();
  });

  it("exposes actual mobile disclosure state while close preserves desktop preferences and the draft", () => {
    const onLeftRailVisibleChange = vi.fn();
    const onRightRailVisibleChange = vi.fn();
    render(
      <PlaygroundCockpitShell
        mode="cockpit"
        onLeftRailVisibleChange={onLeftRailVisibleChange}
        onRightRailVisibleChange={onRightRailVisibleChange}
        leftRail={<div>Context panel</div>}
        rightRail={<div>Runtime panel</div>}
      >
        <textarea aria-label="Mobile message" defaultValue="Retained message" />
      </PlaygroundCockpitShell>,
    );

    const mobileRails = screen.getByTestId("playground-cockpit-mobile-rails");
    const rails = within(mobileRails);
    const context = rails.getByRole("button", { name: "Context" });
    const runtime = rails.getByRole("button", { name: "Chat settings" });
    const draft = screen.getByRole("textbox", { name: "Mobile message" });
    expect(rails.getAllByRole("button")).toHaveLength(2);
    expect(context).toHaveAttribute("aria-expanded", "true");
    expect(runtime).toHaveAttribute("aria-expanded", "false");
    for (const button of [context, runtime]) {
      const panel = document.getElementById(
        button.getAttribute("aria-controls") ?? "",
      );
      expect(panel).toHaveAttribute("aria-labelledby", button.id);
      expect(button).not.toHaveAttribute("aria-pressed");
    }
    expect(rails.getByRole("region", { name: "Context" })).toBeVisible();

    fireEvent.click(context);
    expect(mobileRails).toHaveAttribute("data-mobile-panel", "none");
    expect(context).toHaveAttribute("aria-expanded", "false");
    expect(runtime).toHaveAttribute("aria-expanded", "false");
    expect(rails.queryAllByRole("region")).toHaveLength(0);

    fireEvent.click(runtime);
    expect(rails.getByRole("region", { name: "Chat settings" })).toBeVisible();
    expect(runtime).toHaveAttribute("aria-expanded", "true");
    fireEvent.click(runtime);
    expect(mobileRails).toHaveAttribute("data-mobile-panel", "none");
    expect(runtime).toHaveAttribute("aria-expanded", "false");
    expect(onLeftRailVisibleChange).not.toHaveBeenCalled();
    expect(onRightRailVisibleChange).not.toHaveBeenCalled();
    expect(
      screen.getByRole("complementary", { name: "Chat cockpit context" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("complementary", { name: "Chat cockpit runtime" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: "Mobile message" })).toBe(draft);
    expect(draft).toHaveValue("Retained message");
  });
});
