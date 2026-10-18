// @vitest-environment jsdom
import React from "react";
import {
  act,
  fireEvent,
  render as renderTestingLibrary,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { historySelectionExpansionPath } from "@/hooks/chat/useHistorySelection";
import type { HistoryBookmark } from "@/db/dexie/types";

import { Playground } from "../Playground";
import { PlaygroundCockpitShell } from "../PlaygroundCockpitShell";
import { usePlaygroundSessionStore } from "@/store/playground-session";
import { useStoreMessageOption } from "@/store/option";
import { resolveEffectiveAssistantState } from "@/hooks/chat/effective-assistant-state";
import { captureChatDraftLoad } from "@/hooks/chat/chat-draft-transition";
import { characterToAssistantSelection } from "@/types/assistant-selection";
import { resolveHistorySelection } from "@/utils/history-selection";

// Keep the real query lifecycle: the shared no-provider fallback intentionally
// does not execute catalog reads and cannot prove root readiness outcomes.
const render = (ui: React.ReactElement) => {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  function QueryFixture({ children }: { children: React.ReactNode }) {
    return (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    );
  }
  return renderTestingLibrary(ui, { wrapper: QueryFixture });
};

const messageOptionState = vi.hoisted(() => ({
  value: {
    messages: [],
    history: [],
    historyId: null as string | null,
    serverChatId: null as string | null,
    isLoading: false,
    setHistoryId: vi.fn(),
    setHistory: vi.fn(),
    setMessages: vi.fn(),
    regenerateLastMessage: vi.fn(),
    selectedSystemPrompt: "" as string | null,
    setSelectedSystemPrompt: vi.fn(),
    selectedQuickPrompt: null as string | null,
    setSelectedQuickPrompt: vi.fn(),
    setSelectedModel: vi.fn(),
    setServerChatId: vi.fn(),
    serverChatCharacterId: null as string | number | null,
    setServerChatCharacterId: vi.fn(),
    serverChatAssistantKind: null as "character" | "persona" | null,
    setServerChatAssistantKind: vi.fn(),
    serverChatAssistantId: null as string | null,
    setServerChatAssistantId: vi.fn(),
    contextFiles: [] as Array<{ id: string; name: string }>,
    setContextFiles: vi.fn(),
    createChatBranch: vi.fn(),
    streaming: false,
    selectedModel: "openai:gpt-4.1-mini",
    selectedCharacter: null,
    setSelectedCharacter: vi.fn(),
    assistantSelectionMeta: { isLoading: false, assistantKey: "account-a", isCurrent: () => true },
    selectedAssistant: null as {
      kind: "character" | "persona";
      id: string;
      name: string;
      metadata?: Record<string, unknown>;
    } | null,
    serverChatPersonaMemoryMode: null as "read_only" | "read_write" | null,
    serverChatMetaLoaded: false,
    setServerChatPersonaMemoryMode: vi.fn(),
    setServerChatMetaLoaded: vi.fn(),
    setSelectedAssistant: vi.fn(),
    compareMode: false,
    compareFeatureEnabled: false,
    temporaryChat: false,
    webSearch: false,
    selectedKnowledge: [],
    ragMediaIds: [],
  },
}));

const sessionPersistenceState = vi.hoisted(() => ({
  value: {
    restoreSession: vi.fn(async (): Promise<"restored" | "not-restored" | "cancelled"> => "not-restored"),
    clearPersistedSession: vi.fn(async () => undefined),
    sessionScopeReady: true,
    hasPersistedSession: false,
    hasPersistedDraftHandle: false,
    hasPersistedFreshDraft: false,
    persistedHistoryId: null as string | null,
    persistedServerChatId: null as string | null,
  },
}));

const storageState = vi.hoisted(() => ({
  values: new Map<string, unknown>(),
}));
const connectionState = vi.hoisted(() => ({ isConnected: true, phase: "connected" }))
vi.mock("@/hooks/useConnectionState", () => ({ useConnectionState: () => connectionState }))
const recentChatState = vi.hoisted(() => ({
  enabled: false,
  recent: vi.fn(),
  capture: vi.fn(),
  bookmark: vi.fn(),
}));
vi.mock("@/db/dexie/chat", () => ({
  PageAssistDatabase: class {
    getHistoryInfo = async (id: string) => ({ id, local_owner_key: "local-owner" });
  }
}));
vi.mock("@/db/dexie/history-selection", () => ({
  ensureLocalProfileId: async () => "profile",
  getLocalHistoryOwner: async (id: string) => ({ kind: "local", profile_id: "profile", owner_key: "local-owner", conversation_id: id }),
  loadHistoryTurnRecoveries: async () => [],
  dismissHistoryTurnRecovery: async () => {},
  loadHistoryBookmark: (...args: unknown[]) => recentChatState.bookmark(...args),
  saveHistoryBookmark: async () => {}
}));
vi.mock("@/db/dexie/fork-operations", () => ({ findForkCandidate: async () => null, loadForkOperations: async () => [] }));
vi.mock("@/services/chat-history-selection", () => ({ captureHistorySnapshot: (...args: unknown[]) => recentChatState.capture(...args) }));

const darkModeState = vi.hoisted(() => ({
  mode: "dark" as "system" | "dark" | "light",
  toggleDarkMode: vi.fn(),
}));

const chatSettingsState = vi.hoisted(() => ({
  syncChatSettingsForServerChat: vi.fn(async (_params: unknown) => null),
}));

const cockpitChatRenderState = vi.hoisted(() => ({
  starterDeckSignals: [] as Array<boolean | undefined>,
}));

const tldwClientState = vi.hoisted(() => ({
  getCharacter: vi.fn(async (id: string | number) => ({
    id,
    name: "Route Character",
  })),
  initialize: vi.fn(async () => null),
  getProvidersStatus: vi.fn(async () => null),
  getResearchBundle: vi.fn(async () => null),
}));

const tldwServerState = vi.hoisted(() => ({
  fetchChatModels: vi.fn(async () => [
    {
      model: "openai:gpt-4.1-mini",
      provider: "openai",
      is_configured: true,
      provider_is_configured: true,
    },
  ]),
}));

const characterSessionsPanelState = vi.hoisted(() => ({
  props: [] as Array<Record<string, unknown>>,
}));

const routerLocationState = vi.hoisted(() => ({
  value: null as {
    pathname?: string;
    search?: string;
    hash?: string;
    key?: string;
  } | null,
}));

const routerNavigateState = vi.hoisted(() => ({
  navigate: vi.fn(),
}));

vi.mock("@/services/service-prompts", async importOriginal => ({
  ...await importOriginal<typeof import("@/services/service-prompts")>(),
  resolveServicePromptScope: async () => ({ config: { serverUrl: "http://chat.test", authMode: "multi-user" }, userId: "account-a" }),
  loadServicePromptSnapshot: async () => ({ requestScope: { config: { serverUrl: "http://chat.test", authMode: "multi-user" }, userId: "account-a" }, scopeSignal: new AbortController().signal, scopeInvalidatedSignal: new AbortController().signal, release: vi.fn() })
}));

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?:
        | string
        | { defaultValue?: string; [key: string]: unknown },
    ) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions;
      const template = fallbackOrOptions?.defaultValue || key;
      return template.replace(/\{\{(\w+)\}\}/g, (_, token: string) => {
        const value = fallbackOrOptions?.[token];
        return value == null ? `{{${token}}}` : String(value);
      });
    },
  }),
}));

// The root fixture stubs the Form acceptance boundary; actual preflight/retention
// behavior is exercised in PlaygroundForm.role-play-starter.integration.test.tsx.
const routeApplyOutcome = vi.hoisted(() => ({ value: null as string | null }))
vi.mock("@/components/Option/Playground/PlaygroundForm", () => ({
  PlaygroundForm: ({ onDraftPresenceChange, onCharacterRouteApplyReady }: React.ComponentProps<typeof import("../PlaygroundForm").PlaygroundForm>) => {
    React.useEffect(() => {
      onCharacterRouteApplyReady?.(async (id: string, isCurrent: () => boolean) => {
        const character = await tldwClientState.getCharacter(id, { requestScope: { config: { serverUrl: "http://chat.test", authMode: "multi-user" }, userId: "account-a" } })
        if (!isCurrent()) return "retired"
        if (routeApplyOutcome.value) return routeApplyOutcome.value
        if (!character) return "missing"
        await messageOptionState.value.setSelectedCharacter(character)
        if (!isCurrent()) return "retired"
        messageOptionState.value.setHistoryId(null, { preserveServerChatId: false })
        messageOptionState.value.setHistory([])
        messageOptionState.value.setMessages([])
        messageOptionState.value.setServerChatId(null)
        return "applied"
      })
      return () => onCharacterRouteApplyReady?.(null)
    }, [onCharacterRouteApplyReady])
    return <div data-testid="playground-form"><textarea aria-label="Composer draft" data-testid="composer-textarea"
      onChange={event => onDraftPresenceChange?.(event.currentTarget.value.trim().length > 0)} /></div>
  },
}));

vi.mock("@/components/Option/Playground/PlaygroundChat", () => ({
  PlaygroundChat: ({ showStarterDeck }: { showStarterDeck?: boolean }) => {
    cockpitChatRenderState.starterDeckSignals.push(showStarterDeck);
    const legacyWouldShowStarterDeck =
      messageOptionState.value.messages.length === 0;

    return (
      <div data-testid="playground-chat">
        {(showStarterDeck ?? legacyWouldShowStarterDeck) && (
          <div data-testid="playground-empty-mode-deck" />
        )}
      </div>
    );
  },
}));

vi.mock("@/components/Sidepanel/Chat/ArtifactsPanel", () => ({
  ArtifactsPanel: () => null,
}));

vi.mock("@/hooks/useMessageOption", () => ({
  useMessageOption: () => {
    // The real hook derives selectedCharacter from the unified assistant. Older
    // cases use the character shorthand to arrange a tracked draft selection.
    const character = characterToAssistantSelection(messageOptionState.value.selectedCharacter);
    const selectedAssistant = messageOptionState.value.selectedAssistant ?? (character && {
      ...character, metadata: { ...character.metadata, selectionMode: "tracked" }
    });
    return { ...messageOptionState.value, selectedAssistant, effectiveAssistantState: resolveEffectiveAssistantState({
      tracked: { assistantKind: messageOptionState.value.serverChatAssistantKind, assistantId: messageOptionState.value.serverChatAssistantId, characterId: messageOptionState.value.serverChatCharacterId },
      draftSelection: selectedAssistant
    }) };
  },
}));

vi.mock("@/hooks/usePlaygroundSessionPersistence", () => ({
  usePlaygroundSessionPersistence: () => sessionPersistenceState.value,
}));


vi.mock("@/services/app", () => ({
  webUIResumeLastChat: vi.fn(async () => recentChatState.enabled),
}));

vi.mock("@/services/chat-settings", async importOriginal => ({
  ...await importOriginal<typeof import("@/services/chat-settings")>(),
  syncChatSettingsForServerChat: (params: unknown) =>
    chatSettingsState.syncChatSettingsForServerChat(params),
}));

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: tldwClientState,
}));

vi.mock("@/services/tldw-server", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/services/tldw-server")>()
  return {
    ...actual,
    fetchChatModels: tldwServerState.fetchChatModels
  }
})

vi.mock("@/components/Option/Playground/CharacterChatSessionsPanel", () => ({
  CharacterChatSessionsPanel: (props: Record<string, unknown>) => {
    characterSessionsPanelState.props.push(props);
    return <section data-testid="character-chat-sessions-panel" />;
  },
}));

vi.mock("@/db/dexie/helpers", async importOriginal => ({
  ...await importOriginal<typeof import("@/db/dexie/helpers")>(),
  formatToChatHistory: vi.fn(),
  formatToMessage: vi.fn(),
  getHistoryByServerChatId: vi.fn(async () => null),
  getPromptById: vi.fn(async () => null),
  getRecentChatFromWebUI: (...args: unknown[]) => recentChatState.recent(...args),
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
  HEADER_SHORTCUTS_EXPANDED_SETTING: "headerShortcutsExpanded",
  HEADER_SHORTCUTS_LAUNCHER_VIEW_SETTING: "headerShortcutsLauncherView",
  HEADER_SHORTCUT_SELECTION_SETTING: "headerShortcutSelection",
  HEADER_SHORTCUT_IDS: [],
  SIDEBAR_SHORTCUT_IDS: [],
  CHAT_WINDOW_OPACITY_SETTING: "chatWindowOpacity",
  CHAT_MESSAGE_OPACITY_SETTING: "chatMessageOpacity",
  CHAT_CHARACTER_IMAGE_OPACITY_SETTING: "chatCharacterImageOpacity",
  resolveOpacityAlpha: (value: unknown, fallback = 35) =>
    typeof value === "number" && Number.isFinite(value)
      ? value / 100
      : fallback / 100,
}));

vi.mock("../Knowledge/utils/unsupported-types", () => ({
  otherUnsupportedTypes: [],
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
  useSetting: (setting: string) => {
    if (setting === "chatWindowOpacity") return [35];
    if (setting === "chatMessageOpacity") return [60];
    if (setting === "chatCharacterImageOpacity") return [100];
    return [""];
  },
}));

vi.mock("@/hooks/useDarkmode", () => ({
  useDarkMode: () => ({
    mode: darkModeState.mode,
    toggleDarkMode: darkModeState.toggleDarkMode,
  }),
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

vi.mock("@/hooks/useLoadLocalConversation", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("@/hooks/useLoadLocalConversation")
  >()),
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
    useNavigate: () => routerNavigateState.navigate,
	    useLocation: () => ({
	      pathname:
	        routerLocationState.value?.pathname ??
	        window.location.pathname ??
	        "/chat",
	      search: routerLocationState.value?.search ?? window.location.search ?? "",
	      hash: routerLocationState.value?.hash ?? window.location.hash ?? "",
	      state: null,
	      key: routerLocationState.value?.key ?? "test-location",
	    }),
	  };
	});

describe("Playground cockpit shell", () => {
  beforeEach(() => {
    connectionState.isConnected = true
    connectionState.phase = "connected"
    routeApplyOutcome.value = null
    vi.clearAllMocks();
    recentChatState.enabled = false;
    recentChatState.recent.mockReset().mockResolvedValue(null);
    recentChatState.bookmark.mockReset().mockResolvedValue(null);
    recentChatState.capture.mockReset().mockImplementation(async (owner, request) => {
      const view = { ...request, owner_key: "local-owner" };
      const snapshot = { version: 1, owner_key: "local-owner", conversation_id: owner.conversation_id, source_digest: "source", storage_context_digest: "storage", fences: {}, interpretation_status: { kind: "parent_graph_v1" }, nodes: [] };
      const result = resolveHistorySelection(snapshot, view, "send", "");
      if (result.status !== "ready") return { ...result, view, snapshot };
      return { status: "captured", view, snapshot, rows: result.rows, selected_content: [], storage_context_digest: "storage", purpose: "send" };
    });
    window.history.replaceState({}, "", "/chat");
    storageState.values.clear();
    storageState.values.set("playgroundChatContextRailVisible", true);
    storageState.values.set("playgroundChatRuntimeRailVisible", true);
    messageOptionState.value.messages = [];
    messageOptionState.value.history = [];
    messageOptionState.value.historyId = null;
    messageOptionState.value.serverChatId = null;
    messageOptionState.value.serverChatCharacterId = null;
    messageOptionState.value.serverChatAssistantKind = null;
    messageOptionState.value.serverChatAssistantId = null;
    messageOptionState.value.serverChatMetaLoaded = false;
    messageOptionState.value.streaming = false;
    messageOptionState.value.selectedModel = "openai:gpt-4.1-mini";
    messageOptionState.value.selectedSystemPrompt = "";
    messageOptionState.value.selectedQuickPrompt = null;
    messageOptionState.value.selectedAssistant = null;
    messageOptionState.value.selectedCharacter = null;
    messageOptionState.value.setSelectedCharacter = vi.fn();
    messageOptionState.value.setServerChatId = vi.fn();
    messageOptionState.value.setServerChatCharacterId = vi.fn();
    messageOptionState.value.setServerChatAssistantKind = vi.fn();
    messageOptionState.value.setServerChatAssistantId = vi.fn();
    messageOptionState.value.setServerChatPersonaMemoryMode = vi.fn();
    messageOptionState.value.setServerChatMetaLoaded = vi.fn();
    messageOptionState.value.contextFiles = [];
    messageOptionState.value.regenerateLastMessage = vi.fn();
    sessionPersistenceState.value.sessionScopeReady = true;
    sessionPersistenceState.value.hasPersistedSession = false;
    sessionPersistenceState.value.hasPersistedDraftHandle = false;
    sessionPersistenceState.value.hasPersistedFreshDraft = false;
    sessionPersistenceState.value.restoreSession.mockReset().mockResolvedValue("not-restored");
    sessionStorage.removeItem("tldw-h1-playground-reference");
    useStoreMessageOption.setState({ chatDraftAddress: null, chatDraftRevision: 0, chatDraftTransition: null, historyId: null, serverChatId: null, compareParentByHistory: {} });
    usePlaygroundSessionStore.getState().clearSession();
    sessionPersistenceState.value.clearPersistedSession.mockClear();
    chatSettingsState.syncChatSettingsForServerChat.mockClear();
    darkModeState.mode = "dark";
    darkModeState.toggleDarkMode.mockClear();
	    cockpitChatRenderState.starterDeckSignals = [];
	    routerLocationState.value = null;
    routerNavigateState.navigate.mockClear();
	    tldwServerState.fetchChatModels.mockResolvedValue([
      {
        model: "openai:gpt-4.1-mini",
        provider: "openai",
        is_configured: true,
        provider_is_configured: true,
      },
    ]);
    tldwClientState.getCharacter.mockImplementation(
      async (id: string | number) => ({
        id,
        name: "Route Character",
      }),
    );
    characterSessionsPanelState.props = [];
  });

  it("does not add a second themed wash on the cockpit shell", () => {
    render(
      <PlaygroundCockpitShell
        mode="cockpit"
        themedBackdrop
        leftRail={<aside>Context</aside>}
        rightRail={<aside>Runtime</aside>}
      >
        <main>Chat</main>
      </PlaygroundCockpitShell>,
    );

    const shell = screen.getByTestId("playground-cockpit-shell");
    expect(shell).toHaveClass("backdrop-blur-[1px]");
    expect(shell.style.backgroundColor).toBe("");
  });

  it("renders the cockpit rails and main chat surface without a bottom status strip by default", async () => {
    render(<Playground />);

    expect(
      await screen.findByTestId("playground-cockpit-shell"),
    ).toHaveAttribute("data-mode", "cockpit");
    expect(
      screen.getByTestId("playground-cockpit-left-rail"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("playground-cockpit-right-rail"),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId("playground-cockpit-status-strip"),
    ).not.toBeInTheDocument();
    expect(
      screen.getByTestId("playground-cockpit-mode-summary"),
    ).toHaveTextContent("Context and chat settings visible.");
    expect(screen.getByTestId("playground-chat")).toBeInTheDocument();
    expect(screen.getByTestId("playground-form")).toBeInTheDocument();
    expect(
      screen.queryByTestId("character-chat-sessions-panel"),
    ).not.toBeInTheDocument();
    await waitFor(() => {
      expect(tldwServerState.fetchChatModels).toHaveBeenCalledWith({
        requireFresh: true,
      });
    });
  });

  it("exposes the shared theme toggle contract in the cockpit header", async () => {
    render(<Playground />);

    const themeToggle = await screen.findByTestId("chat-header-theme-toggle");
    expect(themeToggle).toHaveAccessibleName("Switch to light theme");

    fireEvent.click(themeToggle);

    expect(darkModeState.toggleDarkMode).toHaveBeenCalledTimes(1);
  });

  it("renders Character Chat recent sessions only for the active character workflow", async () => {
    storageState.values.set("playgroundChatWorkflowMode", "character");
    messageOptionState.value.selectedCharacter = {
      id: "char-1",
      name: "Ariadne",
    };
    messageOptionState.value.serverChatId = "server-chat-1";

    render(<Playground />);

    await waitFor(() => {
      expect(
        screen.getAllByTestId("character-chat-sessions-panel").length,
      ).toBeGreaterThan(0);
    });
    expect(characterSessionsPanelState.props.at(-1)).toMatchObject({
      activeCharacterId: "char-1",
      activeCharacterName: "Ariadne",
      activeServerChatId: "server-chat-1",
    });
  });

  it("uses a tracked assistant selection as the active character when legacy character state is empty", async () => {
    storageState.values.set("playgroundChatWorkflowMode", "character");
    messageOptionState.value.selectedAssistant = {
      kind: "character",
      id: "char-route",
      name: "Route Character",
      metadata: { selectionMode: "tracked" },
    };
    messageOptionState.value.selectedCharacter = null;

    render(<Playground />);

    expect(
      await screen.findByTestId("playground-active-chat-mode"),
    ).toHaveTextContent("Route Character");
    expect(
      screen.queryByText("Choose a character to start character chat"),
    ).not.toBeInTheDocument();
    await waitFor(() => {
      expect(characterSessionsPanelState.props.at(-1)).toMatchObject({
        activeCharacterId: "char-route",
        activeCharacterName: "Route Character",
      });
    });
  });

  it("preserves the loaded character chat during background selection hydration", async () => {
    storageState.values.set("playgroundChatWorkflowMode", "character");
    messageOptionState.value.serverChatId = "chat-character-a";
    messageOptionState.value.serverChatCharacterId = "character-a";
    messageOptionState.value.serverChatAssistantKind = "character";
    messageOptionState.value.serverChatAssistantId = "character-a";
    messageOptionState.value.serverChatMetaLoaded = true;
    messageOptionState.value.historyId = "local-chat-a";
    messageOptionState.value.messages = [
      {
        role: "assistant",
        isBot: true,
        message: "Prior character reply",
      },
    ];
    messageOptionState.value.history = [
      {
        role: "assistant",
        content: "Prior character reply",
      },
    ];
    messageOptionState.value.selectedAssistant = {
      kind: "character",
      id: "character-a",
      name: "Character A",
      metadata: { selectionMode: "tracked" },
    };
    messageOptionState.value.selectedCharacter = {
      id: "character-a",
      name: "Character A",
    };

    const { rerender } = render(<Playground />);

    await screen.findByTestId("playground-cockpit-shell");
    await waitFor(() => {
      expect(characterSessionsPanelState.props.at(-1)).toMatchObject({
        activeServerChatId: "chat-character-a",
      });
    });
    vi.mocked(messageOptionState.value.setServerChatId).mockClear();
    vi.mocked(messageOptionState.value.setHistoryId).mockClear();
    vi.mocked(messageOptionState.value.setHistory).mockClear();
    vi.mocked(messageOptionState.value.setMessages).mockClear();
    vi.mocked(messageOptionState.value.setServerChatCharacterId).mockClear();
    vi.mocked(messageOptionState.value.setServerChatAssistantKind).mockClear();
    vi.mocked(messageOptionState.value.setServerChatAssistantId).mockClear();
    vi.mocked(messageOptionState.value.setServerChatPersonaMemoryMode).mockClear();
    vi.mocked(messageOptionState.value.setServerChatMetaLoaded).mockClear();

    // useMessageOption keeps selectedAssistant bound to the loaded chat;
    // its raw selectedCharacter can still hydrate a different preference.
    messageOptionState.value.selectedCharacter = {
      id: "character-b",
      name: "Character B",
    };

    rerender(<Playground />);

    // Explicit picker changes are exercised by the coordinator integration tests.
    // A hydrated preference does not replace the loaded conversation's identity.
    expect(characterSessionsPanelState.props.at(-1)).toMatchObject({
      activeServerChatId: "chat-character-a",
      activeCharacterId: "character-a",
    });
    for (const setter of [
      messageOptionState.value.setServerChatId,
      messageOptionState.value.setHistoryId,
      messageOptionState.value.setHistory,
      messageOptionState.value.setMessages,
      messageOptionState.value.setServerChatCharacterId,
      messageOptionState.value.setServerChatAssistantKind,
      messageOptionState.value.setServerChatAssistantId,
      messageOptionState.value.setServerChatPersonaMemoryMode,
      messageOptionState.value.setServerChatMetaLoaded,
    ]) expect(setter).not.toHaveBeenCalled();
  });

  it("shows the starter deck for a true blank chat state", async () => {
    render(<Playground />);

    expect(
      await screen.findByTestId("playground-empty-mode-deck"),
    ).toBeInTheDocument();
  });

  it("restores a verified fresh handle despite a malformed older H1 bookmark", async () => {
    sessionPersistenceState.value.hasPersistedSession = true;
    sessionPersistenceState.value.hasPersistedDraftHandle = true;
    sessionPersistenceState.value.hasPersistedFreshDraft = true;
    sessionPersistenceState.value.restoreSession.mockResolvedValue("restored");
    sessionStorage.setItem("tldw-h1-playground-reference", "{broken");
    render(<Playground />);
    await waitFor(() => expect(sessionPersistenceState.value.restoreSession).toHaveBeenCalledTimes(1));
  });

  it("publishes the canonical draft address when the actual H1 recent-chat fallback opens a local conversation", async () => {
    useStoreMessageOption.getState().startNewChatDraft();
    recentChatState.enabled = true;
    recentChatState.recent.mockResolvedValue({ history: { id: "local-resume", local_owner_key: "local-owner" }, messages: [] });
    messageOptionState.value.setHistoryId = vi.fn((id, options) => useStoreMessageOption.getState().setHistoryId(id, options));
    messageOptionState.value.setServerChatId = vi.fn((id) => useStoreMessageOption.getState().setServerChatId(id));
    render(<Playground />);
    await waitFor(() => expect(useStoreMessageOption.getState()).toMatchObject({ historyId: "local-resume", serverChatId: null, chatDraftAddress: { kind: "local", id: "local-resume", profileId: "profile" } }));
    expect(recentChatState.capture).toHaveBeenCalledTimes(1);
    expect(useStoreMessageOption.getState().chatDraftTransition?.kind).toBe("switch");
  });

  // Actual route/H1/core; the Form action, database rows, owner resolution and
  // backend reads remain the existing declared fixture boundaries.
  it.each([
    [
      "same query character",
      "/chat?mode=character&characterId=foreign-card&handoff=opaque-copy",
    ],
    [
      "hash-router character",
      "/options.html#/chat?mode=character&characterId=foreign-card&handoff=opaque-copy",
    ],
    [
      "search copy with hash character",
      "/chat?handoff=opaque-copy#/chat?mode=character&characterId=foreign-card",
    ],
    [
      "bare fragment copy with search character",
      "/chat?mode=character&characterId=foreign-card#handoff=opaque-copy",
    ],
    [
      "same query chatId",
      "/chat?mode=character&characterId=foreign-card&chatId=foreign-chat&handoff=opaque-copy",
    ],
    [
      "hash chat_id with search copy",
      "/chat?handoff=opaque-copy#/chat?mode=character&chat_id=foreign-chat",
    ],
    [
      "search serverChatId with hash-query copy",
      "/chat?mode=character&serverChatId=foreign-chat#?handoff=opaque-copy",
    ],
    [
      "search server_chat_id with bare fragment copy",
      "/chat?mode=character&server_chat_id=foreign-chat#handoff=opaque-copy",
    ],
  ])(
    "keeps %s in literal-copy review without adopting bare character or chat identity",
    async (_case, url) => {
      window.history.replaceState({}, "", url);
      render(<Playground />);
      await screen.findByTestId("playground-cockpit-shell");
      await act(async () => {});

      expect(tldwClientState.getCharacter).not.toHaveBeenCalled();
      expect(
        messageOptionState.value.setSelectedCharacter,
      ).not.toHaveBeenCalled();
      expect(messageOptionState.value.setHistoryId).not.toHaveBeenCalled();
      expect(messageOptionState.value.setServerChatId).not.toHaveBeenCalled();
      expect(
        usePlaygroundSessionStore.getState().serverChatSelectionIntent,
      ).toBeNull();
      expect(
        screen.getByTestId("playground-active-chat-mode"),
      ).toHaveTextContent("Standard chat");
    },
  );

  it.each([
    ["outer character pointer", ""],
    ["outer character pointer and copy token", "&handoff=ambient-copy"],
  ])(
    "retires %s after copy cleanup while allowing a new explicit router character link",
    async (_case, outerCopy) => {
      const ambientSearch =
        "?mode=character&characterId=foreign-card" + outerCopy;
      window.history.replaceState(
        {},
        "",
        "/options.html" + ambientSearch + "#/chat?handoff=opaque-copy",
      );
      routerLocationState.value = {
        pathname: "/chat",
        search: "?handoff=opaque-copy",
        hash: "",
        key: "copy",
      };
      const { rerender } = render(<Playground />);
      await screen.findByTestId("playground-cockpit-shell");
      await act(async () => {});
      expect(tldwClientState.getCharacter).not.toHaveBeenCalled();

      window.history.replaceState(
        {},
        "",
        "/options.html" + ambientSearch + "#/chat",
      );
      routerLocationState.value = {
        pathname: "/chat",
        search: "",
        hash: "",
        key: "copy-cleaned",
      };
      rerender(<Playground />);
      await act(async () => {});
      expect(tldwClientState.getCharacter).not.toHaveBeenCalled();
      expect(
        usePlaygroundSessionStore.getState().serverChatSelectionIntent,
      ).toBeNull();
      expect(
        screen.getByTestId("playground-active-chat-mode"),
      ).toHaveTextContent("Standard chat");

      window.history.replaceState(
        {},
        "",
        "/options.html" +
          ambientSearch +
          "#/chat?mode=character&characterId=chosen-card",
      );
      routerLocationState.value = {
        pathname: "/chat",
        search: "?mode=character&characterId=chosen-card",
        hash: "",
        key: "explicit-character",
      };
      rerender(<Playground />);
      await waitFor(() =>
        expect(tldwClientState.getCharacter).toHaveBeenCalledOnce(),
      );
      expect(tldwClientState.getCharacter.mock.calls[0]?.[0]).toBe(
        "chosen-card",
      );
      expect(
        messageOptionState.value.setSelectedCharacter,
      ).toHaveBeenCalledWith({ id: "chosen-card", name: "Route Character" });
    },
  );

  it("does not introduce bare-fragment character commands without a copy", async () => {
    window.history.replaceState(
      {},
      "",
      "/chat#mode=character&characterId=foreign-card&chatId=foreign-chat",
    );
    render(<Playground />);
    await screen.findByTestId("playground-cockpit-shell");
    await act(async () => {});
    expect(tldwClientState.getCharacter).not.toHaveBeenCalled();
    expect(
      usePlaygroundSessionStore.getState().serverChatSelectionIntent,
    ).toBeNull();
    expect(messageOptionState.value.setServerChatId).not.toHaveBeenCalled();
    expect(screen.getByTestId("playground-active-chat-mode")).toHaveTextContent(
      "Standard chat",
    );
  });

  it("still loads an owner-qualified history selection while a literal-copy token awaits review", async () => {
    useStoreMessageOption.getState().startNewChatDraft();
    const reference = {
      profile_id: "profile",
      client_session_id: "source-session",
      owner_key: "local-owner",
      conversation_id: "qualified-local-chat",
    };
    const bookmark: HistoryBookmark = {
      profile_id: reference.profile_id,
      client_session_id: reference.client_session_id,
      owner_key: reference.owner_key,
      conversation_id: reference.conversation_id,
      view: {
        view_session_id: "source-view",
        owner_key: reference.owner_key,
        conversation_id: reference.conversation_id,
        interpretation: { kind: "parent_graph_v1" },
        cursor: { kind: "empty" },
        selection_revision: 1,
      },
    };
    recentChatState.bookmark.mockResolvedValue(bookmark);
    const path = historySelectionExpansionPath({
      reference,
      owner: {
        kind: "local",
        profile_id: "profile",
        owner_key: "local-owner",
        conversation_id: "qualified-local-chat",
      },
    });
    window.history.replaceState({}, "", path + "&handoff=opaque-copy");
    messageOptionState.value.setHistoryId = vi.fn((id, options) =>
      useStoreMessageOption.getState().setHistoryId(id, options),
    );
    messageOptionState.value.setServerChatId = vi.fn((id) =>
      useStoreMessageOption.getState().setServerChatId(id),
    );
    render(<Playground />);

    await waitFor(() =>
      expect(useStoreMessageOption.getState()).toMatchObject({
        historyId: "qualified-local-chat",
        serverChatId: null,
        chatDraftAddress: {
          kind: "local",
          id: "qualified-local-chat",
          profileId: "profile",
        },
      }),
    );
    expect(recentChatState.capture).toHaveBeenCalledOnce();
    expect(recentChatState.capture.mock.calls[0]?.[0]).toMatchObject({
      kind: "local",
      owner_key: "local-owner",
      conversation_id: "qualified-local-chat",
    });
    expect(tldwClientState.getCharacter).not.toHaveBeenCalled();
    expect(
      usePlaygroundSessionStore.getState().serverChatSelectionIntent,
    ).toBeNull();
  });

  it("qualifies an explicit saved-character URL through the shared draft selection callback", async () => {
    window.history.replaceState({}, "", "/chat?mode=character&characterId=char-route&chatId=route-chat");
    render(<Playground />);
    await waitFor(() => expect(usePlaygroundSessionStore.getState().serverChatSelectionIntent).toMatchObject({ chatId: "route-chat", onLoaded: expect.any(Function), isCurrent: expect.any(Function) }));
    const intent = usePlaygroundSessionStore.getState().serverChatSelectionIntent!;
    expect(intent.isCurrent?.()).toBe(true);
    act(() => useStoreMessageOption.getState().startNewChatDraft());
    expect(intent.isCurrent?.()).toBe(false);
  });

  it("lets a current incoming sidebar selection own initialization before a fresh restore", async () => {
    sessionPersistenceState.value.hasPersistedSession = true;
    sessionPersistenceState.value.hasPersistedDraftHandle = true;
    sessionPersistenceState.value.hasPersistedFreshDraft = true;
    useStoreMessageOption.getState().startNewChatDraft();
    const accept = captureChatDraftLoad("switch");
    usePlaygroundSessionStore.getState().requestServerChatSelection("sidebar-B", accept);
    useStoreMessageOption.getState().setServerChatId("sidebar-B");
    messageOptionState.value.serverChatId = "sidebar-B";
    render(<Playground />);
    await screen.findByTestId("playground-cockpit-shell");
    await act(async () => { await Promise.resolve(); });
    expect(sessionPersistenceState.value.restoreSession).not.toHaveBeenCalled();
    expect(accept.isCurrent()).toBe(true);
  });

  it("does not report standard chat runtime ready when no chat models are usable", async () => {
    messageOptionState.value.selectedModel = "tldw:gpt-4o";
    tldwServerState.fetchChatModels.mockResolvedValue([]);

    render(<Playground />);

    await waitFor(() => {
      expect(tldwServerState.fetchChatModels).toHaveBeenCalledWith({
        requireFresh: true,
      });
    });

    const runtimeRail = screen.getByTestId("playground-cockpit-right-rail");
    expect(runtimeRail).toHaveTextContent("Error");
    expect(runtimeRail).toHaveTextContent("No chat models configured");
    expect(runtimeRail).not.toHaveTextContent("Ready");

    expect(
      screen.queryByRole("status", { name: "Chat status" }),
    ).not.toBeInTheDocument();
  });

  it('keeps a current unapplied character route and lets Retry accept it', async () => {
    window.history.replaceState({}, '', '/chat?mode=character&characterId=char-route&keep=1');
    messageOptionState.value.serverChatId = 'ordinary-source';
    messageOptionState.value.historyId = 'ordinary-history';
    routeApplyOutcome.value = 'retired';
    render(<Playground />);
    const status = await screen.findByRole('status', { name: 'Character Chat setup status' });
    expect(status).toHaveTextContent('Character setup was not applied');
    expect(routerNavigateState.navigate).not.toHaveBeenCalled();
    expect(messageOptionState.value.setSelectedCharacter).not.toHaveBeenCalled();
    expect(messageOptionState.value.setMessages).not.toHaveBeenCalled();
    expect(messageOptionState.value.serverChatId).toBe('ordinary-source');
    routeApplyOutcome.value = null;
    fireEvent.click(within(status).getByRole('button', { name: 'Retry', exact: true }));
    await waitFor(() =>
      expect(routerNavigateState.navigate).toHaveBeenCalledWith(
        expect.objectContaining({ search: '?mode=character&keep=1' }),
        { replace: true }
      )
    );
    expect(tldwClientState.getCharacter).toHaveBeenCalledTimes(2);
    expect(messageOptionState.value.setSelectedCharacter).toHaveBeenCalledOnce();
  });

  it('keeps catalog discovery and a character route pending until the connection is ready', async () => {
    window.history.replaceState({}, '', '/chat?mode=character&characterId=char-route');
    connectionState.isConnected = false;
    connectionState.phase = 'checking';
    const view = render(<Playground />);
    await act(async () => { await Promise.resolve(); await Promise.resolve(); });
    expect(tldwServerState.fetchChatModels).not.toHaveBeenCalled();
    expect(tldwClientState.getCharacter).not.toHaveBeenCalled();
    connectionState.isConnected = true;
    view.rerender(<Playground />);
    await act(async () => { await Promise.resolve(); });
    expect(tldwServerState.fetchChatModels).not.toHaveBeenCalled();
    expect(tldwClientState.getCharacter).not.toHaveBeenCalled();
    connectionState.phase = 'connected';
    view.rerender(<Playground />);
    await waitFor(() => expect(tldwServerState.fetchChatModels).toHaveBeenCalledOnce());
    await waitFor(() => expect(tldwClientState.getCharacter).toHaveBeenCalledOnce());
  });

  it('waits for the initial model catalog before applying a fresh character route', async () => {
    window.history.replaceState({}, '', '/chat?mode=character&characterId=char-route');
    let finish!: (models: Awaited<ReturnType<typeof tldwServerState.fetchChatModels>>) => void;
    tldwServerState.fetchChatModels.mockReturnValueOnce(
      new Promise((resolve) => {
        finish = resolve;
      })
    );
    render(<Playground />);
    try {
      await waitFor(() => expect(tldwServerState.fetchChatModels).toHaveBeenCalledOnce());
      await act(async () => {
        await Promise.resolve();
      });
      expect(tldwClientState.getCharacter).not.toHaveBeenCalled();
      expect(messageOptionState.value.setMessages).not.toHaveBeenCalled();
      expect(routerNavigateState.navigate).not.toHaveBeenCalled();
    } finally {
      await act(async () => {
        finish([
          {
            model: 'openai:gpt-4.1-mini',
            provider: 'openai',
            is_configured: true,
            provider_is_configured: true,
          },
        ]);
      });
    }
    await waitFor(() =>
      expect(routerNavigateState.navigate).toHaveBeenCalledWith(
        expect.objectContaining({ search: '?mode=character' }),
        { replace: true }
      )
    );
    expect(tldwClientState.getCharacter).toHaveBeenCalledOnce();
  });

  it("keeps an accepted fresh command consumed after the router commits", async () => {
    window.history.replaceState({}, "", "/chat?mode=character&characterId=char-route")
    messageOptionState.value.setSelectedCharacter.mockImplementation((next: { id: string | number; name: string }) => {
      messageOptionState.value.selectedCharacter = next
      messageOptionState.value.selectedAssistant = { kind: "character", id: String(next.id), name: next.name, metadata: { selectionMode: "tracked" } }
    })
    const { rerender } = render(<Playground />)
    await waitFor(() => expect(routerNavigateState.navigate).toHaveBeenCalledWith(expect.objectContaining({ search: "?mode=character" }), { replace: true }))
    routerNavigateState.navigate.mockClear()
    window.history.replaceState({}, "", "/chat?mode=character")
    routerLocationState.value = { pathname: "/chat", search: "?mode=character", hash: "", key: "consumed-entry" }
    rerender(<Playground />)
    await act(async () => { await Promise.resolve(); await Promise.resolve() })
    expect(routerNavigateState.navigate).not.toHaveBeenCalled()
    expect(tldwClientState.getCharacter).toHaveBeenCalledOnce()
  })

  it.each(["held", "missing"])("preserves the source before a %s fresh route card resolves", async outcome => {
    window.history.replaceState({}, "", "/chat?mode=character&characterId=char-route")
    messageOptionState.value.serverChatId = "source-chat"
    messageOptionState.value.historyId = "source-history"
    messageOptionState.value.messages = [{ message: "Source transcript", isBot: true }]
    let finish!: (value: unknown) => void
    const pending = new Promise(resolve => { finish = resolve })
    tldwClientState.getCharacter.mockReturnValueOnce(outcome === "held" ? pending : Promise.resolve(null))
    render(<Playground />)
    await waitFor(() => expect(tldwClientState.getCharacter).toHaveBeenCalledOnce())
    await act(async () => { await Promise.resolve() })
    expect(messageOptionState.value.setHistoryId).not.toHaveBeenCalled()
    expect(messageOptionState.value.setHistory).not.toHaveBeenCalled()
    expect(messageOptionState.value.setMessages).not.toHaveBeenCalled()
    expect(messageOptionState.value.setServerChatId).not.toHaveBeenCalled()
    expect(messageOptionState.value.setSelectedCharacter).not.toHaveBeenCalled()
    if (outcome === "held") await act(async () => finish(null))
    await screen.findByText("Character char-route could not be loaded")
    expect(messageOptionState.value.setSelectedCharacter).not.toHaveBeenCalled()
  })

	  it("honors first-class character route intent in the chat shell", async () => {
    window.history.replaceState(
      {},
      "",
      "/chat?mode=character&characterId=char-route",
    );

    render(<Playground />);

    await waitFor(() => {
      expect(
        screen.getByTestId("playground-active-chat-mode"),
      ).toHaveTextContent("Character Chat");
    });
    await waitFor(() => {
      expect(tldwClientState.getCharacter).toHaveBeenCalledWith("char-route", { requestScope: { config: { serverUrl: "http://chat.test", authMode: "multi-user" }, userId: "account-a" } });
    });
    expect(messageOptionState.value.setSelectedCharacter).toHaveBeenCalledWith(
      expect.objectContaining({
        id: "char-route",
        name: "Route Character",
      }),
    );
	  });

  it("does not replace an explicit character route with a stale selected character before hydration", async () => {
    window.history.replaceState(
      {},
      "",
      "/chat?mode=character&characterId=char-route",
    );
    messageOptionState.value.selectedCharacter = {
      id: "previous-character",
      name: "Previous Character",
    };
    messageOptionState.value.selectedAssistant = {
      kind: "character",
      id: "previous-character",
      name: "Previous Character",
      metadata: { selectionMode: "tracked" },
    };

    render(<Playground />);

    await waitFor(() => {
      expect(tldwClientState.getCharacter).toHaveBeenCalledWith("char-route", { requestScope: { config: { serverUrl: "http://chat.test", authMode: "multi-user" }, userId: "account-a" } });
    });
    expect(routerNavigateState.navigate).not.toHaveBeenCalledWith(
      {
        pathname: "/chat",
        search: "?mode=character&characterId=previous-character",
        hash: "",
      },
      { replace: true },
    );
  });

  it("does not rehydrate the same explicit character route under a fresh location key", async () => {
    window.history.replaceState(
      {},
      "",
      "/chat?mode=character&characterId=char-route",
    );
    messageOptionState.value.setSelectedCharacter = vi.fn((next) => {
      messageOptionState.value.selectedCharacter = next;
      messageOptionState.value.selectedAssistant = next
        ? {
            kind: "character",
            id: String(next.id),
            name: next.name,
            metadata: { selectionMode: "tracked" },
          }
        : null;
    });

    const { rerender } = render(<Playground />);

    await waitFor(() => {
      expect(tldwClientState.getCharacter).toHaveBeenCalledWith("char-route", { requestScope: { config: { serverUrl: "http://chat.test", authMode: "multi-user" }, userId: "account-a" } });
    });
    await waitFor(() => {
      expect(
        messageOptionState.value.setSelectedCharacter,
      ).toHaveBeenCalledWith(
        expect.objectContaining({
          id: "char-route",
          name: "Route Character",
        }),
      );
    });

    messageOptionState.value.selectedCharacter = {
      id: "previous-character",
      name: "Previous Character",
    };
    messageOptionState.value.selectedAssistant = {
      kind: "character",
      id: "previous-character",
      name: "Previous Character",
      metadata: { selectionMode: "tracked" },
    };
    routerLocationState.value = {
      pathname: "/chat",
      search: "?mode=character&characterId=char-route",
      hash: "",
      key: "fresh-char-route-location",
    };
    routerNavigateState.navigate.mockClear();

    rerender(<Playground />);

    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(tldwClientState.getCharacter).toHaveBeenCalledTimes(1);
  });

  it("hydrates character route intent from the browser URL when router search is empty", async () => {
	    window.history.replaceState(
	      {},
	      "",
	      "/chat?mode=character&characterId=char-route",
	    );
	    routerLocationState.value = {
	      pathname: "/chat",
	      search: "",
	      hash: "",
	      key: "next-chat-empty-search",
	    };
	    messageOptionState.value.serverChatId = "stale-chat";
	    messageOptionState.value.historyId = "stale-history";
	    messageOptionState.value.messages = [
	      { role: "assistant", isBot: true, message: "Stale message" },
	    ];
	    messageOptionState.value.history = [
	      { role: "assistant", content: "Stale message" },
	    ];
	    messageOptionState.value.setSelectedCharacter = vi.fn((next) => {
	      messageOptionState.value.selectedCharacter = next;
	      messageOptionState.value.selectedAssistant = next
	        ? {
	            kind: "character",
	            id: String(next.id),
	            name: next.name,
	            metadata: { selectionMode: "tracked" },
	          }
	        : null;
	    });

	    const { rerender } = render(<Playground />);

	    await waitFor(() => {
	      expect(tldwClientState.getCharacter).toHaveBeenCalledWith("char-route", { requestScope: { config: { serverUrl: "http://chat.test", authMode: "multi-user" }, userId: "account-a" } });
	    });
	    expect(messageOptionState.value.setHistoryId).toHaveBeenCalledWith(null, {
	      preserveServerChatId: false,
	    });
	    expect(messageOptionState.value.setHistory).toHaveBeenCalledWith([]);
	    expect(messageOptionState.value.setMessages).toHaveBeenCalledWith([]);
	    expect(messageOptionState.value.setServerChatId).toHaveBeenCalledWith(null);

	    rerender(<Playground />);

	    await waitFor(() => {
	      expect(screen.getByTestId("playground-active-chat-mode")).toHaveTextContent(
	        "Route Character",
	      );
	    });
    expect(
      screen.queryByText("Choose a character to start character chat"),
    ).not.toBeInTheDocument();
  });

  it("keeps route character hydration alive while clearing stale chat state", async () => {
    window.history.replaceState(
      {},
      "",
      "/chat?mode=character&characterId=char-route",
    );
    routerLocationState.value = {
      pathname: "/chat",
      search: "",
      hash: "",
      key: "next-chat-empty-search",
    };
    messageOptionState.value.serverChatId = "stale-chat";
    messageOptionState.value.historyId = "stale-history";
    messageOptionState.value.messages = [
      { role: "assistant", isBot: true, message: "Stale message" },
    ];
    messageOptionState.value.history = [
      { role: "assistant", content: "Stale message" },
    ];
    messageOptionState.value.setSelectedCharacter = vi.fn((next) => {
      messageOptionState.value.selectedCharacter = next;
      messageOptionState.value.selectedAssistant = next
        ? {
            kind: "character",
            id: String(next.id),
            name: next.name,
            metadata: { selectionMode: "tracked" },
          }
        : null;
    });

    let resolveCharacter:
      | ((character: { id: string; name: string }) => void)
      | null = null;
    tldwClientState.getCharacter.mockImplementationOnce(
      (id: string | number) =>
        new Promise((resolve) => {
          resolveCharacter = () =>
            resolve({ id: String(id), name: "Route Character" });
        }),
    );

    const { rerender } = render(<Playground />);

    await waitFor(() => {
      expect(tldwClientState.getCharacter).toHaveBeenCalledWith("char-route", { requestScope: { config: { serverUrl: "http://chat.test", authMode: "multi-user" }, userId: "account-a" } });
    });

    messageOptionState.value.serverChatId = null;
    messageOptionState.value.historyId = null;
    messageOptionState.value.messages = [];
    messageOptionState.value.history = [];
    rerender(<Playground />);

    await act(async () => {
      resolveCharacter?.({ id: "char-route", name: "Route Character" });
      await Promise.resolve();
    });

    await waitFor(() => {
      expect(messageOptionState.value.setSelectedCharacter).toHaveBeenCalledWith(
        expect.objectContaining({
          id: "char-route",
          name: "Route Character",
        }),
      );
    });
    expect(tldwClientState.getCharacter).toHaveBeenCalledTimes(1);
  });

  it("does not re-enforce route character intent after a manual character switch", async () => {
    window.history.replaceState(
      {},
      "",
      "/chat?mode=character&characterId=char-route",
    );

    const { rerender } = render(<Playground />);

    await waitFor(() => {
      expect(tldwClientState.getCharacter).toHaveBeenCalledTimes(1);
    });
    await waitFor(() => {
      expect(
        messageOptionState.value.setSelectedCharacter,
      ).toHaveBeenCalledWith(
        expect.objectContaining({
          id: "char-route",
          name: "Route Character",
        }),
      );
    });

    tldwClientState.getCharacter.mockClear();
    messageOptionState.value.setSelectedCharacter.mockClear();
    messageOptionState.value.selectedCharacter = {
      id: "manual-character",
      name: "Manual Character",
    };

    rerender(<Playground />);

    await waitFor(() => {
      expect(
        screen.getByTestId("playground-active-chat-mode"),
      ).toHaveTextContent("Manual Character");
    });
    expect(tldwClientState.getCharacter).not.toHaveBeenCalled();
    expect(
      messageOptionState.value.setSelectedCharacter,
    ).not.toHaveBeenCalled();
    await waitFor(() => {
      expect(routerNavigateState.navigate).toHaveBeenCalledWith(
        {
          pathname: "/chat",
          search: "?mode=character&characterId=manual-character",
          hash: "",
        },
        { replace: true },
      );
    });
  });

  it("removes a stale route character id when the hydrated character is cleared", async () => {
    window.history.replaceState(
      {},
      "",
      "/chat?mode=character&characterId=char-route",
    );
    messageOptionState.value.setSelectedCharacter = vi.fn((next) => {
      messageOptionState.value.selectedCharacter = next;
      messageOptionState.value.selectedAssistant = next
        ? {
            kind: "character",
            id: String(next.id),
            name: next.name,
            metadata: { selectionMode: "tracked" },
          }
        : null;
    });

    const { rerender } = render(<Playground />);

    await waitFor(() => {
      expect(tldwClientState.getCharacter).toHaveBeenCalledTimes(1);
    });
    await waitFor(() => {
      expect(
        messageOptionState.value.setSelectedCharacter,
      ).toHaveBeenCalledWith(
        expect.objectContaining({
          id: "char-route",
          name: "Route Character",
        }),
      );
    });
    rerender(<Playground />);
    await waitFor(() => {
      expect(
        screen.getByTestId("playground-active-chat-mode"),
      ).toHaveTextContent("Route Character");
    });

    messageOptionState.value.selectedCharacter = null;
    messageOptionState.value.selectedAssistant = null;
    routerNavigateState.navigate.mockClear();

    rerender(<Playground />);

    await waitFor(() => {
      expect(routerNavigateState.navigate).toHaveBeenCalledWith(
        {
          pathname: "/chat",
          search: "?mode=character",
          hash: "",
        },
        { replace: true },
      );
    });
  });

  it("keeps a consumed mode-only route without recreating an unsaved entry command", async () => {
    window.history.replaceState({}, "", "/chat?mode=character");
    messageOptionState.value.selectedCharacter = { id: "manual-character", name: "Manual Character" };
    render(<Playground />);
    await act(async () => { await Promise.resolve(); });
    expect(routerNavigateState.navigate).not.toHaveBeenCalled();
    expect(tldwClientState.getCharacter).not.toHaveBeenCalled();
  });

  it("offers recovery without inventing a character when route loading returns none", async () => {
    window.history.replaceState({}, "", "/chat?mode=character&characterId=missing-character");
    tldwClientState.getCharacter.mockResolvedValueOnce(null);
    render(<Playground />);
    await screen.findByText("Character missing-character could not be loaded");
    expect(messageOptionState.value.setSelectedCharacter).not.toHaveBeenCalled();
  });

  it("shows recovery when a restored route character can no longer be loaded", async () => {
    window.history.replaceState(
      {},
      "",
      "/chat?mode=character&characterId=missing-character",
    );
    tldwClientState.getCharacter.mockResolvedValueOnce(null);
    const assistantSelectListener = vi.fn();
    window.addEventListener(
      "tldw:open-assistant-select",
      assistantSelectListener,
    );

    try {
      render(<Playground />);

      expect(
        await screen.findByText(
          "Character missing-character could not be loaded",
        ),
      ).toBeInTheDocument();

      fireEvent.click(screen.getByRole("button", { name: "Choose character" }));
      expect(assistantSelectListener).toHaveBeenCalledWith(
        expect.objectContaining({
          detail: expect.objectContaining({ tab: "character" }),
        }),
      );
    } finally {
      window.removeEventListener(
        "tldw:open-assistant-select",
        assistantSelectListener,
      );
    }
  });

  it("keeps restored route character recovery visible when an old character was selected", async () => {
    window.history.replaceState(
      {},
      "",
      "/chat?mode=character&characterId=missing-character",
    );
    messageOptionState.value.selectedCharacter = {
      id: "old-character",
      name: "Old Character",
    };
    tldwClientState.getCharacter.mockRejectedValueOnce(new Error("missing"));

    render(<Playground />);

    expect(
      await screen.findByText(
        "Character missing-character could not be loaded",
      ),
    ).toBeInTheDocument();
    expect(
      screen.queryByText("Choose a character to start character chat"),
    ).not.toBeInTheDocument();
  });

  it("keeps route character recovery valid through strict-mode effect replays", async () => {
    window.history.replaceState(
      {},
      "",
      "/chat?mode=character&characterId=missing-character",
    );
    tldwClientState.getCharacter.mockRejectedValue(new Error("missing"));

    render(
      <React.StrictMode>
        <Playground />
      </React.StrictMode>,
    );

    expect(
      await screen.findByText(
        "Character missing-character could not be loaded",
      ),
    ).toBeInTheDocument();
  });

  it("keeps the selected character while opening model settings from readiness recovery", async () => {
    storageState.values.set("playgroundChatWorkflowMode", "character");
    messageOptionState.value.selectedCharacter = {
      id: "char-1",
      name: "Ariadne",
    };
    messageOptionState.value.selectedModel = null;
    const modelSettingsListener = vi.fn();
    window.addEventListener("tldw:open-model-settings", modelSettingsListener);

    try {
      render(<Playground />);

      expect(
        within(
          await screen.findByTestId("character-chat-readiness-panel"),
        ).getByText("Choose a chat model before chatting as Ariadne"),
      ).toBeInTheDocument();

      fireEvent.click(
        within(screen.getByTestId("character-chat-readiness-panel")).getByRole(
          "button",
          { name: "Open model settings" },
        ),
      );

      expect(modelSettingsListener).toHaveBeenCalledTimes(1);
      expect(
        messageOptionState.value.setSelectedCharacter,
      ).not.toHaveBeenCalled();
      expect(messageOptionState.value.selectedCharacter).toEqual({
        id: "char-1",
        name: "Ariadne",
      });
    } finally {
      window.removeEventListener(
        "tldw:open-model-settings",
        modelSettingsListener,
      );
    }
  });

  it("retries character chat model catalog loading from readiness recovery", async () => {
    storageState.values.set("playgroundChatWorkflowMode", "character");
    messageOptionState.value.selectedCharacter = {
      id: "char-1",
      name: "Ariadne",
    };
    messageOptionState.value.selectedModel = "openai:gpt-4.1-mini";
    let rejectInitialModels: ((reason?: unknown) => void) | null = null;
    const initialModels = new Promise<
      Array<{
        model: string;
        provider: string;
        is_configured: boolean;
        provider_is_configured: boolean;
      }>
    >((_resolve, reject) => {
      rejectInitialModels = reject;
    });
    tldwServerState.fetchChatModels
      .mockImplementationOnce(() => initialModels)
      .mockResolvedValueOnce([
        {
          model: "openai:gpt-4.1-mini",
          provider: "openai",
          is_configured: true,
          provider_is_configured: true,
        },
      ]);

    render(<Playground />);

    expect(
      within(
        await screen.findByTestId("character-chat-readiness-panel"),
      ).getByText("Checking chat model readiness"),
    ).toBeInTheDocument();

    await act(async () =>
      rejectInitialModels?.(new Error("Fixture catalog read failed")),
    );
    const readinessPanel = within(
      screen.getByTestId("character-chat-readiness-panel"),
    );
    await readinessPanel.findByText("Could not check chat models");
    fireEvent.click(readinessPanel.getByRole("button", { name: "Try again" }));

    await waitFor(() => {
      expect(tldwServerState.fetchChatModels).toHaveBeenCalledTimes(2);
    });
    expect(tldwServerState.fetchChatModels).toHaveBeenLastCalledWith({
      requireFresh: true,
    });
    await waitFor(() =>
      expect(
        screen.queryByTestId("character-chat-readiness-panel"),
      ).not.toBeInTheDocument(),
    );
  });

  it("surfaces unavailable selected chat models from the model catalog", async () => {
    storageState.values.set("playgroundChatWorkflowMode", "character");
    messageOptionState.value.selectedCharacter = {
      id: "char-1",
      name: "Ariadne",
    };
    messageOptionState.value.selectedModel = "openai:missing-model";
    tldwServerState.fetchChatModels.mockResolvedValueOnce([
      {
        model: "openai:gpt-4.1-mini",
        provider: "openai",
        is_configured: true,
        provider_is_configured: true,
      },
    ]);

    render(<Playground />);

    expect(
      within(
        await screen.findByTestId("character-chat-readiness-panel"),
      ).getByText("Choose an available chat model before chatting as Ariadne"),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("status", { name: "Chat status" }),
    ).not.toBeInTheDocument();
  });

  it("keeps empty character chat model selection in the missing-model status", async () => {
    storageState.values.set("playgroundChatWorkflowMode", "character");
    messageOptionState.value.selectedCharacter = {
      id: "char-1",
      name: "Ariadne",
    };
    messageOptionState.value.selectedModel = "";

    render(<Playground />);

    expect(
      within(
        await screen.findByTestId("character-chat-readiness-panel"),
      ).getByText("Choose a chat model before chatting as Ariadne"),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("status", { name: "Chat status" }),
    ).not.toBeInTheDocument();
  });

  it("does not treat catalog-only backend models as ready for character chat", async () => {
    storageState.values.set("playgroundChatWorkflowMode", "character");
    messageOptionState.value.selectedCharacter = {
      id: "char-1",
      name: "Ariadne",
    };
    messageOptionState.value.selectedModel = "tldw:gpt-4o";
    tldwServerState.fetchChatModels.mockResolvedValueOnce([
      {
        id: "gpt-4o",
        model: "tldw:gpt-4o",
        provider: "openai",
        is_configured: false,
        provider_is_configured: false,
        catalog_only: true,
      } as any,
      {
        id: "gemma3:1b",
        model: "tldw:gemma3:1b",
        provider: "ollama",
        is_configured: true,
        provider_is_configured: true,
      } as any,
    ]);

    render(<Playground />);

    expect(
      within(
        await screen.findByTestId("character-chat-readiness-panel"),
      ).getByText("Choose a model and provider before starting character chat"),
    ).toBeInTheDocument();
  });

  it("does not treat provider-qualified catalog-only backend models as ready for character chat", async () => {
    storageState.values.set("playgroundChatWorkflowMode", "character");
    messageOptionState.value.selectedCharacter = {
      id: "char-1",
      name: "Ariadne",
    };
    messageOptionState.value.selectedModel = "tldw:openai:gpt-4o";
    tldwServerState.fetchChatModels.mockResolvedValueOnce([
      {
        id: "gpt-4o",
        model: "tldw:gpt-4o",
        provider: "openai",
        is_configured: false,
        provider_is_configured: false,
        catalog_only: true,
      } as any,
      {
        id: "gemma3:1b",
        model: "tldw:gemma3:1b",
        provider: "ollama",
        is_configured: true,
        provider_is_configured: true,
      } as any,
    ]);

    render(<Playground />);

    expect(
      within(
        await screen.findByTestId("character-chat-readiness-panel"),
      ).getByText("Configure the selected model provider before chatting as Ariadne"),
    ).toBeInTheDocument();
  });

  it("surfaces send-blocked readiness while character chat is streaming", async () => {
    storageState.values.set("playgroundChatWorkflowMode", "character");
    messageOptionState.value.selectedCharacter = {
      id: "char-1",
      name: "Ariadne",
    };
    messageOptionState.value.selectedModel = "openai:gpt-4.1-mini";
    messageOptionState.value.streaming = true;

    render(<Playground />);

    expect(
      within(
        await screen.findByTestId("character-chat-readiness-panel"),
      ).getByText("Character chat is preparing"),
    ).toBeInTheDocument();
    const runtimeRail = screen.getByTestId("playground-cockpit-right-rail");
    expect(within(runtimeRail).getByText("Streaming")).toBeInTheDocument();
    expect(within(runtimeRail).queryByText("Error")).not.toBeInTheDocument();
  });

  it("surfaces blocked server readiness locally in Character Chat mode", async () => {
    storageState.values.set("playgroundChatWorkflowMode", "character");
    messageOptionState.value.selectedCharacter = {
      id: "char-1",
      name: "Ariadne",
    };
    const modelSettingsListener = vi.fn();
    const serverSettingsListener = vi.fn();
    window.addEventListener("tldw:settings-navigation-request", serverSettingsListener);
    window.addEventListener("tldw:open-model-settings", modelSettingsListener);

    try {
      render(<Playground />);

      fireEvent(
        window,
        new CustomEvent("tldw:server-readiness-state", {
          detail: { state: "blocked" },
        }),
      );

      expect(
        within(
          await screen.findByTestId("character-chat-readiness-panel"),
        ).getByText("Connect to tldw_server before starting character chat"),
      ).toBeInTheDocument();

      fireEvent.click(
        screen.getByRole("button", { name: "Open server settings" }),
      );

      expect(modelSettingsListener).not.toHaveBeenCalled();
      const event = serverSettingsListener.mock.calls[0]?.[0] as CustomEvent;
      expect(event.detail).toMatchObject({
        destination: "/settings/tldw",
      });
    } finally {
      window.removeEventListener("tldw:settings-navigation-request", serverSettingsListener);
      window.removeEventListener(
        "tldw:open-model-settings",
        modelSettingsListener,
      );
    }
  });

  it("does not write a partial character when header intent only changes mode", async () => {
    render(<Playground />);

    await screen.findByTestId("playground-cockpit-shell");
    fireEvent(
      window,
      new CustomEvent("tldw:character-chat-mode-intent", {
        detail: { characterId: "char-event" },
      }),
    );

    await waitFor(() => {
      expect(
        screen.getByTestId("playground-active-chat-mode"),
      ).toHaveTextContent("Character Chat");
    });
    expect(
      messageOptionState.value.setSelectedCharacter,
    ).not.toHaveBeenCalled();
  });

  it("resets character workflow for non-character starter modes", async () => {
    storageState.values.set("playgroundChatWorkflowMode", "character");

    render(<Playground />);

    expect(
      await screen.findByTestId("playground-active-chat-mode"),
    ).toHaveTextContent("Character Chat");

    fireEvent(
      window,
      new CustomEvent("tldw:playground-starter-selected", {
        detail: { mode: "compare" },
      }),
    );

    await waitFor(() => {
      expect(
        screen.getByTestId("playground-active-chat-mode"),
      ).toHaveTextContent("Standard chat");
    });
    expect(storageState.values.get("playgroundChatWorkflowMode")).toBe(
      "standard",
    );
  });

  it("does not show persona names under the Character Chat mode chip", async () => {
    storageState.values.set("playgroundChatWorkflowMode", "character");
    messageOptionState.value.selectedAssistant = {
      kind: "persona",
      id: "persona-1",
      name: "Persona One",
    };

    render(<Playground />);

    const modeChip = await screen.findByTestId("playground-active-chat-mode");
    expect(modeChip).toHaveTextContent("Character Chat");
    expect(modeChip).not.toHaveTextContent("Persona One");
  });

  it("hides the starter deck when the composer has unsent draft text", async () => {
    render(<Playground />);

    expect(
      await screen.findByTestId("playground-empty-mode-deck"),
    ).toBeInTheDocument();

    fireEvent.change(screen.getByTestId("composer-textarea"), {
      target: { value: "Draft a cockpit-ready message" },
    });

    await waitFor(() => {
      expect(screen.queryByTestId("playground-empty-mode-deck")).toBeNull();
    });
  });

  it("keeps the starter deck hidden for an active conversation without rendered messages", async () => {
    messageOptionState.value.historyId = "local-history-1";
    messageOptionState.value.serverChatId = "server-chat-1";

    render(<Playground />);

    await screen.findByTestId("playground-cockpit-shell");

    expect(screen.queryByTestId("playground-empty-mode-deck")).toBeNull();
  });

  it("restores the starter deck when draft text is cleared before send", async () => {
    render(<Playground />);

    expect(
      await screen.findByTestId("playground-empty-mode-deck"),
    ).toBeInTheDocument();

    const composer = screen.getByTestId("composer-textarea");
    fireEvent.change(composer, {
      target: { value: "Temporary draft" },
    });

    await waitFor(() => {
      expect(screen.queryByTestId("playground-empty-mode-deck")).toBeNull();
    });

    fireEvent.change(composer, {
      target: { value: "" },
    });

    await waitFor(() => {
      expect(
        screen.getByTestId("playground-empty-mode-deck"),
      ).toBeInTheDocument();
    });
  });

  it("does not re-render the chat surface for draft edits that keep the same blankness", async () => {
    render(<Playground />);

    expect(
      await screen.findByTestId("playground-empty-mode-deck"),
    ).toBeInTheDocument();

    const composer = screen.getByTestId("composer-textarea");
    fireEvent.change(composer, {
      target: { value: "First non-empty draft" },
    });

    await waitFor(() => {
      expect(screen.queryByTestId("playground-empty-mode-deck")).toBeNull();
    });
    const renderCountAfterIntentChange =
      cockpitChatRenderState.starterDeckSignals.length;

    fireEvent.change(composer, {
      target: { value: "First non-empty draft with more text" },
    });

    await waitFor(() => {
      expect(cockpitChatRenderState.starterDeckSignals).toHaveLength(
        renderCountAfterIntentChange,
      );
    });
  });

  it("keeps cockpit tooltip ids unique across multiple shell instances", async () => {
    render(
      <>
        <Playground />
        <Playground />
      </>,
    );

    const contextCollapseTooltips = await screen.findAllByRole("tooltip", {
      name: "Hide context",
    });
    const tooltipIds = contextCollapseTooltips.map((tooltip) => tooltip.id);

    expect(contextCollapseTooltips).toHaveLength(2);
    expect(new Set(tooltipIds).size).toBe(tooltipIds.length);
  });

  it("restores focus mode from storage while preserving chat and composer", async () => {
    storageState.values.set("playgroundChatLayoutMode", "focus");

    render(<Playground />);

    expect(
      await screen.findByTestId("playground-cockpit-shell"),
    ).toHaveAttribute("data-mode", "focus");
    expect(screen.queryByTestId("playground-cockpit-left-rail")).toBeNull();
    expect(screen.queryByTestId("playground-cockpit-right-rail")).toBeNull();
    expect(
      screen.getByTestId("playground-cockpit-mode-summary"),
    ).toHaveTextContent(
      "Focus mode hides rails. Chat and composer remain active.",
    );
    expect(screen.getByTestId("playground-chat")).toBeInTheDocument();
    expect(screen.getByTestId("playground-form")).toBeInTheDocument();
  });

  it("persists layout mode changes from a keyboard-reachable toggle", async () => {
    render(<Playground />);

    const toggle = await screen.findByRole("button", {
      name: /enter focus chat/i,
    });
    expect(toggle).toBeEnabled();

    fireEvent.click(toggle);

    await waitFor(() => {
      expect(screen.getByTestId("playground-cockpit-shell")).toHaveAttribute(
        "data-mode",
        "focus",
      );
    });
    expect(storageState.values.get("playgroundChatLayoutMode")).toBe("focus");
    expect(
      screen.getByRole("button", { name: /exit focus/i }),
    ).toBeInTheDocument();
  });

  it("preserves both intentionally collapsed rails across focus and cockpit mode toggles", async () => {
    render(<Playground />);

    expect(
      await screen.findByTestId("playground-cockpit-shell"),
    ).toHaveAttribute("data-mode", "cockpit");

    fireEvent.click(
      within(screen.getByTestId("playground-cockpit-left-rail")).getByRole(
        "button",
        { name: /hide context/i },
      ),
    );
    fireEvent.click(
      within(screen.getByTestId("playground-cockpit-right-rail")).getByRole(
        "button",
        { name: /hide chat settings/i },
      ),
    );

    await waitFor(() => {
      expect(screen.queryByTestId("playground-cockpit-left-rail")).toBeNull();
      expect(screen.queryByTestId("playground-cockpit-right-rail")).toBeNull();
    });
    expect(storageState.values.get("playgroundChatContextRailVisible")).toBe(
      false,
    );
    expect(storageState.values.get("playgroundChatRuntimeRailVisible")).toBe(
      false,
    );

    fireEvent.click(screen.getByRole("button", { name: /enter focus chat/i }));

    await waitFor(() => {
      expect(screen.getByTestId("playground-cockpit-shell")).toHaveAttribute(
        "data-mode",
        "focus",
      );
    });

    fireEvent.click(
      screen.getByRole("button", { name: /exit focus/i }),
    );

    await waitFor(() => {
      expect(screen.getByTestId("playground-cockpit-shell")).toHaveAttribute(
        "data-mode",
        "cockpit",
      );
    });
    expect(screen.queryByTestId("playground-cockpit-left-rail")).toBeNull();
    expect(screen.queryByTestId("playground-cockpit-right-rail")).toBeNull();
    expect(
      screen.getByTestId("playground-cockpit-left-rail-restore"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("playground-cockpit-right-rail-restore"),
    ).toBeInTheDocument();
    expect(storageState.values.get("playgroundChatContextRailVisible")).toBe(
      false,
    );
    expect(storageState.values.get("playgroundChatRuntimeRailVisible")).toBe(
      false,
    );
  });

  it("persists independent context and runtime rail visibility in cockpit mode", async () => {
    render(<Playground />);

    expect(
      await screen.findByTestId("playground-cockpit-shell"),
    ).toHaveAttribute("data-mode", "cockpit");
    expect(
      screen.getByTestId("playground-cockpit-left-rail"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("playground-cockpit-right-rail"),
    ).toBeInTheDocument();

    fireEvent.click(
      within(screen.getByTestId("playground-cockpit-left-rail")).getByRole(
        "button",
        { name: /hide context/i },
      ),
    );

    await waitFor(() => {
      expect(screen.queryByTestId("playground-cockpit-left-rail")).toBeNull();
    });
    expect(
      screen.getByTestId("playground-cockpit-mode-summary"),
    ).toHaveTextContent("Context hidden. Chat settings visible.");
    expect(
      screen.getByTestId("playground-cockpit-right-rail"),
    ).toBeInTheDocument();
    expect(storageState.values.get("playgroundChatContextRailVisible")).toBe(
      false,
    );
    expect(
      screen.getByRole("button", { name: /show context/i }),
    ).toBeInTheDocument();

    fireEvent.click(
      within(screen.getByTestId("playground-cockpit-right-rail")).getByRole(
        "button",
        { name: /hide chat settings/i },
      ),
    );

    await waitFor(() => {
      expect(screen.queryByTestId("playground-cockpit-right-rail")).toBeNull();
    });
    expect(
      screen.getByTestId("playground-cockpit-mode-summary"),
    ).toHaveTextContent("Cockpit rails hidden. Chat and composer remain active.");
    expect(storageState.values.get("playgroundChatRuntimeRailVisible")).toBe(
      false,
    );
    expect(screen.getByTestId("playground-chat")).toBeInTheDocument();
    expect(screen.getByTestId("playground-form")).toBeInTheDocument();

    fireEvent.click(
      screen.getByRole("button", { name: /show context/i }),
    );
    fireEvent.click(
      screen.getByRole("button", { name: /show chat settings/i }),
    );

    await waitFor(() => {
      expect(
        screen.getByTestId("playground-cockpit-left-rail"),
      ).toBeInTheDocument();
      expect(
        screen.getByTestId("playground-cockpit-right-rail"),
      ).toBeInTheDocument();
    });
    expect(storageState.values.get("playgroundChatContextRailVisible")).toBe(
      true,
    );
    expect(storageState.values.get("playgroundChatRuntimeRailVisible")).toBe(
      true,
    );
  });

  it("persists sidechannel collapse and restore from rail-local controls", async () => {
    render(<Playground />);

    const contextRail = await screen.findByTestId(
      "playground-cockpit-left-rail",
    );
    const collapseContextButton = within(contextRail).getByRole("button", {
      name: /hide context/i,
    });
    const collapseContextTooltip = screen.getByRole("tooltip", {
      name: "Hide context",
    });
    expect(collapseContextButton).toHaveAttribute(
      "aria-describedby",
      collapseContextTooltip.id,
    );

    fireEvent.click(collapseContextButton);

    await waitFor(() => {
      expect(screen.queryByTestId("playground-cockpit-left-rail")).toBeNull();
    });
    const restoreContextButton = screen.getByRole("button", {
      name: /show context/i,
    });
    const restoreContextTooltip = screen.getByRole("tooltip", {
      name: "Show context",
    });
    expect(restoreContextButton).toHaveAttribute(
      "aria-describedby",
      restoreContextTooltip.id,
    );
    expect(storageState.values.get("playgroundChatContextRailVisible")).toBe(
      false,
    );
    expect(screen.getByTestId("playground-chat")).toBeInTheDocument();
    expect(screen.getByTestId("playground-form")).toBeInTheDocument();
    expect(
      screen.queryByTestId("playground-collapsed-composition-summary"),
    ).toBeNull();

    fireEvent.click(restoreContextButton);

    await waitFor(() => {
      expect(
        screen.getByTestId("playground-cockpit-left-rail"),
      ).toBeInTheDocument();
    });
    expect(storageState.values.get("playgroundChatContextRailVisible")).toBe(
      true,
    );

    const runtimeRailAfterRestore = screen.getByTestId(
      "playground-cockpit-right-rail",
    );
    const collapseRuntimeButton = within(runtimeRailAfterRestore).getByRole(
      "button",
      {
        name: /hide chat settings/i,
      },
    );
    const collapseRuntimeTooltip = screen.getByRole("tooltip", {
      name: "Hide chat settings",
    });
    expect(collapseRuntimeButton).toHaveAttribute(
      "aria-describedby",
      collapseRuntimeTooltip.id,
    );

    fireEvent.click(collapseRuntimeButton);

    await waitFor(() => {
      expect(screen.queryByTestId("playground-cockpit-right-rail")).toBeNull();
    });
    const restoreRuntimeButton = screen.getByRole("button", {
      name: /show chat settings/i,
    });
    const restoreRuntimeTooltip = screen.getByRole("tooltip", {
      name: "Show chat settings",
    });
    expect(restoreRuntimeButton).toHaveAttribute(
      "aria-describedby",
      restoreRuntimeTooltip.id,
    );
    expect(storageState.values.get("playgroundChatRuntimeRailVisible")).toBe(
      false,
    );
    expect(
      screen.queryByTestId("playground-cockpit-status-strip"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("playground-collapsed-composition-summary"),
    ).toBeNull();

    fireEvent.click(restoreRuntimeButton);

    await waitFor(() => {
      expect(
        screen.getByTestId("playground-cockpit-right-rail"),
      ).toBeInTheDocument();
    });
    expect(storageState.values.get("playgroundChatRuntimeRailVisible")).toBe(
      true,
    );
  });

  it("surfaces empty assistant response recovery in the runtime sidechannel", async () => {
    messageOptionState.value.selectedModel = "anthropic:claude-3-haiku";
    tldwServerState.fetchChatModels.mockResolvedValue([
      {
        model: "openai:gpt-4.1-mini",
        provider: "openai",
        is_configured: true,
        provider_is_configured: true,
      },
      {
        model: "anthropic:claude-3-haiku",
        provider: "anthropic",
        is_configured: true,
        provider_is_configured: true,
      },
    ]);
    messageOptionState.value.messages = [
      {
        id: "user-1",
        role: "user",
        isBot: false,
        name: "You",
        message: "Say something",
      },
      {
        id: "assistant-1",
        isBot: true,
        name: "openai:gpt-4.1-mini",
        message: "",
      },
    ];

    render(<Playground />);

    const runtimeRail = await screen.findByTestId(
      "playground-cockpit-right-rail",
    );
    const runControls = within(runtimeRail).getByRole("region", {
      name: "Run controls",
    });
    expect(
      within(runControls).getByRole("status", {
        name: "Empty assistant response",
      }),
    ).toHaveTextContent(
      "openai:gpt-4.1-mini returned no response text.",
    );

    fireEvent.click(
      within(runControls).getByRole("button", {
        name: "Regenerate last response",
      }),
    );

    expect(
      messageOptionState.value.regenerateLastMessage,
    ).toHaveBeenCalledTimes(1);
  });
});
