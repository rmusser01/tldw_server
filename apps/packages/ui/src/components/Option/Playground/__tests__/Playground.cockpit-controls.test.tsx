// @vitest-environment jsdom
import React from "react";
import { fireEvent, render, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { Playground } from "../Playground";

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
    setSelectedSystemPrompt: vi.fn(),
    selectedModel: "openai:gpt-4.1-mini" as string | null,
    setSelectedModel: vi.fn(),
    setServerChatId: vi.fn(),
    contextFiles: [{ id: "file-1", name: "brief.pdf" }],
    setContextFiles: vi.fn(),
    createChatBranch: vi.fn(),
    streaming: true,
    selectedCharacter: { id: "character-1", name: "Mira Vale" },
    setSelectedCharacter: vi.fn(),
    compareMode: false,
    compareFeatureEnabled: false,
    temporaryChat: true,
    webSearch: true,
    selectedKnowledge: [{ id: "knowledge-1", title: "Research notes" }],
    ragMediaIds: [101, 202],
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
  useStoreChatModelSettings: () => ({ setSystemPrompt: vi.fn() }),
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
    messageOptionState.value.messages = [
      { id: "message-1", role: "user", content: "Hello" },
      { id: "message-2", role: "assistant", content: "Hi" },
    ];
    messageOptionState.value.historyId = "history-1";
    messageOptionState.value.serverChatId = "chat-1";
    messageOptionState.value.streaming = true;
    messageOptionState.value.selectedModel = "openai:gpt-4.1-mini";
    messageOptionState.value.selectedCharacter = {
      id: "character-1",
      name: "Mira Vale",
    };
    messageOptionState.value.temporaryChat = true;
    messageOptionState.value.webSearch = true;
    messageOptionState.value.contextFiles = [{ id: "file-1", name: "brief.pdf" }];
    messageOptionState.value.selectedKnowledge = [
      { id: "knowledge-1", title: "Research notes" },
    ];
    messageOptionState.value.ragMediaIds = [101, 202];
  });

  it("surfaces existing context and runtime state in cockpit rails", async () => {
    const openKnowledgePanel = vi.fn();
    const openModelSettings = vi.fn();
    const openActorSettings = vi.fn();
    window.addEventListener("tldw:open-knowledge-panel", openKnowledgePanel);
    window.addEventListener("tldw:open-model-settings", openModelSettings);
    window.addEventListener("tldw:open-actor-settings", openActorSettings);

    try {
      render(<Playground />);

      const leftRail = await screen.findByTestId("playground-cockpit-left-rail");
      const contextRail = within(leftRail).getByTestId("playground-context-rail");
      expect(within(contextRail).getByText("Context active")).toBeInTheDocument();
      expect(within(contextRail).getByText("Web search")).toBeInTheDocument();
      expect(within(contextRail).getByText("1 file(s)")).toBeInTheDocument();
      expect(
        within(contextRail).getByText("1 knowledge item(s)"),
      ).toBeInTheDocument();
      expect(within(contextRail).getByText("2 media scope(s)")).toBeInTheDocument();
      expect(within(contextRail).getByText("Temporary chat")).toBeInTheDocument();
      expect(within(contextRail).getByText("History linked")).toBeInTheDocument();
      expect(within(contextRail).queryByTestId("web-search-toggle")).toBeNull();

      fireEvent.click(
        within(contextRail).getByRole("button", {
          name: /open search & context/i,
        }),
      );
      expect(openKnowledgePanel).toHaveBeenCalledTimes(1);

      const rightRail = screen.getByTestId("playground-cockpit-right-rail");
      const runtimeInspector = within(rightRail).getByTestId(
        "playground-runtime-inspector",
      );
      expect(within(runtimeInspector).getByText("Streaming")).toBeInTheDocument();
      expect(
        within(runtimeInspector).getByText("openai:gpt-4.1-mini"),
      ).toBeInTheDocument();
      expect(within(runtimeInspector).getByText("2 messages")).toBeInTheDocument();
      expect(within(runtimeInspector).getByText("Mira Vale")).toBeInTheDocument();

      fireEvent.click(
        within(runtimeInspector).getByRole("button", {
          name: /open model settings/i,
        }),
      );
      fireEvent.click(
        within(runtimeInspector).getByRole("button", {
          name: /open character settings/i,
        }),
      );

      expect(openModelSettings).toHaveBeenCalledTimes(1);
      expect(openActorSettings).toHaveBeenCalledTimes(1);
    } finally {
      window.removeEventListener("tldw:open-knowledge-panel", openKnowledgePanel);
      window.removeEventListener("tldw:open-model-settings", openModelSettings);
      window.removeEventListener("tldw:open-actor-settings", openActorSettings);
    }
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
});
