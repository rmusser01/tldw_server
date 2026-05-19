// @vitest-environment jsdom
import React from "react";
import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { Playground } from "../Playground";

const messageOptionState = vi.hoisted(() => ({
  value: {
    messages: [],
    history: [],
    historyId: "history-1",
    serverChatId: "chat-1",
    isLoading: false,
    setHistoryId: vi.fn(),
    setHistory: vi.fn(),
    setMessages: vi.fn(),
    setSelectedSystemPrompt: vi.fn(),
    setSelectedModel: vi.fn(),
    setServerChatId: vi.fn(),
    setContextFiles: vi.fn(),
    createChatBranch: vi.fn(),
    streaming: false,
    selectedCharacter: null,
    setSelectedCharacter: vi.fn(),
    compareMode: false,
    compareFeatureEnabled: false,
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

const smartScrollState = vi.hoisted(() => ({
  value: {
    containerRef: {
      current: null,
    } as React.MutableRefObject<HTMLDivElement | null>,
    isAutoScrollToBottom: true,
    autoScrollToBottom: vi.fn(),
  },
}));

const artifactsState = vi.hoisted(() => ({
  value: {
    isOpen: false,
    active: null,
    isPinned: false,
    history: [],
    unreadCount: 0,
    setOpen: vi.fn(),
    closeArtifact: vi.fn(),
    markRead: vi.fn(),
  },
}));

const storeOptionState = vi.hoisted(() => ({
  value: {
    compareParentByHistory: {} as Record<
      string,
      { parentHistoryId: string; clusterId?: string }
    >,
    setSelectedQuickPrompt: vi.fn(),
  },
}));

const storageState = vi.hoisted(() => ({
  value: {
    stickyChatInput: true,
  },
}));

const chatSettingsState = vi.hoisted(() => ({
  syncChatSettingsForServerChat: vi.fn(async () => null),
  applyChatSettingsPatch: vi.fn(async () => null),
}));

const researchClientMocks = vi.hoisted(() => ({
  initialize: vi.fn().mockResolvedValue(undefined),
  getResearchBundle: vi.fn().mockResolvedValue(null),
}));

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, defaultValue?: string) => defaultValue || key,
  }),
}));

vi.mock("@/components/Option/Playground/PlaygroundForm", () => ({
  PlaygroundForm: () => <div data-testid="playground-form">Mock composer</div>,
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
  useSmartScroll: () => smartScrollState.value,
}));

vi.mock("@/services/settings/ui-settings", () => ({
  CHAT_BACKGROUND_IMAGE_SETTING: "chatBackgroundImage",
}));

vi.mock("../Knowledge/utils/unsupported-types", () => ({
  otherUnsupportedTypes: [],
}));

vi.mock("@/store/option", () => ({
  useStoreMessageOption: (
    selector: (state: typeof storeOptionState.value) => unknown,
  ) => selector(storeOptionState.value),
}));

vi.mock("@/store/artifacts", () => ({
  useArtifactsStore: (
    selector: (state: typeof artifactsState.value) => unknown,
  ) => selector(artifactsState.value),
}));

vi.mock("@/hooks/useSetting", () => ({
  useSetting: () => [""],
}));

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (key: string, defaultValue: unknown) => {
    if (key === "stickyChatInput") {
      return [storageState.value.stickyChatInput];
    }
    return [defaultValue];
  },
}));

vi.mock("@/hooks/useMediaQuery", () => ({
  useMobile: () => false,
}));

vi.mock("@/services/chat-settings", () => ({
  syncChatSettingsForServerChat: (...args: unknown[]) =>
    chatSettingsState.syncChatSettingsForServerChat(...args),
  applyChatSettingsPatch: (...args: unknown[]) =>
    chatSettingsState.applyChatSettingsPatch(...args),
}));

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: researchClientMocks,
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
    useLocation: () => ({
      pathname: window.location.pathname || "/chat",
      search: window.location.search || "",
      hash: window.location.hash || "",
      state: null,
      key: "test-location",
    }),
  };
});

const findCenteredWidthContract = (
  root: HTMLElement,
): HTMLElement | undefined =>
  Array.from(root.querySelectorAll<HTMLElement>("div")).find(
    (node) =>
      typeof node.className === "string" &&
      node.className.includes("max-w-[64rem]"),
  );

describe("Playground sticky composer layout integration", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    storageState.value.stickyChatInput = true;
    storeOptionState.value.setSelectedQuickPrompt = vi.fn();
  });

  it("renders a separate transcript region and persistent composer dock when sticky chat input is enabled", () => {
    render(<Playground />);

    const transcript = screen.getByTestId("playground-chat-transcript");
    const dock = screen.getByTestId("playground-chat-composer-dock");

    expect(transcript).toBeInTheDocument();
    expect(dock).toBeInTheDocument();
    expect(dock.className).toContain("sticky");
    expect(dock.className).toContain("bottom-0");
    expect(
      transcript.compareDocumentPosition(dock) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    expect(findCenteredWidthContract(dock)).toBeDefined();
  });

  it("keeps the legacy non-docked composer branch when sticky chat input is disabled", () => {
    storageState.value.stickyChatInput = false;

    render(<Playground />);

    expect(
      screen.queryByTestId("playground-chat-composer-dock"),
    ).not.toBeInTheDocument();
    expect(screen.getByTestId("playground-form")).toBeInTheDocument();
  });
});
