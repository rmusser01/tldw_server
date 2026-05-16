// @vitest-environment jsdom
import React from "react";
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { Playground } from "../Playground";

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
    setSelectedSystemPrompt: vi.fn(),
    setSelectedModel: vi.fn(),
    setServerChatId: vi.fn(),
    setContextFiles: vi.fn(),
    createChatBranch: vi.fn(),
    streaming: false,
    selectedModel: "openai:gpt-4.1-mini",
    selectedCharacter: null,
    setSelectedCharacter: vi.fn(),
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

describe("Playground cockpit shell", () => {
  beforeEach(() => {
    storageState.values.clear();
    messageOptionState.value.messages = [];
    messageOptionState.value.history = [];
    messageOptionState.value.historyId = null;
    messageOptionState.value.serverChatId = null;
    messageOptionState.value.streaming = false;
    messageOptionState.value.selectedModel = "openai:gpt-4.1-mini";
    messageOptionState.value.regenerateLastMessage = vi.fn();
    sessionPersistenceState.value.sessionScopeReady = true;
  });

  it("renders the cockpit rails, main chat surface, and status strip by default", async () => {
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
      screen.getByTestId("playground-cockpit-status-strip"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("playground-cockpit-mode-summary"),
    ).toHaveTextContent("Context and runtime rails visible.");
    expect(screen.getByTestId("playground-chat")).toBeInTheDocument();
    expect(screen.getByTestId("playground-form")).toBeInTheDocument();
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
      screen.getByRole("button", { name: /show cockpit panels/i }),
    ).toBeInTheDocument();
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

    fireEvent.click(screen.getByRole("button", { name: /hide context rail/i }));

    await waitFor(() => {
      expect(screen.queryByTestId("playground-cockpit-left-rail")).toBeNull();
    });
    expect(
      screen.getByTestId("playground-cockpit-mode-summary"),
    ).toHaveTextContent("Context rail hidden. Runtime rail visible.");
    expect(
      screen.getByTestId("playground-cockpit-right-rail"),
    ).toBeInTheDocument();
    expect(storageState.values.get("playgroundChatContextRailVisible")).toBe(
      false,
    );
    expect(
      screen.getByRole("button", { name: /show context rail/i }),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /hide runtime rail/i }));

    await waitFor(() => {
      expect(screen.queryByTestId("playground-cockpit-right-rail")).toBeNull();
    });
    expect(
      screen.getByTestId("playground-cockpit-mode-summary"),
    ).toHaveTextContent("Cockpit rails hidden. Status remains visible.");
    expect(storageState.values.get("playgroundChatRuntimeRailVisible")).toBe(
      false,
    );
    expect(screen.getByTestId("playground-chat")).toBeInTheDocument();
    expect(screen.getByTestId("playground-form")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /show context rail/i }));
    fireEvent.click(screen.getByRole("button", { name: /show runtime rail/i }));

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
    const runtimeRail = screen.getByTestId("playground-cockpit-right-rail");

    fireEvent.click(
      within(contextRail).getByRole("button", {
        name: /collapse context sidechannel/i,
      }),
    );

    await waitFor(() => {
      expect(screen.queryByTestId("playground-cockpit-left-rail")).toBeNull();
    });
    expect(storageState.values.get("playgroundChatContextRailVisible")).toBe(
      false,
    );
    expect(screen.getByTestId("playground-chat")).toBeInTheDocument();
    expect(screen.getByTestId("playground-form")).toBeInTheDocument();

    fireEvent.click(
      screen.getByRole("button", { name: /restore context sidechannel/i }),
    );

    await waitFor(() => {
      expect(
        screen.getByTestId("playground-cockpit-left-rail"),
      ).toBeInTheDocument();
    });
    expect(storageState.values.get("playgroundChatContextRailVisible")).toBe(
      true,
    );

    fireEvent.click(
      within(runtimeRail).getByRole("button", {
        name: /collapse runtime sidechannel/i,
      }),
    );

    await waitFor(() => {
      expect(screen.queryByTestId("playground-cockpit-right-rail")).toBeNull();
    });
    expect(storageState.values.get("playgroundChatRuntimeRailVisible")).toBe(
      false,
    );
    expect(
      screen.getByTestId("playground-cockpit-status-strip"),
    ).toBeInTheDocument();

    fireEvent.click(
      screen.getByRole("button", { name: /restore runtime sidechannel/i }),
    );

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
    ).toHaveTextContent("No response text returned.");

    fireEvent.click(
      within(runControls).getByRole("button", {
        name: "Regenerate last response",
      }),
    );

    expect(messageOptionState.value.regenerateLastMessage).toHaveBeenCalledTimes(
      1,
    );
  });
});
