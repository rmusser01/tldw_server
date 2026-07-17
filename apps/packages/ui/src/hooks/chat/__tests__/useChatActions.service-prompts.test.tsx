// @vitest-environment jsdom
import React from "react";
import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { useChatActions } from "../useChatActions";

const {
  events,
  generateTitleMock,
  loadServicePromptSnapshotMock,
  normalChatModeMock,
  saveHistoryMock,
  saveMessageMock,
  tabChatModeMock,
} = vi.hoisted(() => ({
  events: [] as string[],
  generateTitleMock: vi.fn(),
  loadServicePromptSnapshotMock: vi.fn(),
  normalChatModeMock: vi.fn(),
  saveHistoryMock: vi.fn(),
  saveMessageMock: vi.fn(),
  tabChatModeMock: vi.fn(),
}));

vi.mock("@/services/service-prompts", () => ({
  loadServicePromptSnapshot: loadServicePromptSnapshotMock,
}));

vi.mock("@/hooks/chat-modes/normalChatMode", () => ({
  normalChatMode: normalChatModeMock,
}));

vi.mock("@/hooks/chat-modes/continueChatMode", () => ({
  continueChatMode: vi.fn(),
}));

vi.mock("@/hooks/chat-modes/ragMode", () => ({
  ragMode: vi.fn(),
}));

vi.mock("@/hooks/chat-modes/tabChatMode", () => ({
  tabChatMode: tabChatModeMock,
}));

vi.mock("@/hooks/chat-modes/documentChatMode", () => ({
  documentChatMode: vi.fn(),
}));

vi.mock("@/hooks/utils/messageHelpers", () => ({
  validateBeforeSubmit: vi.fn(() => true),
  createSaveMessageOnSuccess: vi.fn(
    () => async (): Promise<string | null> => "history-compare",
  ),
  createSaveMessageOnError: vi.fn(
    () => async (): Promise<string | null> => "history-compare",
  ),
}));

vi.mock("@/hooks/handlers/messageHandlers", () => ({
  createRegenerateLastMessage: vi.fn(() => vi.fn()),
  createEditMessage: vi.fn(() => vi.fn()),
  createStopStreamingRequest: vi.fn(() => vi.fn()),
  createBranchMessage: vi.fn(() => vi.fn()),
}));

vi.mock("@/db/dexie/helpers", () => ({
  generateID: vi.fn(() => "generated-id"),
  saveHistory: saveHistoryMock,
  saveMessage: saveMessageMock,
  updateHistory: vi.fn(),
  updateMessage: vi.fn(),
  updateMessageMedia: vi.fn(async () => null),
  removeMessageByIndex: vi.fn(),
  formatToChatHistory: vi.fn((items: unknown) => items),
  formatToMessage: vi.fn((items: unknown) => items),
  getSessionFiles: vi.fn(async () => []),
  getPromptById: vi.fn(async () => null),
}));

vi.mock("@/db/dexie/nickname", () => ({
  getModelNicknameByID: vi.fn(async () => null),
}));

vi.mock("@/db/dexie/branch", () => ({
  generateBranchFromMessageIds: vi.fn(async () => null),
}));

vi.mock("@/services/actor-settings", () => ({
  getActorSettingsForChat: vi.fn(async () => null),
}));

vi.mock("@/services/title", () => ({
  generateTitle: generateTitleMock,
}));

vi.mock("@/utils/update-page-title", () => ({
  updatePageTitle: vi.fn(),
}));

vi.mock("@/utils/selected-character-storage", () => ({
  SELECTED_CHARACTER_STORAGE_KEY: "selected_character",
  selectedCharacterStorage: {
    get: vi.fn(async () => null),
    set: vi.fn(async () => null),
  },
  selectedCharacterSyncStorage: {
    get: vi.fn(async () => null),
  },
  parseSelectedCharacterValue: vi.fn(() => null),
}));

vi.mock("@/hooks/chat/useChatSettingsRecord", () => ({
  useChatSettingsRecord: () => ({
    settings: {},
    updateSettings: vi.fn(),
  }),
}));

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: string, defaultValue: unknown) => {
    const [value] = React.useState(defaultValue);
    return [value, vi.fn()] as const;
  },
}));

vi.mock("@/store/option", () => ({
  useStoreMessageOption: {
    getState: () => ({ selectedModel: "model-a" as string | null }),
  },
}));

vi.mock("@/services/tldw/server-capabilities", () => ({
  getServerCapabilities: vi.fn(async () => ({ hasChatSaveToDb: false })),
}));

vi.mock("@/services/chat-settings", () => ({
  syncChatSettingsForServerChat: vi.fn(async () => null),
}));

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    createChat: vi.fn(),
    streamCharacterChatCompletion: vi.fn(),
    initialize: vi.fn(async () => null),
  },
}));

const snapshot = Object.freeze({
  scopeKey: "http://server.test|user:7",
  capability: "supported" as const,
  definitions: Object.freeze({
    "chat.web_search.answer": Object.freeze({
      definition: Object.freeze({
        id: "chat.web_search.answer",
        parts: Object.freeze([
          Object.freeze({
            key: "template",
            mode: "template" as const,
            required_variables: Object.freeze([
              "current_date_time",
              "search_results",
            ]),
          }),
        ]),
      }),
      parts: Object.freeze({ template: "revision-specific template" }),
      source: "user" as const,
      revision: "revision-compare-7",
    }),
  }),
});

const deferred = <T,>() => {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, reject, resolve };
};

const createHookOptions = ({
  webSearch = true,
  historyId = null,
  serverChatId = null,
}: {
  webSearch?: boolean;
  historyId?: string | null;
  serverChatId?: string | null;
} = {}) => ({
  t: (_key: string, fallback?: string) => fallback || _key,
  notification: {
    error: vi.fn(),
    warning: vi.fn(),
    info: vi.fn(),
    success: vi.fn(),
  },
  abortController: null,
  setAbortController: vi.fn(),
  messages: [],
  setMessages: vi.fn(() => events.push("setMessages")),
  history: [],
  setHistory: vi.fn(() => events.push("setHistory")),
  historyId,
  setHistoryId: vi.fn(),
  temporaryChat: false,
  selectedModel: "model-a",
  useOCR: false,
  selectedSystemPrompt: null,
  selectedKnowledge: null,
  toolChoice: "auto" as const,
  webSearch,
  currentChatModelSettings: {
    apiProvider: "openai",
    setSystemPrompt: vi.fn(),
  },
  setIsSearchingInternet: vi.fn(),
  setIsProcessing: vi.fn(),
  setStreaming: vi.fn(),
  setActionInfo: vi.fn(),
  fileRetrievalEnabled: false,
  ragMediaIds: null,
  ragSearchMode: "hybrid" as const,
  ragTopK: 8,
  ragEnableGeneration: true,
  ragEnableCitations: true,
  ragSources: [],
  ragAdvancedOptions: {},
  serverChatId,
  serverChatTitle: null,
  serverChatCharacterId: null,
  serverChatAssistantKind: null,
  serverChatAssistantId: null,
  serverChatPersonaMemoryMode: null,
  serverChatMetaLoaded: false,
  serverChatState: "in-progress" as const,
  serverChatTopic: null,
  serverChatClusterId: null,
  serverChatSource: null,
  serverChatExternalRef: null,
  setServerChatId: vi.fn(),
  setServerChatTitle: vi.fn(),
  setServerChatCharacterId: vi.fn(),
  setServerChatAssistantKind: vi.fn(),
  setServerChatAssistantId: vi.fn(),
  setServerChatPersonaMemoryMode: vi.fn(),
  setServerChatMetaLoaded: vi.fn(),
  setServerChatState: vi.fn(),
  setServerChatVersion: vi.fn(),
  setServerChatTopic: vi.fn(),
  setServerChatClusterId: vi.fn(),
  setServerChatSource: vi.fn(),
  setServerChatExternalRef: vi.fn(),
  ensureServerChatHistoryId: vi.fn(async () => historyId),
  contextFiles: [],
  setContextFiles: vi.fn(),
  documentContext: null,
  setDocumentContext: vi.fn(),
  uploadedFiles: [],
  compareModeActive: true,
  compareSelectedModels: ["model-a", "model-b"],
  compareMaxModels: 3,
  compareFeatureEnabled: true,
  markCompareHistoryCreated: vi.fn(),
  replyTarget: null,
  clearReplyTarget: vi.fn(),
  messageSteeringPrompts: null,
  setSelectedQuickPrompt: vi.fn(),
  setSelectedSystemPrompt: vi.fn(),
  invalidateServerChatHistory: vi.fn(),
  selectedCharacter: null,
  selectedAssistant: null,
  messageSteeringMode: "none" as const,
  messageSteeringForceNarrate: false,
  clearMessageSteering: vi.fn(),
});

describe("useChatActions Compare service prompt snapshot", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    events.length = 0;
    loadServicePromptSnapshotMock.mockImplementation(async () => {
      events.push("loadSnapshot");
      return snapshot;
    });
    normalChatModeMock.mockImplementation(async (...args: unknown[]) => {
      const params = args[6] as { selectedModel: string };
      events.push(`normalChatMode:${params.selectedModel}`);
    });
    generateTitleMock.mockImplementation(async () => {
      events.push("generateTitle");
      return "Compare title";
    });
    saveHistoryMock.mockImplementation(async () => {
      events.push("saveHistory");
      return { id: "history-compare" };
    });
    saveMessageMock.mockImplementation(async () => {
      events.push("saveMessage");
    });
  });

  it("loads before shared side effects and gives every web-search branch the same snapshot", async () => {
    const options = createHookOptions();
    const { result } = renderHook(() =>
      useChatActions(
        options as unknown as Parameters<typeof useChatActions>[0],
      ),
    );
    events.length = 0;
    options.setMessages.mockClear();
    options.setHistory.mockClear();

    await act(async () => {
      await result.current.onSubmit({ message: "Compare this", image: "" });
    });

    expect(loadServicePromptSnapshotMock).toHaveBeenCalledTimes(1);
    expect(loadServicePromptSnapshotMock).toHaveBeenCalledWith(
      ["chat.web_search.answer"],
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    );
    for (const sideEffect of [
      "setMessages",
      "setHistory",
      "generateTitle",
      "saveHistory",
      "saveMessage",
      "normalChatMode:model-a",
      "normalChatMode:model-b",
    ]) {
      expect(events.indexOf("loadSnapshot")).toBeLessThan(
        events.indexOf(sideEffect),
      );
    }

    expect(normalChatModeMock).toHaveBeenCalledTimes(2);
    const firstParams = normalChatModeMock.mock.calls[0]?.[6];
    const secondParams = normalChatModeMock.mock.calls[1]?.[6];
    expect(firstParams.servicePromptSnapshot).toBe(snapshot);
    expect(secondParams.servicePromptSnapshot).toBe(snapshot);
    expect(
      firstParams.servicePromptSnapshot.definitions["chat.web_search.answer"]
        .revision,
    ).toBe("revision-compare-7");
  });

  it("does not create a shared user message or history when snapshot loading fails", async () => {
    loadServicePromptSnapshotMock.mockRejectedValueOnce(
      new Error("Workflow prompts are unavailable"),
    );
    const options = createHookOptions();
    const { result } = renderHook(() =>
      useChatActions(
        options as unknown as Parameters<typeof useChatActions>[0],
      ),
    );
    events.length = 0;
    options.setMessages.mockClear();
    options.setHistory.mockClear();

    await act(async () => {
      await result.current.onSubmit({ message: "Compare this", image: "" });
    });

    expect(options.setMessages).not.toHaveBeenCalled();
    expect(options.setHistory).not.toHaveBeenCalled();
    expect(generateTitleMock).not.toHaveBeenCalled();
    expect(saveHistoryMock).not.toHaveBeenCalled();
    expect(saveMessageMock).not.toHaveBeenCalled();
    expect(normalChatModeMock).not.toHaveBeenCalled();
  });

  it("gates server-backed Compare history creation on the exact turn signal", async () => {
    const pendingSnapshot = deferred<typeof snapshot>();
    loadServicePromptSnapshotMock.mockReturnValueOnce(pendingSnapshot.promise);
    const options = createHookOptions({ serverChatId: "server-chat-7" });
    const { result } = renderHook(() =>
      useChatActions(
        options as unknown as Parameters<typeof useChatActions>[0],
      ),
    );
    events.length = 0;
    options.setMessages.mockClear();
    options.setHistory.mockClear();
    const controller = new AbortController();
    let submission!: ReturnType<typeof result.current.onSubmit>;

    await act(async () => {
      submission = result.current.onSubmit({
        message: "Compare this",
        image: "",
        controller,
      });
      await vi.waitFor(() => {
        expect(loadServicePromptSnapshotMock).toHaveBeenCalledTimes(1);
      });
    });

    expect(loadServicePromptSnapshotMock).toHaveBeenCalledWith(
      ["chat.web_search.answer"],
      { signal: controller.signal },
    );
    expect(options.ensureServerChatHistoryId).not.toHaveBeenCalled();
    expect(options.setMessages).not.toHaveBeenCalled();
    expect(options.setHistory).not.toHaveBeenCalled();
    expect(generateTitleMock).not.toHaveBeenCalled();
    expect(saveHistoryMock).not.toHaveBeenCalled();
    expect(saveMessageMock).not.toHaveBeenCalled();

    await act(async () => {
      pendingSnapshot.reject(new Error("Workflow prompts are unavailable"));
      await submission;
    });

    expect(options.ensureServerChatHistoryId).not.toHaveBeenCalled();
    expect(options.setMessages).not.toHaveBeenCalled();
    expect(options.setHistory).not.toHaveBeenCalled();
    expect(generateTitleMock).not.toHaveBeenCalled();
    expect(saveHistoryMock).not.toHaveBeenCalled();
    expect(saveMessageMock).not.toHaveBeenCalled();
    expect(normalChatModeMock).not.toHaveBeenCalled();
  });

  it("uses the ensured server history only after the snapshot gate succeeds", async () => {
    const pendingSnapshot = deferred<typeof snapshot>();
    loadServicePromptSnapshotMock.mockReturnValueOnce(pendingSnapshot.promise);
    const options = createHookOptions({ serverChatId: "server-chat-7" });
    options.ensureServerChatHistoryId.mockImplementation(async () => {
      events.push("ensureHistory");
      return "history-from-server";
    });
    const { result } = renderHook(() =>
      useChatActions(
        options as unknown as Parameters<typeof useChatActions>[0],
      ),
    );
    events.length = 0;
    options.setMessages.mockClear();
    options.setHistory.mockClear();
    const controller = new AbortController();
    let submission!: ReturnType<typeof result.current.onSubmit>;

    await act(async () => {
      submission = result.current.onSubmit({
        message: "Compare this",
        image: "",
        controller,
      });
      await vi.waitFor(() => {
        expect(loadServicePromptSnapshotMock).toHaveBeenCalledTimes(1);
      });
    });

    expect(loadServicePromptSnapshotMock).toHaveBeenCalledWith(
      ["chat.web_search.answer"],
      { signal: controller.signal },
    );
    expect(options.ensureServerChatHistoryId).not.toHaveBeenCalled();
    expect(options.setMessages).not.toHaveBeenCalled();
    expect(options.setHistory).not.toHaveBeenCalled();
    expect(saveMessageMock).not.toHaveBeenCalled();

    await act(async () => {
      pendingSnapshot.resolve(snapshot);
      await submission;
    });

    expect(options.ensureServerChatHistoryId).toHaveBeenCalledTimes(1);
    expect(events.indexOf("loadSnapshot")).toBeLessThan(
      events.indexOf("ensureHistory"),
    );
    for (const sideEffect of ["setMessages", "setHistory", "saveMessage"]) {
      expect(events.indexOf("ensureHistory")).toBeLessThan(
        events.indexOf(sideEffect),
      );
    }
    expect(generateTitleMock).not.toHaveBeenCalled();
    expect(saveHistoryMock).not.toHaveBeenCalled();
    expect(saveMessageMock).toHaveBeenCalledWith(
      expect.objectContaining({ history_id: "history-from-server" }),
    );
    expect(normalChatModeMock).toHaveBeenCalledTimes(2);
    const firstParams = normalChatModeMock.mock.calls[0]?.[6];
    const secondParams = normalChatModeMock.mock.calls[1]?.[6];
    expect(firstParams.historyId).toBe("history-from-server");
    expect(secondParams.historyId).toBe("history-from-server");
    expect(firstParams.servicePromptSnapshot).toBe(snapshot);
    expect(secondParams.servicePromptSnapshot).toBe(snapshot);
  });

  it("rejects a loaded snapshot missing the required Compare prompt before side effects", async () => {
    loadServicePromptSnapshotMock.mockResolvedValueOnce(
      Object.freeze({
        ...snapshot,
        definitions: Object.freeze({}),
      }),
    );
    const options = createHookOptions({ serverChatId: "server-chat-7" });
    const { result } = renderHook(() =>
      useChatActions(
        options as unknown as Parameters<typeof useChatActions>[0],
      ),
    );
    events.length = 0;
    options.setMessages.mockClear();
    options.setHistory.mockClear();

    await act(async () => {
      await result.current.onSubmit({ message: "Compare this", image: "" });
    });

    expect(options.ensureServerChatHistoryId).not.toHaveBeenCalled();
    expect(options.setMessages).not.toHaveBeenCalled();
    expect(options.setHistory).not.toHaveBeenCalled();
    expect(generateTitleMock).not.toHaveBeenCalled();
    expect(saveHistoryMock).not.toHaveBeenCalled();
    expect(saveMessageMock).not.toHaveBeenCalled();
    expect(normalChatModeMock).not.toHaveBeenCalled();
  });

  it("does not read a Service Prompt and preserves server history when Compare web search is disabled", async () => {
    const options = createHookOptions({
      webSearch: false,
      serverChatId: "server-chat-7",
    });
    options.ensureServerChatHistoryId.mockResolvedValueOnce(
      "history-from-server",
    );
    const { result } = renderHook(() =>
      useChatActions(
        options as unknown as Parameters<typeof useChatActions>[0],
      ),
    );
    events.length = 0;
    options.setMessages.mockClear();
    options.setHistory.mockClear();

    await act(async () => {
      await result.current.onSubmit({ message: "Compare this", image: "" });
    });

    expect(loadServicePromptSnapshotMock).not.toHaveBeenCalled();
    expect(options.ensureServerChatHistoryId).toHaveBeenCalledTimes(1);
    expect(saveMessageMock).toHaveBeenCalledWith(
      expect.objectContaining({ history_id: "history-from-server" }),
    );
    expect(normalChatModeMock).toHaveBeenCalledTimes(2);
    expect(normalChatModeMock.mock.calls[0]?.[6].historyId).toBe(
      "history-from-server",
    );
    expect(normalChatModeMock.mock.calls[1]?.[6].historyId).toBe(
      "history-from-server",
    );
  });

  it("uses the request-level web-search override for the Compare prompt gate", async () => {
    const disabledOptions = createHookOptions({
      webSearch: true,
      historyId: "history-1",
    });
    const disabledHook = renderHook(() =>
      useChatActions(
        disabledOptions as unknown as Parameters<typeof useChatActions>[0],
      ),
    );

    await act(async () => {
      await disabledHook.result.current.onSubmit({
        message: "Compare without search",
        image: "",
        requestOverrides: { webSearch: false },
      });
    });

    expect(loadServicePromptSnapshotMock).not.toHaveBeenCalled();

    vi.clearAllMocks();
    events.length = 0;
    loadServicePromptSnapshotMock.mockResolvedValue(snapshot);
    normalChatModeMock.mockResolvedValue(undefined);
    const enabledOptions = createHookOptions({
      webSearch: false,
      historyId: "history-1",
    });
    const enabledHook = renderHook(() =>
      useChatActions(
        enabledOptions as unknown as Parameters<typeof useChatActions>[0],
      ),
    );

    await act(async () => {
      await enabledHook.result.current.onSubmit({
        message: "Compare with search",
        image: "",
        requestOverrides: { webSearch: true },
      });
    });

    expect(loadServicePromptSnapshotMock).toHaveBeenCalledTimes(1);
  });

  it("passes stored tab context when request-level docs is an empty array", async () => {
    const storedTabs = [
      {
        type: "tab" as const,
        tabId: 7,
        title: "Stored tab",
        url: "https://example.test/stored-tab",
      },
    ];
    const options = {
      ...createHookOptions({ webSearch: false, historyId: "history-1" }),
      compareModeActive: false,
      documentContext: storedTabs,
    };
    const { result } = renderHook(() =>
      useChatActions(
        options as unknown as Parameters<typeof useChatActions>[0],
      ),
    );

    await act(async () => {
      await result.current.onSubmit({
        message: "Use the stored tab",
        image: "",
        docs: [],
      });
    });

    expect(tabChatModeMock).toHaveBeenCalledTimes(1);
    expect(tabChatModeMock.mock.calls[0]?.[2]).toEqual(storedTabs);
  });
});
