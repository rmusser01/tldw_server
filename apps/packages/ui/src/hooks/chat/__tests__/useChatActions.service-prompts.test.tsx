// @vitest-environment jsdom
import React from "react";
import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useChatActions } from "../useChatActions";
import {
  HistorySelectionContext,
  useHistorySelection,
} from "../useHistorySelection";
import { useStoreMessageOption } from "@/store/option";
import type { ChatHistory, Message } from "@/store/option";

const {
  addChatMessageMock,
  events,
  createChatMock,
  continueChatModeMock,
  continuationStoreBoundary,
  documentChatModeMock,
  generateTitleMock,
  loadServicePromptSnapshotMock,
  normalChatModeMock,
  ragModeMock,
  releaseSnapshotMock,
  rollbackScopedComparePersistenceMock,
  runChatPersistenceTransactionMock,
  saveHistoryMock,
  saveMessageMock,
  streamCharacterChatCompletionMock,
  setLastUsedModelMock,
  setLastUsedPromptMock,
  tabChatModeMock,
  updateCreatedAtMock,
  updateMessageMock,
  updatePageTitleMock,
} = vi.hoisted(() => ({
  addChatMessageMock: vi.fn(),
  events: [] as string[],
  createChatMock: vi.fn(),
  continueChatModeMock:
    vi.fn<
      typeof import("@/hooks/chat-modes/continueChatMode").continueChatMode
    >(),
  continuationStoreBoundary: { actual: false, actualSave: false },
  documentChatModeMock: vi.fn(),
  generateTitleMock: vi.fn(),
  loadServicePromptSnapshotMock: vi.fn(),
  normalChatModeMock: vi.fn(),
  ragModeMock: vi.fn(),
  releaseSnapshotMock: vi.fn(),
  rollbackScopedComparePersistenceMock: vi.fn(async () => undefined),
  runChatPersistenceTransactionMock: vi.fn(
    async (
      _signal: AbortSignal | undefined,
      operation: () => Promise<unknown>,
    ) => operation(),
  ),
  saveHistoryMock: vi.fn(),
  saveMessageMock: vi.fn(),
  streamCharacterChatCompletionMock: vi.fn(),
  setLastUsedModelMock: vi.fn(async () => undefined),
  setLastUsedPromptMock: vi.fn(async () => undefined),
  tabChatModeMock: vi.fn(),
  updateCreatedAtMock: vi.fn(async () => undefined),
  updateMessageMock: vi.fn(async () => undefined),
  updatePageTitleMock: vi.fn(),
}));

vi.mock("@/services/service-prompts", () => ({
  loadServicePromptSnapshot: loadServicePromptSnapshotMock,
}));

vi.mock("@/hooks/chat-modes/normalChatMode", () => ({
  normalChatMode: normalChatModeMock,
}));

vi.mock("@/hooks/chat-modes/continueChatMode", () => ({
  continueChatMode: continueChatModeMock,
}));

vi.mock("@/hooks/chat-modes/ragMode", () => ({
  ragMode: ragModeMock,
}));

vi.mock("@/hooks/chat-modes/tabChatMode", () => ({
  tabChatMode: tabChatModeMock,
}));

vi.mock("@/hooks/chat-modes/documentChatMode", () => ({
  documentChatMode: documentChatModeMock,
}));

vi.mock("@/hooks/utils/messageHelpers", async (original) => {
  const actual =
    await original<typeof import("@/hooks/utils/messageHelpers")>();
  return {
    ...actual,
    validateBeforeSubmit: vi.fn(() => true),
    createSaveMessageOnSuccess: vi.fn(
      (...args: Parameters<typeof actual.createSaveMessageOnSuccess>) =>
        continuationStoreBoundary.actualSave
          ? actual.createSaveMessageOnSuccess(...args)
          : async (): Promise<string | null> => "history-compare",
    ),
    createSaveMessageOnError: vi.fn(
      () => async (): Promise<string | null> => "history-compare",
    ),
  };
});

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
  updateMessage: updateMessageMock,
  updateMessageMedia: vi.fn(async () => null),
  removeMessageByIndex: vi.fn(),
  formatToChatHistory: vi.fn((items: unknown) => items),
  formatToMessage: vi.fn((items: unknown) => items),
  getSessionFiles: vi.fn(async () => []),
  getPromptById: vi.fn(async () => null),
  updateLastUsedModel: setLastUsedModelMock,
  updateLastUsedPrompt: setLastUsedPromptMock,
  updateChatHistoryCreatedAt: updateCreatedAtMock,
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
  updatePageTitle: updatePageTitleMock,
}));

vi.mock("@/db/dexie/chat-persistence-transaction", () => ({
  rollbackScopedComparePersistence: rollbackScopedComparePersistenceMock,
  runChatPersistenceTransaction: runChatPersistenceTransactionMock,
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

vi.mock("@/store/option", async (original) => {
  const actual = await original<typeof import("@/store/option")>();
  return {
    ...actual,
    useStoreMessageOption: Object.assign(actual.useStoreMessageOption, {
      getState: (() => {
        const readActual = actual.useStoreMessageOption.getState;
        return () =>
          continuationStoreBoundary.actual
            ? readActual()
            : { selectedModel: "model-a" as string | null };
      })(),
    }),
  };
});

vi.mock("@/services/tldw/server-capabilities", () => ({
  getServerCapabilities: vi.fn(async () => ({ hasChatSaveToDb: false })),
}));

vi.mock("@/services/chat-settings", () => ({
  syncChatSettingsForServerChat: vi.fn(async () => null),
}));

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    addChatMessage: addChatMessageMock,
    createChat: createChatMock,
    streamCharacterChatCompletion: streamCharacterChatCompletionMock,
    initialize: vi.fn(async () => null),
  },
}));

const snapshot = Object.freeze({
  scopeKey: "http://server.test|user:7",
  requestScope: Object.freeze({
    config: Object.freeze({
      serverUrl: "http://server.test",
      authMode: "multi-user" as const,
      authSource: "manual" as const,
      orgId: 3,
    }),
    userId: 7,
  }),
  capability: "supported" as const,
  scopeSignal: new AbortController().signal,
  scopeInvalidatedSignal: new AbortController().signal,
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
    "chat.rag.answer": Object.freeze({
      definition: Object.freeze({
        id: "chat.rag.answer",
        parts: Object.freeze([
          Object.freeze({
            key: "template",
            mode: "template" as const,
            required_variables: Object.freeze(["context", "question"]),
          }),
        ]),
      }),
      parts: Object.freeze({
        template: "Context: {context}\nQuestion: {question}",
      }),
      source: "packaged" as const,
      revision: null,
    }),
    "chat.rag.question_rewrite": Object.freeze({
      definition: Object.freeze({
        id: "chat.rag.question_rewrite",
        parts: Object.freeze([
          Object.freeze({
            key: "template",
            mode: "template" as const,
            required_variables: Object.freeze(["question"]),
          }),
        ]),
      }),
      parts: Object.freeze({ template: "Rewrite: {question}" }),
      source: "packaged" as const,
      revision: null,
    }),
  }),
  release: releaseSnapshotMock,
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

const rejectWhenAborted = (signal: AbortSignal): Promise<never> =>
  new Promise((_, reject) => {
    const rejectAbort = () => {
      const error = new Error("Service Prompt request was aborted.");
      error.name = "AbortError";
      reject(error);
    };
    if (signal.aborted) {
      rejectAbort();
      return;
    }
    signal.addEventListener("abort", rejectAbort, { once: true });
  });

const createAbortError = () => {
  const error = new Error("AbortError");
  error.name = "AbortError";
  return error;
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
    documentChatModeMock.mockResolvedValue(undefined);
    ragModeMock.mockResolvedValue(undefined);
    createChatMock.mockResolvedValue({ id: "server-chat-created" });
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

  it.each([false, true])(
    "stamps a new persisted Compare with its captured owner (webSearch=%s)",
    async (webSearch) => {
      const options = createHookOptions({ webSearch });
      const { result } = renderHook(() =>
        useChatActions(
          options as unknown as Parameters<typeof useChatActions>[0],
        ),
      );
      await act(async () => {
        await result.current.onSubmit({
          message: "Compare my private question",
          image: "",
        });
      });
      expect(saveHistoryMock).toHaveBeenCalledWith(
        expect.any(String),
        false,
        "web-ui",
        undefined,
        undefined,
        snapshot.requestScope,
      );
      expect(
        normalChatModeMock.mock.calls.every(
          (call) => call[6].servicePromptSnapshot === snapshot,
        ),
      ).toBe(true);
    },
  );

  it.each([false, true])(
    "does not dispatch normal or Compare modes after the actual scene reader fails (compare=%s)",
    async (compare) => {
      const { Storage } = await import("@plasmohq/storage");
      const service = await vi.importActual<
        typeof import("@/services/actor-settings")
      >("@/services/actor-settings");
      const reader = vi.mocked(
        (await import("@/services/actor-settings")).getActorSettingsForChat
      );
      reader.mockImplementationOnce(service.getActorSettingsForChat);
      const read = vi
        .spyOn(Storage.prototype, "get")
        .mockRejectedValueOnce(new Error("scene read unavailable"));
      const options = {
        ...createHookOptions({ webSearch: false }),
        compareModeActive: compare
      };
      const { result } = renderHook(() =>
        useChatActions(options as unknown as Parameters<typeof useChatActions>[0])
      );
      let outcome: unknown;
      await act(async () => {
        outcome = await result.current.onSubmit({
          message: "Retained exact question",
          image: ""
        });
      });
      expect(outcome).toMatchObject({
        status: "failed",
        errorMessage: "actor_settings_unavailable"
      });
      expect(options.notification.error).toHaveBeenCalledWith(
        expect.objectContaining({
          description:
            "Scene settings could not be loaded. Existing chat setup was left unchanged."
        })
      );
      expect(normalChatModeMock).not.toHaveBeenCalled();
      expect(ragModeMock).not.toHaveBeenCalled();
      expect(createChatMock).not.toHaveBeenCalled();
      read.mockRestore();
    }
  );
  it.each([false, true])("retires a captured turn during real prompt preparation (rag=%s)", async (rag) => {
    const pending = deferred<typeof snapshot>();
    loadServicePromptSnapshotMock.mockReturnValueOnce(pending.promise);
    const options = { ...createHookOptions({ webSearch: false }), compareModeActive: false, ragMediaIds: rag ? [101] : null, fileRetrievalEnabled: rag };
    const { result } = renderHook(() => useChatActions(options as unknown as Parameters<typeof useChatActions>[0]));
    const controller = new AbortController();
    let submission!: ReturnType<typeof result.current.onSubmit>;
    await act(async () => {
      submission = result.current.onSubmit({ message: "Captured question", image: "", controller });
      await vi.waitFor(() => expect(loadServicePromptSnapshotMock).toHaveBeenCalledOnce());
    });
    controller.abort();
    await act(async () => { pending.resolve(snapshot); await submission; });
    expect(normalChatModeMock).not.toHaveBeenCalled();
    expect(ragModeMock).not.toHaveBeenCalled();
    expect(createChatMock).not.toHaveBeenCalled();
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
    expect(releaseSnapshotMock).toHaveBeenCalledTimes(1);
  });

  it("binds title and the atomic shared-user commit to the captured Compare scope", async () => {
    const options = createHookOptions();
    const { result } = renderHook(() =>
      useChatActions(
        options as unknown as Parameters<typeof useChatActions>[0],
      ),
    );

    await act(async () => {
      await result.current.onSubmit({ message: "Compare this", image: "" });
    });

    expect(generateTitleMock).toHaveBeenCalledWith(
      "model-a",
      "Compare this",
      "Compare this",
      {
        signal: snapshot.scopeSignal,
        requestScope: snapshot.requestScope,
      },
    );
    expect(runChatPersistenceTransactionMock).toHaveBeenCalledWith(
      snapshot.scopeSignal,
      expect.any(Function),
    );
  });

  it("leaves no shared UI or title state when scoped title generation returns 412", async () => {
    const scopeChangedError = Object.assign(new Error("scope changed"), {
      status: 412,
      details: { code: "request_config_scope_changed" },
    });
    generateTitleMock.mockRejectedValueOnce(scopeChangedError);
    const options = createHookOptions();
    const { result } = renderHook(() =>
      useChatActions(
        options as unknown as Parameters<typeof useChatActions>[0],
      ),
    );
    options.setMessages.mockClear();
    options.setHistory.mockClear();

    await act(async () => {
      await result.current.onSubmit({ message: "Compare this", image: "" });
    });

    expect(options.setMessages).not.toHaveBeenCalled();
    expect(options.setHistory).not.toHaveBeenCalled();
    expect(runChatPersistenceTransactionMock).not.toHaveBeenCalled();
    expect(saveHistoryMock).not.toHaveBeenCalled();
    expect(saveMessageMock).not.toHaveBeenCalled();
    expect(updatePageTitleMock).not.toHaveBeenCalled();
    expect(options.setHistoryId).not.toHaveBeenCalled();
    expect(options.markCompareHistoryCreated).not.toHaveBeenCalled();
    expect(normalChatModeMock).not.toHaveBeenCalled();
  });

  it("leaves no shared UI or title state when the scoped atomic commit aborts", async () => {
    const abortError = new Error("scope changed");
    abortError.name = "AbortError";
    runChatPersistenceTransactionMock.mockRejectedValueOnce(abortError);
    const options = createHookOptions();
    const { result } = renderHook(() =>
      useChatActions(
        options as unknown as Parameters<typeof useChatActions>[0],
      ),
    );
    options.setMessages.mockClear();
    options.setHistory.mockClear();

    await act(async () => {
      await result.current.onSubmit({ message: "Compare this", image: "" });
    });

    expect(options.setMessages).not.toHaveBeenCalled();
    expect(options.setHistory).not.toHaveBeenCalled();
    expect(saveHistoryMock).not.toHaveBeenCalled();
    expect(saveMessageMock).not.toHaveBeenCalled();
    expect(updatePageTitleMock).not.toHaveBeenCalled();
    expect(options.setHistoryId).not.toHaveBeenCalled();
    expect(options.markCompareHistoryCreated).not.toHaveBeenCalled();
    expect(normalChatModeMock).not.toHaveBeenCalled();
  });

  it("rolls back the shared UI turn when any Compare branch rejects the request scope", async () => {
    normalChatModeMock.mockImplementation(async (...args: unknown[]) => {
      const params = args[6] as {
        selectedModel: string;
        saveMessageOnSuccess: (
          payload: Record<string, unknown>,
        ) => Promise<string | null>;
      };
      await params.saveMessageOnSuccess({
        historyId: "history-compare",
        isRegenerate: true,
        selectedModel: params.selectedModel,
        message: "Compare this",
        image: "",
        fullText: `${params.selectedModel} answer`,
        source: [],
        modelId: params.selectedModel,
        assistantMessageId: `${params.selectedModel}-assistant`,
        reasoning_time_taken: 0,
        prompt_id: "prompt-1",
        scopeSignal: snapshot.scopeSignal,
        scopeInvalidatedSignal: snapshot.scopeInvalidatedSignal,
        requestScope: snapshot.requestScope,
      });
      return params.selectedModel === "model-b"
        ? { status: "skipped", reason: "Request scope changed" }
        : { status: "submitted" };
    });
    const options = createHookOptions();
    const { result } = renderHook(() =>
      useChatActions(
        options as unknown as Parameters<typeof useChatActions>[0],
      ),
    );
    options.setMessages.mockClear();
    options.setHistory.mockClear();

    let submissionResult: Awaited<ReturnType<typeof result.current.onSubmit>>;
    await act(async () => {
      submissionResult = await result.current.onSubmit({
        message: "Compare this",
        image: "",
      });
    });

    expect(submissionResult!).toEqual({
      status: "skipped",
      reason: "Request scope changed",
    });
    expect(options.setMessages).toHaveBeenLastCalledWith(options.messages);
    expect(options.setHistory).toHaveBeenLastCalledWith(options.history);
    expect(options.setHistory).toHaveBeenCalledTimes(2);
    expect(rollbackScopedComparePersistenceMock).toHaveBeenCalledWith({
      clusterId: "generated-id",
      historyId: "history-compare",
      removeHistory: true,
    });
    expect(updatePageTitleMock).not.toHaveBeenCalled();
    expect(options.setHistoryId).not.toHaveBeenCalled();
    expect(options.markCompareHistoryCreated).not.toHaveBeenCalled();
    expect(setLastUsedModelMock).not.toHaveBeenCalled();
    expect(setLastUsedPromptMock).not.toHaveBeenCalled();
    expect(updateCreatedAtMock).not.toHaveBeenCalled();
  });

  it("removes only the rejected Compare cluster from an existing history", async () => {
    normalChatModeMock.mockImplementation(async (...args: unknown[]) => {
      const params = args[6] as { selectedModel: string };
      return params.selectedModel === "model-b"
        ? { status: "skipped", reason: "Request scope changed" }
        : { status: "submitted" };
    });
    const options = createHookOptions({ historyId: "history-existing" });
    const { result } = renderHook(() =>
      useChatActions(
        options as unknown as Parameters<typeof useChatActions>[0],
      ),
    );

    await act(async () => {
      await result.current.onSubmit({ message: "Compare this", image: "" });
    });

    expect(rollbackScopedComparePersistenceMock).toHaveBeenCalledWith({
      clusterId: "generated-id",
      historyId: "history-existing",
      removeHistory: false,
    });
  });

  it("does not publish old Compare branches or rollback over a replacement account", async () => {
    const owner = new AbortController();
    loadServicePromptSnapshotMock.mockResolvedValueOnce({
      ...snapshot,
      scopeSignal: owner.signal,
      scopeInvalidatedSignal: owner.signal,
    });
    let finish!: () => void;
    const held = new Promise<void>((resolve) => {
      finish = resolve;
    });
    const aliceMessages = [
      {
        id: "alice-private",
        role: "user",
        message: "Alice private transcript",
        isBot: false,
        sources: [],
      },
    ];
    const aliceHistory = [
      { role: "user", content: "Alice private transcript" },
    ];
    let currentMessages: unknown = aliceMessages;
    let currentHistory: unknown = aliceHistory;
    const options = {
      ...createHookOptions({ webSearch: false }),
      messages: aliceMessages,
      history: aliceHistory,
      setMessages: vi.fn((next: unknown) => {
        currentMessages = next;
      }),
      setHistory: vi.fn((next: unknown) => {
        currentHistory = next;
      }),
    };
    normalChatModeMock.mockImplementation(async (...args: unknown[]) => {
      const params = args[6] as {
        setMessages: (next: unknown) => void;
        setHistory: (next: unknown) => void;
      };
      await held;
      params.setMessages(aliceMessages);
      params.setHistory(aliceHistory);
      return { status: "skipped", reason: "Request scope changed" };
    });
    const { result } = renderHook(() =>
      useChatActions(
        options as unknown as Parameters<typeof useChatActions>[0],
      ),
    );
    let submission!: ReturnType<typeof result.current.onSubmit>;
    act(() => {
      submission = result.current.onSubmit({
        message: "Compare this",
        image: "",
      });
    });
    await waitFor(() => expect(normalChatModeMock).toHaveBeenCalledTimes(2));
    const bobMessages = [{ message: "Bob replacement transcript" }];
    const bobHistory = [
      { role: "user", content: "Bob replacement transcript" },
    ];
    currentMessages = bobMessages;
    currentHistory = bobHistory;
    options.setAbortController.mockClear();
    options.setIsProcessing.mockClear();
    options.setStreaming.mockClear();
    await act(async () => {
      owner.abort();
      finish();
      await submission;
    });
    expect(currentMessages).toBe(bobMessages);
    expect(currentHistory).toBe(bobHistory);
    expect(options.setAbortController).not.toHaveBeenCalled();
    expect(options.setIsProcessing).not.toHaveBeenCalled();
    expect(options.setStreaming).not.toHaveBeenCalled();
    expect(rollbackScopedComparePersistenceMock).toHaveBeenCalled();
  });

  it("applies existing-history metadata only after every scoped Compare branch succeeds", async () => {
    normalChatModeMock.mockImplementation(async (...args: unknown[]) => {
      const params = args[6] as {
        selectedModel: string;
        saveMessageOnSuccess: (
          payload: Record<string, unknown>,
        ) => Promise<string | null>;
      };
      await params.saveMessageOnSuccess({
        historyId: "history-existing",
        isRegenerate: true,
        selectedModel: params.selectedModel,
        message: "Compare this",
        image: "",
        fullText: `${params.selectedModel} answer`,
        source: [],
        modelId: params.selectedModel,
        assistantMessageId: `${params.selectedModel}-assistant`,
        reasoning_time_taken: 0,
        prompt_id: "prompt-1",
        scopeSignal: snapshot.scopeSignal,
        scopeInvalidatedSignal: snapshot.scopeInvalidatedSignal,
        requestScope: snapshot.requestScope,
      });
      return { status: "submitted" };
    });
    const options = createHookOptions({ historyId: "history-existing" });
    const { result } = renderHook(() =>
      useChatActions(
        options as unknown as Parameters<typeof useChatActions>[0],
      ),
    );

    await act(async () => {
      await result.current.onSubmit({ message: "Compare this", image: "" });
    });

    expect(setLastUsedModelMock).toHaveBeenCalledTimes(1);
    expect(setLastUsedModelMock).toHaveBeenCalledWith(
      "history-existing",
      "model-b",
    );
    expect(setLastUsedPromptMock).toHaveBeenCalledWith("history-existing", {
      prompt_content: undefined,
      prompt_id: "prompt-1",
    });
    expect(updateCreatedAtMock).toHaveBeenCalledWith("history-existing");
    expect(rollbackScopedComparePersistenceMock).not.toHaveBeenCalled();
  });

  it("applies deferred Compare metadata when Stop preserves partial branch output", async () => {
    const controller = new AbortController();
    const scopeInvalidatedController = new AbortController();
    loadServicePromptSnapshotMock.mockResolvedValueOnce(
      Object.freeze({
        ...snapshot,
        scopeSignal: controller.signal,
        scopeInvalidatedSignal: scopeInvalidatedController.signal,
      }),
    );
    normalChatModeMock.mockImplementation(async (...args: unknown[]) => {
      const params = args[6] as {
        selectedModel: string;
        saveMessageOnError: (
          payload: Record<string, unknown>,
        ) => Promise<string | null>;
      };
      controller.abort();
      await params.saveMessageOnError({
        historyId: "history-existing",
        selectedModel: params.selectedModel,
        prompt_id: "prompt-1",
      });
      return { status: "skipped", reason: "Request cancelled" };
    });
    const options = createHookOptions({ historyId: "history-existing" });
    const { result } = renderHook(() =>
      useChatActions(
        options as unknown as Parameters<typeof useChatActions>[0],
      ),
    );

    await act(async () => {
      await result.current.onSubmit({
        message: "Compare this",
        image: "",
        controller,
      });
    });

    expect(setLastUsedModelMock).toHaveBeenCalledOnce();
    expect(setLastUsedModelMock).toHaveBeenCalledWith(
      "history-existing",
      "model-b",
    );
    expect(setLastUsedPromptMock).toHaveBeenCalledWith("history-existing", {
      prompt_content: undefined,
      prompt_id: "prompt-1",
    });
    expect(updateCreatedAtMock).toHaveBeenCalledWith("history-existing");
    expect(rollbackScopedComparePersistenceMock).not.toHaveBeenCalled();
  });

  it("preserves user cancellation instead of misclassifying it as a scope change", async () => {
    const controller = new AbortController();
    loadServicePromptSnapshotMock.mockResolvedValueOnce(
      Object.freeze({ ...snapshot, scopeSignal: controller.signal }),
    );
    normalChatModeMock.mockImplementation(async () => {
      controller.abort();
      return { status: "skipped", reason: "Request cancelled" };
    });
    const options = createHookOptions();
    const { result } = renderHook(() =>
      useChatActions(
        options as unknown as Parameters<typeof useChatActions>[0],
      ),
    );

    let submissionResult: Awaited<ReturnType<typeof result.current.onSubmit>>;
    await act(async () => {
      submissionResult = await result.current.onSubmit({
        message: "Compare this",
        image: "",
        controller,
      });
    });

    expect(submissionResult!).toEqual({
      status: "skipped",
      reason: "Request cancelled",
    });
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
    expect(releaseSnapshotMock).not.toHaveBeenCalled();
    expect(options.notification.error).toHaveBeenCalledWith({
      message: "error",
      description: "Workflow prompts are unavailable",
    });
  });

  it("treats Stop during normal prompt preflight as cancellation without an error notification", async () => {
    loadServicePromptSnapshotMock.mockImplementationOnce(
      async (_ids: unknown, { signal }: { signal: AbortSignal }) => {
        events.push("loadSnapshot");
        return rejectWhenAborted(signal);
      },
    );
    const options = {
      ...createHookOptions({ webSearch: true }),
      compareModeActive: false,
      compareSelectedModels: [],
    };
    const { result } = renderHook(() => {
      const [abortController, setAbortController] =
        React.useState<AbortController | null>(null);
      return useChatActions({
        ...options,
        abortController,
        setAbortController,
      } as unknown as Parameters<typeof useChatActions>[0]);
    });
    let submission!: ReturnType<typeof result.current.onSubmit>;
    let submissionResult!: Awaited<typeof submission>;

    await act(async () => {
      submission = result.current.onSubmit({
        message: "Stop before prompts load",
        image: "",
      });
      await vi.waitFor(() => {
        expect(loadServicePromptSnapshotMock).toHaveBeenCalledTimes(1);
      });
    });

    await act(async () => {
      result.current.stopStreamingRequest();
      submissionResult = await submission;
    });

    expect(submissionResult).toEqual({
      status: "skipped",
      reason: "Request cancelled",
    });
    expect(options.notification.error).not.toHaveBeenCalled();
    expect(normalChatModeMock).not.toHaveBeenCalled();
  });

  it("keeps a newer turn active when an older aborted prompt preflight settles late", async () => {
    const olderSnapshot = deferred<typeof snapshot>();
    const newerSnapshot = deferred<typeof snapshot>();
    loadServicePromptSnapshotMock
      .mockReturnValueOnce(olderSnapshot.promise)
      .mockReturnValueOnce(newerSnapshot.promise);
    const options = {
      ...createHookOptions({ webSearch: true }),
      compareModeActive: false,
      compareSelectedModels: [],
    };
    const { result } = renderHook(() =>
      useChatActions(
        options as unknown as Parameters<typeof useChatActions>[0],
      ),
    );
    const olderController = new AbortController();
    const newerController = new AbortController();
    let olderSubmission!: ReturnType<typeof result.current.onSubmit>;
    let newerSubmission!: ReturnType<typeof result.current.onSubmit>;
    let olderResult!: Awaited<typeof olderSubmission>;

    await act(async () => {
      olderSubmission = result.current.onSubmit({
        message: "Older turn",
        image: "",
        controller: olderController,
      });
      await vi.waitFor(() => {
        expect(loadServicePromptSnapshotMock).toHaveBeenCalledTimes(1);
      });
    });

    olderController.abort();
    await act(async () => {
      newerSubmission = result.current.onSubmit({
        message: "Newer turn",
        image: "",
        controller: newerController,
      });
      await vi.waitFor(() => {
        expect(loadServicePromptSnapshotMock).toHaveBeenCalledTimes(2);
      });
    });
    options.setStreaming.mockClear();
    options.setIsProcessing.mockClear();
    options.setAbortController.mockClear();

    await act(async () => {
      olderSnapshot.reject(createAbortError());
      olderResult = await olderSubmission;
    });

    expect(olderResult).toEqual({
      status: "skipped",
      reason: "Request cancelled",
    });
    expect(options.setStreaming).not.toHaveBeenCalledWith(false);
    expect(options.setIsProcessing).not.toHaveBeenCalledWith(false);
    expect(options.setAbortController).not.toHaveBeenCalledWith(null);

    await act(async () => {
      newerController.abort();
      newerSnapshot.reject(createAbortError());
      await newerSubmission;
    });
  });

  it("does not carry a discarded prompt preflight into a later character turn", async () => {
    loadServicePromptSnapshotMock.mockImplementationOnce(
      async (_ids: unknown, { signal }: { signal: AbortSignal }) =>
        rejectWhenAborted(signal),
    );
    addChatMessageMock.mockResolvedValueOnce({
      id: "user-server-1",
      version: 1,
    });
    streamCharacterChatCompletionMock.mockImplementationOnce(
      async function* () {
        yield await Promise.reject(createAbortError());
      },
    );
    const options = {
      ...createHookOptions({ webSearch: true }),
      compareModeActive: false,
      compareSelectedModels: [],
    };
    const laterCharacter = {
      id: "12",
      name: "Later Character",
      system_prompt: "Stay in character",
    };
    const laterAssistant = {
      kind: "character" as const,
      id: "12",
      name: "Later Character",
      system_prompt: "Stay in character",
      metadata: { selectionMode: "tracked" },
    };
    const { result, rerender } = renderHook(
      ({ characterMode }: { characterMode: boolean }) => {
        const [abortController, setAbortController] =
          React.useState<AbortController | null>(null);
        return useChatActions({
          ...options,
          abortController,
          setAbortController,
          selectedCharacter: characterMode ? laterCharacter : null,
          selectedAssistant: characterMode ? laterAssistant : null,
        } as unknown as Parameters<typeof useChatActions>[0]);
      },
      { initialProps: { characterMode: false } },
    );
    let preflightSubmission!: ReturnType<typeof result.current.onSubmit>;
    let characterResult!: Awaited<ReturnType<typeof result.current.onSubmit>>;

    await act(async () => {
      preflightSubmission = result.current.onSubmit({
        message: "Queued turn being replaced",
        image: "",
      });
      await vi.waitFor(() => {
        expect(loadServicePromptSnapshotMock).toHaveBeenCalledTimes(1);
      });
    });

    await act(async () => {
      result.current.stopStreamingRequest({ discardTurn: true });
      await preflightSubmission;
    });
    rerender({ characterMode: true });

    await act(async () => {
      characterResult = await result.current.onSubmit({
        message: "Later character turn",
        image: "",
      });
    });

    expect(streamCharacterChatCompletionMock).toHaveBeenCalledTimes(1);
    expect(characterResult).toEqual({
      status: "failed",
      errorMessage: "AbortError",
    });
  });

  it.each([
    {
      label: "document",
      options: {
        contextFiles: [{ id: "file-1", filename: "notes.pdf" }],
        selectedKnowledge: null,
      },
      submit: {},
      ids: ["chat.rag.answer", "chat.rag.question_rewrite"],
      modeMock: documentChatModeMock,
    },
    {
      label: "tab",
      options: { contextFiles: [], selectedKnowledge: null },
      submit: { docs: [{ type: "tab", tabId: 7 }] },
      ids: ["chat.rag.answer"],
      modeMock: tabChatModeMock,
    },
    {
      label: "RAG",
      options: { contextFiles: [], selectedKnowledge: "knowledge-1" },
      submit: {},
      ids: ["chat.rag.answer", "chat.rag.question_rewrite"],
      modeMock: ragModeMock,
    },
    {
      label: "normal web-search",
      options: { contextFiles: [], selectedKnowledge: null },
      submit: {},
      ids: ["chat.web_search.answer"],
      modeMock: normalChatModeMock,
      webSearch: true,
    },
  ])(
    "gates $label history work on prompt preflight",
    async ({
      options: optionOverrides,
      submit,
      ids,
      modeMock,
      webSearch: caseWebSearch = false,
    }) => {
      loadServicePromptSnapshotMock.mockRejectedValueOnce(
        new Error("Workflow prompts are unavailable"),
      );
      const options = {
        ...createHookOptions({
          webSearch: caseWebSearch,
          serverChatId: "server-chat-7",
        }),
        compareModeActive: false,
        compareSelectedModels: [],
        ...optionOverrides,
      };
      const { result } = renderHook(() =>
        useChatActions(
          options as unknown as Parameters<typeof useChatActions>[0],
        ),
      );
      options.setMessages.mockClear();
      options.setHistory.mockClear();

      await act(async () => {
        await result.current.onSubmit({
          message: "Prompt-gated turn",
          image: "",
          ...submit,
        });
      });

      expect(loadServicePromptSnapshotMock).toHaveBeenCalledWith(ids, {
        signal: expect.any(AbortSignal),
      });
      expect(options.ensureServerChatHistoryId).not.toHaveBeenCalled();
      expect(options.setMessages).not.toHaveBeenCalled();
      expect(options.setHistory).not.toHaveBeenCalled();
      expect(saveHistoryMock).not.toHaveBeenCalled();
      expect(saveMessageMock).not.toHaveBeenCalled();
      expect(modeMock).not.toHaveBeenCalled();
      expect(options.notification.error).toHaveBeenCalledWith({
        message: "error",
        description: "Workflow prompts are unavailable",
      });
    },
  );

  it("treats Stop during per-model prompt preflight as cancellation without an error notification", async () => {
    loadServicePromptSnapshotMock.mockImplementationOnce(
      async (_ids: unknown, { signal }: { signal: AbortSignal }) => {
        events.push("loadSnapshot");
        return rejectWhenAborted(signal);
      },
    );
    const options = createHookOptions({ webSearch: true });
    const { result } = renderHook(() => {
      const [abortController, setAbortController] =
        React.useState<AbortController | null>(null);
      return useChatActions({
        ...options,
        abortController,
        setAbortController,
      } as unknown as Parameters<typeof useChatActions>[0]);
    });
    let submission!: ReturnType<typeof result.current.sendPerModelReply>;
    let submissionResult!: Awaited<typeof submission>;

    await act(async () => {
      submission = result.current.sendPerModelReply({
        clusterId: "cluster-1",
        modelId: "model-a",
        message: "Stop before prompts load",
      });
      await vi.waitFor(() => {
        expect(loadServicePromptSnapshotMock).toHaveBeenCalledTimes(1);
      });
    });

    await act(async () => {
      result.current.stopStreamingRequest();
      submissionResult = await submission;
    });

    expect(submissionResult).toBeUndefined();
    expect(options.notification.error).not.toHaveBeenCalled();
    expect(normalChatModeMock).not.toHaveBeenCalled();
  });

  it.each([
    {
      label: "document",
      options: {
        contextFiles: [{ id: "file-1", filename: "notes.pdf" }],
        documentContext: null,
        selectedKnowledge: null,
      },
      ids: ["chat.rag.answer", "chat.rag.question_rewrite"],
      modeMock: documentChatModeMock,
    },
    {
      label: "tab",
      options: {
        contextFiles: [],
        documentContext: [{ type: "tab", tabId: 7 }],
        selectedKnowledge: null,
      },
      ids: ["chat.rag.answer"],
      modeMock: tabChatModeMock,
    },
    {
      label: "RAG",
      options: {
        contextFiles: [],
        documentContext: null,
        selectedKnowledge: "knowledge-1",
      },
      ids: ["chat.rag.answer", "chat.rag.question_rewrite"],
      modeMock: ragModeMock,
    },
    {
      label: "normal web-search",
      options: {
        contextFiles: [],
        documentContext: null,
        selectedKnowledge: null,
        webSearch: true,
      },
      ids: ["chat.web_search.answer"],
      modeMock: normalChatModeMock,
    },
  ])(
    "gates per-model $label replies on prompt preflight",
    async ({ options: optionOverrides, ids, modeMock }) => {
      loadServicePromptSnapshotMock.mockRejectedValueOnce(
        new Error("Workflow prompts are unavailable"),
      );
      const options = {
        ...createHookOptions({
          webSearch: false,
          serverChatId: "server-chat-7",
        }),
        ...optionOverrides,
      };
      const { result } = renderHook(() =>
        useChatActions(
          options as unknown as Parameters<typeof useChatActions>[0],
        ),
      );

      await act(async () => {
        await result.current.sendPerModelReply({
          clusterId: "cluster-1",
          modelId: "model-a",
          message: "Follow up",
        });
      });

      expect(loadServicePromptSnapshotMock).toHaveBeenCalledWith(ids, {
        signal: expect.any(AbortSignal),
      });
      expect(options.ensureServerChatHistoryId).not.toHaveBeenCalled();
      expect(saveHistoryMock).not.toHaveBeenCalled();
      expect(saveMessageMock).not.toHaveBeenCalled();
      expect(modeMock).not.toHaveBeenCalled();
    },
  );

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
    expect(releaseSnapshotMock).not.toHaveBeenCalled();
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
    expect(options.ensureServerChatHistoryId).toHaveBeenCalledWith(
      "server-chat-7",
      undefined,
      snapshot.scopeInvalidatedSignal,
      snapshot,
    );
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
    expect(releaseSnapshotMock).toHaveBeenCalledTimes(1);
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
    expect(releaseSnapshotMock).toHaveBeenCalledTimes(1);
  });

  it("captures only owner scope and preserves server history when Compare web search is disabled", async () => {
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

    expect(loadServicePromptSnapshotMock).toHaveBeenCalledWith(
      [],
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    );
    expect(options.ensureServerChatHistoryId.mock.calls[0][3]).toBe(snapshot);
    expect(releaseSnapshotMock).toHaveBeenCalledTimes(1);
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

    expect(loadServicePromptSnapshotMock).toHaveBeenCalledWith(
      [],
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    );
    expect(
      normalChatModeMock.mock.calls.every(
        (call) => call[6].servicePromptSnapshot === snapshot,
      ),
    ).toBe(true);

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

// These tests exercise the actual useChatActions continuation branch, live
// option core/account watcher and H1 controller. Mode execution, service scope,
// database persistence and unrelated UI services remain declared I/O doubles;
// late mode emissions model a response already in flight, not native transport.
describe("useChatActions owned continuation prompt publication", () => {
  const originalMessages: Message[] = [
    {
      id: "source-answer",
      name: "Assistant",
      isBot: true,
      message: "Existing answer.",
    },
  ];
  const originalHistory: ChatHistory = [
    { role: "assistant", content: "Existing answer." },
  ];
  const continuedMessages: Message[] = [
    {
      ...originalMessages[0],
      message: "Existing answer. A suggested next question?",
    },
  ];
  const continuedHistory: ChatHistory = [
    {
      role: "assistant",
      content: continuedMessages[0].message,
    },
  ];
  const replacementMessages: Message[] = [
    {
      id: "replacement-answer",
      name: "Assistant",
      isBot: true,
      message: "Replacement source answer.",
    },
  ];
  const replacementHistory: ChatHistory = [
    {
      role: "assistant",
      content: "Replacement source answer.",
    },
  ];

  const mountContinuation = (persistAfterGate = false) => {
    const gate = deferred<void>();
    let pipeline: Parameters<typeof continueChatModeMock>[3] | undefined;
    let controller: ReturnType<typeof useHistorySelection> | undefined;
    continueChatModeMock.mockImplementation(
      async (_messages, _history, _signal, params) => {
        pipeline = params;
        params.setMessages(structuredClone(continuedMessages));
        params.setHistory(structuredClone(continuedHistory));
        await gate.promise;
        if (persistAfterGate) {
          await params.saveMessageOnSuccess({
            historyId: params.historyId,
            setHistoryId: params.setHistoryId,
            selectedModel: params.selectedModel,
            modelId: params.selectedModel,
            image: "",
            assistantMessageId: "source-answer",
            isRegenerate: false,
            message: "",
            fullText: continuedMessages[0].message,
            source: [],
            reasoning_time_taken: 0,
            isContinue: true,
            saveToDb: false,
          });
          params.setStreaming(false);
          params.setIsProcessing(false);
          params.setAbortController(null);
        }
        return { status: "submitted" };
      },
    );
    const core = useStoreMessageOption.getState();
    const messages = vi.fn(core.setMessages);
    const history = vi.fn(core.setHistory);
    const quick = vi.fn(core.setSelectedQuickPrompt);
    const historyId = vi.fn(core.setHistoryId);
    const options = createHookOptions({
      webSearch: false,
      historyId: "history-source",
    });
    const Wrapper = ({ children }: { children: React.ReactNode }) => {
      const selection = useHistorySelection();
      React.useLayoutEffect(() => {
        controller = selection;
      }, [selection]);
      return (
        <HistorySelectionContext.Provider value={selection}>
          {children}
        </HistorySelectionContext.Provider>
      );
    };
    const view = renderHook(
      () => {
        const state = useStoreMessageOption();
        return useChatActions({
          ...options,
          messages: state.messages,
          history: state.history,
          historyId: state.historyId,
          serverChatId: state.serverChatId,
          temporaryChat: state.temporaryChat,
          selectedModel: state.selectedModel,
          setMessages: messages,
          setHistory: history,
          setHistoryId: historyId,
          setServerChatId: state.setServerChatId,
          setSelectedQuickPrompt: quick,
          compareModeActive: false,
          compareSelectedModels: [],
        } as unknown as Parameters<typeof useChatActions>[0]);
      },
      { wrapper: Wrapper },
    );
    return {
      ...view,
      gate,
      messages,
      history,
      quick,
      historyId,
      streaming: options.setStreaming,
      processing: options.setIsProcessing,
      abortController: options.setAbortController,
      get controller() {
        return controller!;
      },
      get pipeline() {
        return pipeline!;
      },
    };
  };
  const start = async (view: ReturnType<typeof mountContinuation>) => {
    let pending!: ReturnType<typeof view.result.current.onSubmit>;
    act(() => {
      pending = view.result.current.onSubmit({
        message: "",
        image: "",
        isContinue: true,
        continueOutputTarget: "composer_input",
        messageSteeringOverride: {
          mode: "impersonate_user",
          forceNarrate: false,
        },
      });
    });
    await waitFor(() => {
      expect(continueChatModeMock).toHaveBeenCalledOnce();
      expect(useStoreMessageOption.getState().messages).toEqual(
        continuedMessages,
      );
    });
    view.messages.mockClear();
    view.history.mockClear();
    view.quick.mockClear();
    return { pending };
  };
  beforeEach(() => {
    vi.clearAllMocks();
    continuationStoreBoundary.actual = true;
    continuationStoreBoundary.actualSave = false;
    const core = useStoreMessageOption.getState();
    core.startNewChatDraft(false);
    core.setChatDraftAddress({
      kind: "local",
      id: "history-source",
      profileId: "profile-test",
    });
    useStoreMessageOption.setState({
      historyId: "history-source",
      serverChatId: null,
      temporaryChat: false,
      messages: structuredClone(originalMessages),
      history: structuredClone(originalHistory),
      selectedModel: "model-a",
      selectedQuickPrompt: null,
      quickPromptRevision: 0,
      compareMode: false,
      compareSelectedModels: [],
      streaming: false,
      isProcessing: false,
    });
    loadServicePromptSnapshotMock.mockResolvedValue({
      ...snapshot,
      scopeSignal: new AbortController().signal,
      scopeInvalidatedSignal: new AbortController().signal,
    });
  });
  afterEach(() => {
    cleanup();
    continuationStoreBoundary.actual = false;
    continuationStoreBoundary.actualSave = false;
  });

  it("publishes the current continuation once and restores its original transcript", async () => {
    const view = mountContinuation();
    const { pending } = await start(view);
    await act(async () => {
      view.gate.resolve();
      await pending;
    });
    expect(view.quick).toHaveBeenCalledOnce();
    expect(useStoreMessageOption.getState().selectedQuickPrompt).toBe(
      "A suggested next question?",
    );
    expect(useStoreMessageOption.getState().messages).toEqual(originalMessages);
    expect(useStoreMessageOption.getState().history).toEqual(originalHistory);
    expect(updateMessageMock).toHaveBeenCalledWith(
      "history-source",
      "source-answer",
      "Existing answer.",
    );
  });

  it.each([false, true])(
    "keeps actual temporary-success UI publication with its source (retired=%s)",
    async (retired) => {
      continuationStoreBoundary.actualSave = true;
      const core = useStoreMessageOption.getState();
      core.startNewChatDraft(true);
      core.setHistoryId("temp");
      const view = mountContinuation(true);
      const { pending } = await start(view);
      view.historyId.mockClear();
      if (retired) {
        act(() => {
          const current = useStoreMessageOption.getState();
          current.startNewChatDraft(false);
          current.setHistoryId(null);
          current.setMessages(structuredClone(replacementMessages));
          current.setHistory(structuredClone(replacementHistory));
        });
      }
      // The actual temporary helper is composed here; its persistence policy
      // remains no-write, while supplied UI setters retain source ownership.
      await act(async () => {
        view.gate.resolve();
        await pending;
      });
      if (retired) {
        expect(view.historyId).not.toHaveBeenCalled();
        expect(useStoreMessageOption.getState().historyId).toBeNull();
        expect(useStoreMessageOption.getState().temporaryChat).toBe(false);
        expect(useStoreMessageOption.getState().messages).toEqual(
          replacementMessages,
        );
        expect(view.quick).not.toHaveBeenCalled();
      } else {
        expect(view.historyId).toHaveBeenCalledExactlyOnceWith("temp");
        expect(useStoreMessageOption.getState().historyId).toBe("temp");
        expect(useStoreMessageOption.getState().temporaryChat).toBe(true);
        expect(view.quick).toHaveBeenCalledExactlyOnceWith(
          "A suggested next question?",
        );
        expect(useStoreMessageOption.getState().messages).toEqual(
          originalMessages,
        );
      }
      expect(saveHistoryMock).not.toHaveBeenCalled();
      expect(saveMessageMock).not.toHaveBeenCalled();
    },
  );

  it.each([false, true])(
    "permits only its own temporary null-to-temp bind and cleanup (retired=%s)",
    async (retired) => {
      continuationStoreBoundary.actualSave = true;
      const core = useStoreMessageOption.getState();
      core.startNewChatDraft(true);
      core.setHistoryId(null);
      const view = mountContinuation(true);
      const { pending } = await start(view);
      const sourceAddress = useStoreMessageOption.getState().chatDraftAddress;
      view.historyId.mockClear();
      view.streaming.mockClear();
      view.processing.mockClear();
      view.abortController.mockClear();
      if (retired) {
        act(() => {
          const current = useStoreMessageOption.getState();
          current.startNewChatDraft(false);
          current.setHistoryId(null);
          current.setMessages(structuredClone(replacementMessages));
          current.setHistory(structuredClone(replacementHistory));
        });
      }
      await act(async () => {
        view.gate.resolve();
        await pending;
      });
      if (retired) {
        expect(view.historyId).not.toHaveBeenCalled();
        expect(view.quick).not.toHaveBeenCalled();
        expect(view.streaming).not.toHaveBeenCalled();
        expect(view.processing).not.toHaveBeenCalled();
        expect(view.abortController).not.toHaveBeenCalled();
        expect(useStoreMessageOption.getState().historyId).toBeNull();
        expect(useStoreMessageOption.getState().messages).toEqual(
          replacementMessages,
        );
      } else {
        expect(view.historyId).toHaveBeenCalledExactlyOnceWith("temp");
        expect(useStoreMessageOption.getState().chatDraftAddress).toBe(
          sourceAddress,
        );
        expect(view.quick).toHaveBeenCalledExactlyOnceWith(
          "A suggested next question?",
        );
        expect(view.streaming).toHaveBeenCalledExactlyOnceWith(false);
        expect(view.processing).toHaveBeenCalledExactlyOnceWith(false);
        expect(view.abortController).toHaveBeenCalledExactlyOnceWith(null);
        expect(useStoreMessageOption.getState().messages).toEqual(
          originalMessages,
        );
        expect(useStoreMessageOption.getState().historyId).toBe("temp");
      }
      expect(saveHistoryMock).not.toHaveBeenCalled();
      expect(saveMessageMock).not.toHaveBeenCalled();
      expect(updateMessageMock).not.toHaveBeenCalled();
    },
  );

  it("does not roll back after a synchronous core subscriber replaces its published command", async () => {
    const view = mountContinuation();
    const { pending } = await start(view);
    let intervened = false;
    const stop = useStoreMessageOption.subscribe((state) => {
      if (
        !intervened &&
        state.selectedQuickPrompt === "A suggested next question?"
      ) {
        intervened = true;
        state.setSelectedQuickPrompt("Newer synchronous command");
      }
    });
    try {
      await act(async () => {
        view.gate.resolve();
        await pending;
      });
      expect(intervened).toBe(true);
      expect(useStoreMessageOption.getState().selectedQuickPrompt).toBe(
        "Newer synchronous command",
      );
      expect(view.messages).not.toHaveBeenCalled();
      expect(view.history).not.toHaveBeenCalled();
      expect(updateMessageMock).not.toHaveBeenCalled();
      expect(useStoreMessageOption.getState().messages).toEqual(
        continuedMessages,
      );
      expect(useStoreMessageOption.getState().history).toEqual(
        continuedHistory,
      );
    } finally {
      stop();
    }
  });

  it("refuses a retained continuation callback after same-source H1 departure and return", async () => {
    const view = mountContinuation();
    const submit = view.result.current.onSubmit;
    const source = useStoreMessageOption.getState();
    let pending!: ReturnType<typeof submit>;
    act(() => {
      view.controller.beginLoad();
      view.controller.reset();
      // Invoke the rendered callback before React commits a replacement one.
      pending = submit({
        message: "",
        image: "",
        isContinue: true,
        continueOutputTarget: "composer_input",
      });
    });
    await act(async () => {
      view.gate.resolve();
      await pending;
    });
    expect(useStoreMessageOption.getState().chatDraftAddress).toBe(
      source.chatDraftAddress,
    );
    expect(useStoreMessageOption.getState().chatDraftRevision).toBe(
      source.chatDraftRevision,
    );
    expect(continueChatModeMock).not.toHaveBeenCalled();
    expect(view.quick).not.toHaveBeenCalled();
    expect(useStoreMessageOption.getState().messages).toEqual(originalMessages);
    expect(useStoreMessageOption.getState().history).toEqual(originalHistory);
  });

  it("does not restore old history if its rollback messages synchronously replace the source", async () => {
    const view = mountContinuation();
    const { pending } = await start(view);
    let intervened = false;
    const stop = useStoreMessageOption.subscribe((state) => {
      if (!intervened && state.messages[0]?.message === "Existing answer.") {
        intervened = true;
        state.startNewChatDraft(false);
        state.setMessages(structuredClone(replacementMessages));
        state.setHistory(structuredClone(replacementHistory));
      }
    });
    try {
      await act(async () => {
        view.gate.resolve();
        await pending;
      });
      expect(intervened).toBe(true);
      expect(useStoreMessageOption.getState().messages).toEqual(
        replacementMessages,
      );
      expect(view.history).not.toHaveBeenCalled();
      expect(useStoreMessageOption.getState().history).toEqual(
        replacementHistory,
      );
      expect(updateMessageMock).not.toHaveBeenCalled();
    } finally {
      stop();
    }
  });

  it.each(["New", "account", "H1"] as const)(
    "refuses pipeline UI writes and final prompt/rollback after %s retires the source",
    async (boundary) => {
      const view = mountContinuation();
      const { pending } = await start(view);
      const before = useStoreMessageOption.getState();
      act(() => {
        if (boundary === "New") before.startNewChatDraft(false);
        else if (boundary === "account")
          window.dispatchEvent(new Event("tldw:auth-principal-changed"));
        else view.controller.beginLoad();
        useStoreMessageOption.setState({
          messages: structuredClone(replacementMessages),
          history: structuredClone(replacementHistory),
        });
      });
      if (boundary !== "H1")
        expect(
          useStoreMessageOption.getState().chatDraftRevision,
        ).toBeGreaterThan(before.chatDraftRevision);
      try {
        act(() => {
          view.pipeline.setMessages(structuredClone(continuedMessages));
          view.pipeline.setHistory(structuredClone(continuedHistory));
        });
        expect(useStoreMessageOption.getState().messages).toEqual(
          replacementMessages,
        );
        expect(useStoreMessageOption.getState().history).toEqual(
          replacementHistory,
        );
      } finally {
        await act(async () => {
          view.gate.resolve();
          await pending;
        });
      }
      expect(view.quick).not.toHaveBeenCalled();
      expect(updateMessageMock).not.toHaveBeenCalled();
      expect(useStoreMessageOption.getState().messages).toEqual(
        replacementMessages,
      );
      expect(useStoreMessageOption.getState().history).toEqual(
        replacementHistory,
      );
    },
  );

  it.each(["New", "account", "H1"] as const)(
    "refuses final command publication and transcript rollback after %s with no late stream emission",
    async (boundary) => {
      const view = mountContinuation();
      const { pending } = await start(view);
      act(() => {
        const core = useStoreMessageOption.getState();
        if (boundary === "New") core.startNewChatDraft(false);
        else if (boundary === "account")
          window.dispatchEvent(new Event("tldw:auth-principal-changed"));
        else view.controller.beginLoad();
        useStoreMessageOption.setState({
          messages: structuredClone(replacementMessages),
          history: structuredClone(replacementHistory),
        });
      });
      // The held mode simply settles here. This isolates post-submit publication
      // and rollback from the separate late-setter regressions above.
      await act(async () => {
        view.gate.resolve();
        await pending;
      });
      expect(view.quick).not.toHaveBeenCalled();
      expect(view.messages).not.toHaveBeenCalled();
      expect(view.history).not.toHaveBeenCalled();
      expect(updateMessageMock).not.toHaveBeenCalled();
      expect(useStoreMessageOption.getState().messages).toEqual(
        replacementMessages,
      );
      expect(useStoreMessageOption.getState().history).toEqual(
        replacementHistory,
      );
    },
  );

  it("does not overwrite a newer command or roll back its current transcript after command ABA", async () => {
    const view = mountContinuation();
    const { pending } = await start(view);
    act(() => {
      const core = useStoreMessageOption.getState();
      core.setSelectedQuickPrompt("newer explicit command");
      core.setSelectedQuickPrompt("intermediate command");
      core.setSelectedQuickPrompt("newer explicit command");
    });
    await act(async () => {
      view.gate.resolve();
      await pending;
    });
    expect(view.quick).not.toHaveBeenCalled();
    expect(updateMessageMock).not.toHaveBeenCalled();
    expect(useStoreMessageOption.getState().selectedQuickPrompt).toBe(
      "newer explicit command",
    );
    expect(useStoreMessageOption.getState().messages).toEqual(
      continuedMessages,
    );
    expect(useStoreMessageOption.getState().history).toEqual(continuedHistory);
  });
});
