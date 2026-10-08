// @vitest-environment jsdom
import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useSelectServerChat } from "../useSelectServerChat";
import type { ServerChatSummary } from "@/services/tldw/TldwApiClient";
import { usePlaygroundSessionStore } from "@/store/playground-session";

const navigateMock = vi.hoisted(() => vi.fn());
const setSelectedAssistantMock = vi.hoisted(() => vi.fn(async () => undefined));

const chatBaseState = vi.hoisted(() => ({
  value: {
    setHistory: vi.fn(),
    setHistoryId: vi.fn(),
    setMessages: vi.fn(),
    setIsLoading: vi.fn(),
    setIsProcessing: vi.fn(),
    setStreaming: vi.fn(),
    setIsEmbedding: vi.fn(),
  },
}));

const optionState = vi.hoisted(() => ({
  value: {
    temporaryChat: false, historyId: null as string | null, serverChatId: null as string | null,
    messages: [] as unknown[], history: [] as unknown[],
    chatDraftAddress: { kind: "fresh", id: "draft-a", scope: { type: "global" }, temporary: false },
    chatDraftRevision: 3,
    setChatDraftAddress: vi.fn(() => true),
    setIsSearchingInternet: vi.fn(),
    clearReplyTarget: vi.fn(),
    setServerChatId: vi.fn(),
    setServerChatTitle: vi.fn(),
    setServerChatCharacterId: vi.fn(),
    setServerChatAssistantKind: vi.fn(),
    setServerChatAssistantId: vi.fn(),
    setServerChatPersonaMemoryMode: vi.fn(),
    setServerChatVersion: vi.fn(),
    setServerChatLoadState: vi.fn(),
    setServerChatLoadError: vi.fn(),
    setServerChatState: vi.fn(),
    setServerChatTopic: vi.fn(),
    setServerChatClusterId: vi.fn(),
    setServerChatSource: vi.fn(),
    setServerChatExternalRef: vi.fn(),
    setServerChatMetaLoaded: vi.fn(),
    setWebSearch: vi.fn(),
    setSelectedSystemPrompt: vi.fn(),
    setSelectedQuickPrompt: vi.fn(),
    setContextFiles: vi.fn(),
    setSelectedKnowledge: vi.fn(),
    setRagMediaIds: vi.fn(),
  },
}));

vi.mock("antd", () => ({
  Modal: {
    destroyAll: vi.fn(),
  },
}));

vi.mock("react-router-dom", () => ({
  useNavigate: () => navigateMock,
}));

vi.mock("@/hooks/chat/useChatBaseState", () => ({
  useChatBaseState: () => chatBaseState.value,
}));

vi.mock("@/hooks/useSelectedAssistant", () => ({
  useSelectedAssistant: () => [null, setSelectedAssistantMock],
}));

vi.mock("@/store/option", () => ({
  useStoreMessageOption: Object.assign((selector?: (state: unknown) => unknown) =>
    typeof selector === "function"
      ? selector(optionState.value)
      : optionState.value, { getState: () => optionState.value }),
}));

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: vi.fn(async () => null),
    getCharacter: vi.fn(async () => null),
    getPersonaProfile: vi.fn(async () => null),
  },
}));

vi.mock("@/utils/cleanup-ant-overlays", () => ({
  cleanupAntOverlays: vi.fn(),
}));

vi.mock("@/utils/update-page-title", () => ({
  updatePageTitle: vi.fn(),
}));

describe("useSelectServerChat context reset", () => {
  beforeEach(() => {
    Object.assign(optionState.value, { temporaryChat: false, historyId: null, serverChatId: null, messages: [], history: [] });
    vi.spyOn(window, "confirm").mockReturnValue(false);
    navigateMock.mockClear();
    setSelectedAssistantMock.mockClear();
    Object.values(chatBaseState.value).forEach((value) => {
      if (vi.isMockFunction(value)) value.mockClear();
    });
    Object.values(optionState.value).forEach((value) => {
      if (vi.isMockFunction(value)) value.mockClear();
    });
  });

  it("clears stale next-send context when switching to a server chat", () => {
    const { result } = renderHook(() => useSelectServerChat());

    act(() => {
      result.current({
        id: "chat-2",
        title: "Fresh server chat",
        version: 3,
        state: "active",
        topic_label: "Fresh topic",
        cluster_id: "cluster-1",
        source: "webui",
        external_ref: null,
      } as ServerChatSummary);
    });
    const intent = usePlaygroundSessionStore.getState().serverChatSelectionIntent;
    expect(intent).toMatchObject({ chatId: "chat-2", onLoaded: expect.any(Function) });
    expect(optionState.value.setChatDraftAddress).not.toHaveBeenCalled();
    intent?.onLoaded?.({
      owner: { kind: "native", conversation_id: "chat-2", request_scope: {} as never, validate_lease: () => true },
      view: { owner_key: "native-owner", conversation_id: "chat-2", view_session_id: "v", selection_revision: 1, cursor: { kind: "empty" }, interpretation: { kind: "parent_graph_v1" } }
    });
    expect(optionState.value.setChatDraftAddress).toHaveBeenCalledWith(
      { kind: "native", id: "chat-2", ownerKey: "native-owner", scope: { type: "global" } },
      { kind: "switch", expectedRevision: 3, sourceAddress: optionState.value.chatDraftAddress }
    );

    expect(optionState.value.setWebSearch).toHaveBeenCalledWith(false);
    expect(optionState.value.setSelectedSystemPrompt).toHaveBeenCalledWith("");
    expect(optionState.value.setSelectedQuickPrompt).toHaveBeenCalledWith(null);
    expect(optionState.value.setContextFiles).toHaveBeenCalledWith([]);
    expect(optionState.value.setSelectedKnowledge).toHaveBeenCalledWith(null);
    expect(optionState.value.setRagMediaIds).toHaveBeenCalledWith(null);
    expect(optionState.value.setServerChatId).toHaveBeenCalledWith("chat-2");
    expect(optionState.value.setServerChatTitle).toHaveBeenCalledWith(
      "Fresh server chat",
    );
  });
  afterEach(() => vi.restoreAllMocks());

  it("keeps Temporary rows and the pending restore untouched when saved selection is cancelled", () => {
    Object.assign(optionState.value, { temporaryChat: true, historyId: "temp", messages: [{ message: "Temporary answer" }], history: [{ content: "Temporary answer" }] });
    usePlaygroundSessionStore.setState({ restoreRevision: 8 });
    const { result } = renderHook(() => useSelectServerChat());
    act(() => expect(result.current({ id: "next-saved", title: "Next saved" } as ServerChatSummary)).toBe(false));
    expect(window.confirm).toHaveBeenCalledOnce();
    expect(chatBaseState.value.setMessages).not.toHaveBeenCalled();
    expect(optionState.value.setServerChatId).not.toHaveBeenCalled();
    expect(navigateMock).not.toHaveBeenCalled();
    expect(usePlaygroundSessionStore.getState().restoreRevision).toBe(8);
  });

  it("admits acknowledged Temporary replacement once and reports acceptance", () => {
    Object.assign(optionState.value, { temporaryChat: true, historyId: "temp", messages: [{ message: "Temporary answer" }] });
    vi.mocked(window.confirm).mockReturnValue(true);
    const { result } = renderHook(() => useSelectServerChat());
    act(() => expect(result.current({ id: "next-saved", title: "Next saved" } as ServerChatSummary)).toBe(true));
    expect(window.confirm).toHaveBeenCalledOnce();
    expect(chatBaseState.value.setMessages).toHaveBeenCalledOnce();
    expect(optionState.value.setServerChatId).toHaveBeenCalledWith("next-saved");
    expect(navigateMock).toHaveBeenCalledWith("/chat");
  });

});
