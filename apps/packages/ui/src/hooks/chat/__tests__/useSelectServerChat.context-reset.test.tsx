// @vitest-environment jsdom
import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { useSelectServerChat } from "../useSelectServerChat";

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
  useStoreMessageOption: (selector?: (state: unknown) => unknown) =>
    typeof selector === "function"
      ? selector(optionState.value)
      : optionState.value,
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
      } as any);
    });

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
});
