// @vitest-environment jsdom
import React from "react";
import { act, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { expect, vi } from "vitest";
import { useConnectionStore } from "@/store/connection";
import { ConnectionPhase } from "@/types/connection";
import { useStoreMessageOption } from "@/store/option";
import { useUiModeStore, type UiMode } from "@/store/ui-mode";
import { tldwClient } from "@/services/tldw/TldwApiClient";
import { tldwModels } from "@/services/tldw";
import { SidepanelForm } from "../form";

const offline = vi.hoisted(() => ({
  capabilities: [] as Promise<never>[],
}));
const ready = vi.hoisted(() => ({
  account: "alice",
  submissions: [] as Promise<void>[],
  unexpected: [] as string[],
  setImage: null as null | ((value: string) => void),
  embedding: vi.fn<() => Promise<string | null>>(),
  simpleSearch: vi.fn<() => Promise<boolean>>(),
  models:
    vi.fn<
      () => Promise<
        Array<{ id: string; name: string; model: string; provider: string }>
      >
    >(),
}));
export { offline, ready };

export const deferred = <T,>() => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
};

export const persistedDraft = (draftKey: string): unknown => {
  const key = Array.from({ length: localStorage.length }, (_, index) =>
    localStorage.key(index),
  ).find((entry) => entry?.startsWith(`registry:draft:${draftKey}:owner:`));
  return key ? JSON.parse(localStorage.getItem(key) || "null")?.value : null;
};
export const persistedReadyDraft = () =>
  persistedDraft("sidepanel-ownership-ready");

const storageSnapshot = (storage: Storage) =>
  Array.from({ length: storage.length }, (_, index) => {
    const key = storage.key(index)!;
    return [key, storage.getItem(key)!] as const;
  });
const restoreStorage = (
  storage: Storage,
  entries: readonly (readonly [string, string])[],
) => {
  storage.clear();
  for (const [key, value] of entries) storage.setItem(key, value);
};

export const mountDisconnected = async () => {
  offline.capabilities.length = 0;
  ready.submissions.length = 0;
  ready.unexpected.length = 0;
  ready.models.mockClear();
  const originalLocalStorage = storageSnapshot(localStorage);
  const originalSessionStorage = storageSnapshot(sessionStorage);
  localStorage.clear();
  sessionStorage.clear();
  localStorage.setItem("__tldw_force_unconfigured", "true");
  const originalConnection = useConnectionStore.getState();
  const originalMessage = useStoreMessageOption.getState();
  const originalUiMode = useUiModeStore.getState();
  const connectionChecks: Promise<void>[] = [];
  useConnectionStore.setState({
    state: {
      ...originalConnection.state,
      phase: ConnectionPhase.UNCONFIGURED,
      serverUrl: null,
      isConnected: false,
      isChecking: false,
    },
    checkOnce: (...args) => {
      const completion = originalConnection.checkOnce(...args);
      connectionChecks.push(completion);
      return completion;
    },
  });
  useStoreMessageOption.getState().setSelectedModel("tldw:ready-model");
  useUiModeStore.getState().setMode("casual");
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => {
      ready.unexpected.push("fetch");
      throw new Error("External fetch is disabled in this Form fixture");
    }),
  );
  vi.stubGlobal(
    "WebSocket",
    class {
      constructor() {
        ready.unexpected.push("WebSocket");
        throw new Error("External WebSocket is disabled in this Form fixture");
      }
    },
  );
  const provider = vi
    .spyOn(tldwClient, "streamChatCompletion")
    .mockImplementation(async function* () {
      yield* [];
      ready.unexpected.push("provider");
      throw new Error(
        "Provider dispatch is disabled in disconnected Form fixture",
      );
    });
  let view: ReturnType<typeof render> | undefined;
  const cleanup = async () => {
    view?.unmount();
    await act(async () => {
      await Promise.allSettled(ready.submissions);
      await Promise.allSettled(connectionChecks);
      await Promise.allSettled(offline.capabilities);
    });
    queryClient.clear();
    useStoreMessageOption.setState(originalMessage);
    useUiModeStore.setState(originalUiMode);
    useConnectionStore.setState({
      state: originalConnection.state,
      checkOnce: originalConnection.checkOnce,
    });
    restoreStorage(localStorage, originalLocalStorage);
    restoreStorage(sessionStorage, originalSessionStorage);
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    expect(ready.unexpected, JSON.stringify(ready.unexpected)).toEqual([]);
  };
  try {
    view = render(
      <QueryClientProvider client={queryClient}>
        <SidepanelForm
          dropedFile={undefined}
          draftKey="sidepanel-ownership-smoke"
        />
      </QueryClientProvider>,
    );
    return { cleanup, provider };
  } catch (error) {
    await cleanup();
    throw error;
  }
};

vi.mock(
  "@plasmohq/storage",
  () =>
    import("../../../../../../../tldw-frontend/extension/shims/plasmo-storage"),
);

vi.mock("@/services/service-prompts", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/services/service-prompts")>()),
  resolveServicePromptScope: async () => ({
    scopeKey: ready.account,
    userId: ready.account,
    config: {
      serverUrl: "http://localhost:8000",
      authMode: "multi-user",
      authSource: "manual",
      orgId: null,
    },
  }),
}));

vi.mock("@/hooks/useSimpleForm", async (importOriginal) => {
  const original =
    await importOriginal<typeof import("@/hooks/useSimpleForm")>();
  return {
    ...original,
    useSimpleForm: (options: Parameters<typeof original.useSimpleForm>[0]) => {
      const form = original.useSimpleForm(options);
      if (
        "message" in options.initialValues &&
        "image" in options.initialValues
      ) {
        ready.setImage = (value: string) => form.setFieldValue("image", value);
      }
      return {
        ...form,
        onSubmit: (handler: Parameters<typeof form.onSubmit>[0]) => {
          const invoke = form.onSubmit(handler);
          return (event?: React.FormEvent) => {
            const completion = invoke(event);
            ready.submissions.push(completion);
            void completion.catch(() => undefined);
            return completion;
          };
        },
      };
    },
  };
});

vi.mock("@/services/tldw-server", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/services/tldw-server")>()),
  fetchChatModels: () => ready.models(),
  defaultEmbeddingModelForRag: () => ready.embedding(),
  getDefaultApiProvider: async () => "openai",
}));

vi.mock("@/services/search", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/services/search")>()),
  getIsSimpleInternetSearch: () => ready.simpleSearch(),
}));

vi.mock("@/services/api-send", () => ({
  apiSend: async (request: { path: string }) => {
    if (request.path === "/api/v1/auth/sessions") {
      return { ok: true, status: 200, data: { user_id: ready.account } };
    }
    if (request.path === "/api/v1/prompts/capabilities") {
      return {
        ok: false,
        status: 404,
        error: "Prompt capabilities unavailable",
      };
    }
    ready.unexpected.push(`apiSend ${request.path}`);
    throw new Error(
      `Unexpected external request in Sidepanel fixture: ${request.path}`,
    );
  },
}));

vi.mock("@/services/background-proxy", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/services/background-proxy")>()),
  bgRequest: async (request: { path: string }) => {
    if (request.path === "/api/v1/auth/me") {
      return { id: ready.account, username: ready.account };
    }
    if (request.path === "/api/v1/users/me/profile?sections=preferences") {
      return { preferences: {} };
    }
    if (
      request.path === "/api/v1/service-prompts" ||
      request.path === "/api/v1/service-prompts/chat.web_search.answer"
    ) {
      const { LEGACY_SERVICE_PROMPT_DEFAULTS } =
        await import("@/services/tldw-server");
      const template =
        LEGACY_SERVICE_PROMPT_DEFAULTS["chat.web_search.answer"].template;
      const catalog = {
        id: "chat.web_search.answer",
        label: "Web search answer",
        description: "Answer with web search context",
        parts: [
          {
            key: "template",
            label: "Template",
            mode: "template",
            required_variables: ["current_date_time", "search_results"],
          },
        ],
        affected_workflows: [],
      };
      if (request.path === "/api/v1/service-prompts") return [catalog];
      return {
        ...catalog,
        default_parts: { template },
        saved_parts: null,
        effective_parts: { template },
        source: "packaged",
        revision: null,
      };
    }
    if (request.path === "/api/v1/chat/commands") return [];
    ready.unexpected.push(`bgRequest ${request.path}`);
    throw new Error(
      `Unexpected background request in Sidepanel fixture: ${request.path}`,
    );
  },
}));

vi.mock("@/services/tldw/server-capabilities", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("@/services/tldw/server-capabilities")
  >()),
  getServerCapabilities: () => {
    const request = Promise.reject(
      new Error("Capability probe is offline in this Form fixture"),
    );
    offline.capabilities.push(request);
    return request;
  },
}));

export const mountReady = async ({
  model = "tldw:ready-model",
  mode = "casual",
}: {
  model?: string | null;
  mode?: UiMode;
} = {}) => {
  ready.account = "alice";
  ready.submissions.length = 0;
  ready.unexpected.length = 0;
  ready.setImage = null;
  ready.embedding.mockReset().mockResolvedValue("embedding-model");
  ready.simpleSearch.mockReset().mockResolvedValue(false);
  ready.models.mockReset().mockResolvedValue([
    {
      id: "ready-model",
      name: "tldw:ready-model",
      model: "tldw:ready-model",
      provider: "openai",
    },
  ]);
  offline.capabilities.length = 0;
  const originalLocalStorage = storageSnapshot(localStorage);
  const originalSessionStorage = storageSnapshot(sessionStorage);
  localStorage.clear();
  sessionStorage.clear();
  localStorage.setItem(
    "tldwConfig",
    JSON.stringify({
      serverUrl: "http://localhost:8000",
      authMode: "multi-user",
      authSource: "manual",
      accessToken: `test.${btoa(JSON.stringify({ sub: "alice" }))}.signature`,
      refreshToken: "synthetic-refresh",
    }),
  );
  const originalConnection = useConnectionStore.getState();
  const originalMessage = useStoreMessageOption.getState();
  const originalUiMode = useUiModeStore.getState();
  const connectionChecks: Promise<void>[] = [];
  useConnectionStore.setState({
    state: {
      ...originalConnection.state,
      phase: ConnectionPhase.UNCONFIGURED,
      isConnected: false,
      isChecking: false,
      lastCheckedAt: null,
      offlineBypass: false,
    },
    checkOnce: (...args) => {
      const completion = originalConnection.checkOnce(...args);
      connectionChecks.push(completion);
      return completion;
    },
  });
  useStoreMessageOption.getState().setSelectedModel(model);
  useStoreMessageOption.getState().setTemporaryChat(true);
  useUiModeStore.getState().setMode(mode);
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => {
      ready.unexpected.push("fetch");
      throw new Error("Unexpected external fetch in ready Sidepanel fixture");
    }),
  );
  vi.stubGlobal(
    "WebSocket",
    class {
      constructor() {
        ready.unexpected.push("WebSocket");
        throw new Error("Unexpected WebSocket in ready Sidepanel fixture");
      }
    },
  );
  vi.spyOn(tldwClient, "ragHealth").mockResolvedValue({ status: "ok" });
  vi.spyOn(tldwClient, "webSearch").mockResolvedValue({
    web_search_results_dict: { results: [] },
  });
  const modelInfo = {
    id: "ready-model",
    name: "ready-model",
    provider: "openai",
    type: "chat" as const,
    capabilities: [],
  };
  vi.spyOn(tldwModels, "getModels").mockResolvedValue([modelInfo]);
  vi.spyOn(tldwModels, "getModel").mockResolvedValue(modelInfo);
  const provider = vi
    .spyOn(tldwClient, "streamChatCompletion")
    .mockImplementation(async function* () {
      yield* [];
      ready.unexpected.push("provider");
      throw new Error(
        "Unexpected provider dispatch in ready Sidepanel fixture",
      );
    });
  let view: ReturnType<typeof render> | undefined;
  const formElement = () => (
    <QueryClientProvider client={queryClient}>
      <SidepanelForm
        dropedFile={undefined}
        draftKey="sidepanel-ownership-ready"
      />
    </QueryClientProvider>
  );
  const cleanup = async () => {
    view?.unmount();
    await act(async () => {
      await Promise.allSettled(ready.submissions);
      await Promise.allSettled(connectionChecks);
      await Promise.allSettled(offline.capabilities);
    });
    queryClient.clear();
    useStoreMessageOption.setState(originalMessage);
    useUiModeStore.setState(originalUiMode);
    useConnectionStore.setState({
      state: originalConnection.state,
      checkOnce: originalConnection.checkOnce,
    });
    localStorage.clear();
    sessionStorage.clear();
    await tldwClient.initialize();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    restoreStorage(localStorage, originalLocalStorage);
    restoreStorage(sessionStorage, originalSessionStorage);
    ready.setImage = null;
    expect(
      ready.unexpected,
      `Unexpected external routes: ${JSON.stringify(ready.unexpected)}`,
    ).toEqual([]);
  };
  try {
    await act(async () => {
      await useConnectionStore.getState().checkOnce({ force: true });
    });
    expect(useConnectionStore.getState().state.isConnected).toBe(true);
    expect(useConnectionStore.getState().state.offlineBypass).toBe(false);
    view = render(formElement());
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "Send message" }),
      ).toBeInTheDocument(),
    );
    return { cleanup, provider, rerender: () => view?.rerender(formElement()) };
  } catch (error) {
    await cleanup();
    throw error;
  }
};
